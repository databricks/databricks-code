"""Pi coding agent: writes the user's ~/.pi/agent/models.json with Databricks-backed providers.

Pi (https://pi.dev) is a multi-provider coding agent. We register four
providers in its `models.json`, each speaking the API dialect best suited to
that family's gateway path:

- `databricks-claude`  (api: anthropic-messages)       → /ai-gateway/anthropic
- `databricks-openai`  (api: openai-responses)         → /ai-gateway/openai/v1
- `databricks-gemini`  (api: google-generative-ai)     → /ai-gateway/gemini/v1beta
- `databricks-mlflow`  (api: openai-completions)       → /ai-gateway/mlflow/v1

Per-provider `compat` flags work around fields the gateway translators reject:

- claude: `supportsEagerToolInputStreaming: false` — the Anthropic translator
  rejects `tools[].eager_input_streaming` on the streaming + tools path that
  pi uses for every request. With this flag pi omits the per-tool field and
  sends the legacy `anthropic-beta: fine-grained-tool-streaming-...` header
  instead, which the gateway accepts.
- mlflow: `supportsStore: false` and `supportsStrictMode: false` — the MLflow
  chat-completions gateway rejects OpenAI's `store` field and
  `tools[].function.strict`.
- openai: no `compat` flags needed, but the per-model `thinkingLevelMap`
  matters — see `_pi_gpt_model_entry`. Declaring `reasoning: true` without an
  off-state makes Pi send `reasoning: {effort: "none"}`, which `gpt-5`,
  `gpt-5-mini`, `gpt-5-nano` and `gpt-5-5-pro` reject with a 400.

The `databricks-mlflow` provider carries validated MLflow chat-completions
models discovered upstream. Per-model reasoning and token metadata comes from
the persisted gateway capability specs, with conservative static fallback.
At launch this provider alone is routed through a loopback repair proxy because
some models (notably Inkling) omit the terminal `finish_reason` Pi requires.

The bearer token is baked into the file and refreshed by a background thread
while the session runs (same pattern as OpenCode/Copilot).
"""

from __future__ import annotations

import ipaddress
import os
import signal
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any, cast
from urllib.parse import urlparse

from ucode.agents import _mlflow_proxy
from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    deep_merge_dict,
    read_json_safe,
    write_json_file,
)
from ucode.databricks import (
    ANTHROPIC_FAMILIES,
    TOKEN_REFRESH_INTERVAL_SECONDS,
    build_pi_base_urls,
    classify_model_family,
    claude_model_capabilities,
    discover_claude_models_unbucketed,
    get_databricks_token,
    gpt_model_token_limits,
    model_is_reasoning,
    model_token_limits,
    preferred_gpt_model,
)
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version
from ucode.ui import print_warning

# Use Pi's standard user configuration directory. Keep ucode's backups and
# state under APP_DIR, but let Pi share the user's normal ~/.pi/agent config.
PI_CONFIG_DIR = Path.home() / ".pi" / "agent"
PI_CONFIG_PATH = PI_CONFIG_DIR / "models.json"
PI_SETTINGS_PATH = PI_CONFIG_DIR / "settings.json"
# Do not reuse the legacy backup names from ucode's private Pi home. On upgrade,
# those files can contain an unrelated old private config and must never be
# restored over the user's standard ~/.pi/agent files.
PI_BACKUP_PATH = APP_DIR / "pi-agent-models.backup.json"
PI_SETTINGS_BACKUP_PATH = APP_DIR / "pi-agent-settings.backup.json"
_CONFIG_WRITE_LOCK = threading.RLock()

SPEC: ToolSpec = {
    "binary": "pi",
    "package": "@earendil-works/pi-coding-agent",
    "display": "Pi",
    "config_path": PI_CONFIG_PATH,
    "backup_path": PI_BACKUP_PATH,
}

PROVIDER_NAMES = (
    "databricks-claude",
    "databricks-openai",
    "databricks-gemini",
    "databricks-mlflow",
)

PROVIDER_KEYS: list[list[str]] = [["providers", name] for name in PROVIDER_NAMES]

# Old provider names earlier ucode versions wrote; cleaned up on each write so
# users don't end up with stale entries pointing at routes that 400.
LEGACY_PROVIDER_NAMES = ("databricks-anthropic", "databricks-codex", "databricks-oss")


def _resolve_model_selector(
    model: str,
    claude_models: dict[str, str],
    codex_models: list[str],
    gemini_models: list[str],
    oss_models: list[str],
    claude_model_ids: list[str] | None = None,
) -> str:
    """Return a Pi model selector in `<provider>/<model>` form when possible."""
    for name in PROVIDER_NAMES:
        if model.startswith(f"{name}/"):
            return model
    all_claude_models = set(claude_models.values())
    all_claude_models.update(claude_model_ids or [])
    if model in all_claude_models:
        return f"databricks-claude/{model}"
    if model in codex_models:
        return f"databricks-openai/{model}"
    if model in gemini_models:
        return f"databricks-gemini/{model}"
    if model in oss_models:
        return f"databricks-mlflow/{model}"
    return model


def _pi_claude_model_entry(model_id: str) -> dict:
    """Build a Claude entry with explicit limits.

    Databricks model ids do not match Pi's built-in Anthropic ids, so a bare
    custom entry silently gets Pi's 128k context / 4k output defaults.
    """
    capabilities = claude_model_capabilities(model_id)
    entry: dict = {
        "id": model_id,
        "reasoning": True,
        "input": ["text", "image"],
        "contextWindow": capabilities.context,
        "maxTokens": capabilities.output,
    }
    if capabilities.force_adaptive_thinking:
        entry["compat"] = {"forceAdaptiveThinking": True}
        # Pi hides its extended levels unless custom models declare them. All
        # adaptive Claude models support `max`; native `xhigh` is limited to
        # Opus 4.7/4.8, Sonnet 5, and Fable 5.
        entry["thinkingLevelMap"] = {"max": "max"}
        if capabilities.supports_xhigh_thinking:
            entry["thinkingLevelMap"]["xhigh"] = "xhigh"
    return entry


_OSS_SAFE_LIMITS = {"context": 128_000, "output": 8_192}


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _oss_specs_by_id(raw_specs: object) -> dict[str, dict[str, object]]:
    if not isinstance(raw_specs, list):
        return {}
    specs: dict[str, dict[str, object]] = {}
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, dict):
            continue
        typed_spec = cast(dict[str, object], raw_spec)
        model_id = typed_spec.get("id")
        reasoning = typed_spec.get("reasoning")
        context = typed_spec.get("context_window")
        output = typed_spec.get("max_tokens")
        valid_limits = all(
            value is None or _positive_int(value) is not None for value in (context, output)
        )
        if (
            isinstance(model_id, str)
            and model_id
            and isinstance(reasoning, bool)
            and "context_window" in typed_spec
            and "max_tokens" in typed_spec
            and valid_limits
            and model_id not in specs
        ):
            specs[model_id] = typed_spec
    return specs


def _pi_oss_model_entry(model_id: str, spec: dict[str, object] | None = None) -> dict:
    """Build a Pi MLflow model entry from discovered or static capabilities.

    A valid discovered boolean overrides static reasoning. Any discovered spec
    receives a complete conservative limit pair, so missing capability fields
    cannot leave a validated model effectively uncapped. Missing specs retain
    the existing static GLM/Kimi/DeepSeek behavior, while unknown models remain bare.
    """
    entry: dict = {"id": model_id}
    static_limits = model_token_limits(model_id)
    static_reasoning = model_is_reasoning(model_id)

    reasoning = spec.get("reasoning") if isinstance(spec, dict) else None
    if not isinstance(reasoning, bool):
        reasoning = static_reasoning
    if reasoning:
        entry["reasoning"] = True

    context = _positive_int(spec.get("context_window")) if isinstance(spec, dict) else None
    output = _positive_int(spec.get("max_tokens")) if isinstance(spec, dict) else None
    if isinstance(spec, dict):
        entry["contextWindow"] = context or (
            static_limits.get("context") if static_limits else _OSS_SAFE_LIMITS["context"]
        )
        entry["maxTokens"] = output or (
            static_limits.get("output") if static_limits else _OSS_SAFE_LIMITS["output"]
        )
    elif static_limits:
        entry["contextWindow"] = static_limits["context"]
        entry["maxTokens"] = static_limits["output"]
    return entry


def _responses_specs_by_id(raw_specs: object) -> dict[str, dict[str, object]]:
    if not isinstance(raw_specs, list):
        return {}
    specs: dict[str, dict[str, object]] = {}
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, dict):
            continue
        typed_spec = cast(dict[str, object], raw_spec)
        model_id = typed_spec.get("id")
        context = _positive_int(typed_spec.get("context_window"))
        if isinstance(model_id, str) and model_id and context is not None:
            specs.setdefault(model_id, typed_spec)
    return specs


def _pi_gpt_model_entry(model_id: str, spec: dict[str, object] | None = None) -> dict:
    """Build a Pi Responses model entry with explicit limits and reasoning.

    Gateway ids aren't in Pi's built-in catalog, so capabilities must be
    declared here rather than inherited from Pi's vendor model registry.

    `thinkingLevelMap: {"off": None}` is required alongside `reasoning: True`.
    When a model declares `reasoning` but no off-state, Pi's Responses builder
    falls back to `reasoning: {effort: "none"}` for the thinking-off case
    (`pi-ai/dist/api/openai-responses.js`, the `thinkingLevelMap?.off !== null`
    branch). `"none"` is only valid on gpt-5.1+, so `gpt-5`, `gpt-5-mini`,
    `gpt-5-nano` and `gpt-5-5-pro` reject every request with
    `BAD_REQUEST: Unsupported value: 'none' is not supported with the 'gpt-5'
    model`. An explicit `None` makes Pi omit `reasoning` entirely, which the
    gateway accepts for all ids.
    """
    limits = gpt_model_token_limits(model_id)
    discovered_context = (
        _positive_int(spec.get("context_window")) if isinstance(spec, dict) else None
    )
    entry: dict = {
        "id": model_id,
        "contextWindow": discovered_context or limits["context"],
        "maxTokens": limits["output"],
    }
    normalized_id = model_id.rsplit("/", 1)[-1].lower()
    for prefix in ("system.ai.", "databricks-"):
        if normalized_id.startswith(prefix):
            normalized_id = normalized_id[len(prefix) :]
            break
    normalized_id = normalized_id.replace(".", "-")
    if normalized_id == "grok-4-6":
        # Grok 4.6 is reasoning-only. Databricks supports low/medium/high/xhigh;
        # hide Pi's unsupported off/minimal/max choices instead of mapping them
        # to values the gateway would reject.
        entry["reasoning"] = True
        entry["thinkingLevelMap"] = {
            "off": None,
            "minimal": None,
            "xhigh": "xhigh",
            "max": None,
        }
    elif "gpt-5" in normalized_id:
        entry["reasoning"] = True
        entry["input"] = ["text", "image"]
        entry["thinkingLevelMap"] = {"off": None}
    return entry


def render_overlay(
    model: str,
    token: str,
    pi_base_urls: dict[str, str],
    claude_models: dict[str, str],
    codex_models: list[str],
    gemini_models: list[str],
    oss_models: list[str],
    oss_specs: list[dict] | None = None,
    claude_model_ids: list[str] | None = None,
    codex_specs: list[dict] | None = None,
) -> tuple[dict, list[list[str]]]:
    """Return (overlay, managed_key_paths) for Pi's user agent config."""
    providers: dict = {}
    keys: list[list[str]] = [["model"]]
    # Pi expands header values that match an env var name. Our UA contains
    # `/` and a space so it can never collide — safe to pass as a literal.
    ua_headers = {"User-Agent": f"ucode/{ucode_version()} pi/{agent_version('pi')}"}

    claude_ids = sorted(set(claude_models.values()) | set(claude_model_ids or []))
    if claude_ids:
        providers["databricks-claude"] = {
            "baseUrl": pi_base_urls["claude"],
            "api": "anthropic-messages",
            "apiKey": token,
            "authHeader": True,
            # Gateway's Anthropic translator rejects per-tool
            # `eager_input_streaming` on the streaming + tools path. Pi sends
            # the legacy beta header instead when this is false. Session
            # affinity keeps a Pi conversation on one AI Gateway destination,
            # which is important when traffic splitting is configured because
            # prompt caches are destination-local.
            "compat": {
                "supportsEagerToolInputStreaming": False,
                "sendSessionAffinityHeaders": True,
            },
            "headers": ua_headers,
            "models": [_pi_claude_model_entry(m) for m in claude_ids],
        }
        keys.append(["providers", "databricks-claude"])
    if codex_models:
        codex_specs_by_id = _responses_specs_by_id(codex_specs)
        providers["databricks-openai"] = {
            "baseUrl": pi_base_urls["openai"],
            "api": "openai-responses",
            "apiKey": token,
            "authHeader": True,
            "headers": ua_headers,
            "models": [
                _pi_gpt_model_entry(model_id, codex_specs_by_id.get(model_id))
                for model_id in codex_models
            ],
        }
        keys.append(["providers", "databricks-openai"])
    if gemini_models:
        providers["databricks-gemini"] = {
            "baseUrl": pi_base_urls["gemini"],
            "api": "google-generative-ai",
            "apiKey": token,
            "authHeader": True,
            "headers": ua_headers,
            "models": [{"id": m} for m in gemini_models],
        }
        keys.append(["providers", "databricks-gemini"])
    if oss_models:
        specs_by_id = _oss_specs_by_id(oss_specs)
        providers["databricks-mlflow"] = {
            "baseUrl": pi_base_urls["oss"],
            "api": "openai-completions",
            "apiKey": token,
            "authHeader": True,
            # MLflow chat-completions gateway rejects OpenAI's `store` field
            # and per-tool `strict`. Pi omits both when these are false.
            "compat": {"supportsStore": False, "supportsStrictMode": False},
            "headers": ua_headers,
            "models": [_pi_oss_model_entry(m, specs_by_id.get(m)) for m in oss_models],
        }
        keys.append(["providers", "databricks-mlflow"])
    overlay: dict = {
        "model": _resolve_model_selector(
            model,
            claude_models,
            codex_models,
            gemini_models,
            oss_models,
            claude_model_ids,
        ),
    }
    if providers:
        overlay["providers"] = providers
    return overlay, keys


def write_tool_config(
    state: dict,
    model: str,
    token: str | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[dict, str]:
    with _CONFIG_WRITE_LOCK:
        return _write_tool_config_unlocked(state, model, token, force_refresh=force_refresh)


def _write_tool_config_unlocked(
    state: dict,
    model: str,
    token: str | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[dict, str]:
    backup_existing_file(PI_CONFIG_PATH, PI_BACKUP_PATH)
    if token is None:
        token = get_databricks_token(
            state["workspace"], state.get("profile"), force_refresh=force_refresh
        )
    pi_base_urls = state.get("base_urls", {}).get("pi") or build_pi_base_urls(state["workspace"])
    managed_families = _managed_model_families(state)
    claude_model_ids: list[str] | None = None
    if managed_families is None:
        claude_models = state.get("claude_models") or {}
        codex_models = state.get("codex_models") or []
        gemini_models = state.get("gemini_models") or []
        oss_models = state.get("oss_models") or []
        # The shared map keeps one model per family; Pi supplements it with
        # the full inventory for every family that discovery enabled.
        claude_model_ids = (
            _discover_pi_claude_models(state, token, claude_models) if claude_models else None
        )
    else:
        claude_models, codex_models, gemini_models, oss_models = managed_families
        claude_model_ids = _managed_pi_claude_models(state)
    overlay, managed_keys = render_overlay(
        model,
        token,
        pi_base_urls,
        claude_models,
        codex_models,
        gemini_models,
        oss_models,
        state.get("oss_model_specs") or [],
        claude_model_ids,
        state.get("codex_model_specs") or [],
    )
    existing = read_json_safe(PI_CONFIG_PATH)
    providers = existing.get("providers")
    if isinstance(providers, dict):
        for stale in (*PROVIDER_NAMES, *LEGACY_PROVIDER_NAMES):
            providers.pop(stale, None)
    merged = deep_merge_dict(existing, overlay)
    write_json_file(PI_CONFIG_PATH, merged)
    _write_settings(overlay["model"], clear_unresolved=managed_families is not None)
    state = mark_tool_managed(state, "pi", managed_keys)
    save_state(state)
    return state, token


def _managed_pi_claude_models(state: dict) -> list[str]:
    """Return every Claude id explicitly allowed by a managed Pi config."""
    managed = state.get("pi_models")
    if not isinstance(managed, list):
        return []
    return [
        model
        for model in managed
        if isinstance(model, str) and classify_model_family(model) in ANTHROPIC_FAMILIES
    ]


def _discover_pi_claude_models(state: dict, token: str, claude_models: dict[str, str]) -> list[str]:
    """Return all Claude ids Pi may offer without changing shared routing state.

    Shared discovery intentionally keeps one model per Claude family and pins
    Opus to 4.8 while the smart router requires that arm. Pi is a model picker,
    so it can safely expose newer versions as long as its default remains the
    shared pinned id.
    Cache the supplemental inventory in state after the first Pi config write;
    a failed supplemental request degrades to the shared family picks.
    """
    allowed_families = set(claude_models)
    cached = state.get("pi_claude_models")
    if isinstance(cached, list):
        cached_models = [
            model
            for model in cached
            if isinstance(model, str) and classify_model_family(model) in allowed_families
        ]
        return list(dict.fromkeys([*claude_models.values(), *cached_models]))

    try:
        discovered, _ = discover_claude_models_unbucketed(state["workspace"], token)
    except (RuntimeError, OSError):
        discovered = []
    if discovered:
        state["pi_claude_models"] = discovered
        discovered_models = [
            model for model in discovered if classify_model_family(model) in allowed_families
        ]
        return list(dict.fromkeys([*claude_models.values(), *discovered_models]))
    return list(claude_models.values())


def _write_settings(model_selector: str, *, clear_unresolved: bool = False) -> None:
    # Pin defaultProvider/defaultModel in settings.json so Pi doesn't fall
    # through to an env-key-backed provider (e.g. HF_TOKEN exposing
    # huggingface) in `findInitialModel` when no --model is passed.
    provider, _, model_id = model_selector.partition("/")
    if not model_id and not clear_unresolved:
        return
    backup_existing_file(PI_SETTINGS_PATH, PI_SETTINGS_BACKUP_PATH)
    existing = read_json_safe(PI_SETTINGS_PATH)
    if not model_id:
        # A non-empty managed allowlist with no servable model must not retain a
        # stale default that bypasses the current administrator policy.
        existing.pop("defaultProvider", None)
        existing.pop("defaultModel", None)
        write_json_file(PI_SETTINGS_PATH, existing)
        return
    merged = deep_merge_dict(existing, {"defaultProvider": provider, "defaultModel": model_id})
    write_json_file(PI_SETTINGS_PATH, merged)


def _managed_model_families(
    state: dict,
) -> tuple[dict[str, str], list[str], list[str], list[str]] | None:
    """Split a managed config's ``pi_models`` into Pi provider inputs.

    ``None`` means there is no managed allowlist and workspace discovery may be
    used. A present non-empty allowlist always returns four collections—even if
    every entry is malformed or unsupported—so unlisted discovered models can
    never leak back into managed Pi configuration.
    """
    managed = state.get("pi_models")
    if not isinstance(managed, list) or not managed:
        return None
    claude: dict[str, str] = {}
    codex: list[str] = []
    gemini: list[str] = []
    oss: list[str] = []
    for model in managed:
        if not isinstance(model, str) or not model.strip():
            continue
        family = classify_model_family(model)
        if family in ANTHROPIC_FAMILIES:
            claude.setdefault(family, model)
        elif family == "codex":
            codex.append(model)
        elif family == "gemini":
            gemini.append(model)
        elif family == "oss":
            oss.append(model)
    return claude, codex, gemini, oss


def default_model(state: dict) -> str | None:
    """Prefer a managed Pi default/allowlist, then Claude opus → sonnet → haiku;
    fall back to codex, Gemini, then OSS.

    A managed config's ``pi_default_model`` and ``pi_models`` both win outright: the former is
    the admin's chosen session start, the latter their allowlist. Workspace-wide discovery falls back.
    """
    if isinstance(state.get("pi_default_model"), str):
        return state.get("pi_default_model")
    managed = state.get("pi_models")
    if isinstance(managed, list) and managed:
        return managed[0]
    claude_models = state.get("claude_models") or {}
    for family in ("opus", "sonnet", "haiku"):
        if claude_models.get(family):
            return claude_models[family]
    codex_model = preferred_gpt_model(state.get("codex_models") or [])
    if codex_model:
        return codex_model
    gemini_models = state.get("gemini_models") or []
    if gemini_models:
        return gemini_models[0]
    oss_models = state.get("oss_models") or []
    return oss_models[0] if oss_models else None


def _refresh_token_once(state: dict, *, force_refresh: bool = False) -> str:
    model = default_model(state)
    if not model:
        raise RuntimeError("No Pi model is available on this workspace.")
    _, token = write_tool_config(state, model, force_refresh=force_refresh)
    return token


def _refresh_forever(state: dict, stop_event: threading.Event) -> None:
    while not stop_event.wait(TOKEN_REFRESH_INTERVAL_SECONDS):
        try:
            _refresh_token_once(state, force_refresh=True)
        except RuntimeError:
            continue


def build_runtime_env(token: str) -> dict[str, str]:
    env = os.environ.copy()
    env["OAUTH_TOKEN"] = token
    env["PI_CODING_AGENT_DIR"] = str(PI_CONFIG_DIR)
    return env


def _is_loopback_origin(origin: str) -> bool:
    hostname = urlparse(origin).hostname
    if not hostname:
        return True
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _start_oss_proxy(
    state: dict,
) -> tuple[threading.Thread, _mlflow_proxy.ThreadingHTTPServer] | None:
    """Start Pi's MLflow repair proxy and rewrite only the in-memory base URL.

    The upstream is always derived from ``workspace`` rather than an existing
    base URL, so a stale loopback port from an old config can never become the
    next proxy's upstream.
    """
    managed_families = _managed_model_families(state)
    oss_models = state.get("oss_models") or [] if managed_families is None else managed_families[3]
    if not oss_models:
        return None
    direct_oss_url = build_pi_base_urls(state["workspace"])["oss"]
    origin = direct_oss_url.split("/ai-gateway/", 1)[0]
    if _is_loopback_origin(origin):
        print_warning("MLflow stream repair proxy skipped for a loopback workspace URL.")
        return None
    started = _mlflow_proxy.start(origin)
    if started is None:
        return None
    server, proxy_origin = started
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    try:
        thread.start()
    except RuntimeError:
        server.server_close()
        print_warning("MLflow stream repair proxy could not start; using the direct gateway URL.")
        return None
    pi_urls = state.setdefault("base_urls", {}).setdefault(
        "pi", build_pi_base_urls(state["workspace"])
    )
    pi_urls["oss"] = f"{proxy_origin}/ai-gateway/mlflow/v1"
    return thread, server


def _restore_direct_oss_config(state: dict, token: str | None) -> None:
    """Replace the session-only proxy URL after any in-flight config write."""
    with _CONFIG_WRITE_LOCK:
        _restore_direct_oss_config_unlocked(state, token)


def _restore_direct_oss_config_unlocked(state: dict, token: str | None) -> None:
    pi_urls = state.setdefault("base_urls", {}).setdefault(
        "pi", build_pi_base_urls(state["workspace"])
    )
    pi_urls["oss"] = build_pi_base_urls(state["workspace"])["oss"]
    model = default_model(state)
    if model and token is not None:
        write_tool_config(state, model, token=token)
        return

    # Token acquisition can fail before the normal config rewrite returns a
    # token. Repair an existing generated provider in place without changing
    # its credential, so a stale loopback port is never left behind.
    existing = read_json_safe(PI_CONFIG_PATH)
    providers = existing.get("providers")
    mlflow = providers.get("databricks-mlflow") if isinstance(providers, dict) else None
    if isinstance(mlflow, dict):
        mlflow["baseUrl"] = pi_urls["oss"]
        write_json_file(PI_CONFIG_PATH, existing)


def launch(state: dict, tool_args: list[str]) -> None:
    proxy: tuple[threading.Thread, _mlflow_proxy.ThreadingHTTPServer] | None = None
    stop_event = threading.Event()
    refresher: threading.Thread | None = None
    proc: subprocess.Popen | None = None
    token: str | None = None
    primary_error: BaseException | None = None
    previous_signal_handlers: dict[
        signal.Signals, int | Callable[[int, FrameType | None], Any] | None
    ] = {}
    termination_requested = False

    def handle_termination(signum: int, _frame: FrameType | None) -> None:
        """Forward catchable terminal shutdowns and unwind through proxy cleanup."""
        nonlocal termination_requested
        if termination_requested:
            return
        termination_requested = True
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signum)
            except OSError:
                pass
        raise SystemExit(128 + signum)

    try:
        for signal_name in ("SIGTERM", "SIGHUP"):
            signum = getattr(signal, signal_name, None)
            if signum is None:
                continue
            # Record the previous handler only AFTER a successful install, so the
            # restore loop below can never put back a handler we never replaced
            # (and so a failed install isn't masked by a failing restore).
            previous_handler = signal.getsignal(signum)
            signal.signal(signum, handle_termination)
            previous_signal_handlers[signum] = previous_handler
        try:
            # The proxy must be live and its URL in state before the first config
            # write; refreshes then keep writing the same live loopback endpoint.
            proxy = _start_oss_proxy(state)
            token = _refresh_token_once(state)
            env = build_runtime_env(token)

            refresher = threading.Thread(
                target=_refresh_forever,
                args=(state, stop_event),
                daemon=True,
            )
            refresher.start()

            proc = subprocess.Popen([SPEC["binary"], *tool_args], env=env)
            try:
                returncode = proc.wait()
            except KeyboardInterrupt:
                proc.send_signal(signal.SIGINT)
                returncode = proc.wait()
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            stop_event.set()
            if refresher is not None:
                refresher.join(timeout=1)
            if proxy is not None:
                proxy_thread, server = proxy
                restore_error: Exception | None = None
                try:
                    _restore_direct_oss_config(state, token)
                except Exception as exc:
                    restore_error = exc
                    print_warning(f"Pi MLflow direct configuration could not be restored ({exc}).")
                finally:
                    server.shutdown()
                    server.server_close()
                    proxy_thread.join(timeout=1)
                if restore_error is not None and primary_error is None:
                    raise restore_error
    finally:
        for signum, previous_handler in previous_signal_handlers.items():
            signal.signal(signum, previous_handler)

    raise SystemExit(returncode)


def validate_cmd(binary: str) -> list[str]:
    return [binary, "--print", "say hi in 5 words or less"]


def validate_env(state: dict) -> dict[str, str]:
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured.")
    return build_runtime_env(get_databricks_token(workspace, state.get("profile")))
