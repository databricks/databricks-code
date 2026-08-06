"""Pi coding agent: writes ~/.pi/agent/models.json with Databricks-backed providers.

Pi (https://pi.dev) is a multi-provider coding agent. We register three
providers in its `models.json`, each speaking the API dialect best suited to
that family's gateway path:

- `databricks-claude`  (api: anthropic-messages)       → /ai-gateway/anthropic
- `databricks-openai`  (api: openai-responses)         → /ai-gateway/codex/v1
- `databricks-gemini`  (api: google-generative-ai)     → /ai-gateway/gemini/v1beta

A fourth provider, `databricks-custom` (api: openai-completions →
/ai-gateway/openai/v1), is written only when launching through a `custom` Model
Provider Service — a self-hosted, OpenAI-compatible model registered in Unity
Catalog. The service is selected by the `Databricks-Model-Provider-Service`
header and its targets are the models, so no Databricks model is pinned.

Per-provider `compat` flags work around fields the gateway translators reject:

- claude: `supportsEagerToolInputStreaming: false` — the Anthropic translator
  rejects `tools[].eager_input_streaming` on the streaming + tools path that
  pi uses for every request. With this flag pi omits the per-tool field and
  sends the legacy `anthropic-beta: fine-grained-tool-streaming-...` header
  instead, which the gateway accepts.
- custom: three flags turn off assumptions pi makes for openai.com. Pi's own
  auto-detection already gets the rest right for an unrecognized base URL, so
  restating them would be noise.

OSS / Databricks-foundation models (Llama, Qwen, etc.) are not exposed via
pi today — they live behind /ai-gateway/mlflow/v1 with per-model
`max_tokens` caps that pi has no global way to honor without per-model
config we don't currently maintain.

The bearer token is baked into the file and refreshed by a background thread
while the session runs (same pattern as OpenCode/Copilot).
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading

from ucode.agent_updates import available_npm_package_update
from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    deep_merge_dict,
    read_json_safe,
    write_json_file,
)
from ucode.databricks import (
    OPENAI_CHAT_NATIVE_API_TYPE,
    TOKEN_REFRESH_INTERVAL_SECONDS,
    build_native_api_base_url,
    build_pi_base_urls,
    get_databricks_token,
)
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version

PI_UCODE_HOME = APP_DIR / "pi-home"
PI_CONFIG_DIR = PI_UCODE_HOME / ".pi" / "agent"
PI_CONFIG_PATH = PI_CONFIG_DIR / "models.json"
PI_SETTINGS_PATH = PI_CONFIG_DIR / "settings.json"
PI_BACKUP_PATH = APP_DIR / "pi-models.backup.json"
PI_SETTINGS_BACKUP_PATH = APP_DIR / "pi-settings.backup.json"

SPEC: ToolSpec = {
    "binary": "pi",
    "package": "@earendil-works/pi-coding-agent",
    "display": "Pi",
    "config_path": PI_CONFIG_PATH,
    "backup_path": PI_BACKUP_PATH,
}

CUSTOM_PROVIDER_NAME = "databricks-custom"

PROVIDER_NAMES = (
    "databricks-claude",
    "databricks-openai",
    "databricks-gemini",
    # Written only under a custom Model Provider Service, but listed here
    # unconditionally so it's stripped on every write: a later launch without
    # `--provider` (or against a different service) must not leave a stale
    # provider pointing at the old service's header.
    CUSTOM_PROVIDER_NAME,
)

PROVIDER_KEYS: list[list[str]] = [["providers", name] for name in PROVIDER_NAMES]

# The Model Provider Service API exposes no context-window metadata, so ucode has
# to assume one for a custom target. The two directions are not symmetric:
# understating costs earlier compaction and shorter replies, while overstating is
# unrecoverable — pi compacts to `contextWindow - reserveTokens`, so a window
# above the server's real limit makes the compact-and-retry overflow again and
# the turn ends with "Context overflow recovery failed after one
# compact-and-retry attempt." Assume the conservative floor common to
# self-hosted OpenAI-compatible servers and let users raise it with
# `ucode configure --provider-context-window`.
PROVIDER_CONTEXT_WINDOW = 32768
PROVIDER_MAX_OUTPUT_TOKENS = 8192

# Old provider names earlier ucode versions wrote; cleaned up on each write so
# users don't end up with stale entries pointing at routes that 400.
LEGACY_PROVIDER_NAMES = ("databricks-anthropic", "databricks-codex", "databricks-oss")


def is_update_available() -> tuple[str, str] | None:
    return available_npm_package_update(SPEC["package"])


def _resolve_model_selector(
    model: str | None,
    claude_models: dict[str, str],
    codex_models: list[str],
    gemini_models: list[str],
    provider_models: list[str] | None = None,
) -> str:
    """Return a Pi model selector in `<provider>/<model>` form when possible.

    The provider-qualified form matters: a bare target id would match Pi's own
    built-in provider of the same name (e.g. `deepseek`) and fail with "No API
    key found for deepseek" rather than routing through the gateway.
    """
    provider_models = provider_models or []
    # Under a custom Model Provider Service no Databricks model is resolved, so
    # the service's first target is the default.
    if not model:
        return f"{CUSTOM_PROVIDER_NAME}/{provider_models[0]}" if provider_models else ""
    for name in PROVIDER_NAMES:
        if model.startswith(f"{name}/"):
            return model
    if model in provider_models:
        return f"{CUSTOM_PROVIDER_NAME}/{model}"
    if model in claude_models.values():
        return f"databricks-claude/{model}"
    if model in codex_models:
        return f"databricks-openai/{model}"
    if model in gemini_models:
        return f"databricks-gemini/{model}"
    return model


def render_overlay(
    model: str | None,
    token: str,
    pi_base_urls: dict[str, str],
    claude_models: dict[str, str],
    codex_models: list[str],
    gemini_models: list[str],
    *,
    provider: str | None = None,
    provider_models: list[str] | None = None,
    provider_base_url: str | None = None,
    context_window: int | None = None,
) -> tuple[dict, list[list[str]]]:
    """Return (overlay, managed_key_paths) for ~/.pi/agent/models.json.

    ``provider`` is a `custom` Model Provider Service name; ``provider_models``
    its routable targets. When both are set (plus a base URL for the dialect) a
    `databricks-custom` provider is emitted alongside whichever Databricks
    providers the workspace exposes.
    """
    providers: dict = {}
    keys: list[list[str]] = [["model"]]
    # Pi expands header values that match an env var name. Our UA contains
    # `/` and a space so it can never collide — safe to pass as a literal.
    ua_headers = {"User-Agent": f"ucode/{ucode_version()} pi/{agent_version('pi')}"}

    claude_ids = sorted(set(claude_models.values()))
    if claude_ids:
        providers["databricks-claude"] = {
            "baseUrl": pi_base_urls["claude"],
            "api": "anthropic-messages",
            "apiKey": token,
            "authHeader": True,
            # Gateway's Anthropic translator rejects per-tool
            # `eager_input_streaming` on the streaming + tools path. Pi sends
            # the legacy beta header instead when this is false.
            "compat": {"supportsEagerToolInputStreaming": False},
            "headers": ua_headers,
            "models": [{"id": m} for m in claude_ids],
        }
        keys.append(["providers", "databricks-claude"])
    if codex_models:
        providers["databricks-openai"] = {
            "baseUrl": pi_base_urls["openai"],
            "api": "openai-responses",
            "apiKey": token,
            "authHeader": True,
            "headers": ua_headers,
            "models": [{"id": m} for m in codex_models],
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
    if provider and provider_models and provider_base_url:
        window = context_window or PROVIDER_CONTEXT_WINDOW
        providers[CUSTOM_PROVIDER_NAME] = {
            "baseUrl": provider_base_url,
            "api": "openai-completions",
            "apiKey": token,
            "authHeader": True,
            # The header selects the service; the body's `model` carries the bare
            # target name (the fully-qualified service name returns NOT_FOUND).
            "headers": {**ua_headers, "Databricks-Model-Provider-Service": provider},
            # Only the flags that change pi's behavior against an unknown
            # OpenAI-compatible backend. Pi already auto-detects
            # supportsReasoningEffort / supportsUsageInStreaming /
            # supportsStrictMode correctly for an unrecognized base URL.
            "compat": {
                # Older OSS servers accept `max_tokens`, not the newer
                # `max_completion_tokens` pi would otherwise send.
                "maxTokensField": "max_tokens",
                # The `developer` role is an OpenAI-ism; `system` is universal.
                "supportsDeveloperRole": False,
                # Many OSS servers reject unknown body fields such as `store`.
                "supportsStore": False,
            },
            # No `reasoning`/`thinkingLevelMap`: the service exposes no capability
            # metadata, and claiming reasoning support would make pi send
            # `reasoning_effort` on every request, which a non-reasoning server
            # can reject outright.
            "models": [
                {
                    "id": target,
                    "contextWindow": window,
                    "maxTokens": min(PROVIDER_MAX_OUTPUT_TOKENS, window // 4),
                }
                for target in provider_models
            ],
        }
        keys.append(["providers", CUSTOM_PROVIDER_NAME])
    overlay: dict = {
        "model": _resolve_model_selector(
            model, claude_models, codex_models, gemini_models, provider_models
        ),
    }
    if providers:
        overlay["providers"] = providers
    return overlay, keys


def write_tool_config(
    state: dict,
    model: str | None,
    token: str | None = None,
    *,
    provider: str | None = None,
    provider_models: list[str] | None = None,
    force_refresh: bool = False,
) -> tuple[dict, str]:
    backup_existing_file(PI_CONFIG_PATH, PI_BACKUP_PATH)
    if token is None:
        token = get_databricks_token(
            state["workspace"], state.get("profile"), force_refresh=force_refresh
        )
    pi_base_urls = state.get("base_urls", {}).get("pi") or build_pi_base_urls(state["workspace"])
    overlay, managed_keys = render_overlay(
        model,
        token,
        pi_base_urls,
        state.get("claude_models") or {},
        state.get("codex_models") or [],
        state.get("gemini_models") or [],
        provider=provider,
        provider_models=provider_models,
        provider_base_url=build_native_api_base_url(
            state["workspace"], OPENAI_CHAT_NATIVE_API_TYPE
        ),
        context_window=state.get("provider_context_window"),
    )
    # Persist the resolved pair so the background refresh thread can rewrite the
    # same provider config without re-resolving it, and clear it on a launch
    # without a provider so the next refresh doesn't resurrect a stale one.
    if provider and provider_models:
        state["pi_provider"] = provider
        state["pi_provider_models"] = list(provider_models)
    else:
        state.pop("pi_provider", None)
        state.pop("pi_provider_models", None)
    existing = read_json_safe(PI_CONFIG_PATH)
    providers = existing.get("providers")
    if isinstance(providers, dict):
        for stale in (*PROVIDER_NAMES, *LEGACY_PROVIDER_NAMES):
            providers.pop(stale, None)
    merged = deep_merge_dict(existing, overlay)
    write_json_file(PI_CONFIG_PATH, merged)
    _write_settings(overlay["model"])
    state = mark_tool_managed(state, "pi", managed_keys)
    save_state(state)
    return state, token


def _write_settings(model_selector: str) -> None:
    # Pin defaultProvider/defaultModel in settings.json so Pi doesn't fall
    # through to an env-key-backed provider (e.g. HF_TOKEN exposing
    # huggingface) in `findInitialModel` when no --model is passed.
    provider, _, model_id = model_selector.partition("/")
    if not model_id:
        return
    backup_existing_file(PI_SETTINGS_PATH, PI_SETTINGS_BACKUP_PATH)
    existing = read_json_safe(PI_SETTINGS_PATH)
    merged = deep_merge_dict(existing, {"defaultProvider": provider, "defaultModel": model_id})
    write_json_file(PI_SETTINGS_PATH, merged)


def default_model(state: dict) -> str | None:
    """Prefer Claude opus → sonnet → haiku; fall back to codex, gemini."""
    claude_models = state.get("claude_models") or {}
    for family in ("opus", "sonnet", "haiku"):
        if claude_models.get(family):
            return claude_models[family]
    codex_models = state.get("codex_models") or []
    if codex_models:
        return codex_models[0]
    gemini_models = state.get("gemini_models") or []
    return gemini_models[0] if gemini_models else None


def _refresh_token_once(state: dict, *, force_refresh: bool = False) -> str:
    # Under a Model Provider Service the workspace may expose no Databricks model
    # at all — the service's targets are the models. Requiring one here would
    # raise, and `_refresh_forever` swallows that, so the token would silently
    # stop refreshing and the session would die when it expired.
    provider = state.get("pi_provider")
    provider_models = state.get("pi_provider_models") or []
    model = None if provider else default_model(state)
    if not model and not (provider and provider_models):
        raise RuntimeError("No Pi model is available on this workspace.")
    _, token = write_tool_config(
        state,
        model,
        provider=provider,
        provider_models=provider_models or None,
        force_refresh=force_refresh,
    )
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
    env["HOME"] = str(PI_UCODE_HOME)
    return env


def launch(state: dict, tool_args: list[str]) -> None:
    token = _refresh_token_once(state)
    env = build_runtime_env(token)

    stop_event = threading.Event()
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
    finally:
        stop_event.set()
        refresher.join(timeout=1)

    raise SystemExit(returncode)


def validate_cmd(binary: str) -> list[str]:
    return [binary, "--print", "say hi in 5 words or less"]


def validate_env(state: dict) -> dict[str, str]:
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured.")
    return build_runtime_env(get_databricks_token(workspace, state.get("profile")))
