"""Pi coding agent: writes ~/.pi/agent/models.json with Databricks-backed providers.

Pi (https://pi.dev) is a multi-provider coding agent. We register four
providers in its `models.json`, each speaking the API dialect best suited to
that family's gateway path:

- `databricks-claude`  (api: anthropic-messages)       → /ai-gateway/anthropic
- `databricks-openai`  (api: openai-responses)         → /ai-gateway/codex/v1
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

The `databricks-mlflow` provider carries the OSS chat-completions-only
foundation models (Llama, Qwen, GLM, inkling, ...) discovered upstream. Per
model it sets `contextWindow`/`maxTokens` from `databricks.model_token_limits`
and `reasoning` from `databricks.model_is_reasoning` (so Pi renders the
gateway's streamed reasoning_content as thinking).

At launch the mlflow provider is routed through a local SSE-repair proxy (see
`_mlflow_proxy`): some OSS models (inkling) omit the `finish_reason` on the
streaming terminal chunk, which Pi's parser rejects. The proxy injects it and
is a no-op for well-behaved models. Removable once the gateway is fixed.

The bearer token is baked into the file and refreshed by a background thread
while the session runs (same pattern as OpenCode/Copilot).
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading

from ucode.agent_updates import available_npm_package_update
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
    TOKEN_REFRESH_INTERVAL_SECONDS,
    build_pi_base_urls,
    get_databricks_token,
    model_is_reasoning,
    model_token_limits,
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


def is_update_available() -> tuple[str, str] | None:
    return available_npm_package_update(SPEC["package"])


def _resolve_model_selector(
    model: str,
    claude_models: dict[str, str],
    codex_models: list[str],
    gemini_models: list[str],
    oss_models: list[str],
) -> str:
    """Return a Pi model selector in `<provider>/<model>` form when possible."""
    for name in PROVIDER_NAMES:
        if model.startswith(f"{name}/"):
            return model
    if model in claude_models.values():
        return f"databricks-claude/{model}"
    if model in codex_models:
        return f"databricks-openai/{model}"
    if model in gemini_models:
        return f"databricks-gemini/{model}"
    if model in oss_models:
        return f"databricks-mlflow/{model}"
    return model


def _pi_oss_model_entry(model_id: str) -> dict:
    """Build a Pi mlflow model entry enriched from the shared limits/reasoning
    tables: `reasoning:true` for reasoning models (Pi renders their streamed
    reasoning_content as thinking), and `contextWindow`/`maxTokens` from
    `model_token_limits`. Fields are omitted when unknown so Pi keeps its
    default."""
    entry: dict = {"id": model_id}
    if model_is_reasoning(model_id):
        entry["reasoning"] = True
    limits = model_token_limits(model_id)
    if limits:
        if limits.get("context"):
            entry["contextWindow"] = limits["context"]
        if limits.get("output"):
            entry["maxTokens"] = limits["output"]
    return entry


def render_overlay(
    model: str,
    token: str,
    pi_base_urls: dict[str, str],
    claude_models: dict[str, str],
    codex_models: list[str],
    gemini_models: list[str],
    oss_models: list[str],
) -> tuple[dict, list[list[str]]]:
    """Return (overlay, managed_key_paths) for ~/.pi/agent/models.json."""
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
    if oss_models:
        providers["databricks-mlflow"] = {
            "baseUrl": pi_base_urls["oss"],
            "api": "openai-completions",
            "apiKey": token,
            "authHeader": True,
            # MLflow chat-completions gateway rejects OpenAI's `store` field
            # and per-tool `strict`. Pi omits both when these are false.
            "compat": {"supportsStore": False, "supportsStrictMode": False},
            "headers": ua_headers,
            "models": [_pi_oss_model_entry(m) for m in oss_models],
        }
        keys.append(["providers", "databricks-mlflow"])
    overlay: dict = {
        "model": _resolve_model_selector(
            model, claude_models, codex_models, gemini_models, oss_models
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
        state.get("oss_models") or [],
    )
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
    env["HOME"] = str(PI_UCODE_HOME)
    return env


def _start_oss_proxy(state: dict) -> threading.Event | None:
    """Route the mlflow provider through the local SSE-repair proxy.

    Rewrites ``state["base_urls"]["pi"]["oss"]`` in-memory to the proxy origin
    so both the initial config write and every token-refresh rewrite point Pi
    at the proxy. The gateway origin is always re-derived from the workspace
    (never the persisted `oss` URL, which may hold a dead proxy port from a
    prior session). Returns the proxy's stop event, or None when no OSS models
    are configured. See `_mlflow_proxy`.
    """
    if not (state.get("oss_models") or []):
        return None
    pi_urls = state.setdefault("base_urls", {}).setdefault(
        "pi", build_pi_base_urls(state["workspace"])
    )
    origin = build_pi_base_urls(state["workspace"])["oss"].split("/ai-gateway/", 1)[0]
    proxy_base, stop_event = _mlflow_proxy.start(origin)
    pi_urls["oss"] = f"{proxy_base}/ai-gateway/mlflow/v1"
    return stop_event


def launch(state: dict, tool_args: list[str]) -> None:
    proxy_stop = _start_oss_proxy(state)
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
        if proxy_stop is not None:
            proxy_stop.set()

    raise SystemExit(returncode)


def validate_cmd(binary: str) -> list[str]:
    return [binary, "--print", "say hi in 5 words or less"]


def validate_env(state: dict) -> dict[str, str]:
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured.")
    return build_runtime_env(get_databricks_token(workspace, state.get("profile")))
