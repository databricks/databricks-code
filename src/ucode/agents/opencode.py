"""OpenCode agent: writes opencode.json with Databricks-backed providers."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from typing import cast

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
    build_opencode_base_urls,
    get_databricks_token,
    gpt_model_token_limits,
    model_token_limits,
    preferred_gpt_model,
)
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version

OPENCODE_XDG_CONFIG_HOME = APP_DIR / "opencode-xdg"
OPENCODE_CONFIG_DIR = OPENCODE_XDG_CONFIG_HOME / "opencode"
OPENCODE_CONFIG_PATH = OPENCODE_CONFIG_DIR / "opencode.json"
OPENCODE_BACKUP_PATH = APP_DIR / "opencode-config.backup.json"

SPEC: ToolSpec = {
    "binary": "opencode",
    "package": "opencode-ai",
    "display": "OpenCode",
    "config_path": OPENCODE_CONFIG_PATH,
    "backup_path": OPENCODE_BACKUP_PATH,
}

PROVIDER_KEYS: list[list[str]] = [
    ["provider", "databricks-anthropic"],
    ["provider", "databricks-google"],
    ["provider", "databricks-openai"],
    ["provider", "databricks-oss"],
]


def _resolve_model_selector(model: str, opencode_models: dict[str, list[str]]) -> str:
    """Return an OpenCode model selector in provider/model form when possible."""
    if model.startswith(
        (
            "databricks-anthropic/",
            "databricks-google/",
            "databricks-openai/",
            "databricks-oss/",
        )
    ):
        return model

    anthropic_models = opencode_models.get("anthropic") or []
    if model in anthropic_models:
        return f"databricks-anthropic/{model}"

    gemini_models = opencode_models.get("gemini") or []
    if model in gemini_models:
        return f"databricks-google/{model}"

    openai_models = opencode_models.get("openai") or []
    if model in openai_models:
        return f"databricks-openai/{model}"

    oss_models = opencode_models.get("oss") or []
    if model in oss_models:
        return f"databricks-oss/{model}"

    return model


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


def _oss_model_overlay(
    model: str, ua_header: dict[str, str], spec: dict[str, object] | None = None
) -> dict:
    """Per-model OSS overlay from discovered or static capabilities.

    OpenCode requires context and output limits together. Every discovered spec
    therefore receives a complete conservative pair. Missing specs retain
    static GLM/Kimi/DeepSeek metadata, and unknown no-spec models remain uncapped.
    """
    overlay: dict = {"headers": ua_header}
    static_limits = model_token_limits(model)
    context = _positive_int(spec.get("context_window")) if isinstance(spec, dict) else None
    output = _positive_int(spec.get("max_tokens")) if isinstance(spec, dict) else None
    if isinstance(spec, dict):
        overlay["limit"] = {
            "context": context
            or (static_limits.get("context") if static_limits else _OSS_SAFE_LIMITS["context"]),
            "output": output
            or (static_limits.get("output") if static_limits else _OSS_SAFE_LIMITS["output"]),
        }
    elif static_limits is not None:
        overlay["limit"] = static_limits

    reasoning = spec.get("reasoning") if isinstance(spec, dict) else None
    if isinstance(reasoning, bool):
        overlay["reasoning"] = reasoning
    return overlay


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


def _openai_model_overlay(
    model: str, ua_header: dict[str, str], spec: dict[str, object] | None = None
) -> dict:
    """Per-model Responses API options and explicit token limits."""
    limits = gpt_model_token_limits(model)
    discovered_context = (
        _positive_int(spec.get("context_window")) if isinstance(spec, dict) else None
    )
    if discovered_context is not None:
        limits["context"] = discovered_context
    return {
        "headers": ua_header,
        "limit": limits,
        "options": {"useResponsesApi": True},
    }


def render_overlay(
    model: str,
    token: str,
    opencode_base_urls: dict[str, str],
    opencode_models: dict[str, list[str]],
    oss_specs: list[dict] | None = None,
    codex_specs: list[dict] | None = None,
) -> tuple[dict, list[list[str]]]:
    """Return (overlay, managed_key_paths) for opencode.json."""
    auth_headers = {"Authorization": f"Bearer {token}"}
    # OpenCode hardcodes `User-Agent: opencode/<ver>` in session/llm.ts for
    # every provider, after the AI SDK's combineHeaders. The provider-level
    # `headers` are clobbered by that injection, but per-model `headers` are
    # merged AFTER and win — so the UA must live on each model entry.
    ua_header = {
        "User-Agent": f"ucode/{ucode_version()} opencode/{agent_version('opencode')}",
    }

    anthropic_models = opencode_models.get("anthropic") or []
    gemini_models = opencode_models.get("gemini") or []
    openai_models = opencode_models.get("openai") or []
    oss_models = opencode_models.get("oss") or []

    providers: dict = {}
    keys: list[list[str]] = [["model"]]
    if anthropic_models:
        # @ai-sdk/anthropic injects `eager_input_streaming: true` on tool defs;
        # the Databricks gateway's strict validator rejects it. opencode's
        # auto-disable in transform.ts skips models whose id contains "claude",
        # so we opt out per-model. The setting lives in per-call providerOptions,
        # which opencode reads from `models.<m>.options`, not provider `options`.
        anthropic_model_overlay = {
            "headers": ua_header,
            "options": {"toolStreaming": False},
        }
        providers["databricks-anthropic"] = {
            "npm": "@ai-sdk/anthropic",
            "options": {
                "baseURL": opencode_base_urls["anthropic"],
                "apiKey": token,
                "headers": auth_headers,
            },
            "models": dict.fromkeys(anthropic_models, anthropic_model_overlay),
        }
        keys.append(["provider", "databricks-anthropic"])
    if gemini_models:
        providers["databricks-google"] = {
            "npm": "@ai-sdk/google",
            "options": {
                "baseURL": opencode_base_urls["gemini"],
                "apiKey": token,
                "headers": auth_headers,
            },
            "models": {m: {"headers": ua_header} for m in gemini_models},
        }
        keys.append(["provider", "databricks-google"])
    if openai_models:
        codex_specs_by_id = _responses_specs_by_id(codex_specs)
        # @ai-sdk/openai supports both Responses and legacy chat completions.
        # These models use the native `/ai-gateway/openai/v1/responses` route,
        # so `useResponsesApi: true` lives in models.<m>.options where OpenCode
        # reads it (provider-level options is read by the SDK only).
        providers["databricks-openai"] = {
            "npm": "@ai-sdk/openai",
            "options": {
                "baseURL": opencode_base_urls["openai"],
                "apiKey": token,
                "headers": auth_headers,
            },
            "models": {
                model_id: _openai_model_overlay(
                    model_id, ua_header, codex_specs_by_id.get(model_id)
                )
                for model_id in openai_models
            },
        }
        keys.append(["provider", "databricks-openai"])
    if oss_models:
        specs_by_id = _oss_specs_by_id(oss_specs)
        providers["databricks-oss"] = {
            "npm": "@ai-sdk/openai",
            "options": {
                "baseURL": opencode_base_urls["oss"],
                "apiKey": token,
                "headers": auth_headers,
                # OpenCode otherwise adds `prompt_cache_key`, which the MLflow
                # chat-completions gateway rejects as an unknown field.
                "setCacheKey": False,
            },
            "models": {m: _oss_model_overlay(m, ua_header, specs_by_id.get(m)) for m in oss_models},
        }
        keys.append(["provider", "databricks-oss"])

    overlay: dict = {"model": _resolve_model_selector(model, opencode_models)}
    if providers:
        overlay["provider"] = providers
    return overlay, keys


def write_tool_config(
    state: dict,
    model: str,
    token: str | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[dict, str]:
    backup_existing_file(OPENCODE_CONFIG_PATH, OPENCODE_BACKUP_PATH)
    if token is None:
        token = get_databricks_token(
            state["workspace"], state.get("profile"), force_refresh=force_refresh
        )
    opencode_base_urls = state.get("base_urls", {}).get("opencode") or build_opencode_base_urls(
        state["workspace"]
    )
    overlay, managed_keys = render_overlay(
        model,
        token,
        opencode_base_urls,
        state.get("opencode_models") or {},
        state.get("oss_model_specs") or [],
        state.get("codex_model_specs") or [],
    )
    existing = read_json_safe(OPENCODE_CONFIG_PATH)
    providers = existing.get("provider")
    if isinstance(providers, dict):
        for stale in (
            "databricks-anthropic",
            "databricks-google",
            "databricks-openai",
            "databricks-oss",
        ):
            providers.pop(stale, None)
    merged = deep_merge_dict(existing, overlay)
    write_json_file(OPENCODE_CONFIG_PATH, merged)
    state = mark_tool_managed(state, "opencode", managed_keys)
    save_state(state)
    return state, token


def build_mcp_server_entry(argv: list[str]) -> dict:
    # A `local` MCP server runs a command over stdio; `command` is the full
    # argv. ucode registers the `ucode mcp-proxy ...` bridge here so OpenCode
    # never speaks HTTP+bearer directly — the proxy mints fresh tokens itself.
    return {
        "type": "local",
        "command": list(argv),
        "enabled": True,
    }


def write_mcp_server_config(name: str, argv: list[str]) -> bool:
    backup_existing_file(OPENCODE_CONFIG_PATH, OPENCODE_BACKUP_PATH)
    existing = read_json_safe(OPENCODE_CONFIG_PATH)
    mcp_servers = existing.get("mcp")
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
    removed = name in mcp_servers
    mcp_servers[name] = build_mcp_server_entry(argv)
    existing["mcp"] = mcp_servers
    write_json_file(OPENCODE_CONFIG_PATH, existing)
    return removed


def remove_mcp_server_config(name: str) -> bool:
    existing = read_json_safe(OPENCODE_CONFIG_PATH)
    mcp_servers = existing.get("mcp")
    if not isinstance(mcp_servers, dict) or name not in mcp_servers:
        return False
    mcp_servers.pop(name)
    existing["mcp"] = mcp_servers
    write_json_file(OPENCODE_CONFIG_PATH, existing)
    return True


def default_model(state: dict) -> str | None:
    if isinstance(state.get("opencode_default_model"), str):
        return state.get("opencode_default_model")
    opencode_models = state.get("opencode_models") or {}
    anthropic = opencode_models.get("anthropic") or []
    if anthropic:
        return anthropic[0]
    openai = preferred_gpt_model(opencode_models.get("openai") or [])
    if openai:
        return openai
    gemini = opencode_models.get("gemini") or []
    if gemini:
        return gemini[0]
    oss = opencode_models.get("oss") or []
    return oss[0] if oss else None


def _refresh_token_once(state: dict, *, force_refresh: bool = False) -> str:
    model = default_model(state)
    if not model:
        raise RuntimeError("No OpenCode model is configured.")
    _, token = write_tool_config(state, model, force_refresh=force_refresh)
    return token


def _refresh_forever(state: dict, stop_event: threading.Event) -> None:
    while not stop_event.wait(TOKEN_REFRESH_INTERVAL_SECONDS):
        try:
            _refresh_token_once(state, force_refresh=True)
        except RuntimeError:
            continue


def build_runtime_env(token: str, state: dict | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["OAUTH_TOKEN"] = token
    env["XDG_CONFIG_HOME"] = str(OPENCODE_XDG_CONFIG_HOME)
    return env


def launch(state: dict, tool_args: list[str]) -> None:
    """Launch opencode with background token refresh (same pattern as Gemini)."""
    token = _refresh_token_once(state)
    env = build_runtime_env(token, state)

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
    return [binary, "run", "say hi in 5 words or less"]


def validate_env(state: dict) -> dict[str, str]:
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured.")
    return build_runtime_env(get_databricks_token(workspace, state.get("profile")), state)
