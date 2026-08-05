"""OpenCode agent: writes Databricks providers and a request-time auth plugin."""

from __future__ import annotations

import json
import os

from ucode.agent_updates import available_npm_package_update
from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    deep_merge_dict,
    read_json_safe,
    write_json_file,
    write_text_file,
)
from ucode.databricks import (
    build_auth_token_argv,
    build_opencode_base_urls,
    model_token_limits,
)
from ucode.launcher import exec_or_spawn
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version

OPENCODE_XDG_CONFIG_HOME = APP_DIR / "opencode-xdg"
OPENCODE_CONFIG_DIR = OPENCODE_XDG_CONFIG_HOME / "opencode"
OPENCODE_CONFIG_PATH = OPENCODE_CONFIG_DIR / "opencode.json"
OPENCODE_AUTH_PLUGIN_PATH = OPENCODE_CONFIG_DIR / "plugins" / "ucode-databricks-auth.js"
OPENCODE_BACKUP_PATH = APP_DIR / "opencode-config.backup.json"

SPEC: ToolSpec = {
    "binary": "opencode",
    "package": "opencode-ai",
    "display": "OpenCode",
    "config_path": OPENCODE_CONFIG_PATH,
    "backup_path": OPENCODE_BACKUP_PATH,
}

PROVIDER_NAMES = ("databricks-anthropic", "databricks-google", "databricks-oss")
PROVIDER_KEYS: list[list[str]] = [["provider", name] for name in PROVIDER_NAMES]


def is_update_available() -> tuple[str, str] | None:
    return available_npm_package_update(SPEC["package"])


def _resolve_model_selector(model: str, opencode_models: dict[str, list[str]]) -> str:
    """Return an OpenCode model selector in provider/model form when possible."""
    if model.startswith(("databricks-anthropic/", "databricks-google/", "databricks-oss/")):
        return model

    anthropic_models = opencode_models.get("anthropic") or []
    if model in anthropic_models:
        return f"databricks-anthropic/{model}"

    gemini_models = opencode_models.get("gemini") or []
    if model in gemini_models:
        return f"databricks-google/{model}"

    oss_models = opencode_models.get("oss") or []
    if model in oss_models:
        return f"databricks-oss/{model}"

    return model


def _oss_model_overlay(model: str, ua_header: dict[str, str]) -> dict:
    """Per-model overlay for an OSS model entry.

    All OSS models carry the User-Agent header; models with known token limits
    also pin `limit` (context + output) so OpenCode clamps `max_tokens` to a
    value the gateway accepts. OpenCode's schema requires both fields together,
    so the limits table always supplies both."""
    overlay: dict = {"headers": ua_header}
    limits = model_token_limits(model)
    if limits is not None:
        overlay["limit"] = limits
    return overlay


def render_overlay(
    model: str,
    opencode_base_urls: dict[str, str],
    opencode_models: dict[str, list[str]],
) -> tuple[dict, list[list[str]]]:
    """Return (overlay, managed_key_paths) for opencode.json."""
    # OpenCode hardcodes `User-Agent: opencode/<ver>` in session/llm.ts for
    # every provider, after the AI SDK's combineHeaders. The provider-level
    # `headers` are clobbered by that injection, but per-model `headers` are
    # merged AFTER and win — so the UA must live on each model entry.
    ua_header = {
        "User-Agent": f"ucode/{ucode_version()} opencode/{agent_version('opencode')}",
    }

    anthropic_models = opencode_models.get("anthropic") or []
    gemini_models = opencode_models.get("gemini") or []
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
            },
            "models": dict.fromkeys(anthropic_models, anthropic_model_overlay),
        }
        keys.append(["provider", "databricks-anthropic"])
    if gemini_models:
        providers["databricks-google"] = {
            "npm": "@ai-sdk/google",
            "options": {
                "baseURL": opencode_base_urls["gemini"],
            },
            "models": {m: {"headers": ua_header} for m in gemini_models},
        }
        keys.append(["provider", "databricks-google"])
    if oss_models:
        providers["databricks-oss"] = {
            "npm": "@ai-sdk/openai",
            "options": {
                "baseURL": opencode_base_urls["oss"],
            },
            "models": {m: _oss_model_overlay(m, ua_header) for m in oss_models},
        }
        keys.append(["provider", "databricks-oss"])

    overlay: dict = {"model": _resolve_model_selector(model, opencode_models)}
    if providers:
        overlay["provider"] = providers
    return overlay, keys


def render_auth_plugin(auth_argv: list[str]) -> str:
    """Render an OpenCode plugin that obtains a fresh bearer per request."""
    return f"""import {{ execFileSync }} from "node:child_process"

const [command, ...args] = {json.dumps(auth_argv)}
const providers = {json.dumps(PROVIDER_NAMES)}

function token() {{
  const value = execFileSync(command, args, {{ encoding: "utf8" }}).trim()
  if (!value) throw new Error("ucode auth-token returned no access token")
  return value
}}

async function authenticatedFetch(input, init) {{
  const headers = new Headers(init?.headers)
  headers.set("Authorization", `Bearer ${{token()}}`)
  return fetch(input, {{ ...init, headers }})
}}

export const UcodeDatabricksAuth = async () => ({{
  async config(config) {{
    for (const providerName of providers) {{
      const provider = config.provider?.[providerName]
      if (!provider) continue
      provider.options ??= {{}}
      provider.options.apiKey = "databricks-cli"
      provider.options.fetch = authenticatedFetch
    }}
  }},
}})
"""


def write_tool_config(state: dict, model: str) -> dict:
    backup_existing_file(OPENCODE_CONFIG_PATH, OPENCODE_BACKUP_PATH)
    auth_argv = build_auth_token_argv(
        state["workspace"], state.get("profile"), use_pat=bool(state.get("use_pat"))
    )
    write_text_file(OPENCODE_AUTH_PLUGIN_PATH, render_auth_plugin(auth_argv))
    opencode_base_urls = state.get("base_urls", {}).get("opencode") or build_opencode_base_urls(
        state["workspace"]
    )
    overlay, managed_keys = render_overlay(
        model,
        opencode_base_urls,
        state.get("opencode_models") or {},
    )
    existing = read_json_safe(OPENCODE_CONFIG_PATH)
    providers = existing.get("provider")
    if isinstance(providers, dict):
        for stale in (*PROVIDER_NAMES, "databricks-openai"):
            providers.pop(stale, None)
    merged = deep_merge_dict(existing, overlay)
    write_json_file(OPENCODE_CONFIG_PATH, merged)
    state = mark_tool_managed(state, "opencode", managed_keys)
    save_state(state)
    return state


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
    opencode_models = state.get("opencode_models") or {}
    anthropic = opencode_models.get("anthropic") or []
    if anthropic:
        return anthropic[0]
    gemini = opencode_models.get("gemini") or []
    if gemini:
        return gemini[0]
    oss = opencode_models.get("oss") or []
    return oss[0] if oss else None


def build_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env["XDG_CONFIG_HOME"] = str(OPENCODE_XDG_CONFIG_HOME)
    return env


def launch(state: dict, tool_args: list[str]) -> None:
    _ = state
    os.environ["XDG_CONFIG_HOME"] = str(OPENCODE_XDG_CONFIG_HOME)
    exec_or_spawn([SPEC["binary"], *tool_args])


def validate_cmd(binary: str) -> list[str]:
    return [binary, "run", "say hi in 5 words or less"]


def validate_env(state: dict) -> dict[str, str]:
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured.")
    return build_runtime_env()
