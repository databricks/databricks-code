"""Persistent state for ucode (per-workspace, versioned)."""

from __future__ import annotations

import json

from ucode.config_io import APP_DIR, is_dry_run
from ucode.databricks import (
    build_auth_shell_command,
    build_auth_token_argv,
    build_shared_base_urls,
)

STATE_PATH = APP_DIR / "state.json"
STATE_VERSION = 3
AUTH_COMMAND_TIMEOUT_MS = 5000
AUTH_REFRESH_INTERVAL_MS = 900_000


def load_full_state() -> dict:
    """Load the entire state file. Returns empty structure if missing or wrong version."""
    if not STATE_PATH.exists():
        return {"state_version": STATE_VERSION, "current_workspace": None, "workspaces": {}}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state_version": STATE_VERSION, "current_workspace": None, "workspaces": {}}
    if not isinstance(data, dict) or data.get("state_version") != STATE_VERSION:
        return {"state_version": STATE_VERSION, "current_workspace": None, "workspaces": {}}
    return data


def load_state() -> dict:
    """Load the current workspace's state as a flat dict."""
    full = load_full_state()
    workspace = full.get("current_workspace")
    if not workspace:
        return {}
    ws_state = full.get("workspaces", {}).get(workspace, {})
    ws_state["workspace"] = workspace
    return hydrate_state(ws_state)


def save_state(state: dict) -> None:
    """Save workspace state back into the per-workspace structure."""
    if is_dry_run():
        return
    full = load_full_state()
    workspace = state.get("workspace") or full.get("current_workspace")
    if workspace:
        full["current_workspace"] = workspace
        full["workspaces"][workspace] = hydrate_state(state)
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(full, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write state file: {STATE_PATH}") from exc


def set_current_workspace(workspace: str | None) -> None:
    """Set ``current_workspace`` without touching the per-workspace blocks.

    Used by flows like ``configure tracing`` that operate on a non-current
    workspace and must not silently change which workspace ``ucode launch``
    targets afterwards."""
    if is_dry_run():
        return
    full = load_full_state()
    if full.get("current_workspace") == workspace:
        return
    full["current_workspace"] = workspace
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(full, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write state file: {STATE_PATH}") from exc


def hydrate_state(state: dict) -> dict:
    """Normalize a workspace state entry and add derived harness config.

    :param state: Raw workspace state entry from ``state.json``.
    :returns: Hydrated workspace state with stable ``managed_configs``,
        ``base_urls``, and per-agent ``agents`` entries.
    """
    if not isinstance(state, dict):
        return {}

    hydrated = dict(state)
    managed_configs = hydrated.get("managed_configs")
    if not isinstance(managed_configs, dict):
        managed_configs = {}
    normalized: dict[str, dict] = {}
    for tool, entry in managed_configs.items():
        if isinstance(entry, dict):
            keys = entry.get("keys") if isinstance(entry.get("keys"), list) else []
            normalized[tool] = {"keys": keys}
        elif entry:
            normalized[tool] = {"keys": []}
    hydrated["managed_configs"] = normalized

    workspace = hydrated.get("workspace")
    if workspace:
        hydrated["base_urls"] = build_shared_base_urls(workspace)
        hydrated["agents"] = build_agent_state(hydrated)
    else:
        hydrated["base_urls"] = {}
        hydrated["agents"] = {}

    return hydrated


def build_agent_state(state: dict) -> dict[str, dict]:
    """Build per-agent harness configuration for a workspace.

    The returned shape is intended for downstream tools that want to reuse
    ucode's configured gateway URLs and auth command without duplicating
    endpoint construction logic.

    :param state: Hydrated workspace state containing ``workspace``,
        ``base_urls``, and discovered model lists.
    :returns: Mapping from agent name to its reusable configuration.
    """
    workspace = state.get("workspace")
    if not isinstance(workspace, str) or not workspace:
        return {}

    profile = state.get("profile") if isinstance(state.get("profile"), str) else None
    base_urls_value = state.get("base_urls")
    base_urls = base_urls_value if isinstance(base_urls_value, dict) else {}
    use_pat = bool(state.get("use_pat"))
    auth_command = build_auth_shell_command(workspace, profile, use_pat=use_pat)
    auth_argv = build_auth_token_argv(workspace, profile, use_pat=use_pat)
    claude_models_value = state.get("claude_models")
    claude_models: dict = claude_models_value if isinstance(claude_models_value, dict) else {}
    codex_models_value = state.get("codex_models")
    codex_models = codex_models_value if isinstance(codex_models_value, list) else []
    gemini_models_value = state.get("gemini_models")
    gemini_models = gemini_models_value if isinstance(gemini_models_value, list) else []

    claude_model = (
        claude_models.get("opus") or claude_models.get("sonnet") or claude_models.get("haiku")
    )
    codex_model = codex_models[0] if codex_models else None
    pi_model = claude_model or codex_model or (gemini_models[0] if gemini_models else None)

    agents: dict[str, dict] = {
        "claude": {
            "model": claude_model,
            "base_url": base_urls.get("claude"),
            "auth_command": auth_command,
            "auth_refresh_interval_ms": AUTH_REFRESH_INTERVAL_MS,
            "env": {
                "ANTHROPIC_BASE_URL": base_urls.get("claude"),
                "CLAUDE_CODE_API_KEY_HELPER_TTL_MS": str(AUTH_REFRESH_INTERVAL_MS),
                # Kept in sync with agents/claude.py render_overlay.
                "ENABLE_PROMPT_CACHING_1H": "1",
                "ENABLE_TOOL_SEARCH": "1",
                "CLAUDE_CODE_USE_GATEWAY": "1",
            },
        },
        "codex": {
            "model": codex_model,
            "base_url": base_urls.get("codex"),
            "auth_command": auth_command,
            "auth": {
                "command": auth_argv[0],
                "args": auth_argv[1:],
                "timeout_ms": AUTH_COMMAND_TIMEOUT_MS,
                "refresh_interval_ms": AUTH_REFRESH_INTERVAL_MS,
            },
        },
        "pi": {
            "model": pi_model,
            "base_urls": base_urls.get("pi") if isinstance(base_urls.get("pi"), dict) else {},
            "auth_command": auth_command,
            "auth_refresh_interval_ms": AUTH_REFRESH_INTERVAL_MS,
        },
    }
    return {
        name: {key: value for key, value in config.items() if value is not None}
        for name, config in agents.items()
    }


def clear_state() -> None:
    """Remove the current workspace entry from state."""
    full = load_full_state()
    workspace = full.get("current_workspace")
    if workspace:
        full.get("workspaces", {}).pop(workspace, None)
        full["current_workspace"] = None
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(full, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to clear state file: {STATE_PATH}") from exc


def mark_tool_managed(state: dict, tool: str, managed_keys: list) -> dict:
    managed_configs = dict(state.get("managed_configs") or {})
    managed_configs[tool] = {"keys": list(managed_keys)}
    state["managed_configs"] = managed_configs
    state["last_tool"] = tool
    return state


def get_provider_service(state: dict, tool: str) -> str | None:
    """Return the persisted Model Provider Service for ``tool``, if any.

    Launches route through this provider (skipping Databricks model pinning)
    unless overridden by an explicit ``--provider`` flag.
    """
    providers = state.get("provider_services")
    if not isinstance(providers, dict):
        return None
    name = providers.get(tool)
    return name if isinstance(name, str) and name else None


def set_provider_service(state: dict, tool: str, full_name: str | None) -> dict:
    """Persist (or clear, when ``full_name`` is None) ``tool``'s provider service."""
    providers = dict(state.get("provider_services") or {})
    if full_name:
        providers[tool] = full_name
    else:
        providers.pop(tool, None)
    if providers:
        state["provider_services"] = providers
    else:
        state.pop("provider_services", None)
    return state
