"""Continue agent: writes ~/.continue/config.yaml with a Databricks-backed model.

Continue.dev's `cn` CLI and its VS Code/JetBrains extensions read the same
`~/.continue/config.yaml`, so the file ucode writes configures both. We point
Continue's OpenAI-compatible provider at the Databricks MLflow chat-completions
gateway (the same endpoint Copilot uses), which serves Claude and codex (gpt-5)
models behind one URL. `provider: openai` makes Continue append
`/chat/completions` to the configured `apiBase`.

The gateway bearer token is baked into the config file (Continue has no
command-based auth refresh), so — like OpenCode/Copilot/Gemini — the token is
short-lived: `ucode continue` rewrites a fresh one on every launch and a
background thread refreshes it during the session. This is why Continue is not
in `GLOBAL_SETTINGS_AGENTS`: a bare `cn` launched outside ucode would eventually
see an expired token.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from pathlib import Path

from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    read_yaml_safe,
    write_yaml_file,
)
from ucode.databricks import (
    TOKEN_REFRESH_INTERVAL_SECONDS,
    build_continue_base_url,
    get_databricks_token,
)
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version

CONTINUE_CONFIG_DIR = Path.home() / ".continue"
CONTINUE_CONFIG_PATH = CONTINUE_CONFIG_DIR / "config.yaml"
CONTINUE_BACKUP_PATH = APP_DIR / "continue-ucode-config.backup.yaml"

# ucode's model entries in the shared `models:` list carry this name prefix so a
# rewrite can drop the stale ones without disturbing models the user added.
UCODE_MODEL_NAME_PREFIX = "Databricks (ucode)"

SPEC: ToolSpec = {
    "binary": "cn",
    "package": "@continuedev/cli",
    "display": "Continue",
    "config_path": CONTINUE_CONFIG_PATH,
    "backup_path": CONTINUE_BACKUP_PATH,
}

# Informational: Continue's config is a list-shaped YAML document, so revert
# restores the backup (or deletes the ucode-created file) rather than pruning
# key paths. Recorded so `restore_file` sees the tool as managed.
MANAGED_KEYS: list[list[str]] = [["models"]]


def default_model(state: dict) -> str | None:
    """Pick the best available Continue model.

    A managed config's ``continue_default_model`` wins outright. Otherwise prefer
    Claude sonnet, then opus/haiku, then the first codex model — the same order
    Copilot uses, since both draw from the shared chat-completions gateway.
    """
    if isinstance(state.get("continue_default_model"), str):
        return state.get("continue_default_model")
    claude_models = state.get("claude_models") or {}
    for family in ("sonnet", "opus", "haiku"):
        if claude_models.get(family):
            return claude_models[family]
    codex_models = state.get("codex_models") or []
    if codex_models:
        return codex_models[0]
    return None


def _ucode_model_entry(model: str, token: str, workspace: str) -> dict:
    """Build the ucode-managed `models:` entry for chat/edit/apply roles."""
    return {
        "name": f"{UCODE_MODEL_NAME_PREFIX} {model}",
        "provider": "openai",
        "model": model,
        "apiBase": build_continue_base_url(workspace),
        "apiKey": token,
        "roles": ["chat", "edit", "apply"],
        "requestOptions": {
            "headers": {
                "User-Agent": f"ucode/{ucode_version()} continue/{agent_version('cn')}",
            },
        },
    }


def _ensure_schema_header(doc: dict) -> None:
    """Set Continue's required top-level keys, keeping the user's if present."""
    doc.setdefault("name", "ucode")
    doc.setdefault("version", "0.0.1")
    doc.setdefault("schema", "v1")


def render_config(model: str, token: str, workspace: str) -> dict:
    """Return a complete Continue config document pinning the ucode model."""
    return {
        "name": "ucode",
        "version": "0.0.1",
        "schema": "v1",
        "models": [_ucode_model_entry(model, token, workspace)],
    }


def write_tool_config(
    state: dict,
    model: str,
    token: str | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[dict, str]:
    backup_existing_file(CONTINUE_CONFIG_PATH, CONTINUE_BACKUP_PATH)
    if token is None:
        token = get_databricks_token(
            state["workspace"], state.get("profile"), force_refresh=force_refresh
        )
    existing = read_yaml_safe(CONTINUE_CONFIG_PATH)
    _ensure_schema_header(existing)
    models = existing.get("models")
    if not isinstance(models, list):
        models = []
    # Drop any prior ucode entry so a rewrite replaces (not duplicates) it, while
    # leaving the user's own models untouched.
    models = [
        m
        for m in models
        if not (isinstance(m, dict) and str(m.get("name", "")).startswith(UCODE_MODEL_NAME_PREFIX))
    ]
    models.append(_ucode_model_entry(model, token, state["workspace"]))
    existing["models"] = models
    write_yaml_file(CONTINUE_CONFIG_PATH, existing)
    state = mark_tool_managed(state, "continue", MANAGED_KEYS)
    save_state(state)
    return state, token


def build_mcp_server_entry(name: str, argv: list[str]) -> dict:
    # A local stdio MCP server: `command`/`args` run the `ucode mcp-proxy ...`
    # bridge, which mints a fresh OAuth token per request — so Continue never
    # speaks HTTP+bearer directly (same proxy pattern as Cursor/OpenCode).
    # Continue's `mcpServers` is a list, so each entry carries its own `name`.
    return {
        "name": name,
        "type": "stdio",
        "command": argv[0],
        "args": list(argv[1:]),
    }


def write_mcp_server_config(name: str, argv: list[str]) -> bool:
    backup_existing_file(CONTINUE_CONFIG_PATH, CONTINUE_BACKUP_PATH)
    existing = read_yaml_safe(CONTINUE_CONFIG_PATH)
    _ensure_schema_header(existing)
    servers = existing.get("mcpServers")
    if not isinstance(servers, list):
        servers = []
    # `mcpServers` is a list keyed by `name`; drop any prior entry with this name
    # so a re-register replaces it, leaving the user's own servers untouched.
    removed = any(isinstance(s, dict) and s.get("name") == name for s in servers)
    servers = [s for s in servers if not (isinstance(s, dict) and s.get("name") == name)]
    servers.append(build_mcp_server_entry(name, argv))
    existing["mcpServers"] = servers
    write_yaml_file(CONTINUE_CONFIG_PATH, existing)
    return removed


def remove_mcp_server_config(name: str) -> bool:
    existing = read_yaml_safe(CONTINUE_CONFIG_PATH)
    servers = existing.get("mcpServers")
    if not isinstance(servers, list):
        return False
    filtered = [s for s in servers if not (isinstance(s, dict) and s.get("name") == name)]
    if len(filtered) == len(servers):
        return False
    existing["mcpServers"] = filtered
    write_yaml_file(CONTINUE_CONFIG_PATH, existing)
    return True


def _refresh_token_once(state: dict, *, force_refresh: bool = False) -> str:
    model = default_model(state)
    if not model:
        raise RuntimeError("No Continue model is available on this workspace.")
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
    return env


def launch(state: dict, tool_args: list[str]) -> None:
    """Launch `cn` with background token refresh (same pattern as OpenCode)."""
    token = _refresh_token_once(state)
    env = build_runtime_env(token)

    stop_event = threading.Event()
    refresher = threading.Thread(
        target=_refresh_forever,
        args=(state, stop_event),
        daemon=True,
    )
    refresher.start()

    # `--config` forces ucode's config over any hub assistant the user last
    # selected, so `ucode continue` always routes through the Databricks gateway.
    argv = [SPEC["binary"], "--config", str(CONTINUE_CONFIG_PATH), *tool_args]
    proc = subprocess.Popen(argv, env=env)
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
    # `-p` runs headless (TTY-less); `--config` pins ucode's config for the probe.
    return [
        binary,
        "-p",
        "say hi in 5 words or less",
        "--config",
        str(CONTINUE_CONFIG_PATH),
    ]
