"""Claude Code agent: writes ~/.claude/settings.json env block."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import cast

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
    build_auth_shell_command,
    build_tool_base_url,
    get_databricks_token,
)
from ucode.launcher import exec_or_spawn
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version
from ucode.tracing import tracing_env
from ucode.ui import print_note, print_success, print_warning

CLAUDE_CONFIG_DIR = Path.home() / ".claude"
CLAUDE_SETTINGS_PATH = CLAUDE_CONFIG_DIR / "ucode-settings.json"
CLAUDE_BACKUP_PATH = APP_DIR / "claude-ucode-settings.backup.json"

SPEC: ToolSpec = {
    "binary": "claude",
    "package": "@anthropic-ai/claude-code",
    "display": "Claude Code",
    "config_path": CLAUDE_SETTINGS_PATH,
    "backup_path": CLAUDE_BACKUP_PATH,
}


def is_update_available() -> tuple[str, str] | None:
    return available_npm_package_update(SPEC["package"])


def _resolve_web_search_model(state: dict) -> str | None:
    """Pick the model the web_search MCP server should call. Prefers an
    explicit override in state, otherwise the first endpoint discovered as
    Responses-API-capable. Returns None if no GPT endpoint is available —
    callers should skip the MCP wiring in that case."""
    override = state.get("web_search_model")
    if isinstance(override, str) and override.strip():
        return override.strip()
    codex_models = state.get("codex_models") or []
    if isinstance(codex_models, list) and codex_models:
        first = codex_models[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    return None


WEB_SEARCH_MCP_NAME = "web_search"
# Matches both the AI Gateway form (`databricks-claude-opus-4-8`) and the UC
# model-services form (`system.ai.claude-opus-4-8`).
_CLAUDE_MODEL_RE = re.compile(
    r"^(?:system\.ai\.)?(?:databricks-)?claude-(opus|sonnet)-(\d+)-(\d+)(.*)$"
)

# Env keys the MLflow Stop hook reads to route traces. Written into the
# settings `env` block alongside the hook itself.
CLAUDE_TRACING_ENV_KEYS = (
    "MLFLOW_CLAUDE_TRACING_ENABLED",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_ID",
    "MLFLOW_TRACING_SQL_WAREHOUSE_ID",
)
# Model-selection env keys ucode owns end-to-end. Anything in this tuple that
# isn't written by render_overlay gets actively pruned from settings.json on
# every launch, so stale values from older ucode versions never linger.
CLAUDE_MANAGED_MODEL_ENV_KEYS = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL_NAME",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME",
)
# Env keys ucode used to write but no longer does; stripped from the managed
# settings file on every launch so stale values never linger.
CLAUDE_REMOVED_ENV_KEYS = ("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",)
CLAUDE_TRACING_STOP_HOOK_SUFFIX = " autolog claude stop-hook"
# Tracing is driven by an `mlflow autolog claude stop-hook` Stop hook, run by
# the `mlflow` CLI on each session end. Pin to 3.11.x: 3.12 dropped the Unity
# Catalog trace-write path, so traces silently land in the classic store
# instead of the experiment's UC table. ucode installs this via `uv tool` at
# `configure tracing` time (where UV_INDEX_URL is set), then writes the hook
# with the resolved absolute path — so the hook needs no uv or index at run
# time, and can't be shadowed by a project venv's mlflow.
MLFLOW_CLI_SPEC = "mlflow[databricks]>=3.11,<3.12"
MINIMUM_MLFLOW_VERSION = (3, 11)
# Upper bound (exclusive) — an installed mlflow at or above this is too new and
# must be replaced, not just left alone.
MAXIMUM_MLFLOW_VERSION = (3, 12)


def _web_search_mcp_entry(workspace: str, search_model: str, profile: str | None = None) -> dict:
    """Stdio MCP server entry pointing at `ucode mcp web-search`. Resolves
    the absolute path to the `ucode` binary so launchers without the right
    PATH (e.g. desktop GUI launchers) still find it."""
    ucode_binary = shutil.which("ucode") or "ucode"
    env: dict[str, str] = {
        "DATABRICKS_HOST": workspace,
        "UCODE_WEB_SEARCH_MODEL": search_model,
    }
    if profile:
        env["DATABRICKS_CONFIG_PROFILE"] = profile
    return {
        "type": "stdio",
        "command": ucode_binary,
        "args": ["mcp", "web-search"],
        "env": env,
    }


def render_overlay(
    workspace: str,
    model: str | None,
    claude_models: dict[str, str] | None = None,
    disable_web_search: bool = False,
    profile: str | None = None,
    use_pat: bool = False,
    provider: str | None = None,
    provider_models: dict[str, str] | None = None,
    fable_enabled: bool = False,
) -> tuple[dict, list[list[str]]]:
    """Return (overlay, managed_key_paths) for Claude settings.json.

    NOTE: MCP servers are NOT written here. Claude Code reads `mcpServers`
    from `~/.claude.json`, not `~/.claude/settings.json` — registration goes
    through `claude mcp add-json` (see `_register_web_search_mcp`).

    When `provider` is set (a `<catalog>.<schema>.<name>` Model Provider
    Service), the request is routed to that external provider via the
    `Databricks-Model-Provider-Service` header. An Anthropic-backed provider
    understands Claude Code's own canonical model names, so no model id is
    pinned. A Bedrock-backed provider exposes different model ids (e.g.
    `us.anthropic.claude-sonnet-4-6`), passed in `provider_models` by family —
    those get pinned via the `ANTHROPIC_DEFAULT_*_MODEL` env vars."""
    base_url = build_tool_base_url("claude", workspace)
    # ANTHROPIC_CUSTOM_HEADERS is parsed as `key: value` pairs separated by
    # newlines (Anthropic SDK convention). Setting User-Agent here overrides
    # the SDK's default UA on outbound requests so the gateway can attribute
    # traffic to ucode.
    header_lines = [
        "x-databricks-use-coding-agent-mode: true",
        f"User-Agent: ucode/{ucode_version()} claude/{agent_version('claude')}",
    ]
    if provider:
        header_lines.append(f"Databricks-Model-Provider-Service: {provider}")
    custom_headers = "\n".join(header_lines)
    env: dict[str, str] = {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_CUSTOM_HEADERS": custom_headers,
        "CLAUDE_CODE_API_KEY_HELPER_TTL_MS": "900000",
        # 1h prompt caching needs the extended-cache-ttl beta header, which
        # Claude Code only sends when experimental betas are enabled — so we must
        # not set CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS (see CLAUDE_REMOVED_ENV_KEYS).
        "ENABLE_PROMPT_CACHING_1H": "1",
        "ENABLE_TOOL_SEARCH": "1",
        "CLAUDE_CODE_USE_GATEWAY": "1",
    }
    # Intentionally NOT setting ANTHROPIC_MODEL. Setting it produces a duplicate
    # catalog row in Claude Code's /model picker (e.g. "Opus 4.8 (1M context) ✓")
    # on top of the family-alias row from ANTHROPIC_DEFAULT_OPUS_MODEL. Without
    # it, Default resolves through the pinned family alias and the picker shows
    # only one row per model. `ucode claude -- --model X` still overrides for a
    # single session via Claude Code's own --model flag.
    _ = model  # API stability; no longer pinned via env.
    # A Bedrock-backed provider needs its provider-side ids pinned verbatim
    # (Claude Code's canonical names aren't routable there). These come from the
    # service's targets, already de-duped to one id per family upstream.
    if provider and provider_models:
        if provider_models.get("opus"):
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = provider_models["opus"]
        if provider_models.get("sonnet"):
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = provider_models["sonnet"]
        if provider_models.get("haiku"):
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = provider_models["haiku"]
    # With an Anthropic Model Provider Service, the header routes to the external
    # provider and Claude Code's own canonical model names are sent verbatim —
    # pinning a Databricks model id here would mislabel the picker and isn't
    # routable.
    elif claude_models and not provider:
        # Picker rows show the raw routable id (e.g. "system.ai.claude-opus-4-8[1m]")
        # so users can see which gateway-routable model is behind each shortcut.
        # We deliberately don't set the `_NAME` companion env vars — the raw id
        # is more useful than a friendly label for debugging gateway routing.
        #
        # Fable is opt-in only (`ucode configure --enable-fable`): it's a premium
        # model, so we don't pin the family alias unless the user asked for it.
        # When off, ANTHROPIC_DEFAULT_FABLE_MODEL is simply never written — and
        # since it's in CLAUDE_MANAGED_MODEL_ENV_KEYS, any stale value from a
        # prior `--enable-fable` run is pruned from settings.json on next launch.
        if fable_enabled and claude_models.get("fable"):
            env["ANTHROPIC_DEFAULT_FABLE_MODEL"] = claude_models["fable"]
        if claude_models.get("opus"):
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = _maybe_add_1m_suffix(claude_models["opus"])
        if claude_models.get("sonnet"):
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = _maybe_add_1m_suffix(claude_models["sonnet"])
        if claude_models.get("haiku"):
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = claude_models["haiku"]
    overlay: dict = {
        "apiKeyHelper": build_auth_shell_command(workspace, profile, use_pat=use_pat),
        "env": env,
    }
    keys: list[list[str]] = [["apiKeyHelper"]] + [["env", k] for k in env]

    # Disable Claude Code's built-in WebSearch: it declares Anthropic's hosted
    # `web_search_20250305` server tool, which the Databricks gateway rejects
    # (HTTP 400: "Input tag 'web_search_20250305' ... does not match"), so the
    # model wastes a turn on it before falling back. A *bare* `permissions.deny`
    # entry removes the tool from Claude's context entirely, so it is never
    # advertised to the model nor sent to the gateway. (Claude Code has no
    # `disabledTools` setting — the `permissions` block is the only settings.json
    # mechanism for built-in tools; a bare tool name in `deny` drops it, whereas
    # a scoped rule like `WebSearch(*)` would leave it advertised.) The
    # replacement `web_search` MCP server is registered separately via the
    # claude CLI.
    if disable_web_search:
        overlay["permissions"] = {"deny": ["WebSearch"]}
        keys.append(["permissions", "deny"])

    return overlay, keys


def _maybe_add_1m_suffix(model: str) -> str:
    if model.endswith("[1m]"):
        return model
    match = _CLAUDE_MODEL_RE.match(model)
    if not match:
        return model

    family, major_raw, minor_raw, _ = match.groups()
    major = int(major_raw)
    minor = int(minor_raw)
    should_suffix = (family == "opus" and (major, minor) >= (4, 6)) or (
        family == "sonnet" and (major, minor) >= (4, 6)
    )
    return f"{model}[1m]" if should_suffix else model


def _register_web_search_mcp(workspace: str, search_model: str, profile: str | None = None) -> bool:
    """Register (or replace) the web_search MCP server in Claude Code's user
    scope via `claude mcp add-json`. Removes any prior entry first so re-runs
    pick up changes to the workspace, model, or ucode binary path.

    Returns True if registration succeeded. Failures are non-blocking: we warn
    and return False so the rest of `ucode claude` setup can complete.
    """
    # Imported lazily to avoid a circular import via ucode.mcp -> ucode.agents.
    from ucode.mcp import (
        MCP_CLEANUP_SCOPES,
        add_claude_mcp_server,
        remove_claude_mcp_server,
    )

    for scope in MCP_CLEANUP_SCOPES:
        try:
            remove_claude_mcp_server(WEB_SEARCH_MCP_NAME, scope)
        except RuntimeError:
            # Best-effort cleanup of stale entries — keep going.
            pass
    entry = _web_search_mcp_entry(workspace, search_model, profile)
    try:
        add_claude_mcp_server(WEB_SEARCH_MCP_NAME, entry)
    except RuntimeError as exc:
        print_warning(f"{exc} Web search will be unavailable; re-run `ucode claude` to retry.")
        return False
    return True


def _unregister_web_search_mcp() -> None:
    """Remove the web_search MCP server from all scopes. Used by revert."""
    from ucode.mcp import MCP_CLEANUP_SCOPES, remove_claude_mcp_server

    for scope in MCP_CLEANUP_SCOPES:
        try:
            remove_claude_mcp_server(WEB_SEARCH_MCP_NAME, scope)
        except RuntimeError:
            pass


def write_tool_config(
    state: dict,
    model: str | None,
    provider: str | None = None,
    provider_models: dict[str, str] | None = None,
) -> dict:
    backup_existing_file(CLAUDE_SETTINGS_PATH, CLAUDE_BACKUP_PATH)
    web_search_model = _resolve_web_search_model(state)
    overlay, managed_keys = render_overlay(
        state["workspace"],
        model,
        state.get("claude_models") or {},
        disable_web_search=web_search_model is not None,
        profile=state.get("profile"),
        use_pat=bool(state.get("use_pat")),
        provider=provider,
        provider_models=provider_models,
        fable_enabled=bool(state.get("fable_enabled")),
    )
    tracing_env_vars = tracing_env(state, "claude")
    stop_hook_command = claude_tracing_stop_hook_command() if tracing_env_vars else None
    if tracing_env_vars:
        overlay["env"]["MLFLOW_CLAUDE_TRACING_ENABLED"] = "true"
        overlay["env"].update(tracing_env_vars)
        managed_keys = managed_keys + [["env", key] for key in CLAUDE_TRACING_ENV_KEYS]
        if stop_hook_command:
            managed_keys = managed_keys + [["hooks", "Stop"]]
        else:
            print_warning(
                "MLflow tracing env was written, but the `mlflow` CLI could not be located "
                "to install the Claude Stop hook — traces won't be emitted. Re-run "
                "`ucode configure tracing`."
            )

    existing = read_json_safe(CLAUDE_SETTINGS_PATH)
    merged = deep_merge_dict(existing, overlay)
    if tracing_env_vars and stop_hook_command:
        _upsert_tracing_stop_hook(merged, stop_hook_command)
    if not tracing_env_vars:
        env_block = merged.get("env")
        if isinstance(env_block, dict):
            for key in CLAUDE_TRACING_ENV_KEYS:
                env_block.pop(key, None)
        # Strip only ucode's tracing Stop hook so user hooks stay intact.
        _remove_tracing_stop_hook(merged)
    # Prune ucode-managed model env keys we deliberately don't write this run
    # (e.g. ANTHROPIC_MODEL — see render_overlay).
    overlay_env = overlay.get("env", {})
    merged_env = merged.get("env")
    if isinstance(merged_env, dict):
        for key in CLAUDE_MANAGED_MODEL_ENV_KEYS:
            if key not in overlay_env:
                merged_env.pop(key, None)
    # deep_merge_dict keeps keys already in the file, so drop the ones ucode no
    # longer writes.
    if isinstance(merged_env, dict):
        for key in CLAUDE_REMOVED_ENV_KEYS:
            merged_env.pop(key, None)
    write_json_file(CLAUDE_SETTINGS_PATH, merged)

    if web_search_model:
        _register_web_search_mcp(state["workspace"], web_search_model, state.get("profile"))

    state = mark_tool_managed(state, "claude", managed_keys)
    save_state(state)
    return state


def _is_tracing_stop_hook(hook: object) -> bool:
    if not isinstance(hook, dict):
        return False
    hook = cast(dict, hook)
    if hook.get("type") != "command":
        return False
    command = hook.get("command")
    return isinstance(command, str) and command.endswith(CLAUDE_TRACING_STOP_HOOK_SUFFIX)


def _remove_tracing_stop_hook(settings: dict) -> None:
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return
    stop_entries = hooks.get("Stop")
    if not isinstance(stop_entries, list):
        return

    cleaned_entries = []
    for entry in stop_entries:
        if not isinstance(entry, dict):
            cleaned_entries.append(entry)
            continue
        hook_list = entry.get("hooks")
        if not isinstance(hook_list, list):
            cleaned_entries.append(entry)
            continue
        cleaned_hooks = [hook for hook in hook_list if not _is_tracing_stop_hook(hook)]
        if cleaned_hooks:
            cleaned_entry = dict(entry)
            cleaned_entry["hooks"] = cleaned_hooks
            cleaned_entries.append(cleaned_entry)

    if cleaned_entries:
        hooks["Stop"] = cleaned_entries
    else:
        hooks.pop("Stop", None)
    if not hooks:
        settings.pop("hooks", None)


def _upsert_tracing_stop_hook(settings: dict, command: str) -> None:
    _remove_tracing_stop_hook(settings)
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}
        settings["hooks"] = hooks
    stop_entries = hooks.get("Stop")
    if not isinstance(stop_entries, list):
        stop_entries = []
        hooks["Stop"] = stop_entries
    stop_entries.append({"hooks": [{"type": "command", "command": command}]})


def ensure_tracing_runtime() -> bool:
    """Ensure the MLflow tracing runtime is ready: a pinned `mlflow` CLI (3.11.x)
    installed via `uv tool`, whose absolute path the Stop hook will call.

    Best-effort — warns and returns False if it can't be set up, so
    `ucode configure tracing` can still finish for other agents."""
    return _ensure_mlflow_cli()


def _parse_mlflow_version(text: str) -> tuple[int, int] | None:
    match = re.search(r"(\d+)\.(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _uv_tool_mlflow_path() -> str | None:
    """Absolute path to the `mlflow` installed by `uv tool`, or None.

    Resolved from `uv tool dir --bin` rather than ``shutil.which`` so a project
    venv's (possibly wrong-versioned) mlflow can't shadow the one ucode pins —
    the Stop hook must always run the uv-tool copy."""
    if not shutil.which("uv"):
        return None
    try:
        result = subprocess.run(
            ["uv", "tool", "dir", "--bin"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    bin_dir = (result.stdout or "").strip()
    if result.returncode != 0 or not bin_dir:
        return None
    candidate = Path(bin_dir) / "mlflow"
    return str(candidate) if candidate.exists() else None


def _installed_mlflow_version() -> tuple[int, int] | None:
    """The (major, minor) of the uv-tool `mlflow`, or None if absent."""
    path = _uv_tool_mlflow_path()
    if not path:
        return None
    try:
        result = subprocess.run(
            [path, "--version"], check=False, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _parse_mlflow_version(result.stdout or result.stderr or "")


def claude_tracing_stop_hook_command() -> str | None:
    """The Stop hook command string: the absolute uv-tool `mlflow` invoking its
    `autolog claude stop-hook` handler. None when mlflow isn't installed.

    Using the absolute path means the hook needs neither `uv` nor a package
    index at run time (the minimal env Claude runs hooks in lacks UV_INDEX_URL),
    and can't be shadowed by another mlflow on PATH."""
    path = _uv_tool_mlflow_path()
    if not path:
        return None
    return f"{path} autolog claude stop-hook"


def _ensure_mlflow_cli() -> bool:
    """Ensure the pinned `mlflow` CLI (3.11.x) is installed via `uv tool`,
    installing or replacing an out-of-range version when needed."""
    current = _installed_mlflow_version()
    if current and MINIMUM_MLFLOW_VERSION <= current < MAXIMUM_MLFLOW_VERSION:
        return True

    if not shutil.which("uv"):
        verb = "replace" if current else "install"
        print_warning(
            f"Claude tracing needs the `mlflow` CLI ({MLFLOW_CLI_SPEC}), but `uv` is not "
            f'available to {verb} it. Run `uv tool install "{MLFLOW_CLI_SPEC}"`, then '
            "re-run `ucode configure tracing`."
        )
        return False

    print_note(f"{'Replacing' if current else 'Installing'} the mlflow CLI ({MLFLOW_CLI_SPEC})...")
    # Always --force: it installs fresh when absent and replaces in place when
    # present. Keying it on `current` broke when an mlflow existed but its
    # version couldn't be parsed — uv still errors "Executable already exists".
    cmd = ["uv", "tool", "install", "--force", MLFLOW_CLI_SPEC]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print_warning(f"Could not install the mlflow CLI automatically: {exc}")
        return False

    if not _uv_tool_mlflow_path():
        print_warning(
            "Installed mlflow via `uv tool`, but its binary could not be located. "
            "Re-run `ucode configure tracing`."
        )
        return False
    print_success("mlflow CLI ready")
    return True


def default_model(state: dict) -> str | None:
    claude_models = state.get("claude_models") or {}
    return claude_models.get("opus") or claude_models.get("sonnet") or claude_models.get("haiku")


def _extract_caller_settings(tool_args: list[str]) -> tuple[list[str], list[str]]:
    """Split caller-supplied ``--settings`` values out of *tool_args*.

    Returns ``(values, remaining_args)``, handling both ``--settings <value>``
    and ``--settings=<value>`` spellings. Each value is either a JSON string or
    a path to a settings file — Claude Code accepts either.
    """
    values: list[str] = []
    remaining: list[str] = []
    i = 0
    while i < len(tool_args):
        arg = tool_args[i]
        if arg == "--settings" and i + 1 < len(tool_args):
            values.append(tool_args[i + 1])
            i += 2
            continue
        if arg.startswith("--settings="):
            values.append(arg[len("--settings=") :])
            i += 1
            continue
        remaining.append(arg)
        i += 1
    return values, remaining


def _load_caller_settings(value: str) -> dict:
    """Resolve a ``--settings`` value (inline JSON or file path) to a dict.

    Claude Code accepts either inline JSON or a path to a JSON file. Raises
    ``RuntimeError`` (surfaced by the CLI as an actionable error) when the value
    is neither, rather than silently dropping it: a dropped value would also be
    passed through as a second ``--settings`` flag, and Claude Code honors only
    one — so either the caller's settings or ucode's gateway config would be
    silently ignored. Failing loudly lets the caller fix their input.
    """
    text = value.strip()
    if text.startswith("{"):
        source, malformed = text, "value is not valid JSON"
    else:
        path = Path(text)
        if not path.exists():
            raise RuntimeError(
                f"--settings file not found: {value!r}. "
                "Pass inline JSON or a path to an existing JSON file."
            )
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"--settings file could not be read: {value!r} ({exc}).") from exc
        malformed = "file is not valid JSON"
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"--settings {malformed} ({exc}): {value!r}. Pass inline JSON or a path to a JSON file."
        ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"--settings must be a JSON object, got {type(parsed).__name__}: {value!r}."
        )
    return parsed


def _union_claude_hooks(base: dict, overlay: dict) -> dict:
    """Union two Claude Code ``hooks`` maps.

    ``hooks`` is ``{event: [entry, ...]}``. Per event we concatenate the entry
    lists so hooks from BOTH settings sources fire, rather than one replacing
    the other (which is what a plain deep-merge does to lists). This is what
    lets ucode's tracing Stop hook and a caller's own hooks coexist.
    """
    result: dict = {}
    for event in [*base, *(e for e in overlay if e not in base)]:
        entries: list = []
        for src in (base, overlay):
            val = src.get(event)
            if isinstance(val, list):
                entries.extend(val)
        result[event] = entries
    return result


def _merge_claude_settings(base: dict, overlay: dict) -> dict:
    """Deep-merge *overlay* onto *base* (overlay wins on conflicting leaves),
    but UNION the ``hooks`` so neither side's hooks are dropped. Inputs are not
    mutated.
    """
    merged = deep_merge_dict(copy.deepcopy(base), overlay)
    base_hooks = base.get("hooks")
    overlay_hooks = overlay.get("hooks")
    if isinstance(base_hooks, dict) or isinstance(overlay_hooks, dict):
        merged["hooks"] = _union_claude_hooks(
            base_hooks if isinstance(base_hooks, dict) else {},
            overlay_hooks if isinstance(overlay_hooks, dict) else {},
        )
    return merged


def _build_claude_argv(binary: str, tool_args: list[str]) -> list[str]:
    """Build the ``claude`` argv, composing any caller ``--settings`` with
    ucode's managed settings.

    ucode needs its own settings (gateway ``apiKeyHelper`` + env) to reach
    Claude, and normally passes ``--settings <ucode-file>``. But Claude Code
    honors only ONE ``--settings`` flag, so a caller that ALSO passes
    ``--settings`` (e.g. an integration injecting hooks) would have exactly one
    of the two silently dropped. To let ucode compose with any prior command,
    we merge a caller-supplied ``--settings`` with ucode's — ucode's gateway
    keys win, hooks from both are unioned — and hand Claude a single merged
    ``--settings`` (inline JSON). The merge is per-launch and is never written
    back to the shared ucode settings file, so concurrent launches cannot
    accumulate one another's hooks. A caller ``--settings`` value ucode cannot
    resolve raises (see :func:`_load_caller_settings`) rather than being passed
    through as a second, colliding flag.
    """
    caller_values, remaining = _extract_caller_settings(tool_args)
    if not caller_values:
        # No caller --settings: hand Claude ucode's settings file directly (the
        # common path; behavior unchanged).
        return [binary, "--settings", str(CLAUDE_SETTINGS_PATH), *tool_args]
    caller_settings: dict = {}
    for value in caller_values:
        caller_settings = _merge_claude_settings(caller_settings, _load_caller_settings(value))
    # ucode wins over the caller for conflicting keys (protects gateway auth);
    # hooks from both sides survive.
    merged = _merge_claude_settings(caller_settings, read_json_safe(CLAUDE_SETTINGS_PATH))
    return [binary, "--settings", json.dumps(merged, separators=(",", ":")), *remaining]


def launch(state: dict, tool_args: list[str]) -> None:
    binary = SPEC["binary"]
    workspace = state.get("workspace")
    if workspace:
        os.environ["OAUTH_TOKEN"] = get_databricks_token(workspace, state.get("profile"))
    exec_or_spawn(_build_claude_argv(binary, tool_args))


def validate_cmd(binary: str) -> list[str]:
    return [
        binary,
        "--settings",
        str(CLAUDE_SETTINGS_PATH),
        "-p",
        "say hi in 5 words or less",
        "--max-turns",
        "1",
    ]
