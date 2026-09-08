"""MLflow tracing: route Claude Code sessions to a Databricks experiment.

ucode points Claude Code's MLflow integration at a single, pre-provisioned
experiment named ``ucode-traces`` whose traces are stored in a Unity Catalog
table. ucode asserts the experiment exists and is UC-backed (it does not create
it), resolves a SQL warehouse for UC writes, and persists the tracking URI,
experiment id, and warehouse id. Auth reuses the workspace profile ucode
already configures in ``~/.databrickscfg``.

Scope: Claude Code only. Its `mlflow autolog claude` Stop hook writes traces to
the UC table. Codex and OpenCode used the `@mlflow/codex`/`@mlflow/opencode` JS
clients, which only reach the classic (non-UC) trace store, so their tracing was
removed. Gemini's exporter is OTLP-only and was never wired here.

This module must not import ``mlflow`` (the heavy optional dependency) or
``ucode.agents`` at import time: agents import the small helpers here, and the
configure command imports agents lazily to avoid a cycle.
"""

from __future__ import annotations

from ucode.databricks import (
    apply_pat_environment,
    ensure_databricks_auth,
    find_uc_backed_experiment,
    get_databricks_token,
    resolve_sql_warehouse_id,
)
from ucode.state import hydrate_state, load_full_state, save_state, set_current_workspace
from ucode.ui import (
    print_kv,
    print_note,
    print_section,
    print_success,
    prompt_for_workspace,
    spinner,
)

# Agents whose MLflow integration routes to a Databricks tracking URI. Only
# Claude Code is supported: its `mlflow autolog claude` Stop hook writes traces
# to the experiment's Unity Catalog table. Codex and OpenCode used the
# `@mlflow/codex`/`@mlflow/opencode` JS clients, which only reach the classic
# (non-UC) trace store, so their tracing was removed.
TRACING_AGENTS: tuple[str, ...] = ("claude",)

# Leaf name of the experiment every agent and every user routes to. ucode does
# not create it: an admin provisions an experiment with this name whose traces
# are backed by Unity Catalog, and ucode asserts it exists. The full path can be
# anything (e.g. `/Users/<email>/ucode-traces` or `/Shared/ucode-traces`) — only
# the final segment must match, and the traces must land in a UC table.
EXPERIMENT_LEAF_NAME = "ucode-traces"


def tracking_uri_for_state(state: dict) -> str:
    """The MLflow tracking URI for this workspace. ``databricks://<profile>``
    selects the matching ``~/.databrickscfg`` entry; bare ``databricks`` uses
    the default profile."""
    profile = state.get("profile")
    return f"databricks://{profile}" if profile else "databricks"


def experiment_name() -> str:
    """The leaf name of the shared, UC-backed experiment ucode requires."""
    return EXPERIMENT_LEAF_NAME


def _missing_experiment_error(name: str, reason: str | None) -> str:
    """Actionable error when no UC-backed ``ucode-traces`` experiment is found.
    ucode no longer creates the experiment, so the fix is for an admin to set
    one up with Unity Catalog trace storage."""
    detail = f" ({reason})" if reason else ""
    return (
        f"No Unity Catalog-backed MLflow experiment named '{name}' was found on this "
        f"workspace{detail}.\n"
        "ucode does not create it — an admin must provision one whose traces are "
        "stored in Unity Catalog:\n"
        f"  1. In the workspace, create an MLflow experiment named '{name}'.\n"
        "  2. Configure its trace storage to a Unity Catalog table "
        "(any catalog.schema.table you have access to).\n"
        "  3. Re-run `ucode configure tracing`."
    )


def _missing_warehouse_error(reason: str | None) -> str:
    """Actionable error when no SQL warehouse can back UC trace writes. The
    warehouse is mandatory: traces to a UC table are silently dropped without
    ``MLFLOW_TRACING_SQL_WAREHOUSE_ID``."""
    detail = f" ({reason})" if reason else ""
    return (
        f"No SQL warehouse is available to write traces to Unity Catalog{detail}.\n"
        "Writing traces to a UC-backed experiment requires a SQL warehouse:\n"
        "  1. Create a SQL warehouse in the workspace (SQL > SQL Warehouses).\n"
        "  2. Re-run `ucode configure tracing`."
    )


def tracing_config(state: dict) -> dict | None:
    """Return the persisted tracing block iff tracing is enabled."""
    cfg = state.get("tracing")
    if isinstance(cfg, dict) and cfg.get("enabled"):
        return cfg
    return None


def agent_tracing(state: dict, tool: str) -> dict | None:
    """The resolved shared tracing entry ({experiment_id, experiment_name}) when
    ``tool`` should be traced, else None. All tracing-capable agents share one
    experiment, so this returns the same entry for each; it still gates per-tool
    (Gemini/Copilot/Pi never trace) and on the experiment having resolved."""
    cfg = tracing_config(state)
    if not cfg or tool not in TRACING_AGENTS:
        return None
    if not cfg.get("experiment_id"):
        return None
    return {
        "experiment_id": cfg["experiment_id"],
        "experiment_name": cfg.get("experiment_name"),
    }


def tracing_env(state: dict, tool: str) -> dict[str, str]:
    """MLflow env vars for one agent. Empty when tracing is disabled for it. The
    tracking URI carries the profile, so auth resolves from ``~/.databrickscfg``
    without extra vars.

    ``MLFLOW_TRACING_SQL_WAREHOUSE_ID`` is required for the experiment's
    Unity Catalog trace table: without it the MLflow exporter silently drops
    every trace."""
    cfg = tracing_config(state)
    entry = agent_tracing(state, tool)
    if not cfg or not entry:
        return {}
    env = {
        "MLFLOW_TRACKING_URI": str(cfg["tracking_uri"]),
        "MLFLOW_EXPERIMENT_ID": str(entry["experiment_id"]),
        # The trace is emitted by the `mlflow autolog claude stop-hook` Stop
        # hook, which is always a short-lived one-shot process. With async
        # trace logging (MLflow's default) the root `claude_code_conversation`
        # span is queued and the hook's best-effort flush can lose it if the
        # export hasn't drained before the process exits — observed on CI
        # runners, where it leaves an orphaned `llm` span and no queryable
        # trace. Force synchronous export so the span is written before exit.
        "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING": "false",
    }
    warehouse_id = cfg.get("sql_warehouse_id")
    if warehouse_id:
        env["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = str(warehouse_id)
    return env


def disable_tracing(state: dict) -> dict:
    """Mark tracing disabled and rewrite each configured agent's config so the
    injected tracing keys are stripped."""
    cfg = state.get("tracing")
    if isinstance(cfg, dict):
        cfg["enabled"] = False
        state["tracing"] = cfg
    save_state(state)
    return _rewrite_agent_configs(state)


def _configured_tracing_agents(state: dict) -> list[str]:
    available = set(state.get("available_tools") or [])
    return [tool for tool in TRACING_AGENTS if tool in available]


def _tracing_capable_workspaces(full: dict) -> list[str]:
    """Configured workspaces that have at least one tracing-capable agent."""
    workspaces = full.get("workspaces") or {}
    out: list[str] = []
    for ws, st in workspaces.items():
        if set((st or {}).get("available_tools") or []) & set(TRACING_AGENTS):
            out.append(ws)
    return out


def _tracing_enabled_workspaces(full: dict) -> list[str]:
    """Configured workspaces that currently have tracing enabled."""
    workspaces = full.get("workspaces") or {}
    return [ws for ws, st in workspaces.items() if ((st or {}).get("tracing") or {}).get("enabled")]


def _hydrate_workspace_entry(full: dict, workspace: str, profile: str | None) -> dict:
    workspaces = full.get("workspaces") or {}
    entry = dict(workspaces.get(workspace) or {})
    entry["workspace"] = workspace
    if profile:
        entry["profile"] = profile
    return hydrate_state(entry)


def _select_tracing_workspace(*, only_enabled: bool = False) -> dict:
    """Prompt for which workspace's tracing to configure, current first. Returns
    that workspace's hydrated flat state.

    ``only_enabled=True`` restricts to workspaces that currently have tracing
    enabled (used by ``--disable``) and skips the prompt entirely when there's
    only one match — the user has nothing meaningful to choose."""
    full = load_full_state()
    workspaces = full.get("workspaces") or {}
    if only_enabled:
        candidates = _tracing_enabled_workspaces(full)
        if not candidates:
            return {}
    else:
        candidates = _tracing_capable_workspaces(full)
        if not candidates:
            raise RuntimeError(
                "Claude Code is not configured. Run `ucode configure` for Claude Code first."
            )

    current = full.get("current_workspace")
    candidates.sort(key=lambda ws: (ws != current, ws))

    if len(candidates) == 1:
        # Single match — no choice to present.
        workspace = candidates[0]
        profile = (workspaces.get(workspace) or {}).get("profile") or ""
    else:
        # Coerce a missing profile to "" (falsy) so the type matches and
        # downstream resolves the default ~/.databrickscfg profile.
        profiles = [(ws, (workspaces.get(ws) or {}).get("profile") or "") for ws in candidates]
        prompt = (
            "Tracing is enabled on multiple workspaces — pick which to disable"
            if only_enabled
            else "Select the workspace to configure MLflow tracing for"
        )
        workspace, profile = prompt_for_workspace(prompt, profiles)

    if not only_enabled:
        entry_check = workspaces.get(workspace) or {}
        if not (set(entry_check.get("available_tools") or []) & set(TRACING_AGENTS)):
            raise RuntimeError(
                f"{workspace} has no tracing-capable agents configured. "
                "Run `ucode configure` for it first."
            )
    return _hydrate_workspace_entry(full, workspace, profile or None)


def _rewrite_agent_configs(state: dict) -> dict:
    """Re-run each configured agent's config writer so it folds the current
    tracing state into its config files (adds keys when enabled, strips them
    when disabled)."""
    from ucode.agents import configure_tool, default_model_for_tool

    for tool in _configured_tracing_agents(state):
        model = default_model_for_tool(tool, state)
        state = configure_tool(tool, state, model)
    return state


def _install_agent_tracing_deps(state: dict) -> None:
    """Install the Claude tracing runtime (pinned mlflow CLI) when Claude is
    configured on this workspace and has tracing on. Claude is the only
    tracing-capable agent."""
    from ucode.agents import claude

    configured = _configured_tracing_agents(state)
    if "claude" in configured and agent_tracing(state, "claude"):
        claude.ensure_tracing_runtime()


def configure_tracing_command(
    disable: bool = False, workspaces: list[tuple[str, str | None]] | None = None
) -> int:
    # `save_state` (called by us and by every agent config writer underneath
    # us) flips `current_workspace` to the workspace it's saving. Tracing can
    # be configured on a non-current workspace, so snapshot here and restore
    # at the end — running `configure tracing` must not change which workspace
    # `ucode launch` targets.
    original_current = load_full_state().get("current_workspace")
    try:
        if workspaces is not None:
            return _enable_tracing_for_workspaces(workspaces)
        return _configure_tracing(disable=disable)
    finally:
        set_current_workspace(original_current)


def _enable_tracing_for_workspaces(workspaces: list[tuple[str, str | None]]) -> int:
    """Enable tracing for an explicit set of (url, profile) workspaces without
    prompting — used by `configure --tracing`, which already knows which
    workspace(s) it just set up. Workspaces with no tracing-capable agent are
    skipped with a note rather than treated as an error."""
    full = load_full_state()
    enabled_any = False
    for workspace, profile in workspaces:
        state = _hydrate_workspace_entry(full, workspace, profile)
        if not _configured_tracing_agents(state):
            print_section("MLflow Tracing")
            print_note(f"{workspace}: no tracing-capable agents configured — skipping.")
            continue
        _enable_tracing_for_state(state)
        enabled_any = True
    return 0 if enabled_any else 1


def _configure_tracing(disable: bool) -> int:
    if disable:
        return _disable_tracing_command()

    state = _select_tracing_workspace()
    _enable_tracing_for_state(state)
    return 0


def _enable_tracing_for_state(state: dict) -> dict:
    """Resolve the shared experiment, persist tracing config, install deps, and
    rewrite agent configs for one already-selected, hydrated workspace state."""
    workspace = state["workspace"]
    configured = _configured_tracing_agents(state)
    profile = state.get("profile")
    apply_pat_environment(state)
    ensure_databricks_auth(workspace, profile)

    print_section("MLflow Tracing")
    print_kv("Workspace", workspace)

    # Running `ucode configure tracing` is itself the opt-in, so there's no
    # confirmation prompt; `--disable` is the explicit way back off.
    token = get_databricks_token(workspace, profile)

    # ucode does not create the experiment: an admin must have already
    # provisioned a `ucode-traces` experiment whose traces are backed by Unity
    # Catalog. Assert it exists (and is UC-backed), failing with setup steps if
    # not, so every agent and user routes to the same UC-backed sink.
    name = experiment_name()
    with spinner("Looking for the ucode-traces experiment..."):
        experiment, reason = find_uc_backed_experiment(workspace, token, name)
    if not experiment:
        raise RuntimeError(_missing_experiment_error(name, reason))

    # A UC-backed experiment needs a SQL warehouse to write traces — without
    # `MLFLOW_TRACING_SQL_WAREHOUSE_ID` the exporter silently drops them — so a
    # warehouse is mandatory, not optional.
    with spinner("Resolving a SQL warehouse for trace storage..."):
        warehouse_id, wh_reason = resolve_sql_warehouse_id(workspace, token)
    if not warehouse_id:
        raise RuntimeError(_missing_warehouse_error(wh_reason))

    state["tracing"] = {
        "enabled": True,
        "tracking_uri": tracking_uri_for_state(state),
        "experiment_id": experiment["experiment_id"],
        "experiment_name": experiment["experiment_name"],
        "uc_destination": experiment["uc_destination"],
        "sql_warehouse_id": warehouse_id,
    }
    save_state(state)

    print_kv("Tracking URI", str(state["tracing"]["tracking_uri"]))
    print_kv(
        "Experiment",
        f"{experiment['experiment_name']} (id {experiment['experiment_id']})",
    )
    print_kv("Unity Catalog", experiment["uc_destination"])
    print_kv("SQL warehouse", warehouse_id)

    _install_agent_tracing_deps(state)
    state = _rewrite_agent_configs(state)

    print_success(f"Tracing configured for: {', '.join(configured)}")
    return state


def _disable_tracing_command() -> int:
    """``--disable`` flow: pick (or auto-select) a workspace that has tracing
    enabled, then strip the tracing config from its agent files."""
    state = _select_tracing_workspace(only_enabled=True)
    if not state:
        print_section("MLflow Tracing")
        print_note("Tracing is not enabled on any configured workspace — nothing to do.")
        return 0

    workspace = state["workspace"]
    profile = state.get("profile")
    apply_pat_environment(state)
    ensure_databricks_auth(workspace, profile)

    print_section("MLflow Tracing")
    print_kv("Workspace", workspace)
    disable_tracing(state)
    print_success("Tracing disabled")
    return 0
