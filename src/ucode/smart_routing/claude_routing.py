"""Databricks AI Gateway routing helpers for Claude Code sessions and subagents.

Claude-specific configuration on top of the shared
:mod:`ucode.smart_routing.routing` core. The ``task_v1`` router infers a
"Claude Code" (``cc``) scenario when it is offered only Claude arms, and that
scenario REQUIRES its full menu — both ``claude-opus-4-8`` and
``claude-sonnet-5`` — or it returns BAD_REQUEST. So both arms are always offered.
"""

from __future__ import annotations

# Re-exported so tests can patch the shared ``urlopen`` seam via
# ``claude_routing.urllib.request`` — the call lives in ``routing``, but Python
# modules are singletons so patching this name patches the one call site.
import urllib.request  # noqa: F401
from typing import Any

from ucode.config_io import APP_DIR
from ucode.databricks import get_databricks_token
from ucode.smart_routing import routing
from ucode.smart_routing.routing import RoutingDecision

ROUTER_NAME = routing.ROUTER_NAME
REQUEST_TIMEOUT_S = routing.REQUEST_TIMEOUT_S
# Frozen task_v1 "cc" scenario menu (ai-gateway CanonicalModelNames): the router
# rejects the request unless BOTH are offered as route_options.
CLAUDE_ROUTE_ARMS = ("claude-opus-4-8", "claude-sonnet-5")
# Claude Code's subagent-spawn tool (renamed Task -> Agent); match both.
SPAWN_AGENT_TOOL_NAMES = ("agent", "task")
CANARY_PATH = APP_DIR / "claude-smart-routing-canary.json"
AUDIT_PATH = APP_DIR / "claude-smart-routing-audit.jsonl"
DECISIONS_PATH = APP_DIR / "claude-smart-routing-decisions.jsonl"

_normalize_model = routing.normalize_model


def route_launch_model(state: dict, tool_args: list[str]):
    """Route a root Claude Code launch on the launch-time prompt, if there is one.

    Returns (None, None) when the launch carries no prompt (a bare interactive
    session): with no task signal the router can only return its floor arm, so
    routing would just add a round-trip and silently override the user's default
    model. In that case we don't route and keep the configured default. Routing
    on a typed-in first prompt is out of scope — no hook/MCP can retarget the
    root model once the session is running.
    """
    task = _launch_routing_task(tool_args)
    if task is None:
        return None, None
    workspace = state.get("workspace")
    models = state.get("claude_models")
    if not isinstance(workspace, str) or not isinstance(models, dict):
        return None, "workspace model metadata is unavailable"
    try:
        token = get_databricks_token(workspace, state.get("profile"))
    except RuntimeError as exc:
        return None, f"could not authenticate the routing request: {exc}"
    available = [m for m in models.values() if isinstance(m, str) and m]
    return request_routing_decision(workspace, token, task, available)


# Claude Code CLI options that consume a following value (from `claude --help`);
# their values must not be mistaken for the seed prompt. Options whose value is
# optional (`-c`/`-d`/`-r`/`-w`) are intentionally omitted — treating them as
# booleans is safe here (we only skip the flag, never a following positional).
_CLAUDE_VALUE_OPTIONS = frozenset(
    {
        "-n",
        "--name",
        "-m",
        "--model",
        "--add-dir",
        "--agent",
        "--agents",
        "--allowed-tools",
        "--disallowed-tools",
        "--tools",
        "--append-system-prompt",
        "--system-prompt",
        "--betas",
        "--debug-file",
        "--effort",
        "--fallback-model",
        "--file",
        "--input-format",
        "--output-format",
        "--json-schema",
        "--max-budget-usd",
        "--mcp-config",
        "--permission-mode",
        "--plugin-dir",
        "--plugin-url",
        "--remote-control-session-name-prefix",
        "--session-id",
        "--setting-sources",
        "--settings",
    }
)


def _launch_routing_task(tool_args: list[str]) -> str | None:
    # The routing task is the user's real first prompt when it's on the command
    # line (`claude "<prompt>"` or `claude -p "<prompt>"`, or after `--`). A bare
    # interactive launch has no prompt yet → None, and the caller skips routing
    # (the root model can't be re-routed once the session is running).
    return routing.extract_seed_prompt(tool_args, _CLAUDE_VALUE_OPTIONS)


def request_routing_decision(
    workspace: str,
    token: str,
    task: str,
    available_models: list[str],
    *,
    timeout: float = REQUEST_TIMEOUT_S,
) -> tuple[RoutingDecision | None, str | None]:
    """Ask the workspace ``task_v1`` router for a servable Claude model.

    Offers the full ``cc`` menu; resolves the router's pick back to the
    workspace's routable id (e.g. ``system.ai.claude-opus-4-8``).
    """
    available = {_normalize_model(m): m for m in available_models if isinstance(m, str) and m}
    missing = [arm for arm in CLAUDE_ROUTE_ARMS if arm not in available]
    if missing:
        return None, f"required Claude routing models are unavailable: {', '.join(missing)}"

    return routing.select_route(
        workspace,
        token,
        task,
        [(arm, "claude") for arm in CLAUDE_ROUTE_ARMS],
        lambda raw_model: available.get(_normalize_model(raw_model)),
        timeout=timeout,
    )


def route_pre_tool_use(
    payload: dict[str, Any],
    *,
    workspace: str,
    token: str,
    available_models: list[str],
    timeout: float = REQUEST_TIMEOUT_S,
    audit_decision: bool = False,
) -> dict[str, Any] | None:
    """Route one Claude Code ``Agent`` (subagent-spawn) call, rewriting its model."""
    record = None
    if audit_decision:

        def record(payload, task, decision, requested):
            routing.write_decision_record(DECISIONS_PATH, payload, task, decision, requested)

    return routing.route_spawn_tool(
        payload,
        is_spawn_agent=is_spawn_agent_tool,
        decision_fn=lambda task: request_routing_decision(
            workspace, token, task, available_models, timeout=timeout
        ),
        default_task_label="Claude Code subagent task",
        model_id_mapper=_claude_model_id,
        record_decision=record,
    )


def is_spawn_agent_tool(tool_name: Any) -> bool:
    """Return whether a hook payload names Claude Code's subagent spawn tool."""
    if not isinstance(tool_name, str):
        return False
    return tool_name.strip().lower() in SPAWN_AGENT_TOOL_NAMES


def record_session_start(payload: dict[str, Any]) -> None:
    """Write a canary proving Claude Code trusted and ran the routing hooks."""
    routing.record_session_start(CANARY_PATH, payload)


def record_subagent_start(payload: dict[str, Any]) -> dict[str, Any]:
    """Append the model Claude Code actually selected for a routed subagent."""
    return routing.record_subagent_start(DECISIONS_PATH, AUDIT_PATH, payload)


def clear_routing_artifacts() -> None:
    """Remove ucode-owned routing canary and audit files."""
    routing.clear_artifacts((CANARY_PATH, AUDIT_PATH, DECISIONS_PATH))


def _claude_model_id(model: str) -> str:
    """The model id Claude Code should launch the subagent with.

    Claude Code's ``Agent`` tool ``model`` field accepts only short family
    names (``sonnet``, ``opus``, ``haiku``, ``fable``), not full workspace
    ids. Map the router's pick (e.g. ``system.ai.claude-sonnet-5``) back to
    its family name.
    """
    normalized = _normalize_model(model)
    for family in ("fable", "opus", "sonnet", "haiku"):
        if f"claude-{family}-" in normalized:
            return family
    return model
