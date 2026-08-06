"""Databricks AI Gateway routing helpers for Codex sessions and subagents.

Codex-specific configuration on top of the shared :mod:`ucode.smart_routing.routing`
core: the frozen ``task_v1`` Codex route arms, the ``spawn_agent`` tool detector,
the Codex model-id translation, and the artifact paths.
"""

from __future__ import annotations

import re

# Re-exported so tests can patch the shared ``urlopen`` seam via
# ``codex_routing.urllib.request`` — the actual call lives in ``routing``, but
# Python modules are singletons so patching this name patches the one call site.
import urllib.request  # noqa: F401
from typing import Any

from ucode.config_io import APP_DIR
from ucode.databricks import get_databricks_token
from ucode.smart_routing import routing
from ucode.smart_routing.routing import RoutingDecision

ROUTER_NAME = routing.ROUTER_NAME
ROUTING_PATH = routing.ROUTING_PATH
REQUEST_TIMEOUT_S = routing.REQUEST_TIMEOUT_S
CODEX_ROUTE_ARMS = ("glm-5-2", "gpt-5-6-sol", "gpt-5-6-luna")
GLM_ROUTE_ARM = "glm-5-2"
GLM_GATEWAY_MODEL = "system.ai.glm-5-2"
SPAWN_AGENT_TOOL_SUFFIX = "spawn_agent"
CANARY_PATH = APP_DIR / "codex-smart-routing-canary.json"
AUDIT_PATH = APP_DIR / "codex-smart-routing-audit.jsonl"
DECISIONS_PATH = APP_DIR / "codex-smart-routing-decisions.jsonl"

_GPT_RE = re.compile(r"gpt-(\d+)(?:[.-](\d+))?(?:[.-](\d+))?(-.+|[a-z].*)?")

_normalize_model = routing.normalize_model


def route_launch_model(state: dict, tool_args: list[str]):
    """Route a root Codex launch on the launch-time prompt, if there is one.

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
    models = state.get("codex_models")
    if not isinstance(workspace, str) or not isinstance(models, list):
        return None, "workspace model metadata is unavailable"
    try:
        token = get_databricks_token(workspace, state.get("profile"))
    except RuntimeError as exc:
        return None, f"could not authenticate the routing request: {exc}"
    return request_routing_decision(workspace, token, task, models)


# Codex CLI options that consume a following value (from `codex --help`); their
# values must not be mistaken for the seed prompt when parsing launch args.
_CODEX_VALUE_OPTIONS = frozenset(
    {
        "-c",
        "--config",
        "-i",
        "--image",
        "-m",
        "--model",
        "-p",
        "--profile",
        "-s",
        "--sandbox",
        "-a",
        "--ask-for-approval",
        "-C",
        "--cd",
        "--add-dir",
        "--enable",
        "--disable",
        "--local-provider",
        "--remote",
        "--remote-auth-token-env",
    }
)


def _launch_routing_task(tool_args: list[str]) -> str | None:
    # The routing task is the user's real first prompt when it's on the command
    # line (`codex "<prompt>"`, `codex exec "<prompt>"`, or after `--`). A bare
    # interactive launch has no prompt yet → None, and the caller skips routing
    # (the root model can't be re-routed once the TUI is running).
    return routing.extract_seed_prompt(tool_args, _CODEX_VALUE_OPTIONS)


def request_routing_decision(
    workspace: str,
    token: str,
    task: str,
    available_models: list[str],
    *,
    timeout: float = REQUEST_TIMEOUT_S,
) -> tuple[RoutingDecision | None, str | None]:
    """Ask the workspace ``task_v1`` router for a servable Codex model."""
    candidates = _routing_candidates(available_models)
    missing = [
        arm for arm in CODEX_ROUTE_ARMS if arm not in {_normalize_model(m) for m in candidates}
    ]
    if missing:
        return None, f"required Codex routing models are unavailable: {', '.join(missing)}"

    return routing.select_route(
        workspace,
        token,
        task,
        [(arm, "codex") for arm in CODEX_ROUTE_ARMS],
        lambda raw_model: resolve_routed_model(raw_model, candidates),
        timeout=timeout,
    )


def resolve_routed_model(raw_model: str, available_models: list[str]) -> str | None:
    """Map a ``task_v1`` arm to a model the configured workspace can serve."""
    candidates = _routing_candidates(available_models)
    normalized = {_normalize_model(model): model for model in candidates}
    return normalized.get(_normalize_model(raw_model))


def route_pre_tool_use(
    payload: dict[str, Any],
    *,
    workspace: str,
    token: str,
    available_models: list[str],
    timeout: float = REQUEST_TIMEOUT_S,
    audit_decision: bool = False,
) -> dict[str, Any] | None:
    """Route one Codex ``spawn_agent`` call and rewrite its model."""
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
        default_task_label="Codex subagent task",
        model_id_mapper=_codex_model_id,
        record_decision=record,
    )


def is_spawn_agent_tool(tool_name: Any) -> bool:
    """Return whether a hook payload names Codex's subagent spawn tool."""
    if not isinstance(tool_name, str):
        return False
    normalized = tool_name.strip().lower()
    return normalized == "agent" or normalized.endswith(SPAWN_AGENT_TOOL_SUFFIX)


def record_session_start(payload: dict[str, Any]) -> None:
    """Write a canary proving Codex trusted and ran the routing hooks."""
    routing.record_session_start(CANARY_PATH, payload)


def record_subagent_start(payload: dict[str, Any]) -> dict[str, Any]:
    """Append the model Codex actually selected for a routed subagent."""
    return routing.record_subagent_start(DECISIONS_PATH, AUDIT_PATH, payload)


def clear_routing_artifacts() -> None:
    """Remove ucode-owned routing canary and audit files."""
    routing.clear_artifacts((CANARY_PATH, AUDIT_PATH, DECISIONS_PATH))


def _routing_candidates(models: list[str]) -> list[str]:
    candidates = [model for model in models if isinstance(model, str) and model]
    if GLM_ROUTE_ARM not in {_normalize_model(model) for model in candidates}:
        candidates.append(GLM_GATEWAY_MODEL)
    return candidates


def _parse_gpt(model: str) -> tuple[int, int, int, str] | None:
    match = _GPT_RE.fullmatch(_normalize_model(model))
    if not match:
        return None
    major, minor, patch, suffix = match.groups()
    return int(major), int(minor or 0), int(patch or 0), suffix or ""


def _model_strength(model: str) -> tuple[int, int, int, int]:
    parsed = _parse_gpt(model)
    if parsed is None:
        return (0, 0, 0, 0)
    major, minor, patch, suffix = parsed
    return major, minor, patch, 1 if not suffix else 0


def _codex_model_id(model: str) -> str:
    tail = model.rsplit("/", 1)[-1]
    if tail in {"databricks-gpt-5-2-codex", "databricks-gpt-5-4-nano"}:
        return tail
    if _normalize_model(model) == GLM_ROUTE_ARM:
        return GLM_GATEWAY_MODEL
    if model.startswith("system.ai."):
        bare = model.removeprefix("system.ai.")
    elif tail.startswith("databricks-"):
        bare = tail.removeprefix("databricks-")
    else:
        return model
    match = _GPT_RE.fullmatch(bare)
    if not match:
        return bare
    major, minor, patch, suffix = match.groups()
    version = major
    if minor is not None:
        version += f".{minor}"
    if patch is not None:
        version += f".{patch}"
    return f"gpt-{version}{suffix or ''}"
