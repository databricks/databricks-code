"""Databricks AI Gateway routing helpers for Codex sessions and subagents."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ucode.config_io import APP_DIR
from ucode.databricks import get_databricks_token

ROUTER_NAME = "task_v1"
ROUTING_PATH = "/ai-gateway/routing/v1/routes:select"
REQUEST_TIMEOUT_S = 30.0
CODEX_ROUTE_ARMS = ("glm-5-2", "gpt-5-6-sol", "gpt-5-6-luna")
GLM_ROUTE_ARM = "glm-5-2"
GLM_GATEWAY_MODEL = "system.ai.glm-5-2"
SPAWN_AGENT_TOOL_SUFFIX = "spawn_agent"
CANARY_PATH = APP_DIR / "codex-intelligent-routing-canary.json"
AUDIT_PATH = APP_DIR / "codex-intelligent-routing-audit.jsonl"
DECISIONS_PATH = APP_DIR / "codex-intelligent-routing-decisions.jsonl"

_GPT_RE = re.compile(r"gpt-(\d+)(?:[.-](\d+))?(?:[.-](\d+))?(-.+|[a-z].*)?")


@dataclass(frozen=True)
class RoutingDecision:
    """One model selection returned by the AI Gateway router."""

    model: str
    raw_model: str
    rationale: str = ""


def route_launch_model(state: dict, tool_args: list[str]):
    """Route a root Codex launch before the Codex process starts."""
    workspace = state.get("workspace")
    models = state.get("codex_models")
    if not isinstance(workspace, str) or not isinstance(models, list):
        return None, "workspace model metadata is unavailable"
    try:
        token = get_databricks_token(workspace, state.get("profile"))
    except RuntimeError as exc:
        return None, f"could not authenticate the routing request: {exc}"
    task = _launch_routing_task(tool_args)
    return request_routing_decision(workspace, token, task, models)


def _launch_routing_task(tool_args: list[str]) -> str:
    if "exec" in tool_args:
        prompt_parts = tool_args[tool_args.index("exec") + 1 :]
        if prompt_parts:
            return " ".join(prompt_parts)
    if tool_args:
        return "Start a Codex session with options: " + " ".join(tool_args)
    return f"Start an interactive Codex coding session in {Path.cwd().name}."


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

    body = {
        "route_options": [{"model": model, "harness": "codex"} for model in CODEX_ROUTE_ARMS],
        "task": {"prompt": task[:4000]},
        "route_selector": {"router_name": ROUTER_NAME},
    }
    request = urllib.request.Request(
        workspace.rstrip("/") + ROUTING_PATH,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except OSError:
            pass
        reason = f"router returned HTTP {exc.code}"
        if detail:
            reason = f"{reason}: {detail[:300]}"
        return None, reason
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        return None, f"router request failed: {exc}"

    raw_model = _selected_model(payload)
    if raw_model is None:
        return None, "router returned no model selection"
    model = resolve_routed_model(raw_model, candidates)
    if model is None:
        return None, f"router selected unsupported model {raw_model!r}"
    rationale = payload.get("rationale") if isinstance(payload, dict) else None
    return (
        RoutingDecision(
            model=model,
            raw_model=raw_model,
            rationale=rationale if isinstance(rationale, str) else "",
        ),
        None,
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
    if not is_spawn_agent_tool(payload.get("tool_name")):
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    task_name = tool_input.get("task_name") or tool_input.get("agent_name")
    task = task_name if isinstance(task_name, str) and task_name else "Codex subagent task"
    decision, _ = request_routing_decision(
        workspace,
        token,
        task,
        available_models,
        timeout=timeout,
    )
    if decision is None:
        return None
    if decision.raw_model == GLM_ROUTE_ARM:
        return {
            "systemMessage": (
                "Intelligent Routing selected GLM 5.2, which is not enabled for Codex "
                "subagents. Keeping the original subagent model."
            )
        }
    routed_model = _codex_model_id(decision.model)
    if audit_decision:
        _record_routing_decision(payload, task, decision, routed_model)
    routing_message = f"Using Intelligent Routing. Routing to {routed_model}."
    output: dict[str, Any] = {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "updatedInput": {**tool_input, "model": routed_model},
        "permissionDecisionReason": routing_message,
    }
    if decision.rationale:
        output["permissionDecisionReason"] += f" {decision.rationale}"
    return {"systemMessage": routing_message, "hookSpecificOutput": output}


def is_spawn_agent_tool(tool_name: Any) -> bool:
    """Return whether a hook payload names Codex's subagent spawn tool."""
    if not isinstance(tool_name, str):
        return False
    normalized = tool_name.strip().lower()
    return normalized == "agent" or normalized.endswith(SPAWN_AGENT_TOOL_SUFFIX)


def record_session_start(payload: dict[str, Any]) -> None:
    """Write a canary proving Codex trusted and ran the routing hooks."""
    _write_json(
        CANARY_PATH,
        {
            "session_id": payload.get("session_id"),
            "model": payload.get("model"),
            "at": time.time(),
        },
    )


def record_subagent_start(payload: dict[str, Any]) -> dict[str, Any]:
    """Append the model Codex actually selected for a routed subagent."""
    actual_model = payload.get("model")
    decision = _pending_decision(payload.get("session_id"), actual_model)
    record = {
        "agent_id": payload.get("agent_id"),
        "agent_type": payload.get("agent_type"),
        "model": actual_model,
        "session_id": payload.get("session_id"),
        "at": time.time(),
    }
    if decision is not None:
        record.update(
            {
                "decision_id": decision.get("decision_id"),
                "router_model": decision.get("router_model"),
                "requested_model": decision.get("requested_model"),
                "matches_router_decision": decision.get("requested_model") == actual_model,
            }
        )
    _append_jsonl(AUDIT_PATH, record)
    return record


def clear_routing_artifacts() -> None:
    """Remove ucode-owned routing canary and audit files."""
    for path in (CANARY_PATH, AUDIT_PATH, DECISIONS_PATH):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


def _selected_model(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    selections = payload.get("route_selection")
    if not isinstance(selections, list) or not selections:
        return None
    selection = selections[0]
    if not isinstance(selection, dict):
        return None
    option = selection.get("route_option")
    if not isinstance(option, dict):
        return None
    model = option.get("model")
    return model if isinstance(model, str) and model else None


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


def _normalize_model(model: str) -> str:
    tail = model.rsplit("/", 1)[-1]
    for prefix in ("databricks-", "system.ai."):
        if tail.startswith(prefix):
            tail = tail[len(prefix) :]
            break
    return tail.lower()


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


def _record_routing_decision(
    payload: dict[str, Any],
    task_name: str,
    decision: RoutingDecision,
    requested_model: str,
) -> None:
    _append_jsonl(
        DECISIONS_PATH,
        {
            "decision_id": uuid.uuid4().hex,
            "session_id": payload.get("session_id"),
            "task_name": task_name,
            "router_model": decision.raw_model,
            "requested_model": requested_model,
            "at": time.time(),
        },
    )


def _pending_decision(session_id: Any, actual_model: Any) -> dict[str, Any] | None:
    used = {
        record.get("decision_id") for record in _read_jsonl(AUDIT_PATH) if record.get("decision_id")
    }
    pending = [
        decision
        for decision in _read_jsonl(DECISIONS_PATH)
        if decision.get("session_id") == session_id and decision.get("decision_id") not in used
    ]
    return next(
        (decision for decision in pending if decision.get("requested_model") == actual_model),
        pending[0] if pending else None,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
    except OSError:
        return


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        return
