"""Tests for Codex intelligent-routing hooks."""

from __future__ import annotations

import json
import urllib.error

from ucode import codex_routing

WS = "https://example.databricks.com"


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_routes_with_task_v1_codex_menu(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response(
            {
                "route_selection": [{"route_option": {"model": "gpt-5-6-sol", "harness": "codex"}}],
                "rationale": "Needs the strongest coding model.",
            }
        )

    monkeypatch.setattr(codex_routing.urllib.request, "urlopen", fake_urlopen)

    decision, error = codex_routing.request_routing_decision(
        WS,
        "token",
        "Refactor the parser",
        ["system.ai.gpt-5-6-luna", "system.ai.gpt-5-6-sol"],
    )

    assert error is None
    assert decision == codex_routing.RoutingDecision(
        model="system.ai.gpt-5-6-sol",
        raw_model="gpt-5-6-sol",
        rationale="Needs the strongest coding model.",
    )
    assert captured["url"] == f"{WS}/ai-gateway/routing/v1/routes:select"
    assert captured["headers"]["Authorization"] == "Bearer token"
    assert captured["body"] == {
        "route_options": [
            {"model": "glm-5-2", "harness": "codex"},
            {"model": "gpt-5-6-sol", "harness": "codex"},
            {"model": "gpt-5-6-luna", "harness": "codex"},
        ],
        "task": {"prompt": "Refactor the parser"},
        "route_selector": {"router_name": "task_v1"},
    }


def test_router_model_is_not_substituted_when_exact_model_is_unavailable():
    model = codex_routing.resolve_routed_model(
        "gpt-5-6-luna",
        ["databricks-gpt-5-4-nano", "databricks-gpt-5", "databricks-gpt-5-5"],
    )

    assert model is None


def test_glm_maps_to_databricks_gateway_model():
    model = codex_routing.resolve_routed_model(
        "glm-5-2",
        ["system.ai.gpt-5-6-luna", "system.ai.gpt-5-6-sol"],
    )

    assert model == "system.ai.glm-5-2"


def test_exact_router_model_is_preserved():
    model = codex_routing.resolve_routed_model(
        "gpt-5-6-sol",
        ["databricks-gpt-5-5", "databricks-gpt-5-6-sol"],
    )

    assert model == "databricks-gpt-5-6-sol"


def test_router_failure_fails_open(monkeypatch):
    monkeypatch.setattr(
        codex_routing.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )

    decision, error = codex_routing.request_routing_decision(
        WS,
        "token",
        "task",
        ["system.ai.gpt-5-6-luna", "system.ai.gpt-5-6-sol"],
    )

    assert decision is None
    assert "offline" in str(error)


def test_spawn_rewrite_preserves_original_input(monkeypatch):
    encrypted_message = {"encrypted": "opaque-ciphertext"}
    payload = {
        "tool_name": "collaborationspawn_agent",
        "tool_input": {
            "task_name": "reviewer",
            "message": encrypted_message,
            "fork": False,
        },
    }
    monkeypatch.setattr(
        codex_routing,
        "request_routing_decision",
        lambda *args, **kwargs: (
            codex_routing.RoutingDecision(
                model="databricks-gpt-5-5",
                raw_model="gpt-5-6-sol",
                rationale="Review needs deeper reasoning.",
            ),
            None,
        ),
    )

    output = codex_routing.route_pre_tool_use(
        payload,
        workspace=WS,
        token="token",
        available_models=["databricks-gpt-5-5"],
    )

    hook = output["hookSpecificOutput"]
    assert output["systemMessage"] == "Using Intelligent Routing. Routing to gpt-5.5."
    assert hook["permissionDecision"] == "allow"
    assert hook["updatedInput"] == {
        "task_name": "reviewer",
        "message": encrypted_message,
        "fork": False,
        "model": "gpt-5.5",
    }
    assert hook["permissionDecisionReason"] == (
        "Using Intelligent Routing. Routing to gpt-5.5. Review needs deeper reasoning."
    )


def test_spawn_rewrite_uses_codex_model_id_for_uc_endpoint(monkeypatch):
    monkeypatch.setattr(
        codex_routing,
        "request_routing_decision",
        lambda *args, **kwargs: (
            codex_routing.RoutingDecision(
                model="system.ai.gpt-5-6-luna",
                raw_model="gpt-5-6-luna",
            ),
            None,
        ),
    )

    output = codex_routing.route_pre_tool_use(
        {
            "tool_name": "collaborationspawn_agent",
            "tool_input": {"task_name": "routing-smoke-test", "message": "encrypted"},
        },
        workspace=WS,
        token="token",
        available_models=["system.ai.gpt-5-6-luna"],
    )

    assert output["systemMessage"] == "Using Intelligent Routing. Routing to gpt-5.6-luna."
    assert output["hookSpecificOutput"]["updatedInput"]["model"] == "gpt-5.6-luna"


def test_spawn_glm_decision_keeps_original_model(monkeypatch):
    monkeypatch.setattr(
        codex_routing,
        "request_routing_decision",
        lambda *args, **kwargs: (
            codex_routing.RoutingDecision(
                model="system.ai.glm-5-2",
                raw_model="glm-5-2",
            ),
            None,
        ),
    )

    output = codex_routing.route_pre_tool_use(
        {
            "tool_name": "collaborationspawn_agent",
            "tool_input": {"task_name": "glm-task", "message": "encrypted"},
        },
        workspace=WS,
        token="token",
        available_models=["system.ai.gpt-5-6-luna", "system.ai.gpt-5-6-sol"],
    )

    assert output == {
        "systemMessage": (
            "Intelligent Routing selected GLM 5.2, which is not enabled for Codex "
            "subagents. Keeping the original subagent model."
        )
    }


def test_non_spawn_tool_has_no_opinion():
    assert (
        codex_routing.route_pre_tool_use(
            {"tool_name": "Bash", "tool_input": {"command": "true"}},
            workspace=WS,
            token="token",
            available_models=["databricks-gpt-5"],
        )
        is None
    )


def test_canary_and_audit_are_written(tmp_path, monkeypatch):
    canary = tmp_path / "canary.json"
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(codex_routing, "CANARY_PATH", canary)
    monkeypatch.setattr(codex_routing, "AUDIT_PATH", audit)

    codex_routing.record_session_start({"session_id": "s1", "model": "gpt-5.5"})
    codex_routing.record_subagent_start(
        {"session_id": "s1", "agent_id": "a1", "agent_type": "reviewer", "model": "gpt-5"}
    )

    assert json.loads(canary.read_text())["session_id"] == "s1"
    assert json.loads(audit.read_text().strip())["agent_id"] == "a1"


def test_decision_is_reconciled_with_actual_subagent_model(tmp_path, monkeypatch):
    decisions = tmp_path / "decisions.jsonl"
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(codex_routing, "DECISIONS_PATH", decisions)
    monkeypatch.setattr(codex_routing, "AUDIT_PATH", audit)
    monkeypatch.setattr(
        codex_routing,
        "request_routing_decision",
        lambda *args, **kwargs: (
            codex_routing.RoutingDecision(
                model="system.ai.gpt-5-6-luna",
                raw_model="gpt-5-6-luna",
            ),
            None,
        ),
    )

    codex_routing.route_pre_tool_use(
        {
            "session_id": "s1",
            "tool_name": "collaborationspawn_agent",
            "tool_input": {"task_name": "glm-task", "message": "encrypted"},
        },
        workspace=WS,
        token="token",
        available_models=["system.ai.gpt-5-6-luna", "system.ai.gpt-5-6-sol"],
        audit_decision=True,
    )
    record = codex_routing.record_subagent_start(
        {"session_id": "s1", "agent_id": "a1", "model": "gpt-5.6-luna"}
    )

    assert record["router_model"] == "gpt-5-6-luna"
    assert record["requested_model"] == "gpt-5.6-luna"
    assert record["matches_router_decision"] is True
