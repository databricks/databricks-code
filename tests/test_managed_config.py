"""Tests for managed_config.py — fetch/normalize/persist of the admin-authored managed config."""

from __future__ import annotations

import os
import stat

import pytest

import ucode.databricks as db_mod
import ucode.managed_config as mc_mod
from ucode.managed_config import (
    get_managed_config,
    load_managed_state,
    normalize_managed_config,
    save_managed_state,
)

# A representative raw CodingAgentConfig proto-JSON manifest (mirrors what the API returns).
RAW_MANIFEST = {
    "name": "coding-agent-configs/abc-123",
    "workspace_id": 1653573648247579,
    "default_agent": "CODING_AGENT_CLAUDE_CODE",
    "enabled_agents": [
        {
            "agent": "CODING_AGENT_CLAUDE_CODE",
            "config": {
                "use_as_global_settings": True,
                "custom_headers": {"x-databricks-workspace": "eng-ml-inference"},
                "tracing_config": {"table": "main.default.ucode_traces"},
                "model_config": {
                    "claude": {
                        "default_model": "system.ai.claude-opus-4-8",
                        "models": {
                            "default_opus_model": "system.ai.claude-opus-4-8",
                            "default_sonnet_model": "system.ai.claude-sonnet-4-6",
                            "default_haiku_model": "system.ai.claude-haiku-4-5",
                        },
                    }
                },
            },
        },
        {
            "agent": "CODING_AGENT_OPENCODE",
            "config": {
                "model_config": {
                    "opencode": {
                        "default_model": "system.ai.claude-opus-4-8",
                        "models": ["system.ai.claude-opus-4-8", "system.ai.kimi-k2-7-code"],
                    }
                }
            },
        },
    ],
    "mcp_servers": [
        {"name": "system.ai.github", "type": "MCP_SERVER_TYPE_UC_SERVICE"},
        {"name": "some-space-id", "type": "MCP_SERVER_TYPE_GENIE"},
    ],
    "skills": {"names": ["system.ai.pdf-extraction"]},
    "tracing": {"table": "main.default.ucode_traces"},
    "budget_policy": {
        "display_name": "paved-path",
        "budget_id": "c6563b45-df9a-4b19-afb2-d42dc2b52576",
        "tiers": [
            {
                "spending_percentage": 0.8,
                "default_agent": "CODING_AGENT_CLAUDE_CODE",
                "default_model": "system.ai.claude-sonnet-4-6",
            },
            {
                "spending_percentage": 1.0,
                "default_agent": "CODING_AGENT_OPENCODE",
                "default_model": "system.ai.kimi-k2-7-code",
            },
        ],
    },
}


class TestNormalize:
    def test_full_manifest_maps_enums_to_tool_names(self):
        cfg = normalize_managed_config(RAW_MANIFEST)
        assert cfg["name"] == "coding-agent-configs/abc-123"
        assert cfg["default_agent"] == "claude"
        assert set(cfg["enabled_agents"]) == {"claude", "opencode"}

    def test_claude_agent_config_fields(self):
        claude = normalize_managed_config(RAW_MANIFEST)["enabled_agents"]["claude"]
        assert claude["use_as_global_settings"] is True
        assert claude["custom_headers"] == {"x-databricks-workspace": "eng-ml-inference"}
        assert claude["tracing_table"] == "main.default.ucode_traces"
        assert claude["model_config"]["default_model"] == "system.ai.claude-opus-4-8"
        assert claude["model_config"]["models"]["default_opus_model"] == "system.ai.claude-opus-4-8"

    def test_opencode_model_list_is_flat(self):
        opencode = normalize_managed_config(RAW_MANIFEST)["enabled_agents"]["opencode"]
        assert opencode["model_config"]["models"] == [
            "system.ai.claude-opus-4-8",
            "system.ai.kimi-k2-7-code",
        ]

    def test_mcp_servers_map_type_enums_to_tags(self):
        mcp = normalize_managed_config(RAW_MANIFEST)["mcp_servers"]
        assert mcp == [
            {"name": "system.ai.github", "type": "mcp-service"},
            {"name": "some-space-id", "type": "genie-space"},
        ]

    def test_skills_and_tracing_and_budget(self):
        cfg = normalize_managed_config(RAW_MANIFEST)
        assert cfg["skills"] == {"names": ["system.ai.pdf-extraction"]}
        assert cfg["tracing_table"] == "main.default.ucode_traces"
        assert cfg["budget_policy"]["budget_id"] == "c6563b45-df9a-4b19-afb2-d42dc2b52576"
        assert cfg["budget_policy"]["tiers"][1]["default_agent"] == "opencode"

    @pytest.mark.parametrize("agent_enum", ["CODING_AGENT_FUTURE", "CODING_AGENT_UNSPECIFIED"])
    def test_unrecognized_agent_enum_dropped(self, agent_enum):
        raw = {"enabled_agents": [{"agent": agent_enum, "config": {}}]}
        assert "enabled_agents" not in normalize_managed_config(raw)

    def test_unknown_mcp_type_dropped(self):
        raw = {"mcp_servers": [{"name": "x", "type": "MCP_SERVER_TYPE_UNSPECIFIED"}]}
        assert "mcp_servers" not in normalize_managed_config(raw)

    def test_empty_manifest_yields_empty_dict(self):
        assert normalize_managed_config({}) == {}


class TestGetManagedConfig:
    def test_returns_normalized_first_config(self, monkeypatch):
        monkeypatch.setattr(
            mc_mod, "fetch_managed_coding_agent_configs", lambda ws, tok: ([RAW_MANIFEST], None)
        )
        cfg, reason = get_managed_config("https://ws", "tok")
        assert reason is None
        assert cfg["default_agent"] == "claude"

    def test_no_config_is_not_an_error(self, monkeypatch):
        monkeypatch.setattr(
            mc_mod, "fetch_managed_coding_agent_configs", lambda ws, tok: ([], None)
        )
        cfg, reason = get_managed_config("https://ws", "tok")
        assert cfg is None
        assert reason is None

    def test_fetch_failure_surfaces_reason(self, monkeypatch):
        monkeypatch.setattr(
            mc_mod,
            "fetch_managed_coding_agent_configs",
            lambda ws, tok: ([], "HTTP 500 Server Error"),
        )
        cfg, reason = get_managed_config("https://ws", "tok")
        assert cfg is None
        assert reason == "HTTP 500 Server Error"

    @pytest.mark.parametrize(
        "not_found_reason",
        [
            "HTTP 404 Not Found",
            'HTTP 404 Not Found: {"error_code":"NOT_FOUND","message":"..."}',
            'HTTP 400 Bad Request: {"error_code":"NOT_FOUND"}',
        ],
    )
    def test_not_found_is_treated_as_no_config(self, monkeypatch, not_found_reason):
        # A NOT_FOUND from the read means the admin hasn't defined a config — the normal
        # no-config case, not an error, so it collapses to (None, None).
        monkeypatch.setattr(
            mc_mod,
            "fetch_managed_coding_agent_configs",
            lambda ws, tok: ([], not_found_reason),
        )
        cfg, reason = get_managed_config("https://ws", "tok")
        assert cfg is None
        assert reason is None


class TestPersistence:
    @pytest.fixture(autouse=True)
    def _managed_path(self, tmp_path, monkeypatch):
        path = tmp_path / ".ucode" / "managed-state.json"
        monkeypatch.setattr(mc_mod, "MANAGED_STATE_PATH", path)
        return path

    def test_save_then_load_round_trips(self, _managed_path):
        cfg = normalize_managed_config(RAW_MANIFEST)
        save_managed_state("https://ws.example.com", cfg)
        loaded = load_managed_state("https://ws.example.com")
        assert loaded == cfg

    def test_saved_file_is_0600(self, _managed_path):
        save_managed_state("https://ws.example.com", {"default_agent": "claude"})
        mode = stat.S_IMODE(os.stat(_managed_path).st_mode)
        # Owner-only read/write; no group/other bits.
        assert mode == 0o600

    def test_load_ignores_other_workspace(self, _managed_path):
        save_managed_state("https://ws-a.example.com", {"default_agent": "claude"})
        assert load_managed_state("https://ws-b.example.com") is None

    def test_load_missing_returns_none(self, _managed_path):
        assert load_managed_state("https://ws.example.com") is None

    def test_load_none_workspace_returns_none(self, _managed_path):
        assert load_managed_state(None) is None

    def test_delete_removes_file(self, _managed_path):
        save_managed_state("https://ws.example.com", {"default_agent": "claude"})
        assert _managed_path.exists()
        mc_mod.delete_managed_state()
        assert not _managed_path.exists()

    def test_delete_missing_is_noop(self, _managed_path):
        mc_mod.delete_managed_state()  # should not raise


class TestFetchClient:
    """fetch_managed_coding_agent_configs lives in databricks.py; test its response parsing."""

    def test_extracts_configs_list(self, monkeypatch):
        payload = {"coding_agent_configs": [RAW_MANIFEST]}
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, timeout=10: (payload, None),
        )
        configs, reason = db_mod.fetch_managed_coding_agent_configs("https://ws", "tok")
        assert reason is None
        assert len(configs) == 1
        assert configs[0]["default_agent"] == "CODING_AGENT_CLAUDE_CODE"

    def test_empty_list_when_no_configs(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, timeout=10: ({}, None),
        )
        configs, reason = db_mod.fetch_managed_coding_agent_configs("https://ws", "tok")
        assert configs == []
        assert reason is None

    def test_http_failure_surfaces_reason(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, timeout=10: (None, "HTTP 403 Forbidden"),
        )
        configs, reason = db_mod.fetch_managed_coding_agent_configs("https://ws", "tok")
        assert configs == []
        assert reason == "HTTP 403 Forbidden"
