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
    managed_launch_state,
    normalize_managed_config,
    refresh_managed_config,
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

    def test_empty_config_overwrites_a_previous_one(self, _managed_path):
        # Saving an empty config is how "the admin removed it" is recorded: the stored config must
        # be replaced, not left behind for the read-failure fallback to reapply.
        save_managed_state("https://ws.example.com", {"default_agent": "claude"})
        save_managed_state("https://ws.example.com", {})
        assert load_managed_state("https://ws.example.com") == {}


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


WORKSPACE = "https://ws.example.com"

# A normalized managed config, as `normalize_managed_config` produces it.
MANAGED = {
    "default_agent": "claude",
    "enabled_agents": {
        "claude": {
            "model_config": {
                "default_model": "system.ai.claude-opus-5",
                "models": {"default_opus_model": "system.ai.claude-opus-5"},
            }
        }
    },
}


def _state(**overrides) -> dict:
    state = {"workspace": WORKSPACE, "managed_configs": {"claude": {"keys": []}}}
    state.update(overrides)
    return state


class TestRefreshManagedConfig:
    """The per-launch re-read, so an admin's edits land without re-running `ucode configure`."""

    @pytest.fixture(autouse=True)
    def _stub_token(self, monkeypatch):
        monkeypatch.setattr(mc_mod, "get_databricks_token", lambda ws, profile: "tok")

    def test_persists_and_returns_the_manifest(self, monkeypatch):
        saved: list[tuple] = []
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (MANAGED, None))
        monkeypatch.setattr(mc_mod, "save_managed_state", lambda ws, cfg: saved.append((ws, cfg)))
        assert refresh_managed_config(_state()) == MANAGED
        assert saved == [(WORKSPACE, MANAGED)]

    def test_no_managed_config_returns_none(self, monkeypatch):
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, None))
        monkeypatch.setattr(mc_mod, "save_managed_state", lambda ws, cfg: None)
        assert refresh_managed_config(_state()) is None

    def test_read_failure_falls_back_to_the_persisted_config(self, monkeypatch):
        # The admin's last known policy beats no policy, so a failed fetch reuses what we saved.
        warnings: list[str] = []
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, "HTTP 500"))
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: MANAGED)
        monkeypatch.setattr(mc_mod, "print_warning", lambda msg: warnings.append(msg))
        assert refresh_managed_config(_state()) == MANAGED
        assert "HTTP 500" in warnings[0]
        assert "last one saved" in warnings[0]

    def test_read_failure_without_persisted_config_is_silent(self, monkeypatch):
        # Nothing persisted means no managed config is in play, so an expired session shouldn't
        # produce a warning about a feature this developer doesn't use.
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, "HTTP 500"))
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: None)
        monkeypatch.setattr(
            mc_mod, "print_warning", lambda msg: pytest.fail(f"should not warn: {msg}")
        )
        assert refresh_managed_config(_state()) is None

    def test_auth_failure_falls_back_to_the_persisted_config(self, monkeypatch):
        warnings: list[str] = []

        def boom(ws, profile):
            raise RuntimeError("no token")

        monkeypatch.setattr(mc_mod, "get_databricks_token", boom)
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: MANAGED)
        monkeypatch.setattr(mc_mod, "print_warning", lambda msg: warnings.append(msg))
        assert refresh_managed_config(_state()) == MANAGED
        assert "no token" in warnings[0]

    def test_auth_failure_without_persisted_config_is_silent(self, monkeypatch):
        def boom(ws, profile):
            raise RuntimeError("no token")

        monkeypatch.setattr(mc_mod, "get_databricks_token", boom)
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: None)
        monkeypatch.setattr(
            mc_mod, "print_warning", lambda msg: pytest.fail(f"should not warn: {msg}")
        )
        assert refresh_managed_config(_state()) is None

    def test_permission_denied_without_cache_is_silent(self, monkeypatch):
        # A refusal is no evidence a config exists, so with nothing cached there is no managed
        # config in play and warning would be a false positive.
        denied = 'HTTP 403 Forbidden: {"error_code":"PERMISSION_DENIED"}'
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, denied))
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: None)
        monkeypatch.setattr(
            mc_mod, "print_warning", lambda msg: pytest.fail(f"should not warn: {msg}")
        )
        assert refresh_managed_config(_state()) is None

    def test_permission_denied_warns_and_keeps_the_cached_config(self, monkeypatch):
        # A refused read is worth surfacing: an admin may have published a config that isn't
        # reaching this developer. It says nothing about whether one exists, so the cache stands.
        warnings: list[str] = []
        denied = 'HTTP 403 Forbidden: {"error_code":"PERMISSION_DENIED"}'
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, denied))
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: MANAGED)
        monkeypatch.setattr(mc_mod, "print_warning", lambda msg: warnings.append(msg))
        monkeypatch.setattr(
            mc_mod, "save_managed_state", lambda ws, cfg: pytest.fail("must not clear the cache")
        )
        assert refresh_managed_config(_state()) == MANAGED
        assert "not readable by you" in warnings[0]

    def test_no_config_on_the_server_does_not_use_a_stale_persisted_file(self, monkeypatch):
        # A successful read saying "no config" means the admin removed it — that's authoritative,
        # so a previously persisted file must not resurrect the old policy.
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, None))
        monkeypatch.setattr(mc_mod, "save_managed_state", lambda ws, cfg: None)
        monkeypatch.setattr(
            mc_mod, "load_managed_state", lambda ws: pytest.fail("must not fall back")
        )
        assert refresh_managed_config(_state()) is None

    def test_no_config_on_the_server_clears_the_persisted_one(self, monkeypatch):
        # Without this, removing the config server-side would leave the old one on disk and the next
        # failed read would put a dead policy back into force.
        saved: list[tuple] = []
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, None))
        monkeypatch.setattr(mc_mod, "save_managed_state", lambda ws, cfg: saved.append((ws, cfg)))
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: None)
        assert refresh_managed_config(_state()) is None
        assert saved == [(WORKSPACE, {})]

    def test_empty_persisted_config_is_not_treated_as_a_fallback(self, monkeypatch):
        # The empty marker means "no admin policy", so a later failed read falls through to the
        # developer's own settings rather than reporting a managed config.
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, "HTTP 500"))
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: {})
        monkeypatch.setattr(
            mc_mod, "print_warning", lambda msg: pytest.fail(f"should not warn: {msg}")
        )
        assert refresh_managed_config(_state()) is None

    def test_no_workspace_is_a_noop(self, monkeypatch):
        monkeypatch.setattr(
            mc_mod, "get_managed_config", lambda ws, tok: pytest.fail("should not fetch")
        )
        assert refresh_managed_config({}) is None


class TestManagedLaunchState:
    @pytest.fixture(autouse=True)
    def _stub_token(self, monkeypatch):
        monkeypatch.setattr(mc_mod, "get_databricks_token", lambda ws, profile: "tok")
        monkeypatch.setattr(mc_mod, "save_managed_state", lambda ws, cfg: None)
        monkeypatch.setenv(mc_mod.MANAGED_CONFIG_ENV_VAR, "1")

    def test_layers_managed_models_when_a_config_exists(self, monkeypatch):
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (MANAGED, None))
        state = _state(claude_models={"opus": "local-opus"})
        resolved, managed = managed_launch_state(state, "claude")
        assert managed == MANAGED
        assert resolved["claude_models"]["opus"] == "system.ai.claude-opus-5"
        # The developer's own state is untouched — precedence is resolved in memory.
        assert state["claude_models"]["opus"] == "local-opus"

    def test_state_untouched_when_no_managed_config(self, monkeypatch):
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, None))
        state = _state(claude_models={"opus": "local-opus"})
        resolved, managed = managed_launch_state(state, "claude")
        assert managed is None
        assert resolved is state

    @pytest.mark.parametrize("env_value", [None, "", "0", "off", "no"])
    def test_disabled_does_nothing_at_all(self, monkeypatch, env_value):
        """While the feature is opt-in, a disabled launch must behave exactly as it did before.

        Every side effect the managed path can have is trip-wired, so this fails if any future
        change reaches the network, the cache, or the developer's state without the env var set.
        """
        if env_value is None:
            monkeypatch.delenv(mc_mod.MANAGED_CONFIG_ENV_VAR, raising=False)
        else:
            monkeypatch.setenv(mc_mod.MANAGED_CONFIG_ENV_VAR, env_value)
        for name in (
            "get_databricks_token",
            "fetch_managed_coding_agent_configs",
            "get_managed_config",
            "load_managed_state",
            "save_managed_state",
            "resolve_state",
            "print_warning",
        ):
            monkeypatch.setattr(
                mc_mod,
                name,
                lambda *a, called=name, **k: pytest.fail(f"{called} must not run when disabled"),
            )

        assert mc_mod.managed_agent_config_enabled() is False
        state = _state(claude_models={"opus": "local-opus"})
        resolved, managed = managed_launch_state(state, "claude")
        assert managed is None
        # Same object back, so nothing downstream can see a layered value.
        assert resolved is state
        assert state["claude_models"] == {"opus": "local-opus"}

    @pytest.mark.parametrize("env_value", ["1", "true", "TRUE", "yes"])
    def test_enabled_values(self, monkeypatch, env_value):
        monkeypatch.setenv(mc_mod.MANAGED_CONFIG_ENV_VAR, env_value)
        assert mc_mod.managed_agent_config_enabled() is True

    def test_skip_preflight_reads_the_cache_without_fetching(self, monkeypatch):
        # Headless launchers pass --skip-preflight to avoid per-launch network calls, so the config
        # comes from the last persisted copy rather than a fresh read.
        monkeypatch.setattr(
            mc_mod, "get_managed_config", lambda ws, tok: pytest.fail("should not fetch")
        )
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: MANAGED)
        resolved, managed = managed_launch_state(_state(), "claude", skip_preflight=True)
        assert managed == MANAGED
        assert resolved["claude_models"]["opus"] == "system.ai.claude-opus-5"

    def test_skip_preflight_with_no_cache_is_a_noop(self, monkeypatch):
        monkeypatch.setattr(
            mc_mod, "get_managed_config", lambda ws, tok: pytest.fail("should not fetch")
        )
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: None)
        state = _state()
        resolved, managed = managed_launch_state(state, "claude", skip_preflight=True)
        assert managed is None
        assert resolved is state
