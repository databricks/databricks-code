"""Tests for managed_resolve.py / managed_apply.py — resolving and writing managed agent settings."""

from __future__ import annotations

import json

import pytest

import ucode.agents.claude as claude
import ucode.config_io as config_io
import ucode.state as state_mod
from ucode.managed_resolve import (
    effective_agent_models,
    managed_default_model,
    managed_provider_service,
    resolve_state,
)
from ucode.state import MANAGED_OVERLAY_KEY

WORKSPACE = "https://ws.example.com"

# A normalized managed config, as `managed_config.normalize_managed_config` produces it.
MANAGED = {
    "name": "coding-agent-configs/abc-123",
    "default_agent": "claude",
    "enabled_agents": {
        "claude": {
            "use_as_global_settings": True,
            "model_config": {
                "default_model": "system.ai.claude-opus-5",
                "models": {
                    "default_opus_model": "system.ai.claude-opus-5",
                    "default_sonnet_model": "system.ai.claude-sonnet-4-6",
                    "default_haiku_model": "system.ai.claude-haiku-4-5",
                },
            },
        },
        "codex": {
            "model_config": {
                "default_model": "databricks-gpt-5-3-codex",
                "models": ["databricks-gpt-5-3-codex", "databricks-gpt-5-2-codex"],
            }
        },
    },
    "budget_policy": {"display_name": "paved-path", "tiers": []},
}


def _state(**overrides) -> dict:
    state = {
        "workspace": WORKSPACE,
        "managed_configs": {"claude": {"keys": []}, "codex": {"keys": []}},
    }
    state.update(overrides)
    return state


class TestClaudeModels:
    def test_proto_slots_map_to_families(self):
        # The manifest keeps proto spelling (`default_opus_model`); render_overlay reads `opus`.
        models = effective_agent_models(MANAGED, _state(), "claude")
        assert models == {
            "opus": "system.ai.claude-opus-5",
            "sonnet": "system.ai.claude-sonnet-4-6",
            "haiku": "system.ai.claude-haiku-4-5",
        }

    def test_manifest_wins_over_local_per_family(self):
        state = _state(claude_models={"opus": "system.ai.claude-opus-4-8"})
        models = effective_agent_models(MANAGED, state, "claude")
        assert models["opus"] == "system.ai.claude-opus-5"

    def test_family_absent_from_manifest_keeps_local_value(self):
        # Claude resolves per family, so a family the admin didn't pin keeps the developer's choice.
        managed = {
            "enabled_agents": {
                "claude": {"model_config": {"models": {"default_opus_model": "managed-opus"}}}
            }
        }
        state = _state(claude_models={"opus": "local-opus", "fable": "local-fable"})
        models = effective_agent_models(managed, state, "claude")
        assert models == {"opus": "managed-opus", "fable": "local-fable"}

    def test_no_manifest_models_falls_back_to_local(self):
        state = _state(claude_models={"sonnet": "local-sonnet"})
        assert effective_agent_models({}, state, "claude") == {"sonnet": "local-sonnet"}

    def test_none_when_neither_side_has_models(self):
        assert effective_agent_models({}, _state(), "claude") is None


class TestListModels:
    def test_manifest_list_replaces_local(self):
        # A flat list has no per-key identity to merge on, so the manifest's list wins outright.
        state = _state(codex_models=["local-codex"])
        assert effective_agent_models(MANAGED, state, "codex") == [
            "databricks-gpt-5-3-codex",
            "databricks-gpt-5-2-codex",
        ]

    def test_local_list_stands_when_manifest_silent(self):
        state = _state(codex_models=["local-codex"])
        assert effective_agent_models({}, state, "codex") == ["local-codex"]

    def test_blank_entries_dropped(self):
        managed = {"enabled_agents": {"codex": {"model_config": {"models": ["  ", "real", ""]}}}}
        assert effective_agent_models(managed, _state(), "codex") == ["real"]


class TestManagedProviderService:
    """The manifest-only read: needed to attribute a provider to the admin, not to local state."""

    def test_returns_the_manifest_provider(self):
        managed = {
            "enabled_agents": {
                "claude": {"model_config": {"model_provider_service": "main.default.managed"}}
            }
        }
        assert managed_provider_service(managed, "claude") == "main.default.managed"

    def test_ignores_locally_persisted_provider(self):
        # No fallback to local state: otherwise a developer's own provider would be misreported as
        # the admin's when rejecting a conflicting --provider.
        assert managed_provider_service({}, "claude") is None

    def test_none_for_agent_not_in_manifest(self):
        assert managed_provider_service(MANAGED, "gemini") is None


class TestResolveState:
    def test_does_not_mutate_input_state(self):
        # managed-state.json and state.json stay separate files: resolution is per-write and
        # in-memory, so the developer's own state must come back untouched.
        state = _state(claude_models={"opus": "local-opus"})
        before = json.dumps(state, sort_keys=True)
        resolve_state(MANAGED, state, "claude")
        assert json.dumps(state, sort_keys=True) == before

    def test_layers_managed_models_onto_copy(self):
        resolved = resolve_state(MANAGED, _state(), "claude")
        assert resolved["claude_models"]["opus"] == "system.ai.claude-opus-5"

    def test_preserves_unrelated_state_keys(self):
        resolved = resolve_state(MANAGED, _state(profile="my-profile"), "claude")
        assert resolved["profile"] == "my-profile"
        assert resolved["workspace"] == WORKSPACE

    def test_layers_provider_without_dropping_other_tools(self):
        managed = {
            "enabled_agents": {
                "codex": {"model_config": {"model_provider_service": "main.default.managed"}}
            }
        }
        state = _state(provider_services={"claude": "main.default.keep"})
        resolved = resolve_state(managed, state, "codex")
        assert resolved["provider_services"] == {
            "claude": "main.default.keep",
            "codex": "main.default.managed",
        }


class TestStateFileIsNotRewritten:
    """The managed config must win by precedence, not by overwriting the developer's state file.

    managed-state.json and state.json stay separate on disk: resolution happens in memory and only
    the generated agent settings file reflects it. These tests deliberately let the real
    ``save_state`` run against a temp ``state.json`` — stubbing it out is what let this regress,
    because the overwrite happens inside ``write_tool_config``, one layer below the resolver.
    """

    @pytest.fixture
    def real_state_file(self, tmp_path, monkeypatch):
        """Redirect state.json and the claude settings file into tmp_path, unstubbed."""
        monkeypatch.setattr(config_io, "APP_DIR", tmp_path)
        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", tmp_path / "ucode-settings.json")
        monkeypatch.setattr(claude, "CLAUDE_BACKUP_PATH", tmp_path / "backup.json")
        # Seed a developer whose own opus choice differs from the manifest's.
        state_mod.save_state(
            {
                "workspace": WORKSPACE,
                "managed_configs": {"claude": {"keys": []}},
                "claude_models": {"opus": "system.ai.claude-opus-4-8"},
            }
        )
        return tmp_path

    @staticmethod
    def _persisted_claude_models(tmp_path) -> dict:
        full = json.loads((tmp_path / "state.json").read_text())
        return full["workspaces"][WORKSPACE].get("claude_models") or {}

    def test_developers_state_file_keeps_their_own_model(self, real_state_file):
        # The developer picked opus-4-8; the manifest says opus-5. After configuring under the
        # managed config, state.json must still say opus-4-8 — the admin's value belongs only in
        # the generated settings file, so removing the managed config restores their own choice.
        assert self._persisted_claude_models(real_state_file)["opus"] == "system.ai.claude-opus-4-8"

        resolved_state = resolve_state(MANAGED, state_mod.load_state(), "claude")
        claude.write_tool_config(resolved_state, None)

        assert self._persisted_claude_models(real_state_file)["opus"] == "system.ai.claude-opus-4-8"

    def test_settings_file_gets_the_managed_model(self, real_state_file):
        # The other half of the contract: precedence must actually reach the generated file.
        resolved_state = resolve_state(MANAGED, state_mod.load_state(), "claude")
        claude.write_tool_config(resolved_state, None)

        env = json.loads((real_state_file / "ucode-settings.json").read_text())["env"]
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"].startswith("system.ai.claude-opus-5")

    def test_overlay_bookkeeping_never_lands_on_disk(self, real_state_file):
        resolved_state = resolve_state(MANAGED, state_mod.load_state(), "claude")
        claude.write_tool_config(resolved_state, None)

        raw = (real_state_file / "state.json").read_text()
        assert MANAGED_OVERLAY_KEY not in raw

    def test_repeated_saves_still_restore_the_developers_value(self, real_state_file):
        # A launch can save twice from the same dict (the relayed proxy rewrites its port after
        # configure), so the swap-back must be idempotent rather than consuming the overlay.
        resolved_state = resolve_state(MANAGED, state_mod.load_state(), "claude")
        state_mod.save_state(resolved_state)
        state_mod.save_state(resolved_state)

        assert self._persisted_claude_models(real_state_file)["opus"] == "system.ai.claude-opus-4-8"
        # The in-memory dict still carries the managed value for rendering.
        assert resolved_state["claude_models"]["opus"] == "system.ai.claude-opus-5"

    def test_managed_provider_does_not_overwrite_the_developers_provider(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(config_io, "APP_DIR", tmp_path)
        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        state_mod.save_state(
            {"workspace": WORKSPACE, "provider_services": {"claude": "main.default.mine"}}
        )
        managed = {
            "enabled_agents": {
                "claude": {"model_config": {"model_provider_service": "main.default.admin"}}
            }
        }
        resolved_state = resolve_state(managed, state_mod.load_state(), "claude")
        state_mod.save_state(resolved_state)

        full = json.loads((tmp_path / "state.json").read_text())
        assert full["workspaces"][WORKSPACE]["provider_services"] == {"claude": "main.default.mine"}
        assert resolved_state["provider_services"]["claude"] == "main.default.admin"

    def test_developer_with_no_prior_value_is_not_given_one(self, tmp_path, monkeypatch):
        # The developer never configured claude models; the manifest supplies them for this launch
        # only, so state.json must not gain a key recording the admin's choice as theirs.
        monkeypatch.setattr(config_io, "APP_DIR", tmp_path)
        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        state_mod.save_state({"workspace": WORKSPACE})

        resolved_state = resolve_state(MANAGED, state_mod.load_state(), "claude")
        state_mod.save_state(resolved_state)

        full = json.loads((tmp_path / "state.json").read_text())
        assert not full["workspaces"][WORKSPACE].get("claude_models")

    @pytest.mark.parametrize(
        ("tool", "models_key", "managed_models"),
        [
            ("codex", "codex_models", ["managed-codex"]),
            ("gemini", "gemini_models", ["managed-gemini"]),
        ],
    )
    def test_other_agents_state_is_also_preserved(
        self, tmp_path, monkeypatch, tool, models_key, managed_models
    ):
        # Every agent's write_tool_config calls save_state, so the swap-back has to hold for all of
        # them — not just claude.
        monkeypatch.setattr(config_io, "APP_DIR", tmp_path)
        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        state_mod.save_state({"workspace": WORKSPACE, models_key: ["mine"]})
        managed = {"enabled_agents": {tool: {"model_config": {"models": managed_models}}}}

        resolved_state = resolve_state(managed, state_mod.load_state(), tool)
        state_mod.save_state(resolved_state)

        full = json.loads((tmp_path / "state.json").read_text())
        assert full["workspaces"][WORKSPACE][models_key] == ["mine"]
        assert resolved_state[models_key] == managed_models


class TestManagedDefaultModel:
    """The model a launch starts on, which is separate from the family slots."""

    def test_returns_the_manifest_default_model(self):
        assert managed_default_model(MANAGED, "claude") == "system.ai.claude-opus-5"

    def test_none_when_the_manifest_names_no_default(self):
        managed = {"enabled_agents": {"claude": {"model_config": {"models": {}}}}}
        assert managed_default_model(managed, "claude") is None

    def test_none_for_agent_not_in_manifest(self):
        assert managed_default_model({}, "codex") is None

    def test_survives_a_config_with_no_model_list(self):
        # CodexModelConfig has no `models` field at all, so default_model is the only model an
        # admin can set — it has to be usable on its own or a codex launch can't honor the config.
        managed = {"enabled_agents": {"codex": {"model_config": {"default_model": "admin-codex"}}}}
        state = {"workspace": WORKSPACE, "managed_configs": {"codex": {"keys": []}}}
        assert managed_default_model(managed, "codex") == "admin-codex"
        # Nothing lands in the model list, so the launch path must pass the default model into
        # resolve_launch_model rather than relying on state having one.
        assert resolve_state(managed, state, "codex").get("codex_models") is None
