"""Tests for agents/pi.py."""

from __future__ import annotations

import json
import threading
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from ucode.agents import pi

WS = "https://example.databricks.com"


def _base_urls() -> dict[str, str]:
    # Native API per family — see agents/pi.py docstring for path conventions.
    return {
        "claude": f"{WS}/ai-gateway/anthropic",
        "openai": f"{WS}/ai-gateway/codex/v1",
        "gemini": f"{WS}/ai-gateway/gemini/v1beta",
        "oss": f"{WS}/ai-gateway/mlflow/v1",
    }


def _empty() -> dict:
    """No-models input bundle for render_overlay."""
    return {
        "claude_models": {},
        "codex_models": [],
        "gemini_models": [],
        "oss_models": [],
        "oss_specs": [],
    }


def _overlay(model: str, token: str = "tok", **kwargs):
    """Wrapper to call render_overlay with sensible defaults so tests stay terse."""
    bundle = {**_empty(), **kwargs}
    return pi.render_overlay(
        model,
        token,
        _base_urls(),
        bundle["claude_models"],
        bundle["codex_models"],
        bundle["gemini_models"],
        bundle["oss_models"],
        bundle["oss_specs"],
    )


class TestPiSpec:
    def test_binary(self):
        assert pi.SPEC["binary"] == "pi"

    def test_package(self):
        assert pi.SPEC["package"] == "@earendil-works/pi-coding-agent"

    def test_display(self):
        assert pi.SPEC["display"] == "Pi"

    def test_config_path_under_pi_agent_dir(self):
        assert pi.SPEC["config_path"].name == "models.json"
        assert pi.SPEC["config_path"].parent.name == "agent"
        assert pi.PI_UCODE_HOME in pi.SPEC["config_path"].parents


class TestRenderOverlayProviders:
    def test_no_providers_when_no_models(self):
        overlay, _ = _overlay("foo")
        assert "providers" not in overlay

    def test_claude_provider_uses_anthropic_messages(self):
        overlay, _ = _overlay("claude-sonnet", claude_models={"sonnet": "claude-sonnet"})
        provider = overlay["providers"]["databricks-claude"]
        assert provider["api"] == "anthropic-messages"
        assert provider["baseUrl"] == f"{WS}/ai-gateway/anthropic"

    def test_openai_provider_uses_openai_responses(self):
        overlay, _ = _overlay("gpt-5", codex_models=["gpt-5"])
        provider = overlay["providers"]["databricks-openai"]
        assert provider["api"] == "openai-responses"
        assert provider["baseUrl"] == f"{WS}/ai-gateway/codex/v1"

    def test_gpt56_sol_model_entry_pins_1m_context(self):
        # Gateway ids are custom to Pi, so explicit metadata is required to
        # avoid its 128k custom-model default.
        overlay, _ = _overlay("gpt-5-6-sol", codex_models=["gpt-5-6-sol"])
        entry = overlay["providers"]["databricks-openai"]["models"][0]
        assert entry["id"] == "gpt-5-6-sol"
        assert entry["contextWindow"] == 1_050_000
        assert entry["maxTokens"] == 128_000
        assert entry["reasoning"] is True
        assert entry["input"] == ["text", "image"]

    def test_gpt_entries_pin_off_thinking_level_to_none(self):
        # `reasoning: True` without an off-state makes Pi send
        # `reasoning: {effort: "none"}`, which gpt-5 / -mini / -nano / -5-5-pro
        # reject with a 400. `{"off": None}` makes Pi omit `reasoning`.
        overlay, _ = _overlay(
            "system.ai.gpt-5",
            codex_models=[
                "system.ai.gpt-5",
                "system.ai.gpt-5-mini",
                "system.ai.gpt-5-nano",
                "system.ai.gpt-5-5-pro",
                "system.ai.gpt-5-6-luna",
            ],
        )
        entries = overlay["providers"]["databricks-openai"]["models"]
        assert entries, "expected gpt entries"
        for entry in entries:
            assert entry["reasoning"] is True
            assert entry["thinkingLevelMap"] == {"off": None}, entry["id"]

    def test_non_gpt_codex_entry_has_no_thinking_level_map(self):
        # Only the gpt-5 family declares `reasoning`, so only it needs the
        # off-state override.
        overlay, _ = _overlay("gpt-oss-120b", codex_models=["gpt-oss-120b"])
        entry = overlay["providers"]["databricks-openai"]["models"][0]
        assert "reasoning" not in entry
        assert "thinkingLevelMap" not in entry

    def test_gpt_model_entries_use_model_specific_windows(self):
        overlay, _ = _overlay(
            "system.ai.gpt-5-2",
            codex_models=[
                "system.ai.gpt-5-2",
                "databricks-gpt-5-4-nano",
                "databricks-gpt-5-6-sol",
            ],
        )
        windows = {
            m["id"]: m["contextWindow"] for m in overlay["providers"]["databricks-openai"]["models"]
        }
        assert windows == {
            "system.ai.gpt-5-2": 400_000,
            "databricks-gpt-5-4-nano": 400_000,
            "databricks-gpt-5-6-sol": 1_050_000,
        }

    def test_claude_entries_pin_limits_and_capabilities(self):
        overlay, _ = _overlay(
            "databricks-claude-opus-4-8",
            claude_models={
                "opus": "databricks-claude-opus-4-8",
                "sonnet": "system.ai.claude-sonnet-4-5",
                "haiku": "databricks-claude-haiku-4-5",
                "fable": "system.ai.claude-fable-5",
            },
        )
        entries = {m["id"]: m for m in overlay["providers"]["databricks-claude"]["models"]}
        opus = entries["databricks-claude-opus-4-8"]
        assert opus["contextWindow"] == 1_000_000
        assert opus["maxTokens"] == 128_000
        assert opus["reasoning"] is True
        assert opus["input"] == ["text", "image"]
        assert opus["compat"] == {"forceAdaptiveThinking": True}
        assert entries["system.ai.claude-sonnet-4-5"]["contextWindow"] == 1_000_000
        assert entries["system.ai.claude-sonnet-4-5"]["maxTokens"] == 64_000
        assert entries["databricks-claude-haiku-4-5"]["contextWindow"] == 200_000
        fable = entries["system.ai.claude-fable-5"]
        assert fable["contextWindow"] == 1_000_000
        assert fable["maxTokens"] == 128_000
        assert fable["compat"] == {"forceAdaptiveThinking": True}

    def test_gemini_provider_uses_google_generative_ai(self):
        overlay, _ = _overlay("gemini-2", gemini_models=["gemini-2"])
        provider = overlay["providers"]["databricks-gemini"]
        assert provider["api"] == "google-generative-ai"
        assert provider["baseUrl"] == f"{WS}/ai-gateway/gemini/v1beta"

    def test_mlflow_provider_uses_openai_completions(self):
        overlay, _ = _overlay("system.ai.glm-5-2", oss_models=["system.ai.glm-5-2"])
        provider = overlay["providers"]["databricks-mlflow"]
        assert provider["api"] == "openai-completions"
        assert provider["baseUrl"] == f"{WS}/ai-gateway/mlflow/v1"
        assert provider["compat"] == {"supportsStore": False, "supportsStrictMode": False}

    def test_no_mlflow_provider_when_no_oss_models(self):
        overlay, _ = _overlay("gpt-5", codex_models=["gpt-5"])
        assert "databricks-mlflow" not in overlay.get("providers", {})

    def test_all_four_providers_when_all_present(self):
        overlay, _ = _overlay(
            "claude-sonnet",
            claude_models={"sonnet": "claude-sonnet"},
            codex_models=["gpt-5"],
            gemini_models=["gemini-2"],
            oss_models=["system.ai.glm-5-2"],
        )
        assert set(overlay["providers"].keys()) == {
            "databricks-claude",
            "databricks-openai",
            "databricks-gemini",
            "databricks-mlflow",
        }


class TestRenderOverlayOssEnrichment:
    """OSS mlflow model entries carry reasoning + contextWindow + maxTokens
    from the shared databricks.model_token_limits / model_is_reasoning tables."""

    def test_reasoning_model_enriched(self):
        overlay, _ = _overlay("system.ai.glm-5-2", oss_models=["system.ai.glm-5-2"])
        entry = overlay["providers"]["databricks-mlflow"]["models"][0]
        assert entry["id"] == "system.ai.glm-5-2"
        assert entry["reasoning"] is True
        assert entry["contextWindow"] == 1_000_000
        assert entry["maxTokens"] == 65_536

    def test_unvalidated_model_has_no_inferred_metadata(self):
        # Discovery does not offer this model; even if supplied directly, Pi
        # must not infer capabilities for an unvalidated coding model.
        overlay, _ = _overlay("system.ai.inkling", oss_models=["system.ai.inkling"])
        entry = overlay["providers"]["databricks-mlflow"]["models"][0]
        assert entry == {"id": "system.ai.inkling"}

    def test_unknown_oss_model_bare(self):
        # No limits/reasoning table entry -> only id, client keeps defaults.
        overlay, _ = _overlay("system.ai.mystery-7b", oss_models=["system.ai.mystery-7b"])
        assert overlay["providers"]["databricks-mlflow"]["models"][0] == {
            "id": "system.ai.mystery-7b"
        }

    def test_dynamic_full_spec_overrides_static_metadata(self):
        specs = [
            {
                "id": "system.ai.glm-5-2",
                "reasoning": False,
                "context_window": 256_000,
                "max_tokens": 12_345,
            }
        ]
        overlay, _ = _overlay(
            "system.ai.glm-5-2", oss_models=["system.ai.glm-5-2"], oss_specs=specs
        )
        entry = overlay["providers"]["databricks-mlflow"]["models"][0]
        assert entry == {
            "id": "system.ai.glm-5-2",
            "contextWindow": 256_000,
            "maxTokens": 12_345,
        }

    def test_dynamic_reasoning_true_is_applied_with_safe_unknown_limits(self):
        specs = [
            {
                "id": "system.ai.inkling",
                "reasoning": True,
                "context_window": None,
                "max_tokens": None,
            }
        ]
        overlay, _ = _overlay(
            "system.ai.inkling", oss_models=["system.ai.inkling"], oss_specs=specs
        )
        assert overlay["providers"]["databricks-mlflow"]["models"][0] == {
            "id": "system.ai.inkling",
            "reasoning": True,
            "contextWindow": 128_000,
            "maxTokens": 8_192,
        }

    def test_partial_dynamic_limits_are_completed_conservatively(self):
        specs = [
            {
                "id": "system.ai.inkling",
                "reasoning": True,
                "context_window": None,
                "max_tokens": 65_536,
            }
        ]
        overlay, _ = _overlay(
            "system.ai.inkling", oss_models=["system.ai.inkling"], oss_specs=specs
        )
        entry = overlay["providers"]["databricks-mlflow"]["models"][0]
        assert entry["contextWindow"] == 128_000
        assert entry["maxTokens"] == 65_536

    def test_malformed_spec_is_ignored_safely(self):
        specs = [
            None,
            {"id": 12, "reasoning": True},
            {
                "id": "system.ai.mystery-7b",
                "reasoning": "yes",
                "context_window": -1,
                "max_tokens": True,
            },
        ]
        overlay, _ = _overlay(
            "system.ai.mystery-7b",
            oss_models=["system.ai.mystery-7b"],
            oss_specs=specs,
        )
        assert overlay["providers"]["databricks-mlflow"]["models"][0] == {
            "id": "system.ai.mystery-7b"
        }


class TestRenderOverlayUserAgent:
    def test_user_agent_set_on_all_three_providers(self, monkeypatch):
        monkeypatch.setattr(pi, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(pi, "agent_version", lambda binary: "0.74.0")
        overlay, _ = _overlay(
            "claude-sonnet",
            claude_models={"sonnet": "claude-sonnet"},
            codex_models=["gpt-5"],
            gemini_models=["gemini-2"],
        )
        expected = "ucode/0.1.0 pi/0.74.0"
        for name in ("databricks-claude", "databricks-openai", "databricks-gemini"):
            assert overlay["providers"][name]["headers"]["User-Agent"] == expected


class TestRenderOverlayCompatFlags:
    def test_claude_disables_eager_tool_input_streaming(self):
        # Gateway's Anthropic translator rejects per-tool
        # `eager_input_streaming`; this flag makes pi send the legacy beta
        # header instead.
        overlay, _ = _overlay("claude-sonnet", claude_models={"sonnet": "claude-sonnet"})
        compat = overlay["providers"]["databricks-claude"]["compat"]
        assert compat["supportsEagerToolInputStreaming"] is False

    def test_openai_and_gemini_have_no_compat_flags(self):
        # Their gateway routes accept pi's request shape as-is.
        overlay, _ = _overlay(
            "gpt-5",
            codex_models=["gpt-5"],
            gemini_models=["gemini-2"],
        )
        assert "compat" not in overlay["providers"]["databricks-openai"]
        assert "compat" not in overlay["providers"]["databricks-gemini"]


class TestRenderOverlayAuthAndModels:
    def test_token_in_api_key(self):
        overlay, _ = _overlay(
            "claude-sonnet", token="mytoken", claude_models={"sonnet": "claude-sonnet"}
        )
        assert overlay["providers"]["databricks-claude"]["apiKey"] == "mytoken"

    def test_auth_header_flag_set_on_all_providers(self):
        overlay, _ = _overlay(
            "claude-sonnet",
            claude_models={"sonnet": "claude-sonnet"},
            codex_models=["gpt-5"],
            gemini_models=["gemini-2"],
        )
        for name in ("databricks-claude", "databricks-openai", "databricks-gemini"):
            assert overlay["providers"][name]["authHeader"] is True

    def test_claude_models_listed(self):
        claude_models = {"opus": "claude-opus", "sonnet": "claude-sonnet"}
        overlay, _ = _overlay("claude-sonnet", claude_models=claude_models)
        ids = {m["id"] for m in overlay["providers"]["databricks-claude"]["models"]}
        assert ids == {"claude-opus", "claude-sonnet"}

    def test_openai_models_listed(self):
        overlay, _ = _overlay("gpt-5", codex_models=["gpt-5", "gpt-5-mini"])
        ids = {m["id"] for m in overlay["providers"]["databricks-openai"]["models"]}
        assert ids == {"gpt-5", "gpt-5-mini"}

    def test_gemini_models_listed(self):
        overlay, _ = _overlay("gemini-2", gemini_models=["gemini-2", "gemini-2-pro"])
        ids = {m["id"] for m in overlay["providers"]["databricks-gemini"]["models"]}
        assert ids == {"gemini-2", "gemini-2-pro"}


class TestRenderOverlayManagedKeys:
    def test_managed_keys_include_model(self):
        _, keys = _overlay("foo")
        assert ["model"] in keys

    def test_managed_keys_include_each_provider_present(self):
        _, keys = _overlay(
            "claude-sonnet",
            claude_models={"sonnet": "claude-sonnet"},
            codex_models=["gpt-5"],
            gemini_models=["gemini-2"],
        )
        for name in ("databricks-claude", "databricks-openai", "databricks-gemini"):
            assert ["providers", name] in keys


class TestRenderOverlayModelSelector:
    def test_prefixes_claude_model(self):
        overlay, _ = _overlay("claude-sonnet", claude_models={"sonnet": "claude-sonnet"})
        assert overlay["model"] == "databricks-claude/claude-sonnet"

    def test_prefixes_openai_model(self):
        overlay, _ = _overlay("gpt-5", codex_models=["gpt-5"])
        assert overlay["model"] == "databricks-openai/gpt-5"

    def test_prefixes_gemini_model(self):
        overlay, _ = _overlay("gemini-2", gemini_models=["gemini-2"])
        assert overlay["model"] == "databricks-gemini/gemini-2"

    def test_prefixes_oss_model(self):
        overlay, _ = _overlay("system.ai.glm-5-2", oss_models=["system.ai.glm-5-2"])
        assert overlay["model"] == "databricks-mlflow/system.ai.glm-5-2"

    def test_preserves_already_prefixed_model(self):
        overlay, _ = _overlay(
            "databricks-claude/claude-sonnet",
            claude_models={"sonnet": "claude-sonnet"},
        )
        assert overlay["model"] == "databricks-claude/claude-sonnet"

    def test_unknown_model_passes_through_unprefixed(self):
        # Lets a user override `model` to whatever pi accepts even if we
        # didn't classify it.
        overlay, _ = _overlay("custom/whatever")
        assert overlay["model"] == "custom/whatever"


class TestPiDefaultModel:
    def test_prefers_claude_opus(self):
        state = {"claude_models": {"opus": "o4", "sonnet": "s4", "haiku": "h4"}}
        assert pi.default_model(state) == "o4"

    def test_falls_back_to_sonnet(self):
        state = {"claude_models": {"sonnet": "s4"}}
        assert pi.default_model(state) == "s4"

    def test_falls_back_to_haiku(self):
        state = {"claude_models": {"haiku": "h4"}}
        assert pi.default_model(state) == "h4"

    def test_falls_back_to_newest_codex_model(self):
        state = {
            "claude_models": {},
            "codex_models": ["databricks-gpt-5", "system.ai.gpt-5-6-sol", "gpt-5-5"],
        }
        assert pi.default_model(state) == "system.ai.gpt-5-6-sol"

    def test_falls_back_to_generic_responses_endpoint(self):
        state = {"claude_models": {}, "codex_models": ["c1"]}
        assert pi.default_model(state) == "c1"

    def test_does_not_route_gpt_oss_to_responses(self):
        state = {
            "claude_models": {},
            "codex_models": ["gpt-oss-120b"],
            "gemini_models": ["gemini-2"],
        }
        assert pi.default_model(state) == "gemini-2"

    def test_falls_back_to_gemini(self):
        state = {"claude_models": {}, "codex_models": [], "gemini_models": ["gemini-2"]}
        assert pi.default_model(state) == "gemini-2"

    def test_falls_back_to_oss_last(self):
        state = {
            "claude_models": {},
            "codex_models": [],
            "gemini_models": [],
            "oss_models": ["system.ai.glm-5-2"],
        }
        assert pi.default_model(state) == "system.ai.glm-5-2"

    def test_returns_none_when_empty(self):
        assert pi.default_model({}) is None
        assert (
            pi.default_model({"claude_models": {}, "codex_models": [], "gemini_models": []}) is None
        )


class TestBuildRuntimeEnv:
    def test_sets_oauth_token(self):
        env = pi.build_runtime_env("tok")
        assert env["OAUTH_TOKEN"] == "tok"

    def test_sets_private_agent_dir_without_replacing_home(self, monkeypatch):
        monkeypatch.setenv("HOME", "/real-user-home")

        env = pi.build_runtime_env("tok")

        assert env["PI_CODING_AGENT_DIR"] == str(pi.PI_CONFIG_DIR)
        assert env["HOME"] == "/real-user-home"


class TestPiValidateCmd:
    def test_starts_with_binary(self):
        cmd = pi.validate_cmd("pi")
        assert cmd[0] == "pi"

    def test_uses_print_flag(self):
        # `--print` puts pi in non-interactive mode; without it the TUI hangs on stdin.
        cmd = pi.validate_cmd("pi")
        assert "--print" in cmd

    def test_has_prompt(self):
        cmd = pi.validate_cmd("pi")
        assert len(cmd) > 2


class TestWriteToolConfig:
    def _setup(self, tmp_path, monkeypatch):
        import ucode.agents.pi as pi_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "models.json"
        backup_file = tmp_path / "pi-backup.json"
        settings_file = tmp_path / "settings.json"
        settings_backup_file = tmp_path / "pi-settings-backup.json"
        monkeypatch.setattr(pi_mod, "PI_CONFIG_PATH", config_file)
        monkeypatch.setattr(pi_mod, "PI_SETTINGS_PATH", settings_file)
        monkeypatch.setattr(pi_mod, "PI_BACKUP_PATH", backup_file)
        monkeypatch.setattr(pi_mod, "PI_SETTINGS_BACKUP_PATH", settings_backup_file)
        return pi_mod, config_file, settings_file, settings_backup_file

    def _state(self, **overrides) -> dict:
        state = {
            "workspace": WS,
            "base_urls": {"pi": _base_urls()},
            "claude_models": {"sonnet": "claude-sonnet"},
            "codex_models": [],
            "gemini_models": [],
            "managed_configs": {},
        }
        state.update(overrides)
        return state

    def test_stale_managed_providers_removed_before_merge(self, tmp_path, monkeypatch):
        pi_mod, config_file, _, _ = self._setup(tmp_path, monkeypatch)

        stale = {
            "providers": {
                "databricks-claude": {"old": True},
                "databricks-openai": {"old": True},
                "databricks-gemini": {"old": True},
                "databricks-mlflow": {"old": True},
                "user-provider": {"keep": True},
            }
        }
        config_file.write_text(json.dumps(stale), encoding="utf-8")

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(self._state(), "claude-sonnet", token="tok")

        written = json.loads(config_file.read_text())
        providers = written.get("providers", {})
        assert providers.get("databricks-claude") != {"old": True}
        assert "old" not in providers.get("databricks-claude", {})
        assert "databricks-mlflow" not in providers
        assert providers.get("user-provider") == {"keep": True}

    def test_legacy_providers_removed_on_upgrade(self, tmp_path, monkeypatch):
        """Earlier ucode versions wrote `databricks-anthropic`, `databricks-codex`,
        and `databricks-oss` providers. They must be stripped on the next write
        so users don't end up with stale entries pointing at routes that 400."""
        pi_mod, config_file, _, _ = self._setup(tmp_path, monkeypatch)

        config_file.write_text(
            json.dumps(
                {
                    "providers": {
                        "databricks-anthropic": {"api": "anthropic-messages"},
                        "databricks-codex": {"api": "openai-responses"},
                        "databricks-oss": {"api": "openai-completions"},
                    }
                }
            ),
            encoding="utf-8",
        )

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(self._state(), "claude-sonnet", token="tok")

        written_providers = json.loads(config_file.read_text()).get("providers", {})
        for legacy in ("databricks-anthropic", "databricks-codex", "databricks-oss"):
            assert legacy not in written_providers
        assert "databricks-claude" in written_providers

    def test_config_written_with_correct_model_and_token(self, tmp_path, monkeypatch):
        pi_mod, config_file, _, _ = self._setup(tmp_path, monkeypatch)

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(self._state(), "claude-sonnet", token="tok")

        written = json.loads(config_file.read_text())
        assert written["model"] == "databricks-claude/claude-sonnet"
        assert written["providers"]["databricks-claude"]["apiKey"] == "tok"

    def test_state_oss_specs_reach_written_model_entry(self, tmp_path, monkeypatch):
        pi_mod, config_file, _, _ = self._setup(tmp_path, monkeypatch)
        state = self._state(
            claude_models={},
            oss_models=["system.ai.inkling"],
            oss_model_specs=[
                {
                    "id": "system.ai.inkling",
                    "reasoning": True,
                    "context_window": 256_000,
                    "max_tokens": 65_536,
                }
            ],
        )

        with patch("ucode.agents.pi.save_state"):
            pi_mod.write_tool_config(state, "system.ai.inkling", token="tok")

        entry = json.loads(config_file.read_text())["providers"]["databricks-mlflow"]["models"][0]
        assert entry["reasoning"] is True
        assert entry["contextWindow"] == 256_000
        assert entry["maxTokens"] == 65_536

    def test_settings_pins_default_provider_and_model(self, tmp_path, monkeypatch):
        # Without this, Pi's `findInitialModel` can fall through to a built-in
        # provider when an unrelated env var (e.g. HF_TOKEN) makes one look
        # auth-configured. Pinning the default keeps Pi on our provider.
        pi_mod, _, settings_file, _ = self._setup(tmp_path, monkeypatch)

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(self._state(), "claude-sonnet", token="tok")

        settings = json.loads(settings_file.read_text())
        assert settings["defaultProvider"] == "databricks-claude"
        assert settings["defaultModel"] == "claude-sonnet"

    def test_pre_existing_settings_are_backed_up_before_first_write(self, tmp_path, monkeypatch):
        pi_mod, _, settings_file, settings_backup_file = self._setup(tmp_path, monkeypatch)

        original = '{"theme": "Default Dark", "defaultProvider": "openai"}'
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(original, encoding="utf-8")

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(self._state(), "claude-sonnet", token="tok")

        assert settings_backup_file.read_text(encoding="utf-8") == original
        # The on-disk settings still get the ucode pin applied via deep_merge.
        merged = json.loads(settings_file.read_text())
        assert merged["defaultProvider"] == "databricks-claude"
        assert merged["theme"] == "Default Dark"


class TestValidateAllToolsPiRollback:
    def test_failed_pi_validation_rolls_back_settings(self, tmp_path, monkeypatch):
        import ucode.agents as agents_mod
        import ucode.agents.pi as pi_mod

        settings_file = tmp_path / "settings.json"
        settings_file.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(pi_mod, "PI_SETTINGS_PATH", settings_file)
        monkeypatch.setattr(pi_mod, "PI_SETTINGS_BACKUP_PATH", tmp_path / "settings.backup.json")
        # Keep the generic models.json rollback off the user's real config dir.
        monkeypatch.setitem(agents_mod.TOOL_SPECS["pi"], "config_path", tmp_path / "models.json")
        monkeypatch.setitem(
            agents_mod.TOOL_SPECS["pi"], "backup_path", tmp_path / "models.backup.json"
        )
        monkeypatch.setattr(agents_mod, "validate_tool", lambda tool: (False, "boom"))
        monkeypatch.setattr(agents_mod, "save_state", lambda s: None)
        monkeypatch.setattr(agents_mod, "spinner", lambda *_a, **_kw: nullcontext())

        agents_mod.validate_all_tools({"available_tools": ["pi"], "managed_configs": {"pi": True}})

        assert not settings_file.exists()


class TestManagedModels:
    """A managed config's models arrive as `pi_models` and must not come from the shared keys."""

    def test_managed_models_win_over_the_shared_discovery_lists(self):
        state = {
            "pi_models": ["system.ai.claude-opus-4-8"],
            "claude_models": {"opus": "shared-should-not-win"},
        }
        assert pi.default_model(state) == "system.ai.claude-opus-4-8"

    def test_falls_back_to_the_shared_lists_without_a_managed_config(self):
        assert pi.default_model({"claude_models": {"opus": "discovered"}}) == "discovered"

    def test_managed_models_split_into_pis_per_provider_inputs(self):
        # Pi builds one provider block per family, so a flat list has to be classified back out.
        state = {
            "pi_models": [
                "system.ai.claude-opus-4-8",
                "system.ai.gpt-5",
                "system.ai.gemini-3-flash",
            ]
        }
        assert pi._managed_model_families(state) == (
            {"opus": "system.ai.claude-opus-4-8"},
            ["system.ai.gpt-5"],
            ["system.ai.gemini-3-flash"],
        )

    def test_no_split_without_managed_models(self):
        assert pi._managed_model_families({"claude_models": {"opus": "x"}}) is None

    def test_none_when_no_managed_model_is_servable(self):
        # Pi has no OSS provider, so an oss-only list yields no families. Returning an all-empty
        # tuple would be truthy and suppress the fallback, writing a config with zero providers.
        assert pi._managed_model_families({"pi_models": ["system.ai.kimi-k2-7-code"]}) is None

    def test_partially_servable_list_still_splits(self):
        families = pi._managed_model_families(
            {"pi_models": ["system.ai.kimi-k2-7-code", "system.ai.claude-opus-4-8"]}
        )
        assert families == ({"opus": "system.ai.claude-opus-4-8"}, [], [])


class TestMlflowProxyLifecycle:
    def test_not_started_without_oss_models(self):
        state = {"workspace": WS, "oss_models": [], "base_urls": {"pi": _base_urls()}}
        with patch.object(pi._mlflow_proxy, "start") as start:
            assert pi._start_oss_proxy(state) is None
        start.assert_not_called()

    def test_stale_loopback_url_is_replaced_and_real_workspace_is_upstream(self):
        server = MagicMock()
        state = {
            "workspace": WS,
            "oss_models": ["system.ai.inkling"],
            "base_urls": {
                "pi": {**_base_urls(), "oss": "http://127.0.0.1:54321/ai-gateway/mlflow/v1"}
            },
        }
        with patch.object(
            pi._mlflow_proxy,
            "start",
            return_value=(server, "http://127.0.0.1:60000"),
        ) as start:
            running = pi._start_oss_proxy(state)
        assert running is not None
        start.assert_called_once_with(WS)
        assert state["base_urls"]["pi"]["oss"] == ("http://127.0.0.1:60000/ai-gateway/mlflow/v1")

    def test_loopback_workspace_is_not_recursively_proxied(self):
        state = {
            "workspace": "http://127.0.0.1:9999",
            "oss_models": ["system.ai.inkling"],
        }
        with patch.object(pi._mlflow_proxy, "start") as start:
            assert pi._start_oss_proxy(state) is None
        start.assert_not_called()

    @staticmethod
    def _proxy_pair():
        proxy_thread = MagicMock()
        server = MagicMock()
        return (proxy_thread, server), proxy_thread, server

    def test_restore_rewrites_persistent_config_to_direct_gateway(self):
        state = {
            "workspace": WS,
            "oss_models": ["system.ai.inkling"],
            "base_urls": {
                "pi": {**_base_urls(), "oss": "http://127.0.0.1:54321/ai-gateway/mlflow/v1"}
            },
        }
        with patch.object(pi, "write_tool_config") as write:
            pi._restore_direct_oss_config(state, "tok")
        assert state["base_urls"]["pi"]["oss"] == f"{WS}/ai-gateway/mlflow/v1"
        write.assert_called_once_with(state, "system.ai.inkling", token="tok")

    def test_restore_without_token_clears_state_and_existing_config_url(
        self, tmp_path, monkeypatch
    ):
        config_path = tmp_path / "models.json"
        config_path.write_text(
            json.dumps(
                {
                    "providers": {
                        "databricks-mlflow": {
                            "baseUrl": "http://127.0.0.1:54321/ai-gateway/mlflow/v1",
                            "apiKey": "existing-token",
                        }
                    }
                }
            )
        )
        monkeypatch.setattr(pi, "PI_CONFIG_PATH", config_path)
        state = {
            "workspace": WS,
            "oss_models": ["system.ai.inkling"],
            "base_urls": {
                "pi": {**_base_urls(), "oss": "http://127.0.0.1:54321/ai-gateway/mlflow/v1"}
            },
        }
        with patch.object(pi, "write_tool_config") as write:
            pi._restore_direct_oss_config(state, None)
        assert state["base_urls"]["pi"]["oss"] == f"{WS}/ai-gateway/mlflow/v1"
        write.assert_not_called()
        restored = json.loads(config_path.read_text())
        assert restored["providers"]["databricks-mlflow"]["baseUrl"] == (
            f"{WS}/ai-gateway/mlflow/v1"
        )
        assert restored["providers"]["databricks-mlflow"]["apiKey"] == "existing-token"

    def test_refresh_keeps_live_proxy_url(self, tmp_path, monkeypatch):
        config_path = tmp_path / "models.json"
        settings_path = tmp_path / "settings.json"
        monkeypatch.setattr(pi, "PI_CONFIG_PATH", config_path)
        monkeypatch.setattr(pi, "PI_SETTINGS_PATH", settings_path)
        monkeypatch.setattr(pi, "PI_BACKUP_PATH", tmp_path / "models.backup.json")
        monkeypatch.setattr(pi, "PI_SETTINGS_BACKUP_PATH", tmp_path / "settings.backup.json")
        state = {
            "workspace": WS,
            "oss_models": ["system.ai.inkling"],
            "base_urls": {
                "pi": {**_base_urls(), "oss": "http://127.0.0.1:54321/ai-gateway/mlflow/v1"}
            },
        }
        with (
            patch.object(pi, "get_databricks_token", return_value="refreshed-token"),
            patch.object(pi, "save_state"),
        ):
            pi._refresh_token_once(state, force_refresh=True)
        provider = json.loads(config_path.read_text())["providers"]["databricks-mlflow"]
        assert provider["baseUrl"] == "http://127.0.0.1:54321/ai-gateway/mlflow/v1"
        assert provider["apiKey"] == "refreshed-token"

    def test_restore_waits_for_inflight_refresh_and_writes_direct_last(self, monkeypatch):
        state = {
            "workspace": WS,
            "oss_models": ["system.ai.inkling"],
            "base_urls": {
                "pi": {**_base_urls(), "oss": "http://127.0.0.1:54321/ai-gateway/mlflow/v1"}
            },
        }
        refresh_started = threading.Event()
        release_refresh = threading.Event()
        restore_done = threading.Event()
        write_order: list[str] = []

        def fake_write(current, model, token=None, *, force_refresh=False):
            write_order.append(str(token))
            if token == "refresh-token":
                refresh_started.set()
                assert release_refresh.wait(timeout=2)
            return current, str(token)

        monkeypatch.setattr(pi, "_write_tool_config_unlocked", fake_write)
        refresher = threading.Thread(
            target=pi.write_tool_config,
            args=(state, "system.ai.inkling", "refresh-token"),
        )
        refresher.start()
        assert refresh_started.wait(timeout=2)
        restorer = threading.Thread(
            target=lambda: (
                pi._restore_direct_oss_config(state, "restore-token"),
                restore_done.set(),
            )
        )
        restorer.start()
        assert not restore_done.wait(timeout=0.05)
        release_refresh.set()
        refresher.join(timeout=2)
        restorer.join(timeout=2)
        assert not refresher.is_alive()
        assert not restorer.is_alive()
        assert write_order == ["refresh-token", "restore-token"]
        assert state["base_urls"]["pi"]["oss"] == f"{WS}/ai-gateway/mlflow/v1"

    def test_proxy_precedes_first_config_write_and_is_cleaned_on_normal_exit(self):
        proxy, proxy_thread, server = self._proxy_pair()
        state = {"workspace": WS, "oss_models": ["system.ai.inkling"]}
        order = []

        def start(current):
            current.setdefault("base_urls", {}).setdefault("pi", {})["oss"] = "http://live"
            order.append("proxy")
            return proxy

        def refresh(current, *, force_refresh=False):
            assert current["base_urls"]["pi"]["oss"] == "http://live"
            order.append("config")
            return "tok"

        proc = MagicMock()
        proc.wait.return_value = 0
        with (
            patch.object(pi, "_start_oss_proxy", side_effect=start),
            patch.object(pi, "_refresh_token_once", side_effect=refresh),
            patch.object(pi, "_refresh_forever", return_value=None),
            patch.object(pi, "_restore_direct_oss_config") as restore,
            patch.object(pi.subprocess, "Popen", return_value=proc),
            pytest.raises(SystemExit) as exit_info,
        ):
            pi.launch(state, [])
        assert exit_info.value.code == 0
        assert order[:2] == ["proxy", "config"]
        restore.assert_called_once_with(state, "tok")
        server.shutdown.assert_called_once()
        server.server_close.assert_called_once()
        proxy_thread.join.assert_called_once_with(timeout=1)

    def test_interrupt_forwards_sigint_and_cleans_proxy(self):
        proxy, _, server = self._proxy_pair()
        proc = MagicMock()
        proc.wait.side_effect = [KeyboardInterrupt, 130]
        state = {"workspace": WS, "oss_models": ["system.ai.inkling"]}
        with (
            patch.object(pi, "_start_oss_proxy", return_value=proxy),
            patch.object(pi, "_refresh_token_once", return_value="tok"),
            patch.object(pi, "_refresh_forever", return_value=None),
            patch.object(pi, "_restore_direct_oss_config") as restore,
            patch.object(pi.subprocess, "Popen", return_value=proc),
            pytest.raises(SystemExit) as exit_info,
        ):
            pi.launch(state, [])
        assert exit_info.value.code == 130
        proc.send_signal.assert_called_once_with(pi.signal.SIGINT)
        restore.assert_called_once_with(state, "tok")
        server.shutdown.assert_called_once()
        server.server_close.assert_called_once()

    @pytest.mark.parametrize("failure_stage", ["config", "popen"])
    def test_setup_failure_still_cleans_proxy(self, failure_stage):
        proxy, _, server = self._proxy_pair()
        state = {"workspace": WS, "oss_models": ["system.ai.inkling"]}
        refresh = MagicMock(return_value="tok")
        popen = MagicMock(return_value=MagicMock())
        if failure_stage == "config":
            refresh.side_effect = RuntimeError("token failed")
        else:
            popen.side_effect = OSError("binary missing")
        with (
            patch.object(pi, "_start_oss_proxy", return_value=proxy),
            patch.object(pi, "_refresh_token_once", refresh),
            patch.object(pi, "_refresh_forever", return_value=None),
            patch.object(pi, "_restore_direct_oss_config") as restore,
            patch.object(pi.subprocess, "Popen", popen),
            pytest.raises((RuntimeError, OSError)) as exc_info,
        ):
            pi.launch(state, [])
        expected = "token failed" if failure_stage == "config" else "binary missing"
        assert str(exc_info.value) == expected
        expected_token = None if failure_stage == "config" else "tok"
        restore.assert_called_once_with(state, expected_token)
        server.shutdown.assert_called_once()
        server.server_close.assert_called_once()

    def test_setup_failure_remains_primary_when_restore_also_fails(self):
        proxy, _, server = self._proxy_pair()
        state = {"workspace": WS, "oss_models": ["system.ai.inkling"]}
        with (
            patch.object(pi, "_start_oss_proxy", return_value=proxy),
            patch.object(pi, "_refresh_token_once", return_value="tok"),
            patch.object(pi, "_refresh_forever", return_value=None),
            patch.object(
                pi, "_restore_direct_oss_config", side_effect=RuntimeError("restore failed")
            ),
            patch.object(pi, "print_warning") as warning,
            patch.object(pi.subprocess, "Popen", side_effect=OSError("binary missing")),
            pytest.raises(OSError, match="binary missing"),
        ):
            pi.launch(state, [])
        warning.assert_called_once()
        server.shutdown.assert_called_once()
        server.server_close.assert_called_once()

    def test_restore_failure_does_not_skip_proxy_shutdown(self):
        proxy, _, server = self._proxy_pair()
        proc = MagicMock()
        proc.wait.return_value = 0
        state = {"workspace": WS, "oss_models": ["system.ai.inkling"]}
        with (
            patch.object(pi, "_start_oss_proxy", return_value=proxy),
            patch.object(pi, "_refresh_token_once", return_value="tok"),
            patch.object(pi, "_refresh_forever", return_value=None),
            patch.object(pi, "_restore_direct_oss_config", side_effect=OSError("restore failed")),
            patch.object(pi.subprocess, "Popen", return_value=proc),
            pytest.raises(OSError, match="restore failed"),
        ):
            pi.launch(state, [])
        server.shutdown.assert_called_once()
        server.server_close.assert_called_once()


class TestManagedDefaultModel:
    """A managed config's `pi_default_model` takes priority over the allowlist."""

    def test_pi_default_model_wins_over_allowlist(self):
        state = {
            "pi_default_model": "admin-chosen-default",
            "pi_models": ["system.ai.claude-opus-4-8", "system.ai.gpt-5"],
        }
        assert pi.default_model(state) == "admin-chosen-default"

    def test_falls_back_to_pi_models_without_default(self):
        state = {"pi_models": ["system.ai.claude-opus-4-8"]}
        assert pi.default_model(state) == "system.ai.claude-opus-4-8"
