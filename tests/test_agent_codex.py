"""Tests for agents/codex.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ucode.agents import codex
from ucode.config_io import read_toml_safe
from ucode.smart_routing import codex_routing

WS = "https://example.databricks.com"


class TestCodexSpec:
    def test_binary(self):
        assert codex.SPEC["binary"] == "codex"

    def test_package(self):
        assert codex.SPEC["package"] == "@openai/codex"

    def test_display(self):
        assert codex.SPEC["display"] == "Codex"


class TestHasUcodeConfig:
    def test_detects_profile_config(self, tmp_path, monkeypatch):
        config_path = tmp_path / "ucode.config.toml"
        legacy_path = tmp_path / "config.toml"
        legacy_path.write_text(
            'profile = "ucode"\n\n[profiles.ucode]\nmodel_provider = "ucode-databricks"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "LEGACY_CODEX_CONFIG_PATH", legacy_path)

        assert codex.has_ucode_config() is True

    def test_ignores_unrelated_legacy_config(self, tmp_path, monkeypatch):
        config_path = tmp_path / "ucode.config.toml"
        legacy_path = tmp_path / "config.toml"
        legacy_path.write_text('profile = "default"\n', encoding="utf-8")
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "LEGACY_CODEX_CONFIG_PATH", legacy_path)

        assert codex.has_ucode_config() is False


class TestRenderOverlay:
    def test_uses_profile_file_shape_without_legacy_profiles(self):
        overlay = codex.render_overlay(WS)
        assert "profile" not in overlay
        assert "profiles" not in overlay

    def test_sets_model_provider(self):
        overlay = codex.render_overlay(WS)
        assert overlay["model_provider"] == "ucode-databricks"

    def test_sets_model_when_provided(self):
        overlay = codex.render_overlay(WS, "databricks-gpt-5")
        assert overlay["model"] == "databricks-gpt-5"

    def test_provider_base_url(self):
        overlay = codex.render_overlay(WS)
        provider = overlay["model_providers"]["ucode-databricks"]
        assert provider["base_url"] == f"{WS}/ai-gateway/codex/v1"

    def test_provider_wire_api(self):
        overlay = codex.render_overlay(WS)
        provider = overlay["model_providers"]["ucode-databricks"]
        assert provider["wire_api"] == "responses"

    def test_auth_runs_ucode_auth_token(self):
        # The auth command runs the `ucode auth-token` executable directly
        # (not `sh -c`), so it works on Windows where there is no POSIX shell.
        overlay = codex.render_overlay(WS)
        auth = overlay["model_providers"]["ucode-databricks"]["auth"]
        assert auth["command"].endswith("ucode") or auth["command"] == "ucode"
        assert auth["args"][0] == "auth-token"
        assert auth["command"] != "sh"

    def test_auth_contains_workspace(self):
        overlay = codex.render_overlay(WS)
        auth = overlay["model_providers"]["ucode-databricks"]["auth"]
        assert any(WS in arg for arg in auth["args"])

    def test_auth_refresh_interval(self):
        overlay = codex.render_overlay(WS)
        auth = overlay["model_providers"]["ucode-databricks"]["auth"]
        assert auth["refresh_interval_ms"] == 900_000

    def test_provider_adds_routing_header(self):
        overlay = codex.render_overlay(WS, provider="main.aarushi.aarushi-openai")
        headers = overlay["model_providers"]["ucode-databricks"]["http_headers"]
        assert headers["Databricks-Model-Provider-Service"] == "main.aarushi.aarushi-openai"

    def test_provider_omits_model(self):
        overlay = codex.render_overlay(WS, model=None, provider="main.aarushi.aarushi-openai")
        assert "model" not in overlay

    def test_no_provider_header_without_flag(self):
        overlay = codex.render_overlay(WS)
        headers = overlay["model_providers"]["ucode-databricks"]["http_headers"]
        assert "Databricks-Model-Provider-Service" not in headers


class TestRenderOverlayUserAgent:
    def test_user_agent_set_on_provider(self, monkeypatch):
        monkeypatch.setattr(codex, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.123.0")
        overlay = codex.render_overlay(WS)
        provider = overlay["model_providers"]["ucode-databricks"]
        assert provider["http_headers"]["User-Agent"] == "ucode/0.1.0 codex/0.123.0"

    def test_managed_keys_include_http_headers(self):
        # Revert must clean up the new key.
        assert ["model_providers", "ucode-databricks", "http_headers"] in codex.MANAGED_KEYS


class TestCodexWriteConfig:
    def test_writes_ucode_profile_config_file(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config({"workspace": WS, "codex_models": ["gpt-5"]})

        doc = read_toml_safe(config_path)
        assert doc["model_provider"] == "ucode-databricks"
        assert "model" not in doc
        assert "model_reasoning_effort" not in doc
        assert "profiles" not in doc

    def test_removes_discovered_model_id(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config(
            {"workspace": WS, "codex_models": ["databricks-gpt-5", "databricks-gpt-5-5"]}
        )

        doc = read_toml_safe(config_path)
        assert "model" not in doc

    def test_removes_uc_model_services_id(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config(
            {"workspace": WS, "codex_models": ["system.ai.gpt-5", "system.ai.gpt-5-5"]}
        )

        doc = read_toml_safe(config_path)
        assert "model" not in doc

    def test_provider_writes_header_and_drops_stale_model(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        # An earlier run pinned a model.
        codex.write_tool_config({"workspace": WS, "codex_models": ["gpt-5"]})
        assert "model" not in read_toml_safe(config_path)

        # A provider run must clear it and add the routing header.
        codex.write_tool_config(
            {"workspace": WS, "codex_models": ["gpt-5"]},
            provider="main.aarushi.aarushi-openai",
        )

        doc = read_toml_safe(config_path)
        assert "model" not in doc
        headers = doc["model_providers"]["ucode-databricks"]["http_headers"]
        assert headers["Databricks-Model-Provider-Service"] == "main.aarushi.aarushi-openai"

    def test_clears_profile_model_preferences_before_launch(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        config_path.parent.mkdir()
        config_path.write_text(
            'model = "system.ai.gpt-5-6-luna"\nmodel_reasoning_effort = "medium"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", tmp_path / "backup.toml")

        assert codex.clear_model_preferences({}) is True

        doc = read_toml_safe(config_path)
        assert "model" not in doc
        assert "model_reasoning_effort" not in doc

    def test_preserves_profile_without_model_preferences(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        config_path.parent.mkdir()
        config_path.write_text('model_provider = "ucode-databricks"\n', encoding="utf-8")
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", tmp_path / "backup.toml")

        assert codex.clear_model_preferences({}) is False
        assert not (tmp_path / "backup.toml").exists()

    def test_preserves_profile_model_for_managed_default(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        config_path.parent.mkdir()
        config_path.write_text('model = "managed-default"\n', encoding="utf-8")
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)

        assert codex.clear_model_preferences({"codex_default_model": "managed-default"}) is False
        assert read_toml_safe(config_path)["model"] == "managed-default"

    def test_removes_legacy_ucode_profile_from_shared_config(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        profile_path = config_dir / "ucode.config.toml"
        legacy_path = config_dir / "config.toml"
        legacy_path.write_text(
            'profile = "ucode"\n\n'
            "[profiles.ucode]\n"
            'model_provider = "old"\n\n'
            "[profiles.other]\n"
            'model_provider = "keep"\n',
            encoding="utf-8",
        )
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        legacy_backup_path = tmp_path / "codex-legacy-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", profile_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config({"workspace": WS, "codex_models": ["gpt-5"]})

        doc = read_toml_safe(legacy_path)
        assert "profile" not in doc
        assert "ucode" not in doc["profiles"]
        assert doc["profiles"]["other"]["model_provider"] == "keep"
        assert legacy_backup_path.exists()

    def test_writes_legacy_shared_config_when_codex_too_old(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".codex"
        legacy_path = config_dir / "config.toml"
        profile_path = config_dir / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        legacy_backup_path = tmp_path / "codex-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", profile_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "LEGACY_CODEX_CONFIG_PATH", legacy_path)
        monkeypatch.setattr(codex, "LEGACY_CODEX_BACKUP_PATH", legacy_backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.133.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config({"workspace": WS, "codex_models": ["gpt-5"]})

        # Per-profile file must not be written for old Codex.
        assert not profile_path.exists()
        doc = read_toml_safe(legacy_path)
        assert doc["profile"] == "ucode"
        assert doc["profiles"]["ucode"]["model_provider"] == "ucode-databricks"
        assert "model" not in doc["profiles"]["ucode"]
        provider = doc["model_providers"]["ucode-databricks"]
        assert provider["base_url"] == f"{WS}/ai-gateway/codex/v1"
        assert provider["wire_api"] == "responses"

    def test_smart_routing_writes_profile_scoped_hooks(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        config_path.parent.mkdir()
        config_path.write_text(
            "[[hooks.PreToolUse]]\n"
            'matcher = "Bash"\n'
            "[[hooks.PreToolUse.hooks]]\n"
            'type = "command"\n'
            'command = "user-policy"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", tmp_path / "backup.toml")
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.145.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config(
            {
                "workspace": WS,
                "profile": "prod",
                "codex_models": ["databricks-gpt-5", "databricks-gpt-5-5"],
                "oss_models": ["system.ai.glm-5-2"],
                codex.SMART_ROUTING_STATE_KEY: True,
            }
        )

        doc = read_toml_safe(config_path)
        assert set(doc["hooks"]) == {"PreToolUse", "SessionStart", "SubagentStart"}
        pre_tool_commands = [
            hook["command"] for group in doc["hooks"]["PreToolUse"] for hook in group["hooks"]
        ]
        assert "user-policy" in pre_tool_commands
        route_command = next(
            command for command in pre_tool_commands if "codex-router-hook" in command
        )
        assert "route-subagent" in route_command
        assert "--host https://example.databricks.com" in route_command
        assert "--profile prod" in route_command
        assert "--model databricks-gpt-5-5" in route_command
        assert "--model system.ai.glm-5-2" in route_command

    def test_provider_launch_removes_routing_hooks(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        backup_path = tmp_path / "backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.145.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)
        state = {
            "workspace": WS,
            "codex_models": ["databricks-gpt-5"],
            codex.SMART_ROUTING_STATE_KEY: True,
        }

        codex.write_tool_config(state)
        assert "hooks" in read_toml_safe(config_path)

        codex.write_tool_config(state, provider="main.schema.provider")

        assert "hooks" not in read_toml_safe(config_path)

    def test_legacy_write_preserves_other_profiles_in_shared_config(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        legacy_path = config_dir / "config.toml"
        legacy_path.write_text(
            '[profiles.other]\nmodel_provider = "keep"\n',
            encoding="utf-8",
        )
        profile_path = config_dir / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        legacy_backup_path = tmp_path / "codex-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", profile_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "LEGACY_CODEX_CONFIG_PATH", legacy_path)
        monkeypatch.setattr(codex, "LEGACY_CODEX_BACKUP_PATH", legacy_backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.133.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config({"workspace": WS, "codex_models": ["gpt-5"]})

        doc = read_toml_safe(legacy_path)
        assert doc["profiles"]["other"]["model_provider"] == "keep"
        assert doc["profiles"]["ucode"]["model_provider"] == "ucode-databricks"


class TestCodexLegacyLayoutDetection:
    def test_new_codex_uses_modern_layout(self, monkeypatch):
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")

        assert codex._use_legacy_layout() is False

    def test_old_codex_uses_legacy_layout(self, monkeypatch):
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.133.0")

        assert codex._use_legacy_layout() is True

    def test_unknown_version_uses_modern_layout(self, monkeypatch):
        monkeypatch.setattr(codex, "agent_version", lambda binary: "unknown")

        assert codex._use_legacy_layout() is False


class TestCodexSmartRouting:
    def test_enable_requires_supported_codex(self, monkeypatch):
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.144.0")

        try:
            codex.enable_smart_routing({})
            assert False
        except RuntimeError as exc:
            assert "0.145.0 or newer" in str(exc)

    def test_disable_removes_only_ucode_hooks(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        legacy_path = tmp_path / ".codex" / "config.toml"
        config_path.parent.mkdir()
        config_path.write_text(
            "[[hooks.PreToolUse]]\n"
            'matcher = "Bash"\n'
            "[[hooks.PreToolUse.hooks]]\n"
            'type = "command"\n'
            'command = "user-policy"\n\n'
            "[[hooks.PreToolUse]]\n"
            'matcher = "Agent"\n'
            "[[hooks.PreToolUse.hooks]]\n"
            'type = "command"\n'
            'command = "ucode codex-router-hook route-subagent"\n\n'
            "[[hooks.SessionStart]]\n"
            "[[hooks.SessionStart.hooks]]\n"
            'type = "command"\n'
            'command = "ucode codex-router-hook session-start"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "LEGACY_CODEX_CONFIG_PATH", legacy_path)
        monkeypatch.setattr(codex, "save_state", lambda state: None)
        monkeypatch.setattr(codex_routing, "clear_routing_artifacts", lambda: None)
        state = {"workspace": WS, codex.SMART_ROUTING_STATE_KEY: True}

        assert codex.disable_smart_routing(state) is True

        doc = read_toml_safe(config_path)
        assert state.get(codex.SMART_ROUTING_STATE_KEY) is None
        assert list(doc["hooks"]) == ["PreToolUse"]
        assert doc["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "user-policy"

    def test_launch_task_uses_exec_prompt(self):
        assert codex_routing._launch_routing_task(["exec", "fix the parser"]) == "fix the parser"

    def test_launch_task_uses_positional_interactive_prompt(self):
        # `codex "fix the parser"` — the seed prompt is routed directly, not
        # wrapped in a placeholder.
        assert codex_routing._launch_routing_task(["fix the parser"]) == "fix the parser"

    def test_launch_task_skips_value_option_before_prompt(self):
        # `-m <model>` consumes its value; the model id must not be taken as the
        # prompt.
        assert (
            codex_routing._launch_routing_task(["-m", "gpt-5", "refactor the parser"])
            == "refactor the parser"
        )

    def test_launch_task_honors_double_dash(self):
        assert (
            codex_routing._launch_routing_task(["--", "--not-a-flag prompt"])
            == "--not-a-flag prompt"
        )

    def test_launch_task_bare_launch_returns_none(self):
        # No prompt on the command line → None, so the caller skips routing and
        # keeps the user's default model (root model can't be re-routed once the
        # TUI is up).
        assert codex_routing._launch_routing_task([]) is None

    def test_launch_task_flags_only_returns_none(self):
        assert codex_routing._launch_routing_task(["--search", "-m", "gpt-5"]) is None

    def test_route_launch_model_skips_routing_without_prompt(self, monkeypatch):
        # Bare launch: no router call at all, no decision, no error.
        def fail(*args, **kwargs):
            raise AssertionError("router must not be called on a bare launch")

        monkeypatch.setattr(codex_routing, "request_routing_decision", fail)
        decision, error = codex_routing.route_launch_model(
            {"workspace": WS, "codex_models": ["system.ai.gpt-5-6-sol"]}, []
        )
        assert decision is None
        assert error is None

    def test_route_launch_model_includes_codex_and_oss_models(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(codex_routing, "get_databricks_token", lambda *args: "token")

        def request(workspace, token, task, models):
            captured["models"] = models
            return None, "expected test stop"

        monkeypatch.setattr(codex_routing, "request_routing_decision", request)

        codex_routing.route_launch_model(
            {
                "workspace": WS,
                "codex_models": ["system.ai.gpt-5-6-sol"],
                "oss_models": ["system.ai.glm-5-2"],
            },
            ["Fix the parser"],
        )

        assert captured["models"] == ["system.ai.gpt-5-6-sol", "system.ai.glm-5-2"]


class TestCodexRemoveLegacyProfile:
    def test_drops_provider_block_on_modern_path(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        profile_path = config_dir / "ucode.config.toml"
        legacy_path = config_dir / "config.toml"
        legacy_path.write_text(
            'profile = "ucode"\n\n'
            "[profiles.ucode]\n"
            'model_provider = "ucode-databricks"\n\n'
            "[model_providers.ucode-databricks]\n"
            'name = "Databricks AI Gateway"\n\n'
            "[model_providers.other]\n"
            'name = "keep"\n',
            encoding="utf-8",
        )
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", profile_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config({"workspace": WS, "codex_models": ["gpt-5"]})

        doc = read_toml_safe(legacy_path)
        assert "profile" not in doc
        assert "ucode" not in doc.get("profiles", {})
        assert "ucode-databricks" not in doc["model_providers"]
        assert doc["model_providers"]["other"]["name"] == "keep"


class TestCodexRevertLegacySharedConfig:
    def test_strips_all_ucode_entries(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        profile_path = config_dir / "ucode.config.toml"
        legacy_path = config_dir / "config.toml"
        legacy_path.write_text(
            'profile = "ucode"\n\n'
            "[profiles.ucode]\n"
            'model_provider = "ucode-databricks"\n\n'
            "[profiles.other]\n"
            'model_provider = "keep"\n\n'
            "[model_providers.ucode-databricks]\n"
            'name = "Databricks AI Gateway"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", profile_path)

        assert codex.revert_legacy_shared_config() is True

        doc = read_toml_safe(legacy_path)
        assert "profile" not in doc
        assert "ucode" not in doc["profiles"]
        assert doc["profiles"]["other"]["model_provider"] == "keep"
        assert "model_providers" not in doc

    def test_returns_false_when_no_ucode_entries(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        profile_path = config_dir / "ucode.config.toml"
        legacy_path = config_dir / "config.toml"
        legacy_path.write_text('[profiles.other]\nmodel_provider = "keep"\n', encoding="utf-8")
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", profile_path)

        assert codex.revert_legacy_shared_config() is False

        doc = read_toml_safe(legacy_path)
        assert doc["profiles"]["other"]["model_provider"] == "keep"

    def test_returns_false_when_no_shared_config(self, tmp_path, monkeypatch):
        profile_path = tmp_path / ".codex" / "ucode.config.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", profile_path)

        assert codex.revert_legacy_shared_config() is False


class TestCodexDefaultModel:
    @pytest.fixture(autouse=True)
    def _isolate_profile_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", tmp_path / "ucode.config.toml")
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", tmp_path / "backup.toml")

    def test_clears_profile_model_preferences(self, tmp_path):
        codex.CODEX_CONFIG_PATH.write_text(
            'model = "gpt-5.6-sol"\nmodel_reasoning_effort = "medium"\n', encoding="utf-8"
        )

        assert codex.default_model({"codex_models": ["system.ai.gpt-5-6-luna"]}) is None
        doc = read_toml_safe(codex.CODEX_CONFIG_PATH)
        assert "model" not in doc
        assert "model_reasoning_effort" not in doc

    def test_none_when_no_configured_model(self):
        assert codex.default_model({}) is None

    def test_managed_default_model_takes_priority(self):
        state = {
            "codex_default_model": "admin-chosen-default",
            "codex_models": ["databricks-gpt-5-5"],
        }
        assert codex.default_model(state) == "admin-chosen-default"

    def test_starting_model_uses_cached_codex_model(self):
        state = {
            "codex_models": ["system.ai.gpt-5-6-luna"],
            "oss_models": ["system.ai.glm-5-2"],
        }

        assert codex.starting_model(state) == "system.ai.gpt-5-6-luna"

    def test_starting_model_prefers_managed_default(self):
        state = {
            "codex_default_model": "admin-chosen-default",
            "codex_models": ["system.ai.gpt-5-6-luna"],
        }

        assert codex.starting_model(state) == "admin-chosen-default"

    def test_starting_model_falls_back_to_codex_catalog_default(self):
        assert codex.starting_model({}) == "gpt-5.6-luna"


class TestCodexValidateCmd:
    def test_starts_with_binary(self):
        cmd = codex.validate_cmd("codex")
        assert cmd[0] == "codex"

    def test_uses_exec_subcommand(self):
        cmd = codex.validate_cmd("codex")
        assert "exec" in cmd

    def test_uses_ucode_profile(self):
        cmd = codex.validate_cmd("codex")
        assert cmd[:3] == ["codex", "--profile", "ucode"]

    def test_has_prompt(self):
        cmd = codex.validate_cmd("codex")
        assert len(cmd) > 2

    def test_skips_git_repo_check(self):
        # Validation runs in arbitrary cwd (e.g., ~/Documents); without this
        # flag Codex refuses to run outside a trusted/git directory.
        cmd = codex.validate_cmd("codex")
        assert "--skip-git-repo-check" in cmd


class TestCodexLaunch:
    """launch() runs codex with --profile first and relaunches without it only
    when that attempt fails *fast* — codex's --profile rejection is a parse-time
    error, so a fast nonzero exit means the subcommand didn't accept --profile.
    A slow failure is a real session error and is propagated unchanged."""

    @staticmethod
    def _patch(monkeypatch, *, returncode: int, elapsed: float):
        """Stub subprocess.run to return `returncode` and make launch() perceive
        `elapsed` seconds between its two time.monotonic() reads."""
        runs: list[list[str]] = []
        fallbacks: list[list[str]] = []

        def fake_run(argv, **kwargs):
            runs.append(argv)
            return codex.subprocess.CompletedProcess(argv, returncode)

        # launch() reads time.monotonic() once before run and once after.
        clock = iter([100.0, 100.0 + elapsed])
        monkeypatch.setattr(codex.subprocess, "run", fake_run)
        monkeypatch.setattr(codex.time, "monotonic", lambda: next(clock))
        monkeypatch.setattr(codex, "exec_or_spawn", lambda argv: fallbacks.append(argv))
        monkeypatch.setattr(codex, "get_databricks_token", lambda workspace, profile=None: "tok")
        monkeypatch.setattr(codex, "clear_model_preferences", lambda state: False)
        return runs, fallbacks

    def test_sets_oauth_token_and_runs_with_profile(self, monkeypatch):
        monkeypatch.delenv("OAUTH_TOKEN", raising=False)
        runs, fallbacks = self._patch(monkeypatch, returncode=0, elapsed=0.5)
        monkeypatch.setattr(
            codex, "get_databricks_token", lambda workspace, profile=None: "fresh-token"
        )
        with pytest.raises(SystemExit) as exc:
            codex.launch({"workspace": WS}, ["--search"])
        assert exc.value.code == 0
        assert os.environ["OAUTH_TOKEN"] == "fresh-token"
        assert runs == [["codex", "--profile", "ucode", "--search"]]
        assert fallbacks == []

    def test_success_propagates_exit_without_retry(self, monkeypatch):
        runs, fallbacks = self._patch(monkeypatch, returncode=0, elapsed=0.2)
        with pytest.raises(SystemExit) as exc:
            codex.launch({"workspace": WS}, ["exec", "hi"])
        assert exc.value.code == 0
        assert runs == [["codex", "--profile", "ucode", "exec", "hi"]]
        assert fallbacks == []

    def test_fast_failure_relaunches_without_profile(self, monkeypatch):
        # codex rejects --profile on server-family subcommands at parse time —
        # a fast nonzero exit → relaunch without --profile.
        for args in (["app-server", "--listen", "u"], ["mcp-server"]):
            runs, fallbacks = self._patch(monkeypatch, returncode=1, elapsed=0.15)
            codex.launch({"workspace": WS}, args)
            assert runs == [["codex", "--profile", "ucode", *args]]
            assert fallbacks == [["codex", *args]]

    def test_fallback_warns_on_stderr_before_handoff(self, monkeypatch, capsys):
        # The fallback drops ucode's Databricks routing, so it must say so. The
        # warning goes to *stderr*: `codex app-server` speaks JSON-RPC on stdout,
        # and a warning there would corrupt the stream its caller parses.
        warned_before_handoff = []
        runs, fallbacks = self._patch(monkeypatch, returncode=1, elapsed=0.15)
        monkeypatch.setattr(
            codex,
            "exec_or_spawn",
            lambda argv: warned_before_handoff.append(capsys.readouterr()),
        )
        codex.launch({"workspace": WS}, ["app-server"])

        # execvp replaces the process, so the warning must already be out by then.
        assert len(warned_before_handoff) == 1
        captured = warned_before_handoff[0]
        err = " ".join(captured.err.split())  # unwrap Rich's width-based wrapping
        # Attributes the flag to ucode (users never type --profile themselves)
        # and points at codex's own error so it doesn't read as their mistake.
        assert "ucode's `--profile`" in err
        assert "error above" in err
        # Names the fallback config and that Databricks routing is lost.
        assert str(codex.LEGACY_CODEX_CONFIG_PATH) in err
        assert "NOT the Databricks gateway" in err
        assert captured.out == ""

    def test_slow_failure_does_not_retry(self, monkeypatch):
        # A session that started and then failed (seconds) must NOT be re-run
        # without --profile — that would silently drop ucode's Databricks routing
        # (relaunching the user's prompt on their own OpenAI login).
        runs, fallbacks = self._patch(monkeypatch, returncode=1, elapsed=8.0)
        with pytest.raises(SystemExit) as exc:
            codex.launch({"workspace": WS}, ["exec", "hi"])
        assert exc.value.code == 1
        assert fallbacks == []

    def test_fast_success_does_not_retry(self, monkeypatch):
        # The retry is gated on a *nonzero* exit; a fast clean exit just returns.
        runs, fallbacks = self._patch(monkeypatch, returncode=0, elapsed=0.15)
        with pytest.raises(SystemExit) as exc:
            codex.launch({"workspace": WS}, [])
        assert exc.value.code == 0
        assert fallbacks == []


class TestCodexManagedConfig:
    """use_as_global_settings: also write Codex's OS managed_config.toml (via sudo, mocked)."""

    def _patch(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        managed_path = tmp_path / "etc-codex" / "managed_config.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", tmp_path / "codex-ucode-config.backup.toml")
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)
        # Deterministic managed path + a mocked sudo writer that writes straight to disk, so the test
        # can read the TOML back and NO real sudo/`/etc` write ever happens.
        monkeypatch.setattr(codex, "_managed_config_path", lambda: managed_path)

        def fake_write_managed(path, text, *, display):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
            return "written"

        monkeypatch.setattr(codex, "write_managed_file", fake_write_managed)
        return config_path, managed_path

    def test_writes_managed_config_when_flagged(self, tmp_path, monkeypatch):
        _, managed_path = self._patch(tmp_path, monkeypatch)
        state = {"workspace": WS, "codex_models": ["gpt-5"], "write_managed_config": True}
        codex.write_tool_config(state)

        doc = read_toml_safe(managed_path)
        assert doc["model_provider"] == "ucode-databricks"
        assert "model" not in doc
        assert "ucode-databricks" in doc["model_providers"]

    def test_managed_config_preserves_other_keys(self, tmp_path, monkeypatch):
        _, managed_path = self._patch(tmp_path, monkeypatch)
        managed_path.parent.mkdir(parents=True, exist_ok=True)
        managed_path.write_text(
            'model = "my-own"\napproval_policy = "on-request"\n', encoding="utf-8"
        )
        state = {"workspace": WS, "codex_models": ["gpt-5"], "write_managed_config": True}
        codex.write_tool_config(state)

        doc = read_toml_safe(managed_path)
        # ucode removes its stale model pin, but other keys already in the managed file survive.
        assert doc["approval_policy"] == "on-request"
        assert "model" not in doc

    def test_no_managed_write_by_default(self, tmp_path, monkeypatch):
        _, managed_path = self._patch(tmp_path, monkeypatch)
        state = {"workspace": WS, "codex_models": ["gpt-5"]}
        codex.write_tool_config(state)
        assert not managed_path.exists()
