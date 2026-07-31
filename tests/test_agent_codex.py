"""Tests for agents/codex.py."""

from __future__ import annotations

import os

from ucode.agents import codex
from ucode.config_io import read_toml_safe

WS = "https://example.databricks.com"


class TestCodexSpec:
    def test_binary(self):
        assert codex.SPEC["binary"] == "codex"

    def test_package(self):
        assert codex.SPEC["package"] == "@openai/codex"

    def test_display(self):
        assert codex.SPEC["display"] == "Codex"


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
        assert doc["model"] == "gpt-5"
        assert "profiles" not in doc

    def test_writes_openai_model_id_for_databricks_gpt_endpoint(self, tmp_path, monkeypatch):
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
        assert doc["model"] == "gpt-5.5"

    def test_preserves_databricks_model_id_when_openai_id_is_incompatible(
        self, tmp_path, monkeypatch
    ):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        codex.write_tool_config(
            {"workspace": WS, "codex_models": ["databricks-gpt-5-2-codex"]},
            "databricks-gpt-5-2-codex",
        )

        doc = read_toml_safe(config_path)
        assert doc["model"] == "databricks-gpt-5-2-codex"

    def test_provider_writes_header_and_drops_stale_model(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "ucode.config.toml"
        backup_path = tmp_path / "codex-ucode-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.134.0")
        monkeypatch.setattr(codex, "save_state", lambda state: None)

        # An earlier non-provider run pinned a model.
        codex.write_tool_config({"workspace": WS, "codex_models": ["gpt-5"]})
        assert read_toml_safe(config_path)["model"] == "gpt-5"

        # A provider run must clear it and add the routing header.
        codex.write_tool_config(
            {"workspace": WS, "codex_models": ["gpt-5"]},
            provider="main.aarushi.aarushi-openai",
        )

        doc = read_toml_safe(config_path)
        assert "model" not in doc
        headers = doc["model_providers"]["ucode-databricks"]["http_headers"]
        assert headers["Databricks-Model-Provider-Service"] == "main.aarushi.aarushi-openai"

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
        assert doc["profiles"]["ucode"]["model"] == "gpt-5"
        provider = doc["model_providers"]["ucode-databricks"]
        assert provider["base_url"] == f"{WS}/ai-gateway/codex/v1"
        assert provider["wire_api"] == "responses"

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
    def test_picks_highest_semver_over_alpha(self):
        state = {"codex_models": ["databricks-gpt-5", "databricks-gpt-5-5"]}

        assert codex.default_model(state) == "databricks-gpt-5-5"

    def test_none_when_no_models(self):
        assert codex.default_model({}) is None

    def test_none_when_no_gpt_parseable_models(self):
        # A workspace whose responses-capable models aren't GPT (e.g. kimi)
        # must not pin an unroutable id as the Codex model.
        state = {"codex_models": ["moonshotai/kimi-k2.5", "claude-sonnet-4"]}

        assert codex.default_model(state) is None

    def test_ignores_non_gpt_candidates(self):
        state = {"codex_models": ["moonshotai/kimi-k2.5", "databricks-gpt-5-5"]}

        assert codex.default_model(state) == "databricks-gpt-5-5"

    def test_prefers_base_over_suffixed_same_version(self):
        models = ["gpt-5-5-mini", "gpt-5-5", "gpt-5"]

        assert codex.default_model({"codex_models": models}) == "gpt-5-5"

    def test_namespaced_models_use_same_version_parser(self):
        models = ["served-models/databricks-gpt-5", "served-models/databricks-gpt-5-5"]

        assert codex.default_model({"codex_models": models}) == "served-models/databricks-gpt-5-5"

    def test_openai_model_id_maps_databricks_naming(self):
        assert codex._openai_model_id("databricks-gpt-5-5") == "gpt-5.5"
        assert codex._openai_model_id("databricks-gpt-5-5-mini") == "gpt-5.5-mini"
        assert codex._openai_model_id("databricks-gpt-4o") == "gpt-4o"
        assert codex._openai_model_id("served-models/databricks-gpt-5-5") == "gpt-5.5"
        assert codex._openai_model_id("gpt-5.5") == "gpt-5.5"

    def test_codex_model_id_preserves_openai_incompatible_models(self):
        assert codex._codex_model_id("databricks-gpt-5-2-codex") == "databricks-gpt-5-2-codex"
        assert codex._codex_model_id("databricks-gpt-5-4-nano") == "databricks-gpt-5-4-nano"

    def test_codex_model_id_passes_model_services_id_verbatim(self):
        # UC model-services ids route by name, so they must not be rewritten
        # to the OpenAI id form.
        assert codex._codex_model_id("system.ai.gpt-5") == "system.ai.gpt-5"
        assert codex._codex_model_id("system.ai.gpt-5-2-codex") == "system.ai.gpt-5-2-codex"

    def test_default_model_selects_model_services_gpt(self):
        models = ["system.ai.gpt-5", "system.ai.gpt-5-5", "system.ai.claude-opus-4-8"]

        assert codex.default_model({"codex_models": models}) == "system.ai.gpt-5-5"
        assert codex._codex_model_id("databricks-gpt-5-5") == "gpt-5.5"


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
    def test_sets_oauth_token_and_ucode_profile_before_exec(self, monkeypatch):
        exec_calls: list[tuple[str, list[str]]] = []

        def fake_execvp(binary: str, args: list[str]) -> None:
            exec_calls.append((binary, args))
            raise RuntimeError("stop")

        monkeypatch.delenv("OAUTH_TOKEN", raising=False)
        monkeypatch.setattr(
            codex, "get_databricks_token", lambda workspace, profile=None: "fresh-token"
        )
        monkeypatch.setattr(os, "execvp", fake_execvp)

        try:
            codex.launch({"workspace": WS}, ["--search"])
        except RuntimeError as exc:
            assert str(exc) == "stop"

        assert os.environ["OAUTH_TOKEN"] == "fresh-token"
        assert exec_calls == [("codex", ["codex", "--profile", "ucode", "--search"])]


class TestCodexAppWriteConfig:
    def _patch_paths(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "config.toml"
        backup_path = tmp_path / "codex-app-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_APP_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_APP_BACKUP_PATH", backup_path)
        monkeypatch.setattr(codex, "save_state", lambda state: None)
        return config_path, backup_path

    def test_writes_root_config_with_app_provider(self, tmp_path, monkeypatch):
        config_path, _ = self._patch_paths(tmp_path, monkeypatch)

        codex.write_app_config({"workspace": WS, "codex_models": ["gpt-5"]})

        doc = read_toml_safe(config_path)
        # The app reads root-level keys, not a [profiles.*] table.
        assert doc["model_provider"] == "ucode-databricks-app"
        assert doc["model"] == "gpt-5"
        assert "profiles" not in doc
        provider = doc["model_providers"]["ucode-databricks-app"]
        assert provider["base_url"] == f"{WS}/ai-gateway/codex/v1"
        assert provider["wire_api"] == "responses"

    def test_uses_distinct_provider_from_cli(self, tmp_path, monkeypatch):
        # The app must not reuse the CLI provider name, so the two features can
        # coexist on the same ~/.codex/config.toml without clobbering.
        assert codex.CODEX_APP_MODEL_PROVIDER_NAME != codex.CODEX_MODEL_PROVIDER_NAME

    def test_backs_up_existing_config_before_editing(self, tmp_path, monkeypatch):
        config_path, backup_path = self._patch_paths(tmp_path, monkeypatch)
        config_path.parent.mkdir(parents=True)
        config_path.write_text('model = "o1"\n[foo]\nbar = "baz"\n', encoding="utf-8")

        codex.write_app_config({"workspace": WS, "codex_models": ["gpt-5"]})

        # Backup captures the pristine original.
        assert 'model = "o1"' in backup_path.read_text(encoding="utf-8")
        # The user's unrelated section is preserved by the deep merge.
        assert read_toml_safe(config_path)["foo"]["bar"] == "baz"

    def test_marks_codex_app_managed(self, tmp_path, monkeypatch):
        self._patch_paths(tmp_path, monkeypatch)

        state = codex.write_app_config({"workspace": WS, "codex_models": ["gpt-5"]})

        assert "codex-app" in state["managed_configs"]

    def test_provider_drops_stale_model_and_adds_header(self, tmp_path, monkeypatch):
        config_path, _ = self._patch_paths(tmp_path, monkeypatch)

        codex.write_app_config({"workspace": WS, "codex_models": ["gpt-5"]})
        assert read_toml_safe(config_path)["model"] == "gpt-5"

        codex.write_app_config(
            {"workspace": WS, "codex_models": ["gpt-5"]},
            provider="main.aarushi.aarushi-openai",
        )

        doc = read_toml_safe(config_path)
        assert "model" not in doc
        headers = doc["model_providers"]["ucode-databricks-app"]["http_headers"]
        assert headers["Databricks-Model-Provider-Service"] == "main.aarushi.aarushi-openai"


class TestCodexAppRevert:
    def test_restores_from_backup(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "config.toml"
        backup_path = tmp_path / "codex-app-config.backup.toml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text('model = "ucode-wrote-this"\n', encoding="utf-8")
        backup_path.write_text('model = "original"\n', encoding="utf-8")
        monkeypatch.setattr(codex, "CODEX_APP_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_APP_BACKUP_PATH", backup_path)

        assert codex.revert_app_config(managed=True) is True

        assert 'model = "original"' in config_path.read_text(encoding="utf-8")
        assert not backup_path.exists()

    def test_removes_config_when_managed_and_no_backup(self, tmp_path, monkeypatch):
        # ucode created config.toml (no prior file to back up), so revert deletes it.
        config_path = tmp_path / ".codex" / "config.toml"
        backup_path = tmp_path / "codex-app-config.backup.toml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text('model = "ucode-wrote-this"\n', encoding="utf-8")
        monkeypatch.setattr(codex, "CODEX_APP_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_APP_BACKUP_PATH", backup_path)

        assert codex.revert_app_config(managed=True) is True
        assert not config_path.exists()

    def test_unchanged_when_not_managed_and_no_backup(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".codex" / "config.toml"
        backup_path = tmp_path / "codex-app-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_APP_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_APP_BACKUP_PATH", backup_path)

        assert codex.revert_app_config(managed=False) is False


class TestCodexAppLaunch:
    def test_unsupported_platform_raises(self, monkeypatch):
        monkeypatch.setattr(codex.platform, "system", lambda: "Linux")

        try:
            codex.launch_app({"workspace": WS})
            raise AssertionError("expected RuntimeError")
        except RuntimeError as exc:
            assert "macOS and Windows" in str(exc)

    def test_opens_app_in_cwd_when_not_running(self, tmp_path, monkeypatch):
        opened: list[str | None] = []
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(codex.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(codex, "_app_is_running", lambda: False)
        monkeypatch.setattr(codex, "_open_app", lambda workdir=None: opened.append(workdir) or True)

        codex.launch_app({"workspace": WS})

        # The app is opened in the directory ucode was run from, not its default.
        assert opened == [str(tmp_path)]

    def test_prompts_restart_when_running(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(codex.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(codex, "_app_is_running", lambda: True)
        monkeypatch.setattr(codex, "prompt_yes_no", lambda prompt: True)
        monkeypatch.setattr(codex, "_quit_app", lambda: calls.append("quit"))
        monkeypatch.setattr(codex, "_wait_for_app_exit", lambda timeout: True)
        monkeypatch.setattr(codex, "_open_app", lambda workdir=None: calls.append("open") or True)

        codex.launch_app({"workspace": WS})

        assert calls == ["quit", "open"]

    def test_declining_restart_leaves_app_alone(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(codex.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(codex, "_app_is_running", lambda: True)
        monkeypatch.setattr(codex, "prompt_yes_no", lambda prompt: False)
        monkeypatch.setattr(codex, "_quit_app", lambda: calls.append("quit"))
        monkeypatch.setattr(codex, "_open_app", lambda workdir=None: calls.append("open") or True)

        codex.launch_app({"workspace": WS})

        assert calls == []
