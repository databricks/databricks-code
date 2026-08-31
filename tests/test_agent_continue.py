"""Tests for agents/continue_dev.py."""

from __future__ import annotations

import yaml

import ucode.config_io as config_io_mod
from ucode.agents import continue_dev

WS = "https://example.databricks.com"


class TestContinueSpec:
    def test_binary(self):
        assert continue_dev.SPEC["binary"] == "cn"

    def test_package(self):
        assert continue_dev.SPEC["package"] == "@continuedev/cli"

    def test_display(self):
        assert continue_dev.SPEC["display"] == "Continue"

    def test_config_path_is_continue_config_yaml(self):
        assert continue_dev.SPEC["config_path"].name == "config.yaml"
        assert continue_dev.SPEC["config_path"].parent.name == ".continue"


class TestDefaultModel:
    def test_prefers_claude_sonnet(self):
        state = {"claude_models": {"sonnet": "s4", "opus": "o4", "haiku": "h4"}}
        assert continue_dev.default_model(state) == "s4"

    def test_falls_back_to_opus(self):
        assert continue_dev.default_model({"claude_models": {"opus": "o4"}}) == "o4"

    def test_falls_back_to_haiku(self):
        assert continue_dev.default_model({"claude_models": {"haiku": "h4"}}) == "h4"

    def test_falls_back_to_codex_when_no_claude(self):
        state = {"claude_models": {}, "codex_models": ["gpt-5", "gpt-4"]}
        assert continue_dev.default_model(state) == "gpt-5"

    def test_returns_none_when_no_models(self):
        assert continue_dev.default_model({}) is None

    def test_managed_default_wins(self):
        state = {"continue_default_model": "pinned", "claude_models": {"sonnet": "s4"}}
        assert continue_dev.default_model(state) == "pinned"


class TestUcodeModelEntry:
    # `_ucode_model_entry` is the primitive the production write path uses, so
    # asserting on it directly (rather than a test-only builder) can't drift.
    def test_provider_is_openai(self):
        assert continue_dev._ucode_model_entry("m", "tok", WS)["provider"] == "openai"

    def test_api_base_is_mlflow_gateway(self):
        entry = continue_dev._ucode_model_entry("m", "tok", WS)
        assert entry["apiBase"] == f"{WS}/ai-gateway/mlflow/v1"

    def test_model_id_verbatim(self):
        assert continue_dev._ucode_model_entry("databricks-gpt-5", "tok", WS)["model"] == (
            "databricks-gpt-5"
        )

    def test_api_key_is_token(self):
        assert continue_dev._ucode_model_entry("m", "tok123", WS)["apiKey"] == "tok123"

    def test_roles_are_chat_edit_apply(self):
        assert continue_dev._ucode_model_entry("m", "tok", WS)["roles"] == ["chat", "edit", "apply"]

    def test_name_carries_ucode_prefix(self):
        entry = continue_dev._ucode_model_entry("m", "tok", WS)
        assert entry["name"].startswith(continue_dev.UCODE_MODEL_NAME_PREFIX)

    def test_user_agent_header_present(self):
        entry = continue_dev._ucode_model_entry("m", "tok", WS)
        assert "User-Agent" in entry["requestOptions"]["headers"]


class TestSchemaHeader:
    def test_sets_required_keys_on_empty(self):
        doc: dict = {}
        continue_dev._ensure_schema_header(doc)
        assert doc == {"name": "ucode", "version": "0.0.1", "schema": "v1"}

    def test_keeps_user_values(self):
        doc = {"name": "mine", "version": "9.9", "schema": "v1"}
        continue_dev._ensure_schema_header(doc)
        assert doc["name"] == "mine"
        assert doc["version"] == "9.9"


class TestWriteToolConfig:
    def _patch_paths(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        backup_file = tmp_path / "continue-backup.yaml"
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr(continue_dev, "CONTINUE_CONFIG_PATH", config_file)
        monkeypatch.setattr(continue_dev, "CONTINUE_BACKUP_PATH", backup_file)
        monkeypatch.setattr(continue_dev, "get_databricks_token", lambda w, p, **k: "minted-token")
        monkeypatch.setattr(continue_dev, "save_state", lambda s: None)
        return config_file

    def test_writes_valid_yaml_with_ucode_model(self, tmp_path, monkeypatch):
        config_file = self._patch_paths(tmp_path, monkeypatch)
        state, token = continue_dev.write_tool_config(
            {"workspace": WS, "profile": None}, "claude-sonnet-4-6"
        )
        assert token == "minted-token"
        doc = yaml.safe_load(config_file.read_text())
        assert doc["schema"] == "v1"
        ucode_models = [
            m for m in doc["models"] if m["name"].startswith(continue_dev.UCODE_MODEL_NAME_PREFIX)
        ]
        assert len(ucode_models) == 1
        assert ucode_models[0]["model"] == "claude-sonnet-4-6"
        assert ucode_models[0]["apiKey"] == "minted-token"

    def test_uses_passed_token_without_minting(self, tmp_path, monkeypatch):
        config_file = self._patch_paths(tmp_path, monkeypatch)
        continue_dev.write_tool_config({"workspace": WS, "profile": None}, "m", token="given")
        doc = yaml.safe_load(config_file.read_text())
        assert doc["models"][0]["apiKey"] == "given"

    def test_rewrite_replaces_not_duplicates_ucode_entry(self, tmp_path, monkeypatch):
        config_file = self._patch_paths(tmp_path, monkeypatch)
        st = {"workspace": WS, "profile": None}
        continue_dev.write_tool_config(st, "model-a", token="t1")
        continue_dev.write_tool_config(st, "model-b", token="t2")
        doc = yaml.safe_load(config_file.read_text())
        ucode_models = [
            m for m in doc["models"] if m["name"].startswith(continue_dev.UCODE_MODEL_NAME_PREFIX)
        ]
        assert len(ucode_models) == 1
        assert ucode_models[0]["model"] == "model-b"

    def test_preserves_user_authored_models(self, tmp_path, monkeypatch):
        config_file = self._patch_paths(tmp_path, monkeypatch)
        config_file.write_text(
            yaml.safe_dump(
                {
                    "name": "my-config",
                    "version": "1.0.0",
                    "schema": "v1",
                    "models": [{"name": "My Local Model", "provider": "ollama", "model": "llama"}],
                }
            )
        )
        continue_dev.write_tool_config({"workspace": WS, "profile": None}, "m", token="t")
        doc = yaml.safe_load(config_file.read_text())
        names = [m["name"] for m in doc["models"]]
        assert "My Local Model" in names
        assert any(n.startswith(continue_dev.UCODE_MODEL_NAME_PREFIX) for n in names)
        # The user's own top-level name is kept (setdefault does not clobber it).
        assert doc["name"] == "my-config"

    def test_marks_tool_managed(self, tmp_path, monkeypatch):
        self._patch_paths(tmp_path, monkeypatch)
        state, _ = continue_dev.write_tool_config(
            {"workspace": WS, "profile": None}, "m", token="t"
        )
        assert state["managed_configs"]["continue"]["keys"] == [["models"]]


class TestMcpServerConfig:
    _ARGV = [
        "/usr/local/bin/ucode",
        "mcp-proxy",
        "--url",
        "https://ws/mcp/x",
        "--host",
        "https://ws",
    ]

    def _patch_paths(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr(continue_dev, "CONTINUE_CONFIG_PATH", config_file)
        monkeypatch.setattr(continue_dev, "CONTINUE_BACKUP_PATH", tmp_path / "continue-backup.yaml")
        return config_file

    def test_builds_stdio_entry_from_proxy_argv(self):
        entry = continue_dev.build_mcp_server_entry("databricks-slack", self._ARGV)
        assert entry["name"] == "databricks-slack"
        assert entry["type"] == "stdio"
        assert entry["command"] == "/usr/local/bin/ucode"
        assert entry["args"] == ["mcp-proxy", "--url", "https://ws/mcp/x", "--host", "https://ws"]

    def test_writes_server_without_clobbering_model(self, tmp_path, monkeypatch):
        config_file = self._patch_paths(tmp_path, monkeypatch)
        config_file.write_text(
            yaml.safe_dump(
                {
                    "name": "ucode",
                    "version": "0.0.1",
                    "schema": "v1",
                    "models": [
                        {"name": "Databricks (ucode) m", "provider": "openai", "model": "m"}
                    ],
                }
            )
        )
        removed = continue_dev.write_mcp_server_config("databricks-slack", self._ARGV)
        assert removed is False
        doc = yaml.safe_load(config_file.read_text())
        assert [s["name"] for s in doc["mcpServers"]] == ["databricks-slack"]
        # The model entry the writer produced is left intact.
        assert doc["models"][0]["name"] == "Databricks (ucode) m"

    def test_rewrite_replaces_reports_removed(self, tmp_path, monkeypatch):
        config_file = self._patch_paths(tmp_path, monkeypatch)
        continue_dev.write_mcp_server_config("databricks-slack", self._ARGV)
        removed = continue_dev.write_mcp_server_config("databricks-slack", self._ARGV)
        assert removed is True
        doc = yaml.safe_load(config_file.read_text())
        assert len(doc["mcpServers"]) == 1

    def test_removes_without_clobbering_others(self, tmp_path, monkeypatch):
        config_file = self._patch_paths(tmp_path, monkeypatch)
        config_file.write_text(
            yaml.safe_dump(
                {
                    "schema": "v1",
                    "mcpServers": [
                        {"name": "mine", "command": "x"},
                        {"name": "databricks-slack", "command": "y"},
                    ],
                }
            )
        )
        assert continue_dev.remove_mcp_server_config("databricks-slack") is True
        doc = yaml.safe_load(config_file.read_text())
        assert [s["name"] for s in doc["mcpServers"]] == ["mine"]

    def test_remove_absent_returns_false(self, tmp_path, monkeypatch):
        self._patch_paths(tmp_path, monkeypatch)
        assert continue_dev.remove_mcp_server_config("nope") is False


class TestRevertConfig:
    def _patch_paths(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr(continue_dev, "CONTINUE_CONFIG_PATH", config_file)
        monkeypatch.setattr(continue_dev, "CONTINUE_BACKUP_PATH", tmp_path / "continue-backup.yaml")
        return config_file

    def test_strips_only_ucode_models_keeping_user_content(self, tmp_path, monkeypatch):
        config_file = self._patch_paths(tmp_path, monkeypatch)
        config_file.write_text(
            yaml.safe_dump(
                {
                    "name": "mine",
                    "schema": "v1",
                    "models": [
                        {"name": "My Local Model", "provider": "ollama", "model": "llama"},
                        {"name": f"{continue_dev.UCODE_MODEL_NAME_PREFIX} m", "model": "m"},
                    ],
                    "mcpServers": [{"name": "mine-mcp", "command": "x"}],
                }
            )
        )
        assert continue_dev.revert_config() is True
        doc = yaml.safe_load(config_file.read_text())
        # ucode's model is gone; the user's model, servers, and name survive.
        assert [m["name"] for m in doc["models"]] == ["My Local Model"]
        assert [s["name"] for s in doc["mcpServers"]] == ["mine-mcp"]
        assert doc["name"] == "mine"

    def test_drops_models_key_when_only_ucode(self, tmp_path, monkeypatch):
        config_file = self._patch_paths(tmp_path, monkeypatch)
        config_file.write_text(
            yaml.safe_dump(
                {
                    "name": "ucode",
                    "schema": "v1",
                    "models": [{"name": f"{continue_dev.UCODE_MODEL_NAME_PREFIX} m", "model": "m"}],
                }
            )
        )
        assert continue_dev.revert_config() is True
        assert "models" not in yaml.safe_load(config_file.read_text())

    def test_returns_false_when_no_ucode_models(self, tmp_path, monkeypatch):
        config_file = self._patch_paths(tmp_path, monkeypatch)
        config_file.write_text(yaml.safe_dump({"schema": "v1", "models": [{"name": "Mine"}]}))
        assert continue_dev.revert_config() is False

    def test_returns_false_when_no_config(self, tmp_path, monkeypatch):
        self._patch_paths(tmp_path, monkeypatch)
        assert continue_dev.revert_config() is False


class TestValidateCmd:
    def test_starts_with_binary(self):
        cmd = continue_dev.validate_cmd("cn")
        assert cmd[0] == "cn"

    def test_has_headless_and_config_flags(self):
        cmd = continue_dev.validate_cmd("cn")
        assert "-p" in cmd
        assert "--config" in cmd
