"""Tests for non-destructive config merge/unmerge logic.

Verifies that coding-gateway's configure and logout flows preserve
user settings that the gateway does not manage.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CLAUDE_SETTINGS = {
    "$schema": "https://example.com/schema",
    "model": "claude-sonnet-4-20250514",
    "env": {
        "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1",
        "MY_CUSTOM_VAR": "keep-me",
    },
    "mcpServers": {
        "slack": {"type": "http", "url": "https://slack.example.com"},
        "jira": {"type": "http", "url": "https://jira.example.com"},
    },
    "permissions": {"allow": ["read"], "deny": ["write"]},
    "hooks": {"PreToolUse": {"command": "echo hi"}},
    "allow": ["Bash", "Read", "Write"],
    "enabledPlugins": {"plugin-a": True, "plugin-b": True},
    "statusLine": {"enabled": True},
    "alwaysThinkingEnabled": True,
    "voiceEnabled": False,
}

GATEWAY_CLAUDE_SETTINGS = {
    "apiKeyHelper": "sh -c 'databricks auth token --host https://test.cloud.databricks.com'",
    "env": {
        "ANTHROPIC_MODEL": "databricks-claude-opus-4-6",
        "ANTHROPIC_BASE_URL": "https://test.cloud.databricks.com/ai-gateway/anthropic/v1",
        "ANTHROPIC_CUSTOM_HEADERS": "x-databricks-use-coding-agent-mode: true",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_API_KEY_HELPER_TTL_MS": "1800000",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "databricks-claude-opus-4-6",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "databricks-claude-sonnet-4-6",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "databricks-claude-haiku-4-5",
    },
}

SAMPLE_GEMINI_ENV = textwrap.dedent("""\
    MY_CUSTOM_KEY="some-value"
    ANOTHER_VAR="keep-this"
""")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Set up an isolated config home and patch module-level paths."""
    import coding_tool_gateway.cli as cli

    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    gemini_dir = tmp_path / ".gemini"
    gemini_dir.mkdir()

    monkeypatch.setattr(cli, "CLAUDE_SETTINGS_PATH", claude_dir / "settings.json")
    monkeypatch.setattr(cli, "GEMINI_ENV_PATH", gemini_dir / ".env")

    return tmp_path


# ---------------------------------------------------------------------------
# Claude merge tests
# ---------------------------------------------------------------------------

class TestClaudeMerge:
    def test_merge_preserves_all_existing_keys(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.CLAUDE_SETTINGS_PATH.write_text(json.dumps(SAMPLE_CLAUDE_SETTINGS, indent=2))

        merged = cli._merge_claude_settings(GATEWAY_CLAUDE_SETTINGS)

        for key in SAMPLE_CLAUDE_SETTINGS:
            assert key in merged, f"Key '{key}' was lost during merge"

    def test_merge_adds_gateway_keys(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.CLAUDE_SETTINGS_PATH.write_text(json.dumps(SAMPLE_CLAUDE_SETTINGS, indent=2))

        merged = cli._merge_claude_settings(GATEWAY_CLAUDE_SETTINGS)

        assert merged["apiKeyHelper"] == GATEWAY_CLAUDE_SETTINGS["apiKeyHelper"]
        assert merged["env"]["ANTHROPIC_MODEL"] == "databricks-claude-opus-4-6"
        assert merged["env"]["ANTHROPIC_BASE_URL"].endswith("/ai-gateway/anthropic/v1")

    def test_merge_preserves_user_env_vars(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.CLAUDE_SETTINGS_PATH.write_text(json.dumps(SAMPLE_CLAUDE_SETTINGS, indent=2))

        merged = cli._merge_claude_settings(GATEWAY_CLAUDE_SETTINGS)

        assert merged["env"]["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] == "1"
        assert merged["env"]["MY_CUSTOM_VAR"] == "keep-me"

    def test_merge_preserves_mcp_servers(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.CLAUDE_SETTINGS_PATH.write_text(json.dumps(SAMPLE_CLAUDE_SETTINGS, indent=2))

        merged = cli._merge_claude_settings(GATEWAY_CLAUDE_SETTINGS)

        assert set(merged["mcpServers"].keys()) == {"slack", "jira"}

    def test_merge_preserves_allow_rules(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.CLAUDE_SETTINGS_PATH.write_text(json.dumps(SAMPLE_CLAUDE_SETTINGS, indent=2))

        merged = cli._merge_claude_settings(GATEWAY_CLAUDE_SETTINGS)

        assert merged["allow"] == ["Bash", "Read", "Write"]

    def test_merge_preserves_hooks(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.CLAUDE_SETTINGS_PATH.write_text(json.dumps(SAMPLE_CLAUDE_SETTINGS, indent=2))

        merged = cli._merge_claude_settings(GATEWAY_CLAUDE_SETTINGS)

        assert "PreToolUse" in merged["hooks"]

    def test_merge_with_no_existing_file(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.CLAUDE_SETTINGS_PATH.unlink(missing_ok=True)

        merged = cli._merge_claude_settings(GATEWAY_CLAUDE_SETTINGS)

        assert merged["apiKeyHelper"] == GATEWAY_CLAUDE_SETTINGS["apiKeyHelper"]
        assert "ANTHROPIC_MODEL" in merged["env"]

    def test_merge_with_corrupt_json(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.CLAUDE_SETTINGS_PATH.write_text("{ not valid json !!!")

        merged = cli._merge_claude_settings(GATEWAY_CLAUDE_SETTINGS)

        assert merged["apiKeyHelper"] == GATEWAY_CLAUDE_SETTINGS["apiKeyHelper"]

    def test_merge_with_empty_file(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.CLAUDE_SETTINGS_PATH.write_text("{}")

        merged = cli._merge_claude_settings(GATEWAY_CLAUDE_SETTINGS)

        assert merged["apiKeyHelper"] == GATEWAY_CLAUDE_SETTINGS["apiKeyHelper"]
        assert "ANTHROPIC_MODEL" in merged["env"]

    def test_merge_is_idempotent(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.CLAUDE_SETTINGS_PATH.write_text(json.dumps(SAMPLE_CLAUDE_SETTINGS, indent=2))

        first = cli._merge_claude_settings(GATEWAY_CLAUDE_SETTINGS)
        cli.CLAUDE_SETTINGS_PATH.write_text(json.dumps(first, indent=2))
        second = cli._merge_claude_settings(GATEWAY_CLAUDE_SETTINGS)

        assert first == second


# ---------------------------------------------------------------------------
# Claude unmerge tests
# ---------------------------------------------------------------------------

class TestClaudeUnmerge:
    def test_unmerge_removes_gateway_keys(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.CLAUDE_SETTINGS_PATH.write_text(json.dumps(SAMPLE_CLAUDE_SETTINGS, indent=2))

        merged = cli._merge_claude_settings(GATEWAY_CLAUDE_SETTINGS)
        cli.CLAUDE_SETTINGS_PATH.write_text(json.dumps(merged, indent=2))
        cli._unmerge_claude_settings()

        restored = json.loads(cli.CLAUDE_SETTINGS_PATH.read_text())
        assert "apiKeyHelper" not in restored
        assert "ANTHROPIC_MODEL" not in restored.get("env", {})
        assert "ANTHROPIC_BASE_URL" not in restored.get("env", {})

    def test_unmerge_preserves_user_settings(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.CLAUDE_SETTINGS_PATH.write_text(json.dumps(SAMPLE_CLAUDE_SETTINGS, indent=2))

        merged = cli._merge_claude_settings(GATEWAY_CLAUDE_SETTINGS)
        cli.CLAUDE_SETTINGS_PATH.write_text(json.dumps(merged, indent=2))
        cli._unmerge_claude_settings()

        restored = json.loads(cli.CLAUDE_SETTINGS_PATH.read_text())
        assert set(restored.keys()) == set(SAMPLE_CLAUDE_SETTINGS.keys())
        assert restored["mcpServers"] == SAMPLE_CLAUDE_SETTINGS["mcpServers"]
        assert restored["allow"] == SAMPLE_CLAUDE_SETTINGS["allow"]
        assert restored["hooks"] == SAMPLE_CLAUDE_SETTINGS["hooks"]

    def test_unmerge_preserves_user_env_vars(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.CLAUDE_SETTINGS_PATH.write_text(json.dumps(SAMPLE_CLAUDE_SETTINGS, indent=2))

        merged = cli._merge_claude_settings(GATEWAY_CLAUDE_SETTINGS)
        cli.CLAUDE_SETTINGS_PATH.write_text(json.dumps(merged, indent=2))
        cli._unmerge_claude_settings()

        restored = json.loads(cli.CLAUDE_SETTINGS_PATH.read_text())
        assert restored["env"]["MY_CUSTOM_VAR"] == "keep-me"
        assert restored["env"]["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] == "1"

    def test_unmerge_no_file_returns_false(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.CLAUDE_SETTINGS_PATH.unlink(missing_ok=True)

        assert cli._unmerge_claude_settings() is False

    def test_full_roundtrip(self, sandbox):
        """configure -> launch -> launch -> logout should leave settings intact."""
        import coding_tool_gateway.cli as cli
        original = dict(SAMPLE_CLAUDE_SETTINGS)
        cli.CLAUDE_SETTINGS_PATH.write_text(json.dumps(original, indent=2))

        # Simulate multiple configures (merges)
        for _ in range(3):
            merged = cli._merge_claude_settings(GATEWAY_CLAUDE_SETTINGS)
            cli.CLAUDE_SETTINGS_PATH.write_text(json.dumps(merged, indent=2))

        # Simulate logout (unmerge)
        cli._unmerge_claude_settings()

        restored = json.loads(cli.CLAUDE_SETTINGS_PATH.read_text())
        assert set(restored.keys()) == set(original.keys())
        assert restored["mcpServers"] == original["mcpServers"]
        assert restored["env"] == original["env"]


# ---------------------------------------------------------------------------
# Gemini merge tests
# ---------------------------------------------------------------------------

class TestGeminiMerge:
    def test_merge_preserves_existing_vars(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.GEMINI_ENV_PATH.write_text(SAMPLE_GEMINI_ENV)

        new_vars = {
            "GEMINI_MODEL": "databricks-gemini-3-1-pro",
            "GOOGLE_GEMINI_BASE_URL": "https://test.cloud.databricks.com/ai-gateway/gemini/v1",
            "GEMINI_API_KEY_AUTH_MECHANISM": "bearer",
            "GEMINI_API_KEY": "dapi-test-token",
        }
        result = cli._merge_gemini_env(new_vars)

        assert 'MY_CUSTOM_KEY="some-value"' in result
        assert 'ANOTHER_VAR="keep-this"' in result
        assert 'GEMINI_MODEL="databricks-gemini-3-1-pro"' in result
        assert 'GEMINI_API_KEY="dapi-test-token"' in result

    def test_merge_updates_existing_gateway_vars(self, sandbox):
        import coding_tool_gateway.cli as cli
        existing = 'GEMINI_MODEL="old-model"\nMY_VAR="keep"\n'
        cli.GEMINI_ENV_PATH.write_text(existing)

        new_vars = {"GEMINI_MODEL": "new-model"}
        result = cli._merge_gemini_env(new_vars)

        assert result.count("GEMINI_MODEL") == 1
        assert 'GEMINI_MODEL="new-model"' in result
        assert 'MY_VAR="keep"' in result

    def test_merge_with_no_existing_file(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.GEMINI_ENV_PATH.unlink(missing_ok=True)

        new_vars = {"GEMINI_MODEL": "test-model", "GEMINI_API_KEY": "token"}
        result = cli._merge_gemini_env(new_vars)

        assert 'GEMINI_MODEL="test-model"' in result
        assert 'GEMINI_API_KEY="token"' in result


class TestGeminiUnmerge:
    def test_unmerge_removes_only_gateway_vars(self, sandbox):
        import coding_tool_gateway.cli as cli
        content = textwrap.dedent("""\
            # Managed by coding-gateway. Run `coding-gateway logout` to restore the prior config.
            MY_CUSTOM_KEY="keep"
            GEMINI_MODEL="databricks-gemini-3-1-pro"
            GEMINI_API_KEY="secret-token"
            ANOTHER_VAR="also-keep"
        """)
        cli.GEMINI_ENV_PATH.write_text(content)

        cli._unmerge_gemini_env()

        result = cli.GEMINI_ENV_PATH.read_text()
        assert "MY_CUSTOM_KEY" in result
        assert "ANOTHER_VAR" in result
        assert "GEMINI_MODEL" not in result
        assert "GEMINI_API_KEY" not in result
        assert "Managed by coding-gateway" not in result

    def test_unmerge_deletes_file_if_only_gateway_vars(self, sandbox):
        import coding_tool_gateway.cli as cli
        content = textwrap.dedent("""\
            # Managed by coding-gateway. Run `coding-gateway logout` to restore the prior config.
            GEMINI_MODEL="model"
            GEMINI_API_KEY="token"
            GOOGLE_GEMINI_BASE_URL="url"
            GEMINI_API_KEY_AUTH_MECHANISM="bearer"
        """)
        cli.GEMINI_ENV_PATH.write_text(content)

        cli._unmerge_gemini_env()

        assert not cli.GEMINI_ENV_PATH.exists()

    def test_unmerge_no_file_returns_false(self, sandbox):
        import coding_tool_gateway.cli as cli
        cli.GEMINI_ENV_PATH.unlink(missing_ok=True)

        assert cli._unmerge_gemini_env() is False
