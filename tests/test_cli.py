"""Tests for CLI subcommand routing and passthrough args."""

from __future__ import annotations

import contextlib
import os
import re
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ucode.cli import app

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Drop SGR escape sequences so substring assertions match regardless of
    whether the runner forces color rendering (e.g. CI sets FORCE_COLOR=1,
    which makes rich split styled tokens like ``--agents`` with ANSI codes)."""
    return _ANSI_RE.sub("", text)


runner = CliRunner()

TOOLS = ["codex", "claude", "gemini", "opencode"]


@pytest.fixture(autouse=True)
def no_state_writes():
    """Prevent any test from writing to the real state file on disk."""
    with (
        patch("ucode.state.save_state"),
        patch("ucode.cli.save_state"),
        patch("ucode.agents.__init__.save_state"),
        patch("ucode.agents.codex.save_state"),
        patch("ucode.agents.claude.save_state"),
        patch("ucode.agents.gemini.save_state"),
        patch("ucode.agents.opencode.save_state"),
    ):
        yield


@pytest.fixture(autouse=True)
def no_blocking_ai_tools_prompt():
    """The interactive configure flow prompts for AI Tools; default it to yes so
    tests that drive that path don't block reading stdin. Tests that assert on the
    prompt override this with their own patch."""
    with patch("ucode.cli.prompt_yes_no_default", lambda msg, *, default: default):
        yield


MINIMAL_STATE = {
    "workspace": "https://example.databricks.com",
    "base_urls": {
        "codex": "https://example.databricks.com/ai-gateway/codex",
        "claude": "https://example.databricks.com/ai-gateway/anthropic",
        "gemini": "https://example.databricks.com/ai-gateway/gemini",
        "opencode": "https://example.databricks.com/ai-gateway/opencode",
    },
    "claude_models": {"sonnet": "databricks-claude-sonnet-4"},
    "gemini_models": ["gemini-2.0-flash"],
    "codex_models": ["codex-mini"],
    "opencode_models": {"anthropic": ["databricks-claude-sonnet-4"]},
    "managed_configs": {},
    "available_tools": TOOLS,
}


class TestHelp:
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # no_args_is_help=True exits with code 0 or 2 depending on typer version
        assert result.exit_code in (0, 2)
        assert "Usage:" in result.output

    def test_help_lists_all_agent_subcommands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for tool in TOOLS:
            assert tool in result.output

    @pytest.mark.parametrize("tool", TOOLS)
    def test_subcommand_help(self, tool):
        result = runner.invoke(app, [tool, "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_configure_help_lists_agents_flag(self):
        result = runner.invoke(app, ["configure", "--help"])
        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        # Typer wraps long help text across lines and pads with box-drawing
        # characters; collapse whitespace + box chars before substring-matching.
        flat = re.sub(r"[│╭╮╯╰─\s]+", " ", output)
        assert "--agents" in output
        assert "comma-separated list of agents" in flat
        assert "--workspaces" in output


def _patch_launch(tool: str):
    """Return a context-manager stack that makes _launch_tool a no-op.

    load_state returns MINIMAL_STATE (workspace + tool already configured) so
    the auto-configure path is skipped entirely. configure_shared_state is
    also stubbed to avoid the launch-time refetch hitting the network.
    """
    return [
        patch("ucode.cli.ensure_bootstrap_dependencies"),
        patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
        patch(
            "ucode.cli.ensure_provider_state",
            return_value=MINIMAL_STATE,
        ),
        patch(
            "ucode.cli.configure_shared_state",
            return_value=MINIMAL_STATE,
        ),
        patch(
            "ucode.cli.resolve_launch_model",
            return_value=(MINIMAL_STATE, "databricks-claude-sonnet-4"),
        ),
        patch(
            "ucode.cli.configure_tool",
            return_value=MINIMAL_STATE,
        ),
        patch("ucode.cli.launch_agent"),
    ]


class TestSubcommandRouting:
    @pytest.mark.parametrize("tool", TOOLS)
    def test_subcommand_calls_correct_tool(self, tool):
        patches = _patch_launch(tool)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6] as mock_launch,
        ):
            result = runner.invoke(app, [tool])
        assert result.exit_code == 0, result.output
        mock_launch.assert_called_once()
        called_tool = mock_launch.call_args[0][0]
        assert called_tool == tool

    def test_no_agent_flag(self):
        """--agent flag must no longer exist."""
        result = runner.invoke(app, ["--agent", "claude"])
        assert result.exit_code != 0


class TestMcpSubcommands:
    def test_web_search_subcommand_help(self):
        result = runner.invoke(app, ["mcp", "web-search", "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_mcp_group_lists_web_search(self):
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "web-search" in result.output


class TestAuthTokenCommand:
    """`ucode auth-token` is the cross-platform apiKeyHelper (#116)."""

    @pytest.fixture(autouse=True)
    def _isolated_bearer(self):
        # The --use-pat path writes DATABRICKS_BEARER directly; restore it so
        # writes by code under test don't leak into other tests.
        original = os.environ.pop("DATABRICKS_BEARER", None)
        yield
        if original is None:
            os.environ.pop("DATABRICKS_BEARER", None)
        else:
            os.environ["DATABRICKS_BEARER"] = original

    def test_prints_only_the_token_to_stdout(self):
        with (
            patch("ucode.cli.load_state", return_value={"workspace": "https://ws"}),
            patch("ucode.cli.get_databricks_token", return_value="tok-123") as fetch,
        ):
            result = runner.invoke(app, ["auth-token"])
        assert result.exit_code == 0
        # Nothing but the bare token (plus trailing newline) may reach stdout,
        # or the consuming agent will treat the noise as part of the token.
        assert result.stdout == "tok-123\n"
        fetch.assert_called_once_with("https://ws", None)

    def test_host_and_profile_override_state(self):
        with (
            patch("ucode.cli.load_state", return_value={"workspace": "https://saved"}),
            patch("ucode.cli.get_databricks_token", return_value="tok") as fetch,
        ):
            result = runner.invoke(
                app, ["auth-token", "--host", "https://override", "--profile", "prod"]
            )
        assert result.exit_code == 0
        fetch.assert_called_once_with("https://override", "prod")

    def test_errors_without_workspace(self):
        with patch("ucode.cli.load_state", return_value={}):
            result = runner.invoke(app, ["auth-token"])
        assert result.exit_code == 1
        # The error goes to stderr, never stdout.
        assert result.stdout == ""

    def test_hidden_from_top_level_help(self):
        result = runner.invoke(app, ["--help"])
        assert "auth-token" not in _strip_ansi(result.output)

    def test_use_pat_emits_resolved_pat(self, monkeypatch):
        # --use-pat reads the profile's static PAT, exports it as
        # DATABRICKS_BEARER, and get_databricks_token returns it directly.
        monkeypatch.delenv("DATABRICKS_BEARER", raising=False)
        monkeypatch.setattr("ucode.databricks.resolve_pat_token", lambda p: "dapi-pat")
        with (
            patch("ucode.cli.load_state", return_value={"workspace": "https://ws"}),
            patch(
                "ucode.cli.get_databricks_token",
                side_effect=lambda w, p: os.environ.get("DATABRICKS_BEARER", ""),
            ),
        ):
            result = runner.invoke(app, ["auth-token", "--use-pat", "--profile", "p"])
        assert result.exit_code == 0
        assert result.stdout == "dapi-pat\n"

    def test_use_pat_ignores_empty_bearer_env(self, monkeypatch):
        # A stray empty DATABRICKS_BEARER must not shadow the PAT and force the
        # OAuth path (the regression that motivated ensure_pat_bearer).
        monkeypatch.setenv("DATABRICKS_BEARER", "")
        monkeypatch.setattr("ucode.databricks.resolve_pat_token", lambda p: "dapi-pat")
        with (
            patch("ucode.cli.load_state", return_value={"workspace": "https://ws"}),
            patch(
                "ucode.cli.get_databricks_token",
                side_effect=lambda w, p: os.environ.get("DATABRICKS_BEARER", ""),
            ),
        ):
            result = runner.invoke(app, ["auth-token", "--use-pat", "--profile", "p"])
        assert result.exit_code == 0
        assert result.stdout == "dapi-pat\n"

    def test_use_pat_fails_closed_without_pat(self, monkeypatch):
        # --use-pat with no resolvable PAT must error, NOT fall through to OAuth
        # (which can't serve a PAT-only profile and yields a misleading message).
        monkeypatch.delenv("DATABRICKS_BEARER", raising=False)
        monkeypatch.setattr("ucode.databricks.resolve_pat_token", lambda p: None)
        with (
            patch("ucode.cli.load_state", return_value={"workspace": "https://ws"}),
            patch("ucode.cli.get_databricks_token", return_value="oauth-tok") as fetch,
        ):
            result = runner.invoke(app, ["auth-token", "--use-pat", "--profile", "p"])
        assert result.exit_code == 1
        # Never attempted OAuth, and nothing leaked to stdout.
        fetch.assert_not_called()
        assert result.stdout == ""

    def test_use_pat_honors_non_empty_bearer_env(self, monkeypatch):
        # A real pre-set bearer (CI escape hatch) wins over the profile PAT.
        monkeypatch.setenv("DATABRICKS_BEARER", "ci-bearer")
        monkeypatch.setattr("ucode.databricks.resolve_pat_token", lambda p: "dapi-pat")
        with (
            patch("ucode.cli.load_state", return_value={"workspace": "https://ws"}),
            patch(
                "ucode.cli.get_databricks_token",
                side_effect=lambda w, p: os.environ.get("DATABRICKS_BEARER", ""),
            ),
        ):
            result = runner.invoke(app, ["auth-token", "--use-pat", "--profile", "p"])
        assert result.exit_code == 0
        assert result.stdout == "ci-bearer\n"


class TestStatus:
    def test_shows_mcp_list_commands(self):
        with patch("ucode.cli.load_state", return_value=MINIMAL_STATE):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "Managed by Databricks" not in result.output
        assert "MCP list command:" in result.output
        assert "claude mcp list" in result.output
        assert "codex mcp list" in result.output
        assert "gemini mcp list" in result.output
        assert "opencode mcp list" in result.output
        assert "copilot mcp list" not in result.output

    def test_shows_mcp_servers_configured_by_ucode(self):
        state = {
            **MINIMAL_STATE,
            "mcp_servers": [
                {
                    "name": "github-mcp",
                    "url": "https://example.databricks.com/api/2.0/mcp/external/github-mcp",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["claude", "codex"],
                },
                {
                    "name": "databricks-sql",
                    "url": "https://example.databricks.com/api/2.0/mcp/sql",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["gemini"],
                },
            ],
        }
        with patch("ucode.cli.load_state", return_value=state):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "github-mcp" in result.output
        assert "MCP servers: github-mcp" in result.output
        assert "databricks-sql" in result.output
        assert "MCP servers: databricks-sql" in result.output
        assert "MCP Servers" not in result.output
        assert "MCP Server:" not in result.output
        assert "Configured tools:" not in result.output

    def test_status_treats_available_tools_as_configured_agents(self):
        state = {
            **MINIMAL_STATE,
            "available_tools": ["copilot"],
            "base_urls": {
                **MINIMAL_STATE["base_urls"],
                "copilot": "https://example.databricks.com/ai-gateway/copilot",
            },
            "mcp_servers": [
                {
                    "name": "databricks-sql",
                    "url": "https://example.databricks.com/api/2.0/mcp/sql",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["copilot"],
                }
            ],
        }
        with patch("ucode.cli.load_state", return_value=state):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "copilot mcp list" in result.output
        assert "MCP servers: databricks-sql" in result.output
        assert "codex mcp list" not in result.output
        assert "claude mcp list" not in result.output
        assert "gemini mcp list" not in result.output
        assert "https://example.databricks.com/ai-gateway/anthropic" not in result.output
        assert "https://example.databricks.com/ai-gateway/gemini" not in result.output


class TestConfigureSkillsCommand:
    def test_mcp_flag_dispatches_location_set(self):
        with patch("ucode.cli.configure_skills_mcp_command") as mock_mcp:
            result = runner.invoke(app, ["configure", "skills", "--location", "a.b", "--mcp"])
        assert result.exit_code == 0, result.output
        mock_mcp.assert_called_once_with(["a.b"])

    def test_comma_location_yields_multiple_schemas(self):
        with patch("ucode.cli.configure_skills_mcp_command") as mock_mcp:
            result = runner.invoke(app, ["configure", "skills", "--location", "a.b, c.d", "--mcp"])
        assert result.exit_code == 0, result.output
        mock_mcp.assert_called_once_with(["a.b", "c.d"])

    def test_without_mcp_is_not_implemented_exit_1(self):
        with patch("ucode.cli.configure_skills_mcp_command") as mock_mcp:
            result = runner.invoke(app, ["configure", "skills", "--location", "a.b"])
        assert result.exit_code == 1
        mock_mcp.assert_not_called()

    def test_three_part_location_exit_1(self):
        with patch("ucode.cli.configure_skills_mcp_command") as mock_mcp:
            result = runner.invoke(app, ["configure", "skills", "--location", "a.b.c", "--mcp"])
        assert result.exit_code == 1
        mock_mcp.assert_not_called()

    def test_malformed_location_exit_1_names_location(self):
        with patch("ucode.cli.configure_skills_mcp_command") as mock_mcp:
            result = runner.invoke(app, ["configure", "skills", "--location", "justone", "--mcp"])
        assert result.exit_code == 1
        assert "--location" in _strip_ansi(result.output)
        mock_mcp.assert_not_called()

    def test_missing_location_is_typer_usage_error(self):
        result = runner.invoke(app, ["configure", "skills"])
        assert result.exit_code == 2


class TestStatusSkillsSection:
    def _run(self, state):
        with patch("ucode.cli.load_state", return_value=state):
            return runner.invoke(app, ["status"])

    def test_not_configured_when_no_skills_entry(self):
        result = self._run(MINIMAL_STATE)
        assert result.exit_code == 0, result.output
        out = _strip_ansi(result.output)
        assert "Skills" in out
        assert "not configured" in out

    def test_renders_locations_and_configured_agents(self):
        state = {
            **MINIMAL_STATE,
            "mcp_servers": [
                {
                    "name": "databricks-skill-registry",
                    "kind": "skills",
                    "skill_locations": ["main.default", "ml.prod"],
                    "url": "https://example.databricks.com/ai-gateway/skills/?schema=main.default&schema=ml.prod",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["claude", "codex"],
                }
            ],
        }
        result = self._run(state)
        assert result.exit_code == 0, result.output
        out = _strip_ansi(result.output)
        assert "Skill MCP Locations: main.default, ml.prod" in out
        assert "Configured: Claude Code, Codex" in out

    def test_renders_placeholder_when_no_locations(self):
        state = {
            **MINIMAL_STATE,
            "mcp_servers": [
                {
                    "name": "databricks-skill-registry",
                    "kind": "skills",
                    "skill_locations": [],
                    "url": "https://example.databricks.com/ai-gateway/skills/",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["claude"],
                }
            ],
        }
        result = self._run(state)
        assert result.exit_code == 0, result.output
        out = _strip_ansi(result.output)
        assert "Skill MCP Locations: none — utility tools only" in out

    def test_skills_entry_absent_from_per_client_mcp_lines(self):
        state = {
            **MINIMAL_STATE,
            "mcp_servers": [
                {
                    "name": "github-mcp",
                    "url": "https://example.databricks.com/api/2.0/mcp/external/github-mcp",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["claude"],
                },
                {
                    "name": "databricks-skill-registry",
                    "kind": "skills",
                    "skill_locations": ["main.default"],
                    "url": "https://example.databricks.com/ai-gateway/skills/?schema=main.default",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["claude"],
                },
            ],
        }
        result = self._run(state)
        assert result.exit_code == 0, result.output
        out = _strip_ansi(result.output)
        # The skills registry is managed in the Skills section, never listed on
        # a per-client "MCP servers:" line.
        for line in out.splitlines():
            if "MCP servers:" in line:
                assert "databricks-skill-registry" not in line
        assert "Skill MCP Locations: main.default" in out


class TestRevert:
    def test_reverts_mcp_configs_before_clearing_state(self):
        state = {
            **MINIMAL_STATE,
            "mcp_servers": [{"name": "github-mcp", "clients": ["claude"]}],
        }
        reverted_mcp: list[dict] = []
        cleared: list[bool] = []

        with (
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.restore_file", return_value=False),
            patch(
                "ucode.cli.revert_mcp_configs",
                side_effect=lambda loaded_state: (
                    reverted_mcp.append(loaded_state) or {"claude": True}
                ),
            ),
            patch("ucode.cli.clear_state", side_effect=lambda: cleared.append(True)),
        ):
            result = runner.invoke(app, ["revert"])

        assert result.exit_code == 0, result.output
        assert reverted_mcp == [state]
        assert cleared == [True]
        assert "Claude Code MCP config: restored" in result.output


class TestAutoConfigureOnFirstRun:
    def test_triggers_when_no_workspace(self):
        """Auto-configure runs when state has no workspace."""
        empty_state = {}
        configured_state = {**MINIMAL_STATE}
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies") as mock_bootstrap,
            patch("ucode.cli.load_state", return_value=empty_state),
            patch("ucode.cli._auto_configure_tool") as mock_auto,
            patch("ucode.cli.configure_shared_state", return_value=MINIMAL_STATE),
            patch(
                "ucode.cli.ensure_provider_state",
                return_value=configured_state,
            ),
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(configured_state, "databricks-claude-sonnet-4"),
            ),
            patch("ucode.cli.configure_tool", return_value=configured_state),
            patch("ucode.cli.launch_agent"),
        ):
            result = runner.invoke(app, ["claude"])
        assert result.exit_code == 0, result.output
        mock_bootstrap.assert_called_once_with("claude", update_existing=True)
        mock_auto.assert_called_once_with("claude")

    def test_triggers_when_tool_not_in_available_tools(self):
        """Auto-configure runs when workspace exists but the tool wasn't configured."""
        state_without_tool = {**MINIMAL_STATE, "available_tools": ["codex"]}
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies") as mock_bootstrap,
            patch("ucode.cli.load_state", return_value=state_without_tool),
            patch("ucode.cli._auto_configure_tool") as mock_auto,
            patch("ucode.cli.configure_shared_state", return_value=MINIMAL_STATE),
            patch(
                "ucode.cli.ensure_provider_state",
                return_value=MINIMAL_STATE,
            ),
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(MINIMAL_STATE, "databricks-claude-sonnet-4"),
            ),
            patch("ucode.cli.configure_tool", return_value=MINIMAL_STATE),
            patch("ucode.cli.launch_agent"),
        ):
            result = runner.invoke(app, ["claude"])
        assert result.exit_code == 0, result.output
        mock_bootstrap.assert_called_once_with("claude", update_existing=True)
        mock_auto.assert_called_once_with("claude")

    def test_skipped_when_already_configured(self):
        """Auto-configure is skipped when workspace and tool are already set up."""
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies") as mock_bootstrap,
            patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
            patch("ucode.cli._auto_configure_tool") as mock_auto,
            patch("ucode.cli.configure_shared_state", return_value=MINIMAL_STATE),
            patch(
                "ucode.cli.ensure_provider_state",
                return_value=MINIMAL_STATE,
            ),
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(MINIMAL_STATE, "databricks-claude-sonnet-4"),
            ),
            patch("ucode.cli.configure_tool", return_value=MINIMAL_STATE),
            patch("ucode.cli.launch_agent"),
        ):
            runner.invoke(app, ["claude"])
        mock_bootstrap.assert_called_once_with("claude", update_existing=False)
        mock_auto.assert_not_called()


class TestPassthroughArgs:
    @pytest.mark.parametrize(
        "tool,extra_args",
        [
            ("claude", ["-r"]),
            ("claude", ["--resume"]),
            ("codex", ["--full-auto"]),
            ("gemini", ["--debug"]),
            ("opencode", ["--model", "my-model"]),
            ("claude", ["-r", "--some-flag", "value"]),
        ],
    )
    def test_extra_args_forwarded(self, tool, extra_args):
        patches = _patch_launch(tool)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6] as mock_launch,
        ):
            result = runner.invoke(app, [tool, *extra_args])
        assert result.exit_code == 0, result.output
        forwarded = mock_launch.call_args[0][2]
        assert forwarded == extra_args

    def test_no_extra_args_passes_empty_list(self):
        patches = _patch_launch("claude")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6] as mock_launch,
        ):
            runner.invoke(app, ["claude"])
        forwarded = mock_launch.call_args[0][2]
        assert forwarded == []


class TestConfigureAgentFlag:
    def test_no_flag_calls_configure_all(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            # No flag: the AI Tools prompt happens later, inside
            # configure_workspace_command, so nothing is forwarded here.
            result = runner.invoke(app, ["configure"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(prompt_optional_updates=True)

    def test_agents_flag_calls_configure_with_tools(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary") as mock_install,
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agents", "claude,codex"])
        assert result.exit_code == 0, result.output
        mock_install.assert_not_called()
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            prompt_optional_updates=True,
        )

    def test_agents_flag_normalizes_aliases_and_dedupes(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agents", " claude-code, codex,claude "])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            prompt_optional_updates=True,
        )

    def test_workspaces_flag_calls_configure_with_workspaces(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app,
                [
                    "configure",
                    "--workspaces",
                    "first.databricks.com,https://second.databricks.com/",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            workspaces=[
                ("https://first.databricks.com", None),
                ("https://second.databricks.com", None),
            ],
            prompt_optional_updates=True,
        )

    def test_agents_and_workspaces_flags_call_configure_with_both(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app,
                ["configure", "--agents", "claude,codex", "--workspaces", "https://first.com"],
            )
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            workspaces=[("https://first.com", None)],
            prompt_optional_updates=True,
        )

    def test_agent_and_workspaces_flags_call_configure_with_both(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary") as mock_install,
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app,
                ["configure", "--agent", "claude", "--workspaces", "https://first.com"],
            )
        assert result.exit_code == 0, result.output
        mock_install.assert_called_once_with(
            "claude", strict=True, update_existing=True, prompt_optional_updates=True
        )
        mock_cfg.assert_called_once_with("claude", workspaces=[("https://first.com", None)])

    def test_agent_flag_calls_configure_with_tool(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary") as mock_install,
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agent", "claude"])
        assert result.exit_code == 0, result.output
        mock_install.assert_called_once_with(
            "claude", strict=True, update_existing=True, prompt_optional_updates=True
        )
        mock_cfg.assert_called_once_with("claude")

    def test_disable_fable_alone_implicitly_targets_claude(self):
        # Fable is Claude-only, so `--disable-fable` on its own should configure
        # claude directly instead of dropping into the interactive agent picker.
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary") as mock_install,
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--disable-fable"])
        assert result.exit_code == 0, result.output
        mock_install.assert_called_once_with(
            "claude", strict=True, update_existing=True, prompt_optional_updates=True
        )
        mock_cfg.assert_called_once_with("claude", fable_enabled=False)

    def test_enable_fable_alone_implicitly_targets_claude(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary") as mock_install,
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--enable-fable"])
        assert result.exit_code == 0, result.output
        mock_install.assert_called_once_with(
            "claude", strict=True, update_existing=True, prompt_optional_updates=True
        )
        mock_cfg.assert_called_once_with("claude", fable_enabled=True)

    def test_enable_fable_with_explicit_agents_does_not_override(self):
        # An explicit --agents selection wins; the fable flag rides along without
        # forcing the claude-only single-agent path.
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--enable-fable", "--agents", "claude,codex"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            prompt_optional_updates=True,
            fable_enabled=True,
        )

    def test_skip_upgrade_flag_disables_optional_update_prompt(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--skip-upgrade"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(prompt_optional_updates=False)

    def test_disable_databricks_ai_tools_forwards_false_and_skips_prompt(self):
        # An explicit flag suppresses the interactive prompt and forwards the choice.
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
            patch("ucode.cli.prompt_yes_no_default") as mock_prompt,
        ):
            result = runner.invoke(app, ["configure", "--disable-databricks-ai-tools"])
        assert result.exit_code == 0, result.output
        mock_prompt.assert_not_called()
        mock_cfg.assert_called_once_with(
            prompt_optional_updates=True, databricks_ai_tools_enabled=False
        )

    def test_enable_databricks_ai_tools_with_agents_forwards_true(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app, ["configure", "--enable-databricks-ai-tools", "--agents", "claude,codex"]
            )
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            prompt_optional_updates=True,
            databricks_ai_tools_enabled=True,
        )

    def _stub_interactive_configure(self, monkeypatch, shared_state):
        """Wire configure_workspace_command's interactive path; return captured info."""
        import ucode.cli as cli_mod

        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *a, **k: shared_state)
        monkeypatch.setattr(cli_mod, "check_gateway_endpoint", lambda s, t: t == "claude")
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda *a, **k: True)
        monkeypatch.setattr(cli_mod, "_maybe_select_provider_service", lambda tool, s: s)
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda s: None)
        monkeypatch.setattr(cli_mod, "validate_tool", lambda t: (True, None))
        monkeypatch.setattr(
            cli_mod, "_prompt_for_configuration", lambda tool=None: ("https://w", None)
        )
        monkeypatch.setattr(cli_mod, "prompt_for_tools", lambda options: ["claude"])
        captured = {}
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda s, tools: captured.update(state=dict(s)) or s,
        )
        prompt_calls = []
        monkeypatch.setattr(
            cli_mod,
            "prompt_yes_no_default",
            lambda msg, *, default: prompt_calls.append(default) or default,
        )
        return cli_mod, captured, prompt_calls

    def test_interactive_prompt_default_yes_when_no_prior_optout(self, monkeypatch):
        # No prior opt-out -> prompt defaults to yes; state carries True into install.
        state = {**MINIMAL_STATE, "available_tools": [], "databricks_ai_tools_enabled": True}
        cli_mod, captured, prompt_calls = self._stub_interactive_configure(monkeypatch, state)
        cli_mod.configure_workspace_command()
        assert prompt_calls == [True]  # default derived from resolved prior choice
        assert captured["state"]["databricks_ai_tools_enabled"] is True

    def test_interactive_prompt_defaults_to_prior_optout(self, monkeypatch):
        # configure_shared_state resolved a prior --disable to False; the prompt must
        # default to no so Enter doesn't silently re-enable it.
        state = {**MINIMAL_STATE, "available_tools": [], "databricks_ai_tools_enabled": False}
        cli_mod, captured, prompt_calls = self._stub_interactive_configure(monkeypatch, state)
        cli_mod.configure_workspace_command()
        assert prompt_calls == [False]
        assert captured["state"]["databricks_ai_tools_enabled"] is False

    def test_skip_upgrade_flag_with_agent_skips_optional_update(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary") as mock_install,
            patch("ucode.cli.configure_workspace_command"),
        ):
            result = runner.invoke(app, ["configure", "--agent", "claude", "--skip-upgrade"])
        assert result.exit_code == 0, result.output
        mock_install.assert_called_once_with(
            "claude", strict=True, update_existing=True, prompt_optional_updates=False
        )

    def test_skip_upgrade_flag_with_agents_forwards_to_configure(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agents", "claude,codex", "--skip-upgrade"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            prompt_optional_updates=False,
        )

    def test_agent_flag_normalizes_alias(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agent", "claude-code"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with("claude")

    def test_upgrade_runs_uv_tool_install(self):
        with patch("subprocess.run") as mock_run:
            result = runner.invoke(app, ["upgrade"])
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[:3] == ["uv", "tool", "install"]
        assert "--reinstall" in cmd
        assert any("github.com/databricks/ucode" in s for s in cmd)

    def test_upgrade_handles_uv_missing(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = runner.invoke(app, ["upgrade"])
        assert result.exit_code != 0
        assert "uv" in result.output.lower()

    def test_agent_flag_rejects_unknown(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agent", "bogus"])
        assert result.exit_code != 0
        mock_cfg.assert_not_called()

    def test_agents_flag_rejects_unknown(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agents", "claude,bogus"])
        assert result.exit_code != 0
        assert "Unsupported tool 'bogus'" in result.output
        assert "codex, claude, gemini, opencode, copilot, pi" in " ".join(result.output.split())
        mock_cfg.assert_not_called()

    def test_agents_flag_rejects_empty_list(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agents", ","])
        assert result.exit_code != 0
        mock_cfg.assert_not_called()

    def test_agent_and_agents_flags_are_mutually_exclusive(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agent", "claude", "--agents", "codex"])
        assert result.exit_code != 0
        mock_cfg.assert_not_called()

    def test_workspaces_flag_rejects_empty_list(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--workspaces", ","])
        assert result.exit_code != 0
        mock_cfg.assert_not_called()


class TestConfigureMcpFlag:
    def test_mcp_with_agents_configures_then_registers_services(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
            patch("ucode.cli.configure_mcp_command") as mock_mcp,
        ):
            result = runner.invoke(
                app,
                ["configure", "--agents", "claude", "--mcp", "system.ai.slack,system.ai.github"],
            )
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude"],
            prompt_optional_updates=True,
        )
        mock_mcp.assert_called_once_with(services={"system.ai.slack", "system.ai.github"})

    def test_mcp_only_configures_workspace_without_agent_picker(self):
        # `--mcp` with no --agents (e.g. Cursor): configure the workspace directly,
        # never the interactive agent picker, then register the MCP service.
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
            patch("ucode.cli._configure_shared_workspace_states") as mock_shared,
            patch("ucode.cli.configure_mcp_command") as mock_mcp,
        ):
            result = runner.invoke(
                app,
                [
                    "configure",
                    "--workspaces",
                    "https://ws.databricks.com",
                    "--mcp",
                    "system.ai.slack",
                ],
            )
        assert result.exit_code == 0, result.output
        # Never the model-agent picker path.
        mock_cfg.assert_not_called()
        mock_shared.assert_called_once()
        # Workspace-only: no model tools fetched.
        assert (
            mock_shared.call_args.kwargs.get("tools") == [] or mock_shared.call_args.args[1] == []
        )
        mock_mcp.assert_called_once_with(services={"system.ai.slack"})

    def test_mcp_rejects_bare_short_name(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.configure_workspace_command"),
            patch("ucode.cli._configure_shared_workspace_states"),
            patch("ucode.cli.configure_mcp_command") as mock_mcp,
        ):
            result = runner.invoke(
                app, ["configure", "--workspaces", "https://ws.databricks.com", "--mcp", "slack"]
            )
        assert result.exit_code != 0
        mock_mcp.assert_not_called()


class TestConfigureAgentsSelection:
    def test_selected_tools_skip_picker(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {**MINIMAL_STATE, "available_tools": []}
        monkeypatch.setattr(
            cli_mod,
            "_prompt_for_configuration",
            lambda tool=None: ("https://example.com", None),
        )
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *args, **kwargs: state)
        monkeypatch.setattr(
            cli_mod, "check_gateway_endpoint", lambda state, tool: tool in {"claude", "codex"}
        )
        monkeypatch.setattr(
            cli_mod,
            "prompt_for_tools",
            lambda available: pytest.fail("prompt_for_tools should not be called"),
        )
        install_calls: list[str] = []
        monkeypatch.setattr(
            cli_mod,
            "install_tool_binary",
            lambda tool, strict=False, update_existing=False, prompt_optional_updates=True: (
                install_calls.append(tool) or True
            ),
        )
        configured: list[list[str]] = []
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda state, tools: configured.append(tools) or {**state, "available_tools": tools},
        )
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda state: None)

        assert cli_mod.configure_workspace_command(selected_tools=["claude", "codex"]) == 0
        assert install_calls == ["claude", "codex"]
        assert configured == [["claude", "codex"]]

    def test_provider_picker_gated_by_interactive_path(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {**MINIMAL_STATE, "available_tools": []}
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *a, **k: state)
        monkeypatch.setattr(cli_mod, "check_gateway_endpoint", lambda s, t: t == "claude")
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda *a, **k: True)
        monkeypatch.setattr(
            cli_mod, "configure_selected_tools", lambda s, tools: {**s, "available_tools": tools}
        )
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda s: None)
        picked_for: list[str] = []
        monkeypatch.setattr(
            cli_mod,
            "_maybe_select_provider_service",
            lambda tool, s: picked_for.append(tool) or s,
        )

        # Non-interactive (--agents passed): no provider picker.
        cli_mod.configure_workspace_command(
            selected_tools=["claude"], workspaces=[("https://w.com", None)]
        )
        assert picked_for == []

        # Interactive (`ucode configure`): picker offered for each picked tool.
        monkeypatch.setattr(
            cli_mod, "_prompt_for_configuration", lambda tool=None: ("https://w.com", None)
        )
        monkeypatch.setattr(cli_mod, "prompt_for_tools", lambda options: ["claude"])
        cli_mod.configure_workspace_command()
        assert picked_for == ["claude"]

    def test_unavailable_selected_tool_errors_before_configure(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {**MINIMAL_STATE, "available_tools": []}
        monkeypatch.setattr(
            cli_mod,
            "_prompt_for_configuration",
            lambda tool=None: ("https://example.com", None),
        )
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *args, **kwargs: state)
        monkeypatch.setattr(cli_mod, "check_gateway_endpoint", lambda state, tool: tool == "claude")
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda state, tools: pytest.fail("configure_selected_tools should not be called"),
        )

        with pytest.raises(RuntimeError, match="Codex"):
            cli_mod.configure_workspace_command(selected_tools=["claude", "codex"])

    def test_multiple_workspaces_configure_all_and_use_first(self, monkeypatch):
        import ucode.cli as cli_mod

        states = {
            "https://first.com": {**MINIMAL_STATE, "workspace": "https://first.com"},
            "https://second.com": {**MINIMAL_STATE, "workspace": "https://second.com"},
        }
        configured_shared: list[tuple[str, str | None, tuple[str, ...] | None, bool]] = []

        def fake_configure_shared_state(
            workspace,
            profile=None,
            tools=None,
            force_login=False,
            use_pat=False,
            fable_enabled=None,
            databricks_ai_tools_enabled=None,
        ):
            configured_shared.append(
                (workspace, profile, tuple(tools) if tools is not None else None, force_login)
            )
            return states[workspace]

        saved: list[str] = []
        configured_tools: list[tuple[str, list[str]]] = []
        monkeypatch.setattr(cli_mod, "configure_shared_state", fake_configure_shared_state)
        monkeypatch.setattr(cli_mod, "save_state", lambda state: saved.append(state["workspace"]))
        monkeypatch.setattr(cli_mod, "check_gateway_endpoint", lambda state, tool: True)
        monkeypatch.setattr(cli_mod, "prompt_for_tools", lambda available: ["codex"])
        monkeypatch.setattr(cli_mod, "_maybe_select_provider_service", lambda tool, state: state)
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda *args, **kwargs: True)
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda state, tools: (
                configured_tools.append((state["workspace"], tools))
                or {**state, "available_tools": tools}
            ),
        )
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda state: None)

        assert (
            cli_mod.configure_workspace_command(
                workspaces=[("https://first.com", None), ("https://second.com", None)]
            )
            == 0
        )
        assert configured_shared == [
            ("https://first.com", None, None, True),
            ("https://second.com", None, None, True),
        ]
        assert saved == ["https://first.com"]
        assert configured_tools == [("https://first.com", ["codex"])]


class TestParseProfilesOption:
    @staticmethod
    def _patch_profiles(monkeypatch, entries):
        import ucode.cli as cli_mod

        monkeypatch.setattr(cli_mod, "list_profile_entries", lambda: entries)
        return cli_mod

    def test_resolves_profiles_to_workspace_entries(self, monkeypatch):
        cli_mod = self._patch_profiles(
            monkeypatch,
            [
                {"name": "DEFAULT", "host": "https://first.databricks.com/", "auth_type": "pat"},
                {
                    "name": "second",
                    "host": "https://second.databricks.com",
                    "auth_type": "databricks-cli",
                },
            ],
        )
        assert cli_mod._parse_profiles_option("DEFAULT, second") == [
            ("https://first.databricks.com", "DEFAULT"),
            ("https://second.databricks.com", "second"),
        ]

    def test_unknown_profile_raises_with_available_names(self, monkeypatch):
        cli_mod = self._patch_profiles(
            monkeypatch,
            [{"name": "DEFAULT", "host": "https://first.databricks.com", "auth_type": "pat"}],
        )
        with pytest.raises(RuntimeError, match=r"'missing' was not found.*DEFAULT"):
            cli_mod._parse_profiles_option("missing")

    def test_profile_without_host_raises(self, monkeypatch):
        cli_mod = self._patch_profiles(monkeypatch, [{"name": "DEFAULT", "auth_type": "pat"}])
        with pytest.raises(RuntimeError, match="no host configured"):
            cli_mod._parse_profiles_option("DEFAULT")

    def test_empty_value_raises(self, monkeypatch):
        cli_mod = self._patch_profiles(monkeypatch, [])
        with pytest.raises(RuntimeError, match="No profiles provided"):
            cli_mod._parse_profiles_option(" , ")


class TestConfigureProfilesFlag:
    PROFILE_ENTRIES = [
        {"name": "DEFAULT", "host": "https://first.databricks.com", "auth_type": "pat"}
    ]

    def test_profiles_flag_resolves_workspaces(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.list_profile_entries", return_value=self.PROFILE_ENTRIES),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--profiles", "DEFAULT"])
        assert result.exit_code == 0, result.output
        # Auth behaves like --workspaces: no skip flags are forwarded, so the
        # default forced OAuth login applies.
        mock_cfg.assert_called_once_with(
            workspaces=[("https://first.databricks.com", "DEFAULT")],
            prompt_optional_updates=True,
        )

    def test_profiles_flag_with_agents(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.list_profile_entries", return_value=self.PROFILE_ENTRIES),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app, ["configure", "--agents", "claude,codex", "--profiles", "DEFAULT"]
            )
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            workspaces=[("https://first.databricks.com", "DEFAULT")],
            prompt_optional_updates=True,
        )

    def test_profiles_flag_with_agent(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.list_profile_entries", return_value=self.PROFILE_ENTRIES),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agent", "claude", "--profiles", "DEFAULT"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            "claude",
            workspaces=[("https://first.databricks.com", "DEFAULT")],
        )

    def test_use_pat_and_skip_validate_are_forwarded(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.list_profile_entries", return_value=self.PROFILE_ENTRIES),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app,
                [
                    "configure",
                    "--agents",
                    "claude,codex",
                    "--profiles",
                    "DEFAULT",
                    "--use-pat",
                    "--skip-validate",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            workspaces=[("https://first.databricks.com", "DEFAULT")],
            prompt_optional_updates=True,
            use_pat=True,
            skip_validate=True,
        )

    def test_use_pat_requires_profiles(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app,
                ["configure", "--workspaces", "https://first.databricks.com", "--use-pat"],
            )
        assert result.exit_code == 1
        assert "--use-pat requires --profiles" in _strip_ansi(result.output)
        mock_cfg.assert_not_called()

    def test_profiles_and_workspaces_are_mutually_exclusive(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app,
                [
                    "configure",
                    "--profiles",
                    "DEFAULT",
                    "--workspaces",
                    "https://first.databricks.com",
                ],
            )
        assert result.exit_code == 1
        assert "not both" in _strip_ansi(result.output)
        mock_cfg.assert_not_called()


class TestConfigureSharedStateUsePat:
    """--use-pat reads the profile's PAT from ~/.databrickscfg, exports it as
    DATABRICKS_BEARER, persists the mode, and never opens a browser."""

    WS = "https://example.databricks.com"

    @pytest.fixture(autouse=True)
    def _isolated_bearer(self):
        # configure_shared_state writes DATABRICKS_BEARER directly; restore it
        # since monkeypatch can't track writes made by code under test.
        import os as os_mod

        original = os_mod.environ.pop("DATABRICKS_BEARER", None)
        yield
        if original is None:
            os_mod.environ.pop("DATABRICKS_BEARER", None)
        else:
            os_mod.environ["DATABRICKS_BEARER"] = original

    @staticmethod
    def _stub_deps(monkeypatch, *, pat_token, existing_state=None):
        import ucode.cli as cli_mod

        logins: list[tuple] = []
        ensures: list[tuple] = []
        saved: list[dict] = []
        monkeypatch.setattr(cli_mod, "load_state", lambda: dict(existing_state or {}))
        monkeypatch.setattr(cli_mod, "save_state", lambda s: saved.append(dict(s)))
        monkeypatch.setattr(cli_mod, "run_databricks_login", lambda w, p: logins.append((w, p)))
        monkeypatch.setattr(
            cli_mod, "ensure_databricks_auth", lambda w, p=None: ensures.append((w, p))
        )
        monkeypatch.setattr(cli_mod, "resolve_pat_token", lambda p: pat_token)
        monkeypatch.setattr(cli_mod, "find_profile_name_for_host", lambda w: None)
        monkeypatch.setattr(cli_mod, "get_databricks_token", lambda w, p: "token")
        monkeypatch.setattr(cli_mod, "ensure_ai_gateway_v2", lambda w, t: None)
        monkeypatch.setattr(cli_mod, "discover_model_services", lambda w, t: ({}, [], [], [], None))
        monkeypatch.setattr(cli_mod, "discover_claude_models", lambda w, t: ({}, None))
        monkeypatch.setattr(cli_mod, "discover_gemini_models", lambda w, t: ([], None))
        monkeypatch.setattr(cli_mod, "discover_codex_models", lambda w, t: ([], None))
        monkeypatch.setattr(cli_mod, "build_shared_base_urls", lambda w: {})
        return cli_mod, logins, ensures, saved

    def test_use_pat_exports_bearer_and_skips_login(self, monkeypatch):
        import os as os_mod

        cli_mod, logins, ensures, saved = self._stub_deps(monkeypatch, pat_token="dapi-pat")

        state = cli_mod.configure_shared_state(
            self.WS, profile="DEFAULT", force_login=True, use_pat=True
        )

        assert logins == []
        assert ensures == [(self.WS, "DEFAULT")]
        assert os_mod.environ["DATABRICKS_BEARER"] == "dapi-pat"
        assert state["use_pat"] is True
        assert saved and saved[-1]["use_pat"] is True

    def test_use_pat_without_pat_profile_raises(self, monkeypatch):
        cli_mod, logins, _, _ = self._stub_deps(monkeypatch, pat_token=None)

        with pytest.raises(RuntimeError, match="no personal access token"):
            cli_mod.configure_shared_state(
                self.WS, profile="oauth-profile", force_login=True, use_pat=True
            )
        assert logins == []

    def test_use_pat_without_profile_raises(self, monkeypatch):
        cli_mod, _, _, _ = self._stub_deps(monkeypatch, pat_token="dapi-pat")

        with pytest.raises(RuntimeError, match="requires a Databricks CLI profile"):
            cli_mod.configure_shared_state(self.WS, force_login=True, use_pat=True)

    def test_launch_inherits_persisted_use_pat(self, monkeypatch):
        # A launch re-run passes use_pat=None; the persisted mode for the same
        # workspace must apply so no OAuth login is forced.
        cli_mod, logins, ensures, _ = self._stub_deps(
            monkeypatch,
            pat_token="dapi-pat",
            existing_state={"workspace": self.WS, "profile": "DEFAULT", "use_pat": True},
        )

        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT", force_login=False)

        assert logins == []
        assert state["use_pat"] is True

    def test_reconfigure_without_flag_clears_use_pat(self, monkeypatch):
        cli_mod, logins, _, _ = self._stub_deps(
            monkeypatch,
            pat_token="dapi-pat",
            existing_state={"workspace": self.WS, "profile": "DEFAULT", "use_pat": True},
        )

        state = cli_mod.configure_shared_state(
            self.WS, profile="DEFAULT", force_login=True, use_pat=False
        )

        assert logins == [(self.WS, "DEFAULT")]
        assert "use_pat" not in state

    def test_uc_models_used_without_legacy_fallback(self, monkeypatch):
        # When model-services returns models, they're used and the legacy
        # per-family discovery is never consulted.
        cli_mod, *_ = self._stub_deps(monkeypatch, pat_token="dapi-pat")
        monkeypatch.setattr(
            cli_mod,
            "discover_model_services",
            lambda w, t: (
                {"opus": "system.ai.claude-opus-4-8"},
                ["system.ai.gpt-5"],
                [],
                [],
                None,
            ),
        )
        legacy_called: list[str] = []
        monkeypatch.setattr(
            cli_mod,
            "discover_claude_models",
            lambda w, t: legacy_called.append("claude") or ({}, None),
        )

        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT")

        assert state["claude_models"] == {"opus": "system.ai.claude-opus-4-8"}
        assert state["codex_models"] == ["system.ai.gpt-5"]
        assert legacy_called == []
        assert "uc_enabled" not in state

    def _stub_with_fable(self, monkeypatch):
        cli_mod, *_ = self._stub_deps(monkeypatch, pat_token="dapi-pat")
        monkeypatch.setattr(
            cli_mod,
            "discover_model_services",
            lambda w, t: (
                {"fable": "system.ai.claude-fable-5", "opus": "system.ai.claude-opus-4-8"},
                [],
                [],
                [],
                None,
            ),
        )
        return cli_mod

    def test_fable_stripped_from_discovery_when_not_enabled(self, monkeypatch):
        # Discovery buckets fable, but without --enable-fable it's dropped from
        # the persisted bundle so it never reaches any agent's config.
        cli_mod = self._stub_with_fable(monkeypatch)

        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT")

        assert state["claude_models"] == {"opus": "system.ai.claude-opus-4-8"}
        assert "fable_enabled" not in state

    def test_fable_retained_and_persisted_when_enabled(self, monkeypatch):
        cli_mod = self._stub_with_fable(monkeypatch)

        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT", fable_enabled=True)

        assert state["claude_models"]["fable"] == "system.ai.claude-fable-5"
        assert state["fable_enabled"] is True

    def test_launch_inherits_persisted_fable_opt_in(self, monkeypatch):
        # A launch re-run passes fable_enabled=None; the persisted opt-in for the
        # same workspace applies, so fable stays in the discovered bundle.
        cli_mod, *_ = self._stub_deps(
            monkeypatch,
            pat_token="dapi-pat",
            existing_state={"workspace": self.WS, "profile": "DEFAULT", "fable_enabled": True},
        )
        monkeypatch.setattr(
            cli_mod,
            "discover_model_services",
            lambda w, t: ({"fable": "system.ai.claude-fable-5"}, [], [], [], None),
        )

        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT")

        assert state["claude_models"]["fable"] == "system.ai.claude-fable-5"
        assert state["fable_enabled"] is True

    def test_reconfigure_with_disable_fable_clears_opt_in(self, monkeypatch):
        cli_mod, *_ = self._stub_deps(
            monkeypatch,
            pat_token="dapi-pat",
            existing_state={"workspace": self.WS, "profile": "DEFAULT", "fable_enabled": True},
        )
        monkeypatch.setattr(
            cli_mod,
            "discover_model_services",
            lambda w, t: ({"fable": "system.ai.claude-fable-5"}, [], [], [], None),
        )

        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT", fable_enabled=False)

        assert "fable_enabled" not in state
        assert "fable" not in state["claude_models"]

    def test_ai_tools_disable_persists(self, monkeypatch):
        cli_mod, *_ = self._stub_deps(monkeypatch, pat_token="dapi-pat")
        state = cli_mod.configure_shared_state(
            self.WS, profile="DEFAULT", databricks_ai_tools_enabled=False
        )
        assert state["databricks_ai_tools_enabled"] is False

    def test_ai_tools_enable_persists_explicit_true(self, monkeypatch):
        # We ask explicitly, so store the on choice too (not just absent-default).
        cli_mod, *_ = self._stub_deps(monkeypatch, pat_token="dapi-pat")
        state = cli_mod.configure_shared_state(
            self.WS, profile="DEFAULT", databricks_ai_tools_enabled=True
        )
        assert state["databricks_ai_tools_enabled"] is True

    def test_ai_tools_disable_inherited_same_workspace(self, monkeypatch):
        # No flag on a re-configure of the same workspace keeps the prior opt-out.
        cli_mod, *_ = self._stub_deps(
            monkeypatch,
            pat_token="dapi-pat",
            existing_state={
                "workspace": self.WS,
                "profile": "DEFAULT",
                "databricks_ai_tools_enabled": False,
            },
        )
        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT")
        assert state["databricks_ai_tools_enabled"] is False

    def test_ai_tools_disable_does_not_leak_across_workspaces(self, monkeypatch):
        # A different workspace's opt-out must NOT carry into this one; no flag
        # here resolves to the default (install=True), matching use_pat/fable scoping.
        cli_mod, *_ = self._stub_deps(
            monkeypatch,
            pat_token="dapi-pat",
            existing_state={
                "workspace": "https://other.databricks.com",
                "profile": "DEFAULT",
                "databricks_ai_tools_enabled": False,
            },
        )
        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT")
        assert state["databricks_ai_tools_enabled"] is True

    def test_falls_back_to_legacy_when_uc_empty(self, monkeypatch):
        # No UC model-services: each family falls back to the legacy listing.
        cli_mod, *_ = self._stub_deps(monkeypatch, pat_token="dapi-pat")
        monkeypatch.setattr(
            cli_mod, "discover_model_services", lambda w, t: ({}, [], [], [], "no model services")
        )
        monkeypatch.setattr(
            cli_mod,
            "discover_claude_models",
            lambda w, t: (
                {"opus": "databricks-claude-opus-4-8", "sonnet": "databricks-claude-sonnet-4-6"},
                None,
            ),
        )

        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT")

        assert state["claude_models"] == {
            "opus": "databricks-claude-opus-4-8",
            "sonnet": "databricks-claude-sonnet-4-6",
        }


class TestConfigureSkipValidate:
    def test_skip_validate_skips_agent_validation(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {**MINIMAL_STATE, "workspace": "https://first.com"}
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *a, **k: state)
        monkeypatch.setattr(cli_mod, "save_state", lambda s: None)
        monkeypatch.setattr(cli_mod, "check_gateway_endpoint", lambda s, t: True)
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda *a, **k: True)
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda s, tools: {**s, "available_tools": tools},
        )
        validated: list = []
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda s: validated.append(s))

        result = cli_mod.configure_workspace_command(
            selected_tools=["codex"],
            workspaces=[("https://first.com", None)],
            skip_validate=True,
        )

        assert result == 0
        assert validated == []

    def test_skip_validate_skips_single_tool_validation(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {**MINIMAL_STATE, "workspace": "https://first.com"}
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *a, **k: state)
        monkeypatch.setattr(cli_mod, "configure_single_tool", lambda t, s: s)
        validated: list = []
        monkeypatch.setattr(cli_mod, "validate_tool", lambda t: validated.append(t) or (True, ""))

        result = cli_mod.configure_workspace_command(
            "claude",
            workspaces=[("https://first.com", None)],
            skip_validate=True,
        )

        assert result == 0
        assert validated == []


class TestConfigureSharedStateMcpCleanup:
    """A workspace switch should scrub the previous workspace's MCP entries from
    installed client configs. Switching to the same workspace must not."""

    @staticmethod
    def _stub_external_deps(monkeypatch):
        import ucode.cli as cli_mod

        monkeypatch.setattr(cli_mod, "normalize_workspace_url", lambda w: w)
        monkeypatch.setattr(cli_mod, "run_databricks_login", lambda w, p: None)
        monkeypatch.setattr(cli_mod, "ensure_databricks_auth", lambda w, p=None: None)
        monkeypatch.setattr(cli_mod, "find_profile_name_for_host", lambda w: None)
        monkeypatch.setattr(cli_mod, "get_databricks_token", lambda w, p: "token")
        monkeypatch.setattr(cli_mod, "ensure_ai_gateway_v2", lambda w, t: None)
        monkeypatch.setattr(cli_mod, "discover_model_services", lambda w, t: ({}, [], [], [], None))
        monkeypatch.setattr(cli_mod, "discover_claude_models", lambda w, t: ({}, None))
        monkeypatch.setattr(cli_mod, "discover_gemini_models", lambda w, t: ([], None))
        monkeypatch.setattr(cli_mod, "discover_codex_models", lambda w, t: ([], None))
        monkeypatch.setattr(cli_mod, "build_shared_base_urls", lambda w: {})

    def test_purges_residue_when_workspace_changes(self, monkeypatch):
        import ucode.cli as cli_mod

        self._stub_external_deps(monkeypatch)
        monkeypatch.setattr(
            cli_mod, "load_state", lambda: {"workspace": "https://old.databricks.com"}
        )
        purge_calls: list[tuple[dict, str]] = []
        monkeypatch.setattr(
            cli_mod,
            "purge_cross_workspace_mcp_residue",
            lambda state, workspace: purge_calls.append((state, workspace)),
        )

        cli_mod.configure_shared_state("https://new.databricks.com")

        assert len(purge_calls) == 1
        _, called_workspace = purge_calls[0]
        assert called_workspace == "https://new.databricks.com"

    def test_skips_purge_when_workspace_unchanged(self, monkeypatch):
        import ucode.cli as cli_mod

        self._stub_external_deps(monkeypatch)
        monkeypatch.setattr(
            cli_mod, "load_state", lambda: {"workspace": "https://same.databricks.com"}
        )
        purge_calls: list = []
        monkeypatch.setattr(
            cli_mod,
            "purge_cross_workspace_mcp_residue",
            lambda state, workspace: purge_calls.append((state, workspace)),
        )

        cli_mod.configure_shared_state("https://same.databricks.com")

        assert purge_calls == []


class TestConfigureSharedStateSkipDiscovery:
    """With skip_model_discovery (provider mode), the heavy family discovery is
    skipped; only a single web-search model is fetched, and existing model lists
    are preserved rather than clobbered."""

    @staticmethod
    def _stub(monkeypatch):
        import ucode.cli as cli_mod

        monkeypatch.setattr(cli_mod, "normalize_workspace_url", lambda w: w)
        monkeypatch.setattr(cli_mod, "ensure_databricks_auth", lambda w, p=None: None)
        monkeypatch.setattr(cli_mod, "run_databricks_login", lambda w, p: None)
        monkeypatch.setattr(cli_mod, "find_profile_name_for_host", lambda w: None)
        monkeypatch.setattr(cli_mod, "get_databricks_token", lambda w, p: "token")
        monkeypatch.setattr(cli_mod, "ensure_ai_gateway_v2", lambda w, t: None)
        monkeypatch.setattr(cli_mod, "build_shared_base_urls", lambda w: {})
        monkeypatch.setattr(cli_mod, "save_state", lambda s: None)

    def test_skips_family_discovery_and_fetches_web_search_model(self, monkeypatch):
        import ucode.cli as cli_mod

        ws = "https://prov.databricks.com"
        self._stub(monkeypatch)
        # Pretend a prior Databricks configure left models behind.
        monkeypatch.setattr(
            cli_mod,
            "load_state",
            lambda: {"workspace": ws, "claude_models": {"opus": "databricks-claude-opus-4-8"}},
        )

        def _boom(*a, **k):
            raise AssertionError("discover_model_services must not run in provider mode")

        monkeypatch.setattr(cli_mod, "discover_model_services", _boom)
        codex_calls: list = []
        monkeypatch.setattr(
            cli_mod,
            "discover_codex_models",
            lambda w, t: codex_calls.append((w, t)) or (["databricks-gpt-5"], None),
        )

        state = cli_mod.configure_shared_state(ws, tools=["claude"], skip_model_discovery=True)

        assert codex_calls == [(ws, "token")]
        assert state["web_search_model"] == "databricks-gpt-5"
        # Existing model list preserved, not overwritten to {}.
        assert state["claude_models"] == {"opus": "databricks-claude-opus-4-8"}


class TestConfigureSharedStateSkipPreflight:
    """With skip_preflight (--skip-preflight), a prior configure is trusted:
    no auth login, token fetch, gateway probe, or model discovery runs — but the
    profile and base URLs are still resolved and state is persisted."""

    WS = "https://cfg.databricks.com"

    @staticmethod
    def _stub(monkeypatch):
        import ucode.cli as cli_mod

        def _boom(name):
            def _f(*a, **k):
                raise AssertionError(f"{name} must not run under skip_preflight")

            return _f

        monkeypatch.setattr(cli_mod, "normalize_workspace_url", lambda w: w)
        # Any network round-trip is a hard failure in this mode.
        monkeypatch.setattr(cli_mod, "ensure_databricks_auth", _boom("ensure_databricks_auth"))
        monkeypatch.setattr(cli_mod, "run_databricks_login", _boom("run_databricks_login"))
        monkeypatch.setattr(cli_mod, "ensure_pat_bearer", _boom("ensure_pat_bearer"))
        monkeypatch.setattr(cli_mod, "get_databricks_token", _boom("get_databricks_token"))
        monkeypatch.setattr(cli_mod, "ensure_ai_gateway_v2", _boom("ensure_ai_gateway_v2"))
        monkeypatch.setattr(cli_mod, "discover_model_services", _boom("discover_model_services"))
        monkeypatch.setattr(cli_mod, "discover_codex_models", _boom("discover_codex_models"))
        monkeypatch.setattr(cli_mod, "find_profile_name_for_host", lambda w: "resolved")
        monkeypatch.setattr(cli_mod, "build_shared_base_urls", lambda w: {"codex": "u/codex"})
        saved: list[dict] = []
        monkeypatch.setattr(cli_mod, "save_state", lambda s: saved.append(dict(s)))
        return cli_mod, saved

    def test_skips_auth_gateway_and_discovery_but_persists(self, monkeypatch):
        cli_mod, saved = self._stub(monkeypatch)
        monkeypatch.setattr(
            cli_mod,
            "load_state",
            lambda: {"workspace": self.WS, "codex_models": ["databricks-gpt-5"]},
        )

        state = cli_mod.configure_shared_state(
            self.WS, profile="DEFAULT", tools=["codex"], skip_preflight=True
        )

        # base_urls rebuilt and state saved, but the prior model list is left intact.
        assert state["base_urls"] == {"codex": "u/codex"}
        assert state["codex_models"] == ["databricks-gpt-5"]
        assert saved and saved[-1]["base_urls"] == {"codex": "u/codex"}

    def test_resolves_profile_locally_when_missing(self, monkeypatch):
        cli_mod, _ = self._stub(monkeypatch)
        monkeypatch.setattr(cli_mod, "load_state", lambda: {"workspace": self.WS})

        state = cli_mod.configure_shared_state(self.WS, profile=None, skip_preflight=True)

        # find_profile_name_for_host is a local ~/.databrickscfg lookup (no network).
        assert state["profile"] == "resolved"


class TestSkipPreflightFlag:
    """`--skip-preflight` on a launch command threads through _launch_tool to
    configure_shared_state as skip_preflight."""

    LAUNCH_TOOLS = ["codex", "claude", "gemini", "opencode", "copilot", "pi"]

    @staticmethod
    def _patches(cfg):
        return [
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli._auto_configure_tool"),
            patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.ensure_provider_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.configure_shared_state", cfg),
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(MINIMAL_STATE, "databricks-claude-sonnet-4"),
            ),
            patch("ucode.cli.configure_tool", return_value=MINIMAL_STATE),
            patch("ucode.cli.launch_agent"),
        ]

    @pytest.mark.parametrize("tool", LAUNCH_TOOLS)
    def test_flag_sets_skip_preflight_true(self, tool):
        cfg = MagicMock(return_value=MINIMAL_STATE)
        with contextlib.ExitStack() as stack:
            for p in self._patches(cfg):
                stack.enter_context(p)
            result = runner.invoke(app, [tool, "--skip-preflight"])
        assert result.exit_code == 0, result.output
        assert cfg.call_args.kwargs["skip_preflight"] is True

    @pytest.mark.parametrize("tool", ["codex", "gemini"])
    def test_absent_flag_defaults_false(self, tool):
        cfg = MagicMock(return_value=MINIMAL_STATE)
        with contextlib.ExitStack() as stack:
            for p in self._patches(cfg):
                stack.enter_context(p)
            result = runner.invoke(app, [tool])
        assert result.exit_code == 0, result.output
        assert cfg.call_args.kwargs["skip_preflight"] is False
