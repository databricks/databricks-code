"""Tests for CLI subcommand routing and passthrough args."""

from __future__ import annotations

import contextlib
import os
import re
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ucode.cli import app
from ucode.smart_routing.routing import RoutingDecision

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


class TestVersion:
    @pytest.mark.parametrize("flag", ["--version", "-V"])
    def test_prints_version_and_exits(self, flag):
        result = runner.invoke(app, [flag])
        assert result.exit_code == 0
        # Matches the derived version reported by importlib.metadata — either a
        # real string like "0.1.0" / "0.1.0+2.g93986a8" or the "unknown" fallback.
        assert _strip_ansi(result.output).strip() != ""

    def test_matches_telemetry_version(self):
        from ucode.telemetry import ucode_version

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert ucode_version() in _strip_ansi(result.output)


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
        patch("ucode.cli._fetch_managed_config", return_value=None),
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
            patches[6],
            patches[7] as mock_launch,
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

    def test_workspace_flag_sets_current_workspace(self):
        """--workspace targets that workspace (normalized) before launch."""
        patches = _patch_launch("claude")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patch("ucode.cli.set_current_workspace") as mock_set,
        ):
            result = runner.invoke(
                app,
                ["claude", "--workspace", "https://eng-ml-inference.staging.cloud.databricks.com/"],
            )
        assert result.exit_code == 0, result.output
        mock_set.assert_called_once_with("https://eng-ml-inference.staging.cloud.databricks.com")

    def test_no_workspace_flag_leaves_current_workspace(self):
        """Without --workspace, launch never reassigns the current workspace."""
        patches = _patch_launch("claude")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patch("ucode.cli.set_current_workspace") as mock_set,
        ):
            result = runner.invoke(app, ["claude"])
        assert result.exit_code == 0, result.output
        mock_set.assert_not_called()

    def test_codex_enable_smart_routing_is_consumed_by_ucode(self):
        with patch("ucode.cli._launch_tool") as mock_launch:
            result = runner.invoke(app, ["codex", "--enable-smart-routing"])

        assert result.exit_code == 0, result.output
        assert mock_launch.call_args.kwargs["enable_smart_routing_flag"] is True
        assert mock_launch.call_args.args[1].args == []

    def test_codex_disable_removes_hooks_without_launching(self):
        with (
            patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.codex_agent.disable_smart_routing") as mock_disable,
            patch("ucode.cli._launch_tool") as mock_launch,
        ):
            result = runner.invoke(app, ["codex", "--disable-smart-routing"])

        assert result.exit_code == 0, result.output
        mock_disable.assert_called_once_with(MINIMAL_STATE)
        mock_launch.assert_not_called()
        assert "routing hooks removed" in result.output

    def test_codex_routing_flags_are_mutually_exclusive(self):
        result = runner.invoke(
            app,
            ["codex", "--enable-smart-routing", "--disable-smart-routing"],
        )

        assert result.exit_code == 1
        assert "Use only one" in result.output

    def test_enabled_codex_launch_uses_routed_root_model(self):
        state = {
            **MINIMAL_STATE,
            "smart_routing_enabled": True,
            "codex_models": ["databricks-gpt-5", "databricks-gpt-5-5"],
        }
        decision = RoutingDecision(
            model="databricks-gpt-5-5",
            raw_model="gpt-5-6-sol",
            rationale="Cross-cutting refactor.",
        )
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.ensure_provider_state", return_value=state),
            patch("ucode.cli.configure_shared_state", return_value=state),
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(state, "databricks-gpt-5"),
            ),
            patch(
                "ucode.cli.codex_routing.route_launch_model",
                return_value=(decision, None),
            ),
            patch("ucode.cli.configure_tool", return_value=state) as mock_configure,
            patch("ucode.cli._fetch_managed_config", return_value=None),
            patch("ucode.cli.launch_agent"),
        ):
            result = runner.invoke(app, ["codex"])

        assert result.exit_code == 0, result.output
        assert mock_configure.call_args.args[2] == "databricks-gpt-5-5"
        # The launch notice surfaces both the routed model and the rationale.
        assert (
            "Using Smart Routing. Routing to databricks-gpt-5-5. Cross-cutting refactor."
            in _strip_ansi(result.output)
        )


class TestClaudeModelFlag:
    """`ucode claude --model <id>` pins the id into the family aliases so the gateway resolves any
    Databricks model id, instead of Claude Code's own --model flag rejecting non-catalog ids."""

    def test_model_threads_through_to_launch(self):
        with patch("ucode.cli._launch_tool") as mock_launch:
            result = runner.invoke(app, ["claude", "--model", "cat.schema.claude-opus-5"])
        assert result.exit_code == 0, result.output
        assert mock_launch.call_args.kwargs["model"] == "cat.schema.claude-opus-5"

    def test_model_threads_to_claude_as_custom_model(self):
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.ensure_provider_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.configure_shared_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.resolve_launch_model", return_value=(MINIMAL_STATE, "system.ai.opus")),
            patch("ucode.cli.configure_tool", return_value=MINIMAL_STATE) as mock_configure,
            patch("ucode.cli._fetch_managed_config", return_value=None),
            patch("ucode.cli.launch_agent"),
        ):
            result = runner.invoke(app, ["claude", "--model", "cat.schema.claude-opus-5"])
        assert result.exit_code == 0, result.output
        # Claude routes --model as custom_model (pinned into the family aliases by render_overlay),
        # NOT as ANTHROPIC_MODEL — Claude Code validates that value and rejects a raw id.
        assert mock_configure.call_args.kwargs["custom_model"] == "cat.schema.claude-opus-5"
        assert mock_configure.call_args.kwargs["route_root_model"] is None

    def test_model_and_provider_are_mutually_exclusive(self):
        result = runner.invoke(
            app, ["claude", "--model", "cat.schema.m", "--provider", "cat.schema.svc"]
        )
        assert result.exit_code == 1
        assert "Use either --model or --provider" in result.output

    def test_warns_when_enterprise_settings_pin_the_model(self):
        # Claude Code's enterprise managed-settings scope outranks the --settings file ucode writes,
        # so --model is silently ignored; warn instead of launching on the "wrong" model unexplained.
        from pathlib import Path

        with (
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.ensure_provider_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.configure_shared_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.resolve_launch_model", return_value=(MINIMAL_STATE, "system.ai.opus")),
            patch("ucode.cli.configure_tool", return_value=MINIMAL_STATE),
            patch("ucode.cli._fetch_managed_config", return_value=None),
            patch(
                "ucode.cli.claude_agent.managed_settings_model_overrides",
                return_value=Path("/etc/claude-code/managed-settings.json"),
            ),
            patch("ucode.cli.launch_agent"),
        ):
            result = runner.invoke(app, ["claude", "--model", "main.aarushi.claude-opus-5"])
        assert result.exit_code == 0, result.output
        assert "enterprise managed settings" in _strip_ansi(result.output)
        assert "overrides `--model main.aarushi.claude-opus-5`" in _strip_ansi(result.output)

    def test_no_enterprise_warning_when_no_managed_settings(self):
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.ensure_provider_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.configure_shared_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.resolve_launch_model", return_value=(MINIMAL_STATE, "system.ai.opus")),
            patch("ucode.cli.configure_tool", return_value=MINIMAL_STATE),
            patch("ucode.cli._fetch_managed_config", return_value=None),
            patch("ucode.cli.claude_agent.managed_settings_model_overrides", return_value=None),
            patch("ucode.cli.launch_agent"),
        ):
            result = runner.invoke(app, ["claude", "--model", "main.aarushi.claude-opus-5"])
        assert result.exit_code == 0, result.output
        assert "enterprise managed settings" not in _strip_ansi(result.output)


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

    def test_default_mode_dispatches_download_with_path(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(
                app, ["configure", "skills", "--location", "a.b", "--path", "/tmp/skills"]
            )
        assert result.exit_code == 0, result.output
        mock_download.assert_called_once_with(["a.b"], path="/tmp/skills", skills=None)

    def test_default_mode_without_path_dispatches_download(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(app, ["configure", "skills", "--location", "a.b"])
        assert result.exit_code == 0, result.output
        mock_download.assert_called_once_with(["a.b"], path=None, skills=None)

    def test_skill_filter_dispatches_download_with_subset(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(
                app, ["configure", "skills", "--location", "a.b", "--skill", "my_skill"]
            )
        assert result.exit_code == 0, result.output
        mock_download.assert_called_once_with(["a.b"], path=None, skills={"my_skill"})

    def test_skill_filter_parses_comma_list(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(
                app, ["configure", "skills", "--location", "a.b", "--skill", "s1, s2"]
            )
        assert result.exit_code == 0, result.output
        mock_download.assert_called_once_with(["a.b"], path=None, skills={"s1", "s2"})

    def test_skill_with_mcp_exit_1(self):
        with (
            patch("ucode.cli.configure_skills_mcp_command") as mock_mcp,
            patch("ucode.cli.configure_skills_download_command") as mock_download,
        ):
            result = runner.invoke(
                app, ["configure", "skills", "--location", "a.b", "--mcp", "--skill", "my_skill"]
            )
        assert result.exit_code == 1
        assert "--skill" in _strip_ansi(result.output)
        mock_mcp.assert_not_called()
        mock_download.assert_not_called()

    def test_skill_without_location_exit_1(self):
        with (
            patch("ucode.cli.configure_skills_mcp_command") as mock_mcp,
            patch("ucode.cli.configure_skills_download_command") as mock_download,
        ):
            result = runner.invoke(app, ["configure", "skills", "--skill", "my_skill"])
        assert result.exit_code == 1
        assert "--skill" in _strip_ansi(result.output)
        mock_mcp.assert_not_called()
        mock_download.assert_not_called()

    def test_skill_with_multiple_locations_exit_1(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(
                app, ["configure", "skills", "--location", "a.b, c.d", "--skill", "my_skill"]
            )
        assert result.exit_code == 1
        output = _strip_ansi(result.output)
        assert "--skill requires a single --location" in output
        mock_download.assert_not_called()

    def test_path_with_mcp_exit_1(self):
        with (
            patch("ucode.cli.configure_skills_mcp_command") as mock_mcp,
            patch("ucode.cli.configure_skills_download_command") as mock_download,
        ):
            result = runner.invoke(
                app, ["configure", "skills", "--location", "a.b", "--mcp", "--path", "/tmp/skills"]
            )
        assert result.exit_code == 1
        assert "--path" in _strip_ansi(result.output)
        mock_mcp.assert_not_called()
        mock_download.assert_not_called()

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

    def test_bare_command_registers_schemaless_connection(self):
        with patch("ucode.cli.configure_skills_mcp_command") as mock_mcp:
            result = runner.invoke(app, ["configure", "skills"])
        assert result.exit_code == 0, result.output
        mock_mcp.assert_called_once_with([])

    def test_mcp_without_location_registers_schemaless_connection(self):
        with patch("ucode.cli.configure_skills_mcp_command") as mock_mcp:
            result = runner.invoke(app, ["configure", "skills", "--mcp"])
        assert result.exit_code == 0, result.output
        mock_mcp.assert_called_once_with([])

    def test_path_without_location_exit_1(self):
        with (
            patch("ucode.cli.configure_skills_mcp_command") as mock_mcp,
            patch("ucode.cli.configure_skills_download_command") as mock_download,
        ):
            result = runner.invoke(app, ["configure", "skills", "--path", "/tmp/skills"])
        assert result.exit_code == 1
        assert "--path" in _strip_ansi(result.output)
        mock_mcp.assert_not_called()
        mock_download.assert_not_called()


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
            patch("ucode.cli._fetch_managed_config", return_value=None),
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
            patch("ucode.cli._fetch_managed_config", return_value=None),
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
            patch("ucode.cli._fetch_managed_config", return_value=None),
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
            patches[6],
            patches[7] as mock_launch,
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
            patches[6],
            patches[7] as mock_launch,
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
            # Fully-interactive configure ends by offering the MCP step; decline it.
            patch("ucode.cli.prompt_yes_no", return_value=False) as mock_mcp_prompt,
            patch("ucode.cli.configure_mcp_command") as mock_mcp,
        ):
            # No flag: the AI Tools prompt happens later, inside
            # configure_workspace_command, so nothing is forwarded here.
            result = runner.invoke(app, ["configure"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(prompt_optional_updates=True)
        mock_mcp_prompt.assert_called_once()
        mock_mcp.assert_not_called()

    def test_interactive_accepting_mcp_prompt_runs_mcp_config(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command"),
            patch("ucode.cli.prompt_yes_no", return_value=True),
            patch("ucode.cli.configure_mcp_command") as mock_mcp,
        ):
            result = runner.invoke(app, ["configure"])
        assert result.exit_code == 0, result.output
        mock_mcp.assert_called_once_with()

    def test_agents_flag_skips_mcp_prompt(self):
        # Flag-driven (non-interactive) runs must stay scriptable: no MCP prompt.
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command"),
            patch("ucode.cli.prompt_yes_no") as mock_prompt,
            patch("ucode.cli.configure_mcp_command") as mock_mcp,
        ):
            result = runner.invoke(app, ["configure", "--agents", "claude,codex"])
        assert result.exit_code == 0, result.output
        mock_prompt.assert_not_called()
        mock_mcp.assert_not_called()

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
            # Fully-interactive configure ends by offering the MCP step; decline it.
            patch("ucode.cli.prompt_yes_no", return_value=False),
            patch("ucode.cli.configure_mcp_command"),
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
            # Fully-interactive configure ends by offering the MCP step; decline it.
            patch("ucode.cli.prompt_yes_no", return_value=False),
            patch("ucode.cli.configure_mcp_command"),
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
            patch("ucode.cli._fetch_managed_config", return_value=None),
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


class TestRejectDisabledAgent:
    """`enabled_agents` is an allowlist: an agent the admin didn't enable would launch unmanaged."""

    @staticmethod
    def _reject(managed, tool):
        import ucode.cli as cli_mod

        cli_mod._reject_disabled_agent(managed, tool)

    def test_raises_naming_the_enabled_agents(self):
        managed = {"enabled_agents": {"claude": {}, "opencode": {}}}
        with pytest.raises(RuntimeError, match="doesn't enable Gemini CLI") as exc:
            self._reject(managed, "gemini")
        assert "Claude Code, OpenCode" in str(exc.value)

    def test_allows_an_enabled_agent(self):
        self._reject({"enabled_agents": {"claude": {}}}, "claude")

    @pytest.mark.parametrize("managed", [None, {}, {"budget_policy": {}}])
    def test_a_config_naming_no_agents_blocks_nothing(self, managed):
        # No managed config, or one that only sets a budget policy, expresses no opinion on agents.
        self._reject(managed, "gemini")


class TestFetchManagedConfig:
    """The launch path's managed-config read, which gates both the allowlist and model discovery."""

    @staticmethod
    def _fetch(state, *, skip_preflight=False):
        import ucode.cli as cli_mod

        return cli_mod._fetch_managed_config(state, skip_preflight=skip_preflight)

    def test_fetches_fresh_when_enabled(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        monkeypatch.setattr(
            "ucode.cli.refresh_managed_config", lambda state: {"enabled_agents": {}}
        )
        assert self._fetch({"workspace": "https://w"}) == {"enabled_agents": {}}

    @pytest.mark.parametrize("env_value", [None, "", "0", "off", "no"])
    def test_disabled_reads_nothing_at_all(self, monkeypatch, env_value):
        """While the feature is opt-in, a disabled launch must not read the config or the network."""
        if env_value is None:
            monkeypatch.delenv("ENABLE_MANAGED_AGENT_CONFIG", raising=False)
        else:
            monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", env_value)
        for name in ("refresh_managed_config", "load_managed_state"):
            monkeypatch.setattr(
                f"ucode.cli.{name}",
                lambda *a, called=name, **k: pytest.fail(f"{called} must not run when disabled"),
            )
        assert self._fetch({"workspace": "https://w"}) is None

    def test_skip_preflight_reads_the_cache_without_fetching(self, monkeypatch):
        # Headless launchers pass --skip-preflight to avoid per-launch network calls.
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        monkeypatch.setattr(
            "ucode.cli.refresh_managed_config", lambda state: pytest.fail("should not fetch")
        )
        monkeypatch.setattr("ucode.cli.load_managed_state", lambda ws: {"enabled_agents": {}})
        assert self._fetch({"workspace": "https://w"}, skip_preflight=True) == {
            "enabled_agents": {}
        }

    def test_skip_preflight_with_an_empty_cache_is_none(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        monkeypatch.setattr("ucode.cli.load_managed_state", lambda ws: {})
        assert self._fetch({"workspace": "https://w"}, skip_preflight=True) is None


class TestManagedConfigDecidesDiscoveryFromFreshRead:
    def test_a_removed_model_list_no_longer_skips_discovery(self, monkeypatch):
        """The sweep decision must come from the fetched config, not the cached one.

        An admin who removes a previously published model list leaves a cache that still names
        models. Deciding from that cache would skip discovery for a config that no longer supplies
        models, so the launch would have neither.
        """
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        stale_cache = {
            "enabled_agents": {"claude": {"model_config": {"models": {"default_opus_model": "m"}}}}
        }
        fresh = {"enabled_agents": {"claude": {"model_config": {}}}}
        monkeypatch.setattr("ucode.cli.load_managed_state", lambda ws: stale_cache)
        monkeypatch.setattr("ucode.cli.refresh_managed_config", lambda state: fresh)

        state = dict(MINIMAL_STATE)
        with (
            patch("ucode.cli.normalize_tool", return_value="claude"),
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.apply_pat_environment"),
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.ensure_provider_state", return_value=state),
            patch("ucode.cli.configure_shared_state", return_value=state) as mock_shared,
            patch("ucode.cli.configure_tool", return_value=state),
            patch("ucode.cli.launch_agent"),
        ):
            result = runner.invoke(app, ["claude"])

        assert result.exit_code == 0, result.output
        assert mock_shared.call_args.kwargs["skip_model_discovery"] is False


class TestConfigureDeprecation:
    """`ucode configure` resolves the target workspace first, then short-circuits once a managed
    config exists for it, since the admin's wins anyway."""

    @staticmethod
    def _resolve(entries=None):
        import ucode.cli as cli_mod

        return cli_mod._resolve_workspace_then_maybe_reject(entries)

    def test_shows_summary_and_exits_when_a_managed_config_exists(self, monkeypatch, capsys):
        import typer

        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        monkeypatch.setattr("ucode.cli.set_current_workspace", lambda ws: None)
        monkeypatch.setattr("ucode.cli.load_state", lambda: {"workspace": "https://w"})
        monkeypatch.setattr(
            "ucode.cli.refresh_managed_config",
            lambda state: {"enabled_agents": {"claude": {}}},
        )
        with pytest.raises(typer.Exit) as exc:
            self._resolve([("https://w", None)])
        assert exc.value.exit_code == 0
        out = capsys.readouterr().out
        assert "managed config has been detected" in out
        assert "run `ucode`" in out

    def test_fetches_the_config_rather_than_reading_a_cold_cache(self, monkeypatch):
        # The gap this guards: on a fresh machine the local cache is empty until the first launch,
        # so a cache read would miss a config the workspace does publish. The resolver must fetch.
        import typer

        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        monkeypatch.setattr("ucode.cli.set_current_workspace", lambda ws: None)
        monkeypatch.setattr("ucode.cli.load_state", lambda: {"workspace": "https://w"})
        # Cold cache — a cache read would wrongly fall through to the local configure flow.
        monkeypatch.setattr("ucode.cli.load_managed_state", lambda ws: None)
        monkeypatch.setattr(
            "ucode.cli.refresh_managed_config",
            lambda state: {"enabled_agents": {"claude": {}}},
        )
        with pytest.raises(typer.Exit) as exc:
            self._resolve([("https://w", None)])
        assert exc.value.exit_code == 0

    @staticmethod
    def _stub_not_admin(monkeypatch):
        # No managed config -> the resolver now checks admin status; keep the developer case simple.
        monkeypatch.setattr("ucode.cli.get_databricks_token", lambda ws, profile=None: "tok")
        monkeypatch.setattr("ucode.cli.is_workspace_admin", lambda ws, tok: False)

    def test_prompts_for_the_workspace_before_checking_the_config(self, monkeypatch):
        # The whole point: even under a managed config the developer can still switch workspaces,
        # so the prompt runs (and the picked workspace is made current) before the config check.
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        picked = []
        monkeypatch.setattr(
            "ucode.cli._prompt_for_configuration", lambda tool=None: ("https://picked", None)
        )
        monkeypatch.setattr("ucode.cli.set_current_workspace", lambda ws: picked.append(ws))
        monkeypatch.setattr("ucode.cli.refresh_managed_config", lambda state: None)
        self._stub_not_admin(monkeypatch)
        entries = self._resolve(None)
        assert picked == ["https://picked"]
        assert entries == [("https://picked", None)]

    def test_returns_flag_entries_and_proceeds_without_a_managed_config(self, monkeypatch):
        # Setting up a new workspace still goes through `ucode configure`, so hand the resolved
        # workspace back to the caller instead of re-prompting.
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        monkeypatch.setattr("ucode.cli.set_current_workspace", lambda ws: None)
        monkeypatch.setattr("ucode.cli.refresh_managed_config", lambda state: None)
        self._stub_not_admin(monkeypatch)
        entries = self._resolve([("https://w", None)])
        assert entries == [("https://w", None)]

    def test_admin_who_declines_proceeds_with_configure(self, monkeypatch):
        # An admin on a config-less workspace is offered `ucode setup`; declining continues the
        # normal configure flow and returns the resolved workspace.
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        monkeypatch.setattr("ucode.cli.set_current_workspace", lambda ws: None)
        monkeypatch.setattr("ucode.cli.refresh_managed_config", lambda state: None)
        monkeypatch.setattr("ucode.cli.get_databricks_token", lambda ws, profile=None: "tok")
        monkeypatch.setattr("ucode.cli.is_workspace_admin", lambda ws, tok: True)
        monkeypatch.setattr("ucode.cli.prompt_yes_no", lambda prompt: False)
        entries = self._resolve([("https://w", None)])
        assert entries == [("https://w", None)]

    def test_admin_who_accepts_launches_setup_in_place(self, monkeypatch):
        # Accepting the offer runs `setup_command` right there — reusing the workspace/profile we
        # already resolved so setup doesn't re-prompt for them — then exits with its code.
        import typer

        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        monkeypatch.setattr("ucode.cli.set_current_workspace", lambda ws: None)
        monkeypatch.setattr("ucode.cli.refresh_managed_config", lambda state: None)
        monkeypatch.setattr("ucode.cli.get_databricks_token", lambda ws, profile=None: "tok")
        monkeypatch.setattr("ucode.cli.is_workspace_admin", lambda ws, tok: True)
        monkeypatch.setattr("ucode.cli.prompt_yes_no", lambda prompt: True)
        calls = {}
        monkeypatch.setattr(
            "ucode.cli.setup_command",
            lambda **kwargs: calls.update(kwargs) or 0,
        )
        with pytest.raises(typer.Exit) as exc:
            self._resolve([("https://w", None)])
        assert exc.value.exit_code == 0
        assert calls == {"workspace": "https://w", "profile": None}

    def test_admin_who_accepts_propagates_setup_failure(self, monkeypatch):
        # A RuntimeError from setup (e.g. discovery failed) surfaces as a non-zero exit, not a stack
        # trace bubbling out of configure.
        import typer

        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        monkeypatch.setattr("ucode.cli.set_current_workspace", lambda ws: None)
        monkeypatch.setattr("ucode.cli.refresh_managed_config", lambda state: None)
        monkeypatch.setattr("ucode.cli.get_databricks_token", lambda ws, profile=None: "tok")
        monkeypatch.setattr("ucode.cli.is_workspace_admin", lambda ws, tok: True)
        monkeypatch.setattr("ucode.cli.prompt_yes_no", lambda prompt: True)
        monkeypatch.setattr(
            "ucode.cli.setup_command",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("no agents")),
        )
        with pytest.raises(typer.Exit) as exc:
            self._resolve([("https://w", None)])
        assert exc.value.exit_code == 1

    def test_non_admin_is_never_prompted_for_setup(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        monkeypatch.setattr("ucode.cli.set_current_workspace", lambda ws: None)
        monkeypatch.setattr("ucode.cli.refresh_managed_config", lambda state: None)
        monkeypatch.setattr("ucode.cli.get_databricks_token", lambda ws, profile=None: "tok")
        monkeypatch.setattr("ucode.cli.is_workspace_admin", lambda ws, tok: False)
        monkeypatch.setattr(
            "ucode.cli.prompt_yes_no",
            lambda prompt: pytest.fail("non-admins must not be prompted for setup"),
        )
        entries = self._resolve([("https://w", None)])
        assert entries == [("https://w", None)]

    def test_admin_check_failure_skips_the_setup_prompt(self, monkeypatch):
        # `is_workspace_admin` returns None when the check itself fails; treat as "don't prompt".
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        monkeypatch.setattr("ucode.cli.set_current_workspace", lambda ws: None)
        monkeypatch.setattr("ucode.cli.refresh_managed_config", lambda state: None)
        monkeypatch.setattr("ucode.cli.get_databricks_token", lambda ws, profile=None: "tok")
        monkeypatch.setattr("ucode.cli.is_workspace_admin", lambda ws, tok: None)
        monkeypatch.setattr(
            "ucode.cli.prompt_yes_no",
            lambda prompt: pytest.fail("must not prompt when admin status is unknown"),
        )
        entries = self._resolve([("https://w", None)])
        assert entries == [("https://w", None)]

    def test_configure_command_exits_zero_without_erroring(self, monkeypatch):
        # `typer.Exit(0)` subclasses RuntimeError, so the command's own RuntimeError handler must
        # not catch the clean exit and print `str(exc)` -> a bare, meaningless "ERROR 0".
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        monkeypatch.setattr("ucode.cli.set_current_workspace", lambda ws: None)
        monkeypatch.setattr("ucode.cli.load_state", lambda: {"workspace": "https://w"})
        monkeypatch.setattr(
            "ucode.cli.refresh_managed_config",
            lambda state: {"enabled_agents": {"claude": {}}},
        )
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli._prompt_for_configuration", return_value=("https://w", None)),
        ):
            result = runner.invoke(app, ["configure"])
        assert result.exit_code == 0, result.output
        assert "ERROR" not in result.output

    @pytest.mark.parametrize("env_value", [None, "", "0"])
    def test_passes_entries_through_when_the_env_var_is_off(self, monkeypatch, capsys, env_value):
        if env_value is None:
            monkeypatch.delenv("ENABLE_MANAGED_AGENT_CONFIG", raising=False)
        else:
            monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", env_value)
        monkeypatch.setattr(
            "ucode.cli.load_managed_state",
            lambda ws: pytest.fail("must not read the config when disabled"),
        )
        monkeypatch.setattr(
            "ucode.cli._prompt_for_configuration",
            lambda tool=None: pytest.fail("must not prompt when disabled"),
        )
        assert self._resolve(None) is None
        assert capsys.readouterr().out == ""


class TestPolicySummary:
    """The box shown to a developer when their admin's config is applied."""

    MANAGED = {
        "default_agent": "claude",
        "enabled_agents": {"claude": {"model_config": {"default_model": "system.ai.opus"}}},
        "budget_policy": {
            "display_name": "paved-path",
            # A fraction of the budget, as the API validates it: 0.8 renders as "at 80%".
            "tiers": [
                {"spending_percentage": 0.8, "default_agent": "opencode", "default_model": "haiku"}
            ],
        },
    }

    def test_lists_the_tiers_and_the_applied_model(self, capsys):
        import ucode.cli as cli_mod

        cli_mod._print_managed_summary(self.MANAGED, {"workspace": "https://w"}, "claude")
        out = capsys.readouterr().out
        assert "paved-path" in out
        assert "at 80%" in out and "OpenCode" in out and "haiku" in out
        assert "system.ai.opus" in out

    def test_lists_managed_mcps_and_skills(self, capsys):
        import ucode.cli as cli_mod

        managed = {
            **self.MANAGED,
            "mcp_servers": [{"name": "system.ai.slack", "type": "mcp-service"}],
            "skills": {"names": ["main.default.my_skill"]},
        }
        cli_mod._print_managed_summary(managed, {"workspace": "https://w"}, "claude")
        out = capsys.readouterr().out
        assert "system.ai.slack" in out
        assert "main.default.my_skill" in out
        # Marked pending until ucode registers them locally.
        assert "pending" in out

    def test_mcp_and_skill_rows_say_none_when_the_config_names_none(self, capsys):
        import ucode.cli as cli_mod

        # Shown rather than omitted: a missing row leaves "my admin set none" ambiguous.
        cli_mod._print_managed_summary(self.MANAGED, {"workspace": "https://w"}, "claude")
        out = capsys.readouterr().out
        assert "MCPs:" in out and "Skills:" in out
        assert out.count("none configured") == 2
        assert "pending" not in out

    def test_no_policy_rows_without_a_budget_policy(self, capsys):
        import ucode.cli as cli_mod

        cli_mod._print_managed_summary(
            {"enabled_agents": {"claude": {}}}, {"workspace": "w"}, "claude"
        )
        out = capsys.readouterr().out
        assert "Policy:" not in out
        assert "Claude Code" in out


class TestBareUcode:
    """Bare `ucode` launches the managed default agent, or explains why it can't."""

    MANAGED = {"default_agent": "claude", "enabled_agents": {"claude": {}, "opencode": {}}}

    @staticmethod
    def _run(monkeypatch, *, managed, is_admin=False, args=None, cached=None):
        launched: list[tuple] = []
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        monkeypatch.setattr("ucode.cli.install_databricks_cli", lambda *a, **k: None)
        monkeypatch.setattr("ucode.cli.apply_pat_environment", lambda *a, **k: None)
        monkeypatch.setattr("ucode.cli.load_state", lambda: {"workspace": "https://w"})
        monkeypatch.setattr("ucode.cli.refresh_managed_config", lambda state: managed)
        monkeypatch.setattr("ucode.cli.load_managed_state", lambda ws: cached)
        monkeypatch.setattr("ucode.cli.get_databricks_token", lambda *a, **k: "tok")
        monkeypatch.setattr("ucode.cli.is_workspace_admin", lambda *a, **k: is_admin)
        monkeypatch.setattr(
            "ucode.cli._launch_tool",
            lambda tool, ctx, **kw: launched.append((tool, kw)),
        )
        result = runner.invoke(app, args or [])
        return result, launched

    def test_launches_the_managed_default_agent(self, monkeypatch):
        result, launched = self._run(monkeypatch, managed=self.MANAGED)
        assert result.exit_code == 0, result.output
        assert launched and launched[0][0] == "claude"
        assert "paved" not in result.output  # no policy set in this config
        assert "Claude Code" in result.output

    def test_falls_back_to_the_first_enabled_agent(self, monkeypatch):
        managed = {"enabled_agents": {"opencode": {}}}
        result, launched = self._run(monkeypatch, managed=managed)
        assert result.exit_code == 0, result.output
        assert launched[0][0] == "opencode"

    def test_admin_without_a_config_is_pointed_at_setup(self, monkeypatch):
        result, launched = self._run(monkeypatch, managed=None, is_admin=True)
        assert result.exit_code == 0, result.output
        assert launched == []
        assert "ucode setup" in result.output

    def test_non_admin_without_a_config_is_told_to_ask(self, monkeypatch):
        result, launched = self._run(monkeypatch, managed=None, is_admin=False)
        assert result.exit_code == 0, result.output
        assert launched == []
        assert "Ask a workspace admin" in result.output

    def test_dry_run_uses_the_cache_and_does_not_fetch(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        monkeypatch.setattr("ucode.cli.install_databricks_cli", lambda *a, **k: None)
        monkeypatch.setattr("ucode.cli.apply_pat_environment", lambda *a, **k: None)
        monkeypatch.setattr("ucode.cli.load_state", lambda: {"workspace": "https://w"})
        monkeypatch.setattr(
            "ucode.cli.refresh_managed_config",
            lambda state: pytest.fail("--dry-run must not fetch"),
        )
        monkeypatch.setattr("ucode.cli.load_managed_state", lambda ws: self.MANAGED)
        launched: list[tuple] = []
        monkeypatch.setattr(
            "ucode.cli._launch_tool", lambda tool, ctx, **kw: launched.append((tool, kw))
        )
        result = runner.invoke(app, ["--dry-run"])
        assert result.exit_code == 0, result.output
        # The config bare `ucode` already read is handed down, so the launch path does not refetch.
        assert launched[0][1]["managed"] == self.MANAGED

    def test_skip_preflight_has_no_config_to_pick_an_agent_from(self, monkeypatch):
        # --skip-preflight is deliberately unmanaged, so bare `ucode` cannot resolve an agent. It
        # must say that rather than report "no config found", which would be wrong.
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        monkeypatch.setattr("ucode.cli.install_databricks_cli", lambda *a, **k: None)
        monkeypatch.setattr("ucode.cli.apply_pat_environment", lambda *a, **k: None)
        monkeypatch.setattr("ucode.cli.load_state", lambda: {"workspace": "https://w"})
        monkeypatch.setattr(
            "ucode.cli.refresh_managed_config",
            lambda state: pytest.fail("--skip-preflight must not fetch"),
        )
        monkeypatch.setattr(
            "ucode.cli.load_managed_state",
            lambda ws: pytest.fail("--skip-preflight must not read the cache"),
        )
        result = runner.invoke(app, ["--skip-preflight"])
        assert result.exit_code == 1
        assert "ucode <agent> --skip-preflight" in result.output
        assert "No managed coding agent config was found" not in result.output

    @pytest.mark.parametrize("env_value", [None, "", "0"])
    def test_prints_help_when_the_env_var_is_off(self, monkeypatch, env_value):
        if env_value is None:
            monkeypatch.delenv("ENABLE_MANAGED_AGENT_CONFIG", raising=False)
        else:
            monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", env_value)
        monkeypatch.setattr(
            "ucode.cli.refresh_managed_config",
            lambda state: pytest.fail("must not fetch when disabled"),
        )
        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output
        assert "Usage:" in result.output

    def test_subcommands_still_work(self, monkeypatch):
        # The callback runs for every invocation, so it must not intercept `ucode status`.
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        monkeypatch.setattr(
            "ucode.cli.refresh_managed_config",
            lambda state: pytest.fail("the callback must not run for a subcommand"),
        )
        monkeypatch.setattr("ucode.cli.load_state", lambda: {"workspace": "https://w"})
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0, result.output


class TestBudgetRecommendationAtLaunch:
    """The budget read informs the launch; it never blocks it."""

    @staticmethod
    def _launch(monkeypatch, *, tool="claude", managed, recommendation=None, reason=None):
        state = dict(MINIMAL_STATE)
        calls: list[str] = []

        def fake_recommendation(workspace, token):
            calls.append(workspace)
            return recommendation, reason

        monkeypatch.setattr("ucode.cli.get_model_recommendation", fake_recommendation)
        with (
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.apply_pat_environment"),
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.ensure_provider_state", return_value=state),
            patch("ucode.cli.configure_shared_state", return_value=state),
            patch("ucode.cli.configure_tool", return_value=state) as cfg,
            patch("ucode.cli.get_databricks_token", return_value="tok"),
            patch("ucode.cli._fetch_managed_config", return_value=managed),
            patch("ucode.cli.launch_agent"),
        ):
            result = runner.invoke(app, [tool])
        return result, calls, cfg

    def test_not_checked_without_a_managed_config(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        result, calls, _ = self._launch(monkeypatch, managed=None)
        assert result.exit_code == 0, result.output
        assert calls == []

    def test_the_recommended_agent_gets_the_recommended_model(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        managed = {
            "enabled_agents": {
                "claude": {"model_config": {"default_model": "system.ai.claude-opus-4-8"}}
            }
        }
        _result, _calls, cfg = self._launch(
            monkeypatch,
            managed=managed,
            recommendation={"agent": "claude", "model": "system.ai.claude-haiku-4-5"},
        )
        assert cfg.call_args.args[2] == "system.ai.claude-haiku-4-5"

    def test_another_agent_keeps_its_own_model_and_is_told_why(self, monkeypatch):
        # A tier's model belongs to the tier's agent; pinning it on claude would land a Kimi id in
        # ANTHROPIC_MODEL, which the Anthropic-dialect endpoint cannot serve.
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        managed = {
            "enabled_agents": {
                "claude": {"model_config": {"default_model": "system.ai.claude-opus-4-8"}},
                "opencode": {},
            }
        }
        result, _calls, cfg = self._launch(
            monkeypatch,
            managed=managed,
            recommendation={
                "agent": "opencode",
                "model": "system.ai.kimi-k2-7-code",
                "current_spend": 412.5,
                "effective_threshold": 500.0,
            },
        )
        assert result.exit_code == 0, result.output
        assert cfg.call_args.args[2] == "system.ai.claude-opus-4-8"
        assert "recommends OpenCode" in result.output

    def test_a_failed_read_does_not_block_the_launch(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        result, _calls, _cfg = self._launch(
            monkeypatch,
            managed={"enabled_agents": {"claude": {}}},
            recommendation=None,
            reason="HTTP 500",
        )
        assert result.exit_code == 0, result.output
        assert "Could not check your budget" in result.output

    def test_a_token_failure_does_not_block_the_launch(self, monkeypatch):
        # Auth can lapse between the config refresh and the budget check.
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        state = dict(MINIMAL_STATE)
        monkeypatch.setattr("ucode.cli.get_model_recommendation", lambda ws, tok: (None, None))
        with (
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.apply_pat_environment"),
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.ensure_provider_state", return_value=state),
            patch("ucode.cli.configure_shared_state", return_value=state),
            patch("ucode.cli.configure_tool", return_value=state),
            patch("ucode.cli.get_databricks_token", side_effect=RuntimeError("token expired")),
            patch(
                "ucode.cli._fetch_managed_config",
                return_value={"enabled_agents": {"claude": {}}},
            ),
            patch("ucode.cli.launch_agent"),
        ):
            result = runner.invoke(app, ["claude"])
        assert result.exit_code == 0, result.output
        assert "Could not check your budget" in result.output

    def test_shows_the_budget_bar(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", "1")
        result, _calls, _cfg = self._launch(
            monkeypatch,
            managed={"enabled_agents": {"claude": {}}},
            recommendation={
                "agent": "claude",
                "model": "m",
                "current_spend": 412.5,
                "effective_threshold": 500.0,
            },
        )
        assert result.exit_code == 0, result.output
        assert "83% used" in result.output
        assert "█" in result.output

    @pytest.mark.parametrize("env_value", [None, "", "0"])
    def test_not_checked_when_the_env_var_is_off(self, monkeypatch, env_value):
        if env_value is None:
            monkeypatch.delenv("ENABLE_MANAGED_AGENT_CONFIG", raising=False)
        else:
            monkeypatch.setenv("ENABLE_MANAGED_AGENT_CONFIG", env_value)
        result, calls, _ = self._launch(monkeypatch, managed=None)
        assert result.exit_code == 0, result.output
        assert calls == []
