"""Tests for MCP server registration."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock

from ucode import mcp

WS = "https://example.databricks.com"
CLAUDE_STATE = {"workspace": WS, "available_tools": ["claude"]}
ALL_MCP_CLIENTS = ["claude", "codex", "gemini", "opencode", "copilot"]


class TestBuildMcpHttpEntry:
    def test_uses_http_url(self):
        entry = mcp.build_mcp_http_entry(f"{WS}/api/2.0/mcp/external/github")
        assert entry["type"] == "http"
        assert entry["url"] == f"{WS}/api/2.0/mcp/external/github"

    def test_uses_oauth_token_header_reference(self):
        entry = mcp.build_mcp_http_entry(f"{WS}/api/2.0/mcp/external/github")
        assert entry["headers"]["Authorization"] == "Bearer ${OAUTH_TOKEN}"
        assert "oauth" not in entry
        assert "headersHelper" not in entry

    def test_omits_always_load_by_default(self):
        entry = mcp.build_mcp_http_entry(f"{WS}/ai-gateway/skills/")
        assert "alwaysLoad" not in entry

    def test_sets_always_load_when_requested(self):
        entry = mcp.build_mcp_http_entry(f"{WS}/ai-gateway/skills/", always_load=True)
        assert entry["alwaysLoad"] is True


class TestAddClaudeMcpServer:
    def test_adds_user_scoped_json(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return MagicMock(returncode=0)

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        entry = mcp.build_mcp_http_entry(f"{WS}/api/2.0/mcp/external/github")
        mcp.add_claude_mcp_server("github", entry)

        assert calls
        args = calls[0]["args"]
        assert args[:4] == ["claude", "mcp", "add-json", "github"]
        assert json.loads(args[4]) == entry
        assert args[5:] == ["-s", "user"]
        assert "--client-secret" not in args
        assert "env" not in calls[0]["kwargs"]


class TestAddCodexMcpServer:
    def test_adds_http_server_with_bearer_token_env(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return MagicMock(returncode=0)

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        mcp.add_codex_mcp_server("github", f"{WS}/api/2.0/mcp/external/github")

        assert calls == [
            {
                "args": [
                    "codex",
                    "mcp",
                    "add",
                    "github",
                    "--url",
                    f"{WS}/api/2.0/mcp/external/github",
                    "--bearer-token-env-var",
                    "OAUTH_TOKEN",
                ],
                "kwargs": {
                    "check": True,
                    "capture_output": True,
                    "text": True,
                    "timeout": 30,
                },
            }
        ]


class TestAddGeminiMcpServer:
    def test_adds_user_scoped_http_server_with_auth_header(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return MagicMock(returncode=0)

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        mcp.add_gemini_mcp_server("github", f"{WS}/api/2.0/mcp/external/github")

        assert len(calls) == 1
        call = calls[0]
        assert call["args"] == [
            "gemini",
            "mcp",
            "add",
            "github",
            f"{WS}/api/2.0/mcp/external/github",
            "--type",
            "http",
            "--scope",
            "user",
            "--header",
            "Authorization: Bearer ${OAUTH_TOKEN}",
        ]
        kwargs = call["kwargs"]
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["timeout"] == 30
        # GEMINI_CLI_HOME must point at the launcher's home so `gemini mcp
        # add` writes the same settings.json the ucode session reads from.
        assert kwargs["env"]["GEMINI_CLI_HOME"] == str(mcp.gemini.GEMINI_HOME_DIR)


class TestRemoveClaudeMcpServer:
    def test_returns_true_when_server_removed(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return MagicMock(returncode=0)

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        assert mcp.remove_claude_mcp_server("github", "user") is True
        assert calls == [["claude", "mcp", "remove", "github", "-s", "user"]]

    def test_returns_false_when_server_missing(self, monkeypatch):
        def fake_run(args, **kwargs):
            raise subprocess.CalledProcessError(1, args, stderr="No MCP server named github found")

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        assert mcp.remove_claude_mcp_server("github", "user") is False

    def test_returns_false_when_project_local_server_missing(self, monkeypatch):
        def fake_run(args, **kwargs):
            raise subprocess.CalledProcessError(
                1,
                args,
                stderr="No project-local MCP server found with name: github",
            )

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        assert mcp.remove_claude_mcp_server("github", "project") is False

    def test_returns_false_when_user_scoped_server_missing(self, monkeypatch):
        def fake_run(args, **kwargs):
            raise subprocess.CalledProcessError(
                1,
                args,
                stderr="No user-scoped MCP server found with name: github",
            )

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        assert mcp.remove_claude_mcp_server("github", "user") is False

    def test_unexpected_failure_raises(self, monkeypatch):
        def fake_run(args, **kwargs):
            raise subprocess.CalledProcessError(1, args, stderr="permission denied")

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        try:
            mcp.remove_claude_mcp_server("github", "user")
        except RuntimeError as exc:
            assert "Failed to remove MCP server 'github'" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")


class TestExternalMcpConnectionNames:
    def test_returns_sorted_http_connection_names(self):
        assert mcp.external_mcp_connection_names(
            [
                {"name": "jira-mcp", "connection_type": "HTTP"},
                {"name": "not-http", "connection_type": "POSTGRESQL"},
                {"name": "confluence-mcp", "connection_type": "http"},
                {"name": "jira-mcp", "connection_type": "HTTP"},
            ]
        ) == ["confluence-mcp", "jira-mcp"]

    def test_excludes_explicit_non_mcp_http_connections(self):
        assert mcp.external_mcp_connection_names(
            [
                {
                    "name": "analytics-api",
                    "connection_type": "HTTP",
                    "options": {"is_mcp": "false"},
                },
                {"name": "github-mcp", "connection_type": "HTTP", "options": {"is_mcp": "true"}},
            ]
        ) == ["github-mcp"]


class TestConfigureClientMcpServer:
    def test_configures_copilot_mcp_server(self, monkeypatch):
        calls: list[tuple[str, str]] = []

        monkeypatch.setattr(
            mcp.copilot,
            "write_mcp_server_config",
            lambda name, url: calls.append((name, url)) or False,
        )

        removed_scopes = mcp.configure_client_mcp_server(
            "copilot",
            "github",
            f"{WS}/api/2.0/mcp/external/github",
            mcp.build_mcp_http_entry(f"{WS}/api/2.0/mcp/external/github"),
        )

        assert removed_scopes == []
        assert calls == [("github", f"{WS}/api/2.0/mcp/external/github")]


class TestMcpPicker:
    def test_prompt_uses_scrolling_checkbox_selector(self, monkeypatch):
        checkbox_calls: list[dict] = []

        class FakePrompt:
            def ask(self):
                return [f"{mcp.MCP_ADD_PREFIX}external:github-mcp"]

        def fake_checkbox(*args, **kwargs):
            checkbox_calls.append({"args": args, "kwargs": kwargs})
            return FakePrompt()

        monkeypatch.setattr(mcp, "_scrolling_checkbox", fake_checkbox)

        assert mcp.prompt_for_mcp_server_choices(["github-mcp"], [], [], []) == [
            f"{mcp.MCP_ADD_PREFIX}external:github-mcp"
        ]

        assert checkbox_calls
        choices = checkbox_calls[0]["kwargs"]["choices"]
        choice_text = [choice.title for choice in choices]
        assert "External connections" not in choice_text
        assert "Databricks managed services" not in choice_text
        assert "Custom servers" not in choice_text
        assert choice_text == [
            "Databricks SQL",
            "github-mcp",
        ]
        assert "Built-in AI tools" not in choice_text
        assert checkbox_calls[0]["kwargs"]["instruction"] == (
            "(space to toggle, enter to save, type to filter)"
        )

    def test_prompt_returns_none_when_cancelled(self, monkeypatch):
        class FakePrompt:
            def ask(self):
                return None

        monkeypatch.setattr(mcp, "_scrolling_checkbox", lambda *args, **kwargs: FakePrompt())

        assert mcp.prompt_for_mcp_server_choices(["github-mcp"], [], [], []) is None

    def test_picker_marks_configured_servers(self):
        choices = mcp.build_mcp_picker_choices(
            ["github-mcp"],
            [],
            [],
            [{"name": "github-mcp", "url": f"{WS}/api/2.0/mcp/external/github-mcp"}],
        )
        choices_by_title = {choice.title: choice for choice in choices}
        assert choices_by_title["github-mcp"].checked is True
        assert choices_by_title["Databricks SQL"].checked is False

    def test_picker_keeps_databricks_sql_when_nothing_discovered(self):
        choices = mcp.build_mcp_picker_choices([], [], [], [])
        assert [choice.title for choice in choices] == ["Databricks SQL"]
        assert choices[0].value == f"{mcp.MCP_ADD_PREFIX}managed:sql"

    def test_discovers_genie_spaces_as_mcp_servers(self):
        assert mcp.genie_mcp_servers(
            [
                {"space_id": "space-2", "title": "Second Space"},
                {"space_id": "space-1", "title": "First Space"},
                {"title": "Missing ID"},
            ],
            WS,
        ) == [
            {
                "name": "databricks-genie-first-space",
                "title": "First Space",
                "url": f"{WS}/api/2.0/mcp/genie/space-1",
            },
            {
                "name": "databricks-genie-second-space",
                "title": "Second Space",
                "url": f"{WS}/api/2.0/mcp/genie/space-2",
            },
        ]

    def test_genie_server_name_falls_back_to_space_id_on_slug_collision(self):
        assert mcp.genie_mcp_servers(
            [
                {"space_id": "space-1", "title": "New Space"},
                {"space_id": "space-2", "title": "new space"},
                {"space_id": "space-3", "title": ""},
            ],
            WS,
        ) == [
            {
                "name": "databricks-genie-new-space",
                "title": "New Space",
                "url": f"{WS}/api/2.0/mcp/genie/space-1",
            },
            {
                "name": "databricks-genie-space-2",
                "title": "new space",
                "url": f"{WS}/api/2.0/mcp/genie/space-2",
            },
            {
                "name": "databricks-genie-space-3",
                "title": "space-3",
                "url": f"{WS}/api/2.0/mcp/genie/space-3",
            },
        ]

    def test_picker_lists_discovered_genie_spaces(self):
        choices = mcp.build_mcp_picker_choices(
            ["github-mcp"],
            [
                {
                    "name": "databricks-genie-space-1",
                    "title": "First Space",
                    "url": f"{WS}/api/2.0/mcp/genie/space-1",
                }
            ],
            [],
            [],
        )

        choices_by_title = {choice.title: choice for choice in choices}
        assert choices_by_title["Genie: First Space"].value == (
            f"{mcp.MCP_ADD_PREFIX}genie-space:space-1"
        )

    def test_discovers_apps_as_mcp_servers(self):
        assert mcp.app_mcp_servers(
            [
                {
                    "name": "mcp-my-app",
                    "url": "https://mcp-my-app.example.databricksapps.com",
                },
                {
                    "name": "regular-app",
                    "url": "https://regular-app.example.databricksapps.com",
                },
                {"name": "missing-url"},
            ]
        ) == [
            {
                "name": "databricks-app-mcp-my-app",
                "title": "mcp-my-app",
                "url": "https://mcp-my-app.example.databricksapps.com/mcp",
            }
        ]

    def test_picker_lists_discovered_app_mcps(self):
        choices = mcp.build_mcp_picker_choices(
            ["github-mcp"],
            [],
            [
                {
                    "name": "databricks-app-mcp-my-app",
                    "title": "mcp-my-app",
                    "url": "https://mcp-my-app.example.databricksapps.com/mcp",
                }
            ],
            [],
        )

        choices_by_title = {choice.title: choice for choice in choices}
        assert choices_by_title["App: mcp-my-app"].value == f"{mcp.MCP_ADD_PREFIX}app:mcp-my-app"

    def test_vector_search_mcp_servers_emit_managed_url_per_pair(self):
        servers = mcp.vector_search_mcp_servers(
            [("main", "search"), ("Marketing", "Docs")],
            WS,
        )
        assert servers == [
            {
                "name": "databricks-vector-search-main-search",
                "title": "main.search",
                "catalog": "main",
                "schema": "search",
                "url": f"{WS}/api/2.0/mcp/vector-search/main/search",
            },
            {
                "name": "databricks-vector-search-marketing-docs",
                "title": "Marketing.Docs",
                "catalog": "Marketing",
                "schema": "Docs",
                "url": f"{WS}/api/2.0/mcp/vector-search/Marketing/Docs",
            },
        ]

    def test_uc_functions_mcp_servers_emit_managed_url_per_pair(self):
        servers = mcp.uc_functions_mcp_servers(
            [("analytics", "tools"), ("ml", "udfs")],
            WS,
        )
        assert servers == [
            {
                "name": "databricks-functions-analytics-tools",
                "title": "analytics.tools",
                "catalog": "analytics",
                "schema": "tools",
                "url": f"{WS}/api/2.0/mcp/functions/analytics/tools",
            },
            {
                "name": "databricks-functions-ml-udfs",
                "title": "ml.udfs",
                "catalog": "ml",
                "schema": "udfs",
                "url": f"{WS}/api/2.0/mcp/functions/ml/udfs",
            },
        ]

    def test_picker_lists_discovered_vector_search_and_uc_functions(self):
        choices = mcp.build_mcp_picker_choices(
            [],
            [],
            [],
            [],
            available_vector_search_servers=mcp.vector_search_mcp_servers([("main", "search")], WS),
            available_uc_functions_servers=mcp.uc_functions_mcp_servers(
                [("analytics", "tools")], WS
            ),
        )
        choices_by_title = {choice.title: choice for choice in choices}
        assert (
            choices_by_title["Vector Search: main.search"].value
            == f"{mcp.MCP_ADD_PREFIX}vector-search:main.search"
        )
        assert (
            choices_by_title["UC Functions: analytics.tools"].value
            == f"{mcp.MCP_ADD_PREFIX}uc-functions:analytics.tools"
        )

    def test_picker_keeps_saved_legacy_servers_for_removal(self):
        choices = mcp.build_mcp_picker_choices(
            [],
            [],
            [],
            [
                {
                    "name": "databricks-vector-search-main-search-docs",
                    "url": f"{WS}/api/2.0/mcp/vector-search/main/search/docs",
                }
            ],
        )

        choices_by_title = {choice.title: choice for choice in choices}
        assert choices_by_title["databricks-vector-search-main-search-docs"].checked is True


def _patch_mcp_choices(monkeypatch, *values: str) -> None:
    monkeypatch.setattr(
        mcp,
        "prompt_for_mcp_server_choices",
        lambda *args, **kwargs: list(values),
    )
    # Stub the always-on discoveries so configure_mcp_command tests don't hit
    # real APIs. Individual tests override these after calling the helper.
    monkeypatch.setattr(mcp, "discover_mcp_service_names", lambda workspace, profile=None: [])
    monkeypatch.setattr(
        mcp, "discover_vector_search_mcp_servers", lambda workspace, profile=None: []
    )
    monkeypatch.setattr(
        mcp, "discover_uc_functions_mcp_servers", lambda workspace, profile=None: []
    )


class TestConfigureMcpCommand:
    def test_skips_existing_server_state_by_name(self, monkeypatch):
        saved_states: list[dict] = []
        monkeypatch.setattr(
            mcp,
            "load_state",
            lambda: {
                "workspace": WS,
                "available_tools": ["claude"],
                "mcp_servers": [
                    {
                        "name": "github",
                        "url": f"{WS}/old",
                        "client_id": "old-client-id",
                        "client_secret": "old-client-secret",
                    }
                ],
            },
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, "github")
        monkeypatch.setattr(mcp, "remove_claude_mcp_server", lambda name, scope: False)
        monkeypatch.setattr(mcp, "add_claude_mcp_server", lambda name, entry, scope: None)
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert saved_states == []

    def test_registers_discovered_external_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []

        monkeypatch.setattr(
            mcp,
            "load_state",
            lambda: {"workspace": WS, "available_tools": ALL_MCP_CLIENTS},
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(
            mcp,
            "available_mcp_clients",
            lambda: ALL_MCP_CLIENTS,
        )
        monkeypatch.setattr(
            mcp,
            "discover_external_mcp_connection_names",
            lambda workspace, profile=None: ["confluence-mcp", "github-mcp"],
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, f"{mcp.MCP_ADD_PREFIX}external:github-mcp")

        def fake_configure_client_mcp_server(client, name, url, entry):
            configured.append((client, name, url, entry))
            return []

        monkeypatch.setattr(mcp, "configure_client_mcp_server", fake_configure_client_mcp_server)
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        expected_entry = {
            "type": "http",
            "url": f"{WS}/api/2.0/mcp/external/github-mcp",
            "headers": {"Authorization": "Bearer ${OAUTH_TOKEN}"},
        }
        assert configured == [
            (
                "claude",
                "github-mcp",
                f"{WS}/api/2.0/mcp/external/github-mcp",
                expected_entry,
            ),
            ("codex", "github-mcp", f"{WS}/api/2.0/mcp/external/github-mcp", expected_entry),
            ("gemini", "github-mcp", f"{WS}/api/2.0/mcp/external/github-mcp", expected_entry),
            ("opencode", "github-mcp", f"{WS}/api/2.0/mcp/external/github-mcp", expected_entry),
            ("copilot", "github-mcp", f"{WS}/api/2.0/mcp/external/github-mcp", expected_entry),
        ]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "github-mcp",
                "url": f"{WS}/api/2.0/mcp/external/github-mcp",
                "auth": "env:OAUTH_TOKEN",
                "clients": ["claude", "codex", "gemini", "opencode", "copilot"],
            }
        ]

    def test_registers_discovered_genie_space_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(
            mcp,
            "discover_genie_mcp_servers",
            lambda workspace, profile=None: [
                {
                    "name": "databricks-genie-space-123",
                    "title": "Sales Genie",
                    "url": f"{WS}/api/2.0/mcp/genie/space-123",
                }
            ],
        )
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, f"{mcp.MCP_ADD_PREFIX}genie-space:space-123")
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert configured == [
            (
                "claude",
                "databricks-genie-space-123",
                f"{WS}/api/2.0/mcp/genie/space-123",
                {
                    "type": "http",
                    "url": f"{WS}/api/2.0/mcp/genie/space-123",
                    "headers": {"Authorization": "Bearer ${OAUTH_TOKEN}"},
                },
            )
        ]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "databricks-genie-space-123",
                "url": f"{WS}/api/2.0/mcp/genie/space-123",
                "auth": "env:OAUTH_TOKEN",
                "clients": ["claude"],
            }
        ]

    def test_registers_discovered_vector_search_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(
            monkeypatch, f"{mcp.MCP_ADD_PREFIX}{mcp.VECTOR_SEARCH_SELECTION_PREFIX}main.search"
        )
        monkeypatch.setattr(
            mcp,
            "discover_vector_search_mcp_servers",
            lambda workspace, profile=None: mcp.vector_search_mcp_servers(
                [("main", "search")], workspace
            ),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert configured == [
            (
                "claude",
                "databricks-vector-search-main-search",
                f"{WS}/api/2.0/mcp/vector-search/main/search",
                {
                    "type": "http",
                    "url": f"{WS}/api/2.0/mcp/vector-search/main/search",
                    "headers": {"Authorization": "Bearer ${OAUTH_TOKEN}"},
                },
            )
        ]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "databricks-vector-search-main-search",
                "url": f"{WS}/api/2.0/mcp/vector-search/main/search",
                "auth": "env:OAUTH_TOKEN",
                "clients": ["claude"],
            }
        ]

    def test_registers_discovered_uc_functions_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(
            monkeypatch,
            f"{mcp.MCP_ADD_PREFIX}{mcp.UC_FUNCTIONS_SELECTION_PREFIX}analytics.tools",
        )
        monkeypatch.setattr(
            mcp,
            "discover_uc_functions_mcp_servers",
            lambda workspace, profile=None: mcp.uc_functions_mcp_servers(
                [("analytics", "tools")], workspace
            ),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert configured == [
            (
                "claude",
                "databricks-functions-analytics-tools",
                f"{WS}/api/2.0/mcp/functions/analytics/tools",
                {
                    "type": "http",
                    "url": f"{WS}/api/2.0/mcp/functions/analytics/tools",
                    "headers": {"Authorization": "Bearer ${OAUTH_TOKEN}"},
                },
            )
        ]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "databricks-functions-analytics-tools",
                "url": f"{WS}/api/2.0/mcp/functions/analytics/tools",
                "auth": "env:OAUTH_TOKEN",
                "clients": ["claude"],
            }
        ]

    def test_registers_discovered_app_mcp_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(
            mcp,
            "discover_app_mcp_servers",
            lambda workspace, profile=None: [
                {
                    "name": "databricks-app-mcp-my-app",
                    "title": "mcp-my-app",
                    "url": "https://mcp-my-app.example.databricksapps.com/mcp",
                }
            ],
        )
        _patch_mcp_choices(monkeypatch, f"{mcp.MCP_ADD_PREFIX}app:mcp-my-app")
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert configured == [
            (
                "claude",
                "databricks-app-mcp-my-app",
                "https://mcp-my-app.example.databricksapps.com/mcp",
                {
                    "type": "http",
                    "url": "https://mcp-my-app.example.databricksapps.com/mcp",
                    "headers": {"Authorization": "Bearer ${OAUTH_TOKEN}"},
                },
            )
        ]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "databricks-app-mcp-my-app",
                "url": "https://mcp-my-app.example.databricksapps.com/mcp",
                "auth": "env:OAUTH_TOKEN",
                "clients": ["claude"],
            }
        ]

    def test_hints_when_no_selections_and_no_existing_servers(self, monkeypatch, capsys):
        saved_states: list[dict] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch)
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        output = capsys.readouterr().out
        assert "No MCP servers selected" in output
        assert "space to toggle" in output
        assert saved_states == []

    def test_preserves_skills_connection_and_hides_it_from_picker(self, monkeypatch):
        """The picker manages mcp-services only; a skills connection is kept out
        of the choices and never removed. Selecting an mcp-service saves both."""
        saved_states: list[dict] = []
        picker_servers: list[list[dict]] = []
        removed: list[tuple[str, str]] = []
        skills_entry = {
            "name": mcp.SKILLS_MCP_SERVER_NAME,
            "kind": mcp.SKILLS_MCP_KIND,
            "skill_locations": ["main.default"],
            "url": f"{WS}/ai-gateway/skills/?schema=main.default",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }

        monkeypatch.setattr(
            mcp, "load_state", lambda: {**CLAUDE_STATE, "mcp_servers": [skills_entry]}
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_mcp_service_names", lambda workspace, profile=None: [])
        monkeypatch.setattr(
            mcp, "discover_vector_search_mcp_servers", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(
            mcp, "discover_uc_functions_mcp_servers", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(
            mcp,
            "prompt_for_mcp_server_choices",
            lambda ext, genie, app, servers, *a, **kw: (
                picker_servers.append(servers) or [f"{mcp.MCP_ADD_PREFIX}{mcp.SQL_MCP_VALUE}"]
            ),
        )
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a: [])
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert picker_servers == [[]]
        assert removed == []
        assert skills_entry in saved_states[-1]["mcp_servers"]

    def test_drops_stale_foreign_workspace_mcp_entries(self, monkeypatch, capsys):
        saved_states: list[dict] = []
        cleanup_calls: list[tuple[str, str]] = []
        other_ws = "https://other-workspace.cloud.databricks.com"
        stale_entry = {
            "name": "databricks-genie-foreign",
            "url": f"{other_ws}/api/2.0/mcp/genie/foreign",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude", "codex"],
        }
        kept_entry = {
            "name": "databricks-sql",
            "url": f"{WS}/api/2.0/mcp/sql",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }

        monkeypatch.setattr(
            mcp,
            "load_state",
            lambda: {
                "workspace": WS,
                "available_tools": ["claude"],
                "mcp_servers": [stale_entry, kept_entry],
            },
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, "databricks-sql")
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: cleanup_calls.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        output = capsys.readouterr().out
        assert "Dropping 1 stale MCP entry" in output
        assert "databricks-genie-foreign" in output
        # codex is listed on the stale entry but not installed -> skipped.
        assert cleanup_calls == [("claude", "databricks-genie-foreign")]
        assert saved_states, "expected sanitized state to be persisted"
        assert saved_states[0]["mcp_servers"] == [kept_entry]

    def test_removes_orphan_mcp_entries_from_other_workspace_buckets(self, monkeypatch, capsys):
        saved_states: list[dict] = []
        cleanup_calls: list[tuple[str, str]] = []
        other_ws = "https://other-workspace.cloud.databricks.com"
        current_entry = {
            "name": "databricks-sql",
            "url": f"{WS}/api/2.0/mcp/sql",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }
        orphan_entry = {
            "name": "orphan-mcp",
            "url": f"{other_ws}/api/2.0/mcp/external/orphan-mcp",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude", "codex"],
        }

        monkeypatch.setattr(
            mcp,
            "load_state",
            lambda: {
                "workspace": WS,
                "available_tools": ["claude"],
                "mcp_servers": [current_entry],
            },
        )
        monkeypatch.setattr(
            mcp,
            "load_full_state",
            lambda: {
                "current_workspace": WS,
                "workspaces": {
                    WS: {"mcp_servers": [current_entry]},
                    other_ws: {"mcp_servers": [orphan_entry]},
                },
            },
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, "databricks-sql")
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: cleanup_calls.append((client, name)) or [mcp.MCP_USER_SCOPE],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        output = capsys.readouterr().out
        assert "left over from previously-configured workspaces" in output
        assert "orphan-mcp" in output
        # codex was in orphan-mcp's clients but isn't installed -> skipped.
        assert cleanup_calls == [("claude", "orphan-mcp")]

    def test_skips_orphan_warning_when_nothing_was_actually_removed(self, monkeypatch, capsys):
        """Re-running configure mcp on the same workspace shouldn't repeat the warning
        if the leftover entries were already removed by a previous run."""
        cleanup_calls: list[tuple[str, str]] = []
        other_ws = "https://other-workspace.cloud.databricks.com"
        orphan_entry = {
            "name": "orphan-mcp",
            "url": f"{other_ws}/api/2.0/mcp/external/orphan-mcp",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }

        monkeypatch.setattr(
            mcp,
            "load_state",
            lambda: {"workspace": WS, "available_tools": ["claude"]},
        )
        monkeypatch.setattr(
            mcp,
            "load_full_state",
            lambda: {
                "current_workspace": WS,
                "workspaces": {
                    WS: {},
                    other_ws: {"mcp_servers": [orphan_entry]},
                },
            },
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch)
        # Stub returns empty list -> "entry wasn't in this agent's config".
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: cleanup_calls.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: None)

        assert mcp.configure_mcp_command() == 0

        output = capsys.readouterr().out
        assert "left over from previously-configured workspaces" not in output
        # The removal attempt was still made (cheap and safe); we just don't announce it.
        assert cleanup_calls == [("claude", "orphan-mcp")]

    def test_warns_when_app_selection_is_no_longer_discoverable(self, monkeypatch, capsys):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, f"{mcp.MCP_ADD_PREFIX}app:mcp-vanished")
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        output = capsys.readouterr().out
        assert "Skipped MCP selection `app:mcp-vanished`" in output
        assert "mcp-vanished" in output
        assert configured == []

    def test_warns_for_unrecognized_selection_prefix(self, monkeypatch, capsys):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, f"{mcp.MCP_ADD_PREFIX}bogus:value")
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        output = capsys.readouterr().out
        assert "Skipped MCP selection `bogus:value`" in output
        assert "unrecognized" in output
        assert configured == []

    def test_continues_when_optional_discovery_fails(self, monkeypatch, capsys):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp,
            "discover_external_mcp_connection_names",
            lambda workspace, profile=None: (_ for _ in ()).throw(
                RuntimeError("permission denied")
            ),
        )
        monkeypatch.setattr(
            mcp,
            "discover_genie_mcp_servers",
            lambda workspace, profile=None: (_ for _ in ()).throw(
                RuntimeError("permission denied")
            ),
        )
        monkeypatch.setattr(
            mcp,
            "discover_app_mcp_servers",
            lambda workspace, profile=None: (_ for _ in ()).throw(
                RuntimeError("permission denied")
            ),
        )
        _patch_mcp_choices(monkeypatch, f"{mcp.MCP_ADD_PREFIX}managed:sql")
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        output = capsys.readouterr().out
        assert "Skipped external connections." in output
        assert "Skipped Genie spaces." in output
        assert "Skipped Databricks apps." in output
        assert configured[0][1] == "databricks-sql"
        assert saved_states[-1]["mcp_servers"][0]["name"] == "databricks-sql"

    def test_forwards_profile_to_discovery(self, monkeypatch):
        saved_states: list[dict] = []
        seen_profiles: dict[str, str | None] = {}

        monkeypatch.setattr(
            mcp,
            "load_state",
            lambda: {**CLAUDE_STATE, "profile": "my-profile"},
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])

        def fake_external(workspace, profile=None):
            seen_profiles["external"] = profile
            return []

        def fake_genie(workspace, profile=None):
            seen_profiles["genie"] = profile
            return []

        def fake_apps(workspace, profile=None):
            seen_profiles["apps"] = profile
            return []

        monkeypatch.setattr(mcp, "discover_external_mcp_connection_names", fake_external)
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", fake_genie)
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", fake_apps)
        _patch_mcp_choices(monkeypatch)
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0
        assert seen_profiles == {
            "external": "my-profile",
            "genie": "my-profile",
            "apps": "my-profile",
        }

    def test_configures_only_ucode_configured_clients(self, monkeypatch, capsys):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []
        monkeypatch.setattr(
            mcp,
            "load_state",
            lambda: {"workspace": WS, "available_tools": ["claude", "codex"]},
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ALL_MCP_CLIENTS)
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, f"{mcp.MCP_ADD_PREFIX}managed:sql")
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        output = capsys.readouterr().out
        assert "Configuring for: Claude Code, Codex" in output
        assert [call[0] for call in configured] == ["claude", "codex"]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "databricks-sql",
                "url": f"{WS}/api/2.0/mcp/sql",
                "auth": "env:OAUTH_TOKEN",
                "clients": ["claude", "codex"],
            }
        ]

    def test_registers_databricks_sql_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []
        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, f"{mcp.MCP_ADD_PREFIX}managed:sql")
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert configured == [
            (
                "claude",
                "databricks-sql",
                f"{WS}/api/2.0/mcp/sql",
                {
                    "type": "http",
                    "url": f"{WS}/api/2.0/mcp/sql",
                    "headers": {"Authorization": "Bearer ${OAUTH_TOKEN}"},
                },
            )
        ]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "databricks-sql",
                "url": f"{WS}/api/2.0/mcp/sql",
                "auth": "env:OAUTH_TOKEN",
                "clients": ["claude"],
            }
        ]

    def test_removes_saved_server(self, monkeypatch):
        state = {
            "workspace": WS,
            "available_tools": ["claude"],
            "mcp_servers": [
                {
                    "name": "github-mcp",
                    "url": f"{WS}/api/2.0/mcp/external/github-mcp",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["claude"],
                }
            ],
        }
        saved_states: list[dict] = []
        removed: list[tuple[str, str]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: state)
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch)
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or ["user"],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert removed == [("claude", "github-mcp")]
        assert saved_states[-1]["mcp_servers"] == []


def _stub_location_base(monkeypatch, state):
    """Shared scaffolding for --location tests: no installed agents missing,
    auth no-ops, no cross-workspace residue."""
    monkeypatch.setattr(mcp, "load_state", lambda: state)
    monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
    monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
    monkeypatch.setattr(mcp, "purge_cross_workspace_mcp_residue", lambda state, workspace: None)
    monkeypatch.setattr(mcp, "get_databricks_token", lambda workspace, profile=None: "token")


class TestConfigureMcpFromLocation:
    def test_rejects_malformed_location(self, monkeypatch):
        _stub_location_base(monkeypatch, {**CLAUDE_STATE})
        monkeypatch.setattr(mcp, "list_mcp_services", lambda *a, **kw: ([], None))

        for bad in ("system", "system.ai.extra", ".ai", "system.", ""):
            try:
                mcp.configure_mcp_command(location=bad)
            except RuntimeError as exc:
                assert "--location" in str(exc)
            else:
                raise AssertionError(f"expected RuntimeError for `{bad}`")

    def test_invalid_location_raises_with_clear_message(self, monkeypatch):
        _stub_location_base(monkeypatch, {**CLAUDE_STATE})
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: ([], "HTTP 404 Not Found: NOT_FOUND"),
        )
        try:
            mcp.configure_mcp_command(location="nope.nope")
        except RuntimeError as exc:
            assert "Invalid location" in str(exc)
            assert "nope.nope" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")

    def test_other_listing_failure_surfaces_reason(self, monkeypatch):
        _stub_location_base(monkeypatch, {**CLAUDE_STATE})
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: ([], "HTTP 500 Server Error"),
        )
        try:
            mcp.configure_mcp_command(location="system.ai")
        except RuntimeError as exc:
            assert "HTTP 500" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")

    def test_registers_every_discovered_service(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []
        seen: dict[str, str] = {}
        picker_called: list[bool] = []
        _stub_location_base(monkeypatch, {**CLAUDE_STATE})

        def fake_list(workspace, token, parent):
            seen["parent"] = parent
            return ["system.ai.github", "system.ai.slack"], None

        monkeypatch.setattr(mcp, "list_mcp_services", fake_list)
        monkeypatch.setattr(
            mcp,
            "prompt_for_mcp_server_choices",
            lambda *a, **kw: picker_called.append(True) or [],
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command(location="system.ai") == 0

        assert seen == {"parent": "system.ai"}
        assert picker_called == []
        assert [c[1] for c in configured] == ["system-ai-github", "system-ai-slack"]
        assert configured[0][2] == f"{WS}/ai-gateway/mcp-services/system.ai.github"
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "system-ai-github",
                "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
                "auth": "env:OAUTH_TOKEN",
                "clients": ["claude"],
            },
            {
                "name": "system-ai-slack",
                "url": f"{WS}/ai-gateway/mcp-services/system.ai.slack",
                "auth": "env:OAUTH_TOKEN",
                "clients": ["claude"],
            },
        ]

    def test_replaces_servers_outside_location(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []
        removed: list[tuple[str, str]] = []
        outside_entry = {
            "name": "databricks-sql",
            "url": f"{WS}/api/2.0/mcp/sql",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }
        _stub_location_base(
            monkeypatch,
            {**CLAUDE_STATE, "mcp_servers": [outside_entry]},
        )
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (["system.ai.github"], None),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command(location="system.ai") == 0

        assert removed == [("claude", "databricks-sql")]
        assert [c[1] for c in configured] == ["system-ai-github"]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "system-ai-github",
                "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
                "auth": "env:OAUTH_TOKEN",
                "clients": ["claude"],
            },
        ]

    def test_preserves_skills_connection(self, monkeypatch):
        """A skills connection is owned by `configure skills`, so `configure mcp
        --location` must leave it registered rather than treating it as a removal."""
        saved_states: list[dict] = []
        removed: list[tuple[str, str]] = []
        skills_entry = {
            "name": mcp.SKILLS_MCP_SERVER_NAME,
            "kind": mcp.SKILLS_MCP_KIND,
            "skill_locations": ["main.default"],
            "url": f"{WS}/ai-gateway/skills/?schema=main.default",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }
        _stub_location_base(
            monkeypatch,
            {**CLAUDE_STATE, "mcp_servers": [skills_entry]},
        )
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (["system.ai.github"], None),
        )
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a: [])
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command(location="system.ai") == 0

        assert removed == []
        assert _find_skills(saved_states[-1]["mcp_servers"]) == [skills_entry]

    def test_existing_entry_gets_reconfigured_for_newly_added_clients(self, monkeypatch):
        """An entry registered before a second agent was configured should
        get registered for that agent on the next --location run."""
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []
        existing = {
            "name": "system-ai-github",
            "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }
        _stub_location_base(
            monkeypatch,
            {
                "workspace": WS,
                "available_tools": ["claude", "codex"],
                "mcp_servers": [existing],
            },
        )
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude", "codex"])
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (["system.ai.github"], None),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command(location="system.ai") == 0

        assert [c[0] for c in configured] == ["claude", "codex"]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "system-ai-github",
                "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
                "auth": "env:OAUTH_TOKEN",
                "clients": ["claude", "codex"],
            }
        ]


class TestConfigureMcpServicesSubset:
    """`--location <schema> --services a,b,...` configures exactly the named subset."""

    def test_configures_only_the_requested_subset(self, monkeypatch):
        configured: list[tuple[str, str, str, dict]] = []
        saved_states: list[dict] = []
        _stub_location_base(monkeypatch, {**CLAUDE_STATE})
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (
                ["system.ai.github", "system.ai.slack", "system.ai.gmail"],
                None,
            ),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert (
            mcp.configure_mcp_command(
                location="system.ai", services={"system.ai.github", "system.ai.gmail"}
            )
            == 0
        )

        # slack is dropped; only the two requested services are configured.
        assert sorted(c[1] for c in configured) == ["system-ai-github", "system-ai-gmail"]
        assert sorted(s["name"] for s in saved_states[-1]["mcp_servers"]) == [
            "system-ai-github",
            "system-ai-gmail",
        ]

    def test_matches_bare_short_names(self, monkeypatch):
        configured: list[tuple[str, str, str, dict]] = []
        _stub_location_base(monkeypatch, {**CLAUDE_STATE})
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (["system.ai.github", "system.ai.slack"], None),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: None)

        assert mcp.configure_mcp_command(location="system.ai", services={"github"}) == 0

        assert [c[1] for c in configured] == ["system-ai-github"]

    def test_unknown_requested_service_warns_and_skips(self, monkeypatch):
        configured: list[tuple[str, str, str, dict]] = []
        warnings: list[str] = []
        _stub_location_base(monkeypatch, {**CLAUDE_STATE})
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (["system.ai.github"], None),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: None)
        monkeypatch.setattr(mcp, "print_warning", lambda msg: warnings.append(msg))

        assert (
            mcp.configure_mcp_command(
                location="system.ai", services={"system.ai.github", "system.ai.ghost"}
            )
            == 0
        )

        # The known service is still configured; the unknown one is reported, not fatal.
        assert [c[1] for c in configured] == ["system-ai-github"]
        assert any("system.ai.ghost" in w for w in warnings)

    def test_empty_services_removes_everything(self, monkeypatch):
        existing = {
            "name": "system-ai-github",
            "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }
        configured: list[tuple[str, str, str, dict]] = []
        removed: list[tuple[str, str]] = []
        saved_states: list[dict] = []
        _stub_location_base(monkeypatch, {**CLAUDE_STATE, "mcp_servers": [existing]})
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (["system.ai.github"], None),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command(location="system.ai", services=set()) == 0

        assert configured == []
        assert removed == [("claude", "system-ai-github")]
        assert saved_states[-1]["mcp_servers"] == []

    def test_adds_and_removes_to_match_new_selection(self, monkeypatch):
        # The live case teammates want mid-session: started with github+slack,
        # then the user deselects slack and selects gmail.
        github = {
            "name": "system-ai-github",
            "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }
        slack = {
            "name": "system-ai-slack",
            "url": f"{WS}/ai-gateway/mcp-services/system.ai.slack",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }
        configured: list[tuple[str, str, str, dict]] = []
        removed: list[tuple[str, str]] = []
        saved_states: list[dict] = []
        _stub_location_base(monkeypatch, {**CLAUDE_STATE, "mcp_servers": [github, slack]})
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (
                ["system.ai.github", "system.ai.slack", "system.ai.gmail"],
                None,
            ),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert (
            mcp.configure_mcp_command(
                location="system.ai", services={"system.ai.github", "system.ai.gmail"}
            )
            == 0
        )

        # slack removed, gmail added, github untouched (entry unchanged).
        assert removed == [("claude", "system-ai-slack")]
        assert [c[1] for c in configured] == ["system-ai-gmail"]
        assert sorted(s["name"] for s in saved_states[-1]["mcp_servers"]) == [
            "system-ai-github",
            "system-ai-gmail",
        ]

    def test_full_names_without_location_derive_the_schema(self, monkeypatch):
        configured: list[tuple[str, str, str, dict]] = []
        seen: dict[str, str] = {}
        _stub_location_base(monkeypatch, {**CLAUDE_STATE})

        def fake_list(workspace, token, parent):
            seen["parent"] = parent
            return ["system.ai.github", "system.ai.slack"], None

        monkeypatch.setattr(mcp, "list_mcp_services", fake_list)
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: None)

        # No --location: the `<catalog>.<schema>` is derived from the full names.
        assert mcp.configure_mcp_command(services={"system.ai.github", "system.ai.slack"}) == 0

        assert seen == {"parent": "system.ai"}
        assert sorted(c[1] for c in configured) == ["system-ai-github", "system-ai-slack"]

    def test_short_name_without_location_raises(self):
        try:
            mcp.configure_mcp_command(services={"github"})
        except RuntimeError as exc:
            assert "--location" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for a bare short name without --location")

    def test_full_names_spanning_multiple_schemas_without_location_raises(self):
        try:
            mcp.configure_mcp_command(services={"system.ai.github", "other.cat.thing"})
        except RuntimeError as exc:
            assert "--location" in str(exc)
        else:
            raise AssertionError(
                "expected RuntimeError for multi-schema services without --location"
            )


def _find_skills(servers):
    return [s for s in servers if s.get("kind") == mcp.SKILLS_MCP_KIND]


class TestResolveSkillsMcpServers:
    def test_builds_single_canonical_entry(self):
        servers = mcp._resolve_skills_mcp_servers(WS, ["claude"], ["main.default"], [])
        assert _find_skills(servers) == servers
        entry = servers[0]
        assert entry["name"] == mcp.SKILLS_MCP_SERVER_NAME
        assert entry["kind"] == mcp.SKILLS_MCP_KIND
        assert entry["skill_locations"] == ["main.default"]
        assert entry["url"] == f"{WS}/ai-gateway/skills/?schema=main.default"
        assert entry["auth"] == "env:OAUTH_TOKEN"
        assert entry["clients"] == ["claude"]

    def test_keeps_other_entries_and_rebuilds_to_one_skills_entry(self):
        service_entry = {
            "name": "system-ai-github",
            "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }
        stale_skills = {
            "name": mcp.SKILLS_MCP_SERVER_NAME,
            "kind": mcp.SKILLS_MCP_KIND,
            "skill_locations": ["old.one"],
            "url": f"{WS}/ai-gateway/skills/?schema=old.one",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["codex"],
        }
        servers = mcp._resolve_skills_mcp_servers(
            WS, ["claude"], ["a.b"], [service_entry, stale_skills]
        )
        assert service_entry in servers
        skills = _find_skills(servers)
        assert len(skills) == 1
        # Prior skills clients merge with the newly-configured ones (order-stable).
        assert skills[0]["clients"] == ["codex", "claude"]

    def test_drops_entry_matching_skills_server_name_even_without_kind(self):
        old_named = {
            "name": mcp.SKILLS_MCP_SERVER_NAME,
            "url": f"{WS}/ai-gateway/skills/",
            "clients": ["claude"],
        }
        servers = mcp._resolve_skills_mcp_servers(WS, ["claude"], ["a.b"], [old_named])
        assert len(servers) == 1
        assert servers[0]["kind"] == mcp.SKILLS_MCP_KIND

    def test_url_derives_from_locations_not_stale_url(self):
        stale = {
            "name": mcp.SKILLS_MCP_SERVER_NAME,
            "kind": mcp.SKILLS_MCP_KIND,
            "skill_locations": ["old.one"],
            "url": f"{WS}/ai-gateway/skills/?schema=stale.value",
            "clients": ["claude"],
        }
        servers = mcp._resolve_skills_mcp_servers(WS, ["claude"], ["new.two"], [stale])
        assert servers[0]["url"] == f"{WS}/ai-gateway/skills/?schema=new.two"

    def test_empty_locations_yields_bare_route(self):
        servers = mcp._resolve_skills_mcp_servers(WS, ["claude"], [], [])
        assert servers[0]["skill_locations"] == []
        assert servers[0]["url"] == f"{WS}/ai-gateway/skills/"


def _skills_state(mcp_servers=None):
    state = {"workspace": WS, "available_tools": ["claude"]}
    if mcp_servers is not None:
        state["mcp_servers"] = mcp_servers
    return state


class TestConfigureSkillsMcpCommand:
    def test_set_on_empty_registers_connection(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []
        _stub_location_base(monkeypatch, _skills_state())
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, entry: configured.append((client, name, url, entry)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_skills_mcp_command(["a.b"]) == 0

        skills = _find_skills(saved_states[-1]["mcp_servers"])
        assert len(skills) == 1
        assert skills[0]["skill_locations"] == ["a.b"]
        assert skills[0]["url"] == f"{WS}/ai-gateway/skills/?schema=a.b"
        # alwaysLoad rides the per-entry apply (Claude-only).
        assert configured[0][3]["alwaysLoad"] is True

    def test_location_replaces_prior_set(self, monkeypatch):
        saved_states: list[dict] = []
        prior = mcp._resolve_skills_mcp_servers(WS, ["claude"], ["A.a", "B.b"], [])
        _stub_location_base(monkeypatch, _skills_state(prior))
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **kw: [])
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_skills_mcp_command(["X.x"]) == 0

        assert _find_skills(saved_states[-1]["mcp_servers"])[0]["skill_locations"] == ["X.x"]

    def test_multiple_locations_set_in_order(self, monkeypatch):
        saved_states: list[dict] = []
        prior = mcp._resolve_skills_mcp_servers(WS, ["claude"], ["A.a"], [])
        _stub_location_base(monkeypatch, _skills_state(prior))
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **kw: [])
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_skills_mcp_command(["X.x", "Y.y"]) == 0

        assert _find_skills(saved_states[-1]["mcp_servers"])[0]["skill_locations"] == ["X.x", "Y.y"]

    def test_preserves_mcp_service_entries_across_set(self, monkeypatch):
        saved_states: list[dict] = []
        service_entry = {
            "name": "system-ai-github",
            "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }
        prior = mcp._resolve_skills_mcp_servers(WS, ["claude"], ["A.a"], [service_entry])
        _stub_location_base(monkeypatch, _skills_state(prior))
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **kw: [])
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_skills_mcp_command(["B.b"]) == 0

        names = [s["name"] for s in saved_states[-1]["mcp_servers"]]
        assert "system-ai-github" in names
        assert names.count(mcp.SKILLS_MCP_SERVER_NAME) == 1


class TestSkillMcpLocations:
    def test_reads_locations_off_skills_entry(self):
        state = _skills_state(mcp._resolve_skills_mcp_servers(WS, ["claude"], ["A.a", "B.b"], []))
        assert mcp._skill_mcp_locations(state) == ["A.a", "B.b"]

    def test_empty_when_no_skills_entry(self):
        assert mcp._skill_mcp_locations(_skills_state([])) == []
        assert mcp._skill_mcp_locations(_skills_state()) == []


class TestConfigureSkillsDownloadCommand:
    def _stub_download(self, monkeypatch, state, *, leaves, bundle=(None, None)):
        saved_states: list[dict] = []
        written: list[tuple[str, str]] = []
        _stub_location_base(monkeypatch, state)
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **kw: [])
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))
        monkeypatch.setattr(mcp, "skill_dir_roots", lambda path: [f"{path}/.claude/skills"])
        monkeypatch.setattr(mcp, "list_schema_skills", lambda *a, **kw: leaves)
        monkeypatch.setattr(mcp, "fetch_skill_bundle", lambda *a, **kw: bundle)
        monkeypatch.setattr(
            mcp,
            "write_skill",
            lambda roots, leaf, files, *, location: written.append((location, leaf)) or "written",
        )
        return saved_states, written

    def test_downloads_leaves_and_registers_schemaless_connection(self, monkeypatch):
        saved_states, written = self._stub_download(
            monkeypatch,
            _skills_state(),
            leaves=(["pii-handling", "triage"], None),
            bundle=({"SKILL.md": b"body"}, None),
        )

        assert mcp.configure_skills_download_command(["a.b"], path="/tmp/skills") == 0

        assert written == [("a.b", "pii-handling"), ("a.b", "triage")]
        skills = _find_skills(saved_states[-1]["mcp_servers"])
        assert len(skills) == 1
        assert skills[0]["skill_locations"] == []
        assert skills[0]["url"] == f"{WS}/ai-gateway/skills/"

    def test_preserves_prior_mcp_location_set(self, monkeypatch):
        prior = mcp._resolve_skills_mcp_servers(WS, ["claude"], ["X.x", "Y.y"], [])
        state = _skills_state(prior)
        self._stub_download(
            monkeypatch,
            state,
            leaves=(["pii-handling"], None),
            bundle=({"SKILL.md": b"body"}, None),
        )

        assert mcp.configure_skills_download_command(["a.b"], path="/tmp/skills") == 0

        assert _find_skills(state["mcp_servers"])[0]["skill_locations"] == ["X.x", "Y.y"]

    def test_bundle_failure_warns_and_still_exits_0(self, monkeypatch):
        saved_states, written = self._stub_download(
            monkeypatch,
            _skills_state(),
            leaves=(["pii-handling"], None),
            bundle=(None, "HTTP 500 Server Error"),
        )

        assert mcp.configure_skills_download_command(["a.b"], path="/tmp/skills") == 0

        assert written == []
        assert _find_skills(saved_states[-1]["mcp_servers"])[0]["skill_locations"] == []

    def test_list_failure_skips_location(self, monkeypatch):
        saved_states, written = self._stub_download(
            monkeypatch,
            _skills_state(),
            leaves=([], "HTTP 404 Not Found"),
        )

        assert mcp.configure_skills_download_command(["a.b"], path="/tmp/skills") == 0

        assert written == []


class TestRevertMcpConfigs:
    def test_removes_cli_registered_servers_and_restores_copilot_config(self, monkeypatch):
        removed: list[tuple[str, str]] = []
        restored: list[tuple[object, object, bool]] = []

        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or ["user"],
        )
        monkeypatch.setattr(
            mcp,
            "restore_file",
            lambda config_path, backup_path, managed: (
                restored.append((config_path, backup_path, managed)) or True
            ),
        )

        result = mcp.revert_mcp_configs(
            {
                "mcp_servers": [
                    {
                        "name": "github-mcp",
                        "clients": ["claude", "codex", "gemini", "opencode", "copilot"],
                    }
                ]
            }
        )

        assert removed == [
            ("claude", "github-mcp"),
            ("codex", "github-mcp"),
            ("gemini", "github-mcp"),
            ("opencode", "github-mcp"),
            ("copilot", "github-mcp"),
        ]
        assert restored == [
            (mcp.copilot.COPILOT_MCP_CONFIG_PATH, mcp.copilot.COPILOT_MCP_BACKUP_PATH, True)
        ]
        assert result == {
            "claude": True,
            "codex": True,
            "gemini": True,
            "opencode": True,
            "copilot": True,
        }

    def test_removes_skills_registry_across_its_clients(self, monkeypatch):
        removed: list[tuple[str, str]] = []
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or ["user"],
        )
        monkeypatch.setattr(mcp, "restore_file", lambda *a, **kw: False)

        skills_entry = mcp._resolve_skills_mcp_servers(WS, ["claude", "codex"], ["a.b"], [])[0]
        mcp.revert_mcp_configs({"mcp_servers": [skills_entry]})

        assert removed == [
            ("claude", mcp.SKILLS_MCP_SERVER_NAME),
            ("codex", mcp.SKILLS_MCP_SERVER_NAME),
        ]


class TestPurgeCrossWorkspaceSkillsEntry:
    def test_drops_foreign_workspace_skills_entry(self, monkeypatch):
        removed: list[tuple[str, str]] = []
        saved_states: list[dict] = []
        foreign = "https://other.databricks.com"
        skills_entry = mcp._resolve_skills_mcp_servers(foreign, ["claude"], ["a.b"], [])[0]
        # The skills URL carries a `?schema=` query; its host must still parse.
        assert mcp._mcp_entry_url_host(skills_entry) == "other.databricks.com"
        state = {"mcp_servers": [skills_entry]}

        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(mcp, "load_full_state", lambda: {})
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or ["user"],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        mcp.purge_cross_workspace_mcp_residue(state, WS)

        assert removed == [("claude", mcp.SKILLS_MCP_SERVER_NAME)]
        assert state["mcp_servers"] == []
