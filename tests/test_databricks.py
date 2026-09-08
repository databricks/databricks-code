"""Tests for databricks.py — pure helpers and URL builders that don't hit the network."""

from __future__ import annotations

import json
import os
import subprocess

import pytest

import ucode.databricks as db_mod
from ucode.databricks import (
    AI_GATEWAY_V2_DOCS_URL,
    _format_subprocess_result,
    _parse_databricks_cli_version,
    _run_databricks_cli_installer,
    _scrub_databrickscfg,
    _scrub_json,
    build_auth_shell_command,
    build_databricks_cli_env,
    build_opencode_base_urls,
    build_shared_base_urls,
    build_tool_base_url,
    ensure_databricks_cli_version,
    get_databricks_token,
    list_databricks_apps,
    list_databricks_connections,
    list_genie_spaces,
    workspace_hostname,
)

WS = "https://example.databricks.com"


class TestWorkspaceHostname:
    def test_extracts_hostname(self):
        assert workspace_hostname(WS) == "example.databricks.com"

    def test_handles_path(self):
        assert (
            workspace_hostname("https://foo.azuredatabricks.net/some/path")
            == "foo.azuredatabricks.net"
        )

    def test_invalid_url_raises(self):
        with pytest.raises((RuntimeError, ValueError)):
            workspace_hostname("")


class TestBuildDatabricksCliEnv:
    def test_sets_databricks_host(self):
        env = build_databricks_cli_env(WS)
        assert env["DATABRICKS_HOST"] == WS

    def test_strips_ambient_profile_without_explicit_profile(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "other-workspace")

        env = build_databricks_cli_env(WS)

        assert env["DATABRICKS_HOST"] == WS
        assert "DATABRICKS_CONFIG_PROFILE" not in env

    def test_preserves_ambient_profile_with_explicit_profile(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "other-workspace")

        env = build_databricks_cli_env(WS, profile="stablebox")

        assert env["DATABRICKS_HOST"] == WS
        assert env["DATABRICKS_CONFIG_PROFILE"] == "other-workspace"


class TestBuildToolBaseUrl:
    def test_codex(self):
        url = build_tool_base_url("codex", WS)
        assert url == f"{WS}/ai-gateway/codex/v1"

    def test_claude(self):
        url = build_tool_base_url("claude", WS)
        assert url == f"{WS}/ai-gateway/anthropic"

    def test_gemini(self):
        url = build_tool_base_url("gemini", WS)
        assert url == f"{WS}/ai-gateway/gemini"

    def test_opencode_raises(self):
        with pytest.raises(RuntimeError, match="multiple base URLs"):
            build_tool_base_url("opencode", WS)

    def test_unsupported_tool_raises(self):
        with pytest.raises(RuntimeError, match="Unsupported"):
            build_tool_base_url("unknown", WS)


class TestBuildOpencodeBaseUrls:
    def test_returns_anthropic_and_gemini(self):
        urls = build_opencode_base_urls(WS)
        assert urls["anthropic"] == f"{WS}/ai-gateway/anthropic/v1"
        assert urls["gemini"] == f"{WS}/ai-gateway/gemini/v1beta"


class TestBuildSharedBaseUrls:
    def test_contains_all_tools(self):
        urls = build_shared_base_urls(WS)
        assert "codex" in urls
        assert "claude" in urls
        assert "gemini" in urls
        assert "opencode" in urls

    def test_opencode_is_dict(self):
        urls = build_shared_base_urls(WS)
        assert isinstance(urls["opencode"], dict)

    def test_codex_url_format(self):
        urls = build_shared_base_urls(WS)
        assert urls["codex"] == f"{WS}/ai-gateway/codex/v1"


class TestDiscoverClaudeModels:
    def test_selects_opus_4_8_when_advertised(self, monkeypatch):
        payload = {
            "data": [
                {"id": "databricks-claude-opus-4-7"},
                {"id": "databricks-claude-opus-4-8"},
                {"id": "databricks-claude-sonnet-4-6"},
            ]
        }
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: (payload, None))

        models, reason = db_mod.discover_claude_models(WS, "token")

        assert reason is None
        assert models["opus"] == "databricks-claude-opus-4-8"


def _model_service(model_id: str) -> dict:
    """A model-services entry whose `name` strips to `model_id`."""
    return {"name": f"model-services/{model_id}"}


class TestDiscoverModelServices:
    def test_buckets_families_by_name(self, monkeypatch):
        payload = {
            "model_services": [
                _model_service("system.ai.claude-opus-4-7"),
                _model_service("system.ai.claude-opus-4-8"),
                _model_service("system.ai.claude-sonnet-4-6"),
                _model_service("system.ai.gpt-5"),
                _model_service("system.ai.gemini-2-5-flash"),
                _model_service("system.ai.gemini-3-5-flash"),
                _model_service("system.ai.llama-4-maverick"),
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        claude, codex, gemini, reason = db_mod.discover_model_services(WS, "token")

        assert reason is None
        # Newest opus wins; sonnet bucketed; haiku absent.
        assert claude == {
            "opus": "system.ai.claude-opus-4-8",
            "sonnet": "system.ai.claude-sonnet-4-6",
        }
        assert codex == ["system.ai.gpt-5"]
        # Gemini ordered newest-first via the shared sort key.
        assert gemini[0] == "system.ai.gemini-3-5-flash"
        # llama is not bucketed into any of the three families.
        assert "system.ai.llama-4-maverick" not in codex + gemini

    def test_paginates_via_next_page_token(self, monkeypatch):
        pages = {
            None: {
                "model_services": [_model_service("system.ai.gpt-5")],
                "next_page_token": "tok2",
            },
            "tok2": {
                "model_services": [_model_service("system.ai.claude-opus-4-8")],
            },
        }

        def fake_get(url, token, timeout=10):
            token_param = None
            if "page_token=" in url:
                token_param = url.split("page_token=")[1].split("&")[0]
            return pages[token_param], None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        claude, codex, _, reason = db_mod.discover_model_services(WS, "token")

        assert reason is None
        assert codex == ["system.ai.gpt-5"]
        assert claude == {"opus": "system.ai.claude-opus-4-8"}

    def test_http_failure_returns_reason(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (None, "HTTP 500 Server Error")
        )

        claude, codex, gemini, reason = db_mod.discover_model_services(WS, "token")

        assert (claude, codex, gemini) == ({}, [], [])
        assert reason == "HTTP 500 Server Error"

    def test_no_matching_families_reports_sample(self, monkeypatch):
        payload = {"model_services": [_model_service("system.ai.llama-4-maverick")]}
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        claude, codex, gemini, reason = db_mod.discover_model_services(WS, "token")

        assert (claude, codex, gemini) == ({}, [], [])
        assert reason is not None and "llama-4-maverick" in reason

    def test_ignores_non_system_ai_schemas(self, monkeypatch):
        # The metastore listing returns services from every schema; only
        # system.ai.* foundation models should be picked up.
        payload = {
            "model_services": [
                _model_service("system.ai.gpt-5"),
                _model_service("main.svenwb.gpt-5-5"),
                _model_service("temp.erni.claude-opus-4-8"),
                _model_service("dnasi_agent_cuj.default.dnasi-gpt55-test"),
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        claude, codex, gemini, reason = db_mod.discover_model_services(WS, "token")

        assert reason is None
        assert codex == ["system.ai.gpt-5"]
        assert claude == {}  # temp.erni.claude-* must not be bucketed
        assert gemini == []

    def test_requests_bounded_page_size(self, monkeypatch):
        # The endpoint 499s without a bounded page_size, so every request must
        # carry one.
        urls: list[str] = []

        def fake_get(url, token, timeout=10):
            urls.append(url)
            return {"model_services": [_model_service("system.ai.gpt-5")]}, None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        ids, reason = db_mod.list_model_services(WS, "token")

        assert ids == ["system.ai.gpt-5"]
        assert reason is None
        assert all("page_size=" in u for u in urls)

    def test_retries_page_before_giving_up(self, monkeypatch):
        payload = {"model_services": [_model_service("system.ai.gpt-5")]}
        calls = {"n": 0}

        def flaky_get(url, token, timeout=10):
            calls["n"] += 1
            if calls["n"] < 3:
                return None, "HTTP 499 Unknown"
            return payload, None

        monkeypatch.setattr(db_mod, "_http_get_json", flaky_get)

        ids, reason = db_mod.list_model_services(WS, "token")

        assert reason is None
        assert ids == ["system.ai.gpt-5"]
        assert calls["n"] == 3  # two failures, third succeeds


class TestListMcpServices:
    def test_accepts_entries_without_connection_status(self, monkeypatch):
        payload = {
            "mcp_services": [
                {
                    "name": "mcp-services/system.ai.github",
                    "config": {"usage_tracking": {"enabled": True}, "tracing": {"enabled": True}},
                },
                {
                    "name": "mcp-services/system.ai.atlassian",
                    "config": {},
                },
                {
                    "name": "mcp-services/system.ai.slack",
                },
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, None)
        )

        names, reason = db_mod.list_mcp_services(WS, "token")

        assert reason is None
        assert names == ["system.ai.atlassian", "system.ai.github", "system.ai.slack"]

    def test_accepts_legacy_active_status(self, monkeypatch):
        payload = {
            "mcp_services": [
                {
                    "name": "mcp-services/system.ai.github",
                    "config": {"connection": {"status": "ACTIVE"}},
                },
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, None)
        )

        names, reason = db_mod.list_mcp_services(WS, "token")

        assert reason is None
        assert names == ["system.ai.github"]

    def test_rejects_explicit_non_active_status(self, monkeypatch):
        # If the field is present and non-ACTIVE, drop the entry — the
        # backing connection is broken and the proxy will fail.
        payload = {
            "mcp_services": [
                {
                    "name": "mcp-services/system.ai.github",
                    "config": {"connection": {"status": "ACTIVE"}},
                },
                {
                    "name": "mcp-services/system.ai.broken",
                    "config": {"connection": {"status": "FAILED"}},
                },
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, None)
        )

        names, _reason = db_mod.list_mcp_services(WS, "token")

        assert names == ["system.ai.github"]

    def test_ignores_non_system_ai_entries(self, monkeypatch):
        payload = {
            "mcp_services": [
                {"name": "mcp-services/system.ai.github"},
                {"name": "mcp-services/main.svenwb.github_mcp"},
                {"name": "mcp-services/temp.erni.github_mcp"},
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, None)
        )

        names, _reason = db_mod.list_mcp_services(WS, "token")

        assert names == ["system.ai.github"]

    def test_http_failure_propagates_reason(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, timeout=30: (None, "HTTP 500 Server Error"),
        )

        names, reason = db_mod.list_mcp_services(WS, "token")

        assert names == []
        assert reason == "HTTP 500 Server Error"

    def test_empty_payload_reports_no_results(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: ({"mcp_services": []}, None)
        )

        names, reason = db_mod.list_mcp_services(WS, "token")

        assert names == []
        assert reason and "no `system.ai.*`" in reason


def _foundation_models_payload(names):
    return {
        "endpoints": [
            {
                "name": name,
                "config": {
                    "served_entities": [
                        {
                            "foundation_model": {
                                "ai_gateway_v2_supported": True,
                                "api_types": ["gemini/v1/generateContent"],
                            }
                        }
                    ]
                },
            }
            for name in names
        ]
    }


class TestModelVersionSortKey:
    def test_orders_newest_version_first(self):
        names = [
            "databricks-gemini-2-5-flash",
            "databricks-gemini-2-5-pro",
            "databricks-gemini-3-1-flash-lite",
            "databricks-gemini-3-1-pro",
            "databricks-gemini-3-5-flash",
            "databricks-gemini-3-flash",
            "databricks-gemini-3-pro",
        ]
        ordered = sorted(names, key=db_mod.model_version_sort_key)
        assert ordered[0] == "databricks-gemini-3-5-flash"

    def test_treats_bare_major_as_dot_zero(self):
        # 3-flash is 3.0, so 3-5-flash (3.5) must sort ahead of it.
        names = ["databricks-gemini-3-flash", "databricks-gemini-3-5-flash"]
        ordered = sorted(names, key=db_mod.model_version_sort_key)
        assert ordered == [
            "databricks-gemini-3-5-flash",
            "databricks-gemini-3-flash",
        ]

    def test_unversioned_names_sort_last_alphabetically(self):
        names = ["databricks-gemini-2-5-flash", "custom-endpoint", "another-endpoint"]
        ordered = sorted(names, key=db_mod.model_version_sort_key)
        assert ordered[0] == "databricks-gemini-2-5-flash"
        assert ordered[1:] == ["another-endpoint", "custom-endpoint"]


class TestDiscoverGeminiModels:
    def test_returns_newest_flash_first(self, monkeypatch):
        payload = _foundation_models_payload(
            [
                "databricks-gemini-2-5-flash",
                "databricks-gemini-3-5-flash",
                "databricks-gemini-3-flash",
            ]
        )
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: (payload, None))

        models, reason = db_mod.discover_gemini_models(WS, "token")

        assert reason is None
        assert models[0] == "databricks-gemini-3-5-flash"

    def test_codex_discovery_keeps_alphabetical_order(self, monkeypatch):
        # Codex passes no sort_key, so ordering must stay the plain alphabetical
        # default — guarding against the gemini change leaking across tools.
        payload = {
            "endpoints": [
                {
                    "name": name,
                    "config": {
                        "served_entities": [
                            {
                                "foundation_model": {
                                    "ai_gateway_v2_supported": True,
                                    "api_types": ["openai/v1/responses"],
                                }
                            }
                        ]
                    },
                }
                for name in ["databricks-gpt-5-2-codex", "databricks-gpt-4-1"]
            ]
        }
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: (payload, None))

        models, reason = db_mod.discover_codex_models(WS, "token")

        assert reason is None
        assert models == ["databricks-gpt-4-1", "databricks-gpt-5-2-codex"]


class TestResolvePatToken:
    def test_reads_pat_profile_token_from_cfg(self, monkeypatch, tmp_path):
        cfg = tmp_path / "databrickscfg"
        cfg.write_text(f"[lakebox]\nhost = {WS}\ntoken = dapi-from-cfg\n")
        monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(cfg))
        monkeypatch.setattr(
            db_mod,
            "list_profile_entries",
            lambda: [{"name": "lakebox", "host": WS, "auth_type": "pat"}],
        )
        assert db_mod.resolve_pat_token("lakebox") == "dapi-from-cfg"

    def test_default_section_token_does_not_leak_into_named_profiles(self, monkeypatch, tmp_path):
        cfg = tmp_path / "databrickscfg"
        cfg.write_text(
            f"[DEFAULT]\nhost = {WS}\ntoken = dapi-default\n"
            "[other]\nhost = https://other.databricks.com\n"
        )
        monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(cfg))
        monkeypatch.setattr(
            db_mod,
            "list_profile_entries",
            lambda: [
                {"name": "DEFAULT", "host": WS, "auth_type": "pat"},
                {"name": "other", "host": "https://other.databricks.com", "auth_type": "pat"},
            ],
        )
        assert db_mod.resolve_pat_token("DEFAULT") == "dapi-default"
        assert db_mod.resolve_pat_token("other") is None

    def test_returns_none_for_oauth_profile(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "list_profile_entries",
            lambda: [{"name": "oauth", "host": WS, "auth_type": "databricks-cli"}],
        )
        assert db_mod.resolve_pat_token("oauth") is None

    def test_returns_none_without_profile(self):
        assert db_mod.resolve_pat_token(None) is None


class TestApplyPatEnvironment:
    @pytest.fixture(autouse=True)
    def _isolated_bearer(self):
        # apply_pat_environment writes os.environ directly; restore it even
        # though monkeypatch can't track writes made by code under test.
        original = os.environ.pop("DATABRICKS_BEARER", None)
        yield
        if original is None:
            os.environ.pop("DATABRICKS_BEARER", None)
        else:
            os.environ["DATABRICKS_BEARER"] = original

    def test_exports_bearer_for_use_pat_state(self, monkeypatch):
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: "dapi-pat")

        db_mod.apply_pat_environment({"use_pat": True, "profile": "DEFAULT"})

        assert os.environ["DATABRICKS_BEARER"] == "dapi-pat"

    def test_noop_without_use_pat(self, monkeypatch):
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: "dapi-pat")

        db_mod.apply_pat_environment({"profile": "DEFAULT"})

        assert "DATABRICKS_BEARER" not in os.environ

    def test_existing_bearer_wins(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_BEARER", "explicit-bearer")
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: "dapi-pat")

        db_mod.apply_pat_environment({"use_pat": True, "profile": "DEFAULT"})

        assert os.environ["DATABRICKS_BEARER"] == "explicit-bearer"


class TestBuildAuthShellCommand:
    def test_contains_workspace(self):
        cmd = build_auth_shell_command(WS)
        assert WS in cmd

    def test_parses_access_token(self):
        cmd = build_auth_shell_command(WS)
        assert "jq" in cmd
        assert ".access_token" in cmd
        assert "--force-refresh" in cmd
        assert "DATABRICKS_BEARER" in cmd
        assert "DATABRICKS_CONFIG_PROFILE" in cmd

    def test_returns_token_when_auth_succeeds(self, tmp_path):
        # Fake databricks binary that always returns a valid token JSON.
        fake = tmp_path / "databricks"
        fake.write_text(
            '#!/bin/sh\necho \'{"access_token": "good-token", "token_type": "Bearer"}\'\n'
        )
        fake.chmod(0o755)
        cmd = build_auth_shell_command(WS)
        result = subprocess.run(
            ["sh", "-c", cmd],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{tmp_path}:{os.environ['PATH']}",
                "DATABRICKS_BEARER": "",
            },
        )
        assert result.stdout.strip() == "good-token"

    def test_prefers_databricks_bearer(self, tmp_path):
        fake = tmp_path / "databricks"
        fake.write_text("#!/bin/sh\nexit 1\n")
        fake.chmod(0o755)
        cmd = build_auth_shell_command(WS)
        result = subprocess.run(
            ["sh", "-c", cmd],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{tmp_path}:{os.environ['PATH']}",
                "DATABRICKS_BEARER": "bearer-token",
            },
        )
        assert result.stdout.strip() == "bearer-token"

    def test_embeds_profile_when_provided(self):
        cmd = build_auth_shell_command(WS, profile="stablebox")
        assert "--profile stablebox" in cmd
        # We do not strip DATABRICKS_CONFIG_PROFILE when we are explicit about
        # which profile to use — the --profile flag wins.
        assert "env -u DATABRICKS_CONFIG_PROFILE" not in cmd

    def test_quotes_profile_shell_metacharacters(self):
        cmd = build_auth_shell_command(WS, profile="weird name; rm -rf /")
        # shlex.quote should wrap the value so the rest of the command cannot
        # be interpreted as a shell injection.
        assert "rm -rf /" in cmd
        assert "'weird name; rm -rf /'" in cmd

    def test_use_pat_reads_profile_token_via_describe(self):
        cmd = build_auth_shell_command(WS, profile="DEFAULT", use_pat=True)
        assert "auth describe --profile DEFAULT --sensitive" in cmd
        assert ".details.configuration.token.value" in cmd
        # No OAuth attempt for PAT profiles — `auth token` cannot serve them.
        assert "auth token" not in cmd
        # The DATABRICKS_BEARER escape hatch still takes precedence.
        assert "DATABRICKS_BEARER" in cmd

    def test_use_pat_command_emits_token(self, tmp_path):
        fake = tmp_path / "databricks"
        fake.write_text(
            "#!/bin/sh\n"
            'case "$*" in\n'
            '  *"auth describe"*) echo \'{"details": {"configuration": '
            '{"token": {"value": "dapi-pat"}}}}\' ;;\n'
            "  *) exit 1 ;;\n"
            "esac\n"
        )
        fake.chmod(0o755)
        cmd = build_auth_shell_command(WS, profile="DEFAULT", use_pat=True)
        result = subprocess.run(
            ["sh", "-c", cmd],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{tmp_path}:{os.environ['PATH']}",
                "DATABRICKS_BEARER": "",
            },
        )
        assert result.stdout.strip() == "dapi-pat"

    def test_use_pat_without_profile_falls_back_to_oauth_command(self):
        cmd = build_auth_shell_command(WS, use_pat=True)
        assert "auth token" in cmd


class TestFormatSubprocessResult:
    def test_suppresses_stdout_on_success(self):
        result = subprocess.CompletedProcess(
            args=["databricks", "auth", "token"],
            returncode=0,
            stdout='{"access_token": "dapi-secret-do-not-leak", "token_type": "Bearer"}',
            stderr="",
        )
        formatted = _format_subprocess_result(result)
        assert "dapi-secret-do-not-leak" not in formatted
        assert "rc=0" in formatted

    def test_includes_stdout_on_failure(self):
        result = subprocess.CompletedProcess(
            args=["databricks", "auth", "token"],
            returncode=1,
            stdout="useful diagnostic output",
            stderr="error: no matching profile",
        )
        formatted = _format_subprocess_result(result)
        assert "rc=1" in formatted
        assert "useful diagnostic output" in formatted
        assert "no matching profile" in formatted


class TestScrubDatabrickscfg:
    def test_redacts_token_value(self):
        text = "[DEFAULT]\nhost = https://example.databricks.com\ntoken = dapi-secret\n"
        scrubbed = _scrub_databrickscfg(text)
        assert "dapi-secret" not in scrubbed
        assert "token = <redacted>" in scrubbed
        assert "host = https://example.databricks.com" in scrubbed

    def test_redacts_various_secret_keys(self):
        text = (
            "[p]\n"
            "client_secret = secret-val-1\n"
            "bearer_token = secret-val-2\n"
            "api_key = secret-val-3\n"
            "password = secret-val-4\n"
            "auth_type = oauth-u2m\n"
        )
        scrubbed = _scrub_databrickscfg(text)
        for secret in ("secret-val-1", "secret-val-2", "secret-val-3", "secret-val-4"):
            assert secret not in scrubbed
        assert "auth_type = oauth-u2m" in scrubbed

    def test_preserves_comments_and_sections(self):
        text = "# comment\n[DEFAULT]\nhost = https://x\n; another comment with token = leak\n"
        scrubbed = _scrub_databrickscfg(text)
        assert "# comment" in scrubbed
        assert "[DEFAULT]" in scrubbed
        assert "; another comment with token = leak" in scrubbed

    def test_key_matching_is_case_insensitive(self):
        text = "[p]\nTOKEN = upper\nAccess_Token = mixed\n"
        scrubbed = _scrub_databrickscfg(text)
        assert "upper" not in scrubbed
        assert "mixed" not in scrubbed


class TestScrubJson:
    def test_redacts_secret_keys(self):
        payload = {
            "access_token": "dapi-secret",
            "host": "https://example.databricks.com",
        }
        scrubbed = _scrub_json(payload)
        assert isinstance(scrubbed, dict)
        assert scrubbed["access_token"] == "<redacted>"
        assert scrubbed["host"] == "https://example.databricks.com"

    def test_recurses_into_nested_structures(self):
        payload = {
            "profiles": [
                {"name": "DEFAULT", "client_secret": "abc"},
                {"name": "other", "password": "pw"},
            ]
        }
        scrubbed = _scrub_json(payload)
        assert scrubbed == {
            "profiles": [
                {"name": "DEFAULT", "client_secret": "<redacted>"},
                {"name": "other", "password": "<redacted>"},
            ]
        }

    def test_passes_through_scalars_and_non_secret_keys(self):
        assert _scrub_json("plain") == "plain"
        assert _scrub_json(42) == 42
        assert _scrub_json({"host": "x", "auth_type": "pat"}) == {
            "host": "x",
            "auth_type": "pat",
        }


class TestGetDatabricksToken:
    def _fake_databricks(self, tmp_path, script: str) -> dict:
        fake = tmp_path / "databricks"
        fake.write_text(f"#!/bin/sh\n{script}\n")
        fake.chmod(0o755)
        return {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}

    def test_returns_token_on_success(self, tmp_path, monkeypatch):
        env = self._fake_databricks(
            tmp_path,
            'echo \'{"access_token": "good-token", "token_type": "Bearer"}\'',
        )
        monkeypatch.setattr("os.environ", env)
        token = get_databricks_token(WS)
        assert token == "good-token"

    def test_strips_ambient_profile_when_profile_not_provided(self, tmp_path, monkeypatch):
        profile_log = tmp_path / "profile"
        env = self._fake_databricks(
            tmp_path,
            f'printf "%s" "${{DATABRICKS_CONFIG_PROFILE:-}}" > {profile_log}\n'
            'echo \'{"access_token": "good-token", "token_type": "Bearer"}\'',
        )
        env["DATABRICKS_CONFIG_PROFILE"] = "other-workspace"
        monkeypatch.setattr("os.environ", env)

        token = get_databricks_token(WS)

        assert token == "good-token"
        assert profile_log.read_text() == ""

    def test_has_valid_auth_strips_ambient_profile_without_explicit_profile(
        self, tmp_path, monkeypatch
    ):
        profile_log = tmp_path / "profile"
        env = self._fake_databricks(
            tmp_path,
            f'printf "%s" "${{DATABRICKS_CONFIG_PROFILE:-}}" > {profile_log}\n'
            'echo \'{"access_token": "good-token", "token_type": "Bearer"}\'',
        )
        env["DATABRICKS_CONFIG_PROFILE"] = "other-workspace"
        monkeypatch.setattr("os.environ", env)

        assert db_mod.has_valid_databricks_auth(WS)
        assert profile_log.read_text() == ""

    def test_reauths_and_retries_when_token_empty(self, tmp_path, monkeypatch):
        call_count = tmp_path / "calls"
        call_count.write_text("0")
        env = self._fake_databricks(
            tmp_path,
            f"count=$(cat {call_count})\n"
            f"echo $((count + 1)) > {call_count}\n"
            'case "$*" in\n'
            '  *"auth login"*) exit 0 ;;\n'
            "esac\n"
            'if [ "$count" -eq 0 ]; then\n'
            '  echo \'{"access_token": "", "token_type": "Bearer"}\'\n'
            "else\n"
            '  echo \'{"access_token": "refreshed-token", "token_type": "Bearer"}\'\n'
            "fi",
        )
        monkeypatch.setattr("os.environ", env)
        token = get_databricks_token(WS)
        assert token == "refreshed-token"

    def test_raises_when_reauth_also_fails(self, tmp_path, monkeypatch):
        env = self._fake_databricks(
            tmp_path,
            'echo \'{"access_token": "", "token_type": "Bearer"}\'',
        )
        monkeypatch.setattr("os.environ", env)
        with pytest.raises(RuntimeError, match="no access token"):
            get_databricks_token(WS)

    def test_passes_profile_flag_when_provided(self, tmp_path, monkeypatch):
        # Fake CLI that records its argv to a file so we can assert the
        # --profile flag is forwarded to `databricks auth token`.
        argv_log = tmp_path / "argv"
        env = self._fake_databricks(
            tmp_path,
            f'printf "%s\\n" "$@" >> {argv_log}\n'
            'echo \'{"access_token": "good-token", "token_type": "Bearer"}\'',
        )
        monkeypatch.setattr("os.environ", env)
        token = get_databricks_token(WS, profile="stablebox")
        assert token == "good-token"
        argv = argv_log.read_text().splitlines()
        assert "--profile" in argv
        assert argv[argv.index("--profile") + 1] == "stablebox"

    def test_error_suggests_logout_when_matching_profile_exists(self, tmp_path, monkeypatch):
        env = self._fake_databricks(
            tmp_path,
            'case "$*" in\n'
            '  *"auth profiles"*) echo \'{"profiles": [{"host": "'
            + WS
            + '", "name": "example-profile", "auth_type": "databricks-cli"}]}\'; exit 0 ;;\n'
            '  *"auth login"*) exit 0 ;;\n'
            "esac\n"
            'echo \'{"access_token": "", "token_type": "Bearer"}\'',
        )
        monkeypatch.setattr("os.environ", env)

        with pytest.raises(RuntimeError) as exc_info:
            get_databricks_token(WS)

        message = str(exc_info.value)
        assert "stale or invalid" in message
        assert "databricks auth logout --profile example-profile" in message
        assert f"databricks auth login --host {WS} --profile example-profile" in message


class TestListDatabricksConnections:
    def test_lists_paginated_connections_with_workspace_env(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            if "--page-token" in args:
                payload = {"connections": [{"name": "jira-mcp", "connection_type": "HTTP"}]}
            else:
                payload = {
                    "connections": [{"name": "confluence-mcp", "connection_type": "HTTP"}],
                    "next_page_token": "next-page",
                }
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload))

        monkeypatch.setattr(db_mod, "run", fake_run)

        assert list_databricks_connections(WS) == [
            {"name": "confluence-mcp", "connection_type": "HTTP"},
            {"name": "jira-mcp", "connection_type": "HTTP"},
        ]
        assert calls[0]["args"] == [
            "databricks",
            "connections",
            "list",
            "--max-results",
            "0",
            "--output",
            "json",
        ]
        assert calls[0]["kwargs"]["env"]["DATABRICKS_HOST"] == WS
        assert calls[1]["args"][-2:] == ["--page-token", "next-page"]

    def test_passes_profile_when_provided(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"connections": []}))

        monkeypatch.setattr(db_mod, "run", fake_run)

        list_databricks_connections(WS, "my-profile")

        assert "--profile" in calls[0]
        assert calls[0][calls[0].index("--profile") + 1] == "my-profile"

    def test_raises_on_invalid_json(self, monkeypatch):
        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="not-json")

        monkeypatch.setattr(db_mod, "run", fake_run)

        with pytest.raises(RuntimeError, match="invalid JSON"):
            list_databricks_connections(WS)


class TestListGenieSpaces:
    def test_lists_paginated_spaces_with_workspace_env(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            if "--page-token" in args:
                payload = {"spaces": [{"space_id": "space-2", "title": "Second"}]}
            else:
                payload = {
                    "spaces": [{"space_id": "space-1", "title": "First"}],
                    "next_page_token": "next-page",
                }
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload))

        monkeypatch.setattr(db_mod, "run", fake_run)

        assert list_genie_spaces(WS) == [
            {"space_id": "space-1", "title": "First"},
            {"space_id": "space-2", "title": "Second"},
        ]
        assert calls[0]["args"] == [
            "databricks",
            "genie",
            "list-spaces",
            "--page-size",
            "100",
            "--output",
            "json",
        ]
        assert calls[0]["kwargs"]["env"]["DATABRICKS_HOST"] == WS
        assert calls[1]["args"][-2:] == ["--page-token", "next-page"]

    def test_passes_profile_when_provided(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"spaces": []}))

        monkeypatch.setattr(db_mod, "run", fake_run)

        list_genie_spaces(WS, "my-profile")

        assert "--profile" in calls[0]
        assert calls[0][calls[0].index("--profile") + 1] == "my-profile"

    def test_raises_on_invalid_json(self, monkeypatch):
        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="not-json")

        monkeypatch.setattr(db_mod, "run", fake_run)

        with pytest.raises(RuntimeError, match="invalid JSON"):
            list_genie_spaces(WS)


class TestListDatabricksApps:
    def test_lists_apps_with_workspace_env(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            payload = [
                {
                    "name": "my-app",
                    "url": "https://my-app.example.databricksapps.com",
                }
            ]
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload))

        monkeypatch.setattr(db_mod, "run", fake_run)

        assert list_databricks_apps(WS) == [
            {
                "name": "my-app",
                "url": "https://my-app.example.databricksapps.com",
            }
        ]
        assert calls[0]["args"] == [
            "databricks",
            "apps",
            "list",
            "--limit",
            "1000",
            "--output",
            "json",
        ]
        assert calls[0]["kwargs"]["env"]["DATABRICKS_HOST"] == WS

    def test_passes_profile_when_provided(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps([]))

        monkeypatch.setattr(db_mod, "run", fake_run)

        list_databricks_apps(WS, "my-profile")

        assert "--profile" in calls[0]
        assert calls[0][calls[0].index("--profile") + 1] == "my-profile"

    def test_accepts_object_wrapped_apps(self, monkeypatch):
        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps({"apps": [{"name": "my-app", "url": "https://example.com"}]}),
            )

        monkeypatch.setattr(db_mod, "run", fake_run)

        assert list_databricks_apps(WS) == [{"name": "my-app", "url": "https://example.com"}]

    def test_raises_on_invalid_json(self, monkeypatch):
        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="not-json")

        monkeypatch.setattr(db_mod, "run", fake_run)

        with pytest.raises(RuntimeError, match="invalid JSON"):
            list_databricks_apps(WS)


class TestEnsureAiGatewayV2:
    """Test ensure_ai_gateway_v2 without real network calls.

    The probe is `GET /api/ai-gateway/v2/endpoints`: a successful JSON
    response means v2 is wired up (even if `endpoints` is empty), while
    404/401/403/network errors all raise a RuntimeError with the docs URL.
    """

    @staticmethod
    def _mock_json_response(body: str):
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = body.encode("utf-8")
        return mock_resp

    @staticmethod
    def _http_error(code: int, msg: str, body: str = ""):
        import io
        from unittest.mock import MagicMock
        from urllib.error import HTTPError

        fp = io.BytesIO(body.encode("utf-8")) if body else None
        return HTTPError(url="", code=code, msg=msg, hdrs=MagicMock(), fp=fp)

    def test_raises_on_404(self):
        from unittest.mock import patch

        exc = self._http_error(404, "Not Found")
        with patch("ucode.databricks.urllib_request.urlopen", side_effect=exc):
            from ucode.databricks import ensure_ai_gateway_v2

            with pytest.raises(RuntimeError, match=AI_GATEWAY_V2_DOCS_URL) as excinfo:
                ensure_ai_gateway_v2(WS, "fake-token")
            assert "not enabled" in str(excinfo.value)

    def test_raises_on_401_with_auth_hint(self):
        from unittest.mock import patch

        exc = self._http_error(401, "Unauthorized")
        with patch("ucode.databricks.urllib_request.urlopen", side_effect=exc):
            from ucode.databricks import ensure_ai_gateway_v2

            with pytest.raises(RuntimeError, match="401") as excinfo:
                ensure_ai_gateway_v2(WS, "fake-token")
            message = str(excinfo.value)
            assert "rejected" in message.lower()
            assert "databricks auth login" in message

    def test_raises_on_400_invalid_token_with_auth_hint(self):
        """400 + body `Invalid Token` is the misleading-error case from issue #84."""
        from unittest.mock import patch

        exc = self._http_error(400, "Bad Request", body="Invalid Token")
        with patch("ucode.databricks.urllib_request.urlopen", side_effect=exc):
            from ucode.databricks import ensure_ai_gateway_v2

            with pytest.raises(RuntimeError) as excinfo:
                ensure_ai_gateway_v2(WS, "fake-token")
            message = str(excinfo.value)
            # The bug we are fixing: must NOT collapse to the generic
            # "v2 not available" message — must call out the auth failure
            # and point at re-login.
            assert "Invalid Token" in message
            assert "rejected" in message.lower()
            assert "databricks auth login" in message

    def test_400_without_invalid_token_falls_through_to_generic(self):
        """A 400 that is *not* an auth failure should still surface the body."""
        from unittest.mock import patch

        exc = self._http_error(400, "Bad Request", body="some other detail")
        with patch("ucode.databricks.urllib_request.urlopen", side_effect=exc):
            from ucode.databricks import ensure_ai_gateway_v2

            with pytest.raises(RuntimeError, match=AI_GATEWAY_V2_DOCS_URL) as excinfo:
                ensure_ai_gateway_v2(WS, "fake-token")
            assert "some other detail" in str(excinfo.value)

    def test_raises_on_url_error(self):
        from unittest.mock import patch
        from urllib.error import URLError

        with patch(
            "ucode.databricks.urllib_request.urlopen",
            side_effect=URLError("connection refused"),
        ):
            from ucode.databricks import ensure_ai_gateway_v2

            with pytest.raises(RuntimeError, match=AI_GATEWAY_V2_DOCS_URL):
                ensure_ai_gateway_v2(WS, "fake-token")

    def test_succeeds_with_endpoints_list(self):
        from unittest.mock import patch

        with patch(
            "ucode.databricks.urllib_request.urlopen",
            return_value=self._mock_json_response('{"endpoints": [{"name": "foo"}]}'),
        ):
            from ucode.databricks import ensure_ai_gateway_v2

            ensure_ai_gateway_v2(WS, "fake-token")  # should not raise

    def test_succeeds_with_empty_endpoints_list(self):
        from unittest.mock import patch

        # A 200 with no endpoints still means v2 is wired up on this workspace —
        # downstream discovery will surface "no models" with a clearer reason.
        with patch(
            "ucode.databricks.urllib_request.urlopen",
            return_value=self._mock_json_response('{"endpoints": []}'),
        ):
            from ucode.databricks import ensure_ai_gateway_v2

            ensure_ai_gateway_v2(WS, "fake-token")  # should not raise


class TestHttpGetJsonReason:
    """The `reason` string returned by `_http_get_json` must include the response body
    so callers (e.g. ensure_ai_gateway_v2) can route on it. Before issue #84's fix
    the body was logged only when UCODE_DEBUG=1 and dropped from the bubbled error."""

    @staticmethod
    def _http_error(code: int, msg: str, body: str = ""):
        import io
        from unittest.mock import MagicMock
        from urllib.error import HTTPError

        fp = io.BytesIO(body.encode("utf-8")) if body else None
        return HTTPError(url="", code=code, msg=msg, hdrs=MagicMock(), fp=fp)

    def test_reason_includes_body_on_http_error(self):
        from unittest.mock import patch

        from ucode.databricks import _http_get_json

        exc = self._http_error(400, "Bad Request", body="Invalid Token")
        with patch("ucode.databricks.urllib_request.urlopen", side_effect=exc):
            payload, reason = _http_get_json("https://x/y", "tok")
        assert payload is None
        assert "HTTP 400" in reason
        assert "Invalid Token" in reason

    def test_reason_without_body_is_status_only(self):
        from unittest.mock import patch

        from ucode.databricks import _http_get_json

        exc = self._http_error(404, "Not Found")
        with patch("ucode.databricks.urllib_request.urlopen", side_effect=exc):
            payload, reason = _http_get_json("https://x/y", "tok")
        assert payload is None
        assert reason == "HTTP 404 Not Found"


class TestMakeSslContext:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        db_mod._make_ssl_context.cache_clear()
        yield
        db_mod._make_ssl_context.cache_clear()

    def test_context_is_cached(self, monkeypatch):
        from unittest.mock import Mock

        context = Mock()
        create_default_context = Mock(return_value=context)
        monkeypatch.setattr(db_mod.ssl, "create_default_context", create_default_context)

        assert db_mod._make_ssl_context() is context
        assert db_mod._make_ssl_context() is context
        create_default_context.assert_called_once_with()

    @pytest.mark.parametrize("load_error", [OSError("permission denied"), db_mod.ssl.SSLError()])
    def test_failed_bundle_falls_back_to_next_one(self, monkeypatch, load_error):
        from unittest.mock import Mock

        monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/first.pem")
        monkeypatch.setenv("CURL_CA_BUNDLE", "/second.pem")
        monkeypatch.setattr(db_mod.Path, "is_file", lambda self: True)
        context = Mock()
        context.load_verify_locations.side_effect = [load_error, None]
        monkeypatch.setattr(db_mod.ssl, "create_default_context", Mock(return_value=context))

        assert db_mod._make_ssl_context() is context
        assert context.load_verify_locations.call_args_list == [
            ((), {"cafile": "/first.pem"}),
            ((), {"cafile": "/second.pem"}),
        ]


class TestParseDatabricksCliVersion:
    def test_parses_standard_format(self):
        assert _parse_databricks_cli_version("Databricks CLI v0.299.2") == (0, 299, 2)

    def test_parses_without_v_prefix(self):
        assert _parse_databricks_cli_version("Databricks CLI 0.298.0") == (0, 298, 0)

    def test_returns_none_on_garbage(self):
        assert _parse_databricks_cli_version("not a version") is None


class TestEnsureDatabricksCliVersion:
    def _fake_databricks(self, tmp_path, version_output: str) -> dict:
        fake = tmp_path / "databricks"
        fake.write_text(f"#!/bin/sh\necho '{version_output}'\n")
        fake.chmod(0o755)
        return {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}

    def test_passes_when_version_meets_minimum(self, tmp_path, monkeypatch):
        env = self._fake_databricks(tmp_path, "Databricks CLI v0.298.0")
        monkeypatch.setattr("os.environ", env)
        ensure_databricks_cli_version()  # should not raise

    def test_passes_when_version_exceeds_minimum(self, tmp_path, monkeypatch):
        env = self._fake_databricks(tmp_path, "Databricks CLI v0.299.2")
        monkeypatch.setattr("os.environ", env)
        ensure_databricks_cli_version()

    def test_auto_upgrades_when_version_too_old(self, tmp_path, monkeypatch):
        import ucode.databricks as db_mod

        env = self._fake_databricks(tmp_path, "Databricks CLI v0.297.0")
        monkeypatch.setattr("os.environ", env)
        upgraded = []
        monkeypatch.setattr(
            db_mod,
            "_run_databricks_cli_installer",
            lambda brew_subcommand="install": upgraded.append(brew_subcommand),
        )
        # Stop the recursive re-check after upgrade
        call_count = [0]
        original = db_mod.ensure_databricks_cli_version

        def once(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                original()

        monkeypatch.setattr(db_mod, "ensure_databricks_cli_version", once)
        once()
        assert upgraded == ["upgrade"]

    def test_raises_when_version_unparseable(self, tmp_path, monkeypatch):
        env = self._fake_databricks(tmp_path, "completely broken output")
        monkeypatch.setattr("os.environ", env)
        with pytest.raises(RuntimeError, match="Could not parse"):
            ensure_databricks_cli_version()


class TestRunDatabricksCliInstaller:
    @pytest.mark.parametrize("brew_subcommand", ["install", "upgrade"])
    def test_macos_uses_fully_qualified_tap_formula(self, monkeypatch, brew_subcommand):
        calls = []
        monkeypatch.setattr(db_mod.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(db_mod.shutil, "which", lambda cmd: "/opt/homebrew/bin/brew")
        monkeypatch.setattr(db_mod, "run", lambda cmd, **kw: calls.append(cmd))

        _run_databricks_cli_installer(brew_subcommand=brew_subcommand)

        # The fully-qualified formula forces Homebrew to the Databricks CLI in
        # databricks/tap and fails if absent, rather than falling back to the
        # unrelated `databricks` cask.
        assert calls == [["brew", brew_subcommand, "databricks/tap/databricks"]]


class TestIsUsageTableAccessError:
    """Pin which `ServerOperationError` strings trigger the friendly
    `system.ai_gateway.usage` permissions hint vs. fall through to the
    generic `Usage query failed: ...` arm."""

    @staticmethod
    def _err(msg: str):
        from databricks.sql.exc import ServerOperationError

        return ServerOperationError(msg)

    def test_table_level_select_denial_matches(self):
        msg = (
            "[INSUFFICIENT_PERMISSIONS] Insufficient privileges: "
            "User does not have SELECT on Table 'system.ai_gateway.usage'. "
            "SQLSTATE: 42501"
        )
        assert db_mod._is_usage_table_access_error(self._err(msg)) is True

    def test_schema_level_use_schema_denial_matches(self):
        msg = (
            "[INSUFFICIENT_PERMISSIONS] Insufficient privileges: "
            "User does not have USE SCHEMA on Schema 'system.ai_gateway'. "
            "SQLSTATE: 42501"
        )
        assert db_mod._is_usage_table_access_error(self._err(msg)) is True

    def test_unrelated_catalog_denial_falls_through(self):
        msg = (
            "[INSUFFICIENT_PERMISSIONS] Insufficient privileges: "
            "User does not have USE CATALOG on Catalog 'aarushi'. "
            "SQLSTATE: 42501"
        )
        assert db_mod._is_usage_table_access_error(self._err(msg)) is False

    def test_other_error_code_on_same_table_falls_through(self):
        """Different code on the right table must not trip the gate — the
        helper requires INSUFFICIENT_PERMISSIONS specifically so we don't
        mask e.g. missing-table failures with a permissions-shaped hint."""
        msg = (
            "[TABLE_OR_VIEW_NOT_FOUND] The table or view "
            "`system`.`ai_gateway`.`usage` cannot be found. SQLSTATE: 42P01"
        )
        assert db_mod._is_usage_table_access_error(self._err(msg)) is False

    @pytest.mark.parametrize(
        "quoted",
        [
            "`system`.`ai_gateway`.`usage`",
            "[system].[ai_gateway].[usage]",
        ],
    )
    def test_identifier_quoting_variants_all_match(self, quoted):
        msg = (
            f"[INSUFFICIENT_PERMISSIONS] User does not have SELECT on Table "
            f"{quoted}. SQLSTATE: 42501"
        )
        assert db_mod._is_usage_table_access_error(self._err(msg)) is True


class TestRunUsageQuery:
    """Cover the two control-flow arms `_is_usage_table_access_error` gates:
    friendly RuntimeError for matching errors, raw-text fallback for the rest.
    `from exc` chaining is also pinned so `--debug` still surfaces the
    underlying connector error."""

    @staticmethod
    def _patch_connect_to_raise(monkeypatch, exc):
        import databricks.sql as sql_mod

        def fake_connect(*args, **kwargs):
            raise exc

        monkeypatch.setattr(sql_mod, "connect", fake_connect)

    def test_raises_actionable_message_for_table_access_error(self, monkeypatch):
        from databricks.sql.exc import ServerOperationError

        original = ServerOperationError(
            "[INSUFFICIENT_PERMISSIONS] Insufficient privileges: "
            "User does not have SELECT on Table 'system.ai_gateway.usage'. "
            "SQLSTATE: 42501"
        )
        self._patch_connect_to_raise(monkeypatch, original)

        with pytest.raises(RuntimeError, match="Ask your workspace admin") as exc_info:
            db_mod.run_usage_query(WS, "/sql/1.0/warehouses/abc", "tok", "SELECT 1")
        assert "system.ai_gateway.usage" in str(exc_info.value)
        # The original ServerOperationError must survive on __cause__ so
        # `--debug` / stack traces still show the underlying connector error.
        assert exc_info.value.__cause__ is original

    def test_falls_through_for_unrelated_permission_error(self, monkeypatch):
        from databricks.sql.exc import ServerOperationError

        original = ServerOperationError(
            "[INSUFFICIENT_PERMISSIONS] Insufficient privileges: "
            "User does not have USE CATALOG on Catalog 'aarushi'. SQLSTATE: 42501"
        )
        self._patch_connect_to_raise(monkeypatch, original)

        with pytest.raises(RuntimeError, match="aarushi") as exc_info:
            db_mod.run_usage_query(WS, "/sql/1.0/warehouses/abc", "tok", "SELECT 1")
        assert "Ask your workspace admin" not in str(exc_info.value)
        assert str(exc_info.value).startswith("Usage query failed:")
