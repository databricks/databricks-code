"""Tests for agents/claude.py."""

from __future__ import annotations

import json
import os

import pytest

from ucode.agents import claude

WS = "https://example.databricks.com"


class TestClaudeSpec:
    def test_binary(self):
        assert claude.SPEC["binary"] == "claude"

    def test_package(self):
        assert claude.SPEC["package"] == "@anthropic-ai/claude-code"

    def test_display(self):
        assert claude.SPEC["display"] == "Claude Code"


class TestRenderOverlay:
    def test_does_not_set_anthropic_model_env(self):
        # We deliberately don't pin ANTHROPIC_MODEL: when set, Claude Code's
        # /model picker surfaces a duplicate catalog row on top of the family
        # alias from ANTHROPIC_DEFAULT_OPUS_MODEL. Default falls back to the
        # active family alias instead.
        overlay, _ = claude.render_overlay(
            WS, "databricks-claude-opus-4-7", claude_models={"opus": "databricks-claude-opus-4-7"}
        )
        assert "ANTHROPIC_MODEL" not in overlay["env"]

    def test_adds_1m_suffix_for_opus_4_6_and_later(self):
        overlay, _ = claude.render_overlay(
            WS, "s4", claude_models={"opus": "databricks-claude-opus-4-7"}
        )
        assert overlay["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "databricks-claude-opus-4-7[1m]"

    def test_adds_1m_suffix_for_sonnet_4_6_and_later(self):
        overlay, _ = claude.render_overlay(
            WS, "s4", claude_models={"sonnet": "databricks-claude-sonnet-4-7"}
        )
        assert (
            overlay["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "databricks-claude-sonnet-4-7[1m]"
        )

    def test_does_not_add_1m_suffix_for_haiku(self):
        overlay, _ = claude.render_overlay(
            WS, "s4", claude_models={"haiku": "databricks-claude-haiku-4-6"}
        )
        assert overlay["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "databricks-claude-haiku-4-6"

    def test_does_not_duplicate_1m_suffix(self):
        overlay, _ = claude.render_overlay(
            WS, "s4", claude_models={"opus": "databricks-claude-opus-4-7[1m]"}
        )
        assert overlay["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "databricks-claude-opus-4-7[1m]"

    def test_adds_1m_suffix_for_model_services_name(self):
        overlay, _ = claude.render_overlay(
            WS, "s4", claude_models={"opus": "system.ai.claude-opus-4-8"}
        )
        assert overlay["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "system.ai.claude-opus-4-8[1m]"

    def test_no_1m_suffix_for_model_services_haiku(self):
        overlay, _ = claude.render_overlay(
            WS, "s4", claude_models={"haiku": "system.ai.claude-haiku-4-6"}
        )
        assert overlay["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "system.ai.claude-haiku-4-6"

    def test_sets_anthropic_base_url(self):
        overlay, _ = claude.render_overlay(WS, "s4")
        assert overlay["env"]["ANTHROPIC_BASE_URL"] == f"{WS}/ai-gateway/anthropic"

    def test_sets_custom_headers(self):
        overlay, _ = claude.render_overlay(WS, "s4")
        assert "x-databricks-use-coding-agent-mode" in overlay["env"]["ANTHROPIC_CUSTOM_HEADERS"]

    def test_does_not_disable_experimental_betas(self):
        # Would suppress the beta header 1h prompt caching needs.
        overlay, _ = claude.render_overlay(WS, "s4")
        assert "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS" not in overlay["env"]

    def test_enables_prompt_caching_1h(self):
        overlay, _ = claude.render_overlay(WS, "s4")
        assert overlay["env"]["ENABLE_PROMPT_CACHING_1H"] == "1"

    def test_enables_tool_search(self):
        overlay, _ = claude.render_overlay(WS, "s4")
        assert overlay["env"]["ENABLE_TOOL_SEARCH"] == "1"

    def test_enables_use_gateway(self):
        overlay, _ = claude.render_overlay(WS, "s4")
        assert overlay["env"]["CLAUDE_CODE_USE_GATEWAY"] == "1"

    def test_sets_api_key_helper(self):
        overlay, _ = claude.render_overlay(WS, "s4")
        assert "apiKeyHelper" in overlay
        assert WS in overlay["apiKeyHelper"]

    def test_model_overrides_when_all_provided(self):
        models = {
            "sonnet": "databricks-claude-sonnet-4-6",
            "opus": "databricks-claude-opus-4-7",
            "haiku": "databricks-claude-haiku-4-6",
        }
        overlay, _ = claude.render_overlay(WS, "s4", claude_models=models)
        env = overlay["env"]
        assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "databricks-claude-sonnet-4-6[1m]"
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "databricks-claude-opus-4-7[1m]"
        assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "databricks-claude-haiku-4-6"

    def test_model_overrides_partial(self):
        models = {"sonnet": "s4"}
        overlay, _ = claude.render_overlay(WS, "s4", claude_models=models)
        env = overlay["env"]
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" in env
        assert "ANTHROPIC_DEFAULT_OPUS_MODEL" not in env

    def test_model_overrides_not_set_when_no_models(self):
        overlay, _ = claude.render_overlay(WS, "s4")
        env = overlay["env"]
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" not in env

    def test_fable_not_pinned_by_default(self):
        # Fable is opt-in: even when the workspace advertises it, the env var is
        # absent unless fable_enabled is passed.
        models = {"fable": "databricks-claude-fable-5", "opus": "databricks-claude-opus-4-8"}
        overlay, _ = claude.render_overlay(WS, "s4", claude_models=models)
        env = overlay["env"]
        assert "ANTHROPIC_DEFAULT_FABLE_MODEL" not in env
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "databricks-claude-opus-4-8[1m]"

    def test_fable_pinned_when_enabled_and_discovered(self):
        models = {"fable": "system.ai.claude-fable-5"}
        overlay, _ = claude.render_overlay(WS, "s4", claude_models=models, fable_enabled=True)
        env = overlay["env"]
        # Fable 5 is 1M-context by default, so no `[1m]` suffix is appended.
        assert env["ANTHROPIC_DEFAULT_FABLE_MODEL"] == "system.ai.claude-fable-5"

    def test_fable_not_pinned_when_enabled_but_not_discovered(self):
        # --enable-fable is a no-op when the workspace advertises no fable model,
        # mirroring the opus/sonnet/haiku "only if discovered" behavior.
        models = {"opus": "databricks-claude-opus-4-8"}
        overlay, _ = claude.render_overlay(WS, "s4", claude_models=models, fable_enabled=True)
        assert "ANTHROPIC_DEFAULT_FABLE_MODEL" not in overlay["env"]

    def test_fable_not_pinned_under_provider(self):
        # A Model Provider Service routes by header and pins no Databricks model.
        models = {"fable": "databricks-claude-fable-5"}
        overlay, _ = claude.render_overlay(
            WS, "s4", claude_models=models, fable_enabled=True, provider="main.x.claude-svc"
        )
        assert "ANTHROPIC_DEFAULT_FABLE_MODEL" not in overlay["env"]

    def test_provider_adds_routing_header(self):
        overlay, _ = claude.render_overlay(WS, "s4", provider="main.aarushi.aarushi-claude")
        assert (
            "Databricks-Model-Provider-Service: main.aarushi.aarushi-claude"
            in overlay["env"]["ANTHROPIC_CUSTOM_HEADERS"]
        )

    def test_provider_skips_model_pinning(self):
        models = {
            "opus": "databricks-claude-opus-4-7",
            "sonnet": "databricks-claude-sonnet-4-6",
            "haiku": "databricks-claude-haiku-4-6",
        }
        overlay, _ = claude.render_overlay(
            WS, "s4", claude_models=models, provider="main.aarushi.aarushi-claude"
        )
        env = overlay["env"]
        assert "ANTHROPIC_DEFAULT_OPUS_MODEL" not in env
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" not in env
        assert "ANTHROPIC_DEFAULT_HAIKU_MODEL" not in env

    def test_no_provider_header_without_flag(self):
        overlay, _ = claude.render_overlay(WS, "s4")
        assert "Databricks-Model-Provider-Service" not in overlay["env"]["ANTHROPIC_CUSTOM_HEADERS"]

    def test_bedrock_provider_pins_model_ids(self):
        provider_models = {
            "opus": "global.anthropic.claude-opus-4-8",
            "sonnet": "us.anthropic.claude-sonnet-4-6",
            "haiku": "anthropic.claude-haiku-4-5",
        }
        overlay, _ = claude.render_overlay(
            WS,
            None,
            provider="main.bob.bedrock-svc",
            provider_models=provider_models,
        )
        env = overlay["env"]
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "global.anthropic.claude-opus-4-8"
        assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "us.anthropic.claude-sonnet-4-6"
        assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "anthropic.claude-haiku-4-5"
        # Bedrock ids are pinned verbatim — no `[1m]` suffix mangling.
        assert "[1m]" not in env["ANTHROPIC_DEFAULT_OPUS_MODEL"]
        assert (
            "Databricks-Model-Provider-Service: main.bob.bedrock-svc"
            in env["ANTHROPIC_CUSTOM_HEADERS"]
        )

    def test_picker_labels_show_raw_routable_id(self):
        # We deliberately don't set the `_NAME` companion env vars. Showing the
        # raw `system.ai.…` / `databricks-…` id in the picker label tells users
        # exactly which gateway-routable model is behind each shortcut, which is
        # more useful than a friendly catalog label for Databricks routing.
        models = {
            "opus": "system.ai.claude-opus-4-8",
            "sonnet": "databricks-claude-sonnet-4-6",
            "haiku": "system.ai.claude-haiku-4-5",
        }
        overlay, _ = claude.render_overlay(WS, "s4", claude_models=models)
        env = overlay["env"]
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "system.ai.claude-opus-4-8[1m]"
        assert "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME" not in env
        assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "databricks-claude-sonnet-4-6[1m]"
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME" not in env
        assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "system.ai.claude-haiku-4-5"
        assert "ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME" not in env

    def test_managed_keys_include_api_key_helper(self):
        _, keys = claude.render_overlay(WS, "s4")
        assert ["apiKeyHelper"] in keys

    def test_managed_keys_include_env_entries(self):
        _, keys = claude.render_overlay(WS, "s4")
        env_keys = [k for k in keys if len(k) == 2 and k[0] == "env"]
        assert len(env_keys) > 0


class TestRenderOverlayUserAgent:
    def _ua(self, monkeypatch) -> str:
        monkeypatch.setattr(claude, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(claude, "agent_version", lambda binary: "2.1.136")
        overlay, _ = claude.render_overlay(WS, "s4")
        return overlay["env"]["ANTHROPIC_CUSTOM_HEADERS"]

    def test_user_agent_present(self, monkeypatch):
        assert "User-Agent: ucode/0.1.0 claude/2.1.136" in self._ua(monkeypatch)

    def test_existing_databricks_header_preserved(self, monkeypatch):
        assert "x-databricks-use-coding-agent-mode: true" in self._ua(monkeypatch)

    def test_headers_newline_delimited(self, monkeypatch):
        assert "\n" in self._ua(monkeypatch)


class TestRenderOverlayWebSearchDisable:
    def test_settings_overlay_never_includes_mcp_servers(self):
        # MCP servers belong in ~/.claude.json, not settings.json.
        overlay, _ = claude.render_overlay(WS, "s4", disable_web_search=True)
        assert "mcpServers" not in overlay

    def test_disables_builtin_websearch_when_requested(self):
        # A bare `permissions.deny` entry removes the built-in WebSearch tool
        # from Claude's context (Claude Code has no `disabledTools` setting).
        overlay, _ = claude.render_overlay(WS, "s4", disable_web_search=True)
        assert overlay["permissions"] == {"deny": ["WebSearch"]}

    def test_no_disable_when_not_requested(self):
        overlay, _ = claude.render_overlay(WS, "s4", disable_web_search=False)
        assert "permissions" not in overlay

    def test_managed_keys_include_disabled_tools_when_set(self):
        _, keys = claude.render_overlay(WS, "s4", disable_web_search=True)
        assert ["permissions", "deny"] in keys

    def test_managed_keys_omit_disabled_tools_when_not_set(self):
        _, keys = claude.render_overlay(WS, "s4", disable_web_search=False)
        assert ["permissions", "deny"] not in keys


class TestWebSearchMcpEntry:
    def test_entry_shape(self):
        entry = claude._web_search_mcp_entry(WS, "databricks-gpt-5")
        assert entry["type"] == "stdio"
        assert entry["args"] == ["mcp", "web-search"]
        assert entry["env"]["DATABRICKS_HOST"] == WS
        assert entry["env"]["UCODE_WEB_SEARCH_MODEL"] == "databricks-gpt-5"
        assert isinstance(entry["command"], str) and entry["command"]


class TestResolveWebSearchModel:
    def test_uses_explicit_override(self):
        assert claude._resolve_web_search_model({"web_search_model": "explicit"}) == "explicit"

    def test_falls_back_to_first_codex_model(self):
        state = {"codex_models": ["m1", "m2"]}
        assert claude._resolve_web_search_model(state) == "m1"

    def test_returns_none_when_no_codex_models(self):
        assert claude._resolve_web_search_model({}) is None
        assert claude._resolve_web_search_model({"codex_models": []}) is None

    def test_override_wins_over_codex_models(self):
        state = {"web_search_model": "winner", "codex_models": ["loser"]}
        assert claude._resolve_web_search_model(state) == "winner"


class TestClaudeDefaultModel:
    def test_prefers_opus(self):
        state = {"claude_models": {"sonnet": "s4", "opus": "o4", "haiku": "h4"}}
        assert claude.default_model(state) == "o4"

    def test_falls_back_to_sonnet(self):
        state = {"claude_models": {"sonnet": "s4", "haiku": "h4"}}
        assert claude.default_model(state) == "s4"

    def test_falls_back_to_haiku(self):
        state = {"claude_models": {"haiku": "h4"}}
        assert claude.default_model(state) == "h4"

    def test_returns_none_when_no_models(self):
        assert claude.default_model({}) is None
        assert claude.default_model({"claude_models": {}}) is None


class TestClaudeValidateCmd:
    def test_starts_with_binary(self):
        cmd = claude.validate_cmd("claude")
        assert cmd[0] == "claude"

    def test_has_p_flag(self):
        cmd = claude.validate_cmd("claude")
        assert "-p" in cmd

    def test_uses_ucode_settings_file(self):
        cmd = claude.validate_cmd("claude")
        assert cmd[:3] == ["claude", "--settings", str(claude.CLAUDE_SETTINGS_PATH)]

    def test_has_max_turns(self):
        cmd = claude.validate_cmd("claude")
        assert "--max-turns" in cmd
        idx = cmd.index("--max-turns")
        assert cmd[idx + 1] == "1"


class TestWriteToolConfigMcpRegistration:
    def _common_patches(self, monkeypatch, calls):
        monkeypatch.setattr(claude, "backup_existing_file", lambda *a, **kw: True)
        monkeypatch.setattr(claude, "read_json_safe", lambda path: {})
        monkeypatch.setattr(claude, "write_json_file", lambda path, payload: None)
        monkeypatch.setattr(claude, "save_state", lambda state: None)
        monkeypatch.setattr(
            claude,
            "_register_web_search_mcp",
            lambda ws, model, profile=None: calls.append(("register", ws, model)),
        )

    def test_registers_mcp_when_codex_model_available(self, monkeypatch):
        calls: list = []
        self._common_patches(monkeypatch, calls)
        state = {"workspace": WS, "codex_models": ["databricks-gpt-5"]}
        claude.write_tool_config(state, "databricks-claude-sonnet-4")
        assert calls == [("register", WS, "databricks-gpt-5")]

    def test_skips_registration_without_codex_model(self, monkeypatch):
        calls: list = []
        self._common_patches(monkeypatch, calls)
        state = {"workspace": WS, "codex_models": []}
        claude.write_tool_config(state, "databricks-claude-sonnet-4")
        assert calls == []

    def test_explicit_override_used_over_codex_models(self, monkeypatch):
        calls: list = []
        self._common_patches(monkeypatch, calls)
        state = {
            "workspace": WS,
            "web_search_model": "explicit-model",
            "codex_models": ["other-model"],
        }
        claude.write_tool_config(state, "databricks-claude-sonnet-4")
        assert calls == [("register", WS, "explicit-model")]


class TestWriteToolConfigStripsRemovedEnvKeys:
    """Stale keys ucode no longer writes are dropped from the merged settings."""

    def _patch(self, monkeypatch, existing, written):
        monkeypatch.setattr(claude, "backup_existing_file", lambda *a, **kw: True)
        monkeypatch.setattr(claude, "read_json_safe", lambda path: existing)
        monkeypatch.setattr(
            claude, "write_json_file", lambda path, payload: written.append(payload)
        )
        monkeypatch.setattr(claude, "save_state", lambda state: None)
        monkeypatch.setattr(claude, "_register_web_search_mcp", lambda *a, **kw: True)

    def test_strips_stale_disable_experimental_betas(self, monkeypatch):
        existing = {"env": {"CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1"}}
        written: list = []
        self._patch(monkeypatch, existing, written)
        state = {"workspace": WS, "codex_models": []}
        claude.write_tool_config(state, "databricks-claude-sonnet-4")
        assert "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS" not in written[0]["env"]
        assert written[0]["env"]["ENABLE_PROMPT_CACHING_1H"] == "1"
        assert written[0]["env"]["ENABLE_TOOL_SEARCH"] == "1"
        assert written[0]["env"]["CLAUDE_CODE_USE_GATEWAY"] == "1"


class TestRegisterWebSearchMcp:
    def test_clears_existing_then_adds(self, monkeypatch):
        import ucode.mcp as mcp_mod

        removed: list[str] = []
        added: list = []
        monkeypatch.setattr(
            mcp_mod, "remove_claude_mcp_server", lambda name, scope: removed.append(scope) or True
        )
        monkeypatch.setattr(
            mcp_mod,
            "add_claude_mcp_server",
            lambda name, entry, scope=mcp_mod.MCP_USER_SCOPE: added.append((name, entry, scope)),
        )
        claude._register_web_search_mcp(WS, "databricks-gpt-5")
        assert removed == list(mcp_mod.MCP_CLEANUP_SCOPES)
        assert len(added) == 1
        name, entry, _ = added[0]
        assert name == "web_search"
        assert entry["env"]["UCODE_WEB_SEARCH_MODEL"] == "databricks-gpt-5"

    def test_remove_failures_are_swallowed(self, monkeypatch):
        import ucode.mcp as mcp_mod

        def boom(name, scope):
            raise RuntimeError("nope")

        added: list = []
        monkeypatch.setattr(mcp_mod, "remove_claude_mcp_server", boom)
        monkeypatch.setattr(
            mcp_mod,
            "add_claude_mcp_server",
            lambda name, entry, scope=mcp_mod.MCP_USER_SCOPE: added.append(name),
        )
        claude._register_web_search_mcp(WS, "m")
        assert added == ["web_search"]

    def test_add_failure_is_non_blocking_and_warns(self, monkeypatch, capsys):
        # Regression: a failing `claude mcp add-json` used to abort the whole
        # `ucode claude` setup. It must now warn and return False instead.
        import ucode.mcp as mcp_mod

        monkeypatch.setattr(mcp_mod, "remove_claude_mcp_server", lambda name, scope: False)

        def boom(name, entry, scope=mcp_mod.MCP_USER_SCOPE):
            raise RuntimeError("Failed to add MCP server 'web_search' via claude CLI.")

        monkeypatch.setattr(mcp_mod, "add_claude_mcp_server", boom)
        result = claude._register_web_search_mcp(WS, "m")
        assert result is False
        captured = capsys.readouterr()
        assert "web_search" in captured.out.lower() or "web search" in captured.out.lower()

    def test_add_success_returns_true(self, monkeypatch):
        import ucode.mcp as mcp_mod

        monkeypatch.setattr(mcp_mod, "remove_claude_mcp_server", lambda name, scope: False)
        monkeypatch.setattr(
            mcp_mod,
            "add_claude_mcp_server",
            lambda name, entry, scope=mcp_mod.MCP_USER_SCOPE: None,
        )
        assert claude._register_web_search_mcp(WS, "m") is True

    def test_write_tool_config_completes_when_mcp_registration_fails(self, monkeypatch):
        # Regression for issue #100: a `claude mcp add-json` failure must not
        # block the rest of `ucode claude` setup (state save, managed-key
        # marking, etc.) from completing.
        import ucode.mcp as mcp_mod

        monkeypatch.setattr(claude, "backup_existing_file", lambda *a, **kw: True)
        monkeypatch.setattr(claude, "read_json_safe", lambda path: {})
        monkeypatch.setattr(claude, "write_json_file", lambda path, payload: None)
        saved: list[dict] = []
        monkeypatch.setattr(claude, "save_state", lambda state: saved.append(state))
        monkeypatch.setattr(mcp_mod, "remove_claude_mcp_server", lambda name, scope: False)

        def boom(name, entry, scope=mcp_mod.MCP_USER_SCOPE):
            raise RuntimeError("Failed to add MCP server 'web_search' via claude CLI.")

        monkeypatch.setattr(mcp_mod, "add_claude_mcp_server", boom)

        state = {"workspace": WS, "codex_models": ["databricks-gpt-5"]}
        result = claude.write_tool_config(state, "databricks-claude-sonnet-4")
        assert saved, "save_state should still be called when MCP registration fails"
        assert result["workspace"] == WS


class TestClaudeLaunch:
    def test_sets_oauth_token_before_exec(self, monkeypatch):
        exec_calls: list[tuple[str, list[str]]] = []

        def fake_execvp(binary: str, args: list[str]) -> None:
            exec_calls.append((binary, args))
            raise RuntimeError("stop")

        monkeypatch.delenv("OAUTH_TOKEN", raising=False)
        monkeypatch.setattr(
            claude, "get_databricks_token", lambda workspace, profile=None: "fresh-token"
        )
        monkeypatch.setattr(os, "execvp", fake_execvp)

        try:
            claude.launch({"workspace": WS}, ["--debug"])
        except RuntimeError as exc:
            assert str(exc) == "stop"

        assert os.environ["OAUTH_TOKEN"] == "fresh-token"
        assert exec_calls == [
            (
                "claude",
                ["claude", "--settings", str(claude.CLAUDE_SETTINGS_PATH), "--debug"],
            )
        ]


class TestWriteToolConfigPrunesStaleModelEnv:
    """Stale ucode-managed model env keys (ANTHROPIC_MODEL, etc.) from earlier
    ucode versions must be removed on every launch — otherwise they linger in
    settings.json and re-introduce the duplicate /model picker row that this
    change is meant to remove.
    """

    def _patch(self, monkeypatch, existing_settings):
        monkeypatch.setattr(claude, "backup_existing_file", lambda *a, **kw: True)
        monkeypatch.setattr(claude, "read_json_safe", lambda path: existing_settings)
        written: dict = {}

        def fake_write(path, payload):
            written["payload"] = payload

        monkeypatch.setattr(claude, "write_json_file", fake_write)
        monkeypatch.setattr(claude, "save_state", lambda state: None)
        monkeypatch.setattr(claude, "_register_web_search_mcp", lambda *a, **kw: True)
        return written

    def test_prunes_stale_anthropic_model_from_prior_run(self, monkeypatch):
        existing = {
            "env": {
                "ANTHROPIC_MODEL": "system.ai.claude-opus-4-8[1m]",
                "ANTHROPIC_DEFAULT_OPUS_MODEL": "system.ai.claude-opus-4-8[1m]",
                "MY_CUSTOM_VAR": "keep-me",
            }
        }
        written = self._patch(monkeypatch, existing)
        state = {
            "workspace": WS,
            "claude_models": {"opus": "system.ai.claude-opus-4-8"},
        }
        claude.write_tool_config(state, "system.ai.claude-opus-4-8")
        env = written["payload"]["env"]
        assert "ANTHROPIC_MODEL" not in env
        # Family default we still write this run is preserved.
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "system.ai.claude-opus-4-8[1m]"
        # User-owned keys are untouched.
        assert env["MY_CUSTOM_VAR"] == "keep-me"

    def test_prunes_unused_family_default_when_models_change(self, monkeypatch):
        # Earlier launch wrote a sonnet default; the new state only has opus.
        # The stale sonnet keys should be removed.
        existing = {
            "env": {
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "databricks-claude-sonnet-4-6[1m]",
            }
        }
        written = self._patch(monkeypatch, existing)
        state = {"workspace": WS, "claude_models": {"opus": "system.ai.claude-opus-4-8"}}
        claude.write_tool_config(state, "system.ai.claude-opus-4-8")
        env = written["payload"]["env"]
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" not in env
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "system.ai.claude-opus-4-8[1m]"

    def test_prunes_stale_name_companion_keys_from_older_ucode(self, monkeypatch):
        # An older ucode build briefly wrote `_NAME` companion env vars to give
        # the picker friendly labels. The current build only writes the raw id,
        # so any leftover `_NAME` keys must be pruned — otherwise users who
        # tested the in-between version would see stale labels.
        existing = {
            "env": {
                "ANTHROPIC_DEFAULT_OPUS_MODEL": "system.ai.claude-opus-4-8[1m]",
                "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME": "Opus 4.8 (1M)",
                "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME": "Sonnet 4.6 (1M)",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME": "Haiku 4.5",
            }
        }
        written = self._patch(monkeypatch, existing)
        state = {"workspace": WS, "claude_models": {"opus": "system.ai.claude-opus-4-8"}}
        claude.write_tool_config(state, "system.ai.claude-opus-4-8")
        env = written["payload"]["env"]
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "system.ai.claude-opus-4-8[1m]"
        assert "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME" not in env
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME" not in env
        assert "ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME" not in env


class TestBuildClaudeArgv:
    def test_no_caller_settings_uses_ucode_file(self, monkeypatch):
        monkeypatch.setattr(claude, "read_json_safe", lambda p: {"apiKeyHelper": "u"})
        argv = claude._build_claude_argv("claude", ["-p", "hi"])
        assert argv == ["claude", "--settings", str(claude.CLAUDE_SETTINGS_PATH), "-p", "hi"]

    def test_inline_caller_settings_merged_into_single_flag(self, monkeypatch):
        ucode_settings = {
            "apiKeyHelper": "ucode-helper",
            "env": {"ANTHROPIC_BASE_URL": "https://gw"},
            "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "ucode-stop"}]}]},
        }
        monkeypatch.setattr(claude, "read_json_safe", lambda p: ucode_settings)
        caller = json.dumps(
            {
                "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "caller-stop"}]}]},
                "statusLine": {"type": "command", "command": "sl"},
            }
        )
        argv = claude._build_claude_argv("claude", ["--settings", caller, "-p", "hi"])
        # Exactly one --settings reaches Claude, and the caller's raw flag is gone.
        assert argv.count("--settings") == 1
        assert argv[:2] == ["claude", "--settings"]
        assert argv[3:] == ["-p", "hi"]
        merged = json.loads(argv[2])
        # ucode's gateway config survives.
        assert merged["apiKeyHelper"] == "ucode-helper"
        assert merged["env"]["ANTHROPIC_BASE_URL"] == "https://gw"
        # The caller's own (non-hook) settings pass through.
        assert merged["statusLine"] == {"type": "command", "command": "sl"}
        # Hooks from BOTH sides fire (unioned, not clobbered).
        stop_cmds = [h["command"] for e in merged["hooks"]["Stop"] for h in e["hooks"]]
        assert "ucode-stop" in stop_cmds
        assert "caller-stop" in stop_cmds

    def test_equals_form_is_handled(self, monkeypatch):
        monkeypatch.setattr(claude, "read_json_safe", lambda p: {"apiKeyHelper": "u"})
        caller = json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "c"}]}]}})
        argv = claude._build_claude_argv("claude", [f"--settings={caller}"])
        assert argv.count("--settings") == 1
        merged = json.loads(argv[2])
        assert merged["apiKeyHelper"] == "u"
        assert merged["hooks"]["Stop"][0]["hooks"][0]["command"] == "c"

    def test_ucode_wins_on_conflicting_env(self, monkeypatch):
        monkeypatch.setattr(
            claude, "read_json_safe", lambda p: {"env": {"ANTHROPIC_BASE_URL": "https://ucode"}}
        )
        caller = json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://caller", "FOO": "bar"}})
        argv = claude._build_claude_argv("claude", ["--settings", caller])
        merged = json.loads(argv[2])
        assert merged["env"]["ANTHROPIC_BASE_URL"] == "https://ucode"  # ucode wins
        assert merged["env"]["FOO"] == "bar"  # caller's non-conflicting key kept

    def test_file_path_caller_settings(self, tmp_path, monkeypatch):
        caller_file = tmp_path / "caller.json"
        caller_file.write_text(
            json.dumps(
                {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "cs"}]}]}}
            )
        )
        # The caller file is read directly; read_json_safe is only used for
        # ucode's own settings file.
        monkeypatch.setattr(claude, "read_json_safe", lambda p: {"apiKeyHelper": "u"})
        argv = claude._build_claude_argv("claude", ["--settings", str(caller_file)])
        assert argv.count("--settings") == 1
        merged = json.loads(argv[2])
        assert merged["apiKeyHelper"] == "u"
        assert merged["hooks"]["SessionStart"][0]["hooks"][0]["command"] == "cs"

    def test_malformed_inline_json_raises(self, monkeypatch):
        monkeypatch.setattr(claude, "read_json_safe", lambda p: {"apiKeyHelper": "u"})
        # Clearly-intended-as-JSON but broken: fail loudly rather than pass it
        # through as a second, colliding --settings flag.
        with pytest.raises(RuntimeError, match="not valid JSON"):
            claude._build_claude_argv("claude", ["--settings", '{"hooks": '])

    def test_nonexistent_file_raises(self, monkeypatch):
        monkeypatch.setattr(claude, "read_json_safe", lambda p: {"apiKeyHelper": "u"})
        with pytest.raises(RuntimeError, match="file not found"):
            claude._build_claude_argv("claude", ["--settings", "/no/such/settings.json"])

    def test_non_object_file_json_raises(self, tmp_path, monkeypatch):
        # A --settings file whose JSON is not an object (e.g. an array) can't be
        # merged; fail loudly. (An inline value only enters the JSON branch when
        # it starts with "{", so the non-object case is reachable via a file.)
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("[1, 2, 3]")
        monkeypatch.setattr(claude, "read_json_safe", lambda p: {"apiKeyHelper": "u"})
        with pytest.raises(RuntimeError, match="must be a JSON object"):
            claude._build_claude_argv("claude", ["--settings", str(bad_file)])

    def test_malformed_file_json_raises(self, tmp_path, monkeypatch):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text('{"hooks": ')
        monkeypatch.setattr(claude, "read_json_safe", lambda p: {"apiKeyHelper": "u"})
        with pytest.raises(RuntimeError, match="not valid JSON"):
            claude._build_claude_argv("claude", ["--settings", str(bad_file)])
