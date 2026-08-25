"""Tests for agents/claude.py."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ucode.agents import claude
from ucode.smart_routing import claude_routing

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

    def test_custom_model_pins_all_family_aliases(self):
        # `ucode claude --model` pins the id into every family alias so it takes effect whichever
        # slot Claude Code resolves — and NOT into ANTHROPIC_MODEL, which Claude Code validates and
        # rejects for a raw Databricks id. It overrides the discovered-model aliases.
        overlay, _ = claude.render_overlay(
            WS,
            "s4",
            claude_models={"opus": "system.ai.claude-opus-4-8", "sonnet": "system.ai.sonnet"},
            custom_model="main.aarushi.claude-opus-5",
        )
        env = overlay["env"]
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "main.aarushi.claude-opus-5"
        assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "main.aarushi.claude-opus-5"
        assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "main.aarushi.claude-opus-5"
        assert "ANTHROPIC_MODEL" not in env
        # No [1m] suffix is appended to the custom id — it's passed through verbatim.
        assert "[1m]" not in env["ANTHROPIC_DEFAULT_OPUS_MODEL"]

    def test_custom_model_pins_fable_alias_only_when_fable_enabled(self):
        without = claude.render_overlay(WS, "s4", claude_models={}, custom_model="main.x.m")[0][
            "env"
        ]
        assert "ANTHROPIC_DEFAULT_FABLE_MODEL" not in without
        with_fable = claude.render_overlay(
            WS, "s4", claude_models={}, custom_model="main.x.m", fable_enabled=True
        )[0]["env"]
        assert with_fable["ANTHROPIC_DEFAULT_FABLE_MODEL"] == "main.x.m"

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

    @pytest.mark.parametrize("env_value", [None, "", "0", "true", "yes"])
    def test_gateway_model_discovery_disabled_unless_opted_in(self, monkeypatch, env_value):
        if env_value is not None:
            monkeypatch.setenv("ENABLE_CLAUDE_CODE_GATEWAY_MODEL_DISCOVERY", env_value)
        overlay, _ = claude.render_overlay(WS, "s4")
        assert "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY" not in overlay["env"]

    def test_enables_gateway_model_discovery(self, monkeypatch):
        monkeypatch.setenv("ENABLE_CLAUDE_CODE_GATEWAY_MODEL_DISCOVERY", "1")
        overlay, _ = claude.render_overlay(WS, "s4")
        assert overlay["env"]["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"

    def test_gateway_model_discovery_skipped_under_provider(self, monkeypatch):
        # A Model Provider Service routes every request to the external provider,
        # so a discovered gateway endpoint id would reach a provider that can't
        # resolve it — discovery must be off in that mode.
        monkeypatch.setenv("ENABLE_CLAUDE_CODE_GATEWAY_MODEL_DISCOVERY", "1")
        overlay, _ = claude.render_overlay(WS, "s4", provider="main.x.claude-svc")
        assert "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY" not in overlay["env"]

    def test_sets_api_key_helper(self):
        overlay, _ = claude.render_overlay(WS, "s4")
        assert "apiKeyHelper" in overlay
        assert WS in overlay["apiKeyHelper"]

    def test_relayed_omits_api_key_helper(self):
        # Claude Code's own subscription OAuth must own Authorization; an
        # apiKeyHelper would outrank it.
        overlay, _ = claude.render_overlay(
            WS,
            None,
            provider="c.s.mps",
            relayed=True,
            relayed_base_url="http://127.0.0.1:9",
        )
        assert "apiKeyHelper" not in overlay

    def test_relayed_points_base_url_at_proxy(self):
        overlay, _ = claude.render_overlay(
            WS,
            None,
            provider="c.s.mps",
            relayed=True,
            relayed_base_url="http://127.0.0.1:9",
        )
        assert overlay["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9"

    def test_relayed_sends_mps_header_but_not_swap_token(self):
        # The MPS header selects the service; the swap token is injected by the
        # proxy, never written into settings.
        overlay, _ = claude.render_overlay(
            WS,
            None,
            provider="c.s.mps",
            relayed=True,
            relayed_base_url="http://127.0.0.1:9",
        )
        headers = overlay["env"]["ANTHROPIC_CUSTOM_HEADERS"]
        assert "Databricks-Model-Provider-Service: c.s.mps" in headers
        assert "X-Databricks-AI-Gateway-Token" not in headers

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


FAKE_MANAGED_PATH = Path("/tmp/ucode-test/managed-settings.json")


class TestWriteToolConfigManagedSettings:
    """use_as_global_settings: also write Claude Code's OS managed-settings.json (via sudo, mocked)."""

    def _patch(self, monkeypatch, private_writes, managed_writes, existing_by_path=None):
        existing_by_path = existing_by_path or {}
        monkeypatch.setattr(claude, "backup_existing_file", lambda *a, **kw: True)
        # Deep-copy the seeded existing content so the compose step can't mutate the fixture.
        monkeypatch.setattr(
            claude,
            "read_json_safe",
            lambda path: json.loads(json.dumps(existing_by_path.get(str(path), {}))),
        )
        monkeypatch.setattr(
            claude,
            "write_json_file",
            lambda path, payload: private_writes.append((str(path), payload)),
        )
        monkeypatch.setattr(claude, "save_state", lambda state: None)
        monkeypatch.setattr(claude, "_register_web_search_mcp", lambda *a, **kw: True)
        # Deterministic managed path, and a mocked sudo writer so NO real sudo/`/etc` write happens.
        monkeypatch.setattr(claude, "_managed_settings_path", lambda: FAKE_MANAGED_PATH)

        def fake_write_managed(path, text, *, display):
            managed_writes.append((str(path), text))
            return "written"

        monkeypatch.setattr(claude, "write_managed_file", fake_write_managed)

    def test_writes_managed_file_when_flagged(self, monkeypatch):
        private_writes: list = []
        managed_writes: list = []
        self._patch(monkeypatch, private_writes, managed_writes)
        state = {"workspace": WS, "codex_models": [], "write_managed_config": True}
        claude.write_tool_config(state, "databricks-claude-sonnet-4")
        # Private file still written; managed file written too.
        assert str(claude.CLAUDE_SETTINGS_PATH) in [p for p, _ in private_writes]
        assert [p for p, _ in managed_writes] == [str(FAKE_MANAGED_PATH)]

    def test_managed_file_preserves_other_keys(self, monkeypatch):
        private_writes: list = []
        managed_writes: list = []
        # An IT-authored key already in the managed file must survive the merge.
        existing = {str(FAKE_MANAGED_PATH): {"env": {"MY_OWN": "keep"}}}
        self._patch(monkeypatch, private_writes, managed_writes, existing)
        state = {"workspace": WS, "codex_models": [], "write_managed_config": True}
        claude.write_tool_config(state, "databricks-claude-sonnet-4")
        _, text = managed_writes[0]
        written = json.loads(text)
        assert written["env"]["MY_OWN"] == "keep"
        assert written["env"]["ANTHROPIC_BASE_URL"]
        assert written["apiKeyHelper"]

    def test_no_managed_write_by_default(self, monkeypatch):
        private_writes: list = []
        managed_writes: list = []
        self._patch(monkeypatch, private_writes, managed_writes)
        state = {"workspace": WS, "codex_models": []}
        claude.write_tool_config(state, "databricks-claude-sonnet-4")
        assert managed_writes == []

    def test_relayed_skips_managed_write(self, monkeypatch):
        private_writes: list = []
        managed_writes: list = []
        warns: list = []
        self._patch(monkeypatch, private_writes, managed_writes)
        monkeypatch.setattr(claude, "print_warning", lambda msg: warns.append(msg))
        monkeypatch.setattr(claude, "relayed_proxy_base_url", lambda state: "http://127.0.0.1:9999")
        state = {"workspace": WS, "codex_models": [], "write_managed_config": True}
        claude.write_tool_config(state, "databricks-claude-sonnet-4", relayed=True)
        assert managed_writes == []
        assert any("bare `claude`" in w for w in warns)


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
    def test_default_launch_keeps_existing_auth_path(self, monkeypatch):
        calls: list[list[str]] = []
        monkeypatch.delenv(claude.GATEWAY_MODEL_DISCOVERY_ENV_VAR, raising=False)
        monkeypatch.delenv("OAUTH_TOKEN", raising=False)
        monkeypatch.setattr(claude, "get_databricks_token", lambda *_args: "token")
        monkeypatch.setattr(claude, "exec_or_spawn", lambda argv: calls.append(argv))

        claude.launch({"workspace": WS, "profile": "test"}, ["--debug"])

        assert os.environ["OAUTH_TOKEN"] == "token"
        assert calls == [["claude", "--settings", str(claude.CLAUDE_SETTINGS_PATH), "--debug"]]

    def test_runs_through_refresh_proxy(self, monkeypatch):
        calls: list[tuple] = []

        class Server:
            server_address = ("127.0.0.1", 12345)

            def serve_forever(self):
                calls.append(("serve",))

            def shutdown(self):
                calls.append(("shutdown",))

        class Cache:
            token = "fresh-token"

            def stop(self):
                calls.append(("stop",))

        class Client:
            def close(self):
                calls.append(("close",))

        class Process:
            def __init__(self, argv):
                calls.append(("popen", argv))

            def wait(self):
                return 0

        def start_proxy(workspace, profile, port, token_header, force_refresh_near_expiry):
            calls.append(
                (
                    "proxy",
                    workspace,
                    profile,
                    port,
                    token_header,
                    force_refresh_near_expiry,
                )
            )
            return Server(), Cache(), Client()

        monkeypatch.setenv(claude.GATEWAY_MODEL_DISCOVERY_ENV_VAR, "1")
        monkeypatch.delenv("OAUTH_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_GATEWAY", raising=False)
        monkeypatch.setattr(claude, "start_proxy", start_proxy)
        monkeypatch.setattr(claude.subprocess, "Popen", Process)

        with pytest.raises(SystemExit) as exc:
            claude.launch({"workspace": WS, "profile": "test"}, ["--debug"])

        assert exc.value.code == 0
        assert os.environ["OAUTH_TOKEN"] == "fresh-token"
        assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "fresh-token"
        assert os.environ["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:12345"
        assert os.environ["CLAUDE_CODE_USE_GATEWAY"] == "1"
        assert calls[:2] == [
            ("proxy", WS, "test", 0, claude.AUTHORIZATION_HEADER, True),
            ("serve",),
        ]
        assert calls[2][0] == "popen"
        argv = calls[2][1]
        assert argv[:2] == ["claude", "--settings"]
        assert json.loads(argv[2])["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:12345"
        assert argv[3:] == ["--debug"]
        assert calls[3:] == [
            ("stop",),
            ("shutdown",),
            ("close",),
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

    def test_non_relayed_does_not_set_setting_sources(self, monkeypatch):
        # Normal launches must keep loading user settings (hooks/permissions) —
        # no --setting-sources so nothing changes for the stored-key path.
        monkeypatch.setattr(claude, "read_json_safe", lambda p: {"apiKeyHelper": "u"})
        argv = claude._build_claude_argv("claude", ["-p", "hi"], relayed=False)
        assert "--setting-sources" not in argv

    def test_relayed_excludes_user_scope_via_setting_sources(self, monkeypatch):
        # Relayed must drop the user scope so a stale ~/.claude/settings.json
        # apiKeyHelper can't merge through and shadow the subscription OAuth.
        monkeypatch.setattr(claude, "read_json_safe", lambda p: {"env": {}})
        argv = claude._build_claude_argv("claude", ["-p", "hi"], relayed=True)
        assert "--setting-sources" in argv
        src = argv[argv.index("--setting-sources") + 1]
        assert src == claude._RELAYED_SETTING_SOURCES
        assert "user" not in src
        # ucode's own settings file is still passed.
        assert "--settings" in argv
        assert str(claude.CLAUDE_SETTINGS_PATH) in argv

    def test_relayed_with_caller_settings_keeps_setting_sources(self, monkeypatch):
        # Even when composing a caller --settings, relayed still excludes user scope.
        monkeypatch.setattr(claude, "read_json_safe", lambda p: {"env": {}})
        caller = json.dumps({"statusLine": {"type": "command", "command": "sl"}})
        argv = claude._build_claude_argv("claude", ["--settings", caller], relayed=True)
        assert argv[:3] == ["claude", "--setting-sources", claude._RELAYED_SETTING_SOURCES]
        assert argv.count("--settings") == 1

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


class TestClaudeSmartRouting:
    def _capture_write(self, monkeypatch, existing, written):
        monkeypatch.setattr(claude, "backup_existing_file", lambda *a, **kw: True)
        monkeypatch.setattr(claude, "read_json_safe", lambda path: existing)
        monkeypatch.setattr(
            claude, "write_json_file", lambda path, payload: written.append(payload)
        )
        monkeypatch.setattr(claude, "save_state", lambda state: None)
        monkeypatch.setattr(claude, "_register_web_search_mcp", lambda *a, **kw: True)

    def test_enable_sets_state_key(self):
        state = claude.enable_smart_routing({})
        assert state[claude.SMART_ROUTING_STATE_KEY] is True

    def test_enabled_reads_state_key(self):
        assert claude.smart_routing_enabled({claude.SMART_ROUTING_STATE_KEY: True})
        assert not claude.smart_routing_enabled({})

    def test_write_config_installs_routing_hooks(self, monkeypatch):
        written: list = []
        self._capture_write(monkeypatch, {}, written)
        state = {
            "workspace": WS,
            "claude_models": {"opus": "system.ai.claude-opus-4-8"},
            claude.SMART_ROUTING_STATE_KEY: True,
        }
        claude.write_tool_config(state, "system.ai.claude-opus-4-8", route_root_model=None)
        hooks = written[0]["hooks"]
        assert set(hooks) == {"PreToolUse", "SessionStart", "SubagentStart"}
        commands = [hook["command"] for group in hooks["PreToolUse"] for hook in group["hooks"]]
        assert any("claude-router-hook" in command for command in commands)

    def test_root_model_pins_anthropic_model(self, monkeypatch):
        written: list = []
        self._capture_write(monkeypatch, {}, written)
        state = {
            "workspace": WS,
            "claude_models": {"opus": "system.ai.claude-opus-4-8"},
            claude.SMART_ROUTING_STATE_KEY: True,
        }
        claude.write_tool_config(
            state, "system.ai.claude-opus-4-8", route_root_model="system.ai.claude-sonnet-5"
        )
        assert written[0]["env"]["ANTHROPIC_MODEL"] == "system.ai.claude-sonnet-5"

    def test_provider_suppresses_routing_hooks(self, monkeypatch):
        written: list = []
        self._capture_write(monkeypatch, {}, written)
        state = {"workspace": WS, claude.SMART_ROUTING_STATE_KEY: True}
        # Under a Model Provider Service no Databricks model is pinned, so routing
        # is inapplicable — hooks must not be installed even when the flag is set.
        claude.write_tool_config(state, None, provider="cat.sch.svc")
        assert "hooks" not in written[0] or "PreToolUse" not in written[0]["hooks"]

    def test_disable_removes_only_ucode_hooks(self, tmp_path, monkeypatch):
        settings_path = tmp_path / "ucode-settings.json"
        settings_path.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {
                                "matcher": "Bash",
                                "hooks": [{"type": "command", "command": "user-policy"}],
                            },
                            {
                                "matcher": "Agent|Task",
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "ucode claude-router-hook route-subagent",
                                    }
                                ],
                            },
                        ],
                        "SessionStart": [
                            {
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "ucode claude-router-hook session-start",
                                    }
                                ]
                            }
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", settings_path)
        monkeypatch.setattr(claude, "save_state", lambda state: None)
        monkeypatch.setattr(claude_routing, "clear_routing_artifacts", lambda: None)
        state = {"workspace": WS, claude.SMART_ROUTING_STATE_KEY: True}

        assert claude.disable_smart_routing(state) is True

        doc = json.loads(settings_path.read_text())
        assert state.get(claude.SMART_ROUTING_STATE_KEY) is None
        assert list(doc["hooks"]) == ["PreToolUse"]
        assert doc["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "user-policy"


class TestManagedSettingsModelOverrides:
    """Enterprise managed settings outrank ucode's --settings, so a model pinned there beats the
    one an admin published — worth pointing a developer at the file."""

    @staticmethod
    def _write(monkeypatch, tmp_path, payload):
        path = tmp_path / "managed-settings.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(claude, "_managed_settings_path", lambda: path)
        return path

    @pytest.mark.parametrize(
        "key",
        ["ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL"],
    )
    def test_reports_the_path_when_a_model_is_pinned(self, monkeypatch, tmp_path, key):
        path = self._write(monkeypatch, tmp_path, {"env": {key: "system.ai.claude-opus-5"}})
        assert claude.managed_settings_model_overrides() == path

    def test_none_for_name_companions_that_select_nothing(self, monkeypatch, tmp_path):
        # The `_NAME` keys are picker labels, so an enterprise value there overrides no model.
        self._write(monkeypatch, tmp_path, {"env": {"ANTHROPIC_DEFAULT_OPUS_MODEL_NAME": "Opus 5"}})
        assert claude.managed_settings_model_overrides() is None

    def test_none_when_no_model_keys_are_set(self, monkeypatch, tmp_path):
        self._write(monkeypatch, tmp_path, {"env": {"SOMETHING_ELSE": "1"}})
        assert claude.managed_settings_model_overrides() is None

    def test_none_when_env_block_is_absent(self, monkeypatch, tmp_path):
        self._write(monkeypatch, tmp_path, {"permissions": {}})
        assert claude.managed_settings_model_overrides() is None

    def test_none_on_platforms_without_managed_settings(self, monkeypatch):
        monkeypatch.setattr(claude, "_managed_settings_path", lambda: None)
        assert claude.managed_settings_model_overrides() is None

    def test_none_when_the_file_does_not_exist(self, monkeypatch, tmp_path):
        monkeypatch.setattr(claude, "_managed_settings_path", lambda: tmp_path / "missing.json")
        assert claude.managed_settings_model_overrides() is None
