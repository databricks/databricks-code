"""Tests for agents/pi.py."""

from __future__ import annotations

import json
from contextlib import nullcontext
from unittest.mock import patch

from ucode.agents import pi

WS = "https://example.databricks.com"


def _base_urls() -> dict[str, str]:
    # Native API per family — see agents/pi.py docstring for path conventions.
    return {
        "claude": f"{WS}/ai-gateway/anthropic",
        "openai": f"{WS}/ai-gateway/codex/v1",
        "gemini": f"{WS}/ai-gateway/gemini/v1beta",
    }


def _empty() -> dict:
    """No-models input bundle for render_overlay."""
    return {
        "claude_models": {},
        "codex_models": [],
        "gemini_models": [],
    }


def _overlay(model: str, token: str = "tok", **kwargs):
    """Wrapper to call render_overlay with sensible defaults so tests stay terse."""
    bundle = {**_empty(), **kwargs}
    return pi.render_overlay(
        model,
        token,
        _base_urls(),
        bundle["claude_models"],
        bundle["codex_models"],
        bundle["gemini_models"],
    )


class TestPiSpec:
    def test_binary(self):
        assert pi.SPEC["binary"] == "pi"

    def test_package(self):
        assert pi.SPEC["package"] == "@earendil-works/pi-coding-agent"

    def test_display(self):
        assert pi.SPEC["display"] == "Pi"

    def test_config_path_under_pi_agent_dir(self):
        assert pi.SPEC["config_path"].name == "models.json"
        assert pi.SPEC["config_path"].parent.name == "agent"
        assert pi.PI_UCODE_HOME in pi.SPEC["config_path"].parents


class TestRenderOverlayProviders:
    def test_no_providers_when_no_models(self):
        overlay, _ = _overlay("foo")
        assert "providers" not in overlay

    def test_claude_provider_uses_anthropic_messages(self):
        overlay, _ = _overlay("claude-sonnet", claude_models={"sonnet": "claude-sonnet"})
        provider = overlay["providers"]["databricks-claude"]
        assert provider["api"] == "anthropic-messages"
        assert provider["baseUrl"] == f"{WS}/ai-gateway/anthropic"

    def test_openai_provider_uses_openai_responses(self):
        overlay, _ = _overlay("gpt-5", codex_models=["gpt-5"])
        provider = overlay["providers"]["databricks-openai"]
        assert provider["api"] == "openai-responses"
        assert provider["baseUrl"] == f"{WS}/ai-gateway/codex/v1"

    def test_gemini_provider_uses_google_generative_ai(self):
        overlay, _ = _overlay("gemini-2", gemini_models=["gemini-2"])
        provider = overlay["providers"]["databricks-gemini"]
        assert provider["api"] == "google-generative-ai"
        assert provider["baseUrl"] == f"{WS}/ai-gateway/gemini/v1beta"

    def test_all_three_providers_when_all_present(self):
        overlay, _ = _overlay(
            "claude-sonnet",
            claude_models={"sonnet": "claude-sonnet"},
            codex_models=["gpt-5"],
            gemini_models=["gemini-2"],
        )
        assert set(overlay["providers"].keys()) == {
            "databricks-claude",
            "databricks-openai",
            "databricks-gemini",
        }


class TestRenderOverlayUserAgent:
    def test_user_agent_set_on_all_three_providers(self, monkeypatch):
        monkeypatch.setattr(pi, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(pi, "agent_version", lambda binary: "0.74.0")
        overlay, _ = _overlay(
            "claude-sonnet",
            claude_models={"sonnet": "claude-sonnet"},
            codex_models=["gpt-5"],
            gemini_models=["gemini-2"],
        )
        expected = "ucode/0.1.0 pi/0.74.0"
        for name in ("databricks-claude", "databricks-openai", "databricks-gemini"):
            assert overlay["providers"][name]["headers"]["User-Agent"] == expected


class TestRenderOverlayCompatFlags:
    def test_claude_disables_eager_tool_input_streaming(self):
        # Gateway's Anthropic translator rejects per-tool
        # `eager_input_streaming`; this flag makes pi send the legacy beta
        # header instead.
        overlay, _ = _overlay("claude-sonnet", claude_models={"sonnet": "claude-sonnet"})
        compat = overlay["providers"]["databricks-claude"]["compat"]
        assert compat["supportsEagerToolInputStreaming"] is False

    def test_openai_and_gemini_have_no_compat_flags(self):
        # Their gateway routes accept pi's request shape as-is.
        overlay, _ = _overlay(
            "gpt-5",
            codex_models=["gpt-5"],
            gemini_models=["gemini-2"],
        )
        assert "compat" not in overlay["providers"]["databricks-openai"]
        assert "compat" not in overlay["providers"]["databricks-gemini"]


class TestRenderOverlayAuthAndModels:
    def test_token_in_api_key(self):
        overlay, _ = _overlay(
            "claude-sonnet", token="mytoken", claude_models={"sonnet": "claude-sonnet"}
        )
        assert overlay["providers"]["databricks-claude"]["apiKey"] == "mytoken"

    def test_auth_header_flag_set_on_all_providers(self):
        overlay, _ = _overlay(
            "claude-sonnet",
            claude_models={"sonnet": "claude-sonnet"},
            codex_models=["gpt-5"],
            gemini_models=["gemini-2"],
        )
        for name in ("databricks-claude", "databricks-openai", "databricks-gemini"):
            assert overlay["providers"][name]["authHeader"] is True

    def test_claude_models_listed(self):
        claude_models = {"opus": "claude-opus", "sonnet": "claude-sonnet"}
        overlay, _ = _overlay("claude-sonnet", claude_models=claude_models)
        ids = {m["id"] for m in overlay["providers"]["databricks-claude"]["models"]}
        assert ids == {"claude-opus", "claude-sonnet"}

    def test_openai_models_listed(self):
        overlay, _ = _overlay("gpt-5", codex_models=["gpt-5", "gpt-5-mini"])
        ids = {m["id"] for m in overlay["providers"]["databricks-openai"]["models"]}
        assert ids == {"gpt-5", "gpt-5-mini"}

    def test_gemini_models_listed(self):
        overlay, _ = _overlay("gemini-2", gemini_models=["gemini-2", "gemini-2-pro"])
        ids = {m["id"] for m in overlay["providers"]["databricks-gemini"]["models"]}
        assert ids == {"gemini-2", "gemini-2-pro"}


class TestRenderOverlayManagedKeys:
    def test_managed_keys_include_model(self):
        _, keys = _overlay("foo")
        assert ["model"] in keys

    def test_managed_keys_include_each_provider_present(self):
        _, keys = _overlay(
            "claude-sonnet",
            claude_models={"sonnet": "claude-sonnet"},
            codex_models=["gpt-5"],
            gemini_models=["gemini-2"],
        )
        for name in ("databricks-claude", "databricks-openai", "databricks-gemini"):
            assert ["providers", name] in keys


class TestRenderOverlayModelSelector:
    def test_prefixes_claude_model(self):
        overlay, _ = _overlay("claude-sonnet", claude_models={"sonnet": "claude-sonnet"})
        assert overlay["model"] == "databricks-claude/claude-sonnet"

    def test_prefixes_openai_model(self):
        overlay, _ = _overlay("gpt-5", codex_models=["gpt-5"])
        assert overlay["model"] == "databricks-openai/gpt-5"

    def test_prefixes_gemini_model(self):
        overlay, _ = _overlay("gemini-2", gemini_models=["gemini-2"])
        assert overlay["model"] == "databricks-gemini/gemini-2"

    def test_preserves_already_prefixed_model(self):
        overlay, _ = _overlay(
            "databricks-claude/claude-sonnet",
            claude_models={"sonnet": "claude-sonnet"},
        )
        assert overlay["model"] == "databricks-claude/claude-sonnet"

    def test_unknown_model_passes_through_unprefixed(self):
        # Lets a user override `model` to whatever pi accepts even if we
        # didn't classify it.
        overlay, _ = _overlay("custom/whatever")
        assert overlay["model"] == "custom/whatever"


PROVIDER = "main.gateway.custom-svc"
PROVIDER_BASE_URL = f"{WS}/ai-gateway/openai/v1"


def _provider_overlay(model: str | None = None, token: str = "tok", **kwargs):
    """render_overlay under a custom Model Provider Service.

    Separate from `_overlay` so the provider-specific keyword arguments stay in
    one place.
    """
    bundle = {**_empty(), **kwargs}
    return pi.render_overlay(
        model,
        token,
        _base_urls(),
        bundle["claude_models"],
        bundle["codex_models"],
        bundle["gemini_models"],
        provider=kwargs.get("provider", PROVIDER),
        provider_models=kwargs.get("provider_models", ["deepseek-v4-flash"]),
        provider_base_url=kwargs.get("provider_base_url", PROVIDER_BASE_URL),
        context_window=kwargs.get("context_window"),
    )


class TestRenderOverlayCustomProvider:
    def _provider(self, **kwargs) -> dict:
        overlay, _ = _provider_overlay(**kwargs)
        return overlay["providers"][pi.CUSTOM_PROVIDER_NAME]

    def test_uses_openai_completions_on_the_openai_gateway_path(self):
        provider = self._provider()
        assert provider["api"] == "openai-completions"
        assert provider["baseUrl"] == PROVIDER_BASE_URL

    def test_routes_by_provider_service_header(self):
        # The header selects the service; without it the gateway can't tell which
        # provider to forward to.
        headers = self._provider()["headers"]
        assert headers["Databricks-Model-Provider-Service"] == PROVIDER
        assert headers["User-Agent"].startswith("ucode/")

    def test_compat_is_exactly_the_behavior_changing_flags(self):
        # Pinned deliberately: pi already auto-detects supportsReasoningEffort,
        # supportsUsageInStreaming and supportsStrictMode correctly for an
        # unrecognized base URL, and `thinkingFormat` has a closed enum that
        # "reasoning_effort" is not a member of. Restating either would be dead
        # config at best and invalid at worst.
        assert self._provider()["compat"] == {
            "maxTokensField": "max_tokens",
            "supportsDeveloperRole": False,
            "supportsStore": False,
        }

    def test_targets_become_models_with_conservative_context_window(self):
        assert self._provider()["models"] == [
            {
                "id": "deepseek-v4-flash",
                "contextWindow": pi.PROVIDER_CONTEXT_WINDOW,
                "maxTokens": pi.PROVIDER_MAX_OUTPUT_TOKENS,
            }
        ]

    def test_explicit_context_window_is_honored(self):
        models = self._provider(context_window=327680)["models"]
        assert models[0]["contextWindow"] == 327680
        assert models[0]["maxTokens"] == pi.PROVIDER_MAX_OUTPUT_TOKENS

    def test_max_tokens_clamped_for_a_small_window(self):
        models = self._provider(context_window=8192)["models"]
        assert models[0]["maxTokens"] == 2048

    def test_no_reasoning_claimed_without_capability_metadata(self):
        # The service exposes no capability metadata, and claiming reasoning would
        # make pi send `reasoning_effort` to a server that may reject it.
        model = self._provider()["models"][0]
        assert "reasoning" not in model
        assert "thinkingLevelMap" not in model

    def test_all_chat_targets_are_exposed(self):
        provider = self._provider(provider_models=["model-a", "model-b"])
        assert [m["id"] for m in provider["models"]] == ["model-a", "model-b"]

    def test_databricks_providers_absent_when_workspace_has_no_models(self):
        overlay, _ = _provider_overlay()
        assert set(overlay["providers"]) == {pi.CUSTOM_PROVIDER_NAME}

    def test_coexists_with_databricks_providers(self):
        overlay, _ = _provider_overlay(claude_models={"sonnet": "claude-sonnet"})
        assert set(overlay["providers"]) == {
            pi.CUSTOM_PROVIDER_NAME,
            "databricks-claude",
        }

    def test_provider_is_a_managed_key(self):
        _, keys = _provider_overlay()
        assert ["providers", pi.CUSTOM_PROVIDER_NAME] in keys

    def test_provider_name_is_always_stripped_on_write(self):
        # Membership in PROVIDER_NAMES is what removes a stale provider on a
        # later launch without --provider.
        assert pi.CUSTOM_PROVIDER_NAME in pi.PROVIDER_NAMES

    def test_omitted_without_a_routable_base_url(self):
        overlay, keys = _provider_overlay(provider_base_url=None)
        assert pi.CUSTOM_PROVIDER_NAME not in overlay.get("providers", {})
        assert ["providers", pi.CUSTOM_PROVIDER_NAME] not in keys


class TestRenderOverlayCustomProviderSelector:
    def test_defaults_to_first_target_when_no_model_resolved(self):
        # Under a provider no Databricks model is resolved, so the selector has to
        # come from the service's targets.
        overlay, _ = _provider_overlay(None, provider_models=["model-a", "model-b"])
        assert overlay["model"] == f"{pi.CUSTOM_PROVIDER_NAME}/model-a"

    def test_prefixes_a_bare_target_id(self):
        # A bare id would match pi's own built-in provider of the same name and
        # fail with "No API key found for <provider>".
        overlay, _ = _provider_overlay("deepseek-v4-flash")
        assert overlay["model"] == f"{pi.CUSTOM_PROVIDER_NAME}/deepseek-v4-flash"

    def test_preserves_already_prefixed_target(self):
        overlay, _ = _provider_overlay(f"{pi.CUSTOM_PROVIDER_NAME}/deepseek-v4-flash")
        assert overlay["model"] == f"{pi.CUSTOM_PROVIDER_NAME}/deepseek-v4-flash"


class TestPiDefaultModel:
    def test_prefers_claude_opus(self):
        state = {"claude_models": {"opus": "o4", "sonnet": "s4", "haiku": "h4"}}
        assert pi.default_model(state) == "o4"

    def test_falls_back_to_sonnet(self):
        state = {"claude_models": {"sonnet": "s4"}}
        assert pi.default_model(state) == "s4"

    def test_falls_back_to_haiku(self):
        state = {"claude_models": {"haiku": "h4"}}
        assert pi.default_model(state) == "h4"

    def test_falls_back_to_codex(self):
        state = {"claude_models": {}, "codex_models": ["gpt-5"]}
        assert pi.default_model(state) == "gpt-5"

    def test_falls_back_to_gemini(self):
        state = {"claude_models": {}, "codex_models": [], "gemini_models": ["gemini-2"]}
        assert pi.default_model(state) == "gemini-2"

    def test_returns_none_when_empty(self):
        assert pi.default_model({}) is None
        assert (
            pi.default_model({"claude_models": {}, "codex_models": [], "gemini_models": []}) is None
        )


class TestBuildRuntimeEnv:
    def test_sets_oauth_token(self):
        env = pi.build_runtime_env("tok")
        assert env["OAUTH_TOKEN"] == "tok"

    def test_sets_ucode_home(self):
        env = pi.build_runtime_env("tok")
        assert env["HOME"] == str(pi.PI_UCODE_HOME)


class TestPiValidateCmd:
    def test_starts_with_binary(self):
        cmd = pi.validate_cmd("pi")
        assert cmd[0] == "pi"

    def test_uses_print_flag(self):
        # `--print` puts pi in non-interactive mode; without it the TUI hangs on stdin.
        cmd = pi.validate_cmd("pi")
        assert "--print" in cmd

    def test_has_prompt(self):
        cmd = pi.validate_cmd("pi")
        assert len(cmd) > 2


class TestWriteToolConfig:
    def _setup(self, tmp_path, monkeypatch):
        import ucode.agents.pi as pi_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "models.json"
        backup_file = tmp_path / "pi-backup.json"
        settings_file = tmp_path / "settings.json"
        settings_backup_file = tmp_path / "pi-settings-backup.json"
        monkeypatch.setattr(pi_mod, "PI_CONFIG_PATH", config_file)
        monkeypatch.setattr(pi_mod, "PI_SETTINGS_PATH", settings_file)
        monkeypatch.setattr(pi_mod, "PI_BACKUP_PATH", backup_file)
        monkeypatch.setattr(pi_mod, "PI_SETTINGS_BACKUP_PATH", settings_backup_file)
        return pi_mod, config_file, settings_file, settings_backup_file

    def _state(self, **overrides) -> dict:
        state = {
            "workspace": WS,
            "base_urls": {"pi": _base_urls()},
            "claude_models": {"sonnet": "claude-sonnet"},
            "codex_models": [],
            "gemini_models": [],
            "managed_configs": {},
        }
        state.update(overrides)
        return state

    def test_stale_managed_providers_removed_before_merge(self, tmp_path, monkeypatch):
        pi_mod, config_file, _, _ = self._setup(tmp_path, monkeypatch)

        stale = {
            "providers": {
                "databricks-claude": {"old": True},
                "databricks-openai": {"old": True},
                "databricks-gemini": {"old": True},
                "user-provider": {"keep": True},
            }
        }
        config_file.write_text(json.dumps(stale), encoding="utf-8")

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(self._state(), "claude-sonnet", token="tok")

        written = json.loads(config_file.read_text())
        providers = written.get("providers", {})
        assert providers.get("databricks-claude") != {"old": True}
        assert "old" not in providers.get("databricks-claude", {})
        assert providers.get("user-provider") == {"keep": True}

    def test_legacy_providers_removed_on_upgrade(self, tmp_path, monkeypatch):
        """Earlier ucode versions wrote `databricks-anthropic`, `databricks-codex`,
        and `databricks-oss` providers. They must be stripped on the next write
        so users don't end up with stale entries pointing at routes that 400."""
        pi_mod, config_file, _, _ = self._setup(tmp_path, monkeypatch)

        config_file.write_text(
            json.dumps(
                {
                    "providers": {
                        "databricks-anthropic": {"api": "anthropic-messages"},
                        "databricks-codex": {"api": "openai-responses"},
                        "databricks-oss": {"api": "openai-completions"},
                    }
                }
            ),
            encoding="utf-8",
        )

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(self._state(), "claude-sonnet", token="tok")

        written_providers = json.loads(config_file.read_text()).get("providers", {})
        for legacy in ("databricks-anthropic", "databricks-codex", "databricks-oss"):
            assert legacy not in written_providers
        assert "databricks-claude" in written_providers

    def test_config_written_with_correct_model_and_token(self, tmp_path, monkeypatch):
        pi_mod, config_file, _, _ = self._setup(tmp_path, monkeypatch)

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(self._state(), "claude-sonnet", token="tok")

        written = json.loads(config_file.read_text())
        assert written["model"] == "databricks-claude/claude-sonnet"
        assert written["providers"]["databricks-claude"]["apiKey"] == "tok"

    def test_settings_pins_default_provider_and_model(self, tmp_path, monkeypatch):
        # Without this, Pi's `findInitialModel` can fall through to a built-in
        # provider when an unrelated env var (e.g. HF_TOKEN) makes one look
        # auth-configured. Pinning the default keeps Pi on our provider.
        pi_mod, _, settings_file, _ = self._setup(tmp_path, monkeypatch)

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(self._state(), "claude-sonnet", token="tok")

        settings = json.loads(settings_file.read_text())
        assert settings["defaultProvider"] == "databricks-claude"
        assert settings["defaultModel"] == "claude-sonnet"

    def test_pre_existing_settings_are_backed_up_before_first_write(self, tmp_path, monkeypatch):
        pi_mod, _, settings_file, settings_backup_file = self._setup(tmp_path, monkeypatch)

        original = '{"theme": "Default Dark", "defaultProvider": "openai"}'
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(original, encoding="utf-8")

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(self._state(), "claude-sonnet", token="tok")

        assert settings_backup_file.read_text(encoding="utf-8") == original
        # The on-disk settings still get the ucode pin applied via deep_merge.
        merged = json.loads(settings_file.read_text())
        assert merged["defaultProvider"] == "databricks-claude"
        assert merged["theme"] == "Default Dark"

    def _write_with_provider(self, pi_mod, state):
        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            return pi_mod.write_tool_config(
                state,
                None,
                token="tok",
                provider=PROVIDER,
                provider_models=["deepseek-v4-flash"],
            )

    def test_provider_write_emits_custom_provider_and_pins_settings(self, tmp_path, monkeypatch):
        pi_mod, config_file, settings_file, _ = self._setup(tmp_path, monkeypatch)
        state = self._state(claude_models={}, codex_models=[], gemini_models=[])

        self._write_with_provider(pi_mod, state)

        providers = json.loads(config_file.read_text())["providers"]
        assert pi_mod.CUSTOM_PROVIDER_NAME in providers
        settings = json.loads(settings_file.read_text())
        assert settings["defaultProvider"] == pi_mod.CUSTOM_PROVIDER_NAME
        assert settings["defaultModel"] == "deepseek-v4-flash"

    def test_provider_write_persists_state_for_the_refresh_thread(self, tmp_path, monkeypatch):
        pi_mod, _, _, _ = self._setup(tmp_path, monkeypatch)
        state = self._state()

        new_state, _ = self._write_with_provider(pi_mod, state)

        assert new_state["pi_provider"] == PROVIDER
        assert new_state["pi_provider_models"] == ["deepseek-v4-flash"]

    def test_context_window_override_flows_from_state(self, tmp_path, monkeypatch):
        pi_mod, config_file, _, _ = self._setup(tmp_path, monkeypatch)
        state = self._state(provider_context_window=327680)

        self._write_with_provider(pi_mod, state)

        providers = json.loads(config_file.read_text())["providers"]
        models = providers[pi_mod.CUSTOM_PROVIDER_NAME]["models"]
        assert models[0]["contextWindow"] == 327680

    def test_later_launch_without_provider_clears_it(self, tmp_path, monkeypatch):
        """A stale custom provider would keep routing to the old service's header,
        so a plain `ucode pi` has to remove it and clear the persisted state."""
        pi_mod, config_file, _, _ = self._setup(tmp_path, monkeypatch)
        state = self._state()
        state, _ = self._write_with_provider(pi_mod, state)
        # A hand-added provider must survive; only ucode's own are managed.
        written = json.loads(config_file.read_text())
        written["providers"]["user-provider"] = {"keep": True}
        config_file.write_text(json.dumps(written), encoding="utf-8")

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            state, _ = pi_mod.write_tool_config(state, "claude-sonnet", token="tok")

        providers = json.loads(config_file.read_text())["providers"]
        assert pi_mod.CUSTOM_PROVIDER_NAME not in providers
        assert providers["user-provider"] == {"keep": True}
        assert "pi_provider" not in state
        assert "pi_provider_models" not in state


class TestRefreshTokenOnce:
    def _state(self, **overrides) -> dict:
        state = {
            "workspace": WS,
            "base_urls": {"pi": _base_urls()},
            "claude_models": {},
            "codex_models": [],
            "gemini_models": [],
            "managed_configs": {},
        }
        state.update(overrides)
        return state

    def test_succeeds_under_a_provider_with_no_databricks_models(self, tmp_path, monkeypatch):
        """The regression this guards: requiring a Databricks model here raised
        RuntimeError, `_refresh_forever` swallowed it, and a provider-only session
        silently stopped refreshing until the token expired mid-session."""
        import ucode.agents.pi as pi_mod

        monkeypatch.setattr(pi_mod, "PI_CONFIG_PATH", tmp_path / "models.json")
        monkeypatch.setattr(pi_mod, "PI_SETTINGS_PATH", tmp_path / "settings.json")
        monkeypatch.setattr(pi_mod, "PI_BACKUP_PATH", tmp_path / "backup.json")
        monkeypatch.setattr(pi_mod, "PI_SETTINGS_BACKUP_PATH", tmp_path / "s-backup.json")
        state = self._state(pi_provider=PROVIDER, pi_provider_models=["deepseek-v4-flash"])

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="fresh-tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            assert pi_mod._refresh_token_once(state) == "fresh-tok"

    def test_raises_without_a_model_or_a_provider(self):
        import pytest

        import ucode.agents.pi as pi_mod

        with pytest.raises(RuntimeError, match="No Pi model is available"):
            pi_mod._refresh_token_once(self._state())


class TestValidateAllToolsPiRollback:
    def test_failed_pi_validation_rolls_back_settings(self, tmp_path, monkeypatch):
        import ucode.agents as agents_mod
        import ucode.agents.pi as pi_mod

        settings_file = tmp_path / "settings.json"
        settings_file.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(pi_mod, "PI_SETTINGS_PATH", settings_file)
        monkeypatch.setattr(pi_mod, "PI_SETTINGS_BACKUP_PATH", tmp_path / "settings.backup.json")
        # Keep the generic models.json rollback off the user's real config dir.
        monkeypatch.setitem(agents_mod.TOOL_SPECS["pi"], "config_path", tmp_path / "models.json")
        monkeypatch.setitem(
            agents_mod.TOOL_SPECS["pi"], "backup_path", tmp_path / "models.backup.json"
        )
        monkeypatch.setattr(agents_mod, "validate_tool", lambda tool: (False, "boom"))
        monkeypatch.setattr(agents_mod, "save_state", lambda s: None)
        monkeypatch.setattr(agents_mod, "spinner", lambda *_a, **_kw: nullcontext())

        agents_mod.validate_all_tools({"available_tools": ["pi"], "managed_configs": {"pi": True}})

        assert not settings_file.exists()
