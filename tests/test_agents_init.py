"""Tests for agents/__init__.py — registry, dispatchers, normalize_tool."""

from __future__ import annotations

import subprocess

import pytest

import ucode.agents as agents_mod
from ucode.agents import (
    DEFAULT_TOOL,
    TOOL_SPECS,
    check_gateway_endpoint,
    configure_selected_tools,
    default_model_for_tool,
    ensure_tool_binary_available,
    install_ai_tools_for_agents,
    install_tool_binary,
    normalize_tool,
    provider_permission_error,
    resolve_launch_model,
)


class TestProviderPermissionError:
    _CONN_ERR = (
        "User does not have USE CONNECTION on SCHEMA_CONNECTION "
        "'299433db-cb91-4b08-9761-edab72a27836'."
    )

    def test_rewrites_when_provider_configured(self):
        state = {"provider_services": {"codex": "main.aarushi.aarushi-test-openai"}}
        out = provider_permission_error("codex", state, self._CONN_ERR)
        assert "main.aarushi.aarushi-test-openai" in out
        assert "EXECUTE" in out
        assert "SCHEMA_CONNECTION" not in out

    def test_passthrough_without_provider(self):
        assert provider_permission_error("codex", {}, self._CONN_ERR) == self._CONN_ERR

    def test_passthrough_for_unrelated_error(self):
        state = {"provider_services": {"codex": "main.a.b"}}
        assert provider_permission_error("codex", state, "boom") == "boom"


class TestToolSpecs:
    def test_all_tools_present(self):
        assert set(TOOL_SPECS) == {"codex", "claude", "gemini", "opencode", "copilot", "pi"}

    def test_each_spec_has_required_keys(self):
        required = {"binary", "package", "display", "config_path", "backup_path"}
        for tool, spec in TOOL_SPECS.items():
            missing = required - set(spec)
            assert not missing, f"{tool} spec missing: {missing}"

    def test_default_tool_is_codex(self):
        assert DEFAULT_TOOL == "codex"

    def test_each_agent_exposes_update_check(self):
        for tool, module in agents_mod._MODULES.items():
            assert callable(module.is_update_available), f"{tool} missing is_update_available"


class TestInstallAiToolsForAgents:
    def _capture(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            agents_mod,
            "install_ai_tools",
            lambda agents, profile: captured.update(agents=agents, profile=profile),
        )
        return captured

    def test_maps_supported_tools_and_drops_others(self, monkeypatch):
        captured = self._capture(monkeypatch)
        # gemini and pi aren't supported by `databricks aitools`, so they drop.
        install_ai_tools_for_agents(["claude", "codex", "gemini", "pi"], {"profile": "prof"})
        assert captured == {"agents": ["claude-code", "codex"], "profile": "prof"}

    def test_installed_by_default(self, monkeypatch):
        # Opt-out: absent flag means install.
        captured = self._capture(monkeypatch)
        install_ai_tools_for_agents(["claude"], {"profile": "p"})
        assert captured == {"agents": ["claude-code"], "profile": "p"}

    def test_skipped_when_disabled(self, monkeypatch):
        # `configure --disable-databricks-ai-tools` persists this False.
        captured = self._capture(monkeypatch)
        install_ai_tools_for_agents(
            ["claude"], {"profile": "p", "databricks_ai_tools_enabled": False}
        )
        assert captured == {}  # install_ai_tools never called


class TestConfigureWiresAiToolsInstall:
    """Both configure chokepoints must trigger AI Tools install."""

    def _stub_configure(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(agents_mod, "configure_tool", lambda tool, state, model=None: state)
        monkeypatch.setattr(agents_mod, "save_state", lambda state: None)
        monkeypatch.setattr(
            agents_mod,
            "install_ai_tools",
            lambda agents, profile: captured.update(agents=agents, profile=profile),
        )
        return captured

    def test_configure_single_tool_triggers_install(self, monkeypatch):
        captured = self._stub_configure(monkeypatch)
        agents_mod.configure_single_tool("codex", {"codex_models": ["m"], "profile": "myprof"})
        assert captured == {"agents": ["codex"], "profile": "myprof"}

    def test_configure_selected_tools_triggers_install(self, monkeypatch):
        captured = self._stub_configure(monkeypatch)
        agents_mod.configure_selected_tools({"profile": "myprof"}, ["codex"])
        assert captured == {"agents": ["codex"], "profile": "myprof"}

    def test_configure_single_tool_respects_disable(self, monkeypatch):
        captured = self._stub_configure(monkeypatch)
        agents_mod.configure_single_tool(
            "codex",
            {"codex_models": ["m"], "profile": "myprof", "databricks_ai_tools_enabled": False},
        )
        assert captured == {}


class TestNormalizeTool:
    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("codex", "codex"),
            ("claude", "claude"),
            ("claude-code", "claude"),
            ("gemini", "gemini"),
            ("gemini-cli", "gemini"),
            ("opencode", "opencode"),
            ("copilot", "copilot"),
            ("pi", "pi"),
            ("CODEX", "codex"),
            ("  Claude  ", "claude"),
        ],
    )
    def test_known_aliases(self, alias, expected):
        assert normalize_tool(alias) == expected

    def test_unknown_raises(self):
        with pytest.raises(RuntimeError, match="Unsupported"):
            normalize_tool("unknown-agent")


class TestCheckGatewayEndpoint:
    def test_claude_available_when_models_present(self):
        assert check_gateway_endpoint({"claude_models": {"sonnet": "s4"}}, "claude") is True

    def test_claude_unavailable_when_no_models(self):
        assert check_gateway_endpoint({"claude_models": {}}, "claude") is False
        assert check_gateway_endpoint({}, "claude") is False

    def test_codex_available(self):
        assert check_gateway_endpoint({"codex_models": ["model-a"]}, "codex") is True

    def test_gemini_available(self):
        assert check_gateway_endpoint({"gemini_models": ["gemini-2"]}, "gemini") is True

    def test_opencode_available(self):
        state = {"opencode_models": {"anthropic": ["claude-sonnet"]}}
        assert check_gateway_endpoint(state, "opencode") is True

    def test_copilot_available_with_claude(self):
        assert check_gateway_endpoint({"claude_models": {"sonnet": "s4"}}, "copilot") is True

    def test_copilot_available_with_codex(self):
        assert check_gateway_endpoint({"codex_models": ["m"]}, "copilot") is True

    def test_copilot_unavailable_with_only_gemini(self):
        # Gemini is intentionally excluded from Copilot.
        assert check_gateway_endpoint({"gemini_models": ["g"]}, "copilot") is False

    def test_copilot_unavailable_when_no_models(self):
        assert check_gateway_endpoint({}, "copilot") is False

    def test_pi_available_with_claude(self):
        assert check_gateway_endpoint({"claude_models": {"sonnet": "s4"}}, "pi") is True

    def test_pi_available_with_codex(self):
        assert check_gateway_endpoint({"codex_models": ["m"]}, "pi") is True

    def test_pi_available_with_gemini(self):
        assert check_gateway_endpoint({"gemini_models": ["gemini-2"]}, "pi") is True

    def test_pi_unavailable_when_no_models(self):
        assert check_gateway_endpoint({}, "pi") is False


class TestDefaultModelForTool:
    def test_codex_returns_highest_gpt_model(self):
        models = ["databricks-gpt-5", "databricks-gpt-5-5"]
        assert default_model_for_tool("codex", {"codex_models": models}) == "databricks-gpt-5-5"

    def test_codex_returns_none_when_no_models(self):
        assert default_model_for_tool("codex", {}) is None

    def test_claude_prefers_opus(self):
        state = {"claude_models": {"sonnet": "s4", "opus": "o4", "haiku": "h4"}}
        assert default_model_for_tool("claude", state) == "o4"

    def test_claude_falls_back_to_sonnet(self):
        state = {"claude_models": {"sonnet": "s4"}}
        assert default_model_for_tool("claude", state) == "s4"

    def test_claude_falls_back_to_haiku(self):
        state = {"claude_models": {"haiku": "h4"}}
        assert default_model_for_tool("claude", state) == "h4"

    def test_claude_returns_none_when_no_models(self):
        assert default_model_for_tool("claude", {}) is None

    def test_gemini_returns_first_model(self):
        state = {"gemini_models": ["gemini-2", "gemini-1"]}
        assert default_model_for_tool("gemini", state) == "gemini-2"

    def test_gemini_returns_none_when_no_models(self):
        assert default_model_for_tool("gemini", {}) is None

    def test_opencode_prefers_anthropic(self):
        state = {"opencode_models": {"anthropic": ["claude-sonnet"], "gemini": ["gemini-2"]}}
        assert default_model_for_tool("opencode", state) == "claude-sonnet"

    def test_opencode_falls_back_to_gemini(self):
        state = {"opencode_models": {"gemini": ["gemini-2"]}}
        assert default_model_for_tool("opencode", state) == "gemini-2"

    def test_pi_prefers_claude_opus(self):
        state = {"claude_models": {"opus": "o4", "sonnet": "s4"}, "codex_models": ["c"]}
        assert default_model_for_tool("pi", state) == "o4"

    def test_pi_falls_back_to_codex(self):
        state = {"claude_models": {}, "codex_models": ["c1"]}
        assert default_model_for_tool("pi", state) == "c1"

    def test_pi_falls_back_to_gemini(self):
        state = {"claude_models": {}, "codex_models": [], "gemini_models": ["gemini-2"]}
        assert default_model_for_tool("pi", state) == "gemini-2"

    def test_pi_returns_none_when_no_models(self):
        assert default_model_for_tool("pi", {}) is None


class TestResolveLaunchModel:
    def test_codex_default_model_used_when_no_explicit(self):
        state = {"codex_models": ["databricks-gpt-5"]}
        _, model = resolve_launch_model("codex", state, None)
        assert model == "databricks-gpt-5"

    def test_explicit_model_used_when_provided(self):
        _, model = resolve_launch_model("claude", {}, "my-model")
        assert model == "my-model"

    def test_default_model_used_when_no_explicit(self):
        state = {"claude_models": {"sonnet": "s4"}}
        _, model = resolve_launch_model("claude", state, None)
        assert model == "s4"

    def test_raises_when_no_models_available(self):
        with pytest.raises(RuntimeError, match="No models available"):
            resolve_launch_model("claude", {}, None)


class TestResolveProviderModels:
    _STATE = {"workspace": "https://ws.databricks.com", "profile": None}

    def _patch(self, monkeypatch, service, error):
        monkeypatch.setattr(agents_mod, "get_databricks_token", lambda w, p: "token")
        monkeypatch.setattr(
            agents_mod, "resolve_provider_service", lambda t, n, w, tok: (service, error)
        )

    def test_none_provider_returns_none(self):
        models, error = agents_mod.resolve_provider_models("claude", self._STATE, None)
        assert (models, error) == (None, None)

    def test_anthropic_returns_no_models(self, monkeypatch):
        self._patch(monkeypatch, {"provider_type": "anthropic", "targets": []}, None)
        models, error = agents_mod.resolve_provider_models("claude", self._STATE, "main.a.svc")
        assert error is None
        assert models is None

    def test_bedrock_returns_pinned_models(self, monkeypatch):
        service = {
            "provider_type": "amazon_bedrock",
            "targets": ["us.anthropic.claude-sonnet-4-6", "global.anthropic.claude-opus-4-8"],
        }
        self._patch(monkeypatch, service, None)
        models, error = agents_mod.resolve_provider_models("claude", self._STATE, "main.b.svc")
        assert error is None
        assert models == {
            "sonnet": "us.anthropic.claude-sonnet-4-6",
            "opus": "global.anthropic.claude-opus-4-8",
        }

    def test_invalid_provider_returns_error(self, monkeypatch):
        self._patch(monkeypatch, None, "boom")
        models, error = agents_mod.resolve_provider_models("claude", self._STATE, "main.x.svc")
        assert models is None
        assert error == "boom"


class TestInstallToolBinary:
    def test_non_strict_returns_false_when_npm_missing(self, monkeypatch):
        monkeypatch.setattr("ucode.agents.shutil.which", lambda _: None)

        assert install_tool_binary("opencode", strict=False) is False

    def test_non_strict_returns_false_when_install_fails(self, monkeypatch):
        def fake_which(binary: str) -> str | None:
            if binary == "npm":
                return "/usr/bin/npm"
            return None

        def fake_run(*args, **kwargs):
            raise subprocess.CalledProcessError(1, args[0])

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)

        assert install_tool_binary("opencode", strict=False) is False

    def test_updates_existing_binary_when_requested(self, monkeypatch, capsys):
        calls: list[list[str]] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents._confirm_update_installed_tool_binary", lambda _: True)

        assert install_tool_binary("opencode", strict=False, update_existing=True) is True
        assert calls == [["npm", "install", "-g", "opencode-ai"]]
        output = capsys.readouterr().out
        assert "Updating OpenCode..." in output
        assert "OpenCode is up to date" in output

    def test_skips_existing_binary_update_when_latest_is_not_newer(self, monkeypatch, capsys):
        calls: list[list[str]] = []
        prompt_calls: list[str] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents.opencode.is_update_available", lambda: None)
        monkeypatch.setattr(
            "ucode.agents.prompt_yes_no", lambda prompt: prompt_calls.append(prompt) or True
        )

        assert install_tool_binary("opencode", strict=False, update_existing=True) is True
        assert calls == []
        assert prompt_calls == []
        assert "Updating OpenCode..." not in capsys.readouterr().out

    def test_prompts_and_updates_existing_binary_when_newer_version_exists(
        self, monkeypatch, capsys
    ):
        calls: list[list[str]] = []
        prompt_calls: list[str] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents.opencode.is_update_available", lambda: ("1.2.3", "1.2.4"))
        monkeypatch.setattr(
            "ucode.agents.prompt_yes_no", lambda prompt: prompt_calls.append(prompt) or True
        )

        assert install_tool_binary("opencode", strict=False, update_existing=True) is True
        assert prompt_calls == ["(Optional) Update OpenCode from 1.2.3 to 1.2.4?"]
        assert calls == [["npm", "install", "-g", "opencode-ai"]]
        assert "Updating OpenCode..." in capsys.readouterr().out

    def test_skips_existing_binary_update_when_user_declines(self, monkeypatch, capsys):
        calls: list[list[str]] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents._confirm_update_installed_tool_binary", lambda _: False)

        assert install_tool_binary("opencode", strict=False, update_existing=True) is True
        assert calls == []
        assert "Updating OpenCode..." not in capsys.readouterr().out

    def test_optional_update_prompt_suppressed_when_disabled(self, monkeypatch):
        """prompt_optional_updates=False must skip the optional update check
        entirely — the confirm prompt should never be reached."""

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents._minimum_version_error", lambda _: None)
        monkeypatch.setattr("ucode.agents._required_update_message", lambda _: None)

        def boom(_tool: str) -> bool:
            raise AssertionError("optional update prompt should not be reached")

        monkeypatch.setattr("ucode.agents._confirm_update_installed_tool_binary", boom)

        assert (
            install_tool_binary(
                "opencode",
                strict=False,
                update_existing=True,
                prompt_optional_updates=False,
            )
            is True
        )

    def test_required_update_runs_even_when_optional_prompt_disabled(self, monkeypatch):
        """A required (minimum-version) update is forced regardless of the
        prompt_optional_updates preference."""
        calls: list[list[str]] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents._required_update_message", lambda _: "must upgrade")
        monkeypatch.setattr("ucode.agents._minimum_version_error", lambda _: None)

        assert (
            install_tool_binary(
                "opencode",
                strict=True,
                update_existing=True,
                prompt_optional_updates=False,
            )
            is True
        )
        assert calls and calls[0][:3] == ["npm", "install", "-g"]

    def test_too_new_tool_warns_and_downgrades_on_confirm(self, monkeypatch, capsys):
        """An installed build past its supported ceiling is offered as a
        downgrade (to a pinned working version), not an upgrade."""
        calls: list[list[str]] = []
        prompt_calls: list[str] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents.gemini.too_new_downgrade", lambda: ("0.45.0", "0.44.1"))
        # The optional-update path must never be reached for a too-new tool.
        monkeypatch.setattr(
            "ucode.agents._confirm_update_installed_tool_binary",
            lambda _: (_ for _ in ()).throw(AssertionError("should not reach optional update")),
        )
        monkeypatch.setattr(
            "ucode.agents.prompt_yes_no", lambda prompt: prompt_calls.append(prompt) or True
        )

        assert install_tool_binary("gemini", strict=False, update_existing=True) is True
        assert prompt_calls == ["Downgrade Gemini CLI from 0.45.0 to 0.44.1?"]
        assert calls == [["npm", "install", "-g", "@google/gemini-cli@0.44.1"]]
        out = capsys.readouterr().out
        assert "newer than the latest version known to work" in out

    def test_too_new_tool_warns_but_keeps_version_on_decline(self, monkeypatch, capsys):
        calls: list[list[str]] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents.gemini.too_new_downgrade", lambda: ("0.45.0", "0.44.1"))
        monkeypatch.setattr("ucode.agents.prompt_yes_no", lambda prompt: False)

        assert install_tool_binary("gemini", strict=False, update_existing=True) is True
        assert calls == []
        assert "newer than the latest version known to work" in capsys.readouterr().out

    def test_too_new_tool_warns_without_prompt_when_updates_disabled(self, monkeypatch, capsys):
        """With prompts suppressed we still warn, but never downgrade."""
        calls: list[list[str]] = []

        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents.gemini.too_new_downgrade", lambda: ("0.45.0", "0.44.1"))
        monkeypatch.setattr(
            "ucode.agents.prompt_yes_no",
            lambda prompt: (_ for _ in ()).throw(AssertionError("should not prompt")),
        )

        assert (
            install_tool_binary(
                "gemini",
                strict=False,
                update_existing=True,
                prompt_optional_updates=False,
            )
            is True
        )
        assert calls == []
        assert "newer than the latest version known to work" in capsys.readouterr().out

    def test_update_failure_keeps_existing_binary_available(self, monkeypatch):
        def fake_which(binary: str) -> str | None:
            return f"/usr/bin/{binary}"

        def fake_run(*args, **kwargs):
            raise subprocess.CalledProcessError(1, args[0])

        monkeypatch.setattr("ucode.agents.shutil.which", fake_which)
        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr("ucode.agents._confirm_update_installed_tool_binary", lambda _: True)

        assert install_tool_binary("opencode", strict=True, update_existing=True) is True

    def test_ensure_tool_binary_available_raises_when_missing(self, monkeypatch):
        monkeypatch.setattr("ucode.agents.shutil.which", lambda _: None)

        with pytest.raises(RuntimeError, match="OpenCode is not installed"):
            ensure_tool_binary_available("opencode")


class TestConfigureSelectedTools:
    def test_merges_with_existing_available_tools(self, monkeypatch):
        """Configuring a new tool should not drop previously-configured tools
        from state['available_tools']."""
        monkeypatch.setattr("ucode.agents.configure_tool", lambda tool, state, model=None: state)
        monkeypatch.setattr("ucode.agents.save_state", lambda s: None)

        state = {
            "workspace": "https://x.databricks.com",
            "available_tools": ["codex", "claude"],
            "claude_models": {"sonnet": "s4"},
        }
        result = configure_selected_tools(state, ["claude"])
        assert set(result["available_tools"]) == {"codex", "claude"}

    def test_adds_new_tool_to_available_tools(self, monkeypatch):
        monkeypatch.setattr("ucode.agents.configure_tool", lambda tool, state, model=None: state)
        monkeypatch.setattr("ucode.agents.save_state", lambda s: None)

        state = {
            "workspace": "https://x.databricks.com",
            "available_tools": ["codex"],
            "claude_models": {"sonnet": "s4"},
        }
        result = configure_selected_tools(state, ["claude"])
        assert set(result["available_tools"]) == {"codex", "claude"}

    def test_empty_selection_preserves_existing(self, monkeypatch):
        monkeypatch.setattr("ucode.agents.configure_tool", lambda tool, state, model=None: state)
        monkeypatch.setattr("ucode.agents.save_state", lambda s: None)

        state = {"workspace": "https://x.databricks.com", "available_tools": ["codex"]}
        result = configure_selected_tools(state, [])
        assert result["available_tools"] == ["codex"]


class TestValidateAllToolsVerbosity:
    def _run(self, monkeypatch, capsys):
        from contextlib import nullcontext

        monkeypatch.setattr(agents_mod, "validate_tool", lambda tool: (True, ""))
        monkeypatch.setattr(agents_mod, "save_state", lambda s: None)
        monkeypatch.setattr(agents_mod, "spinner", lambda *_a, **_kw: nullcontext())
        agents_mod.validate_all_tools({"available_tools": ["codex"], "managed_configs": {}})
        return capsys.readouterr().out

    def test_normal_verbosity_renders_panels(self, monkeypatch, capsys):
        import ucode.ui as ui_mod

        monkeypatch.setattr(ui_mod, "_verbosity", "normal")
        out = self._run(monkeypatch, capsys)
        assert "Testing each tool with a quick message" in out
        assert "Ready" in out
        assert "Codex is working" in out

    def test_low_verbosity_omits_panels(self, monkeypatch, capsys):
        import ucode.ui as ui_mod

        monkeypatch.setattr(ui_mod, "_verbosity", "low")
        out = self._run(monkeypatch, capsys)
        assert "Validating..." in out
        assert "Testing each tool with a quick message" not in out
        assert "Ready" not in out
        # Per-tool success line is still printed.
        assert "Codex is working" in out


class TestValidateTool:
    def test_runs_validate_command_with_stdin_devnull(self, monkeypatch):
        # Regression guard: the validation smoke test must never inherit the
        # caller's stdin, or it hangs to the timeout when ucode is launched
        # from a non-interactive parent whose stdin is an open pipe.
        captured: dict = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr(agents_mod, "load_state", lambda: {})

        ok, err = agents_mod.validate_tool("codex")

        assert ok is True
        assert err == ""
        assert captured["kwargs"].get("stdin") is subprocess.DEVNULL

    def test_reports_timed_out_on_timeout(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 60)

        monkeypatch.setattr("ucode.agents.subprocess.run", fake_run)
        monkeypatch.setattr(agents_mod, "load_state", lambda: {})

        ok, err = agents_mod.validate_tool("codex")

        assert ok is False
        assert err == "timed out"
