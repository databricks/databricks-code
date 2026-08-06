"""Tests for the interactive `ucode setup` flow and its CLI wiring.

The wizard is mostly orchestration, so these focus on the parts where it can silently produce a
wrong manifest: reading tracing/MCP/skills back out of ``state.json``, classifying MCP URLs into
managed-config types, the admin gate, and the per-agent model-config shapes.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import typer.main
from typer.testing import CliRunner

import ucode.config_io as config_io_mod
import ucode.managed_setup as managed_setup_mod
import ucode.managed_wizard as wizard
from ucode.cli import app
from ucode.managed_setup import validate_manifest

runner = CliRunner()

WORKSPACE = "https://ws.example.com"

STATE = {
    "workspace": WORKSPACE,
    "claude_models": {
        "opus": "system.ai.claude-opus-4-8",
        "sonnet": "system.ai.claude-sonnet-4-6",
    },
    "codex_models": ["system.ai.gpt-5-6"],
    "gemini_models": ["system.ai.gemini-3-flash"],
    "oss_models": ["system.ai.kimi-k2-6"],
}


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path, monkeypatch):
    """Point the manifest path at a tmp dir so no test touches the real ~/.ucode."""
    monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
    monkeypatch.setattr(
        managed_setup_mod, "MANAGED_SETTINGS_PATH", tmp_path / "managed-settings.json"
    )
    monkeypatch.setattr(config_io_mod, "_dry_run", False)


class TestTracingReadback:
    def test_reads_uc_destination(self):
        state = {"tracing": {"enabled": True, "uc_destination": "main.default.ucode-traces"}}
        assert wizard._tracing_table_from_state(state) == "main.default.ucode-traces"

    def test_disabled_tracing_yields_none(self):
        state = {"tracing": {"enabled": False, "uc_destination": "main.default.t"}}
        assert wizard._tracing_table_from_state(state) is None

    def test_enabled_without_destination_yields_none(self):
        # A non-UC-backed experiment has no table to publish, so the manifest must omit tracing
        # rather than carry an empty value the server would reject.
        assert wizard._tracing_table_from_state({"tracing": {"enabled": True}}) is None

    def test_missing_tracing_yields_none(self):
        assert wizard._tracing_table_from_state({}) is None

    def test_malformed_tracing_yields_none(self):
        assert wizard._tracing_table_from_state({"tracing": "on"}) is None


class TestMcpUrlClassification:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://ws.example.com/ai-gateway/mcp-services/system.ai.github", "mcp-service"),
            ("https://ws.example.com/api/2.0/mcp/external/jira-prod", "external"),
            ("https://ws.example.com/api/2.0/mcp/genie/01ef", "genie-space"),
            ("https://ws.example.com/api/2.0/mcp/vector-search/main/default", "vector-search"),
            ("https://ws.example.com/api/2.0/mcp/functions/main/default", "uc-functions"),
            ("https://ws.example.com/api/2.0/mcp/sql", "sql"),
            ("https://mcp-myapp-123.aws.databricksapps.com/mcp", "app"),
        ],
    )
    def test_known_urls(self, url, expected):
        assert wizard._mcp_type_for_url(url) == expected

    def test_trailing_slash_is_tolerated(self):
        assert wizard._mcp_type_for_url("https://ws.example.com/api/2.0/mcp/sql/") == "sql"

    def test_unknown_url_yields_none(self):
        # Better to skip a server than publish it with a guessed type.
        assert wizard._mcp_type_for_url("https://example.com/something/else") is None

    def test_sql_is_not_confused_for_app(self):
        # Both end in a fixed segment; sql must win since it is checked first.
        assert wizard._mcp_type_for_url("https://ws.example.com/api/2.0/mcp/sql") == "sql"


class TestMcpServersFromState:
    def test_maps_registered_servers_to_name_and_type(self):
        state = {
            "mcp_servers": [
                {
                    "name": "databricks-github",
                    "url": f"{WORKSPACE}/ai-gateway/mcp-services/system.ai.github",
                },
                {"name": "databricks-sql", "url": f"{WORKSPACE}/api/2.0/mcp/sql"},
            ]
        }
        assert wizard._mcp_servers_from_state(state) == [
            {"name": "databricks-github", "type": "mcp-service"},
            {"name": "databricks-sql", "type": "sql"},
        ]

    def test_skips_the_skills_registry_entry(self):
        # Skills are published under the manifest's own `skills` field; including the MCP entry too
        # would configure them twice.
        from ucode.mcp import SKILLS_MCP_KIND

        state = {
            "mcp_servers": [
                {
                    "name": "databricks-skill-registry",
                    "kind": SKILLS_MCP_KIND,
                    "url": f"{WORKSPACE}/api/2.0/mcp/sql",
                },
                {"name": "databricks-sql", "url": f"{WORKSPACE}/api/2.0/mcp/sql"},
            ]
        }
        assert wizard._mcp_servers_from_state(state) == [{"name": "databricks-sql", "type": "sql"}]

    def test_skips_unclassifiable_servers(self):
        state = {"mcp_servers": [{"name": "mystery", "url": "https://example.com/nope"}]}
        assert wizard._mcp_servers_from_state(state) == []

    def test_skips_entries_missing_name_or_url(self):
        state = {
            "mcp_servers": [
                {"url": f"{WORKSPACE}/api/2.0/mcp/sql"},
                {"name": "no-url"},
                "not-a-dict",
            ]
        }
        assert wizard._mcp_servers_from_state(state) == []

    def test_empty_state_yields_nothing(self):
        assert wizard._mcp_servers_from_state({}) == []

    def test_output_validates_as_a_manifest(self):
        state = {
            "mcp_servers": [
                {"name": "databricks-sql", "url": f"{WORKSPACE}/api/2.0/mcp/sql"},
            ]
        }
        servers = wizard._mcp_servers_from_state(state)
        assert validate_manifest({"mcp_servers": servers}) == []


class TestAdminGate:
    def test_non_admin_is_rejected(self):
        with patch.object(wizard, "is_workspace_admin", return_value=False):
            with pytest.raises(RuntimeError, match="not an admin"):
                wizard._require_admin(WORKSPACE, "token")

    def test_admin_passes(self):
        with patch.object(wizard, "is_workspace_admin", return_value=True):
            wizard._require_admin(WORKSPACE, "token")  # must not raise

    def test_unverifiable_check_warns_and_continues(self):
        # A failed SCIM call must not block a legitimate admin — the API enforces the same rule.
        with (
            patch.object(wizard, "is_workspace_admin", return_value=None),
            patch.object(wizard, "print_warning") as warn,
        ):
            wizard._require_admin(WORKSPACE, "token")
        assert warn.called


class TestExistingConfigWarning:
    @staticmethod
    def _warn(existing: dict) -> str:
        with (
            patch.object(wizard, "get_managed_config", return_value=(existing, None)),
            patch.object(wizard, "print_warning") as warn,
        ):
            wizard._warn_on_existing_config(WORKSPACE, "token")
        assert warn.called
        return warn.call_args[0][0]

    def test_warns_that_publishing_replaces_the_whole_config(self):
        message = self._warn({"enabled_agents": {"claude": {}, "codex": {}}})
        # There is one config per workspace covering everything, so the warning says that rather
        # than reading like a per-agent notice.
        assert "one config covers every agent" in message
        assert "replaces all of it" in message
        assert "everything you want to keep" in message

    def test_warning_does_not_itemize_the_existing_config(self):
        # The message is the same whatever the config holds: an inventory doesn't change what the
        # admin should do, and `ucode setup show` prints the real thing for comparison.
        rich = self._warn(
            {
                "enabled_agents": {"claude": {}, "opencode": {}, "pi": {}},
                "mcp_servers": [{"name": "a", "type": "sql"}],
                "skills": {"names": ["main.default"]},
                "tracing_table": "main.default.traces",
                "budget_policy": {"display_name": "lillys_budget", "budget_id": "abc"},
            }
        )
        assert rich == self._warn({"enabled_agents": {}})
        for leaked in ("Claude Code", "OpenCode", "lillys_budget", "main.default"):
            assert leaked not in rich, leaked

    def test_silent_when_no_config_exists(self):
        with (
            patch.object(wizard, "get_managed_config", return_value=(None, None)),
            patch.object(wizard, "print_warning") as warn,
        ):
            wizard._warn_on_existing_config(WORKSPACE, "token")
        assert not warn.called

    def test_read_failure_is_a_note_not_a_warning(self):
        # Can't check isn't the same as "there is one"; don't imply data loss.
        with (
            patch.object(wizard, "get_managed_config", return_value=(None, "HTTP 403 Forbidden")),
            patch.object(wizard, "print_warning") as warn,
            patch.object(wizard, "print_note") as note,
        ):
            wizard._warn_on_existing_config(WORKSPACE, "token")
        assert not warn.called
        assert note.called


class TestModelPrompting:
    def test_codex_takes_a_single_model(self):
        with patch.object(wizard, "prompt_for_selection", return_value="system.ai.gpt-5-6"):
            config = wizard._prompt_models_for_agent("codex", STATE, None)
        # CodexModelConfig has no model list, so the wizard must not build one.
        assert config == {"default_model": "system.ai.gpt-5-6"}

    def test_claude_prompts_one_slot_per_family(self):
        # Claude Code selects by family alias, so each `ClaudeDefaultModels` slot gets its own
        # prompt — and each shows that family's real alternatives, not just the newest.
        candidates = {
            "opus": ["system.ai.claude-opus-5", "system.ai.claude-opus-4-8"],
            "sonnet": ["system.ai.claude-sonnet-5"],
        }
        asked: list[str] = []

        def fake_sel(prompt, options, **kwargs):
            asked.append(prompt)
            return [v for v, _ in options][0]

        with (
            patch.object(wizard, "_claude_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)

        assert config["models"] == {
            "default_opus_model": "system.ai.claude-opus-5",
            "default_sonnet_model": "system.ai.claude-sonnet-5",
        }
        assert any("opus" in p for p in asked) and any("sonnet" in p for p in asked)

    def test_claude_offers_every_version_in_a_family(self):
        candidates = {"opus": ["system.ai.claude-opus-5", "system.ai.claude-opus-4-8"]}
        offered: list[list[str]] = []

        def fake_sel(prompt, options, **kwargs):
            values = [v for v, _ in options]
            offered.append(values)
            return values[1]  # pick the older opus on purpose

        with (
            patch.object(wizard, "_claude_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)

        # Pinning a known-good older version has to be expressible.
        assert config["models"] == {"default_opus_model": "system.ai.claude-opus-4-8"}
        assert "system.ai.claude-opus-4-8" in offered[0]

    def test_claude_families_can_be_skipped(self):
        candidates = {
            "opus": ["system.ai.claude-opus-5"],
            "sonnet": ["system.ai.claude-sonnet-5"],
        }

        def fake_sel(prompt, options, **kwargs):
            values = [v for v, _ in options]
            return wizard._SKIP_FAMILY if "sonnet" in prompt else values[0]

        with (
            patch.object(wizard, "_claude_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)

        # Every slot is optional in the proto; a skipped one must simply be absent.
        assert config["models"] == {"default_opus_model": "system.ai.claude-opus-5"}

    def test_claude_overall_default_comes_from_the_filled_slots(self):
        candidates = {
            "opus": ["system.ai.claude-opus-5"],
            "sonnet": ["system.ai.claude-sonnet-5"],
        }
        prompts: list[list[str]] = []

        def fake_sel(prompt, options, **kwargs):
            values = [v for v, _ in options]
            prompts.append(values)
            return values[0]

        with (
            patch.object(wizard, "_claude_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)

        # The last prompt is the overall default, offering only what the slots hold — so it can
        # never name a model the config doesn't carry.
        assert set(prompts[-1]) == {"system.ai.claude-opus-5", "system.ai.claude-sonnet-5"}
        assert config["default_model"] in config["models"].values()

    def test_claude_single_slot_skips_the_default_prompt(self):
        candidates = {"opus": ["system.ai.claude-opus-5"]}
        calls = {"n": 0}

        def fake_sel(prompt, options, **kwargs):
            calls["n"] += 1
            return [v for v, _ in options][0]

        with (
            patch.object(wizard, "_claude_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)

        assert calls["n"] == 1  # only the opus prompt; no redundant default question
        assert config["default_model"] == "system.ai.claude-opus-5"

    def test_claude_falls_back_to_text_when_nothing_discovered(self):
        with (
            patch.object(wizard, "_claude_candidates", return_value={}),
            patch.object(wizard, "prompt_for_text", return_value="some-claude"),
            patch.object(wizard, "print_warning"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)
        assert config == {"default_model": "some-claude"}

    def test_single_slot_announces_the_inferred_default(self):
        # The one-option prompt is skipped, but silence reads as a dropped step — the admin has to
        # learn that the default is set, and to what.
        candidates = {"opus": ["system.ai.claude-opus-4-8", "system.ai.claude-opus-5"]}

        def fake_sel(prompt, options, **kwargs):
            if prompt.startswith("Default opus"):
                return "system.ai.claude-opus-4-8"
            return wizard._SKIP_FAMILY

        with (
            patch.object(wizard, "_claude_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "print_note"),
            patch.object(wizard, "print_success") as success,
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)

        assert config["default_model"] == "system.ai.claude-opus-4-8"
        assert success.called
        assert "system.ai.claude-opus-4-8" in success.call_args[0][0]

    def test_claude_all_families_skipped_still_picks_from_the_candidates(self):
        # Skipping every slot is a legitimate minimal config — `models` is optional and each unset
        # slot falls back to `default_model`, so one model covers every family. The admin shouldn't
        # have to type an id we already have.
        candidates = {
            "opus": ["system.ai.claude-opus-5", "system.ai.claude-opus-4-8"],
            "sonnet": ["system.ai.claude-sonnet-5"],
        }
        offered: list[list[str]] = []

        def fake_sel(prompt, options, **kwargs):
            values = [v for v, _ in options]
            offered.append(values)
            if prompt.startswith("Default "):
                return wizard._SKIP_FAMILY
            return "system.ai.claude-opus-4-8"

        with (
            patch.object(wizard, "_claude_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "prompt_for_text") as text,
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)

        assert config == {"default_model": "system.ai.claude-opus-4-8"}
        assert "models" not in config
        assert not text.called, "should pick from candidates, not ask for free text"
        # The final prompt offers every candidate across all families.
        assert set(offered[-1]) == {
            "system.ai.claude-opus-5",
            "system.ai.claude-opus-4-8",
            "system.ai.claude-sonnet-5",
        }

    def test_older_claude_version_passes_validation(self):
        # The picker offers every version in a family, but `claude_models` holds only the newest —
        # so validation has to learn about the rest or it rejects a legitimate pick.
        candidates = {"opus": ["system.ai.claude-opus-5", "system.ai.claude-opus-4-8"]}
        state = {
            "workspace": "https://ws.example.com",
            "claude_models": {"opus": "system.ai.claude-opus-5"},
        }

        def fake_unbucketed(workspace, token):
            return ["system.ai.claude-opus-5", "system.ai.claude-opus-4-8"], None

        with (
            patch.object(wizard, "get_databricks_token", lambda *a, **k: "tok"),
            patch.object(wizard, "discover_claude_models_unbucketed", fake_unbucketed),
            patch.object(wizard, "claude_family_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", return_value="system.ai.claude-opus-4-8"),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", state, None)

        manifest = {
            "default_agent": "claude",
            "enabled_agents": {"claude": {"model_config": config}},
        }
        assert validate_manifest(manifest, state) == []

    def test_claude_cancelled_family_prompt_aborts(self):
        with (
            patch.object(
                wizard, "_claude_candidates", return_value={"opus": ["system.ai.claude-opus-5"]}
            ),
            patch.object(wizard, "prompt_for_selection", return_value=None),
            patch.object(wizard, "print_note"),
        ):
            with pytest.raises(KeyboardInterrupt):
                wizard._prompt_models_for_agent("claude", STATE, None)

    def test_single_model_agents_get_one_prompt(self):
        # Gemini and Copilot declare `repeated string models` in the proto, but their config writers
        # take one model and write one env var — a published list would be read by nothing.
        for tool in ("codex", "gemini", "copilot"):
            options = wizard.model_options_for_agent(tool, STATE)
            with (
                patch.object(wizard, "prompt_for_selection", return_value=options[0]) as select,
                patch.object(wizard, "prompt_for_multi_selection") as multi,
            ):
                config = wizard._prompt_models_for_agent(tool, STATE, None)
            assert select.called, tool
            assert not multi.called, tool
            assert config == {"default_model": options[0]}, tool
            assert "models" not in config, tool

    def test_claude_catalog_is_fetched_once(self):
        # `configure_shared_state` already paged the whole catalog; re-fetching per claude prompt
        # pages it again for no new information.
        calls = {"n": 0}

        def fake_fetch(workspace, token):
            calls["n"] += 1
            return ["system.ai.claude-opus-5", "system.ai.claude-opus-4-8"], None

        state = {"workspace": "https://ws.example.com", "profile": "p"}
        with (
            patch.object(wizard, "discover_claude_models_unbucketed", side_effect=fake_fetch),
            patch.object(wizard, "get_databricks_token", lambda *a, **k: "tok"),
            patch.object(wizard, "prompt_for_selection", return_value="system.ai.claude-opus-5"),
            patch.object(wizard, "print_note"),
            patch.object(wizard, "print_success"),
        ):
            wizard._prompt_models_for_agent("claude", state, None)
            wizard._prompt_models_for_agent("claude", state, None)

        assert calls["n"] == 1
        assert state["all_claude_models"]

    def test_nothing_is_prechecked(self):
        # The first option is whatever discovery sorted first, not a recommendation — for pi it is a
        # Claude model, for codex the oldest GPT. Pre-checking it made "hit Enter" produce an
        # arbitrary config.
        captured: dict = {}

        def fake_multi(prompt, options, preselected=None, **kwargs):
            captured["preselected"] = preselected
            return [v for v, _ in options][:1]

        with (
            patch.object(wizard, "prompt_for_multi_selection", side_effect=fake_multi),
            patch.object(wizard, "prompt_for_selection", return_value="x"),
        ):
            wizard._prompt_models_for_agent("pi", STATE, None)

        assert not captured["preselected"]

    def test_list_agents_still_multi_select(self):
        # OpenCode and Pi really do show a model picker, so their lists are honoured.
        for tool in ("opencode", "pi"):
            options = wizard.model_options_for_agent(tool, STATE)
            with (
                patch.object(wizard, "prompt_for_multi_selection", return_value=options[:2]),
                patch.object(wizard, "prompt_for_selection", return_value=options[0]),
            ):
                config = wizard._prompt_models_for_agent(tool, STATE, None)
            assert config["models"] == options[:2], tool

    def test_flat_list_agents_keep_the_picked_list(self):
        picked = ["system.ai.claude-opus-4-8", "system.ai.kimi-k2-6"]
        with (
            patch.object(wizard, "prompt_for_multi_selection", return_value=picked),
            patch.object(wizard, "prompt_for_selection", return_value=picked[0]),
        ):
            config = wizard._prompt_models_for_agent("opencode", STATE, None)
        assert config["models"] == picked

    def test_single_pick_skips_the_default_prompt(self):
        with (
            patch.object(
                wizard, "prompt_for_multi_selection", return_value=["system.ai.claude-opus-4-8"]
            ),
            patch.object(wizard, "prompt_for_selection") as select,
        ):
            config = wizard._prompt_models_for_agent("pi", STATE, None)
        assert config["default_model"] == "system.ai.claude-opus-4-8"
        assert not select.called

    def test_provider_service_offers_its_targets(self):
        # The service's own targets are the model vocabulary the manifest must use, so the admin
        # picks from them rather than typing an id from memory.
        service = {
            "name": "main.default.anthropic-mps",
            "provider_type": "anthropic",
            "targets": ["claude-sonnet-4-6", "claude-opus-4-8"],
            "allow_all_targets": False,
        }
        with (
            patch.object(wizard, "prompt_for_selection", return_value="claude-opus-4-8") as select,
            patch.object(wizard, "prompt_for_text") as text,
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, service)
        assert config == {
            "model_provider_service": "main.default.anthropic-mps",
            "default_model": "claude-opus-4-8",
        }
        assert not text.called, "should not fall back to free text when targets are known"
        # Offered sorted, so the picker order is stable run to run.
        assert [value for value, _ in select.call_args[0][1]] == [
            "claude-opus-4-8",
            "claude-sonnet-4-6",
        ]

    def test_provider_service_falls_back_to_text_when_targets_unknown(self):
        # allow_all_targets passes the provider's whole catalog through; there is nothing to list.
        service = {
            "name": "main.default.anthropic-mps",
            "provider_type": "anthropic",
            "targets": [],
            "allow_all_targets": True,
        }
        with (
            patch.object(wizard, "prompt_for_text", return_value="claude-sonnet-5"),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, service)
        assert config == {
            "model_provider_service": "main.default.anthropic-mps",
            "default_model": "claude-sonnet-5",
        }

    def test_relayed_service_falls_back_to_text(self):
        # A relayed Anthropic subscription service routes by canonical name, with no target list.
        service = {
            "name": "main.default.lilly-anthropic",
            "provider_type": "anthropic",
            "targets": [],
            "allow_all_targets": False,
            "relayed": True,
        }
        with (
            patch.object(wizard, "prompt_for_text", return_value="claude-sonnet-4-6"),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, service)
        assert config["default_model"] == "claude-sonnet-4-6"

    def test_falls_back_to_free_text_when_nothing_discovered(self):
        with (
            patch.object(wizard, "prompt_for_text", return_value="some-model"),
            patch.object(wizard, "print_warning"),
        ):
            config = wizard._prompt_models_for_agent("pi", {}, None)
        assert config == {"default_model": "some-model"}

    def test_empty_selection_is_re_prompted(self):
        # An agent with no default_model can't be the config's default_agent (the server rejects
        # it) and gives developers nothing to launch, so "none" is re-asked rather than accepted.
        with (
            patch.object(
                wizard,
                "prompt_for_multi_selection",
                side_effect=[[], ["system.ai.claude-opus-4-8"]],
            ) as picker,
            patch.object(wizard, "print_err") as err,
        ):
            config = wizard._prompt_models_for_agent("pi", STATE, None)
        assert picker.call_count == 2
        assert err.called
        assert config["default_model"] == "system.ai.claude-opus-4-8"

    def test_every_agent_always_gets_a_default_model(self):
        # The invariant the late-validation bug violated: no agent can come back model-less.
        for tool in ("claude", "codex", "gemini", "opencode", "pi", "copilot"):
            options = wizard.model_options_for_agent(tool, STATE)
            with (
                patch.object(wizard, "prompt_for_multi_selection", return_value=[options[0]]),
                patch.object(wizard, "prompt_for_selection", return_value=options[0]),
                # Without this the claude pass reaches for the real catalog, which shells out to
                # `databricks auth token` and depends on the machine's CLI and credentials.
                patch.object(wizard, "discover_claude_models_unbucketed", return_value=([], None)),
                patch.object(wizard, "get_databricks_token", lambda *a, **k: "tok"),
                patch.object(wizard, "print_note"),
                patch.object(wizard, "print_success"),
            ):
                config = wizard._prompt_models_for_agent(tool, STATE, None)
            assert config.get("default_model"), tool

    def test_claude_candidates_survive_a_missing_databricks_cli(self):
        # `get_databricks_token` shells out, so a machine without the CLI on PATH raises
        # FileNotFoundError, not RuntimeError. That must degrade to the bucketed per-family picks
        # rather than aborting the wizard mid-flow.
        def no_cli(*args, **kwargs):
            raise FileNotFoundError(2, "No such file or directory", "databricks")

        with patch.object(wizard, "get_databricks_token", side_effect=no_cli):
            candidates = wizard._claude_candidates(dict(STATE))

        # STATE's bucketed `claude_models` still supplies one model per family.
        assert candidates["opus"] == ["system.ai.claude-opus-4-8"]
        assert candidates["sonnet"] == ["system.ai.claude-sonnet-4-6"]

    def test_default_model_is_a_bare_uc_id(self):
        # Provider prefixes (e.g. opencode's `databricks-anthropic/`) are added by each agent's own
        # writer, so the manifest stays agent-neutral.
        with (
            patch.object(
                wizard, "prompt_for_multi_selection", return_value=["system.ai.claude-opus-4-8"]
            ),
        ):
            config = wizard._prompt_models_for_agent("opencode", STATE, None)
        assert config["default_model"] == "system.ai.claude-opus-4-8"
        assert "/" not in config["default_model"]

    def test_empty_single_select_is_re_prompted(self):
        with (
            patch.object(
                wizard, "prompt_for_selection", side_effect=[None, "system.ai.gpt-5-6"]
            ) as picker,
            patch.object(wizard, "print_err"),
        ):
            config = wizard._prompt_models_for_agent("codex", STATE, None)
        assert picker.call_count == 2
        assert config == {"default_model": "system.ai.gpt-5-6"}

    def test_empty_free_text_is_re_prompted(self):
        with (
            patch.object(wizard, "prompt_for_text", side_effect=[None, "some-model"]) as text,
            patch.object(wizard, "print_err"),
            patch.object(wizard, "print_warning"),
        ):
            config = wizard._prompt_models_for_agent("pi", {}, None)
        assert text.call_count == 2
        assert config == {"default_model": "some-model"}

    def test_cancelled_picker_aborts(self):
        with patch.object(wizard, "prompt_for_multi_selection", return_value=None):
            with pytest.raises(KeyboardInterrupt):
                wizard._prompt_models_for_agent("pi", STATE, None)


ANTHROPIC_SERVICE = {
    "name": "main.default.lilly-anthropic",
    "provider_type": "anthropic",
    "targets": ["claude-sonnet-4-6"],
    "allow_all_targets": False,
    "relayed": False,
}
OPENAI_SERVICE = {
    "name": "main.default.openai-mps",
    "provider_type": "openai",
    "targets": ["gpt-5-6"],
    "allow_all_targets": False,
    "relayed": False,
}


class TestProviderServiceSpinner:
    """The MPS listing is cached per workspace, so only the first agent's lookup does any I/O."""

    SERVICES = [
        {
            "name": "main.j.ant",
            "provider_type": "anthropic",
            "targets": ["claude-opus-5"],
            "allow_all_targets": False,
            "relayed": False,
        }
    ]

    def test_spinner_shows_once_not_once_per_agent(self):
        # The reported symptom: "Checking for model provider services for <agent>..." appeared for
        # every configured agent even though the listing had already been fetched.
        spins: list[str] = []
        cached = {"yes": False}

        def fake_list(workspace, token, **kwargs):
            cached["yes"] = True
            return list(self.SERVICES), None

        def fake_spinner(message):
            spins.append(message)
            from contextlib import nullcontext

            return nullcontext()

        with (
            patch.object(wizard, "list_model_provider_services", side_effect=fake_list),
            patch.object(
                wizard, "has_cached_model_provider_services", side_effect=lambda ws: cached["yes"]
            ),
            patch.object(wizard, "spinner", side_effect=fake_spinner),
            patch.object(wizard, "prompt_for_selection", return_value="databricks"),
        ):
            wizard._select_provider_service("claude", WORKSPACE, "tok")
            wizard._select_provider_service("codex", WORKSPACE, "tok")

        listing_spins = [m for m in spins if "provider service" in m]
        assert len(listing_spins) == 1, listing_spins
        # And it doesn't name an agent, since one lookup covers them all.
        assert "Claude Code" not in listing_spins[0]


class TestProviderServiceSelection:
    def test_agents_without_provider_support_skip_the_prompt(self):
        with patch.object(wizard, "list_model_provider_services") as listing:
            assert wizard._select_provider_service("opencode", WORKSPACE, "token") is None
        assert not listing.called

    def test_feature_disabled_is_silent(self):
        # The common case on most workspaces; a warning here would be noise.
        with (
            patch.object(wizard, "list_model_provider_services", return_value=([], "HTTP 404")),
            patch.object(wizard, "is_model_provider_feature_unavailable", return_value=True),
            patch.object(wizard, "print_warning") as warn,
        ):
            assert wizard._select_provider_service("claude", WORKSPACE, "token") is None
        assert not warn.called

    def test_unexpected_listing_failure_warns(self):
        # Without this the admin silently loses the MPS option with no idea why.
        with (
            patch.object(
                wizard, "list_model_provider_services", return_value=([], "HTTP 403 Forbidden")
            ),
            patch.object(wizard, "is_model_provider_feature_unavailable", return_value=False),
            patch.object(wizard, "print_warning") as warn,
            patch.object(wizard, "print_note"),
        ):
            assert wizard._select_provider_service("claude", WORKSPACE, "token") is None
        assert warn.called
        assert "403" in warn.call_args[0][0]

    def test_services_exist_but_none_match_the_agent_explains_why(self):
        # An openai-only workspace offers claude nothing; say so rather than showing no picker.
        with (
            patch.object(
                wizard, "list_model_provider_services", return_value=([OPENAI_SERVICE], None)
            ),
            patch.object(wizard, "print_note") as note,
        ):
            assert wizard._select_provider_service("claude", WORKSPACE, "token") is None
        assert note.called
        assert "API dialect" in note.call_args[0][0]

    def test_choosing_databricks_returns_none(self):
        with (
            patch.object(
                wizard, "list_model_provider_services", return_value=([ANTHROPIC_SERVICE], None)
            ),
            patch.object(wizard, "prompt_for_selection", return_value="databricks"),
        ):
            assert wizard._select_provider_service("claude", WORKSPACE, "token") is None

    def test_choosing_mps_returns_the_whole_service(self):
        # The dict (not just the name) is returned so the model prompt can offer its targets.
        with (
            patch.object(
                wizard, "list_model_provider_services", return_value=([ANTHROPIC_SERVICE], None)
            ),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=["mps", "main.default.lilly-anthropic"],
            ),
        ):
            service = wizard._select_provider_service("claude", WORKSPACE, "token")
        assert service == ANTHROPIC_SERVICE

    def test_only_matching_services_are_offered(self):
        with (
            patch.object(
                wizard,
                "list_model_provider_services",
                return_value=([ANTHROPIC_SERVICE, OPENAI_SERVICE], None),
            ),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=["mps", "main.default.lilly-anthropic"],
            ) as select,
        ):
            wizard._select_provider_service("claude", WORKSPACE, "token")
        offered = [value for value, _ in select.call_args_list[1][0][1]]
        assert offered == ["main.default.lilly-anthropic"]

    def test_cancelling_the_service_picker_returns_none(self):
        with (
            patch.object(
                wizard, "list_model_provider_services", return_value=([ANTHROPIC_SERVICE], None)
            ),
            patch.object(wizard, "prompt_for_selection", side_effect=["mps", None]),
        ):
            assert wizard._select_provider_service("claude", WORKSPACE, "token") is None


class TestProviderServiceModelOptions:
    def test_returns_sorted_targets(self):
        service = {"targets": ["b-model", "a-model"], "allow_all_targets": False}
        assert wizard.provider_service_model_options(service) == ["a-model", "b-model"]

    def test_deduplicates(self):
        service = {"targets": ["m", "m"], "allow_all_targets": False}
        assert wizard.provider_service_model_options(service) == ["m"]

    def test_allow_all_targets_yields_nothing(self):
        service = {"targets": ["m"], "allow_all_targets": True}
        assert wizard.provider_service_model_options(service) == []

    def test_missing_targets_yields_nothing(self):
        assert wizard.provider_service_model_options({}) == []

    def test_malformed_targets_yield_nothing(self):
        assert wizard.provider_service_model_options({"targets": "m"}) == []


# Agents as the wizard configures them: the tier picker must offer these, not the workspace catalog.
CLAUDE_ONLY = {"claude": {"model_config": {"default_model": "system.ai.claude-opus-4-8"}}}


class TestBudgetPolicy:
    def test_declining_yields_none(self):
        with patch.object(wizard, "prompt_yes_no_default", return_value=False):
            assert wizard._prompt_budget_policy(WORKSPACE, "token", CLAUDE_ONLY, STATE) is None

    def test_no_budgets_warns_and_yields_none(self):
        with (
            patch.object(wizard, "prompt_yes_no_default", return_value=True),
            patch.object(wizard, "list_workspace_budgets", return_value=([], "none found")),
            patch.object(wizard, "print_warning") as warn,
        ):
            assert wizard._prompt_budget_policy(WORKSPACE, "token", CLAUDE_ONLY, STATE) is None
        assert warn.called

    def test_percentages_are_stored_as_fractions(self):
        budgets = [{"id": "budget-1", "display_name": "eng"}]
        with (
            patch.object(wizard, "prompt_yes_no_default", side_effect=[True, False]),
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=["budget-1", "claude", "system.ai.claude-opus-4-8"],
            ),
            patch.object(wizard, "prompt_for_text", return_value="tiered"),
            # prompt_for_percentage already converts; it returns the fraction.
            patch.object(wizard, "prompt_for_percentage", return_value=0.8),
        ):
            policy = wizard._prompt_budget_policy(WORKSPACE, "token", CLAUDE_ONLY, STATE)
        assert policy is not None
        assert policy["budget_id"] == "budget-1"
        assert policy["tiers"] == [
            {
                "spending_percentage": 0.8,
                "default_agent": "claude",
                "default_model": "system.ai.claude-opus-4-8",
            }
        ]

    def test_offers_only_the_models_the_agent_was_configured_with(self):
        # Pi's catalog spans every family, so offering the workspace catalog would present four
        # models it was never given — and a tier naming one of them silently misroutes developers.
        enabled = {
            "pi": {
                "model_config": {
                    "default_model": "system.ai.kimi-k2-6",
                    "models": ["system.ai.kimi-k2-6"],
                }
            }
        }
        budgets = [{"id": "budget-1", "display_name": "eng"}]
        with (
            patch.object(wizard, "prompt_yes_no_default", side_effect=[True, False]),
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=["budget-1", "pi", "system.ai.kimi-k2-6"],
            ) as select,
            patch.object(wizard, "prompt_for_text", return_value="tiered"),
            patch.object(wizard, "prompt_for_percentage", return_value=0.8),
        ):
            wizard._prompt_budget_policy(WORKSPACE, "token", enabled, STATE)
        # Third call is the model picker.
        offered = [value for value, _ in select.call_args_list[2][0][1]]
        assert offered == ["system.ai.kimi-k2-6"]

    def test_claude_family_slots_are_flattened_for_the_picker(self):
        enabled = {
            "claude": {
                "model_config": {
                    "default_model": "system.ai.claude-opus-4-8",
                    "models": {
                        "default_opus_model": "system.ai.claude-opus-4-8",
                        "default_sonnet_model": "system.ai.claude-sonnet-4-6",
                    },
                }
            }
        }
        budgets = [{"id": "budget-1", "display_name": "eng"}]
        with (
            patch.object(wizard, "prompt_yes_no_default", side_effect=[True, False]),
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=["budget-1", "claude", "system.ai.claude-opus-4-8"],
            ) as select,
            patch.object(wizard, "prompt_for_text", return_value="tiered"),
            patch.object(wizard, "prompt_for_percentage", return_value=0.8),
        ):
            wizard._prompt_budget_policy(WORKSPACE, "token", enabled, STATE)
        offered = [value for value, _ in select.call_args_list[2][0][1]]
        assert set(offered) == {"system.ai.claude-opus-4-8", "system.ai.claude-sonnet-4-6"}

    def test_falls_back_to_the_catalog_when_an_agent_lists_nothing(self):
        # An agent configured through a provider service has no enumerable list; better to offer the
        # catalog than nothing at all.
        budgets = [{"id": "budget-1", "display_name": "eng"}]
        with (
            patch.object(wizard, "prompt_yes_no_default", side_effect=[True, False]),
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=["budget-1", "gemini", "system.ai.gemini-3-flash"],
            ) as select,
            patch.object(wizard, "prompt_for_text", return_value="tiered"),
            patch.object(wizard, "prompt_for_percentage", return_value=0.8),
        ):
            wizard._prompt_budget_policy(WORKSPACE, "token", {"gemini": {}}, STATE)
        offered = [value for value, _ in select.call_args_list[2][0][1]]
        assert offered == ["system.ai.gemini-3-flash"]

    def test_authored_policy_validates(self):
        budgets = [{"id": "budget-1", "display_name": "eng"}]
        with (
            patch.object(wizard, "prompt_yes_no_default", side_effect=[True, False]),
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=["budget-1", "claude", "system.ai.claude-opus-4-8"],
            ),
            patch.object(wizard, "prompt_for_text", return_value="tiered"),
            patch.object(wizard, "prompt_for_percentage", return_value=0.8),
        ):
            policy = wizard._prompt_budget_policy(WORKSPACE, "token", CLAUDE_ONLY, STATE)
        manifest = {
            "default_agent": "claude",
            "enabled_agents": CLAUDE_ONLY,
            "budget_policy": policy,
        }
        assert validate_manifest(manifest, STATE) == []


class TestConfiguredModelsForAgent:
    def test_flat_list_plus_default(self):
        agent = {"model_config": {"default_model": "b", "models": ["a", "b"]}}
        assert wizard.configured_models_for_agent(agent) == ["a", "b"]

    def test_claude_slots_are_flattened(self):
        agent = {
            "model_config": {
                "default_model": "opus",
                "models": {"default_opus_model": "opus", "default_sonnet_model": "sonnet"},
            }
        }
        assert set(wizard.configured_models_for_agent(agent)) == {"opus", "sonnet"}

    def test_codex_has_only_a_default(self):
        # CodexModelConfig carries no model list, so the default is the whole set.
        assert wizard.configured_models_for_agent({"model_config": {"default_model": "gpt-5"}}) == [
            "gpt-5"
        ]

    def test_no_model_config_yields_nothing(self):
        assert wizard.configured_models_for_agent({}) == []


class TestSummary:
    def test_lists_claude_family_slots(self, capsys):
        # The one-line default hides which families were configured, which is most of the choice.
        manifest = {
            "default_agent": "claude",
            "enabled_agents": {
                "claude": {
                    "model_config": {
                        "default_model": "system.ai.claude-opus-4-8",
                        "models": {
                            "default_opus_model": "system.ai.claude-opus-4-8",
                            "default_haiku_model": "system.ai.claude-haiku-4-5",
                        },
                    }
                }
            },
        }
        wizard._render_summary(WORKSPACE, manifest)
        out = capsys.readouterr().out
        assert "opus" in out and "haiku" in out
        assert "system.ai.claude-haiku-4-5" in out

    def test_lists_a_multi_model_agents_models(self, capsys):
        manifest = {
            "default_agent": "pi",
            "enabled_agents": {
                "pi": {
                    "model_config": {
                        "default_model": "system.ai.kimi-k2-6",
                        "models": ["system.ai.kimi-k2-6", "system.ai.gpt-5-6"],
                    }
                }
            },
        }
        wizard._render_summary(WORKSPACE, manifest)
        assert "system.ai.gpt-5-6" in capsys.readouterr().out

    def test_single_model_agent_needs_no_extra_line(self, capsys):
        manifest = {
            "default_agent": "gemini",
            "enabled_agents": {
                "gemini": {"model_config": {"default_model": "system.ai.gemini-3-flash"}}
            },
        }
        wizard._render_summary(WORKSPACE, manifest)
        out = capsys.readouterr().out
        assert "system.ai.gemini-3-flash" in out
        assert "models:" not in out


class TestSetupFromFile:
    def _write(self, tmp_path, payload):
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _valid(self):
        return {
            "default_agent": "claude",
            "enabled_agents": {
                "claude": {"model_config": {"default_model": "system.ai.claude-opus-4-8"}}
            },
        }

    def test_valid_manifest_is_saved(self, tmp_path):
        path = self._write(tmp_path, self._valid())
        with patch.object(wizard, "load_state", return_value=STATE):
            assert wizard.setup_from_file(str(path)) == 0
        assert managed_setup_mod.load_managed_settings(WORKSPACE) == self._valid()

    def test_invalid_manifest_returns_1_and_saves_nothing(self, tmp_path):
        path = self._write(tmp_path, {"enabled_agents": {"claude": {}}})
        with patch.object(wizard, "load_state", return_value=STATE):
            assert wizard.setup_from_file(str(path)) == 1
        assert managed_setup_mod.load_managed_settings(WORKSPACE) is None

    def test_missing_file_is_actionable(self, tmp_path):
        with patch.object(wizard, "load_state", return_value=STATE):
            with pytest.raises(RuntimeError, match="Could not read manifest file"):
                wizard.setup_from_file(str(tmp_path / "nope.json"))

    def test_malformed_json_names_the_line(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{oops", encoding="utf-8")
        with patch.object(wizard, "load_state", return_value=STATE):
            with pytest.raises(RuntimeError, match="not valid JSON"):
                wizard.setup_from_file(str(path))

    def test_non_object_json_is_rejected(self, tmp_path):
        path = self._write(tmp_path, ["not", "an", "object"])
        with patch.object(wizard, "load_state", return_value=STATE):
            with pytest.raises(RuntimeError, match="must contain a JSON object"):
                wizard.setup_from_file(str(path))

    def test_unconfigured_workspace_is_actionable(self, tmp_path):
        path = self._write(tmp_path, self._valid())
        with patch.object(wizard, "load_state", return_value={}):
            with pytest.raises(RuntimeError, match="No workspace is configured"):
                wizard.setup_from_file(str(path))


class TestShowCommand:
    def test_reports_nothing_when_unauthored(self):
        with patch.object(wizard, "load_state", return_value={"workspace": WORKSPACE}):
            assert wizard.show_command() == 0

    def test_prints_the_apply_payload(self, capsys):
        manifest = {
            "default_agent": "claude",
            "enabled_agents": {
                "claude": {"model_config": {"default_model": "system.ai.claude-opus-4-8"}}
            },
        }
        managed_setup_mod.save_managed_settings(WORKSPACE, manifest)
        with patch.object(wizard, "load_state", return_value={"workspace": WORKSPACE}):
            assert wizard.show_command() == 0
        out = capsys.readouterr().out
        # The proto enum spelling is what `apply` sends, so it must appear verbatim.
        assert "CODING_AGENT_CLAUDE_CODE" in out


class TestSummaryPanel:
    def test_summary_is_boxed(self, capsys):
        # The summary is the one block an admin reads as a whole to check against what they
        # intended, and it lands after a long flow of prompts — so it gets a box rather than loose
        # lines that blend into the preceding output.
        wizard._render_summary(
            WORKSPACE,
            {
                "default_agent": "claude",
                "enabled_agents": {
                    "claude": {"model_config": {"default_model": "system.ai.claude-opus-5"}}
                },
            },
        )
        out = capsys.readouterr().out
        assert "Configuration summary" in out
        # Rich box-drawing characters: the panel border.
        assert "╭" in out and "╰" in out
        assert "system.ai.claude-opus-5" in out

    def test_a_bracketed_policy_name_survives_the_summary(self, capsys):
        # Rich reads bracketed text as a style tag and renders nothing for it, so an unescaped
        # `[prod] tiered routing` displayed as `tiered routing` — in the block whose whole purpose
        # is confirming what the admin is about to publish workspace-wide.
        wizard._render_summary(
            WORKSPACE,
            {
                "default_agent": "claude",
                "enabled_agents": {
                    "claude": {"model_config": {"default_model": "system.ai.claude-opus-5"}}
                },
                "budget_policy": {
                    "budget_id": "19165ea4-ff8d-4fbb-b6ce-fc5abe7e1c57",
                    "display_name": "[prod] tiered routing",
                    "tiers": [],
                },
            },
        )
        assert "[prod] tiered routing" in capsys.readouterr().out


class TestSearchablePickers:
    """Long lists (models, provider services, budgets) filter as you type."""

    def test_model_pickers_are_searchable(self):
        seen: list[dict] = []

        def fake_multi(prompt, options, preselected=None, **kwargs):
            seen.append(kwargs)
            return [options[0][0]]

        with patch.object(wizard, "prompt_for_multi_selection", side_effect=fake_multi):
            wizard._require_multi_selection("pick", [("a", "a"), ("b", "b")])
        assert seen[0].get("searchable") is True

    def test_single_select_pickers_are_searchable(self):
        seen: list[dict] = []

        def fake_sel(prompt, options, **kwargs):
            seen.append(kwargs)
            return options[0][0]

        with patch.object(wizard, "prompt_for_selection", side_effect=fake_sel):
            wizard._require_selection("pick", [("a", "a"), ("b", "b")])
        assert seen[0].get("searchable") is True

    def test_budget_and_tier_pickers_are_searchable(self):
        budgets = [{"id": "budget-1", "display_name": "eng"}]
        searchable_prompts: list[str] = []

        def fake_sel(prompt, options, **kwargs):
            if kwargs.get("searchable"):
                searchable_prompts.append(prompt)
            if "budget" in prompt:
                return "budget-1"
            if "agent" in prompt:
                return "claude"
            return "system.ai.claude-opus-4-8"

        with (
            patch.object(wizard, "prompt_yes_no_default", side_effect=[True, False]),
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "prompt_for_text", return_value="tiered"),
            patch.object(wizard, "prompt_for_percentage", return_value=0.8),
        ):
            wizard._prompt_budget_policy(WORKSPACE, "token", CLAUDE_ONLY, STATE)
        # Both the budget list and the tier's model list filter as you type.
        assert any("budget" in p for p in searchable_prompts), searchable_prompts
        assert any("model" in p for p in searchable_prompts), searchable_prompts


class TestCliWiring:
    def test_setup_is_registered(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "setup" in result.output

    def test_setup_help_lists_from_file(self):
        # Assert on the declared option rather than the rendered help text: Rich ellipsizes option
        # names to fit the terminal ("--fro…" below ~40 columns), and CI runners report no width, so
        # grepping `--from-file` out of the output fails there while passing on a wide local one.
        group = typer.main.get_command(app).commands["setup"]  # type: ignore[attr-defined]
        declared = {opt for param in group.params for opt in param.opts}
        assert "--from-file" in declared
        result = runner.invoke(app, ["setup", "--help"])
        assert result.exit_code == 0

    def test_setup_show_is_registered(self):
        result = runner.invoke(app, ["setup", "--help"])
        assert result.exit_code == 0
        assert "show" in result.output

    def test_successful_setup_exits_zero(self):
        # `typer.Exit` subclasses RuntimeError, so a success code must not be caught and reported
        # as an error by the command's own RuntimeError handler.
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.setup_command", return_value=0) as setup,
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        assert setup.called
        assert "ERROR" not in _out(result)

    def test_nonzero_setup_propagates(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.setup_command", return_value=1),
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 1

    def test_runtime_error_is_reported_and_exits_1(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.setup_command", side_effect=RuntimeError("you are not an admin")),
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 1
        assert "not an admin" in _out(result)

    def test_interrupt_exits_130(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.setup_command", side_effect=KeyboardInterrupt),
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 130

    def test_from_file_is_forwarded(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.setup_command", return_value=0) as setup,
        ):
            runner.invoke(app, ["setup", "--from-file", "/tmp/x.json"])
        assert setup.call_args.kwargs["from_file"] == "/tmp/x.json"

    def test_dry_run_sets_the_flag(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.setup_command", return_value=0),
            patch("ucode.cli.set_dry_run") as set_flag,
        ):
            runner.invoke(app, ["setup", "--dry-run"])
        set_flag.assert_called_once_with(True)

    def test_show_exits_zero(self):
        with patch("ucode.cli.show_command", return_value=0):
            result = runner.invoke(app, ["setup", "show"])
        assert result.exit_code == 0


def _out(result) -> str:
    """CliRunner output with stderr folded in, since print_err writes to a stderr console."""
    return result.output + (result.stderr if result.stderr_bytes else "")
