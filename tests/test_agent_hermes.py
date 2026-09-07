"""Contract tests for the Hermes agent configuration adapter."""

from __future__ import annotations

import copy
import json
import shlex
import subprocess
from pathlib import Path

import pytest

from ucode.agents import hermes
from ucode.config_io import set_dry_run
from ucode.databricks import build_auth_token_argv

WS = "https://example.databricks.com"
MODEL = "system.ai.gpt-5-6"
PROVIDER_ID = "ucode-databricks-codex"
ANTHROPIC_PROVIDER_ID = "ucode-databricks-anthropic"
OSS_PROVIDER_ID = "ucode-databricks-oss"
GEMINI_PROVIDER_ID = "ucode-databricks-gemini"


def _state(**overrides):
    state = {
        "workspace": WS,
        "profile": "team prod",
        "codex_models": ["system.ai.gpt-5", MODEL],
        "claude_models": {
            "sonnet": "system.ai.claude-sonnet-4-6",
            "haiku": "system.ai.claude-haiku-4-5",
        },
        "gemini_models": ["system.ai.gemini-3-1-pro"],
        "oss_models": [
            "system.ai.deepseek-v3-2",
            "system.ai.gpt-oss-120b",
            "system.ai.gpt-oss-20b",
        ],
    }
    state.update(overrides)
    return state


class TestHermesSpec:
    def test_is_an_externally_installed_hermes_binary(self):
        assert hermes.SPEC["binary"] == "hermes"
        assert hermes.SPEC["display"] == "Hermes"
        assert hermes.SPEC.get("install_method") == "external"


class TestRenderConfigPatch:
    def test_initial_provider_writes_require_managed_ids_to_be_absent(self):
        patch = hermes.render_config_patch(_state())

        provider_paths = {path for path in patch["set"] if path.startswith("providers.")}
        assert set(patch["expect_missing"]) == provider_paths
        assert patch["expect_hashes"] == {}

    def test_reconfiguration_requires_owned_provider_fingerprint(self):
        previous = hermes.render_config_patch(_state())
        provider_path = f"providers.{PROVIDER_ID}"
        fingerprint = hermes.config_value_fingerprint(previous["set"][provider_path])
        patch = hermes.render_config_patch(
            _state(
                managed_configs={
                    "hermes": {
                        "hermes_home": str(Path.home() / ".hermes"),
                        "provider_fingerprints": {PROVIDER_ID: fingerprint},
                    }
                }
            )
        )

        assert patch["expect_hashes"] == {provider_path: fingerprint}
        assert provider_path not in patch["expect_missing"]

    def test_reconfiguration_requires_owned_active_model_pair(self):
        owned_model = {"provider": PROVIDER_ID, "default": MODEL}
        patch = hermes.render_config_patch(
            _state(
                managed_configs={
                    "hermes": {
                        "hermes_home": str(Path.home() / ".hermes"),
                        "active_model": owned_model,
                    }
                }
            )
        )

        assert patch["expect_hashes"]["model.provider"] == hermes.config_value_fingerprint(
            owned_model["provider"]
        )
        assert patch["expect_hashes"]["model.default"] == hermes.config_value_fingerprint(
            owned_model["default"]
        )

    def test_responses_provider_uses_gateway_codex_route(self):
        patch = hermes.render_config_patch(_state())
        provider = patch["set"][f"providers.{PROVIDER_ID}"]

        assert provider["api"] == f"{WS}/ai-gateway/codex/v1"
        assert provider["transport"] == "codex_responses"

    def test_uses_current_model_selector_and_mapping(self):
        patch = hermes.render_config_patch(_state())
        provider = patch["set"][f"providers.{PROVIDER_ID}"]

        assert patch["set"]["model.provider"] == PROVIDER_ID
        assert patch["set"]["model.default"] == MODEL
        assert provider["default_model"] == MODEL
        assert provider["models"] == {
            "system.ai.gpt-5": {},
            MODEL: {},
        }

    def test_auth_helper_is_shell_escaped_command_string(self):
        patch = hermes.render_config_patch(_state())
        key_cmd = patch["set"][f"providers.{PROVIDER_ID}"]["key_cmd"]

        assert isinstance(key_cmd, str)
        argv = shlex.split(key_cmd)
        assert argv[0].endswith("ucode") or argv[0] == "ucode"
        assert argv[1] == "auth-token"
        assert "team prod" in argv

    def test_auth_helper_uses_windows_command_line_quoting(self, monkeypatch):
        monkeypatch.setattr("ucode.databricks.platform.system", lambda: "Windows")
        patch = hermes.render_config_patch(_state())
        key_cmd = patch["set"][f"providers.{PROVIDER_ID}"]["key_cmd"]

        expected = subprocess.list2cmdline(build_auth_token_argv(WS, "team prod", use_pat=False))
        assert key_cmd == expected

    def test_generates_only_wire_compatible_route_providers(self):
        patch = hermes.render_config_patch(_state())

        responses = patch["set"][f"providers.{PROVIDER_ID}"]
        anthropic = patch["set"][f"providers.{ANTHROPIC_PROVIDER_ID}"]
        oss = patch["set"][f"providers.{OSS_PROVIDER_ID}"]
        gemini = patch["set"][f"providers.{GEMINI_PROVIDER_ID}"]

        assert responses["api"] == f"{WS}/ai-gateway/codex/v1"
        assert responses["transport"] == "codex_responses"
        assert set(responses["models"]) == {"system.ai.gpt-5", MODEL}
        assert anthropic["api"] == f"{WS}/ai-gateway/anthropic"
        assert anthropic["transport"] == "anthropic_messages"
        assert set(anthropic["models"]) == {
            "system.ai.claude-sonnet-4-6",
            "system.ai.claude-haiku-4-5",
        }
        assert oss["api"] == f"{WS}/ai-gateway/mlflow/v1"
        assert oss["transport"] == "openai_chat"
        assert set(oss["models"]) == {"system.ai.deepseek-v3-2"}

        assert gemini["api"] == f"{WS}/ai-gateway/gemini/v1beta"
        assert gemini["transport"] == "gemini-native"
        assert set(gemini["models"]) == {"system.ai.gemini-3-1-pro"}

    def test_stale_overlapping_state_omits_unsupported_gpt_oss(self):
        model = "system.ai.gpt-oss-120b"
        state = _state(codex_models=["system.ai.gpt-5", model])

        patch = hermes.render_config_patch(state)

        providers = [
            patch["set"][f"providers.{provider_id}"]
            for provider_id in (
                PROVIDER_ID,
                ANTHROPIC_PROVIDER_ID,
                OSS_PROVIDER_ID,
                GEMINI_PROVIDER_ID,
            )
        ]
        model_sets = [set(provider["models"]) for provider in providers]
        for index, current in enumerate(model_sets):
            for other in model_sets[index + 1 :]:
                assert current.isdisjoint(other)
        assert all(model not in models for models in model_sets)

    def test_default_model_skips_unsupported_gpt_oss(self):
        state = _state(
            codex_default_model="system.ai.gpt-oss-120b",
            codex_models=["system.ai.gpt-oss-120b"],
            claude_models={"opus": "system.ai.claude-opus-4-8"},
            oss_models=["system.ai.gpt-oss-120b", "system.ai.deepseek-v3-2"],
        )

        assert hermes.default_model(state) == "system.ai.claude-opus-4-8"

    @pytest.mark.parametrize(
        "model",
        ["system.ai.gpt-oss-20b", "system.ai.gpt-oss-120b"],
    )
    def test_explicit_unsupported_gpt_oss_is_rejected(self, model):
        with pytest.raises(RuntimeError, match="not supported by Hermes"):
            hermes.render_config_patch(_state(), model=model)

    def test_explicit_compatible_model_selects_its_route(self):
        patch = hermes.render_config_patch(_state(), model="system.ai.claude-haiku-4-5")

        assert patch["set"]["model.provider"] == ANTHROPIC_PROVIDER_ID
        assert patch["set"]["model.default"] == "system.ai.claude-haiku-4-5"

        patch = hermes.render_config_patch(_state(), model="system.ai.deepseek-v3-2")

        assert patch["set"]["model.provider"] == OSS_PROVIDER_ID
        assert patch["set"]["model.default"] == "system.ai.deepseek-v3-2"

    def test_explicit_gemini_model_selects_native_route(self):
        patch = hermes.render_config_patch(_state(), model="system.ai.gemini-3-1-pro")

        assert patch["set"]["model.provider"] == GEMINI_PROVIDER_ID
        assert patch["set"]["model.default"] == "system.ai.gemini-3-1-pro"

    def test_rendering_is_deterministic_and_token_free(self):
        state = _state(access_token="sentinel-access-token", token="sentinel-pat")

        first = hermes.render_config_patch(state)
        second = hermes.render_config_patch(state)

        assert first == second
        encoded = json.dumps(first, sort_keys=True)
        assert "sentinel-access-token" not in encoded
        assert "sentinel-pat" not in encoded

    def test_patch_only_manages_generated_provider_and_active_model(self):
        patch = hermes.render_config_patch(_state())

        assert set(patch["set"]) == {
            f"providers.{PROVIDER_ID}",
            f"providers.{ANTHROPIC_PROVIDER_ID}",
            f"providers.{OSS_PROVIDER_ID}",
            f"providers.{GEMINI_PROVIDER_ID}",
            "model.provider",
            "model.default",
        }
        assert patch["unset"] == []

    def test_reconfigure_unsets_only_disappeared_managed_family(self):
        prior = hermes.render_config_patch(_state())
        gemini_value = prior["set"][f"providers.{GEMINI_PROVIDER_ID}"]
        fingerprint = hermes.config_value_fingerprint(gemini_value)
        patch = hermes.render_config_patch(
            _state(
                gemini_models=[],
                managed_configs={
                    "hermes": {"provider_fingerprints": {GEMINI_PROVIDER_ID: fingerprint}}
                },
            )
        )

        assert patch["unset"] == []
        assert patch["unset_if_hash"] == {f"providers.{GEMINI_PROVIDER_ID}": fingerprint}

    def test_unconfigure_is_surgical(self):
        owned = {"provider": PROVIDER_ID, "default": MODEL}
        fingerprints = dict.fromkeys(hermes.MANAGED_PROVIDER_IDS, "0" * 64)
        patch = hermes.render_unconfigure_patch(
            current_model=owned,
            owned_model=owned,
            owned_provider_fingerprints=fingerprints,
            current_provider_fingerprint="0" * 64,
        )

        assert patch["set"] == {}
        assert set(patch["unset"]) == {
            f"providers.{PROVIDER_ID}",
            "model.provider",
            "model.default",
        }
        assert set(patch["unset_if_hash"]) == set(hermes.MANAGED_PATHS[1:4])

    def test_unconfigure_preserves_user_switched_active_pair(self):
        patch = hermes.render_unconfigure_patch(
            current_model={"provider": "vertex", "default": "gemini-user"},
            owned_model={"provider": PROVIDER_ID, "default": MODEL},
            owned_provider_fingerprints=dict.fromkeys(hermes.MANAGED_PROVIDER_IDS, "0" * 64),
            current_provider_fingerprint=None,
        )

        assert "model.provider" not in patch["unset"]
        assert "model.default" not in patch["unset"]
        assert set(patch["unset_if_hash"]) == set(hermes.MANAGED_PATHS[:4])

    def test_unconfigure_preserves_changed_model_and_its_managed_provider(self):
        patch = hermes.render_unconfigure_patch(
            current_model={"provider": PROVIDER_ID, "default": "system.ai.user-choice"},
            owned_model={"provider": PROVIDER_ID, "default": MODEL},
            owned_provider_fingerprints=dict.fromkeys(hermes.MANAGED_PROVIDER_IDS, "0" * 64),
            current_provider_fingerprint="0" * 64,
        )

        assert f"providers.{PROVIDER_ID}" not in patch["unset_if_hash"]
        assert "model.provider" not in patch["unset"]
        assert "model.default" not in patch["unset"]

    def test_unconfigure_preserves_active_pair_for_same_id_user_replacement(self):
        owned = {"provider": PROVIDER_ID, "default": MODEL}
        patch = hermes.render_unconfigure_patch(
            current_model=owned,
            owned_model=owned,
            owned_provider_fingerprints={PROVIDER_ID: "a" * 64},
            current_provider_fingerprint="b" * 64,
        )

        assert patch["unset"] == []
        assert patch["expect_hashes"] == {}
        assert f"providers.{PROVIDER_ID}" not in patch["unset_if_hash"]


class TestUnconfigure:
    def test_read_active_model_preserves_partial_user_configuration(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            hermes,
            "read_config_value",
            lambda _key, *, hermes_home: {"provider": "user-provider"},
        )

        assert hermes.read_active_model(hermes_home=tmp_path) == {"provider": "user-provider"}

    def test_reads_and_applies_against_explicit_home(self, monkeypatch, tmp_path):
        owned = {"provider": PROVIDER_ID, "default": MODEL}
        provider_value = {"api": "https://owned.example/v1"}
        fingerprint = hermes.config_value_fingerprint(provider_value)
        seen = {}

        def read(*, hermes_home):
            seen["read_home"] = hermes_home
            return owned

        def apply(patch, *, hermes_home):
            seen["patch"] = patch
            seen["apply_home"] = hermes_home
            return {"status": "applied"}

        monkeypatch.setattr(hermes, "read_active_model", read)
        monkeypatch.setattr(
            hermes,
            "read_config_value",
            lambda _key, *, hermes_home: provider_value,
        )
        monkeypatch.setattr(hermes, "apply_config_patch", apply)

        hermes.unconfigure(
            hermes_home=tmp_path,
            owned_model=owned,
            owned_provider_fingerprints={PROVIDER_ID: fingerprint},
        )

        assert seen["read_home"] == tmp_path
        assert seen["apply_home"] == tmp_path
        assert "model.provider" in seen["patch"]["unset"]


class TestWriteToolConfig:
    def test_cross_home_ownership_is_not_reused(self):
        state = _state(
            gemini_models=[],
            managed_configs={
                "hermes": {
                    "keys": [],
                    "hermes_home": "/profiles/a",
                    "active_model": {"provider": PROVIDER_ID, "default": MODEL},
                    "provider_fingerprints": {GEMINI_PROVIDER_ID: "a" * 64},
                }
            },
        )

        patch = hermes.render_config_patch(
            hermes.state_scoped_to_home(state, "/profiles/b"),
            model=MODEL,
        )

        assert patch["unset_if_hash"] == {}

    def test_records_home_and_active_pair_after_success(self, monkeypatch, tmp_path):
        state = _state()
        events = []
        monkeypatch.setattr("ucode.state.save_state", lambda value: events.append(("save", value)))

        def apply(*_args, **_kwargs):
            assert events and events[0][0] == "save"
            events.append(("apply", None))

        monkeypatch.setattr(hermes, "apply_config_patch", apply)

        result = hermes.write_tool_config(state, hermes_home=tmp_path)

        ownership = result["managed_configs"]["hermes"]
        assert ownership["keys"] == []
        assert ownership["hermes_home"] == str(tmp_path.resolve())
        assert ownership["active_model"] == {
            "provider": PROVIDER_ID,
            "default": MODEL,
        }
        assert set(ownership["provider_fingerprints"]) == set(hermes.MANAGED_PROVIDER_IDS)
        assert all(len(value) == 64 for value in ownership["provider_fingerprints"].values())
        assert [event[0] for event in events] == ["save", "apply"]

    def test_failed_apply_preserves_prior_ownership(self, monkeypatch, tmp_path):
        prior = {
            "keys": [],
            "hermes_home": "/prior/home",
            "active_model": {"provider": "prior", "default": "prior-model"},
        }
        state = _state(managed_configs={"hermes": prior})

        def fail(*_args, **_kwargs):
            raise RuntimeError("apply failed")

        monkeypatch.setattr(hermes, "apply_config_patch", fail)
        saved = []
        monkeypatch.setattr(
            "ucode.state.save_state", lambda value: saved.append(copy.deepcopy(value))
        )

        with pytest.raises(RuntimeError, match="apply failed"):
            hermes.write_tool_config(state, hermes_home=tmp_path)

        assert state["managed_configs"]["hermes"] == prior
        assert saved[-1]["managed_configs"]["hermes"] == prior

    def test_partial_apply_retains_staged_ownership(self, monkeypatch, tmp_path):
        state = _state()

        def fail(*_args, **_kwargs):
            raise hermes.HermesConfigApplyError("apply may be partial")

        monkeypatch.setattr(hermes, "apply_config_patch", fail)
        saved = []
        monkeypatch.setattr(
            "ucode.state.save_state", lambda value: saved.append(copy.deepcopy(value))
        )

        with pytest.raises(hermes.HermesConfigApplyError, match="may be partial"):
            hermes.write_tool_config(state, hermes_home=tmp_path)

        ownership = saved[-1]["managed_configs"]["hermes"]
        assert ownership["hermes_home"] == str(tmp_path.resolve())
        assert ownership["provider_fingerprints"]


class TestApplyPatch:
    def test_dry_run_prints_plan_without_starting_hermes(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            hermes.subprocess,
            "run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("dry run must not start Hermes")
            ),
        )
        set_dry_run(True)
        try:
            receipt = hermes.apply_config_patch(
                {
                    "set": {"providers.managed": {"api": "https://example.invalid"}},
                    "unset": ["providers.old"],
                },
                hermes_home=tmp_path,
            )
        finally:
            set_dry_run(False)

        assert receipt == {
            "status": "dry_run",
            "paths": {"set": ["providers.managed"], "unset": ["providers.old"]},
        }
        output = capsys.readouterr().out
        assert "dry run" in output.lower()
        assert "providers.managed" in output

    def test_spawn_failure_is_not_marked_partial(self, tmp_path, monkeypatch):
        def fail_to_spawn(_argv, **_kwargs):
            raise OSError("binary unavailable")

        monkeypatch.setattr(hermes.subprocess, "run", fail_to_spawn)

        with pytest.raises(RuntimeError) as exc_info:
            hermes.apply_config_patch(
                {"set": {"providers.first": {"api": "https://first.example"}}, "unset": []},
                hermes_home=tmp_path,
            )

        assert not isinstance(exc_info.value, hermes.HermesConfigApplyError)

    def test_write_failure_after_mutation_is_marked_partial(self, tmp_path, monkeypatch):
        writes = 0

        def fake_run(argv, **_kwargs):
            nonlocal writes
            if argv[2] == "get":
                return hermes.subprocess.CompletedProcess(argv, 1, stdout="", stderr="missing")
            if argv[2] == "set":
                writes += 1
                return hermes.subprocess.CompletedProcess(
                    argv, 0 if writes == 1 else 1, stdout="", stderr="secret output"
                )
            raise AssertionError(argv)

        monkeypatch.setattr(hermes.subprocess, "run", fake_run)

        with pytest.raises(hermes.HermesConfigApplyError, match="write failed"):
            hermes.apply_config_patch(
                {
                    "set": {
                        "providers.first": {"api": "https://first.example"},
                        "providers.second": {"api": "https://second.example"},
                    },
                    "unset": [],
                },
                hermes_home=tmp_path,
            )

    def test_uses_public_commands_and_activates_model_last(self, tmp_path, monkeypatch):
        calls = []
        values = {}

        def fake_run(argv, **kwargs):
            calls.append((argv, kwargs))
            command, key = argv[2:4]
            if command == "get":
                if key not in values:
                    return hermes.subprocess.CompletedProcess(argv, 1, stdout="", stderr="missing")
                return hermes.subprocess.CompletedProcess(
                    argv, 0, stdout=json.dumps(values[key]), stderr=""
                )
            if command == "set":
                try:
                    values[key] = json.loads(argv[4])
                except json.JSONDecodeError:
                    values[key] = argv[4]
                return hermes.subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")
            if command == "unset":
                values.pop(key, None)
                return hermes.subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")
            raise AssertionError(argv)

        monkeypatch.setattr(hermes.subprocess, "run", fake_run)

        receipt = hermes.apply_config_patch(
            hermes.render_config_patch(_state()), hermes_home=tmp_path
        )

        assert receipt["status"] == "applied"
        mutating = [argv for argv, _kwargs in calls if argv[2] in {"set", "unset"}]
        assert all(argv[2] != "apply" for argv, _ in calls)
        assert [argv[3] for argv in mutating][-2:] == ["model.provider", "model.default"]
        assert [argv[4] for argv in mutating][-2:] == [PROVIDER_ID, MODEL]
        assert all(kwargs["shell"] is False for _argv, kwargs in calls)
        assert all(
            kwargs["env"]["HERMES_HOME"] == str(tmp_path.resolve()) for _argv, kwargs in calls
        )

    def test_conditional_delete_preserves_changed_value(self, tmp_path, monkeypatch):
        path = f"providers.{PROVIDER_ID}"
        replacement = {"api": "https://user.example/v1"}
        calls = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            if argv[1:3] == ["config", "get"]:
                return hermes.subprocess.CompletedProcess(
                    argv, 0, stdout=json.dumps(replacement), stderr=""
                )
            raise AssertionError("changed value must not be unset")

        monkeypatch.setattr(hermes.subprocess, "run", fake_run)

        receipt = hermes.apply_config_patch(
            {
                "set": {},
                "unset": [],
                "unset_if_hash": {path: hermes.config_value_fingerprint({"api": "owned"})},
            },
            hermes_home=tmp_path,
        )

        assert receipt["paths"]["unset"] == []
        assert calls == [["hermes", "config", "get", path, "--json"]]

    def test_conditional_delete_rechecks_value_before_unset(self, tmp_path, monkeypatch):
        path = f"providers.{PROVIDER_ID}"
        owned = {"api": "https://owned.example/v1"}
        replacement = {"api": "https://user.example/v1"}
        reads = iter((owned, replacement))

        def fake_run(argv, **kwargs):
            if argv[1:3] == ["config", "get"]:
                return hermes.subprocess.CompletedProcess(
                    argv, 0, stdout=json.dumps(next(reads)), stderr=""
                )
            raise AssertionError("replacement must not be unset")

        monkeypatch.setattr(hermes.subprocess, "run", fake_run)

        receipt = hermes.apply_config_patch(
            {
                "set": {},
                "unset": [],
                "unset_if_hash": {path: hermes.config_value_fingerprint(owned)},
            },
            hermes_home=tmp_path,
        )

        assert receipt["paths"]["unset"] == []

    def test_unconditional_delete_rechecks_expected_hash_before_unset(self, tmp_path, monkeypatch):
        path = f"providers.{PROVIDER_ID}"
        owned = {"api": "https://owned.example/v1"}
        replacement = {"api": "https://user.example/v1"}
        reads = iter((owned, replacement))

        def fake_run(argv, **_kwargs):
            if argv[1:3] == ["config", "get"]:
                return hermes.subprocess.CompletedProcess(
                    argv, 0, stdout=json.dumps(next(reads)), stderr=""
                )
            raise AssertionError("replacement must not be unset")

        monkeypatch.setattr(hermes.subprocess, "run", fake_run)

        receipt = hermes.apply_config_patch(
            {
                "set": {},
                "unset": [path],
                "expect_hashes": {path: hermes.config_value_fingerprint(owned)},
            },
            hermes_home=tmp_path,
        )

        assert receipt["paths"]["unset"] == []

    def test_expect_missing_rejects_existing_mcp_without_writing(self, tmp_path, monkeypatch):
        path = "mcp_servers.databricks-system-ai"

        def fake_run(argv, **kwargs):
            if argv[1:3] == ["config", "get"]:
                return hermes.subprocess.CompletedProcess(
                    argv, 0, stdout=json.dumps({"command": "user-server"}), stderr=""
                )
            raise AssertionError("collision must fail before mutation")

        monkeypatch.setattr(hermes.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="already exists"):
            hermes.apply_config_patch(
                {
                    "set": {path: {"command": "ucode", "args": []}},
                    "unset": [],
                    "expect_missing": [path],
                },
                hermes_home=tmp_path,
            )

    def test_failed_public_command_does_not_echo_child_output(self, tmp_path, monkeypatch):
        def fake_run(argv, **kwargs):
            return hermes.subprocess.CompletedProcess(
                argv,
                2,
                stdout="sentinel-access-token",
                stderr="sentinel-refresh-token",
            )

        monkeypatch.setattr(hermes.subprocess, "run", fake_run)

        try:
            hermes.apply_config_patch(hermes.render_config_patch(_state()), hermes_home=tmp_path)
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("failed config command must raise")

        assert "sentinel-access-token" not in message
        assert "sentinel-refresh-token" not in message


class TestManagedMcp:
    PROXY_ARGV = [
        "ucode",
        "mcp-proxy",
        "--url",
        f"{WS}/api/2.0/mcp/functions/system/ai",
        "--profile",
        "team prod",
    ]

    def test_entry_uses_stdio_proxy_and_contains_no_bearer(self):
        patch = hermes.render_mcp_server_patch("databricks-system-ai", self.PROXY_ARGV)

        assert patch == {
            "set": {
                "mcp_servers.databricks-system-ai": {
                    "command": "ucode",
                    "args": self.PROXY_ARGV[1:],
                }
            },
            "unset": [],
            "expect_missing": ["mcp_servers.databricks-system-ai"],
        }
        assert "bearer" not in json.dumps(patch).lower()

    def test_configure_scopes_apply_to_active_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
        calls = []

        def fake_apply(patch, *, hermes_home):
            calls.append((patch, hermes_home))
            return {"status": "applied"}

        monkeypatch.setattr(hermes, "apply_config_patch", fake_apply)

        hermes.write_mcp_server_config("databricks-system-ai", self.PROXY_ARGV)

        assert calls[0][1] == str(tmp_path / "profile-a")
        assert calls[0][0]["expect_missing"] == ["mcp_servers.databricks-system-ai"]

    def test_managed_update_requires_prior_fingerprint(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            hermes,
            "apply_config_patch",
            lambda patch, *, hermes_home: calls.append(patch) or {"status": "applied"},
        )

        hermes.write_mcp_server_config(
            "databricks-system-ai",
            self.PROXY_ARGV,
            hermes_home=tmp_path,
            expected_fingerprint="a" * 64,
        )

        assert calls[0]["expect_hashes"] == {"mcp_servers.databricks-system-ai": "a" * 64}

    def test_remove_only_unsets_named_managed_entry(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-b"))
        value = hermes.mcp_value_for_argv(self.PROXY_ARGV)
        calls = []
        monkeypatch.setattr(
            hermes,
            "apply_config_patch",
            lambda patch, *, hermes_home: (
                calls.append((patch, hermes_home))
                or {
                    "status": "applied",
                    "paths": {"unset": ["mcp_servers.databricks-system-ai"]},
                }
            ),
        )

        assert (
            hermes.remove_mcp_server_config(
                "databricks-system-ai",
                expected_fingerprint=hermes.mcp_server_fingerprint(value),
            )
            is True
        )
        assert calls == [
            (
                {
                    "set": {},
                    "unset": [],
                    "unset_if_hash": {
                        "mcp_servers.databricks-system-ai": hermes.mcp_server_fingerprint(value)
                    },
                },
                tmp_path / "profile-b",
            )
        ]

    def test_remove_preserves_user_replacement(self, tmp_path, monkeypatch):
        managed = hermes.mcp_value_for_argv(self.PROXY_ARGV)
        calls = []
        monkeypatch.setattr(
            hermes,
            "apply_config_patch",
            lambda patch, **kwargs: (
                calls.append(patch) or {"status": "applied", "paths": {"unset": []}}
            ),
        )

        assert (
            hermes.remove_mcp_server_config(
                "databricks-system-ai",
                hermes_home=tmp_path,
                expected_fingerprint=hermes.mcp_server_fingerprint(managed),
            )
            is False
        )
        assert calls[0]["unset_if_hash"]


class TestModelSelection:
    def test_default_model_falls_back_across_supported_families(self):
        assert (
            hermes.default_model(
                _state(
                    codex_models=[],
                    claude_models={"sonnet": "system.ai.claude-sonnet-4-6"},
                )
            )
            == "system.ai.claude-sonnet-4-6"
        )

    def test_default_model_reuses_codex_discovery_semantics(self):
        assert hermes.default_model(_state()) == MODEL
