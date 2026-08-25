from __future__ import annotations

import json

import pytest

from ucode.agents import codex
from ucode.config_io import read_toml_safe
from ucode.smart_routing import codex_interposer, v2

WS = "https://example.databricks.com"


class TestGenerateV2Home:
    def test_writes_provider_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(v2, "CODEX_APP_SERVER_HOME", tmp_path / "v2home")
        monkeypatch.setattr(codex, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.148.0")

        home = v2._generate_codex_app_server_home(
            {"workspace": WS, "profile": "myprof"},
            "gpt-5.6-luna",
            codex.render_overlay,
        )

        assert home == tmp_path / "v2home"
        doc = read_toml_safe(home / "config.toml")
        assert doc["model_provider"] == codex.CODEX_MODEL_PROVIDER_NAME
        assert doc["model"] == "gpt-5.6-luna"
        provider = doc["model_providers"][codex.CODEX_MODEL_PROVIDER_NAME]
        assert provider["base_url"].endswith("/ai-gateway/codex/v1")
        assert provider["auth"]["command"].endswith("ucode")
        assert "myprof" in provider["auth"]["args"]


def test_smart_routing_switch_message_is_boxed():
    message = v2._switch_message("model-x", "Because X.")

    assert message == (
        "┌───────────────────────────────────┐\n"
        "│ Using Unity Gateway Smart Router. │\n"
        "│ Selected Model : model-x          │\n"
        "│ Reason : Because X.               │\n"
        "└───────────────────────────────────┘"
    )


class TestLaunchCodex:
    def test_owns_app_server_interposer_and_tui_lifecycle(self, tmp_path, monkeypatch):
        processes = []
        interposer_args = {}
        stopped = []

        class FakeProcess:
            def __init__(self, argv, **kwargs):
                self.argv = argv
                self.kwargs = kwargs
                self.terminated = False
                processes.append(self)

            def wait(self, timeout=None):
                return 0 if timeout is not None else 7

            def terminate(self):
                self.terminated = True

            def kill(self):
                raise AssertionError("clean shutdown should not need kill")

            def send_signal(self, _signal):
                raise AssertionError("test does not interrupt the TUI")

        monkeypatch.setattr(v2.subprocess, "Popen", FakeProcess)
        monkeypatch.setattr(v2, "get_databricks_token", lambda workspace, profile: "token")
        monkeypatch.setattr(
            v2,
            "_generate_codex_app_server_home",
            lambda state, model, render_overlay: tmp_path,
        )
        monkeypatch.setattr(v2, "_free_port", lambda: 41001)
        monkeypatch.setattr(v2, "_wait_for_app_server", lambda port, timeout: True)

        def start_interposer(*args, **kwargs):
            interposer_args["args"] = args
            interposer_args["kwargs"] = kwargs
            return 41002, lambda: stopped.append(True)

        monkeypatch.setattr(codex_interposer, "start_interposer_thread", start_interposer)

        with pytest.raises(SystemExit) as exc:
            v2.launch_codex(
                {"workspace": WS, "profile": "myprof"},
                ["--search"],
                binary="codex",
                start_model="gpt-start",
                render_overlay=codex.render_overlay,
            )

        assert exc.value.code == 7
        assert processes[0].argv == [
            "codex",
            "app-server",
            "--listen",
            "ws://127.0.0.1:41001",
        ]
        assert processes[0].kwargs["env"][v2.OAUTH_TOKEN_ENV_VAR] == "token"
        assert processes[0].kwargs["env"][v2.CODEX_HOME_ENV_VAR] == str(tmp_path)
        assert processes[1].argv == [
            "codex",
            "--remote",
            "ws://127.0.0.1:41002",
            "--model",
            "gpt-start",
            "--search",
        ]
        assert interposer_args["args"] == (
            v2.LOOPBACK_HOST,
            "ws://127.0.0.1:41001",
            v2.CODEX_TARGET_MODEL,
        )
        assert "Using Unity Gateway Smart Router." in interposer_args["kwargs"]["switch_message"]
        assert stopped == [True]
        assert processes[0].terminated is True


def test_interposer_startup_failure_is_propagated(monkeypatch):
    async def fail_to_serve(*args, **kwargs):
        raise OSError("bind failed")

    monkeypatch.setattr(codex_interposer, "_serve", fail_to_serve)

    with pytest.raises(RuntimeError, match="failed to start") as exc:
        codex_interposer.start_interposer_thread(
            v2.LOOPBACK_HOST,
            "ws://127.0.0.1:41001",
            "model-x",
        )

    assert isinstance(exc.value.__cause__, OSError)


class TestInterposerSession:
    def _turn_start(self, model: str, thread_id: str = "t1") -> str:
        return json.dumps(
            {
                "method": codex_interposer.TURN_START,
                "id": 1,
                "params": {"threadId": thread_id, "input": [], "model": model},
            }
        )

    def test_switches_first_turn(self):
        sess = codex_interposer._Session("gpt-5.5", log=lambda _m: None)
        output = sess.on_tui_frame(self._turn_start("system.ai.gpt-5-6-luna"))
        assert json.loads(output)["params"]["model"] == "gpt-5.5"

    def test_does_not_schedule_notification_when_model_is_already_selected(self):
        sess = codex_interposer._Session("gpt-5.5", log=lambda _m: None)
        frame = self._turn_start("gpt-5.5")
        assert sess.on_tui_frame(frame) == frame
        assert sess.on_engine_frame(self._turn_started("turn-1")) == []

    def test_non_turn_frames_pass_through(self):
        sess = codex_interposer._Session("gpt-5.5", log=lambda _m: None)
        frame = json.dumps({"method": "initialize", "id": 1, "params": {}})
        assert sess.on_tui_frame(frame) == frame

    def _turn_started(self, turn_id: str, thread_id: str = "t1") -> str:
        return json.dumps(
            {
                "method": codex_interposer.TURN_STARTED,
                "params": {"threadId": thread_id, "turn": {"id": turn_id}},
            }
        )

    def test_injects_note_when_switched_turn_starts(self):
        sess = codex_interposer._Session("gpt-5.5", log=lambda _m: None)
        sess.on_tui_frame(self._turn_start("luna"))
        injected = sess.on_engine_frame(self._turn_started("turn-1"))
        settings = next(m for m in injected if m["method"] == codex_interposer.SETTINGS_UPDATED)
        assert settings["params"]["threadId"] == "t1"
        assert settings["params"]["threadSettings"]["model"] == "gpt-5.5"

    def test_injects_switch_note_as_agent_message_when_message_set(self):
        sess = codex_interposer._Session(
            "gpt-5.5", log=lambda _m: None, switch_message="selected glm-5-2 because X"
        )
        sess.on_tui_frame(self._turn_start("luna"))
        injected = sess.on_engine_frame(self._turn_started("turn-1"))
        started = next(m for m in injected if m["method"] == codex_interposer.ITEM_STARTED)
        completed = next(m for m in injected if m["method"] == codex_interposer.ITEM_COMPLETED)
        assert started["params"]["turnId"] == "turn-1"
        assert completed["params"]["turnId"] == "turn-1"
        for frame in (started, completed):
            item = frame["params"]["item"]
            assert item["type"] == "agentMessage"
            assert item["text"] == "selected glm-5-2 because X"
        assert started["params"]["item"]["id"] == completed["params"]["item"]["id"]

    def test_no_note_without_message(self):
        sess = codex_interposer._Session("gpt-5.5", log=lambda _m: None)
        sess.on_tui_frame(self._turn_start("luna"))
        injected = sess.on_engine_frame(self._turn_started("turn-1"))
        assert [m["method"] for m in injected] == [codex_interposer.SETTINGS_UPDATED]

    def test_injects_only_once(self):
        sess = codex_interposer._Session("gpt-5.5", log=lambda _m: None)
        sess.on_tui_frame(self._turn_start("luna"))
        assert sess.on_engine_frame(self._turn_started("turn-1"))
        sess.on_tui_frame(self._turn_start("luna"))
        assert sess.on_engine_frame(self._turn_started("turn-2")) == []
