"""Tests for the experimental ENABLE_SMART_ROUTING_V2 Codex launch path."""

from __future__ import annotations

import json

from ucode.agents import codex
from ucode.config_io import read_toml_safe
from ucode.smart_routing import codex_interposer

WS = "https://example.databricks.com"


class TestGenerateV2Home:
    def test_writes_provider_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(codex, "SMART_ROUTING_V2_HOME", tmp_path / "v2home")
        monkeypatch.setattr(codex, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.148.0")

        home = codex._generate_v2_app_server_home(
            {"workspace": WS, "profile": "myprof"}, "gpt-5.6-luna"
        )

        assert home == tmp_path / "v2home"
        doc = read_toml_safe(home / "config.toml")
        assert doc["model_provider"] == codex.CODEX_MODEL_PROVIDER_NAME
        assert doc["model"] == "gpt-5.6-luna"
        provider = doc["model_providers"][codex.CODEX_MODEL_PROVIDER_NAME]
        assert provider["base_url"].endswith("/ai-gateway/codex/v1")
        # Self-refreshing auth command is preserved (app-server rejects --profile).
        assert provider["auth"]["command"].endswith("ucode")
        assert "myprof" in provider["auth"]["args"]


def test_smart_routing_switch_message_is_boxed():
    message = codex._smart_routing_switch_message("model-x", "Because X.")

    assert message == (
        "┌───────────────────────────────────┐\n"
        "│ Using Unity Gateway Smart Router. │\n"
        "│ Selected Model : model-x          │\n"
        "│ Reason : Because X.               │\n"
        "└───────────────────────────────────┘"
    )


class TestInterposerSession:
    """The interposer's hold-then-switch + settings-injection logic (the novel behavior)."""

    def _turn_start(self, model: str, thread_id: str = "t1") -> str:
        return json.dumps(
            {
                "method": "turn/start",
                "id": 1,
                "params": {"threadId": thread_id, "input": [], "model": model},
            }
        )

    def test_holds_first_turn_then_switches(self):
        sess = codex_interposer._Session("gpt-5.5", after=1, log=lambda _m: None)
        # Turn 1 passes through unchanged (still on the TUI's model).
        out1 = sess.on_tui_frame(self._turn_start("system.ai.gpt-5-6-luna"))
        assert json.loads(out1)["params"]["model"] == "system.ai.gpt-5-6-luna"
        # Turn 2 is rewritten to the target.
        out2 = sess.on_tui_frame(self._turn_start("system.ai.gpt-5-6-luna"))
        assert json.loads(out2)["params"]["model"] == "gpt-5.5"

    def test_after_zero_switches_immediately(self):
        sess = codex_interposer._Session("gpt-5.5", after=0, log=lambda _m: None)
        out1 = sess.on_tui_frame(self._turn_start("luna"))
        assert json.loads(out1)["params"]["model"] == "gpt-5.5"

    def test_non_turn_frames_pass_through(self):
        sess = codex_interposer._Session("gpt-5.5", after=1, log=lambda _m: None)
        frame = json.dumps({"method": "initialize", "id": 1, "params": {}})
        assert sess.on_tui_frame(frame) == frame

    def _turn_started(self, turn_id: str, thread_id: str = "t1") -> str:
        return json.dumps(
            {"method": "turn/started", "params": {"threadId": thread_id, "turn": {"id": turn_id}}}
        )

    def test_holds_note_until_switched_turn_starts(self):
        # The note/chip-flip fire on the SWITCHED turn's turn/started (before its
        # response), never on the held turn.
        sess = codex_interposer._Session("gpt-5.5", after=1, log=lambda _m: None)
        sess.on_tui_frame(self._turn_start("luna"))  # turn 1 (held)
        assert sess.on_engine_frame(self._turn_started("turn-1")) == []  # held: no inject
        sess.on_tui_frame(self._turn_start("luna"))  # turn 2 (switched)
        injected = sess.on_engine_frame(self._turn_started("turn-2"))
        settings = next(m for m in injected if m["method"] == codex_interposer.SETTINGS_UPDATED)
        assert settings["params"]["threadId"] == "t1"
        assert settings["params"]["threadSettings"]["model"] == "gpt-5.5"

    def test_injects_switch_note_as_agent_message_when_message_set(self):
        sess = codex_interposer._Session(
            "gpt-5.5", after=1, log=lambda _m: None, switch_message="selected glm-5-2 because X"
        )
        sess.on_tui_frame(self._turn_start("luna"))  # turn 1 (held)
        sess.on_engine_frame(self._turn_started("turn-1"))
        sess.on_tui_frame(self._turn_start("luna"))  # turn 2 (switched)
        injected = sess.on_engine_frame(self._turn_started("turn-2"))
        # The note is a full agentMessage lifecycle: item/started THEN item/completed,
        # both carrying the same item (a lone item/completed renders nothing in the TUI).
        started = next(m for m in injected if m["method"] == codex_interposer.ITEM_STARTED)
        completed = next(m for m in injected if m["method"] == codex_interposer.ITEM_COMPLETED)
        assert started["params"]["turnId"] == "turn-2"
        assert completed["params"]["turnId"] == "turn-2"
        for frame in (started, completed):
            item = frame["params"]["item"]
            # An agentMessage renders as plain chat text, not a yellow warning banner.
            assert item["type"] == "agentMessage"
            assert item["text"] == "selected glm-5-2 because X"
        assert started["params"]["item"]["id"] == completed["params"]["item"]["id"]

    def test_after_zero_injects_on_first_turn_start(self):
        sess = codex_interposer._Session(
            "gpt-5.5", after=0, log=lambda _m: None, switch_message="switched"
        )
        sess.on_tui_frame(self._turn_start("luna"))  # turn 1 (switched immediately)
        injected = sess.on_engine_frame(self._turn_started("turn-1"))
        methods = [m["method"] for m in injected]
        assert codex_interposer.ITEM_STARTED in methods
        assert codex_interposer.ITEM_COMPLETED in methods

    def test_no_note_without_message(self):
        sess = codex_interposer._Session("gpt-5.5", after=1, log=lambda _m: None)
        sess.on_tui_frame(self._turn_start("luna"))
        sess.on_tui_frame(self._turn_start("luna"))
        injected = sess.on_engine_frame(self._turn_started("turn-2"))
        assert [m["method"] for m in injected] == [codex_interposer.SETTINGS_UPDATED]

    def test_injects_only_once(self):
        sess = codex_interposer._Session("gpt-5.5", after=1, log=lambda _m: None)
        sess.on_tui_frame(self._turn_start("luna"))
        sess.on_tui_frame(self._turn_start("luna"))
        assert sess.on_engine_frame(self._turn_started("turn-2"))  # switched turn: injects
        assert sess.on_engine_frame(self._turn_started("turn-3")) == []  # later turn: no re-inject
