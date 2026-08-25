"""Tests for Claude's experimental first-prompt PTY routing path."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from ucode.agents import claude
from ucode.smart_routing import claude_hooks, claude_pty


class TestDirectModelCommand:
    @pytest.mark.parametrize(
        "name",
        ["system.ai.claude-opus-4-8[1m]", "databricks-claude-sonnet-5", "opus"],
    )
    def test_accepts_model_names(self, name):
        assert claude_pty.valid_model_name(name)

    @pytest.mark.parametrize("name", ["", "a b", "a\nb", "x" * 201, None])
    def test_rejects_unsafe_model_names(self, name):
        assert not claude_pty.valid_model_name(name)

    def test_types_direct_model_command(self):
        read_fd, write_fd = os.pipe()
        try:
            claude_pty.inject_model_switch(write_fd, "system.ai.claude-sonnet-5")
            assert os.read(read_fd, 200) == b"/model system.ai.claude-sonnet-5\r"
        finally:
            os.close(read_fd)
            os.close(write_fd)


class TestFirstPromptHook:
    def test_blocks_once_then_allows_replay(self, tmp_path):
        socket_path = tmp_path / "first.sock"
        blocked: list[tuple[str, str]] = []
        stop = threading.Event()
        claude_pty.serve_first_prompt_socket(
            socket_path,
            lambda _prompt: "sonnet",
            lambda prompt, model: blocked.append((prompt, model)),
            stop,
        )
        try:
            deadline = time.monotonic() + 5
            while not socket_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            first = claude_pty.request_first_prompt_route(
                socket_path, {"session_id": "s1", "prompt": "fix the parser"}
            )
            replay = claude_pty.request_first_prompt_route(
                socket_path, {"session_id": "s1", "prompt": "fix the parser"}
            )
            assert first == {"action": "block", "model": "sonnet"}
            assert replay == {"action": "allow"}
            assert blocked == [("fix the parser", "sonnet")]
        finally:
            stop.set()

    def test_first_prompt_hook_is_per_launch(self):
        settings = {"hooks": {"PreToolUse": [{"hooks": [{"command": "user-policy"}]}]}}
        claude_hooks.sync_first_prompt_hook(settings, "/bin/ucode")
        claude_hooks.sync_first_prompt_hook(settings, "/bin/ucode")
        command = settings["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
        assert command == "/bin/ucode claude-router-hook route-first-prompt"
        assert len(settings["hooks"]["UserPromptSubmit"]) == 1
        assert "user-policy" in str(settings["hooks"]["PreToolUse"])


class TestV2Launch:
    def test_saves_default_passes_model_and_restores_only_model(self, tmp_path, monkeypatch):
        ucode_settings = tmp_path / "ucode-settings.json"
        user_settings = tmp_path / "settings.json"
        snapshot_path = tmp_path / "model-snapshot.json"
        ucode_settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://gw"}}))
        user_settings.write_text(json.dumps({"model": "opus", "theme": "dark"}))
        monkeypatch.setattr(claude, "APP_DIR", tmp_path)
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", ucode_settings)
        monkeypatch.setattr(claude, "CLAUDE_USER_SETTINGS_PATH", user_settings)
        monkeypatch.setattr(claude, "CLAUDE_MODEL_SNAPSHOT_PATH", snapshot_path)
        monkeypatch.setattr(claude, "SMART_ROUTING_V2_CLAUDE_LOG", tmp_path / "v2.log")
        monkeypatch.setattr(claude, "get_databricks_token", lambda *_args, **_kwargs: "token")
        monkeypatch.setattr(claude, "build_auth_token_argv", lambda *_args, **_kwargs: ["ucode"])
        captured: dict = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            generated = Path(argv[argv.index("--settings") + 1])
            captured["settings"] = json.loads(generated.read_text())
            # Simulate `/model` changing the user file, plus an unrelated concurrent edit.
            user_settings.write_text(json.dumps({"model": "routed", "theme": "light", "new": True}))
            return 0

        monkeypatch.setattr(claude_pty, "run_claude_pty", fake_run)
        snapshot = claude._snapshot_user_model_setting()
        with pytest.raises(SystemExit) as exc:
            claude._launch_smart_routing_v2(
                {"workspace": "https://example.com"},
                ["--debug"],
                model_snapshot=snapshot,
                launch_model="opus",
            )

        assert exc.value.code == 0
        assert captured["argv"][-3:] == ["--model", "opus", "--debug"]
        assert claude_hooks.FIRST_PROMPT_SOCKET_ENV in captured["settings"]["env"]
        assert "modelPicker" not in captured["settings"]
        assert json.loads(user_settings.read_text()) == {
            "model": "opus",
            "theme": "light",
            "new": True,
        }
        assert not snapshot_path.exists()
        assert not list(tmp_path.glob("claude-default-model.*.snapshot.json"))


class TestPtyFlow:
    def test_direct_switch_restore_and_replay(self, tmp_path):
        fake_claude = tmp_path / "fake_claude.py"
        capture = tmp_path / "capture.json"
        socket_path = tmp_path / "first.sock"
        fake_claude.write_text(
            """
import json
import os
import socket
import sys
import tty
from pathlib import Path

socket_path = sys.argv[1]
capture_path = Path(sys.argv[2])
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect(socket_path)
client.sendall((json.dumps({
    "method": "route_first_prompt",
    "prompt": "fix\\nthe parser",
    "session_id": "s1",
}) + "\\n").encode())
response = client.makefile("rb").readline()
client.close()
assert json.loads(response)["action"] == "block"
print("Smart Routing blocked the prompt", flush=True)
tty.setraw(0)

def read_until(suffix):
    data = b""
    while not data.endswith(suffix):
        data += os.read(0, 1)
    return data

model_command = read_until(b"\\r")
print("Set model to system.ai.claude-sonnet-5", flush=True)
replayed = read_until(b"\\x1b[201~\\r")
capture_path.write_text(json.dumps({
    "command": model_command.decode(),
    "replayed": replayed.decode(),
}))
""".lstrip()
        )
        result = claude_pty.run_claude_pty(
            [sys.executable, str(fake_claude), str(socket_path), str(capture)],
            route_prompt=lambda _prompt: "system.ai.claude-sonnet-5",
            switch_message="router selected sonnet",
            socket_path=socket_path,
        )

        assert result == 0
        assert json.loads(capture.read_text()) == {
            "command": "/model system.ai.claude-sonnet-5\r",
            "replayed": "\x1b[200~fix\nthe parser\x1b[201~\r",
        }
