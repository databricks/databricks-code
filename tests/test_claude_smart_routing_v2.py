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
from ucode.smart_routing import claude_hooks, claude_pty, v2


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
    def test_renders_boxed_router_notice(self):
        model = "system.ai.claude-sonnet-4-6[1m]"
        reason = "Low complexity, unclear intent, and no code reference."
        result = claude_pty.first_prompt_hook_output({"action": "block", "model": model})

        assert result == {"decision": "block", "reason": v2._switch_message(model, reason)}
        assert claude_pty.switch_message(model, reason) == v2._switch_message(model, reason)

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
    def test_restores_model_captured_immediately_before_switch(self, tmp_path, monkeypatch):
        ucode_settings = tmp_path / "ucode-settings.json"
        user_settings = tmp_path / "settings.json"
        ucode_settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://gw"}}))
        user_settings.write_text(json.dumps({"model": "opus", "theme": "dark"}))
        monkeypatch.setattr(claude, "APP_DIR", tmp_path)
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", ucode_settings)
        monkeypatch.setattr(claude, "CLAUDE_USER_SETTINGS_PATH", user_settings)
        monkeypatch.setattr(v2, "APP_DIR", tmp_path)
        monkeypatch.setattr(v2, "CLAUDE_PTY_LOG", tmp_path / "v2.log")
        monkeypatch.setattr(v2, "build_auth_token_argv", lambda *_args, **_kwargs: ["ucode"])
        captured: dict = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            generated = Path(argv[argv.index("--settings") + 1])
            captured["settings"] = json.loads(generated.read_text())
            # A user-selected model before the first prompt becomes the value to restore.
            user_settings.write_text(json.dumps({"model": "haiku", "theme": "dark"}))
            kwargs["prepare_model_switch"]()
            # Simulate `/model` changing the user file, plus an unrelated concurrent edit.
            user_settings.write_text(json.dumps({"model": "routed", "theme": "light", "new": True}))
            kwargs["restore_model_setting"]()
            captured["restored_during_run"] = json.loads(user_settings.read_text())
            # A later user choice must survive session exit.
            user_settings.write_text(
                json.dumps({"model": "user-selected", "theme": "light", "new": True})
            )
            return 0

        monkeypatch.setattr(claude_pty, "run_claude_pty", fake_run)
        with pytest.raises(SystemExit) as exc:
            v2.launch_claude(
                {"workspace": "https://example.com"},
                ["--debug"],
                binary="claude",
                user_settings_path=user_settings,
                launch_model="opus",
                compose_settings=claude._compose_v2_settings,
                launch_model_args=claude._launch_model_args,
            )

        assert exc.value.code == 0
        assert captured["argv"][-3:] == ["--model", "opus", "--debug"]
        assert claude_hooks.FIRST_PROMPT_SOCKET_ENV in captured["settings"]["env"]
        assert "modelPicker" not in captured["settings"]
        assert captured["restored_during_run"] == {
            "model": "haiku",
            "theme": "light",
            "new": True,
        }
        assert json.loads(user_settings.read_text()) == {
            "model": "user-selected",
            "theme": "light",
            "new": True,
        }

    def test_does_not_restore_when_wrapper_never_switches(self, tmp_path, monkeypatch):
        user_settings = tmp_path / "settings.json"
        user_settings.write_text(json.dumps({"model": "opus"}))
        monkeypatch.setattr(v2, "APP_DIR", tmp_path)
        monkeypatch.setattr(v2, "build_auth_token_argv", lambda *_args, **_kwargs: ["ucode"])

        def fake_run(_argv, **_kwargs):
            user_settings.write_text(json.dumps({"model": "user-selected"}))
            return 0

        monkeypatch.setattr(claude_pty, "run_claude_pty", fake_run)
        with pytest.raises(SystemExit):
            v2.launch_claude(
                {"workspace": "https://example.com"},
                [],
                binary="claude",
                user_settings_path=user_settings,
                launch_model="opus",
                compose_settings=lambda _args: ({}, []),
                launch_model_args=claude._launch_model_args,
            )

        assert json.loads(user_settings.read_text()) == {"model": "user-selected"}

    def test_restores_after_routed_model_persists_and_preserves_later_choice(
        self, tmp_path, monkeypatch
    ):
        user_settings = tmp_path / "settings.json"
        user_settings.write_text(json.dumps({"model": "haiku", "theme": "dark"}))
        monkeypatch.setattr(v2, "APP_DIR", tmp_path)
        monkeypatch.setattr(v2, "build_auth_token_argv", lambda *_args, **_kwargs: ["ucode"])

        def fake_run(_argv, **kwargs):
            kwargs["prepare_model_switch"]()
            user_settings.write_text(json.dumps({"model": v2.CLAUDE_TARGET_MODEL, "theme": "dark"}))
            assert kwargs["model_switch_persisted"]() is True
            kwargs["restore_model_setting"]()
            assert json.loads(user_settings.read_text()) == {"model": "haiku", "theme": "dark"}
            # The same model chosen explicitly later in the session must survive.
            user_settings.write_text(json.dumps({"model": v2.CLAUDE_TARGET_MODEL, "theme": "dark"}))
            return 0

        monkeypatch.setattr(claude_pty, "run_claude_pty", fake_run)
        with pytest.raises(SystemExit):
            v2.launch_claude(
                {"workspace": "https://example.com"},
                [],
                binary="claude",
                user_settings_path=user_settings,
                launch_model=None,
                compose_settings=lambda _args: ({}, []),
                launch_model_args=claude._launch_model_args,
            )

        assert json.loads(user_settings.read_text()) == {
            "model": v2.CLAUDE_TARGET_MODEL,
            "theme": "dark",
        }

    def test_model_switch_lock_serializes_routed_sessions(self, tmp_path, monkeypatch):
        user_settings = tmp_path / "settings.json"
        user_settings.write_text(json.dumps({"model": "haiku"}))
        monkeypatch.setattr(v2, "APP_DIR", tmp_path)
        first = v2._ClaudeModelSettingGuard(user_settings)
        second = v2._ClaudeModelSettingGuard(user_settings)
        captured: list[str] = []

        first.begin()
        user_settings.write_text(json.dumps({"model": v2.CLAUDE_TARGET_MODEL}))

        def run_second() -> None:
            second.begin()
            captured.append(json.loads(user_settings.read_text())["model"])
            second.restore()

        thread = threading.Thread(target=run_second)
        thread.start()
        first.restore()
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert captured == ["haiku"]
        assert json.loads(user_settings.read_text()) == {"model": "haiku"}


class TestPtyFlow:
    def test_does_not_launch_when_socket_startup_fails(self, tmp_path, monkeypatch):
        class StoppedThread:
            @staticmethod
            def is_alive():
                return False

        monkeypatch.setattr(
            claude_pty,
            "serve_first_prompt_socket",
            lambda *_args, **_kwargs: StoppedThread(),
        )
        monkeypatch.setattr(
            claude_pty.pty,
            "fork",
            lambda: pytest.fail("Claude must not launch without the routing socket"),
        )

        with pytest.raises(RuntimeError, match="Claude was not launched"):
            claude_pty.run_claude_pty(
                ["claude"],
                route_prompt=lambda _prompt: "sonnet",
                switch_message="router selected sonnet",
                socket_path=tmp_path / "missing.sock",
            )

    def test_direct_switch_restore_and_replay(self, tmp_path):
        fake_claude = tmp_path / "fake_claude.py"
        capture = tmp_path / "capture.json"
        restored = tmp_path / "restored"
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
restored_path = Path(sys.argv[3])
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
    "restored_before_replay": restored_path.exists(),
}))
""".lstrip()
        )
        result = claude_pty.run_claude_pty(
            [
                sys.executable,
                str(fake_claude),
                str(socket_path),
                str(capture),
                str(restored),
            ],
            route_prompt=lambda _prompt: "system.ai.claude-sonnet-5",
            switch_message="router selected sonnet",
            socket_path=socket_path,
            restore_model_setting=lambda: restored.write_text("restored"),
        )

        assert result == 0
        assert json.loads(capture.read_text()) == {
            "command": "/model system.ai.claude-sonnet-5\r",
            "replayed": "\x1b[200~fix\nthe parser\x1b[201~\r",
            "restored_before_replay": True,
        }
