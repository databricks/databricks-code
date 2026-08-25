from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

from ucode.config_io import (
    APP_DIR,
    deep_merge_dict,
    read_json_safe,
    read_toml_safe,
    write_json_file,
    write_toml_file,
)
from ucode.databricks import build_auth_token_argv, get_databricks_token
from ucode.smart_routing import claude_pty, codex_interposer
from ucode.smart_routing.claude_hooks import FIRST_PROMPT_SOCKET_ENV, sync_first_prompt_hook
from ucode.ui import print_note

ENV_VAR = "ENABLE_SMART_ROUTING_V2"

CODEX_TARGET_MODEL = "system.ai.glm-5-2"  # TODO(lilly): replace with smart router.
CODEX_APP_SERVER_HOME = APP_DIR / "codex-v2-home"
CODEX_INTERPOSER_LOG = APP_DIR / "codex-v2-interposer.log"
CODEX_SWITCH_REASON = "Low complexity, unclear intent, and no code reference."  # TODO(lilly): replace with smart router rationale.

CLAUDE_TARGET_MODEL = "system.ai.claude-sonnet-4-6[1m]"  # TODO(lilly): replace with smart router.
CLAUDE_PTY_LOG = APP_DIR / "claude-v2-pty.log"
CLAUDE_MODEL_SNAPSHOT_PATH = APP_DIR / "claude-default-model.snapshot.json"

APP_SERVER_READY_TIMEOUT_SECONDS = 30
PROCESS_SHUTDOWN_TIMEOUT_SECONDS = 5
OAUTH_TOKEN_ENV_VAR = "OAUTH_TOKEN"
CODEX_HOME_ENV_VAR = "CODEX_HOME"
LOOPBACK_HOST = "127.0.0.1"
HEALTH_REQUEST_TIMEOUT_SECONDS = 1
HEALTH_POLL_INTERVAL_SECONDS = 0.25


def enabled() -> bool:
    return os.environ.get(ENV_VAR) == "1"


def snapshot_claude_model_setting(user_settings_path: Path) -> dict:
    """Capture only Claude's user-level ``model`` setting."""
    settings = read_json_safe(user_settings_path)
    return {"present": "model" in settings, "value": settings.get("model")}


def _save_claude_model_snapshot(snapshot: dict, snapshot_path: Path) -> None:
    """Journal the pre-switch model so the next launch can recover after a crash."""
    write_json_file(snapshot_path, snapshot)


def restore_claude_model_snapshot(
    user_settings_path: Path,
    snapshot_path: Path | None = None,
    snapshot: dict | None = None,
) -> bool:
    """Restore only the journaled ``model`` field, preserving every sibling setting."""
    snapshot_path = snapshot_path or CLAUDE_MODEL_SNAPSHOT_PATH
    if snapshot is None and not snapshot_path.exists():
        return False
    snapshot = snapshot if snapshot is not None else read_json_safe(snapshot_path)
    present = snapshot.get("present")
    if not isinstance(present, bool):
        raise RuntimeError(f"Claude model recovery snapshot is invalid: {snapshot_path}")
    settings = read_json_safe(user_settings_path)
    if present:
        settings["model"] = snapshot.get("value")
    else:
        settings.pop("model", None)
    write_json_file(user_settings_path, settings)
    snapshot_path.unlink(missing_ok=True)
    return True


def recover_claude_model_snapshots(user_settings_path: Path) -> None:
    """Repair defaults left by interrupted Claude PTY launches."""
    candidates = {CLAUDE_MODEL_SNAPSHOT_PATH}
    candidates.update(APP_DIR.glob("claude-default-model.*.snapshot.json"))
    for path in sorted(candidates):
        restore_claude_model_snapshot(user_settings_path, path)


def _loopback_websocket_url(port: int) -> str:
    return f"ws://{LOOPBACK_HOST}:{port}"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind((LOOPBACK_HOST, 0))
        return sock.getsockname()[1]


def _wait_for_app_server(port: int, timeout: float) -> bool:
    url = f"http://{LOOPBACK_HOST}:{port}/healthz"
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(  # noqa: S310
                url, timeout=HEALTH_REQUEST_TIMEOUT_SECONDS
            ) as response:
                if response.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(HEALTH_POLL_INTERVAL_SECONDS)
    return False


def _switch_message(model: str, reason: str) -> str:
    lines = [
        "Using Unity Gateway Smart Router.",
        f"Selected Model : {model}",
        f"Reason : {reason}",
    ]
    width = max(len(line) for line in lines)
    border = "─" * (width + 2)
    return "\n".join([f"┌{border}┐", *(f"│ {line:<{width}} │" for line in lines), f"└{border}┘"])


def _generate_codex_app_server_home(
    state: dict,
    model: str,
    render_overlay: Callable[..., dict],
) -> Path:
    CODEX_APP_SERVER_HOME.mkdir(parents=True, exist_ok=True)
    config_path = CODEX_APP_SERVER_HOME / "config.toml"
    overlay = render_overlay(
        state["workspace"],
        model,
        state.get("profile"),
        use_pat=bool(state.get("use_pat")),
    )
    doc = read_toml_safe(config_path)
    deep_merge_dict(doc, overlay)
    write_toml_file(config_path, doc)
    return CODEX_APP_SERVER_HOME


def _route_claude_prompt(_prompt: str) -> str:
    return CLAUDE_TARGET_MODEL


def launch_claude(
    state: dict,
    tool_args: list[str],
    *,
    binary: str,
    user_settings_path: Path,
    model_snapshot: dict,
    launch_model: str | None,
    compose_settings: Callable[[list[str]], tuple[dict, list[str]]],
    launch_model_args: Callable[[list[str], str | None], list[str]],
) -> NoReturn:
    """Launch Claude in the first-prompt routing PTY wrapper."""
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError(
            "Smart routing v2 needs a configured workspace; run `ucode configure claude` first."
        )
    os.environ[OAUTH_TOKEN_ENV_VAR] = get_databricks_token(workspace, state.get("profile"))

    run_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    socket_path = APP_DIR / f"claude-v2-{run_id}.sock"
    settings_path = APP_DIR / f"claude-v2-{run_id}.json"
    model_snapshot_path = APP_DIR / f"claude-default-model.{run_id}.snapshot.json"

    settings, remaining = compose_settings(tool_args)
    hook_executable = build_auth_token_argv(
        workspace, state.get("profile"), use_pat=bool(state.get("use_pat"))
    )[0]
    env = settings.setdefault("env", {})
    if not isinstance(env, dict):
        raise RuntimeError("Claude settings 'env' must be an object for smart routing.")
    env[FIRST_PROMPT_SOCKET_ENV] = str(socket_path)
    sync_first_prompt_hook(settings, hook_executable)
    write_json_file(settings_path, settings)
    model_args = launch_model_args(remaining, launch_model)
    argv = [binary, "--settings", str(settings_path), *model_args, *remaining]

    _save_claude_model_snapshot(model_snapshot, model_snapshot_path)
    restored = False

    def restore_default_model() -> bool:
        nonlocal restored
        if restored:
            return False
        result = restore_claude_model_snapshot(
            user_settings_path, model_snapshot_path, model_snapshot
        )
        restored = True
        return result

    print_note(
        "Smart routing v2: the first submitted prompt will select Claude Code's "
        f"model ({CLAUDE_TARGET_MODEL}); log: {CLAUDE_PTY_LOG}."
    )
    try:
        returncode = claude_pty.run_claude_pty(
            argv,
            route_prompt=_route_claude_prompt,
            switch_message=(
                f"✨ Databricks Smart Router selected {CLAUDE_TARGET_MODEL} due to "
                "low complexity, unclear intent, and no code reference."
            ),
            socket_path=socket_path,
            log_path=CLAUDE_PTY_LOG,
        )
    finally:
        # Restore only after Claude exits. Claude persists `/model` asynchronously;
        # restoring as soon as its success message renders races that delayed write.
        # The journal repairs hard-killed and concurrent launches before they start.
        restore_default_model()
        settings_path.unlink(missing_ok=True)
        socket_path.unlink(missing_ok=True)
    sys.exit(returncode)


def launch_codex(
    state: dict,
    tool_args: list[str],
    *,
    binary: str,
    start_model: str | None,
    render_overlay: Callable[..., dict],
) -> NoReturn:
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError(
            "Smart routing v2 needs a configured workspace; run `ucode configure codex` first."
        )
    if not start_model:
        raise RuntimeError(
            "Smart routing v2 could not determine a starting Codex model for this workspace."
        )

    os.environ[OAUTH_TOKEN_ENV_VAR] = get_databricks_token(workspace, state.get("profile"))
    home = _generate_codex_app_server_home(state, start_model, render_overlay)
    app_port = _free_port()
    app_server_url = _loopback_websocket_url(app_port)

    app_server = subprocess.Popen(
        [binary, "app-server", "--listen", app_server_url],
        env={**os.environ, CODEX_HOME_ENV_VAR: str(home)},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    stop_interposer = None
    try:
        if not _wait_for_app_server(app_port, timeout=APP_SERVER_READY_TIMEOUT_SECONDS):
            raise RuntimeError(
                "Codex app-server did not become ready for smart routing v2; check workspace auth."
            )
        tui_port, stop_interposer = codex_interposer.start_interposer_thread(
            LOOPBACK_HOST,
            app_server_url,
            CODEX_TARGET_MODEL,
            switch_message=_switch_message(CODEX_TARGET_MODEL, CODEX_SWITCH_REASON),
            log_path=CODEX_INTERPOSER_LOG,
        )
        tui_url = _loopback_websocket_url(tui_port)
        tui = subprocess.Popen([binary, "--remote", tui_url, "--model", start_model, *tool_args])
        try:
            returncode = tui.wait()
        except KeyboardInterrupt:
            tui.send_signal(signal.SIGINT)
            returncode = tui.wait()
    finally:
        if stop_interposer is not None:
            stop_interposer()
        app_server.terminate()
        try:
            app_server.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)
        except Exception:  # noqa: BLE001
            app_server.kill()
    sys.exit(returncode)
