"""Runtime model-switching launch path for smart routing v2.

Smart routing v2 launches an agent's real TUI against a ucode-run app-server with a
WebSocket interposer that switches the model mid-session (see e.g.
``smart_routing.codex_interposer``). The enable flag, launch configuration, and process
lifecycle live here so agent modules only need to supply their provider overlay.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

from ucode.config_io import APP_DIR, deep_merge_dict, read_toml_safe, write_toml_file
from ucode.databricks import get_databricks_token
from ucode.smart_routing import codex_interposer

# Single env var that enables the v2 launch path for every routing-capable agent.
ENV_VAR = "ENABLE_SMART_ROUTING_V2"

CODEX_TARGET_MODEL = "system.ai.glm-5-2"  # TODO(lilly): replace with smart router.
CODEX_APP_SERVER_HOME = APP_DIR / "codex-v2-home"
CODEX_INTERPOSER_LOG = APP_DIR / "codex-v2-interposer.log"
CODEX_SWITCH_REASON = "Low complexity, unclear intent, and no code reference."  # TODO(lilly): replace with smart router rationale.

APP_SERVER_READY_TIMEOUT_SECONDS = 30
PROCESS_SHUTDOWN_TIMEOUT_SECONDS = 5
OAUTH_TOKEN_ENV_VAR = "OAUTH_TOKEN"
CODEX_HOME_ENV_VAR = "CODEX_HOME"
LOOPBACK_HOST = "127.0.0.1"
HEALTH_REQUEST_TIMEOUT_SECONDS = 1
HEALTH_POLL_INTERVAL_SECONDS = 0.25


def enabled() -> bool:
    """Return whether the smart-routing-v2 launch path is enabled via the env var."""
    return os.environ.get(ENV_VAR) == "1"


def _loopback_websocket_url(port: int) -> str:
    return f"ws://{LOOPBACK_HOST}:{port}"


def _free_port() -> int:
    """Return an available loopback port for the app-server to bind."""
    with socket.socket() as sock:
        sock.bind((LOOPBACK_HOST, 0))
        return sock.getsockname()[1]


def _wait_for_app_server(port: int, timeout: float) -> bool:
    """Poll the app-server's health endpoint until it is ready or times out."""
    url = f"http://{LOOPBACK_HOST}:{port}/healthz"
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(  # noqa: S310 - fixed loopback URL
                url, timeout=HEALTH_REQUEST_TIMEOUT_SECONDS
            ) as response:
                if response.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - the app-server is not ready yet
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
    """Write the isolated CODEX_HOME used by the ucode-run app-server."""
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


def launch_codex(
    state: dict,
    tool_args: list[str],
    *,
    binary: str,
    start_model: str | None,
    render_overlay: Callable[..., dict],
) -> NoReturn:
    """Launch the Codex app-server, interposer, and remote TUI as one lifecycle."""
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
        # Keep ucode alive while the TUI runs so it can tear down the app-server and interposer.
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
        except Exception:  # noqa: BLE001 - the app-server must never linger
            app_server.kill()
    sys.exit(returncode)
