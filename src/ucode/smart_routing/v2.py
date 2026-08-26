from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from typing import NoReturn

import tomlkit

from ucode.config_io import APP_DIR
from ucode.databricks import get_databricks_token
from ucode.smart_routing import codex_interposer

ENV_VAR = "ENABLE_SMART_ROUTING_V2"

CODEX_TARGET_MODEL = "system.ai.glm-5-2"  # TODO(lilly): replace with smart router.
CODEX_INTERPOSER_LOG = APP_DIR / "codex-v2-interposer.log"
CODEX_SWITCH_REASON = "Low complexity, unclear intent, and no code reference."  # TODO(lilly): replace with smart router rationale.

APP_SERVER_READY_TIMEOUT_SECONDS = 30
PROCESS_SHUTDOWN_TIMEOUT_SECONDS = 5
OAUTH_TOKEN_ENV_VAR = "OAUTH_TOKEN"
LOOPBACK_HOST = "127.0.0.1"
HEALTH_REQUEST_TIMEOUT_SECONDS = 1
HEALTH_POLL_INTERVAL_SECONDS = 0.25


def enabled() -> bool:
    return os.environ.get(ENV_VAR) == "1"


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


def _toml_value(value: str | int | float | bool | list[object] | dict[str, object]) -> str:
    if isinstance(value, dict):
        item = tomlkit.inline_table()
        item.update(value)
        return item.as_string()
    return tomlkit.item(value).as_string()


def _codex_config_args(overlay: dict) -> list[str]:
    args: list[str] = []
    for key, value in overlay.items():
        if key == "model_providers" and isinstance(value, dict):
            for provider_name, provider_config in value.items():
                args.extend(
                    [
                        "--config",
                        f"model_providers.{provider_name}={_toml_value(provider_config)}",
                    ]
                )
        else:
            args.extend(["--config", f"{key}={_toml_value(value)}"])
    return args


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
    overlay = render_overlay(
        workspace,
        start_model,
        state.get("profile"),
        use_pat=bool(state.get("use_pat")),
    )
    config_args = _codex_config_args(overlay)
    app_port = _free_port()
    app_server_url = _loopback_websocket_url(app_port)

    # Preserve the user's normal CODEX_HOME (including MCP servers, skills, and
    # preferences) and layer only ucode's gateway settings at CLI precedence.
    app_server = subprocess.Popen(
        [binary, "app-server", *config_args, "--listen", app_server_url],
        env=os.environ.copy(),
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
