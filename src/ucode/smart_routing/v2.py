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
from typing import NoReturn, TextIO

import tomlkit

from ucode.config_io import APP_DIR, read_json_safe, write_json_file
from ucode.constants import LOOPBACK_HOST
from ucode.databricks import (
    build_auth_token_argv,
    get_databricks_token,
    list_anthropic_models,
)
from ucode.smart_routing import codex_interposer, routing
from ucode.smart_routing.claude_hooks import FIRST_PROMPT_SOCKET_ENV, sync_first_prompt_hook
from ucode.ui import print_note

ENV_VAR = "ENABLE_SMART_ROUTING_V2"

CODEX_INTERPOSER_LOG = APP_DIR / "codex-v2-interposer.log"
STUBBED_SWITCH_REASON = "Low complexity, unclear intent, and no code reference."  # TODO(lilly): replace with smart router rationale.

CLAUDE_TARGET_MODEL = "system.ai.claude-sonnet-4-6[1m]"  # TODO(lilly): replace with smart router.
CLAUDE_PTY_LOG = APP_DIR / "claude-v2-pty.log"

APP_SERVER_READY_TIMEOUT_SECONDS = 30
PROCESS_SHUTDOWN_TIMEOUT_SECONDS = 5
OAUTH_TOKEN_ENV_VAR = "OAUTH_TOKEN"
HEALTH_REQUEST_TIMEOUT_SECONDS = 1
HEALTH_POLL_INTERVAL_SECONDS = 0.25
CLAUDE_ROUTE_SELECTION_TIMEOUT_S = 20.0


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


def _canonical_claude_model_id(model: str) -> str:
    """Use the system.ai id when Anthropic discovery returns a legacy alias."""
    prefix = "databricks-claude-"
    if model.startswith(prefix):
        return f"system.ai.claude-{model[len(prefix) :]}"
    return model


def _route_claude_prompt(state: dict, token: str, prompt: str) -> routing.RoutingDecision:
    workspace = state.get("workspace")
    if not isinstance(workspace, str):
        raise RuntimeError("workspace metadata is unavailable")

    model_ids, discovery_error = list_anthropic_models(workspace, token)
    if not model_ids:
        raise RuntimeError(discovery_error or "Anthropic models endpoint returned no Claude models")

    available: dict[str, str] = {}
    for model in model_ids:
        if isinstance(model, str) and model:
            canonical = _canonical_claude_model_id(model)
            available.setdefault(routing.normalize_model(canonical), canonical)
    decision, error = routing.select_route(
        workspace,
        token,
        prompt,
        [(model, "claude") for model in available],
        lambda selected: available.get(routing.normalize_model(selected)),
        timeout=CLAUDE_ROUTE_SELECTION_TIMEOUT_S,
    )
    if decision is None:
        raise RuntimeError(error or "router returned no Claude model selection")
    return decision


def _is_claude_target_model(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return value.removesuffix("[1m]") == CLAUDE_TARGET_MODEL.removesuffix("[1m]")


class _ClaudeModelSettingGuard:
    def __init__(self, settings_path: Path) -> None:
        self.settings_path = settings_path
        self._before: dict | None = None
        self._lock: TextIO | None = None

    def begin(self) -> None:
        import fcntl

        APP_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = open(APP_DIR / "claude-v2-model.lock", "a+", encoding="utf-8")
        fcntl.flock(self._lock, fcntl.LOCK_EX)
        self._before = read_json_safe(self.settings_path)

    def is_routed(self) -> bool:
        return _is_claude_target_model(read_json_safe(self.settings_path).get("model"))

    def restore(self) -> None:
        import fcntl

        if self._before is None:
            return
        try:
            current = read_json_safe(self.settings_path)
            if "model" in self._before:
                current["model"] = self._before["model"]
            else:
                current.pop("model", None)
            write_json_file(self.settings_path, current)
            self._before = None
        finally:
            if self._lock is not None:
                fcntl.flock(self._lock, fcntl.LOCK_UN)
                self._lock.close()
                self._lock = None


def launch_claude(
    state: dict,
    tool_args: list[str],
    *,
    binary: str,
    user_settings_path: Path,
    launch_model: str | None,
    compose_settings: Callable[[list[str]], tuple[dict, list[str]]],
    launch_model_args: Callable[[list[str], str | None], list[str]],
) -> NoReturn:
    """Launch Claude in the first-prompt routing PTY wrapper."""
    from ucode.agents.claude import GATEWAY_MODEL_DISCOVERY_ENV_VAR
    from ucode.smart_routing import claude_pty

    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError(
            "Smart routing v2 needs a configured workspace; run `ucode configure claude` first."
        )
    token = get_databricks_token(workspace, state.get("profile"))
    os.environ[OAUTH_TOKEN_ENV_VAR] = token
    os.environ[GATEWAY_MODEL_DISCOVERY_ENV_VAR] = "1"

    run_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    socket_path = APP_DIR / f"claude-v2-{run_id}.sock"
    settings_path = APP_DIR / f"claude-v2-{run_id}.json"

    settings, remaining = compose_settings(tool_args)
    hook_executable = build_auth_token_argv(
        workspace, state.get("profile"), use_pat=bool(state.get("use_pat"))
    )[0]
    env = settings.setdefault("env", {})
    if not isinstance(env, dict):
        raise RuntimeError("Claude settings 'env' must be an object for smart routing.")
    env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    env[FIRST_PROMPT_SOCKET_ENV] = str(socket_path)
    sync_first_prompt_hook(settings, hook_executable)
    write_json_file(settings_path, settings)
    model_args = launch_model_args(remaining, launch_model)
    argv = [binary, "--settings", str(settings_path), *model_args, *remaining]

    model_setting = _ClaudeModelSettingGuard(user_settings_path)

    print_note(
        "Smart routing v2: the first submitted prompt will select Claude Code's "
        f"model ({CLAUDE_TARGET_MODEL}); log: {CLAUDE_PTY_LOG}."
    )
    try:
        returncode = claude_pty.run_claude_pty(
            argv,
            route_prompt=lambda prompt: _route_claude_prompt(state, token, prompt),
            socket_path=socket_path,
            prepare_model_switch=model_setting.begin,
            model_switch_persisted=model_setting.is_routed,
            restore_model_setting=model_setting.restore,
            log_path=CLAUDE_PTY_LOG,
        )
    finally:
        model_setting.restore()
        settings_path.unlink(missing_ok=True)
        socket_path.unlink(missing_ok=True)
    sys.exit(returncode)


def _toml_value(value: str | int | float | bool | list[object] | dict[str, object]) -> str:
    if isinstance(value, dict):
        item = tomlkit.inline_table()
        item.update(value)
        return item.as_string()
    return tomlkit.item(value).as_string()


def _codex_config_args(overlay: dict) -> list[str]:
    args: list[str] = []
    for key, value in overlay.items():
        # This is Codex's AI Gateway transport definition, not Unity Catalog
        # Model Provider Service support; smart routing still cannot use --provider.
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


# TODO: Replace with /codex/v1/models once /codex/v1/models can send GPT models as well.
def _cached_routing_models(state: dict) -> list[str]:
    """Return the persisted UC model-service ids usable by Codex routing."""
    models: list[str] = []
    for key in ("codex_models", "oss_models"):
        values = state.get(key)
        if isinstance(values, list):
            models.extend(value for value in values if isinstance(value, str) and value)
    return list(dict.fromkeys(models))


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

    profile = state.get("profile")
    os.environ[OAUTH_TOKEN_ENV_VAR] = get_databricks_token(workspace, profile)
    available_models = _cached_routing_models(state)
    if not available_models:
        raise RuntimeError(
            "Smart routing v2 has no cached Unity Catalog model services; "
            "run `ucode configure codex` to refresh them."
        )
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
            available_models=available_models,
            workspace=workspace,
            token_provider=lambda: get_databricks_token(workspace, profile),
            switch_message_fn=_switch_message,
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
