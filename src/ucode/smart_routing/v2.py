from __future__ import annotations

import hashlib
import json
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

from ucode.config_io import APP_DIR, read_json_safe, read_toml_safe, write_json_file
from ucode.constants import LOOPBACK_HOST
from ucode.databricks import (
    build_auth_token_argv,
    get_databricks_token,
    list_anthropic_models,
)
from ucode.smart_routing import claude_routing, codex_interposer, routing
from ucode.smart_routing.claude_hooks import (
    FIRST_PROMPT_SOCKET_ENV,
    sync_first_prompt_hook,
    sync_smart_routing_hooks,
)
from ucode.smart_routing.codex_hooks import merge_pre_tool_use_hooks, routing_models
from ucode.ui import print_note

ENV_VAR = "ENABLE_SMART_ROUTING_V2"
LEGACY_STATE_KEY = "smart_routing_enabled"

CODEX_INTERPOSER_LOG = APP_DIR / "codex-v2-interposer.log"

CLAUDE_TARGET_MODEL = "system.ai.claude-sonnet-4-6[1m]"  # TODO(lilly): replace with smart router.
CLAUDE_PTY_LOG = APP_DIR / "claude-v2-pty.log"

APP_SERVER_READY_TIMEOUT_SECONDS = 30
PROCESS_SHUTDOWN_TIMEOUT_SECONDS = 5
OAUTH_TOKEN_ENV_VAR = "OAUTH_TOKEN"
HEALTH_REQUEST_TIMEOUT_SECONDS = 1
HEALTH_POLL_INTERVAL_SECONDS = 0.25
CLAUDE_ROUTE_SELECTION_TIMEOUT_S = 20.0
CLAUDE_ROUTED_AGENT_PREFIX = "ucode-route-"
CLAUDE_ROUTED_AGENT_PROMPT = (
    "Complete the delegated task exactly as requested. Follow the parent agent's instructions and "
    "return a concise report of your findings or changes."
)


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
    return routing.format_switch_message(model, reason)


def format_routing_notice(model: str, reason: str | None) -> str:
    return routing.format_switch_message(model, reason or "")


def _canonical_claude_model_id(model: str) -> str:
    """Use the system.ai id when Anthropic discovery returns a legacy alias."""
    prefix = "databricks-claude-"
    if model.startswith(prefix):
        return f"system.ai.claude-{model[len(prefix) :]}"
    return model


def _canonical_claude_models(model_ids: list[str]) -> list[str]:
    return list(
        dict.fromkeys(
            _canonical_claude_model_id(model)
            for model in model_ids
            if isinstance(model, str) and model
        )
    )


def _claude_model_overrides(model_ids: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    prefix = "system.ai."
    for model in _canonical_claude_models(model_ids):
        if model.startswith(f"{prefix}claude-"):
            overrides[model[len(prefix) :]] = model
    return overrides


def _routed_claude_agent_name(model: str) -> str:
    canonical = _canonical_claude_model_id(model)
    normalized = routing.normalize_model(canonical)
    safe = "".join(character if character.isalnum() else "-" for character in normalized)
    slug = "-".join(part for part in safe.split("-") if part)
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:8]
    return f"{CLAUDE_ROUTED_AGENT_PREFIX}{slug[:36]}-{digest}"


def _routed_claude_agent_definitions(model_ids: list[str]) -> dict[str, dict[str, str]]:
    return {
        _routed_claude_agent_name(model): {
            "description": f"Smart-routed coding agent using {model}",
            "prompt": CLAUDE_ROUTED_AGENT_PROMPT,
            "model": model,
        }
        for model in _canonical_claude_models(model_ids)
    }


def _with_routed_claude_agents(tool_args: list[str], model_ids: list[str]) -> list[str]:
    definitions = _routed_claude_agent_definitions(model_ids)
    caller_definitions: dict = {}
    remaining: list[str] = []
    index = 0
    while index < len(tool_args):
        arg = tool_args[index]
        if arg == "--":
            remaining.extend(tool_args[index:])
            break
        if arg == "--agents":
            if index + 1 >= len(tool_args):
                raise RuntimeError("Claude's --agents option requires a JSON object.")
            raw = tool_args[index + 1]
            index += 2
        elif arg.startswith("--agents="):
            raw = arg.partition("=")[2]
            index += 1
        else:
            remaining.append(arg)
            index += 1
            continue
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise RuntimeError("Claude's --agents option must contain valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Claude's --agents option must contain a JSON object.")
        caller_definitions.update(parsed)

    collisions = definitions.keys() & caller_definitions.keys()
    if collisions:
        names = ", ".join(sorted(collisions))
        raise RuntimeError(f"Claude --agents names conflict with smart routing: {names}.")
    combined = {**caller_definitions, **definitions}
    return ["--agents", json.dumps(combined, separators=(",", ":")), *remaining]


def _request_claude_routing_decision(
    workspace: str,
    token: str,
    prompt: str,
    model_ids: list[str],
) -> tuple[routing.RoutingDecision | None, str | None]:
    available: dict[str, str] = {}
    for model in _canonical_claude_models(model_ids):
        available.setdefault(routing.normalize_model(model), model)
    if not available:
        return None, "Anthropic models endpoint returned no Claude models"
    return routing.select_route(
        workspace,
        token,
        prompt,
        [(model, "claude") for model in available],
        lambda selected: available.get(routing.normalize_model(selected)),
        timeout=CLAUDE_ROUTE_SELECTION_TIMEOUT_S,
    )


def _route_claude_prompt(
    state: dict,
    token: str,
    prompt: str,
    model_ids: list[str] | None = None,
) -> routing.RoutingDecision:
    workspace = state.get("workspace")
    if not isinstance(workspace, str):
        raise RuntimeError("workspace metadata is unavailable")

    if model_ids is None:
        model_ids, discovery_error = list_anthropic_models(workspace, token)
        if not model_ids:
            raise RuntimeError(
                discovery_error or "Anthropic models endpoint returned no Claude models"
            )
    decision, error = _request_claude_routing_decision(workspace, token, prompt, model_ids)
    if decision is None:
        raise RuntimeError(error or "router returned no Claude model selection")
    return decision


def route_claude_pre_tool_use(
    payload: dict,
    *,
    workspace: str,
    token: str,
    available_models: list[str],
    audit_decision: bool = False,
) -> dict | None:
    """Route a Claude Agent call through a transient exact-model agent definition."""
    route = routing.resolve_spawn_route(
        payload,
        is_spawn_agent=claude_routing.is_spawn_agent_tool,
        decision_fn=lambda task: _request_claude_routing_decision(
            workspace, token, task, available_models
        ),
        default_task_label="Claude Code subagent task",
        model_id_mapper=lambda model: model,
    )
    if route is None:
        return None
    if audit_decision:
        routing.write_decision_record(
            claude_routing.DECISIONS_PATH,
            payload,
            route.task,
            route.decision,
            route.routed_model,
        )
    routing_message = routing.format_subagent_message(
        route.routed_model,
        route.decision.rationale,
    )
    updated_input = {
        **{key: value for key, value in route.tool_input.items() if key != "model"},
        "subagent_type": _routed_claude_agent_name(route.routed_model),
    }
    hook_output = {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "updatedInput": updated_input,
        "permissionDecisionReason": routing_message,
    }
    return {"systemMessage": routing_message, "hookSpecificOutput": hook_output}


def _is_claude_target_model(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return value.removesuffix("[1m]") == CLAUDE_TARGET_MODEL.removesuffix("[1m]")


class _ClaudeModelSettingGuard:
    def __init__(self, settings_path: Path) -> None:
        self.settings_path = settings_path
        self._before: dict | None = None
        self._routed_model: str | None = None
        self._lock: TextIO | None = None

    def begin(self, routed_model: str) -> None:
        import fcntl

        APP_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = open(APP_DIR / "claude-v2-model.lock", "a+", encoding="utf-8")
        fcntl.flock(self._lock, fcntl.LOCK_EX)
        self._before = read_json_safe(self.settings_path)
        self._routed_model = routed_model

    def is_routed(self) -> bool:
        value = read_json_safe(self.settings_path).get("model")
        return isinstance(value, str) and value == self._routed_model

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
            self._routed_model = None
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
    model_name: Callable[[str], str],
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
    model_ids, discovery_error = list_anthropic_models(workspace, token)
    if not model_ids:
        raise RuntimeError(discovery_error or "Anthropic models endpoint returned no Claude models")

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
    model_overrides = settings.setdefault("modelOverrides", {})
    if not isinstance(model_overrides, dict):
        raise RuntimeError("Claude settings 'modelOverrides' must be an object for smart routing.")
    model_overrides.update(_claude_model_overrides(model_ids))
    routing_state = {
        **state,
        "claude_models": {str(index): model for index, model in enumerate(model_ids)},
    }
    sync_smart_routing_hooks(settings, routing_state, enabled=True)
    sync_first_prompt_hook(settings, hook_executable)
    write_json_file(settings_path, settings)
    model_args = launch_model_args(remaining, launch_model)
    routed_agent_args = _with_routed_claude_agents(remaining, model_ids)
    argv = [binary, "--settings", str(settings_path), *model_args, *routed_agent_args]

    model_setting = _ClaudeModelSettingGuard(user_settings_path)

    def route_prompt(prompt: str) -> tuple[str, str]:
        decision = _route_claude_prompt(state, token, prompt, model_ids)
        return model_name(decision.model), decision.rationale

    print_note(
        "Smart routing v2: the first submitted prompt will select Claude Code's "
        f"model; log: {CLAUDE_PTY_LOG}."
    )
    try:
        returncode = claude_pty.run_claude_pty(
            argv,
            route_prompt=route_prompt,
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
    if isinstance(value, list) and any(isinstance(entry, dict) for entry in value):
        wrapper = tomlkit.inline_table()
        wrapper["value"] = value
        rendered = wrapper.as_string()
        return rendered.removeprefix("{value = ").removesuffix("}")
    return tomlkit.item(value).as_string()


def _codex_config_args(overlay: dict) -> list[str]:
    args: list[str] = []
    for key, value in overlay.items():
        # This is Codex's AI Gateway transport definition, not Unity Catalog
        # Model Provider Service support; smart routing still cannot use --provider.
        if key in {"hooks", "model_providers"} and isinstance(value, dict):
            for provider_name, provider_config in value.items():
                args.extend(
                    [
                        "--config",
                        f"{key}.{provider_name}={_toml_value(provider_config)}",
                    ]
                )
        else:
            args.extend(["--config", f"{key}={_toml_value(value)}"])
    return args


# TODO: Replace with /codex/v1/models once /codex/v1/models can send GPT models as well.
def _cached_routing_models(state: dict) -> list[str]:
    """Return the persisted UC model-service ids usable by Codex routing."""
    return routing_models(state)


def _codex_home_config_path() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "config.toml"
    return Path.home() / ".codex" / "config.toml"


def _v2_pre_tool_use_hooks(state: dict, available_models: list[str]) -> list[dict]:
    doc = read_toml_safe(_codex_home_config_path())
    configured_hooks = doc.get("hooks")
    existing = configured_hooks.get("PreToolUse") if isinstance(configured_hooks, dict) else None
    return merge_pre_tool_use_hooks(
        existing if isinstance(existing, list) else [],
        state,
        available_models=available_models,
    )


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
        print_note(
            "Smart routing model metadata is unavailable; starting Codex on gpt-5.6-luna "
            "without automatic model switching. Run `ucode configure codex` to enable routing."
        )
    overlay = render_overlay(
        workspace,
        start_model,
        state.get("profile"),
        use_pat=bool(state.get("use_pat")),
    )
    overlay["hooks"] = {
        "PreToolUse": _v2_pre_tool_use_hooks(state, available_models),
    }
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
            switch_message_fn=format_routing_notice,
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
