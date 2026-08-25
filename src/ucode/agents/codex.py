"""Codex agent: writes ~/.codex/ucode.config.toml for Databricks-backed Codex."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import tomlkit

from ucode.agent_updates import available_npm_package_update
from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    deep_merge_dict,
    read_toml_safe,
    write_toml_file,
)
from ucode.databricks import (
    build_auth_token_argv,
    build_tool_base_url,
    get_databricks_token,
)
from ucode.launcher import exec_or_spawn
from ucode.managed_files import OS, current_os, write_managed_file
from ucode.smart_routing import v2 as smart_routing_v2
from ucode.smart_routing.codex_hooks import (
    remove_smart_routing_hooks,
    sync_smart_routing_hooks,
)
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version
from ucode.ui import print_warning_err

CODEX_CONFIG_DIR = Path.home() / ".codex"
CODEX_PROFILE_NAME = "ucode"
CODEX_CONFIG_PATH = CODEX_CONFIG_DIR / f"{CODEX_PROFILE_NAME}.config.toml"
CODEX_BACKUP_PATH = APP_DIR / "codex-ucode-config.backup.toml"
LEGACY_CODEX_CONFIG_PATH = CODEX_CONFIG_DIR / "config.toml"
LEGACY_CODEX_BACKUP_PATH = APP_DIR / "codex-config.backup.toml"
CODEX_MODEL_PROVIDER_NAME = "ucode-databricks"
MINIMUM_CODEX_VERSION = (0, 134, 0)
MINIMUM_CODEX_VERSION_TEXT = "0.134.0"
MINIMUM_ROUTING_CODEX_VERSION = (0, 145, 0)
MINIMUM_ROUTING_CODEX_VERSION_TEXT = "0.145.0"
# Shared across agents: one opt-in enables smart routing for every routing-capable
# tool (codex, claude), so a workspace turns it on once.
SMART_ROUTING_STATE_KEY = "smart_routing_enabled"

SMART_ROUTING_V2_TARGET_MODEL = "system.ai.glm-5-2"  # hardcoded switch-to model for now
SMART_ROUTING_V2_HOME = APP_DIR / "codex-v2-home"  # CODEX_HOME for the ucode-run app-server
SMART_ROUTING_V2_LOG = (
    APP_DIR / "codex-v2-interposer.log"
)  # interposer log (not stdout: TUI owns it)
SMART_ROUTING_V2_REASON = "Low complexity, unclear intent, and no code reference."


def _smart_routing_switch_message(model: str, reason: str) -> str:
    lines = [
        "Using Unity Gateway Smart Router.",
        f"Selected Model : {model}",
        f"Reason : {reason}",
    ]
    width = max(len(line) for line in lines)
    border = "─" * (width + 2)
    return "\n".join(
        [f"┌{border}┐", *(f"│ {line:<{width}} │" for line in lines), f"└{border}┘"]
    )


SPEC: ToolSpec = {
    "binary": "codex",
    "package": "@openai/codex",
    "display": "Codex",
    "config_path": CODEX_CONFIG_PATH,
    "backup_path": CODEX_BACKUP_PATH,
}

MANAGED_KEYS: list[list[str]] = [
    ["model_provider"],
    ["model"],
    ["model_providers", CODEX_MODEL_PROVIDER_NAME],
    ["model_providers", CODEX_MODEL_PROVIDER_NAME, "http_headers"],
]

LEGACY_MANAGED_KEYS: list[list[str]] = [
    ["profile"],
    ["profiles", CODEX_PROFILE_NAME],
    ["model_providers", CODEX_MODEL_PROVIDER_NAME],
    ["model_providers", CODEX_MODEL_PROVIDER_NAME, "http_headers"],
]

_GPT_RE = re.compile(r"(?:databricks-)?gpt-(\d+)(?:[.-](\d+))?(?:[.-](\d+))?(-.+|[a-z].*)?")


def is_update_available() -> tuple[str, str] | None:
    return available_npm_package_update(SPEC["package"])


def _parse_version(value: str) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _installed_version_status() -> tuple[str, bool] | None:
    version = agent_version(SPEC["binary"])
    parsed = _parse_version(version)
    if parsed is None:
        return None
    return version, parsed < MINIMUM_CODEX_VERSION


def _use_legacy_layout() -> bool:
    """Return True when the installed Codex CLI predates per-profile config files.

    Codex 0.134.0 introduced support for `--profile <name>` resolving to
    `~/.codex/<name>.config.toml`. Older releases only honor a single
    `~/.codex/config.toml` with `[profiles.<name>]` sections. When the version
    is unknown we keep the new layout (matches the prior "unknown does not
    block" semantic).
    """
    parsed = _parse_version(agent_version(SPEC["binary"]))
    if parsed is None:
        return False
    return parsed < MINIMUM_CODEX_VERSION


def _provider_block(
    workspace: str,
    databricks_profile: str | None,
    use_pat: bool = False,
    provider: str | None = None,
) -> dict:
    auth_argv = build_auth_token_argv(workspace, databricks_profile, use_pat=use_pat)
    base_url = build_tool_base_url("codex", workspace)
    http_headers = {
        "User-Agent": f"ucode/{ucode_version()} codex/{agent_version('codex')}",
    }
    # Route to an external Model Provider Service; the gateway selects the
    # provider from this header on every request.
    if provider:
        http_headers["Databricks-Model-Provider-Service"] = provider
    return {
        "name": "Databricks AI Gateway",
        "base_url": base_url,
        "wire_api": "responses",
        "http_headers": http_headers,
        # Run the `ucode auth-token` executable directly (not via `sh -c`) so the
        # helper works on Windows, where there is no POSIX shell (issue #116).
        "auth": {
            "command": auth_argv[0],
            "args": auth_argv[1:],
            "timeout_ms": 5000,
            "refresh_interval_ms": 900000,
        },
    }


def render_overlay(
    workspace: str,
    model: str | None = None,
    databricks_profile: str | None = None,
    use_pat: bool = False,
    provider: str | None = None,
) -> dict:
    overlay: dict = {"model_provider": CODEX_MODEL_PROVIDER_NAME}
    if model:
        overlay["model"] = model
    overlay["model_providers"] = {
        CODEX_MODEL_PROVIDER_NAME: _provider_block(
            workspace, databricks_profile, use_pat, provider
        ),
    }
    return overlay


def render_legacy_overlay(
    workspace: str,
    model: str | None = None,
    databricks_profile: str | None = None,
    use_pat: bool = False,
    provider: str | None = None,
) -> dict:
    """Overlay for Codex CLI < 0.134.0, which only reads `~/.codex/config.toml`.

    The shared file uses `profile = "ucode"` to select `[profiles.ucode]`, which
    points at the shared `[model_providers.ucode-databricks]` block.
    """
    profile_block: dict = {"model_provider": CODEX_MODEL_PROVIDER_NAME}
    if model:
        profile_block["model"] = model
    return {
        "profile": CODEX_PROFILE_NAME,
        "profiles": {CODEX_PROFILE_NAME: profile_block},
        "model_providers": {
            CODEX_MODEL_PROVIDER_NAME: _provider_block(
                workspace, databricks_profile, use_pat, provider
            ),
        },
    }


def _legacy_config_path() -> Path:
    return CODEX_CONFIG_PATH.parent / "config.toml"


def _legacy_backup_path() -> Path:
    return CODEX_BACKUP_PATH.with_name("codex-legacy-config.backup.toml")


def _has_legacy_ucode_entries(doc: dict) -> bool:
    profiles = doc.get("profiles")
    providers = doc.get("model_providers")
    return (
        doc.get("profile") == CODEX_PROFILE_NAME
        or (isinstance(profiles, dict) and CODEX_PROFILE_NAME in profiles)
        or (isinstance(providers, dict) and CODEX_MODEL_PROVIDER_NAME in providers)
    )


def _strip_legacy_ucode_entries(path: Path) -> bool:
    """Surgically remove ucode's keys from a shared Codex config.

    Drops the top-level ``profile = "ucode"`` selector, ``[profiles.ucode]``,
    and ``[model_providers.ucode-databricks]`` while leaving everything else the
    user has in the file untouched. Returns True if anything was removed.

    Surgical removal beats restoring the backup: ``backup_existing_file`` only
    keeps the first-ever snapshot, so a whole-file restore would clobber edits
    made since ucode first ran.
    """
    if not path.exists():
        return False

    doc = read_toml_safe(path)
    changed = False

    if doc.get("profile") == CODEX_PROFILE_NAME:
        doc.pop("profile", None)
        changed = True

    profiles = doc.get("profiles")
    if isinstance(profiles, dict) and CODEX_PROFILE_NAME in profiles:
        profiles.pop(CODEX_PROFILE_NAME, None)
        if not profiles:
            doc.pop("profiles", None)
        changed = True

    providers = doc.get("model_providers")
    if isinstance(providers, dict) and CODEX_MODEL_PROVIDER_NAME in providers:
        providers.pop(CODEX_MODEL_PROVIDER_NAME, None)
        if not providers:
            doc.pop("model_providers", None)
        changed = True

    if changed:
        write_toml_file(path, doc)
    return changed


def _remove_legacy_ucode_profile() -> None:
    """Remove ucode's old shared-config entries when configuring modern Codex.

    Strips the legacy ``profile``/``[profiles.ucode]`` selector and the
    ``[model_providers.ucode-databricks]`` provider block that older ucode
    versions deep-merged into ``~/.codex/config.toml``.
    """
    path = _legacy_config_path()
    if path == CODEX_CONFIG_PATH or not path.exists():
        return

    if _has_legacy_ucode_entries(read_toml_safe(path)):
        backup_existing_file(path, _legacy_backup_path())
        _strip_legacy_ucode_entries(path)


def revert_legacy_shared_config() -> bool:
    """Undo legacy in-place edits to ``~/.codex/config.toml`` on revert.

    Codex CLI < 0.134.0 had ucode deep-merge ``profile = "ucode"``,
    ``[profiles.ucode]``, and ``[model_providers.ucode-databricks]`` into the
    user's real shared config, which routes every bare ``codex`` invocation
    through the workspace gateway. ``ucode revert`` only restored the
    per-profile file, leaving those edits in place. Surgically strip them here.

    Returns True if anything was removed.
    """
    return _strip_legacy_ucode_entries(_legacy_config_path())


def _parse_gpt(model: str | None) -> tuple[int, int | None, int | None, str] | None:
    if not model:
        return None
    # Strip the UC model-services prefix so `system.ai.gpt-5` parses for version
    # selection; the original id is preserved by callers that need it verbatim.
    tail = model.split("/")[-1]
    if tail.startswith("system.ai."):
        tail = tail[len("system.ai.") :]
    match = _GPT_RE.fullmatch(tail)
    if not match:
        return None
    major, minor, patch, suffix = match.groups()
    return (
        int(major),
        int(minor) if minor is not None else None,
        int(patch) if patch is not None else None,
        suffix or "",
    )


def write_tool_config(state: dict, model: str | None = None, provider: str | None = None) -> dict:
    workspace = state["workspace"]
    # With a Model Provider Service the gateway routes by header and Codex sends
    # its own canonical model name (e.g. `gpt-5`) — leave `model` unset so no
    # Databricks endpoint id is pinned. Otherwise pin the discovered endpoint id
    # verbatim: the gateway routes by that exact name (whether `databricks-gpt-5`
    # from the AI Gateway listing or `system.ai.gpt-5` from UC model-services), so
    # rewriting it to an OpenAI id (`gpt-5`) makes the gateway resolve a
    # non-existent `system.ai.*` alias and 404.
    chosen_model = None if provider else (model or default_model(state))
    databricks_profile = state.get("profile")

    if _use_legacy_layout():
        if smart_routing_enabled(state) and provider is None:
            raise RuntimeError(
                f"Codex smart routing requires Codex {MINIMUM_ROUTING_CODEX_VERSION_TEXT} or newer."
            )
        # Codex < 0.134.0 only reads ~/.codex/config.toml. Write the shared
        # config with [profiles.ucode] + shared [model_providers.ucode-databricks]
        # and skip the per-profile-file cleanup that would normally strip
        # ucode's entry from the shared file.
        backup_existing_file(LEGACY_CODEX_CONFIG_PATH, LEGACY_CODEX_BACKUP_PATH)
        overlay = render_legacy_overlay(
            workspace,
            chosen_model,
            databricks_profile,
            use_pat=bool(state.get("use_pat")),
            provider=provider,
        )
        doc = read_toml_safe(LEGACY_CODEX_CONFIG_PATH)
        deep_merge_dict(doc, overlay)
        if provider:
            # deep_merge can't drop keys, so clear a `model` pinned by an
            # earlier non-provider run that the provider overlay omits.
            profiles = doc.get("profiles")
            if isinstance(profiles, dict) and isinstance(profiles.get(CODEX_PROFILE_NAME), dict):
                profiles[CODEX_PROFILE_NAME].pop("model", None)
        write_toml_file(LEGACY_CODEX_CONFIG_PATH, doc)
        state = mark_tool_managed(state, "codex", LEGACY_MANAGED_KEYS)
        save_state(state)
        return state

    _remove_legacy_ucode_profile()
    backup_existing_file(CODEX_CONFIG_PATH, CODEX_BACKUP_PATH)
    overlay = render_overlay(
        workspace,
        chosen_model,
        databricks_profile,
        use_pat=bool(state.get("use_pat")),
        provider=provider,
    )
    doc = read_toml_safe(CODEX_CONFIG_PATH)
    deep_merge_dict(doc, overlay)
    if provider:
        # deep_merge can't drop keys, so clear a `model` pinned by an earlier
        # non-provider run that the provider overlay omits.
        doc.pop("model", None)
    sync_smart_routing_hooks(
        doc,
        state,
        enabled=smart_routing_enabled(state) and provider is None,
    )
    write_toml_file(CODEX_CONFIG_PATH, doc)
    # use_as_global_settings: also write the modern overlay to Codex's OS managed config
    # (/etc/codex/managed_config.toml), the highest-precedence scope a bare `codex` reads — so it
    # defaults to the gateway without `--profile ucode`. codex auth self-refreshes via
    # `ucode auth-token`, so the file keeps working. The write goes through the sudo path in
    # `managed_files`.
    if state.get("write_managed_config"):
        _write_managed_config(
            workspace, chosen_model, databricks_profile, bool(state.get("use_pat")), provider
        )
    state = mark_tool_managed(state, "codex", MANAGED_KEYS)
    save_state(state)
    return state


def _is_gpt_family(model: str) -> bool:
    """Return True if this id is in the GPT family (versioned or OSS variants)."""
    tail = model.split("/")[-1]
    if tail.startswith("system.ai."):
        tail = tail[len("system.ai.") :]
    return tail.startswith("gpt-")


def _managed_config_path() -> Path | None:
    """OS-level Codex managed config file, or None on unsupported platforms.

    Linux and macOS use ``/etc/codex/managed_config.toml`` (root-owned, highest precedence). See
    https://learn.chatgpt.com/docs/enterprise/managed-configuration. Codex also supports a
    ``~/.codex/managed_config.toml`` on Windows, but ucode's write path is sudo/Unix-only
    (see :func:`managed_files.managed_files_supported`), so Windows returns None here too.
    """
    if current_os() in (OS.LINUX, OS.MACOS):
        return Path("/etc/codex/managed_config.toml")
    return None


def _write_managed_config(
    workspace: str,
    model: str | None,
    databricks_profile: str | None,
    use_pat: bool,
    provider: str | None,
) -> None:
    """Merge the modern overlay into Codex's OS managed_config.toml, preserving any other keys there.

    Written via the sudo path in `managed_files` (drift-suppressed).
    """
    path = _managed_config_path()
    if path is None:
        print_warning_err(
            "Machine-wide Codex settings aren't supported on this platform; skipped the managed "
            "config write."
        )
        return
    overlay = render_overlay(
        workspace, model, databricks_profile, use_pat=use_pat, provider=provider
    )
    doc = read_toml_safe(path)
    deep_merge_dict(doc, overlay)
    if provider:
        # deep_merge can't drop keys; clear a `model` a prior non-provider run pinned.
        doc.pop("model", None)
    write_managed_file(path, tomlkit.dumps(doc), display="Codex")


def default_model(state: dict) -> str | None:
    """Pick the best available codex model.

    A managed config's ``codex_default_model`` takes priority. Among versioned
    GPT ids (e.g. ``system.ai.gpt-5``, ``system.ai.gpt-5-6-luna``) the highest
    semantic version wins. When no versioned GPT is present but other codex-family
    ids are available (e.g. ``system.ai.gpt-oss-120b``), the first of those is
    used — UC model-services only places ids in the codex bucket when they expose
    the responses API, so any id there is routable.
    """
    if isinstance(state.get("codex_default_model"), str):
        return state.get("codex_default_model")
    codex_models = state.get("codex_models") or []
    parsed: list[tuple[str, tuple[int, int | None, int | None, str]]] = [
        (mid, gpt) for mid in codex_models if (gpt := _parse_gpt(mid)) is not None
    ]
    if parsed:

        def _gpt_version_key(entry: tuple[str, tuple[int, int | None, int | None, str]]):
            major, minor, patch, suffix = entry[1]
            base_bonus = 1 if not suffix else 0
            return (major, minor or 0, patch or 0, base_bonus)

        return max(parsed, key=_gpt_version_key)[0]

    # No versioned GPT found. Fall back to the first GPT-family id (gpt-*
    # after stripping the system.ai. prefix). gpt-oss-* models are confirmed
    # routable through the responses API; non-GPT ids (e.g. moonshotai/kimi-k2.5)
    # would be rejected by the gateway, so they stay excluded.
    gpt_family = [m for m in codex_models if _is_gpt_family(m)]
    return gpt_family[0] if gpt_family else None


# codex rejects the global --profile on subcommands that don't accept it
# (app-server, mcp-server, ...) with a CLI *parse-time* error — before it touches
# auth, the gateway, or the network — so the rejection exits almost instantly.
# We use that to decide when to retry without --profile (see launch()). This
# window is well above codex's ~0.15s cold-start floor and far below the seconds
# any real session needs to connect and then fail, so it never catches a genuine
# failure. Its exit code (1) is indistinguishable from an ordinary failure, so
# elapsed time is the signal we key on rather than stderr text.
_PROFILE_REJECTED_MAX_SECONDS = 3.0


def _generate_v2_app_server_home(state: dict, model: str) -> Path:
    """Write an isolated CODEX_HOME whose config.toml carries the ucode gateway
    provider block, for the ucode-run `codex app-server`.

    The app-server rejects the global `--profile`, so a default-config CODEX_HOME
    is how it inherits ucode's gateway (base_url + `ucode auth-token` refresh).
    Reuses `render_overlay` — the same provider block `ucode configure codex` writes."""
    home = SMART_ROUTING_V2_HOME
    home.mkdir(parents=True, exist_ok=True)
    config_path = home / "config.toml"
    overlay = render_overlay(
        state["workspace"],
        model,
        state.get("profile"),
        use_pat=bool(state.get("use_pat")),
    )
    doc = read_toml_safe(config_path)
    deep_merge_dict(doc, overlay)
    write_toml_file(config_path, doc)
    return home


def _launch_smart_routing_v2(state: dict, tool_args: list[str]) -> None:
    """Experimental single-command launch of the real Codex TUI with runtime model switching.

    ucode owns three processes: a `codex app-server` subprocess, the WebSocket interposer
    (daemon thread), and the `codex --remote` TUI (foreground). The interposer holds the first
    turn on the normal model, then rewrites subsequent turns to SMART_ROUTING_V2_TARGET_MODEL and
    injects a settings update so the TUI reflects the switch. The app-server + interposer are torn
    down when the TUI exits. Mirrors the lifecycle of `claude.py::_launch_relayed`.
    """
    from ucode.smart_routing import codex_interposer

    binary = SPEC["binary"]
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError(
            "Smart routing v2 needs a configured workspace; run `ucode configure codex` first."
        )
    start_model = default_model(state)
    if not start_model:
        raise RuntimeError(
            "Smart routing v2 could not determine a starting Codex model for this workspace."
        )

    os.environ["OAUTH_TOKEN"] = get_databricks_token(workspace, state.get("profile"))
    home = _generate_v2_app_server_home(state, start_model)
    app_port = codex_interposer.free_port()
    tui_port = codex_interposer.free_port()

    app_server = subprocess.Popen(
        [binary, "app-server", "--listen", f"ws://127.0.0.1:{app_port}"],
        env={**os.environ, "CODEX_HOME": str(home)},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    stop_interposer = None
    try:
        if not codex_interposer.wait_healthz(app_port, timeout=30):
            raise RuntimeError(
                "Codex app-server did not become ready for smart routing v2; check workspace auth."
            )
        _thread, stop_interposer = codex_interposer.start_interposer_thread(
            "127.0.0.1",
            tui_port,
            f"ws://127.0.0.1:{app_port}",
            SMART_ROUTING_V2_TARGET_MODEL,
            smart_routing_v2.SWITCH_AFTER_TURNS,
            switch_message=_smart_routing_switch_message(
                SMART_ROUTING_V2_TARGET_MODEL,
                SMART_ROUTING_V2_REASON,
            ),
            log_path=SMART_ROUTING_V2_LOG,
        )
        # Foreground TUI. Popen (not exec) so this process stays alive to tear down the
        # app-server + interposer when the TUI exits (see claude.py::_launch_relayed).
        tui = subprocess.Popen(
            [binary, "--remote", f"ws://127.0.0.1:{tui_port}", "--model", start_model, *tool_args]
        )
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
            app_server.wait(timeout=5)
        except Exception:  # noqa: BLE001 - the app-server must never linger
            app_server.kill()
    sys.exit(returncode)


def launch(state: dict, tool_args: list[str]) -> None:
    binary = SPEC["binary"]
    workspace = state.get("workspace")
    if smart_routing_v2.enabled():
        _launch_smart_routing_v2(state, tool_args)
        return
    if workspace:
        os.environ["OAUTH_TOKEN"] = get_databricks_token(workspace, state.get("profile"))
    # Run codex with --profile first — the TUI and runtime subcommands
    # (exec/resume/mcp/...) keep ucode's Databricks routing, including any added
    # by future codex versions. codex rejects the global --profile on
    # server-family subcommands (app-server, mcp-server, ...), which are
    # caller-configured anyway (e.g. omnigent runs `codex app-server` with its
    # own CODEX_HOME); on that rejection we relaunch without --profile.
    #
    # The retry is gated on the attempt failing *fast*: the rejection is a
    # parse-time error (~0.15s), whereas a session that actually starts can only
    # fail after a network round-trip (seconds). Without that gate a genuinely
    # failing `codex exec` would be silently re-run without --profile — i.e. on
    # the user's own OpenAI login instead of the Databricks gateway (ucode writes
    # a *named-profile* file, so no --profile means no ucode routing). stdio is
    # inherited (no capture), so Ctrl-C reaches codex directly and the resulting
    # KeyboardInterrupt propagates past the retry check — quitting an interactive
    # session is never mistaken for a --profile rejection.
    started = time.monotonic()
    returncode = subprocess.run([binary, "--profile", CODEX_PROFILE_NAME, *tool_args]).returncode
    if returncode != 0 and time.monotonic() - started < _PROFILE_REJECTED_MAX_SECONDS:
        # Fast failure: most likely codex rejected --profile on this subcommand.
        # Relaunch without it, handing over the terminal. (A fast failure for
        # any other reason — e.g. a bad flag — just re-fails the same way here,
        # with no ucode routing to lose since the subcommand had none.)
        #
        # Warn on *stderr*: this path is reached by `codex app-server`, whose
        # stdout is a JSON-RPC stream its caller parses. Emit before handing off,
        # since execvp replaces this process.
        print_warning_err(
            "ucode's `--profile` isn't accepted here (error above). Retrying "
            f"without it: this run uses {LEGACY_CODEX_CONFIG_PATH}, NOT the "
            "Databricks gateway."
        )
        exec_or_spawn([binary, *tool_args])
        return  # unreachable in production (exec replaces the process)
    sys.exit(returncode)


def smart_routing_enabled(state: dict) -> bool:
    """Return whether the current workspace opted into Codex routing."""
    return state.get(SMART_ROUTING_STATE_KEY) is True


def enable_smart_routing(state: dict) -> dict:
    """Persist the current workspace's Codex smart-routing opt-in."""
    parsed = _parse_version(agent_version(SPEC["binary"]))
    if parsed is not None and parsed < MINIMUM_ROUTING_CODEX_VERSION:
        raise RuntimeError(
            "Codex smart routing requires Codex "
            f"{MINIMUM_ROUTING_CODEX_VERSION_TEXT} or newer; found "
            f"{agent_version(SPEC['binary'])}."
        )
    state[SMART_ROUTING_STATE_KEY] = True
    return state


def disable_smart_routing(state: dict) -> bool:
    """Disable routing and remove only ucode's Codex routing hooks."""
    state.pop(SMART_ROUTING_STATE_KEY, None)
    if state.get("workspace"):
        save_state(state)
    changed = False
    for path in (CODEX_CONFIG_PATH, LEGACY_CODEX_CONFIG_PATH):
        if not path.exists():
            continue
        doc = read_toml_safe(path)
        if remove_smart_routing_hooks(doc):
            write_toml_file(path, doc)
            changed = True
    from ucode.smart_routing.codex_routing import clear_routing_artifacts

    clear_routing_artifacts()
    return changed


def validate_cmd(binary: str) -> list[str]:
    return [
        binary,
        "--profile",
        CODEX_PROFILE_NAME,
        "exec",
        "--skip-git-repo-check",
        "say hi in 5 words or less",
    ]
