"""Codex agent: writes ~/.codex/ucode.config.toml for Databricks-backed Codex."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import time
from pathlib import Path

from ucode.agent_updates import available_npm_package_update
from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    deep_merge_dict,
    read_toml_safe,
    restore_file,
    write_toml_file,
)
from ucode.databricks import (
    build_auth_token_argv,
    build_tool_base_url,
    get_databricks_token,
)
from ucode.launcher import exec_or_spawn
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version
from ucode.ui import print_note, print_success, print_warning, prompt_yes_no

CODEX_CONFIG_DIR = Path.home() / ".codex"
CODEX_PROFILE_NAME = "ucode"
CODEX_CONFIG_PATH = CODEX_CONFIG_DIR / f"{CODEX_PROFILE_NAME}.config.toml"
CODEX_BACKUP_PATH = APP_DIR / "codex-ucode-config.backup.toml"
LEGACY_CODEX_CONFIG_PATH = CODEX_CONFIG_DIR / "config.toml"
LEGACY_CODEX_BACKUP_PATH = APP_DIR / "codex-config.backup.toml"
CODEX_MODEL_PROVIDER_NAME = "ucode-databricks"
MINIMUM_CODEX_VERSION = (0, 134, 0)
MINIMUM_CODEX_VERSION_TEXT = "0.134.0"

# ── Codex desktop app ──────────────────────────────────────────────────────
# The Codex desktop app has no `--profile`/`-c` command-line surface (unlike the
# CLI), so the only way to steer it is the *root* `~/.codex/config.toml` it reads
# on startup. `ucode codex-app` therefore edits that shared file in place —
# backing it up first — instead of writing an isolated per-profile file. A
# distinct provider name keeps the app's block from colliding with the CLI's
# `ucode-databricks` provider on the same file.
CODEX_APP_CONFIG_PATH = CODEX_CONFIG_DIR / "config.toml"
CODEX_APP_BACKUP_PATH = APP_DIR / "codex-app-config.backup.toml"
CODEX_APP_MODEL_PROVIDER_NAME = "ucode-databricks-app"
CODEX_APP_BUNDLE_ID = "com.openai.codex"
# Seconds to wait for the app to quit before reopening it (mirrors Ollama).
CODEX_APP_QUIT_TIMEOUT = 5.0

CODEX_APP_MANAGED_KEYS: list[list[str]] = [
    ["model_provider"],
    ["model"],
    ["model_providers", CODEX_APP_MODEL_PROVIDER_NAME],
    ["model_providers", CODEX_APP_MODEL_PROVIDER_NAME, "http_headers"],
]


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

# These models should use the Databricks ID, not the OpenAI ID, as the OpenAI
# ID is incompatible with Codex.
CODEX_OPENAI_ID_INCOMPATIBLE_MODELS = {
    "databricks-gpt-5-2-codex",
    "databricks-gpt-5-4-nano",
}


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
    provider_name: str = CODEX_MODEL_PROVIDER_NAME,
) -> dict:
    overlay: dict = {"model_provider": provider_name}
    if model:
        overlay["model"] = model
    overlay["model_providers"] = {
        provider_name: _provider_block(workspace, databricks_profile, use_pat, provider),
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


def _openai_model_id(model: str | None) -> str | None:
    """Map Databricks GPT endpoint ids to OpenAI model ids for Codex metadata."""
    parsed = _parse_gpt(model)
    if parsed is None:
        return model
    major, minor, patch, suffix = parsed
    version = str(major)
    if minor is not None:
        version += f".{minor}"
    if patch is not None:
        version += f".{patch}"
    return f"gpt-{version}{suffix}"


def _codex_model_id(model: str | None) -> str | None:
    # UC model-services ids (`system.ai.gpt-5`) route by name through the
    # gateway, so they must be sent verbatim — not rewritten to an OpenAI id.
    if model and model.startswith("system.ai."):
        return model
    if model in CODEX_OPENAI_ID_INCOMPATIBLE_MODELS:
        return model
    return _openai_model_id(model)


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
    # Databricks endpoint id is pinned.
    chosen_model = None if provider else _codex_model_id(model or default_model(state))
    databricks_profile = state.get("profile")

    if _use_legacy_layout():
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
    write_toml_file(CODEX_CONFIG_PATH, doc)
    state = mark_tool_managed(state, "codex", MANAGED_KEYS)
    save_state(state)
    return state


def default_model(state: dict) -> str | None:
    """Pick the newest GPT model when multiple are available.

    The discovery list is alphabetically sorted, which can put
    "databricks-gpt-5" ahead of "databricks-gpt-5-5". Prefer the
    highest semantic version instead.

    Only GPT-parseable ids are considered. Codex routes the chosen ``model``
    through the gateway as-is, so a non-GPT entry (e.g. ``moonshotai/kimi-k2.5``)
    would be rejected with a Unity Catalog endpoint-name error. When no
    candidate parses as GPT we return None rather than pinning an unroutable id.
    """
    codex_models = state.get("codex_models") or []
    parsed: list[tuple[str, tuple[int, int | None, int | None, str]]] = [
        (mid, gpt) for mid in codex_models if (gpt := _parse_gpt(mid)) is not None
    ]
    if not parsed:
        return None

    def _gpt_version_key(entry: tuple[str, tuple[int, int | None, int | None, str]]):
        major, minor, patch, suffix = entry[1]
        base_bonus = 1 if not suffix else 0
        return (major, minor or 0, patch or 0, base_bonus)

    return max(parsed, key=_gpt_version_key)[0]


def launch(state: dict, tool_args: list[str]) -> None:
    binary = SPEC["binary"]
    workspace = state.get("workspace")
    if workspace:
        os.environ["OAUTH_TOKEN"] = get_databricks_token(workspace, state.get("profile"))
    exec_or_spawn([binary, "--profile", CODEX_PROFILE_NAME, *tool_args])


def validate_cmd(binary: str) -> list[str]:
    return [
        binary,
        "--profile",
        CODEX_PROFILE_NAME,
        "exec",
        "--skip-git-repo-check",
        "say hi in 5 words or less",
    ]


# ── Codex desktop app config + launch ──────────────────────────────────────


def write_app_config(state: dict, model: str | None = None, provider: str | None = None) -> dict:
    """Point the Codex desktop app at the Databricks gateway via its root config.

    The app reads only `~/.codex/config.toml`, so — unlike the CLI's isolated
    per-profile file — we deep-merge ucode's provider block, `model_provider`,
    and (unless routing through a provider) `model` into that shared file. The
    file is backed up first so `ucode revert` / `ucode codex-app --restore` can
    put the user's original config back.
    """
    workspace = state["workspace"]
    # A Model Provider Service routes by header, so pin no Databricks model.
    chosen_model = None if provider else _codex_model_id(model or default_model(state))
    databricks_profile = state.get("profile")

    backup_existing_file(CODEX_APP_CONFIG_PATH, CODEX_APP_BACKUP_PATH)
    overlay = render_overlay(
        workspace,
        chosen_model,
        databricks_profile,
        use_pat=bool(state.get("use_pat")),
        provider=provider,
        provider_name=CODEX_APP_MODEL_PROVIDER_NAME,
    )
    doc = read_toml_safe(CODEX_APP_CONFIG_PATH)
    deep_merge_dict(doc, overlay)
    if provider:
        # deep_merge can't drop keys, so clear a `model` pinned by an earlier
        # non-provider run that the provider overlay omits.
        doc.pop("model", None)
    write_toml_file(CODEX_APP_CONFIG_PATH, doc)
    state = mark_tool_managed(state, "codex-app", CODEX_APP_MANAGED_KEYS)
    save_state(state)
    return state


def revert_app_config(managed: bool = True) -> bool:
    """Restore the desktop app's root config from ucode's backup.

    Returns True if the file was restored (or removed when ucode created it and
    there was no prior config to back up). Mirrors how every other ucode tool
    reverts — whole-file restore from the first-ever snapshot.
    """
    return restore_file(CODEX_APP_CONFIG_PATH, CODEX_APP_BACKUP_PATH, managed)


def _app_is_running() -> bool:
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["osascript", "-e", 'tell application "System Events" to exists process "Codex"'],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() == "true"
        if system == "Windows":
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq Codex.exe"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return "Codex.exe" in result.stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return False


def _quit_app() -> None:
    system = platform.system()
    if system == "Darwin":
        subprocess.run(
            ["osascript", "-e", 'tell application "Codex" to quit'],
            capture_output=True,
            timeout=10,
        )
    elif system == "Windows":
        subprocess.run(
            ["taskkill", "/IM", "Codex.exe"],
            capture_output=True,
            timeout=10,
        )


def _open_app(workdir: str | None = None) -> bool:
    """Open the Codex desktop app, in ``workdir`` when given.

    A GUI app launched via ``open``/``start`` does not inherit the terminal's
    working directory (the launch is handed to launchd / the shell), so Codex
    would otherwise open its own default folder. Passing the directory as a
    positional argument tells Codex which repo to open.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            argv = ["open", "-b", CODEX_APP_BUNDLE_ID]
            if workdir:
                argv.append(workdir)
            result = subprocess.run(argv, capture_output=True, timeout=15)
            return result.returncode == 0
        if system == "Windows":
            # `start` is a cmd builtin; launch the Codex app by name, passing the
            # directory as an argument so it opens there.
            argv = ["cmd", "/c", "start", "", "Codex.exe"]
            if workdir:
                argv.append(workdir)
            result = subprocess.run(argv, capture_output=True, timeout=15)
            return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
    return False


def _warn_app_not_found() -> None:
    print_warning(
        "Couldn't open the Codex desktop app — is it installed? Install it from "
        "https://developers.openai.com/codex, then re-run `ucode codex-app`. "
        "Your config was written, so it will take effect the next time Codex starts."
    )


def _wait_for_app_exit(timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _app_is_running():
            return True
        time.sleep(0.2)
    return not _app_is_running()


def _restart_app(prompt: str, workdir: str | None = None) -> None:
    """Quit and reopen the app (with confirmation) so it re-reads config.toml."""
    if not prompt_yes_no(prompt):
        print_note("Quit and reopen Codex when you're ready for the change to take effect.")
        return
    _quit_app()
    if not _wait_for_app_exit(CODEX_APP_QUIT_TIMEOUT):
        print_warning("Codex did not quit; quit it manually, then reopen it to apply the change.")
        return
    if _open_app(workdir):
        print_success("Codex restarted.")
    else:
        _warn_app_not_found()


def _ensure_app_supported() -> None:
    if platform.system() not in ("Darwin", "Windows"):
        raise RuntimeError(
            "`ucode codex-app` is only supported on macOS and Windows. On Linux, use "
            "`ucode codex` for the CLI instead."
        )


def launch_app(state: dict) -> None:
    """Open the Codex desktop app, restarting it if it's already running.

    The app only re-reads `config.toml` on startup, so when it's already open we
    prompt to quit + reopen; declining leaves the config written for next launch.
    """
    _ensure_app_supported()
    workdir = os.getcwd()
    if not _app_is_running():
        if _open_app(workdir):
            print_success("Codex is now configured to use Databricks. Opening the app...")
        else:
            _warn_app_not_found()
        return
    _restart_app("Codex is running. Restart it to use Databricks?", workdir)


def restart_app_after_restore() -> None:
    """Prompt to restart a running app so it picks up a just-restored config.

    No-op when the app isn't running (it will read the restored config on its
    next launch) or on unsupported platforms."""
    if platform.system() not in ("Darwin", "Windows") or not _app_is_running():
        return
    _restart_app("Codex is running. Restart it to use your restored config?")
