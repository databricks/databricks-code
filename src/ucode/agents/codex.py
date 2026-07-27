"""Codex agent: writes ~/.codex/ucode.config.toml for Databricks-backed Codex."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

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
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version
from ucode.tracing import tracing_env
from ucode.ui import print_note, print_success, print_warning

CODEX_CONFIG_DIR = Path.home() / ".codex"
CODEX_PROFILE_NAME = "ucode"
CODEX_CONFIG_PATH = CODEX_CONFIG_DIR / f"{CODEX_PROFILE_NAME}.config.toml"
CODEX_BACKUP_PATH = APP_DIR / "codex-ucode-config.backup.toml"
LEGACY_CODEX_CONFIG_PATH = CODEX_CONFIG_DIR / "config.toml"
LEGACY_CODEX_BACKUP_PATH = APP_DIR / "codex-config.backup.toml"
CODEX_MODEL_PROVIDER_NAME = "ucode-databricks"
MINIMUM_CODEX_VERSION = (0, 134, 0)
MINIMUM_CODEX_VERSION_TEXT = "0.134.0"
MLFLOW_CODEX_PACKAGE = "@mlflow/codex"
# npm integration for Codex, built on the @mlflow/core TypeScript tracing SDK.
MLFLOW_CODEX_PACKAGE_SPEC = f"{MLFLOW_CODEX_PACKAGE}@^0.3.0"
MLFLOW_CODEX_BINARY = "mlflow-codex"
MINIMUM_MLFLOW_CODEX_VERSION = (0, 3, 0)
CODEX_TRACING_NOTIFY = [MLFLOW_CODEX_BINARY, "notify-hook"]
CODEX_TRACING_ENV_KEYS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_ID",
    "MLFLOW_TRACE_LOCATION",
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
TRACING_MANAGED_KEYS: list[list[str]] = [["notify"]]

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


def _version_at_least(version: tuple[int, int, int] | None, minimum: tuple[int, int, int]) -> bool:
    return version is not None and version >= minimum


def _installed_version_status() -> tuple[str, bool] | None:
    version = agent_version(SPEC["binary"])
    parsed = _parse_version(version)
    if parsed is None:
        return None
    return version, parsed < MINIMUM_CODEX_VERSION


def _installed_mlflow_codex_version() -> tuple[int, int, int] | None:
    if not shutil.which("npm"):
        return None
    try:
        result = subprocess.run(
            ["npm", "list", "-g", MLFLOW_CODEX_PACKAGE, "--json", "--depth=0"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    deps = payload.get("dependencies")
    if not isinstance(deps, dict):
        return None
    package = deps.get(MLFLOW_CODEX_PACKAGE)
    if not isinstance(package, dict):
        return None
    version = package.get("version")
    return _parse_version(version) if isinstance(version, str) else None


def ensure_tracing_runtime() -> bool:
    """Ensure the UC-capable stock MLflow Codex hook is installed globally."""
    current = _installed_mlflow_codex_version()
    if _version_at_least(current, MINIMUM_MLFLOW_CODEX_VERSION) and shutil.which(
        MLFLOW_CODEX_BINARY
    ):
        return True

    if not shutil.which("npm"):
        print_warning(
            f"Codex tracing needs {MLFLOW_CODEX_PACKAGE_SPEC}, but npm is not available. "
            "Install npm, then re-run `ucode configure tracing`."
        )
        return False

    print_note(f"Installing {MLFLOW_CODEX_PACKAGE_SPEC} for Codex tracing...")
    try:
        subprocess.run(
            ["npm", "install", "-g", MLFLOW_CODEX_PACKAGE_SPEC],
            check=True,
            timeout=300,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print_warning(f"Could not install {MLFLOW_CODEX_PACKAGE_SPEC}: {exc}")
        return False

    installed = _installed_mlflow_codex_version()
    if not _version_at_least(installed, MINIMUM_MLFLOW_CODEX_VERSION):
        print_warning(f"npm did not install a compatible {MLFLOW_CODEX_PACKAGE} runtime")
        return False
    if not shutil.which(MLFLOW_CODEX_BINARY):
        print_warning(
            f"Installed {MLFLOW_CODEX_PACKAGE_SPEC}, but `{MLFLOW_CODEX_BINARY}` is not on PATH"
        )
        return False

    print_success("MLflow Codex tracing runtime ready")
    return True


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


def _is_tracing_notify(value: object) -> bool:
    if not isinstance(value, Iterable):
        return False
    items = list(value)  # tomlkit arrays are iterable but not plain lists.
    if not all(isinstance(item, str) for item in items):
        return False
    return items == CODEX_TRACING_NOTIFY


def _apply_tracing_notify(doc: dict, enabled: bool) -> bool:
    existing = doc.get("notify")
    if enabled:
        if existing is not None and not _is_tracing_notify(existing):
            print_warning(
                "Replacing existing Codex `notify` hook with MLflow tracing; "
                "the previous value is preserved in ucode's config backup."
            )
        doc["notify"] = list(CODEX_TRACING_NOTIFY)
        return True

    if _is_tracing_notify(existing):
        doc.pop("notify", None)
        return True
    return False


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
    tracing_enabled = bool(tracing_env(state, "codex"))

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
        _apply_tracing_notify(doc, tracing_enabled)
        if provider:
            # deep_merge can't drop keys, so clear a `model` pinned by an
            # earlier non-provider run that the provider overlay omits.
            profiles = doc.get("profiles")
            if isinstance(profiles, dict) and isinstance(profiles.get(CODEX_PROFILE_NAME), dict):
                profiles[CODEX_PROFILE_NAME].pop("model", None)
        write_toml_file(LEGACY_CODEX_CONFIG_PATH, doc)
        managed_keys = list(LEGACY_MANAGED_KEYS)
        if tracing_enabled:
            managed_keys += TRACING_MANAGED_KEYS
        state = mark_tool_managed(state, "codex", managed_keys)
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
    _apply_tracing_notify(doc, tracing_enabled)
    if provider:
        # deep_merge can't drop keys, so clear a `model` pinned by an earlier
        # non-provider run that the provider overlay omits.
        doc.pop("model", None)
    write_toml_file(CODEX_CONFIG_PATH, doc)
    managed_keys = list(MANAGED_KEYS)
    if tracing_enabled:
        managed_keys += TRACING_MANAGED_KEYS
    state = mark_tool_managed(state, "codex", managed_keys)
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
    for key in CODEX_TRACING_ENV_KEYS:
        os.environ.pop(key, None)
    if workspace:
        token = get_databricks_token(workspace, state.get("profile"))
        os.environ["OAUTH_TOKEN"] = token
        tracing_env_vars = tracing_env(state, "codex")
        if tracing_env_vars:
            if not shutil.which(MLFLOW_CODEX_BINARY):
                raise RuntimeError(
                    f"Codex tracing is enabled, but `{MLFLOW_CODEX_BINARY}` is not on PATH. "
                    f"Run `ucode configure tracing` or `npm install -g {MLFLOW_CODEX_PACKAGE_SPEC}`."
                )
            os.environ["DATABRICKS_HOST"] = workspace
            os.environ["DATABRICKS_TOKEN"] = token
            if state.get("profile"):
                os.environ["DATABRICKS_CONFIG_PROFILE"] = str(state["profile"])
            os.environ.update(tracing_env_vars)
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
