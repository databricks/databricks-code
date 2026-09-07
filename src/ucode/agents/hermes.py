"""Hermes adapter: render Databricks Gateway config and apply it through Hermes."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import cast

from ucode.config_io import APP_DIR, ToolSpec, is_dry_run
from ucode.databricks import build_auth_shell_command, build_tool_base_url
from ucode.launcher import exec_or_spawn
from ucode.ui import console

PROVIDER_ID = "ucode-databricks-codex"
ANTHROPIC_PROVIDER_ID = "ucode-databricks-anthropic"
OSS_PROVIDER_ID = "ucode-databricks-oss"
GEMINI_PROVIDER_ID = "ucode-databricks-gemini"
MANAGED_PROVIDER_IDS = (
    PROVIDER_ID,
    ANTHROPIC_PROVIDER_ID,
    OSS_PROVIDER_ID,
    GEMINI_PROVIDER_ID,
)
DISPLAY_NAME = "Databricks Model Serving"
UNSUPPORTED_MODELS = frozenset(
    {
        "system.ai.gpt-oss-20b",
        "system.ai.gpt-oss-120b",
    }
)


class HermesConfigApplyError(RuntimeError):
    """Hermes configuration may have changed before the operation failed."""


SPEC: ToolSpec = {
    "binary": "hermes",
    "package": "",
    "display": "Hermes",
    "config_path": Path.home() / ".hermes" / "config.yaml",
    "backup_path": APP_DIR / "hermes-config.backup.yaml",
    "install_method": "external",
}

MANAGED_PATHS = (
    f"providers.{PROVIDER_ID}",
    f"providers.{ANTHROPIC_PROVIDER_ID}",
    f"providers.{OSS_PROVIDER_ID}",
    f"providers.{GEMINI_PROVIDER_ID}",
    "model.provider",
    "model.default",
)


def default_model(state: dict) -> str | None:
    """Choose the first model from Hermes' supported protocol families."""
    responses_model = state.get("codex_default_model")
    if not isinstance(responses_model, str):
        codex_models = state.get("codex_models")
        responses_model = (
            max(
                (model for model in codex_models if isinstance(model, str)),
                key=_gpt_model_version,
                default=None,
            )
            if isinstance(codex_models, list)
            else None
        )
    if responses_model and responses_model not in UNSUPPORTED_MODELS:
        return responses_model
    claude_models = state.get("claude_models")
    if isinstance(claude_models, dict):
        for model in claude_models.values():
            if isinstance(model, str) and model and model not in UNSUPPORTED_MODELS:
                return model
    for key in ("gemini_models", "oss_models"):
        models = state.get(key)
        if isinstance(models, list):
            for model in models:
                if isinstance(model, str) and model and model not in UNSUPPORTED_MODELS:
                    return model
    return None


def _gpt_model_version(model: str) -> tuple[int, int, int, int]:
    """Order discovered GPT Responses models without changing Codex defaults."""
    match = re.fullmatch(r"(?:system\.ai\.)?gpt-(\d+)(?:-(\d+))?(?:-(\d+))?(-.*)?", model)
    if not match:
        return (0, 0, 0, 0)
    major, minor, patch, suffix = match.groups()
    return int(major), int(minor or 0), int(patch or 0), 1 if suffix is None else 0


def state_scoped_to_home(state: dict, hermes_home: str | Path) -> dict:
    """Hide ownership established for a different Hermes profile."""
    managed = state.get("managed_configs")
    hermes_managed = managed.get("hermes") if isinstance(managed, dict) else None
    recorded_home = hermes_managed.get("hermes_home") if isinstance(hermes_managed, dict) else None
    target = str(Path(hermes_home).expanduser().resolve())
    if isinstance(recorded_home, str) and str(Path(recorded_home).expanduser().resolve()) == target:
        return state
    scoped = copy.deepcopy(state)
    scoped_managed = scoped.get("managed_configs")
    if isinstance(scoped_managed, dict):
        scoped_managed.pop("hermes", None)
    return scoped


def is_update_available() -> None:
    """Hermes is installed externally; ucode never manages its version."""
    return None


def render_config_patch(
    state: dict,
    model: str | None = None,
) -> dict:
    """Return the non-secret, Hermes-owned multi-protocol transaction patch.

    Each model family is isolated behind its matching wire transport.
    """
    workspace = state["workspace"]
    claude_models = state.get("claude_models")
    anthropic_models = [
        candidate
        for candidate in _unique_models(
            claude_models.values() if isinstance(claude_models, dict) else None
        )
        if candidate not in UNSUPPORTED_MODELS
    ]
    anthropic_set = set(anthropic_models)
    oss_models = [
        candidate
        for candidate in _unique_models(state.get("oss_models"))
        if candidate not in anthropic_set and candidate not in UNSUPPORTED_MODELS
    ]
    specific_models = anthropic_set | set(oss_models)
    gemini_models = [
        model
        for model in _unique_models(state.get("gemini_models"))
        if model not in specific_models and model not in UNSUPPORTED_MODELS
    ]
    specific_models.update(gemini_models)
    responses_models = [
        candidate
        for candidate in _unique_models(state.get("codex_models"))
        if candidate not in specific_models and candidate not in UNSUPPORTED_MODELS
    ]
    if model in UNSUPPORTED_MODELS:
        raise RuntimeError(f"Model {model!r} is not supported by Hermes.")
    selected = model or default_model(state)
    if selected in UNSUPPORTED_MODELS:
        selected = None
    selected = selected or next(
        (
            candidate
            for candidates in (
                responses_models,
                anthropic_models,
                gemini_models,
                oss_models,
            )
            for candidate in candidates
        ),
        None,
    )
    if not selected:
        raise RuntimeError("No supported models are available for Hermes.")

    provider_specs = [
        (PROVIDER_ID, build_tool_base_url("codex", workspace), "codex_responses", responses_models),
        (
            ANTHROPIC_PROVIDER_ID,
            build_tool_base_url("claude", workspace),
            "anthropic_messages",
            anthropic_models,
        ),
        (OSS_PROVIDER_ID, f"{workspace}/ai-gateway/mlflow/v1", "openai_chat", oss_models),
        (
            GEMINI_PROVIDER_ID,
            f"{build_tool_base_url('gemini', workspace)}/v1beta",
            "gemini-native",
            gemini_models,
        ),
    ]
    selected_provider = PROVIDER_ID
    for provider_id, _, _, compatible_models in provider_specs[1:]:
        if selected in compatible_models:
            selected_provider = provider_id
            break
    if selected_provider == PROVIDER_ID and selected not in responses_models:
        # Preserve Phase 1's explicit-model behavior without guessing a route
        # from the model name. Only discovered family membership may switch
        # the wire protocol.
        responses_models.append(selected)

    key_cmd = build_auth_shell_command(
        workspace,
        state.get("profile"),
        use_pat=bool(state.get("use_pat")),
    )
    provider_paths = {}
    for provider_id, api, transport, compatible_models in provider_specs:
        if not compatible_models:
            continue
        default = selected if provider_id == selected_provider else compatible_models[0]
        provider_config: dict = {
            "name": DISPLAY_NAME,
            "api": api,
            "transport": transport,
            "key_cmd": key_cmd,
            "default_model": default,
            "models": {model_id: {} for model_id in compatible_models},
            "discover_models": False,
        }
        provider_paths[f"providers.{provider_id}"] = provider_config

    managed = state.get("managed_configs")
    hermes_managed = managed.get("hermes") if isinstance(managed, dict) else None
    raw_fingerprints = (
        hermes_managed.get("provider_fingerprints") if isinstance(hermes_managed, dict) else None
    )
    owned_fingerprints: dict = raw_fingerprints if isinstance(raw_fingerprints, dict) else {}
    provider_expect_hashes = {
        path: owned_fingerprints[provider_id]
        for path in provider_paths
        if (provider_id := path.removeprefix("providers.")) in owned_fingerprints
    }
    owned_model = hermes_managed.get("active_model") if isinstance(hermes_managed, dict) else None
    if (
        isinstance(owned_model, dict)
        and isinstance(owned_model.get("provider"), str)
        and isinstance(owned_model.get("default"), str)
    ):
        provider_expect_hashes.update(
            {
                "model.provider": config_value_fingerprint(owned_model["provider"]),
                "model.default": config_value_fingerprint(owned_model["default"]),
            }
        )
    provider_expect_missing = [
        path for path in provider_paths if path not in provider_expect_hashes
    ]
    missing_provider_paths = [path for path in MANAGED_PATHS[:4] if path not in provider_paths]
    return {
        "set": {
            **provider_paths,
            "model.provider": selected_provider,
            "model.default": selected,
        },
        "unset": [],
        "expect_missing": provider_expect_missing,
        "expect_hashes": provider_expect_hashes,
        "unset_if_hash": {
            path: owned_fingerprints[path.removeprefix("providers.")]
            for path in missing_provider_paths
            if path.removeprefix("providers.") in owned_fingerprints
        },
    }


def _unique_models(models) -> list[str]:
    """Preserve Gateway discovery order while dropping invalid duplicates."""
    return list(dict.fromkeys(model for model in models or [] if isinstance(model, str) and model))


def render_unconfigure_patch(
    *,
    current_model: dict,
    owned_model: dict,
    owned_provider_fingerprints: dict,
    current_provider_fingerprint: str | None,
) -> dict:
    """Remove managed paths without clobbering a later user selection."""
    current_provider = current_model.get("provider")
    owned_pair_matches = current_model == owned_model
    owned_provider_fingerprint = owned_provider_fingerprints.get(current_provider)
    provider_still_owned = (
        isinstance(owned_provider_fingerprint, str)
        and current_provider_fingerprint == owned_provider_fingerprint
    )
    provider_paths = list(MANAGED_PATHS[:4])
    if owned_pair_matches and provider_still_owned:
        active_provider_path = f"providers.{current_provider}"
        unset = [active_provider_path, *MANAGED_PATHS[4:]]
        expect_hashes = {
            active_provider_path: owned_provider_fingerprint,
            "model.provider": config_value_fingerprint(owned_model["provider"]),
            "model.default": config_value_fingerprint(owned_model["default"]),
        }
        provider_paths = [path for path in provider_paths if path != active_provider_path]
    elif isinstance(current_provider, str):
        active_provider_path = f"providers.{current_provider}"
        provider_paths = [path for path in provider_paths if path != active_provider_path]
        unset = []
        expect_hashes = {}
    else:
        unset = []
        expect_hashes = {}
    return {
        "set": {},
        "unset": unset,
        "expect_hashes": expect_hashes,
        "unset_if_hash": {
            path: owned_provider_fingerprints[path.removeprefix("providers.")]
            for path in provider_paths
            if path.removeprefix("providers.") in owned_provider_fingerprints
        },
    }


def render_mcp_server_patch(
    name: str,
    argv: list[str],
    *,
    expected_fingerprint: str | None = None,
) -> dict:
    """Render one Hermes stdio MCP entry backed by ucode's refreshable proxy."""
    if not argv:
        raise ValueError("MCP proxy argv must not be empty.")
    path = f"mcp_servers.{name}"
    patch: dict[str, object] = {
        "set": {
            path: {
                "command": argv[0],
                "args": list(argv[1:]),
            }
        },
        "unset": [],
    }
    if expected_fingerprint is None:
        patch["expect_missing"] = [path]
    else:
        patch["expect_hashes"] = {path: expected_fingerprint}
    return patch


def config_value_fingerprint(value: object) -> str:
    """Hash one canonical Hermes config value without persisting its contents."""

    def validate(item: object) -> None:
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("Config fingerprints require string mapping keys")
            for nested in item.values():
                validate(nested)
            return
        if isinstance(item, list):
            for nested in item:
                validate(nested)
            return
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Config fingerprints require finite numbers")
        if item is not None and not isinstance(item, (str, int, float, bool)):
            raise ValueError("Unsupported config fingerprint value")

    validate(value)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mcp_server_fingerprint(value: dict) -> str:
    return config_value_fingerprint(value)


def mcp_value_for_argv(argv: list[str]) -> dict:
    return {"command": argv[0], "args": argv[1:]}


def apply_config_patch(patch: dict, *, hermes_home: str | Path) -> dict:
    """Apply a conservative patch through Hermes's public config commands.

    Hermes does not currently expose a multi-key transaction. Preconditions
    therefore prevent known collisions but cannot eliminate a concurrent
    read-to-write race. On uncertainty, cleanup leaves the value in place.
    """
    env = os.environ.copy()
    env["HERMES_HOME"] = str(Path(hermes_home).expanduser().resolve())

    def run(arguments: list[str]) -> subprocess.CompletedProcess:
        argv = [SPEC["binary"], "config", *arguments]
        try:
            return subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                env=env,
            )
        except OSError as exc:
            raise RuntimeError("Hermes configuration could not be started.") from exc

    def read_if_set(path: str) -> tuple[bool, object | None]:
        completed = run(["get", path, "--json"])
        if completed.returncode != 0:
            if "Config key not set:" in (completed.stderr or "") or completed.stderr == "missing":
                return False, None
            raise RuntimeError(f"Hermes configuration read failed for {path!r}.")
        try:
            return True, json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Hermes returned invalid configuration for {path!r}.") from exc

    configured = patch.get("set") or {}
    unconditional_unsets = patch.get("unset") or []
    expect_missing = patch.get("expect_missing") or []
    expect_hashes = patch.get("expect_hashes") or {}
    unset_if_hash = patch.get("unset_if_hash") or {}
    if not isinstance(configured, dict) or not isinstance(unconditional_unsets, list):
        raise RuntimeError("Invalid Hermes configuration patch.")

    if is_dry_run():
        plan = {
            "set": list(configured),
            "unset": list(dict.fromkeys([*unconditional_unsets, *unset_if_hash])),
        }
        console.print(
            f"\n[bold]\\[dry run] Hermes config operations[/bold]\n{json.dumps(plan, indent=2)}"
        )
        return {"status": "dry_run", "paths": plan}

    for path in expect_missing:
        found, _value = read_if_set(path)
        if found:
            raise RuntimeError(f"Hermes configuration value {path!r} already exists.")
    for path, expected in expect_hashes.items():
        found, value = read_if_set(path)
        if not found or config_value_fingerprint(value) != expected:
            raise RuntimeError(f"Hermes configuration value {path!r} changed; refusing update.")

    conditional_unsets = []
    for path, expected in unset_if_hash.items():
        found, value = read_if_set(path)
        if found and config_value_fingerprint(value) == expected:
            conditional_unsets.append(path)

    model_paths = {"model.provider", "model.default"}
    provider_sets = [(path, value) for path, value in configured.items() if path not in model_paths]
    model_sets = [
        (path, configured[path])
        for path in ("model.provider", "model.default")
        if path in configured
    ]
    applied_sets = []
    applied_unsets = []

    def cli_value(value: object) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

    mutation_started = False
    try:
        for path, value in provider_sets:
            completed = run(["set", path, cli_value(value), "--force"])
            mutation_started = True
            if completed.returncode != 0:
                raise RuntimeError(f"Hermes configuration write failed for {path!r}.")
            applied_sets.append(path)

        for path in dict.fromkeys([*unconditional_unsets, *conditional_unsets]):
            found, current_value = read_if_set(path)
            if not found:
                continue
            expected = expect_hashes.get(path) or unset_if_hash.get(path)
            if expected is not None and config_value_fingerprint(current_value) != expected:
                continue
            completed = run(["unset", path])
            mutation_started = True
            if completed.returncode != 0:
                raise RuntimeError(f"Hermes configuration cleanup failed for {path!r}.")
            applied_unsets.append(path)

        for path, value in model_sets:
            completed = run(["set", path, cli_value(value), "--force"])
            mutation_started = True
            if completed.returncode != 0:
                raise RuntimeError(f"Hermes configuration write failed for {path!r}.")
            applied_sets.append(path)
    except Exception as exc:
        if mutation_started:
            raise HermesConfigApplyError(str(exc)) from exc
        raise

    return {
        "status": "applied",
        "paths": {"set": applied_sets, "unset": applied_unsets},
    }


def read_config_value(dotted_key: str, *, hermes_home: str | Path) -> object:
    """Read one value from an explicit Hermes profile."""
    target = Path(hermes_home).expanduser().resolve()
    env = dict(os.environ)
    env["HERMES_HOME"] = str(target)
    try:
        completed = subprocess.run(
            [SPEC["binary"], "config", "get", dotted_key, "--json"],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            env=env,
        )
    except OSError as exc:
        raise RuntimeError("Hermes configuration could not be read.") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"Hermes configuration read failed with exit code {completed.returncode}."
        )
    try:
        return json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Hermes returned invalid configuration.") from exc


def read_active_model(*, hermes_home: str | Path) -> dict:
    """Read the remaining active-model fields from one explicit profile."""
    model = read_config_value("model", hermes_home=hermes_home)
    if not isinstance(model, dict):
        raise RuntimeError("Hermes returned invalid model configuration.")
    model_values = cast(dict[str, object], model)
    return {
        key: value
        for key in ("provider", "default")
        if isinstance((value := model_values.get(key)), str)
    }


def write_tool_config(
    state: dict,
    model: str | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> dict:
    """Render and apply Hermes configuration, then record cleanup ownership."""
    target = Path(hermes_home or os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    patch = render_config_patch(
        state_scoped_to_home(state, target),
        model=model,
    )
    previous_state = copy.deepcopy(state)
    pending_state = record_config_ownership(copy.deepcopy(state), patch, target)
    from ucode.state import save_state

    save_state(pending_state)
    try:
        apply_config_patch(patch, hermes_home=target)
    except Exception as exc:
        if not isinstance(exc, HermesConfigApplyError):
            try:
                save_state(previous_state)
            except Exception:
                pass
        raise
    return pending_state


def record_config_ownership(state: dict, patch: dict, hermes_home: str | Path) -> dict:
    """Record only the non-secret values needed for surgical cleanup."""
    from ucode.state import mark_tool_managed

    configured = patch["set"]
    provider_fingerprints = {
        path.removeprefix("providers."): config_value_fingerprint(value)
        for path, value in configured.items()
        if path.startswith("providers.")
    }
    return mark_tool_managed(
        state,
        "hermes",
        [],
        metadata={
            "hermes_home": str(Path(hermes_home).expanduser().resolve()),
            "active_model": {
                "provider": configured["model.provider"],
                "default": configured["model.default"],
            },
            "provider_fingerprints": provider_fingerprints,
        },
    )


def unconfigure(
    *,
    hermes_home: str | Path | None = None,
    owned_model: dict,
    owned_provider_fingerprints: dict,
) -> dict:
    target = Path(hermes_home or os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    current_model = read_active_model(hermes_home=target)
    current_provider = current_model.get("provider")
    current_provider_fingerprint = None
    if isinstance(current_provider, str) and current_provider in owned_provider_fingerprints:
        current_provider_value = read_config_value(
            f"providers.{current_provider}",
            hermes_home=target,
        )
        current_provider_fingerprint = config_value_fingerprint(current_provider_value)
    patch = render_unconfigure_patch(
        current_model=current_model,
        owned_model=owned_model,
        owned_provider_fingerprints=owned_provider_fingerprints,
        current_provider_fingerprint=current_provider_fingerprint,
    )
    return apply_config_patch(patch, hermes_home=target)


def write_mcp_server_config(
    name: str,
    argv: list[str],
    *,
    hermes_home: str | Path | None = None,
    expected_fingerprint: str | None = None,
) -> bool:
    """Idempotently upsert one managed server in the active Hermes profile."""
    target = hermes_home or os.environ.get("HERMES_HOME") or Path.home() / ".hermes"
    apply_config_patch(
        render_mcp_server_patch(name, argv, expected_fingerprint=expected_fingerprint),
        hermes_home=target,
    )
    return True


def remove_mcp_server_config(
    name: str,
    *,
    hermes_home: str | Path | None = None,
    expected_fingerprint: str | None = None,
) -> bool:
    """Remove one server only while its exact managed value still matches."""
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        return False
    target = Path(hermes_home or os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    path = f"mcp_servers.{name}"
    receipt = apply_config_patch(
        {"set": {}, "unset": [], "unset_if_hash": {path: expected_fingerprint}},
        hermes_home=target,
    )
    return path in receipt.get("paths", {}).get("unset", [])


def launch(state: dict, tool_args: list[str]) -> None:
    """Launch ordinary Hermes; configuration does not require a wrapper."""
    del state
    exec_or_spawn([SPEC["binary"], *tool_args])


def validate_cmd(binary: str) -> list[str]:
    return [binary, "--version"]
