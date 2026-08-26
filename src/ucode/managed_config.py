"""Managed coding-agent config: fetch, normalize, and local persistence.

An org admin authors a ``CodingAgentConfig`` through the Databricks AI Gateway; developers read it
(non-admin) and ``ucode`` applies it locally. This module owns the fetch/normalize side and keeps
the admin's local draft separate from the last published config fetched by a launch:

- fetching the raw manifest (via :func:`ucode.databricks.fetch_managed_coding_agent_configs`),
- normalizing the proto-JSON into a stable internal dict keyed by ucode's own tool names,
- ``~/.ucode/managed-state.json`` is the editable draft authored by ``ucode setup`` and published by
  ``ucode apply``;
- ``~/.ucode/managed-cache/<workspace-hash>.json`` is launch-owned and contains only the last
  workspace-published config plus provenance metadata, for outage fallback.

Keeping the files separate is important: an ordinary launch must never overwrite or accidentally
apply an admin's unpublished edits. Only a launch with ``--local`` reads the authored draft.

:func:`refresh_managed_config` is the launch path's entry point. It is called before model discovery,
because the manifest decides whether that discovery is needed at all; the launch path then hands the
manifest to :func:`ucode.managed_resolve.resolve_state` once the state it layers over is final.
Deciding *which* value wins for a given key is :mod:`ucode.managed_resolve`'s job, kept separate so
that logic stays pure and I/O-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import ucode.config_io as config_io
from ucode.databricks import (
    fetch_managed_coding_agent_configs,
    fetch_model_recommendation,
    get_databricks_token,
)
from ucode.ui import console, print_warning

MANAGED_STATE_PATH = config_io.APP_DIR / "managed-state.json"
MANAGED_CACHE_DIR = config_io.APP_DIR / "managed-cache"
# Read-only migration fallback for the single-cache layout used by earlier builds. New writes always
# go to MANAGED_CACHE_DIR so switching workspaces cannot discard another workspace's fallback.
LEGACY_MANAGED_CACHE_PATH = config_io.APP_DIR / "managed-cache.json"
MANAGED_STATE_SCHEMA_VERSION = "1.0"

# Opt-in switch while the feature is in bug bash: unset means launches ignore managed configs
# entirely and behave exactly as they did before.
MANAGED_CONFIG_ENV_VAR = "ENABLE_MANAGED_AGENT_CONFIG"

# Shown to a developer when their workspace has no admin-defined managed config yet — the normal
# case, not an error. Kept here so the CLI (which surfaces it) uses one consistent message.
NO_MANAGED_CONFIG_MESSAGE = "No coding-agent config has been set up by your workspace admin yet."

# CodingAgent proto enum -> ucode tool name. Anything unrecognized (e.g. a newer agent this ucode
# build doesn't know) is dropped during normalization rather than guessed at. Public because the
# admin-write side (``managed_setup``) inverts these maps to serialize, so a new agent or MCP type
# only has to be declared once.
AGENT_ENUM_TO_TOOL: dict[str, str] = {
    "CODING_AGENT_CLAUDE_CODE": "claude",
    "CODING_AGENT_CODEX": "codex",
    "CODING_AGENT_GEMINI": "gemini",
    "CODING_AGENT_COPILOT": "copilot",
    "CODING_AGENT_PI": "pi",
    "CODING_AGENT_OPENCODE": "opencode",
}

# McpServerType proto enum -> ucode's short type tag. Mirrors the selection prefixes in ``mcp.py``;
# the actual name->URL resolution happens there when the manifest is applied (a later change).
MCP_TYPE_ENUM_TO_TAG: dict[str, str] = {
    "MCP_SERVER_TYPE_UC_SERVICE": "mcp-service",
    "MCP_SERVER_TYPE_EXTERNAL": "external",
    "MCP_SERVER_TYPE_GENIE": "genie-space",
    "MCP_SERVER_TYPE_VECTOR_SEARCH": "vector-search",
    "MCP_SERVER_TYPE_UC_FUNCTIONS": "uc-functions",
    "MCP_SERVER_TYPE_DATABRICKS_APP": "app",
    "MCP_SERVER_TYPE_DATABRICKS_SQL": "sql",
}


def _as_dict(value: object) -> dict[str, object]:
    """Return ``value`` as a ``dict[str, object]`` when it is a dict, else an empty dict.

    Centralizes the isinstance-narrowing so downstream ``.get`` calls type-check (a bare
    ``isinstance(x, dict)`` narrows to ``dict[Never, Never]``, which rejects string keys)."""
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def _str(value: object) -> str | None:
    """Return a non-empty stripped string, or None."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        s = _str(item)
        if s:
            out.append(s)
    return out


def _normalize_model_config(model_config: object) -> dict | None:
    """Normalize an ``AgentModelConfig`` oneof into ``{model_provider_service?, default_model?,
    models}``.

    The proto is a oneof over per-agent variants (claude/codex/opencode/pi/gemini/copilot). We
    don't care which variant tag it is here — the enclosing agent already tells us — so we read the
    common fields. Claude's ``models`` is a dict of family slots; the rest are a flat list. Returns
    None when there's no usable model config.
    """
    mc = _as_dict(model_config)
    if not mc:
        return None
    # Unwrap the oneof: take whichever single variant sub-dict is present.
    variant = next((_as_dict(v) for v in mc.values() if isinstance(v, dict)), None)
    if not variant:
        return None
    result: dict = {}
    mps = _str(variant.get("model_provider_service"))
    if mps:
        result["model_provider_service"] = mps
    default_model = _str(variant.get("default_model"))
    if default_model:
        result["default_model"] = default_model
    models = variant.get("models")
    if isinstance(models, dict):
        # Claude family slots (default_opus_model, default_sonnet_model, ...).
        slots = {k: _str(v) for k, v in _as_dict(models).items() if _str(v)}
        if slots:
            result["models"] = slots
    else:
        model_list = _str_list(models)
        if model_list:
            result["models"] = model_list
    return result or None


def _normalize_enabled_agent(entry: object) -> tuple[str, dict] | None:
    """Normalize one ``EnabledAgent`` into ``(tool, agent_config)``, or None if unusable.

    Drops entries whose agent enum is unset/unknown to this ucode build.
    """
    entry_dict = _as_dict(entry)
    if not entry_dict:
        return None
    tool = AGENT_ENUM_TO_TOOL.get(_str(entry_dict.get("agent")) or "")
    if tool is None:
        return None
    config_in = _as_dict(entry_dict.get("config"))
    agent_config: dict = {}
    if isinstance(config_in.get("use_as_global_settings"), bool):
        agent_config["use_as_global_settings"] = config_in["use_as_global_settings"]
    headers = config_in.get("custom_headers")
    if isinstance(headers, dict):
        clean = {
            k: v for k, v in _as_dict(headers).items() if isinstance(k, str) and isinstance(v, str)
        }
        if clean:
            agent_config["custom_headers"] = clean
    tracing_table = _tracing_table(config_in.get("tracing_config"))
    if tracing_table:
        agent_config["tracing_table"] = tracing_table
    model_config = _normalize_model_config(config_in.get("model_config"))
    if model_config is not None:
        agent_config["model_config"] = model_config
    return tool, agent_config


def _tracing_table(tracing: object) -> str | None:
    """Extract ``TracingConfig.table`` (a UC table FQN), or None."""
    return _str(_as_dict(tracing).get("table"))


def _normalize_mcp_servers(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    out: list[dict] = []
    for entry in value:
        entry_dict = _as_dict(entry)
        name = _str(entry_dict.get("name"))
        tag = MCP_TYPE_ENUM_TO_TAG.get(_str(entry_dict.get("type")) or "")
        if name and tag:
            out.append({"name": name, "type": tag})
    return out


def _normalize_budget_policy(value: object) -> dict | None:
    bp = _as_dict(value)
    if not bp:
        return None
    policy: dict = {}
    display_name = _str(bp.get("display_name"))
    if display_name:
        policy["display_name"] = display_name
    budget_id = _str(bp.get("budget_id"))
    if budget_id:
        policy["budget_id"] = budget_id
    tiers: list[dict] = []
    raw_tiers = bp.get("tiers")
    for tier in raw_tiers if isinstance(raw_tiers, list) else []:
        tier_dict = _as_dict(tier)
        pct = tier_dict.get("spending_percentage")
        if not isinstance(pct, (int, float)) or isinstance(pct, bool):
            continue
        tier_out: dict = {"spending_percentage": float(pct)}
        agent = AGENT_ENUM_TO_TOOL.get(_str(tier_dict.get("default_agent")) or "")
        if agent:
            tier_out["default_agent"] = agent
        model = _str(tier_dict.get("default_model"))
        if model:
            tier_out["default_model"] = model
        tiers.append(tier_out)
    if tiers:
        policy["tiers"] = tiers
    return policy or None


def normalize_managed_config(raw: dict) -> dict:
    """Normalize a raw ``CodingAgentConfig`` proto-JSON dict into ucode's internal shape.

    The internal shape uses ucode's own tool names and short MCP type tags so downstream reconcile
    and apply code never touches proto enum spellings. Unknown agents / MCP types are dropped.
    """
    raw = _as_dict(raw)
    result: dict = {}
    name = _str(raw.get("name"))
    if name:
        result["name"] = name
    default_agent = AGENT_ENUM_TO_TOOL.get(_str(raw.get("default_agent")) or "")
    if default_agent:
        result["default_agent"] = default_agent
    enabled_agents: dict[str, dict] = {}
    raw_agents = raw.get("enabled_agents")
    for entry in raw_agents if isinstance(raw_agents, list) else []:
        normalized = _normalize_enabled_agent(entry)
        if normalized is not None:
            tool, agent_config = normalized
            enabled_agents[tool] = agent_config
    if enabled_agents:
        result["enabled_agents"] = enabled_agents
    mcp_servers = _normalize_mcp_servers(raw.get("mcp_servers"))
    if mcp_servers:
        result["mcp_servers"] = mcp_servers
    skill_names = _str_list(_as_dict(raw.get("skills")).get("names"))
    if skill_names:
        result["skills"] = {"names": skill_names}
    tracing_table = _tracing_table(raw.get("tracing"))
    if tracing_table:
        result["tracing_table"] = tracing_table
    budget_policy = _normalize_budget_policy(raw.get("budget_policy"))
    if budget_policy is not None:
        result["budget_policy"] = budget_policy
    return result


def _decimal(value: object) -> float | None:
    """Parse one of the API's decimal-string money fields, or None when absent/unparseable."""
    text = _str(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def get_model_recommendation(workspace: str, token: str) -> tuple[dict | None, str | None]:
    """Fetch the agent and model the caller's budget tier allows, normalized for the launch path.

    Returns ``(recommendation, reason)`` where the recommendation is ``{"agent", "model",
    "current_spend", "effective_threshold"}``. Every field is optional server-side, so each is
    normalized independently: an agent this build doesn't recognize is dropped rather than failing
    the read, and a model can arrive without an agent.
    """
    payload, reason = fetch_model_recommendation(workspace, token)
    if reason is not None:
        return None, reason
    agent = AGENT_ENUM_TO_TOOL.get(_str(payload.get("recommended_agent")) or "")
    model = _str(payload.get("recommended_model"))
    spend = _decimal(payload.get("current_spend"))
    threshold = _decimal(payload.get("effective_threshold"))
    if agent is None and model is None and spend is None and threshold is None:
        return None, None
    return {
        "agent": agent,
        "model": model,
        "current_spend": spend,
        "effective_threshold": threshold,
    }, None


def get_managed_config(workspace: str, token: str) -> tuple[dict | None, str | None]:
    """Fetch and normalize the workspace's managed config.

    Returns ``(config, reason)``:
    - ``(config, None)`` — the normalized manifest for the workspace's single config;
    - ``(None, None)`` — the workspace definitively has no managed config (not an error);
    - ``(None, reason)`` — the read didn't settle the question; ``reason`` says why.

    The distinction matters to callers that cache: only ``(None, None)`` is authoritative enough to
    clear a previously stored config. "No config defined" arrives two ways depending on the backend
    — an empty listing (HTTP 200 with no configs) or a NOT_FOUND — and both collapse to
    ``(None, None)``. Anything else, including a PERMISSION_DENIED, leaves the question unanswered
    and is surfaced as a failure: an admin may have published a config the developer can't read,
    which they need to know about rather than silently launch without.

    v0 stores at most one config per workspace, so the first entry is the workspace's config.
    """
    configs, reason = fetch_managed_coding_agent_configs(workspace, token)
    if reason is not None:
        if _is_feature_disabled(reason):
            return None, reason
        # A NOT_FOUND means the admin hasn't defined a config for this workspace — not a failure.
        if _is_not_found(reason):
            return None, None
        return None, reason
    if not configs:
        return None, None
    return normalize_managed_config(configs[0]), None


def _is_not_found(reason: str) -> bool:
    """True when a read failure reason means the workspace definitively has no managed config.

    ``_http_get_json`` formats failures as ``HTTP <code> <text>[: <body>]``; a NOT_FOUND surfaces
    as an ``HTTP 404`` there (and the API's error body carries ``NOT_FOUND``)."""
    lowered = reason.lower()
    return "http 404" in lowered or "not_found" in lowered


def _is_permission_denied(reason: str) -> bool:
    """True when the read was refused rather than answering whether a config exists.

    The read is meant to be available to any workspace user, so a refusal means the workspace's
    managed config isn't readable by this developer — worth telling them about, since an admin may
    have published a config that silently isn't reaching them. It settles nothing about whether one
    exists, so a cached config is left in place rather than cleared."""
    lowered = reason.lower()
    return "http 403" in lowered or "permission_denied" in lowered


def _config_digest(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _managed_cache_path(workspace: str) -> Path:
    """Return a stable, filesystem-safe cache path for one workspace."""
    workspace_hash = hashlib.sha256(workspace.encode("utf-8")).hexdigest()
    return MANAGED_CACHE_DIR / f"{workspace_hash}.json"


def _save_managed_payload(
    path: Path,
    workspace: str,
    config: dict,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    """Atomically write one versioned managed-config envelope at mode 0600.

    The temporary file is created beside the destination, flushed, and replaced into place. Readers
    therefore observe either the complete old document or the complete new one, including when two
    launches refresh the same cache concurrently.
    """
    # Dict insertion order is preserved by json.dumps: keep the format version first so humans and
    # future readers can identify the file schema before interpreting any payload fields. Version
    # 1.0 is the only managed-state schema this ucode release writes.
    payload: dict[str, object] = {
        "schema_version": MANAGED_STATE_SCHEMA_VERSION,
        "workspace": workspace,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    payload["config"] = config
    if config_io.is_dry_run():
        # Print rather than write, matching how the agent config writers behave under --dry-run.
        console.print(f"\n[bold]\\[dry run] {path}[/bold]\n{json.dumps(payload, indent=2)}\n")
        return
    config_io.ensure_parent_dir(path)
    serialized = json.dumps(payload, indent=2) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        _restrict_permissions(temporary_path)
        os.replace(temporary_path, path)
    except OSError as exc:
        raise RuntimeError(f"Failed to write managed config file: {path}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def save_managed_state(workspace: str, config: dict) -> None:
    """Save the admin-authored draft to ``~/.ucode/managed-state.json``."""
    _save_managed_payload(MANAGED_STATE_PATH, workspace, config)


def save_managed_cache(workspace: str, config: dict) -> None:
    """Save the last workspace-published config for launch fallback.

    An empty config records an authoritative successful read of "no published config", preventing
    an older cached policy from being resurrected after a later transient failure.
    """
    metadata: dict[str, object] = {
        "source": "workspace",
        "fetched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "config_digest": _config_digest(config),
    }
    _save_managed_payload(_managed_cache_path(workspace), workspace, config, metadata=metadata)


def _restrict_permissions(path: Path) -> None:
    """Best-effort chmod 0600. No-op where unsupported (e.g. Windows), where the effective
    read-only guarantee is left to a later change."""
    try:
        os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def _read_managed_state_v1(data: dict) -> tuple[str, dict] | None:
    """Parse the v1.0 managed-state envelope.

    Keeping this version-specific prevents a future schema from being accidentally interpreted as
    v1 just because it happens to reuse a field name. New versions add a new reader and registry
    entry rather than accumulating conditionals in the launch path.
    """
    workspace = data.get("workspace")
    config = data.get("config")
    if not isinstance(workspace, str) or not workspace or not isinstance(config, dict):
        return None
    return workspace, config


_MANAGED_STATE_READERS = {"1.0": _read_managed_state_v1}


def _read_managed_payload(
    path: Path, *, require_cache_metadata: bool = False
) -> tuple[str, dict] | None:
    data = config_io.read_json_safe(path)
    version = data.get("schema_version")
    if version is None:
        # Files written before schema_version was introduced already use the v1.0 envelope. Read
        # them as v1 so upgrading ucode does not discard an otherwise valid cached config.
        version = MANAGED_STATE_SCHEMA_VERSION
    if not isinstance(version, str):
        return None
    reader = _MANAGED_STATE_READERS.get(version)
    if reader is None:
        # An older ucode must not guess how to interpret a future managed-state schema.
        return None
    payload = reader(data)
    if payload is None or not require_cache_metadata:
        return payload
    _workspace, config = payload
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        return None
    if metadata.get("source") != "workspace":
        return None
    if metadata.get("config_digest") != _config_digest(config):
        return None
    fetched_at = metadata.get("fetched_at")
    if not isinstance(fetched_at, str) or not fetched_at:
        return None
    return payload


def load_managed_state(workspace: str | None) -> dict | None:
    """Load the admin-authored draft for ``workspace``, or None if absent/mismatched.

    Returns the normalized config dict (the ``config`` field), only when the stored file is for the
    same workspace — so a stale file from another workspace is ignored rather than misapplied.

    ``ucode setup`` authors this file, ``ucode <agent> --local`` tests it, and ``ucode apply``
    publishes it. Ordinary launches never write or read it.
    """
    if not workspace:
        return None
    payload = _read_managed_payload(MANAGED_STATE_PATH)
    if payload is None:
        return None
    stored_workspace, config = payload
    return config if stored_workspace == workspace else None


def load_managed_cache(workspace: str | None) -> dict | None:
    """Load the last workspace-published config cached by an ordinary launch."""
    if not workspace:
        return None
    cache_path = _managed_cache_path(workspace)
    payload = _read_managed_payload(cache_path, require_cache_metadata=True)
    if payload is None and not cache_path.exists():
        payload = _read_managed_payload(LEGACY_MANAGED_CACHE_PATH)
    if payload is None:
        return None
    stored_workspace, config = payload
    return config if stored_workspace == workspace else None


def managed_state_workspace() -> str | None:
    """The workspace the on-disk draft was authored for, or None when there is none.

    Lets a caller that has no workspace in local ucode state (e.g. ``ucode setup --show`` before
    ``ucode configure``) still find the manifest on disk and report which workspace it belongs to.
    """
    payload = _read_managed_payload(MANAGED_STATE_PATH)
    return payload[0] if payload is not None else None


def refresh_managed_config(state: dict) -> tuple[dict | None, bool]:
    """Fetch and cache the workspace-published config.

    Runs on every launch so a developer picks up an admin's edits without re-running
    ``ucode configure``. The manifest is None when the workspace has no managed config — the normal
    case for a workspace whose admin hasn't published one.

    A failed fetch never blocks the launch: an unreachable control plane shouldn't stop someone from
    coding. Instead it falls back to the last published config cached for this workspace, so the admin's
    most recent known policy still applies; only when there is no persisted config either does the
    launch fall through to the developer's own settings.

    ``coding_agent_config_feature_disabled`` is True when the gateway returned ``FEATURE_DISABLED`` and there was no
    persisted config to fall back on — the coding-agent-configs feature isn't enabled server-side,
    so callers suppress the ``ucode setup`` recommendation.
    """
    workspace = state.get("workspace")
    if not workspace:
        return None, False
    try:
        token = get_databricks_token(workspace, state.get("profile"))
    except RuntimeError as exc:
        return _persisted_fallback(workspace, str(exc)), False
    managed, reason = get_managed_config(workspace, token)
    if reason is not None:
        # A refused read leaves the cached config alone: it says nothing about whether the admin's
        # config still exists, unlike a successful "no config" answer below.
        fallback = _persisted_fallback(workspace, reason, refused=_is_permission_denied(reason))
        return fallback, _is_feature_disabled(reason) and fallback is None
    if managed is None:
        # Record that this workspace has no config, rather than leaving an earlier one on disk:
        # the file doubles as the fallback above, so a removed policy would otherwise come back
        # into force after the next transient outage.
        save_managed_cache(workspace, {})
        return None, False
    save_managed_cache(workspace, managed)
    return managed, False


def _is_feature_disabled(reason: str) -> bool:
    return "feature_disabled" in reason.lower()


def _persisted_fallback(workspace: str, reason: str, *, refused: bool = False) -> dict | None:
    """Return the last persisted config for ``workspace`` after a failed fetch.

    Warns only when there is a config to fall back on, because then the launch proceeds on an admin
    policy that may be out of date. With nothing persisted there is no managed config in play at
    all, so staying quiet keeps someone with (say) an expired session from being told about a
    feature they don't use — including when the read was ``refused``, since a refusal is no evidence
    that a config exists.
    """
    # An empty persisted config means the last successful read found none, so there is no admin
    # policy to fall back to — treat it the same as having no file at all.
    persisted = load_managed_cache(workspace)
    if not persisted:
        return None
    summary = _summarize_read_failure(reason)
    if refused:
        print_warning(
            f"Your workspace's managed config is not readable by you ({summary}); using the last "
            "one saved for this workspace. Ask an admin to grant access."
        )
    else:
        print_warning(
            f"Could not read your workspace's managed config ({summary}); "
            "using the last one saved for this workspace."
        )
    return persisted


def _summarize_read_failure(reason: str) -> str:
    """Condense a read failure into one short line fit for a terminal warning.

    ``_http_get_json`` appends the raw response body, which for a gateway error is a multi-line JSON
    blob (error_code, message, request_id, trace ids). Surface just the status and the API's own
    message; the full text is still available under ``UCODE_DEBUG=1``.
    """
    status, _, body = reason.partition(": ")
    body = body.strip()
    if body.startswith("{"):
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            message = _str(parsed.get("message")) or _str(parsed.get("error_code"))
            if message:
                return f"{status.strip()}: {message}"
        return status.strip()
    condensed = " ".join(reason.split())
    return condensed if len(condensed) <= 160 else condensed[:157] + "..."


def managed_agent_config_enabled() -> bool:
    """True when managed coding-agent configs are switched on for this run.

    Opt-in while the feature is being bug-bashed: without the env var set, launches behave exactly
    as they did before and never read the workspace's config."""
    return os.environ.get(MANAGED_CONFIG_ENV_VAR, "").strip().lower() in ("1", "true", "yes")
