"""Admin-authored managed coding-agent config: fetch, normalize, and local persistence.

An org admin authors a ``CodingAgentConfig`` through the Databricks AI Gateway; developers read it
(non-admin) and ``ucode`` applies it locally. This module owns the fetch/normalize side and the one
local file, ``~/.ucode/managed-state.json`` (0600), that both roles share:

- fetching the raw manifest (via :func:`ucode.databricks.fetch_managed_coding_agent_configs`),
- normalizing the proto-JSON into a stable internal dict keyed by ucode's own tool names,
- persisting it via :func:`save_draft_config` / :func:`save_published_config` (and the matching
  loaders) — the admin-write side (``managed_setup`` / ``managed_wizard``) authors the manifest here,
  and the launch path pulls the published copy back into the same file, and
- re-reading it on each launch, falling back to the persisted copy when the read fails.

One file holds two clearly separated slots per workspace. ``managed-state.json`` is a versioned
per-workspace map — ``{"version": 2, "workspaces": {<url>: {"published": ..., "draft": ...}}}`` — so
the admin's locally authored, unpublished ``draft`` never shares a slot with the launch-fetched
``published`` snapshot. ``ucode configure`` (admin) authors the ``draft``; ``ucode export`` /
``ucode publish`` read the ``draft`` only; a launch refreshes the ``published`` slot and never
touches the ``draft``. Keeping a map (not a single slot) means refreshing one workspace can't clobber
another workspace's draft.

:func:`refresh_managed_config` is the launch path's entry point. It is called before model discovery,
because the manifest decides whether that discovery is needed at all; the launch path then hands the
manifest to :func:`ucode.managed_resolve.resolve_state` once the state it layers over is final.
Deciding *which* value wins for a given key is :mod:`ucode.managed_resolve`'s job, kept separate so
that logic stays pure and I/O-free.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import ucode.config_io as config_io
from ucode.databricks import (
    fetch_managed_coding_agent_configs,
    fetch_model_recommendation,
    get_databricks_token,
)
from ucode.ui import print_warning

MANAGED_STATE_PATH = config_io.APP_DIR / "managed-state.json"

MANAGED_STATE_VERSION = 2

# Opt-in switch while the feature is in bug bash: unset means launches ignore managed configs
# entirely and behave exactly as they did before.
MANAGED_CONFIG_ENV_VAR = "ENABLE_MANAGED_AGENT_CONFIG"

# Shown to a developer when their workspace has no admin-defined managed config yet — the normal
# case, not an error. Kept here so the CLI (which surfaces it) uses one consistent message.
NO_MANAGED_CONFIG_MESSAGE = "No coding-agent config has been set up by your workspace admin yet."

# CodingAgent proto enum -> ucode tool name. Anything unrecognized (e.g. a newer agent this ucode
# build doesn't know) is dropped during normalization rather than guessed at. Public because the
# admin-write side (``managed_setup``) inverts this map to serialize, so a new agent only has to be
# declared once.
AGENT_ENUM_TO_TOOL: dict[str, str] = {
    "CODING_AGENT_CLAUDE_CODE": "claude",
    "CODING_AGENT_CODEX": "codex",
    "CODING_AGENT_GEMINI": "gemini",
    "CODING_AGENT_COPILOT": "copilot",
    "CODING_AGENT_PI": "pi",
    "CODING_AGENT_OPENCODE": "opencode",
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

    The internal shape uses ucode's own tool names so downstream reconcile and apply code never
    touches proto enum spellings. Unknown agents are dropped. MCP servers and skills are personal
    configuration and are deliberately not read from the manifest.
    """
    raw = _as_dict(raw)
    result: dict = {}
    name = _str(raw.get("name"))
    if name:
        result["name"] = name
    display_name = _str(raw.get("display_name"))
    if display_name:
        result["display_name"] = display_name
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


def _migrated_workspaces(data: dict) -> dict[str, dict]:
    """Return the ``workspaces`` map from a raw ``managed-state.json`` dict, migrating v1 in-memory.

    v1 stored a single ``{workspace, config}`` slot shared by both the admin's authored draft and the
    launch-fetched published copy, so a legacy value's provenance can't be recovered. It is migrated
    into the ``published`` slot — the common case is a developer's fetched snapshot, and treating it
    as published keeps ``ucode export`` / ``ucode publish`` (now strictly draft-only) from mistaking a
    cached fetch for authored work. A one-time backup (see :func:`_backup_legacy_file_once`) makes a
    rare unpublished admin draft recoverable. Read-only: never writes.
    """
    if data.get("version") == MANAGED_STATE_VERSION and isinstance(data.get("workspaces"), dict):
        return {k: v for k, v in cast("dict", data["workspaces"]).items() if isinstance(v, dict)}
    workspaces: dict[str, dict] = {}
    legacy_ws = data.get("workspace")
    if isinstance(legacy_ws, str) and legacy_ws:
        legacy_cfg = data.get("config")
        workspaces[legacy_ws] = {"published": legacy_cfg if isinstance(legacy_cfg, dict) else {}}
    return workspaces


def _is_legacy_file(data: dict) -> bool:
    """True when ``data`` is a pre-v2 ``{workspace, config}`` file (not the versioned map)."""
    return data.get("version") != MANAGED_STATE_VERSION and "workspace" in data


def _backup_legacy_file_once() -> None:
    """Copy a pre-v2 ``managed-state.json`` to ``<path>.pre-v2.bak`` before it is overwritten.

    Best-effort and idempotent: migration maps the single legacy slot into ``published``, which can't
    preserve a rare unpublished admin draft, so the original bytes are kept once for recovery. Written
    through a temp file so an interrupted copy cannot leave a truncated backup that the ``exists()``
    guard would then treat as the original."""
    backup = MANAGED_STATE_PATH.with_suffix(MANAGED_STATE_PATH.suffix + ".pre-v2.bak")
    if backup.exists() or not MANAGED_STATE_PATH.exists():
        return
    if not _is_legacy_file(config_io.read_json_safe(MANAGED_STATE_PATH)):
        return
    tmp = backup.with_name(backup.name + ".tmp")
    try:
        tmp.write_bytes(MANAGED_STATE_PATH.read_bytes())
        _restrict_permissions(tmp)
        os.replace(tmp, backup)
    except OSError:
        pass


def _save_slot(workspace: str, slot: str, config: dict) -> None:
    """Write ``config`` to ``workspace``'s ``published`` or ``draft`` slot, preserving everything else.

    Reads the current file, migrates it to the v2 map, sets one slot for one workspace, and writes it
    back — so a refresh of the published slot never disturbs an admin's draft (or other workspaces).
    0600 keeps the org-authored file readable/writable only by the user. No-op write in dry-run.

    An empty ``config`` in the ``published`` slot records "this workspace has no managed config",
    which matters because that slot doubles as the fallback when a later read fails: without it,
    removing a config server-side would leave the old one on disk to be reapplied after an outage.
    """
    workspaces = _migrated_workspaces(config_io.read_json_safe(MANAGED_STATE_PATH))
    workspaces[workspace] = {**workspaces.get(workspace, {}), slot: config}
    payload = {"version": MANAGED_STATE_VERSION, "workspaces": workspaces}
    if config_io.is_dry_run():
        config_io.write_json_file(MANAGED_STATE_PATH, payload)
        return
    _backup_legacy_file_once()
    _write_state_atomically(payload)


def _write_state_atomically(payload: dict) -> None:
    """Write the state map through a sibling temp file and rename it into place.

    The map holds the admin's draft, which nothing can rebuild: a torn write would take the draft
    with it, where before v2 an interrupted write only cost a snapshot the next launch refetches.
    """
    tmp = MANAGED_STATE_PATH.with_name(MANAGED_STATE_PATH.name + ".tmp")
    config_io.write_json_file(tmp, payload)
    _restrict_permissions(tmp)
    try:
        os.replace(tmp, MANAGED_STATE_PATH)
    except OSError as exc:
        raise RuntimeError(f"Failed to write managed state file: {MANAGED_STATE_PATH}") from exc


def save_published_config(workspace: str, config: dict) -> None:
    """Persist the workspace's launch-fetched ``published`` snapshot, leaving any ``draft`` intact."""
    _save_slot(workspace, "published", config)


def save_draft_config(workspace: str, config: dict) -> None:
    """Persist the admin's locally authored ``draft``, leaving the ``published`` snapshot intact."""
    _save_slot(workspace, "draft", config)


def _restrict_permissions(path: Path) -> None:
    """Best-effort chmod 0600. No-op where unsupported (e.g. Windows), where the effective
    read-only guarantee is left to a later change."""
    try:
        os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def _load_slot(workspace: str | None, slot: str) -> dict | None:
    """Return ``workspace``'s ``published`` or ``draft`` config, or None if absent."""
    if not workspace:
        return None
    entry = _migrated_workspaces(config_io.read_json_safe(MANAGED_STATE_PATH)).get(workspace) or {}
    config = entry.get(slot)
    return config if isinstance(config, dict) else None


def load_published_config(workspace: str | None) -> dict | None:
    """Load the launch-fetched ``published`` snapshot for ``workspace``, or None if absent.

    This is what the launch path overlays (:func:`ucode.managed_resolve.resolve_state`) and what
    ``ucode status`` reports — the admin-defined config a developer actually runs under. A stale
    snapshot from a different workspace is ignored rather than misapplied.
    """
    return _load_slot(workspace, "published")


def load_draft_config(workspace: str | None) -> dict | None:
    """Load the admin's locally authored, unpublished ``draft`` for ``workspace``, or None.

    Only ``ucode configure`` (admin authoring) writes this, and only ``ucode export`` / ``ucode
    publish`` read it — never the launch path. A developer who has only ever fetched a published
    snapshot has no draft, which is why export/publish are draft-only rather than falling back to the
    fetched copy: a cached publication is not authored work.
    """
    return _load_slot(workspace, "draft")


def managed_state_workspace() -> str | None:
    """The sole workspace recorded in ``managed-state.json``, or None when absent/ambiguous.

    Lets a caller with no workspace in local ucode state still find the config on disk. With several
    workspaces recorded the answer is ambiguous, so it returns None and the caller reports that a
    workspace must be selected first.
    """
    workspaces = _migrated_workspaces(config_io.read_json_safe(MANAGED_STATE_PATH))
    return next(iter(workspaces)) if len(workspaces) == 1 else None


def refresh_managed_config(state: dict) -> tuple[dict | None, bool]:
    """Fetch the workspace's managed config and persist it, returning ``(manifest, coding_agent_config_feature_disabled)``.

    Runs on every launch so a developer picks up an admin's edits without re-running
    ``ucode configure``. The manifest is None when the workspace has no managed config — the normal
    case for a workspace whose admin hasn't published one.

    A failed fetch never blocks the launch: an unreachable control plane shouldn't stop someone from
    coding. Instead it falls back to the last config persisted for this workspace, so the admin's
    most recent known policy still applies; only when there is no persisted config either does the
    launch fall through to the developer's own settings.

    ``coding_agent_config_feature_disabled`` is True when the gateway returned ``FEATURE_DISABLED`` and there was no
    persisted config to fall back on — the coding-agent-configs feature isn't enabled server-side,
    so callers suppress the ``ucode configure`` publish recommendation.
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
        # the published slot doubles as the fallback above, so a removed policy would otherwise come
        # back into force after the next transient outage. The admin's draft, if any, is untouched.
        save_published_config(workspace, {})
        return None, False
    save_published_config(workspace, managed)
    return managed, False


def fetch_published_config(workspace: str, token: str) -> tuple[dict | None, str | None, bool]:
    """Fetch the workspace's published config for ``ucode configure`` and persist the published slot.

    Returns ``(published_or_None, read_error_or_None, feature_disabled)``. Unlike
    :func:`refresh_managed_config`, ``feature_disabled`` is reported independently of any persisted
    fallback, so ``ucode configure`` can branch on "the managed-config backend is unavailable" — and
    then skip publish advice — even when a stale published snapshot is still on disk. A successful
    read persists the published slot (recording emptiness as ``{}``); the admin's draft is untouched.
    """
    managed, reason = get_managed_config(workspace, token)
    if reason is not None:
        return None, reason, _is_feature_disabled(reason)
    if managed is None:
        save_published_config(workspace, {})
        return None, None, False
    save_published_config(workspace, managed)
    return managed, None, False


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
    persisted = load_published_config(workspace)
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
