"""Resolve the effective agent settings from the managed config plus local ucode state.

The admin-authored manifest (``~/.ucode/managed-state.json``, written by
:mod:`ucode.managed_config`) and the developer's own ucode state (``~/.ucode/state.json``) stay
separate files — they are never merged on disk. Instead this module resolves them *per key* at
config-write time: whatever the manifest specifies wins, and anything it leaves unset falls back to
the developer's ucode state. The resolved view is what gets rendered into the agent config files
(e.g. ``~/.claude/ucode-settings.json``), so managed settings take precedence for every ``ucode``
command without either file being rewritten.

Only settings the developer set *through* ucode participate in the fallback. Settings they wrote by
hand outside ucode (``~/.claude/settings.json``, etc.) are not read here — Claude Code merges
those scopes itself at launch, underneath the file ucode passes via ``--settings``.

Everything here is pure: no I/O, no mutation of the inputs. Fetching and persisting the manifest,
and handing the resolved state to the agent config writers, live in :mod:`ucode.managed_config`.
"""

from __future__ import annotations

from typing import cast

from ucode.state import MANAGED_OVERLAY_KEY

# Proto model-config slot -> the family key `claude.py`'s render_overlay reads. The manifest keeps
# the proto spelling (`default_opus_model`), while ucode state and render_overlay both key claude
# models by bare family (`opus`), so the two have to be bridged before the settings file is written.
_CLAUDE_FAMILY_SLOTS = {
    "default_opus_model": "opus",
    "default_sonnet_model": "sonnet",
    "default_haiku_model": "haiku",
    "default_fable_model": "fable",
}


def _as_dict(value: object) -> dict[str, object]:
    """Return ``value`` as a ``dict[str, object]`` when it is a dict, else an empty dict."""
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def _str(value: object) -> str | None:
    """Return a non-empty stripped string, or None."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _agent_entry(managed: dict, tool: str) -> dict[str, object]:
    """Return the manifest's config for ``tool``, or an empty dict when it isn't enabled."""
    enabled = _as_dict(_as_dict(managed).get("enabled_agents"))
    return _as_dict(enabled.get(tool))


def _agent_model_config(managed: dict, tool: str) -> dict[str, object]:
    """Return the manifest's normalized ``model_config`` for ``tool``, if any."""
    return _as_dict(_agent_entry(managed, tool).get("model_config"))


def effective_agent_models(managed: dict, state: dict, tool: str) -> dict | list | None:
    """Resolve ``tool``'s model list/slots from the manifest, falling back to ucode state.

    Once the manifest says anything about ``tool``'s models it is the whole allowlist: the
    developer's discovered models drop out entirely, so a launch can only reach models the admin
    named. Each claude family resolves only to its own slot — a family the manifest leaves out stays
    unset rather than inheriting ``default_model``, so ucode writes no
    ``ANTHROPIC_DEFAULT_<FAMILY>_MODEL`` for it and the agent falls back to its own default. Omitting
    a family is how an admin steers people off it, and filling it in with ``default_model`` would
    quietly re-enable what they left out. Every other agent stores a flat list, which has no per-key
    identity — there the manifest's list replaces the local one outright. Only when the manifest
    names nothing for ``tool`` does the developer's own list stand.
    """
    model_config = _agent_model_config(managed, tool)
    manifest_models = model_config.get("models")
    if tool == "claude":
        slots: dict[str, str] = {}
        for slot, family in _CLAUDE_FAMILY_SLOTS.items():
            model = _str(_as_dict(manifest_models).get(slot))
            if model:
                slots[family] = model
        if slots:
            return slots
        local = _as_dict(state.get("claude_models"))
        return dict(local) if local else None
    if isinstance(manifest_models, list):
        models = [m for m in (_str(item) for item in manifest_models) if m]
        if models:
            return models
    local_list = state.get(f"{tool}_models")
    return local_list if local_list else None


def managed_supplies_models(managed: dict | None, tool: str) -> bool:
    """True when the managed config already says which models ``tool`` should use.

    Lets the launch path skip Databricks model discovery, whose whole purpose is to find the models
    the config has now specified. Any of the three counts: a provider (the agent routes by header and
    pins no Databricks model), a ``default_model``, or at least one entry in ``models``.
    """
    model_config = _agent_model_config(managed or {}, tool)
    if _str(model_config.get("model_provider_service")) or _str(model_config.get("default_model")):
        return True
    models = model_config.get("models")
    if isinstance(models, dict):
        return any(_str(value) for value in models.values())
    if isinstance(models, list):
        return any(_str(item) for item in models)
    return False


def managed_provider_service(managed: dict, tool: str) -> str | None:
    """Return only the provider the managed config specifies for ``tool``, ignoring local state."""
    return _str(_agent_model_config(managed, tool).get("model_provider_service"))


def managed_default_model(managed: dict, tool: str) -> str | None:
    """Return the model the managed config wants ``tool`` to launch on, if it names one.

    Distinct from the family slots :func:`effective_agent_models` resolves: those set what each
    family shortcut maps to, while this is the model the session actually starts on. The launch path
    pins it explicitly, so the admin's choice holds even for agents that would otherwise pick their
    own default."""
    return _str(_agent_model_config(managed, tool).get("default_model"))


def resolve_state(managed: dict, state: dict, tool: str) -> dict:
    """Return a copy of ``state`` with ``tool``'s managed values layered on top.

    ``write_tool_config`` reads its models and provider out of the state dict it is handed, so
    handing it this resolved copy is what makes managed settings win. Each key the managed config
    displaces is recorded under :data:`~ucode.state.MANAGED_OVERLAY_KEY` with the developer's own
    value (None when they had none), which ``save_state`` swaps back before writing — so the admin's
    settings reach the generated agent config files without ``state.json`` losing what the developer
    configured. The two files are never merged on disk.
    """
    resolved = dict(state)
    overlay: dict[str, object] = {}
    models = effective_agent_models(managed, state, tool)
    models_key = f"{tool}_models"
    if models is not None and models != state.get(models_key):
        overlay[models_key] = state.get(models_key)
        resolved[models_key] = models
    provider = managed_provider_service(managed, tool)
    if provider:
        providers = dict(_as_dict(state.get("provider_services")))
        if providers.get(tool) != provider:
            overlay["provider_services"] = state.get("provider_services")
            providers[tool] = provider
            resolved["provider_services"] = providers
    if overlay:
        resolved[MANAGED_OVERLAY_KEY] = overlay
    return resolved
