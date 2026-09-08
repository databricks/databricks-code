"""Shared add/remove machinery for ucode-managed smart-routing hooks.

Codex config (TOML) and Claude Code settings (JSON) both express hooks as
``{event: [{matcher?, hooks: [{command, ...}]}]}``, and both identify a
ucode-managed hook by a marker substring in its ``command``. This module owns
the surgical merge (add ucode's groups, leave user hooks untouched) and the
surgical removal (drop only hooks whose command carries the marker), so each
harness module supplies just its own per-event hook groups.
"""

from __future__ import annotations


def sync_managed_hooks(doc: dict, marker: str, groups: dict[str, list[dict]]) -> None:
    """Replace ucode's marked hooks with ``groups``, preserving user hooks."""
    remove_managed_hooks(doc, marker)
    if not groups:
        return
    hooks = doc.setdefault("hooks", {})
    for event, event_groups in groups.items():
        existing = hooks.get(event)
        if not isinstance(existing, list):
            existing = []
        hooks[event] = [*existing, *event_groups]


def remove_managed_hooks(doc: dict, marker: str) -> bool:
    """Remove only hooks whose command contains ``marker``. Returns True if any."""
    hooks = doc.get("hooks")
    if not isinstance(hooks, dict):
        return False
    changed = False
    for event in list(hooks):
        groups = hooks.get(event)
        if not isinstance(groups, list):
            continue
        kept_groups = []
        for group in groups:
            if not isinstance(group, dict):
                kept_groups.append(group)
                continue
            handlers = group.get("hooks")
            if not isinstance(handlers, list):
                kept_groups.append(group)
                continue
            kept_handlers = [
                handler
                for handler in handlers
                if not (isinstance(handler, dict) and marker in str(handler.get("command") or ""))
            ]
            if len(kept_handlers) != len(handlers):
                changed = True
            if kept_handlers:
                group["hooks"] = kept_handlers
                kept_groups.append(group)
        if kept_groups:
            hooks[event] = kept_groups
        else:
            hooks.pop(event, None)
    if not hooks:
        doc.pop("hooks", None)
    return changed
