"""Compatibility rewrites for Codex Responses API requests.

Some non-OpenAI FMAPI models return visible ``reasoning.content`` items. Codex
can replay those items while continuing on the same model, but OpenAI reasoning
models require replayable reasoning state in ``encrypted_content`` and reject a
non-empty ``content`` array. Keep the persisted transcript untouched and remove
only that provider-specific field from requests sent to the affected models.
"""

from __future__ import annotations

import json
import re

_STRICT_REASONING_REPLAY_MODEL_KEYS = frozenset(
    {
        "gpt6astra",
        "gpt56sol",
        "gpt56terra",
        "gpt56luna",
    }
)


def _model_key(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", model.strip().lower())


def _needs_strict_reasoning_replay(model: object) -> bool:
    if not isinstance(model, str):
        return False
    normalized = _model_key(model)
    return any(normalized.endswith(key) for key in _STRICT_REASONING_REPLAY_MODEL_KEYS)


def sanitize_reasoning_replay(body: bytes) -> bytes:
    """Remove nonportable visible reasoning from strict-model request history.

    Invalid JSON, unknown models, and already-compatible requests are returned
    byte-for-byte so the shared proxy remains transparent outside this narrow
    Responses compatibility case.
    """
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict) or not _needs_strict_reasoning_replay(payload.get("model")):
        return body

    items = payload.get("input")
    if not isinstance(items, list):
        return body

    changed = False
    for item in items:
        if (
            isinstance(item, dict)
            and item.get("type") == "reasoning"
            and isinstance(item.get("content"), list)
            and item["content"]
        ):
            item.pop("content")
            changed = True

    if not changed:
        return body
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
