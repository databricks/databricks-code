from __future__ import annotations

import json

import pytest

from ucode.codex_request import sanitize_reasoning_replay


def _body(model: str, items: list[dict]) -> bytes:
    return json.dumps({"model": model, "input": items}, indent=2).encode()


@pytest.mark.parametrize(
    "model",
    [
        "gpt-6-astra",
        "system.ai.gpt-5-6-sol",
        "databricks-gpt-5.6-terra",
        "eu/gpt-5-6-luna",
    ],
)
def test_strips_visible_reasoning_for_strict_models(model):
    reasoning = {
        "type": "reasoning",
        "id": "rs_provider",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "private"}],
        "encrypted_content": None,
    }
    message = {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "kept"}],
    }

    result = json.loads(sanitize_reasoning_replay(_body(model, [reasoning, message])))

    assert "content" not in result["input"][0]
    assert result["input"][0]["encrypted_content"] is None
    assert result["input"][1] == message


def test_kimi_visible_reasoning_is_unchanged():
    body = _body(
        "system.ai.kimi-k3",
        [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "needed by Kimi"}],
            }
        ],
    )

    assert sanitize_reasoning_replay(body) is body


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        b'{"model":"gpt-future","input":[]}',
        b'{"model":"gpt-5.6-sol","input":"text"}',
        b'{"model":"gpt-5.6-sol","input":[{"type":"reasoning","content":[]}]}',
        b'{"model":"gpt-5.6-sol","input":[{"type":"reasoning"}]}',
    ],
)
def test_unrelated_or_compatible_requests_are_byte_transparent(body):
    assert sanitize_reasoning_replay(body) is body
