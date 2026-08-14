from __future__ import annotations

import json
from pathlib import Path

from scripts import prompt_cache_repro as repro


def test_retention_mapping():
    assert repro._retention_value("long") == "24h"
    assert repro._retention_value("in_memory") == "in_memory"
    assert repro._retention_value("default") is None


def test_usage_supports_responses_details():
    usage = repro._usage(
        {
            "usage": {
                "input_tokens": 5000,
                "output_tokens": 20,
                "input_tokens_details": {
                    "cached_tokens": 4000,
                    "cache_write_tokens": 900,
                },
            }
        }
    )
    assert usage == {
        "input_tokens": 5000,
        "cached_tokens": 4000,
        "cache_write_tokens": 900,
        "output_tokens": 20,
    }


def test_request_sends_key_and_long_retention(monkeypatch):
    captured: dict = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"usage": {}}'

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data)
        captured["headers"] = dict(request.headers)
        return Response()

    monkeypatch.setattr(repro.urllib.request, "urlopen", fake_urlopen)
    repro._request(
        workspace="https://workspace.example",
        token="secret",
        model="system.ai.gpt-5-6-sol",
        key="cache-test",
        input_items=[{"role": "user", "content": "hello"}],
        retention="long",
    )

    assert captured["body"]["prompt_cache_key"] == "cache-test"
    assert captured["body"]["prompt_cache_retention"] == "24h"
    assert captured["body"]["max_output_tokens"] == 64
    assert "secret" not in json.dumps(captured["body"])


def test_request_uses_explicit_in_memory(monkeypatch):
    captured: dict = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"usage": {}}'

    monkeypatch.setattr(
        repro.urllib.request,
        "urlopen",
        lambda request, timeout: captured.update(body=json.loads(request.data)) or Response(),
    )
    repro._request(
        workspace="https://workspace.example",
        token="secret",
        model="system.ai.gpt-5-6-sol",
        key="cache-test",
        input_items=[{"role": "user", "content": "hello"}],
        retention="in_memory",
    )

    assert captured["body"]["prompt_cache_retention"] == "in_memory"


def test_state_round_trip(tmp_path: Path):
    state_file = tmp_path / "cache.json"
    state_file.write_text(json.dumps({"assistant_text": "ack"}))
    assert repro._load_saved(state_file)["assistant_text"] == "ack"
