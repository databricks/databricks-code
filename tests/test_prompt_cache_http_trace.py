from __future__ import annotations

import io
import json
import urllib.error

import pytest

from scripts import prompt_cache_http_trace as trace


def test_request_body_retention_mapping() -> None:
    inputs = [{"role": "user", "content": "context"}]

    default = trace._request_body(
        model="system.ai.gpt-5-6-sol",
        key="cache-key",
        input_items=inputs,
        retention="default",
    )
    assert "prompt_cache_retention" not in default

    long = trace._request_body(
        model="system.ai.gpt-5-6-sol",
        key="cache-key",
        input_items=inputs,
        retention="long",
    )
    assert long["prompt_cache_retention"] == "24h"

    in_memory = trace._request_body(
        model="system.ai.gpt-5-6-sol",
        key="cache-key",
        input_items=inputs,
        retention="in_memory",
    )
    assert in_memory["prompt_cache_retention"] == "in_memory"


def test_request_prints_body_and_redacts_token(monkeypatch, capsys) -> None:
    token = "dapi-test-secret"

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"echo": f"Bearer {token}", "ok": True}).encode()

    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        captured["headers"] = dict(request.headers)
        return Response()

    monkeypatch.setattr(trace.urllib.request, "urlopen", fake_urlopen)
    trace._request(
        workspace="https://workspace.example",
        token=token,
        body=trace._request_body(
            model="system.ai.gpt-5-6-sol",
            key="cache-key",
            input_items=[{"role": "user", "content": "hello"}],
            retention="long",
        ),
    )

    output = capsys.readouterr().out
    assert captured["body"]["prompt_cache_retention"] == "24h"
    assert "dapi-test-secret" not in output
    assert "<redacted-bearer-token>" in output
    assert "Authorization" not in output


def test_http_error_is_printed_without_token(monkeypatch, capsys) -> None:
    token = "dapi-test-secret"
    error_body = json.dumps({"error": {"message": f"Bearer {token}"}}).encode()

    def fake_urlopen(_request, timeout):
        raise urllib.error.HTTPError(
            "https://workspace.example/ai-gateway/codex/v1/responses",
            400,
            "bad request",
            {},
            io.BytesIO(error_body),
        )

    monkeypatch.setattr(trace.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        trace._request(
            workspace="https://workspace.example",
            token=token,
            body={"prompt_cache_retention": "in_memory"},
        )

    output = capsys.readouterr().out
    assert "dapi-test-secret" not in output
    assert "<redacted-bearer-token>" in output
    assert "response_status: 400" in output
