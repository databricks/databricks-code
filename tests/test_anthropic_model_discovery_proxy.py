"""Tests for Anthropic model discovery transformations."""

from __future__ import annotations

import io
import json

from ucode import anthropic_model_discovery_proxy


class _FakeResponse:
    def __init__(self, status_code: int, headers: dict[str, str], body: bytes):
        self.status_code = status_code
        self.headers = headers
        self._body = body
        self.read_calls = 0
        self.iter_raw_calls = 0

    def read(self):
        self.read_calls += 1
        return self._body

    def iter_raw(self):
        self.iter_raw_calls += 1
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeClient:
    def __init__(self, response):
        self.responses = list(response) if isinstance(response, list) else [response]
        self.request = None
        self.requests = []

    def stream(self, method, url, headers, content):
        self.request = (method, url, headers, content)
        self.requests.append(self.request)
        return self.responses.pop(0)


class _FakeCache:
    token = "databricks-token"

    def refresh(self):
        return None


class _Collect(io.RawIOBase):
    def __init__(self):
        self.data = bytearray()

    def write(self, body):  # type: ignore[override]
        self.data += bytes(body)
        return len(body)

    def flush(self):
        return None


def _handler(wfile, path="/v1/models", command="GET"):
    handler = object.__new__(anthropic_model_discovery_proxy._AnthropicModelDiscoveryHandler)
    handler.wfile = wfile
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"{command} {path} HTTP/1.1"
    handler.command = command
    handler.path = path
    handler._headers_buffer = []
    handler.anthropic_model_aliases = anthropic_model_discovery_proxy._AnthropicModelAliases()
    return handler


class TestAnthropicModelAliases:
    def test_prefixes_custom_model_ids_without_changing_display_name(self):
        aliases = anthropic_model_discovery_proxy._AnthropicModelAliases()
        body = json.dumps(
            {
                "data": [
                    {"id": "catalog.schema.custom", "display_name": "Custom model"},
                    {"id": "system.ai.claude-sonnet"},
                    {"id": "claude-sonnet"},
                    {"id": "catalog.schema.anthropic-provider"},
                    {"id": "anthropic-provider"},
                ],
                "first_id": "catalog.schema.custom",
                "last_id": "catalog.schema.anthropic-provider",
            }
        ).encode()

        payload = json.loads(aliases.prefix_model_ids(body))

        assert payload == {
            "data": [
                {
                    "id": "anthropic-aigw-catalog.schema.custom",
                    "display_name": "Custom model",
                },
                {"id": "system.ai.claude-sonnet"},
                {"id": "claude-sonnet"},
                {"id": "catalog.schema.anthropic-provider"},
                {"id": "anthropic-provider"},
            ],
            "first_id": "anthropic-aigw-catalog.schema.custom",
            "last_id": "catalog.schema.anthropic-provider",
        }

    def test_rewrites_known_alias_in_messages_body(self):
        aliases = anthropic_model_discovery_proxy._AnthropicModelAliases()
        aliases.prefix_model_ids(b'{"data":[{"id":"catalog.schema.custom"}]}')

        body = aliases.rewrite_body(
            "/v1/messages",
            b'{"model":"anthropic-aigw-catalog.schema.custom","messages":[]}',
        )

        assert json.loads(body) == {"model": "catalog.schema.custom", "messages": []}

    def test_rewrites_known_alias_in_pagination_cursor(self):
        aliases = anthropic_model_discovery_proxy._AnthropicModelAliases()
        aliases.prefix_model_ids(b'{"data":[{"id":"catalog.schema.custom"}]}')

        assert (
            aliases.rewrite_path(
                "/v1/models?limit=1000&after_id=anthropic-aigw-catalog.schema.custom"
            )
            == "/v1/models?limit=1000&after_id=catalog.schema.custom"
        )

    def test_ignores_non_anthropic_models_path(self):
        aliases = anthropic_model_discovery_proxy._AnthropicModelAliases()
        path = "/codex/v1/models?after_id=anthropic-aigw-catalog.schema.custom"

        assert aliases.rewrite_path(path) == path

    def test_does_not_strip_unknown_prefixed_id(self):
        aliases = anthropic_model_discovery_proxy._AnthropicModelAliases()
        unknown = "anthropic-aigw-legitimate-upstream-id"

        assert aliases.rewrite_path(f"/v1/models?after_id={unknown}") == (
            f"/v1/models?after_id={unknown}"
        )
        assert (
            aliases.rewrite_body("/v1/messages", json.dumps({"model": unknown}).encode())
            == json.dumps({"model": unknown}).encode()
        )

    def test_leaves_malformed_discovery_response_unchanged(self):
        aliases = anthropic_model_discovery_proxy._AnthropicModelAliases()
        assert aliases.prefix_model_ids(b"not-json") == b"not-json"


class TestAnthropicModelDiscoveryHandler:
    def test_inherits_relayed_auth_and_prefixes_models(self):
        out = _Collect()
        handler = _handler(out)
        handler.headers = {"Authorization": "Bearer subscription-token"}
        handler.rfile = io.BytesIO()
        handler.cache = _FakeCache()
        handler.client = _FakeClient(_FakeResponse(200, {}, b'{"data":[{"id":"custom-model"}]}'))

        handler._handle()

        method, url, headers, body = handler.client.request
        assert (method, url, body) == ("GET", "v1/models", None)
        assert headers["Authorization"] == "Bearer subscription-token"
        assert headers["X-Databricks-AI-Gateway-Token"] == "Bearer databricks-token"
        assert b"anthropic-aigw-custom-model" in bytes(out.data)

    def test_retries_rate_limited_model_discovery(self, monkeypatch):
        out = _Collect()
        handler = _handler(out)
        handler.headers = {"Authorization": "Bearer subscription-token"}
        handler.rfile = io.BytesIO()
        handler.cache = _FakeCache()
        rate_limited = _FakeResponse(429, {"Retry-After": "0"}, b"rate limited")
        success = _FakeResponse(200, {}, b'{"data":[{"id":"custom-model"}]}')
        handler.client = _FakeClient([rate_limited, success])
        monkeypatch.setattr(anthropic_model_discovery_proxy.time, "sleep", lambda _delay: None)

        handler._handle()

        assert len(handler.client.requests) == 2
        assert rate_limited.read_calls == 1
        assert b"429 Too Many Requests" not in bytes(out.data)
        assert b"anthropic-aigw-custom-model" in bytes(out.data)

    def test_relays_rate_limit_after_model_discovery_retries_are_exhausted(self, monkeypatch):
        out = _Collect()
        handler = _handler(out)
        handler.headers = {"Authorization": "Bearer subscription-token"}
        handler.rfile = io.BytesIO()
        handler.cache = _FakeCache()
        responses = [_FakeResponse(429, {"Retry-After": "0"}, b"rate limited") for _ in range(2)]
        handler.client = _FakeClient(responses)
        monkeypatch.setattr(anthropic_model_discovery_proxy.time, "sleep", lambda _delay: None)

        handler._handle()

        assert len(handler.client.requests) == 2
        assert responses[0].read_calls == 1
        assert responses[1].read_calls == 0
        assert b"429 Too Many Requests" in bytes(out.data)
        assert b"rate limited" in bytes(out.data)

    def test_prefixes_successful_model_response_and_drops_content_encoding(self):
        out = _Collect()
        handler = _handler(out)
        response = _FakeResponse(
            200,
            {"Content-Encoding": "gzip"},
            b'{"data":[{"id":"custom-model"}]}',
        )

        handler._relay_response(response)

        assert b"Content-Encoding" not in bytes(out.data)
        assert b"anthropic-aigw-custom-model" in bytes(out.data)

    def test_keeps_content_encoding_for_unchanged_error(self):
        out = _Collect()
        handler = _handler(out)
        response = _FakeResponse(400, {"Content-Encoding": "gzip"}, b"compressed-error")

        handler._relay_response(response)

        assert b"Content-Encoding: gzip" in bytes(out.data)
        assert b"compressed-error" in bytes(out.data)
        assert response.read_calls == 0
        assert response.iter_raw_calls == 1

    def test_streams_relayed_inference_response_without_buffering(self):
        out = _Collect()
        handler = _handler(out, path="/v1/messages", command="POST")
        handler.headers = {"Authorization": "Bearer subscription-token", "Content-Length": "2"}
        handler.rfile = io.BytesIO(b"{}")
        handler.cache = _FakeCache()
        response = _FakeResponse(200, {"Content-Type": "text/event-stream"}, b"data: event\n\n")
        handler.client = _FakeClient(response)

        handler._handle()

        _method, _url, headers, _body = handler.client.request
        assert headers["Authorization"] == "Bearer subscription-token"
        assert headers["X-Databricks-AI-Gateway-Token"] == "Bearer databricks-token"
        assert response.read_calls == 0
        assert response.iter_raw_calls == 1
        assert b"data: event\n\n" in bytes(out.data)

    def test_strips_known_alias_from_message_request(self):
        handler = _handler(_Collect(), path="/v1/messages", command="POST")
        handler.anthropic_model_aliases.prefix_model_ids(
            b'{"data":[{"id":"catalog.schema.custom"}]}'
        )

        url, body = handler._transform_request(b'{"model":"anthropic-aigw-catalog.schema.custom"}')

        assert url == "v1/messages"
        assert json.loads(body) == {"model": "catalog.schema.custom"}


def test_start_proxy_uses_discovery_handler(monkeypatch):
    class _StubCache:
        def run_refresher(self):
            return None

    cache = _StubCache()
    monkeypatch.setattr(
        anthropic_model_discovery_proxy,
        "TokenCache",
        lambda *_args, **_kwargs: cache,
    )

    server, actual_cache, client = anthropic_model_discovery_proxy.start_proxy(
        "https://workspace.example.com", "profile", 0, "header", False
    )
    try:
        handler = server.RequestHandlerClass
        assert issubclass(
            handler,
            anthropic_model_discovery_proxy._AnthropicModelDiscoveryHandler,
        )
        assert handler.cache is cache
        assert isinstance(
            handler.anthropic_model_aliases,
            anthropic_model_discovery_proxy._AnthropicModelAliases,
        )
        assert actual_cache is cache
    finally:
        server.server_close()
        client.close()
