"""Tests for Anthropic model discovery transformations."""

from __future__ import annotations

import io
import json

from ucode import anthropic_gateway_proxy


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
        self.response = response
        self.request = None

    def stream(self, method, url, headers, content):
        self.request = (method, url, headers, content)
        return self.response


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
    handler = object.__new__(anthropic_gateway_proxy._AnthropicGatewayHandler)
    handler.wfile = wfile
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"{command} {path} HTTP/1.1"
    handler.command = command
    handler.path = path
    handler._headers_buffer = []
    handler.anthropic_model_aliases = anthropic_gateway_proxy._AnthropicModelAliases()
    handler.model_cache = anthropic_gateway_proxy._ModelCache(handler.anthropic_model_aliases)
    return handler


class TestAnthropicModelAliases:
    def test_prefixes_custom_model_ids_without_changing_display_name(self):
        aliases = anthropic_gateway_proxy._AnthropicModelAliases()
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
        aliases = anthropic_gateway_proxy._AnthropicModelAliases()
        aliases.prefix_model_ids(b'{"data":[{"id":"catalog.schema.custom"}]}')

        body = aliases.rewrite_body(
            "/v1/messages",
            b'{"model":"anthropic-aigw-catalog.schema.custom","messages":[]}',
        )

        assert json.loads(body) == {"model": "catalog.schema.custom", "messages": []}

    def test_rewrites_known_alias_in_pagination_cursor(self):
        aliases = anthropic_gateway_proxy._AnthropicModelAliases()
        aliases.prefix_model_ids(b'{"data":[{"id":"catalog.schema.custom"}]}')

        assert (
            aliases.rewrite_path(
                "/v1/models?limit=1000&after_id=anthropic-aigw-catalog.schema.custom"
            )
            == "/v1/models?limit=1000&after_id=catalog.schema.custom"
        )

    def test_ignores_non_anthropic_models_path(self):
        aliases = anthropic_gateway_proxy._AnthropicModelAliases()
        path = "/codex/v1/models?after_id=anthropic-aigw-catalog.schema.custom"

        assert aliases.rewrite_path(path) == path

    def test_does_not_strip_unknown_prefixed_id(self):
        aliases = anthropic_gateway_proxy._AnthropicModelAliases()
        unknown = "anthropic-aigw-legitimate-upstream-id"

        assert aliases.rewrite_path(f"/v1/models?after_id={unknown}") == (
            f"/v1/models?after_id={unknown}"
        )
        assert (
            aliases.rewrite_body("/v1/messages", json.dumps({"model": unknown}).encode())
            == json.dumps({"model": unknown}).encode()
        )

    def test_leaves_malformed_discovery_response_unchanged(self):
        aliases = anthropic_gateway_proxy._AnthropicModelAliases()
        assert aliases.prefix_model_ids(b"not-json") == b"not-json"


class _DiscoveryResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _DiscoveryClient:
    def __init__(self, pages: list[dict]):
        self._pages = iter(pages)
        self.requests: list[tuple[str, dict[str, str], dict[str, str | int]]] = []

    def get(self, path, *, headers, params):
        self.requests.append((path, headers, params))
        return _DiscoveryResponse(next(self._pages))


class TestModelCache:
    def test_prefetches_all_pages_and_serves_one_aliased_response(self):
        aliases = anthropic_gateway_proxy._AnthropicModelAliases()
        cache = anthropic_gateway_proxy._ModelCache(aliases)
        client = _DiscoveryClient(
            [
                {
                    "data": [{"id": "catalog.schema.one"}],
                    "has_more": True,
                    "first_id": "catalog.schema.one",
                    "last_id": "catalog.schema.one",
                },
                {
                    "data": [{"id": "system.ai.claude-sonnet"}],
                    "has_more": False,
                    "first_id": "system.ai.claude-sonnet",
                    "last_id": "system.ai.claude-sonnet",
                },
            ]
        )

        cache.refresh(
            client,
            "token",
            anthropic_gateway_proxy.AUTHORIZATION_HEADER,
        )

        payload = json.loads(cache.get("GET", "/v1/models?limit=1000"))
        assert [model["id"] for model in payload["data"]] == [
            "anthropic-aigw-catalog.schema.one",
            "system.ai.claude-sonnet",
        ]
        assert payload["has_more"] is False
        assert payload["last_id"] == "system.ai.claude-sonnet"
        assert [request[2] for request in client.requests] == [
            {"limit": 1000},
            {"limit": 1000, "after_id": "catalog.schema.one"},
        ]

    def test_only_serves_first_page_get_requests(self):
        cache = anthropic_gateway_proxy._ModelCache(
            anthropic_gateway_proxy._AnthropicModelAliases()
        )
        cache.refresh(
            _DiscoveryClient([{"data": [], "has_more": False}]),
            "token",
            anthropic_gateway_proxy.AUTHORIZATION_HEADER,
        )

        assert cache.get("GET", "/v1/models") is not None
        assert cache.get("POST", "/v1/models") is None
        assert cache.get("GET", "/v1/models?after_id=cursor") is None
        assert cache.get("GET", "/v1/models?before_id=cursor") is None
        assert cache.get("GET", "/v1/messages") is None


class TestAnthropicGatewayHandler:
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

    def test_serves_cached_models_without_upstream_request(self):
        out = _Collect()
        handler = _handler(out)
        handler.headers = {}
        handler.rfile = io.BytesIO()
        handler.cache = _FakeCache()
        handler.client = _FakeClient(None)
        handler.model_cache.refresh(
            _DiscoveryClient([{"data": [{"id": "system.ai.claude"}], "has_more": False}]),
            "token",
            anthropic_gateway_proxy.AUTHORIZATION_HEADER,
        )

        handler._handle()

        assert handler.client.request is None
        assert b"system.ai.claude" in bytes(out.data)

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
    monkeypatch.setattr(anthropic_gateway_proxy, "_TokenCache", lambda *_args, **_kwargs: cache)

    server, actual_cache, client = anthropic_gateway_proxy.start_proxy(
        "https://workspace.example.com", "profile", 0, "header", False
    )
    try:
        handler = server.RequestHandlerClass
        assert issubclass(handler, anthropic_gateway_proxy._AnthropicGatewayHandler)
        assert handler.cache is cache
        assert isinstance(
            handler.anthropic_model_aliases,
            anthropic_gateway_proxy._AnthropicModelAliases,
        )
        assert isinstance(handler.model_cache, anthropic_gateway_proxy._ModelCache)
        assert actual_cache is cache
    finally:
        server.server_close()
        client.close()
