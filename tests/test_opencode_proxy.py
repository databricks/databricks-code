"""Tests for the OpenCode loopback refresh proxy."""

from __future__ import annotations

from ucode import opencode_proxy


class _FakeHandler:
    """Minimal stand-in exposing a `.headers` mapping like BaseHTTPRequestHandler."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class _StubCache:
    """Avoids minting a real token when start_proxy runs under test."""

    def run_refresher(self):
        return None

    def stop(self):
        return None


class TestForwardedRequestHeaders:
    def test_overwrites_authorization_with_minted_token(self):
        # OpenCode's baked-in (possibly stale) token must be replaced per request.
        handler = _FakeHandler({"Authorization": "Bearer stale"})
        out = opencode_proxy._forwarded_request_headers(handler, "fresh")
        assert out["Authorization"] == "Bearer fresh"

    def test_injects_authorization_when_client_sent_none(self):
        handler = _FakeHandler({"Content-Type": "application/json"})
        out = opencode_proxy._forwarded_request_headers(handler, "tok")
        assert out["Authorization"] == "Bearer tok"

    def test_strips_hop_by_hop_headers(self):
        handler = _FakeHandler(
            {"Host": "localhost:9", "Content-Length": "5", "Connection": "keep-alive"}
        )
        out = opencode_proxy._forwarded_request_headers(handler, "t")
        assert "Host" not in out
        assert "Content-Length" not in out
        assert "Connection" not in out


class TestStartProxy:
    def test_forwards_to_workspace_root_so_gateway_path_is_preserved(self, monkeypatch):
        monkeypatch.setattr(opencode_proxy, "_TokenCache", lambda workspace, profile: _StubCache())
        server, cache = opencode_proxy.start_proxy("https://ws.databricks.com", None)
        try:
            assert server.RequestHandlerClass.upstream_base == "https://ws.databricks.com/"
        finally:
            cache.stop()
            server.server_close()
