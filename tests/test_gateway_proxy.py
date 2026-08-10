"""Tests for the relayed refresh proxy header handling."""

from __future__ import annotations

import io
import socket
from email.message import Message

from ucode import gateway_proxy


class _FakeHandler:
    """Minimal stand-in exposing a `.headers` mapping like BaseHTTPRequestHandler."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class TestForwardedRequestHeaders:
    def test_injects_swap_header_with_bearer(self):
        handler = _FakeHandler({"Authorization": "Bearer anthropic-oauth"})
        out = gateway_proxy._forwarded_request_headers(handler, "dbx-token")
        assert out["X-Databricks-AI-Gateway-Token"] == "Bearer dbx-token"

    def test_passes_authorization_through_untouched(self):
        # The caller's Anthropic OAuth must survive verbatim — the proxy never
        # reads or rewrites it.
        handler = _FakeHandler({"Authorization": "Bearer anthropic-oauth"})
        out = gateway_proxy._forwarded_request_headers(handler, "dbx-token")
        assert out["Authorization"] == "Bearer anthropic-oauth"

    def test_overwrites_client_supplied_swap_header(self):
        # A stale settings.json value must not survive; the proxy replaces it.
        handler = _FakeHandler({"X-Databricks-AI-Gateway-Token": "Bearer stale"})
        out = gateway_proxy._forwarded_request_headers(handler, "fresh")
        assert out["X-Databricks-AI-Gateway-Token"] == "Bearer fresh"

    def test_strips_hop_by_hop_headers(self):
        handler = _FakeHandler(
            {"Host": "localhost:9", "Content-Length": "5", "Connection": "keep-alive"}
        )
        out = gateway_proxy._forwarded_request_headers(handler, "t")
        assert "Host" not in out
        assert "Content-Length" not in out
        assert "Connection" not in out


class _BrokenPipeWriter(io.RawIOBase):
    """A wfile stand-in that raises BrokenPipeError on write, mimicking a client
    (Claude Code) that closed the connection mid-response."""

    def write(self, _data):  # type: ignore[override]
        raise BrokenPipeError(32, "Broken pipe")


class TestRelayResponseClientDisconnect:
    def _handler(self, wfile) -> gateway_proxy._ProxyHandler:
        # Bypass BaseHTTPRequestHandler.__init__ (which would service a socket);
        # we only exercise _relay_response's write path. Set the few attributes the
        # send_response/send_header machinery reads (normally populated by __init__).
        handler = object.__new__(gateway_proxy._ProxyHandler)
        handler.wfile = wfile
        handler.request_version = "HTTP/1.1"
        handler.requestline = "POST /v1/messages HTTP/1.1"
        handler.command = "POST"
        handler._headers_buffer = []
        return handler

    def test_relay_swallows_broken_pipe_on_headers(self):
        # Client gone before headers flush: end_headers write raises BrokenPipe.
        handler = self._handler(_BrokenPipeWriter())
        stream = io.BytesIO(b'{"ok":true}')
        # Must not raise — a dead client is a routine teardown, not an error.
        handler._relay_response(200, Message(), stream)

    def test_relay_swallows_connection_reset_mid_stream(self):
        # Headers flush ok, then the client resets while streaming body chunks.
        writes: list[bytes] = []

        class _ResetAfterHeaders(io.RawIOBase):
            def write(self, data):  # type: ignore[override]
                writes.append(bytes(data))
                if b"chunk" in bytes(data):
                    raise ConnectionResetError(54, "Connection reset by peer")
                return len(data)

            def flush(self):
                return None

        handler = self._handler(_ResetAfterHeaders())
        handler._relay_response(200, Message(), io.BytesIO(b"chunk-of-sse-data"))


class TestStartProxyPortFallback:
    def test_falls_back_to_free_port_when_cached_port_busy(self, monkeypatch):
        # A stale proxy from a killed session can still hold the cached port; the
        # bind must fall back to an OS-assigned free port rather than crash.
        class _StubCache:
            def run_refresher(self):
                return None

        monkeypatch.setattr(gateway_proxy, "_TokenCache", lambda workspace, profile: _StubCache())
        # Occupy a port to simulate the leftover proxy holding it.
        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        busy_port = occupied.getsockname()[1]
        try:
            server, _cache = gateway_proxy.start_proxy(
                "https://x.staging.cloud.databricks.com", None, busy_port
            )
            try:
                bound = server.server_address[1]
                assert bound != busy_port  # fell back to a different, free port
                assert bound != 0
            finally:
                server.server_close()
        finally:
            occupied.close()
