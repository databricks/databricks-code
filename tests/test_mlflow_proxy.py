"""Behavioral tests for Pi's MLflow SSE-repair proxy."""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ucode.agents import _mlflow_proxy

_STREAM_NO_FINISH = (
    b'data: {"id":"c1","choices":[{"delta":{"content":"ok"},"index":0}]}\n\ndata: [DONE]\n\n'
)
_STREAM_WITH_FINISH = (
    b'data: {"id":"c2","choices":[{"delta":{"content":"ok"},"index":0}]}\n\n'
    b'data: {"id":"c2","choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n'
    b"data: [DONE]\n\n"
)


class _Gateway(HTTPServer):
    response_status = 200
    response_type = "text/event-stream"
    response_body = b""
    truncate = False
    received_headers: dict[str, str]


def _gateway(
    body: bytes,
    *,
    status: int = 200,
    content_type: str = "text/event-stream",
    truncate: bool = False,
) -> tuple[str, _Gateway, threading.Thread]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802
            self.server.received_headers = dict(self.headers.items())  # type: ignore[attr-defined]
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self.send_response(self.server.response_status)  # type: ignore[attr-defined]
            self.send_header("Content-Type", self.server.response_type)  # type: ignore[attr-defined]
            advertised = len(self.server.response_body) + (20 if self.server.truncate else 0)  # type: ignore[attr-defined]
            self.send_header("Content-Length", str(advertised))
            self.send_header("X-Upstream", "yes")
            self.end_headers()
            self.wfile.write(self.server.response_body)  # type: ignore[attr-defined]
            self.wfile.flush()
            if self.server.truncate:  # type: ignore[attr-defined]
                self.close_connection = True

        def log_message(self, format: str, *args: object) -> None:
            return

    server = _Gateway(("127.0.0.1", 0), Handler)
    server.response_status = status
    server.response_type = content_type
    server.response_body = body
    server.truncate = truncate
    server.received_headers = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_address[1]}", server, thread


def _proxy(upstream: str):
    started = _mlflow_proxy.start(upstream)
    assert started is not None
    server, base = started
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return base, server, thread


def _stop(server: HTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    assert not thread.is_alive()


def _post(base: str, *, authorization: str | None = None) -> tuple[int, dict[str, str], bytes]:
    headers = {"Content-Type": "application/json"}
    if authorization:
        headers["Authorization"] = authorization
    request = urllib.request.Request(
        f"{base}/ai-gateway/mlflow/v1/chat/completions",
        data=b'{"stream":true}',
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()


class TestSseRepair:
    def test_healthy_stream_body_is_byte_identical(self):
        upstream, gateway, gateway_thread = _gateway(_STREAM_WITH_FINISH)
        base, proxy, proxy_thread = _proxy(upstream)
        try:
            status, _, body = _post(base)
        finally:
            _stop(proxy, proxy_thread)
            _stop(gateway, gateway_thread)
        assert status == 200
        assert body == _STREAM_WITH_FINISH
        assert body.count(b"finish_reason") == 1

    def test_missing_finish_is_injected_before_done(self):
        upstream, gateway, gateway_thread = _gateway(_STREAM_NO_FINISH)
        base, proxy, proxy_thread = _proxy(upstream)
        try:
            _, headers, body = _post(base)
        finally:
            _stop(proxy, proxy_thread)
            _stop(gateway, gateway_thread)
        assert body.count(b"finish_reason") == 1
        assert body.index(b"finish_reason") < body.index(b"[DONE]")
        assert "Content-Length" not in headers

    def test_data_field_without_space_and_absent_id(self):
        stream = b'data:{"choices":[{"delta":{"content":"ok"}}]}\n\ndata:[DONE]\n\n'
        upstream, gateway, gateway_thread = _gateway(stream)
        base, proxy, proxy_thread = _proxy(upstream)
        try:
            _, _, body = _post(base)
        finally:
            _stop(proxy, proxy_thread)
            _stop(gateway, gateway_thread)
        assert b"finish_reason" in body
        assert b'"id":null' not in body

    def test_truncated_content_stream_gets_finish_and_done(self):
        stream = b'data: {"id":"c3","choices":[{"delta":{"content":"ok"}}]}\n\n'
        upstream, gateway, gateway_thread = _gateway(stream, truncate=True)
        base, proxy, proxy_thread = _proxy(upstream)
        try:
            _, _, body = _post(base)
        finally:
            _stop(proxy, proxy_thread)
            _stop(gateway, gateway_thread)
        assert b"finish_reason" in body
        assert body.rstrip().endswith(b"[DONE]")

    def test_any_choice_finish_reason_suppresses_injection(self):
        stream = (
            b'data: {"choices":[{"delta":{}},{"delta":{},"finish_reason":"length"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        upstream, gateway, gateway_thread = _gateway(stream)
        base, proxy, proxy_thread = _proxy(upstream)
        try:
            _, _, body = _post(base)
        finally:
            _stop(proxy, proxy_thread)
            _stop(gateway, gateway_thread)
        assert body == stream
        assert body.count(b"finish_reason") == 1

    @pytest.mark.parametrize(
        "stream",
        [
            b'data: {"error":{"message":"rate limited"}}\n\n',
            b'event: error\ndata: {"message":"rate limited"}\n\n',
            b'event:error\ndata: {"message":"rate limited"}\n\n',
        ],
    )
    def test_explicit_sse_error_is_not_turned_into_success(self, stream):
        upstream, gateway, gateway_thread = _gateway(stream)
        base, proxy, proxy_thread = _proxy(upstream)
        try:
            _, _, body = _post(base)
        finally:
            _stop(proxy, proxy_thread)
            _stop(gateway, gateway_thread)
        assert body == stream
        assert b"finish_reason" not in body
        assert b"[DONE]" not in body


class TestPassthroughAndFailures:
    def test_non_streaming_json_status_headers_and_body_preserved(self):
        payload = b'{"choices":[{"message":{"content":"ok"}}]}'
        upstream, gateway, gateway_thread = _gateway(payload, content_type="application/json")
        base, proxy, proxy_thread = _proxy(upstream)
        try:
            status, headers, body = _post(base)
        finally:
            _stop(proxy, proxy_thread)
            _stop(gateway, gateway_thread)
        assert status == 200
        assert headers["Content-Type"] == "application/json"
        assert headers["Content-Length"] == str(len(payload))
        assert headers["X-Upstream"] == "yes"
        assert body == payload

    def test_http_error_is_relayed_without_repair(self):
        payload = b'{"error":"rate limited"}'
        upstream, gateway, gateway_thread = _gateway(
            payload, status=429, content_type="application/json"
        )
        base, proxy, proxy_thread = _proxy(upstream)
        try:
            status, headers, body = _post(base)
        finally:
            _stop(proxy, proxy_thread)
            _stop(gateway, gateway_thread)
        assert status == 429
        assert headers["Content-Type"] == "application/json"
        assert body == payload
        assert b"finish_reason" not in body

    def test_connection_refused_returns_controlled_502(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        base, proxy, proxy_thread = _proxy(f"http://127.0.0.1:{port}")
        try:
            status, _, body = _post(base)
        finally:
            _stop(proxy, proxy_thread)
        assert status == 502
        assert json.loads(body)["error"]
        assert b"finish_reason" not in body

    def test_absolute_request_target_is_rejected_without_forwarding_auth(self):
        upstream, gateway, gateway_thread = _gateway(_STREAM_WITH_FINISH)
        base, proxy, proxy_thread = _proxy(upstream)
        proxy_port = int(base.rsplit(":", 1)[1])
        attacker, attacker_server, attacker_thread = _gateway(
            b"captured", content_type="text/plain"
        )
        request = (
            f"POST {attacker}/capture HTTP/1.1\r\n"
            "Host: ignored\r\n"
            "Authorization: Bearer secret-value\r\n"
            "Content-Length: 2\r\n"
            "Connection: close\r\n\r\n{}"
        ).encode()
        sock = socket.create_connection(("127.0.0.1", proxy_port), timeout=5)
        try:
            sock.sendall(request)
            response = b""
            while chunk := sock.recv(4096):
                response += chunk
        finally:
            sock.close()
            _stop(proxy, proxy_thread)
            _stop(gateway, gateway_thread)
            _stop(attacker_server, attacker_thread)
        assert b" 400 " in response.split(b"\r\n", 1)[0]
        assert gateway.received_headers == {}
        assert attacker_server.received_headers == {}

    def test_authorization_forwarded_and_hop_by_hop_headers_removed(self):
        upstream, gateway, gateway_thread = _gateway(_STREAM_WITH_FINISH)
        base, proxy, proxy_thread = _proxy(upstream)
        try:
            _post(base, authorization="Bearer secret-value")
        finally:
            _stop(proxy, proxy_thread)
            _stop(gateway, gateway_thread)
        lowered = {key.lower(): value for key, value in gateway.received_headers.items()}
        assert lowered["authorization"] == "Bearer secret-value"
        # urllib regenerates identity after the client value is stripped, so
        # the parseable SSE cannot arrive gzip-compressed.
        assert lowered["accept-encoding"] == "identity"
        assert lowered["host"].startswith("127.0.0.1:")  # regenerated for upstream


class TestLifecycle:
    def test_shutdown_and_server_close_release_port(self):
        started = _mlflow_proxy.start("https://example.com")
        assert started is not None
        server, _ = started
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _stop(server, thread)
        rebound = socket.socket()
        try:
            rebound.bind(("127.0.0.1", port))
        finally:
            rebound.close()

    def test_repeated_start_stop_uses_fresh_live_servers(self):
        ports = []
        for _ in range(3):
            started = _mlflow_proxy.start("https://example.com")
            assert started is not None
            server, _ = started
            ports.append(server.server_address[1])
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            _stop(server, thread)
        assert all(isinstance(port, int) and port > 0 for port in ports)

    def test_bind_failure_warns_and_degrades_to_direct_gateway(self, monkeypatch):
        warnings = []
        monkeypatch.setattr(
            _mlflow_proxy,
            "_Server",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no sockets")),
        )
        monkeypatch.setattr(_mlflow_proxy, "print_warning", warnings.append)

        assert _mlflow_proxy.start("https://example.com") is None
        assert warnings and "not started" in warnings[0]
