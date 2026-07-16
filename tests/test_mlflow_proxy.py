"""Tests for agents/_mlflow_proxy.py — the SSE finish_reason repair proxy."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from ucode.agents import _mlflow_proxy

# An inkling-style stream: content deltas then [DONE], NO finish_reason anywhere.
_STREAM_NO_FINISH = (
    b'data: {"id":"c1","choices":[{"delta":{"content":"ok"},"index":0}]}\n\ndata: [DONE]\n\n'
)
# A well-behaved stream: terminal chunk carries finish_reason.
_STREAM_WITH_FINISH = (
    b'data: {"id":"c2","choices":[{"delta":{"content":"ok"},"index":0}]}\n\n'
    b'data: {"id":"c2","choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n'
    b"data: [DONE]\n\n"
)
# An abnormally-terminated stream: content delta, then the upstream just stops —
# no finish_reason AND no [DONE] (what a mid-stream 429/drop looks like).
_STREAM_ABRUPT_END = b'data: {"id":"c3","choices":[{"delta":{"content":"ok"},"index":0}]}\n\n'
# SSE data line with NO space after the colon (spec-legal) and no finish_reason.
_STREAM_NO_SPACE = (
    b'data:{"id":"c4","choices":[{"delta":{"content":"ok"},"index":0}]}\n\ndata:[DONE]\n\n'
)
# A non-streaming JSON response (some models / non-stream requests).
_JSON_BODY = b'{"id":"c5","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}'


def _make_fake_gateway(
    payload: bytes, content_type: str = "text/event-stream", status: int = 200
) -> tuple[str, HTTPServer]:
    class _Fake(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


def _post_through_proxy(upstream: str, path: str = "/ai-gateway/mlflow/v1/chat/completions"):
    """POST through the proxy; returns (status, body, content_type)."""
    base, stop = _mlflow_proxy.start(upstream)
    try:
        req = urllib.request.Request(
            f"{base}{path}",
            data=b'{"model":"databricks-inkling","stream":true}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, resp.read().decode("utf-8"), resp.headers.get("Content-Type")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8"), exc.headers.get("Content-Type")
    finally:
        stop.set()


def _body(upstream: str) -> str:
    """Convenience: body text of a normal streaming POST through the proxy."""
    return _post_through_proxy(upstream)[1]


class TestFinishReasonInjection:
    def test_injects_finish_reason_when_absent(self):
        upstream, server = _make_fake_gateway(_STREAM_NO_FINISH)
        try:
            out = _body(upstream)
        finally:
            server.shutdown()
        assert '"finish_reason": "stop"' in out or '"finish_reason":"stop"' in out
        # The injected chunk must come before the terminal [DONE].
        assert out.index("finish_reason") < out.index("[DONE]")
        assert out.rstrip().endswith("[DONE]")

    def test_passes_through_when_finish_reason_present(self):
        upstream, server = _make_fake_gateway(_STREAM_WITH_FINISH)
        try:
            out = _body(upstream)
        finally:
            server.shutdown()
        # Exactly one finish_reason — the proxy did not add a second.
        assert out.count("finish_reason") == 1

    def test_forwards_content_unchanged(self):
        upstream, server = _make_fake_gateway(_STREAM_NO_FINISH)
        try:
            out = _body(upstream)
        finally:
            server.shutdown()
        assert '"content":"ok"' in out

    def test_repairs_abrupt_end_with_finish_and_done(self):
        # Upstream stopped after content with no finish_reason and no [DONE]
        # (mid-stream 429/drop). Proxy must synthesize BOTH so a strict client
        # doesn't error on a truncated stream.
        upstream, server = _make_fake_gateway(_STREAM_ABRUPT_END)
        try:
            out = _body(upstream)
        finally:
            server.shutdown()
        assert "finish_reason" in out
        assert out.rstrip().endswith("[DONE]")
        assert '"content":"ok"' in out

    def test_parses_data_line_without_space(self):
        # SSE allows `data:{...}` (no space). The already-present finish_reason
        # must be detected so no duplicate is injected.
        upstream, server = _make_fake_gateway(_STREAM_NO_SPACE)
        try:
            out = _body(upstream)
        finally:
            server.shutdown()
        # One synthetic finish (none was present) — not two.
        assert out.count("finish_reason") == 1


class TestNonStreamingAndErrors:
    def test_non_streaming_json_relayed_unchanged(self):
        # A non-SSE 200 must pass through verbatim with its content-type — no
        # finish_reason injection, no SSE relabeling.
        upstream, server = _make_fake_gateway(_JSON_BODY, content_type="application/json")
        try:
            status, body, ctype = _post_through_proxy(upstream)
        finally:
            server.shutdown()
        assert status == 200
        assert "application/json" in ctype
        assert body == _JSON_BODY.decode()
        assert "chat.completion.chunk" not in body  # no injected SSE chunk

    def test_gateway_error_relayed_with_status(self):
        # A gateway 400 (e.g. max_tokens exceeded) is relayed with its status.
        err = b'{"error":"max_tokens cannot exceed 8192"}'
        upstream, server = _make_fake_gateway(err, content_type="application/json", status=400)
        try:
            status, body, _ = _post_through_proxy(upstream)
        finally:
            server.shutdown()
        assert status == 400
        assert "max_tokens" in body

    def test_connection_failure_returns_502(self):
        # Point the proxy at a dead port: urlopen raises URLError, and the proxy
        # must return a clean 502, not a dead socket.
        status, body, _ = _post_through_proxy("http://127.0.0.1:1")
        assert status == 502
        assert "upstream unreachable" in body

    def test_disallowed_path_returns_404(self):
        # Only the mlflow route may be forwarded (SSRF guard).
        upstream, server = _make_fake_gateway(_JSON_BODY, content_type="application/json")
        try:
            status, body, _ = _post_through_proxy(upstream, path="/api/2.0/secrets/get")
        finally:
            server.shutdown()
        assert status == 404
        assert "not allowed" in body


class TestFinishChunk:
    def test_shape(self):

        chunk = json.loads(_mlflow_proxy._finish_chunk("abc"))
        assert chunk["id"] == "abc"
        assert chunk["choices"][0]["finish_reason"] == "stop"
        assert chunk["object"] == "chat.completion.chunk"


class TestStart:
    def test_returns_local_base_url_and_stop_event(self):
        upstream, server = _make_fake_gateway(_STREAM_NO_FINISH)
        try:
            base, stop = _mlflow_proxy.start(upstream)
            assert base.startswith("http://127.0.0.1:")
            assert isinstance(stop, threading.Event)
            stop.set()
        finally:
            server.shutdown()
