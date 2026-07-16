"""Tests for agents/_mlflow_proxy.py — the SSE finish_reason repair proxy."""

from __future__ import annotations

import threading
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


def _make_fake_gateway(payload: bytes) -> tuple[str, HTTPServer]:
    class _Fake(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


def _post_through_proxy(upstream: str) -> str:
    base, stop = _mlflow_proxy.start(upstream)
    try:
        req = urllib.request.Request(
            f"{base}/ai-gateway/mlflow/v1/chat/completions",
            data=b'{"model":"databricks-inkling","stream":true}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read().decode("utf-8")
    finally:
        stop.set()


class TestFinishReasonInjection:
    def test_injects_finish_reason_when_absent(self):
        upstream, server = _make_fake_gateway(_STREAM_NO_FINISH)
        try:
            out = _post_through_proxy(upstream)
        finally:
            server.shutdown()
        assert '"finish_reason": "stop"' in out or '"finish_reason":"stop"' in out
        # The injected chunk must come before the terminal [DONE].
        assert out.index("finish_reason") < out.index("[DONE]")
        assert out.rstrip().endswith("[DONE]")

    def test_passes_through_when_finish_reason_present(self):
        upstream, server = _make_fake_gateway(_STREAM_WITH_FINISH)
        try:
            out = _post_through_proxy(upstream)
        finally:
            server.shutdown()
        # Exactly one finish_reason — the proxy did not add a second.
        assert out.count("finish_reason") == 1

    def test_forwards_content_unchanged(self):
        upstream, server = _make_fake_gateway(_STREAM_NO_FINISH)
        try:
            out = _post_through_proxy(upstream)
        finally:
            server.shutdown()
        assert '"content":"ok"' in out

    def test_repairs_abrupt_end_with_finish_and_done(self):
        # Upstream stopped after content with no finish_reason and no [DONE]
        # (mid-stream 429/drop). Proxy must synthesize BOTH so a strict client
        # doesn't error on a truncated stream.
        upstream, server = _make_fake_gateway(_STREAM_ABRUPT_END)
        try:
            out = _post_through_proxy(upstream)
        finally:
            server.shutdown()
        assert "finish_reason" in out
        assert out.rstrip().endswith("[DONE]")
        assert '"content":"ok"' in out


class TestFinishChunk:
    def test_shape(self):
        import json

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
