"""Local SSE-repair proxy for the MLflow chat-completions gateway.

Some MLflow-served models (notably ``databricks-inkling``) stream a
spec-incomplete response: on natural completion the terminal chunk omits the
OpenAI-required ``finish_reason`` and the stream ends with only ``data: [DONE]``.
Strict clients — Pi's ``openai-completions`` parser among them — reject this
with "Stream ended without finish_reason", so the model is unusable.

This proxy sits between Pi and the gateway. It forwards every request verbatim
and, for streaming (``text/event-stream``) responses, repairs the terminator:
when the stream ends — cleanly at ``[DONE]`` OR abnormally (an upstream drop /
mid-stream 429) — after some data but without a ``finish_reason``, it
synthesizes a ``finish_reason: "stop"`` chunk (and a ``[DONE]`` if the upstream
never sent one). Non-streaming and error responses are relayed byte-for-byte
with their original ``Content-Type``. Well-behaved streams never trigger the
injection, so the proxy is a no-op for them.

Tracked upstream as the gateway bug that should make this unnecessary; once the
gateway emits ``finish_reason`` this module can be deleted.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# Request headers we must not forward: hop-by-hop or ones urllib recomputes.
_SKIP_REQUEST_HEADERS = frozenset({"host", "content-length", "accept-encoding", "connection"})

# The proxy only exists to front the mlflow chat-completions route. Refuse any
# other path so a co-located process can't turn the localhost relay into an
# arbitrary authenticated workspace client (SSRF-to-workspace) using the token
# we forward.
_ALLOWED_PATH_PREFIX = "/ai-gateway/mlflow/"


def _make_handler(upstream: str) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            # Read the body regardless so the socket is drained before we close.
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            if not self.path.startswith(_ALLOWED_PATH_PREFIX):
                self._send_bytes(404, b'{"error":"path not allowed"}', "application/json")
                return

            req = urllib.request.Request(upstream + self.path, data=body, method="POST")
            for key, value in self.headers.items():
                if key.lower() not in _SKIP_REQUEST_HEADERS:
                    req.add_header(key, value)
            try:
                upstream_resp = urllib.request.urlopen(req)  # noqa: S310 (trusted upstream)
            except urllib.error.HTTPError as exc:
                # Relay the gateway's own error verbatim (status + body).
                payload = exc.read() if exc.fp else b""
                content_type = exc.headers.get("Content-Type", "application/json")
                self._send_bytes(exc.code, payload, content_type)
                return
            except (urllib.error.URLError, OSError) as exc:
                # Connection-level failure (DNS/TLS/timeout/reset): give the
                # client a clean 502 instead of a dead socket.
                msg = json.dumps({"error": f"upstream unreachable: {exc}"}).encode()
                self._send_bytes(502, msg, "application/json")
                return

            content_type = upstream_resp.headers.get("Content-Type", "")
            if "text/event-stream" not in content_type:
                # Non-streaming (or error-envelope) 200: relay unchanged.
                self._send_bytes(
                    upstream_resp.status, upstream_resp.read(), content_type or "application/json"
                )
                return

            # Streaming: close the connection at end-of-stream so an HTTP/1.1
            # client can detect the message boundary (we send no length/chunking).
            self.close_connection = True
            self.send_response(upstream_resp.status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self._relay_stream(upstream_resp)

        def _relay_stream(self, upstream_resp) -> None:
            saw_finish = False
            saw_done = False
            saw_data = False
            last_id: str | None = None
            try:
                for raw in upstream_resp:  # yields line-by-line as bytes arrive
                    payload = _sse_data_payload(raw)
                    if payload is not None and payload != b"[DONE]":
                        saw_data = True
                        try:
                            obj = json.loads(payload)
                            last_id = obj.get("id", last_id)
                            choice = (obj.get("choices") or [{}])[0]
                            if choice.get("finish_reason"):
                                saw_finish = True
                        except (ValueError, AttributeError, IndexError):
                            pass
                        self._write(raw)
                    elif payload == b"[DONE]":
                        if not saw_finish:
                            self._write(b"data: " + _finish_chunk(last_id) + b"\n\n")
                            saw_finish = True
                        self._write(raw)
                        saw_done = True
                    else:
                        self._write(raw)
                # Clean or abnormal end after data but without a terminator:
                # synthesize one so strict clients don't error on a truncated
                # stream (a mid-stream 429 is the common trigger).
                if saw_data and not saw_finish:
                    self._write(b"data: " + _finish_chunk(last_id) + b"\n\n")
                if saw_data and not saw_done:
                    self._write(b"data: [DONE]\n\n")
            except OSError:
                # Upstream dropped, or the client disconnected mid-write. Either
                # way the socket is gone; swallow so the daemon thread exits
                # quietly rather than tracebacking.
                pass

        def _send_bytes(self, status: int, data: bytes, content_type: str) -> None:
            self.close_connection = True
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()
            except OSError:
                pass

        def _write(self, data: bytes) -> None:
            self.wfile.write(data)
            self.wfile.flush()

        def log_message(self, format: str, *args) -> None:  # silence default stderr logging
            pass

    return _Handler


def _sse_data_payload(raw: bytes) -> bytes | None:
    """Return the payload of an SSE ``data:`` line (the spec allows an optional
    single space after the colon), or None if the line isn't a data line."""
    stripped = raw.strip()
    if not stripped.startswith(b"data:"):
        return None
    payload = stripped[len(b"data:") :]
    if payload.startswith(b" "):
        payload = payload[1:]
    return payload


def _finish_chunk(chunk_id: str | None) -> bytes:
    return json.dumps(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
    ).encode()


class _Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start(upstream: str) -> tuple[str, threading.Event]:
    """Start the repair proxy on a free localhost port.

    :param upstream: gateway origin to forward to, e.g.
        ``https://workspace.databricks.com``.
    :returns: ``(base_url, stop_event)`` where ``base_url`` is the local origin
        Pi should target (``http://127.0.0.1:<port>``) and setting
        ``stop_event`` shuts the server down.
    """
    server = _Server(("127.0.0.1", 0), _make_handler(upstream.rstrip("/")))
    port = server.server_address[1]
    stop_event = threading.Event()

    threading.Thread(target=server.serve_forever, daemon=True).start()

    def _watch_stop() -> None:
        stop_event.wait()
        server.shutdown()

    threading.Thread(target=_watch_stop, daemon=True).start()
    return f"http://127.0.0.1:{port}", stop_event
