"""Local SSE-repair proxy for the MLflow chat-completions gateway.

Some MLflow-served models (notably ``databricks-inkling``) stream a
spec-incomplete response: on natural completion the terminal chunk omits the
OpenAI-required ``finish_reason`` and the stream ends with only ``data: [DONE]``.
Strict clients — Pi's ``openai-completions`` parser among them — reject this
with "Stream ended without finish_reason", so the model is unusable.

This proxy sits between Pi and the gateway. It forwards every request verbatim
and passes every response byte through unchanged, except that it repairs the
stream terminator: when a streaming response ends — cleanly at ``[DONE]`` OR
abnormally (an upstream drop / mid-stream 429, which otherwise reaches Pi as
"Stream ended without finish_reason") — after some data but without a
``finish_reason``, it synthesizes a ``finish_reason: "stop"`` chunk (and a
``[DONE]`` if the upstream never sent one). Models that already terminate
correctly (glm, llama, qwen, ...) never trigger the injection, so the proxy is
a no-op for them.

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


def _make_handler(upstream: str) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            req = urllib.request.Request(upstream + self.path, data=body, method="POST")
            for key, value in self.headers.items():
                if key.lower() not in _SKIP_REQUEST_HEADERS:
                    req.add_header(key, value)
            try:
                upstream_resp = urllib.request.urlopen(req)  # noqa: S310 (trusted upstream)
            except urllib.error.HTTPError as exc:
                payload = exc.read()
                self.send_response(exc.code)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

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
                    stripped = raw.strip()
                    if stripped.startswith(b"data: ") and stripped[6:] != b"[DONE]":
                        saw_data = True
                        try:
                            obj = json.loads(stripped[6:])
                            last_id = obj.get("id", last_id)
                            choice = (obj.get("choices") or [{}])[0]
                            if choice.get("finish_reason"):
                                saw_finish = True
                        except (ValueError, AttributeError, IndexError):
                            pass
                        self._write(raw)
                    elif stripped == b"data: [DONE]":
                        if not saw_finish:
                            self._write(b"data: " + _finish_chunk(last_id) + b"\n\n")
                            saw_finish = True
                        self._write(raw)
                        saw_done = True
                    else:
                        self._write(raw)
            except OSError:
                # Upstream stream dropped mid-flight (throttle/reset). Fall
                # through to the terminator repair below so the client still
                # sees a well-formed end instead of a truncated stream.
                pass
            # If the stream ended (cleanly or abnormally) after some data but
            # without a finish_reason and/or [DONE], synthesize the terminator
            # so strict clients (Pi) don't error on an incomplete stream. A
            # gateway 429 mid-stream is the common trigger.
            if saw_data and not saw_finish:
                self._write(b"data: " + _finish_chunk(last_id) + b"\n\n")
            if saw_data and not saw_done:
                self._write(b"data: [DONE]\n\n")

        def _write(self, data: bytes) -> None:
            self.wfile.write(data)
            self.wfile.flush()

        def log_message(self, format: str, *args) -> None:  # silence default stderr logging
            pass

    return _Handler


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
