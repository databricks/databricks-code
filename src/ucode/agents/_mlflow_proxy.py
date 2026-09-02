"""Loopback SSE-repair proxy for Pi's MLflow chat-completions provider.

Some MLflow-served models omit the terminal OpenAI ``finish_reason``. Pi's
strict ``openai-completions`` parser rejects those streams. This loopback-only
proxy forwards requests without logging credentials or bodies and repairs only
successful SSE responses that have already produced data. Healthy SSE and all
non-streaming/error responses pass through unchanged.
"""

from __future__ import annotations

import json
from email.message import Message
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import IO
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlsplit

from ucode.gateway_proxy import HOP_BY_HOP_HEADERS
from ucode.ui import print_warning

_STREAM_CHUNK = 8192
_CHAT_COMPLETIONS_PATH = "/ai-gateway/mlflow/v1/chat/completions"
_SKIP_REQUEST_HEADERS = HOP_BY_HOP_HEADERS | {"accept-encoding"}
_ERROR_BODY = b'{"error":"MLflow proxy upstream unavailable"}\n'


class _NoRedirect(urllib_request.HTTPRedirectHandler):
    """Keep authenticated requests pinned to the configured workspace origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _finish_chunk(chunk_id: str | None) -> bytes:
    payload: dict = {
        "object": "chat.completion.chunk",
        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
    }
    if chunk_id is not None:
        payload["id"] = chunk_id
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _data_payload(raw_line: bytes) -> bytes | None:
    """Return an SSE data field's payload, accepting the optional one space."""
    stripped = raw_line.rstrip(b"\r\n")
    if not stripped.startswith(b"data:"):
        return None
    payload = stripped[5:]
    return payload[1:] if payload.startswith(b" ") else payload


def _forwarded_request_headers(handler: BaseHTTPRequestHandler) -> dict[str, str]:
    return {
        key: value
        for key, value in handler.headers.items()
        if key.lower() not in _SKIP_REQUEST_HEADERS
    }


def _safe_response_headers(headers: Message, *, streaming: bool) -> list[tuple[str, str]]:
    safe: list[tuple[str, str]] = []
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in HOP_BY_HOP_HEADERS:
            # A non-streaming body is unchanged, so preserving Content-Length
            # avoids relying on EOF framing. Repaired streams can change size.
            if lowered == "content-length" and not streaming:
                safe.append((key, value))
            continue
        safe.append((key, value))
    return safe


class _ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream_origin: str

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 (stdlib handler API)
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
            if length < 0:
                raise ValueError
        except (TypeError, ValueError):
            self._fixed_response(400, b'{"error":"invalid Content-Length"}\n')
            return

        try:
            body = self.rfile.read(length) if length else b""
        except OSError:
            return
        parsed_target = urlsplit(self.path)
        if (
            parsed_target.scheme
            or parsed_target.netloc
            or parsed_target.fragment
            or parsed_target.path != _CHAT_COMPLETIONS_PATH
        ):
            self._fixed_response(400, b'{"error":"invalid MLflow proxy request target"}\n')
            return
        target = self.upstream_origin.rstrip("/") + parsed_target.path
        if parsed_target.query:
            target += f"?{parsed_target.query}"
        request = urllib_request.Request(
            target,
            data=body,
            method="POST",
            headers=_forwarded_request_headers(self),
        )
        try:
            opener = urllib_request.build_opener(_NoRedirect)
            with opener.open(request, timeout=600) as response:  # noqa: S310
                content_type = response.headers.get_content_type().lower()
                if content_type == "text/event-stream":
                    self._relay_sse(response.status, response.headers, response)
                else:
                    self._relay_verbatim(response.status, response.headers, response)
        except urllib_error.HTTPError as exc:
            # Relay upstream status, headers, and bytes verbatim. Never turn an
            # upstream rejection into a successful repaired stream.
            self._relay_verbatim(exc.code, exc.headers, exc)
        except (urllib_error.URLError, OSError):
            self._fixed_response(502, _ERROR_BODY)

    def _send_headers(self, status: int, headers: Message, *, streaming: bool) -> bool:
        try:
            self.send_response(status)
            for key, value in _safe_response_headers(headers, streaming=streaming):
                self.send_header(key, value)
            # The proxy never reuses downstream connections. EOF framing is
            # therefore safe for responses without Content-Length (including
            # 204s and repaired SSE), and shutdown cannot leave a keep-alive
            # client waiting on an otherwise complete response.
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _relay_verbatim(self, status: int, headers: Message, stream: IO[bytes]) -> None:
        if not self._send_headers(status, headers, streaming=False):
            return
        try:
            while chunk := stream.read(_STREAM_CHUNK):
                self.wfile.write(chunk)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError, IncompleteRead):
            return

    def _relay_sse(self, status: int, headers: Message, stream: IO[bytes]) -> None:
        if not self._send_headers(status, headers, streaming=True):
            return
        saw_data = False
        saw_finish = False
        saw_done = False
        saw_error = False
        last_id: str | None = None
        event_data: list[bytes] = []

        def inspect_event() -> None:
            nonlocal saw_error, saw_finish, last_id
            if not event_data:
                return
            payload = b"\n".join(event_data)
            event_data.clear()
            try:
                event = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                return
            if not isinstance(event, dict):
                return
            if "error" in event:
                saw_error = True
            event_id = event.get("id")
            if isinstance(event_id, str):
                last_id = event_id
            choices = event.get("choices")
            if isinstance(choices, list) and any(
                isinstance(choice, dict) and choice.get("finish_reason") is not None
                for choice in choices
            ):
                saw_finish = True

        try:
            for raw_line in stream:
                payload = _data_payload(raw_line)
                event_line = raw_line.rstrip(b"\r\n")
                if not event_line:
                    inspect_event()
                elif event_line.lower().startswith(b"event:"):
                    event_name = event_line[6:]
                    if event_name.startswith(b" "):
                        event_name = event_name[1:]
                    if event_name.lower() == b"error":
                        saw_error = True
                if payload == b"[DONE]":
                    inspect_event()
                    if saw_data and not saw_finish and not saw_error:
                        self._write(b"data: " + _finish_chunk(last_id) + b"\n\n")
                        saw_finish = True
                    self._write(raw_line)
                    saw_done = True
                    continue
                if payload is not None and payload:
                    saw_data = True
                    event_data.append(payload)
                self._write(raw_line)
        except (BrokenPipeError, ConnectionResetError, OSError, IncompleteRead):
            # Never turn a transport-failed partial stream into a successful
            # synthetic completion. EOF without a transport exception remains
            # repairable below because affected gateways can end cleanly after
            # their final data event.
            return

        # ``HTTPResponse`` line iteration can end without raising even when a
        # declared Content-Length was not satisfied. A positive remainder is
        # still a transport truncation, not a clean finish-reason omission.
        remaining = getattr(stream, "length", None)
        if isinstance(remaining, int) and remaining > 0:
            return

        inspect_event()
        if saw_data and not saw_error:
            try:
                if not saw_finish:
                    self._write(b"data: " + _finish_chunk(last_id) + b"\n\n")
                if not saw_done:
                    self._write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

    def _write(self, data: bytes) -> None:
        self.wfile.write(data)
        self.wfile.flush()

    def _fixed_response(self, status: int, body: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start(upstream_origin: str) -> tuple[ThreadingHTTPServer, str] | None:
    """Bind a repair proxy to a fresh loopback port; the caller owns its lifecycle."""
    if not isinstance(upstream_origin, str) or not upstream_origin:
        print_warning("MLflow stream repair proxy was not started: invalid upstream URL.")
        return None
    parsed_origin = urlsplit(upstream_origin)
    if parsed_origin.scheme not in {"http", "https"} or not parsed_origin.netloc:
        print_warning("MLflow stream repair proxy was not started: invalid upstream URL.")
        return None
    handler = type("_BoundProxyHandler", (_ProxyHandler,), {"upstream_origin": upstream_origin})
    try:
        server = _Server(("127.0.0.1", 0), handler)
    except OSError as exc:
        print_warning(f"MLflow stream repair proxy was not started ({exc}).")
        return None
    port = int(server.server_address[1])
    return server, f"http://127.0.0.1:{port}"
