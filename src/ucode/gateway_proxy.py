"""Loopback refresh proxy for relayed Anthropic (Claude Max/Team/Enterprise).

A relayed Model Provider Service authenticates the caller's own Anthropic
subscription OAuth (which Claude Code owns in the `Authorization` header) and
carries a Databricks credential in the `X-Databricks-AI-Gateway-Token` swap
header. That Databricks token is short-lived and a static settings.json header
can't be refreshed, so `ucode claude` points `ANTHROPIC_BASE_URL` at this proxy
instead: it forwards every request to the workspace gateway unchanged except for
adding a freshly-minted swap header, and streams the response back verbatim.

Security invariants (mirroring `databricks.py` token handling):
  - Binds 127.0.0.1 only; never exposed off-host.
  - Never logs header values or bodies. The Databricks token lives in memory,
    refreshed off the request path; the Anthropic OAuth in `Authorization` is
    passed through untouched and never read, stored, or logged.
"""

from __future__ import annotations

import threading
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import IO
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urljoin

from ucode.databricks import TOKEN_REFRESH_INTERVAL_SECONDS, get_databricks_token

# Header we overwrite with the freshly-minted Databricks credential. Any
# client-supplied value is replaced, so a stale settings.json value can't leak.
_SWAP_HEADER = "X-Databricks-AI-Gateway-Token"
# Hop-by-hop headers must not be forwarded across the proxy.
_HOP_BY_HOP = frozenset(
    h.lower()
    for h in ("connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length")
)
_STREAM_CHUNK = 8192


class _TokenCache:
    """Holds the current Databricks token, refreshed by a background thread so
    minting never blocks a request."""

    def __init__(self, workspace: str, profile: str | None) -> None:
        self._workspace = workspace
        self._profile = profile
        self._lock = threading.Lock()
        self._token = get_databricks_token(workspace, profile)
        self._stop = threading.Event()

    @property
    def token(self) -> str:
        with self._lock:
            return self._token

    def refresh(self) -> None:
        token = get_databricks_token(self._workspace, self._profile, force_refresh=True)
        with self._lock:
            self._token = token

    def run_refresher(self) -> None:
        while not self._stop.wait(TOKEN_REFRESH_INTERVAL_SECONDS):
            try:
                self.refresh()
            except RuntimeError:
                continue

    def stop(self) -> None:
        self._stop.set()


def _forwarded_request_headers(handler: BaseHTTPRequestHandler, token: str) -> dict[str, str]:
    headers = {
        key: value
        for key, value in handler.headers.items()
        if key.lower() not in _HOP_BY_HOP and key.lower() != _SWAP_HEADER.lower()
    }
    headers[_SWAP_HEADER] = f"Bearer {token}"
    return headers


class _ProxyHandler(BaseHTTPRequestHandler):
    # Set by the server factory.
    cache: _TokenCache
    upstream_base: str

    def log_message(self, format: str, *args: object) -> None:
        return

    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else None
        target = urljoin(self.upstream_base, self.path.lstrip("/"))
        req = urllib_request.Request(
            target,
            data=body,
            method=self.command,
            headers=_forwarded_request_headers(self, self.cache.token),
        )
        try:
            with urllib_request.urlopen(req, timeout=600) as resp:
                self._relay_response(resp.status, resp.headers, resp)
        except urllib_error.HTTPError as exc:
            # Upstream (gateway/Anthropic) error — relay status + body verbatim so
            # the agent sees the real error (e.g. 429 rate_limit_error).
            self._relay_response(exc.code, exc.headers, exc)
        except (urllib_error.URLError, OSError):
            # The client (Claude Code) may already have disconnected, in which case
            # reporting the error writes to a dead socket and raises again; swallow it.
            try:
                self.send_error(502, "gateway proxy upstream error")
            except OSError:
                pass

    # Streaming passthrough: forward chunks as they arrive so SSE token streaming
    # is not buffered (buffering would add full-response latency to first token).
    def _relay_response(self, status: int, headers: Message, stream: IO[bytes]) -> None:
        try:
            self.send_response(status)
            for key, value in headers.items():
                if key.lower() not in _HOP_BY_HOP:
                    self.send_header(key, value)
            self.end_headers()
            while True:
                chunk = stream.read(_STREAM_CHUNK)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Client (Claude Code) closed the connection mid-response — routine on
            # cancelled turns / SSE teardown. There is nothing left to relay to, so
            # stop quietly rather than crashing the handler thread.
            return

    # Forward every method: this is a transparent pass-through, so routing any
    # `do_<METHOD>` lookup to `_handle` lets the gateway reject unsupported methods.
    def __getattr__(self, name: str):
        if name.startswith("do_"):
            return self._handle
        raise AttributeError(name)


def start_proxy(
    workspace: str, profile: str | None, port: int
) -> tuple[ThreadingHTTPServer, _TokenCache]:
    """Start the loopback refresh proxy + its background token refresher.

    Binds ``port``, falling back to a fresh OS-assigned port when it is already
    in use (e.g. a prior session's proxy that was killed before its teardown ran
    still holds the socket). The caller reads ``server.server_address[1]`` for the
    actual port and points Claude Code at it. Returns (server, cache); the caller
    runs the server (e.g. in a thread) and calls shutdown()/cache.stop() on exit.
    """
    upstream_base = f"{workspace.rstrip('/')}/ai-gateway/anthropic/"
    cache = _TokenCache(workspace, profile)

    handler = type(
        "BoundProxyHandler",
        (_ProxyHandler,),
        {"cache": cache, "upstream_base": upstream_base},
    )
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:
        # Cached port is occupied (stale proxy from a killed session). Port 0 lets
        # the OS pick any free port; the caller reconciles the base URL to it.
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)

    refresher = threading.Thread(target=cache.run_refresher, daemon=True)
    refresher.start()
    return server, cache
