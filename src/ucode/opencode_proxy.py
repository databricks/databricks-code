"""Loopback refresh proxy for OpenCode's Databricks AI Gateway providers.

OpenCode resolves ``opencode.json`` once at process start and never re-reads it,
so a Databricks OAuth token baked into the config (the provider ``Authorization``
header) goes stale at the ~1h token lifetime and every request then fails with
``401 Invalid Token``. ``ucode opencode`` therefore points each provider's
``baseURL`` at this loopback proxy instead: it forwards every request to the
workspace gateway unchanged except for overwriting ``Authorization`` with a
freshly-minted Databricks token, and streams the response back verbatim.

Unlike the relayed-Anthropic proxy (``gateway_proxy``), OpenCode has no
subscription-OAuth credential to preserve: the Databricks token *is* the
``Authorization`` credential, so that is the header this proxy swaps. All
OpenCode provider paths (anthropic ``/v1``, gemini ``/v1beta``, mlflow ``/v1``)
share the workspace host, so one proxy serves them all — it swaps host +
``Authorization`` and forwards the client's path unchanged.

Security invariants (mirroring ``databricks.py`` token handling):
  - Binds 127.0.0.1 only; never exposed off-host.
  - Never logs header values or bodies. The Databricks token lives in memory,
    refreshed off the request path.
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

# Header carrying OpenCode's credential; overwritten with the minted token so a
# stale value baked into opencode.json can never reach the gateway.
_AUTH_HEADER = "Authorization"
# Hop-by-hop headers must not be forwarded across the proxy.
_HOP_BY_HOP = frozenset(
    h.lower()
    for h in (
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    )
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
        if key.lower() not in _HOP_BY_HOP and key.lower() != _AUTH_HEADER.lower()
    }
    headers[_AUTH_HEADER] = f"Bearer {token}"
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
            # Upstream (gateway) error — relay status + body verbatim so the agent
            # sees the real error (e.g. 429 rate_limit_error).
            self._relay_response(exc.code, exc.headers, exc)
        except (urllib_error.URLError, OSError):
            try:
                self.send_error(502, "opencode proxy upstream error")
            except OSError:
                pass

    # Streaming passthrough so SSE token streaming is not buffered.
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
            # Client (opencode) closed the connection mid-response — routine on
            # cancelled turns / SSE teardown; nothing left to relay to.
            return

    # Forward every method: transparent pass-through, so the gateway rejects
    # unsupported methods rather than this proxy.
    def __getattr__(self, name: str):
        if name.startswith("do_"):
            return self._handle
        raise AttributeError(name)


def start_proxy(
    workspace: str, profile: str | None, port: int = 0
) -> tuple[ThreadingHTTPServer, _TokenCache]:
    """Start the OpenCode loopback refresh proxy + its background token refresher.

    Forwards to the workspace host (all OpenCode gateway paths live under it) and
    overwrites ``Authorization`` with a freshly-minted token on every request.
    Binds ``port`` (default 0 = an OS-assigned free port); the caller reads
    ``server.server_address[1]`` and points OpenCode's provider baseURLs at it.
    Returns (server, cache); the caller runs the server (e.g. in a thread) and
    calls shutdown()/cache.stop() on exit.
    """
    upstream_base = f"{workspace.rstrip('/')}/"
    cache = _TokenCache(workspace, profile)

    handler = type(
        "BoundOpencodeProxyHandler",
        (_ProxyHandler,),
        {"cache": cache, "upstream_base": upstream_base},
    )
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)

    refresher = threading.Thread(target=cache.run_refresher, daemon=True)
    refresher.start()
    return server, cache
