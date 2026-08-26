"""Loopback refresh proxy for Claude gateway requests.

A relayed Model Provider Service authenticates the caller's own Anthropic
subscription OAuth (which Claude Code owns in the `Authorization` header) and
carries a Databricks credential in the `X-Databricks-AI-Gateway-Token` swap
header. Native gateway discovery instead carries the Databricks credential in
`Authorization`. The proxy refreshes the applicable header, streams inference
responses verbatim, and rewrites model discovery responses when needed.

Security invariants (mirroring `databricks.py` token handling):
  - Binds 127.0.0.1 only; never exposed off-host.
  - Never logs header values or bodies. The Databricks token lives in memory,
    refreshed off the request path; the Anthropic OAuth in `Authorization` is
    passed through untouched in relayed mode and never logged.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Iterable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from ucode.constants import LOOPBACK_HOST
from ucode.databricks import get_databricks_token

# Header we overwrite with the freshly-minted Databricks credential. Any
# client-supplied value is replaced, so a stale settings.json value can't leak.
AI_GATEWAY_TOKEN_HEADER = "X-Databricks-AI-Gateway-Token"
AUTHORIZATION_HEADER = "Authorization"
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
# Per-operation upstream timeouts. `read` is generous because model turns stream
# over a single response and Anthropic emits SSE pings, so inter-chunk gaps stay
# small; `connect`/`pool` fail fast when the gateway is unreachable.
_UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=600.0, pool=10.0)
# Refresh once the token has less than this many seconds of life left. Databricks
# access tokens live ~1h; a 10-min buffer leaves ample headroom for a retry.
_REFRESH_BUFFER_S = 600
# How often the background thread re-checks freshness. Cheap: it only shells out
# to the CLI when actually within the buffer, otherwise it's a bare clock compare.
_REFRESHER_POLL_S = 120
# Assumed lifetime when a token carries no decodable `exp` (defensive fallback).
_DEFAULT_TTL_S = 3600
# Opt-in transport diagnostics for intermittent streaming failures. Events only
# contain locally-generated request ids, timings, status codes, byte counts,
# and exception class names — never headers, bodies, or credentials.
_DIAGNOSTICS_ENV = "UCODE_RELAYED_PROXY_DIAGNOSTICS"
_DIAGNOSTICS_TRUE = frozenset({"1", "true", "yes", "on"})


def _diagnostics_enabled() -> bool:
    return os.environ.get(_DIAGNOSTICS_ENV, "").strip().lower() in _DIAGNOSTICS_TRUE


def _diagnostic_log(event: str, **fields: object) -> None:
    if not _diagnostics_enabled():
        return
    payload = {"event": event, **fields}
    sys.stderr.write(
        f"[ucode-relay] {json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n"
    )
    sys.stderr.flush()


def _jwt_exp(token: str) -> float | None:
    """Best-effort `exp` (epoch seconds) from a JWT access token, else None."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)  # restore base64 padding
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except (IndexError, ValueError, KeyError, binascii.Error, json.JSONDecodeError):
        return None


def _log_refresh_failure(exc: BaseException) -> None:
    """Surface (never silently swallow) a refresh failure, without leaking any
    token or header value."""
    sys.stderr.write(
        f"[ucode] Databricks token refresh failed: {exc}. If the session stalls, "
        "run `databricks auth login` for your workspace profile.\n"
    )


class _TokenCache:
    """Holds the current Databricks token and its expiry, refreshing lazily as it
    nears expiry.

    A background thread refreshes proactively so the request path rarely blocks,
    but the request path also refreshes on demand — which is what carries the
    token across events the timer can't (laptop sleep suspends the monotonic
    clock, so a fixed interval silently stops advancing). All refreshes are
    single-flighted through ``_refresh_lock`` so a burst of requests at the expiry
    boundary triggers exactly one CLI call, not a thundering herd on the shared
    token cache."""

    def __init__(
        self,
        workspace: str,
        profile: str | None,
        *,
        force_refresh_near_expiry: bool = False,
    ) -> None:
        self._workspace = workspace
        self._profile = profile
        self._force_refresh_near_expiry = force_refresh_near_expiry
        self._state_lock = threading.Lock()  # guards _token / _expiry (brief)
        self._refresh_lock = threading.Lock()  # single-flights the CLI refresh
        self._stop = threading.Event()
        self._token = ""
        self._expiry = 0.0
        # Preserve the existing non-forced relayed-auth fetch. Gateway discovery
        # opts into a forced fetch so its static client token starts with a full TTL.
        self._refresh(force=force_refresh_near_expiry)

    def _refresh(self, *, force: bool) -> None:
        """Mint a token and record its expiry."""
        token = get_databricks_token(self._workspace, self._profile, force_refresh=force)
        expiry = _jwt_exp(token) or (time.time() + _DEFAULT_TTL_S)
        with self._state_lock:
            self._token = token
            self._expiry = expiry

    def _fresh_enough(self) -> bool:
        with self._state_lock:
            return bool(self._token) and time.time() < self._expiry - _REFRESH_BUFFER_S

    def _ensure_fresh(self) -> None:
        if self._fresh_enough():
            return
        with self._refresh_lock:
            if self._fresh_enough():  # another thread refreshed while we waited
                return
            try:
                self._refresh(force=self._force_refresh_near_expiry)
            except RuntimeError as exc:
                # Keep serving the current token; a request that then 401s triggers
                # a forced refresh + retry (see _ProxyHandler._handle).
                _log_refresh_failure(exc)

    @property
    def token(self) -> str:
        self._ensure_fresh()
        with self._state_lock:
            return self._token

    def refresh(self) -> None:
        """Force a fresh mint now (used by the retry-on-401 path)."""
        with self._refresh_lock:
            self._refresh(force=True)

    def run_refresher(self) -> None:
        while not self._stop.wait(_REFRESHER_POLL_S):
            try:
                self._ensure_fresh()
            except Exception as exc:  # noqa: BLE001 - a stray error must NOT kill the thread
                # If this thread dies, nothing refreshes and the session lapses at
                # the ~1h mark until restart. Log and keep looping instead.
                _log_refresh_failure(exc)

    def stop(self) -> None:
        self._stop.set()

    def wait_until_stopped(self, timeout: float) -> bool:
        return self._stop.wait(timeout)


def _forwarded_request_headers(
    handler: BaseHTTPRequestHandler,
    token: str,
    token_header: str = AI_GATEWAY_TOKEN_HEADER,
) -> dict[str, str]:
    strip_on_forward = _HOP_BY_HOP | {token_header.lower()}
    headers = {
        key: value for key, value in handler.headers.items() if key.lower() not in strip_on_forward
    }
    headers[token_header] = f"Bearer {token}"
    return headers


class _ProxyHandler(BaseHTTPRequestHandler):
    # Set by the server factory.
    cache: _TokenCache
    client: httpx.Client
    token_header = AI_GATEWAY_TOKEN_HEADER

    def log_message(self, format: str, *args: object) -> None:
        return

    def _safe_send_error(self, code: int, message: str) -> None:
        # The client (Claude Code) may already have disconnected, in which case
        # reporting the error writes to a dead socket and raises again; swallow it.
        try:
            self.send_error(code, message)
        except OSError:
            pass

    def _transform_request(self, body: bytes | None) -> tuple[str, bytes | None]:
        return self.path.lstrip("/"), body

    def _response_chunks(self, resp: httpx.Response) -> tuple[Iterable[bytes], frozenset[str]]:
        return resp.iter_raw(), frozenset()

    def _handle_cached_response(self, diagnostic_id: str, started: float) -> bool:
        return False

    def _handle(self) -> None:
        diagnostic_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else None
        url, body = self._transform_request(body)
        _diagnostic_log(
            "request_start",
            request_id=diagnostic_id,
            method=self.command,
            path=self.path.split("?", 1)[0],
        )
        if self._handle_cached_response(diagnostic_id, started):
            return
        try:
            # First attempt with the current token.
            headers = _forwarded_request_headers(self, self.cache.token, self.token_header)
            with self.client.stream(self.command, url, headers=headers, content=body) as resp:
                _diagnostic_log(
                    "upstream_headers",
                    request_id=diagnostic_id,
                    attempt=1,
                    status=resp.status_code,
                    elapsed_ms=round((time.monotonic() - started) * 1000),
                )
                if resp.status_code not in (401, 403):
                    self._relay_response(resp, diagnostic_id=diagnostic_id, started=started)
                    return
                # Auth rejected. Drain the (small) error body so the pooled
                # connection can be reused, then fall through to one retry.
                resp.read()
            # A relayed 401/403 may be a stale Databricks swap token rather than a
            # bad Anthropic OAuth — the two are indistinguishable from the status
            # alone. Force-refresh the Databricks token and retry once. If it was the
            # Anthropic layer, the retry still 401s and we relay it verbatim, so a
            # genuine re-auth is triggered; a stale-Databricks 401 self-heals here
            # instead of surfacing to Claude Code as a spurious Anthropic prompt.
            try:
                self.cache.refresh()
            except RuntimeError as exc:
                # Refresh failed: the Databricks OAuth session is dead (not just the
                # access token) and can't be re-minted non-interactively. Surface the
                # `databricks auth login` hint rather than silently relaying a bare 401,
                # which otherwise reads as an Anthropic `/login` prompt and sends the
                # user to the wrong re-auth. Still retry + relay with the existing token.
                _log_refresh_failure(exc)
            headers = _forwarded_request_headers(self, self.cache.token, self.token_header)
            with self.client.stream(self.command, url, headers=headers, content=body) as resp:
                _diagnostic_log(
                    "upstream_headers",
                    request_id=diagnostic_id,
                    attempt=2,
                    status=resp.status_code,
                    elapsed_ms=round((time.monotonic() - started) * 1000),
                )
                self._relay_response(resp, diagnostic_id=diagnostic_id, started=started)
        except (BrokenPipeError, ConnectionResetError):
            # Client closed before/while we relayed headers — routine on cancel.
            _diagnostic_log(
                "client_disconnect",
                request_id=diagnostic_id,
                phase="request",
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
            return
        except httpx.HTTPError as exc:
            # Upstream failed before any bytes reached the client; a 502 is still
            # sendable. (An HTTP *status* like 429 is not an error here — httpx
            # only raises for transport failures — so real gateway errors are
            # relayed verbatim by `_relay_response`.)
            _diagnostic_log(
                "upstream_request_error",
                request_id=diagnostic_id,
                error_type=type(exc).__name__,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
            self._safe_send_error(502, "gateway proxy upstream error")

    # Streaming passthrough: forward chunks as they arrive so SSE token streaming
    # is not buffered (buffering would add full-response latency to first token).
    # `iter_raw` preserves any Content-Encoding verbatim (we relay that header),
    # so the proxy stays byte-transparent.
    def _relay_response(
        self,
        resp: httpx.Response,
        *,
        diagnostic_id: str | None = None,
        started: float | None = None,
    ) -> None:
        started = started if started is not None else time.monotonic()
        chunks = 0
        bytes_relayed = 0
        first_byte_ms: int | None = None
        try:
            # The upstream request has completed through response headers before
            # this hook selects raw streaming or a buffered response body.
            response_chunks, dropped_headers = self._response_chunks(resp)
            self.send_response(resp.status_code)
            for key, value in resp.headers.items():
                header_name = key.lower()
                if header_name not in _HOP_BY_HOP and header_name not in dropped_headers:
                    self.send_header(key, value)
            self.end_headers()
            # Do not pass a fixed chunk size here. httpx accumulates bytes until
            # that size is reached, which can hide small SSE heartbeat frames
            # from Claude Code for minutes during a slow artifact/tool call.
            # With ``chunk_size=None`` (the default), raw upstream chunks are
            # yielded as they arrive and pings keep the downstream connection
            # alive even before the model produces a large content block.
            for chunk in response_chunks:
                if chunk:
                    if first_byte_ms is None:
                        first_byte_ms = round((time.monotonic() - started) * 1000)
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    chunks += 1
                    bytes_relayed += len(chunk)
            _diagnostic_log(
                "response_complete",
                request_id=diagnostic_id,
                status=resp.status_code,
                chunks=chunks,
                bytes=bytes_relayed,
                first_byte_ms=first_byte_ms,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
        except (BrokenPipeError, ConnectionResetError):
            # Client (Claude Code) closed the connection mid-response — routine on
            # cancelled turns / SSE teardown. Nothing left to relay to, so stop
            # quietly rather than crashing the handler thread.
            _diagnostic_log(
                "client_disconnect",
                request_id=diagnostic_id,
                phase="response",
                chunks=chunks,
                bytes=bytes_relayed,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
            return
        except httpx.HTTPError as exc:
            # Upstream dropped mid-stream. Headers (and status) may already be
            # sent, so we can't reliably signal a fresh error — stop and let the
            # client see a truncated stream rather than corrupt the framing.
            _diagnostic_log(
                "upstream_stream_error",
                request_id=diagnostic_id,
                error_type=type(exc).__name__,
                status=resp.status_code,
                chunks=chunks,
                bytes=bytes_relayed,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
            return

    # Forward every method: this is a transparent pass-through, so routing any
    # `do_<METHOD>` lookup to `_handle` lets the gateway reject unsupported methods.
    def __getattr__(self, name: str):
        if name.startswith("do_"):
            return self._handle
        raise AttributeError(name)


_MODEL_ALIAS_PREFIX = "anthropic-aigw-"
_ANTHROPIC_MODELS_PATH = "/v1/models"
_ANTHROPIC_MESSAGES_PATH = "/v1/messages"
_MODEL_DISCOVERY_LIMIT = 1000
_MODEL_CACHE_REFRESH_S = 600


class _AnthropicModelAliases:
    """Maps Claude-compatible discovery IDs back to their gateway model IDs."""

    def __init__(self) -> None:
        self._original_by_alias: dict[str, str] = {}
        self._lock = threading.Lock()

    def prefix_model_ids(self, body: bytes) -> bytes:
        try:
            payload = json.loads(body)
            models = payload["data"]
            if not isinstance(models, list):
                return body
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
            return body

        aliases: dict[str, str] = {}
        for model in models:
            if not isinstance(model, dict) or not isinstance(model.get("id"), str):
                continue
            model_id = model["id"]
            lowered = model_id.lower()
            if "claude" in lowered or "anthropic" in lowered:
                continue
            alias = f"{_MODEL_ALIAS_PREFIX}{model_id}"
            model["id"] = alias
            aliases[alias] = model_id

        with self._lock:
            self._original_by_alias.update(aliases)

        for cursor in ("first_id", "last_id"):
            model_id = payload.get(cursor)
            alias = f"{_MODEL_ALIAS_PREFIX}{model_id}"
            if alias in aliases:
                payload[cursor] = alias
        return json.dumps(payload, separators=(",", ":")).encode()

    def original_id(self, model_id: str) -> str:
        with self._lock:
            return self._original_by_alias.get(model_id, model_id)

    def rewrite_path(self, path: str) -> str:
        parsed = urlsplit(path)
        if parsed.path != _ANTHROPIC_MODELS_PATH:
            return path
        query = [
            (key, self.original_id(value) if key == "after_id" else value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        ]
        return urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
        )

    def rewrite_body(self, path: str, body: bytes | None) -> bytes | None:
        if urlsplit(path).path != _ANTHROPIC_MESSAGES_PATH or body is None:
            return body
        try:
            payload = json.loads(body)
            model_id = payload.get("model")
            if not isinstance(model_id, str):
                return body
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            return body
        original_id = self.original_id(model_id)
        if original_id == model_id:
            return body
        payload["model"] = original_id
        return json.dumps(payload, separators=(",", ":")).encode()


class _ModelCache:
    """Caches the complete model list so Claude's discovery request is local."""

    def __init__(self, aliases: _AnthropicModelAliases) -> None:
        self._aliases = aliases
        self._body: bytes | None = None
        self._lock = threading.Lock()

    def refresh(self, client: httpx.Client, token: str, token_header: str) -> None:
        headers = {
            token_header: f"Bearer {token}",
            "anthropic-version": "2023-06-01",
        }
        models: list[object] = []
        first_page: dict[str, object] | None = None
        last_page: dict[str, object] | None = None
        after_id: str | None = None
        seen_cursors: set[str] = set()

        while True:
            params: dict[str, str | int] = {"limit": _MODEL_DISCOVERY_LIMIT}
            if after_id is not None:
                params["after_id"] = after_id
            response = client.get("v1/models", headers=headers, params=params)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                raise ValueError("invalid model discovery response")
            if first_page is None:
                first_page = payload
            last_page = payload
            models.extend(payload["data"])
            if not payload.get("has_more"):
                break
            after_id = payload.get("last_id")
            if not isinstance(after_id, str) or after_id in seen_cursors:
                raise ValueError("invalid model discovery cursor")
            seen_cursors.add(after_id)

        combined = dict(first_page or {})
        combined["data"] = models
        combined["has_more"] = False
        if last_page is not None:
            combined["last_id"] = last_page.get("last_id")
        body = self._aliases.prefix_model_ids(json.dumps(combined, separators=(",", ":")).encode())
        with self._lock:
            self._body = body

    def get(self, method: str, path: str) -> bytes | None:
        parsed = urlsplit(path)
        if method != "GET" or parsed.path != _ANTHROPIC_MODELS_PATH:
            return None
        if any(
            key in {"after_id", "before_id"}
            for key, _value in parse_qsl(parsed.query, keep_blank_values=True)
        ):
            return None
        with self._lock:
            return self._body

    def run_refresher(
        self,
        client: httpx.Client,
        token_cache: _TokenCache,
        token_header: str,
    ) -> None:
        while not token_cache.wait_until_stopped(_MODEL_CACHE_REFRESH_S):
            try:
                self.refresh(client, token_cache.token, token_header)
            except Exception:  # noqa: BLE001 - refresh failure must not kill the thread
                continue


class _AnthropicGatewayHandler(_ProxyHandler):
    anthropic_model_aliases: _AnthropicModelAliases
    model_cache: _ModelCache

    def _transform_request(self, body: bytes | None) -> tuple[str, bytes | None]:
        body = self.anthropic_model_aliases.rewrite_body(self.path, body)
        url = self.anthropic_model_aliases.rewrite_path(self.path).lstrip("/")
        return url, body

    def _response_chunks(self, resp: httpx.Response) -> tuple[Iterable[bytes], frozenset[str]]:
        should_prefix_model_ids = (
            self.command == "GET"
            and urlsplit(self.path).path == _ANTHROPIC_MODELS_PATH
            and HTTPStatus.OK <= resp.status_code < HTTPStatus.MULTIPLE_CHOICES
        )
        if not should_prefix_model_ids:
            return super()._response_chunks(resp)
        body = self.anthropic_model_aliases.prefix_model_ids(resp.read())
        # resp.read() decodes compression; rewritten JSON is uncompressed.
        return (body,), frozenset({"content-encoding"})

    def _handle_cached_response(self, diagnostic_id: str, started: float) -> bool:
        cached_models = self.model_cache.get(self.command, self.path)
        if cached_models is None:
            return False
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(cached_models)))
            self.end_headers()
            self.wfile.write(cached_models)
            self.wfile.flush()
            _diagnostic_log(
                "model_cache_hit",
                request_id=diagnostic_id,
                bytes=len(cached_models),
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
        except (BrokenPipeError, ConnectionResetError):
            pass
        return True


def start_proxy(
    workspace: str,
    profile: str | None,
    port: int,
    token_header: str,
    force_refresh_near_expiry: bool,
    prefetch_models: bool = False,
) -> tuple[ThreadingHTTPServer, _TokenCache, httpx.Client]:
    """Start the Anthropic loopback proxy and token refresher."""
    aliases = _AnthropicModelAliases()
    model_cache = _ModelCache(aliases)
    upstream_base = f"{workspace.rstrip('/')}/ai-gateway/anthropic/"
    token_cache = _TokenCache(
        workspace,
        profile,
        force_refresh_near_expiry=force_refresh_near_expiry,
    )
    client = httpx.Client(base_url=upstream_base, timeout=_UPSTREAM_TIMEOUT, follow_redirects=False)
    handler = cast(
        type[BaseHTTPRequestHandler],
        type(
            "BoundProxyHandler",
            (_AnthropicGatewayHandler,),
            {
                "cache": token_cache,
                "client": client,
                "token_header": token_header,
                "anthropic_model_aliases": aliases,
                "model_cache": model_cache,
            },
        ),
    )
    try:
        server = ThreadingHTTPServer((LOOPBACK_HOST, port), handler)
    except OSError:
        server = ThreadingHTTPServer((LOOPBACK_HOST, 0), handler)

    refresher = threading.Thread(target=token_cache.run_refresher, daemon=True)
    refresher.start()
    if prefetch_models:
        try:
            model_cache.refresh(client, token_cache.token, token_header)
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            sys.stderr.write(
                "[ucode] Claude model prefetch failed "
                f"({type(exc).__name__}); falling back to live discovery.\n"
            )
        model_refresher = threading.Thread(
            target=model_cache.run_refresher,
            args=(client, token_cache, token_header),
            daemon=True,
        )
        model_refresher.start()
    return server, token_cache, client
