"""Loopback proxy for Claude gateway model discovery.

The proxy refreshes the Databricks credential, streams inference responses
verbatim, and rewrites model discovery responses when needed.

Security invariants (mirroring `databricks.py` token handling):
  - Binds 127.0.0.1 only; never exposed off-host.
  - Never logs header values or bodies. The Databricks token lives in memory
    and is refreshed off the request path.
"""

from __future__ import annotations

import json
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
from ucode.gateway_proxy import (
    AI_GATEWAY_TOKEN_HEADER,
    HOP_BY_HOP_HEADERS,
    UPSTREAM_TIMEOUT,
    TokenCache,
    forwarded_request_headers,
    log_proxy_diagnostic,
    log_token_refresh_failure,
)


class _ProxyHandler(BaseHTTPRequestHandler):
    # Set by the server factory.
    cache: TokenCache | None
    client: httpx.Client
    token_header: str | None = AI_GATEWAY_TOKEN_HEADER

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

    def _handle(self) -> None:
        diagnostic_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else None
        url, body = self._transform_request(body)
        log_proxy_diagnostic(
            "model_discovery_request_start",
            request_id=diagnostic_id,
            method=self.command,
            path=self.path.split("?", 1)[0],
        )
        try:
            # Native gateway discovery uses Claude Code's apiKeyHelper credential
            # verbatim. Relayed mode still injects its separately refreshed swap
            # token while preserving the caller's Anthropic subscription OAuth.
            if self.cache is None or self.token_header is None:
                headers = {
                    key: value
                    for key, value in self.headers.items()
                    if key.lower() not in HOP_BY_HOP_HEADERS
                }
            else:
                headers = forwarded_request_headers(self, self.cache.token, self.token_header)
            with self.client.stream(self.command, url, headers=headers, content=body) as resp:
                log_proxy_diagnostic(
                    "model_discovery_upstream_headers",
                    request_id=diagnostic_id,
                    attempt=1,
                    status=resp.status_code,
                    elapsed_ms=round((time.monotonic() - started) * 1000),
                )
                if resp.status_code not in (401, 403) or self.cache is None:
                    self._relay_response(resp, diagnostic_id=diagnostic_id, started=started)
                    return
                # Auth rejected. Drain the (small) error body so the pooled
                # connection can be reused, then fall through to one retry.
                resp.read()
            # Force-refresh the Databricks token and retry once.
            try:
                self.cache.refresh()
            except RuntimeError as exc:
                # Still retry with the existing token after reporting the failure.
                log_token_refresh_failure(exc)
            assert self.token_header is not None
            headers = forwarded_request_headers(self, self.cache.token, self.token_header)
            with self.client.stream(self.command, url, headers=headers, content=body) as resp:
                log_proxy_diagnostic(
                    "model_discovery_upstream_headers",
                    request_id=diagnostic_id,
                    attempt=2,
                    status=resp.status_code,
                    elapsed_ms=round((time.monotonic() - started) * 1000),
                )
                self._relay_response(resp, diagnostic_id=diagnostic_id, started=started)
        except (BrokenPipeError, ConnectionResetError):
            # Client closed before/while we relayed headers — routine on cancel.
            log_proxy_diagnostic(
                "model_discovery_client_disconnect",
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
            log_proxy_diagnostic(
                "model_discovery_upstream_request_error",
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
                if header_name not in HOP_BY_HOP_HEADERS and header_name not in dropped_headers:
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
            log_proxy_diagnostic(
                "model_discovery_response_complete",
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
            log_proxy_diagnostic(
                "model_discovery_client_disconnect",
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
            log_proxy_diagnostic(
                "model_discovery_upstream_stream_error",
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


class _AnthropicModelDiscoveryHandler(_ProxyHandler):
    anthropic_model_aliases: _AnthropicModelAliases

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


def start_proxy(
    workspace: str,
    profile: str | None,
    port: int,
    token_header: str | None,
    force_refresh_near_expiry: bool,
) -> tuple[ThreadingHTTPServer, TokenCache | None, httpx.Client]:
    """Start the Anthropic model discovery proxy and optional token refresher."""
    upstream_base = f"{workspace.rstrip('/')}/ai-gateway/anthropic/"
    cache = (
        TokenCache(
            workspace,
            profile,
            force_refresh_near_expiry=force_refresh_near_expiry,
        )
        if token_header is not None
        else None
    )
    client = httpx.Client(base_url=upstream_base, timeout=UPSTREAM_TIMEOUT, follow_redirects=False)
    handler = cast(
        type[BaseHTTPRequestHandler],
        type(
            "BoundProxyHandler",
            (_AnthropicModelDiscoveryHandler,),
            {
                "cache": cache,
                "client": client,
                "token_header": token_header,
                "anthropic_model_aliases": _AnthropicModelAliases(),
            },
        ),
    )
    try:
        server = ThreadingHTTPServer((LOOPBACK_HOST, port), handler)
    except OSError:
        server = ThreadingHTTPServer((LOOPBACK_HOST, 0), handler)

    if cache is not None:
        refresher = threading.Thread(target=cache.run_refresher, daemon=True)
        refresher.start()
    return server, cache, client
