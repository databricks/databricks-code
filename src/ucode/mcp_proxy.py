"""`ucode mcp-proxy`: a stdio MCP server that bridges to a Databricks
streamable-HTTP MCP endpoint, injecting a freshly-minted OAuth bearer.

Every coding agent ucode configures points its Databricks MCP servers at this
one command (``ucode mcp-proxy --url <endpoint> --profile <profile>``) as a
local **stdio** server. The agent spawns and reaps the proxy as a child process
— ucode owns no long-lived process and no background refresh thread. The proxy
speaks stdio to the agent and streamable-HTTP to Databricks, and mints a fresh
token from the Databricks CLI profile on **every** upstream HTTP request via an
``httpx.Auth`` hook, so the bearer never goes stale mid-session.

This replaces the previous per-client header auth (static ``Bearer
${OAUTH_TOKEN}``, Claude ``headersHelper``, Cursor literal-token rewrites): one
uniform mechanism, token refresh in a single place, and the proxy is an
invisible implementation detail baked into each client's config.

Auth failures are terminal and are reported *fast*. When the Databricks CLI
can't mint a token (expired refresh token, logged-out profile), the proxy prints
the CLI's own message to stderr and exits ``AUTH_FAILURE_EXIT_CODE`` rather than
letting the client wait out its MCP startup timeout. Every server registered
against the same profile fails at once in that state, so a silent hang is
especially confusing -- the user needs to be told to re-run
``databricks auth login``.
"""

from __future__ import annotations

import sys

import anyio
import httpx
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

# NOTE: `mcp` is capped `<2` in pyproject because 2.x renamed
# `streamablehttp_client` to `streamable_http_client` (and swapped httpx for
# httpx2), so these imports fail against it. That ImportError is caught and
# translated at the `ucode mcp-proxy` cli import site (see cli.py), because this
# module is exactly what can't import and so can't host the handler itself.
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.stdio import stdio_server

from ucode.databricks import get_databricks_token

# Exit code used when the proxy cannot authenticate. MCP clients surface a
# non-zero exit far more usefully than a startup timeout, so bail out with this
# instead of letting the process hang until the client's timeout fires.
AUTH_FAILURE_EXIT_CODE = 2


class ProxyAuthError(RuntimeError):
    """The proxy could not mint a Databricks token, so it cannot serve requests.

    Kept distinct from a transport error: this one is terminal and actionable
    (the user must re-run `databricks auth login`), so `serve` reports it on
    stderr and exits rather than retrying."""


def _fail_fast(message: str) -> None:
    """Report a terminal auth failure on stderr and exit non-zero.

    stdout is the MCP wire, so diagnostics must go to stderr — MCP clients
    surface a child's stderr when it fails to start."""
    print(f"ucode mcp-proxy: {message}", file=sys.stderr, flush=True)
    raise SystemExit(AUTH_FAILURE_EXIT_CODE)


class _DatabricksTokenAuth(httpx.Auth):
    """Injects a fresh Databricks OAuth bearer on every request.

    ``get_databricks_token`` returns a cached token and refreshes it when
    expired, so calling it per request keeps auth current without the proxy
    tracking token lifetimes itself."""

    def __init__(self, workspace: str, profile: str | None, *, use_pat: bool) -> None:
        self._workspace = workspace
        self._profile = profile
        self._use_pat = use_pat

    def auth_flow(self, request: httpx.Request):
        # get_databricks_token honors the DATABRICKS_BEARER short-circuit and PAT
        # profiles internally; --use-pat is surfaced via the env ucode already set.
        # A RuntimeError here means auth is dead (expired refresh token, logged-out
        # profile). Raising it from inside httpx's auth_flow would tear through the
        # transport's task group and stall the process until the client times out,
        # so translate it into a terminal ProxyAuthError the caller reports cleanly.
        try:
            token = get_databricks_token(self._workspace, self._profile)
        except RuntimeError as exc:
            raise ProxyAuthError(str(exc)) from exc
        request.headers["Authorization"] = f"Bearer {token}"
        yield request


async def _pump(
    source: MemoryObjectReceiveStream,
    dest: MemoryObjectSendStream,
) -> None:
    """Forward every message (or transport exception) from ``source`` to ``dest``.

    The proxy is transport-level: it never inspects or rewrites MCP method
    payloads, so new methods and capabilities pass through untouched."""
    async with source, dest:
        async for message in source:
            await dest.send(message)


async def _run(url: str, workspace: str, profile: str | None, use_pat: bool) -> None:
    auth = _DatabricksTokenAuth(workspace, profile, use_pat=use_pat)
    async with streamablehttp_client(url, auth=auth) as (http_read, http_write, _get_session_id):
        async with stdio_server() as (stdio_read, stdio_write):
            # Bidirectional bridge: client stdin -> Databricks, Databricks -> client stdout.
            async with anyio.create_task_group() as tg:
                tg.start_soon(_pump, stdio_read, http_write)
                tg.start_soon(_pump, http_read, stdio_write)


def _preflight_token(workspace: str, profile: str | None) -> None:
    """Verify a Databricks token can be minted before opening the bridge.

    Raises ``RuntimeError`` (with the CLI's own message) when auth is dead. This
    is a plain synchronous call: ``get_databricks_token`` already bounds itself
    with subprocess timeouts, so it returns or fails on its own — the point here
    is only to *locate* the failure before the transport starts, where it can be
    reported instead of stalling the session."""
    get_databricks_token(workspace, profile)


def _unwrap_auth_error(exc: BaseException) -> ProxyAuthError | None:
    """Find a ProxyAuthError anywhere in an exception (or ExceptionGroup) tree.

    anyio task groups wrap failures in ExceptionGroups, so a token failure
    raised inside the transport arrives nested rather than as itself."""
    if isinstance(exc, ProxyAuthError):
        return exc
    for nested in getattr(exc, "exceptions", ()) or ():
        found = _unwrap_auth_error(nested)
        if found is not None:
            return found
    return None


def serve(url: str, workspace: str, profile: str | None = None, *, use_pat: bool = False) -> None:
    """Run the stdio<->streamable-HTTP MCP proxy until the client closes stdin.

    Authentication is checked up front: a dead profile is a terminal condition,
    and failing here (fast, with the CLI's own message) is far better than
    letting the client wait out its MCP startup timeout with no explanation."""
    # Pre-flight the token before opening the bridge. Without this, the first
    # token failure surfaces from inside the transport's task group, where it can
    # stall the process instead of erroring out.
    try:
        _preflight_token(workspace, profile)
    except RuntimeError as exc:
        _fail_fast(str(exc))

    try:
        anyio.run(_run, url, workspace, profile, use_pat)
    except BaseException as exc:  # noqa: BLE001 - re-raised unless it's an auth failure
        # The token can still expire mid-session; report that the same way
        # rather than letting the ExceptionGroup surface as a hang or traceback.
        auth_error = _unwrap_auth_error(exc)
        if auth_error is None:
            raise
        _fail_fast(str(auth_error))


__all__ = ["AUTH_FAILURE_EXIT_CODE", "ProxyAuthError", "serve"]
