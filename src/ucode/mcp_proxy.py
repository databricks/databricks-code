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
"""

from __future__ import annotations

import anyio
import httpx
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.stdio import stdio_server

from ucode.databricks import get_databricks_token


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
        token = get_databricks_token(self._workspace, self._profile)
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


def serve(url: str, workspace: str, profile: str | None = None, *, use_pat: bool = False) -> None:
    """Run the stdio<->streamable-HTTP MCP proxy until the client closes stdin."""
    anyio.run(_run, url, workspace, profile, use_pat)


__all__ = ["serve"]
