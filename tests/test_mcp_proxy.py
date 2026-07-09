"""Tests for the `ucode mcp-proxy` stdio<->streamable-HTTP bridge."""

from __future__ import annotations

import anyio
import httpx
import pytest

from ucode import mcp_proxy

WS = "https://example.databricks.com"
URL = f"{WS}/api/2.0/mcp/functions/system/ai"


class TestDatabricksTokenAuth:
    def test_injects_bearer_from_minted_token(self, monkeypatch):
        monkeypatch.setattr(mcp_proxy, "get_databricks_token", lambda ws, profile: "tok-123")
        auth = mcp_proxy._DatabricksTokenAuth(WS, "uc-dogfood", use_pat=False)

        request = httpx.Request("POST", URL)
        # auth_flow is a generator that yields the (mutated) request.
        list(auth.auth_flow(request))

        assert request.headers["Authorization"] == "Bearer tok-123"

    def test_calls_get_token_with_workspace_and_profile(self, monkeypatch):
        calls: list[tuple[str, str | None]] = []
        monkeypatch.setattr(
            mcp_proxy,
            "get_databricks_token",
            lambda ws, profile: calls.append((ws, profile)) or "t",
        )
        auth = mcp_proxy._DatabricksTokenAuth(WS, "myprofile", use_pat=False)

        list(auth.auth_flow(httpx.Request("POST", URL)))

        assert calls == [(WS, "myprofile")]

    def test_mints_a_fresh_token_per_request(self, monkeypatch):
        # Each request re-invokes get_databricks_token, so a rotated token is
        # picked up mid-session without the proxy tracking expiry itself.
        tokens = iter(["first", "second"])
        monkeypatch.setattr(mcp_proxy, "get_databricks_token", lambda ws, profile: next(tokens))
        auth = mcp_proxy._DatabricksTokenAuth(WS, None, use_pat=False)

        r1 = httpx.Request("POST", URL)
        r2 = httpx.Request("POST", URL)
        list(auth.auth_flow(r1))
        list(auth.auth_flow(r2))

        assert r1.headers["Authorization"] == "Bearer first"
        assert r2.headers["Authorization"] == "Bearer second"

    def test_auth_flow_yields_the_same_request(self, monkeypatch):
        monkeypatch.setattr(mcp_proxy, "get_databricks_token", lambda ws, profile: "t")
        auth = mcp_proxy._DatabricksTokenAuth(WS, None, use_pat=False)

        request = httpx.Request("POST", URL)
        yielded = list(auth.auth_flow(request))

        assert yielded == [request]


class TestPump:
    def test_forwards_all_messages_in_order(self):
        async def scenario() -> list[str]:
            src_send, src_recv = anyio.create_memory_object_stream(10)
            dst_send, dst_recv = anyio.create_memory_object_stream(10)
            # Preload the source, then close its send end so _pump's `async for`
            # terminates once drained.
            for msg in ["a", "b", "c"]:
                await src_send.send(msg)
            await src_send.aclose()

            await mcp_proxy._pump(src_recv, dst_send)

            received: list[str] = []
            # _pump closed dst_send on exit, so this drains then stops.
            async with dst_recv:
                async for msg in dst_recv:
                    received.append(msg)
            return received

        assert anyio.run(scenario) == ["a", "b", "c"]

    def test_closes_destination_when_source_exhausts(self):
        # A closed dest send-stream is what lets the *other* pump's reader
        # terminate, so the bridge tears down cleanly when one side hangs up.
        async def scenario() -> bool:
            src_send, src_recv = anyio.create_memory_object_stream(1)
            dst_send, dst_recv = anyio.create_memory_object_stream(1)
            await src_send.aclose()

            await mcp_proxy._pump(src_recv, dst_send)

            with pytest.raises(anyio.EndOfStream):
                dst_recv.receive_nowait()
            return True

        assert anyio.run(scenario) is True


class TestServe:
    def test_runs_the_bridge_with_parsed_args(self, monkeypatch):
        captured: dict = {}

        def fake_run(func, *args):
            captured["func"] = func
            captured["args"] = args

        monkeypatch.setattr(mcp_proxy.anyio, "run", fake_run)

        mcp_proxy.serve(URL, WS, "uc-dogfood", use_pat=True)

        assert captured["func"] is mcp_proxy._run
        assert captured["args"] == (URL, WS, "uc-dogfood", True)

    def test_defaults_profile_none_and_use_pat_false(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(mcp_proxy.anyio, "run", lambda func, *args: captured.update(args=args))

        mcp_proxy.serve(URL, WS)

        assert captured["args"] == (URL, WS, None, False)
