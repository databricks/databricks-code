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

    def test_dead_auth_becomes_a_terminal_proxy_auth_error(self, monkeypatch):
        # A raw RuntimeError escaping auth_flow tears through httpx's transport
        # task group and stalls the proxy until the client's startup timeout.
        # Translating it keeps the failure reportable by `serve`.
        def boom(ws, profile):
            raise RuntimeError("no access token; run `databricks auth login`")

        monkeypatch.setattr(mcp_proxy, "get_databricks_token", boom)
        auth = mcp_proxy._DatabricksTokenAuth(WS, "p", use_pat=False)

        with pytest.raises(mcp_proxy.ProxyAuthError, match="databricks auth login"):
            list(auth.auth_flow(httpx.Request("POST", URL)))


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

        monkeypatch.setattr(mcp_proxy, "_preflight_token", lambda ws, profile: None)
        monkeypatch.setattr(mcp_proxy.anyio, "run", fake_run)

        mcp_proxy.serve(URL, WS, "uc-dogfood", use_pat=True)

        assert captured["func"] is mcp_proxy._run
        assert captured["args"] == (URL, WS, "uc-dogfood", True)

    def test_defaults_profile_none_and_use_pat_false(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(mcp_proxy, "_preflight_token", lambda ws, profile: None)
        monkeypatch.setattr(mcp_proxy.anyio, "run", lambda func, *args: captured.update(args=args))

        mcp_proxy.serve(URL, WS)

        assert captured["args"] == (URL, WS, None, False)

    def test_preflights_auth_before_opening_the_bridge(self, monkeypatch):
        # Order matters: a dead profile must be caught before the stdio bridge
        # starts, so the failure is a fast exit rather than a stalled session.
        order: list[str] = []
        monkeypatch.setattr(
            mcp_proxy, "_preflight_token", lambda ws, profile: order.append("preflight")
        )
        monkeypatch.setattr(mcp_proxy.anyio, "run", lambda func, *args: order.append("bridge"))

        mcp_proxy.serve(URL, WS, "p")

        assert order == ["preflight", "bridge"]

    def test_dead_auth_exits_fast_without_starting_the_bridge(self, monkeypatch, capsys):
        # The regression this fix targets: previously the token failure surfaced
        # from inside the transport and the proxy hung until the MCP client's
        # startup timeout (~30s) with no explanation.
        started: list[str] = []

        def dead_auth(ws, profile):
            raise RuntimeError("no access token for " + ws + "; run `databricks auth login`")

        monkeypatch.setattr(mcp_proxy, "_preflight_token", dead_auth)
        monkeypatch.setattr(mcp_proxy.anyio, "run", lambda func, *args: started.append("bridge"))

        with pytest.raises(SystemExit) as excinfo:
            mcp_proxy.serve(URL, WS, "p")

        assert excinfo.value.code == mcp_proxy.AUTH_FAILURE_EXIT_CODE
        assert started == []  # the bridge never opened
        # Diagnostics go to stderr; stdout is the MCP wire and must stay clean.
        captured = capsys.readouterr()
        assert "databricks auth login" in captured.err
        assert captured.out == ""

    def test_auth_expiring_mid_session_exits_with_the_actionable_message(self, monkeypatch, capsys):
        # A ProxyAuthError raised once the bridge is running arrives wrapped in an
        # anyio ExceptionGroup; it must still be reported, not surface as a crash.
        def raise_group(func, *args):
            raise BaseExceptionGroup(
                "transport",
                [mcp_proxy.ProxyAuthError("token expired; run `databricks auth login`")],
            )

        monkeypatch.setattr(mcp_proxy, "_preflight_token", lambda ws, profile: None)
        monkeypatch.setattr(mcp_proxy.anyio, "run", raise_group)

        with pytest.raises(SystemExit) as excinfo:
            mcp_proxy.serve(URL, WS, "p")

        assert excinfo.value.code == mcp_proxy.AUTH_FAILURE_EXIT_CODE
        assert "token expired" in capsys.readouterr().err

    def test_non_auth_failures_still_propagate(self, monkeypatch):
        # Only auth failures are converted to a clean exit; genuine transport
        # bugs must keep their traceback so they stay debuggable.
        def raise_other(func, *args):
            raise ValueError("some transport bug")

        monkeypatch.setattr(mcp_proxy, "_preflight_token", lambda ws, profile: None)
        monkeypatch.setattr(mcp_proxy.anyio, "run", raise_other)

        with pytest.raises(ValueError, match="some transport bug"):
            mcp_proxy.serve(URL, WS, "p")


class TestPreflightToken:
    def test_passes_through_when_a_token_is_available(self, monkeypatch):
        monkeypatch.setattr(mcp_proxy, "get_databricks_token", lambda ws, profile: "tok")
        mcp_proxy._preflight_token(WS, "p")  # no exception

    def test_surfaces_the_cli_error_message(self, monkeypatch):
        def boom(ws, profile):
            raise RuntimeError("profile is stale; run `databricks auth logout`")

        monkeypatch.setattr(mcp_proxy, "get_databricks_token", boom)

        with pytest.raises(RuntimeError, match="databricks auth logout"):
            mcp_proxy._preflight_token(WS, "p")

    def test_checks_the_same_workspace_and_profile_the_bridge_will_use(self, monkeypatch):
        # The preflight must validate the exact credentials the request-time auth
        # hook uses, or it could pass while the bridge still fails.
        calls: list[tuple[str, str | None]] = []
        monkeypatch.setattr(
            mcp_proxy,
            "get_databricks_token",
            lambda ws, profile: calls.append((ws, profile)) or "tok",
        )

        mcp_proxy._preflight_token(WS, "myprofile")

        assert calls == [(WS, "myprofile")]
