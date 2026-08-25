"""WebSocket interposer for the Codex TUI's ``--remote`` transport (smart routing v2).

Codex's remote transport (``codex --remote ws://…``) is WebSocket: a plain-JSONL
client is rejected with HTTP 400 ("Connection header did not include 'upgrade'"),
a proper upgrade returns 101, and each JSON-RPC message is one WebSocket text
frame. This module sits between the real TUI and a real ``codex app-server``,
forwarding every frame untouched except:

  - ``turn/start`` (TUI->engine): after an initial hold of ``after`` turns, its
    ``model`` is rewritten. ``turn/start.model`` is documented as "override the
    model for this turn and subsequent turns", so the live session retargets with
    history preserved.
  - When the first switched turn starts — on that turn's ``turn/started``, before
    any response items stream — two things are injected (engine->TUI): a
    ``thread/settings/updated`` carrying the new model, so the TUI's on-screen
    model indicator follows the switch, and — when a ``switch_message`` is
    configured — an ``agentMessage`` item (as an ``item/started`` + ``item/completed``
    pair) that surfaces an explanation of why the model was switched, ahead
    of the model's reply. An ``agentMessage`` renders as ordinary chat text (no
    warning styling); Codex's protocol has no neutral free-text notification
    (``warning``, ``configWarning``, ``deprecationNotice`` all render as warnings),
    so an item is the way to show an informational note. The ``item/started`` is
    required: the TUI creates the message widget on ``item/started``, so a lone
    ``item/completed`` has no widget to finalize and renders nothing.

``ucode.smart_routing.v2`` runs :func:`start_interposer_thread` in a daemon
thread while it owns the app-server subprocess and the ``codex --remote`` TUI,
so the whole thing launches from the single ``ucode codex`` command.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import threading
import time
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

SETTINGS_UPDATED = "thread/settings/updated"
ITEM_STARTED = "item/started"
ITEM_COMPLETED = "item/completed"
LOOPBACK_HOST = "127.0.0.1"

# Set UCODE_INTERPOSER_DEBUG=1 to dump raw engine->TUI item/turn frames (and the frames we
# inject) to the interposer log, for comparing our synthetic note against real assistant frames.
_DEBUG = os.environ.get("UCODE_INTERPOSER_DEBUG") == "1"


class _Session:
    """Per-TUI-connection state: hold the first ``after`` turns, then switch model."""

    def __init__(
        self,
        target_model: str,
        after: int,
        log: Callable[[str], None],
        switch_message: str | None = None,
    ) -> None:
        self.target = target_model
        self.after = after
        self.log = log
        # One-line explanation surfaced in the TUI when the switch fires; None skips it.
        self.switch_message = switch_message
        self.turns = 0
        self.thread_id: str | None = None
        self.settings: dict | None = None
        self.injected = False

    def on_tui_frame(self, raw: str) -> str:
        """TUI->engine: rewrite ``turn/start.model`` once past the hold."""
        try:
            msg = json.loads(raw)
        except ValueError:
            return raw
        if not isinstance(msg, dict):
            return raw
        params = msg.get("params")
        if _DEBUG and isinstance(msg.get("method"), str) and msg["method"].startswith("turn/"):
            self.log(f"[DEBUG->engine] {msg['method']}: {raw[:1200]}")
        if msg.get("method") == "turn/start" and isinstance(params, dict):
            self.turns += 1
            if isinstance(params.get("threadId"), str):
                self.thread_id = params["threadId"]
            if self.turns > self.after:
                old = params.get("model")
                if old != self.target:
                    params["model"] = self.target
                    self.log(f"[REWRITE] turn #{self.turns}: model {old!r} -> {self.target!r}")
                    return json.dumps(msg)
        return raw

    def on_engine_frame(self, raw: str) -> list[dict]:
        """engine->TUI: capture thread id/settings; when the first switched turn
        starts, return the frames to inject (empty list = none).

        On the switched turn's ``turn/started`` — before its response streams —
        this yields a ``thread/settings/updated`` (flips the TUI's model chip)
        and, when ``switch_message`` is set, an ``item/started`` +
        ``item/completed`` pair carrying an ``agentMessage`` — plain chat text
        (no warning styling) that explains why the model changed, shown ahead of
        the model's reply."""
        try:
            msg = json.loads(raw)
        except ValueError:
            return []
        if not isinstance(msg, dict):
            return []
        if _DEBUG:
            method = msg.get("method")
            if isinstance(method, str) and (
                method.startswith("item/") or method.startswith("turn/")
            ):
                self.log(f"[DEBUG<-engine] {method}: {raw[:1800]}")
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
        result = msg.get("result") if isinstance(msg.get("result"), dict) else {}
        for src in (params, result):
            tid = src.get("threadId") or (src.get("thread") or {}).get("id")
            if isinstance(tid, str):
                self.thread_id = tid
            ts = src.get("threadSettings")
            if isinstance(ts, dict):
                self.settings = ts
        if (
            msg.get("method") == "turn/started"
            and not self.injected
            and self.turns > self.after
            and self.thread_id
        ):
            self.injected = True
            settings = dict(self.settings) if isinstance(self.settings, dict) else {}
            settings["model"] = self.target
            self.log(f"[INJECT] {SETTINGS_UPDATED}: model -> {self.target!r} (flip TUI chip)")
            injected: list[dict] = [
                {
                    "method": SETTINGS_UPDATED,
                    "params": {"threadId": self.thread_id, "threadSettings": settings},
                }
            ]
            if self.switch_message:
                params_obj = msg.get("params") if isinstance(msg.get("params"), dict) else {}
                turn = params_obj.get("turn") if isinstance(params_obj, dict) else {}
                turn_id = turn.get("id") if isinstance(turn, dict) else None
                now_ms = int(time.time() * 1000)
                item = {
                    "type": "agentMessage",
                    "id": f"ucode-smart-router-{uuid.uuid4().hex}",
                    "text": self.switch_message,
                    "phase": None,
                    "memoryCitation": None,
                }
                self.log(f"[INJECT] agentMessage note (started+completed): {self.switch_message!r}")
                # The TUI creates the message widget on item/started; a lone item/completed
                # has no widget to finalize and renders nothing. Send the full lifecycle with
                # the text already populated (no deltas needed for a static note).
                injected.append(
                    {
                        "method": ITEM_STARTED,
                        "params": {
                            "item": item,
                            "threadId": self.thread_id,
                            "turnId": turn_id,
                            "startedAtMs": now_ms,
                        },
                    }
                )
                injected.append(
                    {
                        "method": ITEM_COMPLETED,
                        "params": {
                            "item": item,
                            "threadId": self.thread_id,
                            "turnId": turn_id,
                            "completedAtMs": now_ms,
                        },
                    }
                )
            if _DEBUG:
                for frame in injected:
                    self.log(f"[DEBUG-inject] {json.dumps(frame)[:1800]}")
            return injected
        return []


async def _handle_tui(
    tui, upstream_uri: str, target_model: str, after: int, log, switch_message: str | None = None
) -> None:
    path = getattr(getattr(tui, "request", None), "path", "/") or "/"
    uri = upstream_uri.rstrip("/") + path
    log(f"[CONN] TUI connected (path={path}); dialing app-server {uri}")
    sess = _Session(target_model, after, log, switch_message)
    async with connect(uri, max_size=None) as upstream:

        async def tui_to_app():
            async for frame in tui:
                if isinstance(frame, str):
                    frame = sess.on_tui_frame(frame)
                await upstream.send(frame)

        async def app_to_tui():
            async for frame in upstream:
                await tui.send(frame)
                if isinstance(frame, str):
                    for inj in sess.on_engine_frame(frame):
                        await tui.send(json.dumps(inj))

        a = asyncio.create_task(tui_to_app())
        b = asyncio.create_task(app_to_tui())
        _done, pending = await asyncio.wait({a, b}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
    log("[CONN] TUI session closed")


async def _serve(
    host: str,
    port: int,
    upstream_uri: str,
    model: str,
    after: int,
    log,
    switch_message: str | None = None,
):
    async def handler(tui):
        try:
            await _handle_tui(tui, upstream_uri, model, after, log, switch_message)
        except Exception as exc:  # noqa: BLE001 - one session must never kill the server
            log(f"[ERR] session: {exc!r}")

    server = await serve(handler, host, port, max_size=None)
    log(f"[READY] ws://{host}:{port} -> {upstream_uri} (hold {after} turn(s), then -> {model!r})")
    return server


def start_interposer_thread(
    host: str,
    port: int,
    upstream_uri: str,
    model: str,
    after: int,
    *,
    switch_message: str | None = None,
    log_path: Path | None = None,
    ready_timeout: float = 10.0,
) -> tuple[threading.Thread, Callable[[], None]]:
    """Run the interposer's asyncio server in a daemon thread.

    Returns ``(thread, stop)``; ``stop()`` shuts the server down and stops the
    loop. ``switch_message``, when set, is surfaced in the TUI as an
    ``agentMessage`` explaining why the model switched. Logs go to ``log_path`` (appended) when
    given — never to stdout/stderr, which the foreground TUI owns. Blocks until
    the server is listening (or ``ready_timeout`` elapses)."""

    def log(message: str) -> None:
        if log_path is None:
            return
        line = f"{time.strftime('%H:%M:%S')} {message}\n"
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            pass

    loop = asyncio.new_event_loop()
    holder: dict = {}
    ready = threading.Event()

    def run() -> None:
        asyncio.set_event_loop(loop)
        try:
            holder["server"] = loop.run_until_complete(
                _serve(host, port, upstream_uri, model, after, log, switch_message)
            )
        except Exception as exc:  # noqa: BLE001 - surface bind/connect failures to the log
            log(f"[ERR] failed to start interposer: {exc!r}")
            ready.set()
            loop.close()
            return
        ready.set()
        loop.run_forever()
        # Stopped: close the server and drain.
        server = holder.get("server")
        if server is not None:
            server.close()
            with contextlib.suppress(Exception):
                loop.run_until_complete(server.wait_closed())
        loop.close()

    thread = threading.Thread(target=run, name="codex-interposer", daemon=True)
    thread.start()
    ready.wait(timeout=ready_timeout)

    def stop() -> None:
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(loop.stop)

    return thread, stop


def free_port() -> int:
    """Grab an unused loopback TCP port (races are irrelevant for local ephemeral use)."""
    sock = socket.socket()
    sock.bind((LOOPBACK_HOST, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def wait_healthz(port: int, timeout: float = 30.0) -> bool:
    """Poll the app-server's ``/healthz`` until it returns 200, or timeout."""
    url = f"http://{LOOPBACK_HOST}:{port}/healthz"
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:  # noqa: S310 - fixed localhost URL
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - not ready yet; keep polling
            time.sleep(0.25)
    return False
