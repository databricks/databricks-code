from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

SETTINGS_UPDATED = "thread/settings/updated"
ITEM_STARTED = "item/started"
ITEM_COMPLETED = "item/completed"
TURN_START = "turn/start"
TURN_STARTED = "turn/started"


class _Session:
    def __init__(
        self,
        target_model: str,
        log: Callable[[str], None],
        switch_message: str | None = None,
    ) -> None:
        self.target = target_model
        self.log = log
        self.switch_message = switch_message
        self.thread_id: str | None = None
        self.settings: dict | None = None
        self.first_turn_seen = False
        self.switch_pending = False
        self.injected = False

    def on_tui_frame(self, raw: str) -> str:
        try:
            msg = json.loads(raw)
        except ValueError:
            return raw
        if not isinstance(msg, dict):
            return raw
        params = msg.get("params")
        if msg.get("method") == TURN_START and isinstance(params, dict):
            if isinstance(params.get("threadId"), str):
                self.thread_id = params["threadId"]
            # Route only the first turn; later /model selections belong to the user.
            if self.first_turn_seen:
                return raw
            self.first_turn_seen = True
            old = params.get("model")
            if old != self.target:
                params["model"] = self.target
                self.switch_pending = True
                self.log(f"[REWRITE] model {old!r} -> {self.target!r}")
                return json.dumps(msg)
        return raw

    def on_engine_frame(self, raw: str) -> list[dict]:
        try:
            msg = json.loads(raw)
        except ValueError:
            return []
        if not isinstance(msg, dict):
            return []
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
        result = msg.get("result") if isinstance(msg.get("result"), dict) else {}
        for src in (params, result):
            thread = src.get("thread")
            tid = src.get("threadId") or (thread.get("id") if isinstance(thread, dict) else None)
            if isinstance(tid, str):
                self.thread_id = tid
            ts = src.get("threadSettings")
            if isinstance(ts, dict):
                self.settings = ts
        if (
            msg.get("method") == TURN_STARTED
            and not self.injected
            and self.switch_pending
            and self.thread_id
        ):
            self.injected = True
            self.switch_pending = False
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
                turn = params.get("turn")
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
                # Codex renders the message only when it receives both lifecycle events.
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
            return injected
        return []


async def _handle_tui(
    tui, upstream_uri: str, target_model: str, log, switch_message: str | None = None
) -> None:
    path = getattr(getattr(tui, "request", None), "path", "/") or "/"
    uri = upstream_uri.rstrip("/") + path
    log(f"[CONN] TUI connected (path={path}); dialing app-server {uri}")
    sess = _Session(target_model, log, switch_message)
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
        done, pending = await asyncio.wait({a, b}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()
    log("[CONN] TUI session closed")


async def _serve(
    host: str,
    port: int,
    upstream_uri: str,
    model: str,
    log,
    switch_message: str | None = None,
):
    async def handler(tui):
        try:
            await _handle_tui(tui, upstream_uri, model, log, switch_message)
        except Exception as exc:  # noqa: BLE001
            log(f"[ERR] session: {exc!r}")

    server = await serve(handler, host, port, max_size=None)
    bound_port = server.sockets[0].getsockname()[1]
    log(f"[READY] ws://{host}:{bound_port} -> {upstream_uri} (switch -> {model!r})")
    return server


def start_interposer_thread(
    host: str,
    upstream_uri: str,
    model: str,
    *,
    switch_message: str | None = None,
    log_path: Path | None = None,
    ready_timeout: float = 10.0,
) -> tuple[int, Callable[[], None]]:
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
                _serve(host, 0, upstream_uri, model, log, switch_message)
            )
            holder["port"] = holder["server"].sockets[0].getsockname()[1]
        except Exception as exc:  # noqa: BLE001
            holder["error"] = exc
            log(f"[ERR] failed to start interposer: {exc!r}")
            ready.set()
            loop.close()
            return
        ready.set()
        loop.run_forever()
        server = holder.get("server")
        if server is not None:
            server.close()
            with contextlib.suppress(Exception):
                loop.run_until_complete(server.wait_closed())
        loop.close()

    thread = threading.Thread(target=run, name="codex-interposer", daemon=True)
    thread.start()

    if not ready.wait(timeout=ready_timeout):
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=ready_timeout)
        raise RuntimeError("Codex interposer did not become ready in time.")
    if error := holder.get("error"):
        raise RuntimeError("Codex interposer failed to start.") from error

    def stop() -> None:
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=ready_timeout)

    return holder["port"], stop
