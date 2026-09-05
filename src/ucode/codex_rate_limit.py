"""Cross-process input-token limiter for Codex requests sent through FMAPI.

Codex resends its active context on every model round-trip. Several concurrent
sessions can therefore exhaust a model's input-tokens-per-minute allowance even
when each session is new. This module reserves a conservative body-size estimate
in a rolling, per-workspace/model window shared through ``~/.ucode``.

Request bodies are inspected in memory only. The shared state contains model
keys, timestamps, and estimated token counts; prompts and credentials are never
persisted or logged.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import BinaryIO

from ucode import config_io

WINDOW_SECONDS = 60.0
BYTES_PER_ESTIMATED_TOKEN = 3
SAFETY_PERCENT = 90
STATE_FILE_NAME = "codex-rate-limit-state.json"
LOCK_FILE_NAME = "codex-rate-limit.lock"
RETRY_BACKOFF_BASE_SECONDS = 2.0
RETRY_BACKOFF_MAX_SECONDS = 300.0
RETRY_AFTER_JITTER_PERCENT = 10

# Databricks Foundation Model API input-token-per-minute limits. Reservations
# use only 90% so estimation error and requests outside this ucode process have
# some headroom. Model spellings are normalized before lookup, so dotted and
# dashed UC/Codex variants share the same bucket.
PUBLISHED_INPUT_TOKENS_PER_MINUTE = {
    "gpt6astra": 200_000,
    "gpt56sol": 2_000_000,
    "gpt56terra": 2_000_000,
    "gpt56luna": 2_000_000,
    "kimik3": 200_000,
    "qwen35122ba10b": 1_000_000,
    "qwen3next80ba3binstruct": 1_000_000,
}
DEFAULT_TARGET_LIMITS = {
    model: published * SAFETY_PERCENT // 100
    for model, published in PUBLISHED_INPUT_TOKENS_PER_MINUTE.items()
}

_PROCESS_LOCK = threading.Lock()


def _canonical_model_key(model: str) -> str | None:
    """Return the known quota key for a Codex/UC model spelling."""
    normalized = re.sub(r"[^a-z0-9]+", "", model.strip().lower())
    for key in DEFAULT_TARGET_LIMITS:
        if normalized.endswith(key):
            return key
    return None


def estimate_request(body: bytes) -> tuple[str, str, int] | None:
    """Return ``(request model, quota key, estimated input tokens)``.

    Unknown models and non-JSON bodies deliberately pass through. Codex request
    compression is disabled in the launch-scoped proxy config, so a normal
    Responses API request reaches this function as JSON.
    """
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        return None
    model_key = _canonical_model_key(model)
    if model_key is None:
        return None
    estimated_tokens = max(
        1,
        (len(body) + BYTES_PER_ESTIMATED_TOKEN - 1) // BYTES_PER_ESTIMATED_TOKEN,
    )
    return model, model_key, estimated_tokens


def _acquire_file_lock(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release_file_lock(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _locked(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # flock/locking provides process coordination. The Python lock also makes
    # the behavior explicit and portable between threads in one launcher.
    with _PROCESS_LOCK, lock_path.open("a+b") as handle:
        _acquire_file_lock(handle)
        try:
            yield
        finally:
            _release_file_lock(handle)


def _empty_state() -> dict:
    return {"version": 2, "buckets": {}, "cooldowns": {}}


def _read_state(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return _empty_state()
    if not isinstance(payload, dict) or not isinstance(payload.get("buckets"), dict):
        return _empty_state()
    return payload


def _valid_recent_events(raw: object, now: float) -> list[dict[str, float | int]]:
    if not isinstance(raw, list):
        return []
    cutoff = now - WINDOW_SECONDS
    events: list[dict[str, float | int]] = []
    for event in raw:
        if not isinstance(event, dict):
            continue
        at = event.get("at")
        tokens = event.get("tokens")
        if (
            isinstance(at, (int, float))
            and not isinstance(at, bool)
            and isinstance(tokens, int)
            and not isinstance(tokens, bool)
            and tokens > 0
            and cutoff < float(at) <= now + WINDOW_SECONDS
        ):
            events.append({"at": float(at), "tokens": tokens})
    events.sort(key=lambda event: float(event["at"]))
    return events


def _prune_state(state: dict, now: float) -> dict:
    buckets = state.get("buckets")
    clean: dict[str, list[dict[str, float | int]]] = {}
    if isinstance(buckets, dict):
        for key, raw_events in buckets.items():
            if not isinstance(key, str):
                continue
            events = _valid_recent_events(raw_events, now)
            if events:
                clean[key] = events
    cooldowns: dict[str, float] = {}
    raw_cooldowns = state.get("cooldowns")
    if isinstance(raw_cooldowns, dict):
        for key, raw_until in raw_cooldowns.items():
            if (
                isinstance(key, str)
                and isinstance(raw_until, (int, float))
                and not isinstance(raw_until, bool)
                and now < float(raw_until)
            ):
                cooldowns[key] = float(raw_until)
    return {"version": 2, "buckets": clean, "cooldowns": cooldowns}


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Read one response header from either a plain or case-insensitive map."""
    value = headers.get(name)
    if value is not None:
        return value
    lowered = name.lower()
    for key, candidate in headers.items():
        if key.lower() == lowered:
            return candidate
    return None


def _retry_after_seconds(headers: Mapping[str, str], now: float) -> float | None:
    """Parse Retry-After in either delta-seconds or HTTP-date form."""
    value = _header(headers, "Retry-After")
    if value is None:
        return None
    value = value.strip()
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
        seconds = parsed.timestamp() - now
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(state, handle, separators=(",", ":"), sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass


class SharedCodexRateLimiter:
    """Reserve Codex request estimates in a shared rolling 60-second window."""

    def __init__(
        self,
        workspace: str,
        *,
        state_path: Path | None = None,
        lock_path: Path | None = None,
        target_limits: dict[str, int] | None = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
        notice: Callable[[str, float], None] | None = None,
        retry_notice: Callable[[str, float, int, bool], None] | None = None,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        app_dir = config_io.APP_DIR
        self.workspace = workspace.rstrip("/").lower()
        self.state_path = state_path or app_dir / STATE_FILE_NAME
        self.lock_path = lock_path or app_dir / LOCK_FILE_NAME
        self.target_limits = dict(DEFAULT_TARGET_LIMITS if target_limits is None else target_limits)
        self.clock = clock
        self.sleeper = sleeper
        self.notice = notice or self._default_notice
        self.retry_notice = retry_notice or self._default_retry_notice
        self.jitter = jitter
        self._unavailable_notice_sent = False

    @staticmethod
    def _default_notice(model: str, wait_seconds: float) -> None:
        seconds = max(1, math.ceil(wait_seconds))
        sys.stderr.write(
            f"[ucode] Pausing a {model} request for about {seconds}s to stay under the "
            "shared Databricks rate limit.\n"
        )
        sys.stderr.flush()

    @staticmethod
    def _default_retry_notice(
        model: str,
        wait_seconds: float,
        attempt: int,
        used_retry_after: bool,
    ) -> None:
        seconds = max(1, math.ceil(wait_seconds))
        source = "server Retry-After" if used_retry_after else "jittered backoff"
        sys.stderr.write(
            f"[ucode] {model} returned 429; keeping the thread alive and retrying in "
            f"about {seconds}s ({source}, attempt {attempt}).\n"
        )
        sys.stderr.flush()

    def __call__(self, body: bytes) -> None:
        estimate = estimate_request(body)
        try:
            # A 429 whose request body was unreadable creates a workspace-wide
            # cooldown. Honor it even when this request is also unreadable or is
            # for a model whose quota has not been added yet.
            self.wait_for_cooldown(estimate[1] if estimate is not None else None)
            if estimate is None:
                return
            model, model_key, estimated_tokens = estimate
            self.wait_for_capacity(model, model_key, estimated_tokens)
        except OSError as exc:
            # A local permissions/filesystem problem must not make the model
            # endpoint unreachable. Surface it once, then fail open.
            if not self._unavailable_notice_sent:
                self._unavailable_notice_sent = True
                sys.stderr.write(
                    "[ucode] Shared Codex rate limiter is unavailable "
                    f"({type(exc).__name__}); sending without a local throttle.\n"
                )
                sys.stderr.flush()

    def _cooldown_keys(self, model_key: str | None) -> list[str]:
        keys = [f"{self.workspace}|*"]
        if model_key is not None:
            keys.append(f"{self.workspace}|{model_key}")
        return keys

    def wait_for_cooldown(self, model_key: str | None) -> None:
        """Wait for the latest applicable workspace/model 429 cooldown."""
        while True:
            now = self.clock()
            with _locked(self.lock_path):
                state = _prune_state(_read_state(self.state_path), now)
                cooldowns = state["cooldowns"]
                blocked_until = max(
                    (float(cooldowns.get(key, 0.0)) for key in self._cooldown_keys(model_key)),
                    default=0.0,
                )
                _write_state(self.state_path, state)
            if blocked_until <= now:
                return
            self.sleeper(max(0.01, blocked_until - now))

    def _fallback_retry_delay(self, attempt: int) -> float:
        exponent = max(0, attempt - 1)
        ceiling = min(
            RETRY_BACKOFF_MAX_SECONDS,
            RETRY_BACKOFF_BASE_SECONDS * (2 ** min(exponent, 30)),
        )
        # Equal jitter keeps meaningful backoff while preventing many local
        # Codex processes from waking on exactly the same instant.
        return ceiling * (0.5 + 0.5 * min(1.0, max(0.0, self.jitter())))

    def _schedule_cooldown(self, model_key: str | None, delay: float, now: float) -> float:
        """Claim one retry slot at the tail of the shared cooldown queue."""
        bucket_key = self._cooldown_keys(model_key)[-1]
        with _locked(self.lock_path):
            state = _prune_state(_read_state(self.state_path), now)
            cooldowns = state["cooldowns"]
            scheduled_until = max(float(cooldowns.get(bucket_key, 0.0)), now) + delay
            cooldowns[bucket_key] = scheduled_until
            _write_state(self.state_path, state)
        return scheduled_until

    def _sleep_until(self, scheduled_until: float) -> None:
        while True:
            remaining = scheduled_until - self.clock()
            if remaining <= 0:
                return
            self.sleeper(max(0.01, remaining))

    def retry_after_429(
        self,
        body: bytes | None,
        headers: Mapping[str, str],
        attempt: int,
    ) -> None:
        """Publish a shared cooldown, then wait before an internal 429 retry.

        The proxy calls this only after draining a 429 response. Recognized
        models get an isolated bucket; unreadable/compressed bodies and future
        models safely fall back to a workspace-wide bucket. No prompt, response,
        header value, or credential is written to disk.
        """
        estimate = estimate_request(body) if body is not None else None
        model = estimate[0] if estimate is not None else "Codex"
        model_key = estimate[1] if estimate is not None else None
        now = self.clock()
        retry_after = _retry_after_seconds(headers, now)
        used_retry_after = retry_after is not None
        if retry_after is None:
            delay = self._fallback_retry_delay(attempt)
        else:
            # Positive jitter preserves the server's minimum while spreading a
            # local herd. Cap only the added jitter, never Retry-After itself.
            extra_ceiling = min(5.0, max(0.1, retry_after * RETRY_AFTER_JITTER_PERCENT / 100))
            delay = retry_after + extra_ceiling * min(1.0, max(0.0, self.jitter()))
        delay = max(0.01, delay)

        try:
            # Appending rather than overwriting gives simultaneous 429s separate
            # retry slots. Each blocked handler waits for its own slot, while new
            # requests observe the tail and do not jump ahead of the queue.
            scheduled_until = self._schedule_cooldown(model_key, delay, now)
            actual_wait = max(0.01, scheduled_until - now)
            self.retry_notice(model, actual_wait, attempt, used_retry_after)
            self._sleep_until(scheduled_until)
        except OSError as exc:
            self.retry_notice(model, delay, attempt, used_retry_after)
            if not self._unavailable_notice_sent:
                self._unavailable_notice_sent = True
                sys.stderr.write(
                    "[ucode] Shared Codex rate limiter is unavailable "
                    f"({type(exc).__name__}); using local 429 backoff only.\n"
                )
                sys.stderr.flush()
            self.sleeper(delay)

    def wait_for_capacity(self, model: str, model_key: str, estimated_tokens: int) -> None:
        target = self.target_limits.get(model_key)
        if not isinstance(target, int) or target <= 0 or estimated_tokens <= 0:
            return

        # An estimate larger than the local target can never satisfy
        # total+estimate <= target. Reserve one full window instead: it proceeds
        # when the bucket is empty and serializes any following request.
        reservation = min(estimated_tokens, target)
        bucket_key = f"{self.workspace}|{model_key}"
        notice_sent = False

        while True:
            now = self.clock()
            with _locked(self.lock_path):
                state = _prune_state(_read_state(self.state_path), now)
                buckets = state["buckets"]
                events = buckets.setdefault(bucket_key, [])
                used = sum(int(event["tokens"]) for event in events)
                if used + reservation <= target:
                    events.append({"at": now, "tokens": reservation})
                    _write_state(self.state_path, state)
                    return

                oldest = min(float(event["at"]) for event in events)
                wait_seconds = max(0.01, oldest + WINDOW_SECONDS - now)
                _write_state(self.state_path, state)

            # Never hold the cross-process lock while sleeping or printing.
            if not notice_sent:
                self.notice(model, wait_seconds)
                notice_sent = True
            self.sleeper(wait_seconds)
