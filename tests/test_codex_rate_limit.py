from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ucode import codex_rate_limit


def _body(model: str, padding: str = "") -> bytes:
    return json.dumps({"model": model, "input": padding}, separators=(",", ":")).encode()


def _reserve_in_process(state_path: str, lock_path: str, count: int) -> None:
    limiter = codex_rate_limit.SharedCodexRateLimiter(
        "https://workspace",
        state_path=Path(state_path),
        lock_path=Path(lock_path),
        target_limits={"gpt56sol": 1_000},
        clock=lambda: 100.0,
    )
    for _ in range(count):
        limiter.wait_for_capacity("gpt-5.6-sol", "gpt56sol", 10)


@pytest.mark.parametrize(
    ("model", "expected_key", "published"),
    [
        ("gpt-6-astra", "gpt6astra", 200_000),
        ("system.ai.gpt-5-6-sol", "gpt56sol", 2_000_000),
        ("databricks-gpt-5.6-terra", "gpt56terra", 2_000_000),
        ("eu/gpt-5-6-luna", "gpt56luna", 2_000_000),
        ("system.ai.kimi-k3", "kimik3", 200_000),
    ],
)
def test_estimates_every_known_codex_model(model, expected_key, published):
    body = _body(model, "x" * 100)

    assert codex_rate_limit.estimate_request(body) == (
        model,
        expected_key,
        (len(body) + 2) // 3,
    )
    assert codex_rate_limit.PUBLISHED_INPUT_TOKENS_PER_MINUTE[expected_key] == published
    assert codex_rate_limit.DEFAULT_TARGET_LIMITS[expected_key] == published * 90 // 100


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        b'{"input":"hi"}',
        b'{"model":"gpt-future"}',
    ],
)
def test_unknown_or_unreadable_requests_pass_through(body):
    assert codex_rate_limit.estimate_request(body) is None


def test_exact_window_boundary_releases_capacity(tmp_path):
    now = [100.0]
    sleeps = []
    notices = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    limiter = codex_rate_limit.SharedCodexRateLimiter(
        "https://workspace",
        state_path=tmp_path / "state.json",
        lock_path=tmp_path / "state.lock",
        target_limits={"gpt6astra": 10},
        clock=lambda: now[0],
        sleeper=sleep,
        notice=lambda model, seconds: notices.append((model, seconds)),
    )

    limiter.wait_for_capacity("gpt-6-astra", "gpt6astra", 10)
    limiter.wait_for_capacity("gpt-6-astra", "gpt6astra", 1)

    assert sleeps == [60.0]
    assert notices == [("gpt-6-astra", 60.0)]
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["buckets"]["https://workspace|gpt6astra"] == [{"at": 160.0, "tokens": 1}]


def test_oversized_request_uses_one_full_window_reservation(tmp_path):
    limiter = codex_rate_limit.SharedCodexRateLimiter(
        "https://workspace",
        state_path=tmp_path / "state.json",
        lock_path=tmp_path / "state.lock",
        target_limits={"gpt6astra": 10},
        clock=lambda: 100.0,
        sleeper=lambda _seconds: pytest.fail("first oversized request must not wait forever"),
    )

    limiter.wait_for_capacity("gpt-6-astra", "gpt6astra", 100)

    state = json.loads((tmp_path / "state.json").read_text())
    assert state["buckets"]["https://workspace|gpt6astra"] == [{"at": 100.0, "tokens": 10}]


def test_models_and_workspaces_have_independent_buckets(tmp_path):
    common = {
        "state_path": tmp_path / "state.json",
        "lock_path": tmp_path / "state.lock",
        "target_limits": {"gpt6astra": 10, "gpt56sol": 10},
        "clock": lambda: 100.0,
        "sleeper": lambda _seconds: pytest.fail("independent buckets should not wait"),
    }
    first = codex_rate_limit.SharedCodexRateLimiter("https://one", **common)
    second = codex_rate_limit.SharedCodexRateLimiter("https://two", **common)

    first.wait_for_capacity("gpt-6-astra", "gpt6astra", 10)
    first.wait_for_capacity("gpt-5.6-sol", "gpt56sol", 10)
    second.wait_for_capacity("gpt-6-astra", "gpt6astra", 10)

    state = json.loads((tmp_path / "state.json").read_text())
    assert set(state["buckets"]) == {
        "https://one|gpt6astra",
        "https://one|gpt56sol",
        "https://two|gpt6astra",
    }


def test_concurrent_instances_do_not_lose_reservations(tmp_path):
    state_path = tmp_path / "state.json"
    lock_path = tmp_path / "state.lock"

    def reserve(_index):
        limiter = codex_rate_limit.SharedCodexRateLimiter(
            "https://workspace",
            state_path=state_path,
            lock_path=lock_path,
            target_limits={"gpt56sol": 1_000},
            clock=lambda: 100.0,
            sleeper=lambda _seconds: pytest.fail("reservations fit in the window"),
        )
        limiter.wait_for_capacity("gpt-5.6-sol", "gpt56sol", 10)

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(reserve, range(40)))

    state = json.loads(state_path.read_text())
    events = state["buckets"]["https://workspace|gpt56sol"]
    assert len(events) == 40
    assert sum(event["tokens"] for event in events) == 400


def test_concurrent_processes_share_the_same_reservations(tmp_path):
    state_path = tmp_path / "state.json"
    lock_path = tmp_path / "state.lock"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_reserve_in_process,
            args=(str(state_path), str(lock_path), 10),
        )
        for _ in range(4)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    state = json.loads(state_path.read_text())
    events = state["buckets"]["https://workspace|gpt56sol"]
    assert len(events) == 40
    assert sum(event["tokens"] for event in events) == 400


def test_corrupt_state_is_replaced_under_lock(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("truncated{")
    limiter = codex_rate_limit.SharedCodexRateLimiter(
        "https://workspace",
        state_path=state_path,
        lock_path=tmp_path / "state.lock",
        target_limits={"gpt56luna": 10},
        clock=lambda: 100.0,
    )

    limiter.wait_for_capacity("gpt-5.6-luna", "gpt56luna", 3)

    state = json.loads(state_path.read_text())
    assert state["buckets"]["https://workspace|gpt56luna"][0]["tokens"] == 3
