#!/usr/bin/env python3
"""Reproduce GPT Responses prompt-cache retention through Databricks AI Gateway.

Examples:
  # Immediate smoke test (writes, then reads the same conversation prefix)
  uv run python scripts/prompt_cache_repro.py --retention long --phase both

  # Long-retention test; the process sleeps between the two requests
  uv run python scripts/prompt_cache_repro.py --retention long --wait-seconds 2700

  # Control/default test (omits the retention field)
  uv run python scripts/prompt_cache_repro.py --retention default --wait-seconds 2700

  # Explicit in-memory control (not supported by every GPT model)
  uv run python scripts/prompt_cache_repro.py --retention in_memory --wait-seconds 2700

The script persists only the test key, first assistant text, and usage metadata in
``--state-file``. It never writes or prints the Databricks bearer token.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from ucode.databricks import get_databricks_token
from ucode.state import load_state

DEFAULT_MODEL = "system.ai.gpt-5-6-sol"
DEFAULT_CONTEXT_CHARS = 24_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--phase", choices=("write", "read", "both"), default="both")
    parser.add_argument(
        "--retention",
        choices=("long", "default", "in_memory"),
        default="long",
        help="24h, provider default (field omitted), or explicit in_memory",
    )
    parser.add_argument(
        "--wait-seconds", type=float, default=0, help="Delay between write and read in both mode"
    )
    parser.add_argument("--state-file", type=Path, help="State file for write/read phases")
    parser.add_argument("--key", help="Stable prompt_cache_key; generated when omitted")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--workspace", help="Databricks workspace URL; defaults to saved ucode state"
    )
    parser.add_argument("--profile", help="Databricks CLI profile")
    parser.add_argument("--context-chars", type=int, default=DEFAULT_CONTEXT_CHARS)
    return parser


def _retention_value(retention: str) -> str | None:
    if retention == "long":
        return "24h"
    if retention == "in_memory":
        return "in_memory"
    return None


def _fixed_context(key: str, chars: int) -> str:
    if chars < 1024:
        raise ValueError("--context-chars must be at least 1024")
    paragraph = (
        f"This is deterministic prompt-cache reproduction key {key}. "
        "Keep this context unchanged across both requests. "
        "It exists only to create a stable cacheable prefix. "
    )
    repetitions = (chars // len(paragraph)) + 1
    return (paragraph * repetitions)[:chars]


def _first_input(key: str, chars: int) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": _fixed_context(key, chars) + "\n\nReply exactly CACHE_REPRO_WRITE_OK.",
        }
    ]


def _read_input(key: str, chars: int, assistant_text: str) -> list[dict[str, Any]]:
    return [
        *_first_input(key, chars),
        {"role": "assistant", "content": assistant_text or "CACHE_REPRO_WRITE_OK."},
        {"role": "user", "content": "Reply exactly CACHE_REPRO_READ_OK."},
    ]


def _usage(response: dict[str, Any]) -> dict[str, int]:
    raw = response.get("usage") or {}
    details = raw.get("input_tokens_details") or raw.get("prompt_tokens_details") or {}
    return {
        "input_tokens": int(raw.get("input_tokens", raw.get("prompt_tokens", 0)) or 0),
        "cached_tokens": int(
            details.get(
                "cached_tokens", raw.get("cache_read_input_tokens", raw.get("cacheRead", 0))
            )
            or 0
        ),
        "cache_write_tokens": int(
            details.get(
                "cache_write_tokens",
                raw.get("cache_creation_input_tokens", raw.get("cacheWrite", 0)),
            )
            or 0
        ),
        "output_tokens": int(raw.get("output_tokens", raw.get("completion_tokens", 0)) or 0),
    }


def _assistant_text(response: dict[str, Any]) -> str:
    text = response.get("output_text")
    if isinstance(text, str) and text:
        return text
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                return content["text"]
    return "CACHE_REPRO_WRITE_OK."


def _request(
    *,
    workspace: str,
    token: str,
    model: str,
    key: str,
    input_items: list[dict[str, Any]],
    retention: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "max_output_tokens": 64,
        "stream": False,
        "prompt_cache_key": key,
    }
    prompt_retention = _retention_value(retention)
    if prompt_retention is not None:
        body["prompt_cache_retention"] = prompt_retention
    request = urllib.request.Request(
        f"{workspace.rstrip('/')}/ai-gateway/codex/v1/responses",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "ucode-prompt-cache-repro",
            "x-databricks-use-coding-agent-mode": "true",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"Gateway returned HTTP {exc.code}: {detail}") from exc


def _print_result(label: str, response: dict[str, Any], *, retention: str, key: str) -> None:
    print(
        json.dumps(
            {
                "phase": label,
                "model": response.get("model"),
                "prompt_cache_key": key,
                "prompt_cache_retention_requested": _retention_value(retention)
                or "provider-default",
                "usage": _usage(response),
            },
            indent=2,
        )
    )


def _load_saved(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read state file {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("assistant_text"), str):
        raise RuntimeError(f"State file {path} is not a valid cache repro state")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.wait_seconds < 0:
        raise SystemExit("--wait-seconds must be non-negative")
    state = load_state()
    workspace = args.workspace or state.get("workspace") or os.environ.get("DATABRICKS_HOST")
    if not isinstance(workspace, str) or not workspace:
        raise SystemExit("No workspace configured; pass --workspace or run ucode configure first")
    key = args.key or f"ucode-cache-repro-{uuid.uuid4().hex[:12]}"
    state_file = args.state_file or Path(f"/tmp/{key}.json")
    token = get_databricks_token(workspace, args.profile or state.get("profile"))

    if args.phase in ("write", "both"):
        response = _request(
            workspace=workspace,
            token=token,
            model=args.model,
            key=key,
            input_items=_first_input(key, args.context_chars),
            retention=args.retention,
        )
        _print_result("write", response, retention=args.retention, key=key)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(
            json.dumps(
                {
                    "key": key,
                    "model": args.model,
                    "context_chars": args.context_chars,
                    "retention": args.retention,
                    "assistant_text": _assistant_text(response),
                    "write_usage": _usage(response),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"state_file={state_file}")

    if args.phase == "write":
        return 0

    if args.phase == "read":
        saved = _load_saved(state_file)
        key = str(saved["key"])
        model = str(saved["model"])
        context_chars = int(saved["context_chars"])
        assistant_text = str(saved["assistant_text"])
    else:
        saved = _load_saved(state_file)
        model = str(saved["model"])
        context_chars = int(saved["context_chars"])
        assistant_text = str(saved["assistant_text"])
        if args.wait_seconds:
            print(f"waiting_seconds={args.wait_seconds:g}", flush=True)
            time.sleep(args.wait_seconds)

    response = _request(
        workspace=workspace,
        token=token,
        model=model,
        key=key,
        input_items=_read_input(key, context_chars, assistant_text),
        retention=args.retention,
    )
    _print_result("read", response, retention=args.retention, key=key)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
