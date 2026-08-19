#!/usr/bin/env python3
"""Print sanitized UAG prompt-cache request bodies and JSON responses.

This is a shareable diagnostic companion to ``prompt_cache_repro.py``. It sends
requests to the Databricks UAG Responses endpoint, prints each request URL and
body before sending it, and prints the response body afterward. Authorization
headers are never printed; if a response echoes the bearer token, the exact
token is redacted before output.

Examples:
  # Explicit 24-hour retention (the default behavior when no retention flag is used)
  uv run python scripts/prompt_cache_http_trace.py --retention long

  # Provider-default behavior: prompt_cache_retention is omitted
  uv run python scripts/prompt_cache_http_trace.py --retention default

  # Print only one write request/response
  uv run python scripts/prompt_cache_http_trace.py --phase write --retention long
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from ucode.databricks import get_databricks_token
from ucode.state import load_state

try:
    from scripts.prompt_cache_repro import (
        DEFAULT_CONTEXT_CHARS,
        DEFAULT_MODEL,
        _assistant_text,
        _first_input,
        _load_saved,
        _read_input,
        _retention_value,
        _usage,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/prompt_cache_http_trace.py
    from prompt_cache_repro import (
        DEFAULT_CONTEXT_CHARS,
        DEFAULT_MODEL,
        _assistant_text,
        _first_input,
        _load_saved,
        _read_input,
        _retention_value,
        _usage,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("write", "read", "both"), default="both")
    parser.add_argument(
        "--retention",
        choices=("long", "default", "in_memory"),
        default="long",
        help="24h, provider default (field omitted), or explicit in_memory",
    )
    parser.add_argument("--state-file", type=Path, help="State file for write/read phases")
    parser.add_argument("--key", help="Stable prompt_cache_key; generated when omitted")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workspace", help="Databricks workspace URL; defaults to saved state")
    parser.add_argument("--profile", help="Databricks CLI profile")
    parser.add_argument("--context-chars", type=int, default=DEFAULT_CONTEXT_CHARS)
    return parser


def _redact(value: str, token: str) -> str:
    """Redact the exact bearer token without altering other response content."""
    return value.replace(token, "<redacted-bearer-token>")


def _print_json(label: str, value: Any, token: str) -> None:
    rendered = json.dumps(value, indent=2, ensure_ascii=False)
    print(f"{label}:")
    print(_redact(rendered, token))


def _request_body(
    *,
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
    return body


def _request(
    *,
    workspace: str,
    token: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    url = f"{workspace.rstrip('/')}/ai-gateway/codex/v1/responses"
    print(f"request_url: {url}")
    _print_json("request_body", body, token)
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "ucode-prompt-cache-http-trace",
            "x-databricks-use-coding-agent-mode": "true",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read().decode()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        print(f"response_status: {exc.code}")
        try:
            _print_json("response_body", json.loads(raw), token)
        except json.JSONDecodeError:
            print(f"response_body: {_redact(raw, token)}")
        raise RuntimeError(f"Gateway returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach gateway: {exc.reason}") from exc

    print(f"response_status: {status}")
    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"response_body: {_redact(raw, token)}")
        raise RuntimeError("Gateway returned a non-JSON response") from exc
    _print_json("response_body", response, token)
    if not isinstance(response, dict):
        raise RuntimeError("Gateway returned a JSON value instead of an object")
    return response


def _write_state(
    path: Path,
    *,
    key: str,
    model: str,
    context_chars: int,
    retention: str,
    response: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "key": key,
                "model": model,
                "context_chars": context_chars,
                "retention": retention,
                "assistant_text": _assistant_text(response),
                "write_usage": _usage(response),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"state_file={path}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.context_chars < 1024:
        raise SystemExit("--context-chars must be at least 1024")

    state = load_state()
    workspace = args.workspace or state.get("workspace") or os.environ.get("DATABRICKS_HOST")
    if not isinstance(workspace, str) or not workspace:
        raise SystemExit("No workspace configured; pass --workspace or run ucode configure first")
    key = args.key or f"ucode-cache-trace-{uuid.uuid4().hex[:12]}"
    state_file = args.state_file or Path(f"/tmp/{key}.json")
    token = get_databricks_token(workspace, args.profile or state.get("profile"))

    if args.phase in ("write", "both"):
        response = _request(
            workspace=workspace,
            token=token,
            body=_request_body(
                model=args.model,
                key=key,
                input_items=_first_input(key, args.context_chars),
                retention=args.retention,
            ),
        )
        _write_state(
            state_file,
            key=key,
            model=args.model,
            context_chars=args.context_chars,
            retention=args.retention,
            response=response,
        )

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

    _request(
        workspace=workspace,
        token=token,
        body=_request_body(
            model=model,
            key=key,
            input_items=_read_input(key, context_chars, assistant_text),
            retention=args.retention,
        ),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
