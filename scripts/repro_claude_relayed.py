#!/usr/bin/env python3
r"""Reproduce intermittent Claude Code stream failures through relayed auth.

The harness runs real, non-interactive ``ucode claude`` turns repeatedly. Each
turn requests a long deterministic response, enables the relayed proxy's safe
transport diagnostics, and saves stdout/stderr separately. It treats Claude's
retry banner as evidence even when Claude eventually recovers and exits zero.

Example:

    uv run python scripts/repro_claude_relayed.py --profile <PROFILE> \
      --workspace https://example.staging.cloud.databricks.com \
      --provider catalog.schema.service --attempts 50

Artifacts contain the synthetic prompt's model output and Claude diagnostics,
but the proxy diagnostics never record request/response bodies or headers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from ucode.databricks import get_databricks_profiles, normalize_workspace_url
from ucode.state import load_state, save_state, set_current_workspace

PROXY_DIAGNOSTICS_ENV = "UCODE_RELAYED_PROXY_DIAGNOSTICS"
_PROXY_PREFIX = "[ucode-relay] "
_MARKERS = {
    "claude_response_stopped": re.compile(r"response stopped arriving", re.IGNORECASE),
    "claude_api_error": re.compile(r"\bapi error\b", re.IGNORECASE),
    "claude_retry": re.compile(r"retrying in .*attempt\s+\d+", re.IGNORECASE),
    "proxy_upstream_request_error": re.compile(r'"event":"upstream_request_error"'),
    "proxy_upstream_stream_error": re.compile(r'"event":"upstream_stream_error"'),
}


def build_prompt(response_lines: int, attempt: int, input_kib: int = 0) -> str:
    nonce = uuid.uuid4().hex
    instructions = (
        f"Transport test {attempt}, nonce {nonce}. Produce exactly {response_lines} lines. "
        "Every line must be `STREAM_TEST <line number> relayed transport payload`. "
        "Number lines consecutively from 1 with no omissions. After those lines, output "
        "`STREAM_TEST_DONE`. Do not use tools, explain, summarize, or stop early."
    )
    if input_kib == 0:
        return instructions
    target_chars = input_kib * 1024
    seed = "deterministic relayed-auth prefill payload 0123456789 abcdefghijklmnopqrstuvwxyz\n"
    filler = (seed * ((target_chars // len(seed)) + 1))[:target_chars]
    return f"{instructions}\nINPUT_PAYLOAD_BEGIN\n{filler}\nINPUT_PAYLOAD_END"


def build_artifact_prompt(artifact_path: Path, sections: int, attempt: int) -> str:
    nonce = uuid.uuid4().hex
    return (
        f"Artifact streaming test {attempt}, nonce {nonce}. Use the Write tool to create "
        f"the Markdown document at {artifact_path}. Write exactly {sections} numbered sections. "
        "Each section must have a descriptive heading and two substantive paragraphs about "
        "designing, operating, testing, or debugging production HTTP/SSE streaming proxies. "
        "Include concrete failure modes, observability signals, and mitigations; do not use "
        "repetitive filler. Put the complete document in the Write tool call rather than in the "
        "chat response. After the file is written, reply `ARTIFACT_TEST_DONE`."
    )


def select_ucode_profile(workspace: str, profile: str) -> str:
    """Validate and persist the profile chosen for this workspace.

    ``ucode claude --workspace`` has no launch-time ``--profile`` flag. Priming
    its per-workspace state makes each auth call use the caller's explicit
    choice instead of resolving the first profile matching the host.
    """
    normalized_workspace = normalize_workspace_url(workspace)
    profile_hosts = {name: host for host, name in get_databricks_profiles()}
    configured_host = profile_hosts.get(profile)
    if configured_host is None:
        available = ", ".join(sorted(profile_hosts)) or "none"
        raise RuntimeError(
            f"Databricks CLI profile {profile!r} was not found. Available profiles: {available}."
        )
    normalized_profile_host = normalize_workspace_url(configured_host)
    if normalized_profile_host != normalized_workspace:
        raise RuntimeError(
            f"Profile {profile!r} targets {normalized_profile_host}, not "
            f"{normalized_workspace}. Choose a profile for the requested workspace."
        )

    set_current_workspace(normalized_workspace)
    state = load_state()
    state["workspace"] = normalized_workspace
    state["profile"] = profile
    save_state(state)
    return normalized_workspace


def build_command(
    ucode_command: list[str],
    *,
    workspace: str,
    provider: str,
    prompt: str,
    debug_file: Path | None = None,
    skip_preflight: bool = True,
    safe_mode: bool = False,
    tools: str = "",
    add_dir: Path | None = None,
) -> list[str]:
    command = [
        *ucode_command,
        "claude",
        "--provider",
        provider,
        "--workspace",
        workspace,
    ]
    if skip_preflight:
        command.append("--skip-preflight")
    command.extend(
        [
            "--",
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--no-session-persistence",
            "--permission-mode",
            "acceptEdits" if tools else "dontAsk",
            "--effort",
            "low",
        ]
    )
    if add_dir is not None:
        command.extend(["--add-dir", str(add_dir)])
    if safe_mode:
        command.append("--safe-mode")
    if debug_file is not None:
        command.extend(["--debug-file", str(debug_file)])
    # Keep this last: Claude's --tools option accepts a variable number of values.
    command.extend(["--tools", tools])
    return command


def classify_output(stdout: str, stderr: str) -> tuple[dict[str, int], dict[str, int], list[int]]:
    combined = f"{stdout}\n{stderr}"
    markers = {name: len(pattern.findall(combined)) for name, pattern in _MARKERS.items()}

    result_errors = 0
    for line in stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "result" and payload.get("is_error") is True:
            result_errors += 1
    if result_errors:
        markers["claude_error_result"] = result_errors

    proxy_events: Counter[str] = Counter()
    upstream_statuses: list[int] = []
    request_paths: dict[str, str] = {}
    message_non_2xx = 0
    zero_byte_timeouts = 0
    for line in stderr.splitlines():
        if _PROXY_PREFIX not in line:
            continue
        raw = line.split(_PROXY_PREFIX, 1)[1].strip()
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = event.get("event")
        if isinstance(name, str):
            proxy_events[name] += 1
        request_id = event.get("request_id")
        if (
            name == "request_start"
            and isinstance(request_id, str)
            and isinstance(event.get("path"), str)
        ):
            request_paths[request_id] = event["path"]
        if name == "upstream_headers" and isinstance(event.get("status"), int):
            status = event["status"]
            upstream_statuses.append(status)
            # Claude probes /api/hello without its subscription credential and
            # may receive an expected 401 before a healthy model turn. Only a
            # non-2xx on the actual Messages API is stream-failure evidence.
            path = request_paths.get(request_id) if isinstance(request_id, str) else None
            if (path is None or path == "/v1/messages") and (status < 200 or status >= 300):
                message_non_2xx += 1
        if (
            name == "client_disconnect"
            and isinstance(request_id, str)
            and request_paths.get(request_id) == "/v1/messages"
            and event.get("phase") == "response"
            and event.get("bytes") == 0
            and isinstance(event.get("elapsed_ms"), int)
            and event["elapsed_ms"] >= 30_000
        ):
            zero_byte_timeouts += 1
    if message_non_2xx:
        markers["proxy_non_2xx"] = message_non_2xx
    if zero_byte_timeouts:
        markers["proxy_zero_byte_client_disconnect"] = zero_byte_timeouts

    return (
        {name: count for name, count in markers.items() if count},
        dict(proxy_events),
        upstream_statuses,
    )


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait()


def _pump_stream(
    stream,
    destination: Path,
    *,
    show_diagnostics: bool,
    print_lock: threading.Lock,
) -> None:
    with destination.open("w", encoding="utf-8") as output:
        for line in iter(stream.readline, ""):
            output.write(line)
            output.flush()
            if show_diagnostics and (
                _PROXY_PREFIX in line
                or "response stopped arriving" in line.lower()
                or "api error" in line.lower()
                or "retrying in" in line.lower()
            ):
                with print_lock:
                    print(f"    {line.rstrip()}", flush=True)
    stream.close()


def run_attempt(
    command: list[str],
    *,
    attempt: int,
    artifact_dir: Path,
    timeout: float,
) -> dict:
    stdout_path = artifact_dir / f"attempt-{attempt:04d}.stdout.jsonl"
    stderr_path = artifact_dir / f"attempt-{attempt:04d}.stderr.log"
    env = os.environ.copy()
    env[PROXY_DIAGNOSTICS_ENV] = "1"
    env["NO_COLOR"] = "1"
    env["FORCE_COLOR"] = "0"
    env["PYTHONUNBUFFERED"] = "1"

    started_at = datetime.now(UTC)
    monotonic_start = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        stderr_path.write_text(f"Could not start command: {exc}\n", encoding="utf-8")
        stdout_path.write_text("", encoding="utf-8")
        return {
            "attempt": attempt,
            "started_at": started_at.isoformat(),
            "elapsed_s": round(time.monotonic() - monotonic_start, 3),
            "returncode": 127,
            "timed_out": False,
            "markers": {"launch_error": 1},
            "proxy_events": {},
            "upstream_statuses": [],
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }

    assert proc.stdout is not None
    assert proc.stderr is not None
    print_lock = threading.Lock()
    stdout_thread = threading.Thread(
        target=_pump_stream,
        args=(proc.stdout, stdout_path),
        kwargs={"show_diagnostics": False, "print_lock": print_lock},
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_pump_stream,
        args=(proc.stderr, stderr_path),
        kwargs={"show_diagnostics": True, "print_lock": print_lock},
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_group(proc)
        returncode = proc.returncode
    except KeyboardInterrupt:
        _terminate_process_group(proc)
        raise
    finally:
        stdout_thread.join(timeout=10)
        stderr_thread.join(timeout=10)

    stdout = stdout_path.read_text(encoding="utf-8")
    stderr = stderr_path.read_text(encoding="utf-8")
    markers, proxy_events, upstream_statuses = classify_output(stdout, stderr)
    if timed_out:
        markers["timeout"] = 1
    return {
        "attempt": attempt,
        "started_at": started_at.isoformat(),
        "elapsed_s": round(time.monotonic() - monotonic_start, 3),
        "returncode": returncode,
        "timed_out": timed_out,
        "markers": markers,
        "proxy_events": proxy_events,
        "upstream_statuses": upstream_statuses,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="explicit Databricks CLI profile whose host must match --workspace",
    )
    parser.add_argument("--workspace", required=True, help="Databricks workspace URL")
    parser.add_argument("--provider", required=True, help="UC Model Provider Service name")
    parser.add_argument(
        "--attempts", type=int, default=20, help="number of Claude turns (default: 20)"
    )
    parser.add_argument(
        "--response-lines",
        type=int,
        default=400,
        help="requested lines per response; larger values hold streams open longer (default: 400)",
    )
    parser.add_argument(
        "--input-kib",
        type=int,
        default=0,
        help="deterministic input payload size in KiB, excluding instructions (default: 0)",
    )
    parser.add_argument(
        "--artifact-sections",
        type=int,
        default=0,
        help="write a Markdown artifact with this many sections instead of plain text (default: 0)",
    )
    parser.add_argument("--timeout", type=float, default=900, help="seconds allowed per turn")
    parser.add_argument(
        "--output-dir", type=Path, help="artifact directory (default: a new /tmp directory)"
    )
    parser.add_argument(
        "--ucode-command",
        default="uv run ucode",
        help="command used to launch ucode (default: 'uv run ucode')",
    )
    parser.add_argument(
        "--full-preflight",
        action="store_true",
        help="run ucode's gateway/model preflight on every attempt (stream path is unchanged)",
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="disable Claude customizations to separate transport failures from hooks/plugins",
    )
    parser.add_argument(
        "--claude-debug",
        action="store_true",
        help="also save Claude's detailed debug log (inspect it before sharing)",
    )
    parser.add_argument(
        "--stop-on-repro", action="store_true", help="stop after the first suspicious attempt"
    )
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")
    if args.response_lines < 1:
        parser.error("--response-lines must be at least 1")
    if args.input_kib < 0:
        parser.error("--input-kib cannot be negative")
    if args.artifact_sections < 0:
        parser.error("--artifact-sections cannot be negative")
    if args.artifact_sections and args.input_kib:
        parser.error("--artifact-sections cannot be combined with --input-kib")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    args = _parse_args()
    try:
        workspace = select_ucode_profile(args.workspace, args.profile)
    except (RuntimeError, ValueError) as exc:
        print(f"Profile selection failed: {exc}", file=sys.stderr)
        return 2
    artifact_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="ucode-claude-relay-repro-"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ucode_command = shlex.split(args.ucode_command)
    if not ucode_command:
        raise SystemExit("--ucode-command cannot be empty")

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "workspace": workspace,
        "profile": args.profile,
        "provider": args.provider,
        "attempts": args.attempts,
        "response_lines": args.response_lines,
        "input_kib": args.input_kib,
        "artifact_sections": args.artifact_sections,
        "timeout_s": args.timeout,
        "ucode_command": ucode_command,
        "full_preflight": args.full_preflight,
        "safe_mode": args.safe_mode,
        "claude_debug": args.claude_debug,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    summary_path = artifact_dir / "summary.jsonl"
    suspicious_attempts = 0

    print(f"Artifacts: {artifact_dir}")
    print(f"Running {args.attempts} sequential relayed-auth turns...")
    try:
        for attempt in range(1, args.attempts + 1):
            attempt_artifact_dir = None
            if args.artifact_sections:
                attempt_artifact_dir = artifact_dir / f"attempt-{attempt:04d}-files"
                attempt_artifact_dir.mkdir(parents=True, exist_ok=True)
                prompt = build_artifact_prompt(
                    attempt_artifact_dir / "streaming-proxy-design.md",
                    args.artifact_sections,
                    attempt,
                )
            else:
                prompt = build_prompt(args.response_lines, attempt, args.input_kib)
            debug_file = (
                artifact_dir / f"attempt-{attempt:04d}.claude-debug.log"
                if args.claude_debug
                else None
            )
            command = build_command(
                ucode_command,
                workspace=workspace,
                provider=args.provider,
                prompt=prompt,
                debug_file=debug_file,
                skip_preflight=not args.full_preflight,
                safe_mode=args.safe_mode,
                tools="Write" if attempt_artifact_dir is not None else "",
                add_dir=attempt_artifact_dir,
            )
            print(f"[{attempt}/{args.attempts}] starting")
            result = run_attempt(
                command,
                attempt=attempt,
                artifact_dir=artifact_dir,
                timeout=args.timeout,
            )
            result["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
            result["prompt_length"] = len(prompt)
            with summary_path.open("a", encoding="utf-8") as summary:
                summary.write(json.dumps(result, sort_keys=True) + "\n")
            suspicious = bool(result["timed_out"] or result["returncode"] != 0 or result["markers"])
            suspicious_attempts += int(suspicious)
            label = "SUSPICIOUS" if suspicious else "ok"
            print(
                f"[{attempt}/{args.attempts}] {label}: rc={result['returncode']} "
                f"elapsed={result['elapsed_s']}s markers={result['markers']}"
            )
            if suspicious and args.stop_on_repro:
                break
    except KeyboardInterrupt:
        print("\nInterrupted; completed artifacts were preserved.", file=sys.stderr)

    print(f"\nSuspicious attempts: {suspicious_attempts}")
    print(f"Summary: {summary_path}")
    if suspicious_attempts:
        print(
            "Reproduction captured. Share metadata.json, summary.jsonl, and the matching attempt logs."
        )
        return 1
    print("No failure observed; increase --attempts or --response-lines and try again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
