#!/usr/bin/env python3
"""Diagnose ucode / Databricks relayed-auth failures.

Confirms the root causes behind "sessions lag out after ~1h / require a fresh
`databricks auth login` / spuriously trigger Anthropic re-auth":

  1. Access-token TTL          — how long a minted token actually lives.
  2. Duplicate cache entries   — the same workspace keyed by BOTH profile-name
                                 and host-URL, each with its own refresh token.
  3. Refresh-token rotation    — does using the refresh token invalidate the old
                                 one? (drives the concurrent-refresh race)
  4. Live token validity       — does the current token still authenticate?
  5. Concurrency race (--stress) and overnight lapse (--watch).

Secrets (access/refresh tokens) are NEVER printed — only lengths and claims.

SAFE BY DEFAULT: a bare run only READS ~/.databricks/token-cache.json and
decodes token claims. `--live`, `--stress`, and `--force` call the CLI and can
ROTATE your real refresh token; `--stress` may even require a re-login.

Usage:
    python scripts/diagnose_auth.py --host <workspace-url> [--profile <name>]
    python scripts/diagnose_auth.py --host <url> --profile <name> --live
    python scripts/diagnose_auth.py --host <url> --profile <name> --stress 8
    python scripts/diagnose_auth.py --host <url> --profile <name> --watch 300
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

CACHE = Path.home() / ".databricks" / "token-cache.json"


def _decode_jwt_claims(token: str) -> dict:
    """Best-effort decode of a JWT payload (no signature check). {} if not a JWT."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)  # restore base64 padding
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


def _fmt_ttl(exp: float | None) -> str:
    if not exp:
        return "unknown"
    delta = exp - time.time()
    sign = "in" if delta >= 0 else "EXPIRED"
    mins = abs(delta) / 60
    return f"{sign} {mins:.1f} min" if delta >= 0 else f"{sign} {mins:.1f} min ago"


def _load_cache() -> dict:
    if not CACHE.exists():
        return {}
    return json.loads(CACHE.read_text()).get("tokens", {})


def _matching_keys(host: str, profile: str | None) -> list[str]:
    tokens = _load_cache()
    host_frag = host.replace("https://", "").rstrip("/")
    keys = [k for k in tokens if host_frag in k or (profile and k == profile)]
    return keys


def inspect_cache(host: str, profile: str | None) -> None:
    print("== 1/2. Cache entries + access-token TTL ==")
    tokens = _load_cache()
    keys = _matching_keys(host, profile)
    if not keys:
        print(f"  no cache entries match host={host!r} profile={profile!r}")
        return
    for k in keys:
        entry = tokens[k]
        at = entry.get("access_token", "")
        claims = _decode_jwt_claims(at)
        exp = claims.get("exp")
        iat = claims.get("iat")
        lifetime = f"{(exp - iat) / 60:.0f} min" if exp and iat else "?"
        print(f"  key: {k}")
        print(
            f"     expires_in={entry.get('expires_in')}  jwt-lifetime={lifetime}  "
            f"exp {_fmt_ttl(exp)}  refresh_token={len(entry.get('refresh_token', ''))} chars"
        )
    if len(keys) > 1:
        print(
            f"  ⚠️  {len(keys)} SEPARATE entries for this workspace — each holds its own\n"
            "      refresh token and refreshes independently. `--host URL` and\n"
            "      `--profile NAME` invocations can hit different ones, fragmenting auth."
        )


def _cli_token(host: str, profile: str | None, *, force: bool) -> tuple[int, str, str]:
    cmd = ["databricks", "auth", "token", "--host", host, "--output", "json"]
    if profile:
        cmd += ["--profile", profile]
    if force:
        cmd += ["--force-refresh"]
    env = os.environ.copy()
    env["DATABRICKS_HOST"] = host
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)
    return p.returncode, p.stdout, p.stderr


def _refresh_token_for(host: str, profile: str | None) -> str:
    for k in _matching_keys(host, profile):
        rt = _load_cache()[k].get("refresh_token")
        if rt:
            return rt
    return ""


def check_rotation(host: str, profile: str | None) -> None:
    print("\n== 3. Refresh-token rotation (sequential, safe-ish) ==")
    before = _refresh_token_for(host, profile)
    rc, out, err = _cli_token(host, profile, force=True)
    if rc != 0:
        print(f"  force-refresh FAILED (rc={rc}): {err.strip()[:200]}")
        print("  → this is exactly the failure that forces `databricks auth login`.")
        return
    after = _refresh_token_for(host, profile)
    if before and after and before != after:
        print("  ⚠️  refresh token ROTATED on use (old one is now invalid).")
        print("      => two concurrent refreshes will race; the loser gets logged out.")
    elif before and after and before == after:
        print("  refresh token did NOT rotate (reusable) — concurrent refresh is safer.")
    else:
        print("  could not compare refresh tokens (missing before/after).")


def check_live(host: str, profile: str | None) -> None:
    print("\n== 4. Live token validity ==")
    rc, out, err = _cli_token(host, profile, force=False)
    if rc != 0:
        print(f"  `databricks auth token` FAILED (rc={rc}): {err.strip()[:200]}")
        return
    token = json.loads(out or "{}").get("access_token", "")
    claims = _decode_jwt_claims(token)
    print(f"  minted token: exp {_fmt_ttl(claims.get('exp'))}  ({len(token)} chars)")
    try:
        import httpx

        r = httpx.get(
            f"{host.rstrip('/')}/api/2.0/preview/scim/v2/Me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        print(f"  workspace API /scim/v2/Me -> HTTP {r.status_code} "
              f"({'valid' if r.status_code == 200 else 'REJECTED'})")
    except Exception as exc:
        print(f"  probe skipped: {type(exc).__name__}: {exc}")


def stress(host: str, profile: str | None, n: int) -> None:
    print(f"\n== 5a. Concurrency race: {n} simultaneous force-refresh ==")
    print("  ⚠️  may rotate your token repeatedly and could require re-login.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(lambda _: _cli_token(host, profile, force=True)[0], range(n)))
    ok = sum(1 for rc in results if rc == 0)
    print(f"  succeeded: {ok}/{n}   failed: {n - ok}/{n}")
    if n - ok:
        print(
            "  ⚠️  concurrent refreshes RACED (expect 'cache update: exit status 45') —\n"
            "      the CLI serializes token-cache writes with a file lock; the losers\n"
            "      fail and are told to `databricks auth login`. This is the root cause."
        )


def watch(host: str, profile: str | None, interval: int) -> None:
    print(f"\n== 5b. Watch: probing every {interval}s (Ctrl-C to stop) ==")
    print("  Leave running overnight / during an idle session to catch WHEN auth lapses.")
    while True:
        ts = datetime.now(UTC).astimezone().strftime("%H:%M:%S")
        rc, out, err = _cli_token(host, profile, force=False)
        if rc != 0:
            print(f"  [{ts}] auth token FAILED: {err.strip()[:120]}")
        else:
            token = json.loads(out or "{}").get("access_token", "")
            print(f"  [{ts}] ok, exp {_fmt_ttl(_decode_jwt_claims(token).get('exp'))}")
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, help="workspace URL, e.g. https://x.cloud.databricks.com")
    ap.add_argument("--profile", help="Databricks CLI profile name")
    ap.add_argument("--live", action="store_true", help="mint a token via the CLI and probe it (mutates cache)")
    ap.add_argument("--force", action="store_true", help="run the rotation check (force-refresh; mutates cache)")
    ap.add_argument("--stress", type=int, metavar="N", help="N concurrent force-refreshes to reproduce the race")
    ap.add_argument("--watch", type=int, metavar="SECONDS", help="poll validity on an interval (overnight repro)")
    args = ap.parse_args()

    inspect_cache(args.host, args.profile)
    if args.force:
        check_rotation(args.host, args.profile)
    if args.live:
        check_live(args.host, args.profile)
    if args.stress:
        stress(args.host, args.profile, args.stress)
    if args.watch:
        watch(args.host, args.profile, args.watch)
    if not any([args.force, args.live, args.stress, args.watch]):
        print("\n(read-only run — add --live / --force / --stress N / --watch S to probe further)")


if __name__ == "__main__":
    main()
