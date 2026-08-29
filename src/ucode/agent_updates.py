"""Update checks for npm-installed coding agent CLIs."""

from __future__ import annotations

import json
import re
import shutil
import subprocess

_BASE_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_STABLE_VERSION_RE = re.compile(r"v?\d+\.\d+\.\d+$")


def _base_version(value: str) -> tuple[int, int, int] | None:
    """Return the leading (major, minor, patch) of a version, ignoring any
    prerelease/build suffix (e.g. `-nightly.20260515.g928a311fb`)."""
    match = _BASE_VERSION_RE.search(value)
    if not match:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _is_stable(value: str) -> bool:
    """True for plain `X.Y.Z` releases (no prerelease/nightly/preview suffix)."""
    return bool(_STABLE_VERSION_RE.fullmatch(value.strip()))


def published_versions(package: str) -> list[str]:
    """Return every published version of an npm package, in npm's ascending
    order, or an empty list if the registry can't be reached."""
    if not shutil.which("npm"):
        return []
    try:
        result = subprocess.run(
            ["npm", "view", package, "versions", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        versions = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(versions, str):
        versions = [versions]
    if not isinstance(versions, list):
        return []
    return [v for v in versions if isinstance(v, str)]


def latest_version_below(package: str, ceiling: tuple[int, int, int]) -> str | None:
    """Return the newest published version whose base (major, minor, patch) is
    strictly below `ceiling`, preferring a stable release over a prerelease at
    the same base. Returns None when nothing qualifies or npm is unavailable."""
    candidates = [(v, _base_version(v)) for v in published_versions(package)]
    eligible = [(v, base) for v, base in candidates if base is not None and base < ceiling]
    if not eligible:
        return None
    max_base = max(base for _, base in eligible)
    at_max = [v for v, base in eligible if base == max_base]
    stable = [v for v in at_max if _is_stable(v)]
    pool = stable or at_max
    # npm returns versions in ascending order, so the last entry is newest.
    return pool[-1]
