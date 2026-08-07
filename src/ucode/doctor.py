"""`ucode doctor` — diagnose the local ucode setup and offer to fix what it can.

Mirrors the `brew doctor` / `flutter doctor` / `npm doctor` pattern: run a
series of independent checks, print a status line for each, and for any problem
ucode knows how to fix, prompt the user to apply the fix and report whether it
worked. The command is read-only until the user says yes to a specific
suggestion, and a declined or piped run (no tty) changes nothing.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from ucode.agents import (
    TOOL_SPECS,
    tool_binary_installed,
    tool_update_available,
    update_tool_binary,
)
from ucode.databricks import (
    MIN_DATABRICKS_CLI_VERSION,
    databricks_cli_version,
    install_databricks_cli,
    upgrade_databricks_cli,
)
from ucode.state import load_state
from ucode.telemetry import ucode_version
from ucode.ui import (
    console,
    heading,
    label,
    print_note,
    print_success,
    print_warning,
    prompt_yes_no_default,
    spinner,
    status_badge,
)

UCODE_GIT_URL = "git+https://github.com/databricks/ucode"

# status -> (glyph, status_badge kind). "info" is a healthy line that still
# carries an optional suggestion (e.g. the ucode self-upgrade).
_BADGES = {
    "ok": ("✓", "ok"),
    "warn": ("!", "warn"),
    "error": ("✗", "error"),
    "info": ("•", "info"),
}


@dataclass
class Suggestion:
    """A fix ucode offers to apply. ``apply`` returns True on success."""

    prompt: str
    apply: Callable[[], bool]


@dataclass
class Check:
    name: str
    status: str  # one of _BADGES
    detail: str
    suggestion: Suggestion | None = None


def _fmt_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(n) for n in version)


# ── individual checks ──────────────────────────────────────────────────────


def _check_uv() -> Check:
    if shutil.which("uv"):
        return Check("uv", "ok", "found on PATH")
    return Check(
        "uv",
        "error",
        "not found — needed to install and upgrade ucode. Install it from "
        "https://docs.astral.sh/uv/getting-started/installation/.",
    )


def _check_npm() -> Check:
    if shutil.which("npm"):
        return Check("npm", "ok", "found on PATH")
    return Check(
        "npm",
        "warn",
        "not found — needed to install coding-agent CLIs automatically. "
        "Install Node.js/npm from https://nodejs.org/.",
    )


def _install_databricks() -> bool:
    try:
        install_databricks_cli()
    except RuntimeError:
        return False
    return shutil.which("databricks") is not None


def _check_databricks_cli() -> Check:
    if not shutil.which("databricks"):
        return Check(
            "Databricks CLI",
            "error",
            "not installed",
            Suggestion("Install the Databricks CLI?", _install_databricks),
        )
    version = databricks_cli_version()
    if version is None:
        return Check("Databricks CLI", "warn", "installed, but its version could not be read")
    current = _fmt_version(version)
    if version < MIN_DATABRICKS_CLI_VERSION:
        floor = _fmt_version(MIN_DATABRICKS_CLI_VERSION)
        return Check(
            "Databricks CLI",
            "warn",
            f"v{current} is below v{floor}, the release that ships `databricks aitools`",
            Suggestion("Upgrade the Databricks CLI to the latest release?", upgrade_databricks_cli),
        )
    return Check("Databricks CLI", "ok", f"v{current}")


def _check_workspace() -> Check:
    workspace = load_state().get("workspace")
    if workspace:
        return Check("Workspace", "ok", str(workspace))
    return Check(
        "Workspace",
        "warn",
        "not configured — run `ucode configure` to set your Databricks workspace",
    )


def _check_agent_clis() -> list[Check]:
    """One check per configured coding agent: installed and up to date?"""
    tools = load_state().get("available_tools") or []
    checks: list[Check] = []
    for tool in tools:
        if tool not in TOOL_SPECS:
            continue
        spec = TOOL_SPECS[tool]
        display = spec["display"]
        if not tool_binary_installed(tool):
            checks.append(
                Check(
                    display,
                    "warn",
                    f"`{spec['binary']}` not found on PATH",
                    Suggestion(f"Install {display}?", lambda t=tool: update_tool_binary(t)),
                )
            )
            continue
        with spinner(f"Checking {display} for updates..."):
            update = tool_update_available(tool)
        if update:
            current, latest = update
            checks.append(
                Check(
                    display,
                    "warn",
                    f"{current} installed; {latest} available",
                    Suggestion(
                        f"Update {display} to {latest}?", lambda t=tool: update_tool_binary(t)
                    ),
                )
            )
        else:
            checks.append(Check(display, "ok", "installed and up to date"))
    return checks


def _upgrade_ucode() -> bool:
    if not shutil.which("uv"):
        print_warning("`uv` is not on PATH; cannot upgrade ucode.")
        return False
    try:
        subprocess.run(["uv", "tool", "install", "--reinstall", UCODE_GIT_URL], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def _check_ucode() -> Check:
    """ucode installs from GitHub (no release tags), so there's no version to
    diff against. Report the installed build and offer a reinstall-to-latest as
    an optional maintenance action rather than claiming it's out of date."""
    version = ucode_version()
    suggestion = (
        Suggestion("Reinstall ucode from GitHub to pick up the latest changes?", _upgrade_ucode)
        if shutil.which("uv")
        else None
    )
    return Check("ucode", "info", f"v{version} (installed from GitHub)", suggestion)


# ── orchestration ──────────────────────────────────────────────────────────


def _gather_checks() -> list[Check]:
    checks: list[Check] = [_check_uv(), _check_npm(), _check_databricks_cli(), _check_workspace()]
    checks.extend(_check_agent_clis())
    checks.append(_check_ucode())
    return checks


def doctor() -> int:
    """Run every check, print its status, and prompt to apply any offered fix."""
    console.print(heading("ucode doctor"))
    console.print()

    checks = _gather_checks()
    problems = 0
    applied = 0
    for check in checks:
        glyph, kind = _BADGES[check.status]
        console.print(f"  {status_badge(glyph, kind)} {label(check.name)}: {check.detail}")
        if check.status in ("warn", "error"):
            problems += 1
        if check.suggestion is None:
            continue
        if prompt_yes_no_default(f"    {check.suggestion.prompt}", default=False):
            if check.suggestion.apply():
                print_success(f"{check.name}: fixed")
                applied += 1
            else:
                print_warning(f"{check.name}: fix did not complete")

    console.print()
    if problems == 0:
        print_success("No problems detected.")
    else:
        noun = "issue" if problems == 1 else "issues"
        print_note(f"{problems} {noun} found; {applied} fix(es) applied.")
    return 0
