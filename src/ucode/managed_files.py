"""Write agent config into OS-level *managed settings* files.

These files are root-owned and the highest-precedence config scope for their agent — a bare
``claude`` / ``codex`` (launched directly, without ucode) reads them, so writing here is what makes
the gateway config apply outside ``ucode <agent>``:

- Claude Code: ``/etc/claude-code/managed-settings.json`` (Linux),
  ``/Library/Application Support/ClaudeCode/managed-settings.json`` (macOS)
- Codex: ``/etc/codex/managed_config.toml`` (Linux + macOS)

The write is guarded by a **drift check**: it reads the world-readable file WITHOUT sudo and does
nothing when it already matches, so the common no-op launch never prompts for a password; only a
real change shells out to ``sudo`` (temp file → ``sudo cp``), clearing and restoring the immutable
flag (``chattr``/``chflags``) that a fleet golden image may have set. Writing needs root, so the
first write (or one after the config changes) prompts for the developer's sudo password.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from ucode.config_io import is_dry_run
from ucode.ui import console, print_err, print_warning

# Absolute path so a stripped PATH (desktop/GUI launchers) still finds it.
_SUDO = "/usr/bin/sudo"


def managed_files_supported() -> bool:
    """True on the platforms whose managed-settings write path is implemented (Linux, macOS)."""
    return sys.platform == "darwin" or sys.platform.startswith("linux")


def _read_existing(path: Path) -> str:
    """Current file contents, or "" when absent. No sudo — the managed file is world-readable."""
    try:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError:
        return ""


def write_managed_file(path: Path, desired_text: str, *, display: str) -> str:
    """Write ``desired_text`` to a root-owned managed file, only when it differs (drift check).

    Returns ``"written"``, ``"unchanged"``, or ``"skipped"``. Never raises: a permission or immutable
    failure is surfaced as an actionable message and reported as ``"skipped"`` so the launch still
    proceeds (the private ucode config already lets ``ucode <agent>`` work).
    """
    if not managed_files_supported():
        print_warning(
            f"{display}: machine-wide managed settings aren't supported on this platform; "
            f"skipped {path}."
        )
        return "skipped"
    # Drift check first — reading is unprivileged, so an unchanged file never triggers a sudo prompt.
    if _read_existing(path) == desired_text:
        return "unchanged"
    if is_dry_run():
        console.print(f"\n[bold]\\[dry run] {path} (via sudo)[/bold]\n{desired_text}")
        return "written"
    try:
        _sudo_replace(path, desired_text)
    except PermissionError as exc:
        print_err(
            f"{display}: cannot write {path} without root ({exc}). Re-run with `sudo ucode ...` to "
            "apply the config machine-wide."
        )
        return "skipped"
    except subprocess.CalledProcessError as exc:
        _report_sudo_failure(path, display, exc)
        return "skipped"
    return "written"


def _sudo_replace(path: Path, desired_text: str) -> None:
    """Replace ``path`` with ``desired_text`` via sudo (temp file → ``sudo cp``), handling immutability.

    Writes the payload to a user-owned temp file first (no sudo), then copies it into place with
    ``sudo`` and makes it world-readable so the file it lays down is readable by the agent binary
    regardless of who launched it.
    """
    subprocess.run([_SUDO, "mkdir", "-p", str(path.parent)], check=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=path.suffix or ".tmp", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(desired_text)
        tmp_path = tmp.name
    try:
        restore_immutable = _clear_immutable(path)
        try:
            # capture_output so the CalledProcessError on failure (e.g. still-immutable dest) carries
            # cp's stderr for an actionable message.
            subprocess.run(
                [_SUDO, "cp", tmp_path, str(path)], capture_output=True, text=True, check=True
            )
            subprocess.run([_SUDO, "chmod", "a+rx", str(path.parent)], check=True)
            subprocess.run([_SUDO, "chmod", "a+r", str(path)], check=True)
        finally:
            if restore_immutable:
                _restore_immutable(path)
    finally:
        os.unlink(tmp_path)


def _clear_immutable(path: Path) -> bool:
    """Clear an immutable flag a fleet golden image may have set. Returns whether to restore it.

    macOS: preserve JAMF's system-immutable ``schg`` across the update — inspect, unlock only when
    set, and report that it must be restored. Linux: best-effort ``chattr -i`` (not every filesystem
    supports it), never restored.
    """
    try:
        # `path.exists()` stats the file; under a root-locked parent dir (e.g. a 750 /etc/codex we
        # haven't opened yet) that raises PermissionError. There's nothing to unlock we can see, and
        # the subsequent `sudo cp` (as root) overwrites regardless, so treat it as "nothing to clear".
        if not path.exists():
            return False
    except OSError:
        return False
    if sys.platform == "darwin":
        result = subprocess.run(
            ["/usr/bin/stat", "-f", "%Sf", str(path)], capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and "schg" in result.stdout.strip().split(","):
            subprocess.run(
                [_SUDO, "chflags", "noschg", str(path)], capture_output=True, text=True, check=True
            )
            return True
        return False
    subprocess.run([_SUDO, "chattr", "-i", str(path)], capture_output=True, text=True, check=False)
    return False


def _restore_immutable(path: Path) -> None:
    """Re-set macOS's ``schg`` flag after a write. Best-effort so it never masks the write result."""
    result = subprocess.run(
        [_SUDO, "chflags", "schg", str(path)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        print_warning(f"Could not restore the immutable flag on {path}.")


def _report_sudo_failure(path: Path, display: str, exc: subprocess.CalledProcessError) -> None:
    """Surface a sudo helper failure with a concrete fix. An immutable destination is the common
    cause — cp fails with EPERM even under root — so point at the OS-specific clear command."""
    stderr = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
    cmd = exc.cmd or []
    cp_failed = len(cmd) >= 2 and cmd[1] == "cp"
    if cp_failed and "Operation not permitted" in stderr:
        quoted = shlex.quote(str(path))
        clear_cmd = f"sudo {'chflags noschg' if sys.platform == 'darwin' else 'chattr -i'} {quoted}"
        print_err(
            f"{display}: {path} appears to be immutable. Clear the immutable attribute and re-run:\n"
            f"  {clear_cmd}\n  ucode ..."
        )
    else:
        print_err(f"{display}: failed to write managed settings at {path}: {stderr or exc}")


def prune_managed_file(path: Path, pruned_text: str, *, display: str) -> str:
    """Write back a managed file with ucode's keys removed (used by ``ucode revert``).

    ``pruned_text`` is the file's content with ucode's entries stripped. Goes through the same
    drift-suppressed sudo write, so when ucode's keys weren't present the write is a no-op with no
    password prompt.
    """
    return write_managed_file(path, pruned_text, display=display)
