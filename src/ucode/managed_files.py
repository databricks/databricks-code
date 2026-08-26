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

import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from ucode import config_io
from ucode.config_io import is_dry_run
from ucode.ui import console, print_err, print_warning

# Absolute path so a stripped PATH (desktop/GUI launchers) still finds it.
_SUDO = "/usr/bin/sudo"


class OS(Enum):
    """The host OS families this module distinguishes, off `sys.platform`."""

    LINUX = "linux"
    MACOS = "macos"
    WINDOWS = "windows"
    OTHER = "other"


def current_os() -> OS:
    """Map `sys.platform` onto :class:`OS` (lowercased, so a mixed-case value can't slip through)."""
    platform = sys.platform.lower()
    if platform.startswith("linux"):
        return OS.LINUX
    if platform == "darwin":
        return OS.MACOS
    if platform.startswith("win"):
        return OS.WINDOWS
    return OS.OTHER


def managed_files_supported() -> bool:
    """True on the platforms whose managed-settings write path is implemented (Linux, macOS).

    The write path needs `sudo` (`sudo cp`, `chattr`/`chflags`), which is Unix-only — so Windows and
    any other platform are unsupported.
    """
    return current_os() in (OS.LINUX, OS.MACOS)


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
    snapshot = _snapshot_original(path)
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
    _record_original(path, snapshot)
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
    if current_os() is OS.MACOS:
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
        clear_cmd = f"sudo {'chflags noschg' if current_os() is OS.MACOS else 'chattr -i'} {quoted}"
        print_err(
            f"{display}: {path} appears to be immutable. Clear the immutable attribute and re-run:\n"
            f"  {clear_cmd}\n  ucode ..."
        )
    else:
        print_err(f"{display}: failed to write managed settings at {path}: {stderr or exc}")


def _ledger_path() -> Path:
    """Machine-global record of each OS-managed file's pre-ucode state (OS files are machine-wide,
    unlike per-workspace ``state.json``), so reconcile can restore the original instead of pruning keys."""
    return config_io.APP_DIR / "managed-os-backups.json"


def _read_ledger() -> dict:
    path = _ledger_path()
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_ledger(ledger: dict) -> None:
    try:
        config_io.APP_DIR.mkdir(parents=True, exist_ok=True)
        _ledger_path().write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    except OSError:
        pass


def _snapshot_original(path: Path) -> dict:
    """Read ``path``'s pre-ucode state into a ledger entry.

    ``contents`` is the exact text for a readable file, ``None`` for one absent or unreadable (kept
    distinct from a genuinely empty ``""``). When the unprivileged stat or read fails (a root-locked
    parent, or a root-only file under a readable dir) it falls back to a privileged probe via
    :func:`_sudo_snapshot`, because guessing here would either leave a ucode-created file uncleaned or
    delete a pre-existing one. Runs only on the write path, so the fallback adds no extra sudo prompt.
    """
    try:
        existed = path.exists()
    except OSError:
        return _sudo_snapshot(path)
    if not existed:
        return {"existed": False, "contents": None}
    try:
        return {"existed": True, "contents": path.read_text(encoding="utf-8")}
    except OSError:
        return _sudo_snapshot(path)


def _sudo_snapshot(path: Path) -> dict:
    """Determine ``path``'s pre-ucode state with sudo, for a root-locked parent or root-only file.

    A file sudo confirms absent is recorded ``existed=False`` so cleanup removes the file ucode is
    about to create; one readable only as root is captured verbatim; one unreadable even as root stays
    ``existed=True/contents=None`` (left untouched on cleanup). A sudo failure is harmless: the write
    that follows fails the same way, so this snapshot is never recorded.
    """
    present = subprocess.run([_SUDO, "test", "-e", str(path)], capture_output=True, check=False)
    if present.returncode != 0:
        return {"existed": False, "contents": None}
    read = subprocess.run([_SUDO, "cat", str(path)], capture_output=True, text=True, check=False)
    if read.returncode == 0:
        return {"existed": True, "contents": read.stdout}
    return {"existed": True, "contents": None}


def _record_original(path: Path, snapshot: dict) -> None:
    """Persist ``snapshot`` to the ledger under ``path``, once (call only after a successful write, so
    a failed sudo leaves no false ownership record). No-op if already recorded."""
    key = str(path)
    ledger = _read_ledger()
    if key in ledger:
        return
    ledger[key] = snapshot
    _write_ledger(ledger)


def _forget_original(key: str) -> None:
    if is_dry_run():
        return
    ledger = _read_ledger()
    if key in ledger:
        del ledger[key]
        _write_ledger(ledger)


def restore_managed_file(path: Path, *, display: str) -> str:
    """Undo ucode's write to ``path``, restoring the pre-ucode original or removing a ucode-created
    file. Returns ``"restored"``, ``"removed"``, ``"unchanged"``, or ``"skipped"``.

    No ledger entry means ucode never wrote it, so nothing happens. Drift-suppressed (an already-matching
    or already-absent file makes no sudo call) and never raises: sudo failures warn and return ``"skipped"``.
    """
    if not managed_files_supported():
        return "unchanged"
    key = str(path)
    entry = _read_ledger().get(key)
    if not isinstance(entry, dict):
        return "unchanged"
    original = entry.get("contents")
    if entry.get("existed"):
        if not isinstance(original, str):
            _forget_original(key)
            return "unchanged"
        if _read_existing(path) == original:
            _forget_original(key)
            return "unchanged"
        if is_dry_run():
            console.print(f"\n[bold]\\[dry run] restore {path} (via sudo)[/bold]\n{original}")
            return "restored"
        if not _sudo_managed_op(lambda: _sudo_replace(path, original), path, display):
            return "skipped"
        _forget_original(key)
        return "restored"
    if not _path_present(path):
        _forget_original(key)
        return "unchanged"
    if is_dry_run():
        console.print(f"\n[bold]\\[dry run] remove {path} (via sudo)[/bold]")
        return "removed"
    if not _sudo_managed_op(lambda: _sudo_remove(path), path, display):
        return "skipped"
    _forget_original(key)
    return "removed"


def _path_present(path: Path) -> bool:
    """Whether ``path`` exists; a root-locked parent we can't stat is assumed present (let sudo try)."""
    try:
        return path.exists()
    except OSError:
        return True


def _sudo_managed_op(op: Callable[[], None], path: Path, display: str) -> bool:
    """Run a privileged managed-file op, turning its failures into a warning. Returns success."""
    try:
        op()
    except PermissionError as exc:
        print_err(
            f"{display}: cannot update {path} without root ({exc}). Re-run with `sudo ucode ...` to "
            "reconcile the machine-wide config."
        )
        return False
    except subprocess.CalledProcessError as exc:
        _report_sudo_failure(path, display, exc)
        return False
    return True


def _sudo_remove(path: Path) -> None:
    """Remove a root-owned managed file via sudo, clearing any immutable flag first."""
    _clear_immutable(path)
    subprocess.run([_SUDO, "rm", "-f", str(path)], capture_output=True, text=True, check=True)
