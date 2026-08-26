"""Tests for managed_files.py — the isaac-style sudo writer for OS managed settings files.

Every test mocks the actual privileged step (`_sudo_replace`), so NO real `sudo` / `/etc` write
ever runs. The behavior that matters here is the drift check: an unchanged file must not shell out.
"""

from __future__ import annotations

import subprocess

import pytest

import ucode.config_io as config_io
from ucode import managed_files


@pytest.fixture(autouse=True)
def _reset_dry_run():
    config_io.set_dry_run(False)
    yield
    config_io.set_dry_run(False)


@pytest.fixture(autouse=True)
def _supported(monkeypatch):
    # Pin platform support on so tests are deterministic on any host.
    monkeypatch.setattr(managed_files, "managed_files_supported", lambda: True)


def _capture_sudo(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        managed_files, "_sudo_replace", lambda path, text: calls.append((str(path), text))
    )
    return calls


class TestWriteManagedFile:
    def test_unchanged_content_does_not_sudo(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text("same", encoding="utf-8")
        calls = _capture_sudo(monkeypatch)
        assert managed_files.write_managed_file(path, "same", display="X") == "unchanged"
        # The whole point: an unchanged file never prompts for a password.
        assert calls == []

    def test_changed_content_sudo_writes(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text("old", encoding="utf-8")
        calls = _capture_sudo(monkeypatch)
        assert managed_files.write_managed_file(path, "new", display="X") == "written"
        assert calls == [(str(path), "new")]

    def test_absent_file_sudo_writes(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        calls = _capture_sudo(monkeypatch)
        assert managed_files.write_managed_file(path, "new", display="X") == "written"
        assert calls == [(str(path), "new")]

    def test_dry_run_does_not_sudo(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        calls = _capture_sudo(monkeypatch)
        config_io.set_dry_run(True)
        assert managed_files.write_managed_file(path, "new", display="X") == "written"
        assert calls == []

    def test_unsupported_platform_skips(self, tmp_path, monkeypatch):
        monkeypatch.setattr(managed_files, "managed_files_supported", lambda: False)
        calls = _capture_sudo(monkeypatch)
        path = tmp_path / "managed.json"
        assert managed_files.write_managed_file(path, "new", display="X") == "skipped"
        assert calls == []

    def test_permission_error_is_skipped_not_raised(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"

        def boom(path, text):
            raise PermissionError("no root")

        monkeypatch.setattr(managed_files, "_sudo_replace", boom)
        # Never raises — the launch proceeds; the private ucode config still works.
        assert managed_files.write_managed_file(path, "new", display="X") == "skipped"

    def test_sudo_failure_is_skipped_not_raised(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"

        def boom(path, text):
            raise subprocess.CalledProcessError(1, ["/usr/bin/sudo", "cp"], stderr="denied")

        monkeypatch.setattr(managed_files, "_sudo_replace", boom)
        assert managed_files.write_managed_file(path, "new", display="X") == "skipped"


class TestClearImmutableStatDenied:
    def test_stat_denied_path_returns_false_without_raising(self, monkeypatch):
        # Regression: `_clear_immutable` ran an unguarded path.exists() inside the sudo write; under a
        # root-locked /etc/codex that raised PermissionError and aborted the write ("without root").
        class _StatDenied:
            def exists(self):
                raise PermissionError(13, "Permission denied")

        # Ensure no sudo subprocess is attempted if the guard ever regresses.
        monkeypatch.setattr(
            managed_files.subprocess, "run", lambda *a, **k: pytest.fail("should not shell out")
        )
        assert managed_files._clear_immutable(_StatDenied()) is False


def _fake_sudo(monkeypatch):
    """Mock the privileged primitives to act on the tmp file directly (no real sudo/`/etc`).

    Records each op so a test can assert whether a privileged call happened at all.
    """
    calls: list = []

    def _replace(path, text):
        calls.append(("write", str(path), text))
        p = managed_files.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def _remove(path):
        calls.append(("remove", str(path)))
        p = managed_files.Path(path)
        if p.exists():
            p.unlink()

    monkeypatch.setattr(managed_files, "_sudo_replace", _replace)
    monkeypatch.setattr(managed_files, "_sudo_remove", _remove)
    return calls


class TestCaptureOnWrite:
    """`write_managed_file` records the pre-ucode file, once, only on a real change."""

    def test_absent_file_captured_as_not_existing(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        _fake_sudo(monkeypatch)
        managed_files.write_managed_file(path, "new", display="X")
        assert managed_files._read_ledger()[str(path)] == {"existed": False, "contents": None}

    def test_existing_file_captures_original_contents(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text("orig", encoding="utf-8")
        _fake_sudo(monkeypatch)
        managed_files.write_managed_file(path, "new", display="X")
        assert managed_files._read_ledger()[str(path)] == {"existed": True, "contents": "orig"}

    def test_capture_is_idempotent_across_writes(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text("orig", encoding="utf-8")
        _fake_sudo(monkeypatch)
        managed_files.write_managed_file(path, "first", display="X")
        managed_files.write_managed_file(path, "second", display="X")
        assert managed_files._read_ledger()[str(path)] == {"existed": True, "contents": "orig"}

    def test_unchanged_write_does_not_capture(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text("same", encoding="utf-8")
        _fake_sudo(monkeypatch)
        managed_files.write_managed_file(path, "same", display="X")
        assert managed_files._read_ledger() == {}

    def test_dry_run_does_not_capture(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text("orig", encoding="utf-8")
        _fake_sudo(monkeypatch)
        config_io.set_dry_run(True)
        managed_files.write_managed_file(path, "new", display="X")
        assert managed_files._read_ledger() == {}

    def test_failed_sudo_records_no_ledger_entry(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text("orig", encoding="utf-8")

        def boom(path, text):
            raise PermissionError("no root")

        monkeypatch.setattr(managed_files, "_sudo_replace", boom)
        assert managed_files.write_managed_file(path, "new", display="X") == "skipped"
        assert managed_files._read_ledger() == {}


class TestSnapshotOriginal:
    """The captured pre-ucode state distinguishes absent / empty / readable / unreadable."""

    def test_absent_file(self, tmp_path):
        snap = managed_files._snapshot_original(tmp_path / "nope.json")
        assert snap == {"existed": False, "contents": None}

    def test_readable_file(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text("orig", encoding="utf-8")
        assert managed_files._snapshot_original(path) == {"existed": True, "contents": "orig"}

    def test_empty_file_is_empty_string_not_none(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text("", encoding="utf-8")
        assert managed_files._snapshot_original(path) == {"existed": True, "contents": ""}

    def test_read_failure_routes_to_sudo_snapshot(self, tmp_path, monkeypatch):
        path = tmp_path / "m.json"
        path.write_text("secret IT config", encoding="utf-8")
        orig_read = managed_files.Path.read_text

        def fake_read(self, *args, **kwargs):
            if self.name == "m.json":
                raise PermissionError("denied")
            return orig_read(self, *args, **kwargs)

        monkeypatch.setattr(managed_files.Path, "read_text", fake_read)
        monkeypatch.setattr(
            managed_files, "_sudo_snapshot", lambda p: {"existed": True, "contents": "via sudo"}
        )
        assert managed_files._snapshot_original(path) == {"existed": True, "contents": "via sudo"}

    def test_locked_parent_routes_to_sudo_snapshot(self, tmp_path, monkeypatch):
        path = tmp_path / "m.json"
        orig_exists = managed_files.Path.exists

        def fake_exists(self, *args, **kwargs):
            if self.name == "m.json":
                raise PermissionError("denied")
            return orig_exists(self, *args, **kwargs)

        monkeypatch.setattr(managed_files.Path, "exists", fake_exists)
        monkeypatch.setattr(
            managed_files, "_sudo_snapshot", lambda p: {"existed": False, "contents": None}
        )
        assert managed_files._snapshot_original(path) == {"existed": False, "contents": None}


class TestSudoSnapshot:
    """The privileged fallback for a root-locked parent or root-only file."""

    def _fake_run(self, monkeypatch, *, present, cat_rc=0, cat_out=""):
        def run(cmd, *args, **kwargs):
            if cmd[:2] == [managed_files._SUDO, "test"]:
                return subprocess.CompletedProcess(cmd, 0 if present else 1)
            if cmd[:2] == [managed_files._SUDO, "cat"]:
                return subprocess.CompletedProcess(cmd, cat_rc, stdout=cat_out)
            raise AssertionError(f"unexpected command {cmd}")

        monkeypatch.setattr(managed_files.subprocess, "run", run)

    def test_sudo_absent_is_existed_false(self, tmp_path, monkeypatch):
        self._fake_run(monkeypatch, present=False)
        assert managed_files._sudo_snapshot(tmp_path / "m") == {"existed": False, "contents": None}

    def test_root_only_readable_is_captured(self, tmp_path, monkeypatch):
        self._fake_run(monkeypatch, present=True, cat_rc=0, cat_out="it config")
        snap = managed_files._sudo_snapshot(tmp_path / "m")
        assert snap == {"existed": True, "contents": "it config"}

    def test_unreadable_even_as_root_is_none(self, tmp_path, monkeypatch):
        self._fake_run(monkeypatch, present=True, cat_rc=1)
        assert managed_files._sudo_snapshot(tmp_path / "m") == {"existed": True, "contents": None}


class TestRestoreManagedFile:
    """The inverse of `write_managed_file`: put the pre-ucode file back, or remove one ucode made."""

    def test_workspace_a_to_b_restores_original_and_preserves_unrelated(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "managed.json"
        path.write_text('{"itKey": "keep"}', encoding="utf-8")
        _fake_sudo(monkeypatch)
        assert (
            managed_files.write_managed_file(
                path, '{"itKey": "keep", "ucode": 1}', display="Claude Code"
            )
            == "written"
        )
        assert managed_files.restore_managed_file(path, display="Claude Code") == "restored"
        assert path.read_text(encoding="utf-8") == '{"itKey": "keep"}'

    def test_created_file_with_no_original_is_removed(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"  # absent before ucode
        _fake_sudo(monkeypatch)
        managed_files.write_managed_file(path, "ucode", display="Codex")
        assert path.exists()
        assert managed_files.restore_managed_file(path, display="Codex") == "removed"
        assert not path.exists()

    def test_second_restore_is_a_noop_without_privilege(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text("orig", encoding="utf-8")
        calls = _fake_sudo(monkeypatch)
        managed_files.write_managed_file(path, "ucode", display="X")
        assert managed_files.restore_managed_file(path, display="X") == "restored"
        before = len(calls)
        assert managed_files.restore_managed_file(path, display="X") == "unchanged"
        assert len(calls) == before

    def test_no_ledger_entry_is_a_noop(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text("some IT file", encoding="utf-8")
        calls = _fake_sudo(monkeypatch)
        assert managed_files.restore_managed_file(path, display="X") == "unchanged"
        assert calls == []
        assert path.read_text(encoding="utf-8") == "some IT file"

    def test_missing_created_file_is_a_noop(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        managed_files._write_ledger({str(path): {"existed": False, "contents": None}})
        calls = _fake_sudo(monkeypatch)
        assert managed_files.restore_managed_file(path, display="X") == "unchanged"
        assert calls == []

    def test_preexisting_but_unreadable_original_is_left_untouched(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text("some IT file", encoding="utf-8")
        managed_files._write_ledger({str(path): {"existed": True, "contents": None}})
        calls = _fake_sudo(monkeypatch)
        assert managed_files.restore_managed_file(path, display="X") == "unchanged"
        assert calls == []
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "some IT file"

    def test_unsupported_platform_is_safe(self, tmp_path, monkeypatch):
        monkeypatch.setattr(managed_files, "managed_files_supported", lambda: False)
        path = tmp_path / "managed.json"
        managed_files._write_ledger({str(path): {"existed": True, "contents": "orig"}})
        calls = _fake_sudo(monkeypatch)
        assert managed_files.restore_managed_file(path, display="X") == "unchanged"
        assert calls == []

    def test_permission_error_is_skipped_not_raised(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text("ucode", encoding="utf-8")
        managed_files._write_ledger({str(path): {"existed": True, "contents": "orig"}})

        def boom(path, text):
            raise PermissionError("no root")

        monkeypatch.setattr(managed_files, "_sudo_replace", boom)
        assert managed_files.restore_managed_file(path, display="X") == "skipped"
        assert str(path) in managed_files._read_ledger()

    def test_sudo_failure_is_skipped_not_raised(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        managed_files._write_ledger({str(path): {"existed": False, "contents": None}})
        path.write_text("ucode", encoding="utf-8")

        def boom(path):
            raise subprocess.CalledProcessError(1, ["/usr/bin/sudo", "rm"], stderr="denied")

        monkeypatch.setattr(managed_files, "_sudo_remove", boom)
        assert managed_files.restore_managed_file(path, display="X") == "skipped"

    def test_dry_run_restore_makes_no_privileged_call(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text("ucode", encoding="utf-8")
        managed_files._write_ledger({str(path): {"existed": True, "contents": "orig"}})
        calls = _fake_sudo(monkeypatch)
        config_io.set_dry_run(True)
        assert managed_files.restore_managed_file(path, display="X") == "restored"
        assert calls == []
        assert path.read_text(encoding="utf-8") == "ucode"
        assert str(path) in managed_files._read_ledger()
