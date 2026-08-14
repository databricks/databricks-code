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
