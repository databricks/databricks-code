"""Tests for skills_download.py — the on-disk skill writer (no network)."""

from __future__ import annotations

import pytest

import ucode.skills_download as sd
from ucode.skills_download import skill_dir_roots, write_skill


class TestSkillDirRoots:
    def test_roots_under_project_dir(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))
        assert roots == [tmp_path / ".claude/skills", tmp_path / ".agents/skills"]

    def test_relative_path_rejected(self):
        with pytest.raises(ValueError, match="absolute"):
            skill_dir_roots("relative/dir")

    def test_missing_directory_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            skill_dir_roots(str(tmp_path / "nope"))


def _write(roots, leaf, files, *, location="main.default"):
    return write_skill(roots, leaf, files, location=location)


class TestWriteSkill:
    def test_writes_bundle_into_every_root(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))
        files = {"SKILL.md": b"# skill", "scripts/run.py": b"print(1)"}

        status = _write(roots, "triage", files)

        assert status == "written"
        for root in roots:
            assert (root / "triage/SKILL.md").read_bytes() == b"# skill"
            assert (root / "triage/scripts/run.py").read_bytes() == b"print(1)"

    def test_existing_skill_prompt_keep(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        _write(roots, "triage", {"SKILL.md": b"from-main"}, location="main.default")

        monkeypatch.setattr(sd, "prompt_yes_no", lambda _: False)
        status = _write(roots, "triage", {"SKILL.md": b"from-ml"}, location="ml.prod")

        assert status == "kept"
        assert (roots[0] / "triage/SKILL.md").read_bytes() == b"from-main"

    def test_existing_skill_prompt_overwrite(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        _write(roots, "triage", {"SKILL.md": b"from-main"}, location="main.default")

        monkeypatch.setattr(sd, "prompt_yes_no", lambda _: True)
        status = _write(roots, "triage", {"SKILL.md": b"from-ml"}, location="ml.prod")

        assert status == "overwritten"
        assert (roots[0] / "triage/SKILL.md").read_bytes() == b"from-ml"

    def test_invalid_leaf_is_skipped(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))

        status = _write(roots, "Bad_Name", {"SKILL.md": b"x"})

        assert status == "skipped"
        assert not (roots[0] / "Bad_Name").exists()

    def test_path_traversal_is_rejected(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))

        status = _write(
            roots, "triage", {"SKILL.md": b"ok", "../escape.md": b"nope", "/abs.md": b"nope"}
        )

        assert status == "written"
        assert (roots[0] / "triage/SKILL.md").read_bytes() == b"ok"
        assert not (tmp_path / "escape.md").exists()
