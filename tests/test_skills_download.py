"""Tests for skills_download.py — the on-disk skill writer (no network)."""

from __future__ import annotations

from pathlib import Path

import ucode.skills_download as sd
from ucode.skills_download import skill_dir_roots, write_skill


class TestSkillDirRoots:
    def test_user_scope_defaults_to_home(self):
        roots = skill_dir_roots(None)
        assert roots == [Path.home() / ".claude/skills", Path.home() / ".agents/skills"]

    def test_project_scope_roots_under_path(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))
        assert roots == [tmp_path / ".claude/skills", tmp_path / ".agents/skills"]


def _write(roots, leaf, files, *, assume_yes=False, written=None, schema_ref="main.default"):
    if written is None:
        written = {}
    return write_skill(
        roots, leaf, files, assume_yes=assume_yes, written=written, schema_ref=schema_ref
    )


class TestWriteSkill:
    def test_writes_bundle_into_every_root(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))
        files = {"SKILL.md": b"# skill", "scripts/run.py": b"print(1)"}

        status = _write(roots, "triage", files)

        assert status == "written"
        for root in roots:
            assert (root / "triage/SKILL.md").read_bytes() == b"# skill"
            assert (root / "triage/scripts/run.py").read_bytes() == b"print(1)"

    def test_same_skill_twice_in_one_run_overwrites_silently(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        written: dict[str, str] = {}

        _write(roots, "triage", {"SKILL.md": b"v1"}, written=written)
        monkeypatch.setattr(sd, "prompt_yes_no", _fail_if_called)
        status = _write(roots, "triage", {"SKILL.md": b"v2"}, written=written)

        assert status == "overwritten"
        assert (roots[0] / "triage/SKILL.md").read_bytes() == b"v2"

    def test_cross_schema_collision_overwrites_with_yes(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        _write(roots, "triage", {"SKILL.md": b"from-main"}, schema_ref="main.default")

        monkeypatch.setattr(sd, "prompt_yes_no", _fail_if_called)
        status = _write(
            roots, "triage", {"SKILL.md": b"from-ml"}, assume_yes=True, schema_ref="ml.prod"
        )

        assert status == "overwritten"
        assert (roots[0] / "triage/SKILL.md").read_bytes() == b"from-ml"

    def test_cross_schema_collision_prompt_keep(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        _write(roots, "triage", {"SKILL.md": b"from-main"}, schema_ref="main.default")

        monkeypatch.setattr(sd, "prompt_yes_no", lambda _: False)
        status = _write(roots, "triage", {"SKILL.md": b"from-ml"}, schema_ref="ml.prod")

        assert status == "kept"
        assert (roots[0] / "triage/SKILL.md").read_bytes() == b"from-main"

    def test_cross_schema_collision_prompt_overwrite(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        _write(roots, "triage", {"SKILL.md": b"from-main"}, schema_ref="main.default")

        monkeypatch.setattr(sd, "prompt_yes_no", lambda _: True)
        status = _write(roots, "triage", {"SKILL.md": b"from-ml"}, schema_ref="ml.prod")

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


def _fail_if_called(_prompt):
    raise AssertionError("prompt_yes_no should not be called")
