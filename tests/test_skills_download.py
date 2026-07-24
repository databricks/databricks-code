"""Tests for skills_download.py — the UC skill-download client, on-disk writer,
and download orchestration."""

from __future__ import annotations

import pytest

import ucode.skills_download as sd
from ucode.skills_download import skill_dir_roots, write_skill

WS = "https://example.databricks.com"


class TestListSchemaSkills:
    def test_keeps_finalized_skills_and_uses_bundle_name(self, monkeypatch):
        payload = {
            "skills": [
                {
                    "name": "skills/main.default.pii-handling",
                    "bundle_name": "pii-handling",
                    "finalize_time": "2026-06-26T05:58:25Z",
                },
                {
                    "name": "skills/main.default.triage",
                    "bundle_name": "triage",
                    "finalize_time": "2026-06-26T05:58:26Z",
                },
                {"name": "skills/main.default.draft", "bundle_name": "draft"},
            ]
        }
        monkeypatch.setattr(sd, "_http_get_json", lambda url, token, timeout=30: (payload, None))

        leaves, reason = sd.list_schema_skills(WS, "token", "main", "default")

        assert reason is None
        assert leaves == ["pii-handling", "triage"]

    def test_falls_back_to_resource_name_leaf(self, monkeypatch):
        payload = {
            "skills": [
                {
                    "name": "skills/main.default.pii-handling",
                    "finalize_time": "2026-06-26T05:58:25Z",
                }
            ]
        }
        monkeypatch.setattr(sd, "_http_get_json", lambda url, token, timeout=30: (payload, None))

        leaves, reason = sd.list_schema_skills(WS, "token", "main", "default")

        assert reason is None
        assert leaves == ["pii-handling"]

    def test_follows_pagination(self, monkeypatch):
        pages = [
            {"skills": [{"bundle_name": "a", "finalize_time": "t"}], "next_page_token": "tok"},
            {"skills": [{"bundle_name": "b", "finalize_time": "t"}]},
        ]
        captured_tokens = []

        def fake_get(url, token, timeout=30):
            captured_tokens.append("page_token=tok" in url)
            return pages.pop(0), None

        monkeypatch.setattr(sd, "_http_get_json", fake_get)

        leaves, reason = sd.list_schema_skills(WS, "token", "main", "default")

        assert reason is None
        assert leaves == ["a", "b"]
        assert captured_tokens == [False, True]

    def test_targets_uc_skills_api_for_the_schema(self, monkeypatch):
        captured = {}

        def fake_get(url, token, timeout=30):
            captured["url"] = url
            return {"skills": []}, None

        monkeypatch.setattr(sd, "_http_get_json", fake_get)

        sd.list_schema_skills(WS, "token", "main", "default")

        assert "/api/2.1/unity-catalog/skills?" in captured["url"]
        assert "parent=schemas%2Fmain.default" in captured["url"]

    def test_http_failure_propagates_reason(self, monkeypatch):
        monkeypatch.setattr(
            sd, "_http_get_json", lambda url, token, timeout=30: (None, "HTTP 500 Server Error")
        )

        leaves, reason = sd.list_schema_skills(WS, "token", "main", "default")

        assert leaves == []
        assert reason == "HTTP 500 Server Error"


class TestListSkillFiles:
    def test_walks_nested_directories_into_relative_paths(self, monkeypatch):
        # The Files API returns absolute `/Volumes/...` paths.
        vol = "/Volumes/main/default/triage"
        listings = {
            "Volumes/main/default/triage": {
                "contents": [
                    {"path": f"{vol}/SKILL.md", "is_directory": False},
                    {"path": f"{vol}/references/", "is_directory": True},
                ]
            },
            "Volumes/main/default/triage/references": {
                "contents": [{"path": f"{vol}/references/primary.md", "is_directory": False}]
            },
        }

        def fake_get(url, token, timeout=30):
            directory = url.split("/api/2.0/fs/directories/", 1)[1]
            return listings[directory], None

        monkeypatch.setattr(sd, "_http_get_json", fake_get)

        paths, reason = sd.list_skill_files(WS, "token", "main", "default", "triage")

        assert reason is None
        assert sorted(paths) == ["SKILL.md", "references/primary.md"]

    def test_follows_pagination(self, monkeypatch):
        vol = "/Volumes/main/default/triage"
        pages = [
            {
                "contents": [{"path": f"{vol}/a.md", "is_directory": False}],
                "next_page_token": "tok",
            },
            {"contents": [{"path": f"{vol}/b.md", "is_directory": False}]},
        ]

        monkeypatch.setattr(
            sd, "_http_get_json", lambda url, token, timeout=30: (pages.pop(0), None)
        )

        paths, reason = sd.list_skill_files(WS, "token", "main", "default", "triage")

        assert reason is None
        assert sorted(paths) == ["a.md", "b.md"]

    def test_http_failure_propagates_reason(self, monkeypatch):
        monkeypatch.setattr(
            sd, "_http_get_json", lambda url, token, timeout=30: (None, "HTTP 404 Not Found")
        )

        paths, reason = sd.list_skill_files(WS, "token", "main", "default", "triage")

        assert paths == []
        assert reason == "HTTP 404 Not Found"


class TestFetchSkillFile:
    def test_returns_raw_bytes_from_files_api(self, monkeypatch):
        captured = {}

        def fake_get_bytes(url, token, timeout=30):
            captured["url"] = url
            return b"# SKILL\n", None

        monkeypatch.setattr(sd, "_http_get_bytes", fake_get_bytes)

        body, reason = sd.fetch_skill_file(WS, "token", "main", "default", "triage", "SKILL.md")

        assert reason is None
        assert body == b"# SKILL\n"
        assert captured["url"] == f"{WS}/api/2.0/fs/files/Volumes/main/default/triage/SKILL.md"

    def test_http_failure_propagates_reason(self, monkeypatch):
        monkeypatch.setattr(
            sd, "_http_get_bytes", lambda url, token, timeout=30: (None, "HTTP 404 Not Found")
        )

        body, reason = sd.fetch_skill_file(WS, "token", "main", "default", "triage", "gone.md")

        assert body is None
        assert reason == "HTTP 404 Not Found"


class TestFetchSkillBundle:
    def test_assembles_relpath_to_bytes_map(self, monkeypatch):
        contents = {"SKILL.md": b"# skill", "references/a.md": b"aaa"}
        monkeypatch.setattr(sd, "list_skill_files", lambda *a, **k: (list(contents), None))
        monkeypatch.setattr(
            sd, "fetch_skill_file", lambda ws, tok, c, s, leaf, rel: (contents[rel], None)
        )

        bundle, reason = sd.fetch_skill_bundle(WS, "token", "main", "default", "triage")

        assert reason is None
        assert bundle == contents

    def test_listing_failure_propagates_reason(self, monkeypatch):
        monkeypatch.setattr(sd, "list_skill_files", lambda *a, **k: ([], "HTTP 404 Not Found"))

        bundle, reason = sd.fetch_skill_bundle(WS, "token", "main", "default", "triage")

        assert bundle is None
        assert reason == "HTTP 404 Not Found"

    def test_file_failure_aborts_whole_bundle(self, monkeypatch):
        monkeypatch.setattr(
            sd, "list_skill_files", lambda *a, **k: (["SKILL.md", "broken.md"], None)
        )
        monkeypatch.setattr(
            sd,
            "fetch_skill_file",
            lambda ws, tok, c, s, leaf, rel: (
                (b"ok", None) if rel == "SKILL.md" else (None, "HTTP 500 Server Error")
            ),
        )

        bundle, reason = sd.fetch_skill_bundle(WS, "token", "main", "default", "triage")

        assert bundle is None
        assert reason == "HTTP 500 Server Error"


class TestSkillDirRoots:
    def test_roots_under_project_dir(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))
        assert roots == [tmp_path / ".claude/skills", tmp_path / ".agents/skills"]

    def test_defaults_to_home_when_omitted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sd.Path, "home", classmethod(lambda cls: tmp_path))
        roots = skill_dir_roots(None)
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

        _write(roots, "triage", files)

        for root in roots:
            assert (root / "triage/SKILL.md").read_bytes() == b"# skill"
            assert (root / "triage/scripts/run.py").read_bytes() == b"print(1)"

    def test_existing_skill_prompt_keep(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        _write(roots, "triage", {"SKILL.md": b"from-main"}, location="main.default")

        monkeypatch.setattr(sd, "prompt_yes_no", lambda _: False)
        _write(roots, "triage", {"SKILL.md": b"from-ml"}, location="ml.prod")

        assert (roots[0] / "triage/SKILL.md").read_bytes() == b"from-main"

    def test_existing_skill_prompt_overwrite(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        _write(roots, "triage", {"SKILL.md": b"from-main"}, location="main.default")

        monkeypatch.setattr(sd, "prompt_yes_no", lambda _: True)
        _write(roots, "triage", {"SKILL.md": b"from-ml"}, location="ml.prod")

        assert (roots[0] / "triage/SKILL.md").read_bytes() == b"from-ml"

    def test_invalid_leaf_is_skipped(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))

        _write(roots, "Bad_Name", {"SKILL.md": b"x"})

        assert not (roots[0] / "Bad_Name").exists()

    def test_path_traversal_is_rejected(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))

        _write(roots, "triage", {"SKILL.md": b"ok", "../escape.md": b"nope", "/abs.md": b"nope"})

        assert (roots[0] / "triage/SKILL.md").read_bytes() == b"ok"
        assert not (tmp_path / "escape.md").exists()


class TestDownloadSkills:
    def test_fetches_and_writes_each_leaf(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sd, "list_schema_skills", lambda *a, **k: (["pii-handling", "triage"], None)
        )
        bundles = {
            "pii-handling": {"SKILL.md": b"pii"},
            "triage": {"SKILL.md": b"triage"},
        }
        monkeypatch.setattr(
            sd, "fetch_skill_bundle", lambda ws, tok, c, s, leaf: (bundles[leaf], None)
        )

        sd.download_skills(WS, "token", ["main.default"], str(tmp_path))

        assert (tmp_path / ".claude/skills/pii-handling/SKILL.md").read_bytes() == b"pii"
        assert (tmp_path / ".agents/skills/triage/SKILL.md").read_bytes() == b"triage"

    def test_list_failure_skips_location(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: ([], "HTTP 404 Not Found"))
        called = []
        monkeypatch.setattr(
            sd, "fetch_skill_bundle", lambda *a, **k: called.append(1) or (None, None)
        )

        sd.download_skills(WS, "token", ["main.default"], str(tmp_path))

        assert called == []

    def test_bundle_failure_skips_that_skill_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: (["good", "bad"], None))
        monkeypatch.setattr(
            sd,
            "fetch_skill_bundle",
            lambda ws, tok, c, s, leaf: (
                ({"SKILL.md": b"ok"}, None) if leaf == "good" else (None, "HTTP 500 Server Error")
            ),
        )

        sd.download_skills(WS, "token", ["main.default"], str(tmp_path))

        assert (tmp_path / ".claude/skills/good/SKILL.md").read_bytes() == b"ok"
        assert not (tmp_path / ".claude/skills/bad").exists()
