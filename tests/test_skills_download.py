"""Tests for skills_download.py — the UC skill-download client, on-disk writer,
and download orchestration."""

from __future__ import annotations

import pytest

import ucode.skills_download as sd
from ucode.skills_download import (
    SkillRef,
    existing_skill_on_disk,
    should_download_skill,
    skill_dir_roots,
    write_skill,
)

WS = "https://example.databricks.com"


def ref(securable_name: str, bundle_name: str | None = None) -> SkillRef:
    """A SkillRef whose two names match unless a differing bundle name is given."""
    return SkillRef(securable_name=securable_name, bundle_name=bundle_name or securable_name)


class TestListSchemaSkills:
    def test_keeps_finalized_skills_only(self, monkeypatch):
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

        refs, reason = sd.list_schema_skills(WS, "token", "main", "default")

        assert reason is None
        assert refs == [ref("pii-handling"), ref("triage")]

    def test_keeps_both_names_when_bundle_differs_from_securable(self, monkeypatch):
        # bundle_name comes from the bundle's SKILL.md frontmatter, so it can
        # differ from the securable it was created under.
        payload = {
            "skills": [
                {
                    "name": "skills/main.default.task-prioritizer",
                    "bundle_name": "task-triage",
                    "finalize_time": "2026-06-26T05:58:25Z",
                }
            ]
        }
        monkeypatch.setattr(sd, "_http_get_json", lambda url, token, timeout=30: (payload, None))

        refs, reason = sd.list_schema_skills(WS, "token", "main", "default")

        assert reason is None
        assert refs == [SkillRef(securable_name="task-prioritizer", bundle_name="task-triage")]

    def test_bundle_falls_back_to_securable_leaf(self, monkeypatch):
        payload = {
            "skills": [
                {
                    "name": "skills/main.default.pii-handling",
                    "finalize_time": "2026-06-26T05:58:25Z",
                }
            ]
        }
        monkeypatch.setattr(sd, "_http_get_json", lambda url, token, timeout=30: (payload, None))

        refs, reason = sd.list_schema_skills(WS, "token", "main", "default")

        assert reason is None
        assert refs == [ref("pii-handling")]

    def test_skips_skills_without_a_resource_name(self, monkeypatch):
        payload = {"skills": [{"bundle_name": "orphan", "finalize_time": "t"}]}
        monkeypatch.setattr(sd, "_http_get_json", lambda url, token, timeout=30: (payload, None))

        refs, reason = sd.list_schema_skills(WS, "token", "main", "default")

        assert reason is None
        assert refs == []

    def test_follows_pagination(self, monkeypatch):
        pages = [
            {
                "skills": [{"name": "skills/main.default.a", "finalize_time": "t"}],
                "next_page_token": "tok",
            },
            {"skills": [{"name": "skills/main.default.b", "finalize_time": "t"}]},
        ]
        captured_tokens = []

        def fake_get(url, token, timeout=30):
            captured_tokens.append("page_token=tok" in url)
            return pages.pop(0), None

        monkeypatch.setattr(sd, "_http_get_json", fake_get)

        refs, reason = sd.list_schema_skills(WS, "token", "main", "default")

        assert reason is None
        assert refs == [ref("a"), ref("b")]
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
    def test_lists_under_the_skills_place(self, monkeypatch):
        captured = {}

        def fake_get(url, token, timeout=30):
            captured["url"] = url
            return {"contents": []}, None

        monkeypatch.setattr(sd, "_http_get_json", fake_get)

        sd.list_skill_files(WS, "token", "main", "default", "triage")

        assert captured["url"] == f"{WS}/api/2.0/fs/directories/Skills/main/default/triage"

    def test_walks_nested_directories_into_relative_paths(self, monkeypatch):
        # The Files API returns absolute paths.
        skill = "/Skills/main/default/triage"
        listings = {
            "Skills/main/default/triage": {
                "contents": [
                    {"path": f"{skill}/SKILL.md", "is_directory": False},
                    {"path": f"{skill}/references/", "is_directory": True},
                ]
            },
            "Skills/main/default/triage/references": {
                "contents": [{"path": f"{skill}/references/primary.md", "is_directory": False}]
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
        skill = "/Skills/main/default/triage"
        pages = [
            {
                "contents": [{"path": f"{skill}/a.md", "is_directory": False}],
                "next_page_token": "tok",
            },
            {"contents": [{"path": f"{skill}/b.md", "is_directory": False}]},
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
        assert captured["url"] == f"{WS}/api/2.0/fs/files/Skills/main/default/triage/SKILL.md"

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


class TestShouldDownloadSkill:
    def test_new_skill_is_downloaded(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))

        assert should_download_skill(roots, ref("triage"), location="main.default")

    def test_existing_skill_prompt_keep(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        write_skill(roots, ref("triage"), {"SKILL.md": b"from-main"})

        monkeypatch.setattr(sd, "prompt_yes_no", lambda _: False)

        assert not should_download_skill(roots, ref("triage"), location="ml.prod")

    def test_existing_skill_prompt_overwrite(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        write_skill(roots, ref("triage"), {"SKILL.md": b"from-main"})

        monkeypatch.setattr(sd, "prompt_yes_no", lambda _: True)

        assert should_download_skill(roots, ref("triage"), location="ml.prod")

    def test_invalid_leaf_is_skipped(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))

        assert not should_download_skill(roots, ref("Bad_Name"), location="main.default")

    def test_either_unsafe_name_is_skipped(self, tmp_path):
        # Both names reach the filesystem or a URL, so both must be validated.
        roots = skill_dir_roots(str(tmp_path))

        unsafe_bundle = SkillRef(securable_name="ok-name", bundle_name="../escape")
        unsafe_securable = SkillRef(securable_name="../escape", bundle_name="ok-name")

        assert not should_download_skill(roots, unsafe_bundle, location="main.default")
        assert not should_download_skill(roots, unsafe_securable, location="main.default")

    def test_existing_skill_on_disk_checks_every_root(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))
        assert not existing_skill_on_disk(roots, "triage")

        (roots[1] / "triage").mkdir(parents=True)
        assert existing_skill_on_disk(roots, "triage")


class TestWriteSkill:
    def test_writes_bundle_into_every_root(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))
        files = {"SKILL.md": b"# skill", "scripts/run.py": b"print(1)"}

        write_skill(roots, ref("triage"), files)

        for root in roots:
            assert (root / "triage/SKILL.md").read_bytes() == b"# skill"
            assert (root / "triage/scripts/run.py").read_bytes() == b"print(1)"

    def test_path_traversal_is_rejected(self, tmp_path):
        roots = skill_dir_roots(str(tmp_path))

        write_skill(
            roots, ref("triage"), {"SKILL.md": b"ok", "../escape.md": b"nope", "/abs.md": b"nope"}
        )

        assert (roots[0] / "triage/SKILL.md").read_bytes() == b"ok"
        assert not (tmp_path / "escape.md").exists()


class TestFetchBundles:
    def test_empty_leaves_returns_empty_without_pool(self):
        # min(workers, 0) would raise ValueError in ThreadPoolExecutor; the
        # early return keeps _fetch_bundles safe regardless of caller.
        assert sd._fetch_bundles(WS, "token", "main", "default", []) == {}


class TestDownloadSkills:
    def test_fetches_and_writes_each_leaf(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sd, "list_schema_skills", lambda *a, **k: ([ref("pii-handling"), ref("triage")], None)
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

    def test_fetches_by_securable_and_writes_under_bundle_name(self, tmp_path, monkeypatch):
        # The Files API resolves only the securable, while an agent loads the
        # directory matching the bundle's SKILL.md `name:`.
        diverging = SkillRef(securable_name="task-prioritizer", bundle_name="task-triage")
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: ([diverging], None))
        fetched = []
        monkeypatch.setattr(
            sd,
            "fetch_skill_bundle",
            lambda ws, tok, c, s, securable_name: (
                fetched.append(securable_name) or ({"SKILL.md": b"name: task-triage"}, None)
            ),
        )

        sd.download_skills(WS, "token", ["main.default"], str(tmp_path))

        assert fetched == ["task-prioritizer"]
        for base in (".claude/skills", ".agents/skills"):
            assert (tmp_path / base / "task-triage/SKILL.md").read_bytes() == b"name: task-triage"
            assert not (tmp_path / base / "task-prioritizer").exists()

    def test_skill_filter_matches_securable_name_only(self, tmp_path, monkeypatch):
        # `--skill` selects by the name that identifies the skill in UC, so the
        # bundle name is not a selector even when it differs.
        diverging = SkillRef(securable_name="task-prioritizer", bundle_name="task-triage")
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: ([diverging], None))
        monkeypatch.setattr(sd, "fetch_skill_bundle", lambda *a, **k: ({"SKILL.md": b"ok"}, None))

        selected = tmp_path / "by-securable"
        selected.mkdir()
        sd.download_skills(WS, "token", ["main.default"], str(selected), {"task-prioritizer"})
        assert (selected / ".claude/skills/task-triage/SKILL.md").exists()

        ignored = tmp_path / "by-bundle"
        ignored.mkdir()
        sd.download_skills(WS, "token", ["main.default"], str(ignored), {"task-triage"})
        assert not (ignored / ".claude/skills").exists()

    def test_list_failure_skips_location(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: ([], "HTTP 404 Not Found"))
        called = []
        monkeypatch.setattr(
            sd, "fetch_skill_bundle", lambda *a, **k: called.append(1) or (None, None)
        )

        sd.download_skills(WS, "token", ["main.default"], str(tmp_path))

        assert called == []

    def test_declined_skill_is_not_fetched(self, tmp_path, monkeypatch):
        roots = skill_dir_roots(str(tmp_path))
        write_skill(roots, ref("triage"), {"SKILL.md": b"kept"})
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: ([ref("triage")], None))
        monkeypatch.setattr(sd, "prompt_yes_no", lambda _: False)
        fetched = []
        monkeypatch.setattr(
            sd,
            "fetch_skill_bundle",
            lambda ws, tok, c, s, leaf: fetched.append(leaf) or ({"SKILL.md": b"new"}, None),
        )

        sd.download_skills(WS, "token", ["main.default"], str(tmp_path))

        assert fetched == []
        assert (roots[0] / "triage/SKILL.md").read_bytes() == b"kept"

    def test_bundle_failure_skips_that_skill_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sd, "list_schema_skills", lambda *a, **k: ([ref("good"), ref("bad")], None)
        )
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

    def test_prints_downloaded_count_and_roots_summary(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            sd, "list_schema_skills", lambda *a, **k: ([ref("a"), ref("b"), ref("c")], None)
        )
        monkeypatch.setattr(sd, "fetch_skill_bundle", lambda *a, **k: ({"SKILL.md": b"x"}, None))

        sd.download_skills(WS, "token", ["main.default"], str(tmp_path))

        # Rich wraps long paths across lines; strip all whitespace from both sides to compare.
        roots = sd.skill_dir_roots(str(tmp_path))
        expected = f"Downloaded 3/3 skill(s) from `main.default` in {roots[0]} and {roots[1]}."
        printed = "".join(capsys.readouterr().out.split())
        assert "".join(expected.split()) in printed

    def test_summary_counts_only_written_skills(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            sd, "list_schema_skills", lambda *a, **k: ([ref("good"), ref("bad")], None)
        )
        monkeypatch.setattr(
            sd,
            "fetch_skill_bundle",
            lambda ws, tok, c, s, leaf: (
                ({"SKILL.md": b"ok"}, None) if leaf == "good" else (None, "HTTP 500 Server Error")
            ),
        )

        sd.download_skills(WS, "token", ["main.default"], str(tmp_path))

        assert "Downloaded 1/2 skill(s) from `main.default` in" in capsys.readouterr().out

    def test_skill_filter_downloads_only_matching_leaves(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sd, "list_schema_skills", lambda *a, **k: ([ref("pii-handling"), ref("triage")], None)
        )
        monkeypatch.setattr(sd, "fetch_skill_bundle", lambda *a, **k: ({"SKILL.md": b"x"}, None))

        sd.download_skills(WS, "token", ["main.default"], str(tmp_path), {"triage"})

        assert (tmp_path / ".claude/skills/triage/SKILL.md").read_bytes() == b"x"
        assert not (tmp_path / ".claude/skills/pii-handling").exists()

    def test_skill_filter_warns_on_unknown_and_downloads_rest(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: ([ref("triage")], None))
        monkeypatch.setattr(sd, "fetch_skill_bundle", lambda *a, **k: ({"SKILL.md": b"x"}, None))

        sd.download_skills(WS, "token", ["main.default"], str(tmp_path), {"triage", "ghost"})

        out = capsys.readouterr().out
        assert "Skipping requested skill(s) not found in `main.default`: ghost" in out
        assert (tmp_path / ".claude/skills/triage/SKILL.md").read_bytes() == b"x"

    def test_empty_skill_filter_downloads_nothing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: ([ref("triage")], None))
        called = []
        monkeypatch.setattr(
            sd, "fetch_skill_bundle", lambda *a, **k: called.append(1) or ({"SKILL.md": b"x"}, None)
        )

        sd.download_skills(WS, "token", ["main.default"], str(tmp_path), set())

        assert called == []
        assert not (tmp_path / ".claude/skills/triage").exists()
        # The schema has skills; the filter selected none — distinct from the
        # empty-schema note.
        out = capsys.readouterr().out
        assert "No requested skills to download from `main.default`." in out
        assert "No skills found" not in out

    def test_empty_schema_reports_no_skills_found(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: ([], None))

        sd.download_skills(WS, "token", ["main.default"], str(tmp_path), None)

        assert "No skills found in `main.default`." in capsys.readouterr().out

    def test_none_skill_filter_downloads_everything(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sd, "list_schema_skills", lambda *a, **k: ([ref("a"), ref("b")], None))
        monkeypatch.setattr(sd, "fetch_skill_bundle", lambda *a, **k: ({"SKILL.md": b"x"}, None))

        sd.download_skills(WS, "token", ["main.default"], str(tmp_path), None)

        assert (tmp_path / ".claude/skills/a/SKILL.md").exists()
        assert (tmp_path / ".claude/skills/b/SKILL.md").exists()


class TestConfigureSkillsDownloadCommand:
    def _stub(self, monkeypatch):
        calls: dict[str, object] = {}
        monkeypatch.setattr(sd, "load_state", lambda: {"state": True})
        monkeypatch.setattr(
            sd, "setup_mcp_clients", lambda state, section: (WS, "profile", ["claude"])
        )
        monkeypatch.setattr(sd, "get_databricks_token", lambda ws, profile: "token")
        monkeypatch.setattr(
            sd,
            "download_skills",
            lambda ws, tok, locations, path, skills=None: calls.update(
                download=(ws, tok, locations, path, skills)
            ),
        )
        monkeypatch.setattr(
            sd,
            "register_schemaless_skills_connection",
            lambda state, ws, profile, clients: calls.update(register=(ws, profile, clients)),
        )
        return calls

    def test_downloads_then_registers_connection(self, monkeypatch):
        calls = self._stub(monkeypatch)

        assert sd.configure_skills_download_command(["a.b"], path="/tmp/skills") == 0

        assert calls["download"] == (WS, "token", ["a.b"], "/tmp/skills", None)
        assert calls["register"] == (WS, "profile", ["claude"])

    def test_none_path_threads_through(self, monkeypatch):
        calls = self._stub(monkeypatch)

        assert sd.configure_skills_download_command(["a.b"], path=None) == 0

        assert calls["download"] == (WS, "token", ["a.b"], None, None)
        assert calls["register"] == (WS, "profile", ["claude"])

    def test_skills_filter_threads_through(self, monkeypatch):
        calls = self._stub(monkeypatch)

        assert sd.configure_skills_download_command(["a.b"], path=None, skills={"triage"}) == 0

        assert calls["download"] == (WS, "token", ["a.b"], None, {"triage"})
        assert calls["register"] == (WS, "profile", ["claude"])
