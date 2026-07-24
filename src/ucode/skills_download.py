"""Download Unity Catalog skills and write them to disk, one flat dir per skill."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast
from urllib.parse import urlencode

from ucode.databricks import _http_get_bytes, _http_get_json, workspace_hostname
from ucode.ui import print_note, print_success, print_warning, prompt_yes_no

# `.claude/skills` (Claude) + `.agents/skills` (the alias other agents read).
SKILL_BASE_DIR_NAMES = (".claude/skills", ".agents/skills")

SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")

# Parallel skill fetches per schema; writes stay sequential (they prompt).
_MAX_FETCH_WORKERS = 8


# --- Download client (UC skills API + Files API) ---------------------------


def _skill_bundle_name(skill: dict) -> str | None:
    """The downloadable leaf name of a skill, or None if it isn't finalized.

    Only finalized skills (those with a ``finalize_time``) have bundle content
    to download. ``bundle_name`` is the leaf; fall back to the last dotted
    segment of the resource ``name`` (``skills/<cat>.<sch>.<leaf>``).
    """
    if not skill.get("finalize_time"):
        return None
    bundle_name = skill.get("bundle_name")
    if isinstance(bundle_name, str) and bundle_name:
        return bundle_name
    name = skill.get("name")
    return name.rsplit(".", 1)[-1] if isinstance(name, str) else None


def list_schema_skills(
    workspace: str, token: str, catalog: str, schema: str
) -> tuple[list[str], str | None]:
    """List the finalized skill leaf names in ``<catalog>.<schema>``.

    A non-None reason indicates the listing call itself failed.
    """
    hostname = workspace_hostname(workspace)
    base_url = f"https://{hostname}/api/2.1/unity-catalog/skills"
    query = {"parent": f"schemas/{catalog}.{schema}"}

    leaves: list[str] = []
    page_token: str | None = None
    while True:
        if page_token:
            query["page_token"] = page_token
        payload, reason = _http_get_json(f"{base_url}?{urlencode(query)}", token, timeout=30)
        if payload is None:
            return [], reason
        data = cast(dict, payload) if isinstance(payload, dict) else {}
        for skill in data.get("skills") or []:
            leaf = _skill_bundle_name(skill) if isinstance(skill, dict) else None
            if leaf:
                leaves.append(leaf)
        page_token = data.get("next_page_token")
        if not page_token:
            return leaves, None


def list_skill_files(
    workspace: str, token: str, catalog: str, schema: str, leaf: str
) -> tuple[list[str], str | None]:
    """List a skill bundle's files, as paths relative to the skill directory.

    Recursively walks the skill's UC Volume directory (including ``SKILL.md``).
    A non-None reason indicates the listing call itself failed.
    """
    hostname = workspace_hostname(workspace)
    dirs_base = f"https://{hostname}/api/2.0/fs/directories"
    volume_prefix = f"/Volumes/{catalog}/{schema}/{leaf}/"

    relative_paths: list[str] = []
    pending = [f"Volumes/{catalog}/{schema}/{leaf}"]
    while pending:
        directory = pending.pop()
        page_token: str | None = None
        while True:
            url = f"{dirs_base}/{directory}"
            if page_token:
                url = f"{url}?{urlencode({'page_token': page_token})}"
            payload, reason = _http_get_json(url, token, timeout=30)
            if payload is None:
                return [], reason
            data = cast(dict, payload) if isinstance(payload, dict) else {}
            for entry in data.get("contents") or []:
                path = entry.get("path") if isinstance(entry, dict) else None
                if not isinstance(path, str):
                    continue
                if entry.get("is_directory"):
                    pending.append(path.strip("/"))
                else:
                    relative_paths.append(path.removeprefix(volume_prefix))
            page_token = data.get("next_page_token")
            if not page_token:
                break
    return relative_paths, None


def fetch_skill_file(
    workspace: str, token: str, catalog: str, schema: str, leaf: str, relative_path: str
) -> tuple[bytes | None, str | None]:
    """Fetch one skill bundle file's raw bytes from its UC Volume."""
    hostname = workspace_hostname(workspace)
    url = f"https://{hostname}/api/2.0/fs/files/Volumes/{catalog}/{schema}/{leaf}/{relative_path}"
    return _http_get_bytes(url, token, timeout=30)


def fetch_skill_bundle(
    workspace: str, token: str, catalog: str, schema: str, leaf: str
) -> tuple[dict[str, bytes] | None, str | None]:
    """Fetch a whole skill bundle as ``{relative_path: bytes}``.

    Lists the skill's files then fetches each one. All-or-nothing: a non-None
    reason (and None bundle) means the listing or any file fetch failed, so a
    partially-downloaded skill is never written to disk.
    """
    relative_paths, reason = list_skill_files(workspace, token, catalog, schema, leaf)
    if reason:
        return None, reason
    bundle: dict[str, bytes] = {}
    for relative_path in relative_paths:
        content, reason = fetch_skill_file(workspace, token, catalog, schema, leaf, relative_path)
        if content is None:
            return None, reason
        bundle[relative_path] = content
    return bundle, None


# --- On-disk writer --------------------------------------------------------


def skill_dir_roots(project_dir: str | None) -> list[Path]:
    """The ``.claude/skills`` and ``.agents/skills`` roots to download into.

    ``project_dir`` must be an existing absolute directory when given; when
    omitted, roots default to the user's home directory (user scope).
    """
    if project_dir is None:
        base = Path.home()
    else:
        base = Path(project_dir)
        if not base.is_absolute():
            raise ValueError(f"--path must be an absolute path, got `{project_dir}`.")
        if not base.is_dir():
            raise ValueError(f"--path directory does not exist: `{project_dir}`.")
    return [base / name for name in SKILL_BASE_DIR_NAMES]


def _is_valid_leaf(leaf: str) -> bool:
    return bool(SKILL_NAME_PATTERN.match(leaf))


def _safe_relative_path(relative_path: str) -> Path | None:
    """A bundle file's path within its skill dir, or None if it escapes the dir.

    The Files API returns server-controlled paths, but ucode writes them to
    disk, so reject absolute paths and any ``..`` traversal.
    """
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def _write_bundle(skill_dir: Path, leaf: str, files: dict[str, bytes]) -> None:
    for relative_path, content in files.items():
        safe_path = _safe_relative_path(relative_path)
        if safe_path is None:
            print_warning(f"Skipping unsafe path in `{leaf}`: {relative_path}")
            continue
        destination = skill_dir / safe_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def write_skill(roots: list[Path], leaf: str, files: dict[str, bytes], *, location: str) -> None:
    """Write ``leaf``'s bundle (``{relpath: bytes}``) into every root.

    Prompts before overwriting an existing skill dir. ``location`` is the source
    ``<catalog>.<schema>``, shown in that prompt.
    """
    if not _is_valid_leaf(leaf):
        print_warning(f"Skipping `{leaf}`: not a valid skill name (lowercase a-z, 0-9, -).")
        return

    already_on_disk = any((root / leaf).exists() for root in roots)
    if already_on_disk and not prompt_yes_no(
        f"A skill named `{leaf}` already exists. Overwrite it with `{location}.{leaf}`?"
    ):
        print_note(f"Kept existing `{leaf}`.")
        return

    for root in roots:
        _write_bundle(root / leaf, leaf, files)
    verb = "Overwrote" if already_on_disk else "Downloaded"
    print_success(f"{verb} `{leaf}` ({len(files)} file(s)).")


# --- Orchestration ---------------------------------------------------------


def _fetch_bundles(
    workspace: str, token: str, catalog: str, schema: str, leaves: list[str]
) -> dict[str, tuple[dict[str, bytes] | None, str | None]]:
    """Fetch every leaf's bundle concurrently, keyed by leaf name."""
    results: dict[str, tuple[dict[str, bytes] | None, str | None]] = {}
    with ThreadPoolExecutor(max_workers=min(_MAX_FETCH_WORKERS, len(leaves))) as pool:
        futures = {
            pool.submit(fetch_skill_bundle, workspace, token, catalog, schema, leaf): leaf
            for leaf in leaves
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def download_skills(workspace: str, token: str, locations: list[str], path: str | None) -> None:
    """Download every skill in each ``<catalog>.<schema>`` location to disk.

    Bundles are fetched concurrently per schema, then written sequentially (so
    overwrite prompts don't interleave). A failure on one skill warns and skips
    it without aborting the batch.
    """
    roots = skill_dir_roots(path)
    for location in locations:
        catalog, schema = location.split(".")
        leaves, reason = list_schema_skills(workspace, token, catalog, schema)
        if reason:
            print_warning(f"Skipping `{location}`: {reason}.")
            continue
        if not leaves:
            print_note(f"No skills found in `{location}`.")
            continue

        print_note(f"Downloading {len(leaves)} skill(s) from `{location}`...")
        bundles = _fetch_bundles(workspace, token, catalog, schema, leaves)
        for leaf in leaves:
            files, reason = bundles[leaf]
            if reason or files is None:
                print_warning(f"Skipping `{location}.{leaf}`: {reason}.")
                continue
            write_skill(roots, leaf, files, location=location)
