"""Download Unity Catalog skills and write them to disk, one flat dir per skill."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode

from ucode.databricks import (
    _http_get_bytes,
    _http_get_json,
    get_databricks_token,
    workspace_hostname,
)
from ucode.mcp import register_schemaless_skills_connection, setup_mcp_clients
from ucode.state import load_state
from ucode.ui import (
    console,
    print_note,
    print_success,
    print_warning,
    progress_bar,
    prompt_yes_no,
)

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
        data = payload if isinstance(payload, dict) else {}
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
            data = payload if isinstance(payload, dict) else {}
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


def existing_skill_on_disk(roots: list[Path], leaf: str) -> bool:
    """Whether ``leaf`` already has a skill directory under any root."""
    return any((root / leaf).exists() for root in roots)


def should_download_skill(roots: list[Path], leaf: str, *, location: str) -> bool:
    """Whether ``leaf`` should be fetched and written into ``roots``.

    Applies the disk-only checks that need no bundle bytes, so a declined or
    invalid skill is never downloaded: skips invalid leaf names, and prompts
    before overwriting a skill already on disk (``location`` is the source
    ``<catalog>.<schema>`` shown in that prompt).
    """
    if not _is_valid_leaf(leaf):
        print_warning(f"Skipping `{leaf}`: not a valid skill name (lowercase a-z, 0-9, -).")
        return False

    if existing_skill_on_disk(roots, leaf) and not prompt_yes_no(
        f"A skill named `{leaf}` already exists. Overwrite it with `{location}.{leaf}`?"
    ):
        print_note(f"Kept existing `{leaf}`.")
        return False

    return True


def write_skill(roots: list[Path], leaf: str, files: dict[str, bytes]) -> None:
    """Write ``leaf``'s bundle (``{relpath: bytes}``) into every root."""
    for root in roots:
        _write_bundle(root / leaf, leaf, files)


# --- Orchestration ---------------------------------------------------------


def _fetch_bundles(
    workspace: str, token: str, catalog: str, schema: str, leaves: list[str]
) -> dict[str, tuple[dict[str, bytes] | None, str | None]]:
    """Fetch every leaf's bundle concurrently, keyed by leaf name.

    Renders a ``k/n`` progress bar that advances as each fetch completes.
    """
    if not leaves:
        return {}
    results: dict[str, tuple[dict[str, bytes] | None, str | None]] = {}
    with (
        progress_bar(f"Fetching skills from {catalog}.{schema}", len(leaves)) as advance,
        ThreadPoolExecutor(max_workers=min(_MAX_FETCH_WORKERS, len(leaves))) as pool,
    ):
        futures = {
            pool.submit(fetch_skill_bundle, workspace, token, catalog, schema, leaf): leaf
            for leaf in leaves
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            advance()
    return results


def download_skills(
    workspace: str,
    token: str,
    locations: list[str],
    path: str | None,
    skills: set[str] | None = None,
) -> None:
    """Download every skill in each ``<catalog>.<schema>`` location to disk.

    Locations are processed one at a time, and each runs three stages:

    1. **List** the schema's skill leaves. When ``skills`` is given, restrict to
       those leaf names; names absent from the schema warn and are skipped, and
       ``None`` keeps the whole schema.
    2. **Decide** which to download via ``should_download_skill`` (skips invalid
       names and prompts before overwriting a skill already on disk), so a
       declined skill is never fetched.
    3. **Fetch** the survivors' bundles concurrently (with a progress bar) and
       **write** them.

    Finishing one location before starting the next means a skill written for an
    earlier location is already on disk when a same-named skill in a later
    location reaches its decide stage, so the overwrite prompt still fires. A
    failure on one skill warns and skips it without aborting the batch.
    """
    roots = skill_dir_roots(path)
    roots_display = " and ".join(str(root) for root in roots)
    for location in locations:
        catalog, schema = location.split(".")
        leaves, reason = list_schema_skills(workspace, token, catalog, schema)
        if reason:
            print_warning(f"Skipping `{location}`: {reason}.")
            continue
        if skills is not None:
            unknown = skills - set(leaves)
            if unknown:
                print_warning(
                    f"Skipping requested skill(s) not found in `{location}`: "
                    f"{', '.join(sorted(unknown))}."
                )
            leaves = [leaf for leaf in leaves if leaf in skills]
            if not leaves:
                print_note(f"No requested skills to download from `{location}`.")
                continue
        if not leaves:
            print_note(f"No skills found in `{location}`.")
            continue

        to_download = [
            leaf for leaf in leaves if should_download_skill(roots, leaf, location=location)
        ]
        bundles = _fetch_bundles(workspace, token, catalog, schema, to_download)
        written = 0
        for leaf in to_download:
            files, reason = bundles[leaf]
            if reason or files is None:
                print_warning(f"Skipping `{location}.{leaf}`: {reason}.")
                continue
            write_skill(roots, leaf, files)
            written += 1
        console.print()
        print_success(
            f"Downloaded {written}/{len(leaves)} skill(s) from `{location}` in {roots_display}."
        )


def configure_skills_download_command(
    locations: list[str], *, path: str | None, skills: set[str] | None = None
) -> int:
    """Download every skill in each schema to disk and register the skills connection.

    Downloads to ``path`` (or the home dir when None), then registers/keeps the
    schema-less MCP connection. ``skill_locations`` is never touched, so a prior
    ``--mcp`` set survives a download run. ``skills`` narrows the download (see
    ``download_skills``)."""
    state = load_state()
    workspace, profile, clients = setup_mcp_clients(state, "Skills")
    token = get_databricks_token(workspace, profile)

    download_skills(workspace, token, locations, path, skills)

    register_schemaless_skills_connection(state, workspace, profile, clients)
    return 0
