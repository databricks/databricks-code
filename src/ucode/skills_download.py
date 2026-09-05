"""Download Unity Catalog skills and write them to disk, one flat dir per skill."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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

SKILL_FILES_API_PREFIX = "Skills"

# Parallel skill fetches per schema; writes stay sequential (they prompt).
_MAX_FETCH_WORKERS = 8


# --- Download client (UC skills API + Files API) ---------------------------


@dataclass(frozen=True)
class SkillRef:
    """A downloadable skill's two names, which are not interchangeable.

    ``securable_name`` is the UC leaf of ``skills/<cat>.<sch>.<leaf>`` and is the
    only name the Files API resolves, so it addresses the bytes and identifies the
    skill. ``bundle_name`` is the ``name:`` an agent reads from the bundle's
    SKILL.md frontmatter, so it names the on-disk directory. Finalize does not
    require the two to match, so a skill created under a securable that differs
    from its frontmatter carries both.
    """

    securable_name: str
    bundle_name: str


def _non_empty_str(value: object) -> str | None:
    """``value`` when it is a non-empty string, else None."""
    return value if isinstance(value, str) and value else None


def _skill_ref(skill: dict) -> SkillRef | None:
    """A finalized skill's ``SkillRef``, or None if it cannot be downloaded.

    A skill without a ``finalize_time`` has no bundle content yet and is skipped
    quietly, since that is a normal in-progress state.

    A finalized skill is expected to carry both names: ``name`` is immutable from
    creation, and finalize is the sole writer of ``bundle_name``. One missing is
    therefore an anomaly, so warn and skip rather than substituting the other
    name -- the two are not interchangeable, and guessing a directory name that
    doesn't match the bundle's SKILL.md ``name:`` would hide the skill from the
    agent meant to load it.
    """
    if not skill.get("finalize_time"):
        return None

    name = _non_empty_str(skill.get("name"))
    bundle_name = _non_empty_str(skill.get("bundle_name"))
    if name is None or bundle_name is None:
        missing = " or ".join(
            field
            for field, value in (("name", name), ("bundle_name", bundle_name))
            if value is None
        )
        print_warning(
            f"Skipping `{name or '<unnamed skill>'}`: the skills API returned no {missing}."
        )
        return None

    return SkillRef(securable_name=name.rsplit(".", 1)[-1], bundle_name=bundle_name)


def list_schema_skills(
    workspace: str, token: str, catalog: str, schema: str
) -> tuple[list[SkillRef], str | None]:
    """List the finalized skills in ``<catalog>.<schema>``.

    A non-None reason indicates the listing call itself failed.
    """
    hostname = workspace_hostname(workspace)
    base_url = f"https://{hostname}/api/2.1/unity-catalog/skills"
    query = {"parent": f"schemas/{catalog}.{schema}"}

    refs: list[SkillRef] = []
    page_token: str | None = None
    while True:
        if page_token:
            query["page_token"] = page_token
        payload, reason = _http_get_json(f"{base_url}?{urlencode(query)}", token, timeout=30)
        if payload is None:
            return [], reason
        data = payload if isinstance(payload, dict) else {}
        for skill in data.get("skills") or []:
            ref = _skill_ref(skill) if isinstance(skill, dict) else None
            if ref:
                refs.append(ref)
        page_token = data.get("next_page_token")
        if not page_token:
            return refs, None


def list_skill_files(
    workspace: str, token: str, catalog: str, schema: str, securable: str
) -> tuple[list[str], str | None]:
    """List a skill bundle's files, as paths relative to the skill directory.

    Recursively walks the skill's Files API directory (including ``SKILL.md``).
    Takes the securable leaf, the only name the Files API resolves. A non-None
    reason indicates the listing call itself failed.
    """
    hostname = workspace_hostname(workspace)
    dirs_base = f"https://{hostname}/api/2.0/fs/directories"
    skill_prefix = f"/{SKILL_FILES_API_PREFIX}/{catalog}/{schema}/{securable}/"

    relative_paths: list[str] = []
    pending = [f"{SKILL_FILES_API_PREFIX}/{catalog}/{schema}/{securable}"]
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
                    relative_paths.append(path.removeprefix(skill_prefix))
            page_token = data.get("next_page_token")
            if not page_token:
                break
    return relative_paths, None


def fetch_skill_file(
    workspace: str, token: str, catalog: str, schema: str, securable: str, relative_path: str
) -> tuple[bytes | None, str | None]:
    """Fetch one skill bundle file's raw bytes from the Files API."""
    hostname = workspace_hostname(workspace)
    url = (
        f"https://{hostname}/api/2.0/fs/files/"
        f"{SKILL_FILES_API_PREFIX}/{catalog}/{schema}/{securable}/{relative_path}"
    )
    return _http_get_bytes(url, token, timeout=30)


def fetch_skill_bundle(
    workspace: str, token: str, catalog: str, schema: str, securable: str
) -> tuple[dict[str, bytes] | None, str | None]:
    """Fetch a whole skill bundle as ``{relative_path: bytes}``.

    Lists the skill's files then fetches each one. All-or-nothing: a non-None
    reason (and None bundle) means the listing or any file fetch failed, so a
    partially-downloaded skill is never written to disk.
    """
    relative_paths, reason = list_skill_files(workspace, token, catalog, schema, securable)
    if reason:
        return None, reason
    bundle: dict[str, bytes] = {}
    for relative_path in relative_paths:
        content, reason = fetch_skill_file(
            workspace, token, catalog, schema, securable, relative_path
        )
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


def _safe_relative_path(relative_path: str) -> Path | None:
    """A bundle file's path within its skill dir, or None if it escapes the dir.

    The Files API returns server-controlled paths, but ucode writes them to
    disk, so reject absolute paths and any ``..`` traversal.
    """
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def _write_bundle(skill_dir: Path, bundle_name: str, files: dict[str, bytes]) -> None:
    for relative_path, content in files.items():
        safe_path = _safe_relative_path(relative_path)
        if safe_path is None:
            print_warning(f"Skipping unsafe path in `{bundle_name}`: {relative_path}")
            continue
        destination = skill_dir / safe_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def existing_skill_on_disk(roots: list[Path], bundle_name: str) -> bool:
    """Whether ``bundle_name`` already has a skill directory under any root."""
    return any((root / bundle_name).exists() for root in roots)


def should_download_skill(roots: list[Path], ref: SkillRef, *, location: str) -> bool:
    """Whether ``ref`` should be fetched and written into ``roots``.

    Applies the disk-only check that needs no bundle bytes: prompts before
    overwriting a skill already on disk (``location`` is the source
    ``<catalog>.<schema>`` shown in that prompt), so a declined skill is never
    fetched. Dedup keys on the bundle name, since that is the directory an agent
    would load. Name validity is the server's job -- FinalizeSkill enforces the
    Agent Skills naming rules on ``bundle_name`` before we ever see it -- so
    ucode does not re-check it here.
    """
    if existing_skill_on_disk(roots, ref.bundle_name) and not prompt_yes_no(
        f"A skill named `{ref.bundle_name}` already exists. "
        f"Overwrite it with `{location}.{ref.securable_name}`?"
    ):
        print_note(f"Kept existing `{ref.bundle_name}`.")
        return False

    return True


def write_skill(roots: list[Path], ref: SkillRef, files: dict[str, bytes]) -> None:
    """Write ``ref``'s bundle (``{relpath: bytes}``) into every root.

    The directory is named for the bundle, so it matches the ``name:`` an agent
    reads from the written SKILL.md.
    """
    for root in roots:
        _write_bundle(root / ref.bundle_name, ref.bundle_name, files)


# --- Orchestration ---------------------------------------------------------


def _fetch_bundles(
    workspace: str, token: str, catalog: str, schema: str, refs: list[SkillRef]
) -> dict[str, tuple[dict[str, bytes] | None, str | None]]:
    """Fetch every skill's bundle concurrently, keyed by securable leaf.

    Renders a ``k/n`` progress bar that advances as each fetch completes.
    """
    if not refs:
        return {}
    results: dict[str, tuple[dict[str, bytes] | None, str | None]] = {}
    with (
        progress_bar(f"Fetching skills from {catalog}.{schema}", len(refs)) as advance,
        ThreadPoolExecutor(max_workers=min(_MAX_FETCH_WORKERS, len(refs))) as pool,
    ):
        futures = {
            pool.submit(
                fetch_skill_bundle, workspace, token, catalog, schema, ref.securable_name
            ): ref.securable_name
            for ref in refs
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            advance()
    return results


def _reject_bundle_name_collisions(refs: list[SkillRef], *, location: str) -> list[SkillRef]:
    """``refs`` with any later skill that repeats an earlier one's bundle name dropped.

    Only the securable name is unique within a schema; ``bundle_name`` comes from
    each bundle's SKILL.md frontmatter and is never checked against its siblings,
    so one schema can hold two skills claiming the same directory. Writing both
    would land them on top of each other, leaving whichever finished last with no
    sign the other was lost, so keep the first and warn about the rest.
    """
    kept: list[SkillRef] = []
    claimed: dict[str, str] = {}
    for ref in refs:
        winner = claimed.get(ref.bundle_name)
        if winner is not None:
            print_warning(
                f"Skipping `{location}.{ref.securable_name}`: its bundle name "
                f"`{ref.bundle_name}` is already claimed by `{location}.{winner}`. "
                "Rename one skill's SKILL.md `name:` to download both."
            )
            continue
        claimed[ref.bundle_name] = ref.securable_name
        kept.append(ref)
    return kept


def download_skills(
    workspace: str,
    token: str,
    locations: list[str],
    path: str | None,
    skills: set[str] | None = None,
) -> None:
    """Download every skill in each ``<catalog>.<schema>`` location to disk.

    Locations are processed one at a time, and each runs three stages:

    1. **List** the schema's finalized skills. When ``skills`` is given, restrict
       to those securable names (the name that identifies a skill in UC); names
       absent from the schema warn and are skipped, and ``None`` keeps the whole
       schema. Siblings claiming one directory are then reduced to the first (see
       ``_reject_bundle_name_collisions``).
    2. **Decide** which to download via ``should_download_skill`` (prompts before
       overwriting a skill already on disk), so a declined skill is never fetched.
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
        refs, reason = list_schema_skills(workspace, token, catalog, schema)
        if reason:
            print_warning(f"Skipping `{location}`: {reason}.")
            continue
        if skills is not None:
            unknown = skills - {ref.securable_name for ref in refs}
            if unknown:
                print_warning(
                    f"Skipping requested skill(s) not found in `{location}`: "
                    f"{', '.join(sorted(unknown))}."
                )
            refs = [ref for ref in refs if ref.securable_name in skills]
            if not refs:
                print_note(f"No requested skills to download from `{location}`.")
                continue
        if not refs:
            print_note(f"No skills found in `{location}`.")
            continue
        # Before the decide stage, so a dropped sibling is never fetched and the
        # summary's denominator counts only skills that can reach disk.
        refs = _reject_bundle_name_collisions(refs, location=location)

        to_download = [ref for ref in refs if should_download_skill(roots, ref, location=location)]
        bundles = _fetch_bundles(workspace, token, catalog, schema, to_download)
        written = 0
        for ref in to_download:
            files, reason = bundles[ref.securable_name]
            if reason or files is None:
                print_warning(f"Skipping `{location}.{ref.securable_name}`: {reason}.")
                continue
            write_skill(roots, ref, files)
            written += 1
        console.print()
        total = len(refs)
        skipped = f"; {total - written} skipped" if written < total else ""
        print_success(
            f"Downloaded {written}/{total} skill(s){skipped} from `{location}` in {roots_display}."
        )


def download_managed_skills_on_launch(
    workspace: str, token: str, locations: list[str], path: str | None = None
) -> list[str]:
    """Download admin-published skills to disk so the agent's ``/skills`` lists them.

    Runs on the managed launch path: the config only registers the skills MCP
    connection, so nothing else writes the bundles that ``/skills`` reads. Writes
    only skills not already on disk -- no overwrite prompt, so the launch never
    blocks on input and a developer's own same-named skill is never clobbered.
    Best-effort and never raises, so it can't block the launch. Returns the bundle
    names newly written.
    """
    roots = skill_dir_roots(path)
    written: list[str] = []
    for location in locations:
        if location.count(".") != 1:
            continue
        catalog, schema = location.split(".")
        refs, reason = list_schema_skills(workspace, token, catalog, schema)
        if reason:
            print_warning(f"Could not list workspace skills in `{location}`: {reason}.")
            continue
        refs = _reject_bundle_name_collisions(refs, location=location)
        missing = [ref for ref in refs if not existing_skill_on_disk(roots, ref.bundle_name)]
        if not missing:
            continue
        bundles = _fetch_bundles(workspace, token, catalog, schema, missing)
        for ref in missing:
            files, reason = bundles[ref.securable_name]
            if reason or files is None:
                print_warning(f"Skipping `{location}.{ref.securable_name}`: {reason}.")
                continue
            write_skill(roots, ref, files)
            written.append(ref.bundle_name)
    return written


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
    auth_kwargs = (
        {"oauth_client_id": state["oauth_client_id"]} if state.get("oauth_client_id") else {}
    )
    token = get_databricks_token(workspace, profile, **auth_kwargs)

    download_skills(workspace, token, locations, path, skills)

    register_schemaless_skills_connection(state, workspace, profile, clients)
    return 0
