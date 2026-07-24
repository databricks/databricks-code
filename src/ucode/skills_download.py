"""Write downloaded Unity Catalog skills to disk for coding agents to load.

Skills are written flat, one directory per skill (``<root>/<leaf>/SKILL.md`` plus
bundled files), into both `.claude/skills` and `.agents/skills` — the pair that
covers every skills-capable agent ucode configures. The bytes come from the
download client in `databricks.py`; this module owns only the filesystem side.
"""

from __future__ import annotations

import re
from pathlib import Path

from ucode.ui import print_warning, prompt_yes_no

# Cross-agent skill directories: `.claude/skills` (Claude) and `.agents/skills`
# (the Agent Skills alias the other agents read). Both get the same skills.
SKILL_DIR_NAMES = (".claude/skills", ".agents/skills")

# Agent Skills spec: a skill's directory name is its `name`, lowercase a-z 0-9 -.
_LEAF_PATTERN = re.compile(r"^[a-z0-9-]+$")


def skill_dir_roots(project_dir: str) -> list[Path]:
    """The ``.claude/skills`` and ``.agents/skills`` roots under ``project_dir``.

    ``project_dir`` must be an existing absolute directory.
    """
    base = Path(project_dir)
    if not base.is_absolute():
        raise ValueError(f"--path must be an absolute path, got `{project_dir}`.")
    if not base.is_dir():
        raise ValueError(f"--path directory does not exist: `{project_dir}`.")
    return [base / name for name in SKILL_DIR_NAMES]


def _is_valid_leaf(leaf: str) -> bool:
    return bool(_LEAF_PATTERN.match(leaf))


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


def write_skill(roots: list[Path], leaf: str, files: dict[str, bytes], *, schema_ref: str) -> str:
    """Write one skill's bundle into every root, prompting before overwriting.

    ``files`` maps each in-bundle relative path (including ``SKILL.md``) to its
    bytes. If the skill's directory already exists in any root, ask before
    overwriting it — the on-disk skill may be from a different schema, so we
    never clobber it without consent.

    Returns ``"written"``, ``"overwritten"``, ``"kept"``, or ``"skipped"``.
    """
    if not _is_valid_leaf(leaf):
        print_warning(f"Skipping `{leaf}`: not a valid skill name (lowercase a-z, 0-9, -).")
        return "skipped"

    already_on_disk = any((root / leaf).exists() for root in roots)
    if already_on_disk and not prompt_yes_no(
        f"A skill named `{leaf}` already exists. Overwrite it with `{schema_ref}.{leaf}`?"
    ):
        return "kept"

    for root in roots:
        _write_bundle(root / leaf, leaf, files)

    return "overwritten" if already_on_disk else "written"
