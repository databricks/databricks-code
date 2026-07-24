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


def skill_dir_roots(path: str | None) -> list[Path]:
    """Skill directory roots for a download: user scope (``~/``) or ``path``."""
    base = Path(path).expanduser().resolve() if path else Path.home()
    return [base / name for name in SKILL_DIR_NAMES]


def _is_valid_leaf(leaf: str) -> bool:
    return bool(_LEAF_PATTERN.match(leaf))


def _safe_relative_path(relative_path: str) -> Path | None:
    """A file's path within its skill dir, or None if it escapes the dir.

    The Files API returns server-controlled paths, but ucode writes them to
    disk, so reject absolute paths and any ``..`` traversal.
    """
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def write_skill(
    roots: list[Path],
    leaf: str,
    files: dict[str, bytes],
    *,
    assume_yes: bool,
    written: dict[str, str],
    schema_ref: str,
) -> str:
    """Write one skill's files into every root, resolving directory collisions.

    ``files`` maps each in-bundle relative path (including ``SKILL.md``) to its
    bytes. ``written`` tracks leaves already written this run (``leaf ->
    schema_ref``) so re-downloading the same skill is silent while a different
    schema colliding on the same leaf prompts to keep or overwrite. ``--yes``
    (``assume_yes``) overwrites without prompting.

    Returns ``"written"``, ``"overwritten"``, ``"kept"``, or ``"skipped"``.
    """
    if not _is_valid_leaf(leaf):
        print_warning(f"Skipping `{leaf}`: not a valid skill name (lowercase a-z, 0-9, -).")
        return "skipped"

    if leaf in written:
        status = "overwritten"  # same skill re-downloaded this run — rewrite silently
    elif any((root / leaf).exists() for root in roots):
        if not assume_yes and not prompt_yes_no(
            f"A skill named `{leaf}` already exists. Overwrite it with `{schema_ref}.{leaf}`?"
        ):
            return "kept"
        status = "overwritten"
    else:
        status = "written"

    for root in roots:
        skill_dir = root / leaf
        for relative_path, content in files.items():
            safe_path = _safe_relative_path(relative_path)
            if safe_path is None:
                print_warning(f"Skipping unsafe path in `{leaf}`: {relative_path}")
                continue
            destination = skill_dir / safe_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)

    written[leaf] = schema_ref
    return status
