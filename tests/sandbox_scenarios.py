"""Scenario table for the `ucode configure` sandbox.

Each scenario runs in its own subprocess (fresh HOME, fresh module state, fresh caches).
`tests/test_configure_sandbox.py` drives the whole table and holds the expectations; to look
at one scenario by hand, from the repo root:

    python -m tests.sandbox_scenarios <name>            # print its JSON verdict
    SANDBOX_DISCOVER=1 python -m tests.sandbox_scenarios <name>   # auto-answer new prompts

A scenario declares the fake workspace, plus one or more CLI steps with their scripted prompt
answers. It is judged on the real on-disk state and the real printed output.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

from tests import sandbox_harness as harness
from tests import sandbox_workspace as gw

ON = {"ENABLE_MANAGED_AGENT_CONFIG": "1"}

SCENARIOS: dict[str, dict] = {}

CONFIGURE_SUBCOMMANDS = {"tracing", "spend-tiers"}


def scenario(name, *, persona, desc, ws, steps=None, os_managed=None, seed=None, **single):
    """Register a scenario. `steps` for multi-command flows, else argv/env/prompts inline."""
    if steps is None:
        steps = [single]
    SCENARIOS[name] = {
        "persona": persona,
        "desc": desc,
        "ws": ws,
        "steps": steps,
        "os_managed": os_managed,
        "seed": seed,
    }


AUTHOR_CLAUDE = [
    ("multi", ["claude"]),
    ("select", "system.ai.claude-opus-4-8"),
    ("select", "system.ai.claude-sonnet-4-6"),
    ("select", "system.ai.claude-haiku-4-5"),
    ("select", "system.ai.claude-opus-4-8"),
]

# Same picks, but an overall default that differs from what the workspace publishes, so a flow
# that reads the wrong slot produces visibly different output.
AUTHOR_CLAUDE_SONNET_DEFAULT = [*AUTHOR_CLAUDE[:-1], ("select", "system.ai.claude-sonnet-4-6")]


scenario(
    "gate/flag-off",
    persona="either",
    desc="ENABLE_MANAGED_AGENT_CONFIG unset: never reads the workspace config at all",
    ws={"admin": True, "published": "PUBLISHED"},
    env={},
    argv=["configure", "--agents", "claude", "--skip-validate"],
    prompts=[],
)

scenario(
    "gate/feature-disabled",
    persona="either",
    desc="backend feature off: local configure runs, no publish advice",
    ws={"admin": True, "published": None, "feature_disabled": True},
    env=ON,
    argv=["configure", "--agents", "claude", "--skip-validate"],
    prompts=[],
)

scenario(
    "gate/not-found-is-no-config",
    persona="either",
    desc="a NOT_FOUND read is 'no config defined', not a failure: local configure runs quietly",
    ws={"admin": False, "published": None, "config_read_error": "HTTP 404 Not Found: NOT_FOUND"},
    env=ON,
    argv=["configure", "--agents", "claude", "--skip-validate"],
    prompts=[],
)

scenario(
    "gate/read-error-no-cache",
    persona="either",
    desc="config read fails, nothing cached: warns and configures locally",
    ws={
        "admin": False,
        "published": "PUBLISHED",
        "config_read_error": "HTTP 503 Service Unavailable",
    },
    env=ON,
    argv=["configure", "--agents", "claude", "--skip-validate"],
    prompts=[],
)

scenario(
    "gate/read-error-with-cache",
    persona="non-admin",
    desc="config read fails but a published slot is cached: warns, applies the cached copy",
    ws={
        "admin": False,
        "published": "PUBLISHED",
        "config_read_error": "HTTP 503 Service Unavailable",
    },
    env=ON,
    seed="published",
    argv=["configure"],
    prompts=[],
)


scenario(
    "published/non-admin",
    persona="non-admin",
    desc="developer with a published config: applies it and exits, never offered authoring",
    ws={"admin": False, "published": "PUBLISHED"},
    env=ON,
    argv=["configure"],
    prompts=[],
)

scenario(
    "published/admin-declines",
    persona="admin",
    desc="admin declines the update offer: published applied, no draft written",
    ws={"admin": True, "published": "PUBLISHED"},
    env=ON,
    argv=["configure"],
    prompts=[("yes_no", False)],
)

scenario(
    "published/admin-accepts",
    persona="admin",
    desc="admin accepts: authors a draft, is advised to publish, server untouched",
    ws={"admin": True, "published": "PUBLISHED"},
    env=ON,
    argv=["configure"],
    prompts=[("yes_no", True), *AUTHOR_CLAUDE],
)

scenario(
    "published/admin-unknown",
    persona="admin?",
    desc="SCIM check inconclusive: applies published, must not offer authoring",
    ws={"admin": None, "published": "PUBLISHED"},
    env=ON,
    argv=["configure"],
    prompts=[],
)

scenario(
    "published/admin-non-interactive",
    persona="admin",
    desc="admin passing --agents: applies published, points at a flagless re-run",
    ws={"admin": True, "published": "PUBLISHED"},
    env=ON,
    argv=["configure", "--agents", "claude", "--skip-validate"],
    prompts=[],
)

scenario(
    "published/os-managed-conflict",
    persona="non-admin",
    desc="OS-managed Claude settings block the apply: warns, yet still claims completion",
    ws={"admin": False, "published": "PUBLISHED"},
    env=ON,
    os_managed={
        "claude": json.dumps(
            {
                "apiKeyHelper": "/opt/corp/key.sh",
                "env": {"ANTHROPIC_BASE_URL": "https://corp.example.com"},
            }
        )
    },
    argv=["configure"],
    prompts=[],
)


scenario(
    "empty/non-admin",
    persona="non-admin",
    desc="developer, no workspace config: falls through to plain local configure",
    ws={"admin": False, "published": None},
    env=ON,
    argv=["configure", "--agents", "claude", "--skip-validate"],
    prompts=[],
)

scenario(
    "empty/admin-interactive",
    persona="admin",
    desc="admin authors from scratch: draft saved and applied, publish advised, server untouched",
    ws={"admin": True, "published": None},
    env=ON,
    argv=["configure"],
    prompts=[*AUTHOR_CLAUDE],
)

scenario(
    "empty/admin-non-interactive",
    persona="admin",
    desc="admin passing --agents with no workspace config: local configure, no authoring",
    ws={"admin": True, "published": None},
    env=ON,
    argv=["configure", "--agents", "claude", "--skip-validate"],
    prompts=[],
)

scenario(
    "empty/admin-unknown",
    persona="admin?",
    desc="SCIM inconclusive, no workspace config: falls through to local configure",
    ws={"admin": None, "published": None},
    env=ON,
    argv=["configure", "--agents", "claude", "--skip-validate"],
    prompts=[],
)


scenario(
    "draft/survives-a-launch",
    persona="admin",
    desc="author a draft, an admin elsewhere republishes, then launch: the launch refresh must "
    "overwrite the published slot and leave the draft alone",
    ws={"admin": True, "published": "PUBLISHED"},
    env=ON,
    steps=[
        {"argv": ["configure"], "env": ON, "prompts": [("yes_no", True), *AUTHOR_CLAUDE]},
        {"argv": ["claude"], "env": ON, "prompts": [], "server_default": "CODING_AGENT_CODEX"},
        {"argv": ["status"], "env": ON, "prompts": []},
    ],
)

scenario(
    "draft/publish-promotes",
    persona="admin",
    desc="author a draft then `ucode publish`: draft reaches the server, published slot updates",
    ws={"admin": True, "published": None},
    env=ON,
    steps=[
        {"argv": ["configure"], "env": ON, "prompts": [*AUTHOR_CLAUDE]},
        {"argv": ["publish"], "env": ON, "prompts": [("yes_no", True)]},
    ],
)

scenario(
    "draft/publish-declined",
    persona="admin",
    desc="`ucode publish` declined at the diff: server untouched, draft still on disk",
    ws={"admin": True, "published": None},
    env=ON,
    steps=[
        {"argv": ["configure"], "env": ON, "prompts": [*AUTHOR_CLAUDE]},
        {"argv": ["publish"], "env": ON, "prompts": [("yes_no", False)]},
    ],
)

scenario(
    "draft/non-admin-publish-refused",
    persona="non-admin",
    desc="a developer running `ucode publish` is refused before any write",
    ws={"admin": False, "published": None},
    env=ON,
    seed="draft",
    argv=["publish"],
    prompts=[],
)

scenario(
    "draft/export-reads-draft",
    persona="admin",
    desc="`ucode export` emits the authored draft, not the published copy",
    ws={"admin": True, "published": "PUBLISHED"},
    env=ON,
    steps=[
        {
            "argv": ["configure"],
            "env": ON,
            "prompts": [("yes_no", True), *AUTHOR_CLAUDE_SONNET_DEFAULT],
        },
        {"argv": ["export"], "env": ON, "prompts": []},
    ],
)

scenario(
    "draft/legacy-v1-migration",
    persona="either",
    desc="a pre-v2 managed-state.json migrates into the published slot, keeping a .pre-v2.bak",
    ws={"admin": False, "published": "PUBLISHED"},
    env=ON,
    seed="legacy-v1",
    argv=["configure"],
    prompts=[],
)

scenario(
    "draft/legacy-v1-was-a-draft",
    persona="admin",
    desc="the lossy migration case: a pre-v2 slot holding an unpublished draft lands in published, "
    "so `ucode publish` sees nothing authored and .pre-v2.bak is the only copy left",
    ws={"admin": True, "published": None},
    env=ON,
    seed="legacy-v1",
    argv=["publish"],
    prompts=[],
)

scenario(
    "draft/legacy-v1-serves-as-fallback",
    persona="non-admin",
    desc="a pre-v2 file plus a failing read: the migrated published slot is what gets applied",
    ws={
        "admin": False,
        "published": "PUBLISHED",
        "config_read_error": "HTTP 503 Service Unavailable",
    },
    env=ON,
    seed="legacy-v1",
    argv=["configure"],
    prompts=[],
)


scenario(
    "section/non-admin-refused",
    persona="non-admin",
    desc="`ucode configure spend-tiers` refuses a non-admin",
    ws={"admin": False, "published": None},
    env=ON,
    seed="draft",
    argv=["configure", "spend-tiers"],
    prompts=[],
)

scenario(
    "section/tracing-is-local",
    persona="non-admin",
    desc="`ucode configure tracing` is a per-machine setting, not a managed-config section: a "
    "non-admin can run it, unlike `configure spend-tiers`",
    ws={"admin": False, "published": None},
    env=ON,
    seed="draft",
    steps=[
        {"argv": ["configure", "--agents", "claude", "--skip-validate"], "env": ON, "prompts": []},
        {"argv": ["configure", "tracing"], "env": ON, "prompts": []},
    ],
)

scenario(
    "multi/two-workspaces",
    persona="admin",
    desc="--workspaces with two entries skips the managed flow entirely: a managed config is "
    "per-workspace, so both workspaces are configured locally instead",
    ws={"admin": True, "published": "PUBLISHED"},
    env=ON,
    argv=["configure", "--workspaces", f"{gw.WORKSPACE},https://other.cloud.databricks.com"],
    prompts=[("multi", ["claude"])],
)

scenario(
    "section/no-authored-config",
    persona="admin",
    desc="`ucode configure spend-tiers` with nothing authored points back at `ucode configure`",
    ws={"admin": True, "published": None},
    env=ON,
    argv=["configure", "spend-tiers"],
    prompts=[],
)


def _seed_disk(kind, home):
    """Pre-seed ~/.ucode/managed-state.json before the CLI runs.

    Returns the seeded bytes for the pre-v2 file, whose backup must survive the migration
    unchanged; None for the seeds ucode is free to rewrite.
    """
    from ucode.managed_config import save_draft_config, save_published_config

    if kind == "published":
        save_published_config(gw.WORKSPACE, _normalized())
    elif kind == "draft":
        save_draft_config(gw.WORKSPACE, _normalized())
    elif kind == "legacy-v1":
        path = home / ".ucode" / "managed-state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"workspace": gw.WORKSPACE, "config": _normalized()}, indent=2) + "\n",
            encoding="utf-8",
        )
        return path.read_bytes()
    return None


def _legacy_backup_verdict(home, seeded):
    """How the .pre-v2.bak compares to the file that was on disk before the migration."""
    if seeded is None:
        return None
    backup = home / ".ucode" / "managed-state.json.pre-v2.bak"
    if not backup.exists():
        return "absent"
    return "byte-identical" if backup.read_bytes() == seeded else "rewritten"


def _normalized():
    from ucode.managed_config import normalize_managed_config

    return normalize_managed_config(gw.PUBLISHED_MANIFEST)


def _prompts_for_workspace(argv):
    """True when the flow will ask which workspace to configure.

    `_run_managed_configure_flow` prompts whenever neither --workspaces nor --profiles was
    passed, even though state.json already records one. Only bare `ucode configure` reaches
    it: the `configure <section>` subcommands resolve the workspace from state themselves.
    """
    if argv[0] != "configure":
        return False
    if any(token in argv for token in ("--workspaces", "--profiles")):
        return False
    return not any(token in CONFIGURE_SUBCOMMANDS for token in argv[1:])


def _resolve_ws(spec_ws):
    kwargs = dict(spec_ws)
    if kwargs.get("published") == "PUBLISHED":
        kwargs["published"] = gw.PUBLISHED_MANIFEST
    return gw.FakeWorkspace(**kwargs)


def run_one(name: str) -> dict:
    spec = SCENARIOS[name]

    home = harness.make_home()

    ws = _resolve_ws(spec["ws"])

    import ucode.ui  # noqa: F401  (imported before patching so cli binds the patched names)

    gw.install(ws)
    gw.guard_privileged_writes(home, os_managed=spec.get("os_managed"))

    from ucode.state import save_state

    save_state({"workspace": gw.WORKSPACE, "profile": "DEFAULT"})
    seeded_legacy = _seed_disk(spec["seed"], home) if spec.get("seed") else None

    step_results = []
    installed = False
    for step in spec["steps"]:
        argv = step["argv"]
        if step.get("server_default"):
            ws.published["default_agent"] = step["server_default"]
        answers = list(step.get("prompts") or [])
        if _prompts_for_workspace(argv):
            answers.insert(0, ("workspace", (gw.WORKSPACE, "DEFAULT")))
        script = harness.PromptScript(answers, discover=bool(os.environ.get("SANDBOX_DISCOVER")))
        if installed:
            harness.set_script(script)
        else:
            harness.install_prompts(script)
            installed = True
        code, output = harness.run_cli(argv, env=step.get("env") or {})
        step_results.append(
            {
                "argv": argv,
                "exit_code": code,
                "output": output,
                "prompts_asked": script.asked,
                "prompts_left": script.pending,
                "draft_after": harness.draft_slot(home, gw.WORKSPACE),
                "published_after": harness.published_slot(home, gw.WORKSPACE),
            }
        )

    return {
        "name": name,
        "persona": spec["persona"],
        "desc": spec["desc"],
        "steps": step_results,
        "home": str(home),
        "home_tree": harness.home_tree(home),
        "managed_state": harness.managed_state(home),
        "state_shape": harness.state_shape(home),
        "legacy_backup": _legacy_backup_verdict(home, seeded_legacy),
        "gateway": ws.report(),
    }


if __name__ == "__main__":
    target = sys.argv[1]
    try:
        result = run_one(target)
    except Exception:
        result = {"name": target, "harness_error": traceback.format_exc()}
    print("---SANDBOX-JSON---")
    print(json.dumps(result, indent=2, default=str))
