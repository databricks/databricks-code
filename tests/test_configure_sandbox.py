"""End-to-end expectations for every `ucode configure` path, admin and non-admin.

Each scenario in `tests/sandbox_scenarios.py` runs the real CLI in its own subprocess against
a fake workspace inside a throwaway HOME (see `tests/sandbox_harness.py`). Only the HTTP layer,
the subprocess seams and the questionary prompts are faked, so what is asserted here is the
real on-disk state, the real printed output and the real set of server writes.

This is the regression net for the draft/published split: `draft/survives-a-launch` is the
scenario that fails if the two slots are collapsed back into one.

Expectations live in EXPECTATIONS below rather than in the scenario table, so the table stays a
plain description of what is being run. Every scenario must have an entry.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.sandbox_scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parent.parent

ABSENT = "<absent>"
EMPTY = "<empty>"
SANDBOX_MODEL = "system.ai.claude-opus-4-8"
SANDBOX_SONNET_MODEL = "system.ai.claude-sonnet-4-6"
SANDBOX_TRACING_TABLE = "main.default.ucode_traces"


def _claude_slot(
    *, default_agent="claude", tracing_table=SANDBOX_TRACING_TABLE, model=SANDBOX_MODEL
) -> dict:
    """The `_slot` fingerprint of the sandbox workspace's single-agent config."""
    return {
        "agents": ["claude"],
        "default_agent": default_agent,
        "models": {"claude": model},
        "tracing_table": tracing_table,
        "budget_policy": None,
    }


CLAUDE = _claude_slot()
CLAUDE_CODEX_DEFAULT = _claude_slot(default_agent="codex")
# Authored with a different overall default from the published config, so the two slots are
# distinguishable in output that reads only one of them.
CLAUDE_SONNET_DEFAULT = _claude_slot(model=SANDBOX_SONNET_MODEL)
# Authored with no published config to carry a tracing table forward from.
CLAUDE_UNTRACED = _claude_slot(tracing_table=None)

APPLIED_PUBLISHED = (
    "Configuration complete — this machine now uses your workspace's managed config."
)
APPLIED_PUBLISHED_NOT_FINAL = "This machine now uses your workspace's managed config."
APPLIED_AUTHORED = "Configuration complete — this machine now uses your authored config."
AUTHORING_STARTED = "step 1 of 3"
DRAFT_SAVED = "Draft saved to ~/.ucode/managed-state.json"
PUBLISH_ADVICE = "Publish with ucode publish"
NOT_ADMIN = "You are not an admin of"
LOCAL_OPTIONS_NOTE = (
    "To update the workspace configuration, re-run `ucode configure` without local "
    "configuration options."
)

EXPECTATIONS: dict[str, dict] = {
    "gate/flag-off": {
        "steps": [
            {
                "says": ["Configuration Complete", "Claude Code: configured"],
                "silent": ["Applying managed config", AUTHORING_STARTED],
                "draft": ABSENT,
                "published": ABSENT,
            }
        ],
        "state": "absent",
        "forbid_calls": ["coding-agent-config"],
    },
    "gate/feature-disabled": {
        "steps": [
            {
                "says": ["Configuration Complete"],
                "silent": ["Applying managed config", AUTHORING_STARTED, "ucode publish"],
                "draft": ABSENT,
                "published": ABSENT,
            }
        ],
        "state": "absent",
    },
    "gate/not-found-is-no-config": {
        "steps": [
            {
                "says": ["Configuration Complete"],
                "silent": ["Could not read", "Applying managed config"],
                "draft": ABSENT,
                "published": EMPTY,
            }
        ],
        "state": "v2",
    },
    "gate/read-error-no-cache": {
        "steps": [
            {
                "says": [
                    "Could not read your workspace's managed config (HTTP 503 Service Unavailable)",
                    "configuring your own settings for now",
                    "Configuration Complete",
                ],
                "draft": ABSENT,
                "published": ABSENT,
            }
        ],
        "state": "absent",
    },
    "gate/read-error-with-cache": {
        "steps": [
            {
                "says": [
                    "applying the last one saved for this workspace",
                    APPLIED_PUBLISHED,
                ],
                "draft": ABSENT,
                "published": CLAUDE,
            }
        ],
        "state": "v2",
    },
    "published/non-admin": {
        "steps": [
            {
                "says": [
                    "Applying managed config",
                    "Claude Code: default model -> system.ai.claude-opus-4-8",
                    APPLIED_PUBLISHED,
                ],
                "silent": [AUTHORING_STARTED, "ucode publish"],
                "draft": ABSENT,
                "published": CLAUDE,
            }
        ],
    },
    "published/admin-declines": {
        "steps": [
            {
                "says": [APPLIED_PUBLISHED_NOT_FINAL],
                "silent": ["Configuration complete", AUTHORING_STARTED],
                "draft": ABSENT,
                "published": CLAUDE,
            }
        ],
    },
    "published/admin-accepts": {
        "steps": [
            {
                "says": [
                    APPLIED_PUBLISHED_NOT_FINAL,
                    AUTHORING_STARTED,
                    DRAFT_SAVED,
                    APPLIED_AUTHORED,
                    PUBLISH_ADVICE,
                ],
                "order": [APPLIED_AUTHORED, PUBLISH_ADVICE],
                "draft": CLAUDE,
                "published": CLAUDE,
            }
        ],
        "writes": 0,
    },
    "published/admin-unknown": {
        "steps": [
            {
                "says": [APPLIED_PUBLISHED],
                "silent": [AUTHORING_STARTED],
                "draft": ABSENT,
                "published": CLAUDE,
            }
        ],
    },
    "published/admin-non-interactive": {
        "steps": [
            {
                "says": [APPLIED_PUBLISHED, LOCAL_OPTIONS_NOTE],
                "silent": [AUTHORING_STARTED],
                "draft": ABSENT,
                "published": CLAUDE,
            }
        ],
    },
    "published/os-managed-conflict": {
        "steps": [
            {
                "says": [
                    "Could not apply the managed config for Claude Code",
                    "override ucode values: apiKeyHelper",
                    "Could not apply your workspace's managed config to this machine",
                ],
                "silent": ["Configuration complete"],
                "draft": ABSENT,
                "published": CLAUDE,
            }
        ],
        "files": ["os-managed/claude-managed-settings.json"],
    },
    "empty/non-admin": {
        "steps": [
            {
                "says": ["Configuration Complete"],
                "silent": [AUTHORING_STARTED, "ucode publish"],
                "draft": ABSENT,
                "published": EMPTY,
            }
        ],
        "state": "v2",
    },
    "empty/admin-interactive": {
        "steps": [
            {
                "says": [AUTHORING_STARTED, DRAFT_SAVED, APPLIED_AUTHORED, PUBLISH_ADVICE],
                "order": [DRAFT_SAVED, APPLIED_AUTHORED, PUBLISH_ADVICE],
                "draft": CLAUDE_UNTRACED,
                "published": EMPTY,
            }
        ],
        "writes": 0,
    },
    "empty/admin-non-interactive": {
        "steps": [
            {
                "says": [
                    "Using the requested local configuration options",
                    "re-run `ucode configure` without them",
                ],
                "silent": [AUTHORING_STARTED],
                "draft": ABSENT,
                "published": EMPTY,
            }
        ],
    },
    "empty/admin-unknown": {
        "steps": [
            {
                "silent": [AUTHORING_STARTED, "Using the requested local configuration options"],
                "draft": ABSENT,
                "published": EMPTY,
            }
        ],
    },
    "draft/survives-a-launch": {
        "steps": [
            {"says": [APPLIED_AUTHORED], "draft": CLAUDE, "published": CLAUDE},
            {
                "says": ["Applied your workspace's managed coding agent config", "Launching"],
                "draft": CLAUDE,
                "published": CLAUDE_CODEX_DEFAULT,
            },
            {
                "says": ["Workspace-managed config"],
                "draft": CLAUDE,
                "published": CLAUDE_CODEX_DEFAULT,
            },
        ],
        "writes": 0,
    },
    "draft/publish-promotes": {
        "steps": [
            {"says": [DRAFT_SAVED], "draft": CLAUDE_UNTRACED, "published": EMPTY},
            {
                "says": [
                    "Admin permissions verified",
                    "This will create a new managed config",
                    "Published coding-agent-configs/sandbox-1",
                ],
                "draft": CLAUDE_UNTRACED,
                "published": CLAUDE_UNTRACED,
            },
        ],
        "writes": 1,
    },
    "draft/publish-declined": {
        "steps": [
            {"draft": CLAUDE_UNTRACED, "published": EMPTY},
            {
                "exit": 1,
                "says": ["Nothing was published"],
                "draft": CLAUDE_UNTRACED,
                "published": EMPTY,
            },
        ],
        "writes": 0,
    },
    "draft/non-admin-publish-refused": {
        "steps": [
            {
                "exit": 1,
                "says": [NOT_ADMIN, "restricted to workspace admins"],
                "draft": CLAUDE,
                "published": ABSENT,
            }
        ],
        "writes": 0,
    },
    "draft/export-reads-draft": {
        "steps": [
            {"says": [APPLIED_AUTHORED], "draft": CLAUDE_SONNET_DEFAULT, "published": CLAUDE},
            {
                "says": [
                    '"spec_version": 1',
                    '"default_agent": "CODING_AGENT_CLAUDE_CODE"',
                    f'"default_model": "{SANDBOX_SONNET_MODEL}"',
                ],
                "silent": [
                    '"name"',
                    "mcp_servers",
                    "skills",
                    f'"default_model": "{SANDBOX_MODEL}"',
                ],
                "draft": CLAUDE_SONNET_DEFAULT,
                "published": CLAUDE,
            },
        ],
    },
    "draft/legacy-v1-migration": {
        "steps": [{"says": [APPLIED_PUBLISHED], "draft": ABSENT, "published": CLAUDE}],
        "state": "v2",
        "files": [".ucode/managed-state.json.pre-v2.bak"],
        "legacy_backup": "byte-identical",
    },
    "draft/legacy-v1-was-a-draft": {
        "steps": [
            {
                "exit": 1,
                "says": [
                    "No managed config draft found locally",
                    "a fetched published config is not an authoring source",
                ],
                "draft": ABSENT,
                "published": CLAUDE,
            }
        ],
        "state": "pre-v2 (unmigrated on disk)",
    },
    "draft/legacy-v1-serves-as-fallback": {
        "steps": [
            {
                "says": ["applying the last one saved for this workspace", APPLIED_PUBLISHED],
                "draft": ABSENT,
                "published": CLAUDE,
            }
        ],
    },
    "section/non-admin-refused": {
        "steps": [{"exit": 1, "says": [NOT_ADMIN], "draft": CLAUDE, "published": ABSENT}],
        "writes": 0,
    },
    "section/tracing-is-local": {
        "steps": [
            {"says": ["Configuration Complete"]},
            {
                "says": ["Unity Catalog: main.default.ucode_traces", "Tracing configured for"],
            },
        ],
    },
    "multi/two-workspaces": {
        "steps": [
            {
                "says": ["Configuration Complete"],
                "silent": [AUTHORING_STARTED, APPLIED_PUBLISHED, LOCAL_OPTIONS_NOTE],
            }
        ],
        "writes": 0,
        "state": "absent",
        "forbid_calls": ["coding-agent-config"],
    },
    "section/no-authored-config": {
        "steps": [
            {
                "exit": 1,
                "says": [
                    "No managed config has been authored for",
                    "Run `ucode configure` first to pick the agents and models",
                ],
                "draft": ABSENT,
                "published": ABSENT,
            }
        ],
        "state": "absent",
    },
}


def _run_scenario(name: str, home: Path) -> dict:
    env = dict(os.environ)
    env["SANDBOX_HOME"] = str(home)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src")])
    proc = subprocess.run(
        [sys.executable, "-m", "tests.sandbox_scenarios", name],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    marker = "---SANDBOX-JSON---"
    if marker not in proc.stdout:
        return {
            "name": name,
            "harness_error": (
                f"no JSON verdict\nstdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
            ),
        }
    return json.loads(proc.stdout.split(marker)[-1])


@pytest.fixture(scope="session")
def sandbox_results(tmp_path_factory) -> dict[str, dict]:
    """Run the whole scenario table once, concurrently, and cache the verdicts.

    The scenarios are independent subprocesses with their own HOME, so they parallelize cleanly;
    run serially the table costs about 15 seconds.
    """
    names = list(SCENARIOS)
    homes = {name: tmp_path_factory.mktemp("sandbox") for name in names}
    with ThreadPoolExecutor(max_workers=8) as pool:
        verdicts = pool.map(lambda name: _run_scenario(name, homes[name]), names)
    return dict(zip(names, verdicts, strict=True))


def _normalize(text: str) -> str:
    """Collapse whitespace: Rich wraps output at the terminal width, mid-sentence."""
    return " ".join((text or "").split())


def _slot(actual: dict | None) -> str | dict:
    """Fingerprint a slot, including the fields a bad apply or carry-forward would silently drop."""
    if actual is None:
        return ABSENT
    if actual == {}:
        return EMPTY
    agents = actual.get("enabled_agents") or {}
    return {
        "agents": sorted(agents),
        "default_agent": actual.get("default_agent"),
        "models": {
            tool: (config.get("model_config") or {}).get("default_model")
            for tool, config in sorted(agents.items())
        },
        "tracing_table": actual.get("tracing_table"),
        "budget_policy": actual.get("budget_policy"),
    }


def _check_step(step: dict, expected: dict, label: str) -> list[str]:
    problems = []
    output = _normalize(step["output"])
    if step["exit_code"] != expected.get("exit", 0):
        problems.append(f"{label}: exit {step['exit_code']}, expected {expected.get('exit', 0)}")
    for needle in expected.get("says", []):
        if _normalize(needle) not in output:
            problems.append(f"{label}: missing output {needle!r}")
    for needle in expected.get("silent", []):
        if _normalize(needle) in output:
            problems.append(f"{label}: unexpected output {needle!r}")
    positions = [output.find(_normalize(n)) for n in expected.get("order", [])]
    if positions != sorted(positions):
        problems.append(f"{label}: output out of order: {expected['order']}")
    for slot in ("draft", "published"):
        if slot in expected and _slot(step[f"{slot}_after"]) != expected[slot]:
            problems.append(
                f"{label}: {slot} slot is {_slot(step[f'{slot}_after'])}, expected {expected[slot]}"
            )
    if step["prompts_left"]:
        problems.append(f"{label}: scripted answers went unused: {step['prompts_left']}")
    return problems


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_configure_scenario(name, sandbox_results):
    result = sandbox_results[name]
    expected = EXPECTATIONS[name]
    assert not result.get("harness_error"), result.get("harness_error")

    problems = []
    steps = result["steps"]
    if len(steps) != len(expected["steps"]):
        pytest.fail(f"{name}: ran {len(steps)} steps, expected {len(expected['steps'])}")
    for index, (step, step_expected) in enumerate(zip(steps, expected["steps"], strict=True), 1):
        label = f"step {index} (ucode {' '.join(step['argv'])})"
        problems += _check_step(step, step_expected, label)

    gateway = result["gateway"]
    if gateway["unrouted"]:
        problems.append(f"reached an unmodelled endpoint: {gateway['unrouted']}")
    if len(gateway["writes"]) != expected.get("writes", 0):
        problems.append(
            f"{len(gateway['writes'])} server writes, expected {expected.get('writes', 0)}: "
            f"{gateway['writes']}"
        )
    for forbidden in expected.get("forbid_calls", []):
        hit = [call for call in gateway["calls"] if forbidden in call]
        if hit:
            problems.append(f"called {forbidden!r}: {hit}")
    if "state" in expected and result["state_shape"] != expected["state"]:
        problems.append(
            f"managed-state.json is {result['state_shape']!r}, expected {expected['state']!r}"
        )
    for wanted in expected.get("files", []):
        if not any(wanted in path for path in result["home_tree"]):
            problems.append(f"missing file {wanted!r} in HOME: {result['home_tree']}")
    if "legacy_backup" in expected and result["legacy_backup"] != expected["legacy_backup"]:
        problems.append(
            f"pre-v2 backup is {result['legacy_backup']!r} against the seeded file, "
            f"expected {expected['legacy_backup']!r}"
        )

    if problems:
        transcript = "\n\n".join(
            f"$ ucode {' '.join(step['argv'])}  -> exit {step['exit_code']}\n{step['output']}"
            for step in steps
        )
        pytest.fail(
            f"{name}: {SCENARIOS[name]['desc']}\n\n" + "\n".join(problems) + f"\n\n{transcript}"
        )


def test_every_scenario_has_expectations():
    assert sorted(EXPECTATIONS) == sorted(SCENARIOS)
