"""Sandbox harness for `ucode configure`.

Runs the real CLI against a fake Databricks workspace inside a throwaway HOME, so every
file write, state transition and printed line is real. Only two layers are faked:

  * the HTTP layer in ucode.databricks (`_http_get_json` / `_http_send_json`), backed by
    FakeWorkspace, plus the subprocess seams that shell out to the databricks CLI;
  * the questionary prompts in ucode.ui, driven by a scripted answer queue.

Everything above those seams (cli, managed_wizard, managed_config, managed_publish,
agents/*, config_io, state) executes for real.

Import order matters: HOME must be set before ucode is imported (module-level
`Path.home()`), and ui/databricks must be patched before ucode.cli binds
`from ... import name` references.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_home() -> Path:
    """A throwaway HOME, exported before ucode is imported.

    Honors SANDBOX_HOME so the caller (pytest's tmp_path) can own the directory and clean it
    up; falls back to a temp dir when the module is driven by hand.
    """
    override = os.environ.get("SANDBOX_HOME")
    home = Path(override) if override else Path(tempfile.mkdtemp(prefix="ucode-sandbox-home-"))
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(home)
    os.environ["USERPROFILE"] = str(home)
    os.environ["DATABRICKS_CONFIG_FILE"] = str(home / ".databrickscfg")
    (home / ".databrickscfg").write_text(
        "[DEFAULT]\nhost = https://sandbox.cloud.databricks.com\ntoken = sandbox-pat\n",
        encoding="utf-8",
    )
    for var in ("ENABLE_MANAGED_AGENT_CONFIG", "ENABLE_CLAUDE_CODE_GATEWAY_MODEL_DISCOVERY"):
        os.environ.pop(var, None)
    for path in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return home


class PromptScript:
    """Scripted answers for the ui.prompt_* primitives.

    Each entry is (kind, value). A prompt pops the next entry and asserts its kind, so a
    flow that asks something unexpected fails loudly instead of silently taking a default.
    """

    def __init__(self, answers: list[tuple[str, object]], *, discover: bool = False):
        self.pending = list(answers)
        self.asked: list[tuple[str, str]] = []
        self.discover = discover

    def take(self, kind: str, question: str, options=None) -> object:
        self.asked.append((kind, question))
        if not self.pending:
            if self.discover:
                return self._auto(kind, options)
            raise AssertionError(
                f"unscripted {kind} prompt: {question!r}\nasked so far: {self.asked}"
            )
        want, value = self.pending.pop(0)
        if want != kind:
            if self.discover:
                self.pending.insert(0, (want, value))
                return self._auto(kind, options)
            raise AssertionError(
                f"prompt order mismatch: scripted {want!r} but flow asked {kind!r}: {question!r}"
            )
        return value

    @staticmethod
    def _auto(kind, options):
        """Discovery-mode default: keep the flow moving so one run reveals the whole sequence."""
        first = None
        if options:
            head = list(options)[0]
            first = head[0] if isinstance(head, (tuple, list)) else head
        return {
            "yes_no": False,
            "select": first,
            "multi": [first] if first is not None else [],
            "text": "",
            "number": 50.0,
            "workspace": ("https://sandbox.cloud.databricks.com", "DEFAULT"),
        }[kind]

    def drained(self) -> bool:
        return not self.pending


CURRENT: PromptScript | None = None


def set_script(script: PromptScript) -> None:
    global CURRENT
    CURRENT = script


def install_prompts(script: PromptScript) -> None:
    """Patch ucode.ui's prompt primitives. Must run before ucode.cli is imported."""
    import ucode.ui as ui

    set_script(script)

    def take(kind, question, options=None):
        assert CURRENT is not None, "no prompt script installed"
        return CURRENT.take(kind, question, options)

    def yes_no(question, default=True):
        return take("yes_no", question)

    def yes_no_default(question, default=True):
        return take("yes_no", question)

    def selection(prompt, options, **kwargs):
        return take("select", prompt, options)

    def multi(prompt, options, **kwargs):
        return take("multi", prompt, options)

    def tools(available, preselected=None, prompt="Select coding agents to configure:"):
        return take("multi", prompt, available)

    def text(prompt, **kwargs):
        return take("text", prompt)

    def percentage(prompt, **kwargs):
        return take("number", prompt)

    def workspace(*args, **kwargs):
        return take("workspace", "workspace")

    for name, fn in (
        ("prompt_yes_no", yes_no),
        ("prompt_yes_no_default", yes_no_default),
        ("prompt_for_selection", selection),
        ("prompt_for_multi_selection", multi),
        ("prompt_for_tools", tools),
        ("prompt_for_text", text),
        ("prompt_for_percentage", percentage),
        ("prompt_for_workspace", workspace),
    ):
        if hasattr(ui, name):
            setattr(ui, name, fn)


def run_cli(argv: list[str], *, env: dict[str, str] | None = None) -> tuple[int, str]:
    """Invoke the real typer app in-process and return (exit_code, output)."""
    from typer.testing import CliRunner

    from ucode.cli import app

    previous = {}
    for key, value in (env or {}).items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        result = CliRunner().invoke(app, argv)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    output = result.output
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        import traceback

        output += "\n[EXCEPTION] " + "".join(
            traceback.format_exception(
                type(result.exception), result.exception, result.exception.__traceback__
            )
        )
    return result.exit_code, output


def managed_state(home: Path) -> dict:
    path = home / ".ucode" / "managed-state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _entry(home: Path, workspace: str) -> dict:
    """The slots for one workspace, read straight off disk without going through ucode.

    Mirrors the v1 -> v2 read migration so a scenario that leaves a pre-v2 file on disk reports
    the slots ucode would see, not "absent".
    """
    state = managed_state(home)
    if state.get("version") == 2:
        return (state.get("workspaces") or {}).get(workspace) or {}
    if state.get("workspace") == workspace:
        return {"published": state.get("config") or {}}
    return {}


def state_shape(home: Path) -> str:
    state = managed_state(home)
    if not state:
        return "absent"
    return "v2" if state.get("version") == 2 else "pre-v2 (unmigrated on disk)"


def draft_slot(home: Path, workspace: str) -> dict | None:
    return _entry(home, workspace).get("draft")


def published_slot(home: Path, workspace: str) -> dict | None:
    return _entry(home, workspace).get("published")


def home_tree(home: Path) -> list[str]:
    """Every file under the sandbox HOME, relative and sorted, for eyeballing writes."""
    return sorted(
        str(p.relative_to(home))
        for p in home.rglob("*")
        if p.is_file() and "__pycache__" not in str(p)
    )
