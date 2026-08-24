"""Rich/questionary presentation primitives. No project knowledge."""

from __future__ import annotations

import itertools
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal
from functools import wraps
from types import MethodType
from typing import Any, TypedDict

import questionary
from InquirerPy import inquirer
from InquirerPy.base.control import Choice as InquirerChoice
from InquirerPy.separator import Separator as InquirerSeparator
from InquirerPy.utils import InquirerPyStyle, get_style
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.dimension import Dimension
from questionary.prompts.common import InquirerControl
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)

# `ucode setup` opts into InquirerPy while the rest of the CLI stays on Questionary. Context-local
# state keeps nested helpers on the same backend without threading a presentation flag through every
# model/provider/budget function, and won't bleed between concurrent callers or tests.
_prompt_backend: ContextVar[str] = ContextVar("ucode_prompt_backend", default="questionary")
_wizard_rail_open: ContextVar[bool] = ContextVar("ucode_wizard_rail_open", default=False)


def _using_inquirerpy() -> bool:
    return _prompt_backend.get() == "inquirerpy"


def _rail_prefix() -> str:
    return "[dim]│[/dim]  " if _wizard_rail_open.get() else ""


def _print_blank(*, stderr: bool = False) -> None:
    target = err_console if stderr else console
    target.print("[dim]│[/dim]" if _wizard_rail_open.get() else "")


def inquirerpy_wizard[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Run one wizard on InquirerPy and reset all terminal-layout state afterwards."""

    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        backend_token = _prompt_backend.set("inquirerpy")
        rail_token = _wizard_rail_open.set(False)
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            if _wizard_rail_open.get():
                print_wizard_outro("Setup interrupted", kind="warning")
            raise
        except Exception:
            if _wizard_rail_open.get():
                print_wizard_outro("Setup stopped", kind="error")
            raise
        finally:
            _wizard_rail_open.reset(rail_token)
            _prompt_backend.reset(backend_token)

    return wrapped


_INQUIRER_STYLE = get_style(
    {
        "questionmark": "#666666",
        "answermark": "#666666",
        "answer": "#00d7d7 bold",
        "input": "#00d7d7",
        "question": "bold",
        "answered_question": "bold",
        "instruction": "#666666",
        "long_instruction": "#666666",
        "pointer": "#00d7d7 bold",
        "checkbox": "#00d7d7",
        "separator": "#8a8a8a bold",
        "marker": "#00d7d7",
        "fuzzy_prompt": "#00d7d7",
        "fuzzy_info": "#666666",
        "fuzzy_match": "#00d7d7 bold",
    },
    style_override=False,
)


def _inquirer_marks() -> tuple[str, str]:
    # InquirerPy inserts one space after the mark. Include one more while the wizard rail is open so
    # question text aligns with Rich output's ``│  `` gutter.
    return ("│ ", "│ ") if _wizard_rail_open.get() else ("◇", "◆")


class _InquirerCommon(TypedDict):
    style: InquirerPyStyle
    qmark: str
    amark: str
    raise_keyboard_interrupt: bool


def _inquirer_common() -> _InquirerCommon:
    qmark, amark = _inquirer_marks()
    return {
        "style": _INQUIRER_STYLE,
        "qmark": qmark,
        "amark": amark,
        "raise_keyboard_interrupt": True,
    }


def _with_inquirer_rail(question: Any) -> Any:
    """Prefix every row in an InquirerPy choice list with the wizard's live rail.

    InquirerPy applies ``qmark`` only to the question. Its option window starts back at column zero,
    which both breaks the vertical rail and makes the highlighted row look less indented than status
    output. Wrapping the control's two row formatters keeps highlighted, normal, and separator rows
    under the same ``│  `` gutter without changing their values or key handling.

    Test doubles and non-list prompts have no ``content_control`` and pass through unchanged.
    """
    if not _wizard_rail_open.get():
        return question
    control = getattr(question, "content_control", None)
    if control is None:
        return question

    original_hover = control._get_hover_text
    original_normal = control._get_normal_text

    def hover(_: Any, choice: Any) -> list[tuple[str, str]]:
        return [("class:questionmark", "│  "), *original_hover(choice)]

    def normal(_: Any, choice: Any) -> list[tuple[str, str]]:
        return [("class:questionmark", "│  "), *original_normal(choice)]

    control._get_hover_text = MethodType(hover, control)
    control._get_normal_text = MethodType(normal, control)
    return question


def _inquirer_filter_prompt() -> str:
    return "│  Filter" if _wizard_rail_open.get() else "Filter"


def _choice_summary(values: list[str], labels: dict[str, str]) -> str:
    selected = [labels.get(str(value), str(value)) for value in values]
    if len(selected) <= 3:
        return ", ".join(selected)
    return f"{len(selected)} selected"


# Past this many options the choice list is pinned to a fixed-height scrolling viewport (see
# `_cap_choice_viewport`) rather than growing to fill the terminal, and the pickers append a
# "↑/↓ scroll" note to their instruction line. The value is both the boundary and the number of
# rows shown at once; matches mcp.py's MCP_PICKER_VISIBLE_ROWS so both picker families agree.
_SCROLL_HINT_THRESHOLD = 10


def _with_scroll_hint(instruction: str, option_count: int) -> str:
    """Append a scroll affordance when the option list is too long to fully show at once.

    The hint is discoverability only — the viewport already scrolls; users just can't tell there is
    more below the fold. Short lists are left alone so the instruction stays terse.
    """
    if option_count > _SCROLL_HINT_THRESHOLD:
        return f"{instruction[:-1]}, ↑/↓ scroll)" if instruction.endswith(")") else instruction
    return instruction


def _cap_choice_viewport(question: questionary.Question, option_count: int) -> None:
    """Pin a long choice list to a fixed ``_SCROLL_HINT_THRESHOLD``-row scrolling window.

    questionary sizes the choice window to its content, bounded only by the terminal, so a tall
    terminal shows every option and a short one scrolls — the visible count depends on the window
    size. Capping the window height makes the picker show at most ``_SCROLL_HINT_THRESHOLD`` rows and
    scroll through the rest, keeping the highlighted row in view (``InquirerControl`` emits a cursor
    token at the pointer, which prompt_toolkit's ``Window`` tracks). Short lists are left untouched so
    they render at their natural height with no empty rows.

    Best-effort: it mutates questionary's internal layout, so any shape change (missing application,
    no ``InquirerControl`` window) leaves the picker as-is rather than raising — the list still works,
    it just falls back to terminal-fit sizing.
    """
    if option_count <= _SCROLL_HINT_THRESHOLD:
        return
    application = getattr(question, "application", None)
    if application is None:
        return
    visible = Dimension(preferred=_SCROLL_HINT_THRESHOLD, max=_SCROLL_HINT_THRESHOLD)
    for window in application.layout.find_all_windows():
        if isinstance(window, Window) and isinstance(window.content, InquirerControl):
            window.height = visible
            return


# Output verbosity. "normal" (default) renders decorative panels; "low" trades
# them for terse single-line output. Set once at CLI entry via set_verbosity.
_verbosity = "normal"


def set_verbosity(value: str) -> None:
    global _verbosity
    _verbosity = value or "normal"


def get_verbosity() -> str:
    return _verbosity


def is_low_verbosity() -> bool:
    return _verbosity == "low"


def print_section(title: str) -> None:
    console.print()
    console.print(Panel(title, style="bold blue", expand=False))


def print_wizard_header(command: str, description: str) -> None:
    """Open an interactive flow with compact command branding and a continuous rail.

    A wizard is one continuous composition, so a panel around its name makes the following steps
    look like separate cards. Styling the command itself gives the flow an identity while leaving
    the rest of the terminal as one canvas.
    """
    head, separator, tail = command.partition(" ")
    rendered = f"[bold cyan]{escape(head)}[/bold cyan]"
    if separator:
        rendered += f" [bold]{escape(tail)}[/bold]"
    console.print()
    console.print(f"[bold cyan]┌[/bold cyan]  {rendered}")
    console.print(f"[dim]│[/dim]  [dim]{escape(description)}[/dim]")
    _wizard_rail_open.set(True)


def print_wizard_step(index: int, total: int, title: str) -> None:
    """Render one node in a compact vertical stepper.

    The connector is printed immediately before every node after the first. Output and prompts from
    the previous phase remain between the two nodes, so the rail visually joins the whole flow
    without requiring a full-screen TUI or redrawing terminal history.
    """
    console.print("[dim]│[/dim]")
    console.print(
        f"[bold cyan]◇[/bold cyan]  [dim]{index}/{total}[/dim]  [bold]{escape(title)}[/bold]"
    )


def print_wizard_outro(message: str, *, kind: str = "success") -> None:
    """Close the active wizard rail and return subsequent output to the normal left margin."""
    if not _wizard_rail_open.get():
        return
    color = {"success": "green", "warning": "yellow", "error": "red"}.get(kind, "cyan")
    console.print("[dim]│[/dim]")
    console.print(f"[bold {color}]└[/bold {color}]  [bold]{escape(message)}[/bold]")
    _wizard_rail_open.set(False)


def print_heading(text: str) -> None:
    _print_blank()
    console.print(f"{_rail_prefix()}[bold]{text}[/bold]")


def print_kv(key: str, val: str) -> None:
    console.print(f"{_rail_prefix()}[bold]{key}:[/bold] [cyan]{val}[/cyan]")


def kv_line(key: str, val: str) -> str:
    """A `print_kv`-styled line, returned instead of printed, for collecting into a panel.

    The value is markup-escaped. Rich reads bracketed text as a style tag and renders nothing for
    it, so a policy name of ``[prod] tiered routing`` displayed as ``tiered routing`` in the config
    summary — the one block an admin reads to confirm what they are about to publish workspace-wide.
    Values here include admin-typed free text (policy name, skills locations, tracing table).
    """
    return f"[bold]{escape(key)}:[/bold] [cyan]{escape(val)}[/cyan]"


def print_panel(title: str, lines: list[str]) -> None:
    """Render `lines` inside a titled box.

    Unlike :func:`print_section`, which boxes a bare title, this boxes the body — so a block that
    should be read as one unit (a config summary an admin is about to publish) reads as one, rather
    than as loose lines that blend into whatever the flow printed before it.
    """
    if _wizard_rail_open.get():
        print_heading(title)
        for line in lines:
            console.print(f"{_rail_prefix()}{line}")
        return
    console.print()
    console.print(Panel("\n".join(lines), title=title, style="blue", expand=False))


def print_warning_panel(message: str, *, title: str = "Warning") -> None:
    """Render a warning as a boxed panel, for a blocker that should stand out from inline notes.

    A `!` marker keeps it visually of a kind with :func:`print_warning`, but the box gives a
    dead-end message (e.g. "no budgets exist, nothing to do") the weight to be read before the flow
    exits, rather than scrolling past as one more line.
    """
    if _wizard_rail_open.get():
        print_heading(title)
        console.print(f"{_rail_prefix()}[bold yellow]![/bold yellow] {message}")
        return
    console.print()
    console.print(
        Panel(f"[bold yellow]![/bold yellow] {message}", title=title, style="yellow", expand=False)
    )


def print_note(text: str) -> None:
    console.print(f"{_rail_prefix()}[dim]•[/dim] {text}")


def print_success(message: str) -> None:
    console.print(f"{_rail_prefix()}[bold green]✔[/bold green] {message}")


def print_warning(message: str) -> None:
    console.print(f"{_rail_prefix()}[bold yellow]![/bold yellow] {message}")


def print_warning_err(message: str) -> None:
    """``print_warning`` on stderr, for when stdout is a machine-read stream."""
    err_console.print(f"{_rail_prefix()}[bold yellow]![/bold yellow] {message}")


def print_err(message: str) -> None:
    err_console.print(f"{_rail_prefix()}[bold red]ERROR[/bold red] {message}")


def heading(text: str) -> str:
    return f"[bold blue]{text}[/bold blue]"


def label(text: str) -> str:
    return f"[bold]{text}[/bold]"


def value(text: str) -> str:
    return f"[cyan]{text}[/cyan]"


def muted(text: str) -> str:
    return f"[dim]{text}[/dim]"


def status_badge(text: str, kind: str) -> str:
    color = {"ok": "green", "warn": "yellow", "error": "red", "info": "blue"}.get(kind, "bold")
    return f"[bold {color}]{text}[/bold {color}]"


@contextmanager
def spinner(message: str | Callable[[], str]):
    """Show a spinner while the block runs. `message` may be a callable, which
    is re-evaluated on every frame so callers can render live progress (e.g. a
    running count) during a long operation."""
    if not sys.stdout.isatty():
        yield
        return

    if isinstance(message, str):
        static_message = message

        def current_message() -> str:
            return static_message
    else:
        current_message = message

    stop_event = threading.Event()
    # Context variables don't propagate into this animation thread, so capture the wizard gutter
    # now; otherwise the spinner is the one line that jumps outside an otherwise continuous rail.
    spinner_prefix = "\033[2m│\033[0m  " if _wizard_rail_open.get() else ""

    def spin() -> None:
        for frame in itertools.cycle("|/-\\"):
            if stop_event.is_set():
                break
            # `\033[K` erases to end of line so a shrinking dynamic message
            # doesn't leave stale characters behind.
            sys.stdout.write(f"\r{spinner_prefix}\033[2m{frame}\033[0m {current_message()}\033[K")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1)


@contextmanager
def progress_bar(description: str, total: int) -> Iterator[Callable[[], None]]:
    """Yield an ``advance()`` callback that drives a ``k/n`` progress bar.

    Falls back to no live bar off a tty (e.g. CI), so logs stay single-line.
    """
    if total <= 0 or not sys.stdout.isatty():
        yield lambda: None
        return

    with Progress(
        TextColumn("[dim]{task.description}[/dim]"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(description, total=total)
        yield lambda: progress.advance(task)


def render_box_table(
    headers: list[str],
    rows: list[list[str]],
    max_widths: list[int] | None = None,
) -> str:
    wrapped_rows: list[list[list[str]]] = []
    widths = [len(header) for header in headers]

    for row in rows:
        wrapped_row: list[list[str]] = []
        for index, cell in enumerate(row):
            raw_cell = cell if cell else "-"
            width_limit = max_widths[index] if max_widths and index < len(max_widths) else None
            if width_limit:
                cell_lines = textwrap.wrap(raw_cell, width=width_limit) or ["-"]
            else:
                cell_lines = raw_cell.splitlines() or ["-"]
            wrapped_row.append(cell_lines)
            widths[index] = max(widths[index], max(len(line) for line in cell_lines))
        wrapped_rows.append(wrapped_row)

    top = "┏" + "┳".join("━" * (w + 2) for w in widths) + "┓"
    header = "┃ " + " ┃ ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " ┃"
    middle = "┡" + "╇".join("━" * (w + 2) for w in widths) + "┩"
    bottom = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    body_lines: list[str] = []
    for wrapped_row in wrapped_rows:
        row_height = max(len(cell_lines) for cell_lines in wrapped_row)
        for line_index in range(row_height):
            body_lines.append(
                "│ "
                + " │ ".join(
                    (
                        wrapped_row[col][line_index] if line_index < len(wrapped_row[col]) else ""
                    ).ljust(widths[col])
                    for col in range(len(headers))
                )
                + " │"
            )

    return "\n".join([top, header, middle, *body_lines, bottom])


def format_token_count(token_count: int) -> str:
    value_float = float(token_count)
    if token_count >= 1_000_000_000:
        return f"{value_float / 1_000_000_000:.1f}B"
    if token_count >= 1_000_000:
        return f"{value_float / 1_000_000:.1f}M"
    if token_count >= 1_000:
        return f"{value_float / 1_000:.1f}K"
    return str(token_count)


def format_usd(amount: Decimal) -> str:
    return f"${amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,}"


def format_cost_usd(amount: Decimal) -> str:
    """Like `format_usd`, but keeps more precision for sub-cent per-model costs.

    A model that cost a fraction of a cent would round to ``$0.00`` at two
    decimals and read as free, so amounts under a cent show four decimals.
    """
    if Decimal(0) < amount < Decimal("0.01"):
        return f"${amount.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)}"
    return format_usd(amount)


def format_meter(fraction: float, width: int = 30) -> str:
    """Text meter for `fraction` of a whole, clamped to [0, 1]."""
    clamped = min(max(fraction, 0.0), 1.0)
    filled = int(clamped * width)
    # A small-but-real fraction shouldn't read as empty.
    if clamped > 0:
        filled = max(filled, 1)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def format_duration(duration_value: timedelta | None) -> str:
    if not duration_value or duration_value.total_seconds() <= 0:
        return "-"
    total_minutes = duration_value.total_seconds() / 60
    if total_minutes < 60:
        return f"{int(round(total_minutes))}m"
    total_hours = total_minutes / 60
    if total_hours < 10:
        return f"{total_hours:.1f}h"
    if total_hours < 24:
        return f"{round(total_hours):.0f}h"
    return f"{total_hours / 24:.1f}d"


def normalize_workspace_url(workspace: str) -> str:
    workspace = workspace.strip()
    if not workspace:
        raise ValueError("Workspace URL cannot be empty.")
    if not workspace.startswith(("http://", "https://")):
        workspace = f"https://{workspace}"
    return workspace.rstrip("/")


def prompt_for_workspace(
    description: str,
    profiles: list[tuple[str, str]] | None = None,
    *,
    show_section: bool = True,
    prompt: str = "Select workspace:",
) -> tuple[str, str | None]:
    """Ask the user for a workspace URL, offering profiles as quick-select.

    `profiles` is a list of (host_url, profile_name) tuples. Caller fetches
    them — `ui.py` stays Databricks-agnostic. Duplicate hosts (multiple
    profiles pointing at the same workspace) are shown separately; the picker
    returns the exact (host, profile_name) the user selected. ``show_section=False`` lets a larger
    wizard supply its own step banner. Returns ``(url, profile_name)``; profile_name is ``None``
    when the user typed a URL manually.
    """
    # Keep section labels visually consistent with every other CLI phase: the label is the panel
    # body, not a border title. Callers that already printed a larger flow's step banner can suppress
    # this local section rather than showing two boxes back to back.
    if show_section:
        print_section(description)

    if profiles:
        name_header = "Profile Name"
        url_header = "Workspace URL"
        # Clamp so a single very long profile name can't push the URL column off-screen.
        max_name_width = 40
        name_width = min(
            max_name_width,
            max(len(name_header), *(len(name) for _, name in profiles)),
        )
        header_title = f"  {name_header.ljust(name_width)}  {url_header}"

        def profile_row(host: str, profile_name: str) -> str:
            display_name = (
                profile_name
                if len(profile_name) <= name_width
                else profile_name[: name_width - 1] + "…"
            )
            return f"{display_name.ljust(name_width)}  {host}"

        if _using_inquirerpy():
            inquirer_choices: list[InquirerChoice | InquirerSeparator] = [
                InquirerSeparator(header_title),
                *[
                    InquirerChoice(value=(host, profile_name), name=profile_row(host, profile_name))
                    for host, profile_name in profiles
                ],
                InquirerChoice(value=None, name="Enter a different URL"),
            ]

            def workspace_summary(answer: object) -> str:
                # InquirerPy passes the selected row's display name to transformers, not its raw
                # value. A tuple check therefore mislabeled every completed profile selection as
                # "Enter a different URL". Preserve the actual profile-and-workspace display row.
                return str(answer)

            question = inquirer.select(
                message=prompt,
                choices=inquirer_choices,
                pointer="❯",
                instruction="↑↓ move · enter select",
                transformer=workspace_summary,
                max_height=_SCROLL_HINT_THRESHOLD,
                **_inquirer_common(),
            )
            choice = _with_inquirer_rail(question).execute()
        else:
            choices: list[questionary.Choice | questionary.Separator] = [
                questionary.Separator(header_title),
                *[
                    questionary.Choice(
                        title=profile_row(host, profile_name), value=(host, profile_name)
                    )
                    for host, profile_name in profiles
                ],
                questionary.Choice(title="Enter a different URL", value=None),
            ]
            style = questionary.Style(
                [
                    ("highlighted", "fg:cyan bold"),
                    ("pointer", "fg:cyan bold"),
                    ("answer", "fg:cyan"),
                    ("separator", "fg:white bold"),
                ]
            )
            choice = questionary.select(
                prompt, choices=choices, style=style, pointer="›", qmark=""
            ).ask()
        if isinstance(choice, tuple):
            host, profile_name = choice
            return normalize_workspace_url(host), profile_name

    while True:
        if _using_inquirerpy():
            raw_value = inquirer.text(
                message=prompt,
                validate=lambda value: bool(str(value).strip()),
                invalid_message="Enter a workspace URL",
                mandatory=True,
                **_inquirer_common(),
            ).execute()
        else:
            raw_value = console.input(f"  [bold]Workspace URL[/bold] {muted('›')} ").strip()
        try:
            return normalize_workspace_url(str(raw_value)), None
        except ValueError as exc:
            print_err(str(exc))


def prompt_for_tools(
    available: list[tuple[str, str]],
    preselected: list[str] | set[str] | None = None,
    prompt: str = "Select coding agents to configure:",
) -> list[str]:
    """Multi-select picker for coding agents.

    `available` is [(tool_id, display_name), ...]. Returns the chosen tool_ids.
    When ``preselected`` is None every option is checked by default, so hitting
    Enter selects everything; pass a subset to pre-check only those (e.g. the
    agents an existing managed config already enables). Returns [] if the user
    submits an empty selection.
    """
    preselected_set = {str(item) for item in preselected} if preselected is not None else None
    if _using_inquirerpy():
        labels = dict(available)
        choices = [
            InquirerChoice(
                value=tool_id,
                name=display,
                enabled=(preselected_set is None or tool_id in preselected_set),
            )
            for tool_id, display in available
        ]
        question = inquirer.checkbox(
            message=prompt,
            choices=choices,
            pointer="❯",
            enabled_symbol="◉",
            disabled_symbol="○",
            instruction="↑↓ move · space toggle · enter confirm",
            transformer=lambda values: _choice_summary(values, labels),
            max_height=_SCROLL_HINT_THRESHOLD,
            mandatory=False,
            **_inquirer_common(),
        )
        answer = _with_inquirer_rail(question).execute()
        return list(answer or [])

    style = questionary.Style(
        [
            # Theme-agnostic picker: every row renders in the terminal's
            # default foreground colour (`noinherit` strips the
            # prompt_toolkit defaults that would otherwise re-colour the
            # cursor row or every checked row).
            ("pointer", "fg:cyan bold"),
            ("highlighted", "noinherit"),
            ("selected", "noinherit"),
            ("answer", "fg:cyan"),
        ]
    )
    choices = [
        questionary.Choice(
            title=display,
            value=tool_id,
            checked=(preselected_set is None or tool_id in preselected_set),
        )
        for tool_id, display in available
    ]
    answer = questionary.checkbox(
        prompt,
        choices=choices,
        style=style,
        pointer="›",
        qmark="",
        instruction="(space to toggle, enter to confirm)",
    ).ask()
    return list(answer) if answer else []


def prompt_for_multi_selection(
    prompt: str,
    options: list[tuple[str, str]],
    preselected: list[str] | set[str] | None = None,
    *,
    searchable: bool = False,
) -> list[str] | None:
    """Multi-select picker over arbitrary `(value, label)` options.

    Distinct from :func:`prompt_for_tools`, which is agent-specific and defaults to
    everything checked: here nothing is checked unless ``preselected`` says so, since
    an admin picking models wants an explicit choice rather than "all of them".
    Returns the chosen values, [] on an empty submission, or None if cancelled
    (Ctrl-C) so callers can distinguish "chose nothing" from "aborted".

    ``searchable`` lets the user narrow a long list by typing; see
    :func:`prompt_for_selection` for why it trades away j/k navigation.
    """
    preselected_set = {str(item) for item in preselected} if preselected is not None else set()
    if _using_inquirerpy():
        labels = dict(options)
        choices = [
            InquirerChoice(
                value=value,
                name=option_label,
                enabled=value in preselected_set,
            )
            for value, option_label in options
        ]
        # Fuzzy search owns a separate input row. On a short list that row looks like a blank first
        # choice with a second cursor, so reserve it for lists long enough to benefit from filtering.
        if searchable and len(options) > _SCROLL_HINT_THRESHOLD:
            question = inquirer.fuzzy(
                message=prompt,
                choices=choices,
                pointer="❯",
                transformer=lambda values: _choice_summary(values, labels),
                max_height=_SCROLL_HINT_THRESHOLD,
                mandatory=False,
                multiselect=True,
                prompt=_inquirer_filter_prompt(),
                marker="◉",
                marker_pl="○",
                info=False,
                instruction="type filter · space toggle · enter confirm",
                keybindings={"toggle": [{"key": "space"}]},
                **_inquirer_common(),
            )
        else:
            question = inquirer.checkbox(
                message=prompt,
                choices=choices,
                pointer="❯",
                transformer=lambda values: _choice_summary(values, labels),
                max_height=_SCROLL_HINT_THRESHOLD,
                mandatory=False,
                enabled_symbol="◉",
                disabled_symbol="○",
                instruction="↑↓ move · space toggle · enter confirm",
                **_inquirer_common(),
            )
        answer = _with_inquirer_rail(question).execute()
        return None if answer is None else list(answer)

    style = questionary.Style(
        [
            ("pointer", "fg:cyan bold"),
            ("highlighted", "noinherit"),
            ("selected", "noinherit"),
            ("answer", "fg:cyan"),
        ]
    )
    choices = [
        questionary.Choice(title=option_label, value=value, checked=value in preselected_set)
        for value, option_label in options
    ]
    instruction = "(space to toggle, enter to confirm)"
    if searchable:
        instruction = "(type to filter, space to toggle, enter to confirm)"
    instruction = _with_scroll_hint(instruction, len(options))
    question = questionary.checkbox(
        prompt,
        choices=choices,
        style=style,
        pointer="›",
        qmark="",
        instruction=instruction,
        use_search_filter=searchable,
        use_jk_keys=not searchable,
    )
    _cap_choice_viewport(question, len(options))
    answer = question.ask()
    return None if answer is None else list(answer)


def prompt_for_text(
    prompt: str, *, default: str | None = None, required: bool = False
) -> str | None:
    """Free-text prompt, used when model discovery found nothing to pick from.

    Returns the trimmed input, ``default`` on an empty answer, or None when there is no
    default and the user submits nothing (or closes stdin).

    ``required=True`` raises ``KeyboardInterrupt`` on closed stdin instead of returning None, for
    callers that loop until they get a value: returning None to such a caller spins forever on a
    piped or exhausted stdin. Matches :func:`prompt_for_percentage`, which has no default and does
    the same.

    A default is shown as ``[value] (enter to accept)`` rather than the bare ``[value]``: bracketed
    text alone reads as a format example as easily as a value that will be used, so it invited
    retyping what pressing enter would already pick.

    The whole bracketed hint is markup-escaped, brackets included. Rich reads
    ``[coding-agents-tiered-routing]`` as a style tag and prints nothing for it, so an unescaped
    word-like default vanished from the prompt entirely — numeric ones like ``[80]`` are not valid
    tags and survived, which is why this looked fine wherever it was checked.
    """
    if _using_inquirerpy():
        answer = inquirer.text(
            message=prompt,
            default=default or "",
            instruction="enter to accept current value" if default else "",
            validate=lambda value: bool(str(value).strip()),
            invalid_message="Enter a value",
            mandatory=True,
            **_inquirer_common(),
        ).execute()
        return str(answer).strip() if answer is not None else default

    hint = f" {escape(f'[{default}]')} (enter to accept)" if default else ""
    while True:
        try:
            raw_value = console.input(f"{label(prompt)}{muted(hint)} {muted('›')} ").strip()
        except EOFError as exc:
            if required:
                raise KeyboardInterrupt from exc
            return default
        if raw_value:
            return raw_value
        if default is not None:
            return default
        print_err("Please enter a value.")


def prompt_for_percentage(prompt: str, *, default: float | None = None) -> float:
    """Prompt for a percentage (0-100) and return it as a fraction in [0, 1].

    Budget tiers are fractions in the API (the server validates 0..1), but admins think in
    percent — and the spec's own prose says "80%". Prompting in percent and converting here
    keeps that mismatch in one place instead of at every call site.

    No caller passes ``default`` today, and tier thresholds deliberately have none: a threshold
    decides when developers get downgraded, so it should be typed rather than accepted by accident.
    The hint is still formatted (and escaped) the same way :func:`prompt_for_text` formats its own,
    so the two cannot drift if a default is ever introduced.

    Raises ``KeyboardInterrupt`` on closed stdin when there is no default — see the handler below.
    """
    if _using_inquirerpy():
        default_percent = f"{default * 100:g}" if default is not None else ""

        def valid_percentage(value: str) -> bool:
            try:
                percent = float(str(value).rstrip("%"))
            except ValueError:
                return False
            return 0 <= percent <= 100

        answer = inquirer.text(
            message=prompt,
            default=default_percent,
            instruction="0–100",
            validate=valid_percentage,
            invalid_message="Enter a number between 0 and 100",
            transformer=lambda value: f"{float(str(value).rstrip('%')):g}%",
            mandatory=True,
            **_inquirer_common(),
        ).execute()
        return float(str(answer).rstrip("%")) / 100

    hint = f" {escape(f'[{default * 100:g}]')} (enter to accept)" if default is not None else ""
    while True:
        try:
            raw_value = console.input(f"{label(prompt)}{muted(hint)} {muted('› ')}").strip()
        except EOFError as exc:
            if default is not None:
                return default
            # Closed stdin with no default to fall back on is the admin abandoning the prompt, which
            # is what Ctrl-C means here too. Raised as KeyboardInterrupt so the CLI's existing
            # handler prints "Interrupted." and exits 130; a bare EOFError has no handler anywhere
            # above this and reached the admin as a traceback.
            raise KeyboardInterrupt from exc
        if not raw_value and default is not None:
            return default
        try:
            percent = float(raw_value.rstrip("%"))
        except ValueError:
            print_err("Please enter a number between 0 and 100.")
            continue
        if 0 <= percent <= 100:
            return percent / 100
        print_err("Please enter a number between 0 and 100.")


def prompt_for_selection(
    prompt: str, options: list[tuple[str, str]], *, searchable: bool = False
) -> str | None:
    """Single-select arrow-key picker. `options` is [(value, label), ...].

    The prompt renders above the choices (questionary convention). Returns the
    chosen value, or None if the user cancels (Ctrl-C / empty).

    ``searchable`` lets the user narrow a long list by typing. It costs j/k navigation — questionary
    rejects both at once, since j and k are also search characters — so it is opt-in for the pickers
    that are actually long (model and budget lists), leaving short ones on plain arrow keys.
    """
    if _using_inquirerpy():
        labels = dict(options)
        choices = [
            InquirerChoice(value=value, name=option_label) for value, option_label in options
        ]
        # InquirerPy's fuzzy control always adds a search-input row. Use it only when the list
        # actually needs filtering; otherwise that empty row reads as a duplicate selection cursor.
        if searchable and len(options) > _SCROLL_HINT_THRESHOLD:
            question = inquirer.fuzzy(
                message=prompt,
                choices=choices,
                pointer="❯",
                transformer=lambda selected: labels.get(str(selected), str(selected)),
                max_height=_SCROLL_HINT_THRESHOLD,
                prompt=_inquirer_filter_prompt(),
                info=False,
                instruction="type filter · ↑↓ move · enter select",
                **_inquirer_common(),
            )
        else:
            question = inquirer.select(
                message=prompt,
                choices=choices,
                pointer="❯",
                transformer=lambda selected: labels.get(str(selected), str(selected)),
                max_height=_SCROLL_HINT_THRESHOLD,
                instruction="↑↓ move · enter select",
                **_inquirer_common(),
            )
        return _with_inquirer_rail(question).execute()

    style = questionary.Style(
        [
            ("pointer", "fg:cyan bold"),
            ("highlighted", "noinherit"),
            ("selected", "noinherit"),
            ("answer", "fg:cyan"),
        ]
    )
    choices = [questionary.Choice(title=label, value=value) for value, label in options]
    instruction = "(type to filter, arrow keys to move)" if searchable else "(use arrow keys)"
    instruction = _with_scroll_hint(instruction, len(options))
    question = questionary.select(
        prompt,
        choices=choices,
        style=style,
        pointer="›",
        qmark="",
        instruction=instruction,
        use_search_filter=searchable,
        use_jk_keys=not searchable,
    )
    _cap_choice_viewport(question, len(options))
    answer = question.ask()
    return answer


def prompt_yes_no(prompt: str) -> bool:
    while True:
        response = console.input(f"{label(prompt)} {muted('(y/n)')} {muted('›')} ").strip().lower()
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print_err("Please answer yes or no.")


def prompt_yes_no_default(prompt: str, *, default: bool) -> bool:
    """Empty answer or closed stdin (EOF) takes ``default`` (no abort on piped runs)."""
    if _using_inquirerpy():
        default_label = "Yes" if default else "No"
        keys = "Y/n" if default else "y/N"
        return bool(
            inquirer.confirm(
                message=prompt,
                default=default,
                instruction=f"{keys} · enter selects {default_label}",
                transformer=lambda answer: "Yes" if answer else "No",
                **_inquirer_common(),
            ).execute()
        )

    hint = "(Y/n)" if default else "(y/N)"
    while True:
        try:
            response = console.input(f"{label(prompt)} {muted(hint)} {muted('›')} ").strip().lower()
        except EOFError:
            return default
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print_err("Please answer yes or no.")


def prompt_for_choice(prompt: str, options: list[tuple[str, str]]) -> str:
    console.print()
    for index, (_, option_label) in enumerate(options, start=1):
        console.print(f"  [bold]{index}.[/bold] [cyan]{option_label}[/cyan]")

    while True:
        raw_value = console.input(f"{label(prompt)} {muted('›')} ").strip()
        if raw_value.isdigit():
            selected_index = int(raw_value)
            if 1 <= selected_index <= len(options):
                return options[selected_index - 1][0]
        print_err("Please enter a valid option number.")


def prompt_for_client_id() -> str:
    while True:
        client_id = console.input(f"{label('OAuth client ID')} {muted('›')} ").strip()
        if client_id:
            return client_id
        print_err("Client ID cannot be empty.")


def prompt_for_client_secret() -> str:
    while True:
        client_secret = console.input(f"{label('OAuth client secret')} {muted('›')} ").strip()
        if client_secret:
            return client_secret
        print_err("Client secret cannot be empty.")
