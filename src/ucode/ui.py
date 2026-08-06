"""Rich/questionary presentation primitives. No project knowledge."""

from __future__ import annotations

import itertools
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import timedelta

import questionary
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)

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


def print_heading(text: str) -> None:
    console.print()
    console.print(f"[bold]{text}[/bold]")


def print_kv(key: str, val: str) -> None:
    console.print(f"  [bold]{key}:[/bold] [cyan]{val}[/cyan]")


def kv_line(key: str, val: str) -> str:
    """A `print_kv`-styled line, returned instead of printed, for collecting into a panel."""
    return f"[bold]{key}:[/bold] [cyan]{val}[/cyan]"


def print_panel(title: str, lines: list[str]) -> None:
    """Render `lines` inside a titled box.

    Unlike :func:`print_section`, which boxes a bare title, this boxes the body — so a block that
    should be read as one unit (a config summary an admin is about to publish) reads as one, rather
    than as loose lines that blend into whatever the flow printed before it.
    """
    console.print()
    console.print(Panel("\n".join(lines), title=title, style="blue", expand=False))


def print_note(text: str) -> None:
    console.print(f"[dim]•[/dim] {text}")


def print_success(message: str) -> None:
    console.print(f"[bold green]✔[/bold green] {message}")


def print_warning(message: str) -> None:
    console.print(f"[bold yellow]![/bold yellow] {message}")


def print_err(message: str) -> None:
    err_console.print(f"[bold red]ERROR[/bold red] {message}")


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

    def spin() -> None:
        for frame in itertools.cycle("|/-\\"):
            if stop_event.is_set():
                break
            # `\033[K` erases to end of line so a shrinking dynamic message
            # doesn't leave stale characters behind.
            sys.stdout.write(f"\r\033[2m{frame}\033[0m {current_message()}\033[K")
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
) -> tuple[str, str | None]:
    """Ask the user for a workspace URL, offering profiles as quick-select.

    `profiles` is a list of (host_url, profile_name) tuples. Caller fetches
    them — `ui.py` stays Databricks-agnostic. Returns ``(url, profile_name)``;
    profile_name is ``None`` when the user typed a URL manually.
    """
    console.print()
    console.print(Panel(description, title="ucode setup", style="bold blue", expand=False))

    if profiles:
        choices = [
            questionary.Choice(title=host, value=(host, profile_name))
            for host, profile_name in profiles
        ]
        choices.append(questionary.Choice(title="Enter a different URL", value=None))
        style = questionary.Style(
            [
                ("highlighted", "fg:cyan bold"),
                ("pointer", "fg:cyan bold"),
                ("answer", "fg:cyan"),
            ]
        )
        choice = questionary.select(
            "Select workspace:", choices=choices, style=style, pointer="›", qmark=""
        ).ask()
        if isinstance(choice, tuple):
            host, profile_name = choice
            return normalize_workspace_url(host), profile_name

    while True:
        raw_value = console.input(f"  [bold]Workspace URL[/bold] {muted('›')} ").strip()
        try:
            return normalize_workspace_url(raw_value), None
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
    preselected_set = {str(item) for item in preselected} if preselected is not None else None
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
    style = questionary.Style(
        [
            ("pointer", "fg:cyan bold"),
            ("highlighted", "noinherit"),
            ("selected", "noinherit"),
            ("answer", "fg:cyan"),
        ]
    )
    preselected_set = {str(item) for item in preselected} if preselected is not None else set()
    choices = [
        questionary.Choice(title=option_label, value=value, checked=value in preselected_set)
        for value, option_label in options
    ]
    instruction = "(space to toggle, enter to confirm)"
    if searchable:
        instruction = "(type to filter, space to toggle, enter to confirm)"
    answer = questionary.checkbox(
        prompt,
        choices=choices,
        style=style,
        pointer="›",
        qmark="",
        instruction=instruction,
        use_search_filter=searchable,
        use_jk_keys=not searchable,
    ).ask()
    return None if answer is None else list(answer)


def prompt_for_text(prompt: str, *, default: str | None = None) -> str | None:
    """Free-text prompt, used when model discovery found nothing to pick from.

    Returns the trimmed input, ``default`` on an empty answer, or None when there is no
    default and the user submits nothing (or closes stdin).

    A default is shown as ``[value] (enter to accept)`` rather than the bare ``[value]``: bracketed
    text alone reads as a format example as easily as a value that will be used, so it invited
    retyping what pressing enter would already pick.

    The whole bracketed hint is markup-escaped, brackets included. Rich reads
    ``[coding-agents-tiered-routing]`` as a style tag and prints nothing for it, so an unescaped
    word-like default vanished from the prompt entirely — numeric ones like ``[80]`` are not valid
    tags and survived, which is why this looked fine wherever it was checked.
    """
    hint = f" {escape(f'[{default}]')} (enter to accept)" if default else ""
    while True:
        try:
            raw_value = console.input(f"{label(prompt)}{muted(hint)} {muted('›')} ").strip()
        except EOFError:
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
    """
    hint = f" {escape(f'[{default * 100:g}]')} (enter to accept)" if default is not None else ""
    while True:
        try:
            raw_value = console.input(f"{label(prompt)}{muted(hint)} {muted('› ')}").strip()
        except EOFError:
            if default is not None:
                return default
            raise
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
    style = questionary.Style(
        [
            ("pointer", "fg:cyan bold"),
            ("highlighted", "noinherit"),
            ("selected", "noinherit"),
            ("answer", "fg:cyan"),
        ]
    )
    choices = [questionary.Choice(title=label, value=value) for value, label in options]
    answer = questionary.select(
        prompt,
        choices=choices,
        style=style,
        pointer="›",
        qmark="",
        instruction="(type to filter, arrow keys to move)" if searchable else "(use arrow keys)",
        use_search_filter=searchable,
        use_jk_keys=not searchable,
    ).ask()
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
