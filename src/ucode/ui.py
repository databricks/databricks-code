"""Rich/questionary presentation primitives. No project knowledge."""

from __future__ import annotations

import itertools
import sys
import textwrap
import threading
import time
from contextlib import contextmanager
from datetime import timedelta

import questionary
from rich.console import Console
from rich.panel import Panel

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
def spinner(message: str):
    if not sys.stdout.isatty():
        yield
        return

    stop_event = threading.Event()

    def spin() -> None:
        for frame in itertools.cycle("|/-\\"):
            if stop_event.is_set():
                break
            sys.stdout.write(f"\r\033[2m{frame}\033[0m {message}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * (len(message) + 4) + "\r")
        sys.stdout.flush()

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1)


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


def prompt_for_tools(available: list[tuple[str, str]]) -> list[str]:
    """Multi-select picker for coding agents.

    `available` is [(tool_id, display_name), ...]. Returns the chosen tool_ids.
    All options are checked by default so hitting Enter selects everything.
    Returns [] if the user submits an empty selection.
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
    choices = [
        questionary.Choice(title=display, value=tool_id, checked=True)
        for tool_id, display in available
    ]
    answer = questionary.checkbox(
        "Select coding agents to configure:",
        choices=choices,
        style=style,
        pointer="›",
        qmark="",
        instruction="(space to toggle, enter to confirm)",
    ).ask()
    return list(answer) if answer else []


def prompt_for_selection(prompt: str, options: list[tuple[str, str]]) -> str | None:
    """Single-select arrow-key picker. `options` is [(value, label), ...].

    The prompt renders above the choices (questionary convention). Returns the
    chosen value, or None if the user cancels (Ctrl-C / empty).
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
        instruction="(use arrow keys)",
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
