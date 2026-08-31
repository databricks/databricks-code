"""Parsing helpers shared by coding-agent launchers."""

from __future__ import annotations


def explicit_model_arg_value(tool_args: list[str]) -> str | None:
    """Return the last model selected before the harness's ``--`` separator."""
    model: str | None = None
    index = 0
    while index < len(tool_args):
        arg = tool_args[index]
        if arg == "--":
            break
        if arg in {"--model", "-m"}:
            if index + 1 < len(tool_args) and not tool_args[index + 1].startswith("-"):
                model = tool_args[index + 1]
                index += 1
        elif arg.startswith("--model="):
            value = arg.partition("=")[2]
            if value:
                model = value
        index += 1
    return model


def has_explicit_model_arg(tool_args: list[str]) -> bool:
    """Return whether the harness receives a ``--model`` option before ``--``."""
    for arg in tool_args:
        if arg == "--":
            return False
        if arg in {"--model", "-m"} or arg.startswith("--model="):
            return True
    return False
