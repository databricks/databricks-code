"""Shared helpers for passing Codex configuration on the command line."""

from __future__ import annotations

import tomlkit
from tomlkit.items import Item


def _toml_item(value: object) -> Item:
    """Convert nested Python values without creating regular TOML tables."""
    if isinstance(value, dict):
        item = tomlkit.inline_table()
        for key, entry in value.items():
            item[key] = _toml_item(entry)
        return item
    if isinstance(value, list):
        item = tomlkit.array()
        for entry in value:
            item.append(_toml_item(entry))
        return item
    return tomlkit.item(value)


def _toml_value(value: str | int | float | bool | list[object] | dict[str, object]) -> str:
    return _toml_item(value).as_string()


def codex_config_args(config: dict) -> list[str]:
    """Render a Codex config layer as repeatable ``--config`` overrides."""
    args: list[str] = []
    for key, value in config.items():
        # These maps contain named entries. Override each entry individually so
        # the rest of the user's base map remains intact.
        if key in {"hooks", "model_providers"} and isinstance(value, dict):
            for entry_name, entry_config in value.items():
                args.extend(
                    [
                        "--config",
                        f"{key}.{entry_name}={_toml_value(entry_config)}",
                    ]
                )
        else:
            args.extend(["--config", f"{key}={_toml_value(value)}"])
    return args
