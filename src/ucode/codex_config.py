"""Shared helpers for passing Codex configuration on the command line."""

from __future__ import annotations

from collections.abc import Mapping

import tomlkit


def _inline_compatible(value: object) -> object:
    """Convert parsed TOML containers into inline-table-compatible values."""
    if isinstance(value, Mapping):
        return {key: _inline_compatible(entry) for key, entry in value.items()}
    if isinstance(value, list):
        return [_inline_compatible(entry) for entry in value]
    return value


def _toml_value(value: str | int | float | bool | list[object] | dict[str, object]) -> str:
    normalized = _inline_compatible(value)
    if isinstance(normalized, dict):
        item = tomlkit.inline_table()
        item.update(normalized)
        return item.as_string()
    if isinstance(normalized, list) and any(isinstance(entry, dict) for entry in normalized):
        wrapper = tomlkit.inline_table()
        wrapper["value"] = normalized
        rendered = wrapper.as_string()
        return rendered.removeprefix("{value = ").removesuffix("}")
    return tomlkit.item(normalized).as_string()


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
