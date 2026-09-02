"""Shared helpers for passing Codex configuration on the command line."""

from __future__ import annotations

import tomlkit


def _toml_value(value: str | int | float | bool | list[object] | dict[str, object]) -> str:
    if isinstance(value, dict):
        item = tomlkit.inline_table()
        item.update(value)
        return item.as_string()
    if isinstance(value, list) and any(isinstance(entry, dict) for entry in value):
        wrapper = tomlkit.inline_table()
        wrapper["value"] = value
        rendered = wrapper.as_string()
        return rendered.removeprefix("{value = ").removesuffix("}")
    return tomlkit.item(value).as_string()


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
