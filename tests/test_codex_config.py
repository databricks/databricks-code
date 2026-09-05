from __future__ import annotations

import tomllib

import tomlkit

from ucode.agents import codex
from ucode.codex_config import codex_config_args

WS = "https://example.databricks.com"


class TestCodexConfigArgs:
    def test_layers_provider_overrides_without_replacing_user_config(self, monkeypatch):
        monkeypatch.setattr(codex, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.148.0")

        overlay = codex.render_overlay(
            WS,
            "gpt-5.6-luna",
            "myprof",
        )
        args = codex_config_args(overlay)

        assert args[:4] == [
            "--config",
            'model_provider="ucode-databricks"',
            "--config",
            'model="gpt-5.6-luna"',
        ]
        provider_override = args[-1]
        assert provider_override.startswith("model_providers.ucode-databricks={")
        assert "/ai-gateway/codex/v1" in provider_override
        assert 'command = "' in provider_override
        assert '"myprof"' in provider_override

    def test_renders_nested_tables_from_parsed_config_as_inline_tables(self):
        config = tomlkit.parse(
            """
[model_providers.ucode-databricks]
name = "Databricks AI Gateway"

[model_providers.ucode-databricks.http_headers]
User-Agent = "ucode/0.1.0"

[model_providers.ucode-databricks.auth]
command = "ucode"
args = ["auth-token"]
"""
        )

        args = codex_config_args(config)

        assert args[0] == "--config"
        key, rendered = args[1].split("=", 1)
        assert key == "model_providers.ucode-databricks"
        assert tomllib.loads(f"value = {rendered}")["value"] == {
            "name": "Databricks AI Gateway",
            "http_headers": {"User-Agent": "ucode/0.1.0"},
            "auth": {"command": "ucode", "args": ["auth-token"]},
        }
