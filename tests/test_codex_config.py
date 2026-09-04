from __future__ import annotations

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

        key, rendered_provider = provider_override.split("=", 1)
        parsed = tomlkit.parse(f"{key} = {rendered_provider}")
        provider = parsed["model_providers"]["ucode-databricks"]
        assert provider["auth"]["timeout_ms"] == 5000
        assert provider["http_headers"]["User-Agent"].startswith("ucode/0.1.0")

    def test_renders_nested_tables_inside_arrays(self):
        args = codex_config_args(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Read",
                            "hooks": [{"type": "command", "command": "route"}],
                        }
                    ]
                }
            }
        )

        assert args[0] == "--config"
        key, rendered_hooks = args[1].split("=", 1)
        parsed = tomlkit.parse(f"{key} = {rendered_hooks}")
        hook = parsed["hooks"]["PreToolUse"][0]
        assert hook["matcher"] == "Read"
        assert hook["hooks"][0]["command"] == "route"
