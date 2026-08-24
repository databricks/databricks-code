"""End-to-end managed-config workflow with only external systems stubbed.

Exercises the real CLI orchestration, draft/cache persistence, manifest serialization,
managed-state resolution, and Claude config writer across setup, local testing, publishing, and a
normal workspace-managed launch.
"""

from __future__ import annotations

import contextlib
import json
from unittest.mock import patch

from typer.testing import CliRunner

import ucode.managed_config as managed_config
import ucode.managed_wizard as managed_wizard
from ucode.agents import claude
from ucode.cli import app

runner = CliRunner()

WORKSPACE = "https://workspace.example.com"
PUBLISHED_MODEL = "system.ai.claude-opus-4-8"
UNPUBLISHED_MODEL = "system.ai.claude-opus-4-9"


def _manifest(model: str) -> dict:
    return {
        "default_agent": "claude",
        "enabled_agents": {
            "claude": {
                "model_config": {
                    "default_model": model,
                    "models": {"default_opus_model": model},
                }
            }
        },
    }


def _configured_opus(settings: dict) -> str:
    return settings["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"].removesuffix("[1m]")


def test_setup_local_apply_and_normal_launch_keep_sources_separate(tmp_path, monkeypatch):
    state = {
        "workspace": WORKSPACE,
        "profile": "DEFAULT",
        "available_tools": ["claude"],
        "base_urls": {"claude": f"{WORKSPACE}/ai-gateway/anthropic"},
        "claude_models": {"opus": PUBLISHED_MODEL},
        "all_claude_models": [PUBLISHED_MODEL, UNPUBLISHED_MODEL],
        "managed_configs": {},
    }
    manifest_path = tmp_path / "managed-config.json"
    manifest_path.write_text(json.dumps(_manifest(PUBLISHED_MODEL)), encoding="utf-8")
    settings_path = tmp_path / "claude" / "ucode-settings.json"

    monkeypatch.setattr(managed_config, "MANAGED_STATE_PATH", tmp_path / "managed-state.json")
    monkeypatch.setattr(managed_config, "MANAGED_CACHE_DIR", tmp_path / "managed-cache")
    monkeypatch.setattr(
        managed_config, "LEGACY_MANAGED_CACHE_PATH", tmp_path / "legacy-managed-cache.json"
    )
    monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", settings_path)
    monkeypatch.setattr(claude, "CLAUDE_BACKUP_PATH", tmp_path / "claude-settings.backup.json")

    published_payload: dict = {}
    launches: list[str] = []

    def publish(_workspace, _token, payload):
        published_payload.update(payload)
        return {"name": "coding-agent-configs/test"}, None

    patches = [
        patch("ucode.cli.install_databricks_cli"),
        patch("ucode.cli.load_state", return_value=state),
        patch.object(managed_wizard, "load_state", return_value=state),
        patch("ucode.cli.apply_pat_environment"),
        patch("ucode.cli.ensure_bootstrap_dependencies"),
        patch("ucode.cli.ensure_provider_state", return_value=state),
        patch("ucode.cli._require_local_config_admin"),
        patch("ucode.cli.configure_shared_state", return_value=state),
        patch("ucode.cli._fetch_budget_recommendation", return_value=None),
        patch("ucode.cli._register_managed_mcp_servers"),
        patch("ucode.cli._apply_managed_skills"),
        patch("ucode.cli.launch_agent", side_effect=lambda tool, *_args: launches.append(tool)),
        patch.object(claude, "managed_settings_model_overrides", return_value=None),
        patch.object(claude, "agent_version", return_value="test"),
        patch.object(claude, "ucode_version", return_value="test"),
        patch.object(managed_wizard, "ensure_databricks_auth"),
        patch.object(managed_wizard, "get_databricks_token", return_value="token"),
        patch.object(managed_wizard, "is_workspace_admin", return_value=True),
        patch.object(managed_wizard, "get_managed_config", return_value=(None, None)),
        patch.object(managed_wizard, "create_coding_agent_config", side_effect=publish),
    ]
    with contextlib.ExitStack() as stack:
        for managed_patch in patches:
            stack.enter_context(managed_patch)

        setup_result = runner.invoke(app, ["setup", "--from-file", str(manifest_path)])
        assert setup_result.exit_code == 0, setup_result.output
        assert managed_config.load_managed_state(WORKSPACE) == _manifest(PUBLISHED_MODEL)

        local_result = runner.invoke(app, ["claude", "--local"])
        assert local_result.exit_code == 0, local_result.output
        local_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        assert _configured_opus(local_settings) == PUBLISHED_MODEL

        apply_result = runner.invoke(app, ["apply", "--yes"])
        assert apply_result.exit_code == 0, apply_result.output
        published = managed_config.normalize_managed_config(published_payload)
        assert published["enabled_agents"]["claude"]["model_config"]["default_model"] == (
            PUBLISHED_MODEL
        )

        # The admin continues editing after publication. A normal developer launch must still use
        # the workspace-published snapshot, not this newer local draft.
        managed_config.save_managed_state(WORKSPACE, _manifest(UNPUBLISHED_MODEL))
        with (
            patch.dict("os.environ", {managed_config.MANAGED_CONFIG_ENV_VAR: "1"}),
            patch.object(managed_config, "get_databricks_token", return_value="token"),
            patch.object(managed_config, "get_managed_config", return_value=(published, None)),
        ):
            normal_result = runner.invoke(app, ["claude"])

        assert normal_result.exit_code == 0, normal_result.output

    normal_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert _configured_opus(normal_settings) == PUBLISHED_MODEL
    assert managed_config.load_managed_state(WORKSPACE) == _manifest(UNPUBLISHED_MODEL)
    assert managed_config.load_managed_cache(WORKSPACE) == published
    assert launches == ["claude", "claude"]
