"""Shared fixtures for E2E tests + global state-isolation guard."""

from __future__ import annotations

import os

import pytest

from ucode.databricks import (
    build_shared_base_urls,
    fetch_ai_gateway_claude_models,
    fetch_codex_models,
    fetch_gemini_models,
    get_databricks_token,
)
from ucode.ui import normalize_workspace_url


@pytest.fixture(autouse=True)
def _isolate_ucode_state(tmp_path, monkeypatch):
    """Redirect ucode's state file and APP_DIR to a per-test tmp dir.

    Defense in depth: even if an individual test forgets to patch save_state,
    it can never touch the developer's real ~/.ucode/state.json.
    """
    import ucode.config_io as config_io_mod
    import ucode.databricks as databricks_mod
    import ucode.state as state_mod

    state_dir = tmp_path / ".ucode"
    state_dir.mkdir()
    monkeypatch.setattr(state_mod, "STATE_PATH", state_dir / "state.json")
    monkeypatch.setattr(config_io_mod, "APP_DIR", state_dir)
    # Isolate the managed-config opt-in from the developer's own shell: leaving it set would make
    # `ucode configure` prompt for a workspace mid-test. Tests that exercise the managed path set it.
    monkeypatch.delenv("ENABLE_MANAGED_AGENT_CONFIG", raising=False)
    # The model-services listing is memoized for the life of the process, so without this a cached
    # result would leak into the next test and make a stubbed listing look like it was never called.
    databricks_mod.clear_model_services_cache()


def _workspace() -> str:
    ws = os.environ.get("UCODE_TEST_WORKSPACE", "").strip().rstrip("/")
    return normalize_workspace_url(ws) if ws else ""


@pytest.fixture(scope="session")
def e2e_workspace():
    ws = _workspace()
    if not ws:
        pytest.skip("Set UCODE_TEST_WORKSPACE=https://... to run E2E tests")
    return ws


@pytest.fixture(scope="session")
def e2e_token(e2e_workspace):
    return get_databricks_token(e2e_workspace)


@pytest.fixture(scope="session")
def e2e_state(e2e_workspace, e2e_token):
    """Full state dict mirroring what configure_shared_state produces."""
    claude_models = fetch_ai_gateway_claude_models(e2e_workspace, e2e_token)
    gemini_models = fetch_gemini_models(e2e_workspace, e2e_token)
    codex_models = fetch_codex_models(e2e_workspace, e2e_token)

    opencode_models: dict = {}
    if claude_models:
        opencode_models["anthropic"] = list(claude_models.values())
    if gemini_models:
        opencode_models["gemini"] = gemini_models

    return {
        "workspace": e2e_workspace,
        "claude_models": claude_models,
        "gemini_models": gemini_models,
        "codex_models": codex_models,
        "opencode_models": opencode_models,
        "base_urls": build_shared_base_urls(e2e_workspace),
        "managed_configs": {},
    }
