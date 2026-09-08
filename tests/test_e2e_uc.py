"""End-to-end tests for UC-securables discovery (now always-on).

Verifies that `configure_shared_state` discovers models via UC model-services
(`system.ai.*`) by default, falls back to the legacy per-family AI Gateway
listings when UC model-services are absent, and surfaces only `system.ai.*`
entries from the UC primitives.

Run with:
    UCODE_TEST_WORKSPACE=https://your-workspace.databricks.com \
      uv run pytest tests/test_e2e_uc.py -v
"""

from __future__ import annotations

import pytest

from ucode.cli import configure_shared_state
from ucode.databricks import (
    discover_model_services,
    list_mcp_services,
)
from ucode.state import load_state


def _has_uc_models(workspace: str, token: str) -> bool:
    claude, codex, gemini, oss, _reason = discover_model_services(workspace, token)
    return bool(claude or codex or gemini or oss)


def _all_resolved_model_ids(state: dict) -> list[str]:
    ids: list[str] = list((state.get("claude_models") or {}).values())
    ids += state.get("codex_models") or []
    ids += state.get("gemini_models") or []
    ids += state.get("oss_models") or []
    return ids


# ---------------------------------------------------------------------------
# UC discovery primitives — verify the endpoints return only `system.ai.*`
# entries (the per-family/connection filters drop everything else).
# ---------------------------------------------------------------------------


class TestDiscoverModelServicesE2E:
    def test_returns_only_system_ai_models(self, e2e_workspace, e2e_token):
        claude, codex, gemini, oss, reason = discover_model_services(e2e_workspace, e2e_token)
        if not (claude or codex or gemini or oss):
            pytest.skip(f"No system.ai.* model services on workspace: {reason}")
        non_system = sorted(
            {
                m
                for m in _all_resolved_model_ids(
                    {
                        "claude_models": claude,
                        "codex_models": codex,
                        "gemini_models": gemini,
                        "oss_models": oss,
                    }
                )
                if not m.startswith("system.ai.")
            }
        )
        assert not non_system, f"Non-system.ai entries leaked through: {non_system[:5]}"


class TestListMcpServicesE2E:
    def test_returns_only_system_ai_mcp_services(self, e2e_workspace, e2e_token):
        names, reason = list_mcp_services(e2e_workspace, e2e_token)
        if not names:
            pytest.skip(f"No system.ai.* MCP services on workspace: {reason}")
        non_system = sorted({n for n in names if not n.startswith("system.ai.")})
        assert not non_system, f"Non-system.ai entries leaked through: {non_system[:5]}"

    def test_custom_parent_filters_server_side(self, e2e_workspace, e2e_token):
        names, _ = list_mcp_services(e2e_workspace, e2e_token, parent="main.default")
        if not names:
            pytest.skip("No mcp-services in main.default on this workspace.")
        outside = sorted({n for n in names if not n.startswith("main.default.")})
        assert not outside, f"Server returned entries outside main.default: {outside[:5]}"

    def test_invalid_parent_returns_http_404(self, e2e_workspace, e2e_token):
        names, reason = list_mcp_services(
            e2e_workspace, e2e_token, parent="nope_catalog.nope_schema"
        )
        assert names == []
        assert reason and reason.startswith("HTTP 404"), (
            f"Expected HTTP 404 for bogus location, got: {reason}"
        )


# ---------------------------------------------------------------------------
# `configure_shared_state` end-to-end: UC discovery is the default, with a
# best-effort fallback to the legacy `databricks-*` listings per family.
# ---------------------------------------------------------------------------


class TestConfigureSharedStateE2E:
    def test_default_discovers_system_ai(self, monkeypatch, e2e_workspace, e2e_token):
        """No flag, no env: a workspace with UC model-services resolves
        `system.ai.*` ids and never persists a `uc_enabled` flag."""
        if not _has_uc_models(e2e_workspace, e2e_token):
            pytest.skip("Workspace has no system.ai.* model services.")

        state = configure_shared_state(e2e_workspace, force_login=False)
        assert "uc_enabled" not in load_state()
        ids = _all_resolved_model_ids(state)
        assert any(m.startswith("system.ai.") for m in ids), (
            f"Expected at least one system.ai.* model id, got: {ids[:5]}"
        )

    def test_falls_back_to_legacy_when_no_uc_models(self, monkeypatch, e2e_workspace, e2e_token):
        """A workspace without UC model-services must still configure, via the
        legacy per-family AI Gateway listings (`databricks-*` ids)."""
        if _has_uc_models(e2e_workspace, e2e_token):
            pytest.skip("Workspace has system.ai.* model services; fallback not exercised.")

        state = configure_shared_state(e2e_workspace, force_login=False)
        ids = _all_resolved_model_ids(state)
        assert ids, "Fallback discovery returned no models at all."
        assert all(not m.startswith("system.ai.") for m in ids), (
            f"Fallback unexpectedly returned UC ids: "
            f"{[m for m in ids if m.startswith('system.ai.')][:5]}"
        )
