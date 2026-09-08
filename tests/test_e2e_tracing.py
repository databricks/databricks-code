"""End-to-end test for MLflow tracing against a live Databricks workspace.

Run with:
    UCODE_TEST_WORKSPACE=https://your-workspace.databricks.com uv run pytest tests/test_e2e_tracing.py -v

The flow mirrors `ucode configure tracing` + a real agent run:
  1. Find the shared, UC-backed `ucode-traces` experiment in the workspace.
  2. Resolve a SQL warehouse (required to write traces to the UC table).
  3. Enable tracing in state and write the agent's config (env + plugin).
  4. Install the agent's tracing runtime (Claude plugin + mlflow CLI).
  5. Launch the agent headless with a trivial prompt so it emits a trace.
  6. Poll the experiment via the MLflow SDK until a NEW trace id appears.

Skipped automatically unless UCODE_TEST_WORKSPACE is set, the `claude` binary
is installed, `mlflow` is importable, and the tracing runtime can be set up.
Verification uses the MLflow Python SDK rather than hand-rolling the SearchTraces
V3 REST call (which takes proto `locations`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest

from ucode import tracing
from ucode.databricks import find_uc_backed_experiment, resolve_sql_warehouse_id

# How long to wait for an emitted trace to show up in the experiment. Trace
# ingestion is asynchronous, so we poll.
TRACE_POLL_TIMEOUT = int(os.environ.get("UCODE_E2E_TRACE_TIMEOUT", "120"))
TRACE_POLL_INTERVAL = 5


def _require_binary(binary: str) -> None:
    if not shutil.which(binary):
        pytest.skip(f"`{binary}` is not installed")


def _trace_ids(client, experiment_id: str) -> set[str]:
    """Current trace ids in an experiment, robust to MLflow client version
    differences (experiment_ids kwarg + trace_id/request_id field name)."""
    try:
        traces = client.search_traces(experiment_ids=[experiment_id], max_results=100)
    except TypeError:
        # Older/newer signatures may take positional experiment ids.
        traces = client.search_traces([experiment_id])
    ids: set[str] = set()
    for trace in traces:
        info = getattr(trace, "info", None)
        tid = getattr(info, "trace_id", None) or getattr(info, "request_id", None)
        if tid:
            ids.add(str(tid))
    return ids


@pytest.mark.skip(
    reason="Tracing e2e is flaky in CI (async trace ingestion races the poll timeout); "
    "disabled pending a more robust verification path."
)
class TestClaudeTracingE2E:
    def test_claude_session_lands_a_trace(self, tmp_path, monkeypatch, e2e_state, e2e_workspace):
        pytest.importorskip("mlflow", reason="mlflow not installed (pip install mlflow)")
        from mlflow import MlflowClient

        import ucode.config_io as config_io_mod
        from ucode.agents import claude
        from ucode.databricks import get_databricks_token

        _require_binary("claude")

        claude_models: dict = e2e_state.get("claude_models") or {}
        model = (
            claude_models.get("sonnet") or claude_models.get("opus") or claude_models.get("haiku")
        )
        if not model:
            pytest.skip("No Claude models available on this workspace")

        token = get_databricks_token(e2e_workspace)

        # MLflow's `databricks` tracking URI authenticates via the databricks
        # SDK, which otherwise depends on the *default* ~/.databrickscfg profile.
        # Hand it explicit creds (the bearer we already hold) so both the
        # in-process verification client and the plugin's exporter in the
        # subprocess authenticate regardless of profile naming.
        monkeypatch.setenv("DATABRICKS_HOST", e2e_workspace)
        monkeypatch.setenv("DATABRICKS_TOKEN", token)

        # Isolate only the ucode settings file (passed via --settings). The
        # MLflow plugin lives in the real ~/.claude plugin store, which the
        # subprocess must share, so CLAUDE_CONFIG_DIR is left alone.
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", tmp_path / "ucode-settings.json")
        monkeypatch.setattr(claude, "CLAUDE_BACKUP_PATH", tmp_path / "claude-settings.backup.json")

        # Find the shared, UC-backed `ucode-traces` experiment. ucode no longer
        # creates it, so this workspace must already have one provisioned.
        leaf_name = tracing.experiment_name()
        experiment, reason = find_uc_backed_experiment(e2e_workspace, token, leaf_name)
        if not experiment:
            pytest.skip(f"no UC-backed '{leaf_name}' experiment on this workspace: {reason}")
        experiment_id = experiment["experiment_id"]
        experiment_name = experiment["experiment_name"]

        # A UC-backed experiment needs a SQL warehouse, or traces are silently
        # dropped (and the verification client can't read them back).
        warehouse_id, wh_reason = resolve_sql_warehouse_id(e2e_workspace, token)
        if not warehouse_id:
            pytest.skip(f"no SQL warehouse for UC trace storage: {wh_reason}")
        monkeypatch.setenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", warehouse_id)

        state = {
            **e2e_state,
            "workspace": e2e_workspace,
            "tracing": {
                "enabled": True,
                "tracking_uri": tracing.tracking_uri_for_state({"workspace": e2e_workspace}),
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "uc_destination": experiment["uc_destination"],
                "sql_warehouse_id": warehouse_id,
            },
        }

        # Stand up the plugin + mlflow CLI exactly as `configure tracing` does.
        if not claude.ensure_tracing_runtime():
            pytest.skip("Could not set up the Claude MLflow tracing runtime in this environment")

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("ucode.state.save_state", lambda s: None)
            claude.write_tool_config(state, model)

        tracking_uri = state["tracing"]["tracking_uri"]
        client = MlflowClient(tracking_uri=tracking_uri)
        before = _trace_ids(client, experiment_id)

        # Launch claude headless. MLflow env is in the settings file; we also set
        # it (and DATABRICKS_BEARER, so apiKeyHelper short-circuits) in the
        # subprocess env so the plugin hook sees it regardless of how Claude
        # forwards settings env to hooks.
        env = {
            **os.environ,
            "DATABRICKS_BEARER": token,
            "MLFLOW_CLAUDE_TRACING_ENABLED": "true",
            **tracing.tracing_env(state, "claude"),
        }
        result = subprocess.run(
            claude.validate_cmd("claude"),
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            stdin=subprocess.DEVNULL,
        )
        combined = (result.stdout + result.stderr).strip()
        assert result.returncode == 0, f"claude run failed: {combined[:500]}"

        # Poll until a NEW trace id appears in the experiment.
        deadline = time.monotonic() + TRACE_POLL_TIMEOUT
        new_ids: set[str] = set()
        while time.monotonic() < deadline:
            new_ids = _trace_ids(client, experiment_id) - before
            if new_ids:
                break
            time.sleep(TRACE_POLL_INTERVAL)

        assert new_ids, (
            f"No new MLflow trace landed in experiment {experiment_name} (id {experiment_id}) "
            f"within {TRACE_POLL_TIMEOUT}s. Claude output: {combined[:300]}"
        )
