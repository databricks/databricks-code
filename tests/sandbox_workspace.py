"""A fake Databricks workspace for the `ucode configure` sandbox.

Serves the REST surface `ucode configure` touches, from mutable in-memory state, and
records every call. Any URL the router does not recognize is recorded as UNROUTED and
returned as an error, so a flow that reaches an unmodelled endpoint is visible rather
than silently degraded.

Server-shaped payloads only: manifests go out with proto enum names
(CODING_AGENT_CLAUDE_CODE, ...) so the real `normalize_managed_config` runs.
"""

from __future__ import annotations

import copy
import json
from urllib.parse import urlparse

WORKSPACE = "https://sandbox.cloud.databricks.com"

PUBLISHED_MANIFEST = {
    "name": "coding-agent-configs/sandbox-1",
    "workspace_id": 1653573648247579,
    "default_agent": "CODING_AGENT_CLAUDE_CODE",
    "enabled_agents": [
        {
            "agent": "CODING_AGENT_CLAUDE_CODE",
            "config": {
                "model_config": {
                    "claude": {
                        "default_model": "system.ai.claude-opus-4-8",
                        "models": {
                            "default_opus_model": "system.ai.claude-opus-4-8",
                            "default_sonnet_model": "system.ai.claude-sonnet-4-6",
                            "default_haiku_model": "system.ai.claude-haiku-4-5",
                        },
                    }
                },
            },
        },
    ],
    "mcp_servers": [{"name": "system.ai.github", "type": "MCP_SERVER_TYPE_UC_SERVICE"}],
    "skills": {"names": ["system.ai.pdf-extraction"]},
    "tracing": {"table": "main.default.ucode_traces"},
}

CLAUDE_MODELS = [
    "system.ai.claude-opus-4-8",
    "system.ai.claude-sonnet-4-6",
    "system.ai.claude-haiku-4-5",
]

CLAUDE_FAMILIES = {
    "opus": CLAUDE_MODELS[0],
    "sonnet": CLAUDE_MODELS[1],
    "haiku": CLAUDE_MODELS[2],
}

CONFIG_COLLECTION = "/coding-agent-configs"

RECOMMENDATION = {
    "agent": "CODING_AGENT_CLAUDE_CODE",
    "model": CLAUDE_MODELS[0],
    "current_spend": {"amount": "0", "currency": "USD"},
}

EXPERIMENT = {
    "experiment_id": "4242",
    "name": "/Users/sandbox@example.com/ucode-traces",
    "tags": [
        {
            "key": "mlflow.experiment.databricksTraceDestinationPath",
            "value": "main.default.ucode_traces",
        }
    ],
}

_AGENT_BINARIES = frozenset(
    {"claude", "codex", "gemini", "opencode", "copilot", "pi", "cursor-agent"}
)

MODEL_SERVICES = [
    *CLAUDE_MODELS,
    "system.ai.gpt-5-codex",
    "system.ai.gemini-2-5-pro",
    "system.ai.llama-4-maverick",
]


class FakeWorkspace:
    def __init__(
        self,
        *,
        admin: bool | None = True,
        published: dict | None = None,
        config_read_error: str | None = None,
        feature_disabled: bool = False,
        write_error: str | None = None,
        experiment: bool = True,
        warehouses: list[dict] | None = None,
    ):
        self.experiment = experiment
        self.warehouses = (
            warehouses
            if warehouses is not None
            else [{"id": "sandbox-warehouse-1", "name": "Sandbox", "state": "RUNNING"}]
        )
        self.admin = admin
        self.published = copy.deepcopy(published) if published else None
        self.config_read_error = config_read_error
        self.feature_disabled = feature_disabled
        self.write_error = write_error
        self.launches: list[list[str]] = []
        self.calls: list[tuple[str, str]] = []
        self.probes: list[str] = []
        self.recommendation = copy.deepcopy(RECOMMENDATION)
        self.recommendations = 0
        self.writes: list[tuple[str, dict | None]] = []
        self.unrouted: list[tuple[str, str]] = []

    def get(self, url, token, *, timeout=10, max_retries=0):
        path = urlparse(url).path
        self.calls.append(("GET", path))

        if path == "/api/2.0/preview/scim/v2/Me":
            if self.admin is None:
                return None, "HTTP 500 Internal Server Error"
            groups = [{"display": "admins"}] if self.admin else [{"display": "users"}]
            return {"userName": "sandbox@example.com", "groups": groups}, None

        if "coding-agent-config" in path:
            if self.feature_disabled:
                return None, (
                    'HTTP 400 Bad Request: {"error_code": "FEATURE_DISABLED", "message": '
                    '"Coding agent configs are not enabled for this workspace."}'
                )
            if self.config_read_error:
                return None, self.config_read_error
            configs = [self.published] if self.published else []
            return {"coding_agent_configs": configs}, None

        if "workspace-metrics/budgets" in path:
            return {"workspace_ai_gateway_budgets": []}, None

        if path == "/api/2.1/unity-catalog/model-services":
            return {"model_services": [{"name": name} for name in MODEL_SERVICES]}, None

        if path == "/api/2.0/sql/warehouses":
            return {"warehouses": list(self.warehouses)}, None

        return self._unrouted("GET", url)

    def send(self, method, url, token, payload, *, timeout=10, allow_empty_body=False):
        path = urlparse(url).path
        self.calls.append((method, path))

        if path.endswith(":recommendModel"):
            self.recommendations += 1
            return copy.deepcopy(self.recommendation), None

        if path == "/api/2.0/mlflow/experiments/search":
            return {"experiments": [EXPERIMENT] if self.experiment else []}, None

        if path.endswith(CONFIG_COLLECTION) or "/coding-agent-configs/" in path:
            self.writes.append((method, payload))
            if self.write_error:
                return None, self.write_error
            if method in ("POST", "PATCH"):
                self.published = copy.deepcopy(payload or {})
                self.published.setdefault("name", "coding-agent-configs/sandbox-1")
                return self.published, None
            if method == "DELETE":
                self.published = None
                return None, None
        return self._unrouted(method, url)

    def run(self, argv, **kwargs):
        """Fake every subprocess ucode shells out to. Unknown invocations fail loudly."""
        import subprocess

        cmd = " ".join(argv) if isinstance(argv, (list, tuple)) else str(argv)
        self.calls.append(("CLI", cmd))

        def done(stdout="", code=0):
            return subprocess.CompletedProcess(argv, code, stdout=stdout, stderr="")

        head = argv[0] if isinstance(argv, (list, tuple)) and argv else ""
        if str(head).rsplit("/", 1)[-1] in _AGENT_BINARIES:
            self.probes.append(cmd)
            return done("sandbox ok")

        if "auth profiles" in cmd:
            return done(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "name": "DEFAULT",
                                "host": WORKSPACE,
                                "auth_type": "databricks-cli",
                                "valid": True,
                            }
                        ]
                    }
                )
            )
        if "auth token" in cmd:
            return done(json.dumps({"access_token": "sandbox-token", "token_type": "Bearer"}))
        if "auth login" in cmd or "aitools install" in cmd:
            return done()
        if "--version" in cmd:
            return done("Databricks CLI v0.240.0")
        raise AssertionError(f"sandbox: unexpected subprocess call: {cmd}")

    def get_bytes(self, url, token, *, timeout=10):
        self.calls.append(("GET-BYTES", urlparse(url).path))
        return None, "sandbox: binary fetch not modelled"

    def _unrouted(self, method, url):
        self.unrouted.append((method, url))
        return None, f"sandbox: UNROUTED {method} {url}"

    def report(self) -> dict:
        return {
            "calls": [f"{m} {p}" for m, p in self.calls],
            "writes": [{"method": m, "payload": p} for m, p in self.writes],
            "unrouted": [f"{m} {u}" for m, u in self.unrouted],
            "recommendations": self.recommendations,
            "agent_probes": self.probes,
            "launches": self.launches,
            "server_published": self.published,
        }


def install(ws: FakeWorkspace) -> None:
    """Patch the HTTP and subprocess seams in ucode.databricks. Call before importing ucode.cli."""
    import ucode.databricks as db

    db._http_get_json = ws.get
    db._http_send_json = ws.send
    db._http_get_bytes = ws.get_bytes

    db.ensure_databricks_cli_version = lambda *a, **k: None
    db.databricks_cli_version = lambda *a, **k: (0, 240, 0)
    db.install_databricks_cli = lambda *a, **k: None
    db.upgrade_databricks_cli = lambda *a, **k: False
    db.install_ai_tools = lambda *a, **k: None
    db.ensure_databricks_auth = lambda *a, **k: None
    db.run_databricks_login = lambda *a, **k: None
    db.has_valid_databricks_auth = lambda *a, **k: True
    db.get_databricks_token = lambda *a, **k: "sandbox-token"
    db.ensure_pat_bearer = lambda *a, **k: True

    db.run = ws.run

    import subprocess

    subprocess.run = ws.run

    import shutil

    import ucode.launcher as launcher

    launcher.exec_or_spawn = lambda argv: ws.launches.append(list(argv))

    from ucode.agents import claude as claude_mod

    claude_mod._ensure_mlflow_cli = lambda *a, **k: True
    claude_mod._uv_tool_mlflow_path = lambda *a, **k: "/usr/local/bin/mlflow"
    _real_which = shutil.which
    _faked_binaries = (*_AGENT_BINARIES, "databricks")

    def which(cmd, *args, **kwargs):
        if cmd in _faked_binaries:
            return f"/usr/local/bin/{cmd}"
        return _real_which(cmd, *args, **kwargs)

    shutil.which = which

    db.list_model_services = lambda *a, **k: (list(MODEL_SERVICES), None)
    db.discover_claude_models = lambda *a, **k: (dict(CLAUDE_FAMILIES), None)
    db.discover_claude_models_unbucketed = lambda *a, **k: (list(CLAUDE_MODELS), None)
    db.fetch_ai_gateway_claude_models = lambda *a, **k: dict(CLAUDE_FAMILIES)
    db.discover_codex_models = lambda *a, **k: (["system.ai.gpt-5-codex"], None)
    db.discover_gemini_models = lambda *a, **k: (["system.ai.gemini-2-5-pro"], None)
    db.fetch_codex_models = lambda *a, **k: ["system.ai.gpt-5-codex"]
    db.fetch_gemini_models = lambda *a, **k: ["system.ai.gemini-2-5-pro"]
    db.ensure_ai_gateway = lambda *a, **k: None
    db.list_model_provider_services = lambda *a, **k: ([], None)
    db.list_mcp_services = lambda *a, **k: ([], None)
    db.list_all_mcp_services = lambda *a, **k: ([], None)
    db.list_workspace_budgets = lambda *a, **k: ([], None)
    db.model_service_exists = lambda *a, **k: (True, None)
    db.clear_model_services_cache()


def guard_privileged_writes(home, *, os_managed: dict | None = None) -> None:
    """Refuse OS-managed (sudo) config writes and keep the real machine's out of the sandbox.

    The OS-managed settings paths are absolute (/etc/claude-code/..., /Library/...), so without
    redirecting them a developer machine that really has enterprise-managed Claude settings leaks
    into every scenario. Points them under the sandbox HOME instead; `os_managed` seeds one
    deliberately for the scenario that exercises the conflict path.
    """
    import ucode.managed_files as managed_files
    from ucode.agents import claude as claude_mod
    from ucode.agents import codex as codex_mod

    def reject(path, _text):
        raise AssertionError(f"sandbox: attempted privileged write to {path}")

    managed_files._sudo_replace = reject

    os_dir = home / "os-managed"
    os_dir.mkdir(parents=True, exist_ok=True)
    claude_path = os_dir / "claude-managed-settings.json"
    codex_path = os_dir / "codex-managed-config.toml"
    claude_mod._managed_settings_path = lambda: claude_path
    codex_mod._managed_config_path = lambda: None

    for tool, content in (os_managed or {}).items():
        target = {"claude": claude_path, "codex": codex_path}[tool]
        target.write_text(content, encoding="utf-8")


def dump(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)
