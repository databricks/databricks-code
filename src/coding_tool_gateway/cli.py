#!/usr/bin/env python3
"""CLI entry point for coding-gateway."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import itertools
import threading
import time
import textwrap
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.panel import Panel


APP_DIR = Path.home() / ".coding-gateway"
STATE_PATH = APP_DIR / "state.json"

CODEX_CONFIG_DIR = Path.home() / ".codex"
CODEX_CONFIG_PATH = CODEX_CONFIG_DIR / "config.toml"
CLAUDE_CONFIG_DIR = Path.home() / ".claude"
CLAUDE_SETTINGS_PATH = CLAUDE_CONFIG_DIR / "settings.json"
GEMINI_CONFIG_DIR = Path.home() / ".gemini"
GEMINI_ENV_PATH = GEMINI_CONFIG_DIR / ".env"

CODEX_BACKUP_PATH = APP_DIR / "codex-config.backup.toml"
CLAUDE_BACKUP_PATH = APP_DIR / "claude-settings.backup.json"
GEMINI_BACKUP_PATH = APP_DIR / "gemini-env.backup"

UNIX_DATABRICKS_INSTALL_URL = (
    "https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh"
)
WINDOWS_DATABRICKS_INSTALL_URL = (
    "https://raw.githubusercontent.com/databricks/setup-cli/main/install.ps1"
)
AI_GATEWAY_V2_DOCS_URL = "https://docs.databricks.com/aws/en/ai-gateway/overview-beta"
TOKEN_REFRESH_INTERVAL_SECONDS = 1800
SCRUBBED_DATABRICKS_ENV_VARS = (
    "DATABRICKS_TOKEN",
    "DATABRICKS_CLIENT_ID",
    "DATABRICKS_CLIENT_SECRET",
    "DATABRICKS_USERNAME",
    "DATABRICKS_PASSWORD",
    "DATABRICKS_AUTH_TYPE",
)

TOOL_SPECS = {
    "codex": {
        "binary": "codex",
        "package": "@openai/codex",
        "display": "Codex",
        "config_path": CODEX_CONFIG_PATH,
        "backup_path": CODEX_BACKUP_PATH,
    },
    "claude": {
        "binary": "claude",
        "package": "@anthropic-ai/claude-code",
        "display": "Claude Code",
        "config_path": CLAUDE_SETTINGS_PATH,
        "backup_path": CLAUDE_BACKUP_PATH,
    },
    "gemini": {
        "binary": "gemini",
        "package": "@google/gemini-cli",
        "display": "Gemini CLI",
        "config_path": GEMINI_ENV_PATH,
        "backup_path": GEMINI_BACKUP_PATH,
    },
}
TOOL_ALIASES = {
    "codex": "codex",
    "claude": "claude",
    "claude-code": "claude",
    "gemini": "gemini",
    "gemini-cli": "gemini",
}
DEFAULT_TOOL = "codex"
DEFAULT_SELECTED_MODELS = {
    "claude": "databricks-claude-opus-4-6",
    "gemini": "databricks-gemini-3-1-pro",
}
USAGE_BREAKDOWN_DAYS = 7
USAGE_SUMMARY_DAYS = 30
BUNDLE_VERSION = 1

_dry_run = False


console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)


def print_section(title: str) -> None:
    console.print()
    console.print(Panel(title, style="bold blue", expand=False))


def print_kv(key: str, val: str) -> None:
    console.print(f"  [bold]{key}:[/bold] [cyan]{val}[/cyan]")


def print_note(text: str) -> None:
    console.print(f"[dim]•[/dim] {text}")


def print_success(message: str) -> None:
    console.print(f"[bold green]✔[/bold green] {message}")


def print_warning(message: str) -> None:
    console.print(f"[bold yellow]![/bold yellow] {message}")


def print_err(message: str) -> None:
    err_console.print(f"[bold red]ERROR[/bold red] {message}")


def heading(text: str) -> str:
    return f"[bold blue]{text}[/bold blue]"


def label(text: str) -> str:
    return f"[bold]{text}[/bold]"


def value(text: str) -> str:
    return f"[cyan]{text}[/cyan]"


def muted(text: str) -> str:
    return f"[dim]{text}[/dim]"


def status_badge(text: str, kind: str) -> str:
    color = {"ok": "green", "warn": "yellow", "error": "red", "info": "blue"}.get(kind, "bold")
    return f"[bold {color}]{text}[/bold {color}]"


@contextmanager
def spinner(message: str):
    if not sys.stdout.isatty():
        yield
        return

    stop_event = threading.Event()

    def spin() -> None:
        for frame in itertools.cycle("|/-\\"):
            if stop_event.is_set():
                break
            sys.stdout.write(f"\r\033[2m{frame}\033[0m {message}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * (len(message) + 4) + "\r")
        sys.stdout.flush()

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1)


def run(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: bool = False,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        args,
        check=check,
        capture_output=capture_output,
        text=text,
        env=env,
        timeout=timeout,
    )


def build_databricks_cli_env(workspace: str) -> dict[str, str]:
    env = os.environ.copy()
    env["DATABRICKS_HOST"] = workspace
    for key in SCRUBBED_DATABRICKS_ENV_VARS:
        env.pop(key, None)
    return env


def workspace_hostname(workspace: str) -> str:
    parsed = urlparse(normalize_workspace_url(workspace))
    if not parsed.hostname:
        raise RuntimeError(f"Unable to derive hostname from workspace URL: {workspace}")
    return parsed.hostname


def normalize_workspace_url(workspace: str) -> str:
    workspace = workspace.strip()
    if not workspace:
        raise ValueError("Workspace URL cannot be empty.")
    if not workspace.startswith(("http://", "https://")):
        workspace = f"https://{workspace}"
    return workspace.rstrip("/")


def normalize_tool(tool: str) -> str:
    normalized = TOOL_ALIASES.get(tool.strip().lower())
    if not normalized:
        raise RuntimeError(
            f"Unsupported tool '{tool}'. Use one of: codex, claude, gemini."
        )
    return normalized


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return hydrate_state(state)
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    if _dry_run:
        return
    try:
        state = hydrate_state(state)
        APP_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write state file: {STATE_PATH}") from exc



def hydrate_state(state: dict) -> dict:
    if not isinstance(state, dict):
        return {}

    hydrated = dict(state)
    managed_configs = hydrated.get("managed_configs")
    if not isinstance(managed_configs, dict):
        managed_configs = {}
    hydrated["managed_configs"] = managed_configs
    selected_models = hydrated.get("selected_models")
    if not isinstance(selected_models, dict):
        selected_models = {}
    hydrated["selected_models"] = selected_models

    workspace = hydrated.get("workspace")
    if workspace:
        use_ai_gateway_v2 = bool(hydrated.get("use_ai_gateway_v2"))
        hydrated["base_urls"] = build_shared_base_urls(workspace, use_ai_gateway_v2)
    else:
        hydrated["base_urls"] = {}

    return hydrated


def clear_state() -> None:
    try:
        if STATE_PATH.exists():
            STATE_PATH.unlink()
    except OSError as exc:
        raise RuntimeError(f"Failed to clear state file: {STATE_PATH}") from exc


def ensure_parent_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to create directory for {path}") from exc


def backup_existing_file(config_path: Path, backup_path: Path) -> bool:
    if _dry_run:
        return False
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        if backup_path.exists():
            return True
        if not config_path.exists():
            return False
        backup_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        return True
    except OSError as exc:
        raise RuntimeError(f"Failed to back up config from {config_path}") from exc


def restore_file(config_path: Path, backup_path: Path, managed: bool) -> bool:
    try:
        if backup_path.exists():
            ensure_parent_dir(config_path)
            config_path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
            backup_path.unlink()
            return True
        if managed and config_path.exists():
            config_path.unlink()
            return True
        return False
    except OSError as exc:
        raise RuntimeError(f"Failed to restore config at {config_path}") from exc


def write_text_file(path: Path, content: str) -> None:
    if _dry_run:
        console.print(f"\n[bold]\\[dry run] {path}[/bold]\n{content}")
        return
    ensure_parent_dir(path)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write config file: {path}") from exc


def write_json_file(path: Path, payload: dict) -> None:
    content = json.dumps(payload, indent=2) + "\n"
    if _dry_run:
        console.print(f"\n[bold]\\[dry run] {path}[/bold]\n{content}")
        return
    ensure_parent_dir(path)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write config file: {path}") from exc


def install_databricks_cli() -> None:
    if shutil.which("databricks"):
        return

    system = platform.system()
    print_section("Bootstrap")
    print_warning("`databricks` was not found. Installing Databricks CLI...")

    try:
        if system == "Windows":
            run(
                [
                    "powershell",
                    "-Command",
                    f"irm {WINDOWS_DATABRICKS_INSTALL_URL} | iex",
                ],
                timeout=240,
            )
        elif shutil.which("curl"):
            run(
                ["sh", "-c", f"curl -fsSL {UNIX_DATABRICKS_INSTALL_URL} | sh"],
                timeout=240,
            )
        elif shutil.which("wget"):
            run(
                ["sh", "-c", f"wget -qO- {UNIX_DATABRICKS_INSTALL_URL} | sh"],
                timeout=240,
            )
        else:
            raise RuntimeError("Neither curl nor wget is available.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as exc:
        raise RuntimeError("Failed to install Databricks CLI automatically.") from exc

    if not shutil.which("databricks"):
        raise RuntimeError(
            "Databricks CLI install completed, but `databricks` is still not on PATH."
        )


def install_tool_binary(tool: str) -> None:
    spec = TOOL_SPECS[tool]
    binary = spec["binary"]
    package = spec["package"]

    if shutil.which(binary):
        return

    if not shutil.which("npm"):
        raise RuntimeError(
            f"`{binary}` is not installed and npm is not available to install it."
        )

    print_section("Bootstrap")
    print_warning(f"`{binary}` was not found. Installing {spec['display']}...")
    try:
        run(["npm", "install", "-g", package], timeout=300)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Failed to install {spec['display']} automatically.") from exc

    if not shutil.which(binary):
        raise RuntimeError(
            f"{spec['display']} install completed, but `{binary}` is still not on PATH."
        )


def ensure_bootstrap_dependencies(tool: str) -> None:
    install_databricks_cli()
    install_tool_binary(tool)


def has_valid_databricks_auth(workspace: str) -> bool:
    try:
        env = build_databricks_cli_env(workspace)
        result = run(
            ["databricks", "auth", "token", "--host", workspace, "--output", "json"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )
        if result.returncode != 0:
            return False
        data = json.loads(result.stdout or "{}")
        return bool(data.get("access_token"))
    except (json.JSONDecodeError, OSError, subprocess.TimeoutExpired):
        return False


def ensure_databricks_auth(workspace: str) -> None:
    with spinner("Checking Databricks auth..."):
        auth_is_valid = has_valid_databricks_auth(workspace)
    if auth_is_valid:
        print_success(f"Databricks auth already available for {workspace}")
        return

    print_section("Databricks Login")
    print_kv("Workspace", workspace)
    print_note("A browser may open for `databricks auth login`.")
    try:
        run(
            ["databricks", "auth", "login", "--host", workspace],
            env=build_databricks_cli_env(workspace),
            timeout=300,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("`databricks auth login` failed.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("`databricks auth login` timed out.") from exc

    if not has_valid_databricks_auth(workspace):
        raise RuntimeError(
            "Databricks login completed, but no access token is available yet."
        )
    print_success("Databricks authentication complete")


def fetch_ai_gateway_claude_models(workspace: str, token: str) -> dict[str, str]:
    hostname = workspace_hostname(workspace)
    request = urllib_request.Request(
        f"https://{hostname}/ai-gateway/anthropic/v1/models",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urllib_request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, urllib_error.HTTPError, json.JSONDecodeError):
        return {}

    models = [
        m["id"] for m in data.get("data", [])
        if isinstance(m.get("id"), str) and not m["id"].endswith("-anthropic")
    ]

    result = {}
    for family, key in [("opus", "opus"), ("sonnet", "sonnet"), ("haiku", "haiku")]:
        candidates = sorted(
            [m for m in models if f"databricks-claude-{family}-" in m],
            reverse=True,
        )
        if candidates:
            result[key] = candidates[0]
    return result


def detect_ai_gateway_v2(workspace: str, token: str) -> bool:
    hostname = workspace_hostname(workspace)
    request = urllib_request.Request(
        f"https://{hostname}/ai-gateway/anthropic/v1/messages",
        method="HEAD",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        urllib_request.urlopen(request, timeout=10)
        return True
    except urllib_error.HTTPError as exc:
        return exc.code != 404
    except urllib_error.URLError:
        return False


def get_databricks_token(workspace: str) -> str:
    try:
        env = build_databricks_cli_env(workspace)
        result = run(
            ["databricks", "auth", "token", "--host", workspace, "--output", "json"],
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )
        data = json.loads(result.stdout or "{}")
        token = data.get("access_token")
        if not token:
            raise RuntimeError("Databricks CLI returned no access token.")
        return token
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise RuntimeError("Failed to retrieve Databricks access token.") from exc


def discover_sql_warehouse_http_path(
    workspace: str,
    token: str,
    *,
    quiet: bool = False,
) -> str:
    hostname = workspace_hostname(workspace)
    request = urllib_request.Request(
        f"https://{hostname}/api/2.0/sql/warehouses",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )

    try:
        with urllib_request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        detail = body.strip() or f"HTTP {exc.code}"
        raise RuntimeError(f"Failed to list SQL warehouses: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Could not reach workspace hostname {hostname}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Databricks warehouse discovery returned invalid JSON.") from exc

    warehouses = payload.get("warehouses")
    if not isinstance(warehouses, list) or not warehouses:
        raise RuntimeError(
            "No SQL warehouses found in this workspace. Create one or pass `--http-path`."
        )

    running = [item for item in warehouses if isinstance(item, dict) and item.get("state") == "RUNNING"]
    chosen = running[0] if running else next(
        (item for item in warehouses if isinstance(item, dict) and item.get("id")),
        None,
    )
    if not chosen:
        raise RuntimeError("No usable SQL warehouse was returned by Databricks.")

    warehouse_id = chosen.get("id")
    if not isinstance(warehouse_id, str) or not warehouse_id.strip():
        raise RuntimeError("Databricks returned a warehouse without an ID.")

    warehouse_name = chosen.get("name")
    warehouse_state = chosen.get("state", "UNKNOWN")
    label_value = warehouse_name if isinstance(warehouse_name, str) and warehouse_name else warehouse_id
    if not quiet:
        print_note(f"Using SQL warehouse `{label_value}` ({warehouse_state}).")
    return f"/sql/1.0/warehouses/{warehouse_id}"


def run_usage_query(
    workspace: str,
    http_path: str,
    token: str,
    query: str,
) -> tuple[list[str], list[tuple]]:
    try:
        logging.getLogger("databricks.sql").setLevel(logging.ERROR)
        from databricks import sql
    except ImportError as exc:
        raise RuntimeError(
            "`databricks-sql-connector` is not installed. Install it with `pip install databricks-sql-connector`."
        ) from exc

    try:
        with sql.connect(
            server_hostname=workspace_hostname(workspace),
            http_path=http_path,
            access_token=token,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in (cursor.description or [])]
                rows = cursor.fetchall()
    except Exception as exc:
        raise RuntimeError(f"Usage query failed: {exc}") from exc

    return columns, rows


def build_usage_report_query() -> str:
    return f"""
SELECT
  current_user() AS requester_name,
  CASE
    WHEN lower(user_agent) LIKE '%codex%' THEN 'codex'
    WHEN lower(user_agent) LIKE '%claude%' THEN 'claude'
    WHEN lower(user_agent) LIKE '%gemini%' THEN 'gemini'
    ELSE 'other'
  END AS tool,
  date(event_time) AS usage_day,
  SUM(COALESCE(total_tokens, 0)) AS total_tokens_used,
  COUNT(DISTINCT request_id) AS sessions,
  MIN(event_time) AS first_event_time,
  MAX(event_time) AS last_event_time,
  CONCAT_WS(', ', SORT_ARRAY(COLLECT_SET(destination_model))) AS models
FROM system.ai_gateway.usage
WHERE event_time >= current_timestamp() - interval {USAGE_SUMMARY_DAYS} days
  AND requester = current_user()
  AND (
    lower(user_agent) LIKE '%codex%'
    OR lower(user_agent) LIKE '%claude%'
    OR lower(user_agent) LIKE '%gemini%'
  )
GROUP BY 1, 2, 3
ORDER BY usage_day DESC, tool ASC
""".strip()


def build_current_user_query() -> str:
    return "SELECT current_user() AS requester_name"


def parse_usage_rows(columns: list[str], rows: list[tuple]) -> list[dict[str, object]]:
    return [dict(zip(columns, row)) for row in rows]


def coerce_date(value_obj: object) -> date | None:
    if isinstance(value_obj, date) and not isinstance(value_obj, datetime):
        return value_obj
    if isinstance(value_obj, datetime):
        return value_obj.date()
    if isinstance(value_obj, str):
        try:
            return datetime.fromisoformat(value_obj).date()
        except ValueError:
            return None
    return None


def coerce_datetime(value_obj: object) -> datetime | None:
    if isinstance(value_obj, datetime):
        return value_obj
    if isinstance(value_obj, str):
        candidate = value_obj.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            return None
    return None


def format_token_count(token_count: int) -> str:
    value_float = float(token_count)
    if token_count >= 1_000_000_000:
        return f"{value_float / 1_000_000_000:.1f}B"
    if token_count >= 1_000_000:
        return f"{value_float / 1_000_000:.1f}M"
    if token_count >= 1_000:
        return f"{value_float / 1_000:.1f}K"
    return str(token_count)


def format_duration(duration_value: timedelta | None) -> str:
    if not duration_value or duration_value.total_seconds() <= 0:
        return "-"
    total_minutes = duration_value.total_seconds() / 60
    if total_minutes < 60:
        return f"{int(round(total_minutes))}m"
    total_hours = total_minutes / 60
    if total_hours < 10:
        return f"{total_hours:.1f}h"
    if total_hours < 24:
        return f"{round(total_hours):.0f}h"
    return f"{total_hours / 24:.1f}d"


def simplify_model_name(tool: str, model_name: str) -> str:
    normalized = (model_name or "").strip()
    if not normalized:
        return "-"

    prefix = "databricks-"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix):]

    tool_prefixes = {
        "claude": "claude-",
        "gemini": "gemini-",
        "codex": "gpt-",
    }
    tool_prefix = tool_prefixes.get(tool)
    if tool_prefix and normalized.startswith(tool_prefix):
        normalized = normalized[len(tool_prefix):]
    return normalized


def summarize_models(tool: str, raw_models: object) -> str:
    if not isinstance(raw_models, str) or not raw_models.strip():
        return "-"
    models = [
        simplify_model_name(tool, item)
        for item in raw_models.split(",")
        if item.strip()
    ]
    unique_models: list[str] = []
    for model_name in models:
        if model_name not in unique_models:
            unique_models.append(model_name)
    return ", ".join(unique_models) if unique_models else "-"


def extract_model_names(tool: str, raw_models: object) -> list[str]:
    if not isinstance(raw_models, str) or not raw_models.strip():
        return []

    unique_models: list[str] = []
    for item in raw_models.split(","):
        simplified = simplify_model_name(tool, item.strip())
        if simplified != "-" and simplified not in unique_models:
            unique_models.append(simplified)
    return unique_models


def render_box_table(headers: list[str], rows: list[list[str]], max_widths: list[int] | None = None) -> str:
    wrapped_rows: list[list[list[str]]] = []
    widths = [len(header) for header in headers]

    for row in rows:
        wrapped_row: list[list[str]] = []
        for index, cell in enumerate(row):
            raw_cell = cell if cell else "-"
            width_limit = max_widths[index] if max_widths and index < len(max_widths) else None
            if width_limit:
                cell_lines = textwrap.wrap(raw_cell, width=width_limit) or ["-"]
            else:
                cell_lines = raw_cell.splitlines() or ["-"]
            wrapped_row.append(cell_lines)
            widths[index] = max(widths[index], max(len(line) for line in cell_lines))
        wrapped_rows.append(wrapped_row)

    top = "┏" + "┳".join("━" * (width + 2) for width in widths) + "┓"
    header = "┃ " + " ┃ ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " ┃"
    middle = "┡" + "╇".join("━" * (width + 2) for width in widths) + "┩"
    bottom = "└" + "┴".join("─" * (width + 2) for width in widths) + "┘"

    body_lines: list[str] = []
    for wrapped_row in wrapped_rows:
        row_height = max(len(cell_lines) for cell_lines in wrapped_row)
        for line_index in range(row_height):
            body_lines.append(
                "│ "
                + " │ ".join(
                    (
                        wrapped_row[column_index][line_index]
                        if line_index < len(wrapped_row[column_index])
                        else ""
                    ).ljust(widths[column_index])
                    for column_index in range(len(headers))
                )
                + " │"
            )

    return "\n".join([top, header, middle, *body_lines, bottom])


def empty_tool_day(tool: str, usage_day: date) -> dict[str, object]:
    return {
        "tool": tool,
        "usage_day": usage_day,
        "total_tokens_used": 0,
        "sessions": 0,
        "first_event_time": None,
        "last_event_time": None,
        "models": "-",
    }


def build_tool_breakdown_rows(records: list[dict[str, object]], tool: str) -> list[list[str]]:
    today = date.today()
    rows_by_day: dict[date, dict[str, object]] = {}
    for record in records:
        if record.get("tool") != tool:
            continue
        usage_day = coerce_date(record.get("usage_day"))
        if usage_day:
            rows_by_day[usage_day] = record

    rendered_rows: list[list[str]] = []
    for day_offset in range(USAGE_BREAKDOWN_DAYS):
        usage_day = today - timedelta(days=day_offset)
        record = rows_by_day.get(usage_day) or empty_tool_day(tool, usage_day)
        first_event_time = coerce_datetime(record.get("first_event_time"))
        last_event_time = coerce_datetime(record.get("last_event_time"))
        duration = None
        if first_event_time and last_event_time:
            duration = last_event_time - first_event_time
        token_total = int(record.get("total_tokens_used") or 0)
        session_total = int(record.get("sessions") or 0)
        rendered_rows.append(
            [
                usage_day.strftime("%m-%d"),
                usage_day.strftime("%a"),
                format_token_count(token_total) if token_total else "-",
                str(session_total) if session_total else "-",
                format_duration(duration),
                summarize_models(tool, record.get("models")),
            ]
        )

    return rendered_rows


def find_requester_name(
    workspace: str,
    http_path: str,
    token: str,
    records: list[dict[str, object]],
) -> str:
    for record in records:
        requester_name = record.get("requester_name")
        if isinstance(requester_name, str) and requester_name.strip():
            return requester_name.strip()

    columns, rows = run_usage_query(workspace, http_path, token, build_current_user_query())
    parsed_rows = parse_usage_rows(columns, rows)
    if parsed_rows:
        requester_name = parsed_rows[0].get("requester_name")
        if isinstance(requester_name, str) and requester_name.strip():
            return requester_name.strip()
    return "current user"


def render_usage_summary(
    records: list[dict[str, object]],
    requester_name: str,
) -> str:
    today = date.today()
    week_start = today - timedelta(days=USAGE_BREAKDOWN_DAYS - 1)
    month_start = today - timedelta(days=USAGE_SUMMARY_DAYS - 1)

    daily_total = 0
    weekly_total = 0
    monthly_total = 0
    active_tools_last_week: list[str] = []
    weekly_model_tokens: dict[str, int] = {}
    tool_labels = {
        "codex": "Codex",
        "claude": "Claude Code",
        "gemini": "Gemini CLI",
    }

    for record in records:
        usage_day = coerce_date(record.get("usage_day"))
        if not usage_day:
            continue
        token_total = int(record.get("total_tokens_used") or 0)
        tool = record.get("tool")
        if usage_day >= month_start:
            monthly_total += token_total
        if usage_day >= week_start:
            weekly_total += token_total
            if isinstance(tool, str) and tool in tool_labels and tool not in active_tools_last_week:
                active_tools_last_week.append(tool)
            if isinstance(tool, str):
                for model_name in extract_model_names(tool, record.get("models")):
                    weekly_model_tokens[model_name] = (
                        weekly_model_tokens.get(model_name, 0) + token_total
                    )
        if usage_day == today:
            daily_total += token_total

    lines = [
        heading(f"Usage Summary for {requester_name}"),
        "",
        "[bold green]✓[/bold green] Databricks AI Gateway usage",
        f"{label('Today:')} {value(format_token_count(daily_total) + ' tokens')}",
        f"{label('Last 7 days:')} {value(format_token_count(weekly_total) + ' tokens')}",
        f"{label('Last 30 days:')} {value(format_token_count(monthly_total) + ' tokens')}",
    ]
    if active_tools_last_week:
        tool_text = ", ".join(tool_labels[tool] for tool in active_tools_last_week)
        lines.append(f"{label('Active tools:')} {value(tool_text)}")
    if weekly_model_tokens:
        top_models = sorted(
            weekly_model_tokens.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )[:3]
        models_text = ", ".join(
            f"{model_name} ({format_token_count(token_total)})"
            for model_name, token_total in top_models
        )
        lines.append(f"{label('Top models this week:')} {value(models_text)}")
    return "\n".join(lines)


def build_auth_shell_command(workspace: str) -> str:
    python_expr = "import json,sys; print(json.load(sys.stdin).get('access_token', ''))"
    unset_prefix = " ".join(f"-u {key}" for key in SCRUBBED_DATABRICKS_ENV_VARS)
    return (
        f"env {unset_prefix} databricks auth token --host {workspace} --output json "
        f'| python3 -c "{python_expr}"'
    )


def build_tool_base_url(
    tool: str,
    workspace: str,
    use_ai_gateway_v2: bool,
) -> str:
    if tool == "codex":
        return (
            f"{workspace}/ai-gateway/codex/v1"
            if use_ai_gateway_v2
            else f"{workspace}/serving-endpoints/codex/v1"
        )
    if tool == "claude":
        return (
            f"{workspace}/ai-gateway/anthropic"
            if use_ai_gateway_v2
            else f"{workspace}/serving-endpoints/anthropic"
        )
    if tool == "gemini":
        return (
            f"{workspace}/ai-gateway/gemini"
            if use_ai_gateway_v2
            else f"{workspace}/serving-endpoints/gemini"
        )
    raise RuntimeError(f"Unsupported tool '{tool}'.")


def build_shared_base_urls(
    workspace: str,
    use_ai_gateway_v2: bool,
) -> dict[str, str]:
    return {
        tool: build_tool_base_url(tool, workspace, use_ai_gateway_v2)
        for tool in TOOL_SPECS
    }


def classify_tool_from_text(text: str) -> str | None:
    value_lower = text.lower()
    if "claude" in value_lower or "anthropic" in value_lower:
        return "claude"
    if "gemini" in value_lower:
        return "gemini"
    if (
        "gpt" in value_lower
        or "openai" in value_lower
        or "codex" in value_lower
        or " o1" in f" {value_lower}"
        or " o3" in f" {value_lower}"
        or " o4" in f" {value_lower}"
    ):
        return "codex"
    return None


def is_supported_databricks_model_name(name: str) -> bool:
    normalized = name.strip().lower()
    return normalized.startswith("databricks-") and "oss" not in normalized


def run_databricks_json(
    args: list[str],
    workspace: str,
    *,
    timeout: int = 30,
) -> dict | list:
    env = build_databricks_cli_env(workspace)
    result = run(
        ["databricks", *args, "-o", "json"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    if result.returncode != 0:
        error_text_raw = (result.stderr or result.stdout or "").strip()
        first_line = error_text_raw.splitlines()[0] if error_text_raw else "unknown error"
        raise RuntimeError(f"Databricks CLI command failed: {first_line}")

    try:
        return json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("Databricks CLI returned invalid JSON output.") from exc


def extract_model_like_strings(payload: object) -> set[str]:
    names: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value_obj in node.items():
                if isinstance(value_obj, str):
                    key_lower = key.lower()
                    if (
                        "model" in key_lower
                        or "entity" in key_lower
                        or key_lower in {"name", "provider"}
                    ):
                        value_clean = value_obj.strip()
                        if value_clean:
                            names.add(value_clean)
                else:
                    walk(value_obj)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return names


def discover_workspace_models(workspace: str) -> dict[str, set[str]]:
    grouped_models: dict[str, set[str]] = {
        "codex": set(),
        "claude": set(),
        "gemini": set(),
    }

    list_payload = run_databricks_json(["serving-endpoints", "list"], workspace)

    for candidate in extract_model_like_strings(list_payload):
        if not is_supported_databricks_model_name(candidate):
            continue
        tool = classify_tool_from_text(candidate)
        if tool:
            grouped_models[tool].add(candidate)

    return grouped_models


def prompt_for_model_choice(tool: str, models: list[str]) -> str:
    if not models:
        raise RuntimeError(
            f"No {TOOL_SPECS[tool]['display']} models were discovered in the workspace."
        )

    print_section(f"{TOOL_SPECS[tool]['display']} Models")
    print_note("Choose the model to launch with.")
    for index, model_name in enumerate(models, start=1):
        console.print(f"  [bold]{index}.[/bold] [cyan]{model_name}[/cyan]")

    while True:
        raw_value = console.input(f"{label('Select model')} {muted('›')} ").strip()
        if not raw_value:
            print_err("Please enter a model number.")
            continue
        if raw_value.isdigit():
            selected_index = int(raw_value)
            if 1 <= selected_index <= len(models):
                return models[selected_index - 1]
        print_err("Please enter a valid model number from the list.")


def prompt_for_model_value(tool: str) -> str:
    while True:
        model_name = console.input(
            f"{label(f'{TOOL_SPECS[tool]['display']} model')} {muted('›')} "
        ).strip()
        if model_name:
            return model_name
        print_err("Model cannot be empty.")


def resolve_selected_model(
    tool: str,
    state: dict,
    explicit_model: str | None,
    *,
    prefer_saved: bool = True,
) -> tuple[dict, str | None]:
    if tool == "codex":
        return state, None

    selected_models = dict(state.get("selected_models") or {})
    if explicit_model:
        selected_models[tool] = explicit_model
        state["selected_models"] = selected_models
        save_state(state)
        return state, explicit_model

    existing_model = selected_models.get(tool)
    if prefer_saved and existing_model:
        return state, existing_model

    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("Workspace is not configured.")

    try:
        with spinner(f"Loading {TOOL_SPECS[tool]['display']} models from workspace..."):
            grouped_models = discover_workspace_models(workspace)
        available_models = sorted(grouped_models.get(tool, set()))
    except RuntimeError as exc:
        available_models = []
        print_section(f"{TOOL_SPECS[tool]['display']} Models")
        print_warning(f"Automatic model discovery failed: {exc}")
        print_note("Enter the model name manually.")

    if available_models:
        selected_model = prompt_for_model_choice(tool, available_models)
    else:
        print_section(f"{TOOL_SPECS[tool]['display']} Model")
        print_note("No models were discovered automatically. Enter the model name manually.")
        selected_model = prompt_for_model_value(tool)

    selected_models[tool] = selected_model
    state["selected_models"] = selected_models
    save_state(state)
    return state, selected_model


def resolve_launch_model(
    tool: str,
    state: dict,
    explicit_model: str | None,
) -> tuple[dict, str | None]:
    if tool == "codex":
        return state, None

    selected_models = dict(state.get("selected_models") or {})
    if explicit_model:
        selected_models[tool] = explicit_model
        state["selected_models"] = selected_models
        save_state(state)
        return state, explicit_model

    existing_model = selected_models.get(tool)
    if existing_model:
        return state, existing_model

    default_model = DEFAULT_SELECTED_MODELS.get(tool)
    if not default_model:
        raise RuntimeError(f"No default model is configured for {tool}.")

    selected_models[tool] = default_model
    state["selected_models"] = selected_models
    save_state(state)
    return state, default_model


def render_codex_config(workspace: str, use_ai_gateway_v2: bool) -> str:
    auth_command = build_auth_shell_command(workspace)
    base_url = build_tool_base_url("codex", workspace, use_ai_gateway_v2)
    return (
        "# Managed by coding-gateway. Run `coding-gateway logout` to restore the prior config.\n"
        'profile = "default"\n'
        "\n"
        "[profiles.default]\n"
        'model_provider = "Databricks"\n'
        "\n"
        "[model_providers.Databricks]\n"
        'name = "Databricks AI Gateway"\n'
        f'base_url = "{base_url}"\n'
        'wire_api = "responses"\n'
        "\n"
        "[model_providers.Databricks.auth]\n"
        'command = "sh"\n'
        f"args = [{json.dumps('-c')}, {json.dumps(auth_command)}]\n"
        "timeout_ms = 5000\n"
        "refresh_interval_ms = 1800000\n"
    )


def render_claude_settings(
    workspace: str,
    use_ai_gateway_v2: bool,
    model: str,
    claude_models: dict[str, str] | None = None,
) -> dict:
    base_url = build_tool_base_url("claude", workspace, use_ai_gateway_v2)
    env: dict[str, str] = {
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_CUSTOM_HEADERS": "x-databricks-use-coding-agent-mode: true",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_API_KEY_HELPER_TTL_MS": "1800000",
    }
    if claude_models:
        if claude_models.get("opus"):
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = claude_models["opus"]
        if claude_models.get("sonnet"):
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = claude_models["sonnet"]
        if claude_models.get("haiku"):
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = claude_models["haiku"]
    return {"apiKeyHelper": build_auth_shell_command(workspace), "env": env}


def render_gemini_env(
    workspace: str,
    use_ai_gateway_v2: bool,
    model: str,
    token: str,
) -> str:
    base_url = build_tool_base_url("gemini", workspace, use_ai_gateway_v2)
    return (
        "# Managed by coding-gateway. Run `coding-gateway logout` to restore the prior config.\n"
        f'GEMINI_MODEL="{model}"\n'
        f'GOOGLE_GEMINI_BASE_URL="{base_url}"\n'
        'GEMINI_API_KEY_AUTH_MECHANISM="bearer"\n'
        f'GEMINI_API_KEY="{token}"\n'
    )


def build_gemini_runtime_env(
    workspace: str,
    use_ai_gateway_v2: bool,
    model: str,
    token: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env["GEMINI_MODEL"] = model
    env["GOOGLE_GEMINI_BASE_URL"] = build_tool_base_url("gemini", workspace, use_ai_gateway_v2)
    env["GEMINI_API_KEY_AUTH_MECHANISM"] = "bearer"
    env["GEMINI_API_KEY"] = token
    return env


def prompt_for_workspace() -> str:
    while True:
        raw_value = console.input(f"{label('Databricks workspace URL')} {muted('›')} ").strip()
        try:
            return normalize_workspace_url(raw_value)
        except ValueError as exc:
            print_err(str(exc))


def prompt_yes_no(prompt: str) -> bool:
    while True:
        response = console.input(f"{label(prompt)} {muted('[y/n]')} {muted('›')} ").strip().lower()
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print_err("Please answer yes or no.")


def prompt_for_choice(title: str, prompt: str, options: list[tuple[str, str]]) -> str:
    print_section(title)
    for index, (_, option_label) in enumerate(options, start=1):
        console.print(f"  [bold]{index}.[/bold] [cyan]{option_label}[/cyan]")

    while True:
        raw_value = console.input(f"{label(prompt)} {muted('›')} ").strip()
        if raw_value.isdigit():
            selected_index = int(raw_value)
            if 1 <= selected_index <= len(options):
                return options[selected_index - 1][0]
        print_err("Please enter a valid option number.")


def prompt_for_client_id() -> str:
    while True:
        client_id = console.input(
            f"{label('OAuth client ID')} {muted('›')} "
        ).strip()
        if client_id:
            return client_id
        print_err("Client ID cannot be empty.")


def prompt_for_client_secret() -> str:
    while True:
        client_secret = console.input(
            f"{label('OAuth client secret')} {muted('›')} "
        ).strip()
        if client_secret:
            return client_secret
        print_err("Client secret cannot be empty.")


def prompt_for_configuration(tool: str | None = None) -> str:
    print_section("coding-gateway Setup")
    if tool is None:
        print_note("This will configure your Databricks workspace for all supported tools.")
    else:
        print_note(
            f"This will configure {TOOL_SPECS[tool]['display']} to use your Databricks endpoint."
        )
    return prompt_for_workspace()


def mark_tool_managed(state: dict, tool: str) -> dict:
    managed_configs = dict(state.get("managed_configs") or {})
    managed_configs[tool] = True
    state["managed_configs"] = managed_configs
    state["last_tool"] = tool
    return state


def configure_shared_state(workspace: str) -> dict:
    workspace = normalize_workspace_url(workspace)
    ensure_databricks_auth(workspace)
    with spinner("Detecting AI Gateway..."):
        token = get_databricks_token(workspace)
        use_ai_gateway_v2 = detect_ai_gateway_v2(workspace, token)
    if use_ai_gateway_v2:
        print_success("AI Gateway detected — using AI Gateway endpoints")
        with spinner("Fetching available Claude models..."):
            claude_models = fetch_ai_gateway_claude_models(workspace, token)
    else:
        print_note("AI Gateway not detected — using workspace serving endpoints")
        claude_models = {}
    state = load_state()
    state.update(
        {
            "workspace": workspace,
            "use_ai_gateway_v2": use_ai_gateway_v2,
            "claude_models": claude_models,
            "base_urls": build_shared_base_urls(workspace, use_ai_gateway_v2),
        }
    )
    save_state(state)
    return state


def write_codex_tool_config(state: dict) -> dict:
    backup_existing_file(CODEX_CONFIG_PATH, CODEX_BACKUP_PATH)
    write_text_file(
        CODEX_CONFIG_PATH,
        render_codex_config(
            state["workspace"],
            bool(state.get("use_ai_gateway_v2")),
        ),
    )
    state = mark_tool_managed(state, "codex")
    save_state(state)
    return state


def write_claude_tool_config(state: dict, model: str) -> dict:
    backup_existing_file(CLAUDE_SETTINGS_PATH, CLAUDE_BACKUP_PATH)
    write_json_file(
        CLAUDE_SETTINGS_PATH,
        render_claude_settings(
            state["workspace"],
            bool(state.get("use_ai_gateway_v2")),
            model,
            state.get("claude_models") or {},
        ),
    )
    state = mark_tool_managed(state, "claude")
    save_state(state)
    return state


def write_gemini_tool_config(state: dict, model: str) -> dict:
    backup_existing_file(GEMINI_ENV_PATH, GEMINI_BACKUP_PATH)
    token = get_databricks_token(state["workspace"])
    write_text_file(
        GEMINI_ENV_PATH,
        render_gemini_env(
            state["workspace"],
            bool(state.get("use_ai_gateway_v2")),
            model,
            token,
        ),
    )
    state = mark_tool_managed(state, "gemini")
    save_state(state)
    return state


def refresh_gemini_token_once(state: dict) -> str:
    token = get_databricks_token(state["workspace"])
    model = (state.get("selected_models") or {}).get("gemini")
    if not model:
        raise RuntimeError("No Gemini model is configured.")
    write_text_file(
        GEMINI_ENV_PATH,
        render_gemini_env(
            state["workspace"],
            bool(state.get("use_ai_gateway_v2")),
            model,
            token,
        ),
    )
    return token


def refresh_gemini_env_forever(state: dict, stop_event: threading.Event) -> None:
    while not stop_event.wait(TOKEN_REFRESH_INTERVAL_SECONDS):
        try:
            refresh_gemini_token_once(state)
        except RuntimeError:
            # Avoid crashing the user's active Gemini session if refresh fails.
            continue


def configure_tool(tool: str, state: dict, model: str | None = None) -> dict:
    if tool == "codex":
        return write_codex_tool_config(state)
    if tool == "claude":
        if not model:
            raise RuntimeError("A Claude model must be selected before configuration.")
        return write_claude_tool_config(state, model)
    if tool == "gemini":
        if not model:
            raise RuntimeError("A Gemini model must be selected before configuration.")
        return write_gemini_tool_config(state, model)
    raise RuntimeError(f"Unsupported tool '{tool}'.")


def configure_all_tools(state: dict) -> dict:
    state = configure_tool("codex", state, None)
    state, claude_model = resolve_launch_model("claude", state, None)
    state = configure_tool("claude", state, claude_model)
    state, gemini_model = resolve_launch_model("gemini", state, None)
    state = configure_tool("gemini", state, gemini_model)
    return state


def ensure_provider_state(tool: str) -> dict:
    state = load_state()
    workspace = state.get("workspace")
    if workspace:
        ensure_databricks_auth(workspace)
        return state

    workspace = prompt_for_configuration(tool)
    return configure_shared_state(workspace)


def configure_workspace_command() -> int:
    workspace = prompt_for_configuration()
    state = configure_shared_state(workspace)
    state = configure_all_tools(state)

    print_section("Configured")
    print_kv("Workspace", state["workspace"])
    print_kv(
        "Mode",
        "Databricks AI Gateway V2"
        if state.get("use_ai_gateway_v2")
        else "Workspace serving endpoint",
    )
    print_kv("Codex config", str(TOOL_SPECS["codex"]["config_path"]))
    print_kv("Claude config", str(TOOL_SPECS["claude"]["config_path"]))
    print_kv("Gemini config", str(TOOL_SPECS["gemini"]["config_path"]))
    print_success("Workspace configuration saved for all tools")
    return 0


def configure_model_for_tool(tool: str, model: str | None) -> int:
    if tool == "codex":
        raise RuntimeError("Codex model selection is handled inside Codex itself.")

    state = ensure_provider_state(tool)
    state, resolved_model = resolve_selected_model(
        tool,
        state,
        model,
        prefer_saved=False,
    )
    state = configure_tool(tool, state, resolved_model)

    print_section("Configured")
    print_kv("Tool", TOOL_SPECS[tool]["display"])
    print_kv("Workspace", state["workspace"])
    print_kv("Model", resolved_model or "not selected")
    print_kv("Base URL", state["base_urls"][tool])
    print_kv("Config file", str(TOOL_SPECS[tool]["config_path"]))
    print_success(f"{TOOL_SPECS[tool]['display']} model configuration saved")
    print_note(f"Launch via `coding-gateway --tool {tool}`.")
    return 0


def build_mcp_http_entry(url: str, client_id: str, callback_port: int = 8080) -> dict:
    return {
        "type": "http",
        "url": url,
        "oauth": {
            "clientId": client_id,
            "callbackPort": callback_port,
        },
    }


def add_claude_mcp_server(name: str, entry: dict, client_secret: str) -> None:
    env = os.environ.copy()
    env["MCP_CLIENT_SECRET"] = client_secret
    try:
        run(
            ["claude", "mcp", "add-json", name, json.dumps(entry),
             "--client-secret"],
            env=env,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to add MCP server '{name}' via claude CLI.") from exc


def configure_mcp_command() -> int:
    state = load_state()
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError(
            "Workspace is not configured. Run `coding-gateway configure` first."
        )

    if not shutil.which("claude"):
        raise RuntimeError(
            "`claude` CLI is not installed. Install it with: npm install -g @anthropic-ai/claude-code"
        )

    ensure_databricks_auth(workspace)

    print_section("MCP Server Configuration")
    print_note("Configure Claude Code to connect to Databricks MCP servers.")
    print_note(f"Workspace: {workspace}")

    console.print()
    console.print("[bold]OAuth Credentials[/bold]")
    print_note("These will be used for all MCP servers added in this session.")
    client_id = prompt_for_client_id()
    client_secret = prompt_for_client_secret()

    state["mcp_oauth"] = {"client_id": client_id, "client_secret": client_secret}
    mcp_servers: list[dict] = list(state.get("mcp_servers") or [])

    added: list[str] = []

    while True:
        console.print()
        selection = prompt_for_choice(
            "Add MCP Server",
            "Select server type",
            [
                ("external", "External MCP server (e.g. confluence-mcp, jira-mcp)"),
                ("uc-functions", "UC Functions (Unity Catalog AI functions)"),
                ("genie", "Genie (AI/BI dashboard)"),
                ("custom", "Custom MCP server URL"),
                ("done", "Done — exit"),
            ],
        )

        if selection == "done":
            break

        if selection == "external":
            server_name = console.input(
                f"  {label('MCP server name')} {muted('(e.g. confluence-mcp, jira-mcp)')} {muted('›')} "
            ).strip()
            if not server_name:
                print_err("Server name cannot be empty.")
                continue
            url = f"{workspace}/api/2.0/mcp/external/{server_name}"
            entry_name = server_name

        elif selection == "uc-functions":
            catalog = console.input(f"  {label('Catalog name')} {muted('›')} ").strip()
            schema = console.input(f"  {label('Schema name')} {muted('›')} ").strip()
            if not catalog or not schema:
                print_err("Catalog and schema cannot be empty.")
                continue
            url = f"{workspace}/api/2.0/mcp/functions/{catalog}/{schema}"
            entry_name = f"databricks-uc-{catalog}-{schema}"

        elif selection == "genie":
            space_id = console.input(f"  {label('Genie space ID')} {muted('›')} ").strip()
            if not space_id:
                print_err("Space ID cannot be empty.")
                continue
            url = f"{workspace}/api/2.0/mcp/genie/{space_id}"
            entry_name = f"databricks-genie-{space_id}"

        elif selection == "custom":
            url = console.input(f"  {label('Full MCP server URL')} {muted('›')} ").strip()
            if not url:
                print_err("URL cannot be empty.")
                continue
            entry_name = console.input(f"  {label('Server name')} {muted('›')} ").strip()
            if not entry_name:
                print_err("Server name cannot be empty.")
                continue

        else:
            continue

        entry = build_mcp_http_entry(url, client_id)
        add_claude_mcp_server(entry_name, entry, client_secret)
        added.append(entry_name)
        mcp_servers.append(
            {
                "name": entry_name,
                "url": url,
                "client_id": client_id,
                "client_secret": client_secret,
            }
        )
        state["mcp_servers"] = mcp_servers
        save_state(state)
        print_success(f"Added {entry_name}")

    if not added:
        print_note("No MCP servers added.")
        return 0

    print_section("MCP Configured")
    for name in added:
        console.print(f"  [bold green]●[/bold green] [cyan]{name}[/cyan]")
    print_success("MCP servers registered via `claude mcp add-json`")
    print_note("Run `claude mcp list` to see all configured servers.")
    return 0


def usage() -> int:
    state = load_state()
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("Workspace is not configured. Run `coding-gateway configure` first.")
    if not bool(state.get("use_ai_gateway_v2")):
        raise RuntimeError(
            "Usage summary requires Databricks AI Gateway V2. "
            f"Run `coding-gateway configure`, enable AI Gateway V2, then try again. Docs: {AI_GATEWAY_V2_DOCS_URL}"
        )

    ensure_databricks_auth(workspace)
    with spinner("Retrieving Databricks access token..."):
        token = get_databricks_token(workspace)

    with spinner("Discovering SQL warehouse..."):
        resolved_http_path = discover_sql_warehouse_http_path(workspace, token, quiet=False)

    with spinner("Querying system.ai_gateway.usage..."):
        columns, rows = run_usage_query(
            workspace,
            resolved_http_path,
            token,
            build_usage_report_query(),
        )
    records = parse_usage_rows(columns, rows)
    requester_name = find_requester_name(workspace, resolved_http_path, token, records)

    console.print(render_usage_summary(records, requester_name))

    tool_labels = {
        "codex": "Codex",
        "claude": "Claude Code",
        "gemini": "Gemini CLI",
    }
    table_headers = ["Date", "Day", "Tokens", "Sessions", "Duration", "Models"]
    table_widths = [8, 5, 10, 8, 8, 24]

    for tool, section_title in tool_labels.items():
        console.print()
        console.print(f"[bold blue]{section_title} · Last {USAGE_BREAKDOWN_DAYS} Days[/bold blue]")
        console.print(
            render_box_table(
                table_headers,
                build_tool_breakdown_rows(records, tool),
                max_widths=table_widths,
            )
        )
    return 0


def launch_tool(tool: str, tool_args: list[str]) -> None:
    if tool == "gemini":
        raise RuntimeError("Use launch_gemini_tool for Gemini.")
    binary = TOOL_SPECS[tool]["binary"]
    os.execvp(binary, [binary, *tool_args])


def launch_gemini_tool(state: dict, tool_args: list[str]) -> None:
    token = refresh_gemini_token_once(state)
    model = (state.get("selected_models") or {}).get("gemini")
    if not model:
        raise RuntimeError("No Gemini model is configured.")
    env = build_gemini_runtime_env(
        state["workspace"],
        bool(state.get("use_ai_gateway_v2")),
        model,
        token,
    )

    stop_event = threading.Event()
    refresher = threading.Thread(
        target=refresh_gemini_env_forever,
        args=(state, stop_event),
        daemon=True,
    )
    refresher.start()

    proc = subprocess.Popen([TOOL_SPECS["gemini"]["binary"], *tool_args], env=env)
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        returncode = proc.wait()
    finally:
        stop_event.set()
        refresher.join(timeout=1)

    raise SystemExit(returncode)


def status() -> int:
    state = load_state()
    workspace = state.get("workspace")
    use_ai_gateway_v2 = bool(state.get("use_ai_gateway_v2"))
    managed_configs = state.get("managed_configs") or {}
    selected_models = state.get("selected_models") or {}

    console.print(Panel("[bold blue]coding-gateway Status[/bold blue]", expand=False))
    console.print()
    console.print(
        f"  {status_badge('Configured', 'ok') if workspace else status_badge('Not Configured', 'warn')}"
    )

    print_section("Provider")
    print_kv("Workspace URL", workspace or "not configured")
    print_kv(
        "Mode",
        "Databricks AI Gateway V2"
        if use_ai_gateway_v2
        else "Workspace serving endpoint",
    )

    print_section("Tools")
    for tool, spec in TOOL_SPECS.items():
        base_url = "not configured"
        if workspace:
            base_url = build_tool_base_url(tool, workspace, use_ai_gateway_v2)
        managed = bool(managed_configs.get(tool))
        config_path = spec["config_path"]
        print_kv("Tool", spec["display"])
        if tool != "codex":
            print_kv("Selected model", selected_models.get(tool) or "not selected")
        print_kv("Base URL", base_url)
        print_kv("Managed by Databricks", "yes" if managed else "no")
        print_kv("Config file", str(config_path) if config_path.exists() else "missing")
        console.print()

    print_section("MCP Servers (Claude Code)")
    print_note("Run `claude mcp list` to see configured MCP servers.")
    print_note("Run `coding-gateway configure mcp` to add Databricks MCP servers.")

    print_section("State")
    print_kv("State file", str(STATE_PATH) if STATE_PATH.exists() else "missing")
    print_note("Use `coding-gateway configure` to update workspace settings or tool models.")
    print_note("Use `coding-gateway configure mcp` to add Databricks MCP servers to Claude Code.")
    print_note("Use `coding-gateway logout` to clear managed configs and restore prior files.")
    return 0


def logout() -> int:
    state = load_state()
    managed_configs = state.get("managed_configs") or {}

    codex_restored = restore_file(
        CODEX_CONFIG_PATH, CODEX_BACKUP_PATH, bool(managed_configs.get("codex"))
    )
    claude_restored = restore_file(
        CLAUDE_SETTINGS_PATH, CLAUDE_BACKUP_PATH, bool(managed_configs.get("claude"))
    )
    gemini_restored = restore_file(
        GEMINI_ENV_PATH, GEMINI_BACKUP_PATH, bool(managed_configs.get("gemini"))
    )
    clear_state()

    print_section("Logout")
    print_kv("Workspace", state.get("workspace") or "none")
    print_kv("Codex config", "restored" if codex_restored else "unchanged")
    print_kv("Claude config", "restored" if claude_restored else "unchanged")
    print_kv("Gemini config", "restored" if gemini_restored else "unchanged")
    print_success("coding-gateway state cleared")
    return 0


app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
configure_app = typer.Typer(add_completion=False, no_args_is_help=False)
app.add_typer(configure_app, name="configure", help="Configure workspace and tool settings.")


@app.callback(invoke_without_command=True)
def launch(
    ctx: typer.Context,
    tool: Annotated[str, typer.Option("--tool", help="Tool to launch: codex, claude, or gemini.")] = DEFAULT_TOOL,
    model: Annotated[str | None, typer.Option("--model", help="Model override for this session.")] = None,
) -> None:
    """Launch Codex, Claude Code, or Gemini CLI via Databricks."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        _tool = normalize_tool(tool)
        ensure_bootstrap_dependencies(_tool)
        state = ensure_provider_state(_tool)
        state, resolved_model = resolve_launch_model(_tool, state, model)
        state = configure_tool(_tool, state, resolved_model)
        print_section("Launching")
        print_kv("Tool", TOOL_SPECS[_tool]["display"])
        if resolved_model:
            print_kv("Model", resolved_model)
        print_kv("Base URL", state["base_urls"][_tool])
        if _tool == "gemini":
            print_note("Gemini token refresh is managed automatically every 30 minutes while the session is running.")
        print_success(f"Starting {TOOL_SPECS[_tool]['display']}")
        if _tool == "gemini":
            launch_gemini_tool(state, ctx.args)
        else:
            launch_tool(_tool, ctx.args)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1)
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130)


@configure_app.callback(invoke_without_command=True)
def configure(
    ctx: typer.Context,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print config files without writing them.")] = False,
) -> None:
    """Configure workspace URL and auto-detect AI Gateway."""
    if ctx.invoked_subcommand is not None:
        return
    global _dry_run
    _dry_run = dry_run
    try:
        ensure_bootstrap_dependencies("codex")
        ensure_bootstrap_dependencies("claude")
        ensure_bootstrap_dependencies("gemini")
        configure_workspace_command()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1)
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130)


@configure_app.command("mcp")
def configure_mcp() -> None:
    """Add Databricks MCP servers to Claude Code."""
    try:
        configure_mcp_command()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1)
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130)


@app.command("status")
def status_cmd() -> None:
    """Show current workspace, tool configs, and saved model selections."""
    try:
        status()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1)


@app.command("logout")
def logout_cmd() -> None:
    """Clear coding-gateway state and restore backed-up tool config files."""
    try:
        logout()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1)


@app.command("usage")
def usage_cmd() -> None:
    """Show Databricks AI Gateway usage summary (last 7 days)."""
    try:
        install_databricks_cli()
        usage()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
