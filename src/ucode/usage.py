"""Usage report querying & rendering.

Reads from `system.ai_gateway.usage` via a Databricks SQL warehouse.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import cast

from ucode.databricks import (
    SqlWarehouse,
    apply_pat_environment,
    discover_sql_warehouses,
    ensure_databricks_auth,
    get_databricks_token,
    resolve_current_budget_spend,
    run_usage_query,
)
from ucode.state import load_state
from ucode.ui import (
    console,
    format_duration,
    format_meter,
    format_token_count,
    format_usd,
    heading,
    label,
    muted,
    print_heading,
    print_note,
    print_warning,
    render_box_table,
    spinner,
    value,
)

USAGE_BREAKDOWN_DAYS = 7
USAGE_SUMMARY_DAYS = 30


def build_usage_report_query() -> str:
    return f"""
WITH usage_events AS (
SELECT
  current_user() AS requester_name,
  CASE
    WHEN lower(user_agent) LIKE '%codex%' THEN 'codex'
    WHEN lower(user_agent) LIKE '%claude%' THEN 'claude'
    WHEN lower(user_agent) LIKE '%gemini%' THEN 'gemini'
    WHEN lower(user_agent) LIKE '%opencode%' THEN 'opencode'
    ELSE 'other'
  END AS tool,
  date(event_time) AS usage_day,
  request_id,
  event_time,
  destination_model,
  COALESCE(total_tokens, 0) AS total_tokens_used
FROM system.ai_gateway.usage
WHERE event_time >= current_timestamp() - interval {USAGE_SUMMARY_DAYS} days
  AND requester = current_user()
  AND (
    lower(user_agent) LIKE '%codex%'
    OR lower(user_agent) LIKE '%claude%'
    OR lower(user_agent) LIKE '%gemini%'
    OR lower(user_agent) LIKE '%opencode%'
  )
),
daily_usage AS (
  SELECT
    requester_name,
    tool,
    usage_day,
    SUM(total_tokens_used) AS total_tokens_used,
    COUNT(DISTINCT request_id) AS sessions,
    MIN(event_time) AS first_event_time,
    MAX(event_time) AS last_event_time
  FROM usage_events
  GROUP BY 1, 2, 3
),
model_usage AS (
  SELECT
    requester_name,
    tool,
    usage_day,
    destination_model,
    SUM(total_tokens_used) AS model_tokens_used
  FROM usage_events
  WHERE destination_model IS NOT NULL AND destination_model != ''
  GROUP BY 1, 2, 3, 4
),
model_rollup AS (
  SELECT
    requester_name,
    tool,
    usage_day,
    CONCAT_WS(', ', SORT_ARRAY(COLLECT_SET(destination_model))) AS models,
    TO_JSON(
      SORT_ARRAY(
        COLLECT_LIST(
          NAMED_STRUCT('model', destination_model, 'tokens', model_tokens_used)
        )
      )
    ) AS model_tokens
  FROM model_usage
  GROUP BY 1, 2, 3
)
SELECT
  daily_usage.requester_name,
  daily_usage.tool,
  daily_usage.usage_day,
  daily_usage.total_tokens_used,
  daily_usage.sessions,
  daily_usage.first_event_time,
  daily_usage.last_event_time,
  COALESCE(model_rollup.models, '') AS models,
  COALESCE(model_rollup.model_tokens, '[]') AS model_tokens
FROM daily_usage
LEFT JOIN model_rollup
  ON daily_usage.requester_name = model_rollup.requester_name
  AND daily_usage.tool = model_rollup.tool
  AND daily_usage.usage_day = model_rollup.usage_day
ORDER BY daily_usage.usage_day DESC, daily_usage.tool ASC
""".strip()


def build_current_user_query() -> str:
    return "SELECT current_user() AS requester_name"


def parse_usage_rows(columns: list[str], rows: list[tuple]) -> list[dict[str, object]]:
    return [dict(zip(columns, row, strict=False)) for row in rows]


def configured_usage_tools(state: dict, tool_displays: dict[str, str]) -> list[str]:
    configured = state.get("available_tools") or state.get("managed_configs", {}).keys()
    if not isinstance(configured, list):
        configured = list(configured)
    return [tool for tool in tool_displays if tool in configured]


def filter_records_for_tools(
    records: list[dict[str, object]],
    tools: list[str],
) -> list[dict[str, object]]:
    configured = set(tools)
    return [record for record in records if record.get("tool") in configured]


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


def simplify_model_name(tool: str, model_name: str) -> str:
    normalized = (model_name or "").strip()
    if not normalized:
        return "-"

    prefix = "databricks-"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]

    tool_prefixes = {
        "claude": "claude-",
        "gemini": "gemini-",
        "codex": "gpt-",
    }
    tool_prefix = tool_prefixes.get(tool)
    if tool_prefix and normalized.startswith(tool_prefix):
        normalized = normalized[len(tool_prefix) :]
    return normalized


def extract_model_names(tool: str, raw_models: object) -> list[str]:
    if not isinstance(raw_models, str) or not raw_models.strip():
        return []

    unique_models: list[str] = []
    for item in raw_models.split(","):
        simplified = simplify_model_name(tool, item.strip())
        if simplified != "-" and simplified not in unique_models:
            unique_models.append(simplified)
    return unique_models


def summarize_models(tool: str, raw_models: object) -> str:
    if not isinstance(raw_models, str) or not raw_models.strip():
        return "-"
    parts = extract_model_names(tool, raw_models)
    return ", ".join(parts) if parts else "-"


def _coerce_model_token_item(tool: str, item: object) -> tuple[str, int] | None:
    if not isinstance(item, Mapping):
        return None
    item_mapping = cast(Mapping[str, object], item)

    raw_model = item_mapping.get("model")
    if not isinstance(raw_model, str) or not raw_model.strip():
        return None

    raw_tokens = item_mapping.get("tokens")
    try:
        token_total = int(cast(int | float | str, raw_tokens or 0))
    except (TypeError, ValueError):
        token_total = 0

    model_name = simplify_model_name(tool, raw_model)
    if model_name == "-":
        return None
    return model_name, token_total


def extract_model_token_breakdown(
    tool: str,
    raw_model_tokens: object,
    raw_models: object = None,
    total_tokens: int = 0,
) -> list[tuple[str, int]]:
    items: object
    if isinstance(raw_model_tokens, str) and raw_model_tokens.strip():
        try:
            items = json.loads(raw_model_tokens)
        except json.JSONDecodeError:
            items = []
    else:
        items = raw_model_tokens

    model_tokens: dict[str, int] = {}
    if isinstance(items, list):
        for item in items:
            coerced = _coerce_model_token_item(tool, item)
            if not coerced:
                continue
            model_name, token_total = coerced
            model_tokens[model_name] = model_tokens.get(model_name, 0) + token_total

    if model_tokens:
        return sorted(model_tokens.items(), key=lambda item: (-item[1], item[0].lower()))

    models = extract_model_names(tool, raw_models)
    if len(models) == 1 and total_tokens:
        return [(models[0], total_tokens)]
    return [(model_name, 0) for model_name in models]


def summarize_model_tokens(
    tool: str,
    raw_model_tokens: object,
    raw_models: object,
    total_tokens: int,
) -> str:
    model_tokens = extract_model_token_breakdown(
        tool,
        raw_model_tokens,
        raw_models,
        total_tokens,
    )
    if not model_tokens:
        return "-"
    return ", ".join(
        f"{model_name} ({format_token_count(token_total)})" if token_total else model_name
        for model_name, token_total in model_tokens
    )


def empty_tool_day(tool: str, usage_day: date) -> dict[str, object]:
    return {
        "tool": tool,
        "usage_day": usage_day,
        "total_tokens_used": 0,
        "sessions": 0,
        "first_event_time": None,
        "last_event_time": None,
        "models": "-",
        "model_tokens": "[]",
    }


def has_tool_usage_last_week(records: list[dict[str, object]], tool: str) -> bool:
    today = date.today()
    week_start = today - timedelta(days=USAGE_BREAKDOWN_DAYS - 1)
    for record in records:
        if record.get("tool") != tool:
            continue
        usage_day = coerce_date(record.get("usage_day"))
        if not usage_day or usage_day < week_start:
            continue
        token_total = int(cast(int, record.get("total_tokens_used") or 0))
        session_total = int(cast(int, record.get("sessions") or 0))
        if token_total or session_total:
            return True
    return False


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
        token_total = int(cast(int, record.get("total_tokens_used") or 0))
        session_total = int(cast(int, record.get("sessions") or 0))
        rendered_rows.append(
            [
                usage_day.strftime("%m-%d"),
                usage_day.strftime("%a"),
                format_token_count(token_total) if token_total else "-",
                str(session_total) if session_total else "-",
                format_duration(duration),
                summarize_model_tokens(
                    tool,
                    record.get("model_tokens"),
                    record.get("models"),
                    token_total,
                ),
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


def render_budget_lines(budget_spend: tuple[Decimal, Decimal] | None) -> list[str]:
    """Spend-against-threshold lines, or nothing when unavailable."""
    if budget_spend is None:
        return []
    spend, threshold = budget_spend
    # No whole to be a fraction of; dividing would raise.
    if threshold <= 0:
        return [f"{label('Budget spend:')} {value(format_usd(spend))}"]
    fraction = float(spend / threshold)
    summary = f"{format_usd(spend)} of {format_usd(threshold)} ({fraction:.0%})"
    return [
        f"{label('Budget spend:')} {value(summary)}",
        muted(format_meter(fraction)),
    ]


def render_usage_summary(
    records: list[dict[str, object]],
    requester_name: str,
    tool_displays: dict[str, str],
    budget_spend: tuple[Decimal, Decimal] | None = None,
) -> str:
    today = date.today()
    week_start = today - timedelta(days=USAGE_BREAKDOWN_DAYS - 1)
    month_start = today - timedelta(days=USAGE_SUMMARY_DAYS - 1)

    daily_total = 0
    weekly_total = 0
    monthly_total = 0
    active_tools_last_week: list[str] = []
    weekly_model_tokens: dict[str, int] = {}
    for record in records:
        usage_day = coerce_date(record.get("usage_day"))
        if not usage_day:
            continue
        token_total = int(cast(int, record.get("total_tokens_used") or 0))
        tool = record.get("tool")
        if usage_day >= month_start:
            monthly_total += token_total
        if usage_day >= week_start:
            weekly_total += token_total
            if (
                isinstance(tool, str)
                and tool in tool_displays
                and tool not in active_tools_last_week
            ):
                active_tools_last_week.append(tool)
            if isinstance(tool, str):
                for model_name, model_token_total in extract_model_token_breakdown(
                    tool,
                    record.get("model_tokens"),
                    record.get("models"),
                    token_total,
                ):
                    weekly_model_tokens[model_name] = (
                        weekly_model_tokens.get(model_name, 0) + model_token_total
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
        tool_text = ", ".join(tool_displays[tool] for tool in active_tools_last_week)
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
    lines.extend(render_budget_lines(budget_spend))
    return "\n".join(lines)


def run_query_on_first_working_warehouse(
    workspace: str,
    token: str,
    candidates: list[SqlWarehouse],
    query: str,
) -> tuple[str, list[str], list[tuple]]:
    """Run `query` on the first candidate that accepts the connection.

    Returns the warehouse's http path alongside the result so later queries
    reuse it. Raises the last error when every candidate fails.
    """
    last_error: RuntimeError | None = None
    for warehouse in candidates:
        print_note(f"Using SQL warehouse `{warehouse.label}` ({warehouse.state}).")
        try:
            columns, rows = run_usage_query(workspace, warehouse.http_path, token, query)
        except RuntimeError as exc:
            last_error = exc
            print_warning(f"SQL warehouse `{warehouse.label}` is unusable: {exc}")
            continue
        return warehouse.http_path, columns, rows
    raise last_error or RuntimeError("No SQL warehouse could run the usage query.")


def usage(warehouse_id: str | None = None) -> int:
    # Late import to avoid circular import (agents → state, but usage uses TOOL_SPECS for displays).
    from ucode.agents import TOOL_SPECS

    state = load_state()
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("Workspace is not configured. Run `ucode configure` first.")

    profile = state.get("profile")
    apply_pat_environment(state)
    ensure_databricks_auth(workspace, profile)
    with spinner("Retrieving Databricks access token..."):
        token = get_databricks_token(workspace, profile)

    with spinner("Discovering SQL warehouse..."):
        candidates = discover_sql_warehouses(workspace, token, warehouse_id=warehouse_id)

    resolved_http_path, columns, rows = run_query_on_first_working_warehouse(
        workspace, token, candidates, build_usage_report_query()
    )
    records = parse_usage_rows(columns, rows)
    requester_name = find_requester_name(workspace, resolved_http_path, token, records)

    # Opt-in per workspace: omit the lines rather than fail the report.
    with spinner("Checking budget spend..."):
        budget_spend, _ = resolve_current_budget_spend(workspace, token)

    tool_displays = {tool: spec["display"] for tool, spec in TOOL_SPECS.items()}
    configured_tools = configured_usage_tools(state, tool_displays)
    configured_tool_displays = {tool: tool_displays[tool] for tool in configured_tools}
    records = filter_records_for_tools(records, configured_tools)

    console.print(
        render_usage_summary(
            records,
            requester_name,
            configured_tool_displays,
            budget_spend=budget_spend,
        )
    )

    table_headers = ["Date", "Day", "Tokens", "Sessions", "Duration", "Models"]
    table_widths = [8, 5, 10, 8, 8, 24]

    if not configured_tools:
        print_note("No coding agents configured. Run `ucode configure` to set up agents.")
        return 0

    for tool in configured_tools:
        display = tool_displays[tool]
        print_heading(f"{display} · Last {USAGE_BREAKDOWN_DAYS} Days")
        if not has_tool_usage_last_week(records, tool):
            print_note(f"No usage for {display} in the last {USAGE_BREAKDOWN_DAYS} days.")
            continue
        console.print(
            render_box_table(
                table_headers,
                build_tool_breakdown_rows(records, tool),
                max_widths=table_widths,
            )
        )
    return 0
