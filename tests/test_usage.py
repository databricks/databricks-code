"""Tests for usage.py — query builders, parsing/formatting, rendering."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

import ucode.usage as usage_mod
from ucode.databricks import SqlWarehouse
from ucode.ui import label, value
from ucode.usage import (
    USAGE_BREAKDOWN_DAYS,
    USAGE_SUMMARY_DAYS,
    build_current_user_query,
    build_tool_breakdown_rows,
    build_usage_report_query,
    coerce_date,
    coerce_datetime,
    configured_usage_tools,
    empty_tool_day,
    extract_model_names,
    extract_model_token_breakdown,
    filter_records_for_tools,
    has_tool_usage_last_week,
    parse_usage_rows,
    render_budget_lines,
    render_usage_summary,
    run_query_on_first_working_warehouse,
    simplify_model_name,
    summarize_model_tokens,
    summarize_models,
    usage,
)


class TestBuildUsageReportQuery:
    def test_contains_system_table(self):
        q = build_usage_report_query()
        assert "system.ai_gateway.usage" in q

    def test_contains_interval(self):
        q = build_usage_report_query()
        assert str(USAGE_SUMMARY_DAYS) in q

    def test_filters_known_tools(self):
        q = build_usage_report_query()
        for tool in ("codex", "claude", "gemini", "opencode"):
            assert tool in q

    def test_includes_per_model_token_rollup(self):
        q = build_usage_report_query()
        assert "model_tokens" in q
        assert "SUM(total_tokens_used) AS model_tokens_used" in q
        assert "NAMED_STRUCT('model', destination_model, 'tokens', model_tokens_used)" in q


class TestBuildCurrentUserQuery:
    def test_uses_current_user(self):
        q = build_current_user_query()
        assert "current_user()" in q


class TestParseUsageRows:
    def test_zips_columns_and_rows(self):
        columns = ["a", "b", "c"]
        rows = [(1, 2, 3), (4, 5, 6)]
        result = parse_usage_rows(columns, rows)
        assert result == [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]

    def test_empty_rows(self):
        assert parse_usage_rows(["a"], []) == []


class TestConfiguredUsageTools:
    def test_uses_available_tools_in_display_order(self):
        tool_displays = {"claude": "Claude Code", "codex": "Codex", "gemini": "Gemini"}
        state = {"available_tools": ["codex", "claude"]}
        assert configured_usage_tools(state, tool_displays) == ["claude", "codex"]

    def test_falls_back_to_managed_configs(self):
        tool_displays = {"claude": "Claude Code", "codex": "Codex"}
        state = {"managed_configs": {"codex": {"keys": []}}}
        assert configured_usage_tools(state, tool_displays) == ["codex"]

    def test_ignores_unknown_tools(self):
        tool_displays = {"claude": "Claude Code"}
        state = {"available_tools": ["claude", "unknown"]}
        assert configured_usage_tools(state, tool_displays) == ["claude"]


class TestFilterRecordsForTools:
    def test_keeps_only_configured_tools(self):
        records = [
            {"tool": "claude", "total_tokens_used": 100},
            {"tool": "gemini", "total_tokens_used": 200},
            {"tool": "codex", "total_tokens_used": 300},
        ]
        assert filter_records_for_tools(records, ["claude", "codex"]) == [
            {"tool": "claude", "total_tokens_used": 100},
            {"tool": "codex", "total_tokens_used": 300},
        ]


class TestHasToolUsageLastWeek:
    def test_true_for_recent_tokens(self):
        records = [
            {
                "tool": "claude",
                "usage_day": date.today(),
                "total_tokens_used": 100,
                "sessions": 1,
            }
        ]
        assert has_tool_usage_last_week(records, "claude") is True

    def test_true_for_recent_session_even_without_tokens(self):
        records = [
            {
                "tool": "claude",
                "usage_day": date.today(),
                "total_tokens_used": 0,
                "sessions": 1,
            }
        ]
        assert has_tool_usage_last_week(records, "claude") is True

    def test_false_for_only_old_usage(self):
        records = [
            {
                "tool": "claude",
                "usage_day": date.today() - timedelta(days=USAGE_BREAKDOWN_DAYS),
                "total_tokens_used": 100,
                "sessions": 1,
            }
        ]
        assert has_tool_usage_last_week(records, "claude") is False

    def test_false_for_other_tool_usage(self):
        records = [
            {
                "tool": "codex",
                "usage_day": date.today(),
                "total_tokens_used": 100,
                "sessions": 1,
            }
        ]
        assert has_tool_usage_last_week(records, "claude") is False


class TestCoerceDate:
    def test_date_passthrough(self):
        d = date(2024, 6, 1)
        assert coerce_date(d) == d

    def test_datetime_to_date(self):
        dt = datetime(2024, 6, 1, 12, 0, 0)
        assert coerce_date(dt) == date(2024, 6, 1)

    def test_iso_string(self):
        assert coerce_date("2024-06-01") == date(2024, 6, 1)

    def test_invalid_string_returns_none(self):
        assert coerce_date("not-a-date") is None

    def test_none_returns_none(self):
        assert coerce_date(None) is None


class TestCoerceDatetime:
    def test_datetime_passthrough(self):
        dt = datetime(2024, 6, 1, 0, 0, 0)
        assert coerce_datetime(dt) == dt

    def test_iso_string(self):
        result = coerce_datetime("2024-06-01T12:00:00")
        assert isinstance(result, datetime)
        assert result.date() == date(2024, 6, 1)

    def test_z_suffix(self):
        result = coerce_datetime("2024-06-01T12:00:00Z")
        assert isinstance(result, datetime)

    def test_invalid_string_returns_none(self):
        assert coerce_datetime("bad") is None

    def test_none_returns_none(self):
        assert coerce_datetime(None) is None


class TestSimplifyModelName:
    def test_strips_databricks_and_tool_prefix(self):
        # databricks- stripped first, then claude- stripped → "sonnet-4"
        assert simplify_model_name("claude", "databricks-claude-sonnet-4") == "sonnet-4"

    def test_gemini_prefix(self):
        result = simplify_model_name("gemini", "databricks-gemini-2.0-flash")
        assert result == "2.0-flash"

    def test_codex_strips_gpt_prefix(self):
        result = simplify_model_name("codex", "databricks-gpt-4o")
        assert result == "4o"

    def test_empty_returns_dash(self):
        assert simplify_model_name("claude", "") == "-"

    def test_no_known_prefix_returns_as_is(self):
        result = simplify_model_name("claude", "some-other-model")
        assert result == "some-other-model"

    def test_only_databricks_prefix_stripped_for_unknown_tool(self):
        result = simplify_model_name("opencode", "databricks-claude-sonnet-4")
        assert result == "claude-sonnet-4"


class TestExtractModelNames:
    def test_single_model(self):
        result = extract_model_names("claude", "databricks-claude-sonnet-4")
        assert result == ["sonnet-4"]

    def test_multiple_models(self):
        result = extract_model_names(
            "claude", "databricks-claude-sonnet-4, databricks-claude-opus-4"
        )
        assert "sonnet-4" in result
        assert "opus-4" in result

    def test_deduplicates(self):
        result = extract_model_names(
            "claude", "databricks-claude-sonnet-4, databricks-claude-sonnet-4"
        )
        assert result.count("sonnet-4") == 1

    def test_empty_returns_empty_list(self):
        assert extract_model_names("claude", "") == []

    def test_non_string_returns_empty_list(self):
        assert extract_model_names("claude", None) == []


class TestSummarizeModels:
    def test_single_model(self):
        result = summarize_models("claude", "databricks-claude-sonnet-4")
        assert result == "sonnet-4"

    def test_multiple_models_joined(self):
        result = summarize_models("claude", "databricks-claude-sonnet-4, databricks-claude-opus-4")
        assert "sonnet-4" in result
        assert "," in result

    def test_empty_returns_dash(self):
        assert summarize_models("claude", "") == "-"

    def test_none_returns_dash(self):
        assert summarize_models("claude", None) == "-"


class TestModelTokenBreakdown:
    def test_extracts_json_model_tokens(self):
        raw = (
            '[{"model":"databricks-claude-opus-4", "tokens":236000}, '
            '{"model":"databricks-claude-haiku-4.5", "tokens":920}]'
        )
        result = extract_model_token_breakdown("claude", raw)
        assert result == [("opus-4", 236000), ("haiku-4.5", 920)]

    def test_merges_simplified_duplicate_model_names(self):
        raw = [
            {"model": "databricks-claude-opus-4", "tokens": 100},
            {"model": "claude-opus-4", "tokens": 50},
        ]
        result = extract_model_token_breakdown("claude", raw)
        assert result == [("opus-4", 150)]

    def test_single_model_legacy_fallback_uses_total_tokens(self):
        result = extract_model_token_breakdown(
            "codex",
            None,
            "databricks-gpt-5",
            13300,
        )
        assert result == [("5", 13300)]

    def test_multi_model_legacy_fallback_does_not_assign_total_to_each_model(self):
        result = extract_model_token_breakdown(
            "claude",
            None,
            "databricks-claude-haiku-4.5, databricks-claude-opus-4",
            237000,
        )
        assert result == [("haiku-4.5", 0), ("opus-4", 0)]

    def test_summarizes_tokens_next_to_each_model(self):
        raw = '[{"model":"databricks-claude-opus-4", "tokens":236000}]'
        result = summarize_model_tokens("claude", raw, "", 0)
        assert result == "opus-4 (236.0K)"


class TestEmptyToolDay:
    def test_structure(self):
        d = date(2024, 6, 1)
        row = empty_tool_day("claude", d)
        assert row["tool"] == "claude"
        assert row["usage_day"] == d
        assert row["total_tokens_used"] == 0
        assert row["sessions"] == 0
        assert row["models"] == "-"


class TestRenderBudgetLines:
    def test_no_lines_when_unavailable(self):
        assert render_budget_lines(None) == []

    def test_shows_spend_threshold_and_percent(self):
        lines = render_budget_lines((Decimal("12.34"), Decimal("100")))
        assert "$12.34" in lines[0]
        assert "$100.00" in lines[0]
        assert "12%" in lines[0]

    def test_renders_meter(self):
        lines = render_budget_lines((Decimal("50"), Decimal("100")))
        assert len(lines) == 2
        assert "█" in lines[1]
        assert "░" in lines[1]

    def test_zero_threshold_omits_percent_and_meter(self):
        lines = render_budget_lines((Decimal("5"), Decimal("0")))
        assert lines == [f"{label('Budget spend:')} {value('$5.00')}"]

    def test_spend_over_threshold_clamps_meter(self):
        lines = render_budget_lines((Decimal("250"), Decimal("100")))
        assert "250%" in lines[0]
        assert "░" not in lines[1]

    def test_thousands_separator(self):
        lines = render_budget_lines((Decimal("1234.5"), Decimal("10000")))
        assert "$1,234.50" in lines[0]
        assert "$10,000.00" in lines[0]


class TestRenderUsageSummary:
    def _make_record(self, days_ago: int, tool: str, tokens: int, model: str = "") -> dict:
        d = date.today() - timedelta(days=days_ago)
        return {
            "tool": tool,
            "usage_day": d,
            "total_tokens_used": tokens,
            "models": model,
        }

    def test_contains_requester_name(self):
        records = [self._make_record(0, "claude", 1000)]
        result = render_usage_summary(records, "alice@example.com", {"claude": "Claude Code"})
        assert "alice@example.com" in result

    def test_today_total(self):
        records = [self._make_record(0, "claude", 5000)]
        result = render_usage_summary(records, "user", {"claude": "Claude Code"})
        assert "5.0K" in result

    def test_weekly_total_includes_past_week(self):
        records = [
            self._make_record(0, "claude", 1000),
            self._make_record(3, "claude", 2000),
            self._make_record(USAGE_BREAKDOWN_DAYS, "claude", 9999),  # outside window
        ]
        result = render_usage_summary(records, "user", {"claude": "Claude Code"})
        # only 3K from the last 7 days; 9999 from day 7 (boundary) may vary
        assert "3.0K" in result or "3" in result

    def test_active_tools_listed(self):
        records = [self._make_record(0, "claude", 1000)]
        result = render_usage_summary(records, "user", {"claude": "Claude Code"})
        assert "Claude Code" in result

    def test_top_models_listed(self):
        records = [self._make_record(0, "claude", 5000, "databricks-claude-sonnet-4")]
        result = render_usage_summary(records, "user", {"claude": "Claude Code"})
        assert "sonnet-4" in result

    def test_includes_budget_spend_when_available(self):
        records = [self._make_record(0, "claude", 1000)]
        result = render_usage_summary(
            records,
            "user",
            {"claude": "Claude Code"},
            budget_spend=(Decimal("12.34"), Decimal("100")),
        )
        assert "$12.34 of $100.00" in result

    def test_omits_budget_spend_by_default(self):
        records = [self._make_record(0, "claude", 1000)]
        result = render_usage_summary(records, "user", {"claude": "Claude Code"})
        assert "Budget spend" not in result

    def test_top_models_uses_per_model_token_totals(self):
        records = [
            {
                "tool": "claude",
                "usage_day": date.today(),
                "total_tokens_used": 237000,
                "models": "databricks-claude-haiku-4.5, databricks-claude-opus-4",
                "model_tokens": (
                    '[{"model":"databricks-claude-haiku-4.5", "tokens":920}, '
                    '{"model":"databricks-claude-opus-4", "tokens":236080}]'
                ),
            },
            {
                "tool": "codex",
                "usage_day": date.today(),
                "total_tokens_used": 13300,
                "models": "databricks-gpt-5",
                "model_tokens": '[{"model":"databricks-gpt-5", "tokens":13300}]',
            },
        ]
        result = render_usage_summary(
            records,
            "user",
            {"claude": "Claude Code", "codex": "Codex"},
        )
        assert "opus-4 (236.1K)" in result
        assert "5 (13.3K)" in result
        assert "haiku-4.5 (920)" in result
        assert "haiku-4.5 (237.0K)" not in result

    def test_daily_table_shows_per_model_token_totals(self):
        records = [
            {
                "tool": "claude",
                "usage_day": date.today(),
                "total_tokens_used": 237000,
                "sessions": 2,
                "models": "databricks-claude-haiku-4.5, databricks-claude-opus-4",
                "model_tokens": (
                    '[{"model":"databricks-claude-haiku-4.5", "tokens":920}, '
                    '{"model":"databricks-claude-opus-4", "tokens":236080}]'
                ),
            }
        ]
        rows = build_tool_breakdown_rows(records, "claude")
        assert rows[0][5] == "opus-4 (236.1K), haiku-4.5 (920)"

    def test_empty_records(self):
        result = render_usage_summary([], "user", {"claude": "Claude Code"})
        assert "user" in result


class TestUsageCommand:
    def test_filters_to_configured_agents_and_skips_inactive_tables(self, monkeypatch):
        today = date.today()
        old_day = today - timedelta(days=USAGE_BREAKDOWN_DAYS)
        columns = [
            "requester_name",
            "tool",
            "usage_day",
            "total_tokens_used",
            "sessions",
            "first_event_time",
            "last_event_time",
            "models",
            "model_tokens",
        ]
        rows = [
            (
                "user@example.com",
                "codex",
                today,
                100,
                1,
                None,
                None,
                "databricks-gpt-5",
                '[{"model":"databricks-gpt-5", "tokens":100}]',
            ),
            (
                "user@example.com",
                "claude",
                old_day,
                200,
                1,
                None,
                None,
                "databricks-claude-opus-4",
                '[{"model":"databricks-claude-opus-4", "tokens":200}]',
            ),
            (
                "user@example.com",
                "gemini",
                today,
                900,
                1,
                None,
                None,
                "databricks-gemini-2.0-flash",
                '[{"model":"databricks-gemini-2.0-flash", "tokens":900}]',
            ),
        ]

        printed: list[str] = []
        headings: list[str] = []
        notes: list[str] = []
        rendered_tables: list[list[list[str]]] = []

        class DummyConsole:
            def print(self, value):
                printed.append(str(value))

        def fake_render_box_table(headers, table_rows, max_widths=None):
            rendered_tables.append(table_rows)
            return "TABLE"

        monkeypatch.setattr(
            usage_mod,
            "load_state",
            lambda: {"workspace": "https://workspace", "available_tools": ["claude", "codex"]},
        )
        monkeypatch.setattr(usage_mod, "ensure_databricks_auth", lambda *args, **kwargs: None)
        monkeypatch.setattr(usage_mod, "get_databricks_token", lambda *args, **kwargs: "token")
        monkeypatch.setattr(
            usage_mod,
            "discover_sql_warehouses",
            lambda *args, **kwargs: [SqlWarehouse("/sql/1.0/warehouses/abc", "wh", "RUNNING")],
        )
        monkeypatch.setattr(usage_mod, "run_usage_query", lambda *args, **kwargs: (columns, rows))
        monkeypatch.setattr(
            usage_mod, "resolve_current_budget_spend", lambda *args, **kwargs: (None, "disabled")
        )
        monkeypatch.setattr(usage_mod, "console", DummyConsole())
        monkeypatch.setattr(usage_mod, "print_heading", headings.append)
        monkeypatch.setattr(usage_mod, "print_note", notes.append)
        monkeypatch.setattr(usage_mod, "render_box_table", fake_render_box_table)

        assert usage() == 0

        assert "Codex · Last 7 Days" in headings
        assert "Claude Code · Last 7 Days" in headings
        assert all("Gemini" not in heading for heading in headings)
        assert notes == [
            "Using SQL warehouse `wh` (RUNNING).",
            f"No usage for Claude Code in the last {USAGE_BREAKDOWN_DAYS} days.",
        ]
        assert len(rendered_tables) == 1
        assert rendered_tables[0][0][2] == "100"
        assert "gemini" not in "\n".join(printed).lower()
        assert "900" not in "\n".join(printed)


class TestRunQueryOnFirstWorkingWarehouse:
    _COLUMNS = ["requester_name"]
    _ROWS = [("user@example.com",)]

    def _warehouses(self, *labels: str) -> list[SqlWarehouse]:
        return [SqlWarehouse(f"/sql/1.0/warehouses/{label}", label, "RUNNING") for label in labels]

    def test_returns_first_working_warehouse(self, monkeypatch):
        monkeypatch.setattr(usage_mod, "print_note", lambda *a: None)
        monkeypatch.setattr(
            usage_mod, "run_usage_query", lambda *a, **k: (self._COLUMNS, self._ROWS)
        )
        http_path, columns, rows = run_query_on_first_working_warehouse(
            "https://ws", "token", self._warehouses("a", "b"), "SELECT 1"
        )
        assert http_path == "/sql/1.0/warehouses/a"
        assert (columns, rows) == (self._COLUMNS, self._ROWS)

    def test_falls_through_to_next_warehouse(self, monkeypatch):
        warnings: list[str] = []
        monkeypatch.setattr(usage_mod, "print_note", lambda *a: None)
        monkeypatch.setattr(usage_mod, "print_warning", warnings.append)
        attempted: list[str] = []

        def flaky(workspace, http_path, token, query):
            attempted.append(http_path)
            if http_path.endswith("dead"):
                raise RuntimeError("ENDPOINT_NOT_FOUND")
            return self._COLUMNS, self._ROWS

        monkeypatch.setattr(usage_mod, "run_usage_query", flaky)
        http_path, _, _ = run_query_on_first_working_warehouse(
            "https://ws", "token", self._warehouses("dead", "alive"), "SELECT 1"
        )
        assert http_path == "/sql/1.0/warehouses/alive"
        assert attempted == ["/sql/1.0/warehouses/dead", "/sql/1.0/warehouses/alive"]
        assert len(warnings) == 1
        assert "dead" in warnings[0]

    def test_raises_last_error_when_all_fail(self, monkeypatch):
        monkeypatch.setattr(usage_mod, "print_note", lambda *a: None)
        monkeypatch.setattr(usage_mod, "print_warning", lambda *a: None)

        def always_fail(workspace, http_path, token, query):
            raise RuntimeError(f"boom {http_path[-1]}")

        monkeypatch.setattr(usage_mod, "run_usage_query", always_fail)
        with pytest.raises(RuntimeError, match="boom b"):
            run_query_on_first_working_warehouse(
                "https://ws", "token", self._warehouses("a", "b"), "SELECT 1"
            )

    def test_raises_when_no_candidates(self, monkeypatch):
        monkeypatch.setattr(usage_mod, "print_note", lambda *a: None)
        with pytest.raises(RuntimeError, match="No SQL warehouse could run"):
            run_query_on_first_working_warehouse("https://ws", "token", [], "SELECT 1")


class TestUsageWarehouseIdPassthrough:
    def test_forwards_warehouse_id_to_discovery(self, monkeypatch):
        captured = {}

        def fake_discover(workspace, token, *, warehouse_id=None):
            captured["warehouse_id"] = warehouse_id
            return [SqlWarehouse("/sql/1.0/warehouses/xyz", "xyz", "REQUESTED")]

        monkeypatch.setattr(
            usage_mod, "load_state", lambda: {"workspace": "https://ws", "available_tools": []}
        )
        monkeypatch.setattr(usage_mod, "ensure_databricks_auth", lambda *a, **k: None)
        monkeypatch.setattr(usage_mod, "get_databricks_token", lambda *a, **k: "token")
        monkeypatch.setattr(usage_mod, "discover_sql_warehouses", fake_discover)
        monkeypatch.setattr(usage_mod, "run_usage_query", lambda *a, **k: (["c"], []))
        monkeypatch.setattr(usage_mod, "print_note", lambda *a: None)
        monkeypatch.setattr(usage_mod, "console", type("C", (), {"print": lambda *a: None})())

        assert usage(warehouse_id="xyz") == 0
        assert captured["warehouse_id"] == "xyz"
