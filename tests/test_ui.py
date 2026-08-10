"""Tests for ui.py — pure helpers that don't touch I/O or prompts."""

from __future__ import annotations

import io
from datetime import timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
from rich.console import Console

from ucode.ui import (
    format_duration,
    format_meter,
    format_token_count,
    format_usd,
    normalize_workspace_url,
    prompt_for_multi_selection,
    prompt_for_percentage,
    prompt_for_selection,
    prompt_for_text,
    prompt_for_workspace,
    prompt_yes_no_default,
    render_box_table,
    status_badge,
)


class TestPromptYesNoDefault:
    def _answer(self, monkeypatch, value):
        # value: a string the user "types", or EOFError to simulate closed stdin.
        def fake_input(_prompt):
            if value is EOFError:
                raise EOFError
            return value

        monkeypatch.setattr("ucode.ui.console.input", fake_input)

    def test_empty_takes_default_true(self, monkeypatch):
        self._answer(monkeypatch, "")
        assert prompt_yes_no_default("go?", default=True) is True

    def test_empty_takes_default_false(self, monkeypatch):
        self._answer(monkeypatch, "")
        assert prompt_yes_no_default("go?", default=False) is False

    def test_eof_takes_default(self, monkeypatch):
        # Non-interactive / closed stdin must not abort — it takes the default.
        self._answer(monkeypatch, EOFError)
        assert prompt_yes_no_default("go?", default=True) is True

    def test_explicit_no_overrides_default_yes(self, monkeypatch):
        self._answer(monkeypatch, "n")
        assert prompt_yes_no_default("go?", default=True) is False

    def test_explicit_yes_overrides_default_false(self, monkeypatch):
        self._answer(monkeypatch, "yes")
        assert prompt_yes_no_default("go?", default=False) is True


def _visible(markup: str) -> str:
    """What the user actually sees, with Rich markup resolved.

    Asserting on the raw markup string is what let a swallowed default ship: `[tiered]` is present
    in the markup and absent from the output, because Rich reads it as a style tag.
    """
    console = Console(file=io.StringIO(), force_terminal=False, width=200)
    console.print(markup)
    return console.file.getvalue().rstrip()


class TestDefaultsAreLabelledAsAcceptable:
    """A shown default must say that enter takes it, or it reads as a format example."""

    def test_text_default_says_enter_accepts_it(self):
        with patch("ucode.ui.console.input", return_value="") as inp:
            assert prompt_for_text("Policy name", default="tiered") == "tiered"
        rendered = _visible(inp.call_args[0][0])
        assert "[tiered]" in rendered
        assert "enter to accept" in rendered

    def test_a_word_like_default_is_not_eaten_as_markup(self):
        # Rich treats `[coding-agents-tiered-routing]` as a style tag and renders nothing for it, so
        # the real wizard default vanished from the prompt while `[80]` survived.
        with patch("ucode.ui.console.input", return_value="") as inp:
            prompt_for_text("Policy name", default="coding-agents-tiered-routing")
        assert "[coding-agents-tiered-routing]" in _visible(inp.call_args[0][0])

    def test_a_dotted_default_is_not_eaten_as_markup(self):
        with patch("ucode.ui.console.input", return_value="") as inp:
            prompt_for_text("Skills location", default="main.default")
        assert "[main.default]" in _visible(inp.call_args[0][0])

    def test_percentage_default_says_enter_accepts_it(self):
        with patch("ucode.ui.console.input", return_value="") as inp:
            assert prompt_for_percentage("at what percent?", default=0.8) == 0.8
        rendered = _visible(inp.call_args[0][0])
        # Prompted in percent even though the API takes a fraction.
        assert "[80]" in rendered
        assert "enter to accept" in rendered

    def test_no_default_shows_no_hint(self):
        with patch("ucode.ui.console.input", return_value="typed") as inp:
            assert prompt_for_text("Model") == "typed"
        assert "enter to accept" not in _visible(inp.call_args[0][0])

    def test_typing_still_overrides_the_default(self):
        with patch("ucode.ui.console.input", return_value="mine"):
            assert prompt_for_text("Policy name", default="tiered") == "mine"


class TestClosedStdinAborts:
    """Ctrl-D must reach the CLI as an abort, not as a traceback."""

    def test_percentage_without_a_default_raises_keyboard_interrupt(self):
        # `ucode setup`'s tier prompt passes no default. EOFError has no handler above this call —
        # the setup command catches only RuntimeError and KeyboardInterrupt — so a bare EOFError
        # reached the admin as a raw traceback.
        with patch("ucode.ui.console.input", side_effect=EOFError):
            with pytest.raises(KeyboardInterrupt):
                prompt_for_percentage("Tier 1: activates at what percent of budget?")

    def test_percentage_with_a_default_still_takes_it(self):
        with patch("ucode.ui.console.input", side_effect=EOFError):
            assert prompt_for_percentage("at what percent?", default=0.8) == 0.8


class TestNormalizeWorkspaceUrl:
    def test_adds_https_when_missing(self):
        assert normalize_workspace_url("example.databricks.com") == "https://example.databricks.com"

    def test_strips_trailing_slash(self):
        assert (
            normalize_workspace_url("https://example.databricks.com/")
            == "https://example.databricks.com"
        )

    def test_strips_multiple_trailing_slashes(self):
        assert (
            normalize_workspace_url("https://example.databricks.com///")
            == "https://example.databricks.com"
        )

    def test_preserves_https(self):
        assert (
            normalize_workspace_url("https://foo.azuredatabricks.net")
            == "https://foo.azuredatabricks.net"
        )

    def test_preserves_http(self):
        assert normalize_workspace_url("http://localhost:8080") == "http://localhost:8080"

    def test_strips_whitespace(self):
        assert (
            normalize_workspace_url("  https://example.databricks.com  ")
            == "https://example.databricks.com"
        )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_workspace_url("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_workspace_url("   ")


class TestScrollHint:
    """A long picker list scrolls, but questionary doesn't say so — the pickers add the hint."""

    def _opts(self, n):
        return [(f"m{i}", f"m{i}") for i in range(n)]

    def test_short_list_gets_no_hint(self):
        with patch("ucode.ui.questionary.select") as sel:
            sel.return_value.ask.return_value = "m0"
            prompt_for_selection("pick", self._opts(5))
        assert "scroll" not in sel.call_args.kwargs["instruction"]

    def test_long_single_select_gets_the_hint(self):
        with patch("ucode.ui.questionary.select") as sel:
            sel.return_value.ask.return_value = "m0"
            prompt_for_selection("pick", self._opts(16))
        assert "↑/↓ scroll" in sel.call_args.kwargs["instruction"]

    def test_long_multi_select_gets_the_hint(self):
        with patch("ucode.ui.questionary.checkbox") as chk:
            chk.return_value.ask.return_value = []
            prompt_for_multi_selection("pick", self._opts(16))
        assert "↑/↓ scroll" in chk.call_args.kwargs["instruction"]

    def test_hint_preserves_the_filter_affordance(self):
        # The hint must extend the instruction, not replace it — a searchable long list keeps both.
        with patch("ucode.ui.questionary.select") as sel:
            sel.return_value.ask.return_value = "m0"
            prompt_for_selection("pick", self._opts(16), searchable=True)
        instruction = sel.call_args.kwargs["instruction"]
        assert "type to filter" in instruction
        assert "↑/↓ scroll" in instruction

    def test_threshold_is_inclusive_of_ten(self):
        # Exactly the visible-row count still fits, so no hint; one more overflows.
        with patch("ucode.ui.questionary.select") as sel:
            sel.return_value.ask.return_value = "m0"
            prompt_for_selection("pick", self._opts(10))
        assert "scroll" not in sel.call_args.kwargs["instruction"]


class TestFormatTokenCount:
    def test_small(self):
        assert format_token_count(0) == "0"
        assert format_token_count(999) == "999"

    def test_thousands(self):
        assert format_token_count(1000) == "1.0K"
        assert format_token_count(1500) == "1.5K"
        assert format_token_count(999_999) == "1000.0K"

    def test_millions(self):
        assert format_token_count(1_000_000) == "1.0M"
        assert format_token_count(2_500_000) == "2.5M"

    def test_billions(self):
        assert format_token_count(1_000_000_000) == "1.0B"
        assert format_token_count(2_200_000_000) == "2.2B"


class TestFormatDuration:
    def test_none_returns_dash(self):
        assert format_duration(None) == "-"

    def test_zero_returns_dash(self):
        assert format_duration(timedelta(seconds=0)) == "-"

    def test_negative_returns_dash(self):
        assert format_duration(timedelta(seconds=-5)) == "-"

    def test_minutes(self):
        assert format_duration(timedelta(minutes=5)) == "5m"
        assert format_duration(timedelta(minutes=59)) == "59m"

    def test_hours_fractional(self):
        result = format_duration(timedelta(hours=1, minutes=30))
        assert result == "1.5h"

    def test_hours_rounded(self):
        result = format_duration(timedelta(hours=10))
        assert result == "10h"

    def test_days(self):
        result = format_duration(timedelta(hours=48))
        assert result == "2.0d"


class TestStatusBadge:
    def test_ok_is_green(self):
        assert "green" in status_badge("OK", "ok")

    def test_warn_is_yellow(self):
        assert "yellow" in status_badge("Warning", "warn")

    def test_error_is_red(self):
        assert "red" in status_badge("Error", "error")

    def test_unknown_kind_uses_bold(self):
        result = status_badge("X", "unknown")
        assert "bold" in result
        assert "X" in result

    def test_text_is_included(self):
        assert "MyText" in status_badge("MyText", "ok")


class TestRenderBoxTable:
    def test_produces_box_chars(self):
        result = render_box_table(["A", "B"], [["x", "y"]])
        assert "┏" in result
        assert "┗" not in result  # bottom uses └
        assert "└" in result
        assert "A" in result
        assert "x" in result

    def test_empty_rows(self):
        result = render_box_table(["H1", "H2"], [])
        assert "H1" in result
        assert "H2" in result

    def test_cell_wraps_when_max_width_set(self):
        long_text = "a" * 30
        result = render_box_table(["Col"], [[long_text]], max_widths=[10])
        # wrapped lines mean the original 30-char string is broken up
        lines = result.splitlines()
        assert any(len(line.strip()) <= 14 for line in lines)

    def test_dash_for_empty_cell(self):
        result = render_box_table(["A"], [[""]])
        assert "-" in result


class TestPromptForWorkspace:
    """Cover the three things `questionary.select(...).ask()` can return:
    a (host, profile) tuple, None (cancel or "Enter a different URL"),
    or — in some questionary versions — the choice's title string."""

    PROFILES = [("https://a.databricks.com", "prof-a"), ("https://b.databricks.com", "prof-b")]

    def test_returns_selected_profile_tuple(self):
        with patch("ucode.ui.questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = (
                "https://a.databricks.com",
                "prof-a",
            )
            url, profile = prompt_for_workspace("desc", profiles=self.PROFILES)
        assert url == "https://a.databricks.com"
        assert profile == "prof-a"

    def test_none_falls_through_to_manual_prompt(self):
        with (
            patch("ucode.ui.questionary.select") as mock_select,
            patch("ucode.ui.console.input", return_value="https://manual.databricks.com"),
        ):
            mock_select.return_value.ask.return_value = None
            url, profile = prompt_for_workspace("desc", profiles=self.PROFILES)
        assert url == "https://manual.databricks.com"
        assert profile is None

    def test_string_value_falls_through_to_manual_prompt(self):
        # Regression: if questionary returns the choice title (e.g. "Enter a
        # different URL") instead of its value, we must not try to unpack it.
        with (
            patch("ucode.ui.questionary.select") as mock_select,
            patch("ucode.ui.console.input", return_value="https://manual.databricks.com"),
        ):
            mock_select.return_value.ask.return_value = "Enter a different URL"
            url, profile = prompt_for_workspace("desc", profiles=self.PROFILES)
        assert url == "https://manual.databricks.com"
        assert profile is None

    def test_no_profiles_goes_straight_to_manual_prompt(self):
        with patch("ucode.ui.console.input", return_value="example.databricks.com"):
            url, profile = prompt_for_workspace("desc", profiles=None)
        assert url == "https://example.databricks.com"
        assert profile is None


class TestFormatUsd:
    def test_rounds_to_cents(self):
        assert format_usd(Decimal("12.345")) == "$12.35"
        assert format_usd(Decimal("12.344")) == "$12.34"

    def test_pads_to_two_decimals(self):
        assert format_usd(Decimal("5")) == "$5.00"

    def test_thousands_separator(self):
        assert format_usd(Decimal("1234567.5")) == "$1,234,567.50"

    def test_zero(self):
        assert format_usd(Decimal("0")) == "$0.00"


class TestFormatMeter:
    def test_empty(self):
        assert format_meter(0.0, width=10) == "[" + "\u2591" * 10 + "]"

    def test_full(self):
        assert format_meter(1.0, width=10) == "[" + "\u2588" * 10 + "]"

    def test_half(self):
        assert format_meter(0.5, width=10) == "[" + "\u2588" * 5 + "\u2591" * 5 + "]"

    def test_tiny_nonzero_fills_one_cell(self):
        assert format_meter(0.001, width=10) == "[\u2588" + "\u2591" * 9 + "]"

    def test_clamps_above_one(self):
        assert format_meter(2.5, width=10) == "[" + "\u2588" * 10 + "]"

    def test_clamps_below_zero(self):
        assert format_meter(-1.0, width=10) == "[" + "\u2591" * 10 + "]"

    def test_width_is_constant(self):
        for fraction in (0.0, 0.13, 0.5, 0.99, 1.0):
            assert len(format_meter(fraction)) == 32
