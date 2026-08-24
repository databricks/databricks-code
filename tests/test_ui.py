"""Tests for ui.py — pure helpers that don't touch I/O or prompts."""

from __future__ import annotations

import io
from datetime import timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
import questionary
from rich.console import Console

from ucode import ui as ui_mod
from ucode.ui import (
    format_duration,
    format_meter,
    format_token_count,
    format_usd,
    inquirerpy_wizard,
    normalize_workspace_url,
    print_success,
    print_wizard_header,
    print_wizard_outro,
    print_wizard_step,
    prompt_for_multi_selection,
    prompt_for_percentage,
    prompt_for_selection,
    prompt_for_text,
    prompt_for_tools,
    prompt_for_workspace,
    prompt_yes_no_default,
    render_box_table,
    status_badge,
)


class TestWizardLayout:
    def test_header_is_branded_without_a_box(self, capsys):
        print_wizard_header("ucode setup", "Managed defaults")
        print_wizard_outro("Ready")
        out = capsys.readouterr().out
        assert "ucode setup" in out
        assert "Managed defaults" in out
        assert "┌" in out and "└" in out
        assert "╭" not in out and "╰" not in out

    def test_steps_form_a_compact_vertical_rail(self, capsys):
        print_wizard_step(1, 4, "Select the workspace")
        print_wizard_step(2, 4, "Select coding agents")
        out = capsys.readouterr().out
        assert "◇  1/4  Select the workspace" in out
        assert "◇  2/4  Select coding agents" in out
        assert "│" in out
        assert "╭" not in out and "╰" not in out

    def test_status_output_stays_inside_the_rail(self, capsys):
        @inquirerpy_wizard
        def render():
            print_wizard_header("ucode setup", "Managed defaults")
            print_wizard_step(1, 4, "Select the workspace")
            print_success("Admin permissions verified")
            print_wizard_outro("Ready")

        render()
        status_line = next(line for line in capsys.readouterr().out.splitlines() if "Admin" in line)
        assert status_line.startswith("│  ✔")

    def test_inquirer_choice_rows_stay_inside_the_rail(self):
        class FakeControl:
            def _get_hover_text(self, choice):
                return [("class:pointer", "❯"), ("", f" {choice}")]

            def _get_normal_text(self, choice):
                return [("", "  "), ("", str(choice))]

        class FakeQuestion:
            content_control = FakeControl()

        @inquirerpy_wizard
        def render():
            print_wizard_header("ucode setup", "Managed defaults")
            question = ui_mod._with_inquirer_rail(FakeQuestion())
            return (
                question.content_control._get_hover_text("first"),
                question.content_control._get_normal_text("second"),
            )

        hover, normal = render()
        assert "".join(text for _, text in hover) == "│  ❯ first"
        assert "".join(text for _, text in normal) == "│    second"


class TestInquirerPyPrompts:
    def test_agent_picker_uses_clack_style_controls_and_summary(self, monkeypatch):
        captured: dict = {}

        def fake_checkbox(**kwargs):
            captured.update(kwargs)
            return _StubInquirerPrompt(["codex", "claude"])

        monkeypatch.setattr(ui_mod.inquirer, "checkbox", fake_checkbox)

        @inquirerpy_wizard
        def run():
            print_wizard_header("ucode setup", "Managed defaults")
            result = prompt_for_tools(
                [("codex", "Codex"), ("claude", "Claude Code"), ("pi", "Pi")],
                preselected=["codex", "claude"],
                prompt="Coding agents:",
            )
            print_wizard_outro("Ready")
            return result

        assert run() == ["codex", "claude"]
        assert captured["qmark"] == "│ "
        assert captured["enabled_symbol"] == "◉"
        assert captured["disabled_symbol"] == "○"
        assert captured["transformer"](["codex", "claude"]) == "Codex, Claude Code"

    def test_short_searchable_picker_has_one_cursor_and_no_blank_filter_row(self, monkeypatch):
        captured: dict = {}

        def fake_select(**kwargs):
            captured.update(kwargs)
            return _StubInquirerPrompt("custom")

        monkeypatch.setattr(ui_mod.inquirer, "select", fake_select)
        monkeypatch.setattr(
            ui_mod.inquirer,
            "fuzzy",
            lambda **_: pytest.fail("a short picker should not render a fuzzy-search input"),
        )

        @inquirerpy_wizard
        def run():
            return prompt_for_selection(
                "Default fable model:",
                [("fable", "system.ai.claude-fable-5"), ("custom", "Enter a custom model…")],
                searchable=True,
            )

        assert run() == "custom"
        assert captured["pointer"] == "❯"

    def test_long_fuzzy_picker_labels_its_filter_row(self, monkeypatch):
        captured: dict = {}

        def fake_fuzzy(**kwargs):
            captured.update(kwargs)
            return _StubInquirerPrompt("m0")

        monkeypatch.setattr(ui_mod.inquirer, "fuzzy", fake_fuzzy)

        @inquirerpy_wizard
        def run():
            print_wizard_header("ucode setup", "Managed defaults")
            result = prompt_for_selection(
                "Model:", [(f"m{i}", f"model {i}") for i in range(11)], searchable=True
            )
            print_wizard_outro("Ready")
            return result

        assert run() == "m0"
        assert captured["prompt"] == "│  Filter"

    def test_long_fuzzy_multi_picker_toggles_with_space(self, monkeypatch):
        captured: dict = {}

        def fake_fuzzy(**kwargs):
            captured.update(kwargs)
            return _StubInquirerPrompt(["m0"])

        monkeypatch.setattr(ui_mod.inquirer, "fuzzy", fake_fuzzy)

        @inquirerpy_wizard
        def run():
            print_wizard_header("ucode setup", "Managed defaults")
            result = prompt_for_multi_selection(
                "Models:", [(f"m{i}", f"model {i}") for i in range(11)], searchable=True
            )
            print_wizard_outro("Ready")
            return result

        assert run() == ["m0"]
        assert captured["prompt"] == "│  Filter"
        assert captured["keybindings"] == {"toggle": [{"key": "space"}]}

    def test_confirmation_names_the_default_instead_of_claiming_it_is_highlighted(
        self, monkeypatch
    ):
        captured: dict = {}

        def fake_confirm(**kwargs):
            captured.update(kwargs)
            return _StubInquirerPrompt(False)

        monkeypatch.setattr(ui_mod.inquirer, "confirm", fake_confirm)

        @inquirerpy_wizard
        def run():
            return prompt_yes_no_default("Publish?", default=False)

        assert run() is False
        assert captured["instruction"] == "y/N · enter selects No"
        assert "highlight" not in captured["instruction"]


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


class TestPromptForWorkspaceFallbacks:
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


class _StubQuestion:
    def __init__(self, answer):
        self._answer = answer

    def ask(self):
        return self._answer


class _StubInquirerPrompt:
    def __init__(self, answer):
        self._answer = answer

    def execute(self):
        return self._answer


class TestPromptForWorkspace:
    """Capture the choices passed to ``questionary.select`` so we can assert on
    layout (header alignment + duplicate-host preservation) without driving
    real keyboard I/O."""

    def _capture_select(self, monkeypatch, answer):
        captured: dict = {}

        def fake_select(message, choices, **kwargs):
            captured["message"] = message
            captured["choices"] = choices
            captured["kwargs"] = kwargs
            return _StubQuestion(answer)

        monkeypatch.setattr(questionary, "select", fake_select)
        monkeypatch.setattr(ui_mod.questionary, "select", fake_select)
        return captured

    def test_shows_header_and_each_profile_row(self, monkeypatch):
        profiles = [
            ("https://a.cloud.databricks.com", "alpha"),
            ("https://b.cloud.databricks.com", "beta-profile-name"),
        ]
        captured = self._capture_select(monkeypatch, answer=profiles[0])
        url, profile = prompt_for_workspace("setup", profiles)

        assert (url, profile) == profiles[0]
        choices = captured["choices"]
        # Header (separator), 2 rows, "Enter a different URL" entry.
        assert len(choices) == 4
        assert isinstance(choices[0], questionary.Separator)
        header = choices[0].title
        assert "Profile Name" in header
        assert "Workspace URL" in header
        # Profile names ljust-padded to the longest name (17 chars).
        name_width = max(len(name) for _, name in profiles)
        assert "alpha".ljust(name_width) in choices[1].title
        assert profiles[0][0] in choices[1].title
        assert "beta-profile-name".ljust(name_width) in choices[2].title
        assert profiles[1][0] in choices[2].title
        # Final fallback entry still present.
        assert choices[3].title == "Enter a different URL"

    def test_uses_the_standard_section_panel(self, monkeypatch):
        profiles = [("https://a.cloud.databricks.com", "alpha")]
        self._capture_select(monkeypatch, answer=profiles[0])
        with patch.object(ui_mod, "print_section") as section:
            prompt_for_workspace("Databricks workspace", profiles)
        section.assert_called_once_with("Databricks workspace")

    def test_larger_wizard_can_supply_its_own_step_banner(self, monkeypatch):
        profiles = [("https://a.cloud.databricks.com", "alpha")]
        captured = self._capture_select(monkeypatch, answer=profiles[0])
        with patch.object(ui_mod, "print_section") as section:
            prompt_for_workspace("unused", profiles, show_section=False, prompt="Workspace:")
        section.assert_not_called()
        assert captured["message"] == "Workspace:"

    def test_inquirerpy_workspace_prompt_inherits_the_wizard_rail(self, monkeypatch):
        profiles = [("https://a.cloud.databricks.com", "alpha")]
        captured: dict = {}

        def fake_select(**kwargs):
            captured.update(kwargs)
            return _StubInquirerPrompt(profiles[0])

        monkeypatch.setattr(ui_mod.inquirer, "select", fake_select)

        @inquirerpy_wizard
        def run():
            print_wizard_header("ucode setup", "Managed defaults")
            result = prompt_for_workspace(
                "unused", profiles, show_section=False, prompt="Workspace:"
            )
            print_wizard_outro("Ready")
            return result

        assert run() == profiles[0]
        assert captured["qmark"] == "│ "
        assert captured["amark"] == "│ "
        assert captured["pointer"] == "❯"
        selected_row = "alpha  https://a.cloud.databricks.com"
        assert captured["transformer"](selected_row) == selected_row

    def test_keeps_duplicate_hosts_as_separate_rows(self, monkeypatch):
        profiles = [
            ("https://shared.cloud.databricks.com", "first"),
            ("https://shared.cloud.databricks.com", "second"),
        ]
        captured = self._capture_select(monkeypatch, answer=profiles[1])
        url, profile = prompt_for_workspace("setup", profiles)

        assert (url, profile) == profiles[1]
        # Both rows present — duplicates not collapsed.
        choices = captured["choices"]
        # Filter to choices whose value is a (host, profile) tuple — drops the
        # header separator and the trailing "Enter a different URL" entry.
        host_choices = [c for c in choices if isinstance(getattr(c, "value", None), tuple)]
        assert [c.value for c in host_choices] == profiles

    def test_returns_normalized_url_with_profile(self, monkeypatch):
        # Picker handed back a URL with a trailing slash — normalize_workspace_url
        # should strip it before returning.
        profiles = [("https://example.cloud.databricks.com/", "p")]
        self._capture_select(monkeypatch, answer=profiles[0])
        url, profile = prompt_for_workspace("setup", profiles)
        assert url == "https://example.cloud.databricks.com"
        assert profile == "p"

    # ------------------------------------------------------------------
    # Long-name display clamping (PR #114 review feedback)
    # ------------------------------------------------------------------

    def test_long_profile_name_is_truncated_in_display_only(self, monkeypatch):
        # 60-char name — exceeds the 40-char clamp. The displayed row title
        # must be truncated with an ellipsis but the value tuple must carry
        # the full untruncated name through to configure_shared_state.
        long_name = "x" * 60
        profiles = [("https://a.cloud.databricks.com", long_name)]
        captured = self._capture_select(monkeypatch, answer=profiles[0])
        url, profile = prompt_for_workspace("setup", profiles)

        assert (url, profile) == profiles[0]
        choices = captured["choices"]
        # Header + 1 row + "Enter a different URL".
        assert len(choices) == 3
        # Display title is truncated to 40 chars (39 of name + "…").
        row_title = choices[1].title
        assert long_name not in row_title
        assert "…" in row_title
        # Value tuple still carries the full name.
        assert choices[1].value == profiles[0]


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


class TestChoiceViewportCap:
    """`_cap_choice_viewport` pins long picker lists to a fixed scrolling window."""

    @staticmethod
    def _choice_window_height(question):
        from prompt_toolkit.layout.containers import Window
        from questionary.prompts.common import InquirerControl

        for window in question.application.layout.find_all_windows():
            if isinstance(window, Window) and isinstance(window.content, InquirerControl):
                return window.height
        raise AssertionError("no InquirerControl window found")

    def test_short_list_keeps_natural_height(self):
        # At or below the threshold everything fits, so the window is left unbounded (height=None)
        # rather than padded to a fixed size.
        n = ui_mod._SCROLL_HINT_THRESHOLD
        question = questionary.select("p", choices=[f"m{i}" for i in range(n)])
        ui_mod._cap_choice_viewport(question, n)
        assert self._choice_window_height(question) is None

    def test_long_list_is_capped_to_the_threshold(self):
        n = ui_mod._SCROLL_HINT_THRESHOLD + 15
        question = questionary.select("p", choices=[f"m{i}" for i in range(n)])
        ui_mod._cap_choice_viewport(question, n)
        height = self._choice_window_height(question)
        assert height.max == ui_mod._SCROLL_HINT_THRESHOLD
        assert height.preferred == ui_mod._SCROLL_HINT_THRESHOLD

    def test_checkbox_list_is_capped_too(self):
        n = ui_mod._SCROLL_HINT_THRESHOLD + 15
        question = questionary.checkbox("p", choices=[f"m{i}" for i in range(n)])
        ui_mod._cap_choice_viewport(question, n)
        assert self._choice_window_height(question).max == ui_mod._SCROLL_HINT_THRESHOLD

    def test_missing_application_is_a_no_op(self):
        # Best-effort: a question shape without an application must not raise.
        ui_mod._cap_choice_viewport(object(), ui_mod._SCROLL_HINT_THRESHOLD + 5)
