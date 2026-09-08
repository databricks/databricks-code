"""Tests for managed_budget.py — the workspace budget panel shown at launch."""

from __future__ import annotations

import pytest

from ucode.managed_budget import (
    BUDGET_WARN_AT,
    budget_state,
    budget_usage_percent,
    budget_warn_fraction,
    recommendation_line,
    render_budget_panel,
)


class TestBudgetState:
    @pytest.mark.parametrize(
        ("spend", "threshold", "expected"),
        [
            (0.0, 500.0, "ok"),
            (399.0, 500.0, "ok"),
            (400.0, 500.0, "warn"),
            (499.0, 500.0, "warn"),
            (500.0, 500.0, "exceeded"),
            (750.0, 500.0, "exceeded"),
        ],
    )
    def test_classifies_spend(self, spend, threshold, expected):
        assert budget_state(spend, threshold) == expected

    def test_no_threshold_is_never_over_budget(self):
        # A workspace with no budget policy has nothing to exceed.
        assert budget_state(100.0, 0.0) == "ok"

    def test_warn_point_follows_the_policy(self):
        # At 60% spend a policy whose first tier is 50% is already stepping down, so warn.
        assert budget_state(300.0, 500.0, 0.5) == "warn"
        assert budget_state(300.0, 500.0, 0.8) == "ok"


class TestBudgetWarnFraction:
    """The amber point comes from the admin's own tiers, not a fixed 80%."""

    @staticmethod
    def _policy(*percentages):
        return {"budget_policy": {"tiers": [{"spending_percentage": p} for p in percentages]}}

    def test_uses_the_lowest_activating_tier(self):
        assert budget_warn_fraction(self._policy(0.5, 0.8)) == 0.5

    def test_ignores_tiers_at_zero(self):
        # A 0.0 tier activates from the first dollar, so warning on it would be permanent amber.
        assert budget_warn_fraction(self._policy(0.0, 0.5, 0.8)) == 0.5
        assert budget_warn_fraction(self._policy(0.0)) == BUDGET_WARN_AT

    def test_falls_back_without_a_policy(self):
        assert budget_warn_fraction(None) == BUDGET_WARN_AT
        assert budget_warn_fraction({}) == BUDGET_WARN_AT
        assert budget_warn_fraction({"budget_policy": {"tiers": []}}) == BUDGET_WARN_AT

    @pytest.mark.parametrize("bad", [True, "0.5", None, 1.5, -0.2])
    def test_ignores_unusable_percentages(self, bad):
        # `True` would otherwise read as a 1.0 fraction, and out-of-range values aren't meaningful.
        assert budget_warn_fraction(self._policy(bad)) == BUDGET_WARN_AT


class TestBudgetUsagePercent:
    @pytest.mark.parametrize(
        ("spend", "threshold", "expected"),
        [(412.5, 500.0, 83), (0.0, 500.0, 0), (500.0, 500.0, 100), (600.0, 500.0, 120)],
    )
    def test_rounds_half_up(self, spend, threshold, expected):
        assert budget_usage_percent(spend, threshold) == expected

    def test_zero_threshold_is_zero_percent(self):
        assert budget_usage_percent(100.0, 0.0) == 0


class TestRenderBudgetPanel:
    def test_none_without_a_threshold(self):
        # No budget policy on the workspace means no bar to draw.
        assert render_budget_panel({"current_spend": 10.0}) is None
        assert render_budget_panel({"current_spend": 10.0, "effective_threshold": 0.0}) is None

    def test_renders_the_bar_and_spend(self):
        panel = render_budget_panel({"current_spend": 412.5, "effective_threshold": 500.0})
        assert panel is not None
        body = panel.renderable.plain
        assert "$412.50 / $500.00" in body
        assert "83% used" in body
        assert "█" in body and "░" in body
        assert "Remaining" in body and "$87.50" in body

    def test_over_budget_remaining_never_goes_negative(self):
        panel = render_budget_panel({"current_spend": 600.0, "effective_threshold": 500.0})
        assert panel is not None
        assert "$0.00" in panel.renderable.plain

    def test_missing_spend_is_treated_as_zero(self):
        panel = render_budget_panel({"effective_threshold": 500.0})
        assert panel is not None
        assert "$0.00 / $500.00" in panel.renderable.plain


class TestRecommendationLine:
    def test_names_both_agent_and_model(self):
        line = recommendation_line("OpenCode", "system.ai.claude-haiku-4-5", 83)
        assert "83%" in line
        assert "OpenCode" in line and "system.ai.claude-haiku-4-5" in line

    def test_agent_only(self):
        assert "OpenCode" in recommendation_line("OpenCode", None, 50)

    def test_model_only(self):
        # The server can recommend a model without an agent.
        line = recommendation_line(None, "system.ai.gpt-5", 50)
        assert "system.ai.gpt-5" in line

    def test_none_when_nothing_recommended(self):
        assert recommendation_line(None, None, 50) is None
