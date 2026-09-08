"""Render the workspace budget a managed coding-agent config is spending against.

The figures come from the AI Gateway's ``:recommendModel`` response (``current_spend`` and
``effective_threshold``), normalized by :func:`ucode.managed_config.get_model_recommendation`.
Enforcement is entirely server-side — the gateway rejects over-budget requests — so this module
only reports spend and never blocks a launch.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.text import Text

# Fallback amber point for a workspace whose policy defines no tier to derive one from.
BUDGET_WARN_AT = 0.8

_BUDGET_STATE_STYLE = {"ok": "green", "warn": "yellow", "exceeded": "red"}


def budget_warn_fraction(managed: dict | None) -> float:
    """The spend fraction at which to warn: the admin's lowest activating tier, else 0.8.

    Tiers at 0 are skipped — they activate from the first dollar, so warning on one would leave the
    panel permanently amber.
    """
    policy = (managed or {}).get("budget_policy")
    tiers = policy.get("tiers") if isinstance(policy, dict) else None
    fractions = [
        float(pct)
        for tier in (tiers if isinstance(tiers, list) else [])
        if isinstance(tier, dict)
        # `bool` is an `int` subclass, so it would otherwise read as a 0/1 fraction.
        if isinstance(pct := tier.get("spending_percentage"), int | float)
        and not isinstance(pct, bool)
        and 0 < pct <= 1
    ]
    return min(fractions) if fractions else BUDGET_WARN_AT


def budget_state(spend: float, threshold: float, warn_at: float = BUDGET_WARN_AT) -> str:
    """Classify spend against its threshold as ``ok``, ``warn``, or ``exceeded``."""
    if threshold <= 0:
        return "ok"
    if spend >= threshold:
        return "exceeded"
    if spend >= threshold * warn_at:
        return "warn"
    return "ok"


def budget_usage_percent(spend: float, threshold: float) -> int:
    """Spend as a whole percentage of its threshold, rounded half-up and floored at 0."""
    if threshold <= 0:
        return 0
    return max(int(((spend / threshold) * 100) + 0.5), 0)


def _bar_markup(percent: int, color: str, *, width: int = 28) -> str:
    """A Rich-markup fill bar: the filled portion in the state color, the remainder dimmed."""
    capped = min(max(percent, 0), 100)
    filled = min(max((capped * width + 50) // 100, 0), width)
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (width - filled)}[/dim]"


def _header_line(state: str) -> str | None:
    """The status callout shown above the bar for non-ok states."""
    if state == "exceeded":
        return "[bold red]⛔ Workspace budget exceeded[/bold red]"
    if state == "warn":
        return "[bold yellow]⚠️  Nearing workspace budget[/bold yellow]"
    return None


def render_budget_panel(
    recommendation: dict,
    *,
    title: str | None = None,
    extra_lines: list[str] | None = None,
    managed: dict | None = None,
) -> Panel | None:
    """Render the managed budget as a bordered panel with a color-coded fill bar.

    Returns None when the recommendation carries no threshold to measure against, so a workspace
    with no budget policy shows nothing rather than an empty bar. ``managed`` supplies the admin's
    tiers, so the amber point matches where their policy actually starts stepping down.
    """
    threshold = recommendation.get("effective_threshold")
    spend = recommendation.get("current_spend")
    if not isinstance(threshold, int | float) or isinstance(threshold, bool) or threshold <= 0:
        return None
    spend = float(spend) if isinstance(spend, int | float) and not isinstance(spend, bool) else 0.0
    threshold = float(threshold)

    state = budget_state(spend, threshold, budget_warn_fraction(managed))
    color = _BUDGET_STATE_STYLE.get(state, "green")
    percent = budget_usage_percent(spend, threshold)

    lines: list[str] = []
    header = _header_line(state)
    if header:
        lines.append(header)
        lines.append("")
    lines.append(
        f"[bold]${spend:,.2f}[/bold] / ${threshold:,.2f}    [{color}]{percent}% used[/{color}]"
    )
    lines.append(_bar_markup(percent, color))
    lines.append("")
    if extra_lines:
        lines.extend(extra_lines)
        lines.append("")
    lines.append(f"[bold]Remaining[/bold]   ${max(threshold - spend, 0.0):,.2f}")

    return Panel(
        Text.from_markup("\n".join(lines)),
        title=Text(title or "Workspace Budget", style=f"bold {color}"),
        border_style=color,
        expand=False,
        padding=(1, 2, 0, 2),
    )


def recommendation_line(display_agent: str | None, model: str | None, percent: int) -> str | None:
    """The "you've used N%, recommended is X" sentence shown inside the panel."""
    if not display_agent and not model:
        return None
    used = f"You've used [bold]{percent}%[/bold] of the workspace budget. "
    if display_agent and model:
        return f"{used}Recommended agent is [bold]{display_agent}[/bold] with model [bold]{model}[/bold]."
    if display_agent:
        return f"{used}Recommended agent is [bold]{display_agent}[/bold]."
    return f"{used}Recommended model is [bold]{model}[/bold]."
