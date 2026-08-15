"""Tests for Drawdown Overlay risk control logic."""

from __future__ import annotations

from src.domain.futures.portfolio.risk_controls import compute_drawdown_gross_scale


def test_drawdown_overlay_tier2_scaling() -> None:
    """Test that scale drops to Tier 2 (0.40) under -15% return."""
    scale = compute_drawdown_gross_scale(rolling_30d_return=-0.16, current_scale=1.0)
    assert scale == 0.40


def test_drawdown_overlay_tier1_scaling() -> None:
    """Test that scale drops to Tier 1 (0.70) between -15% and -10% return."""
    scale = compute_drawdown_gross_scale(rolling_30d_return=-0.12, current_scale=1.0)
    assert scale == 0.70


def test_drawdown_overlay_no_change_in_intermediate_zone() -> None:
    """Test that scale is maintained between -10% and -5% return."""
    # From 1.0
    scale1 = compute_drawdown_gross_scale(rolling_30d_return=-0.08, current_scale=1.0)
    assert scale1 == 1.0

    # From 0.70
    scale2 = compute_drawdown_gross_scale(rolling_30d_return=-0.08, current_scale=0.70)
    assert scale2 == 0.70

    # From 0.40
    scale3 = compute_drawdown_gross_scale(rolling_30d_return=-0.08, current_scale=0.40)
    assert scale3 == 0.40


def test_drawdown_overlay_stepwise_recovery() -> None:
    """Test stepwise recovery when return is above -5%."""
    # Recovery from Tier 2 (0.40) -> Tier 1 (0.70)
    scale1 = compute_drawdown_gross_scale(rolling_30d_return=-0.03, current_scale=0.40)
    assert scale1 == 0.70

    # Recovery from Tier 1 (0.70) -> Full (1.0)
    scale2 = compute_drawdown_gross_scale(rolling_30d_return=-0.03, current_scale=0.70)
    assert scale2 == 1.0


def test_drawdown_overlay_normal_maintenance() -> None:
    """Test that scale stays at 1.0 in normal conditions."""
    scale = compute_drawdown_gross_scale(rolling_30d_return=0.05, current_scale=1.0)
    assert scale == 1.0
