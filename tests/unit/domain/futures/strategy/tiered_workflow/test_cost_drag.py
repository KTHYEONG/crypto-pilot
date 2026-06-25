"""Unit tests for compute_cost_drag_ratio fix (denom explosion bug + cap)."""
from __future__ import annotations

import pytest

from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    Layer2FoldAttribution,
    compute_cost_drag_ratio,
)


def _make_attr(
    *,
    realized_price: float = 100.0,
    realized_cost: float = 30.0,
) -> Layer2FoldAttribution:
    return Layer2FoldAttribution(
        fold_idx=0,
        oos_bars=100,
        n_rebal=10,
        realized_total=realized_price - realized_cost,
        realized_price=realized_price,
        realized_funding=0.0,
        realized_cost=realized_cost,
        expected_net=50.0,
        alpha_gap=realized_price - realized_cost - 50.0,
        mean_gross_exp=0.5,
        mean_net_exp=0.1,
        sleeves_active_mean=10.0,
        friction_pass_ratio=0.8,
        throttle_mult_mean=1.0,
        dropped_below_cost=0,
        netting_events=0,
    )


class TestCostDragRatio:
    def test_normal_positive_price(self) -> None:
        """Given: 단일 fold, realized_price=100, realized_cost=30.
        When: compute_cost_drag_ratio.
        Then: 30.0 / max(100.0, 1e-9) = 0.30.
        """
        ratio = compute_cost_drag_ratio(
            (_make_attr(realized_price=100.0, realized_cost=30.0),)
        )
        assert ratio == pytest.approx(0.30, rel=1e-3)

    def test_negative_price_uses_abs(self) -> None:
        """Given: realized_price=-50 (net negative gross), realized_cost=10.
        When: compute_cost_drag_ratio.
        Then: 분모가 abs(-50)=50 → 10.0/50.0 = 0.20.
        """
        ratio = compute_cost_drag_ratio(
            (_make_attr(realized_price=-50.0, realized_cost=10.0),)
        )
        assert ratio == pytest.approx(0.20, rel=1e-3)

    def test_zero_price_capped_at_100(self) -> None:
        """Given: realized_price=0, realized_cost=5.0.
        When: compute_cost_drag_ratio.
        Then: total_price_abs=0 → denom=1e-9 → 5e9, capped at 100.0.
        """
        ratio = compute_cost_drag_ratio(
            (_make_attr(realized_price=0.0, realized_cost=5.0),)
        )
        assert ratio == pytest.approx(100.0, rel=1e-3)

    def test_empty_attributions_returns_zero(self) -> None:
        """Given: 빈 tuple.
        When: compute_cost_drag_ratio.
        Then: 0.0 반환.
        """
        ratio = compute_cost_drag_ratio(())
        assert ratio == pytest.approx(0.0)

    def test_multi_fold_long_short_cancels(self) -> None:
        """Given: 2 folds, realized_price=+100 and -100 (long/short cancel).
        When: compute_cost_drag_ratio.
        Then: total_price_abs = |100| + |-100| = 200 → (30+30)/200 = 0.30.
        """
        attrs = (
            _make_attr(realized_price=100.0, realized_cost=30.0),
            _make_attr(realized_price=-100.0, realized_cost=30.0),
        )
        ratio = compute_cost_drag_ratio(attrs)
        assert ratio == pytest.approx(0.30, rel=1e-3)

    def test_minuscule_price_still_capped(self) -> None:
        """Given: realized_price=1e-10 (매우 작음), realized_cost=50.
        When: compute_cost_drag_ratio.
        Then: total_price_abs=1e-10 → 분모=1e-9 → ratio capped at 100.0.
        """
        ratio = compute_cost_drag_ratio(
            (_make_attr(realized_price=1e-10, realized_cost=50.0),)
        )
        assert ratio == pytest.approx(100.0, rel=1e-3)
