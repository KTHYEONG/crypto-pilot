"""Unit tests for compute_mean_trend_efficiency (L2 fit/cal ER aggregation)."""
from __future__ import annotations

import pytest

from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    Layer2FoldAttribution,
    compute_long_short_realized_price,
    compute_mean_trend_efficiency,
)


def _make_attr(
    *,
    oos_bars: int = 100,
    mean_trend_efficiency: float = 0.0,
    trend_efficiency_corr: float = 0.0,
    realized_price_long: float = 0.0,
    realized_price_short: float = 0.0,
) -> Layer2FoldAttribution:
    return Layer2FoldAttribution(
        fold_idx=0,
        oos_bars=oos_bars,
        n_rebal=10,
        realized_total=0.0,
        realized_price=0.0,
        realized_funding=0.0,
        realized_cost=0.0,
        expected_net=0.0,
        alpha_gap=0.0,
        mean_gross_exp=0.5,
        mean_net_exp=0.1,
        sleeves_active_mean=10.0,
        friction_pass_ratio=0.8,
        throttle_mult_mean=1.0,
        dropped_below_cost=0,
        netting_events=0,
        mean_trend_efficiency=mean_trend_efficiency,
        trend_efficiency_corr=trend_efficiency_corr,
        realized_price_long=realized_price_long,
        realized_price_short=realized_price_short,
    )


class TestComputeMeanTrendEfficiency:
    def test_single_fold_returns_its_own_values(self) -> None:
        """Given: 단일 fold, ER=0.30, corr=-0.10.
        When: compute_mean_trend_efficiency.
        Then: 가중평균이 fold 값과 동일.
        """
        mean_er, mean_corr = compute_mean_trend_efficiency(
            (_make_attr(oos_bars=100, mean_trend_efficiency=0.30, trend_efficiency_corr=-0.10),)
        )
        assert mean_er == pytest.approx(0.30)
        assert mean_corr == pytest.approx(-0.10)

    def test_multi_fold_weighted_by_oos_bars(self) -> None:
        """Given: fold A(bars=100, ER=0.10), fold B(bars=300, ER=0.50).
        When: compute_mean_trend_efficiency.
        Then: (100*0.10 + 300*0.50)/400 = 0.40 (bar-count weighted, not simple mean).
        """
        attrs = (
            _make_attr(oos_bars=100, mean_trend_efficiency=0.10),
            _make_attr(oos_bars=300, mean_trend_efficiency=0.50),
        )
        mean_er, _ = compute_mean_trend_efficiency(attrs)
        assert mean_er == pytest.approx(0.40)
        # simple (unweighted) mean would be 0.30 — must not match
        assert mean_er != pytest.approx(0.30)

    def test_empty_attributions_returns_zero(self) -> None:
        """Given: 빈 tuple.
        When: compute_mean_trend_efficiency.
        Then: (0.0, 0.0) 반환.
        """
        mean_er, mean_corr = compute_mean_trend_efficiency(())
        assert mean_er == pytest.approx(0.0)
        assert mean_corr == pytest.approx(0.0)

    def test_all_zero_oos_bars_returns_zero_not_nan(self) -> None:
        """Given: 모든 fold의 oos_bars=0 (분모 0).
        When: compute_mean_trend_efficiency.
        Then: ZeroDivisionError 대신 (0.0, 0.0) 반환.
        """
        attrs = (_make_attr(oos_bars=0, mean_trend_efficiency=0.9),)
        mean_er, mean_corr = compute_mean_trend_efficiency(attrs)
        assert mean_er == pytest.approx(0.0)
        assert mean_corr == pytest.approx(0.0)

    def test_negative_oos_bars_clamped_to_zero_weight(self) -> None:
        """Given: 한 fold의 oos_bars가 음수(방어적 sanity — 실제로는 발생 불가 케이스).
        When: compute_mean_trend_efficiency.
        Then: 음수 weight가 아닌 0으로 clamp되어 다른 정상 fold만 반영.
        """
        attrs = (
            _make_attr(oos_bars=-5, mean_trend_efficiency=0.99),
            _make_attr(oos_bars=50, mean_trend_efficiency=0.20),
        )
        mean_er, _ = compute_mean_trend_efficiency(attrs)
        assert mean_er == pytest.approx(0.20)


class TestComputeLongShortRealizedPrice:
    """compute_long_short_realized_price sum across folds."""

    def test_compute_long_short_realized_price_sums_across_folds(self) -> None:
        """Scenario 1: 두 fold의 long/short realized_price를 단순 합산."""
        attrs = (
            _make_attr(realized_price_long=-0.05, realized_price_short=0.01),
            _make_attr(realized_price_long=-0.03, realized_price_short=0.02),
        )
        total_long, total_short = compute_long_short_realized_price(attrs)
        assert total_long == pytest.approx(-0.08)
        assert total_short == pytest.approx(0.03)

    def test_compute_long_short_realized_price_empty_returns_zero(self) -> None:
        """Scenario 2: 빈 fold_attributions → (0.0, 0.0)."""
        total_long, total_short = compute_long_short_realized_price(())
        assert total_long == pytest.approx(0.0)
        assert total_short == pytest.approx(0.0)
