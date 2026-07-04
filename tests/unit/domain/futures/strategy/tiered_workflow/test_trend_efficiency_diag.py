"""Unit tests for compute_mean_trend_efficiency (L2 fit/cal ER aggregation)."""
from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    Layer2FoldAttribution,
    _assemble_fold_attribution,
    compute_long_short_price_by_symbol,
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


class TestComputeLongShortPriceBySymbol:
    """compute_long_short_price_by_symbol merges per-symbol tuples across folds."""

    def test_compute_long_short_price_by_symbol_merges_across_folds(self) -> None:
        """Scenario 1: overlapping symbols are summed, non-overlapping preserved."""
        attrs = (
            Layer2FoldAttribution(
                fold_idx=0, oos_bars=100, n_rebal=10,
                realized_total=0.0, realized_price=0.0,
                realized_funding=0.0, realized_cost=0.0,
                expected_net=0.0, alpha_gap=0.0,
                mean_gross_exp=0.5, mean_net_exp=0.1,
                sleeves_active_mean=10.0, friction_pass_ratio=0.8,
                throttle_mult_mean=1.0, dropped_below_cost=0,
                netting_events=0,
                realized_price_long_by_symbol=(("BTCUSDT", -0.02), ("ETHUSDT", -0.01)),
                realized_price_short_by_symbol=(),
            ),
            Layer2FoldAttribution(
                fold_idx=1, oos_bars=100, n_rebal=10,
                realized_total=0.0, realized_price=0.0,
                realized_funding=0.0, realized_cost=0.0,
                expected_net=0.0, alpha_gap=0.0,
                mean_gross_exp=0.5, mean_net_exp=0.1,
                sleeves_active_mean=10.0, friction_pass_ratio=0.8,
                throttle_mult_mean=1.0, dropped_below_cost=0,
                netting_events=0,
                realized_price_long_by_symbol=(("BTCUSDT", -0.01), ("SOLUSDT", 0.005)),
                realized_price_short_by_symbol=(),
            ),
        )
        long_totals, short_totals = compute_long_short_price_by_symbol(attrs)
        long_dict = dict(long_totals)
        assert long_dict == pytest.approx({"BTCUSDT": -0.03, "ETHUSDT": -0.01, "SOLUSDT": 0.005})
        assert short_totals == ()

    def test_compute_long_short_price_by_symbol_empty_returns_empty_tuples(self) -> None:
        """Scenario 2: empty fold_attributions → ((), ())."""
        long_totals, short_totals = compute_long_short_price_by_symbol(())
        assert long_totals == ()
        assert short_totals == ()


class TestAssembleFoldAttributionPerSymbol:
    """_assemble_fold_attribution builds per-symbol long/short tuples."""

    def test_assemble_fold_attribution_builds_per_symbol_long_short_tuples(self) -> None:
        """Scenario 3: near-zero contributors filtered, correct pairing."""
        symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
        price_long_by_sym = np.array([-0.02, 0.0, 1e-13])
        price_short_by_sym = np.array([0.0, 0.015, 0.0])
        result = _assemble_fold_attribution(
            fold_idx=0, oos_bars=100, n_rebal=10,
            realized_price=0.0, realized_funding=0.0,
            realized_cost=0.0, expected_net=0.0,
            gross_exps=[], net_exps=[], throttle_mults=[],
            sleeves_active=[], friction_pass_total=0,
            signal_total=0, dropped_below_cost=0,
            netting_events=0,
            symbols=symbols,
            price_long_by_sym=price_long_by_sym,
            price_short_by_sym=price_short_by_sym,
        )
        assert result.realized_price_long_by_symbol == (("BTCUSDT", pytest.approx(-0.02)),)
        assert result.realized_price_short_by_symbol == (("ETHUSDT", pytest.approx(0.015)),)

    def test_assemble_fold_attribution_defaults_per_symbol_to_empty_when_arrays_omitted(self) -> None:
        """Scenario 4: no symbols/price_long_by_sym → empty tuples."""
        result = _assemble_fold_attribution(
            fold_idx=0, oos_bars=100, n_rebal=10,
            realized_price=0.0, realized_funding=0.0,
            realized_cost=0.0, expected_net=0.0,
            gross_exps=[], net_exps=[], throttle_mults=[],
            sleeves_active=[], friction_pass_total=0,
            signal_total=0, dropped_below_cost=0,
            netting_events=0,
        )
        assert result.realized_price_long_by_symbol == ()
        assert result.realized_price_short_by_symbol == ()


class TestSummarizeMajorSymbolSignalSizing:
    """summarize_major_symbol_signal_sizing computes signal/sizing mismatch ratios."""

    def test_summarize_major_symbol_signal_sizing_computes_signal_sizing_mismatch_ratios(self) -> None:
        """Scenario 1: 단일 fold BTCUSDT 4개 스냅샷 → 정확한 비율."""
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            MajorSymbolRebalanceSnapshot,
            MajorSymbolSignalSizingSummary,
            summarize_major_symbol_signal_sizing,
        )
        S = MajorSymbolRebalanceSnapshot
        snapshots = (
            S(t=0, symbol="BTCUSDT", raw_mu=5.0, weight=0.10, regime_code=0, regime_risk_mult=1.0),
            S(t=1, symbol="BTCUSDT", raw_mu=3.0, weight=0.10, regime_code=0, regime_risk_mult=1.0),
            S(t=2, symbol="BTCUSDT", raw_mu=-2.0, weight=0.10, regime_code=1, regime_risk_mult=0.8),
            S(t=3, symbol="BTCUSDT", raw_mu=-1.0, weight=0.0, regime_code=1, regime_risk_mult=0.8),
        )
        fold_attributions = (
            Layer2FoldAttribution(
                fold_idx=0, oos_bars=4, n_rebal=4, realized_total=0.0, realized_price=0.0,
                realized_funding=0.0, realized_cost=0.0, expected_net=0.0, alpha_gap=0.0,
                mean_gross_exp=0.0, mean_net_exp=0.0, sleeves_active_mean=0.0,
                friction_pass_ratio=0.0, throttle_mult_mean=1.0, dropped_below_cost=0,
                netting_events=0, major_symbol_snapshots=snapshots,
            ),
        )
        result = summarize_major_symbol_signal_sizing(fold_attributions)
        assert result == (
            MajorSymbolSignalSizingSummary(
                symbol="BTCUSDT", n_obs=4,
                mu_bullish_pct=pytest.approx(0.5),
                weight_long_pct=pytest.approx(0.75),
                stale_long_pct=pytest.approx(0.25),
                regime_cap_engaged_pct=pytest.approx(0.25),
                mean_regime_risk_mult_when_long=pytest.approx(0.93333, rel=1e-4),
            ),
        )

    def test_summarize_major_symbol_signal_sizing_merges_folds_and_guards_zero_long_bars(self) -> None:
        """Scenario 2a: 여러 fold 병합 + 0-division guard."""
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            MajorSymbolRebalanceSnapshot,
            summarize_major_symbol_signal_sizing,
        )
        S = MajorSymbolRebalanceSnapshot
        fold_a = Layer2FoldAttribution(
            fold_idx=0, oos_bars=2, n_rebal=2, realized_total=0.0, realized_price=0.0,
            realized_funding=0.0, realized_cost=0.0, expected_net=0.0, alpha_gap=0.0,
            mean_gross_exp=0.0, mean_net_exp=0.0, sleeves_active_mean=0.0,
            friction_pass_ratio=0.0, throttle_mult_mean=1.0, dropped_below_cost=0,
            netting_events=0,
            major_symbol_snapshots=(
                S(t=0, symbol="BTCUSDT", raw_mu=1.0, weight=0.0, regime_code=0, regime_risk_mult=1.0),
                S(t=1, symbol="BTCUSDT", raw_mu=2.0, weight=0.0, regime_code=0, regime_risk_mult=1.0),
            ),
        )
        fold_b = Layer2FoldAttribution(
            fold_idx=1, oos_bars=2, n_rebal=2, realized_total=0.0, realized_price=0.0,
            realized_funding=0.0, realized_cost=0.0, expected_net=0.0, alpha_gap=0.0,
            mean_gross_exp=0.0, mean_net_exp=0.0, sleeves_active_mean=0.0,
            friction_pass_ratio=0.0, throttle_mult_mean=1.0, dropped_below_cost=0,
            netting_events=0,
            major_symbol_snapshots=(
                S(t=0, symbol="BTCUSDT", raw_mu=3.0, weight=0.10, regime_code=0, regime_risk_mult=1.0),
                S(t=1, symbol="BTCUSDT", raw_mu=4.0, weight=0.10, regime_code=0, regime_risk_mult=1.0),
                S(t=0, symbol="ETHUSDT", raw_mu=0.0, weight=0.0, regime_code=0, regime_risk_mult=1.0),
                S(t=1, symbol="ETHUSDT", raw_mu=0.0, weight=0.0, regime_code=0, regime_risk_mult=1.0),
            ),
        )
        result = summarize_major_symbol_signal_sizing((fold_a, fold_b))
        result_dict = {r.symbol: r for r in result}
        assert result_dict["BTCUSDT"].n_obs == 4
        assert result_dict["BTCUSDT"].weight_long_pct == pytest.approx(0.5)
        assert result_dict["ETHUSDT"].n_obs == 2
        assert result_dict["ETHUSDT"].weight_long_pct == 0.0
        assert result_dict["ETHUSDT"].regime_cap_engaged_pct == 0.0
        assert result_dict["ETHUSDT"].mean_regime_risk_mult_when_long == 0.0

    def test_summarize_major_symbol_signal_sizing_returns_empty_tuple_for_no_folds(self) -> None:
        """Scenario 2b: 빈 fold_attributions → () ."""
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            summarize_major_symbol_signal_sizing,
        )
        result = summarize_major_symbol_signal_sizing(())
        assert result == ()
