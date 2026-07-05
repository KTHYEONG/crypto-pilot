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


class TestSummarizeMajorSymbolSleeveContribution:
    """summarize_major_symbol_sleeve_contribution computes sleeve-level sign-mismatch ratios."""

    def test_computes_sign_mismatch_and_adverse_conditional_ratio(self) -> None:
        """Scenario 1: BTCUSDT 4개 스냅샷(2x trend_ma 반대부호 vs pooled, regime adverse 2건) -> 정확한 비율."""
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            MajorSymbolSleeveContributionSnapshot,
            summarize_major_symbol_sleeve_contribution,
        )
        SS = MajorSymbolSleeveContributionSnapshot
        snapshots = (
            SS(t=0, symbol="BTCUSDT", strategy_id="trend_ma:ema_12_72_4h",
               raw_mu_sleeve=5.0, quality_weight_sleeve=0.4, pooled_mu_symbol=3.0, regime_code=0),
            SS(t=1, symbol="BTCUSDT", strategy_id="trend_ma:ema_12_72_4h",
               raw_mu_sleeve=-2.0, quality_weight_sleeve=0.3, pooled_mu_symbol=1.0, regime_code=1),
            SS(t=2, symbol="BTCUSDT", strategy_id="trend_ma:ema_12_72_4h",
               raw_mu_sleeve=-1.0, quality_weight_sleeve=0.5, pooled_mu_symbol=0.5, regime_code=1),
            SS(t=3, symbol="BTCUSDT", strategy_id="trend_ma:ema_12_72_4h",
               raw_mu_sleeve=4.0, quality_weight_sleeve=0.2, pooled_mu_symbol=-1.0, regime_code=2),
        )
        fold_attributions = (
            Layer2FoldAttribution(
                fold_idx=0, oos_bars=4, n_rebal=4, realized_total=0.0, realized_price=0.0,
                realized_funding=0.0, realized_cost=0.0, expected_net=0.0, alpha_gap=0.0,
                mean_gross_exp=0.0, mean_net_exp=0.0, sleeves_active_mean=0.0,
                friction_pass_ratio=0.0, throttle_mult_mean=1.0, dropped_below_cost=0,
                netting_events=0, major_symbol_sleeve_snapshots=snapshots,
            ),
        )
        result = summarize_major_symbol_sleeve_contribution(fold_attributions)
        assert len(result) == 1
        r = result[0]
        assert r.symbol == "BTCUSDT"
        assert r.family == "trend_ma"
        assert r.n_obs == 4
        assert r.mean_raw_mu_sleeve == pytest.approx(1.5)
        assert r.mean_quality_weight_sleeve == pytest.approx(0.35)

        # Mismatch: raw_mu_sleeve와 pooled_mu_symbol이 엄격히 반대 부호
        # t=0: +5 vs +3 → same
        # t=1: -2 vs +1 → opposite → mismatch
        # t=2: -1 vs +0.5 → opposite → mismatch
        # t=3: +4 vs -1 → opposite → mismatch
        # n_valid = 4 (all non-zero), mismatches = 3
        # sign_mismatch_pct = 3/4 = 0.75
        assert r.sign_mismatch_pct == pytest.approx(0.75)

        # Adverse conditions (regime_code in {1,2}): t=1,2,3
        # Adverse + valid: t=1(mismatch), t=2(mismatch), t=3(mismatch) → 3/3 = 1.0
        assert r.regime_adverse_sign_mismatch_pct == pytest.approx(1.0)

    def test_awf_simulation_collects_sleeve_snapshots_for_major_symbol(self) -> None:
        """Scenario 1(통합): _run_awf_simulation에서 major 심볼만 스냅샷 수집."""
        from unittest.mock import MagicMock, patch

        import numpy as np

        with (
            patch("src.domain.futures.strategy.market_regime.compress_regime_codes") as mock_compress,
            patch("src.domain.futures.strategy.market_regime.compute_market_regime_context") as mock_regime,
        ):
            mock_regime.return_value = MagicMock(code_1d=np.zeros(50, dtype=np.int8))
            mock_compress.side_effect = lambda x: x

            aligned = MagicMock()
            aligned.symbols = ("BTCUSDT", "ETHUSDT", "XRPUSDT")
            aligned.close_2d = np.ones((50, 3), dtype=np.float64) * 100.0
            aligned.datetimes = np.array(
                [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(50)]
            )
            aligned.beta_vs_market_1d = np.zeros(3, dtype=np.float64)
            aligned.execution_cost_bps_2d = np.full((50, 3), 3.8, dtype=np.float64)
            aligned.funding_2d = np.zeros((50, 3), dtype=np.float64)
            aligned.active_mask = np.ones((50, 3), dtype=np.bool_)
            aligned.warm_mask = np.ones((50, 3), dtype=np.bool_)
            aligned.execution_eligibility_mask = np.ones((50, 3), dtype=np.bool_)
            aligned.strategy_readiness_mask = np.ones((50, 3), dtype=np.bool_)
            aligned.promotion_active_mask = np.ones((50, 3), dtype=np.bool_)
            aligned.entry_block_mask = np.zeros((50, 3), dtype=np.bool_)
            aligned.kill_mask = np.zeros((50, 3), dtype=np.bool_)

            from src.domain.futures.strategy.candidate_contracts import (
                ValidatedSignalBatch,
                ValidatedSignalEvent,
            )
            events = [
                ValidatedSignalEvent(
                    decision_idx=5, decision_time=aligned.datetimes[5],
                    symbol="BTCUSDT", strategy_id="trend_ma:ema_12_72_4h",
                    activation_context="all", side=1,
                    expected_net_bps=10.0, expected_gross_bps=15.0,
                    q10_net_bps=5.0, q10_gross_bps=8.0,
                    q90_net_bps=0.0, q90_gross_bps=20.0,
                    expected_holding_bars=3, reliability=0.9,
                    registry_version="test", model_version="test",
                ),
                ValidatedSignalEvent(
                    decision_idx=5, decision_time=aligned.datetimes[5],
                    symbol="BTCUSDT", strategy_id="mtf_breakout_retest:mtf_bor_20_4h",
                    activation_context="all", side=1,
                    expected_net_bps=8.0, expected_gross_bps=12.0,
                    q10_net_bps=4.0, q10_gross_bps=6.0,
                    q90_net_bps=0.0, q90_gross_bps=16.0,
                    expected_holding_bars=3, reliability=0.9,
                    registry_version="test", model_version="test",
                ),
                ValidatedSignalEvent(
                    decision_idx=5, decision_time=aligned.datetimes[5],
                    symbol="XRPUSDT", strategy_id="trend_ma:ema_12_72_4h",
                    activation_context="all", side=1,
                    expected_net_bps=8.0, expected_gross_bps=12.0,
                    q10_net_bps=4.0, q10_gross_bps=6.0,
                    q90_net_bps=0.0, q90_gross_bps=16.0,
                    expected_holding_bars=3, reliability=0.9,
                    registry_version="test", model_version="test",
                ),
            ]
            signal_batch = ValidatedSignalBatch(
                events=tuple(events), start_idx=0, end_idx=50,
                symbols=aligned.symbols, registry_version="test", model_version="test",
            )
            from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
            from src.domain.futures.strategy.tiered_workflow.awf_sim import (
                _run_awf_simulation,
                build_l2_simulation_cache,
            )
            from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig
            from src.domain.futures.strategy.walk_forward import WFFold

            cache = build_l2_simulation_cache(aligned, signal_batch, "4h")
            awf_folds = (WFFold(fit_start=0, fit_end=5, cal_start=5, cal_end=8, oos_start=8, oos_end=30),)
            config = Layer2AllocationConfig(k_rank=3, rebalance_bars=3)
            caps = PortfolioCaps(gross=3.0, per_symbol=0.15, net=0.5, beta=1.0, target_ann_vol=0.20)

            sim = _run_awf_simulation(
                cache=cache, signal_batch=signal_batch, aligned=aligned,
                awf_folds=awf_folds, config=config, caps=caps, sim_origin="test_sleeve_diag",
            )
            snaps = sim.fold_attributions[0].major_symbol_sleeve_snapshots
            assert len(snaps) > 0
            assert all(s.symbol == "BTCUSDT" for s in snaps)

    def test_neutral_raw_mu_sleeve_excluded_from_mismatch_ratio(self) -> None:
        """Scenario 2: raw_mu_sleeve≈0(1e-12 dead-zone) 스냅샷은 mismatch 분모/분자에서 제외."""
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            MajorSymbolSleeveContributionSnapshot,
            summarize_major_symbol_sleeve_contribution,
        )
        SS = MajorSymbolSleeveContributionSnapshot
        snapshots = (
            SS(t=0, symbol="BTCUSDT", strategy_id="trend_ma:ema_12_72_4h",
               raw_mu_sleeve=1e-13, quality_weight_sleeve=0.5, pooled_mu_symbol=3.0, regime_code=0),
            SS(t=1, symbol="BTCUSDT", strategy_id="trend_ma:ema_12_72_4h",
               raw_mu_sleeve=2.0, quality_weight_sleeve=0.3, pooled_mu_symbol=-1.0, regime_code=0),
        )
        fold_attributions = (
            Layer2FoldAttribution(
                fold_idx=0, oos_bars=2, n_rebal=2, realized_total=0.0, realized_price=0.0,
                realized_funding=0.0, realized_cost=0.0, expected_net=0.0, alpha_gap=0.0,
                mean_gross_exp=0.0, mean_net_exp=0.0, sleeves_active_mean=0.0,
                friction_pass_ratio=0.0, throttle_mult_mean=1.0, dropped_below_cost=0,
                netting_events=0, major_symbol_sleeve_snapshots=snapshots,
            ),
        )
        result = summarize_major_symbol_sleeve_contribution(fold_attributions)
        assert len(result) == 1
        r = result[0]
        assert r.n_obs == 2
        # Only t=1 is non-zero for both sleeve and pooled → 1 valid observation
        # t=1: +2 vs -1 → opposite → mismatch, n_valid=1
        # sign_mismatch_pct = 1/1 = 1.0
        assert r.sign_mismatch_pct == pytest.approx(1.0)

    def test_merges_multiple_folds_per_symbol_family(self) -> None:
        """Scenario 2: 2개 fold에 걸친 동일 (symbol,family) 스냅샷 병합 후 n_obs 합산."""
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            MajorSymbolSleeveContributionSnapshot,
            summarize_major_symbol_sleeve_contribution,
        )
        SS = MajorSymbolSleeveContributionSnapshot
        fold_a = Layer2FoldAttribution(
            fold_idx=0, oos_bars=2, n_rebal=2, realized_total=0.0, realized_price=0.0,
            realized_funding=0.0, realized_cost=0.0, expected_net=0.0, alpha_gap=0.0,
            mean_gross_exp=0.0, mean_net_exp=0.0, sleeves_active_mean=0.0,
            friction_pass_ratio=0.0, throttle_mult_mean=1.0, dropped_below_cost=0,
            netting_events=0,
            major_symbol_sleeve_snapshots=(
                SS(t=0, symbol="BTCUSDT", strategy_id="trend_ma:ema_12_72_4h",
                   raw_mu_sleeve=1.0, quality_weight_sleeve=0.5, pooled_mu_symbol=0.5, regime_code=0),
            ),
        )
        fold_b = Layer2FoldAttribution(
            fold_idx=1, oos_bars=2, n_rebal=2, realized_total=0.0, realized_price=0.0,
            realized_funding=0.0, realized_cost=0.0, expected_net=0.0, alpha_gap=0.0,
            mean_gross_exp=0.0, mean_net_exp=0.0, sleeves_active_mean=0.0,
            friction_pass_ratio=0.0, throttle_mult_mean=1.0, dropped_below_cost=0,
            netting_events=0,
            major_symbol_sleeve_snapshots=(
                SS(t=0, symbol="BTCUSDT", strategy_id="trend_ma:ema_12_72_4h",
                   raw_mu_sleeve=2.0, quality_weight_sleeve=0.3, pooled_mu_symbol=-1.0, regime_code=1),
            ),
        )
        result = summarize_major_symbol_sleeve_contribution((fold_a, fold_b))
        assert len(result) == 1
        assert result[0].n_obs == 2
        assert result[0].mean_raw_mu_sleeve == pytest.approx(1.5)

    def test_returns_empty_tuple_for_no_folds(self) -> None:
        """Scenario 2: 빈 fold_attributions → ()."""
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            summarize_major_symbol_sleeve_contribution,
        )
        result = summarize_major_symbol_sleeve_contribution(())
        assert result == ()

    def test_non_major_symbol_never_collected(self) -> None:
        """Scenario 2: 비-major 심볼 스냅샷은 수집되지 않음."""
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            MajorSymbolSleeveContributionSnapshot,
            summarize_major_symbol_sleeve_contribution,
        )
        SS = MajorSymbolSleeveContributionSnapshot
        snapshots = (
            SS(t=0, symbol="XRPUSDT", strategy_id="trend_ma:ema_12_72_4h",
               raw_mu_sleeve=1.0, quality_weight_sleeve=0.5, pooled_mu_symbol=0.5, regime_code=0),
        )
        fold_attributions = (
            Layer2FoldAttribution(
                fold_idx=0, oos_bars=1, n_rebal=1, realized_total=0.0, realized_price=0.0,
                realized_funding=0.0, realized_cost=0.0, expected_net=0.0, alpha_gap=0.0,
                mean_gross_exp=0.0, mean_net_exp=0.0, sleeves_active_mean=0.0,
                friction_pass_ratio=0.0, throttle_mult_mean=1.0, dropped_below_cost=0,
                netting_events=0, major_symbol_sleeve_snapshots=snapshots,
            ),
        )
        result = summarize_major_symbol_sleeve_contribution(fold_attributions)
        assert result == ()

    def test_sleeve_missing_from_pooled_symbol_result_skips_gracefully(self) -> None:
        """Scenario 3: pooling 후 valid_signals에서 빠진 심볼은 예외 없이 스킵."""
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            MajorSymbolSleeveContributionSnapshot,
            summarize_major_symbol_sleeve_contribution,
        )
        SS = MajorSymbolSleeveContributionSnapshot
        snapshots = (
            SS(t=0, symbol="BTCUSDT", strategy_id="trend_ma:ema_12_72_4h",
               raw_mu_sleeve=1.0, quality_weight_sleeve=0.5, pooled_mu_symbol=0.0, regime_code=0),
        )
        fold_attributions = (
            Layer2FoldAttribution(
                fold_idx=0, oos_bars=1, n_rebal=1, realized_total=0.0, realized_price=0.0,
                realized_funding=0.0, realized_cost=0.0, expected_net=0.0, alpha_gap=0.0,
                mean_gross_exp=0.0, mean_net_exp=0.0, sleeves_active_mean=0.0,
                friction_pass_ratio=0.0, throttle_mult_mean=1.0, dropped_below_cost=0,
                netting_events=0, major_symbol_sleeve_snapshots=snapshots,
            ),
        )
        result = summarize_major_symbol_sleeve_contribution(fold_attributions)
        assert len(result) == 1
        assert result[0].n_obs == 1
        # pooled_mu=0.0 is inside dead-zone → not counted as mismatch
        assert result[0].sign_mismatch_pct == 0.0

    def test_zero_n_obs_family_bucket_returns_no_entry_not_zero_division(self) -> None:
        """Scenario 3: 특정 (symbol,family) 관측치 0이면 결과 튜플에 해당 항목 없음."""
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            summarize_major_symbol_sleeve_contribution,
        )
        result = summarize_major_symbol_sleeve_contribution(
            (
                Layer2FoldAttribution(
                    fold_idx=0, oos_bars=1, n_rebal=1, realized_total=0.0, realized_price=0.0,
                    realized_funding=0.0, realized_cost=0.0, expected_net=0.0, alpha_gap=0.0,
                    mean_gross_exp=0.0, mean_net_exp=0.0, sleeves_active_mean=0.0,
                    friction_pass_ratio=0.0, throttle_mult_mean=1.0, dropped_below_cost=0,
                    netting_events=0,
                    major_symbol_sleeve_snapshots=(),
                    major_symbol_snapshots=(),
                ),
            ),
        )
        assert result == ()


class TestComputeMajorSymbolRegistryCensus:
    """Track 2 — ETH Registry Census 진단."""

    def _make_registry(
        self,
        by_symbol: dict[str, list[tuple[str, float, bool]]],
    ) -> object:
        """Helper to create a mock QualifiedSignalRegistry with minimal fields."""
        from src.domain.futures.signals.contracts import (
            QualifiedSignalRegistry,
            SignalSourceKey,
            SymbolStrategyEvidence,
        )
        registry_data: dict[str, tuple[SymbolStrategyEvidence, ...]] = {}
        for sym, entries in by_symbol.items():
            evs: list[SymbolStrategyEvidence] = []
            for strat, mean_bps, hard_eligible in entries:
                evs.append(SymbolStrategyEvidence(
                    key=SignalSourceKey(symbol=sym, strategy_id=strat, activation_context="all"),
                    mean_gross_bps=0.0,
                    mean_incremental_bps=mean_bps,
                    block_tstat_incremental=0.0,
                    probability_positive=0.0,
                    p_value=1.0,
                    q_value=1.0,
                    positive_fold_ratio=0.0,
                    n_obs=0,
                    effective_n=0.0,
                    n_folds=0,
                    quality_weight=1.0,
                    hard_eligible=hard_eligible,
                ))
            registry_data[sym] = tuple(evs)
        return QualifiedSignalRegistry(
            by_symbol=registry_data,
            ready_symbols=tuple(registry_data.keys()),
            trade_scope_count=len(registry_data),
            registry_version="test",
        )

    def _make_summary(self, symbol: str, family: str, active: bool) -> object:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            MajorSymbolSleeveContributionSummary,
        )
        return MajorSymbolSleeveContributionSummary(
            symbol=symbol, family=family,
            n_obs=10, mean_raw_mu_sleeve=0.5, mean_quality_weight_sleeve=1.0,
            sign_mismatch_pct=0.0, regime_adverse_sign_mismatch_pct=0.0,
        )

    def test_registry_census_flags_activation_gap_for_eth(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            compute_major_symbol_registry_census,
        )
        registry = self._make_registry({
            "ETHUSDT": [("dual_momentum", -2.0, True), ("trend_ma", 3.0, True)],
        })
        # dual_momentum has negative mean_incremental_bps but IS observed in holdout
        # trend_ma has positive mean_incremental_bps but NOT observed
        summaries = (
            self._make_summary("ETHUSDT", "dual_momentum", active=True),
        )
        result = compute_major_symbol_registry_census(
            registry=registry,
            observed_sleeve_summaries=summaries,
            symbols=("ETHUSDT", "BTCUSDT"),
        )
        entries = {e.family: e for e in result if e.symbol == "ETHUSDT"}
        # dual_momentum is observed → observed_active_in_holdout=True
        assert entries["dual_momentum"].observed_active_in_holdout
        assert entries["dual_momentum"].registry_mean_incremental_bps == pytest.approx(-2.0)
        assert entries["dual_momentum"].hard_eligible
        # trend_ma NOT observed → observed_active_in_holdout=False
        assert not entries["trend_ma"].observed_active_in_holdout

    def test_registry_census_admission_gap_when_no_bearish_family_in_registry(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            compute_major_symbol_registry_census,
        )
        registry = self._make_registry({
            "ETHUSDT": [("trend_ma", 3.0, True)],
        })
        result = compute_major_symbol_registry_census(
            registry=registry,
            observed_sleeve_summaries=(),
            symbols=("ETHUSDT",),
        )
        assert len(result) == 1
        assert result[0].family == "trend_ma"
        assert result[0].registry_mean_incremental_bps >= 0.0

    def test_registry_census_empty_registry_returns_empty_tuple(self) -> None:
        from src.domain.futures.signals.contracts import QualifiedSignalRegistry
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            compute_major_symbol_registry_census,
        )
        registry = QualifiedSignalRegistry(by_symbol={}, ready_symbols=(), trade_scope_count=0, registry_version="test")
        result = compute_major_symbol_registry_census(
            registry=registry,
            observed_sleeve_summaries=(),
            symbols=("ETHUSDT",),
        )
        assert result == ()
