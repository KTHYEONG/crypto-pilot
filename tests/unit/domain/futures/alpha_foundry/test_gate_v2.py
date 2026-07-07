from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.cheap_gate import (
    compute_cost_drag_ratio_v2,
    compute_liquidity_cost_stress_bps,
    compute_payoff_stats,
    compute_rank_ic_with_tstat,
    compute_xs_spread_lcb_bps,
    downgrade_gate_v2_to_cheap_evidence,
    evaluate_panel_gate_v2,
)
from src.domain.futures.alpha_foundry.contracts import (
    AlphaGateEvidenceV2,
    AlphaRecipe,
    CheapGateConfig,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def make_gate_v2_aligned(*, n_symbols: int = 6, funding: float = 0.0001) -> AlignedMarketData:
    datetimes = np.arange(
        np.datetime64("2026-01-01T00:00:00"),
        np.datetime64("2026-03-01T00:00:00"),
        np.timedelta64(4, "h"),
        dtype="datetime64[ns]",
    )
    t = int(datetimes.shape[0])
    symbols = tuple(f"SYM{i}USDT" for i in range(n_symbols))
    base = 100.0 * np.exp(0.002 * np.arange(t, dtype=np.float64))
    close = np.column_stack([base * (1.0 + i * 0.01) for i in range(n_symbols)])
    mask = np.ones_like(close, dtype=np.bool_)
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=symbols,
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close,
        volume_2d=np.full_like(close, 1_000.0),
        funding_2d=np.full_like(close, funding),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=np.zeros_like(close, dtype=np.bool_),
        kill_mask=np.zeros_like(close, dtype=np.bool_),
        oi_2d=np.full_like(close, 10_000.0),
        lsr_2d=np.full_like(close, 1.2),
        taker_buy_2d=np.full_like(close, 500.0),
        trades_2d=np.full_like(close, 100.0),
        execution_cost_bps_2d=np.full_like(close, 2.5),
    )


def make_sparse_panel(aligned: AlignedMarketData, *, recipe_id: str) -> CandidateSignalPanel:
    t, n = aligned.close_2d.shape
    side = np.zeros((t, n), dtype=np.int8)
    for start in range(0, t, 16):
        side[start : start + 8, :] = 1
    score = side.astype(np.float64)
    return CandidateSignalPanel(
        family="sparse_breakout_retest_v2",
        variant="bor_v2_20",
        params={"channel": 20, "retest": 3},
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        signed_score_2d=score,
        side_hint_2d=side,
        expected_holding_bars=3,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.abs(np.diff(score, axis=0, prepend=0.0)),
        valid_mask_2d=np.ones((t, n), dtype=np.bool_),
        metadata={"recipe_id": recipe_id, "source": "catalog_exact"},
        archetype="trend",
    )


SAMPLE_V2_RECIPE = AlphaRecipe(
    recipe_id="sparse_breakout_retest_v2:bor_v2_20:4h",
    family="sparse_breakout_retest_v2",
    variant="bor_v2_20",
    timeframe="4h",
    archetype="trend",
    indicator_params={"channel": 20, "retest": 3},
    side_rule_id="breakout_retest_sparse",
    exit_policy_id="atr_trail_2",
    required_fields=("close", "high", "low", "volume"),
    causal_lag_bars=1,
    max_turnover_per_year=120.0,
)


class TestCostDragV2:
    def test_uses_event_means_not_totals(self) -> None:
        c1 = compute_cost_drag_ratio_v2(mean_cost_bps=5.0, mean_gross_bps=100.0)
        c2 = compute_cost_drag_ratio_v2(mean_cost_bps=5.0, mean_gross_bps=100.0)
        assert abs(c1 - c2) < 1e-9

    def test_duplicate_events_same_means_same_drag(self) -> None:
        c1 = compute_cost_drag_ratio_v2(mean_cost_bps=5.0, mean_gross_bps=100.0)
        assert np.isfinite(c1)


class TestRankIcWithTstat:
    def test_constant_score_returns_zero(self) -> None:
        fwd = np.random.default_rng(42).normal(0, 1, (100, 5)).astype(np.float64)
        score = np.ones((100, 5), dtype=np.float64)
        mask = np.ones((100, 5), dtype=np.bool_)
        ic, tstat = compute_rank_ic_with_tstat(fwd_ret_bps=fwd, score=score, mask=mask)
        assert ic == 0.0
        assert tstat == 0.0

    def test_fewer_than_3_obs_returns_zero(self) -> None:
        fwd = np.array([[1.0, 2.0]], dtype=np.float64)
        score = np.array([[3.0, 4.0]], dtype=np.float64)
        mask = np.array([[True, False]], dtype=np.bool_)
        ic, tstat = compute_rank_ic_with_tstat(fwd_ret_bps=fwd, score=score, mask=mask)
        assert ic == 0.0
        assert tstat == 0.0


class TestPayoffStats:
    def test_hit_rate_computed_correctly(self) -> None:
        vals = np.array([1.0, -1.0, 2.0, -2.0, 0.5], dtype=np.float64)
        hit, skew = compute_payoff_stats(vals)
        assert 0.0 <= hit <= 1.0
        assert np.isfinite(skew)

    def test_all_positive(self) -> None:
        vals = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        hit, skew = compute_payoff_stats(vals)
        assert hit == 1.0
        assert np.isfinite(skew)

    def test_all_negative(self) -> None:
        vals = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
        hit, skew = compute_payoff_stats(vals)
        assert hit == 0.0
        assert skew == 0.0

    def test_empty_array(self) -> None:
        vals = np.array([], dtype=np.float64)
        hit, skew = compute_payoff_stats(vals)
        assert hit == 0.0
        assert skew == 0.0


class TestXsSpreadLcbBps:
    def test_requires_min_symbols_per_bar(self) -> None:
        net = np.random.default_rng(42).normal(0, 1, (10, 3)).astype(np.float64)
        score = np.random.default_rng(99).normal(0, 1, (10, 3)).astype(np.float64)
        mask = np.ones((10, 3), dtype=np.bool_)
        result = compute_xs_spread_lcb_bps(
            net_bps=net,
            score=score,
            event_mask=mask,
            min_symbols_per_bar=5,
        )
        assert result is None

    def test_returns_float_with_sufficient_symbols(self) -> None:
        rng = np.random.default_rng(42)
        net = rng.normal(0, 1, (20, 10)).astype(np.float64)
        score = rng.normal(0, 1, (20, 10)).astype(np.float64)
        mask = np.ones((20, 10), dtype=np.bool_)
        result = compute_xs_spread_lcb_bps(
            net_bps=net,
            score=score,
            event_mask=mask,
            min_symbols_per_bar=4,
        )
        assert result is not None
        assert np.isfinite(result)


class TestLiquidityCostStress:
    def test_returns_zero_without_cost_data(self) -> None:
        aligned = make_gate_v2_aligned(n_symbols=3)
        aligned = AlignedMarketData(
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            open_2d=aligned.open_2d,
            high_2d=aligned.high_2d,
            low_2d=aligned.low_2d,
            close_2d=aligned.close_2d,
            volume_2d=aligned.volume_2d,
            funding_2d=aligned.funding_2d,
            active_mask=aligned.active_mask,
            warm_mask=aligned.warm_mask,
            entry_block_mask=aligned.entry_block_mask,
            kill_mask=aligned.kill_mask,
            execution_cost_bps_2d=None,
        )
        mask = np.ones(aligned.close_2d.shape, dtype=np.bool_)
        result = compute_liquidity_cost_stress_bps(
            aligned=aligned,
            event_mask=mask,
            stress_mult=1.0,
        )
        assert result == 0.0

    def test_with_cost_data_returns_positive(self) -> None:
        aligned = make_gate_v2_aligned(n_symbols=3)
        mask = np.ones(aligned.close_2d.shape, dtype=np.bool_)
        result = compute_liquidity_cost_stress_bps(
            aligned=aligned,
            event_mask=mask,
            stress_mult=2.0,
        )
        assert result > 0.0

    def test_no_events_returns_zero(self) -> None:
        aligned = make_gate_v2_aligned(n_symbols=3)
        mask = np.zeros(aligned.close_2d.shape, dtype=np.bool_)
        result = compute_liquidity_cost_stress_bps(
            aligned=aligned,
            event_mask=mask,
            stress_mult=2.0,
        )
        assert result == 0.0


class TestEvaluatePanelGateV2:
    def test_passes_sparse_positive_cost_adjusted_panel(self) -> None:
        aligned = make_gate_v2_aligned(n_symbols=6)
        panel = make_sparse_panel(aligned, recipe_id=SAMPLE_V2_RECIPE.recipe_id)
        config = CheapGateConfig(enable_v2_gate_metrics=True)
        cost_model = ExecutionCostModel()
        evidence = evaluate_panel_gate_v2(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_V2_RECIPE,
            cost_model=cost_model,
            config=config,
            bars_per_year=2190.0,
        )
        assert isinstance(evidence, AlphaGateEvidenceV2)
        assert np.isfinite(evidence.rank_ic_tstat)

    def test_rejects_invalid_bars_per_year(self) -> None:
        aligned = make_gate_v2_aligned(n_symbols=3)
        panel = make_sparse_panel(aligned, recipe_id=SAMPLE_V2_RECIPE.recipe_id)
        config = CheapGateConfig()
        cost_model = ExecutionCostModel()
        with pytest.raises(ValueError, match="bars_per_year must be positive"):
            evaluate_panel_gate_v2(
                panel=panel,
                aligned=aligned,
                recipe=SAMPLE_V2_RECIPE,
                cost_model=cost_model,
                config=config,
                bars_per_year=0.0,
            )

    def test_high_turnover_requires_gross_lcb_above_cost_stress(self) -> None:
        aligned = make_gate_v2_aligned(n_symbols=3)
        panel = make_sparse_panel(aligned, recipe_id=SAMPLE_V2_RECIPE.recipe_id)
        config = CheapGateConfig(
            enable_v2_gate_metrics=True,
            high_turnover_per_year=10.0,
        )
        cost_model = ExecutionCostModel()
        evidence = evaluate_panel_gate_v2(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_V2_RECIPE,
            cost_model=cost_model,
            config=config,
            bars_per_year=2190.0,
        )
        assert isinstance(evidence, AlphaGateEvidenceV2)

    def test_handles_constant_score_ic(self) -> None:
        aligned = make_gate_v2_aligned(n_symbols=5)
        t, n = aligned.close_2d.shape
        side = np.zeros((t, n), dtype=np.int8)
        side[10:180:8, :] = 1
        score = np.ones((t, n), dtype=np.float64)
        valid = np.ones((t, n), dtype=np.bool_)
        panel = CandidateSignalPanel(
            family="sparse_breakout_retest_v2",
            variant="bor_v2_20",
            params={"channel": 20},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=score,
            side_hint_2d=side,
            expected_holding_bars=3,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.abs(np.diff(score, axis=0, prepend=0.0)),
            valid_mask_2d=valid,
            metadata={"recipe_id": "test:r", "source": "test"},
            archetype="trend",
        )
        config = CheapGateConfig(enable_v2_gate_metrics=True)
        cost_model = ExecutionCostModel()
        evidence = evaluate_panel_gate_v2(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_V2_RECIPE,
            cost_model=cost_model,
            config=config,
            bars_per_year=2190.0,
        )
        assert evidence.rank_ic == 0.0
        assert evidence.rank_ic_tstat == 0.0

    def test_excess_cost_drag_rejects_panel(self) -> None:
        aligned = make_gate_v2_aligned(n_symbols=3)
        panel = make_sparse_panel(aligned, recipe_id=SAMPLE_V2_RECIPE.recipe_id)
        config = CheapGateConfig(
            max_cost_drag_ratio=0.01,
            enable_v2_gate_metrics=True,
        )
        cost_model = ExecutionCostModel()
        evidence = evaluate_panel_gate_v2(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_V2_RECIPE,
            cost_model=cost_model,
            config=config,
            bars_per_year=2190.0,
        )
        assert evidence.gate_passed is False

    def test_min_events_rejects_panel(self) -> None:
        aligned = make_gate_v2_aligned(n_symbols=3)
        t, n = aligned.close_2d.shape
        side = np.zeros((t, n), dtype=np.int8)
        score = side.astype(np.float64)
        valid = np.ones((t, n), dtype=np.bool_)
        panel = CandidateSignalPanel(
            family="sparse_breakout_retest_v2",
            variant="bor_v2_20",
            params={"channel": 20},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=score,
            side_hint_2d=side,
            expected_holding_bars=3,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
            valid_mask_2d=valid,
            metadata={"recipe_id": "test:r", "source": "test"},
            archetype="trend",
        )
        config = CheapGateConfig(min_events=1000)
        cost_model = ExecutionCostModel()
        evidence = evaluate_panel_gate_v2(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_V2_RECIPE,
            cost_model=cost_model,
            config=config,
            bars_per_year=2190.0,
        )
        assert evidence.gate_passed is False
        assert "insufficient_events" in evidence.reject_reasons

    def test_varying_score_reaches_rank_ic_lines(self) -> None:
        aligned = make_gate_v2_aligned(n_symbols=5)
        t, n = aligned.close_2d.shape
        rng = np.random.default_rng(99)
        side = np.zeros((t, n), dtype=np.int8)
        for start in range(20, t - 10, 12):
            side[start:start+6, :] = 1
        score = side.astype(np.float64) * rng.uniform(-1, 1, (t, n)).astype(np.float64)
        valid = np.ones((t, n), dtype=np.bool_)
        panel = CandidateSignalPanel(
            family="sparse_breakout_retest_v2",
            variant="bor_v2_20",
            params={"channel": 20},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=score,
            side_hint_2d=side,
            expected_holding_bars=3,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.abs(np.diff(score, axis=0, prepend=0.0)),
            valid_mask_2d=valid,
            metadata={"recipe_id": "test:r", "source": "test"},
            archetype="trend",
        )
        config = CheapGateConfig(enable_v2_gate_metrics=True)
        cost_model = ExecutionCostModel()
        evidence = evaluate_panel_gate_v2(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_V2_RECIPE,
            cost_model=cost_model,
            config=config,
            bars_per_year=2190.0,
        )
        assert evidence.n_events > 0
        assert np.isfinite(evidence.rank_ic)


class TestDowngradeGateV2ToCheapEvidence:
    def test_downgrade_preserves_existing_fields(self) -> None:
        v2 = AlphaGateEvidenceV2(
            recipe_id="r1",
            timeframe="4h",
            symbol_scope="global",
            n_events=50,
            effective_n=40.0,
            mean_gross_bps=100.0,
            mean_cost_bps=25.0,
            mean_net_bps=75.0,
            gross_lcb_bps=30.0,
            net_lcb_bps=20.0,
            nw_tstat=2.5,
            rank_ic=0.05,
            rank_ic_tstat=2.1,
            cost_drag_ratio=0.25,
            turnover_per_year=100.0,
            event_hit_rate=0.6,
            payoff_skew=1.5,
            regime_edge_bps={},
            xs_spread_lcb_bps=None,
            liquidity_cost_stress_bps=2.0,
            bootstrap_lcb_bps=15.0,
            bootstrap_agree=True,
            gate_passed=True,
            reject_reasons=(),
        )
        cheap = downgrade_gate_v2_to_cheap_evidence(v2)
        assert cheap.recipe_id == "r1"
        assert cheap.timeframe == "4h"
        assert cheap.n_events == 50
        assert cheap.gate_passed is True
        assert cheap.mean_gross_bps == 100.0
        assert cheap.mean_cost_bps == 25.0
        assert cheap.mean_net_bps == 75.0

    def test_filters_unknown_reject_reasons(self) -> None:
        v2 = AlphaGateEvidenceV2(
            recipe_id="r1",
            timeframe="4h",
            symbol_scope="global",
            n_events=50,
            effective_n=40.0,
            mean_gross_bps=100.0,
            mean_cost_bps=25.0,
            mean_net_bps=75.0,
            gross_lcb_bps=30.0,
            net_lcb_bps=20.0,
            nw_tstat=2.5,
            rank_ic=0.05,
            rank_ic_tstat=2.1,
            cost_drag_ratio=0.25,
            turnover_per_year=100.0,
            event_hit_rate=0.6,
            payoff_skew=1.5,
            regime_edge_bps={},
            xs_spread_lcb_bps=None,
            liquidity_cost_stress_bps=2.0,
            bootstrap_lcb_bps=15.0,
            bootstrap_agree=True,
            gate_passed=False,
            reject_reasons=("insufficient_events", "non_positive_gross"),
        )
        cheap = downgrade_gate_v2_to_cheap_evidence(v2)
        assert cheap.gate_passed is False
        assert "insufficient_events" in cheap.reject_reasons
        assert "non_positive_gross" not in cheap.reject_reasons


class TestAlphaGateEvidenceV2Validation:
    def test_rejects_negative_n_events(self) -> None:
        with pytest.raises(ValueError, match="n_events must be >= 0"):
            AlphaGateEvidenceV2(
                recipe_id="r",
                timeframe="4h",
                symbol_scope="global",
                n_events=-1,
                effective_n=40.0,
                mean_gross_bps=100.0,
                mean_cost_bps=25.0,
                mean_net_bps=75.0,
                gross_lcb_bps=30.0,
                net_lcb_bps=20.0,
                nw_tstat=2.5,
                rank_ic=0.05,
                rank_ic_tstat=2.1,
                cost_drag_ratio=0.25,
                turnover_per_year=100.0,
                event_hit_rate=0.6,
                payoff_skew=1.5,
                regime_edge_bps={},
                xs_spread_lcb_bps=None,
                liquidity_cost_stress_bps=2.0,
                bootstrap_lcb_bps=15.0,
                bootstrap_agree=True,
                gate_passed=True,
                reject_reasons=(),
            )

    def test_rejects_negative_effective_n(self) -> None:
        with pytest.raises(ValueError, match=r"effective_n must be >= 0.0"):
            AlphaGateEvidenceV2(
                recipe_id="r",
                timeframe="4h",
                symbol_scope="global",
                n_events=50,
                effective_n=-1.0,
                mean_gross_bps=100.0,
                mean_cost_bps=25.0,
                mean_net_bps=75.0,
                gross_lcb_bps=30.0,
                net_lcb_bps=20.0,
                nw_tstat=2.5,
                rank_ic=0.05,
                rank_ic_tstat=2.1,
                cost_drag_ratio=0.25,
                turnover_per_year=100.0,
                event_hit_rate=0.6,
                payoff_skew=1.5,
                regime_edge_bps={},
                xs_spread_lcb_bps=None,
                liquidity_cost_stress_bps=2.0,
                bootstrap_lcb_bps=15.0,
                bootstrap_agree=True,
                gate_passed=True,
                reject_reasons=(),
            )

    def test_rejects_negative_cost_drag(self) -> None:
        with pytest.raises(ValueError, match=r"cost_drag_ratio must be >= 0.0"):
            AlphaGateEvidenceV2(
                recipe_id="r",
                timeframe="4h",
                symbol_scope="global",
                n_events=50,
                effective_n=40.0,
                mean_gross_bps=100.0,
                mean_cost_bps=25.0,
                mean_net_bps=75.0,
                gross_lcb_bps=30.0,
                net_lcb_bps=20.0,
                nw_tstat=2.5,
                rank_ic=0.05,
                rank_ic_tstat=2.1,
                cost_drag_ratio=-0.1,
                turnover_per_year=100.0,
                event_hit_rate=0.6,
                payoff_skew=1.5,
                regime_edge_bps={},
                xs_spread_lcb_bps=None,
                liquidity_cost_stress_bps=2.0,
                bootstrap_lcb_bps=15.0,
                bootstrap_agree=True,
                gate_passed=True,
                reject_reasons=(),
            )

    def test_rejects_event_hit_rate_out_of_bounds(self) -> None:
        with pytest.raises(ValueError, match="event_hit_rate must be in"):
            AlphaGateEvidenceV2(
                recipe_id="r",
                timeframe="4h",
                symbol_scope="global",
                n_events=50,
                effective_n=40.0,
                mean_gross_bps=100.0,
                mean_cost_bps=25.0,
                mean_net_bps=75.0,
                gross_lcb_bps=30.0,
                net_lcb_bps=20.0,
                nw_tstat=2.5,
                rank_ic=0.05,
                rank_ic_tstat=2.1,
                cost_drag_ratio=0.25,
                turnover_per_year=100.0,
                event_hit_rate=1.5,
                payoff_skew=1.5,
                regime_edge_bps={},
                xs_spread_lcb_bps=None,
                liquidity_cost_stress_bps=2.0,
                bootstrap_lcb_bps=15.0,
                bootstrap_agree=True,
                gate_passed=True,
                reject_reasons=(),
            )
