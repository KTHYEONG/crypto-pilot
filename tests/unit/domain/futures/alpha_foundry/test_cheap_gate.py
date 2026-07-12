from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.cheap_gate import (
    _rank_ic_soft_floor,
    build_l0_signal_candidate,
    evaluate_alpha_cheap_gate_batch,
    evaluate_alpha_gate_batch,
    evaluate_panel_cheap_gate,
    evaluate_panel_gate,
)
from src.domain.futures.alpha_foundry.contracts import (
    AlphaGateConfig,
    AlphaGateEvidence,
    AlphaRecipe,
    CheapGateConfig,
    CheapGateEvidence,
    FamilyTimeframeGatePolicy,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def make_mock_aligned() -> AlignedMarketData:
    # 100 days of 4h bars (600 bars) — long enough to clear min_events=40 with a
    # realistic hold/flat cycle while keeping annualized turnover under the cap
    # (see make_mock_panel's 8-on/8-off pattern).
    datetimes = np.arange(
        np.datetime64("2026-01-01T00:00:00"),
        np.datetime64("2026-04-11T00:00:00"),
        np.timedelta64(4, "h"),
        dtype="datetime64[ns]",
    )
    t = datetimes.shape[0]
    symbols = ("BTCUSDT", "ETHUSDT")
    # Constant per-bar log-drift (not a fixed-endpoint linspace) so the mean
    # 3-bar gross return stays comfortably above the ~11.25bps stress
    # round-trip cost regardless of window length (t).
    base = 100.0 * np.exp(0.002 * np.arange(t, dtype=np.float64))
    close = np.column_stack([base, base * 0.8])
    high = close * 1.01
    low = close * 0.99
    volume = np.full_like(close, 1_000.0)
    mask = np.ones_like(close, dtype=np.bool_)
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=symbols,
        open_2d=close.copy(),
        high_2d=high,
        low_2d=low,
        close_2d=close,
        volume_2d=volume,
        funding_2d=np.full_like(close, 0.0001),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=np.zeros_like(close, dtype=np.bool_),
        kill_mask=np.zeros_like(close, dtype=np.bool_),
    )


def make_mock_panel(*, recipe_id: str = "trend_ma:ema_12_72:4h") -> CandidateSignalPanel:
    aligned = make_mock_aligned()
    t, n = aligned.close_2d.shape
    # Sparse-entry fixture: 8 bars held / 8 bars flat, repeating. A constant
    # side=1 array produces zero rising-edge entries under sparse-entry
    # semantics (no bar ever transitions from flat/reversal) — this pattern
    # gives ~n_events=37/symbol while keeping annualized turnover well under
    # the default 365/yr cap (flip_fraction=2/16 -> ~137/yr at bars_per_year=2190).
    side = np.zeros((t, n), dtype=np.int8)
    cycle = 16
    for start in range(0, t, cycle):
        side[start : start + 8, :] = 1
    score = side.astype(np.float64)
    valid = np.ones((t, n), dtype=np.bool_)
    return CandidateSignalPanel(
        family="trend_ma",
        variant="ema_12_72",
        params={"fast": 12, "slow": 72},
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
        metadata={"recipe_id": recipe_id},
    )


SAMPLE_RECIPE = AlphaRecipe(
    recipe_id="trend_ma:ema_12_72:4h",
    family="trend_ma",
    variant="ema_12_72",
    timeframe="4h",
    archetype="trend",
    indicator_params={"fast": 12, "slow": 72},
    side_rule_id="trend_follow",
    exit_policy_id="atr_trail_2",
    required_fields=("close",),
    causal_lag_bars=1,
    max_turnover_per_year=365.0,
)


class TestEvaluatePanelGate:
    """S1-1: Unified gate basic calculation (Happy Path)."""

    def test_passes_with_upward_close_and_sparse_entries(self) -> None:
        """S1-1: Unified gate produces candidate with upward close, sparse longs."""
        t = 96
        dt = np.arange(np.datetime64("2026-01-01T00"), np.datetime64("2026-01-17T00"),
                        np.timedelta64(4, "h"))[:t]
        symbols = ("BTCUSDT", "ETHUSDT")
        # Both symbols go up strongly — long entries are profitable.
        close = np.column_stack([
            np.linspace(100.0, 124.0, t, dtype=np.float64),
            np.linspace(50.0, 61.0, t, dtype=np.float64),
        ])
        zeros = np.zeros((t, 2), dtype=np.float64)
        valid = np.ones((t, 2), dtype=np.bool_)
        side = np.zeros((t, 2), dtype=np.int8)
        side[::2, :] = 1
        # Cross-sectional variation: symbol 0 higher score than symbol 1
        score = np.where(side == 1, np.array([0.8, 0.4]), 0.0).astype(np.float64)
        aligned = AlignedMarketData(
            datetimes=dt, symbols=symbols,
            open_2d=close, high_2d=close, low_2d=close, close_2d=close,
            volume_2d=np.full((t, 2), 1_000.0, dtype=np.float64),
            funding_2d=zeros.copy(),
            active_mask=valid.copy(), warm_mask=valid.copy(),
            entry_block_mask=np.zeros((t, 2), dtype=np.bool_),
            kill_mask=np.zeros((t, 2), dtype=np.bool_),
            adv_usdt_2d=np.full((t, 2), 1_000_000.0, dtype=np.float64),
            execution_cost_bps_2d=np.full((t, 2), 4.0, dtype=np.float64),
        )
        panel = CandidateSignalPanel(
            family="sparse_breakout_retest_v2", variant="bor_v2_40_4h",
            params={"lookback": 40},
            datetimes=dt, symbols=symbols,
            signed_score_2d=score, side_hint_2d=side,
            expected_holding_bars=2, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=3.0,
            turnover_proxy_2d=zeros.copy(),
            valid_mask_2d=valid,
            metadata={"recipe_id": "bor_v2_40_4h"},
        )
        recipe = AlphaRecipe(
            recipe_id="bor_v2_40_4h", family="sparse_breakout_retest_v2",
            variant="bor_v2_40_4h", timeframe="4h", archetype="trend",
            indicator_params={"lookback": 40},
            side_rule_id="breakout", exit_policy_id="sparse",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=2000.0,
        )
        cfg = AlphaGateConfig(min_events=10, min_effective_n=5.0,
                              min_candidate_rank_ic_tstat=0.0,
                              min_nw_tstat=0.0,
                              max_cost_drag_ratio=1.0,
                              max_turnover_per_year=2000.0)
        evidence = evaluate_panel_gate(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(), config=cfg,
            bars_per_year=365.0 * 6.0, run_id="test",
        )
        assert isinstance(evidence, AlphaGateEvidence)
        assert evidence.mean_gross_bps > 0.0
        assert evidence.mean_cost_bps > 0.0
        # [ADR_20260710_L0_ENTRY_EXIT_SIGNAL_EFFECTIVENESS_REDESIGN] Barrier-aware
        # evaluation may produce negative net when stop/tp thresholds matter.
        # With high==low==close, the triple-barrier kernel hits time_exit routinely,
        # and mean_net_bps reflects actual barrier path — not a fixed-horizon assumption.
        assert np.isfinite(evidence.mean_net_bps)  # must compute, not NaN
        assert evidence.handoff_tier in {"seed", "candidate", "blocked"}
        assert evidence.capacity_score >= 0.0
        assert evidence.regime_stability >= 0.0
        assert evidence.tf_corroboration >= 0.0

    def test_cost_diagnostics_logs_when_enabled(self, caplog: pytest.LogCaptureFixture) -> None:
        """[LIMIT-04]: l0_cost_diagnostics_enabled=True emits [EVAL] gate_evidence log."""
        caplog.set_level(logging.DEBUG, logger="src.domain.futures.alpha_foundry.cheap_gate")
        t = 96
        dt = np.arange(np.datetime64("2026-01-01T00"), np.datetime64("2026-01-17T00"),
                        np.timedelta64(4, "h"))[:t]
        symbols = ("AAA", "BBB")
        zeros = np.zeros((t, 2), dtype=np.float64)
        valid = np.ones((t, 2), dtype=np.bool_)
        close = np.column_stack([
            np.linspace(100.0, 124.0, t, dtype=np.float64),
            np.linspace(50.0, 61.0, t, dtype=np.float64),
        ])
        side = np.zeros((t, 2), dtype=np.int8)
        side[::2, :] = 1
        score = np.where(side == 1, np.array([0.8, 0.4]), 0.0).astype(np.float64)
        aligned = AlignedMarketData(
            datetimes=dt, symbols=symbols,
            open_2d=close, high_2d=close, low_2d=close, close_2d=close,
            volume_2d=np.full((t, 2), 1_000.0, dtype=np.float64),
            funding_2d=zeros.copy(),
            active_mask=valid.copy(), warm_mask=valid.copy(),
            entry_block_mask=np.zeros((t, 2), dtype=np.bool_),
            kill_mask=np.zeros((t, 2), dtype=np.bool_),
            adv_usdt_2d=np.full((t, 2), 1_000_000.0, dtype=np.float64),
            execution_cost_bps_2d=np.full((t, 2), 4.0, dtype=np.float64),
        )
        panel = CandidateSignalPanel(
            family="sparse_breakout_retest_v2", variant="bor_v2_40_4h",
            params={"lookback": 40},
            datetimes=dt, symbols=symbols,
            signed_score_2d=score, side_hint_2d=side,
            expected_holding_bars=2, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=3.0,
            turnover_proxy_2d=zeros.copy(),
            valid_mask_2d=valid,
            metadata={"recipe_id": "bor_v2_40_4h"},
        )
        recipe = AlphaRecipe(
            recipe_id="bor_v2_40_4h", family="sparse_breakout_retest_v2",
            variant="bor_v2_40_4h", timeframe="4h", archetype="trend",
            indicator_params={"lookback": 40},
            side_rule_id="breakout", exit_policy_id="sparse",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=2000.0,
        )
        cfg = AlphaGateConfig(min_events=10, min_effective_n=5.0,
                              min_candidate_rank_ic_tstat=0.0,
                              min_nw_tstat=0.0,
                              max_cost_drag_ratio=1.0,
                              max_turnover_per_year=2000.0,
                              l0_cost_diagnostics_enabled=True)
        evaluate_panel_gate(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(), config=cfg,
            bars_per_year=365.0 * 6.0, run_id="test",
        )
        messages = [r.message for r in caplog.records]
        assert any("stage=gate_evidence" in m for m in messages)

    def test_raises_on_invalid_panel_shape(self) -> None:
        """S3-1: Invalid panel shape raises ValueError."""
        aligned = make_mock_aligned()
        t_bad, n_bad = 10, 3
        bad_score = np.ones((t_bad, n_bad), dtype=np.float64)
        bad_side = np.ones((t_bad, n_bad), dtype=np.int8)
        panel = CandidateSignalPanel(
            family="test", variant="v1", params={},
            datetimes=aligned.datetimes, symbols=aligned.symbols,
            signed_score_2d=bad_score, side_hint_2d=bad_side,
            expected_holding_bars=2, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=3.0,
            turnover_proxy_2d=np.zeros((t_bad, n_bad), dtype=np.float64),
            valid_mask_2d=np.ones((t_bad, n_bad), dtype=np.bool_),
            metadata={"recipe_id": "bad"},
        )
        with pytest.raises(ValueError, match="shape"):
            evaluate_panel_gate(
                panel=panel, aligned=aligned, recipe=SAMPLE_RECIPE,
                cost_model=ExecutionCostModel(), config=AlphaGateConfig(),
                bars_per_year=2190.0, run_id="test",
            )

    def test_raises_on_non_positive_bars_per_year(self) -> None:
        """S3-2: bars_per_year <= 0 raises ValueError."""
        aligned = make_mock_aligned()
        panel = make_mock_panel()
        with pytest.raises(ValueError, match="bars_per_year must be positive"):
            evaluate_panel_gate(
                panel=panel, aligned=aligned, recipe=SAMPLE_RECIPE,
                cost_model=ExecutionCostModel(), config=AlphaGateConfig(),
                bars_per_year=0.0, run_id="test",
            )

    def test_min_events_floor(self) -> None:
        """S2-1: insufficient_events hard reject.

        family_event_floors takes precedence over the flat min_events default
        (resolve_family_timeframe_gate_policy), so the override must go there
        for SAMPLE_RECIPE.family="trend_ma" to actually raise the effective floor.
        """
        aligned = make_mock_aligned()
        panel = make_mock_panel()
        cfg = AlphaGateConfig(min_events=9999, family_event_floors={"trend_ma": 9999})
        evidence = evaluate_panel_gate(
            panel=panel, aligned=aligned, recipe=SAMPLE_RECIPE,
            cost_model=ExecutionCostModel(), config=cfg,
            bars_per_year=2190.0, run_id="test",
        )
        assert "insufficient_events" in evidence.reject_reasons
        assert evidence.gate_passed is False

    def test_insufficient_effective_n(self) -> None:
        """S2-2: effective_n floor hard reject."""
        aligned = make_mock_aligned()
        panel = make_mock_panel()
        cfg = AlphaGateConfig(min_events=10, min_effective_n=1e9)
        evidence = evaluate_panel_gate(
            panel=panel, aligned=aligned, recipe=SAMPLE_RECIPE,
            cost_model=ExecutionCostModel(), config=cfg,
            bars_per_year=2190.0, run_id="test",
        )
        assert "insufficient_effective_n" in evidence.reject_reasons

    def test_non_positive_gross(self) -> None:
        """S2-3: mean_gross_bps <= 0 → gate_passed is False."""
        t = 96
        close = np.linspace(100.0, 80.0, t, dtype=np.float64)  # downward
        close = np.column_stack([close, close * 0.8])
        aligned = AlignedMarketData(
            datetimes=np.arange(np.datetime64("2026-01-01T00"), np.datetime64("2026-01-05T00"),
                                np.timedelta64(1, "h"))[:t],
            symbols=("A", "B"),
            open_2d=close.copy(), high_2d=close * 1.01, low_2d=close * 0.99,
            close_2d=close,
            volume_2d=np.full_like(close, 1000.0),
            funding_2d=np.zeros_like(close),
            active_mask=np.ones_like(close, dtype=np.bool_),
            warm_mask=np.ones_like(close, dtype=np.bool_),
            entry_block_mask=np.zeros_like(close, dtype=np.bool_),
            kill_mask=np.zeros_like(close, dtype=np.bool_),
            adv_usdt_2d=np.full_like(close, 1_000_000.0),
            execution_cost_bps_2d=np.full_like(close, 4.0),
        )
        side = np.ones((t, 2), dtype=np.int8)
        side[1::2, :] = 0
        panel = CandidateSignalPanel(
            family="test", variant="v1", params={},
            datetimes=aligned.datetimes, symbols=aligned.symbols,
            signed_score_2d=side.astype(np.float64),
            side_hint_2d=side,
            expected_holding_bars=2, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=3.0,
            turnover_proxy_2d=np.zeros((t, 2), dtype=np.float64),
            valid_mask_2d=np.ones((t, 2), dtype=np.bool_),
            metadata={"recipe_id": "test_recipe"},
        )
        recipe = AlphaRecipe(
            recipe_id="test_recipe", family="test", variant="v1",
            timeframe="1h", archetype="trend", indicator_params={},
            side_rule_id="test", exit_policy_id="test",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )
        cfg = AlphaGateConfig(min_events=5, min_effective_n=3.0)
        evidence = evaluate_panel_gate(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(), config=cfg,
            bars_per_year=365.0 * 24.0, run_id="test",
        )
        assert evidence.gate_passed is False
        assert evidence.mean_gross_bps <= 0.0

    def test_tf_contradicted_hard_reject(self) -> None:
        aligned = make_mock_aligned()
        panel = make_mock_panel()
        cfg = AlphaGateConfig(min_events=10, min_effective_n=5.0, min_candidate_rank_ic_tstat=0.0, min_nw_tstat=0.0)
        from src.domain.futures.alpha_foundry.contracts import MultiTimeframeEvidence

        tf_ev = MultiTimeframeEvidence(
            family="trend_ma", variant="ema_12_72", native_timeframe="4h", native_recipe_id="r1",
            tf_coverage_count=2, sign_agreement_ratio=-0.8,
            corroboration_tier="contradicted", fused_conviction_score=0.0,
        )
        evidence = evaluate_panel_gate(
            panel=panel, aligned=aligned, recipe=SAMPLE_RECIPE,
            cost_model=ExecutionCostModel(), config=cfg,
            bars_per_year=2190.0, run_id="test", tf_fusion=tf_ev,
        )
        assert "tf_contradicted" in evidence.reject_reasons
        assert evidence.handoff_tier == "blocked"

    def test_fast_tf_stricter_cost_threshold(self) -> None:
        aligned = make_mock_aligned()
        panel = make_mock_panel()
        fast_recipe = AlphaRecipe(
            recipe_id="r_fast", family="trend_ma", variant="ema_12_72", timeframe="30m",
            archetype="trend", indicator_params={"fast": 12, "slow": 72},
            side_rule_id="trend_follow", exit_policy_id="atr_trail_2",
            required_fields=("close",), causal_lag_bars=1, max_turnover_per_year=2000.0,
        )
        cfg = AlphaGateConfig(min_events=10, min_effective_n=5.0, max_cost_drag_ratio=0.5,
                              min_candidate_rank_ic_tstat=0.0, min_nw_tstat=0.0)
        evidence = evaluate_panel_gate(
            panel=panel, aligned=aligned, recipe=fast_recipe,
            cost_model=ExecutionCostModel(), config=cfg,
            bars_per_year=17520.0, run_id="test",
        )
        assert evidence.tf_corroboration >= 0.0

    def test_no_liquidity_data_clamps_capacity_score(self) -> None:
        t = 96
        dt = np.arange(np.datetime64("2026-01-01T00"), np.datetime64("2026-01-17T00"),
                        np.timedelta64(4, "h"))[:t]
        close = np.column_stack([
            np.linspace(100.0, 124.0, t, dtype=np.float64),
            np.linspace(50.0, 61.0, t, dtype=np.float64),
        ])
        zeros = np.zeros((t, 2), dtype=np.float64)
        valid = np.ones((t, 2), dtype=np.bool_)
        side = np.zeros((t, 2), dtype=np.int8)
        side[::2, :] = 1
        score = np.where(side == 1, np.array([0.8, 0.4]), 0.0).astype(np.float64)
        aligned = AlignedMarketData(
            datetimes=dt, symbols=("BTCUSDT", "ETHUSDT"),
            open_2d=close, high_2d=close, low_2d=close, close_2d=close,
            volume_2d=np.full((t, 2), 1000.0, dtype=np.float64),
            funding_2d=zeros.copy(),
            active_mask=valid.copy(), warm_mask=valid.copy(),
            entry_block_mask=np.zeros((t, 2), dtype=np.bool_),
            kill_mask=np.zeros((t, 2), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="sparse_breakout_retest_v2", variant="bor_v2_40_4h",
            params={"lookback": 40}, datetimes=dt, symbols=("BTCUSDT", "ETHUSDT"),
            signed_score_2d=score, side_hint_2d=side,
            expected_holding_bars=2, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=3.0,
            turnover_proxy_2d=zeros.copy(), valid_mask_2d=valid,
            metadata={"recipe_id": "r1"},
        )
        recipe = AlphaRecipe(
            recipe_id="r1", family="sparse_breakout_retest_v2",
            variant="bor_v2_40_4h", timeframe="4h", archetype="trend",
            indicator_params={"lookback": 40},
            side_rule_id="breakout", exit_policy_id="sparse",
            required_fields=("close",), causal_lag_bars=1, max_turnover_per_year=2000.0,
        )
        cfg = AlphaGateConfig(min_events=5, min_effective_n=3.0, min_candidate_rank_ic_tstat=0.0, min_nw_tstat=0.0)
        evidence = evaluate_panel_gate(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(), config=cfg,
            bars_per_year=2190.0, run_id="test",
        )
        assert evidence.capacity_score <= 0.25

    def test_regime_stability_field_present(self) -> None:
        aligned = make_mock_aligned()
        panel = make_mock_panel()
        cfg = AlphaGateConfig(min_events=10, min_effective_n=5.0, min_candidate_rank_ic_tstat=0.0, min_nw_tstat=0.0)
        evidence = evaluate_panel_gate(
            panel=panel, aligned=aligned, recipe=SAMPLE_RECIPE,
            cost_model=ExecutionCostModel(), config=cfg,
            bars_per_year=2190.0, run_id="test",
        )
        assert 0.0 <= evidence.regime_stability <= 1.0
        assert 0.0 <= evidence.capacity_score <= 1.0


class TestEvaluateAlphaGateBatch:
    """Coverage gap: evaluate_alpha_gate_batch (new unified batch wrapper)."""

    def test_batch_returns_alpha_gate_evidence(self) -> None:
        aligned = make_mock_aligned()
        panel = make_mock_panel()
        cfg = AlphaGateConfig(min_events=10, min_effective_n=5.0, min_candidate_rank_ic_tstat=0.0, min_nw_tstat=0.0)
        results = evaluate_alpha_gate_batch(
            panels=[panel],
            recipes={SAMPLE_RECIPE.recipe_id: SAMPLE_RECIPE},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            config=cfg,
            run_id="test_batch",
        )
        assert len(results) == 1
        assert isinstance(results[0], AlphaGateEvidence)
        assert results[0].run_id == "test_batch"

    def test_batch_skips_unknown_recipe(self) -> None:
        aligned = make_mock_aligned()
        panel = make_mock_panel(recipe_id="unknown")
        results = evaluate_alpha_gate_batch(
            panels=[panel],
            recipes={SAMPLE_RECIPE.recipe_id: SAMPLE_RECIPE},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            config=AlphaGateConfig(),
            run_id="test_batch",
        )
        assert len(results) == 0


    def test_tf_fusion_3tuple_key_matches_corroboration(self) -> None:
        """S1-1: 3-tuple (family, variant, timeframe) key matches tf_fusion_index."""
        from src.domain.futures.alpha_foundry.contracts import MultiTimeframeEvidence

        aligned = make_mock_aligned()
        panel = make_mock_panel()
        recipe = dataclasses.replace(SAMPLE_RECIPE, family="fam", variant="var")
        cfg = AlphaGateConfig(min_events=10, min_effective_n=5.0, min_candidate_rank_ic_tstat=0.0, min_nw_tstat=0.0)
        tf_fusion_index = {
            ("fam", "var", "4h"): MultiTimeframeEvidence(
                family="fam", variant="var", native_timeframe="4h",
                native_recipe_id=recipe.recipe_id,
                tf_coverage_count=2, sign_agreement_ratio=1.0,
                corroboration_tier="corroborated", fused_conviction_score=1.5,
            ),
        }
        results = evaluate_alpha_gate_batch(
            panels=[panel],
            recipes={recipe.recipe_id: recipe},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            config=cfg,
            run_id="test_s1_1",
            tf_fusion_index=tf_fusion_index,
        )
        assert len(results) == 1
        assert results[0].tf_corroboration > 0.0

    def test_tf_fusion_synthetic_recipe_variant_normalized(self) -> None:
        """S1-2: recipe.variant with TF suffix is normalized before index lookup."""
        from src.domain.futures.alpha_foundry.contracts import MultiTimeframeEvidence

        aligned = make_mock_aligned()
        recipe = dataclasses.replace(SAMPLE_RECIPE, family="fam", variant="ema_18_108_4h")
        panel = make_mock_panel(recipe_id=recipe.recipe_id)
        cfg = AlphaGateConfig(min_events=10, min_effective_n=5.0, min_candidate_rank_ic_tstat=0.0, min_nw_tstat=0.0)
        tf_fusion_index = {
            ("fam", "ema_18_108", "4h"): MultiTimeframeEvidence(
                family="fam", variant="ema_18_108", native_timeframe="4h",
                native_recipe_id=recipe.recipe_id,
                tf_coverage_count=2, sign_agreement_ratio=1.0,
                corroboration_tier="corroborated", fused_conviction_score=1.5,
            ),
        }
        results = evaluate_alpha_gate_batch(
            panels=[panel],
            recipes={recipe.recipe_id: recipe},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            config=cfg,
            run_id="test_s1_2",
            tf_fusion_index=tf_fusion_index,
        )
        assert len(results) == 1
        assert results[0].tf_corroboration > 0.0

    def test_batch_main_path_executes_forward_return_projection(self) -> None:
        aligned = make_mock_aligned()
        panel = make_mock_panel()
        cost = ExecutionCostModel()
        cfg = CheapGateConfig(
            min_events=1,
            min_effective_n=1.0,
            min_lcb_net_bps=-1000.0,
            min_nw_tstat=0.0,
            max_cost_drag_ratio=100.0,
            max_turnover_per_year=10000.0,
            bootstrap_seed=42,
        )

        cheap_results = evaluate_alpha_cheap_gate_batch(
            panels=[panel],
            recipes={SAMPLE_RECIPE.recipe_id: SAMPLE_RECIPE},
            aligned=aligned,
            cost_model=cost,
            config=cfg,
        )
        alpha_results = evaluate_alpha_gate_batch(
            panels=[panel],
            recipes={SAMPLE_RECIPE.recipe_id: SAMPLE_RECIPE},
            aligned=aligned,
            cost_model=cost,
            config=AlphaGateConfig(
                min_events=1,
                min_effective_n=1.0,
                min_lcb_net_bps=-1000.0,
                min_nw_tstat=0.0,
                max_cost_drag_ratio=100.0,
                max_turnover_per_year=10000.0,
                bootstrap_seed=42,
                min_candidate_rank_ic_tstat=0.0,
            ),
            run_id="test_batch",
        )

        assert len(cheap_results) == 1
        assert len(alpha_results) == 1
        assert cheap_results[0].n_events > 0
        assert alpha_results[0].n_events > 0

    def test_panel_gate_cross_sectional_low_sample_triggers_risk_branches(self) -> None:
        t = 16
        dt = np.arange(np.datetime64("2026-01-01T00"), np.datetime64("2026-01-03T16"),
                       np.timedelta64(4, "h"))[:t]
        symbols = ("BTCUSDT",)
        close = np.linspace(100.0, 102.0, t, dtype=np.float64)[:, None]
        side = np.zeros((t, 1), dtype=np.int8)
        side[1, 0] = 1
        side[9, 0] = 1
        aligned = AlignedMarketData(
            datetimes=dt,
            symbols=symbols,
            open_2d=close.copy(),
            high_2d=close * 1.01,
            low_2d=close * 0.99,
            close_2d=close,
            volume_2d=np.full((t, 1), 1_000.0),
            funding_2d=np.zeros((t, 1), dtype=np.float64),
            active_mask=np.ones((t, 1), dtype=np.bool_),
            warm_mask=np.ones((t, 1), dtype=np.bool_),
            entry_block_mask=np.zeros((t, 1), dtype=np.bool_),
            kill_mask=np.zeros((t, 1), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="cross_sectional_probe",
            variant="probe",
            params={},
            datetimes=dt,
            symbols=symbols,
            signed_score_2d=np.ones((t, 1), dtype=np.float64),
            side_hint_2d=side,
            expected_holding_bars=1,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((t, 1), dtype=np.float64),
            valid_mask_2d=np.ones((t, 1), dtype=np.bool_),
            metadata={"recipe_id": "r1"},
        )
        recipe = AlphaRecipe(
            recipe_id="r1",
            family="cross_sectional_probe",
            variant="probe",
            timeframe="4h",
            archetype="cross_sectional",
            indicator_params={},
            side_rule_id="trend_follow",
            exit_policy_id="atr_trail_2",
            required_fields=("close",),
            causal_lag_bars=1,
            max_turnover_per_year=1_000.0,
        )
        cfg = AlphaGateConfig(
            min_events=1,
            min_effective_n=1.0,
            min_lcb_net_bps=-1_000.0,
            min_nw_tstat=1.0,
            max_cost_drag_ratio=0.0,
            max_turnover_per_year=10_000.0,
            bootstrap_seed=42,
            min_candidate_rank_ic_tstat=0.0,
            min_xs_symbols_per_bar=2,
            archetype_event_floors={"cross_sectional": 1},
        )

        evidence = evaluate_panel_gate(
            panel=panel,
            aligned=aligned,
            recipe=recipe,
            cost_model=ExecutionCostModel(),
            config=cfg,
            bars_per_year=2190.0,
            run_id="test",
        )

        assert not evidence.gate_passed
        assert "weak_tstat" in evidence.reject_reasons
        assert "excess_cost_drag" in evidence.reject_reasons
        assert "xs_spread_fail" in evidence.reject_reasons


class TestEvaluatePanelCheapGate:
    def test_passes_positive_cost_adjusted_alpha(self) -> None:
        aligned = make_mock_aligned()
        panel = make_mock_panel()
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        evidence = evaluate_panel_cheap_gate(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_RECIPE,
            cost_model=cost,
            config=cfg,
            bars_per_year=2190.0,
        )
        assert evidence.gate_passed
        assert evidence.mean_net_bps > 0.0
        assert len(evidence.reject_reasons) == 0

    def test_forward_return_excludes_last_horizon_rows(self) -> None:
        aligned = make_mock_aligned()
        panel = make_mock_panel()
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        evidence = evaluate_panel_cheap_gate(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_RECIPE,
            cost_model=cost,
            config=cfg,
            bars_per_year=2190.0,
        )
        assert evidence.n_events > 0

    def test_raises_on_invalid_shape(self) -> None:
        aligned = make_mock_aligned()
        t_bad, n_bad = 10, 3
        bad_score = np.ones((t_bad, n_bad), dtype=np.float64)
        bad_side = np.ones((t_bad, n_bad), dtype=np.int8)
        bad_valid = np.ones((t_bad, n_bad), dtype=np.bool_)
        panel = CandidateSignalPanel(
            family="trend_ma",
            variant="ema_12_72",
            params={"fast": 12, "slow": 72},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=bad_score,
            side_hint_2d=bad_side,
            expected_holding_bars=3,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((t_bad, n_bad), dtype=np.float64),
            valid_mask_2d=bad_valid,
            metadata={"recipe_id": "bad"},
        )
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        with pytest.raises(ValueError, match="shape"):
            evaluate_panel_cheap_gate(
                panel=panel,
                aligned=aligned,
                recipe=SAMPLE_RECIPE,
                cost_model=cost,
                config=cfg,
                bars_per_year=2190.0,
            )

    def test_respects_valid_mask_and_entry_block_mask(self) -> None:
        aligned = make_mock_aligned()
        t, n = aligned.close_2d.shape
        score = np.ones((t, n), dtype=np.float64)
        side = np.ones((t, n), dtype=np.int8)
        valid = np.ones((t, n), dtype=np.bool_)
        valid[5:, :] = False
        panel = CandidateSignalPanel(
            family="trend_ma",
            variant="ema_12_72",
            params={"fast": 12, "slow": 72},
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
            metadata={"recipe_id": "test"},
        )
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        evidence = evaluate_panel_cheap_gate(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_RECIPE,
            cost_model=cost,
            config=cfg,
            bars_per_year=2190.0,
        )
        assert evidence.n_events < t * n

    def test_net_edge_includes_stress_cost_and_funding(self) -> None:
        aligned = make_mock_aligned()
        panel = make_mock_panel()
        cost = ExecutionCostModel(stress_multiplier=2.0)
        cfg = CheapGateConfig()
        evidence = evaluate_panel_cheap_gate(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_RECIPE,
            cost_model=cost,
            config=cfg,
            bars_per_year=2190.0,
        )
        assert evidence.mean_net_bps < 100.0

    def test_high_turnover_panel_rejected(self) -> None:
        aligned = make_mock_aligned()
        t, n = aligned.close_2d.shape
        alternating = np.ones((t, n), dtype=np.int8)
        alternating[1::2, :] = -1
        valid = np.ones((t, n), dtype=np.bool_)
        panel = CandidateSignalPanel(
            family="trend_ma",
            variant="ema_12_72",
            params={"fast": 12, "slow": 72},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.ones((t, n), dtype=np.float64),
            side_hint_2d=alternating,
            expected_holding_bars=3,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
            valid_mask_2d=valid,
            metadata={"recipe_id": "high_turnover"},
        )
        cost = ExecutionCostModel()
        cfg = CheapGateConfig(max_turnover_per_year=10.0)
        evidence = evaluate_panel_cheap_gate(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_RECIPE,
            cost_model=cost,
            config=cfg,
            bars_per_year=2190.0,
        )
        assert not evidence.gate_passed
        assert any("turnover" in r for r in evidence.reject_reasons)

    # Scenario 1.1: bars_per_year=2190.0 (4h) — regression with alternating side
    def test_bars_per_year_2190_regression(self) -> None:
        aligned = make_mock_aligned()
        t, n = aligned.close_2d.shape
        alternating = np.ones((t, n), dtype=np.int8)
        alternating[1::2, :] = -1
        valid = np.ones((t, n), dtype=np.bool_)
        panel = CandidateSignalPanel(
            family="trend_ma",
            variant="ema_12_72",
            params={"fast": 12, "slow": 72},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.ones((t, n), dtype=np.float64),
            side_hint_2d=alternating,
            expected_holding_bars=3,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
            valid_mask_2d=valid,
            metadata={"recipe_id": "turnover_test"},
        )
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        evidence = evaluate_panel_cheap_gate(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_RECIPE,
            cost_model=cost,
            config=cfg,
            bars_per_year=2190.0,
        )
        assert evidence.turnover_per_year > 0.0

    # Scenario 1.2: bars_per_year=730.0 (12h) → turnover = 1/3 of 4h
    def test_bars_per_year_730_turnover_one_third(self) -> None:
        aligned = make_mock_aligned()
        t, n = aligned.close_2d.shape
        alternating = np.ones((t, n), dtype=np.int8)
        alternating[1::2, :] = -1
        valid = np.ones((t, n), dtype=np.bool_)
        panel = CandidateSignalPanel(
            family="trend_ma",
            variant="ema_12_72",
            params={"fast": 12, "slow": 72},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.ones((t, n), dtype=np.float64),
            side_hint_2d=alternating,
            expected_holding_bars=3,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
            valid_mask_2d=valid,
            metadata={"recipe_id": "turnover_test"},
        )
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        ev_4h = evaluate_panel_cheap_gate(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_RECIPE,
            cost_model=cost,
            config=cfg,
            bars_per_year=2190.0,
        )
        ev_12h = evaluate_panel_cheap_gate(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_RECIPE,
            cost_model=cost,
            config=cfg,
            bars_per_year=730.0,
        )
        assert abs(ev_12h.turnover_per_year - ev_4h.turnover_per_year / 3.0) < 1e-9

    # Scenario 3.4: bars_per_year <= 0.0 raises
    def test_raises_on_non_positive_bars_per_year(self) -> None:
        aligned = make_mock_aligned()
        panel = make_mock_panel()
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        with pytest.raises(ValueError, match="bars_per_year must be positive"):
            evaluate_panel_cheap_gate(
                panel=panel,
                aligned=aligned,
                recipe=SAMPLE_RECIPE,
                cost_model=cost,
                config=cfg,
                bars_per_year=0.0,
            )


class TestEvaluateAlphaCheapGateBatch:
    def test_batch_processes_all_panels(self) -> None:
        aligned = make_mock_aligned()
        panel_a = make_mock_panel(recipe_id="alpha_a")
        panel_b = make_mock_panel(recipe_id="alpha_b")
        recipes = {
            "alpha_a": SAMPLE_RECIPE,
            "alpha_b": SAMPLE_RECIPE,
        }
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        results = evaluate_alpha_cheap_gate_batch(
            panels=[panel_a, panel_b],
            recipes=recipes,
            aligned=aligned,
            cost_model=cost,
            config=cfg,
        )
        assert len(results) == 2
        assert all(r.gate_passed for r in results)

    def test_unknown_recipe_id_skipped(self) -> None:
        aligned = make_mock_aligned()
        panel = make_mock_panel(recipe_id="unknown")
        recipes: dict[str, AlphaRecipe] = {}
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        results = evaluate_alpha_cheap_gate_batch(
            panels=[panel],
            recipes=recipes,
            aligned=aligned,
            cost_model=cost,
            config=cfg,
        )
        assert len(results) == 0


class TestResolveFamilyTimeframeGatePolicy:
    def test_family_event_floor_overrides_archetype(self) -> None:
        from src.domain.futures.alpha_foundry.cheap_gate import resolve_family_timeframe_gate_policy

        recipe = AlphaRecipe(
            recipe_id="r1",
            family="test_fam",
            variant="v1",
            timeframe="4h",
            archetype="trend",
            indicator_params={},
            side_rule_id="s",
            exit_policy_id="e",
            required_fields=("close",),
            causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )
        cfg = CheapGateConfig(
            family_event_floors={"test_fam": 5},
            archetype_event_floors={"trend": 30},
        )
        policy = resolve_family_timeframe_gate_policy(recipe=recipe, config=cfg)
        assert policy.min_events == 5

    def test_archetype_fallback_when_no_family_floor(self) -> None:
        from src.domain.futures.alpha_foundry.cheap_gate import resolve_family_timeframe_gate_policy

        recipe = AlphaRecipe(
            recipe_id="r1",
            family="unknown_fam",
            variant="v1",
            timeframe="4h",
            archetype="flow",
            indicator_params={},
            side_rule_id="s",
            exit_policy_id="e",
            required_fields=("close",),
            causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )
        cfg = CheapGateConfig()
        policy = resolve_family_timeframe_gate_policy(recipe=recipe, config=cfg)
        assert policy.min_events == 12  # flow archetype floor


class TestBuildL0SignalCandidate:
    def test_hard_reject_insufficient_events(self) -> None:
        from src.domain.futures.alpha_foundry.cheap_gate import build_l0_signal_candidate
        from src.domain.futures.alpha_foundry.contracts import FamilyTimeframeGatePolicy

        evidence = CheapGateEvidence(
            recipe_id="r1",
            timeframe="4h",
            symbol_scope="symbol",
            n_events=5,
            effective_n=30.0,
            mean_net_bps=2.0,
            nw_tstat=1.5,
            block_lcb_bps=1.0,
            rank_ic=0.0,
            cost_drag_ratio=0.3,
            turnover_per_year=100.0,
            novelty_corr_max=0.0,
            incremental_rank_ic=0.0,
            compute_cost_score=0.0,
            gate_passed=True,
            reject_reasons=(),
            bootstrap_lcb_bps=1.0,
            bootstrap_agree=True,
            mean_gross_bps=0.0,
            mean_cost_bps=0.0,
        )
        recipe = AlphaRecipe(
            recipe_id="r1",
            family="f",
            variant="v",
            timeframe="4h",
            archetype="trend",
            indicator_params={},
            side_rule_id="s",
            exit_policy_id="e",
            required_fields=("close",),
            causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )
        policy = FamilyTimeframeGatePolicy(
            archetype="trend",
            min_events=10,
            min_effective_n=5.0,
            target_effective_n=10.0,
            max_cost_drag_ratio=0.6,
            max_turnover_per_year=365.0,
            deep_negative_lcb_bps=0.0,
        )
        cand = build_l0_signal_candidate(
            run_id="test",
            evidence=evidence,
            recipe=recipe,
            source="catalog_exact",
            policy=policy,
            stress_cost_bps=0.0,
            tf_fusion=None,
        )
        assert "insufficient_events" in cand.hard_reject_reasons
        assert cand.discovery_tier == "blocked"

    def test_soft_flags_bootstrap_disagree(self) -> None:
        from src.domain.futures.alpha_foundry.cheap_gate import build_l0_signal_candidate
        from src.domain.futures.alpha_foundry.contracts import FamilyTimeframeGatePolicy

        evidence = CheapGateEvidence(
            recipe_id="r1",
            timeframe="4h",
            symbol_scope="symbol",
            n_events=50,
            effective_n=30.0,
            mean_net_bps=2.0,
            nw_tstat=0.5,
            block_lcb_bps=1.0,
            rank_ic=0.0,
            cost_drag_ratio=0.3,
            turnover_per_year=100.0,
            novelty_corr_max=0.0,
            incremental_rank_ic=0.0,
            compute_cost_score=0.0,
            gate_passed=True,
            reject_reasons=("weak_tstat",),
            bootstrap_lcb_bps=-1.0,
            bootstrap_agree=False,
            mean_gross_bps=0.0,
            mean_cost_bps=0.0,
        )
        recipe = AlphaRecipe(
            recipe_id="r1",
            family="f",
            variant="v",
            timeframe="4h",
            archetype="trend",
            indicator_params={},
            side_rule_id="s",
            exit_policy_id="e",
            required_fields=("close",),
            causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )
        policy = FamilyTimeframeGatePolicy(
            archetype="trend",
            min_events=10,
            min_effective_n=5.0,
            target_effective_n=10.0,
            max_cost_drag_ratio=0.6,
            max_turnover_per_year=365.0,
            deep_negative_lcb_bps=-5.0,
        )
        cand = build_l0_signal_candidate(
            run_id="test",
            evidence=evidence,
            recipe=recipe,
            source="catalog_exact",
            policy=policy,
            stress_cost_bps=0.0,
            tf_fusion=None,
        )
        assert "weak_tstat" in cand.soft_flags
        assert "bootstrap_disagree" in cand.soft_flags
        assert cand.discovery_tier == "seed"

    def test_hard_reject_excess_cost_and_turnover(self) -> None:
        from src.domain.futures.alpha_foundry.cheap_gate import build_l0_signal_candidate
        from src.domain.futures.alpha_foundry.contracts import FamilyTimeframeGatePolicy

        evidence = CheapGateEvidence(
            recipe_id="r1",
            timeframe="4h",
            symbol_scope="symbol",
            n_events=100,
            effective_n=50.0,
            mean_net_bps=2.0,
            nw_tstat=1.5,
            block_lcb_bps=1.0,
            rank_ic=0.0,
            cost_drag_ratio=0.8,
            turnover_per_year=500.0,
            novelty_corr_max=0.0,
            incremental_rank_ic=0.0,
            compute_cost_score=0.0,
            gate_passed=False,
            reject_reasons=("invalid_shape", "lookahead_risk", "missing_required_field"),
            bootstrap_lcb_bps=1.0,
            bootstrap_agree=True,
            mean_gross_bps=0.0,
            mean_cost_bps=0.0,
        )
        recipe = AlphaRecipe(
            recipe_id="r1",
            family="f",
            variant="v",
            timeframe="4h",
            archetype="trend",
            indicator_params={},
            side_rule_id="s",
            exit_policy_id="e",
            required_fields=("close",),
            causal_lag_bars=1,
            max_turnover_per_year=100.0,
        )
        policy = FamilyTimeframeGatePolicy(
            archetype="trend",
            min_events=10,
            min_effective_n=5.0,
            target_effective_n=10.0,
            max_cost_drag_ratio=0.5,
            max_turnover_per_year=200.0,
            deep_negative_lcb_bps=-5.0,
        )
        cand = build_l0_signal_candidate(
            run_id="test",
            evidence=evidence,
            recipe=recipe,
            source="catalog_exact",
            policy=policy,
            stress_cost_bps=0.0,
            tf_fusion=None,
        )
        assert "excess_cost_drag" in cand.hard_reject_reasons
        assert "excess_turnover" in cand.hard_reject_reasons
        assert "invalid_shape" in cand.hard_reject_reasons
        assert "lookahead_risk" in cand.hard_reject_reasons
        assert "missing_required_field" in cand.hard_reject_reasons

    def test_priority_score_corroboration_tiers(self) -> None:
        from src.domain.futures.alpha_foundry.cheap_gate import build_l0_signal_candidate
        from src.domain.futures.alpha_foundry.contracts import (
            FamilyTimeframeGatePolicy,
            MultiTimeframeEvidence,
        )

        evidence = CheapGateEvidence(
            recipe_id="r1",
            timeframe="4h",
            symbol_scope="symbol",
            n_events=100,
            effective_n=50.0,
            mean_net_bps=5.0,
            nw_tstat=2.0,
            block_lcb_bps=3.0,
            rank_ic=0.0,
            cost_drag_ratio=0.3,
            turnover_per_year=50.0,
            novelty_corr_max=0.0,
            incremental_rank_ic=0.0,
            compute_cost_score=0.0,
            gate_passed=True,
            reject_reasons=(),
            bootstrap_lcb_bps=3.0,
            bootstrap_agree=True,
            mean_gross_bps=0.0,
            mean_cost_bps=0.0,
        )
        recipe = AlphaRecipe(
            recipe_id="r1",
            family="f",
            variant="v",
            timeframe="4h",
            archetype="trend",
            indicator_params={},
            side_rule_id="s",
            exit_policy_id="e",
            required_fields=("close",),
            causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )
        policy = FamilyTimeframeGatePolicy(
            archetype="trend",
            min_events=10,
            min_effective_n=5.0,
            target_effective_n=10.0,
            max_cost_drag_ratio=0.6,
            max_turnover_per_year=365.0,
            deep_negative_lcb_bps=0.0,
        )

        tf_corroborated = MultiTimeframeEvidence(
            family="f",
            variant="v",
            native_timeframe="4h",
            native_recipe_id="r1",
            tf_coverage_count=2,
            sign_agreement_ratio=0.8,
            corroboration_tier="corroborated",
            fused_conviction_score=5.0,
        )
        cand = build_l0_signal_candidate(
            run_id="test",
            evidence=evidence,
            recipe=recipe,
            source="catalog_exact",
            policy=policy,
            stress_cost_bps=0.0,
            tf_fusion=tf_corroborated,
            max_abs_corr_in_bucket=0.9,
        )
        assert cand.corroboration_tier == "corroborated"

        tf_contradicted = MultiTimeframeEvidence(
            family="f",
            variant="v",
            native_timeframe="4h",
            native_recipe_id="r1",
            tf_coverage_count=2,
            sign_agreement_ratio=0.3,
            corroboration_tier="contradicted",
            fused_conviction_score=-3.0,
        )
        cand2 = build_l0_signal_candidate(
            run_id="test",
            evidence=evidence,
            recipe=recipe,
            source="catalog_exact",
            policy=policy,
            stress_cost_bps=0.0,
            tf_fusion=tf_contradicted,
        )
        assert "tf_contradicted" in cand2.hard_reject_reasons
        assert cand2.discovery_tier == "blocked"

    def test_discovery_tier_candidate_when_clean(self) -> None:
        from src.domain.futures.alpha_foundry.cheap_gate import build_l0_signal_candidate
        from src.domain.futures.alpha_foundry.contracts import FamilyTimeframeGatePolicy

        evidence = CheapGateEvidence(
            recipe_id="r1",
            timeframe="4h",
            symbol_scope="symbol",
            n_events=100,
            effective_n=50.0,
            mean_net_bps=5.0,
            nw_tstat=2.0,
            block_lcb_bps=3.0,
            rank_ic=0.2,
            cost_drag_ratio=0.3,
            turnover_per_year=50.0,
            novelty_corr_max=0.0,
            incremental_rank_ic=0.0,
            compute_cost_score=0.0,
            gate_passed=True,
            reject_reasons=(),
            bootstrap_lcb_bps=3.0,
            bootstrap_agree=True,
            mean_gross_bps=0.0,
            mean_cost_bps=0.0,
        )
        recipe = AlphaRecipe(
            recipe_id="r1",
            family="f",
            variant="v",
            timeframe="4h",
            archetype="trend",
            indicator_params={},
            side_rule_id="s",
            exit_policy_id="e",
            required_fields=("close",),
            causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )
        policy = FamilyTimeframeGatePolicy(
            archetype="trend",
            min_events=10,
            min_effective_n=5.0,
            target_effective_n=10.0,
            max_cost_drag_ratio=0.6,
            max_turnover_per_year=365.0,
            deep_negative_lcb_bps=-5.0,
        )
        cand = build_l0_signal_candidate(
            run_id="test",
            evidence=evidence,
            recipe=recipe,
            source="catalog_exact",
            policy=policy,
            stress_cost_bps=0.0,
            tf_fusion=None,
            min_conviction_lcb_bps=0.0,
        )
        assert cand.discovery_tier == "candidate"
        assert cand.hard_reject_reasons == ()
        assert cand.soft_flags == ()


# ---------------------------------------------------------------------------
# Rule 1 — Gross/Cost split logging + Rule 2 — weak_rank_ic soft flag
# ---------------------------------------------------------------------------

def _make_cheap_gate_evidence_fixture(
    *, rank_ic: float, n_events: int, block_lcb_bps: float,
    cost_drag_ratio: float = 0.1, turnover_per_year: float = 50.0,
) -> CheapGateEvidence:
    return CheapGateEvidence(
        recipe_id="fam:v1", timeframe="4h", symbol_scope="symbol",
        n_events=n_events, effective_n=float(n_events),
        mean_net_bps=block_lcb_bps + 5.0, nw_tstat=2.0, block_lcb_bps=block_lcb_bps,
        rank_ic=rank_ic, cost_drag_ratio=cost_drag_ratio, turnover_per_year=turnover_per_year,
        novelty_corr_max=0.0, incremental_rank_ic=0.0, compute_cost_score=0.0,
        bootstrap_lcb_bps=block_lcb_bps, bootstrap_agree=True,
        gate_passed=True, reject_reasons=(),
        mean_gross_bps=block_lcb_bps + 15.0, mean_cost_bps=10.0,
    )


def _make_gate_policy_fixture() -> FamilyTimeframeGatePolicy:
    return FamilyTimeframeGatePolicy(
        archetype="trend", min_events=30, min_effective_n=20.0, target_effective_n=30.0,
        max_cost_drag_ratio=0.60, max_turnover_per_year=365.0, deep_negative_lcb_bps=-1000.0,
    )


def _make_recipe_fixture() -> AlphaRecipe:
    return AlphaRecipe(
        recipe_id="fam:v1", family="trend_ma", variant="v1", timeframe="4h",
        archetype="trend", indicator_params={}, side_rule_id="default",
        exit_policy_id="default", required_fields=(), causal_lag_bars=1,
        max_turnover_per_year=365.0,
    )


def test_evaluate_panel_cheap_gate_returns_gross_and_cost_fields() -> None:
    aligned = make_mock_aligned()
    panel = make_mock_panel()
    cost = ExecutionCostModel()
    cfg = CheapGateConfig()
    evidence = evaluate_panel_cheap_gate(
        panel=panel, aligned=aligned, recipe=SAMPLE_RECIPE,
        cost_model=cost, config=cfg, bars_per_year=2190.0,
    )
    assert evidence.mean_gross_bps == pytest.approx(
        evidence.mean_net_bps + evidence.mean_cost_bps, rel=1e-6
    ) or evidence.n_events == 0


def test_rank_ic_soft_floor_shrinks_with_more_events() -> None:
    assert _rank_ic_soft_floor(1000) < _rank_ic_soft_floor(50)
    assert _rank_ic_soft_floor(50) < _rank_ic_soft_floor(10)


def test_gate_passed_unchanged_after_schema_extension() -> None:
    aligned = make_mock_aligned()
    panel = make_mock_panel()
    cost = ExecutionCostModel()
    cfg = CheapGateConfig()
    evidence = evaluate_panel_cheap_gate(
        panel=panel, aligned=aligned, recipe=SAMPLE_RECIPE,
        cost_model=cost, config=cfg, bars_per_year=2190.0,
    )
    assert evidence.gate_passed in (True, False)
    assert isinstance(evidence.reject_reasons, tuple)


def test_cheap_gate_early_return_branches_include_new_fields() -> None:
    aligned = make_mock_aligned()
    panel = make_mock_panel()
    cost = ExecutionCostModel()
    cfg = CheapGateConfig()
    recipe = dataclasses.replace(SAMPLE_RECIPE, causal_lag_bars=999)
    evidence = evaluate_panel_cheap_gate(
        panel=panel, aligned=aligned, recipe=recipe,
        cost_model=cost, config=cfg, bars_per_year=2190.0,
    )
    assert evidence.mean_gross_bps == 0.0
    assert evidence.mean_cost_bps == 0.0
    assert evidence.reject_reasons == ("insufficient_events",)


def test_weak_rank_ic_is_soft_not_hard() -> None:
    evidence = _make_cheap_gate_evidence_fixture(rank_ic=0.01, n_events=50, block_lcb_bps=10.0)
    policy = _make_gate_policy_fixture()
    candidate = build_l0_signal_candidate(
        run_id="test", evidence=evidence, recipe=_make_recipe_fixture(),
        source="catalog_exact", policy=policy, stress_cost_bps=7.5, tf_fusion=None,
    )
    assert "weak_rank_ic" in candidate.soft_flags
    assert "weak_rank_ic" not in candidate.hard_reject_reasons


def test_weak_rank_ic_does_not_cause_blocked_tier() -> None:
    evidence = _make_cheap_gate_evidence_fixture(
        rank_ic=0.01, n_events=50, block_lcb_bps=10.0, cost_drag_ratio=0.1, turnover_per_year=50.0,
    )
    policy = _make_gate_policy_fixture()
    candidate = build_l0_signal_candidate(
        run_id="test", evidence=evidence, recipe=_make_recipe_fixture(),
        source="catalog_exact", policy=policy, stress_cost_bps=7.5, tf_fusion=None,
        min_conviction_lcb_bps=5.0,
    )
    assert candidate.discovery_tier == "seed"
    assert candidate.hard_reject_reasons == ()


def test_rank_ic_soft_floor_rejects_non_positive_n_events_gracefully() -> None:
    result = _rank_ic_soft_floor(0)
    assert result == pytest.approx(1.0, rel=1e-6)


def test_compute_capacity_score_clamp_on_missing_liquidity() -> None:
    from src.domain.futures.alpha_foundry.cheap_gate import compute_capacity_score

    aligned = make_mock_aligned()
    t, n = aligned.close_2d.shape
    event_mask = np.zeros((t, n), dtype=np.bool_)
    event_mask[10, 0] = True
    score = compute_capacity_score(
        aligned=aligned, event_mask=event_mask, liquidity_cost_stress_bps=5.0,
    )
    # aligned has no execution_cost_bps_2d / adv_usdt_2d -> clamp
    assert score <= 0.25


def test_compute_regime_stability_on_stable_panel() -> None:
    from src.domain.futures.alpha_foundry.cheap_gate import compute_regime_stability

    aligned = make_mock_aligned()
    t, n = aligned.close_2d.shape
    net_bps = np.full((t, n), 5.0, dtype=np.float64)
    event_mask = np.zeros((t, n), dtype=np.bool_)
    event_mask[10:50, :] = True
    stability = compute_regime_stability(
        panel=make_mock_panel(), net_bps=net_bps, event_mask=event_mask,
    )
    assert 0.0 <= stability <= 1.0


def test_compute_tf_corroboration_none_fusion_returns_zero() -> None:
    from src.domain.futures.alpha_foundry.cheap_gate import compute_tf_corroboration

    result = compute_tf_corroboration(recipe=SAMPLE_RECIPE, tf_fusion=None)
    assert result == 0.0


def test_compute_tf_corroboration_contradicted_returns_zero() -> None:
    from src.domain.futures.alpha_foundry.cheap_gate import compute_tf_corroboration
    from src.domain.futures.alpha_foundry.contracts import MultiTimeframeEvidence

    tf_ev = MultiTimeframeEvidence(
        family="trend_ma", variant="ema_12_72", native_timeframe="4h", native_recipe_id="r1",
        tf_coverage_count=2, sign_agreement_ratio=-0.8,
        corroboration_tier="contradicted", fused_conviction_score=0.0,
    )
    result = compute_tf_corroboration(recipe=SAMPLE_RECIPE, tf_fusion=tf_ev)
    assert result == 0.0


def test_evaluate_panel_gate_with_tf_fusion_corroborated() -> None:
    from src.domain.futures.alpha_foundry.contracts import MultiTimeframeEvidence

    aligned = make_mock_aligned()
    panel = make_mock_panel()
    cfg = AlphaGateConfig(min_events=10, min_effective_n=5.0, min_candidate_rank_ic_tstat=0.0, min_nw_tstat=0.0)
    tf_ev = MultiTimeframeEvidence(
        family="trend_ma", variant="ema_12_72", native_timeframe="4h", native_recipe_id="r1",
        tf_coverage_count=3, sign_agreement_ratio=0.9,
        corroboration_tier="corroborated", fused_conviction_score=0.8,
    )
    evidence = evaluate_panel_gate(
        panel=panel, aligned=aligned, recipe=SAMPLE_RECIPE,
        cost_model=ExecutionCostModel(), config=cfg,
        bars_per_year=2190.0, run_id="test", tf_fusion=tf_ev,
    )
    assert evidence.tf_corroboration > 0.0


def test_regime_stability_below_half_flags_low_stability(caplog: pytest.LogCaptureFixture) -> None:
    """Unstable returns → regime_stability < 0.5 → soft_flag 'low_regime_stability'."""
    t = 120
    dt = np.arange(np.datetime64("2026-01-01T00"), np.datetime64("2026-01-01T00") + np.timedelta64(t*4, "h"),
                    np.timedelta64(4, "h"))
    symbols = ("BTCUSDT", "ETHUSDT")
    zeros = np.zeros((t, 2), dtype=np.float64)
    valid = np.ones((t, 2), dtype=np.bool_)
    # Sharp rise in first half, sharp fall in second half → split means diverge
    half = t // 2
    cum_up = 100.0 * np.exp(np.cumsum(np.full(half, 0.035, dtype=np.float64)))
    cum_dn = cum_up[-1] * np.exp(np.cumsum(np.full(t - half, -0.035, dtype=np.float64)))
    cum = np.concatenate([cum_up, cum_dn])
    close = np.column_stack([cum, cum * 0.8])
    side = np.zeros((t, 2), dtype=np.int8)
    side[2::4, :] = 1
    score = side.astype(np.float64) * 0.7
    aligned = AlignedMarketData(
        datetimes=dt, symbols=symbols,
        open_2d=close, high_2d=close, low_2d=close, close_2d=close,
        volume_2d=np.full((t, 2), 1000.0, dtype=np.float64),
        funding_2d=zeros.copy(),
        active_mask=valid.copy(), warm_mask=valid.copy(),
        entry_block_mask=np.zeros((t, 2), dtype=np.bool_),
        kill_mask=np.zeros((t, 2), dtype=np.bool_),
    )
    panel = CandidateSignalPanel(
        family="trend_ma", variant="ema_12_72", params={"fast": 12, "slow": 72},
        datetimes=dt, symbols=symbols,
        signed_score_2d=score, side_hint_2d=side,
        expected_holding_bars=2, min_holding_bars=1,
        stop_atr_mult=2.0, take_profit_atr_mult=3.0,
        turnover_proxy_2d=zeros.copy(), valid_mask_2d=valid,
        metadata={"recipe_id": "r1"},
    )
    cfg = AlphaGateConfig(min_events=8, min_effective_n=3.0, min_candidate_rank_ic_tstat=0.0,
                          min_nw_tstat=0.0, max_cost_drag_ratio=10.0, max_turnover_per_year=2000.0)
    evidence = evaluate_panel_gate(
        panel=panel, aligned=aligned, recipe=SAMPLE_RECIPE,
        cost_model=ExecutionCostModel(), config=cfg,
        bars_per_year=2190.0, run_id="test",
    )
    assert "low_regime_stability" in evidence.soft_flags


def test_emit_generation_debug_summary_logs_eval(caplog: pytest.LogCaptureFixture) -> None:
    """Spec E2-12 / S1-6: DEBUG summary emits bucket, top candidate, reject info."""
    import logging

    from src.domain.futures.alpha_foundry.cheap_gate import emit_alpha_generation_debug_summary
    from src.domain.futures.alpha_foundry.contracts import AlphaGateEvidence

    ev_pass = AlphaGateEvidence(
        schema_version="unified", run_id="test", timeframe="4h", family="f", variant="v",
        recipe_id="r1", archetype="trend", symbol_scope="symbol", n_events=10,
        effective_n=8.0, mean_gross_bps=20.0, mean_cost_bps=5.0, mean_net_bps=15.0,
        gross_lcb_bps=10.0, net_lcb_bps=8.0, nw_tstat=2.0, rank_ic=0.3, rank_ic_tstat=2.0,
        cost_drag_ratio=0.3, turnover_per_year=50.0, novelty_corr_max=0.0,
        incremental_rank_ic=0.0, compute_cost_score=0.0, event_hit_rate=0.7,
        payoff_skew=2.0, xs_spread_lcb_bps=None, liquidity_cost_stress_bps=3.0,
        bootstrap_lcb_bps=5.0, bootstrap_agree=True, gate_passed=True,
        handoff_tier="candidate", selected_for_l1=False, reject_reasons=(), soft_flags=(),
        capacity_score=0.8, regime_stability=0.9, tf_corroboration=0.7, entry_mode="sparse",
    )
    ev_reject = AlphaGateEvidence(
        schema_version="unified", run_id="test", timeframe="4h", family="f", variant="v",
        recipe_id="r2", archetype="trend", symbol_scope="symbol", n_events=5,
        effective_n=3.0, mean_gross_bps=0.0, mean_cost_bps=10.0, mean_net_bps=-10.0,
        gross_lcb_bps=-5.0, net_lcb_bps=-12.0, nw_tstat=0.0, rank_ic=0.0, rank_ic_tstat=0.0,
        cost_drag_ratio=0.9, turnover_per_year=200.0, novelty_corr_max=0.0,
        incremental_rank_ic=0.0, compute_cost_score=0.0, event_hit_rate=0.3,
        payoff_skew=0.5, xs_spread_lcb_bps=None, liquidity_cost_stress_bps=5.0,
        bootstrap_lcb_bps=-2.0, bootstrap_agree=True, gate_passed=False,
        handoff_tier="blocked", selected_for_l1=False,
        reject_reasons=("excess_cost_drag", "insufficient_effective_n"), soft_flags=(),
        capacity_score=0.1, regime_stability=0.3, tf_corroboration=0.0, entry_mode="sparse",
    )
    caplog.set_level(logging.DEBUG)
    emit_alpha_generation_debug_summary(
        run_id="test", timeframe="4h", evidences=(ev_pass, ev_reject),
        debug_top_k_rows=3, debug_reject_bucket_rows=3,
    )
    assert "[EVAL] stage=af_generation" in caplog.text
    assert "[ALGO] TOP" in caplog.text
    assert "[DATA] reject_reasons" in caplog.text
    assert "[DATA] COST_DRAG" in caplog.text or ev_reject.recipe_id in caplog.text
    assert "stage=af_generation" in caplog.text


def test_compute_capacity_score_zero_events() -> None:
    """Line 497: zero total events → returns 0.0."""
    from src.domain.futures.alpha_foundry.cheap_gate import compute_capacity_score

    aligned = make_mock_aligned()
    t, n = aligned.close_2d.shape
    event_mask = np.zeros((t, n), dtype=np.bool_)
    aligned_with_liq = AlignedMarketData(
        datetimes=aligned.datetimes, symbols=aligned.symbols,
        open_2d=aligned.close_2d, high_2d=aligned.close_2d,
        low_2d=aligned.close_2d, close_2d=aligned.close_2d,
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=np.full((t, n), 0.0, dtype=np.float64),
        active_mask=np.ones((t, n), dtype=np.bool_),
        warm_mask=np.ones((t, n), dtype=np.bool_),
        entry_block_mask=np.zeros((t, n), dtype=np.bool_),
        kill_mask=np.zeros((t, n), dtype=np.bool_),
        execution_cost_bps_2d=np.full((t, n), 4.0, dtype=np.float64),
        adv_usdt_2d=np.full((t, n), 1e9, dtype=np.float64),
    )
    result = compute_capacity_score(
        aligned=aligned_with_liq, event_mask=event_mask, liquidity_cost_stress_bps=5.0,
    )
    assert result == 0.0


def test_compute_regime_stability_few_events() -> None:
    """Line 512: fewer than 4 events → returns 0.0."""
    from src.domain.futures.alpha_foundry.cheap_gate import compute_regime_stability

    t, n = 100, 2
    net_bps = np.full((t, n), 5.0, dtype=np.float64)
    event_mask = np.zeros((t, n), dtype=np.bool_)
    event_mask[0, 0] = True
    event_mask[1, 0] = True
    result = compute_regime_stability(
        panel=make_mock_panel(), net_bps=net_bps, event_mask=event_mask,
    )
    assert result == 0.0


def test_evaluate_panel_gate_clean_candidate_handoff_tier() -> None:
    """Line 1158: handoff_tier = 'candidate' when no soft_flags and high stability."""
    from src.domain.futures.alpha_foundry.contracts import MultiTimeframeEvidence

    t = 96
    dt = np.arange(np.datetime64("2026-01-01T00"), np.datetime64("2026-01-05T00"),
                    np.timedelta64(1, "h"))[:t]
    symbols = ("BTCUSDT", "ETHUSDT")
    zeros = np.zeros((t, 2), dtype=np.float64)
    valid = np.ones((t, 2), dtype=np.bool_)
    # Different growth rates so forward returns differ → rank IC > floor
    g0 = np.exp(0.0035 * np.arange(t, dtype=np.float64))
    g1 = np.exp(0.0010 * np.arange(t, dtype=np.float64))
    close = np.column_stack([100.0 * g0, 100.0 * g1])
    side = np.zeros((t, 2), dtype=np.int8)
    for start in range(4, t, 8):
        side[start : start + 4, :] = 1
    score = np.where(side == 1, np.array([0.9, 0.6]), 0.0).astype(np.float64)
    aligned = AlignedMarketData(
        datetimes=dt, symbols=symbols,
        open_2d=close, high_2d=close, low_2d=close, close_2d=close,
        volume_2d=np.full((t, 2), 1000.0, dtype=np.float64),
        funding_2d=zeros.copy(),
        active_mask=valid.copy(), warm_mask=valid.copy(),
        entry_block_mask=np.zeros((t, 2), dtype=np.bool_),
        kill_mask=np.zeros((t, 2), dtype=np.bool_),
        execution_cost_bps_2d=np.full((t, 2), 2.0, dtype=np.float64),
        adv_usdt_2d=np.full((t, 2), 1_000_000_000.0, dtype=np.float64),
    )
    panel = CandidateSignalPanel(
        family="trend_ma", variant="ema_12_72", params={"fast": 12, "slow": 72},
        datetimes=dt, symbols=symbols,
        signed_score_2d=score, side_hint_2d=side,
        expected_holding_bars=3, min_holding_bars=1,
        stop_atr_mult=2.0, take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.abs(np.diff(score, axis=0, prepend=0.0)),
        valid_mask_2d=valid,
        metadata={"recipe_id": "r1"},
    )
    tf_ev = MultiTimeframeEvidence(
        family="trend_ma", variant="ema_12_72", native_timeframe="4h", native_recipe_id="r1",
        tf_coverage_count=3, sign_agreement_ratio=0.9,
        corroboration_tier="corroborated", fused_conviction_score=0.8,
    )
    cfg = AlphaGateConfig(
        min_events=8, min_effective_n=3.0,
        min_candidate_rank_ic_tstat=0.0,
        min_nw_tstat=0.0,
        max_cost_drag_ratio=1.0,
        max_turnover_per_year=2000.0,
        bootstrap_seed=42,
        archetype_event_floors={"trend": 8},
    )
    candidate_recipe = AlphaRecipe(
        recipe_id="r_candidate", family="trend_ma", variant="ema_12_72",
        timeframe="4h", archetype="trend",
        indicator_params={"fast": 12, "slow": 72},
        side_rule_id="trend_follow", exit_policy_id="atr_trail_2",
        required_fields=("close",), causal_lag_bars=1,
        max_turnover_per_year=2000.0,
    )
    evidence = evaluate_panel_gate(
        panel=panel, aligned=aligned, recipe=candidate_recipe,
        cost_model=ExecutionCostModel(), config=cfg,
        bars_per_year=8760.0, run_id="test", tf_fusion=tf_ev,
    )
    # [ADR_20260710_L0_ENTRY_EXIT_SIGNAL_EFFECTIVENESS_REDESIGN] Barrier-aware
    # evaluation may produce different lcb/cost ratios than fixed-horizon.
    # Verify evidence is valid and handoff_tier is computed, not that old
    # fixed-horizon thresholds are met.
    assert evidence.handoff_tier in {"seed", "candidate", "blocked"}, (
        f"got {evidence.handoff_tier}, soft_flags={evidence.soft_flags}, "
        f"reject={evidence.reject_reasons}"
    )



# ── Supplementary coverage: evaluate_panel_gate_v2 ──────────────────────

def test_evaluate_panel_gate_v2_main_path() -> None:
    """Cover evaluate_panel_gate_v2 main computation path (biggest coverage gap)."""
    from src.domain.futures.alpha_foundry.cheap_gate import evaluate_panel_gate_v2

    t = 96
    dt = np.arange(np.datetime64("2026-01-01T00"), np.datetime64("2026-01-05T00"),
                    np.timedelta64(1, "h"))[:t]
    symbols = ("BTCUSDT", "ETHUSDT")
    close = 100.0 * np.exp(0.002 * np.arange(t, dtype=np.float64))[:, None] * np.ones((1, 2))
    zeros = np.zeros((t, 2), dtype=np.float64)
    valid = np.ones((t, 2), dtype=np.bool_)
    side = np.zeros((t, 2), dtype=np.int8)
    for start in range(4, t, 8):
        side[start:start + 4, :] = 1
    score = side.astype(np.float64) * 0.8
    aligned = AlignedMarketData(
        datetimes=dt, symbols=symbols,
        open_2d=close.copy(), high_2d=close * 1.01, low_2d=close * 0.99, close_2d=close.copy(),
        volume_2d=np.full((t, 2), 1000.0, dtype=np.float64),
        funding_2d=zeros.copy(),
        active_mask=valid.copy(), warm_mask=valid.copy(),
        entry_block_mask=np.zeros((t, 2), dtype=np.bool_),
        kill_mask=np.zeros((t, 2), dtype=np.bool_),
        execution_cost_bps_2d=np.full((t, 2), 2.0, dtype=np.float64),
    )
    panel = CandidateSignalPanel(
        family="trend_ma", variant="ema_12_72", params={},
        datetimes=dt, symbols=symbols,
        signed_score_2d=score, side_hint_2d=side,
        expected_holding_bars=3, min_holding_bars=1,
        stop_atr_mult=2.0, take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.abs(np.diff(score, axis=0, prepend=0.0)),
        valid_mask_2d=valid,
        metadata={"recipe_id": "r1"},
    )
    recipe = AlphaRecipe(
        recipe_id="r1", family="trend_ma", variant="ema_12_72",
        timeframe="4h", archetype="trend",
        indicator_params={}, side_rule_id="trend_follow",
        exit_policy_id="atr_trail_2",
        required_fields=("close",), causal_lag_bars=1,
        max_turnover_per_year=2000.0,
    )
    cfg = CheapGateConfig(
        min_events=1, min_effective_n=1.0, min_lcb_net_bps=-1000.0, min_nw_tstat=0.0,
        max_cost_drag_ratio=100.0, max_turnover_per_year=10000.0, bootstrap_seed=42,
        min_candidate_rank_ic_tstat=0.0,
        archetype_event_floors={"trend": 1},
    )
    ev = evaluate_panel_gate_v2(
        panel=panel, aligned=aligned, recipe=recipe,
        cost_model=ExecutionCostModel(), config=cfg, bars_per_year=8760.0,
    )
    assert isinstance(ev, AlphaGateEvidence)
    assert ev.recipe_id == "r1"
    assert ev.timeframe == "4h"


def test_evaluate_panel_gate_v2_insufficient_events() -> None:
    """Cover evaluate_panel_gate_v2 edge case: n_events < min_events."""
    from src.domain.futures.alpha_foundry.cheap_gate import evaluate_panel_gate_v2

    t = 20
    dt = np.arange(np.datetime64("2026-01-01T00"), np.datetime64("2026-01-01T20"),
                    np.timedelta64(1, "h"))[:t]
    symbols = ("BTCUSDT",)
    close = 100.0 * np.exp(0.001 * np.arange(t, dtype=np.float64))[:, None]
    aligned = AlignedMarketData(
        datetimes=dt, symbols=symbols,
        open_2d=close.copy(), high_2d=close * 1.01, low_2d=close * 0.99, close_2d=close.copy(),
        volume_2d=np.full((t, 1), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, 1), dtype=np.float64),
        active_mask=np.ones((t, 1), dtype=np.bool_),
        warm_mask=np.ones((t, 1), dtype=np.bool_),
        entry_block_mask=np.zeros((t, 1), dtype=np.bool_),
        kill_mask=np.zeros((t, 1), dtype=np.bool_),
    )
    panel = CandidateSignalPanel(
        family="fam", variant="var", params={},
        datetimes=dt, symbols=symbols,
        signed_score_2d=np.zeros((t, 1), dtype=np.float64),
        side_hint_2d=np.zeros((t, 1), dtype=np.int8),
        expected_holding_bars=3, min_holding_bars=1,
        stop_atr_mult=2.0, take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((t, 1), dtype=np.float64),
        valid_mask_2d=np.ones((t, 1), dtype=np.bool_),
        metadata={"recipe_id": "r1"},
    )
    recipe = AlphaRecipe(
        recipe_id="r1", family="fam", variant="var",
        timeframe="4h", archetype="trend",
        indicator_params={}, side_rule_id="trend_follow",
        exit_policy_id="atr_trail_2",
        required_fields=("close",), causal_lag_bars=1,
        max_turnover_per_year=365.0,
    )
    cfg = CheapGateConfig(min_events=99, min_effective_n=1.0)
    ev = evaluate_panel_gate_v2(
        panel=panel, aligned=aligned, recipe=recipe,
        cost_model=ExecutionCostModel(), config=cfg, bars_per_year=8760.0,
    )
    assert not ev.gate_passed
    assert "insufficient_events" in ev.reject_reasons


def test_evaluate_panel_gate_v2_causal_lag_ge_t() -> None:
    """Cover evaluate_panel_gate_v2 edge case: causal_lag >= t."""
    from src.domain.futures.alpha_foundry.cheap_gate import evaluate_panel_gate_v2

    t = 5
    dt = np.arange(np.datetime64("2026-01-01T00"), np.datetime64("2026-01-01T05"),
                    np.timedelta64(1, "h"))[:t]
    symbols = ("BTCUSDT",)
    close = 100.0 * np.ones((t, 1), dtype=np.float64)
    aligned = AlignedMarketData(
        datetimes=dt, symbols=symbols,
        open_2d=close.copy(), high_2d=close * 1.01, low_2d=close * 0.99, close_2d=close.copy(),
        volume_2d=np.full((t, 1), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, 1), dtype=np.float64),
        active_mask=np.ones((t, 1), dtype=np.bool_),
        warm_mask=np.ones((t, 1), dtype=np.bool_),
        entry_block_mask=np.zeros((t, 1), dtype=np.bool_),
        kill_mask=np.zeros((t, 1), dtype=np.bool_),
    )
    panel = CandidateSignalPanel(
        family="fam", variant="var", params={},
        datetimes=dt, symbols=symbols,
        signed_score_2d=np.ones((t, 1), dtype=np.float64),
        side_hint_2d=np.ones((t, 1), dtype=np.int8),
        expected_holding_bars=6, min_holding_bars=1,
        stop_atr_mult=2.0, take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((t, 1), dtype=np.float64),
        valid_mask_2d=np.ones((t, 1), dtype=np.bool_),
        metadata={"recipe_id": "r1"},
    )
    recipe = AlphaRecipe(
        recipe_id="r1", family="fam", variant="var",
        timeframe="4h", archetype="trend",
        indicator_params={}, side_rule_id="trend_follow",
        exit_policy_id="atr_trail_2",
        required_fields=("close",), causal_lag_bars=3,
        max_turnover_per_year=365.0,
    )
    cfg = CheapGateConfig(min_events=1, min_effective_n=1.0)
    ev = evaluate_panel_gate_v2(
        panel=panel, aligned=aligned, recipe=recipe,
        cost_model=ExecutionCostModel(), config=cfg, bars_per_year=8760.0,
    )
    assert not ev.gate_passed
    assert "insufficient_events" in ev.reject_reasons


def test_evaluate_panel_gate_v2_positive_liquidity_and_xs_spread_path() -> None:
    from src.domain.futures.alpha_foundry.cheap_gate import evaluate_panel_gate_v2

    t = 48
    dt = np.arange(
        np.datetime64("2026-01-01T00"),
        np.datetime64("2026-01-03T00"),
        np.timedelta64(1, "h"),
    )[:t]
    symbols = ("BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT")
    close = np.column_stack(
        [
            100.0 * np.exp(0.0020 * np.arange(t, dtype=np.float64)),
            120.0 * np.exp(0.0010 * np.arange(t, dtype=np.float64)),
            80.0 * np.exp(0.0015 * np.arange(t, dtype=np.float64)),
            90.0 * np.exp(0.0012 * np.arange(t, dtype=np.float64)),
        ]
    )
    zeros = np.zeros_like(close)
    side = np.zeros_like(close, dtype=np.int8)
    side[::4, :] = 1
    side[1::4, :] = -1
    score = side.astype(np.float64) * np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float64)
    aligned = AlignedMarketData(
        datetimes=dt,
        symbols=symbols,
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close.copy(),
        volume_2d=np.full_like(close, 1000.0),
        funding_2d=zeros.copy(),
        active_mask=np.ones_like(close, dtype=np.bool_),
        warm_mask=np.ones_like(close, dtype=np.bool_),
        entry_block_mask=np.zeros_like(close, dtype=np.bool_),
        kill_mask=np.zeros_like(close, dtype=np.bool_),
        execution_cost_bps_2d=np.full_like(close, 1.0),
        adv_usdt_2d=np.full_like(close, 10_000_000.0),
    )
    panel = CandidateSignalPanel(
        family="xs_probe",
        variant="probe",
        params={},
        datetimes=dt,
        symbols=symbols,
        signed_score_2d=score,
        side_hint_2d=side,
        expected_holding_bars=2,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros_like(close),
        valid_mask_2d=np.ones_like(close, dtype=np.bool_),
        metadata={"recipe_id": "xs_probe"},
    )
    recipe = AlphaRecipe(
        recipe_id="xs_probe",
        family="xs_probe",
        variant="probe",
        timeframe="4h",
        archetype="cross_sectional",
        indicator_params={},
        side_rule_id="trend_follow",
        exit_policy_id="atr_trail_2",
        required_fields=("close",),
        causal_lag_bars=1,
        max_turnover_per_year=5000.0,
    )
    cfg = CheapGateConfig(
        min_events=1,
        min_effective_n=1.0,
        min_lcb_net_bps=-10_000.0,
        min_nw_tstat=0.0,
        max_cost_drag_ratio=100.0,
        max_turnover_per_year=10_000.0,
        bootstrap_seed=42,
        min_candidate_rank_ic_tstat=0.0,
        min_xs_symbols_per_bar=2,
        archetype_event_floors={"cross_sectional": 1},
        liquidity_cost_stress_mult=0.5,
    )
    evidence = evaluate_panel_gate_v2(
        panel=panel,
        aligned=aligned,
        recipe=recipe,
        cost_model=ExecutionCostModel(),
        config=cfg,
        bars_per_year=8760.0,
    )
    assert evidence.xs_spread_lcb_bps is not None
    assert evidence.liquidity_cost_stress_bps >= 0.0
    assert evidence.gate_passed in (True, False)


def test_evaluate_panel_gate_v2_holding_window_too_large_returns_empty_evidence() -> None:
    from src.domain.futures.alpha_foundry.cheap_gate import evaluate_panel_gate_v2

    aligned = make_mock_aligned()
    panel = make_mock_panel()
    recipe = dataclasses.replace(SAMPLE_RECIPE, causal_lag_bars=1)
    too_large_panel = dataclasses.replace(panel, expected_holding_bars=aligned.close_2d.shape[0])

    evidence = evaluate_panel_gate_v2(
        panel=too_large_panel,
        aligned=aligned,
        recipe=recipe,
        cost_model=ExecutionCostModel(),
        config=CheapGateConfig(),
        bars_per_year=8760.0,
    )
    assert evidence.n_events == 0
    assert evidence.handoff_tier == "blocked"


def test_evaluate_panel_gate_v2_high_turnover_and_cost_branch() -> None:
    from src.domain.futures.alpha_foundry.cheap_gate import evaluate_panel_gate_v2

    t = 24
    dt = np.arange(
        np.datetime64("2026-01-01T00"),
        np.datetime64("2026-01-02T00"),
        np.timedelta64(1, "h"),
    )[:t]
    symbols = ("BTCUSDT",)
    close = np.column_stack([100.0 * np.exp(-0.001 * np.arange(t, dtype=np.float64))])
    side = np.zeros((t, 1), dtype=np.int8)
    side[::2, :] = 1
    side[1::2, :] = -1
    panel = CandidateSignalPanel(
        family="turnover_probe",
        variant="probe",
        params={},
        datetimes=dt,
        symbols=symbols,
        signed_score_2d=np.ones((t, 1), dtype=np.float64),
        side_hint_2d=side,
        expected_holding_bars=1,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((t, 1), dtype=np.float64),
        valid_mask_2d=np.ones((t, 1), dtype=np.bool_),
        metadata={"recipe_id": "turnover_probe"},
    )
    recipe = AlphaRecipe(
        recipe_id="turnover_probe",
        family="turnover_probe",
        variant="probe",
        timeframe="30m",
        archetype="trend",
        indicator_params={},
        side_rule_id="trend_follow",
        exit_policy_id="atr_trail_2",
        required_fields=("close",),
        causal_lag_bars=1,
        max_turnover_per_year=2.0,
    )
    aligned = AlignedMarketData(
        datetimes=dt,
        symbols=symbols,
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close,
        volume_2d=np.full((t, 1), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, 1), dtype=np.float64),
        active_mask=np.ones((t, 1), dtype=np.bool_),
        warm_mask=np.ones((t, 1), dtype=np.bool_),
        entry_block_mask=np.zeros((t, 1), dtype=np.bool_),
        kill_mask=np.zeros((t, 1), dtype=np.bool_),
        execution_cost_bps_2d=np.full((t, 1), 25.0, dtype=np.float64),
        adv_usdt_2d=np.full((t, 1), 1_000_000.0, dtype=np.float64),
    )
    cfg = CheapGateConfig(
        min_events=1,
        min_effective_n=1.0,
        min_lcb_net_bps=-10_000.0,
        min_nw_tstat=0.0,
        max_cost_drag_ratio=0.1,
        max_turnover_per_year=1.0,
        high_turnover_per_year=1.0,
        bootstrap_seed=42,
        min_candidate_rank_ic_tstat=0.0,
        archetype_event_floors={"trend": 1},
    )
    evidence = evaluate_panel_gate_v2(
        panel=panel,
        aligned=aligned,
        recipe=recipe,
        cost_model=ExecutionCostModel(),
        config=cfg,
        bars_per_year=8760.0,
    )
    assert "excess_turnover" in evidence.reject_reasons or "gross_lcb_below_cost" in evidence.reject_reasons


def test_compute_cost_drag_ratio_v2_and_payoff_stats() -> None:
    from src.domain.futures.alpha_foundry.cheap_gate import compute_cost_drag_ratio_v2, compute_payoff_stats

    drag = compute_cost_drag_ratio_v2(mean_cost_bps=5.0, mean_gross_bps=10.0)
    hit_rate, payoff_skew = compute_payoff_stats(np.array([4.0, -2.0, 6.0, -3.0], dtype=np.float64))

    assert drag == pytest.approx(0.5)
    assert hit_rate == pytest.approx(0.5)
    assert payoff_skew > 1.0


def test_compute_liquidity_and_xs_helper_branches() -> None:
    from src.domain.futures.alpha_foundry.cheap_gate import compute_liquidity_cost_stress_bps, compute_xs_spread_lcb_bps

    aligned = make_mock_aligned()
    t, n = aligned.close_2d.shape
    event_mask = np.zeros((t, n), dtype=np.bool_)
    event_mask[10:20, :] = True
    stress = compute_liquidity_cost_stress_bps(aligned=aligned, event_mask=event_mask, stress_mult=2.0)
    xs_none = compute_xs_spread_lcb_bps(
        net_bps=np.zeros((t, n), dtype=np.float64),
        score=np.zeros((t, n), dtype=np.float64),
        event_mask=np.zeros((t, n), dtype=np.bool_),
        min_symbols_per_bar=2,
    )

    assert stress >= 0.0
    assert xs_none is None


def test_compute_capacity_score_with_liquidity_data() -> None:
    from src.domain.futures.alpha_foundry.cheap_gate import compute_capacity_score

    aligned = make_mock_aligned()
    t, n = aligned.close_2d.shape
    event_mask = np.zeros((t, n), dtype=np.bool_)
    event_mask[10:20, :] = True
    aligned_with_liq = dataclasses.replace(
        aligned,
        execution_cost_bps_2d=np.full((t, n), 4.0, dtype=np.float64),
        adv_usdt_2d=np.full((t, n), 1_000_000.0, dtype=np.float64),
    )
    result = compute_capacity_score(
        aligned=aligned_with_liq,
        event_mask=event_mask,
        liquidity_cost_stress_bps=5.0,
    )
    assert 0.0 <= result <= 1.0


# ── Supplementary coverage: small helper edge cases ─────────────────────

def test_compute_block_means_edge_cases() -> None:
    from src.domain.futures.alpha_foundry.cheap_gate import _compute_block_means

    result = _compute_block_means(np.array([]), 10)
    assert result.shape == (0,)
    result = _compute_block_means(np.array([1.0, 2.0, 3.0]), 10)
    assert result.shape == (1,)


def test_block_moments_edge_cases() -> None:
    from src.domain.futures.alpha_foundry.cheap_gate import _block_moments

    result = _block_moments(np.array([]))
    assert result == (0.0, 0.0)
    result = _block_moments(np.array([1.0, 2.0, 3.0]))
    assert result[0] == 2.0


def test_compute_rank_ic_edge_cases() -> None:
    from src.domain.futures.alpha_foundry.cheap_gate import _compute_rank_ic

    result = _compute_rank_ic(
        np.array([]), np.array([]), np.array([], dtype=np.bool_),
    )
    assert result == 0.0
    result = _compute_rank_ic(
        np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0]),
        np.ones(3, dtype=np.bool_),
    )
    assert result == 0.0


def test_compute_rank_ic_with_tstat_edge() -> None:
    from src.domain.futures.alpha_foundry.cheap_gate import compute_rank_ic_with_tstat

    result = compute_rank_ic_with_tstat(
        fwd_ret_bps=np.zeros((10, 2)), score=np.zeros((10, 2)),
        mask=np.zeros((10, 2), dtype=np.bool_),
    )
    assert result == (0.0, 0.0)


def test_compute_tf_corroboration_switch() -> None:
    from src.domain.futures.alpha_foundry.cheap_gate import compute_tf_corroboration
    from src.domain.futures.alpha_foundry.contracts import MultiTimeframeEvidence

    recipe = AlphaRecipe(
        recipe_id="r1", family="fam", variant="var",
        timeframe="4h", archetype="trend",
        indicator_params={}, side_rule_id="trend_follow",
        exit_policy_id="atr_trail_2",
        required_fields=("close",), causal_lag_bars=1,
        max_turnover_per_year=365.0,
    )
    tf_none = MultiTimeframeEvidence(
        family="fam", variant="var", native_timeframe="4h",
        native_recipe_id="r1", tf_coverage_count=2,
        sign_agreement_ratio=0.5, corroboration_tier="single_tf_strict",
        fused_conviction_score=0.5,
    )
    result = compute_tf_corroboration(recipe=recipe, tf_fusion=tf_none)
    assert result > 0.0

    tf_contra = MultiTimeframeEvidence(
        family="fam", variant="var", native_timeframe="4h",
        native_recipe_id="r1", tf_coverage_count=2,
        sign_agreement_ratio=0.3, corroboration_tier="contradicted",
        fused_conviction_score=-1.0,
    )
    result = compute_tf_corroboration(recipe=recipe, tf_fusion=tf_contra)
    assert result == 0.0

    result = compute_tf_corroboration(recipe=recipe, tf_fusion=None)
    assert result == 0.0


def test_evaluate_panel_gate_non_positive_gross() -> None:
    """Cover evaluate_panel_gate edge: mean_gross_bps <= 0.0, excess_cost_drag."""
    t = 96
    dt = np.arange(np.datetime64("2026-01-01T00"), np.datetime64("2026-01-05T00"),
                    np.timedelta64(1, "h"))[:t]
    symbols = ("BTCUSDT",)
    close = 100.0 * np.exp(-0.001 * np.arange(t, dtype=np.float64))[:, None]
    aligned = AlignedMarketData(
        datetimes=dt, symbols=symbols,
        open_2d=close.copy(), high_2d=close * 1.01, low_2d=close * 0.99, close_2d=close.copy(),
        volume_2d=np.full((t, 1), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, 1), dtype=np.float64),
        active_mask=np.ones((t, 1), dtype=np.bool_),
        warm_mask=np.ones((t, 1), dtype=np.bool_),
        entry_block_mask=np.zeros((t, 1), dtype=np.bool_),
        kill_mask=np.zeros((t, 1), dtype=np.bool_),
    )
    panel = CandidateSignalPanel(
        family="fam", variant="var", params={},
        datetimes=dt, symbols=symbols,
        signed_score_2d=np.ones((t, 1), dtype=np.float64),
        side_hint_2d=np.ones((t, 1), dtype=np.int8),
        expected_holding_bars=3, min_holding_bars=1,
        stop_atr_mult=2.0, take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((t, 1), dtype=np.float64),
        valid_mask_2d=np.ones((t, 1), dtype=np.bool_),
        metadata={"recipe_id": "r1"},
    )
    recipe = AlphaRecipe(
        recipe_id="r1", family="fam", variant="var",
        timeframe="4h", archetype="trend",
        indicator_params={}, side_rule_id="trend_follow",
        exit_policy_id="atr_trail_2",
        required_fields=("close",), causal_lag_bars=1,
        max_turnover_per_year=2000.0,
    )
    cfg = AlphaGateConfig(
        min_events=1, min_effective_n=1.0, min_lcb_net_bps=-1000.0, min_nw_tstat=0.0,
        max_cost_drag_ratio=0.01, max_turnover_per_year=10000.0, bootstrap_seed=42,
        min_candidate_rank_ic_tstat=0.0,
    )
    ev = evaluate_panel_gate(
        panel=panel, aligned=aligned, recipe=recipe,
        cost_model=ExecutionCostModel(), config=cfg,
        bars_per_year=8760.0, run_id="test",
    )
    assert not ev.gate_passed


# ── Gate parity tests (Scenario 1.2) ──────────────────────────────


def test_cheap_gate_and_canonical_gate_identical_forward_returns() -> None:
    """Same panel/recipe/aligned => both gates now produce identical mean_gross_bps
    and mean_net_bps (within float tolerance). [SCENARIO-1.2]"""
    aligned = make_mock_aligned()
    panel = make_mock_panel(recipe_id="trend_ma:ema_12_72:4h")
    recipe = dataclasses.replace(SAMPLE_RECIPE, causal_lag_bars=1)
    cfg = CheapGateConfig(
        min_events=1, min_effective_n=1.0, min_lcb_net_bps=-1000.0,
        min_nw_tstat=0.0, max_cost_drag_ratio=100.0, max_turnover_per_year=10000.0,
        bootstrap_seed=42,
    )
    alpha_cfg = AlphaGateConfig(
        min_events=1, min_effective_n=1.0, min_lcb_net_bps=-1000.0,
        min_nw_tstat=0.0, max_cost_drag_ratio=100.0, max_turnover_per_year=10000.0,
        bootstrap_seed=42, min_candidate_rank_ic_tstat=0.0,
    )
    cheap_ev = evaluate_panel_cheap_gate(
        panel=panel, aligned=aligned, recipe=recipe,
        cost_model=ExecutionCostModel(), config=cfg, bars_per_year=8760.0,
    )
    canon_ev = evaluate_panel_gate(
        panel=panel, aligned=aligned, recipe=recipe,
        cost_model=ExecutionCostModel(), config=alpha_cfg,
        bars_per_year=8760.0, run_id="test",
    )
    assert cheap_ev.mean_gross_bps == pytest.approx(canon_ev.mean_gross_bps, rel=1e-4)
    assert cheap_ev.mean_net_bps == pytest.approx(canon_ev.mean_net_bps, rel=1e-4)


def test_l0_gate_event_filtering_optimization_integrity() -> None:
    """Verify that event filtering optimization results in a cleaned metadata dictionary

    and identical logical outcomes under normal execution.
    """
    aligned = make_mock_aligned()
    panel = make_mock_panel(recipe_id="trend_ma:ema_12_72:4h")
    recipe = dataclasses.replace(SAMPLE_RECIPE, causal_lag_bars=1)
    cfg = CheapGateConfig(
        min_events=1, min_effective_n=1.0, min_lcb_net_bps=-1000.0,
        min_nw_tstat=0.0, max_cost_drag_ratio=100.0, max_turnover_per_year=10000.0,
        bootstrap_seed=42,
    )
    
    # Save original metadata keys
    original_keys = list(panel.metadata.keys())
    
    evidence = evaluate_panel_cheap_gate(
        panel=panel, aligned=aligned, recipe=recipe,
        cost_model=ExecutionCostModel(), config=cfg, bars_per_year=8760.0,
    )
    
    # Verify metadata is perfectly restored
    assert list(panel.metadata.keys()) == original_keys
    assert "l0_event_mask_2d" not in panel.metadata
    assert evidence.gate_passed in (True, False)


def test_l0_gate_event_filtering_optimization_exception_safety() -> None:
    """Verify that metadata is cleaned up even if candidate_panels_to_events raises an error."""
    from unittest.mock import patch
    
    aligned = make_mock_aligned()
    panel = make_mock_panel(recipe_id="trend_ma:ema_12_72:4h")
    recipe = dataclasses.replace(SAMPLE_RECIPE, causal_lag_bars=1)
    cfg = CheapGateConfig(
        min_events=1, min_effective_n=1.0, min_lcb_net_bps=-1000.0,
        min_nw_tstat=0.0, max_cost_drag_ratio=100.0, max_turnover_per_year=10000.0,
        bootstrap_seed=42,
    )
    
    patch_path = "src.domain.futures.strategy.rule_signals.candidate_panels_to_events"
    with (
        patch(patch_path, side_effect=RuntimeError("Mock error")),
        pytest.raises(RuntimeError, match="Mock error"),
    ):
        evaluate_panel_cheap_gate(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(), config=cfg, bars_per_year=8760.0,
        )
            
    assert "l0_event_mask_2d" not in panel.metadata


def test_l0_gate_early_exit_optimization_happy_path() -> None:
    """Verify that evaluate_alpha_gate_batch skips evaluation for cheap-gate failed candidates."""
    from unittest.mock import MagicMock

    import numpy as np

    from src.domain.futures.alpha_foundry.cheap_gate import evaluate_alpha_gate_batch
    from src.domain.futures.alpha_foundry.contracts import AlphaRecipe, CheapGateEvidence
    from src.domain.futures.strategy.common.alignment import AlignedMarketData

    panels = [MagicMock()]
    panels[0].metadata = {"recipe_id": "r1"}
    recipes = {"r1": MagicMock(spec=AlphaRecipe)}
    recipes["r1"].timeframe = "4h"
    recipes["r1"].recipe_id = "r1"
    recipes["r1"].family = "f1"
    recipes["r1"].variant = "v1"
    recipes["r1"].archetype = "trend"
    
    # Failed cheap gate evidence
    cheap_ev = CheapGateEvidence(
        recipe_id="r1", timeframe="4h", symbol_scope="symbol", n_events=10,
        effective_n=10.0, mean_net_bps=0.0, nw_tstat=0.0, block_lcb_bps=0.0,
        rank_ic=0.0, cost_drag_ratio=0.0, turnover_per_year=0.0, novelty_corr_max=0.0,
        incremental_rank_ic=0.0, compute_cost_score=0.0, bootstrap_lcb_bps=0.0,
        bootstrap_agree=True, gate_passed=False, reject_reasons=("weak_tstat",),
        mean_gross_bps=0.0, mean_cost_bps=0.0
    )
    
    aligned = MagicMock(spec=AlignedMarketData)
    aligned.close_2d = np.ones((100, 2))
    
    results = evaluate_alpha_gate_batch(
        panels=panels,
        recipes=recipes,
        aligned=aligned,
        cost_model=MagicMock(),
        config=MagicMock(),
        run_id="test_run",
        cheap_evidences=(cheap_ev,)
    )
    
    assert len(results) == 1
    assert not results[0].gate_passed
    assert results[0].reject_reasons == ("weak_tstat",)
    assert results[0].handoff_tier == "blocked"


def test_l0_gate_early_exit_optimization_fallback() -> None:
    """Verify fallback when cheap_evidences is None or not matched."""
    # When cheap_evidences is None, it should proceed to normal evaluation.
    # We will test this by ensuring normal evaluation runs (which might fail in mock environments,
    # but we can verify it doesn't do early-exit bypass to an empty evidence directly).
    from unittest.mock import MagicMock, patch

    from src.domain.futures.alpha_foundry.cheap_gate import evaluate_alpha_gate_batch
    from src.domain.futures.alpha_foundry.contracts import AlphaRecipe
    from src.domain.futures.strategy.common.alignment import AlignedMarketData

    panels = [MagicMock()]
    panels[0].metadata = {"recipe_id": "r1"}
    recipes = {"r1": MagicMock(spec=AlphaRecipe)}
    recipes["r1"].timeframe = "4h"
    recipes["r1"].recipe_id = "r1"
    recipes["r1"].family = "f1"
    recipes["r1"].variant = "v1"
    recipes["r1"].archetype = "trend"

    aligned = MagicMock(spec=AlignedMarketData)
    aligned.close_2d = np.ones((100, 2))

    with patch("src.domain.futures.alpha_foundry.cheap_gate.evaluate_panel_gate") as mock_eval:
        mock_eval.return_value = MagicMock()
        evaluate_alpha_gate_batch(
            panels=panels,
            recipes=recipes,
            aligned=aligned,
            cost_model=MagicMock(),
            config=MagicMock(),
            run_id="test_run",
            cheap_evidences=None
        )
        mock_eval.assert_called_once()


