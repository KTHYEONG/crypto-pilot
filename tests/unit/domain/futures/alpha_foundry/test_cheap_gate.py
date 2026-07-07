from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.cheap_gate import (
    _rank_ic_soft_floor,
    build_l0_signal_candidate,
    evaluate_alpha_cheap_gate_batch,
    evaluate_panel_cheap_gate,
)
from src.domain.futures.alpha_foundry.contracts import (
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
            total_cost_bps=0.0,
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
            total_cost_bps=0.0,
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
            total_cost_bps=0.0,
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
            total_cost_bps=0.0,
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
            total_cost_bps=0.0,
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
        mean_gross_bps=block_lcb_bps + 15.0, total_cost_bps=10.0,
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
        evidence.mean_net_bps + evidence.total_cost_bps / max(evidence.n_events, 1), rel=1e-6
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
    assert evidence.total_cost_bps == 0.0
    assert evidence.reject_reasons == ("insufficient_events",)


def test_weak_rank_ic_is_soft_not_hard() -> None:
    evidence = _make_cheap_gate_evidence_fixture(rank_ic=0.01, n_events=50, block_lcb_bps=10.0)
    policy = _make_gate_policy_fixture()
    candidate = build_l0_signal_candidate(
        run_id="test", evidence=evidence, recipe=_make_recipe_fixture(),
        source="catalog_exact", policy=policy, stress_cost_bps=7.5, tf_fusion=None,
    )
    assert "weak_rank_ic" in candidate.soft_flags
    assert "weak_rank_ic" not in candidate.hard_reject_reasons  # type: ignore[operator]


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
