from __future__ import annotations

import pytest

from src.domain.futures.alpha_foundry.budget import (
    build_l1_verification_units,
    update_successive_halving_state,
)
from src.domain.futures.alpha_foundry.contracts import (
    AlphaRecipe,
    CheapGateEvidence,
    L1PosteriorEvidence,
    L1VerificationUnit,
    PosteriorGateConfig,
)

NOOP_RECIPES: dict[str, AlphaRecipe] = {}

SAMPLE_RECIPES = {
    "r1": AlphaRecipe(
        recipe_id="r1",
        family="trend_ma",
        variant="ema_12_72",
        timeframe="4h",
        archetype="trend",
        indicator_params={},
        side_rule_id="trend_follow",
        exit_policy_id="atr_trail_2",
        required_fields=("close",),
        causal_lag_bars=1,
        max_turnover_per_year=365.0,
    ),
}

PASS_EVIDENCE = CheapGateEvidence(
    recipe_id="r1",
    timeframe="4h",
    symbol_scope="symbol",
    n_events=100,
    effective_n=50.0,
    mean_net_bps=5.0,
    nw_tstat=2.5,
    block_lcb_bps=2.0,
    rank_ic=0.05,
    cost_drag_ratio=0.3,
    turnover_per_year=100.0,
    novelty_corr_max=0.0,
    incremental_rank_ic=0.02,
    compute_cost_score=0.0,
    gate_passed=True,
    reject_reasons=(),
    bootstrap_lcb_bps=1.5,
    bootstrap_agree=True,
    mean_gross_bps=0.0,
    total_cost_bps=0.0,
)

FAIL_EVIDENCE = CheapGateEvidence(
    recipe_id="r2",
    timeframe="4h",
    symbol_scope="symbol",
    n_events=5,
    effective_n=3.0,
    mean_net_bps=0.0,
    nw_tstat=0.0,
    block_lcb_bps=-1.0,
    rank_ic=0.0,
    cost_drag_ratio=0.0,
    turnover_per_year=0.0,
    novelty_corr_max=0.0,
    incremental_rank_ic=0.0,
    compute_cost_score=0.0,
    gate_passed=False,
    reject_reasons=("insufficient_events",),
    bootstrap_lcb_bps=0.0,
    bootstrap_agree=True,
    mean_gross_bps=0.0,
    total_cost_bps=0.0,
)


class TestBuildL1VerificationUnits:
    def test_units_built_only_from_gate_survivors(self) -> None:
        units = build_l1_verification_units(
            evidences=[PASS_EVIDENCE, FAIL_EVIDENCE],
            recipes=SAMPLE_RECIPES,
            symbols=("BTCUSDT",),
            top_k_per_family_tf=5,
            initial_fold_budget=3,
        )
        ids = [u.recipe_id for u in units]
        assert "r1" in ids
        assert "r2" not in ids

    def test_all_units_have_positive_fold_budget(self) -> None:
        units = build_l1_verification_units(
            evidences=[PASS_EVIDENCE],
            recipes=SAMPLE_RECIPES,
            symbols=("BTCUSDT",),
            top_k_per_family_tf=5,
            initial_fold_budget=3,
        )
        for u in units:
            assert u.allocated_fold_budget > 0

    def test_units_are_frozen(self) -> None:
        units = build_l1_verification_units(
            evidences=[PASS_EVIDENCE],
            recipes=SAMPLE_RECIPES,
            symbols=("BTCUSDT",),
            top_k_per_family_tf=5,
            initial_fold_budget=3,
        )
        with pytest.raises(AttributeError):
            units[0].allocated_fold_budget = 99  # type: ignore[misc]

    # Scenario 2.6: top_k 초과
    def test_raises_when_top_k_exceeded(self) -> None:
        # 동일 (family, timeframe)에 6건
        evs = [
            CheapGateEvidence(
                recipe_id=f"r{i}",
                timeframe="4h",
                symbol_scope="symbol",
                n_events=100,
                effective_n=50.0,
                mean_net_bps=5.0,
                nw_tstat=2.5,
                block_lcb_bps=2.0,
                rank_ic=0.05,
                cost_drag_ratio=0.3,
                turnover_per_year=100.0,
                novelty_corr_max=0.0,
                incremental_rank_ic=0.02,
                compute_cost_score=0.0,
                gate_passed=True,
                reject_reasons=(),
                bootstrap_lcb_bps=1.5,
                bootstrap_agree=True,
            mean_gross_bps=0.0,
            total_cost_bps=0.0,
        )
            for i in range(6)
        ]
        big_recipes = {
            f"r{i}": AlphaRecipe(
                recipe_id=f"r{i}",
                family="trend_ma",
                variant=f"v{i}",
                timeframe="4h",
                archetype="trend",
                indicator_params={},
                side_rule_id="trend_follow",
                exit_policy_id="atr_trail_2",
                required_fields=("close",),
                causal_lag_bars=1,
                max_turnover_per_year=365.0,
            )
            for i in range(6)
        }
        with pytest.raises(ValueError, match="L0 diversity budget violated"):
            build_l1_verification_units(
                evidences=evs,
                recipes=big_recipes,
                symbols=("BTCUSDT",),
                top_k_per_family_tf=5,
                initial_fold_budget=3,
            )


class TestUpdateSuccessiveHalvingState:
    def test_promotes_high_posterior_utility(self) -> None:
        unit = L1VerificationUnit(
            unit_id="u1",
            recipe_id="r1",
            timeframe="4h",
            scope_symbols=("BTCUSDT",),
            prior_mu_bps=0.0,
            prior_sigma_bps=10.0,
            allocated_fold_budget=3,
            early_stop_state="pending",
        )
        posterior = (
            L1PosteriorEvidence(
                symbol="BTCUSDT",
                recipe_id="r1",
                family="trend_ma",
                timeframe="4h",
                activation_context="pooled",
                posterior_mu_bps=10.0,
                posterior_sigma_bps=5.0,
                prob_mu_gt_cost=0.85,
                lcb_net_bps=2.0,
                q_value=0.05,
                fold_pass_ratio=0.8,
                regime_stability=0.6,
                quality_weight=0.7,
                activation_contract="hard",
            ),
        )
        updated = update_successive_halving_state(
            units=[unit],
            posterior=posterior,
            eta=2,
            max_fold_budget=10,
            config=PosteriorGateConfig(),
        )
        assert updated[0].early_stop_state == "promote"

    def test_drops_low_posterior_utility(self) -> None:
        unit = L1VerificationUnit(
            unit_id="u1",
            recipe_id="r1",
            timeframe="4h",
            scope_symbols=("BTCUSDT",),
            prior_mu_bps=0.0,
            prior_sigma_bps=10.0,
            allocated_fold_budget=3,
            early_stop_state="pending",
        )
        posterior = (
            L1PosteriorEvidence(
                symbol="BTCUSDT",
                recipe_id="r1",
                family="trend_ma",
                timeframe="4h",
                activation_context="pooled",
                posterior_mu_bps=0.0,
                posterior_sigma_bps=5.0,
                prob_mu_gt_cost=0.3,
                lcb_net_bps=-1.0,
                q_value=0.5,
                fold_pass_ratio=0.2,
                regime_stability=0.1,
                quality_weight=0.0,
                activation_contract="observe",
            ),
        )
        updated = update_successive_halving_state(
            units=[unit],
            posterior=posterior,
            eta=2,
            max_fold_budget=10,
            config=PosteriorGateConfig(),
        )
        assert updated[0].early_stop_state == "drop"

    def test_survivor_with_no_recipe_is_skipped(self) -> None:
        units = build_l1_verification_units(
            evidences=[PASS_EVIDENCE],
            recipes=NOOP_RECIPES,
            symbols=("BTCUSDT",),
            top_k_per_family_tf=5,
            initial_fold_budget=3,
        )
        assert len(units) == 0

    def test_unit_with_no_posterior_keeps_pending(self) -> None:
        unit = L1VerificationUnit(
            unit_id="u1",
            recipe_id="r1",
            timeframe="4h",
            scope_symbols=("BTCUSDT",),
            prior_mu_bps=0.0,
            prior_sigma_bps=10.0,
            allocated_fold_budget=3,
            early_stop_state="pending",
        )
        updated = update_successive_halving_state(
            units=[unit],
            posterior=(),
            eta=2,
            max_fold_budget=10,
            config=PosteriorGateConfig(),
        )
        assert len(updated) == 1
        assert updated[0].early_stop_state == "pending"

    def test_continue_state_when_prob_between_thresholds(self) -> None:
        unit = L1VerificationUnit(
            unit_id="u1",
            recipe_id="r1",
            timeframe="4h",
            scope_symbols=("BTCUSDT",),
            prior_mu_bps=0.0,
            prior_sigma_bps=10.0,
            allocated_fold_budget=3,
            early_stop_state="pending",
        )
        posterior = (
            L1PosteriorEvidence(
                symbol="BTCUSDT",
                recipe_id="r1",
                family="trend_ma",
                timeframe="4h",
                activation_context="pooled",
                posterior_mu_bps=3.0,
                posterior_sigma_bps=5.0,
                prob_mu_gt_cost=0.55,
                lcb_net_bps=0.5,
                q_value=0.3,
                fold_pass_ratio=0.5,
                regime_stability=0.3,
                quality_weight=0.3,
                activation_contract="soft",
            ),
        )
        updated = update_successive_halving_state(
            units=[unit],
            posterior=posterior,
            eta=2,
            max_fold_budget=10,
            config=PosteriorGateConfig(),
        )
        assert updated[0].early_stop_state == "continue"
        assert updated[0].allocated_fold_budget == 3

    def test_continue_when_lcb_not_positive_despite_high_prob(self) -> None:
        unit = L1VerificationUnit(
            unit_id="u1",
            recipe_id="r1",
            timeframe="4h",
            scope_symbols=("BTCUSDT",),
            prior_mu_bps=0.0,
            prior_sigma_bps=10.0,
            allocated_fold_budget=3,
            early_stop_state="pending",
        )
        posterior = (
            L1PosteriorEvidence(
                symbol="BTCUSDT",
                recipe_id="r1",
                family="trend_ma",
                timeframe="4h",
                activation_context="pooled",
                posterior_mu_bps=8.0,
                posterior_sigma_bps=5.0,
                prob_mu_gt_cost=0.85,
                lcb_net_bps=-1.0,
                q_value=0.05,
                fold_pass_ratio=0.8,
                regime_stability=0.6,
                quality_weight=0.7,
                activation_contract="observe",
            ),
        )
        updated = update_successive_halving_state(
            units=[unit],
            posterior=posterior,
            eta=2,
            max_fold_budget=10,
            config=PosteriorGateConfig(),
        )
        assert updated[0].early_stop_state == "continue"
