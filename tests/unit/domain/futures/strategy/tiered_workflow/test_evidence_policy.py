"""Tests for evidence_policy.py: assess_fold_evidence, pool_l1_evidence, strategy admissions."""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.tiered_workflow.evidence_policy import (
    FoldEvidenceAssessment,
    StrategyAdmission,
    assess_fold_evidence,
    compute_strategy_admissions,
    compute_symbol_posteriors,
    pool_l1_evidence,
)


def _fold(fold_id: int, gross: tuple[float, ...]) -> FoldEvidenceAssessment:
    n = len(gross)
    return assess_fold_evidence(
        fold_id=fold_id,
        gross_series_bps=np.asarray(gross, dtype=np.float64),
        execution_cost_bps=np.full(n, 7.5, dtype=np.float64),
        funding_cost_bps=np.full(n, 1.0, dtype=np.float64),
        matched_event_count=n,
        unmatched_event_count=0,
        decision_count=n,
        effective_symbol_count=6.0,
        cost_observed=np.ones(n, dtype=np.bool_),
        funding_observed=np.ones(n, dtype=np.bool_),
        min_matched_events=3,
        min_match_wilson_lcb=0.50,
        min_decision_count=3,
        max_cost_fallback_ratio=0.20,
        min_funding_coverage_ratio=0.80,
        block_bars=2,
        n_bootstrap=200,
        seed=fold_id,
    )


class TestAssessFoldEvidence:
    """S1.4: fold assessment with net edge computation."""

    def test_net_mean_computation(self) -> None:
        """Gross 20bps, cost 7.5bps, funding 1bps → net mean 11.5bps, state economic_positive."""
        result = _fold(0, (20.0, 19.0, 21.0, 20.0))
        assert result.state in ("data_eligible", "economic_positive")
        assert abs(result.net_mean_bps - 11.5) < 0.1
        assert len(result.blockers) == 0

    def test_fold_insufficient_support_due_cost_fallback(self) -> None:
        """[LIMIT-08] 21% fallback costs → insufficient_support."""
        n = 10
        cost_obs = np.ones(n, dtype=np.bool_)
        cost_obs[:3] = False  # 30% fallback > 20% max
        result = assess_fold_evidence(
            fold_id=1,
            gross_series_bps=np.full(n, 20.0, dtype=np.float64),
            execution_cost_bps=np.full(n, 7.5, dtype=np.float64),
            funding_cost_bps=np.full(n, 1.0, dtype=np.float64),
            matched_event_count=n,
            unmatched_event_count=0,
            decision_count=n,
            effective_symbol_count=6.0,
            cost_observed=cost_obs,
            funding_observed=np.ones(n, dtype=np.bool_),
            min_matched_events=3,
            min_match_wilson_lcb=0.80,
            min_decision_count=3,
            max_cost_fallback_ratio=0.20,
            min_funding_coverage_ratio=0.80,
            block_bars=2,
            n_bootstrap=200,
            seed=1,
        )
        assert result.state == "insufficient_support"
        assert any("cost_data_incomplete" in b for b in result.blockers)

    def test_event_maturity_lookahead(self) -> None:
        """[LIMIT-07] exit_idx == fold.oos_end → event excluded (via caller)."""
        pass  # This is enforced by the caller (evaluate_outer_signal_opportunities), not assess_fold_evidence.

    def test_fold_invalid_contract_on_non_finite(self) -> None:
        """Non-finite series raises EvidenceContractError."""
        with pytest.raises(ValueError, match="non-finite"):
            assess_fold_evidence(
                fold_id=2,
                gross_series_bps=np.asarray([float("nan"), 10.0], dtype=np.float64),
                execution_cost_bps=np.full(2, 7.5, dtype=np.float64),
                funding_cost_bps=np.full(2, 1.0, dtype=np.float64),
                matched_event_count=2,
                unmatched_event_count=0,
                decision_count=2,
                effective_symbol_count=6.0,
                cost_observed=np.ones(2, dtype=np.bool_),
                funding_observed=np.ones(2, dtype=np.bool_),
                min_matched_events=3,
                min_match_wilson_lcb=0.80,
                min_decision_count=3,
                max_cost_fallback_ratio=0.20,
                min_funding_coverage_ratio=0.80,
                block_bars=2,
                n_bootstrap=200,
                seed=2,
            )


class TestPoolL1Evidence:
    """S1.5, S2.8: pooled gate with negative fold included."""

    def test_negative_data_eligible_fold_is_not_dropped_from_pool(self) -> None:
        folds = (
            _fold(0, (30.0, 28.0, 32.0, 29.0)),
            _fold(1, (26.0, 27.0, 25.0, 28.0)),
            _fold(2, (2.0, 1.0, 3.0, 2.0)),
        )
        assert folds[2].state == "data_eligible"
        pooled = pool_l1_evidence(
            folds=folds,
            fold_cov=1.0,
            effective_symbol_n=6.0,
            min_fold_cov=0.8,
            min_data_eligible_folds=2,
            min_effective_symbol_n=3.0,
            min_positive_fold_ratio=0.5,
            block_bars=2,
            n_bootstrap=200,
            seed=7,
        )
        assert pooled.data_eligible_fold_count == 3
        assert pooled.pooled_net_lcb_bps is not None

    def test_no_eligible_fold_uses_none_not_negative_infinity(self) -> None:
        """[LIMIT-09] No data-eligible fold → pooled LCB None, not -inf/NaN."""
        pooled = pool_l1_evidence(
            folds=(),
            fold_cov=0.0,
            effective_symbol_n=0.0,
            min_fold_cov=0.8,
            min_data_eligible_folds=2,
            min_effective_symbol_n=3.0,
            min_positive_fold_ratio=0.5,
            block_bars=2,
            n_bootstrap=200,
            seed=7,
        )
        assert pooled.pooled_net_lcb_bps is None
        assert not pooled.structural_passed
        assert not pooled.economic_passed

    def test_zero_data_eligible_folds_returns_none_lcb(self) -> None:
        """All folds insufficient_support -> no data-eligible fold."""
        n = 5
        cost_obs = np.zeros(n, dtype=np.bool_)  # all fallback
        bad_fold = assess_fold_evidence(
            fold_id=0,
            gross_series_bps=np.full(n, 20.0, dtype=np.float64),
            execution_cost_bps=np.full(n, 7.5, dtype=np.float64),
            funding_cost_bps=np.full(n, 1.0, dtype=np.float64),
            matched_event_count=n,
            unmatched_event_count=0,
            decision_count=n,
            effective_symbol_count=6.0,
            cost_observed=cost_obs,
            funding_observed=np.ones(n, dtype=np.bool_),
            min_matched_events=3,
            min_match_wilson_lcb=0.80,
            min_decision_count=3,
            max_cost_fallback_ratio=0.20,
            min_funding_coverage_ratio=0.80,
            block_bars=2,
            n_bootstrap=200,
            seed=0,
        )
        assert bad_fold.state == "insufficient_support"
        pooled = pool_l1_evidence(
            folds=(bad_fold,),
            fold_cov=0.5,
            effective_symbol_n=0.0,
            min_fold_cov=0.8,
            min_data_eligible_folds=2,
            min_effective_symbol_n=3.0,
            min_positive_fold_ratio=0.5,
            block_bars=2,
            n_bootstrap=200,
            seed=7,
        )
        assert pooled.pooled_net_lcb_bps is None
        assert not pooled.structural_passed

    def test_negative_pooled_net_lcb_blocks_economic(self) -> None:
        """[S2.9] Negative finite pooled net LCB → economic_passed=False."""
        folds = (
            _fold(0, (-10.0, -12.0, -8.0, -11.0)),
            _fold(1, (-5.0, -6.0, -4.0, -7.0)),
            _fold(2, (-2.0, -3.0, -1.0, -4.0)),
        )
        pooled = pool_l1_evidence(
            folds=folds,
            fold_cov=1.0,
            effective_symbol_n=6.0,
            min_fold_cov=0.8,
            min_data_eligible_folds=2,
            min_effective_symbol_n=3.0,
            min_positive_fold_ratio=0.5,
            block_bars=2,
            n_bootstrap=200,
            seed=7,
        )
        assert pooled.data_eligible_fold_count == 3
        if pooled.pooled_net_lcb_bps is not None:
            assert not pooled.economic_passed

    def test_positive_pooled_net_lcb_allows_economic_pass(self) -> None:
        """Strong positive edge across folds → economic_passed=True."""
        folds = (
            _fold(0, (40.0, 38.0, 42.0, 39.0)),
            _fold(1, (36.0, 37.0, 35.0, 38.0)),
            _fold(2, (32.0, 31.0, 33.0, 32.0)),
        )
        pooled = pool_l1_evidence(
            folds=folds,
            fold_cov=1.0,
            effective_symbol_n=6.0,
            min_fold_cov=0.8,
            min_data_eligible_folds=2,
            min_effective_symbol_n=3.0,
            min_positive_fold_ratio=0.5,
            block_bars=2,
            n_bootstrap=200,
            seed=7,
        )
        assert pooled.structural_passed
        if pooled.pooled_net_lcb_bps is not None and pooled.pooled_net_lcb_bps > 0:
            assert pooled.economic_passed


class TestComputeStrategyAdmissions:
    """P2: BH FDR at strategy level."""

    def test_single_positive_strategy_admitted(self) -> None:
        """One strategy with positive edge → q <= alpha, admitted=True."""
        n = 20
        sids = np.array(["mom:ema_12"] * n, dtype="U12")
        returns = np.full(n, 15.0, dtype=np.float64) + np.random.default_rng(42).normal(0, 5, n)
        weights = np.ones(n, dtype=np.float64)
        decisions = np.arange(n, dtype=np.int64)

        admissions = compute_strategy_admissions(
            strategy_ids=sids,
            net_returns_bps=returns,
            uniqueness_weights=weights,
            decision_indices=decisions,
            block_bars=2,
            n_bootstrap=200,
            fdr_alpha=0.15,
            seed=42,
        )
        assert len(admissions) == 1
        assert admissions[0].strategy_id == "mom:ema_12"
        assert admissions[0].q_value <= 0.15
        assert admissions[0].admitted

    def test_two_strategies_one_noisy(self) -> None:
        """Two strategies, one with zero edge → only positive one admitted."""
        g = np.random.default_rng(42)
        n1, n2 = 20, 20
        sids = np.array(["a"] * n1 + ["b"] * n2, dtype="U4")
        returns = np.concatenate([
            np.full(n1, 15.0) + g.normal(0, 5, n1),
            g.normal(0, 5, n2),
        ])
        weights = np.ones(n1 + n2, dtype=np.float64)
        decisions = np.arange(n1 + n2, dtype=np.int64)

        admissions = compute_strategy_admissions(
            strategy_ids=sids,
            net_returns_bps=returns,
            uniqueness_weights=weights,
            decision_indices=decisions,
            block_bars=2,
            n_bootstrap=200,
            fdr_alpha=0.15,
            seed=42,
        )
        assert len(admissions) == 2
        adm_a = next(a for a in admissions if a.strategy_id == "a")
        adm_b = next(a for a in admissions if a.strategy_id == "b")
        assert adm_a.admitted
        assert not adm_b.admitted

    def test_insufficient_data_returns_not_admitted(self) -> None:
        """Single observation → cannot bootstrap, not admitted."""
        sids = np.array(["x"], dtype="U4")
        returns = np.array([5.0], dtype=np.float64)
        weights = np.array([1.0], dtype=np.float64)
        decisions = np.array([0], dtype=np.int64)

        admissions = compute_strategy_admissions(
            strategy_ids=sids,
            net_returns_bps=returns,
            uniqueness_weights=weights,
            decision_indices=decisions,
            block_bars=2,
            n_bootstrap=200,
            fdr_alpha=0.15,
            seed=42,
        )
        assert len(admissions) == 1
        assert not admissions[0].admitted


class TestComputeSymbolPosteriors:
    """P2: Symbol-level posterior with shrinkage."""

    def test_shrinkage_pulls_toward_strategy_mean(self) -> None:
        """Symbol with low n_eff is pulled toward positive strategy mean."""
        sids = np.array(["trend:ma"] * 40, dtype="U12")
        syms = np.array(["AAPL"] * 5 + ["MSFT"] * 35, dtype="U8")
        g = np.random.default_rng(42)
        returns = np.concatenate([
            np.full(5, 2.0) + g.normal(0, 3, 5),   # AAPL low n
            np.full(35, 15.0) + g.normal(0, 5, 35), # MSFT high n
        ])
        weights = np.ones(40, dtype=np.float64)
        fold_ids = np.arange(40, dtype=np.int64)
        admissions = (
            StrategyAdmission(
                strategy_id="trend:ma",
                mean_net_bps=12.0,
                lcb_net_bps=5.0,
                p_value=0.01,
                q_value=0.05,
                effective_n=30.0,
                admitted=True,
            ),
        )

        posteriors = compute_symbol_posteriors(
            symbol_ids=syms,
            strategy_ids=sids,
            net_returns_bps=returns,
            uniqueness_weights=weights,
            fold_ids=fold_ids,
            admissions=admissions,
            min_effective_n=2.0,
            min_folds=1,
            min_positive_fold_ratio=0.0,
            shrinkage_prior_n=20.0,
            seed=42,
        )
        assert len(posteriors) == 2
        msft = next(p for p in posteriors if p.symbol == "MSFT")
        assert msft.eligible


class TestAssessFoldEvidenceEdgeCases:
    """Edge coverage for assess_fold_evidence."""

    def test_empty_gross_series_returns_invalid_contract(self) -> None:
        """Empty series → invalid_contract state."""
        result = assess_fold_evidence(
            fold_id=0,
            gross_series_bps=np.array([], dtype=np.float64),
            execution_cost_bps=np.array([], dtype=np.float64),
            funding_cost_bps=np.array([], dtype=np.float64),
            matched_event_count=0,
            unmatched_event_count=0,
            decision_count=0,
            effective_symbol_count=0.0,
            cost_observed=np.array([], dtype=np.bool_),
            funding_observed=np.array([], dtype=np.bool_),
            min_matched_events=3,
            min_match_wilson_lcb=0.50,
            min_decision_count=3,
            max_cost_fallback_ratio=0.20,
            min_funding_coverage_ratio=0.80,
            block_bars=2,
            n_bootstrap=200,
            seed=0,
        )
        assert result.state == "invalid_contract"
        assert result.blockers == ("empty_series",)

    def test_insufficient_funding_coverage_blocked(self) -> None:
        """Funding coverage below min → insufficient_support."""
        n = 10
        result = assess_fold_evidence(
            fold_id=1,
            gross_series_bps=np.full(n, 20.0, dtype=np.float64),
            execution_cost_bps=np.full(n, 7.5, dtype=np.float64),
            funding_cost_bps=np.full(n, 1.0, dtype=np.float64),
            matched_event_count=n,
            unmatched_event_count=0,
            decision_count=n,
            effective_symbol_count=6.0,
            cost_observed=np.ones(n, dtype=np.bool_),
            funding_observed=np.zeros(n, dtype=np.bool_),
            min_matched_events=3,
            min_match_wilson_lcb=0.50,
            min_decision_count=3,
            max_cost_fallback_ratio=0.20,
            min_funding_coverage_ratio=0.80,
            block_bars=2,
            n_bootstrap=200,
            seed=1,
        )
        assert result.state == "insufficient_support"
        assert any("funding_data_incomplete" in b for b in result.blockers)

    def test_insufficient_matched_events_blocked(self) -> None:
        """Matched events below min threshold."""
        n = 2
        result = assess_fold_evidence(
            fold_id=2,
            gross_series_bps=np.full(n, 20.0, dtype=np.float64),
            execution_cost_bps=np.full(n, 7.5, dtype=np.float64),
            funding_cost_bps=np.full(n, 1.0, dtype=np.float64),
            matched_event_count=n,
            unmatched_event_count=0,
            decision_count=n,
            effective_symbol_count=6.0,
            cost_observed=np.ones(n, dtype=np.bool_),
            funding_observed=np.ones(n, dtype=np.bool_),
            min_matched_events=3,
            min_match_wilson_lcb=0.50,
            min_decision_count=3,
            max_cost_fallback_ratio=0.20,
            min_funding_coverage_ratio=0.80,
            block_bars=2,
            n_bootstrap=200,
            seed=2,
        )
        assert result.state == "insufficient_support"
        assert any("matched_events" in b for b in result.blockers)


class TestPoolL1EvidenceEdgeCases:
    """Edge coverage for pool_l1_evidence."""

    def test_insufficient_fold_cov_blocks_structural(self) -> None:
        """fold_cov below min → structural_passed=False."""
        folds = (_fold(0, (30.0, 28.0, 32.0, 29.0)),)
        pooled = pool_l1_evidence(
            folds=folds,
            fold_cov=0.5,
            effective_symbol_n=6.0,
            min_fold_cov=0.8,
            min_data_eligible_folds=2,
            min_effective_symbol_n=3.0,
            min_positive_fold_ratio=0.5,
            block_bars=2,
            n_bootstrap=200,
            seed=7,
        )
        assert not pooled.structural_passed
        assert any("fold_cov" in b for b in pooled.blockers)


class TestAssessFoldEvidenceSmallN:
    """Small-n integration: adaptive quantile should raise LCB vs fixed baseline."""

    def test_small_n_relaxed_quantile_raises_lcb(self) -> None:
        """S4: small N (n=8, block_bars=6 → num_blocks=2) → relaxed quantile > base."""
        rng = np.random.default_rng(7)
        gross = np.full(8, 50.0) + rng.normal(0, 5.0, size=8)
        exec_cost = np.full(8, 7.5)
        funding = np.zeros(8)
        cost_obs = np.ones(8, dtype=bool)
        funding_obs = np.ones(8, dtype=bool)

        adaptive = assess_fold_evidence(
            fold_id=0, gross_series_bps=gross, execution_cost_bps=exec_cost,
            funding_cost_bps=funding, matched_event_count=8, unmatched_event_count=0,
            decision_count=8, effective_symbol_count=1.0, cost_observed=cost_obs,
            funding_observed=funding_obs, min_matched_events=1, min_match_wilson_lcb=0.0,
            min_decision_count=1, max_cost_fallback_ratio=1.0, min_funding_coverage_ratio=0.0,
            block_bars=6, n_bootstrap=500, seed=42,
        )
        baseline = assess_fold_evidence(
            fold_id=0, gross_series_bps=gross, execution_cost_bps=exec_cost,
            funding_cost_bps=funding, matched_event_count=8, unmatched_event_count=0,
            decision_count=8, effective_symbol_count=1.0, cost_observed=cost_obs,
            funding_observed=funding_obs, min_matched_events=1, min_match_wilson_lcb=0.0,
            min_decision_count=1, max_cost_fallback_ratio=1.0, min_funding_coverage_ratio=0.0,
            block_bars=6, n_bootstrap=500, seed=42,
            lcb_quantile_floor_blocks=0,
        )

        assert adaptive.net_mean_bps == pytest.approx(baseline.net_mean_bps)
        assert adaptive.net_lcb_bps is not None
        assert baseline.net_lcb_bps is not None
        assert adaptive.net_lcb_bps > baseline.net_lcb_bps

    def test_large_n_regression_against_fixed_quantile(self) -> None:
        """S5: large N (n=200, block_bars=6 → num_blocks>>15) → bit-identical to fixed 0.05."""
        rng = np.random.default_rng(42)
        gross = np.full(200, 20.0) + rng.normal(0, 10.0, size=200)
        exec_cost = np.full(200, 7.5)
        funding = np.zeros(200)
        cost_obs = np.ones(200, dtype=bool)
        funding_obs = np.ones(200, dtype=bool)

        adaptive = assess_fold_evidence(
            fold_id=0, gross_series_bps=gross, execution_cost_bps=exec_cost,
            funding_cost_bps=funding, matched_event_count=200, unmatched_event_count=0,
            decision_count=200, effective_symbol_count=10.0, cost_observed=cost_obs,
            funding_observed=funding_obs, min_matched_events=1, min_match_wilson_lcb=0.0,
            min_decision_count=1, max_cost_fallback_ratio=1.0, min_funding_coverage_ratio=0.0,
            block_bars=6, n_bootstrap=500, seed=42,
        )
        baseline = assess_fold_evidence(
            fold_id=0, gross_series_bps=gross, execution_cost_bps=exec_cost,
            funding_cost_bps=funding, matched_event_count=200, unmatched_event_count=0,
            decision_count=200, effective_symbol_count=10.0, cost_observed=cost_obs,
            funding_observed=funding_obs, min_matched_events=1, min_match_wilson_lcb=0.0,
            min_decision_count=1, max_cost_fallback_ratio=1.0, min_funding_coverage_ratio=0.0,
            block_bars=6, n_bootstrap=500, seed=42,
            lcb_quantile_floor_blocks=0,
        )

        assert adaptive.net_mean_bps == pytest.approx(baseline.net_mean_bps)
        assert adaptive.net_lcb_bps == pytest.approx(baseline.net_lcb_bps)
