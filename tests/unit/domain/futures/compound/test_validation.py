from __future__ import annotations

import math

import numpy as np
import pytest

from src.domain.futures.compound.config import L2GateConfig, L3ValidationConfig
from src.domain.futures.compound.contracts import (
    DeploymentVerdict,
    ExecutionLedger,
    L2BenchmarkSeries,
    L2Evaluation,
    L2GateVerdict,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.multiplicity import TrialMultiplicity
from src.domain.futures.compound.validation import (
    annualized_compound_growth,
    blend_l1_prior_growth_probability,
    build_frozen_control_weights,
    evaluate_l2_walk_forward,
    evaluate_l3_sealed_holdout,
    validate_ledger_before_aggregation,
)


def _ledger(returns: np.ndarray, *, integrity_ok: bool = True) -> ExecutionLedger:
    equity = np.concatenate((np.array([1.0]), np.cumprod(1.0 + returns)))
    n = returns.size
    ns_per_4h = 4 * 3_600_000_000_000
    return ExecutionLedger(
        timestamps_ns=np.arange(n, dtype=np.int64) * ns_per_4h,
        net_returns_1d=returns.astype(np.float64),
        equity_1d=equity.astype(np.float64),
        target_weights_2d=np.zeros((n, 2), dtype=np.float32),
        fee_returns_1d=np.zeros(n, dtype=np.float64),
        slippage_returns_1d=np.zeros(n, dtype=np.float64),
        impact_returns_1d=np.zeros(n, dtype=np.float64),
        funding_returns_1d=np.zeros(n, dtype=np.float64),
        integrity_ok=integrity_ok,
        integrity_reasons=() if integrity_ok else ("execution_integrity",),
    )


def _manifest(holdout_days: int) -> SealedHoldoutManifest:
    return SealedHoldoutManifest(
        holdout_id=f"fixture-{holdout_days}d",
        start_time_ns=0,
        end_time_ns=holdout_days * 86_400_000_000_000,
        holdout_days=holdout_days,
        model_version="fixture-model-v1",
        data_manifest_hash="fixture-data-v1",
    )


class TestEvaluateL2WalkForward:
    def test_returns_l2_evaluation(self) -> None:
        n = 200
        returns = np.random.randn(n).astype(np.float64) * 0.001
        ledger = _ledger(returns)
        daily_timestamps = np.arange(n // 6, dtype=np.int64) * (6 * 4 * 3_600_000_000_000) + (6 * 4 * 3_600_000_000_000 - 1)
        benchmark = L2BenchmarkSeries(
            benchmark_id="test_benchmark",
            timestamps_ns=daily_timestamps[:n // 6],
            daily_returns_1d=np.zeros(n // 6, dtype=np.float64),
            causal_scale_1d=np.ones(n // 6, dtype=np.float64),
        )
        result = evaluate_l2_walk_forward(
            ledger=ledger, fold_ids_1d=np.zeros(n, dtype=np.int16),
            benchmark=benchmark, trial_multiplicity=TrialMultiplicity(10, 10.0, 1.0),
            config=L2GateConfig(), bootstrap_seed=42,
        )
        assert isinstance(result, L2Evaluation)
        assert np.isfinite(result.annualized_log_growth)


class TestEvaluateL3SealedHoldout:
    def test_when_safe_but_inconclusive_returns_shadow(self) -> None:
        prior = np.full(60, 0.0001, dtype=np.float64)
        holdout = _ledger(np.zeros(20, dtype=np.float64))
        manifest = _manifest(20)
        config = L3ValidationConfig(min_holdout_days=30)
        result = evaluate_l3_sealed_holdout(
            l2_prior_returns=prior,
            holdout_ledger=holdout,
            holdout_manifest=manifest,
            config=config,
        )
        assert result.verdict is DeploymentVerdict.SHADOW
        assert "insufficient_holdout_days" in result.reasons

    def test_when_growth_probability_high_returns_promote(self) -> None:
        prior = np.full(60, 0.0010, dtype=np.float64)
        holdout = _ledger(np.full(90, 0.0010, dtype=np.float64))
        manifest = _manifest(90)
        config = L3ValidationConfig(promote_probability=0.65)
        result = evaluate_l3_sealed_holdout(
            l2_prior_returns=prior,
            holdout_ledger=holdout,
            holdout_manifest=manifest,
            config=config,
        )
        assert result.verdict is DeploymentVerdict.PROMOTE
        assert result.posterior_growth_probability >= 0.65

    def test_when_execution_integrity_fails_returns_reject(self) -> None:
        prior = np.full(60, 0.0010, dtype=np.float64)
        holdout = _ledger(np.full(90, 0.0010, dtype=np.float64), integrity_ok=False)
        manifest = _manifest(90)
        config = L3ValidationConfig()
        result = evaluate_l3_sealed_holdout(
            l2_prior_returns=prior,
            holdout_ledger=holdout,
            holdout_manifest=manifest,
            config=config,
        )
        assert result.verdict is DeploymentVerdict.REJECT
        assert "execution_integrity" in result.reasons

    def test_when_liquidated_returns_reject(self) -> None:
        prior = np.full(60, 0.0010, dtype=np.float64)
        holdout = _ledger(np.full(90, -0.01, dtype=np.float64))
        manifest = _manifest(90)
        config = L3ValidationConfig(max_drawdown=0.20)
        result = evaluate_l3_sealed_holdout(
            l2_prior_returns=prior,
            holdout_ledger=holdout,
            holdout_manifest=manifest,
            config=config,
        )
        assert result.verdict is DeploymentVerdict.REJECT


def test_l3_when_safe_but_inconclusive_returns_shadow() -> None:
    prior = np.full(60, 0.0001, dtype=np.float64)
    holdout = _ledger(np.zeros(20, dtype=np.float64))
    manifest = _manifest(20)
    config = L3ValidationConfig(min_holdout_days=30)
    result = evaluate_l3_sealed_holdout(
        l2_prior_returns=prior, holdout_ledger=holdout, holdout_manifest=manifest, config=config,
    )
    assert result.verdict is DeploymentVerdict.SHADOW
    assert "insufficient_holdout_days" in result.reasons


def test_l3_when_growth_probability_high_returns_promote() -> None:
    prior = np.full(60, 0.0010, dtype=np.float64)
    holdout = _ledger(np.full(90, 0.0010, dtype=np.float64))
    manifest = _manifest(90)
    config = L3ValidationConfig(promote_probability=0.65)
    result = evaluate_l3_sealed_holdout(
        l2_prior_returns=prior, holdout_ledger=holdout, holdout_manifest=manifest, config=config,
    )
    assert result.verdict is DeploymentVerdict.PROMOTE
    assert result.posterior_growth_probability >= 0.65


def test_l3_when_liquidated_returns_reject() -> None:
    prior = np.full(60, 0.0010, dtype=np.float64)
    holdout = _ledger(np.full(90, -0.01, dtype=np.float64))
    manifest = _manifest(90)
    config = L3ValidationConfig(max_drawdown=0.20)
    result = evaluate_l3_sealed_holdout(
        l2_prior_returns=prior, holdout_ledger=holdout, holdout_manifest=manifest, config=config,
    )
    assert result.verdict is DeploymentVerdict.REJECT


class TestValidateLedgerBeforeAggregation:
    def test_valid_ledger_returns_empty(self) -> None:
        n = 100
        ledger = _ledger(np.random.randn(n).astype(np.float64) * 0.001)
        reasons = validate_ledger_before_aggregation(ledger)
        assert reasons == ()

    def test_net_return_le_minus_one(self) -> None:
        returns = np.array([0.0, -1.01, 0.0], dtype=np.float64)
        ledger = _ledger(returns)
        reasons = validate_ledger_before_aggregation(ledger)
        assert "net_return_le_minus_one" in reasons

    def test_non_finite_returns(self) -> None:
        returns = np.array([0.0, np.nan, 0.0], dtype=np.float64)
        ledger = _ledger(returns)
        reasons = validate_ledger_before_aggregation(ledger)
        assert "non_finite_returns" in reasons

    def test_integrity_ok_false_propagates_reasons(self) -> None:
        n = 100
        ledger = _ledger(np.random.randn(n).astype(np.float64) * 0.001, integrity_ok=False)
        reasons = validate_ledger_before_aggregation(ledger)
        assert "execution_integrity" in reasons

    def test_non_finite_weights_detected(self) -> None:
        n = 10
        returns = np.zeros(n, dtype=np.float64)
        weights = np.ones((n, 2), dtype=np.float32)
        weights[5, 0] = np.nan
        ledger = ExecutionLedger(
            timestamps_ns=np.arange(n, dtype=np.int64) * (4 * 3_600_000_000_000),
            net_returns_1d=returns,
            equity_1d=np.ones(n, dtype=np.float64),
            target_weights_2d=weights,
            fee_returns_1d=np.zeros(n, dtype=np.float64),
            slippage_returns_1d=np.zeros(n, dtype=np.float64),
            impact_returns_1d=np.zeros(n, dtype=np.float64),
            funding_returns_1d=np.zeros(n, dtype=np.float64),
            integrity_ok=True,
            integrity_reasons=(),
        )
        reasons = validate_ledger_before_aggregation(ledger)
        assert "non_finite_weights" in reasons


class TestL2NoEvidenceOnInvalidLedger:
    def test_net_return_le_minus_one_returns_no_evidence(self) -> None:
        n = 200
        returns = np.zeros(n, dtype=np.float64)
        returns[100] = -1.01
        ledger = _ledger(returns)
        daily_timestamps = np.arange(n // 6, dtype=np.int64) * (6 * 4 * 3_600_000_000_000) + (6 * 4 * 3_600_000_000_000 - 1)
        benchmark = L2BenchmarkSeries(
            benchmark_id="test",
            timestamps_ns=daily_timestamps[:n // 6],
            daily_returns_1d=np.zeros(n // 6, dtype=np.float64),
            causal_scale_1d=np.ones(n // 6, dtype=np.float64),
        )
        result = evaluate_l2_walk_forward(
            ledger=ledger, fold_ids_1d=np.zeros(n, dtype=np.int16),
            benchmark=benchmark, trial_multiplicity=TrialMultiplicity(5, 5.0, 1.0),
            config=L2GateConfig(), bootstrap_seed=42,
        )
        assert result.verdict == L2GateVerdict.NO_EVIDENCE
        assert result.integrity_ok is False
        assert "net_return_le_minus_one" in result.reasons
        assert np.isfinite(result.annualized_log_growth)



class TestL3HoldoutVeto:
    def test_low_holdout_probability_triggers_shadow(self) -> None:
        prior = np.full(60, 0.001, dtype=np.float64)
        holdout = _ledger(np.full(90, 0.0, dtype=np.float64))
        manifest = _manifest(90)
        config = L3ValidationConfig(
            min_holdout_days=30, min_holdout_growth_probability=0.50,
            promote_probability=0.90, reject_probability=0.10,
        )
        result = evaluate_l3_sealed_holdout(
            l2_prior_returns=prior,
            holdout_ledger=holdout,
            holdout_manifest=manifest,
            config=config,
        )
        assert result.verdict is DeploymentVerdict.SHADOW


    def test_evaluate_l3_sealed_holdout_negative_holdout_blocks_promote(self) -> None:
        self.test_low_holdout_probability_triggers_shadow()


class TestL34hBarPriorReject:
    def test_evaluate_l3_sealed_holdout_rejects_4h_bar_prior_length(self) -> None:
        prior = np.full(600, 0.001, dtype=np.float64)
        holdout = _ledger(np.full(90, 0.001, dtype=np.float64))
        manifest = _manifest(90)
        with pytest.raises(ValueError):
            evaluate_l3_sealed_holdout(
                l2_prior_returns=prior, holdout_ledger=holdout,
                holdout_manifest=manifest, config=L3ValidationConfig(),
            )



class TestBenchmarkAlignment:
    def test_identical_strategy_and_benchmark_yields_zero_excess(self) -> None:
        n = 366
        returns = np.full(n, 0.0002, dtype=np.float64)
        ledger = _ledger(returns)
        daily_ts = np.arange(n // 6, dtype=np.int64) * (6 * 4 * 3_600_000_000_000)
        n_daily = len(daily_ts)
        strategy_daily = np.full(n_daily, 0.0012, dtype=np.float64)
        benchmark_daily = strategy_daily.copy()
        benchmark = L2BenchmarkSeries(
            benchmark_id="test",
            timestamps_ns=daily_ts,
            daily_returns_1d=benchmark_daily,
            causal_scale_1d=np.ones(n_daily, dtype=np.float64),
        )
        result = evaluate_l2_walk_forward(
            ledger=ledger, fold_ids_1d=np.zeros(n, dtype=np.int16),
            benchmark=benchmark, trial_multiplicity=TrialMultiplicity(5, 5.0, 1.0),
            config=L2GateConfig(min_oos_days=10), bootstrap_seed=42,
        )
        assert result.excess_growth_lcb90 == pytest.approx(0.0, abs=1e-12)
        assert result.sharpe == pytest.approx(0.0, abs=1e-9)


    def test_evaluate_l2_walk_forward_strategy_equal_to_benchmark_yields_zero_excess(self) -> None:
        self.test_identical_strategy_and_benchmark_yields_zero_excess()


    def test_evaluate_l2_walk_forward_strategy_equal_to_benchmark_yields_zero_excess(self) -> None:
        self.test_identical_strategy_and_benchmark_yields_zero_excess()


class TestAnnualizedCompoundGrowth:
    def test_constant_daily_returns_matches_log_growth_formula(self) -> None:
        n = 365
        daily = np.full(n, 0.01, dtype=np.float64)
        result = annualized_compound_growth(daily, 365.25)
        expected = 365.25 * math.log1p(0.01)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_differs_from_arithmetic_annualization(self) -> None:
        n = 365
        daily = np.full(n, 0.01, dtype=np.float64)
        log_growth = annualized_compound_growth(daily, 365.25)
        arithmetic = 365.25 * 0.01
        assert log_growth != pytest.approx(arithmetic, abs=1e-10)

    def test_negative_one_raises(self) -> None:
        x = np.array([-1.0, 0.001], dtype=np.float64)
        with pytest.raises(ValueError, match="<= -1.0"):
            annualized_compound_growth(x, 365.25)


    def test_annualized_compound_growth_differs_from_arithmetic_mean(self) -> None:
        self.test_constant_daily_returns_matches_log_growth_formula()
        self.test_differs_from_arithmetic_annualization()


class TestBuildFrozenControlWeights:
    def test_freezes_at_given_index(self) -> None:
        w = np.random.default_rng(42).normal(0, 0.1, (100, 5)).astype(np.float64)
        frozen = build_frozen_control_weights(w, 50)
        assert frozen.shape == w.shape
        np.testing.assert_allclose(frozen[0], w[50])
        np.testing.assert_allclose(frozen[-1], w[50])

    def test_freeze_idx_out_of_range_raises(self) -> None:
        w = np.ones((10, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="out of range"):
            build_frozen_control_weights(w, 10)
        with pytest.raises(ValueError, match="out of range"):
            build_frozen_control_weights(w, -1)


def _ledger_with_rebalances(returns: np.ndarray) -> ExecutionLedger:
    n = returns.size
    ledger = _ledger(returns)
    weights = np.zeros((n, 2), dtype=np.float32)
    weights[1::2, 0] = 0.1
    weights[::2, 0] = 0.2
    return ExecutionLedger(
        timestamps_ns=ledger.timestamps_ns,
        net_returns_1d=ledger.net_returns_1d,
        equity_1d=ledger.equity_1d,
        target_weights_2d=weights,
        fee_returns_1d=ledger.fee_returns_1d,
        slippage_returns_1d=ledger.slippage_returns_1d,
        impact_returns_1d=ledger.impact_returns_1d,
        funding_returns_1d=ledger.funding_returns_1d,
        integrity_ok=True,
        integrity_reasons=(),
    )


class TestL2EvaluationNewFields:
    def test_evaluate_l2_walk_forward_sets_spa_and_block_fields(self) -> None:
        n = 2196
        returns = np.random.randn(n).astype(np.float64) * 0.001
        ledger = _ledger_with_rebalances(returns)
        daily_timestamps = np.arange(n // 6, dtype=np.int64) * (6 * 4 * 3_600_000_000_000)
        n_daily = len(daily_timestamps)
        benchmark = L2BenchmarkSeries(
            benchmark_id="test",
            timestamps_ns=daily_timestamps,
            daily_returns_1d=np.zeros(n_daily, dtype=np.float64),
            causal_scale_1d=np.ones(n_daily, dtype=np.float64),
        )
        config = L2GateConfig(min_oos_days=30, min_rebalances=1)
        result = evaluate_l2_walk_forward(
            ledger=ledger, fold_ids_1d=np.zeros(n, dtype=np.int16),
            benchmark=benchmark, trial_multiplicity=TrialMultiplicity(5, 5.0, 1.0),
            config=config, bootstrap_seed=42,
        )
        assert result.spa_pvalue == pytest.approx(1.0, abs=0.5)
        assert result.bootstrap_block_days > 0.0
        assert result.daily_strategy_returns_1d.size > 0
        assert result.daily_benchmark_returns_1d.size > 0
        assert result.daily_excess_returns_1d.size > 0

    def test_frozen_control_mismatch_does_not_crash(self) -> None:
        n = 2196
        returns = np.random.randn(n).astype(np.float64) * 0.001
        ledger = _ledger_with_rebalances(returns)
        daily_timestamps = np.arange(n // 6, dtype=np.int64) * (6 * 4 * 3_600_000_000_000)
        n_daily = len(daily_timestamps)
        benchmark = L2BenchmarkSeries(
            benchmark_id="test",
            timestamps_ns=daily_timestamps,
            daily_returns_1d=np.zeros(n_daily, dtype=np.float64),
            causal_scale_1d=np.ones(n_daily, dtype=np.float64),
        )
        bad_frozen = np.ones(5, dtype=np.float64)
        config = L2GateConfig(min_oos_days=30, min_rebalances=1)
        result = evaluate_l2_walk_forward(
            ledger=ledger, fold_ids_1d=np.zeros(n, dtype=np.int16),
            benchmark=benchmark, trial_multiplicity=TrialMultiplicity(5, 5.0, 1.0),
            config=config, bootstrap_seed=42,
            frozen_control_daily_1d=bad_frozen,
        )
        assert result.spa_pvalue == pytest.approx(1.0, abs=0.5)


    def test_evaluate_l2_walk_forward_wires_spa_and_block_length(self) -> None:
        self.test_evaluate_l2_walk_forward_sets_spa_and_block_fields()


def test_excess_degenerates_to_absolute_when_beta_zero() -> None:
    rng = np.random.default_rng(42)
    n = 100
    strat = rng.normal(0.0, 0.01, n).astype(np.float64)
    bench = rng.normal(0.0, 0.01, n).astype(np.float64)
    excess_zero = np.log1p(strat) - np.zeros(n) * np.log1p(bench)
    np.testing.assert_array_almost_equal(excess_zero, np.log1p(strat))
    excess_one = np.log1p(strat) - np.ones(n) * np.log1p(bench)
    np.testing.assert_array_almost_equal(excess_one, np.log1p(strat) - np.log1p(bench))


class TestBlendL1Prior:
    def test_blend_l1_prior_returns_l2_only_when_prior_empty(self) -> None:
        result = blend_l1_prior_growth_probability(
            np.array([], dtype=np.float64), 0.75, cap_days=90,
        )
        assert result == 0.75

    def test_blend_l1_prior_weight_capped(self) -> None:
        rng = np.random.default_rng(42)
        prior = rng.normal(0.0, 0.01, 200).astype(np.float64)
        result = blend_l1_prior_growth_probability(prior, 0.5, cap_days=90)
        assert 0.0 < result < 1.0


class TestL2GateSkillCategory:
    def test_l2_gate_skill_category_ignores_sharpe_probability(self) -> None:
        n = 2196
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, n).astype(np.float64)
        ledger = _ledger_with_rebalances(returns)
        daily_timestamps = np.arange(n // 6, dtype=np.int64) * (6 * 4 * 3_600_000_000_000)
        n_daily = len(daily_timestamps)
        benchmark = L2BenchmarkSeries(
            benchmark_id="test",
            timestamps_ns=daily_timestamps,
            daily_returns_1d=np.zeros(n_daily, dtype=np.float64),
            causal_scale_1d=np.ones(n_daily, dtype=np.float64),
        )
        config = L2GateConfig(
            min_oos_days=30, min_rebalances=1,
            min_deflated_sharpe_probability=0.001,
        )
        result = evaluate_l2_walk_forward(
            ledger=ledger, fold_ids_1d=np.zeros(n, dtype=np.int16),
            benchmark=benchmark, trial_multiplicity=TrialMultiplicity(5, 5.0, 1.0),
            config=config, bootstrap_seed=42,
        )
        for cat in result.category_results:
            if cat.category == "statistical_skill":
                sharpe_mentioned = any(
                    "sharpe_probability" in r for r in cat.reasons
                )
                assert not sharpe_mentioned, (
                    f"sharpe_probability should not gate skill: {cat.reasons}"
                )
                break
        else:
            pytest.fail("statistical_skill category not found")
