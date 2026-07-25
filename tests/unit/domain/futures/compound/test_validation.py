from __future__ import annotations

import numpy as np

from src.domain.futures.compound.config import L2BenchmarkConfig, L2GateConfig, L3ValidationConfig
from src.domain.futures.compound.contracts import (
    DeploymentVerdict,
    ExecutionLedger,
    L2BenchmarkSeries,
    L2Evaluation,
    L2GateVerdict,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.validation import (
    aggregate_returns_to_utc_days,
    build_causal_l2_benchmark,
    evaluate_l2_walk_forward,
    evaluate_l3_sealed_holdout,
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
            benchmark=benchmark, candidate_count=10,
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
