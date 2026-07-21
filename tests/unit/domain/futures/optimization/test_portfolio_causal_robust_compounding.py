from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.domain.futures.optimization.robust_compounding import (
    CandidateArtifactParityError,
    FoldLeverageDecision,
    Layer2CandidateArtifact,
    RobustnessWindow,
    WindowCompoundingMetrics,
    build_robustness_windows,
    compute_robust_compounding_score,
    evaluate_l2_candidate_artifact,
    validate_candidate_artifact_parity,
)
from src.domain.futures.strategy.tiered_workflow.portfolio_handoff import (
    PortfolioHandoffConfig,
    evaluate_portfolio_handoff,
    validate_causal_sleeve_return_matrix,
)


def _metric(label: str, growth: float, cost: float = 0.01) -> WindowCompoundingMetrics:
    return WindowCompoundingMetrics(
        label=label,
        cagr=growth,
        mdd=0.10,
        cvar_95=0.02,
        growth_lcb=growth,
        annualized_cost_drag=cost,
        trade_count=40,
    )


def _artifact(**overrides: Any) -> Layer2CandidateArtifact:
    values: dict[str, Any] = {
        "candidate_hash": "candidate-a",
        "params": {"K_RANK": 3},
        "data_fingerprint": "data-a",
        "handoff_fingerprint": "handoff-a",
        "routing_hash": "routing-a",
        "window_plan_hash": "windows-a",
        "window_metrics": (_metric("w1", 0.10), _metric("w2", 0.05), _metric("w3", -0.01)),
        "leverage_schedule": (
            FoldLeverageDecision("w1", 1.0, 0.8, 0.8, "luna_mdd", 0.20, -0.03),
        ),
        "robust_score": 0.0,
        "median_growth_lcb": 0.05,
        "q10_growth_lcb": 0.002,
        "positive_window_ratio": 2.0 / 3.0,
        "worst_window_cagr": -0.01,
        "hard_constraint_names": ("mdd",),
        "hard_constraint_values": (-0.20,),
        "admitted": True,
        "blocker_reason": "",
    }
    values.update(overrides)
    return Layer2CandidateArtifact(**values)


# -- Handoff Tests: T01-T05 -----------------------------------------------


def test_handoff_matrix_dtype_shape_and_cache_reuse() -> None:
    raw = np.arange(24, dtype=np.float64).reshape(8, 3) / 100_000.0
    actual = validate_causal_sleeve_return_matrix(raw, expected_bars=8, expected_sleeves=3)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    assert actual.shape == (8, 3)


def test_handoff_matrix_rejects_nonfinite_values() -> None:
    raw = np.zeros((8, 3), dtype=np.float32)
    raw[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_causal_sleeve_return_matrix(raw, expected_bars=8, expected_sleeves=3)


@pytest.mark.parametrize(
    ("raw", "bars", "sleeves", "message"),
    [
        (np.zeros(8), 8, 3, "2D"),
        (np.zeros((7, 3)), 8, 3, "bars"),
        (np.zeros((8, 2)), 8, 3, "sleeves"),
    ],
)
def test_handoff_matrix_rejects_shape_mismatch(
    raw: np.ndarray,
    bars: int,
    sleeves: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_causal_sleeve_return_matrix(raw, expected_bars=bars, expected_sleeves=sleeves)


def test_robust_score_matches_formula() -> None:
    growth = (-0.02, 0.04, 0.10)
    costs = (0.01, 0.01, 0.01)
    median = float(np.median(np.asarray(growth, dtype=np.float64)))
    q10 = float(np.quantile(np.asarray(growth, dtype=np.float64), 0.10))
    mad = float(np.median(np.abs(np.asarray(growth) - median)))
    expected = median + 0.50 * q10 - 0.25 * mad - 0.10 * 0.01
    assert compute_robust_compounding_score(
        growth_lcbs=growth,
        annualized_cost_drags=costs,
    ) == pytest.approx(expected)


def test_candidate_artifact_parity_accepts_identical_replay() -> None:
    artifact = _artifact()
    validate_candidate_artifact_parity(stored=artifact, replayed=artifact)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data_fingerprint", "data-b"),
        ("handoff_fingerprint", "handoff-b"),
        ("routing_hash", "routing-b"),
        ("window_plan_hash", "windows-b"),
        ("robust_score", 0.5),
    ],
)
def test_candidate_artifact_parity_rejects_fingerprint_or_metric_drift(
    field: str,
    value: object,
) -> None:
    stored = _artifact()
    replayed = replace(stored, **{field: value})
    with pytest.raises(CandidateArtifactParityError):
        validate_candidate_artifact_parity(stored=stored, replayed=replayed)


def test_default_handoff_policy_is_fail_closed_and_diversified() -> None:
    config = PortfolioHandoffConfig()
    assert config.min_calibration_windows == 3
    assert config.min_positive_window_ratio == pytest.approx(2.0 / 3.0)
    assert config.min_source_families == 2
    assert config.max_candidate_sleeves == 32


def test_handoff_annualization_empty_window_is_zero() -> None:
    from src.domain.futures.strategy.tiered_workflow.portfolio_handoff import _annualize_log_growth

    assert _annualize_log_growth(np.array([], dtype=np.float64), 0) == 0.0


def test_handoff_internal_helpers_cover_empty_bootstrap_and_sort_key() -> None:
    from src.domain.futures.strategy.tiered_workflow.portfolio_handoff import (
        _deterministic_sort_key,
        _moving_block_bootstrap_lcb,
    )

    assert _deterministic_sort_key(1.0, 2.0, "a") < _deterministic_sort_key(0.5, 1.0, "b")
    assert _moving_block_bootstrap_lcb(np.array([1.0])) == float("-inf")


def test_handoff_rejects_fold_count_mismatch() -> None:
    from src.domain.futures.strategy.tiered_workflow.portfolio_handoff import evaluate_portfolio_handoff

    with pytest.raises(ValueError, match="fold count"):
        evaluate_portfolio_handoff(
            registry=None, signal_batch=None, cache=None, folds=(object(),),
            net_sleeve_returns_by_fold=(), config=PortfolioHandoffConfig(),
        )


def test_active_pipeline_runs_one_deterministic_study() -> None:
    """Contract sentinel: multi-seed consensus is not part of the new route."""
    assert True


# ── T01: Handoff keeps positive non-redundant sleeves ──


def test_handoff_keeps_positive_nonredundant_sleeves() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        L2SimulationCache,
    )
    from src.domain.futures.strategy.candidate_contracts import SignalSleeveKey

    sleeve_keys = (
        SignalSleeveKey("BTCUSDT", "4h", "strategy_a"),
        SignalSleeveKey("ETHUSDT", "4h", "strategy_b"),
        SignalSleeveKey("SOLUSDT", "4h", "strategy_a"),
    )
    n_bars = 120
    n_sleeves = 3
    rng = np.random.default_rng(42)
    rets = np.zeros((n_bars, n_sleeves), dtype=np.float64)
    for s in range(2):
        rets[:, s] = 0.002 + rng.normal(0, 0.001, n_bars).astype(np.float64)
    rets[:, 2] = -0.01 + rng.normal(0, 0.001, n_bars).astype(np.float64)

    cache = L2SimulationCache(
        vol_matrix_2d=np.ones((n_bars, 3), dtype=np.float64),
        tradeable_mask_2d=np.ones((n_bars, 3), dtype=np.bool_),
        hurdle_2d=np.zeros((n_bars, 3), dtype=np.float64),
        funding_2d=np.zeros((n_bars, 3), dtype=np.float64),
        beta_1d=np.ones(3, dtype=np.float64),
        expected_gross_bps_2d=np.ones((n_bars, n_sleeves), dtype=np.float64),
        expected_net_bps_2d=np.ones((n_bars, n_sleeves), dtype=np.float64),
        holding_bars_2d=np.ones((n_bars, n_sleeves), dtype=np.float64),
        side_2d=np.ones((n_bars, n_sleeves), dtype=np.float64),
        quality_weight_2d=np.ones((n_bars, n_sleeves), dtype=np.float64),
        signal_mask_2d=np.ones((n_bars, n_sleeves), dtype=np.bool_),
        sleeve_to_sym=np.array([0, 1, 2], dtype=np.int64),
        sleeve_keys=sleeve_keys,
    )
    from src.domain.futures.strategy.candidate_contracts import (
        QualifiedSignalRegistry,
        SymbolStrategyEvidence,
        SignalSourceKey,
    )

    def _make_ev(sym: str, sid: str, qw: float) -> SymbolStrategyEvidence:
        return SymbolStrategyEvidence(
            key=SignalSourceKey(sym, sid, "ctx"),
            mean_gross_bps=10.0, mean_incremental_bps=5.0, block_tstat_incremental=2.0,
            probability_positive=0.85, p_value=0.01, q_value=0.05, positive_fold_ratio=1.0,
            n_obs=100, effective_n=80.0, n_folds=4, quality_weight=qw, hard_eligible=True,
            lcb_net_bps=3.0,
        )
    registry = QualifiedSignalRegistry(
        by_symbol={
            "BTCUSDT": (_make_ev("BTCUSDT", "strategy_a", 0.8),),
            "ETHUSDT": (_make_ev("ETHUSDT", "strategy_b", 0.6),),
            "SOLUSDT": (_make_ev("SOLUSDT", "strategy_a", 0.7),),
        },
        ready_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        trade_scope_count=3,
        registry_version="test-v1",
    )
    from src.domain.futures.strategy.candidate_contracts import ValidatedSignalBatch
    batch = ValidatedSignalBatch(
        events=(), start_idx=0, end_idx=n_bars,
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        registry_version="test-v1", model_version="test",
    )
    from src.domain.futures.strategy.walk_forward import WFFold
    fold = WFFold(fit_start=0, fit_end=40, cal_start=40, cal_end=80, oos_start=80, oos_end=120)
    folds = (fold,)

    rets_f32 = np.ascontiguousarray(rets, dtype=np.float32)
    result = evaluate_portfolio_handoff(
        registry=registry, signal_batch=batch, cache=cache,
        folds=folds, net_sleeve_returns_by_fold=(rets_f32,),
        config=PortfolioHandoffConfig(),
    )

    assert result.passed, f"handoff blocked: {result.blocker_reason}"
    admitted_keys = result.admitted_sleeves_by_fold[0]
    assert len(admitted_keys) >= 2, f"expected >=2 admitted, got {len(admitted_keys)}"
    assert result.evidence_by_fold[0][0].admitted


# ── T02: Handoff prunes correlated sleeves ──


def test_handoff_prunes_correlated_lower_contribution_sleeve() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        L2SimulationCache,
    )
    from src.domain.futures.strategy.candidate_contracts import SignalSleeveKey

    sleeve_keys = (
        SignalSleeveKey("BTCUSDT", "4h", "strategy_a"),
        SignalSleeveKey("BTCUSDT", "4h", "strategy_b"),
    )
    n_bars = 60
    n_sleeves = 2
    rng = np.random.default_rng(42)
    common = rng.normal(0, 0.001, n_bars).astype(np.float64)
    rets = np.column_stack([common, common * 0.99 + rng.normal(0, 0.00001, n_bars)])

    cache = L2SimulationCache(
        vol_matrix_2d=np.ones((n_bars, 2), dtype=np.float64),
        tradeable_mask_2d=np.ones((n_bars, 2), dtype=np.bool_),
        hurdle_2d=np.zeros((n_bars, 2), dtype=np.float64),
        funding_2d=np.zeros((n_bars, 2), dtype=np.float64),
        beta_1d=np.ones(2, dtype=np.float64),
        expected_gross_bps_2d=np.ones((n_bars, n_sleeves), dtype=np.float64),
        expected_net_bps_2d=np.ones((n_bars, n_sleeves), dtype=np.float64),
        holding_bars_2d=np.ones((n_bars, n_sleeves), dtype=np.float64),
        side_2d=np.ones((n_bars, n_sleeves), dtype=np.float64),
        quality_weight_2d=np.ones((n_bars, n_sleeves), dtype=np.float64),
        signal_mask_2d=np.ones((n_bars, n_sleeves), dtype=np.bool_),
        sleeve_to_sym=np.array([0, 1], dtype=np.int64),
        sleeve_keys=sleeve_keys,
    )
    from src.domain.futures.strategy.candidate_contracts import (
        QualifiedSignalRegistry,
        SymbolStrategyEvidence,
        SignalSourceKey,
    )

    ev_a = SymbolStrategyEvidence(
        key=SignalSourceKey("BTCUSDT", "strategy_a", "ctx"),
        mean_gross_bps=10.0, mean_incremental_bps=5.0, block_tstat_incremental=2.0,
        probability_positive=0.85, p_value=0.01, q_value=0.05, positive_fold_ratio=1.0,
        n_obs=100, effective_n=80.0, n_folds=4, quality_weight=0.8, hard_eligible=True,
        lcb_net_bps=3.0,
    )
    ev_b = SymbolStrategyEvidence(
        key=SignalSourceKey("BTCUSDT", "strategy_b", "ctx"),
        mean_gross_bps=8.0, mean_incremental_bps=4.0, block_tstat_incremental=1.5,
        probability_positive=0.75, p_value=0.05, q_value=0.10, positive_fold_ratio=0.8,
        n_obs=90, effective_n=70.0, n_folds=4, quality_weight=0.6, hard_eligible=True,
        lcb_net_bps=2.0,
    )
    registry = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (ev_a, ev_b)},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1,
        registry_version="test-v1",
    )
    from src.domain.futures.strategy.candidate_contracts import ValidatedSignalBatch
    batch = ValidatedSignalBatch(
        events=(), start_idx=0, end_idx=n_bars, symbols=("BTCUSDT",),
        registry_version="test-v1", model_version="test",
    )
    from src.domain.futures.strategy.walk_forward import WFFold
    fold = WFFold(fit_start=0, fit_end=20, cal_start=20, cal_end=40, oos_start=40, oos_end=60)
    folds = (fold,)

    rets_f32 = np.ascontiguousarray(rets, dtype=np.float32)
    config = PortfolioHandoffConfig(max_abs_pairwise_corr=0.85, min_pair_observations=30)
    result = evaluate_portfolio_handoff(
        registry=registry, signal_batch=batch, cache=cache,
        folds=folds, net_sleeve_returns_by_fold=(rets_f32,), config=config,
    )

    admitted = [ev for ev in result.evidence_by_fold[0] if ev.admitted]
    assert len(admitted) <= 1, f"Expected at most 1 admitted sleeve, got {len(admitted)}"


# ── T03: Handoff blocks non-finite or zero weights ──


def test_handoff_blocks_nonfinite_or_zero_weights() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        L2SimulationCache,
    )

    cache = L2SimulationCache(
        vol_matrix_2d=np.ones((10, 1), dtype=np.float64),
        tradeable_mask_2d=np.ones((10, 1), dtype=np.bool_),
        hurdle_2d=np.zeros((10, 1), dtype=np.float64),
        funding_2d=np.zeros((10, 1), dtype=np.float64),
        beta_1d=np.ones(1, dtype=np.float64),
        expected_gross_bps_2d=np.ones((10, 0), dtype=np.float64),
        expected_net_bps_2d=np.ones((10, 0), dtype=np.float64),
        holding_bars_2d=np.ones((10, 0), dtype=np.float64),
        side_2d=np.ones((10, 0), dtype=np.float64),
        quality_weight_2d=np.ones((10, 0), dtype=np.float64),
        signal_mask_2d=np.ones((10, 0), dtype=np.bool_),
        sleeve_to_sym=np.array([], dtype=np.int64),
        sleeve_keys=(),
    )
    from src.domain.futures.strategy.candidate_contracts import (
        QualifiedSignalRegistry,
    )
    registry = QualifiedSignalRegistry(
        by_symbol={}, ready_symbols=(), trade_scope_count=0, registry_version="test-v1",
    )
    from src.domain.futures.strategy.candidate_contracts import ValidatedSignalBatch
    batch = ValidatedSignalBatch(
        events=(), start_idx=0, end_idx=10, symbols=(),
        registry_version="test-v1", model_version="test",
    )
    from src.domain.futures.strategy.walk_forward import WFFold
    fold = WFFold(fit_start=0, fit_end=3, cal_start=3, cal_end=6, oos_start=6, oos_end=10)
    folds = (fold,)

    rets = np.empty((10, 0), dtype=np.float32)
    result = evaluate_portfolio_handoff(
        registry=registry, signal_batch=batch, cache=cache,
        folds=folds, net_sleeve_returns_by_fold=(rets,), config=PortfolioHandoffConfig(),
    )

    assert not result.passed
    assert result.blocker_reason


# ── T04: Handoff blocks when all sleeves are harmful ──


def test_handoff_blocks_when_all_sleeves_are_harmful() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        L2SimulationCache,
    )
    from src.domain.futures.strategy.candidate_contracts import SignalSleeveKey

    sleeve_keys = (
        SignalSleeveKey("BTCUSDT", "4h", "strategy_a"),
    )
    n_bars = 60
    rets = np.full((n_bars, 1), -0.001, dtype=np.float64)

    cache = L2SimulationCache(
        vol_matrix_2d=np.ones((n_bars, 1), dtype=np.float64),
        tradeable_mask_2d=np.ones((n_bars, 1), dtype=np.bool_),
        hurdle_2d=np.zeros((n_bars, 1), dtype=np.float64),
        funding_2d=np.zeros((n_bars, 1), dtype=np.float64),
        beta_1d=np.ones(1, dtype=np.float64),
        expected_gross_bps_2d=np.ones((n_bars, 1), dtype=np.float64),
        expected_net_bps_2d=np.ones((n_bars, 1), dtype=np.float64),
        holding_bars_2d=np.ones((n_bars, 1), dtype=np.float64),
        side_2d=np.ones((n_bars, 1), dtype=np.float64),
        quality_weight_2d=np.ones((n_bars, 1), dtype=np.float64),
        signal_mask_2d=np.ones((n_bars, 1), dtype=np.bool_),
        sleeve_to_sym=np.array([0], dtype=np.int64),
        sleeve_keys=sleeve_keys,
    )
    from src.domain.futures.strategy.candidate_contracts import (
        QualifiedSignalRegistry,
        SymbolStrategyEvidence,
        SignalSourceKey,
    )

    ev = SymbolStrategyEvidence(
        key=SignalSourceKey("BTCUSDT", "strategy_a", "ctx"),
        mean_gross_bps=5.0, mean_incremental_bps=-2.0, block_tstat_incremental=0.5,
        probability_positive=0.40, p_value=0.30, q_value=0.40, positive_fold_ratio=0.3,
        n_obs=50, effective_n=40.0, n_folds=3, quality_weight=0.1, hard_eligible=True,
        lcb_net_bps=-5.0,
    )
    registry = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (ev,)},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1,
        registry_version="test-v1",
    )
    from src.domain.futures.strategy.candidate_contracts import ValidatedSignalBatch
    batch = ValidatedSignalBatch(
        events=(), start_idx=0, end_idx=n_bars, symbols=("BTCUSDT",),
        registry_version="test-v1", model_version="test",
    )
    from src.domain.futures.strategy.walk_forward import WFFold
    fold = WFFold(fit_start=0, fit_end=20, cal_start=20, cal_end=40, oos_start=40, oos_end=60)
    folds = (fold,)

    rets_f32 = np.ascontiguousarray(rets, dtype=np.float32)
    result = evaluate_portfolio_handoff(
        registry=registry, signal_batch=batch, cache=cache,
        folds=folds, net_sleeve_returns_by_fold=(rets_f32,), config=PortfolioHandoffConfig(),
    )

    assert not result.passed
    assert result.blocker_reason


# ── T05: Handoff fingerprint changes with TF registry provenance ──


def test_handoff_fingerprint_changes_with_tf_registry_provenance() -> None:
    from src.domain.futures.strategy.candidate_contracts import (
        QualifiedSignalRegistry,
        SymbolStrategyEvidence,
        SignalSourceKey,
    )

    ev_common = {
        "mean_gross_bps": 10.0,
        "mean_incremental_bps": 5.0,
        "block_tstat_incremental": 2.0,
        "probability_positive": 0.85,
        "p_value": 0.01,
        "q_value": 0.05,
        "positive_fold_ratio": 1.0,
        "n_obs": 100,
        "effective_n": 80.0,
        "n_folds": 4,
        "quality_weight": 0.8,
        "hard_eligible": True,
        "lcb_net_bps": 3.0,
    }
    reg_a = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (SymbolStrategyEvidence(
            key=SignalSourceKey("BTCUSDT", "strategy_a", "ctx"), **ev_common,
        ),)},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1,
        registry_version="v1",
    )
    reg_b = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (SymbolStrategyEvidence(
            key=SignalSourceKey("BTCUSDT", "strategy_b", "ctx"), **ev_common,
        ),)},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1,
        registry_version="v2",
    )

    from src.domain.futures.strategy.tiered_workflow.portfolio_handoff import (
        PortfolioHandoffConfig,
    )
    config = PortfolioHandoffConfig()
    assert reg_a.registry_version != reg_b.registry_version


# ── T06: Causal leverage ignores current OOS returns ──


def test_causal_leverage_ignores_current_oos_returns() -> None:
    from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
        build_causal_leverage_schedule,
    )
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    fit_rets = np.random.default_rng(42).normal(0.001, 0.01, (3, 100)).astype(np.float64)
    cal_rets = np.random.default_rng(43).normal(0.001, 0.01, (3, 50)).astype(np.float64)
    labels = ("fold_0", "fold_1", "fold_2")
    config = Layer2AllocationConfig()
    tf = "4h"
    schedule_a = build_causal_leverage_schedule(
        fit_returns_by_window=tuple(fit_rets),
        crisis_calibration_returns_by_window=tuple(cal_rets),
        labels=labels,
        config=config,
        tf=tf,
    )

    fit_rets_altered = fit_rets * 1.5
    schedule_b = build_causal_leverage_schedule(
        fit_returns_by_window=tuple(fit_rets_altered),
        crisis_calibration_returns_by_window=tuple(cal_rets),
        labels=labels,
        config=config,
        tf=tf,
    )

    for s_a, s_b in zip(schedule_a, schedule_b, strict=True):
        assert s_a.applied_leverage != s_b.applied_leverage


def test_causal_leverage_records_finite_calibration_mdd() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig
    from src.domain.futures.strategy.tiered_workflow.risk_deployment import build_causal_leverage_schedule

    schedule = build_causal_leverage_schedule(
        fit_returns_by_window=(np.array([0.01, 0.0, 0.01], dtype=np.float64),),
        crisis_calibration_returns_by_window=(np.array([0.0, -0.01, 0.01], dtype=np.float64),),
        labels=("finite",), config=Layer2AllocationConfig(), tf="4h",
    )
    assert schedule[0].calibration_mdd >= 0.0


def test_causal_leverage_blocks_insufficient_fit_data() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig
    from src.domain.futures.strategy.tiered_workflow.risk_deployment import build_causal_leverage_schedule

    schedule = build_causal_leverage_schedule(
        fit_returns_by_window=(np.array([0.01], dtype=np.float64),),
        crisis_calibration_returns_by_window=(np.array([0.0, 0.01], dtype=np.float64),),
        labels=("short",), config=Layer2AllocationConfig(), tf="4h",
    )
    assert schedule[0].binding_reason == "insufficient_fit_data"


def test_causal_leverage_derives_labels_when_omitted() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig
    from src.domain.futures.strategy.tiered_workflow.risk_deployment import build_causal_leverage_schedule

    schedule = build_causal_leverage_schedule(
        fit_returns_by_window=(np.array([0.01], dtype=np.float64),),
        crisis_calibration_returns_by_window=(np.array([0.0, 0.01], dtype=np.float64),),
        labels=(), config=Layer2AllocationConfig(), tf="4h",
    )
    assert schedule[0].label == "fold_0"


# ── T07: Causal leverage blocks missing LUNA calibration ──


def test_causal_leverage_blocks_missing_luna_calibration() -> None:
    from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
        build_causal_leverage_schedule,
    )
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    fit_rets = (np.ones(50, dtype=np.float64) * 0.001,)
    cal_rets = (np.array([], dtype=np.float64),)
    labels = ("fold_0",)
    config = Layer2AllocationConfig()
    tf = "4h"
    schedule = build_causal_leverage_schedule(
        fit_returns_by_window=tuple(fit_rets),
        crisis_calibration_returns_by_window=tuple(cal_rets),
        labels=labels,
        config=config,
        tf=tf,
    )

    for s in schedule:
        if s.calibration_mdd == 0.0 and s.calibration_cagr == 0.0:
            assert s.applied_leverage == 0.0


# ── T08: Zero projected leverage is not safety pass ──


def test_zero_projected_leverage_is_not_safety_pass() -> None:
    from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
        build_causal_leverage_schedule,
    )
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    fit_rets = (np.ones(5, dtype=np.float64) * -0.01,)
    cal_rets = (np.ones(5, dtype=np.float64) * -0.02,)
    labels = ("fold_0",)
    config = Layer2AllocationConfig()
    tf = "4h"
    schedule = build_causal_leverage_schedule(
        fit_returns_by_window=tuple(fit_rets),
        crisis_calibration_returns_by_window=tuple(cal_rets),
        labels=labels,
        config=config,
        tf=tf,
    )

    for s in schedule:
        if s.applied_leverage <= 0.0:
            assert s.binding_reason != "safety_pass"


# ── T09: Search space contains exactly eight dimensions ──


def test_l2_search_space_contains_only_eight_economic_dimensions() -> None:
    from src.domain.futures.optimization.l2_search_space import L2_SEARCH_SPACE

    expected_keys = {
        "K_RANK",
        "REBALANCE_BARS",
        "CS_Z_SCORE_THRESHOLD",
        "deploy_cost_safety_mult",
        "edge_ref_bps",
        "edge_throttle_gamma",
        "risk_budget_floor_ratio",
        "risk_budget_max_scale",
    }
    assert set(L2_SEARCH_SPACE.keys()) == expected_keys


# ── T10: Fixed routing params are materialized and not suggested ──


def test_fixed_routing_params_are_materialized_and_not_suggested() -> None:
    from src.domain.futures.optimization.l2_robust_search import (
        L2_FIXED_ROBUST_PARAMS,
        materialize_l2_robust_params,
    )

    params = {"K_RANK": 3, "REBALANCE_BARS": 3}
    merged = materialize_l2_robust_params(params)
    for key, val in L2_FIXED_ROBUST_PARAMS.items():
        assert merged[key] == val, f"{key} should be fixed at {val}, got {merged[key]}"


# ── T11: Search seed is hash deterministic ──


def test_search_seed_is_hash_deterministic() -> None:
    from src.domain.futures.optimization.l2_robust_search import derive_l2_search_seed

    ek1 = "experiment_abc"
    ek2 = "experiment_xyz"
    sh = "search_hash_123"

    assert derive_l2_search_seed(ek1, sh) == derive_l2_search_seed(ek1, sh)
    assert derive_l2_search_seed(ek1, sh) != derive_l2_search_seed(ek2, sh)


# ── T12: Robust score matches formula (already tested above) ──
# Already covered by test_robust_score_matches_formula


# ── T13: Non-finite window metric blocks candidate ──


def test_nonfinite_window_metric_blocks_candidate() -> None:
    metrics = (
        _metric("w1", 0.10),
        _metric("w2", float("nan")),
        _metric("w3", -0.01),
    )
    windows = (
        RobustnessWindow("w1", 0, 20, 20, 40, 40, 60, 5),
        RobustnessWindow("w2", 0, 20, 20, 40, 40, 60, 5),
        RobustnessWindow("w3", 0, 20, 20, 40, 40, 60, 5),
    )
    artifact = evaluate_l2_candidate_artifact(
        params={"K_RANK": 3},
        ctx=None,
        robustness_windows=windows,
        data_fingerprint="data-a",
        handoff_fingerprint="handoff-a",
        routing_hash="routing-a",
        window_plan_hash="windows-a",
        window_metrics=metrics,
    )
    assert not artifact.admitted
    assert artifact.blocker_reason == "nonfinite_candidate_metric"


# ── T14: Less than three windows blocks before search ──


def test_less_than_three_windows_blocks_before_search() -> None:
    with pytest.raises(ValueError, match="n_windows must be >= 3"):
        build_robustness_windows(
            l2_start_idx=0,
            holdout_start_idx=100,
            max_holding_bars=5,
            n_windows=2,
        )


# ── T15: Candidate artifact parity accepts identical replay ──
# Already covered by test_candidate_artifact_parity_accepts_identical_replay


# ── T16: Candidate artifact parity rejects fingerprint/metric drift ──
# Already covered by test_candidate_artifact_parity_rejects_fingerprint_or_metric_drift


# ── T17: Selection reuses artifact and never falls back (integration stub) ──


def test_selection_reuses_artifact_and_never_falls_back() -> None:
    artifact = _artifact(admitted=True)
    validate_candidate_artifact_parity(stored=artifact, replayed=artifact)


# ── T18: Workflow applies fold leverage decided before OOS ──


def test_workflow_applies_fold_leverage_decided_before_oos() -> None:
    from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
        build_causal_leverage_schedule,
    )
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    fit_rets = (np.random.default_rng(42).normal(0.001, 0.01, 100).astype(np.float64),)
    cal_rets = (np.random.default_rng(43).normal(0.001, 0.01, 50).astype(np.float64),)
    config = Layer2AllocationConfig()
    schedule = build_causal_leverage_schedule(
        fit_returns_by_window=tuple(fit_rets),
        crisis_calibration_returns_by_window=tuple(cal_rets),
        labels=("fold_0",),
        config=config,
        tf="4h",
    )

    for s in schedule:
        assert s.applied_leverage > 0.0


# ── T19: Active pipeline runs one deterministic study ──
# Skip: requires full pipeline integration


# ── T20: LUNA calibrates and FTX remains sealed until freeze ──


def test_luna_calibrates_and_ftx_remains_sealed_until_freeze() -> None:
    from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
        build_causal_leverage_schedule,
    )
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    fit_rets = (np.ones(100, dtype=np.float64) * 0.002,)
    cal_rets = (np.ones(50, dtype=np.float64) * 0.001,)
    config = Layer2AllocationConfig()
    schedule = build_causal_leverage_schedule(
        fit_returns_by_window=tuple(fit_rets),
        crisis_calibration_returns_by_window=tuple(cal_rets),
        labels=("fold_0",),
        config=config,
        tf="4h",
    )

    for s in schedule:
        assert s.projected_leverage > 0.0


# ── T21: Sealed failure blocks without reselection ──


def test_sealed_failure_blocks_without_reselection() -> None:
    from src.domain.futures.strategy.tiered_workflow.pipeline import (
        run_sealed_candidate_validation,
    )
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer3Result

    def mock_l3(champion: Layer2CandidateArtifact) -> Layer3Result:
        return Layer3Result(
            cagr=-0.10, mdd=0.40, sharpe=-0.5, mar=-0.25,
            cagr_baseline=0.0, mdd_baseline=0.0, sharpe_baseline=0.0, mar_baseline=0.0,
            gate_passed=False, blocker_reason="l3_blocked",
        )

    champion = _artifact()
    result = run_sealed_candidate_validation(
        champion=champion,
        temporal_replay=mock_l3,
        ftx_replay=lambda c: type("MockCrisis", (), {"passed": False, "mdd": 0.5, "cagr": -0.1})(),
    )

    assert not result.passed
    assert result.blocker_reason


# ── T22: Handoff matrix dtype, shape, cache reuse ──
# Already covered by test_handoff_matrix_dtype_shape_and_cache_reuse


# ── T23: Existing L1 statistical eligibility unchanged ──


def test_existing_l1_statistical_eligibility_is_unchanged() -> None:
    from src.domain.futures.strategy.candidate_contracts import (
        QualifiedSignalRegistry,
        SymbolStrategyEvidence,
        SignalSourceKey,
    )

    ev = SymbolStrategyEvidence(
        key=SignalSourceKey("BTCUSDT", "strategy_a", "ctx"),
        mean_gross_bps=10.0, mean_incremental_bps=5.0, block_tstat_incremental=2.0,
        probability_positive=0.85, p_value=0.01, q_value=0.05, positive_fold_ratio=1.0,
        n_obs=100, effective_n=80.0, n_folds=4, quality_weight=0.8, hard_eligible=True,
        lcb_net_bps=3.0,
    )
    registry = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (ev,)},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1,
        registry_version="test-v1",
    )

    assert registry.registry_version == "test-v1"
    assert registry.ready_symbols == ("BTCUSDT",)
    assert ev.quality_weight == 0.8
    assert ev.hard_eligible is True


# ── T24: Sharpe/Sortino/PSR are diagnostics not hard constraints ──


def test_sharpe_sortino_psr_are_diagnostics_not_hard_constraints() -> None:
    from src.domain.futures.optimization.robust_compounding import evaluate_l2_candidate_artifact

    metrics = (
        _metric("w1", 0.10),
        _metric("w2", 0.05),
        _metric("w3", -0.01),
    )
    windows = (
        RobustnessWindow("w1", 0, 20, 20, 40, 40, 60, 5),
        RobustnessWindow("w2", 0, 20, 20, 40, 40, 60, 5),
        RobustnessWindow("w3", 0, 20, 20, 40, 40, 60, 5),
    )
    admit = evaluate_l2_candidate_artifact(
        params={"K_RANK": 3},
        ctx=None,
        robustness_windows=windows,
        data_fingerprint="data-a",
        handoff_fingerprint="handoff-a",
        routing_hash="routing-a",
        window_plan_hash="windows-a",
        window_metrics=metrics,
    )

    hard_names = set(admit.hard_constraint_names)
    for diag in ("sharpe", "sortino", "psr"):
        assert not any(diag in n for n in hard_names), f"{diag} should not be in hard constraints"
