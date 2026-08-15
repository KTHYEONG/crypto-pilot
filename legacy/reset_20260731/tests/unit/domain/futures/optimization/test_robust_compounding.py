from __future__ import annotations

from src.domain.futures.optimization.robust_compounding import (
    CandidateArtifactParityError,
    Layer2CandidateArtifact,
    WindowCompoundingMetrics,
    build_robustness_windows,
    compute_robust_compounding_score,
    evaluate_l2_candidate_artifact,
    validate_candidate_artifact_parity,
)
import pytest


def test_compute_robust_compounding_score_smoke() -> None:
    value = compute_robust_compounding_score(
        growth_lcbs=(0.01, 0.02, 0.03),
        annualized_cost_drags=(0.001, 0.001, 0.001),
    )
    assert value > 0.0


def test_build_robustness_windows_validates_bounds_and_embargo() -> None:
    windows = build_robustness_windows(
        l2_start_idx=0,
        holdout_start_idx=30,
        max_holding_bars=4,
    )
    assert len(windows) == 3
    assert windows[-1].oos_end == 30
    assert windows[0].embargo_bars == 4
    with pytest.raises(ValueError, match="n_windows"):
        build_robustness_windows(l2_start_idx=0, holdout_start_idx=30, max_holding_bars=1, n_windows=2)
    with pytest.raises(ValueError, match="insufficient"):
        build_robustness_windows(l2_start_idx=0, holdout_start_idx=4, max_holding_bars=1)


def test_robust_score_empty_and_missing_cost_paths() -> None:
    assert compute_robust_compounding_score(growth_lcbs=(), annualized_cost_drags=()) == float("-inf")
    assert compute_robust_compounding_score(growth_lcbs=(0.1,), annualized_cost_drags=()) == pytest.approx(0.15)


def test_candidate_artifact_blocks_missing_and_nonfinite_metrics() -> None:
    windows = build_robustness_windows(l2_start_idx=0, holdout_start_idx=30, max_holding_bars=1)
    missing = evaluate_l2_candidate_artifact(
        params={"x": 1, "y": 1.0, "z": "a", "flag": True},
        ctx=None,
        robustness_windows=windows[:2],
        data_fingerprint="d",
        handoff_fingerprint="h",
        routing_hash="r",
        window_plan_hash="w",
    )
    assert missing.blocker_reason == "insufficient_robustness_windows"
    metrics = tuple(WindowCompoundingMetrics(f"w{i}", 0.1, 0.1, 0.01, 0.1, 0.01, 50) for i in range(3))
    nonfinite = evaluate_l2_candidate_artifact(
        params={"x": 1}, ctx=None, robustness_windows=windows,
        data_fingerprint="d", handoff_fingerprint="h", routing_hash="r", window_plan_hash="w",
        window_metrics=(*metrics[:2], WindowCompoundingMetrics("w2", float("nan"), 0.1, 0.01, 0.1, 0.01, 50)),
    )
    assert nonfinite.blocker_reason == "nonfinite_candidate_metric"


@pytest.mark.parametrize(
    ("metrics", "blocker"),
    [
        ((WindowCompoundingMetrics("w", -0.1, 0.1, 0.01, -0.1, 0.01, 50),) * 3, "low_positive_window_ratio"),
        ((WindowCompoundingMetrics("w", -0.1, 0.1, 0.01, 0.1, 0.01, 50),) * 3, "worst_window_cagr_below_floor"),
        ((WindowCompoundingMetrics("w", 0.1, 0.4, 0.01, 0.1, 0.01, 50),) * 3, "mdd_budget_exceeded"),
        ((WindowCompoundingMetrics("w", 0.1, 0.1, 0.08, 0.1, 0.01, 50),) * 3, "cvar_budget_exceeded"),
        ((WindowCompoundingMetrics("w", 0.1, 0.1, 0.01, 0.1, 0.01, 1),) * 3, "minimum_trades_not_met"),
    ],
)
def test_candidate_admission_blockers(metrics: tuple[WindowCompoundingMetrics, ...], blocker: str) -> None:
    windows = build_robustness_windows(l2_start_idx=0, holdout_start_idx=30, max_holding_bars=1)
    artifact = evaluate_l2_candidate_artifact(
        params={"x": 1}, ctx=None, robustness_windows=windows,
        data_fingerprint="d", handoff_fingerprint="h", routing_hash="r", window_plan_hash="w",
        window_metrics=metrics,
    )
    assert artifact.blocker_reason == blocker


def test_parity_rejects_nonfinite_and_admitted_mismatch() -> None:
    from dataclasses import replace

    base = Layer2CandidateArtifact(
        candidate_hash="a", params={}, data_fingerprint="d", handoff_fingerprint="h",
        routing_hash="r", window_plan_hash="w", window_metrics=(), leverage_schedule=(),
        robust_score=float("nan"), median_growth_lcb=0.0, q10_growth_lcb=0.0,
        positive_window_ratio=0.0, worst_window_cagr=0.0,
        hard_constraint_names=(), hard_constraint_values=(), admitted=False, blocker_reason="x",
    )
    with pytest.raises(CandidateArtifactParityError):
        validate_candidate_artifact_parity(stored=base, replayed=replace(base, robust_score=0.1))
    with pytest.raises(CandidateArtifactParityError):
        validate_candidate_artifact_parity(stored=base, replayed=replace(base, admitted=True))
