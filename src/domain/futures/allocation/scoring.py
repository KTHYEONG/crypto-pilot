from __future__ import annotations

from typing import Any

import numpy as np

from src.domain.futures.allocation.contracts import (
    Layer2AllocationConfig,
    Layer2DeployableScore,
)


def resolve_worst_fold_cagr(evaluation: Any) -> float:
    worst_fold_cagr = float(getattr(evaluation, "worst_fold_cagr", 0.0))
    if np.isfinite(worst_fold_cagr):
        return worst_fold_cagr
    fold_deployed_cagrs = tuple(getattr(evaluation, "fold_deployed_cagrs", ()))
    finite_fold_cagrs = [
        float(value)
        for value in fold_deployed_cagrs
        if value is not None and np.isfinite(float(value))
    ]
    return min(finite_fold_cagrs) if finite_fold_cagrs else 0.0


def resolve_positive_block_delta_ratio(evaluation: Any) -> float:
    positive_block_delta_ratio = float(getattr(evaluation, "positive_block_delta_ratio", np.nan))
    if np.isfinite(positive_block_delta_ratio):
        return positive_block_delta_ratio
    block_metrics = tuple(getattr(evaluation, "block_metrics", ()))
    return (
        float(
            sum(
                1
                for metric in block_metrics
                if float(metric.log_growth_hybrid) > float(metric.log_growth_baseline)
            )
        )
        / float(len(block_metrics))
        if block_metrics
        else 0.0
    )


def build_layer2_deployable_score(
    *,
    cagr: float,
    sortino: float,
    sharpe: float,
    mdd: float,
    fold_pass_ratio: float,
    worst_fold_cagr: float,
    positive_block_delta_ratio: float,
    total_cost_bps: float,
    bucket_reliability_mean: float,
    entry_spike_penalty: float,
    config: Layer2AllocationConfig,
) -> Layer2DeployableScore:
    calmar = cagr / max(mdd, 1e-9) if np.isfinite(cagr) else 0.0
    cost_drag = max(total_cost_bps, 0.0) / 100.0
    score = float(cagr)
    score += 0.10 * min(float(sortino), 3.0)
    score += 0.05 * min(float(calmar), 3.0)
    score -= float(config.l2_worst_fold_cagr_penalty_weight) * max(0.0, -float(worst_fold_cagr))
    score -= float(config.l2_block_delta_penalty_weight) * max(
        0.0,
        float(config.l2_min_positive_block_delta_ratio) - float(positive_block_delta_ratio),
    )
    score -= 0.20 * float(cost_drag)
    score -= float(entry_spike_penalty)
    return Layer2DeployableScore(
        cagr=float(cagr),
        sortino=float(sortino),
        sharpe=float(sharpe),
        calmar=float(calmar),
        mdd=float(mdd),
        fold_pass_ratio=float(fold_pass_ratio),
        worst_fold_cagr=float(worst_fold_cagr),
        positive_block_delta_ratio=float(positive_block_delta_ratio),
        cost_drag=float(cost_drag),
        bucket_reliability_mean=float(bucket_reliability_mean),
        entry_spike_penalty=float(entry_spike_penalty),
        score=float(score),
    )


def score_layer2_deployable_fallback(
    evaluation: Any,
    *,
    config: Layer2AllocationConfig,
) -> Layer2DeployableScore:
    existing = getattr(evaluation, "deployable_score", None)
    if isinstance(existing, Layer2DeployableScore):
        return existing
    worst_fold_cagr = resolve_worst_fold_cagr(evaluation)
    positive_block_delta_ratio = resolve_positive_block_delta_ratio(evaluation)
    cagr = float(getattr(evaluation, "cagr_hybrid", 0.0))
    sortino = float(getattr(evaluation, "sortino_hybrid", 0.0))
    sharpe = float(
        getattr(
            evaluation,
            "sharpe_hybrid",
            getattr(evaluation, "sharpe_hac_hybrid", 0.0),
        )
    )
    mdd = max(float(getattr(evaluation, "mdd_hybrid", 0.0)), 0.0)
    fold_pass_ratio = float(getattr(evaluation, "fold_pass_ratio", 0.0))
    total_cost_bps = max(float(getattr(evaluation, "total_cost_bps", 0.0)), 0.0)
    bucket_reliability_mean = float(getattr(evaluation, "bucket_reliability_mean", 0.0))
    entry_spike_penalty = float(getattr(evaluation, "entry_spike_penalty", 0.0))
    return build_layer2_deployable_score(
        cagr=cagr,
        sortino=sortino,
        sharpe=sharpe,
        mdd=mdd,
        fold_pass_ratio=fold_pass_ratio,
        worst_fold_cagr=worst_fold_cagr,
        positive_block_delta_ratio=positive_block_delta_ratio,
        total_cost_bps=total_cost_bps,
        bucket_reliability_mean=bucket_reliability_mean,
        entry_spike_penalty=entry_spike_penalty,
        config=config,
    )
