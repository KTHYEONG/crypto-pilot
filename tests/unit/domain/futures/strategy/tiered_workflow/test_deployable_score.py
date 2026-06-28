from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig
from src.domain.futures.strategy.tiered_workflow.deployable_score import (
    build_layer2_deployable_score,
    resolve_positive_block_delta_ratio,
    resolve_worst_fold_cagr,
    score_layer2_deployable_fallback,
)


def test_resolve_worst_fold_cagr_falls_back_to_fold_tuple() -> None:
    evaluation = SimpleNamespace(
        worst_fold_cagr=float("nan"),
        fold_deployed_cagrs=(0.10, None, -0.12, 0.05),
    )

    assert resolve_worst_fold_cagr(evaluation) == pytest.approx(-0.12)


def test_resolve_positive_block_delta_ratio_counts_positive_blocks() -> None:
    evaluation = SimpleNamespace(
        positive_block_delta_ratio=float("nan"),
        block_metrics=(
            SimpleNamespace(log_growth_hybrid=0.02, log_growth_baseline=0.01),
            SimpleNamespace(log_growth_hybrid=0.00, log_growth_baseline=0.01),
            SimpleNamespace(log_growth_hybrid=0.03, log_growth_baseline=0.01),
        ),
    )

    assert resolve_positive_block_delta_ratio(evaluation) == pytest.approx(2.0 / 3.0)


def test_build_layer2_deployable_score_uses_spec_formula() -> None:
    config = Layer2AllocationConfig()

    score = build_layer2_deployable_score(
        cagr=0.20,
        sortino=1.2,
        sharpe=1.5,
        mdd=0.10,
        fold_pass_ratio=0.67,
        worst_fold_cagr=-0.20,
        positive_block_delta_ratio=0.25,
        total_cost_bps=30.0,
        bucket_reliability_mean=0.7,
        entry_spike_penalty=0.15,
        config=config,
    )

    expected = (
        0.20
        + 0.10 * 1.2
        + 0.05 * 2.0
        - 0.50 * 0.20
        - 0.25 * (0.45 - 0.25)
        - 0.20 * 0.30
        - 0.15
    )
    assert score.score == pytest.approx(expected)


def test_score_layer2_deployable_fallback_prefers_existing_score() -> None:
    existing = build_layer2_deployable_score(
        cagr=0.1,
        sortino=1.0,
        sharpe=1.0,
        mdd=0.1,
        fold_pass_ratio=0.5,
        worst_fold_cagr=-0.01,
        positive_block_delta_ratio=0.5,
        total_cost_bps=10.0,
        bucket_reliability_mean=0.5,
        entry_spike_penalty=0.0,
        config=Layer2AllocationConfig(),
    )
    evaluation = SimpleNamespace(deployable_score=existing)

    assert score_layer2_deployable_fallback(
        evaluation,
        config=Layer2AllocationConfig(),
    ) is existing


def test_score_layer2_deployable_fallback_computes_missing_fields() -> None:
    evaluation = SimpleNamespace(
        cagr_hybrid=0.22,
        sortino_hybrid=1.3,
        sharpe_hybrid=1.1,
        mdd_hybrid=0.11,
        fold_pass_ratio=0.70,
        worst_fold_cagr=-0.05,
        positive_block_delta_ratio=0.50,
        total_cost_bps=12.0,
        bucket_reliability_mean=0.6,
        entry_spike_penalty=0.05,
        block_metrics=(MagicMock(),),
    )

    score = score_layer2_deployable_fallback(
        evaluation,
        config=Layer2AllocationConfig(),
    )

    assert score.cagr == pytest.approx(0.22)
    assert score.entry_spike_penalty == pytest.approx(0.05)

