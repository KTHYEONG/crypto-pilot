from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.tiered_workflow.l2_gate import (
    Layer2ConstraintVector,
    compute_ruin_constraint,
)
from src.domain.futures.strategy.tiered_workflow.portfolio_handoff import passes_marginal_growth_gate


def test_low_sharpe_is_diagnostic_when_growth_and_risk_pass() -> None:
    cv = Layer2ConstraintVector(
        deployment=-1.0,
        support_leak=0.0,
        policy_evidence=-1.0,
        active_blocks=-1.0,
        trades=-1.0,
        growth_lcb=-1.0,
        mdd=-1.0,
        cvar_95=-1.0,
        ruin=-1.0,
        crisis_mdd=-1.0,
        fold=-1.0,
        recent_fold=-1.0,
        recency_holdout=-1.0,
        friction=-1.0,
        crisis_measured=True,
    )
    assert cv.non_crisis_feasible()


def test_positive_l1_evidence_cannot_override_negative_marginal_growth() -> None:
    passed, reason = passes_marginal_growth_gate(
        marginal_growth_lcb=-0.01,
        positive_window_ratio=1.0,
        min_marginal_growth_lcb=0.0,
        min_positive_window_ratio=0.5,
    )
    assert not passed
    assert reason == "low_marginal_growth_lcb"


def test_ruin_constraint_blocks_equity_wipeout() -> None:
    value = compute_ruin_constraint(
        deployed_returns=np.array([0.01, -1.0]),
        min_equity_multiplier=1e-6,
    )
    assert value > 0.0


def test_ruin_constraint_accepts_finite_non_ruin_path() -> None:
    value = compute_ruin_constraint(
        deployed_returns=np.array([0.01, -0.20]),
        min_equity_multiplier=1e-6,
    )
    assert value <= 0.0
