"""Integration and comparison tests for compute_robust_score in optimization."""

from __future__ import annotations

import numpy as np

from src.domain.futures.optimization.evaluator import (
    compute_awf_robust_objective_score,
    compute_robust_score,
)


def test_v3_score_vs_legacy_cpcv_score() -> None:
    """Test that compute_robust_score is more conservative than the legacy score."""
    leg_log_tw = np.array([0.05, 0.04, -0.01, 0.06, 0.03, 0.02, 0.05, 0.04])
    worst_mdd = 0.12
    cvar_5 = 0.08
    excess_turnover = 0.15
    funding_drag = 0.05
    aum_impact_penalty = 0.02

    # Legacy score only penalizes downside semideviation.
    # Multiplier is typically lambda_down = 2.0.
    legacy = compute_awf_robust_objective_score(
        leg_log_tw=leg_log_tw,
        max_mdd_pct=worst_mdd * 100.0,  # Legacy takes MDD as percentage scale (0~100)
        lambda_mad=2.0,
    )

    # v3 score introduces 5 extra penalty terms: MDD, CVaR, Turnover, Funding, Capacity
    v3 = compute_robust_score(
        leg_log_tw=leg_log_tw,
        worst_mdd=worst_mdd,
        cvar_5=cvar_5,
        excess_turnover=excess_turnover,
        funding_drag=funding_drag,
        aum_impact_penalty=aum_impact_penalty,
    )

    # v3 score includes multiple subtraction penalties and should be lower.
    assert v3 < legacy


def test_v3_score_parameters_rigidity() -> None:
    """Verify that compute_robust_score does not allow dynamic injection of lambda."""
    import inspect

    sig = inspect.signature(compute_robust_score)
    parameters = list(sig.parameters.keys())

    # We must not have lambda_down, lambda_mdd etc as dynamic parameters.
    assert "lambda_down" not in parameters
    assert "lambda_mdd" not in parameters
    assert "lambda_cvar" not in parameters
    assert "lambda_turnover" not in parameters


def test_v3_score_evaluation_math() -> None:
    """Test mathematical accuracy of compute_robust_score for a known test vector."""
    # Mean of leg_log_tw = 0.05
    leg_log_tw = np.array([0.05] * 8)
    worst_mdd = 0.10
    cvar_5 = 0.05
    excess_turnover = 0.20
    funding_drag = 0.10
    aum_impact_penalty = 0.05

    # Downside semideviation should be 0.0 because there are no negative logs
    score = compute_robust_score(
        leg_log_tw=leg_log_tw,
        worst_mdd=worst_mdd,
        cvar_5=cvar_5,
        excess_turnover=excess_turnover,
        funding_drag=funding_drag,
        aum_impact_penalty=aum_impact_penalty,
    )

    # Expected: 0.05 - 0.5*0 - 1.0*0.10 - 0.3*0.05 - 0.2*0.20 - 0.5*0.10 - 0.4*0.05
    # = 0.05 - 0.0 - 0.10 - 0.015 - 0.04 - 0.05 - 0.02
    # = 0.05 - 0.225 = -0.175
    expected = 0.05 - 0.10 - 0.015 - 0.04 - 0.05 - 0.02
    assert abs(score - expected) < 1e-9
