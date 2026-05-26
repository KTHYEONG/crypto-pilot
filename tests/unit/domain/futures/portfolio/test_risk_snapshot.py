"""RiskSnapshot contract tests (factor-lite)."""

from __future__ import annotations

import numpy as np

from src.domain.futures.portfolio.portfolio_constructor import RiskSnapshot


def test_risk_snapshot_contract_holds_covariance_and_beta() -> None:
    """RiskSnapshot stores covariance + beta metadata as provided."""
    cov = np.tile(np.eye(3, dtype=np.float64)[None, :, :], (4, 1, 1))
    beta = np.ones((4, 3), dtype=np.float64) * 0.25

    snap = RiskSnapshot(covariance_3d=cov, beta_2d=beta)

    assert snap.covariance_3d.shape == (4, 3, 3)
    assert snap.beta_2d is not None
    assert snap.beta_2d.shape == (4, 3)
    assert snap.residual_var_2d is None
