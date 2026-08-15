from __future__ import annotations

import numpy as np

from src.domain.futures.forecast.contracts import CostForecast, RiskForecast
from src.domain.futures.forecast.diagnostics import LabelDiagnostics, cost_diagnostics, risk_diagnostics

_SHAPE = (20, 4)


def _make_cost(bps: float = 14.0) -> CostForecast:
    bps_2d = np.full(_SHAPE, bps)
    return CostForecast(
        execution_cost_bps_2d=bps_2d,
        execution_cost_fraction_2d=bps_2d / 10000.0,
        uncertainty_bps_2d=np.full(_SHAPE, bps * 0.1),
        capacity_notional_2d=None,
        source="test",
    )


def _make_risk() -> RiskForecast:
    t, n = _SHAPE
    cov = np.eye(n)[np.newaxis, :, :].repeat(t, axis=0) * 1e-4
    return RiskForecast(
        covariance_3d=cov,
        beta_2d=np.full(_SHAPE, 0.8),
        residual_var_2d=np.full(_SHAPE, 1e-4),
        forecast_vol_2d=np.full(_SHAPE, 0.02),
        beta_source="trailing_btc",
        source="test",
    )


def test_label_diagnostics_shape() -> None:
    arr = np.ones(_SHAPE)
    ld = LabelDiagnostics(arr, arr * 0.5, arr * 0.3)
    assert ld.cost_clearance_target.shape == _SHAPE


def test_cost_diagnostics_keys() -> None:
    diag = cost_diagnostics(_make_cost())
    assert "cost_mean_bps" in diag


def test_risk_diagnostics_keys() -> None:
    diag = risk_diagnostics(_make_risk())
    assert "cov_psd_fail_count" in diag
