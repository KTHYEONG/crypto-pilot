"""Diagnostic containers and reporting for cost and risk forecasts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.domain.futures.forecast.contracts import CostForecast, RiskForecast


@dataclass(frozen=True)
class LabelDiagnostics:
    """Cost-clearance diagnostics removed from LabelPanel training contract.

    Attributes:
        cost_clearance_target: Absolute EV minus round-trip cost [T, N].
        cost_clearance_target_long: Long-side cost clearance [T, N].
        cost_clearance_target_short: Short-side cost clearance [T, N].

    """

    cost_clearance_target: np.ndarray
    cost_clearance_target_long: np.ndarray
    cost_clearance_target_short: np.ndarray


def cost_diagnostics(cf: CostForecast) -> dict[str, float]:
    """Compute summary diagnostics for a CostForecast.

    Args:
        cf: Typed cost forecast.

    Returns:
        Dict of scalar diagnostic metrics.

    """
    bps = np.asarray(cf.execution_cost_bps_2d, dtype=np.float64).ravel()
    bps_fin = bps[np.isfinite(bps)]
    unc = np.asarray(cf.uncertainty_bps_2d, dtype=np.float64).ravel()
    unc_fin = unc[np.isfinite(unc)]
    return {
        "cost_mean_bps": float(np.nanmean(bps_fin)) if bps_fin.size else 0.0,
        "cost_p95_bps": float(np.nanpercentile(bps_fin, 95)) if bps_fin.size else 0.0,
        "cost_source": 0.0,  # categorical — inspect cf.source directly
        "uncertainty_mean_bps": float(np.nanmean(unc_fin)) if unc_fin.size else 0.0,
    }


def risk_diagnostics(rf: RiskForecast) -> dict[str, float]:
    """Compute summary diagnostics for a RiskForecast.

    Args:
        rf: Typed risk forecast.

    Returns:
        Dict of scalar diagnostic metrics.

    """
    diag: dict[str, float] = {}
    if rf.residual_var_2d is not None:
        rv = np.asarray(rf.residual_var_2d, dtype=np.float64).ravel()
        rv_fin = rv[np.isfinite(rv)]
        diag["residual_var_nan_ratio"] = float(np.sum(~np.isfinite(rv)) / max(rv.size, 1))
        diag["residual_var_mean"] = float(np.nanmean(rv_fin)) if rv_fin.size else 0.0
    if rf.beta_2d is not None:
        b = np.asarray(rf.beta_2d, dtype=np.float64).ravel()
        diag["beta_nonzero_ratio"] = float(np.count_nonzero(np.abs(b) > 1e-9) / max(b.size, 1))
    cov = np.asarray(rf.covariance_3d, dtype=np.float64)
    psd_fails = sum(
        1 for i in range(cov.shape[0]) if np.any(np.linalg.eigvalsh(cov[i]) < -1e-9)
    )
    diag["cov_psd_fail_count"] = float(psd_fails)
    return diag
