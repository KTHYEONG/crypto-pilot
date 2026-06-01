"""Typed forecast contracts for cost and risk layers."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True, frozen=True)
class CostForecast:
    """Typed cost forecast contract — execution friction per symbol per bar."""

    execution_cost_bps_2d: np.ndarray
    execution_cost_fraction_2d: np.ndarray
    uncertainty_bps_2d: np.ndarray
    capacity_notional_2d: np.ndarray | None
    source: str
    components: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RiskForecast:
    """Typed risk forecast contract — covariance, beta, residual variance, vol."""

    covariance_3d: np.ndarray
    beta_2d: np.ndarray | None
    residual_var_2d: np.ndarray | None
    forecast_vol_2d: np.ndarray
    beta_source: str  # raw_column | trailing_btc | trailing_equal_weight_market | unavailable
    source: str
