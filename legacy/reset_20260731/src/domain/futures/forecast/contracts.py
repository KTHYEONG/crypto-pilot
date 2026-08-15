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


@dataclass(slots=True, frozen=True)
class ExitPathRequest:
    """Causal, array-oriented request for intrabar exit labeling."""

    decision_idx: np.ndarray
    entry_idx: np.ndarray
    side: np.ndarray
    horizon_bars: np.ndarray
    stop_atr_mult: np.ndarray
    target_atr_mult: np.ndarray
    min_hold_bars: np.ndarray
    symbol_idx: np.ndarray
    open_2d: np.ndarray
    high_2d: np.ndarray
    low_2d: np.ndarray
    close_2d: np.ndarray
    atr_2d: np.ndarray
    cost_bps_2d: np.ndarray
    funding_2d: np.ndarray
    cost_floor_bps: np.ndarray
    hurdle_bps: np.ndarray
    taker_round_trip_bps: float


@dataclass(slots=True, frozen=True)
class ExitPathLabels:
    """Vectorized realized outcomes for each requested trade event."""

    gross_bps: np.ndarray
    cost_bps: np.ndarray
    funding_bps: np.ndarray
    edge_bps: np.ndarray
    exit_reason: np.ndarray
    exit_idx: np.ndarray
    mae_bps: np.ndarray
    mfe_bps: np.ndarray
    same_bar_collision: np.ndarray
