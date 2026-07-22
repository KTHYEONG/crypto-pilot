from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray


class AlphaLifecycle(StrEnum):
    SHADOW = "shadow"
    ACTIVE = "active"
    RETIRED = "retired"


class DeploymentVerdict(StrEnum):
    PROMOTE = "promote"
    SHADOW = "shadow"
    REJECT = "reject"


@dataclass(slots=True, frozen=True)
class AlphaDefinition:
    recipe_id: str
    family: str
    horizon_bars: int
    lookback_bars: int
    required_fields: tuple[str, ...]
    data_tier: Literal["core", "conditional"]
    causal_lag_bars: int = 1


@dataclass(slots=True, frozen=True)
class MarketFeatureCube:
    timestamps_ns: NDArray[np.int64]
    symbols: tuple[str, ...]
    fields_2d: dict[str, NDArray[np.float32] | NDArray[np.float64]]
    available_2d: dict[str, NDArray[np.bool_]]
    eligible_2d: NDArray[np.bool_]
    entry_block_2d: NDArray[np.bool_]
    capacity_usdt_2d: NDArray[np.float64]
    execution_cost_bps_2d: NDArray[np.float32]
    data_manifest_hash: str


@dataclass(slots=True, frozen=True)
class RawAlphaTape:
    timestamps_ns: NDArray[np.int64]
    symbols: tuple[str, ...]
    recipe_ids: tuple[str, ...]
    scores_3d: NDArray[np.float32]
    valid_3d: NDArray[np.bool_]
    horizon_bars_1d: NDArray[np.int16]


@dataclass(slots=True, frozen=True)
class AlphaForecastTape:
    timestamps_ns: NDArray[np.int64]
    symbols: tuple[str, ...]
    recipe_ids: tuple[str, ...]
    gross_mu_3d: NDArray[np.float32]
    forecast_var_3d: NDArray[np.float32]
    reliability_3d: NDArray[np.float32]
    valid_3d: NDArray[np.bool_]
    horizon_bars_1d: NDArray[np.int16]
    lifecycle_by_recipe: tuple[AlphaLifecycle, ...]
    model_version: str
    data_manifest_hash: str
    fold_manifest_hash: str


@dataclass(slots=True, frozen=True)
class CombinedForecast:
    mu_robust_1d: NDArray[np.float64]
    variance_1d: NDArray[np.float64]
    support_1d: NDArray[np.bool_]


@dataclass(slots=True, frozen=True)
class CausalAlphaFold:
    fold_id: int
    fit_start: int
    fit_end_exclusive: int
    oos_start: int
    oos_end_exclusive: int
    purge_bars: int
    embargo_bars: int


@dataclass(slots=True, frozen=True)
class AlphaLifecycleEvidence:
    recipe_id: str
    effective_n: float
    probability_net_positive: float
    consecutive_negative_versions: int
    data_valid: bool


@dataclass(slots=True, frozen=True)
class PortfolioDecision:
    decision_idx: int
    decision_time_ns: int
    target_weights_1d: NDArray[np.float64]
    gross_exposure: float
    net_exposure: float
    forecast_ann_vol: float
    risk_scale: float
    binding_constraints: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class RiskOverlayResult:
    target_weights_1d: NDArray[np.float64]
    risk_scale: float
    drawdown_scale: float
    volatility_scale: float
    cooldown_remaining: int
    hard_block_reason: str


@dataclass(slots=True, frozen=True)
class ExecutionLedger:
    timestamps_ns: NDArray[np.int64]
    net_returns_1d: NDArray[np.float64]
    equity_1d: NDArray[np.float64]
    target_weights_2d: NDArray[np.float32]
    fee_returns_1d: NDArray[np.float64]
    slippage_returns_1d: NDArray[np.float64]
    impact_returns_1d: NDArray[np.float64]
    funding_returns_1d: NDArray[np.float64]
    integrity_ok: bool
    integrity_reasons: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class L2Evaluation:
    annualized_log_growth: float
    growth_ci90: tuple[float, float]
    equity_multiple: float
    max_drawdown: float
    daily_cvar95: float
    annual_volatility: float
    turnover: float
    safe: bool
    integrity_ok: bool


@dataclass(slots=True, frozen=True)
class L3ValidationResult:
    verdict: DeploymentVerdict
    posterior_growth_probability: float
    holdout_days: int
    max_drawdown: float
    daily_cvar95: float
    reasons: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class SealedHoldoutManifest:
    holdout_id: str
    start_time_ns: int
    end_time_ns: int
    holdout_days: int
    model_version: str
    data_manifest_hash: str
    first_consumed_at_ns: int | None = None


@dataclass(slots=True, frozen=True)
class CompoundEngineResult:
    alpha_tape: AlphaForecastTape
    ledger: ExecutionLedger
    l2: L2Evaluation
    l3: L3ValidationResult


@dataclass(slots=True, frozen=True)
class CompoundPipelineOutcome:
    mode: Literal["legacy", "shadow", "active"]
    engine_result: CompoundEngineResult | None
    order_routed: bool
    reason: str
