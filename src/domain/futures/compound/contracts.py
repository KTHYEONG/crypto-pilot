from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Literal

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from src.domain.futures.universe.contracts import UniverseStateCube
from src.domain.futures.universe.models import UniverseSnapshot


class CausalityError(RuntimeError):
    ...


class InsufficientCoverageError(RuntimeError):
    ...


class AlphaLifecycle(StrEnum):
    SHADOW = "shadow"
    ACTIVE = "active"
    RETIRED = "retired"


class AlphaCandidateState(StrEnum):
    CORE_CANDIDATE = "core_candidate"
    CONDITIONAL_CANDIDATE = "conditional_candidate"
    SHADOW_RESEARCH = "shadow_research"
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
class MultiscaleAlphaDefinition:
    recipe_id: str
    family: str
    native_timeframe: str
    lookback_hours: tuple[int, ...]
    horizon_hours: int
    required_fields: tuple[str, ...]
    initial_state: AlphaCandidateState
    max_half_life_hours: float


@dataclass(slots=True, frozen=True)
class EdgeEvidence:
    recipe_id: str
    outer_folds: int
    positive_folds: int
    effective_days: float
    effective_events: int
    net_growth_lcb90: float
    doubled_cost_growth: float
    probability_positive: float
    sign_consistency: float
    fdr_q_value: float
    max_residual_correlation: float
    incremental_growth_lcb90: float
    capacity_feasible: bool
    admitted: bool
    reasons: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class AlphaEvent:
    recipe_id: str
    family: str
    native_timeframe: str
    symbol: str
    decision_time_ns: int
    first_executable_time_ns: int
    expiry_time_ns: int
    cumulative_net_mu: float
    half_life_hours: float
    alpha_rate_per_hour: float
    mean_edge_variance: float
    residual_variance: float
    reliability: float
    combination_weight: float
    model_version: str
    data_manifest_hash: str
    fold_manifest_hash: str


@dataclass(slots=True, frozen=True)
class AlphaEventTape:
    events: pa.Table
    recipe_definitions: tuple[MultiscaleAlphaDefinition, ...]
    evidence: tuple[EdgeEvidence, ...]
    active_recipe_ids: tuple[str, ...]
    model_version: str
    data_manifest_hash: str
    fold_manifest_hash: str


@dataclass(slots=True, frozen=True)
class ActiveForecastState:
    decision_time_ns: int
    symbols: tuple[str, ...]
    alpha_rate_1d: NDArray[np.float64]
    epistemic_variance_1d: NDArray[np.float64]
    active_event_ids: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class CausalFold:
    fold_id: int
    fit_start: int
    fit_end_exclusive: int
    calibration_start: int
    calibration_end_exclusive: int
    oos_start: int
    oos_end_exclusive: int
    purge_bars: int
    embargo_bars: int


@dataclass(slots=True, frozen=True)
class ForecastFrame:
    timestamps_ns: NDArray[np.int64]
    symbols: tuple[str, ...]
    recipe_id: str
    scores_2d: NDArray[np.float32]
    valid_2d: NDArray[np.bool_]


@dataclass(slots=True, frozen=True)
class ExecutionCostFrame:
    timestamps_ns: NDArray[np.int64]
    symbols: tuple[str, ...]
    execution_cost_bps: NDArray[np.float32]
    funding_cost_bps: NDArray[np.float32]


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
class MarketFeatureCube:
    timestamps_ns: NDArray[np.int64]
    symbols: tuple[str, ...]
    fields_2d: dict[str, NDArray[np.float32] | NDArray[np.float64]]
    available_2d: dict[str, NDArray[np.bool_]]
    eligible_2d: NDArray[np.bool_]
    entry_block_2d: NDArray[np.bool_]
    exit_required_2d: NDArray[np.bool_]
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
    mean_edge_var_3d: NDArray[np.float32]
    residual_var_3d: NDArray[np.float32]
    reliability_3d: NDArray[np.float32]
    estimated_3d: NDArray[np.bool_]
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
    strategy_spec_hash: str = ""
    universe_state_hash: str = ""
    first_consumed_at_ns: int | None = None


@dataclass(slots=True, frozen=True)
class CovariancePath:
    decision_timestamps_ns: NDArray[np.int64]
    covariance_3d: NDArray[np.float64]
    investable_2d: NDArray[np.bool_]


@dataclass(slots=True, frozen=True)
class LadderStageResult:
    stage_id: str
    oos_log_growth: float
    oos_growth_lcb90: float
    sharpe: float
    max_drawdown: float
    turnover_per_year: float
    growth_2x_cost: float
    status: str
    promoted: bool


@dataclass(slots=True, frozen=True)
class CompoundEngineResult:
    handoff: AlphaEventTape
    ledger: ExecutionLedger
    l2: L2Evaluation
    l3: L3ValidationResult


@dataclass(slots=True, frozen=True)
class StrategyDataCoverageEntry:
    dataset: str
    recipe_id: str
    ratio: float
    max_gap_bars: int
    readiness: str
    reason: str


@dataclass(slots=True, frozen=True)
class StrategyDataCoverage:
    entries: tuple[StrategyDataCoverageEntry, ...]
    all_ready: bool
    data_manifest_hash: str


@dataclass(slots=True, frozen=True)
class CompoundPipelineOutcome:
    mode: Literal["legacy", "shadow", "active"]
    engine_result: CompoundEngineResult | None
    order_routed: bool
    reason: str


@dataclass(slots=True, frozen=True)
class UniverseLedgerCoverage:
    requested_start: date
    requested_end: date
    covered_start: date
    covered_end: date
    timeframe: str
    complete: bool
    synced: bool


@dataclass(slots=True, frozen=True)
class CompoundUniverseResult:
    symbols: tuple[str, ...]
    state_cube: UniverseStateCube
    snapshots: tuple[UniverseSnapshot, ...]
    coverage: UniverseLedgerCoverage


@dataclass(slots=True, frozen=True)
class AllocationConstraints:
    gross_cap: float
    net_cap: float
    per_symbol_cap: NDArray[np.float64]
    beta_1d: NDArray[np.float64]
    beta_cap: float
    capacity_weight_1d: NDArray[np.float64]
    cost_bps_1d: NDArray[np.float64]
    entry_block_1d: NDArray[np.bool_]
    exit_required_1d: NDArray[np.bool_]


@dataclass(slots=True, frozen=True)
class TimeframeBarCube:
    timeframe: str
    timestamps_ns: NDArray[np.int64]
    symbols: tuple[str, ...]
    open_2d: NDArray[np.float32]
    high_2d: NDArray[np.float32]
    low_2d: NDArray[np.float32]
    close_2d: NDArray[np.float32]
    quote_volume_2d: NDArray[np.float32]
    complete_2d: NDArray[np.bool_]


@dataclass(slots=True, frozen=True)
class MultiTimeframeBars:
    decision_timestamps_ns: NDArray[np.int64]
    cubes: dict[str, TimeframeBarCube]
    aux_1h_fields: dict[str, NDArray[np.float32]]


@dataclass(slots=True, frozen=True)
class SignalDescriptor:
    signal_id: str
    family: str
    speed: str
    lookback_hours: int
    native_timeframe: str
    target_horizon_hours: int = 4

    def __post_init__(self) -> None:
        if self.native_timeframe == "4h" and self.target_horizon_hours % 4 != 0:
            raise ValueError(
                f"target_horizon_hours must be a multiple of 4 for 4h-native signals, "
                f"got {self.target_horizon_hours}"
            )


@dataclass(slots=True, frozen=True)
class CalibrationTarget:
    decision_timestamps_ns: NDArray[np.int64]
    y_2d: NDArray[np.float32]
    valid_2d: NDArray[np.bool_]


@dataclass(slots=True, frozen=True)
class SignalCalibration:
    signal_id: str
    beta_by_fold: tuple[float, ...]
    beta_se_by_fold: tuple[float, ...]
    n_obs_by_fold: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class SignalAdmissionEvidence:
    signal_id: str
    family: str
    oos_net_growth_lcb90: float
    oos_net_mean_2x_cost: float
    fold_sign_consistency: float
    p_value: float
    fdr_q_value: float
    admitted: bool
    reasons: tuple[str, ...]
    effective_sample_note: str = ""


@dataclass(slots=True, frozen=True)
class CalibratedForecastPanel:
    decision_timestamps_ns: NDArray[np.int64]
    symbols: tuple[str, ...]
    mu_2d: NDArray[np.float32]
    se_2d: NDArray[np.float32]
    family_mu_3d: NDArray[np.float32]
    family_ids: tuple[str, ...]
    admitted_signal_ids: tuple[str, ...]
    fold_manifest_hash: str


@dataclass(slots=True, frozen=True)
class RawSignalPanel:
    decision_timestamps_ns: NDArray[np.int64]
    symbols: tuple[str, ...]
    descriptors: tuple[SignalDescriptor, ...]
    z_3d: NDArray[np.float32]
    valid_3d: NDArray[np.bool_]
    sigma_2d: NDArray[np.float32]


__all__ = [
    "ActiveForecastState",
    "AllocationConstraints",
    "AlphaCandidateState",
    "AlphaDefinition",
    "AlphaEvent",
    "AlphaEventTape",
    "AlphaForecastTape",
    "AlphaLifecycle",
    "AlphaLifecycleEvidence",
    "CalibratedForecastPanel",
    "CalibrationTarget",
    "CausalAlphaFold",
    "CausalFold",
    "CausalityError",
    "CombinedForecast",
    "CompoundEngineResult",
    "CompoundPipelineOutcome",
    "CompoundUniverseResult",
    "CovariancePath",
    "DeploymentVerdict",
    "EdgeEvidence",
    "ExecutionCostFrame",
    "ExecutionLedger",
    "ForecastFrame",
    "InsufficientCoverageError",
    "L2Evaluation",
    "L3ValidationResult",
    "LadderStageResult",
    "MarketFeatureCube",
    "MultiTimeframeBars",
    "MultiscaleAlphaDefinition",
    "PortfolioDecision",
    "RawAlphaTape",
    "RawSignalPanel",
    "RiskOverlayResult",
    "SealedHoldoutManifest",
    "SignalAdmissionEvidence",
    "SignalCalibration",
    "SignalDescriptor",
    "StrategyDataCoverage",
    "StrategyDataCoverageEntry",
    "TimeframeBarCube",
    "UniverseLedgerCoverage",
]
