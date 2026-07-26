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

MAX_ABS_FUNDING_RATE: float = 0.05


class FundingDataIntegrityError(ValueError):
    """Funding partition schema or value integrity failure."""


class ClusteringAlgorithm(StrEnum):
    ROBUST_KMEANS = "robust_kmeans"
    HIERARCHICAL_WARD = "hierarchical_ward"
    DBSCAN = "dbscan"


@dataclass(slots=True, frozen=True)
class ClusterPanel:
    symbols: tuple[str, ...]
    cluster_labels: NDArray[np.int32]
    cluster_centroids: NDArray[np.float64]
    k_clusters: int

    def __post_init__(self) -> None:
        if self.cluster_labels.ndim != 1 or self.cluster_labels.shape[0] != len(self.symbols):
            raise ValueError("cluster_labels must be 1-D with length equal to symbols")
        if self.cluster_centroids.ndim != 2 or self.cluster_centroids.shape[0] != self.k_clusters:
            raise ValueError("cluster_centroids must be 2-D with k_clusters rows")


class L2GateVerdict(StrEnum):
    PASS = "pass"  # noqa: S105
    FAIL = "fail"
    NO_EVIDENCE = "no_evidence"


@dataclass(slots=True, frozen=True)
class L2CategoryResult:
    category: str
    passed: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.category:
            raise ValueError("category must be non-empty")
        if self.passed and self.reasons:
            raise ValueError("passed category must have empty reasons")
        if not self.passed and not self.reasons:
            raise ValueError("failed category must have at least one reason")


@dataclass(slots=True, frozen=True)
class CausalClusterFold:
    fold_id: int
    fit_end_exclusive_4h: int
    fit_end_time_ns: int
    panel: ClusterPanel
    member_hash: str

    def __post_init__(self) -> None:
        if self.fit_end_exclusive_4h <= 0:
            raise ValueError("fit_end_exclusive_4h must be > 0")


@dataclass(slots=True, frozen=True)
class L2BenchmarkSeries:
    benchmark_id: str
    timestamps_ns: NDArray[np.int64]
    daily_returns_1d: NDArray[np.float64]
    causal_scale_1d: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not self.benchmark_id:
            raise ValueError("benchmark_id must be non-empty")
        if self.timestamps_ns.ndim != 1 or self.daily_returns_1d.ndim != 1 or self.causal_scale_1d.ndim != 1:
            raise ValueError("arrays must be 1-D")
        if not (self.timestamps_ns.shape[0] == self.daily_returns_1d.shape[0] == self.causal_scale_1d.shape[0]):
            raise ValueError("arrays must have the same length")
        if len(np.unique(self.timestamps_ns)) != len(self.timestamps_ns):
            raise ValueError("timestamps_ns must be unique")
        if not np.all(np.isfinite(self.daily_returns_1d)):
            raise ValueError("daily_returns_1d must be finite")
        if not np.all(np.isfinite(self.causal_scale_1d)):
            raise ValueError("causal_scale_1d must be finite")


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
    verdict: L2GateVerdict
    benchmark_id: str
    annualized_log_growth: float
    cagr: float
    excess_growth_lcb90: float
    excess_growth_probability: float
    stressed_excess_growth_lcb90: float
    equity_multiple: float
    sharpe: float
    sharpe_probability: float
    deflated_sharpe_probability: float
    candidate_count: int
    calmar: float
    max_drawdown: float
    daily_cvar95: float
    annual_volatility: float
    annual_turnover: float
    cost_drag_ratio: float
    capacity_utilisation_p95: float
    active_days_ratio: float
    rebalance_count: int
    positive_outer_folds: int
    oos_days: int
    category_results: tuple[L2CategoryResult, ...]
    integrity_ok: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for metric in (
            self.annualized_log_growth, self.cagr, self.excess_growth_lcb90,
            self.excess_growth_probability, self.stressed_excess_growth_lcb90,
            self.sharpe, self.sharpe_probability, self.deflated_sharpe_probability,
            self.calmar, self.max_drawdown, self.daily_cvar95,
            self.annual_volatility, self.annual_turnover, self.cost_drag_ratio,
            self.capacity_utilisation_p95, self.active_days_ratio,
        ):
            if not np.isfinite(metric):
                raise ValueError(f"non-finite metric: {metric}")
        if self.verdict == L2GateVerdict.PASS and (not self.category_results or not all(r.passed for r in self.category_results)):
            raise ValueError("PASS verdict requires non-empty category results with all passed")
        if self.verdict != L2GateVerdict.PASS and self.category_results and all(r.passed for r in self.category_results):
            raise ValueError("non-PASS verdict requires at least one failed category")

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict.value,
            "benchmark_id": self.benchmark_id,
            "annualized_log_growth": {"value": self.annualized_log_growth, "unit": "log_returns/year"},
            "cagr": {"value": self.cagr, "unit": "fraction/year"},
            "excess_growth_lcb90": {"value": self.excess_growth_lcb90, "unit": "log_returns/year"},
            "excess_growth_probability": {"value": self.excess_growth_probability, "unit": "probability"},
            "stressed_excess_growth_lcb90": {"value": self.stressed_excess_growth_lcb90, "unit": "log_returns/year"},
            "equity_multiple": {"value": self.equity_multiple, "unit": "ratio"},
            "sharpe": {"value": self.sharpe, "unit": "annualized"},
            "sharpe_probability": {"value": self.sharpe_probability, "unit": "probability"},
            "deflated_sharpe_probability": {"value": self.deflated_sharpe_probability, "unit": "probability"},
            "candidate_count": {"value": self.candidate_count, "unit": "count"},
            "calmar": {"value": self.calmar, "unit": "ratio"},
            "max_drawdown": {"value": self.max_drawdown, "unit": "fraction"},
            "daily_cvar95": {"value": self.daily_cvar95, "unit": "fraction"},
            "annual_volatility": {"value": self.annual_volatility, "unit": "fraction/year"},
            "annual_turnover": {"value": self.annual_turnover, "unit": "turns/year"},
            "cost_drag_ratio": {"value": self.cost_drag_ratio, "unit": "fraction"},
            "capacity_utilisation_p95": {"value": self.capacity_utilisation_p95, "unit": "fraction"},
            "active_days_ratio": {"value": self.active_days_ratio, "unit": "fraction"},
            "rebalance_count": {"value": self.rebalance_count, "unit": "count"},
            "positive_outer_folds": {"value": self.positive_outer_folds, "unit": "count"},
            "oos_days": {"value": self.oos_days, "unit": "days"},
            "integrity_ok": self.integrity_ok,
            "category_results": [
                {"category": r.category, "passed": r.passed, "reasons": list(r.reasons)}
                for r in self.category_results
            ],
            "reasons": list(self.reasons),
        }


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
    archetype: str = ""
    economic_hypothesis: str = ""
    candidate_version: str = "v1"

    def __post_init__(self) -> None:
        if self.native_timeframe == "4h" and self.target_horizon_hours % 4 != 0:
            raise ValueError(
                f"target_horizon_hours must be a multiple of 4 for 4h-native signals, "
                f"got {self.target_horizon_hours}"
            )


class ExitPolicyKind(StrEnum):
    TIME = "time"
    ASYMMETRIC_ATR = "asymmetric_atr"
    ATR_TRAILING = "atr_trailing"


@dataclass(slots=True, frozen=True)
class ExitPolicySpec:
    policy_id: str
    kind: ExitPolicyKind
    stop_atr_mult: float | None
    target_atr_mult: float | None
    trail_atr_mult: float | None
    trail_activation_bars: int
    max_holding_bars: int
    calibration_fold_id: int
    calibration_hash: str

    def __post_init__(self) -> None:
        if self.max_holding_bars <= 0 or self.trail_activation_bars < 0:
            raise ValueError("exit holding periods must be valid")
        for value in (self.stop_atr_mult, self.target_atr_mult, self.trail_atr_mult):
            if value is not None and value <= 0.0:
                raise ValueError("exit multipliers must be positive")
        if not self.calibration_hash:
            raise ValueError("calibration_hash is required")


@dataclass(slots=True, frozen=True)
class PrecomputedExitPaths:
    decision_idx: NDArray[np.int64]
    edge_bps: NDArray[np.float64]
    mae_bps: NDArray[np.float64]
    mfe_bps: NDArray[np.float64]
    horizon_bars: int
    orientation_sign: int


@dataclass(slots=True, frozen=True)
class L1SleevePosterior:
    sleeve_id: str
    signal_id: str
    family: str
    outer_fold_id: int
    cluster_id: int
    member_mask_1d: NDArray[np.bool_]
    member_hash: str
    exit_policy: ExitPolicySpec
    mean_net_return: float
    standard_error: float
    posterior_positive_probability: float
    residual_novelty: float
    fold_net_returns: tuple[float, ...]
    effective_events: int
    admitted: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        values = (self.mean_net_return, self.standard_error, self.posterior_positive_probability, self.residual_novelty)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("posterior fields must be finite")
        if self.standard_error <= 0.0 or not 0.0 <= self.posterior_positive_probability <= 1.0:
            raise ValueError("posterior range is invalid")
        if self.member_mask_1d.ndim != 1:
            raise ValueError("member_mask_1d must be 1-D")
        if int(np.sum(self.member_mask_1d)) == 0:
            raise ValueError("member_mask_1d must have at least one True entry")
        if not self.member_hash:
            raise ValueError("member_hash is required")


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


@dataclass(slots=True, frozen=True)
class HandoffAdmissionEvidence:
    annualized_log_growth: float
    growth_lcb90: float
    growth_2x_cost: float
    max_drawdown: float
    annual_volatility: float
    positive_outer_folds: int
    effective_breadth: float
    active_signal_ids: tuple[str, ...]
    admitted: bool
    reasons: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class HandoffResult:
    forecast: CalibratedForecastPanel
    evidence: HandoffAdmissionEvidence


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
    "CausalClusterFold",
    "CausalFold",
    "CausalityError",
    "ClusterPanel",
    "ClusteringAlgorithm",
    "CombinedForecast",
    "CompoundEngineResult",
    "CompoundPipelineOutcome",
    "CompoundUniverseResult",
    "CovariancePath",
    "DeploymentVerdict",
    "EdgeEvidence",
    "ExecutionCostFrame",
    "ExecutionLedger",
    "ExitPolicyKind",
    "ExitPolicySpec",
    "ForecastFrame",
    "HandoffAdmissionEvidence",
    "HandoffResult",
    "InsufficientCoverageError",
    "L1SleevePosterior",
    "L2BenchmarkSeries",
    "L2CategoryResult",
    "L2Evaluation",
    "L2GateVerdict",
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
