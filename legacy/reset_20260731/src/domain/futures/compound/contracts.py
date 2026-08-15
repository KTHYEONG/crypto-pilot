from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

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


def _empty_f64() -> NDArray[np.float64]:
    return np.array([], dtype=np.float64)


def _empty_i64() -> NDArray[np.int64]:
    return np.array([], dtype=np.int64)


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
    max_name_weight_p95: float
    active_days_ratio: float
    rebalance_count: int
    positive_outer_folds: int
    oos_days: int
    category_results: tuple[L2CategoryResult, ...]
    integrity_ok: bool
    reasons: tuple[str, ...]
    absolute_cagr: float = 0.0
    spa_pvalue: float = 1.0
    bootstrap_block_days: float = 0.0
    l1_prior_active_days: int = 0
    daily_strategy_returns_1d: NDArray[np.float64] = field(default_factory=_empty_f64)
    daily_benchmark_returns_1d: NDArray[np.float64] = field(default_factory=_empty_f64)
    daily_excess_returns_1d: NDArray[np.float64] = field(default_factory=_empty_f64)
    daily_fee_returns_1d: NDArray[np.float64] = field(default_factory=_empty_f64)
    daily_day_start_ns: NDArray[np.int64] = field(default_factory=_empty_i64)

    def __post_init__(self) -> None:
        for metric in (
            self.annualized_log_growth, self.cagr, self.excess_growth_lcb90,
            self.excess_growth_probability, self.stressed_excess_growth_lcb90,
            self.sharpe, self.sharpe_probability, self.deflated_sharpe_probability,
            self.calmar, self.max_drawdown, self.daily_cvar95,
            self.annual_volatility, self.annual_turnover, self.cost_drag_ratio,
            self.max_name_weight_p95, self.absolute_cagr, self.active_days_ratio,
        ):
            if not np.isfinite(metric):
                raise ValueError(f"non-finite metric: {metric}")
        if self.l1_prior_active_days < 0:
            raise ValueError(f"l1_prior_active_days must be >= 0, got {self.l1_prior_active_days}")
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
            "absolute_cagr": {"value": self.absolute_cagr, "unit": "fraction/year"},
            "max_name_weight_p95": {"value": self.max_name_weight_p95, "unit": "fraction"},
            "active_days_ratio": {"value": self.active_days_ratio, "unit": "fraction"},
            "rebalance_count": {"value": self.rebalance_count, "unit": "count"},
            "positive_outer_folds": {"value": self.positive_outer_folds, "unit": "count"},
            "oos_days": {"value": self.oos_days, "unit": "days"},
            "integrity_ok": self.integrity_ok,
            "l1_prior_active_days": {"value": self.l1_prior_active_days, "unit": "days"},
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
    consumed_spec_hash: str = ""


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
class LegScreenDecision:
    economic_eligible: bool
    familywise_supported: bool
    capital_eligible: bool
    economic_reasons: tuple[str, ...]
    familywise_reasons: tuple[str, ...]
    critical_t: float
    n_tested_hypotheses: int

    def __post_init__(self) -> None:
        if self.capital_eligible != self.economic_eligible:
            raise ValueError(
                f"capital_eligible must equal economic_eligible (familywise is diagnostic only), "
                f"got economic={self.economic_eligible} familywise={self.familywise_supported} "
                f"capital={self.capital_eligible}"
            )
        if not np.isfinite(self.critical_t):
            raise ValueError(f"critical_t must be finite, got {self.critical_t}")
        if self.n_tested_hypotheses < 1:
            raise ValueError(f"n_tested_hypotheses must be >= 1, got {self.n_tested_hypotheses}")


@dataclass(slots=True, frozen=True)
class PortfolioAdmissionEvidence:
    admitted: bool
    reasons: tuple[str, ...]
    net_alpha_ann: float
    stressed_net_alpha_ann: float
    posterior_positive: float
    positive_folds: int
    n_folds: int
    n_traded_bars: int
    handoff_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.admitted and self.reasons:
            raise ValueError("admitted must have empty reasons")
        if not self.admitted and not self.reasons:
            raise ValueError("not-admitted must have at least one reason")
        if not np.all(np.isfinite([self.net_alpha_ann, self.stressed_net_alpha_ann, self.posterior_positive])):
            raise ValueError("net_alpha_ann, stressed_net_alpha_ann, posterior_positive must be finite")
        if not 0.0 <= self.posterior_positive <= 1.0:
            raise ValueError(f"posterior_positive must be in [0, 1], got {self.posterior_positive}")
        if self.positive_folds < 0 or self.positive_folds > self.n_folds:
            raise ValueError(f"positive_folds {self.positive_folds} must be in [0, {self.n_folds}]")
        if self.n_traded_bars < 0:
            raise ValueError(f"n_traded_bars must be >= 0, got {self.n_traded_bars}")
        if not np.isfinite(self.handoff_scale) or not (0.0 <= self.handoff_scale <= 1.0):
            raise ValueError(f"handoff_scale must be in [0, 1], got {self.handoff_scale}")
        if self.admitted and self.handoff_scale != 1.0:
            raise ValueError(f"admitted implies handoff_scale == 1.0, got {self.handoff_scale}")


@dataclass(slots=True, frozen=True)
class L1AttributionReport:
    production: PortfolioAdmissionEvidence
    shadow: PortfolioAdmissionEvidence
    economic_candidate_count: int
    capital_candidate_count: int
    bottleneck_code: str
    shadow_available: bool
    production_weights_unchanged: bool

    def __post_init__(self) -> None:
        valid_codes = ("deployable", "partial_evidence_sized", "familywise_power_limited",
                       "signal_economics_absent", "signal_generalization_failed",
                       "diagnostic_unavailable")
        if self.bottleneck_code not in valid_codes:
            raise ValueError(f"bottleneck_code must be one of {valid_codes}, got {self.bottleneck_code}")
        if self.economic_candidate_count < 0:
            raise ValueError(f"economic_candidate_count must be >= 0, got {self.economic_candidate_count}")
        if self.capital_candidate_count < 0:
            raise ValueError(f"capital_candidate_count must be >= 0, got {self.capital_candidate_count}")


@dataclass(slots=True, frozen=True)
class CompoundEngineResult:
    handoff: AlphaEventTape
    ledger: ExecutionLedger
    l2: L2Evaluation
    l3: L3ValidationResult
    deployment_candidate: DeploymentCandidate | None = None
    l1_attribution: L1AttributionReport | None = None


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
    declared_orientation: int = 1

    def __post_init__(self) -> None:
        if self.native_timeframe == "4h" and self.target_horizon_hours % 4 != 0:
            raise ValueError(
                f"target_horizon_hours must be a multiple of 4 for 4h-native signals, "
                f"got {self.target_horizon_hours}"
            )
        if self.declared_orientation not in (-1, 1):
            raise ValueError(
                f"declared_orientation must be -1 or 1, got {self.declared_orientation}"
            )


@dataclass(slots=True, frozen=True)
class FamilyEdgeRecord:
    family: str
    n_signals: int
    n_ic_bars: int
    mean_ic: float
    t_newey_west: float
    p_two_sided: float
    sidak_alpha: float
    declared_orientation: int
    admitted: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("family must be non-empty")
        if not np.isfinite(self.mean_ic):
            raise ValueError(f"mean_ic must be finite, got {self.mean_ic}")
        if self.declared_orientation not in (-1, 1):
            raise ValueError(
                f"declared_orientation must be -1 or 1, got {self.declared_orientation}"
            )


@dataclass(slots=True, frozen=True)
class FamilyEdgeScreen:
    records: tuple[FamilyEdgeRecord, ...]
    n_effective_independent: float
    admitted_families: tuple[str, ...]
    admitted_signal_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.n_effective_independent <= 0:
            raise ValueError(
                f"n_effective_independent must be > 0, got {self.n_effective_independent}"
            )


@dataclass(slots=True, frozen=True)
class SignalEdgeRecord:
    signal_id: str
    family: str
    speed: str
    target_horizon_hours: int
    n_ic_bars: int
    mean_ic: float
    t_newey_west: float
    p_two_sided: float
    sidak_alpha: float
    declared_orientation: int
    admitted: bool
    reasons: tuple[str, ...]
    intrinsic_turnover_per_bar: float = 0.0
    net_growth_ann: float = 0.0
    net_growth_probability: float = 0.0
    edge_per_turnover_bps: float = 0.0
    effective_horizon_hours: int = 0
    effective_orientation: int = 0
    effective_horizon_t_stat: float = 0.0

    def __post_init__(self) -> None:
        if not self.signal_id:
            raise ValueError("signal_id must be non-empty")
        if not np.isfinite(self.mean_ic):
            raise ValueError(f"mean_ic must be finite, got {self.mean_ic}")
        if self.declared_orientation not in (-1, 1):
            raise ValueError(
                f"declared_orientation must be -1 or 1, got {self.declared_orientation}"
            )
        if self.effective_horizon_hours < 0:
            raise ValueError(
                f"effective_horizon_hours must be >= 0, got {self.effective_horizon_hours}"
            )
        if self.effective_orientation not in (-1, 0, 1):
            raise ValueError(
                f"effective_orientation must be -1, 0, or 1, got {self.effective_orientation}"
            )
        if not np.isfinite(self.effective_horizon_t_stat):
            raise ValueError(
                f"effective_horizon_t_stat must be finite, got {self.effective_horizon_t_stat}"
            )


@dataclass(slots=True, frozen=True)
class SignalEdgeScreen:
    records: tuple[SignalEdgeRecord, ...]
    n_effective_independent: float
    admitted_signal_ids: tuple[str, ...]
    admitted_families: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.n_effective_independent <= 0:
            raise ValueError(
                f"n_effective_independent must be > 0, got {self.n_effective_independent}"
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
class ExitPathCache:
    paths_by_signal: dict[str, PrecomputedExitPaths]
    atr: NDArray[np.float64]

    def get(self, signal_id: str) -> PrecomputedExitPaths | None:
        return self.paths_by_signal.get(signal_id)


@dataclass(slots=True, frozen=True)
class SignalFoldRecord:
    gross_1d: NDArray[np.float64]
    cost_1d: NDArray[np.float64]
    funding_1d: NDArray[np.float64]
    net_1d: NDArray[np.float64]
    regime_code_1d: NDArray[np.int8]

    def __post_init__(self) -> None:
        n = self.net_1d.shape[0]
        for arr in (self.gross_1d, self.cost_1d, self.funding_1d, self.regime_code_1d):
            if arr.shape[0] != n:
                raise ValueError(f"all arrays must have length {n}, got {arr.shape[0]}")


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
    fitted_beta: float
    mean_net_return: float
    standard_error: float
    posterior_positive_probability: float
    residual_novelty: float
    fold_net_returns: tuple[float, ...]
    effective_events: int
    admitted: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        values = (self.fitted_beta, self.mean_net_return, self.standard_error, self.posterior_positive_probability, self.residual_novelty)
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
class L1RoutingSleeve:
    sleeve_id: str
    signal_id: str
    family: str
    outer_fold_id: int
    cluster_id: int
    member_mask_1d: NDArray[np.bool_]
    member_hash: str
    declared_orientation: int

    def __post_init__(self) -> None:
        if not self.sleeve_id or not self.signal_id or not self.family:
            raise ValueError("sleeve_id, signal_id, and family must be non-empty")
        if not self.member_hash:
            raise ValueError("member_hash must be non-empty")
        if self.outer_fold_id < 0 or self.cluster_id < 0:
            raise ValueError("outer_fold_id and cluster_id must be >= 0")
        if self.member_mask_1d.ndim != 1 or self.member_mask_1d.dtype != np.bool_:
            raise ValueError("member_mask_1d must be 1-D bool array")
        if int(np.sum(self.member_mask_1d)) < 2:
            raise ValueError("member_mask_1d must have at least two members")
        if self.declared_orientation not in (-1, 1):
            raise ValueError("declared_orientation must be -1 or 1")


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
    robust_fold_growth: float = 0.0
    fold_growths: tuple[float, ...] = ()


@dataclass(slots=True, frozen=True)
class CausalRegimePanel:
    decision_timestamps_ns: NDArray[np.int64]
    code_1d: NDArray[np.int8]
    available_at_ns_1d: NDArray[np.int64]
    names: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class RegimeExpertEvidence:
    signal_id: str
    outer_fold_id: int
    regime_code: int
    effective_blocks: int
    posterior_positive_probability: float
    growth_lcb90: float
    growth_2x_cost: float
    robust_inner_growth: float
    positive_inner_folds: int
    scale: float
    admitted: bool
    reasons: tuple[str, ...]
    annual_volatility: float = 0.0
    regime_mean_net: float = 0.0
    n_evidence_bars: int = 0


@dataclass(slots=True, frozen=True)
class RegimeRoutedForecast:
    forecast: CalibratedForecastPanel
    evidence: tuple[RegimeExpertEvidence, ...]
    active_expert_count_1d: NDArray[np.int16]
    tested_hypotheses: int


@dataclass(slots=True, frozen=True)
class ExpertReturnTape:
    decision_time_ns_1d: NDArray[np.int64]
    execution_time_ns_1d: NDArray[np.int64]
    available_time_ns_1d: NDArray[np.int64]
    signal_id_1d: NDArray[np.str_]
    outer_fold_id_1d: NDArray[np.int16]
    regime_code_1d: NDArray[np.int8]
    gross_return_1d: NDArray[np.float64]
    execution_cost_return_1d: NDArray[np.float64]
    funding_return_1d: NDArray[np.float64]
    net_return_1d: NDArray[np.float64]

    def __post_init__(self) -> None:
        n = self.decision_time_ns_1d.shape[0]
        for arr in (self.execution_time_ns_1d, self.available_time_ns_1d,
                     self.signal_id_1d, self.outer_fold_id_1d, self.regime_code_1d,
                     self.gross_return_1d, self.execution_cost_return_1d,
                     self.funding_return_1d, self.net_return_1d):
            if arr.shape[0] != n:
                raise ValueError(f"all tape arrays must have length {n}, got {arr.shape[0]}")
        if not np.all(self.execution_time_ns_1d <= self.available_time_ns_1d):
            raise CausalityError("execution_time must be <= available_time")
        names = ("gross_return_1d", "execution_cost_return_1d", "funding_return_1d", "net_return_1d")
        for name in names:
            arr = getattr(self, name)
            bad = int(np.sum(~np.isfinite(arr)))
            if bad > 0:
                raise ValueError(f"non-finite return components: {name}={bad}")
        if not np.all(np.isclose(self.net_return_1d,
                                 self.gross_return_1d + self.execution_cost_return_1d + self.funding_return_1d)):
            raise ValueError("net != gross + execution_cost + funding")


@dataclass(slots=True, frozen=True)
class RouteAttribution:
    candidate_experts: int
    unconditional_pass: int
    temporal_pass: int
    regime_pass: int
    active_experts: int
    reason_counts: dict[str, int]


@dataclass(slots=True, frozen=True)
class PrequentialExpertRoute:
    forecast: CalibratedForecastPanel
    tape: ExpertReturnTape
    evidence: tuple[RegimeExpertEvidence, ...]
    attribution: RouteAttribution
    tested_hypotheses: int
    active_expert_count_1d: NDArray[np.int16] | None = None
    is_cash_only: bool = False
    fold_route_scales: dict[int, dict[str, float]] | None = None


@dataclass(slots=True, frozen=True)
class HandoffResult:
    forecast: CalibratedForecastPanel
    evidence: HandoffAdmissionEvidence
    admitted_sleeves: tuple[L1SleevePosterior | L1RoutingSleeve, ...] = ()


@dataclass(slots=True, frozen=True)
class QuarterlyBarBoundaries:
    acquisition_start: int
    l1_start: int
    l2_start: int
    l3_start: int
    cutoff_exclusive: int

    def __post_init__(self) -> None:
        if not (self.acquisition_start < self.l1_start < self.l2_start < self.l3_start < self.cutoff_exclusive):
            raise ValueError(
                f"boundaries must be strictly increasing: "
                f"acq={self.acquisition_start} < l1={self.l1_start} "
                f"< l2={self.l2_start} < l3={self.l3_start} "
                f"< cutoff={self.cutoff_exclusive}"
            )


@dataclass(slots=True, frozen=True)
class CompoundWindowAudit:
    passed: bool
    core_coverage_ratio: float
    dataset_status: tuple[StrategyDataCoverageEntry, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.passed and self.reasons:
            raise ValueError("passed audit must have empty reasons")
        if not self.passed and not self.reasons:
            raise ValueError("failed audit must have at least one reason")


class TrialIntegrityError(RuntimeError):
    ...


@dataclass(slots=True, frozen=True)
class CandidateTrial:
    candidate_hash: str
    strategy_spec_hash: str
    descriptor_ids: tuple[str, ...]
    risk_policy_hash: str
    cutoff_time_ns: int

    def __post_init__(self) -> None:
        if not self.candidate_hash:
            raise ValueError("candidate_hash is required")
        if not self.strategy_spec_hash:
            raise ValueError("strategy_spec_hash is required")


class CandidateTrialLedger:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def _ensure_db(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=EXTRA")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS candidate_trials (
                    candidate_hash TEXT NOT NULL,
                    strategy_spec_hash TEXT NOT NULL,
                    descriptor_ids TEXT NOT NULL,
                    risk_policy_hash TEXT NOT NULL,
                    cutoff_time_ns INTEGER NOT NULL,
                    created_at_ns INTEGER NOT NULL,
                    l2_daily_returns BLOB,
                    PRIMARY KEY (candidate_hash, cutoff_time_ns)
                )
            """)
            columns = {
                str(row[1]) for row in self._conn.execute("PRAGMA table_info(candidate_trials)")
            }
            if "l2_daily_returns" not in columns:
                self._conn.execute(
                    "ALTER TABLE candidate_trials ADD COLUMN l2_daily_returns BLOB"
                )
            self._conn.commit()
        return self._conn

    def register(
        self, trial: CandidateTrial, *,
        l2_daily_returns: NDArray[np.float64] | None = None,
    ) -> int:
        conn = self._ensure_db()
        now_ns = int(time.time_ns())
        blob: bytes | None = l2_daily_returns.tobytes() if l2_daily_returns is not None else None
        try:
            conn.execute(
                """
                INSERT INTO candidate_trials
                    (candidate_hash, strategy_spec_hash, descriptor_ids,
                     risk_policy_hash, cutoff_time_ns, created_at_ns,
                     l2_daily_returns)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial.candidate_hash, trial.strategy_spec_hash,
                    json.dumps(list(trial.descriptor_ids)),
                    trial.risk_policy_hash, trial.cutoff_time_ns, now_ns,
                    blob,
                ),
            )
            conn.commit()
            return 1
        except sqlite3.IntegrityError:
            conn.execute(
                """
                UPDATE candidate_trials
                SET created_at_ns = ?, l2_daily_returns = ?
                WHERE candidate_hash = ? AND cutoff_time_ns = ?
                """,
                (now_ns, blob, trial.candidate_hash, trial.cutoff_time_ns),
            )
            conn.commit()
            return 0

    def distinct_count(self, *, cutoff_time_ns: int, floor: int = 27) -> int:
        conn = self._ensure_db()
        row = conn.execute(
            "SELECT COUNT(DISTINCT candidate_hash) FROM candidate_trials WHERE cutoff_time_ns = ?",
            (cutoff_time_ns,),
        ).fetchone()
        count = int(row[0]) if row is not None else 0
        return max(count, floor)

    def load_trial_returns(
        self, *, cutoff_time_ns: int, exclude_candidate_hash: str = "",
        min_days: int = 30,
    ) -> NDArray[np.float64]:
        conn = self._ensure_db()
        if exclude_candidate_hash:
            rows = conn.execute(
                "SELECT l2_daily_returns FROM candidate_trials WHERE cutoff_time_ns = ? AND candidate_hash != ?",
                (cutoff_time_ns, exclude_candidate_hash),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT l2_daily_returns FROM candidate_trials WHERE cutoff_time_ns = ?",
                (cutoff_time_ns,),
            ).fetchall()
        arrays: list[NDArray[np.float64]] = []
        for (blob,) in rows:
            if blob is None:
                continue
            arr = np.frombuffer(blob, dtype=np.float64)
            if arr.shape[0] < min_days:
                continue
            arrays.append(arr)
        if not arrays:
            return np.zeros((0, 0), dtype=np.float64)
        min_len = min(a.shape[0] for a in arrays)
        stacked = np.stack([a[:min_len] for a in arrays], axis=0)
        return stacked


@dataclass(slots=True, frozen=True)
class DeploymentCandidate:
    active_signal_ids: tuple[str, ...]
    descriptors: tuple[SignalDescriptor, ...]
    orientation_signs: tuple[int, ...]
    vote_weights: tuple[float, ...]
    model_version: str
    strategy_spec_hash: str
    fold_manifest_hash: str
    trial_count: int

    def __post_init__(self) -> None:
        if not self.active_signal_ids:
            raise ValueError("active_signal_ids must be non-empty")
        if len(self.active_signal_ids) != len(self.descriptors):
            raise ValueError("active_signal_ids and descriptors must match")
        if len(self.orientation_signs) != len(self.descriptors):
            raise ValueError("orientation_signs must match descriptors")
        if len(self.vote_weights) != len(self.descriptors):
            raise ValueError("vote_weights must match descriptors")
        if not self.model_version:
            raise ValueError("model_version is required")
        if not self.strategy_spec_hash:
            raise ValueError("strategy_spec_hash is required")
        if not self.fold_manifest_hash:
            raise ValueError("fold_manifest_hash is required")
        if self.trial_count < 0:
            raise ValueError("trial_count must be >= 0")


@dataclass(slots=True, frozen=True)
class DeploymentBundle:
    schema_version: int
    promotion_id: str
    candidate: DeploymentCandidate
    data_manifest_hash: str
    universe_state_hash: str
    config_payload: dict[str, object]
    l2_payload: dict[str, object]
    l3_payload: dict[str, object]
    sha256: str = ""

    def __post_init__(self) -> None:
        if not self.promotion_id:
            raise ValueError("promotion_id is required")


@dataclass(slots=True, frozen=True)
class SignalConceptSpec:
    concept_id: str
    member_signal_ids: tuple[str, ...]
    mode: Literal["xs", "ts"]
    horizon_band_bars: tuple[int, ...]
    declared_orientation: int

    def __post_init__(self) -> None:
        if not self.concept_id:
            raise ValueError("concept_id must be non-empty")
        if not self.member_signal_ids:
            raise ValueError("member_signal_ids must be non-empty")
        if self.mode not in ("xs", "ts"):
            raise ValueError(f"mode must be 'xs' or 'ts', got {self.mode}")
        if not self.horizon_band_bars:
            raise ValueError("horizon_band_bars must be non-empty")
        for h in self.horizon_band_bars:
            if h <= 0:
                raise ValueError(f"horizon_band_bars entries must be > 0, got {h}")
        if self.declared_orientation not in (-1, 1):
            raise ValueError(
                f"declared_orientation must be -1 or 1, got {self.declared_orientation}"
            )


@dataclass(slots=True, frozen=True)
class LegBook:
    spec: SignalConceptSpec
    book_2d: NDArray[np.float64]       # (T, S) unit-gross tranche book
    gross_return_1d: NDArray[np.float64]  # (T,) book_t . asset_return_{t+1}
    turnover_1d: NDArray[np.float64]   # (T,) sum |w_t - w_{t-1}|

    def __post_init__(self) -> None:
        if self.book_2d.ndim != 2:
            raise ValueError(f"book_2d must be 2-D, got {self.book_2d.ndim}")
        t_, _ = self.book_2d.shape
        if self.gross_return_1d.ndim != 1 or self.gross_return_1d.shape[0] != t_:
            raise ValueError(f"gross_return_1d must be 1-D with length {t_}")
        if self.turnover_1d.ndim != 1 or self.turnover_1d.shape[0] != t_:
            raise ValueError(f"turnover_1d must be 1-D with length {t_}")
        if not np.all(np.isfinite(self.book_2d)):
            raise ValueError("book_2d must be finite")
        if not np.all(np.isfinite(self.gross_return_1d)):
            raise ValueError("gross_return_1d must be finite")
        if not np.all(np.isfinite(self.turnover_1d)):
            raise ValueError("turnover_1d must be finite")


@dataclass(slots=True, frozen=True)
class LegEvidence:
    concept_id: str
    mode: str
    n_oos_bars: int
    alpha_ann: float
    beta_market: float
    alpha_sharpe: float
    t_alpha_newey_west: float
    breakeven_cost_bps: float
    mean_turnover_per_bar: float
    positive_folds: int
    n_folds: int
    posterior_positive: float
    evidence_weight: float
    reasons: tuple[str, ...]
    net_alpha_ann: float = 0.0
    net_alpha_sharpe: float = 0.0
    t_net_alpha_newey_west: float = 0.0

    def __post_init__(self) -> None:
        if not self.concept_id:
            raise ValueError("concept_id must be non-empty")
        if self.mode not in ("xs", "ts"):
            raise ValueError(f"mode must be 'xs' or 'ts', got {self.mode}")
        if self.n_oos_bars < 0:
            raise ValueError(f"n_oos_bars must be >= 0, got {self.n_oos_bars}")
        if not np.isfinite(self.alpha_ann):
            raise ValueError(f"alpha_ann must be finite, got {self.alpha_ann}")
        if not np.isfinite(self.beta_market):
            raise ValueError(f"beta_market must be finite, got {self.beta_market}")
        if not np.isfinite(self.alpha_sharpe):
            raise ValueError(f"alpha_sharpe must be finite, got {self.alpha_sharpe}")
        if not np.isfinite(self.t_alpha_newey_west):
            raise ValueError(f"t_alpha_newey_west must be finite, got {self.t_alpha_newey_west}")
        if not np.isfinite(self.breakeven_cost_bps):
            raise ValueError(f"breakeven_cost_bps must be finite, got {self.breakeven_cost_bps}")
        if not np.isfinite(self.mean_turnover_per_bar):
            raise ValueError(f"mean_turnover_per_bar must be finite, got {self.mean_turnover_per_bar}")
        if self.positive_folds < 0 or self.positive_folds > self.n_folds:
            raise ValueError(f"positive_folds {self.positive_folds} must be in [0, {self.n_folds}]")
        if not 0.0 <= self.posterior_positive <= 1.0:
            raise ValueError(f"posterior_positive must be in [0, 1], got {self.posterior_positive}")
        if not np.isfinite(self.evidence_weight):
            raise ValueError(f"evidence_weight must be finite, got {self.evidence_weight}")
        if self.evidence_weight < 0.0:
            raise ValueError(f"evidence_weight must be >= 0, got {self.evidence_weight}")
        if not np.isfinite(self.net_alpha_ann):
            raise ValueError(f"net_alpha_ann must be finite, got {self.net_alpha_ann}")
        if not np.isfinite(self.net_alpha_sharpe):
            raise ValueError(f"net_alpha_sharpe must be finite, got {self.net_alpha_sharpe}")
        if not np.isfinite(self.t_net_alpha_newey_west):
            raise ValueError(f"t_net_alpha_newey_west must be finite, got {self.t_net_alpha_newey_west}")


@dataclass(slots=True, frozen=True)
class L1LegPanel:
    decision_timestamps_ns: NDArray[np.int64]
    symbols: tuple[str, ...]
    leg_specs: tuple[SignalConceptSpec, ...]
    books_3d: NDArray[np.float32]           # (T, S, K)
    leg_weights_2d: NDArray[np.float64]     # (T, K) causal prequential weights
    combined_weights_2d: NDArray[np.float64]  # (T, S)
    evidence: tuple[LegEvidence, ...]
    admitted: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        t_ = self.decision_timestamps_ns.shape[0]
        s_ = len(self.symbols)
        k_ = len(self.leg_specs)
        if self.decision_timestamps_ns.ndim != 1:
            raise ValueError("decision_timestamps_ns must be 1-D")
        if not self.symbols:
            raise ValueError("symbols must be non-empty")
        if not self.leg_specs:
            raise ValueError("leg_specs must be non-empty")
        if self.books_3d.shape != (t_, s_, k_):
            raise ValueError(
                f"books_3d shape {self.books_3d.shape} != ({t_}, {s_}, {k_})"
            )
        if self.leg_weights_2d.shape != (t_, k_):
            raise ValueError(
                f"leg_weights_2d shape {self.leg_weights_2d.shape} != ({t_}, {k_})"
            )
        if self.combined_weights_2d.shape != (t_, s_):
            raise ValueError(
                f"combined_weights_2d shape {self.combined_weights_2d.shape} != ({t_}, {s_})"
            )
        if len(self.evidence) != k_:
            raise ValueError(f"evidence length {len(self.evidence)} != {k_}")
        if self.admitted and self.reasons:
            raise ValueError("admitted panel must have empty reasons")
        if not self.admitted and not self.reasons:
            raise ValueError("not-admitted panel must have at least one reason")


class TargetWeightSink(Protocol):
    def rebalance(
        self, *, target_weights: Mapping[str, float], idempotency_key: str
    ) -> None: ...


@dataclass(slots=True, frozen=True)
class SymbolLegBook:
    concept_id: str
    book_2d: NDArray[np.float64]
    per_symbol_net_2d: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not self.concept_id:
            raise ValueError("concept_id must be non-empty")
        if self.book_2d.ndim != 2:
            raise ValueError(f"book_2d must be 2-D, got {self.book_2d.ndim}")
        if self.per_symbol_net_2d.ndim != 2:
            raise ValueError(f"per_symbol_net_2d must be 2-D, got {self.per_symbol_net_2d.ndim}")
        if self.book_2d.shape != self.per_symbol_net_2d.shape:
            raise ValueError(
                f"book_2d shape {self.book_2d.shape} != per_symbol_net_2d shape {self.per_symbol_net_2d.shape}"
            )
        if not np.all(np.isfinite(self.book_2d)):
            raise ValueError("book_2d must be finite")
        if not np.all(np.isfinite(self.per_symbol_net_2d)):
            raise ValueError("per_symbol_net_2d must be finite")


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
    "CandidateTrial",
    "CandidateTrialLedger",
    "CausalAlphaFold",
    "CausalClusterFold",
    "CausalFold",
    "CausalRegimePanel",
    "CausalityError",
    "ClusterPanel",
    "ClusteringAlgorithm",
    "CombinedForecast",
    "CompoundEngineResult",
    "CompoundPipelineOutcome",
    "CompoundUniverseResult",
    "CompoundWindowAudit",
    "CovariancePath",
    "DeploymentBundle",
    "DeploymentCandidate",
    "DeploymentVerdict",
    "EdgeEvidence",
    "ExecutionCostFrame",
    "ExecutionLedger",
    "ExitPolicyKind",
    "ExitPolicySpec",
    "ExpertReturnTape",
    "FamilyEdgeRecord",
    "FamilyEdgeScreen",
    "HandoffAdmissionEvidence",
    "HandoffResult",
    "InsufficientCoverageError",
    "L1AttributionReport",
    "L1LegPanel",
    "L1RoutingSleeve",
    "L1SleevePosterior",
    "L2BenchmarkSeries",
    "L2CategoryResult",
    "L2Evaluation",
    "L2GateVerdict",
    "L3ValidationResult",
    "LadderStageResult",
    "LegBook",
    "LegEvidence",
    "LegScreenDecision",
    "MarketFeatureCube",
    "MultiTimeframeBars",
    "MultiscaleAlphaDefinition",
    "PortfolioAdmissionEvidence",
    "PortfolioDecision",
    "PrequentialExpertRoute",
    "QuarterlyBarBoundaries",
    "RawAlphaTape",
    "RawSignalPanel",
    "RegimeExpertEvidence",
    "RegimeRoutedForecast",
    "RiskOverlayResult",
    "RouteAttribution",
    "SealedHoldoutManifest",
    "SignalAdmissionEvidence",
    "SignalCalibration",
    "SignalConceptSpec",
    "SignalDescriptor",
    "SignalEdgeRecord",
    "SignalEdgeScreen",
    "SignalFoldRecord",
    "StrategyDataCoverage",
    "StrategyDataCoverageEntry",
    "SymbolLegBook",
    "TargetWeightSink",
    "TimeframeBarCube",
    "TrialIntegrityError",
    "UniverseLedgerCoverage",
]
