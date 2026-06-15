from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from src.domain.futures.strategy.candidate_dataset import CandidateFeatureSchema
    from src.domain.futures.strategy.candidate_ensemble import RegimeConditionalEnsemble

SignalArchetype = Literal[
    "trend",
    "ts_mom",
    "mean_rev",
    "flow_rev",
    "unwind",
    "carry_rev",
    "beta_neut",
]

RegimeName = Literal[
    "bull_quiet",
    "bull_volatile",
    "bear_quiet",
    "bear_volatile",
    "transition",
    "crash",
]


@dataclass(slots=True, frozen=True)
class SignalExitPolicy:
    """Deterministic exit geometry attached to a candidate signal."""

    policy_id: str
    archetype: SignalArchetype
    stop_atr_mult: float
    take_profit_atr_mult: float
    expected_holding_bars: int
    min_holding_bars: int
    description: str = ""


@dataclass(slots=True, frozen=True)
class CandidateSignalPanel:
    """Rule-based candidate signal panel contract."""

    family: str
    variant: str
    params: dict[str, float | int | str]
    datetimes: NDArray[np.datetime64] | NDArray[np.int64]
    symbols: tuple[str, ...]
    signed_score_2d: NDArray[np.float64]
    side_hint_2d: NDArray[np.int8]
    expected_holding_bars: int
    min_holding_bars: int
    stop_atr_mult: float
    take_profit_atr_mult: float
    turnover_proxy_2d: NDArray[np.float64]
    valid_mask_2d: NDArray[np.bool_]
    metadata: dict[str, Any] = field(default_factory=dict)
    archetype: SignalArchetype | str = "mean_rev"
    allowed_regimes: tuple[RegimeName | str, ...] = ()
    exit_policies: tuple[SignalExitPolicy, ...] = ()
    regime_code_1d: NDArray[np.int8] | None = None
    regime_name_by_code: tuple[str, ...] = ()

    @property
    def scores(self) -> NDArray[np.float64]:
        """Backward-compatible score accessor."""
        return self.signed_score_2d

    @property
    def valid_mask(self) -> NDArray[np.bool_]:
        """Backward-compatible valid mask accessor."""
        return self.valid_mask_2d

    @property
    def rule_names(self) -> tuple[str, ...]:
        """Backward-compatible rule name accessor."""
        return (f"{self.family}:{self.variant}",)


class EdgeSource(StrEnum):
    DISABLED = "disabled"
    DIRECT_MODEL = "direct_model"
    PRIOR_ONLY = "prior_only"
    PRIOR_RESIDUAL = "prior_residual"


EdgePredictionMode = Literal[
    "disabled",
    "direct",
    "prior_only",
    "prior_residual",
]


@dataclass(slots=True, frozen=True)
class GateValidationReport:
    enabled: bool
    threshold: float
    raw_brier: float
    calibrated_brier: float
    base_brier: float
    brier_skill: float
    roc_auc: float
    average_precision: float
    decile_lift: float
    incremental_log_growth_lcb: float
    reason: str


@dataclass(slots=True, frozen=True)
class EdgeValidationReport:
    source: EdgeSource
    prior_rank_ic: float
    residual_rank_ic: float
    incremental_log_growth_mean: float
    incremental_log_growth_lcb: float
    selected: bool
    reason: str


@dataclass(slots=True, frozen=True)
class CandidateExecutionPlan:
    target_weights_2d: NDArray[np.float64]
    event_id_2d: NDArray[np.int64]
    stop_atr_mult_2d: NDArray[np.float64]
    take_profit_atr_mult_2d: NDArray[np.float64]
    diagnostics: dict[str, float | int | str]


@dataclass(slots=True, frozen=True)
class CandidateModelOutput:
    """Container for candidate model outputs."""

    events: pd.DataFrame
    p_pass: NDArray[np.float64]
    gate_enabled: bool
    gate_threshold: float
    edge_source: EdgeSource
    expected_return_r: NDArray[np.float64]
    expected_net_bps: NDArray[np.float64]
    expected_gross_bps: NDArray[np.float64]
    q10_return_r: NDArray[np.float64]
    q10_net_bps: NDArray[np.float64]
    q10_gross_bps: NDArray[np.float64]
    q90_return_r: NDArray[np.float64]
    q90_net_bps: NDArray[np.float64]
    q90_gross_bps: NDArray[np.float64]
    selection_score: NDArray[np.float64]
    kelly_fraction: NDArray[np.float64]
    prediction_scale_bps: NDArray[np.float64]
    _has_explicit_expected_gross_bps: bool = field(init=False, repr=False, default=False)
    validation_diagnostics: dict[str, object] = field(
        default_factory=dict
    )

    def __init__(
        self,
        events: pd.DataFrame,
        p_pass: NDArray[np.float64],
        gate_enabled: bool = False,
        gate_threshold: float = 0.5,
        edge_source: EdgeSource = EdgeSource.PRIOR_ONLY,
        expected_return_r: NDArray[np.float64] | None = None,
        expected_net_bps: NDArray[np.float64] | None = None,
        expected_gross_bps: NDArray[np.float64] | None = None,
        q10_return_r: NDArray[np.float64] | None = None,
        q10_net_bps: NDArray[np.float64] | None = None,
        q10_gross_bps: NDArray[np.float64] | None = None,
        q90_return_r: NDArray[np.float64] | None = None,
        q90_net_bps: NDArray[np.float64] | None = None,
        q90_gross_bps: NDArray[np.float64] | None = None,
        selection_score: NDArray[np.float64] | None = None,
        kelly_fraction: NDArray[np.float64] | None = None,
        prediction_scale_bps: NDArray[np.float64] | None = None,
        validation_diagnostics: Mapping[str, object] | None = None,
        **kwargs: Any,
    ) -> None:
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "p_pass", p_pass)
        object.__setattr__(self, "gate_enabled", gate_enabled)
        object.__setattr__(self, "gate_threshold", gate_threshold)
        object.__setattr__(self, "edge_source", edge_source)

        size = p_pass.shape[0] if hasattr(p_pass, "shape") else 0

        net_bps = expected_net_bps if expected_net_bps is not None else kwargs.get("mu_net_decision_bps")
        if net_bps is None:
            net_bps = np.zeros(size, dtype=np.float64)
        object.__setattr__(self, "expected_net_bps", net_bps)
        gross_bps = expected_gross_bps if expected_gross_bps is not None else kwargs.get("mu_gross_bps")
        if gross_bps is None:
            gross_bps = np.zeros(size, dtype=np.float64)
        object.__setattr__(self, "expected_gross_bps", gross_bps)
        object.__setattr__(
            self,
            "_has_explicit_expected_gross_bps",
            bool(expected_gross_bps is not None or "mu_gross_bps" in kwargs),
        )

        ret_r = expected_return_r if expected_return_r is not None else net_bps / 25.0
        object.__setattr__(self, "expected_return_r", ret_r)

        q10_bps_val = q10_net_bps if q10_net_bps is not None else np.zeros(size, dtype=np.float64)
        object.__setattr__(self, "q10_net_bps", q10_bps_val)
        q10_gross_val = q10_gross_bps if q10_gross_bps is not None else np.zeros(size, dtype=np.float64)
        object.__setattr__(self, "q10_gross_bps", q10_gross_val)

        q10_r_val = q10_return_r if q10_return_r is not None else q10_bps_val / 25.0
        object.__setattr__(self, "q10_return_r", q10_r_val)

        q90_bps_val = q90_net_bps if q90_net_bps is not None else np.zeros(size, dtype=np.float64)
        object.__setattr__(self, "q90_net_bps", q90_bps_val)
        q90_gross_val = q90_gross_bps if q90_gross_bps is not None else np.zeros(size, dtype=np.float64)
        object.__setattr__(self, "q90_gross_bps", q90_gross_val)

        q90_r_val = q90_return_r if q90_return_r is not None else q90_bps_val / 25.0
        object.__setattr__(self, "q90_return_r", q90_r_val)

        sel_score = selection_score if selection_score is not None else kwargs.get("utility_score")
        if sel_score is None:
            sel_score = np.zeros(size, dtype=np.float64)
        object.__setattr__(self, "selection_score", sel_score)

        k_frac = kelly_fraction if kelly_fraction is not None else np.zeros(size, dtype=np.float64)
        object.__setattr__(self, "kelly_fraction", k_frac)
        scale = prediction_scale_bps if prediction_scale_bps is not None else np.abs(gross_bps)
        object.__setattr__(self, "prediction_scale_bps", scale)

        val_diag = (
            validation_diagnostics
            if validation_diagnostics is not None
            else kwargs.get("selection_thresholds", {})
        )
        object.__setattr__(self, "validation_diagnostics", dict(val_diag))

    @property
    def mu_gross_bps(self) -> NDArray[np.float64]:
        """Backward-compatible alias for expected_gross_bps."""
        return self.expected_gross_bps

    @property
    def mu_net_decision_bps(self) -> NDArray[np.float64]:
        """Backward-compatible alias for expected_net_bps."""
        return self.expected_net_bps

    @property
    def utility_score(self) -> NDArray[np.float64]:
        """Backward-compatible alias for selection_score."""
        return self.selection_score

    @property
    def selection_thresholds(self) -> dict[str, float | bool]:
        """Backward-compatible alias returning threshold limits."""
        return self.validation_diagnostics  # type: ignore[return-value]


class CandidateWorkflowStatus(StrEnum):
    BLOCKED = "blocked"
    WF_ELIGIBLE = "wf_eligible"
    DEPLOYMENT_PROMOTED = "deployment_promoted"


FoldFitStatus = Literal[
    "trained",
    "insufficient_fit",
    "empty_oos",
    "constant_prediction",
    "failed",
]


@dataclass(slots=True, frozen=True)
class CandidateFoldOutput:
    """Orchestration fold output contract."""

    fold_id: int
    oos_start: int
    oos_end: int
    model_output: CandidateModelOutput
    selected_events: pd.DataFrame
    gate_report: GateValidationReport
    edge_report: EdgeValidationReport
    fit_status: FoldFitStatus
    n_fit: int
    skip_reason: str | None
    gate_model: Any | None = None
    edge_models: Any | None = None
    fit_set: Any | None = None
    calibration_set: Any | None = None
    oos_set: Any | None = None
    timing_profile: dict[str, float] | None = None


GateComparator = Literal["ge", "gt"]


@dataclass(slots=True, frozen=True)
class SignalSourceKey:
    symbol: str
    strategy_id: str
    activation_context: str


@dataclass(slots=True, frozen=True)
class MatchedBaselineKey:
    symbol: str
    side: Literal[-1, 1]
    holding_bucket: int


@dataclass(slots=True, frozen=True, init=False)
class SymbolStrategyEvidence:
    key: SignalSourceKey
    mean_gross_bps: float
    mean_incremental_bps: float
    block_tstat_incremental: float
    probability_positive: float
    p_value: float
    q_value: float
    positive_fold_ratio: float
    n_obs: int
    effective_n: float
    n_folds: int
    quality_weight: float
    hard_eligible: bool
    structural_reasons: tuple[str, ...]
    diagnostic_flags: tuple[str, ...]

    def __init__(
        self,
        *,
        key: SignalSourceKey,
        mean_gross_bps: float,
        mean_incremental_bps: float,
        block_tstat_incremental: float | None = None,
        probability_positive: float = 0.0,
        p_value: float,
        q_value: float,
        positive_fold_ratio: float,
        n_obs: int,
        effective_n: float,
        n_folds: int,
        quality_weight: float | None = None,
        hard_eligible: bool | None = None,
        structural_reasons: tuple[str, ...] = (),
        diagnostic_flags: tuple[str, ...] = (),
        bootstrap_tstat_incremental: float | None = None,
        reliability: float | None = None,
        qualified: bool | None = None,
        rejection_reasons: tuple[str, ...] | None = None,
    ) -> None:
        compat_tstat = (
            block_tstat_incremental
            if block_tstat_incremental is not None
            else float(bootstrap_tstat_incremental or 0.0)
        )
        compat_weight = (
            quality_weight if quality_weight is not None else float(reliability or 0.0)
        )
        compat_hard_eligible = (
            hard_eligible if hard_eligible is not None else bool(qualified)
        )
        compat_structural = (
            structural_reasons if structural_reasons else tuple(rejection_reasons or ())
        )
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "mean_gross_bps", mean_gross_bps)
        object.__setattr__(self, "mean_incremental_bps", mean_incremental_bps)
        object.__setattr__(self, "block_tstat_incremental", float(compat_tstat))
        object.__setattr__(
            self,
            "probability_positive",
            float(max(0.0, min(1.0, probability_positive))),
        )
        object.__setattr__(self, "p_value", float(p_value))
        object.__setattr__(self, "q_value", float(q_value))
        object.__setattr__(self, "positive_fold_ratio", float(positive_fold_ratio))
        object.__setattr__(self, "n_obs", int(n_obs))
        object.__setattr__(self, "effective_n", float(effective_n))
        object.__setattr__(self, "n_folds", int(n_folds))
        object.__setattr__(self, "quality_weight", float(compat_weight))
        object.__setattr__(self, "hard_eligible", bool(compat_hard_eligible))
        object.__setattr__(self, "structural_reasons", tuple(compat_structural))
        object.__setattr__(self, "diagnostic_flags", tuple(diagnostic_flags))

    @property
    def bootstrap_tstat_incremental(self) -> float:
        return self.block_tstat_incremental

    @property
    def reliability(self) -> float:
        return self.quality_weight

    @property
    def qualified(self) -> bool:
        return self.hard_eligible and self.quality_weight > 0.0

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return self.structural_reasons + self.diagnostic_flags


@dataclass(slots=True, frozen=True)
class QualifiedSignalRegistry:
    by_symbol: dict[str, tuple[SymbolStrategyEvidence, ...]]
    ready_symbols: tuple[str, ...]
    trade_scope_count: int
    registry_version: str


@dataclass(slots=True, frozen=True)
class Layer1EvidenceSnapshot:
    as_of_idx: int
    evidence: tuple[SymbolStrategyEvidence, ...]
    registry: QualifiedSignalRegistry
    matured_event_count: int


@dataclass(slots=True, frozen=True, init=False)
class ValidatedSignalEvent:
    decision_idx: int
    decision_time: np.datetime64
    symbol: str
    strategy_id: str
    activation_context: str
    side: Literal[-1, 1]
    expected_net_bps: float
    expected_gross_bps: float
    q10_net_bps: float
    q10_gross_bps: float
    q90_net_bps: float
    q90_gross_bps: float
    expected_holding_bars: int
    quality_weight: float
    registry_version: str
    model_version: str

    def __init__(
        self,
        *,
        decision_idx: int,
        decision_time: np.datetime64,
        symbol: str,
        strategy_id: str,
        activation_context: str,
        side: Literal[-1, 1],
        expected_net_bps: float | None = None,
        expected_gross_bps: float,
        q10_net_bps: float | None = None,
        q10_gross_bps: float,
        q90_net_bps: float | None = None,
        q90_gross_bps: float,
        expected_holding_bars: int,
        quality_weight: float | None = None,
        reliability: float | None = None,
        registry_version: str,
        model_version: str,
    ) -> None:
        object.__setattr__(self, "decision_idx", int(decision_idx))
        object.__setattr__(self, "decision_time", decision_time)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "activation_context", activation_context)
        object.__setattr__(self, "side", side)
        object.__setattr__(
            self,
            "expected_net_bps",
            float(expected_gross_bps if expected_net_bps is None else expected_net_bps),
        )
        object.__setattr__(self, "expected_gross_bps", float(expected_gross_bps))
        object.__setattr__(
            self,
            "q10_net_bps",
            float(q10_gross_bps if q10_net_bps is None else q10_net_bps),
        )
        object.__setattr__(self, "q10_gross_bps", float(q10_gross_bps))
        object.__setattr__(
            self,
            "q90_net_bps",
            float(q90_gross_bps if q90_net_bps is None else q90_net_bps),
        )
        object.__setattr__(self, "q90_gross_bps", float(q90_gross_bps))
        object.__setattr__(self, "expected_holding_bars", int(expected_holding_bars))
        object.__setattr__(
            self,
            "quality_weight",
            float(quality_weight if quality_weight is not None else (reliability or 0.0)),
        )
        object.__setattr__(self, "registry_version", registry_version)
        object.__setattr__(self, "model_version", model_version)

    @property
    def reliability(self) -> float:
        return self.quality_weight


@dataclass(slots=True, frozen=True)
class ValidatedSignalBatch:
    events: tuple[ValidatedSignalEvent, ...]
    start_idx: int
    end_idx: int
    symbols: tuple[str, ...]
    registry_version: str
    model_version: str


@dataclass(slots=True, frozen=True, init=False)
class Layer1FoldReadiness:
    fold_id: int
    registry_source_end_idx: int
    outer_oos_start_idx: int
    outer_oos_end_idx: int
    ready_symbols: tuple[str, ...]
    matched_event_count: int
    unmatched_event_count: int
    realized_match_ratio: float
    unique_decision_count: int
    prediction_unique_count: int
    opportunity_ic: float | None
    opportunity_ic_tstat: float
    probe_bps: float
    probe_lcb_bps: float
    probe_series_bps: tuple[float, ...]
    effective_symbol_count: float
    passed: bool
    blockers: tuple[str, ...]
    _compat_ic_series: tuple[float, ...]

    def __init__(
        self,
        *,
        fold_id: int,
        registry_source_end_idx: int,
        outer_oos_start_idx: int,
        outer_oos_end_idx: int,
        ready_symbols: tuple[str, ...],
        matched_event_count: int = 0,
        unmatched_event_count: int = 0,
        realized_match_ratio: float = 0.0,
        unique_decision_count: int = 0,
        prediction_unique_count: int = 0,
        opportunity_ic: float | None = None,
        opportunity_ic_tstat: float = 0.0,
        probe_bps: float = 0.0,
        probe_lcb_bps: float = 0.0,
        probe_series_bps: tuple[float, ...] = (),
        effective_symbol_count: float = 0.0,
        passed: bool = False,
        blockers: tuple[str, ...] = (),
        valid_opportunity_timestamp_count: int | None = None,
        opportunity_ic_series: tuple[float, ...] | None = None,
        probe_gross_edge_series_bps: tuple[float, ...] | None = None,
    ) -> None:
        compat_ic_series = tuple(opportunity_ic_series or ())
        compat_probe_series = tuple(probe_series_bps or probe_gross_edge_series_bps or ())
        legacy_compat = (
            valid_opportunity_timestamp_count is not None
            or opportunity_ic_series is not None
            or probe_gross_edge_series_bps is not None
        )
        compat_match_count = (
            matched_event_count
            if matched_event_count
            else (valid_opportunity_timestamp_count if valid_opportunity_timestamp_count is not None else 0)
        )
        compat_unique_count = unique_decision_count if unique_decision_count else compat_match_count
        compat_pred_count = prediction_unique_count if prediction_unique_count else compat_match_count
        compat_ratio = float(realized_match_ratio)
        if legacy_compat and compat_ratio <= 0.0 and compat_match_count > 0:
            compat_ratio = 1.0
        compat_effective_symbol_count = (
            float(effective_symbol_count) if effective_symbol_count > 0.0 else float(len(ready_symbols))
        )
        compat_probe_lcb = float(probe_lcb_bps)
        if legacy_compat and compat_probe_lcb == 0.0:
            compat_probe_lcb = float(
                probe_bps
                if probe_bps != 0.0
                else float(np.mean(np.asarray(compat_probe_series, dtype=np.float64)))
            )
        object.__setattr__(self, "fold_id", int(fold_id))
        object.__setattr__(self, "registry_source_end_idx", int(registry_source_end_idx))
        object.__setattr__(self, "outer_oos_start_idx", int(outer_oos_start_idx))
        object.__setattr__(self, "outer_oos_end_idx", int(outer_oos_end_idx))
        object.__setattr__(self, "ready_symbols", tuple(ready_symbols))
        object.__setattr__(self, "matched_event_count", int(compat_match_count))
        object.__setattr__(self, "unmatched_event_count", int(unmatched_event_count))
        object.__setattr__(self, "realized_match_ratio", float(compat_ratio))
        object.__setattr__(self, "unique_decision_count", int(compat_unique_count))
        object.__setattr__(self, "prediction_unique_count", int(compat_pred_count))
        object.__setattr__(self, "opportunity_ic", opportunity_ic)
        object.__setattr__(self, "opportunity_ic_tstat", float(opportunity_ic_tstat))
        object.__setattr__(self, "probe_bps", float(probe_bps))
        object.__setattr__(self, "probe_lcb_bps", float(compat_probe_lcb))
        object.__setattr__(self, "probe_series_bps", compat_probe_series)
        object.__setattr__(self, "effective_symbol_count", float(compat_effective_symbol_count))
        object.__setattr__(self, "passed", bool(passed))
        object.__setattr__(self, "blockers", tuple(blockers))
        object.__setattr__(self, "_compat_ic_series", compat_ic_series)

    @property
    def valid_opportunity_timestamp_count(self) -> int:
        return self.matched_event_count

    @property
    def opportunity_ic_series(self) -> tuple[float, ...]:
        return self._compat_ic_series

    @property
    def probe_gross_edge_series_bps(self) -> tuple[float, ...]:
        return self.probe_series_bps


@dataclass(slots=True, frozen=True)
class Layer1GateCheck:
    key: str
    value: float
    threshold: float
    comparator: GateComparator
    passed: bool
    blocker: str | None = None


@dataclass(slots=True, frozen=True)
class Layer1GateReport:
    checks: tuple[Layer1GateCheck, ...]
    passed: bool
    blockers: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class Layer1InferenceArtifact:
    feature_schema: CandidateFeatureSchema
    model: RegimeConditionalEnsemble
    deployment_registry: QualifiedSignalRegistry
    baseline_by_key: dict[MatchedBaselineKey, float]
    l1_fit_end_idx: int
    model_version: str
    config_hash: str
