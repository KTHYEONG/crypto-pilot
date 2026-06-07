from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SignalArchetype = Literal[
    "trend_continuation",
    "time_series_momentum",
    "mean_reversion",
    "forced_flow_reversal",
    "position_unwind",
    "carry_reversion",
    "beta_neutral_reversion",
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
    archetype: SignalArchetype | str = "mean_reversion"
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
    q10_return_r: NDArray[np.float64]
    q10_net_bps: NDArray[np.float64]
    q90_return_r: NDArray[np.float64]
    q90_net_bps: NDArray[np.float64]
    selection_score: NDArray[np.float64]
    kelly_fraction: NDArray[np.float64]
    validation_diagnostics: dict[str, float | int | str | bool] = field(default_factory=dict)

    def __init__(
        self,
        events: pd.DataFrame,
        p_pass: NDArray[np.float64],
        gate_enabled: bool = False,
        gate_threshold: float = 0.5,
        edge_source: EdgeSource = EdgeSource.PRIOR_ONLY,
        expected_return_r: NDArray[np.float64] | None = None,
        expected_net_bps: NDArray[np.float64] | None = None,
        q10_return_r: NDArray[np.float64] | None = None,
        q10_net_bps: NDArray[np.float64] | None = None,
        q90_return_r: NDArray[np.float64] | None = None,
        q90_net_bps: NDArray[np.float64] | None = None,
        selection_score: NDArray[np.float64] | None = None,
        kelly_fraction: NDArray[np.float64] | None = None,
        validation_diagnostics: dict[str, float | int | str | bool] | None = None,
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
            net_bps = kwargs.get("mu_gross_bps")
        if net_bps is None:
            net_bps = np.zeros(size, dtype=np.float64)
        object.__setattr__(self, "expected_net_bps", net_bps)

        ret_r = expected_return_r if expected_return_r is not None else net_bps / 25.0
        object.__setattr__(self, "expected_return_r", ret_r)

        q10_bps_val = q10_net_bps if q10_net_bps is not None else np.zeros(size, dtype=np.float64)
        object.__setattr__(self, "q10_net_bps", q10_bps_val)

        q10_r_val = q10_return_r if q10_return_r is not None else q10_bps_val / 25.0
        object.__setattr__(self, "q10_return_r", q10_r_val)

        q90_bps_val = q90_net_bps if q90_net_bps is not None else np.zeros(size, dtype=np.float64)
        object.__setattr__(self, "q90_net_bps", q90_bps_val)

        q90_r_val = q90_return_r if q90_return_r is not None else q90_bps_val / 25.0
        object.__setattr__(self, "q90_return_r", q90_r_val)

        sel_score = selection_score if selection_score is not None else kwargs.get("utility_score")
        if sel_score is None:
            sel_score = np.zeros(size, dtype=np.float64)
        object.__setattr__(self, "selection_score", sel_score)

        k_frac = kelly_fraction if kelly_fraction is not None else np.zeros(size, dtype=np.float64)
        object.__setattr__(self, "kelly_fraction", k_frac)

        val_diag = (
            validation_diagnostics
            if validation_diagnostics is not None
            else kwargs.get("selection_thresholds", {})
        )
        object.__setattr__(self, "validation_diagnostics", val_diag)

    @property
    def mu_gross_bps(self) -> NDArray[np.float64]:
        """Backward-compatible alias for expected_net_bps."""
        return self.expected_net_bps

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
    gate_model: Any | None = None
    edge_models: Any | None = None
    fit_set: Any | None = None
    calibration_set: Any | None = None
    oos_set: Any | None = None
    timing_profile: dict[str, float] | None = None
