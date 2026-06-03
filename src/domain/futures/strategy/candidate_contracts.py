from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


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


@dataclass(slots=True, frozen=True)
class CandidateModelOutput:
    """Container for candidate model outputs."""

    events: pd.DataFrame
    p_pass: NDArray[np.float64]
    mu_gross_bps: NDArray[np.float64]
    mu_net_decision_bps: NDArray[np.float64]
    q10_net_bps: NDArray[np.float64]
    q90_net_bps: NDArray[np.float64]
    utility_score: NDArray[np.float64]
    selection_thresholds: dict[str, float | bool] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CandidatePortfolioResult:
    """Candidate portfolio output contract."""

    alpha_panel: pd.DataFrame
    target_weights_2d: NDArray[np.float64]
    selected_events: pd.DataFrame
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)
