"""L0 evidence failure-axis attribution. [ADR_20260708_L0_EDGE_FAILURE_ATTRIBUTION]"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

EdgeFailureAxis = Literal[
    "weak_gross_edge",
    "cost_dominated",
    "turnover_dominated",
    "statistically_unstable",
    "insufficient_sample",
    "heterogeneous_edge",
    "execution_model_mismatch",
    "unknown",
]

_KNOWN_AXES: frozenset[str] = frozenset(
    {
        "weak_gross_edge",
        "cost_dominated",
        "turnover_dominated",
        "statistically_unstable",
        "insufficient_sample",
        "heterogeneous_edge",
        "execution_model_mismatch",
        "unknown",
    }
)


@dataclass(slots=True, frozen=True)
class EdgeFailureAttribution:
    recipe_id: str
    timeframe: str
    family: str
    variant: str
    primary_axis: EdgeFailureAxis
    axes: tuple[EdgeFailureAxis, ...]
    diagnostic: str


def _classify_single_row(
    row: pd.Series,
    *,
    min_gross_lcb_bps: float,
    cost_drag_ratio_floor: float,
    weak_tstat_abs: float,
    high_turnover_per_year: float,
) -> tuple[str, tuple[str, ...], str]:
    axes: list[str] = []

    is_cost_dominated = False
    cost_drag = row.get("cost_drag_ratio", np.nan)
    if np.isfinite(cost_drag) and cost_drag > cost_drag_ratio_floor:
        is_cost_dominated = True

    is_weak_gross_edge = False
    gross_lcb = row.get("gross_lcb_bps", np.nan)
    if not np.isfinite(gross_lcb) or gross_lcb < min_gross_lcb_bps:
        is_weak_gross_edge = True

    if is_cost_dominated and cost_drag > 1.0:
        axes.append("cost_dominated")
        if is_weak_gross_edge:
            axes.append("weak_gross_edge")
    elif is_weak_gross_edge:
        axes.append("weak_gross_edge")
        if is_cost_dominated:
            axes.append("cost_dominated")
    elif is_cost_dominated:
        axes.append("cost_dominated")

    tstat = row.get("nw_tstat", np.nan)
    if np.isfinite(tstat) and abs(tstat) < weak_tstat_abs:
        axes.append("statistically_unstable")

    turnover = row.get("turnover_per_year", np.nan)
    if np.isfinite(turnover) and turnover > high_turnover_per_year:
        axes.append("turnover_dominated")

    eff_n = row.get("effective_n", np.nan)
    if np.isfinite(eff_n) and eff_n < 20.0:
        axes.append("insufficient_sample")

    if not axes:
        primary = "unknown"
        axes_list: tuple[str, ...] = ("unknown",)
        diag = "no detectable failure axis"
    else:
        primary = axes[0]
        axes_list = tuple(axes)
        parts = "; ".join(axes)
        diag = f"failure axes: {parts}"

    return primary, axes_list, diag


def classify_edge_failure_rows(
    evidence: pd.DataFrame,
    *,
    min_gross_lcb_bps: float = 0.0,
    cost_drag_ratio_floor: float = 0.60,
    weak_tstat_abs: float = 1.25,
    high_turnover_per_year: float = 180.0,
) -> pd.DataFrame:
    result = evidence.copy()

    primary_list: list[str] = []
    axes_list: list[tuple[str, ...]] = []
    diag_list: list[str] = []

    for _, row in evidence.iterrows():
        primary, axes, diag = _classify_single_row(
            row,
            min_gross_lcb_bps=min_gross_lcb_bps,
            cost_drag_ratio_floor=cost_drag_ratio_floor,
            weak_tstat_abs=weak_tstat_abs,
            high_turnover_per_year=high_turnover_per_year,
        )
        primary_list.append(primary)
        axes_list.append(axes)
        diag_list.append(diag)

    result["failure_axis"] = primary_list
    result["failure_axes"] = [",".join(a) for a in axes_list]
    result["failure_diagnostic"] = diag_list
    return result
