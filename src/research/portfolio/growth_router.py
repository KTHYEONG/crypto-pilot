"""Pure, fail-closed walk-forward and context-router primitives for the growth engine.

The rolling admission protocol freezes identity, parameters, sleeve weights,
risk, and the context router before each frozen three-calendar-month deployment
segment.  Every function in this module is a pure transformation over
timestamps, market context, and per-sleeve lower-confidence evidence, so the
router can never read the outer deployment or symbol-holdout returns.  A sleeve
with non-positive lower confidence weight is CASH, and the router falls back to
CASH whenever no context-sleeve pair has a positive lower confidence bound.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Calendar-horizon contract for the walk-forward protocol: a deployment segment
#: may use only the immediately preceding 12 calendar months for discovery and
#: is applied to the following three calendar months without refitting.
DISCOVERY_MONTHS = 12
DEPLOYMENT_MONTHS = 3

#: The outer stitched deployment sequence must contain at least three six-month
#: equal-duration folds (18 calendar months) or the engine holds CASH.
MIN_DEPLOYMENT_FOLDS = 3
FOLD_MONTHS = 6

#: Decision-time context is formed over a trailing window of completed bars;
#: a context with fewer than ``MIN_CONTEXT_SAMPLES`` completed bars has no
#: minimum effective sample size and fails closed.
CONTEXT_WINDOW_BARS = 180
MIN_CONTEXT_SAMPLES = 120

_DAYS_PER_MONTH = 30.44


@dataclass(frozen=True, slots=True)
class GrowthSegment:
    """One non-overlapping discovery -> frozen deployment walk-forward window."""

    discovery_dates: tuple[pd.Timestamp, ...]
    deployment_dates: tuple[pd.Timestamp, ...]


def build_rolling_segments(
    dates: Sequence[pd.Timestamp],
    *,
    discovery_months: int = DISCOVERY_MONTHS,
    deployment_months: int = DEPLOYMENT_MONTHS,
) -> list[GrowthSegment]:
    """Chronological, non-overlapping ``discovery_months``/``deployment_months`` segments.

    A deployment window uses only the rebalance dates strictly inside its
    immediately preceding ``discovery_months`` calendar months.  A window with
    fewer than two discovery rebalance dates, or a trailing partial deployment
    window, is dropped fail-closed (the segment is CASH and contributes no
    out-of-sample returns).
    """
    if discovery_months < 1 or deployment_months < 1:
        raise ValueError("discovery_months and deployment_months must be >= 1")
    sorted_dates = list(pd.DatetimeIndex(sorted(dates)))
    segments: list[GrowthSegment] = []
    for start_idx in range(0, len(sorted_dates), deployment_months):
        deploy = sorted_dates[start_idx : start_idx + deployment_months]
        if len(deploy) < deployment_months:
            break
        discovery = [
            d for d in sorted_dates
            if d < deploy[0]
            and d >= deploy[0] - pd.DateOffset(months=discovery_months)
        ]
        if len(discovery) < 2:
            continue
        segments.append(GrowthSegment(tuple(discovery), tuple(deploy)))
    return segments


def enough_deployment_folds(
    segments: Sequence[GrowthSegment],
    *,
    min_folds: int = MIN_DEPLOYMENT_FOLDS,
    fold_months: int = FOLD_MONTHS,
) -> bool:
    """Fail-closed span check: the stitched deployment must admit ``min_folds`` folds."""
    if not segments:
        return False
    if min_folds < 1 or fold_months < 1:
        raise ValueError("min_folds and fold_months must be >= 1")
    elapsed_days = (
        segments[-1].deployment_dates[-1] - segments[0].deployment_dates[0]
    ).days
    months = float(elapsed_days) / _DAYS_PER_MONTH
    return bool(months >= min_folds * fold_months)


@dataclass(frozen=True, slots=True)
class ContextState:
    """Discretized pre-decision context: market return sign, vol regime, breadth."""

    market_ret_up: bool
    high_vol: bool
    wide_breadth: bool


def compute_context_features(
    market_returns: np.ndarray,
    breadth: np.ndarray,
    *,
    end_idx: int,
    window: int = CONTEXT_WINDOW_BARS,
) -> tuple[float, float, float]:
    """Trailing (exclusive of ``end_idx``) mean market return, realised vol, breadth.

    Returns ``(nan, nan, nan)`` when fewer than ``MIN_CONTEXT_SAMPLES`` completed
    bars are available so the caller fails closed on insufficient context.
    """
    start = max(0, int(end_idx) - window)
    market_slice = np.asarray(market_returns, dtype=np.float64)[start : int(end_idx)]
    breadth_slice = np.asarray(breadth, dtype=np.float64)[start : int(end_idx)]
    if len(market_slice) < MIN_CONTEXT_SAMPLES:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(market_slice)),
        float(np.std(market_slice)),
        float(np.mean(breadth_slice)),
    )


def context_state_for(
    features: tuple[float, float, float],
    *,
    vol_threshold: float,
    breadth_threshold: float,
) -> ContextState:
    """Partition context features into a state using discovery-derived thresholds."""
    mean_market_ret, vol, mean_breadth = features
    if not (
        np.isfinite(mean_market_ret)
        and np.isfinite(vol)
        and np.isfinite(mean_breadth)
    ):
        raise ValueError("context features must be finite")
    return ContextState(
        market_ret_up=mean_market_ret > 0.0,
        high_vol=vol > vol_threshold,
        wide_breadth=mean_breadth > breadth_threshold,
    )


def causal_router(sleeve_context_lcbs: Mapping[str, float]) -> str | None:
    """Allocate at most one admitted sleeve, or CASH when none is admissible.

    Only sleeves with a strictly positive, finite lower confidence bound for the
    current context are selectable; the highest such bound wins and ties are
    broken lexicographically by strategy id for exact determinism.  Returns
    ``None`` (CASH) when no context-sleeve pair has a positive LCB.
    """
    admissible: dict[str, float] = {
        sid: float(lcb)
        for sid, lcb in sleeve_context_lcbs.items()
        if np.isfinite(float(lcb)) and float(lcb) > 0.0
    }
    if not admissible:
        return None
    # Ties on the lower confidence bound are broken to the lexicographically
    # smallest strategy id so the allocation is exactly deterministic.
    return max(
        admissible,
        key=lambda sid: (admissible[sid], tuple(-ord(character) for character in sid)),
    )
