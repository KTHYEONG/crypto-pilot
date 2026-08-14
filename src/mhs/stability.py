"""Regime-split stability: the admission gate that sees sign flips.

The discovery gate scores a single window, so ``mom_168h`` (+0.615 in
2021-2023, then -0.690) looks normal. ``regime_split_stability`` splits a PnL
series at explicit regime boundaries and reports per-window Sharpe,
``min_window_sharpe``, ``sign_consistent`` (all finite windows share one strict
sign), and ``decay`` (last minus first window). The Stage 2 admission rule is
``sign_consistent=True AND min_window_sharpe > 0``
(docs/specs/mhs_multi_feature_alpha_architecture.md §2, Stage 2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

_PERIODS_PER_YEAR_1H = 365.0 * 24.0


@dataclass(frozen=True, slots=True)
class StabilityResult:
    """Regime-split stability of one PnL series.

    ``window_sharpes`` is ``(label, annualized Sharpe)`` per contiguous window
    including degenerate (non-finite) windows. ``sign_consistent`` is True only
    when every finite window Sharpe shares one strict sign; ``decay`` is the
    last window Sharpe minus the first.
    """

    window_sharpes: tuple[tuple[str, float], ...]
    min_window_sharpe: float
    sign_consistent: bool
    decay: float


def _window_ann_sharpe(segment: pd.Series, periods_per_year: float) -> float:
    values = segment.dropna()
    if len(values) < 2:
        return float("nan")
    sd = float(values.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float(values.mean() / sd * np.sqrt(periods_per_year))


def regime_split_stability(
    pnl: pd.Series,
    split_points: Sequence[pd.Timestamp],
    periods_per_year: float = _PERIODS_PER_YEAR_1H,
) -> StabilityResult:
    """Split ``pnl`` at ``split_points`` and measure per-window stability.

    Each split point starts a new contiguous window (``[split_i, split_{i+1})``
    in index space; windows are labeled ``window_0``..``window_N``. Per window
    the annualized Sharpe is ``mean/std(ddof=1)*sqrt(periods_per_year)``. A
    window with fewer than 2 finite observations or zero variance yields a
    non-finite Sharpe that is still reported in ``window_sharpes`` but EXCLUDED
    from ``min_window_sharpe`` and ``sign_consistent`` -- a single empty window
    can never manufacture a passing verdict. Raises ``ValueError`` on empty
    ``pnl``, empty or non-monotonic ``split_points``, or
    ``periods_per_year <= 0``.
    """
    if pnl.empty:
        raise ValueError("pnl must not be empty")
    if not split_points:
        raise ValueError("split_points must not be empty")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
    points = [pd.Timestamp(p) for p in split_points]
    for i in range(1, len(points)):
        if not points[i - 1] < points[i]:
            raise ValueError("split_points must be strictly ascending")

    boundaries = [pnl.index[0], *points, pnl.index[-1]]
    window_sharpes: list[tuple[str, float]] = []
    for i in range(len(points) + 1):
        start, end = boundaries[i], boundaries[i + 1]
        segment = pnl.loc[(pnl.index >= start) & (pnl.index < end)]
        window_sharpes.append((f"window_{i}", _window_ann_sharpe(segment, periods_per_year)))

    finite = [sharpe for _, sharpe in window_sharpes if np.isfinite(sharpe)]
    if finite:
        min_window_sharpe = min(finite)
        sign_consistent = all(sharpe > 0 for sharpe in finite) or all(
            sharpe < 0 for sharpe in finite
        )
    else:
        min_window_sharpe = float("nan")
        sign_consistent = False
    decay = window_sharpes[-1][1] - window_sharpes[0][1]
    return StabilityResult(
        window_sharpes=tuple(window_sharpes),
        min_window_sharpe=min_window_sharpe,
        sign_consistent=sign_consistent,
        decay=decay,
    )
