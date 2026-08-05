"""Causal discovery-anchored blend of XS alpha and the frozen Donchian baseline.

``xs_alpha_vol_weighted_v6``'s realized net returns and the frozen
single-symbol Donchian baseline's realized net returns are near-uncorrelated
(measured ``rho`` effectively zero), so blending them is the only lever found
so far that moves the Sharpe/t-stat *regime* rather than just its scale.
This module exposes the pure, deterministic recombination functions behind
that lever:

* :func:`select_baseline_blend_weight` picks a single scalar sleeve weight
  strictly on the discovery window (never reading qualification or holdout
  bars) by argmax annualized Sharpe over a small pre-registered grid -- the
  same discovery-only grid-search pattern as ``select_vol_target_window`` in
  :mod:`src.research.technical_experts.cross_sectional`.
* :func:`select_robust_baseline_blend_weight` is the worst-year-robust sibling:
  identical discovery-only contract, but scored by the *minimum* annualized
  Sharpe across the discovery window's distinct calendar years instead of the
  aggregate, so a grid point is only as good as its worst year.
* :func:`build_blended_ledger` applies that one fixed scalar over the *full*
  history -- the same discovery-select/full-history-apply pattern as
  ``solve_growth_optimal_risk``'s ``selected_risk`` -- compounding the blended
  net returns into an equity ledger and a realized-weight frame whose two
  columns are each leg's gross exposure scaled by its blend weight.
* :func:`apply_fixed_gross_leverage` is a pure linear gross-leverage overlay
  (``scale * net``, ``scale * weights``, elementwise, no path dependence) --
  the deliberate contrast to ``growth_sizing.apply_realised_risk_overlay``.

Both functions are pure (no I/O, no data loading), mirroring the separation of
alpha-construction functions from the application-layer loader in
``xs_trend_screen.py``. No cost or turnover formula is reimplemented here:
the inputs are already realized net returns and realized weight paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Frozen 4h calendar invariant, identical to
# ``GrowthSizingConfig.bars_per_year`` / ``_BARS_PER_YEAR`` in
# :mod:`src.research.technical_experts.cross_sectional`.
_BARS_PER_YEAR = 2190


def _discovery_common(
    xs_alpha_net: pd.Series,
    baseline_net: pd.Series,
    discovery_start: pd.Timestamp,
    discovery_end: pd.Timestamp,
) -> tuple[pd.Series, pd.Series]:
    """Restrict both legs to the inclusive discovery window and inner-join them.

    Fail-closed on the discovery-only contract: fewer than two common finite
    bars, or any non-finite value on the common discovery-window index, raises
    ``ValueError``. Qualification and holdout bars are never read.
    """
    xs_disc = xs_alpha_net[
        (xs_alpha_net.index >= discovery_start) & (xs_alpha_net.index <= discovery_end)
    ]
    bl_disc = baseline_net[
        (baseline_net.index >= discovery_start) & (baseline_net.index <= discovery_end)
    ]
    common = xs_disc.index.intersection(bl_disc.index)
    if len(common) < 2:
        raise ValueError(
            "fewer than 2 common bars remain in the discovery window; "
            "cannot select a blend weight"
        )
    a = xs_disc.reindex(common).astype(np.float64)
    b = bl_disc.reindex(common).astype(np.float64)
    if not np.isfinite(a.to_numpy()).all() or not np.isfinite(b.to_numpy()).all():
        raise ValueError(
            "xs_alpha_net and baseline_net must be finite (non-finite values "
            "found) on the common discovery-window index"
        )
    return a, b


def _blended_sharpe(
    a: pd.Series,
    b: pd.Series,
    xs_alpha_weight: float,
) -> float:
    """Annualized Sharpe of one grid blend, zero-variance safe (``quant.md``).

    A grid point whose blended std is exactly zero is not a ``ZeroDivisionError``:
    it is ``+inf`` Sharpe when the blended mean is positive (dominant, always
    selected) and ``-inf`` when the mean is non-positive.
    """
    blended = xs_alpha_weight * a + (1.0 - xs_alpha_weight) * b
    mean = float(blended.mean())
    std = float(blended.std())
    if std <= 0.0:
        return float("inf") if mean > 0.0 else float("-inf")
    return float(mean / std * np.sqrt(_BARS_PER_YEAR))


def select_baseline_blend_weight(
    xs_alpha_net: pd.Series,
    baseline_net: pd.Series,
    discovery_start: pd.Timestamp,
    discovery_end: pd.Timestamp,
    weight_grid: tuple[float, ...],
) -> float:
    """Select the discovery-only argmax-Sharpe blend weight on a fixed grid.

    Mirrors the discovery-only, grid-based (not matrix-inverted) selection of
    ``select_vol_target_window``: both net-return series are restricted to the
    inclusive ``[discovery_start, discovery_end]`` window and inner-joined on
    their common index; the annualized Sharpe (``bars_per_year=2190``) of every
    grid point is computed on that discovery-only common set, and the argmax
    grid value is returned. On an exact tie the *lowest* ``weight_grid`` value
    wins (mirrors ``solve_growth_optimal_risk``'s min-risk tie-break). The grid
    itself is the only thing iterated -- the return series are never
    row-looped.
    """
    if not weight_grid:
        raise ValueError("weight_grid must not be empty")
    if any(not 0.0 <= w <= 1.0 for w in weight_grid):
        raise ValueError("every weight_grid value must be within [0, 1]")

    a, b = _discovery_common(
        xs_alpha_net, baseline_net, discovery_start, discovery_end,
    )

    best_weight = weight_grid[0]
    best_sharpe = float("-inf")
    for w in weight_grid:
        sharpe = _blended_sharpe(a, b, w)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weight = w
    return best_weight

def select_robust_baseline_blend_weight(
    xs_alpha_net: pd.Series,
    baseline_net: pd.Series,
    discovery_start: pd.Timestamp,
    discovery_end: pd.Timestamp,
    weight_grid: tuple[float, ...],
) -> float:
    """Select the discovery-only worst-year-robust blend weight on a fixed grid.

    Sibling to :func:`select_baseline_blend_weight` with the same
    discovery-only inner-join contract, but scored by the *minimum* annualized
    Sharpe across the distinct calendar years spanned by the discovery window
    instead of the aggregate Sharpe -- a grid point is only as good as its
    worst year, the same robustness philosophy ``evaluate_xs_admission``
    already applies to its own ``annual_sub_sharpe``/``sharpe_floor`` gate.
    This is what stops the aggregate rule from being fooled by a single
    outlier year in one leg (the v8 defect). Fail-closed (``ValueError``) when
    the discovery window spans fewer than 2 distinct calendar years, because
    a per-year minimum is meaningless on a single-year window.
    """
    if not weight_grid:
        raise ValueError("weight_grid must not be empty")
    if any(not 0.0 <= w <= 1.0 for w in weight_grid):
        raise ValueError("every weight_grid value must be within [0, 1]")

    a, b = _discovery_common(
        xs_alpha_net, baseline_net, discovery_start, discovery_end,
    )
    years = np.unique(a.index.year.to_numpy(dtype=np.int64))
    if len(years) < 2:
        raise ValueError(
            "discovery window must span at least 2 distinct calendar years "
            "to select a worst-year-robust blend weight"
        )

    best_weight = weight_grid[0]
    best_min = float("-inf")
    for w in weight_grid:
        yearly: list[float] = []
        for year in years:
            in_year = a.index.year == int(year)
            yearly.append(_blended_sharpe(a[in_year], b[in_year], w))
        min_sharpe = min(yearly)
        if min_sharpe > best_min:
            best_min = min_sharpe
            best_weight = w
    return best_weight


def build_blended_ledger(
    xs_alpha_net: pd.Series,
    xs_alpha_realized_weights: pd.DataFrame,
    baseline_net: pd.Series,
    baseline_realized_weight: pd.Series,
    xs_alpha_weight: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply one fixed scalar weight over the full history and compound it.

    ``blended_net[t] = xs_alpha_weight * xs_alpha_net[t] + (1 - xs_alpha_weight)
    * baseline_net[t]`` on every bar; the equity is the cumulative product of
    ``(1 + blended_net)`` starting from 1.0. The returned weights frame has
    exactly two columns in order ``["xs_alpha", "baseline"]``, where the
    ``xs_alpha`` column is ``xs_alpha_weight * sum(abs(weights), axis=1)`` and
    the ``baseline`` column is ``(1 - xs_alpha_weight) * abs(baseline_realized_weight)``
    -- the realized-weight path that feeds ``count_closed_trades`` inside
    ``evaluate_xs_reliability`` as a coarse sleeve-rebalance proxy. Fail-closed
    on a non-[0, 1] weight, a misaligned index, or any non-finite input.
    """
    if not 0.0 <= xs_alpha_weight <= 1.0:
        raise ValueError(
            f"xs_alpha_weight must be within [0, 1], got {xs_alpha_weight}"
        )
    index = xs_alpha_net.index
    if not (
        isinstance(index, pd.DatetimeIndex)
        and index.equals(baseline_net.index)
        and index.equals(xs_alpha_realized_weights.index)
        and index.equals(baseline_realized_weight.index)
    ):
        raise ValueError(
            "xs_alpha_net, xs_alpha_realized_weights, baseline_net, and "
            "baseline_realized_weight must share an identical DatetimeIndex"
        )

    net_arrays = [
        xs_alpha_net.to_numpy(dtype=np.float64),
        baseline_net.to_numpy(dtype=np.float64),
    ]
    if not all(np.isfinite(arr).all() for arr in net_arrays):
        raise ValueError("xs_alpha_net and baseline_net must be finite")

    blended_net = (
        xs_alpha_weight * xs_alpha_net.to_numpy(dtype=np.float64)
        + (1.0 - xs_alpha_weight) * baseline_net.to_numpy(dtype=np.float64)
    )
    equity = pd.Series(
        np.cumprod(1.0 + blended_net),
        index=index,
        name="blended_equity",
        dtype=np.float64,
    )

    xs_alpha_arr = xs_alpha_realized_weights.to_numpy(dtype=np.float64)
    baseline_arr = baseline_realized_weight.to_numpy(dtype=np.float64)
    if not np.isfinite(xs_alpha_arr).all() or not np.isfinite(baseline_arr).all():
        raise ValueError("realized-weight inputs must be finite")

    weights = pd.DataFrame(
        {
            "xs_alpha": xs_alpha_weight * np.abs(xs_alpha_arr).sum(axis=1),
            "baseline": (1.0 - xs_alpha_weight) * np.abs(baseline_arr),
        },
        index=index,
    )
    return equity, weights

def apply_fixed_gross_leverage(
    net: pd.Series,
    weights: pd.DataFrame,
    scale: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Pure linear gross-leverage overlay on net returns and realised weights.

    ``scaled_net = scale * net`` and ``scaled_weights = scale * weights``
    elementwise -- no path dependence and no drawdown ladder, the deliberate
    contrast to ``growth_sizing.apply_realised_risk_overlay`` whose
    path-dependent ladder was measured to *hurt* LCB90. Fail-closed on a
    non-finite or non-positive ``scale``, a misaligned index, or any
    non-finite input, mirroring ``apply_realised_risk_overlay``'s validation
    minus the ladder step.
    """
    if not isinstance(net.index, pd.DatetimeIndex) or not isinstance(weights.index, pd.DatetimeIndex):
        raise ValueError("net and weights must have a DatetimeIndex")
    if not net.index.equals(weights.index):
        raise ValueError("net and weights must share an identical index")
    if not net.index.is_monotonic_increasing:
        raise ValueError("net index must be monotonic increasing")
    values = net.to_numpy(dtype=np.float64)
    w_arr = weights.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(w_arr).all():
        raise ValueError("net and weights must contain only finite values")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and > 0, got {scale}")

    return (
        pd.Series(scale * values, index=net.index, dtype=np.float64),
        pd.DataFrame(
            scale * w_arr, index=weights.index, columns=weights.columns, dtype=np.float64,
        ),
    )
