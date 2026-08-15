"""Additive time-series trend sleeve on an eligible-market basket.

This module builds a causal, self-vol-normalized time-series momentum position on an
equal-weight market basket, sized by an explicit gross budget and combined additively
with dollar-neutral books.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def market_basket_log_price(
    log_close: pd.DataFrame, eligible: pd.DataFrame,
) -> pd.Series:
    """Causal equal-weight market log-price index over the eligible roster.

    Cumulative sum of the cross-sectional mean of eligible-masked one-bar log
    returns (``log_close.diff().where(eligible).mean(axis=1).fillna(0.0).cumsum()``).
    Reads only bars at or before ``t``, so truncating the panel after bar ``k``
    leaves the first ``k`` values bit-identical (no forward information).
    Ineligible cells never contribute (masked before the mean).
    """
    if not log_close.index.equals(eligible.index) or list(log_close.columns) != list(eligible.columns):
        raise ValueError("log_close and eligible must be identically indexed and columned")
    bar_ret = log_close.diff().where(eligible)
    return bar_ret.mean(axis=1).fillna(0.0).cumsum()


def time_series_trend_position(
    basket_log_price: pd.Series,
    horizons_hours: tuple[int, ...],
    decision_grid: pd.DatetimeIndex,
) -> pd.Series:
    """Self-vol-normalized time-series momentum, ensemble-averaged and held.

    For each horizon ``h`` computes
    ``(logp - logp.shift(h)) / (logp.diff().rolling(h, min_periods=h).std() * sqrt(h))``
    clipped to ``[-1, 1]`` with non-finite cells filled ``0.0``, then returns the
    plain arithmetic mean over horizons, sampled onto ``decision_grid`` and
    forward-held onto the input index (piecewise-constant between grid stamps).
    The result is bounded in ``[-1, 1]``, finite everywhere, exactly ``0.0`` in
    the insufficient-history lead-in, and invariant to an overall rescale of the
    basket's volatility (self normalization -- no threshold or scaling constant
    is ever fit).
    """
    if not horizons_hours:
        raise ValueError("horizons_hours must not be empty")
    if any(h <= 0 for h in horizons_hours):
        raise ValueError(f"horizons_hours must all be > 0, got {horizons_hours}")
    if not decision_grid.is_monotonic_increasing:
        raise ValueError("decision_grid must be monotonically increasing")
    bar_ret = basket_log_price.diff()
    zs: list[pd.Series] = []
    for h in horizons_hours:
        numerator = basket_log_price - basket_log_price.shift(h)
        denominator = bar_ret.rolling(h, min_periods=h).std() * np.sqrt(h)
        zs.append((numerator / denominator).clip(-1.0, 1.0).fillna(0.0))
    position = pd.concat(zs, axis=1).mean(axis=1)
    sampled = position.reindex(decision_grid)
    held = sampled.ffill().reindex(position.index, method="ffill").fillna(0.0)
    return held.astype("float64")


def trend_sleeve_weights(
    position: pd.Series,
    execution_mask: pd.DataFrame,
    gross_budget: float,
    min_symbols: int = 8,
) -> pd.DataFrame:
    """Size the directional trend sleeve against a gross budget.

    Each mask-eligible cell of a row equals ``(1 / n_active) * position * gross_budget``.
    The result is DELIBERATELY not dollar-neutral -- the row sum equals
    ``gross_budget * position``, the net directional exposure the dollar-neutral
    architecture cannot express -- and the per-row gross is ``|position| * gross_budget
    <= gross_budget``, so the sleeve's risk is bounded by an explicit budget rather
    than competing for the alpha books' unit gross. Rows with fewer than
    ``min_symbols`` mask cells return all zeros (fail closed, never NaN).
    """
    if not (0.0 <= gross_budget <= 1.0):
        raise ValueError(f"gross_budget must be in [0.0, 1.0], got {gross_budget}")
    if min_symbols < 2:
        raise ValueError(f"min_symbols must be >= 2, got {min_symbols}")
    if not position.index.equals(execution_mask.index):
        raise ValueError("position and execution_mask must share an identical index")
    n_active = execution_mask.sum(axis=1)
    unit = execution_mask.astype("float64").div(n_active.where(n_active > 0), axis=0)
    out = unit.mul(position, axis=0).mul(gross_budget)
    enough = n_active >= min_symbols
    return out.where(enough, other=0.0).fillna(0.0)
