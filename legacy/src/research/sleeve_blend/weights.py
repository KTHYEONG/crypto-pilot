"""Component labels, symbol caps, and causal inverse-vol risk weights.

Owns the directional component label parsing, the numpy water-fill symbol cap,
and the causal risk-weight contract. No dependency on the execution modules.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

_LONG_SUFFIX = ":long"
_SHORT_SUFFIX = ":short"


def symbol_of_component(component: str) -> str:
    """Map a directional component label back to its symbol.

    Component labels encode the direction as ``"<SYMBOL>:long"`` /
    ``"<SYMBOL>:short"``; any other shape is a malformed contract input.
    """
    if component.endswith(_LONG_SUFFIX):
        return component[: -len(_LONG_SUFFIX)]
    if component.endswith(_SHORT_SUFFIX):
        return component[: -len(_SHORT_SUFFIX)]
    raise ValueError(f"malformed component label: {component}")


def component_labels(symbol: str) -> tuple[str, str]:
    """Long/short component labels for a directional sleeve symbol."""
    return (f"{symbol}{_LONG_SUFFIX}", f"{symbol}{_SHORT_SUFFIX}")


def _zero_weights(active_components: tuple[str, ...]) -> pd.Series:
    return pd.Series(0.0, index=list(active_components), dtype=np.float64)


def _cap_symbol_weights_np(
    weights: np.ndarray,
    symbol_ids: np.ndarray,
    n_symbols: int,
    max_symbol_weight: float,
) -> np.ndarray:
    """Numpy water-fill cap of per-component weights by symbol.

    Symbols whose aggregate free weight exceeds the cap are pinned at the cap
    and the remaining budget is split proportionally among the rest, so no
    symbol ever exceeds the cap (deterministic and convergent). When the cap is
    infeasible for the symbol count, the leftover budget is left unallocated
    rather than pushing any symbol over the cap.
    """
    agg = np.bincount(symbol_ids, weights=weights, minlength=n_symbols)
    order = np.argsort(-agg)
    final = np.zeros(n_symbols, dtype=np.float64)
    budget = 1.0
    i = 0
    while i < n_symbols:
        rem = order[i:]
        remaining_sum = float(agg[rem].sum())
        if remaining_sum <= 0:
            break
        count = 0
        for s in rem:
            if budget * agg[s] / remaining_sum <= max_symbol_weight:
                break
            count += 1
        if count == 0:
            final[rem] = budget * agg[rem] / remaining_sum
            break
        final[order[i : i + count]] = max_symbol_weight
        budget -= max_symbol_weight * count
        i += count

    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.divide(
            final[symbol_ids],
            np.where(agg[symbol_ids] > 0, agg[symbol_ids], 1.0),
        )
    return cast(np.ndarray, np.multiply(weights, scale))


def _symbol_ids_for(components: tuple[str, ...]) -> tuple[np.ndarray, int]:
    symbols = sorted({symbol_of_component(c) for c in components})
    lookup = {s: i for i, s in enumerate(symbols)}
    return (
        np.asarray([lookup[symbol_of_component(c)] for c in components], dtype=np.intp),
        len(symbols),
    )


def _cap_symbol_weights(
    weights: pd.Series,
    max_symbol_weight: float,
) -> pd.Series:
    """Cap each symbol's aggregated long+short weight and renormalize.

    Delegates to the numpy water-fill (see ``_cap_symbol_weights_np``) and
    returns a like-indexed Series, preserving the public contract surface.
    """
    symbol_ids, n_symbols = _symbol_ids_for(tuple(weights.index))
    capped = _cap_symbol_weights_np(
        weights.to_numpy(dtype=np.float64),
        symbol_ids,
        n_symbols,
        max_symbol_weight,
    )
    return pd.Series(capped, index=weights.index, dtype=np.float64)


def compute_causal_risk_weights(
    completed_returns: pd.DataFrame,
    active_components: tuple[str, ...],
    as_of: pd.Timestamp,
    history_days: int = 30,
    max_symbol_weight: float = 0.25,
) -> pd.Series:
    """Causal inverse-volatility risk weights over the trailing completed month.

    Uses strictly earlier marked returns (never the ``as_of`` bar or later
    data), weighting active components by ``1 / std`` of their completed
    30-day window. A symbol's aggregated long+short weight is capped at
    ``max_symbol_weight`` and the remainder is renormalized. An insufficient
    (non-full-month) history, fewer than two completed bars, or all
    zero/non-finite volatilities returns an all-zero weight vector so the
    candidate stays in cash.
    """
    if not isinstance(completed_returns.index, pd.DatetimeIndex):
        raise ValueError("completed_returns must have a DatetimeIndex")
    if not completed_returns.index.is_monotonic_increasing:
        raise ValueError("completed_returns index must be monotonic increasing")
    if len(active_components) == 0:
        raise ValueError("active_components must be non-empty")
    missing = [c for c in active_components if c not in completed_returns.columns]
    if missing:
        raise ValueError(f"active_components missing from returns: {missing}")
    if not isinstance(as_of, pd.Timestamp):
        raise ValueError("as_of must be a pd.Timestamp")
    if as_of.tzinfo is not None and completed_returns.index.tz is None:
        raise ValueError("as_of is tz-aware while returns index is tz-naive")
    if as_of.tzinfo is None and completed_returns.index.tz is not None:
        raise ValueError("as_of is tz-naive while returns index is tz-aware")
    if history_days < 1:
        raise ValueError(f"history_days must be >= 1, got {history_days}")
    if not 0.0 < max_symbol_weight <= 1.0:
        raise ValueError(
            f"max_symbol_weight must be in (0, 1], got {max_symbol_weight}"
        )

    window = completed_returns.loc[
        (completed_returns.index > as_of - pd.Timedelta(days=history_days))
        & (completed_returns.index < as_of)
    ]
    if len(window) < 2:
        return _zero_weights(active_components)
    if (as_of - window.index[0]).days < history_days - 1:
        return _zero_weights(active_components)

    vol = window[list(active_components)].std()
    valid = vol.notna() & (vol > 0)
    if not bool(valid.any()):
        return _zero_weights(active_components)

    weights = _zero_weights(active_components)
    weights.loc[valid.index[valid]] = 1.0 / vol[valid]
    weights = weights / float(weights.sum())
    return _cap_symbol_weights(weights, max_symbol_weight)


def _causal_weight_series(
    component_returns: pd.DataFrame,
    active_components: tuple[str, ...],
    history_days: int,
    max_symbol_weight: float,
) -> pd.DataFrame:
    """Vectorized causal inverse-vol weight series over the completed month.

    Mirrors ``compute_causal_risk_weights`` row by row (strictly-earlier
    completed returns, a full ``history_days`` window, inverse-vol among active
    components, per-symbol 0.25 cap + renormalization) using cumulative sums so
    a long sealed observation window computes in linear time.
    """
    idx = component_returns.index
    cols = list(active_components)
    x = component_returns[cols].to_numpy(dtype=np.float64)
    x_filled = np.where(np.isnan(x), 0.0, x)
    cum = np.cumsum(x_filled, axis=0)
    cum_sq = np.cumsum(x_filled * x_filled, axis=0)

    starts = np.asarray(
        idx.searchsorted(idx - pd.Timedelta(days=history_days), side="right")
    )
    n = len(idx)
    bar_pos = np.arange(n)
    counts = bar_pos - starts
    pos_mask = (bar_pos == 0)[:, None]
    cum_up_to_prev = np.where(pos_mask, 0.0, cum[bar_pos - 1])
    cum_sq_up_to_prev = np.where(pos_mask, 0.0, cum_sq[bar_pos - 1])
    start_mask = (starts == 0)[:, None]
    prev = np.where(start_mask, 0.0, cum[starts - 1])
    prev_sq = np.where(start_mask, 0.0, cum_sq[starts - 1])
    sums = cum_up_to_prev - prev
    sumsq = cum_sq_up_to_prev - prev_sq
    counts_col = np.maximum(counts, 1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sums / counts_col
        var = (sumsq - counts_col * mean * mean) / np.maximum(counts - 1, 1)[:, None]
    std = np.sqrt(np.clip(var, 0.0, None))

    span_days = (idx - pd.DatetimeIndex(idx[starts])).days.astype(np.float64)
    insufficient = (counts < 2) | (span_days < history_days - 1)
    std[insufficient] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / std
    valid = np.isfinite(inv)
    weights = np.zeros((n, len(cols)), dtype=np.float64)
    weights[valid] = inv[valid]
    row_sums = weights.sum(axis=1)
    active = row_sums > 0
    weights[active] = weights[active] / row_sums[active, None]

    symbol_ids, n_symbols = _symbol_ids_for(active_components)
    for i in np.flatnonzero(active):
        weights[i] = _cap_symbol_weights_np(
            weights[i], symbol_ids, n_symbols, max_symbol_weight
        )
    return pd.DataFrame(weights, index=idx, columns=cols)


__all__ = [
    "component_labels",
    "compute_causal_risk_weights",
    "symbol_of_component",
]
