"""Market-wide regime features: fixed-reference-basket trend and drawdown.

Unlike ``horizons.py`` (per-symbol features), these aggregate a caller-supplied
FIXED basket of symbols into market-level regime proxies. The basket is never
derived from a time-varying eligibility panel -- composition drift would
confound the measure (docs/specs/mhs_momentum_strategy_redesign_review.md
§3.2). Both functions are causal (rolling/shift over past rows only) and pure:
they depend solely on ``log_price`` rows at or before the current index.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def reference_basket_trend(
    log_price: pd.DataFrame, reference_symbols: Sequence[str], horizon_bars: int
) -> pd.Series:
    """Horizon log-return of a fixed reference basket (causal lookback).

    Identical construction to ``horizons.horizon_log_return`` applied to the
    mean of a fixed symbol subset instead of every column independently.
    """
    if horizon_bars < 1:
        raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
    _validate_reference_symbols(log_price, reference_symbols)
    basket = log_price[list(reference_symbols)].mean(axis=1)
    return basket - basket.shift(horizon_bars)


def reference_basket_drawdown(
    log_price: pd.DataFrame, reference_symbols: Sequence[str], lookback_bars: int
) -> pd.Series:
    """Trailing drawdown of a fixed reference basket (always <= 0.0, causal).

    ``rolling(max, min_periods=1)`` never looks ahead, so the result uses only
    past bars and equals 0.0 at/after each running high.
    """
    if lookback_bars < 1:
        raise ValueError(f"lookback_bars must be >= 1, got {lookback_bars}")
    _validate_reference_symbols(log_price, reference_symbols)
    basket = log_price[list(reference_symbols)].mean(axis=1)
    trailing_high = basket.rolling(lookback_bars, min_periods=1).max()
    return basket - trailing_high


def crash_regime_tilt_weights(
    rank_neutral_weights: pd.DataFrame,
    log_price: pd.DataFrame,
    eligible: pd.DataFrame,
    reference_symbols: Sequence[str],
    horizon_bars: int,
    alpha: float,
    min_symbols: int = 8,
) -> pd.DataFrame:
    """Convex-blend a dollar-neutral book with a market-direction tilt.

    ``alpha`` in ``[0.0, 1.0]`` is the fraction of unit gross reallocated from
    ``rank_neutral_weights`` to a uniform directional tilt spread evenly
    across the ``eligible`` roster, scaled by a causal, self-vol-normalized,
    [-1, 1]-clipped reference-basket trend z-score (``reference_basket_trend``
    divided by its own trailing realized std over ``horizon_bars`` -- no
    separate arbitrary crash threshold). Both inputs are unit-gross books, so
    by the triangle inequality the convex combination never exceeds unit gross
    (``abs(weights).sum(axis=1) <= 1.0``) on every row -- equality only when
    ``rank_neutral_weights`` and the tilt happen to share the same sign per
    column; a genuine dollar-neutral book has shorts opposing the tilt, so
    gross typically shrinks (the tilt offsets shorts rather than amplifying
    them, the conservative direction). ``alpha=0.0`` returns
    ``rank_neutral_weights`` unchanged (byte-identical passthrough); there is
    no baked-in "recommended" alpha -- see
    docs/specs/mhs_crash_regime_tilt_overlay.md §4-5 for why this is a
    risk-budget policy parameter, not a fitted one.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
    if not rank_neutral_weights.index.equals(log_price.index):
        raise ValueError("rank_neutral_weights and log_price must share an identical index")
    if not rank_neutral_weights.index.equals(eligible.index) or list(rank_neutral_weights.columns) != list(eligible.columns):
        raise ValueError("rank_neutral_weights and eligible must be identically indexed and columned")
    if min_symbols < 2:
        raise ValueError(f"min_symbols must be >= 2, got {min_symbols}")
    if alpha == 0.0:
        return rank_neutral_weights
    trend = reference_basket_trend(log_price, reference_symbols, horizon_bars)
    trend_vol = trend.rolling(horizon_bars, min_periods=horizon_bars).std()
    trend_z = (trend / trend_vol.where(trend_vol > 0)).clip(-1.0, 1.0).fillna(0.0)
    n_active = eligible.sum(axis=1)
    tilt = (
        eligible.astype("float64")
        .div(n_active.where(n_active > 0), axis=0)
        .mul(trend_z, axis=0)
        .fillna(0.0)
    )
    enough = n_active >= min_symbols
    tilt = tilt.where(enough, 0.0)
    return (1.0 - alpha) * rank_neutral_weights + alpha * tilt


def _validate_reference_symbols(
    log_price: pd.DataFrame, reference_symbols: Sequence[str]
) -> None:
    if len(reference_symbols) == 0:
        raise ValueError("reference_symbols must not be empty")
    missing = [s for s in reference_symbols if s not in log_price.columns]
    if missing:
        raise ValueError(
            "reference_symbols not present in log_price.columns: "
            + ", ".join(map(str, missing))
        )
