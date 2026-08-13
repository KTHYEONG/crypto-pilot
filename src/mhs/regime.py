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

from src.mhs.books import renormalize_within_mask


def causal_market_beta(
    log_price: pd.DataFrame,
    eligible: pd.DataFrame,
    lookback_bars: int,
    min_periods: int,
    clip_abs: float = 3.0,
) -> pd.DataFrame:
    """Rolling OLS beta of each symbol's one-bar log return on the market.

    The market return is the equal-weight mean of the eligible-universe one-bar
    returns; each symbol's beta is ``rolling cov(sym, market) / rolling var(market)``
    over ``lookback_bars`` with ``min_periods`` as the minimum sample, clipped to
    ``[-clip_abs, clip_abs]``. Causal: the rolling windows read only bars at or
    before ``t``. Cells with a zero or non-finite market variance are NaN, never
    inf (``var.where(var > 0)`` before division). Used by
    ``beta_neutralize_weights`` to build the parameter-free market-neutral book
    (docs/specs/mhs_alpha_engine.md §4, RC-4).
    """
    if lookback_bars < 2:
        raise ValueError(f"lookback_bars must be >= 2, got {lookback_bars}")
    if min_periods < 2:
        raise ValueError(f"min_periods must be >= 2, got {min_periods}")
    if min_periods > lookback_bars:
        raise ValueError(
            f"min_periods must be <= lookback_bars, got {min_periods} > {lookback_bars}"
        )
    if clip_abs <= 0:
        raise ValueError(f"clip_abs must be > 0, got {clip_abs}")
    if not log_price.index.equals(eligible.index) or list(log_price.columns) != list(eligible.columns):
        raise ValueError("log_price and eligible must be identically indexed and columned")
    rets = log_price.diff()
    market = rets.where(eligible).mean(axis=1)
    cov = rets.rolling(lookback_bars, min_periods=min_periods).cov(market)
    var = market.rolling(lookback_bars, min_periods=min_periods).var()
    beta = cov.div(var.where(var > 0), axis=0)
    return beta.clip(-clip_abs, clip_abs)


def beta_neutralize_weights(
    weights: pd.DataFrame,
    beta: pd.DataFrame,
    mask: pd.DataFrame,
    min_symbols: int,
) -> pd.DataFrame:
    """Project the book onto the subspace of the mask columns orthogonal to
    both the constant vector and ``beta``.

    Demeans ``beta`` over the mask columns (so the constant is removed), then
    removes the beta component ``sum(w * beta_c) / sum(beta_c ** 2)``, then
    restores dollar-neutral unit gross via ``renormalize_within_mask``. Both
    ``sum(w) == 0`` and ``sum(w * beta) == 0`` hold by construction on every
    qualifying row (docs/specs/mhs_alpha_engine.md §4, RC-4). Rows with fewer
    than ``min_symbols`` mask cells, a zero beta dispersion, or all-NaN beta
    return all zeros (fail closed, never NaN).
    """
    if min_symbols < 2:
        raise ValueError(f"min_symbols must be >= 2, got {min_symbols}")
    if not (weights.index.equals(beta.index) and weights.index.equals(mask.index)):
        raise ValueError("weights, beta, and mask must be identically indexed and columned")
    if not (list(weights.columns) == list(beta.columns) == list(mask.columns)):
        raise ValueError("weights, beta, and mask must be identically indexed and columned")
    wm = weights.where(mask, 0.0)
    bm = beta.where(mask)
    bc = bm.sub(bm.mean(axis=1), axis=0).fillna(0.0)
    denom = (bc**2).sum(axis=1)
    coef = (wm * bc).sum(axis=1).div(denom.where(denom > 0)).fillna(0.0)
    projected = wm - bc.mul(coef, axis=0)
    out = renormalize_within_mask(projected, mask, min_symbols)
    qualify = (mask.sum(axis=1) >= min_symbols) & (denom > 0)
    return out.where(qualify, other=0.0)


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


def trend_efficiency_scale(
    mean_efficiency_ratio: pd.Series,
    window_hours: int = 720,
    floor: float = 0.5,
) -> pd.Series:
    """De-risk slow_momentum's exposure in choppy (low efficiency-ratio) regimes.

    ``scale = clip(current / rolling_median(current, window_hours), floor, 1.0)``:
    mirrors ``_regime_cash_scale``'s ratio-to-own-trailing-history design (never
    levers above 1.0), with the ratio inverted relative to that sibling because
    ``efficiency_ratio``'s adverse direction is LOW (choppy/momentum-hostile),
    not HIGH like realized vol. A flat/insufficient-history window carries full
    exposure (never 0/0), matching ``_regime_cash_scale``'s same guarantee
    (docs/specs/mhs_fast_reversal_overlay_redesign.md §2.3).
    """
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"floor must be in (0, 1], got {floor}")
    if window_hours < 1:
        raise ValueError(f"window_hours must be >= 1, got {window_hours}")
    if mean_efficiency_ratio.empty:
        return pd.Series(1.0, index=mean_efficiency_ratio.index)
    median = mean_efficiency_ratio.rolling(
        window_hours, min_periods=min(48, window_hours),
    ).median()
    scale = mean_efficiency_ratio.div(median.where(median > 0))
    scale = scale.clip(lower=floor, upper=1.0)
    return scale.fillna(1.0)


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
