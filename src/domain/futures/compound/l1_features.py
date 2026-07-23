from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.contracts import (
    CausalFold,
    ForecastFrame,
    MarketFeatureCube,
    MultiscaleAlphaDefinition,
)

_logger = logging.getLogger(__name__)


class InsufficientCoverageError(RuntimeError):
    ...


def _ewm(arr: NDArray[np.float64], span: int) -> NDArray[np.float64]:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(arr)
    out[:] = np.nan
    if arr.shape[0] == 0:
        return out
    out[0] = arr[0]
    for t in range(1, arr.shape[0]):
        out[t] = alpha * arr[t] + (1 - alpha) * out[t - 1]
    return out


def _robust_z_score(x: NDArray[np.float64]) -> NDArray[np.float64]:
    med = np.nanmedian(x, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=0, keepdims=True)
    mad = np.where(mad < 1e-12, 1e-12, mad)
    result: NDArray[np.float64] = (x - med) / (1.4826 * mad)
    return result


def _shifted_log_return(
    close: NDArray[np.float64], lookback_bars: int,
) -> NDArray[np.float64]:
    ret = np.full_like(close, np.nan, dtype=np.float64)
    if close.shape[0] <= lookback_bars:
        return ret
    mask = (close[:-lookback_bars] > 0) & (close[lookback_bars:] > 0)
    if np.any(mask):
        ret[lookback_bars:] = np.where(
            mask,
            np.log(close[lookback_bars:] / close[:-lookback_bars]),
            np.nan,
        )
    return ret


def _ts_trend_score(
    close: NDArray[np.float64], lookback_hours: tuple[int, ...],
) -> NDArray[np.float64]:
    lb_bars = [max(1, lb) for lb in lookback_hours]
    scores_list: list[NDArray[np.float64]] = []
    for lb in lb_bars:
        ret = _shifted_log_return(close, lb)
        vol = np.abs(_ewm(ret, lb))
        vol = np.where(vol < 1e-12, 1e-12, vol)
        scores_list.append(ret / vol)
    if len(scores_list) > 1:
        stack = np.dstack(scores_list)
        result: NDArray[np.float64] = np.nansum(stack, axis=2)
    else:
        result = scores_list[0]
    return result


def _xs_resmom_score(
    close: NDArray[np.float64], lookback_hours: tuple[int, ...],
) -> NDArray[np.float64]:
    lb = max(1, lookback_hours[0])
    ret = _shifted_log_return(close, lb)
    return _robust_z_score(ret)


def _breakout_score(
    high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64],
    lookback_hours: tuple[int, ...],
) -> NDArray[np.float64]:
    lb = max(1, lookback_hours[0])
    mid = (high + low) / 2.0
    roll_mid = _ewm(mid, lb)
    prev_close = np.roll(close, 1, axis=0)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    tr[0] = high[0] - low[0]
    atr = _ewm(tr, lb)
    atr = np.where(atr < 1e-12, 1e-12, atr)
    return (close - roll_mid) / atr


def _carry_funding_event_score(
    funding: NDArray[np.float64], premium: NDArray[np.float64],
) -> NDArray[np.float64]:
    combined = funding + premium
    n = combined.shape[0]
    ewm_c = _ewm(combined, min(n, 168))
    z = _robust_z_score(ewm_c)
    return -z


def _basis_reversion_score(
    mark: NDArray[np.float64], index_arr: NDArray[np.float64],
) -> NDArray[np.float64]:
    idx_safe = np.where(index_arr > 0, index_arr, 1.0)
    basis = (mark - index_arr) / idx_safe
    z = _robust_z_score(basis)
    return -z


def _flow_imbalance_score(
    taker_buy_quote: NDArray[np.float64], quote_volume: NDArray[np.float64],
) -> NDArray[np.float64]:
    vol_safe = np.where(quote_volume > 0, quote_volume, 1.0)
    imbalance = (taker_buy_quote * 2.0 - quote_volume) / vol_safe
    z = _robust_z_score(imbalance)
    return z


def build_causal_forecasts(
    *,
    market: MarketFeatureCube,
    catalog: Sequence[MultiscaleAlphaDefinition],
    folds: tuple[CausalFold, ...],
) -> tuple[ForecastFrame, ...]:
    for fold in folds:
        if fold.fit_end_exclusive > fold.oos_start:
            msg = f"fold {fold.fold_id}: fit_end {fold.fit_end_exclusive} > oos_start {fold.oos_start}"
            raise InsufficientCoverageError(msg)

    n_bars = market.timestamps_ns.size
    n_syms = len(market.symbols)
    frames: list[ForecastFrame] = []

    close_raw = market.fields_2d.get("close", None)
    close = close_raw.astype(np.float64) if close_raw is not None else None
    high_raw = market.fields_2d.get("high", None)
    high = high_raw.astype(np.float64) if high_raw is not None else None
    low_raw = market.fields_2d.get("low", None)
    low = low_raw.astype(np.float64) if low_raw is not None else None
    funding_raw = market.fields_2d.get("funding", None)
    funding = funding_raw.astype(np.float64) if funding_raw is not None else None
    premium_raw = market.fields_2d.get("premium", None)
    premium = premium_raw.astype(np.float64) if premium_raw is not None else None
    mark_raw = market.fields_2d.get("mark", None)
    mark = mark_raw.astype(np.float64) if mark_raw is not None else None
    index_raw = market.fields_2d.get("index", None)
    index_arr = index_raw.astype(np.float64) if index_raw is not None else None
    taker_buy_raw = market.fields_2d.get("taker_buy_quote", None)
    taker_buy = taker_buy_raw.astype(np.float64) if taker_buy_raw is not None else None
    volume_raw = market.fields_2d.get("quote_volume", None)
    volume = volume_raw.astype(np.float64) if volume_raw is not None else None

    for recipe in catalog:
        required = set(recipe.required_fields)
        available = set(market.fields_2d.keys())
        if not required.issubset(available):
            _logger.warning("recipe %s: missing fields %s", recipe.recipe_id, required - available)
            raw_arr = np.full((n_bars, n_syms), np.nan, dtype=np.float64)
        else:
            if recipe.family == "trend":
                if close is not None and close.ndim == 2:
                    raw_arr = _ts_trend_score(close, recipe.lookback_hours)
                else:
                    raw_arr = np.full((n_bars, n_syms), np.nan, dtype=np.float64)
            elif recipe.family == "residual_momentum":
                if close is not None:
                    raw_arr = _xs_resmom_score(close, recipe.lookback_hours)
                else:
                    raw_arr = np.full((n_bars, n_syms), np.nan, dtype=np.float64)
            elif recipe.family == "breakout":
                if high is not None and low is not None and close is not None:
                    raw_arr = _breakout_score(high, low, close, recipe.lookback_hours)
                else:
                    raw_arr = np.full((n_bars, n_syms), np.nan, dtype=np.float64)
            elif recipe.family == "carry":
                if funding is not None and premium is not None:
                    raw_arr = _carry_funding_event_score(funding, premium)
                else:
                    raw_arr = np.full((n_bars, n_syms), np.nan, dtype=np.float64)
            elif recipe.family == "basis_reversion":
                if mark is not None and index_arr is not None:
                    raw_arr = _basis_reversion_score(mark, index_arr)
                else:
                    raw_arr = np.full((n_bars, n_syms), np.nan, dtype=np.float64)
            elif recipe.family in ("taker_flow", "flow_oi"):
                if taker_buy is not None and volume is not None:
                    raw_arr = _flow_imbalance_score(taker_buy, volume)
                else:
                    raw_arr = np.full((n_bars, n_syms), np.nan, dtype=np.float64)
            elif recipe.family == "reversal":
                if close is not None:
                    raw_arr = -_robust_z_score(_shifted_log_return(close, max(1, recipe.lookback_hours[0])))
                else:
                    raw_arr = np.full((n_bars, n_syms), np.nan, dtype=np.float64)
            else:
                raw_arr = np.zeros((n_bars, n_syms), dtype=np.float64)

        eligible = market.eligible_2d
        valid_arr = eligible & np.isfinite(raw_arr)
        scores = np.where(valid_arr, raw_arr, np.nan).astype(np.float32)
        valid = valid_arr

        frames.append(ForecastFrame(
            timestamps_ns=market.timestamps_ns,
            symbols=market.symbols,
            recipe_id=recipe.recipe_id,
            scores_2d=scores,
            valid_2d=valid,
        ))

    _logger.info("built %d causal forecast frames", len(frames))
    return tuple(frames)
