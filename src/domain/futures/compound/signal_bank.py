from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

import numpy as np
import psutil
from numba import get_num_threads as _numba_get_num_threads
from numba import njit, prange
from numba import set_num_threads as _numba_set_num_threads
from numpy.typing import NDArray

from src.domain.futures.compound.alpha_catalog import build_canonical_alpha_catalog
from src.domain.futures.compound.contracts import (
    InsufficientCoverageError,
    MultiTimeframeBars,
    RawSignalPanel,
    SignalDescriptor,
)

_logger = logging.getLogger(__name__)

# P3 wiring: bars = build_multi_timeframe_bars(market); panel = build_raw_signal_panel(bars, eligible_2d=eligible_4h)

_numba_fallback_count: int = 0


@njit(cache=True, parallel=True)  # type: ignore[untyped-decorator]
def _rolling_mad_z_numba_kernel(
    arr: NDArray[np.float64],
    window: int,
    min_periods: int,
) -> NDArray[np.float64]:
    n_t, n_s = arr.shape
    z = np.full((n_t, n_s), np.nan, dtype=np.float64)
    for s in prange(n_s):
        max_window = min(window, n_t)
        buf = np.empty(max_window, dtype=np.float64)
        dev_buf = np.empty(max_window, dtype=np.float64)
        for t in range(min_periods - 1, n_t):
            start = max(0, t - window + 1)
            n_valid = 0
            for i in range(start, t + 1):
                v = arr[i, s]
                if np.isfinite(v):
                    buf[n_valid] = v
                    n_valid += 1
            if n_valid == 0:
                continue
            buf_view = buf[:n_valid]
            buf_view.sort()
            if n_valid % 2 == 1:
                med = buf_view[n_valid // 2]
            else:
                med = (buf_view[n_valid // 2 - 1] + buf_view[n_valid // 2]) / 2.0
            for i in range(n_valid):
                dev_buf[i] = np.abs(buf_view[i] - med)
            dev_view = dev_buf[:n_valid]
            dev_view.sort()
            if n_valid % 2 == 1:
                mad = dev_view[n_valid // 2]
            else:
                mad = (dev_view[n_valid // 2 - 1] + dev_view[n_valid // 2]) / 2.0
            if mad < 1e-12:
                continue
            z[t, s] = (arr[t, s] - med) / (1.4826 * mad)
    return z


_SPEED_LADDER_8: tuple[tuple[str, int], ...] = (
    ("fast", 24),
    ("medium", 72),
    ("moderate", 144),
    ("slow", 216),
    ("very_slow", 432),
    ("ultra_slow", 864),
    ("super_slow", 1728),
    ("extreme_slow", 3456),
)

_SPEED_LADDER_6: tuple[tuple[str, int], ...] = (
    ("fast", 24),
    ("medium", 72),
    ("moderate", 144),
    ("slow", 216),
    ("very_slow", 432),
    ("ultra_slow", 864),
)


def _ewm_2d(arr: NDArray[np.float32 | np.float64], span: int) -> NDArray[np.float64]:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(arr, dtype=np.float64)
    out[:] = np.nan
    if arr.shape[0] == 0:
        return out
    out[0] = np.where(np.isfinite(arr[0]), arr[0], 0.0)
    for t in range(1, arr.shape[0]):
        prev = out[t - 1]
        curr = np.where(np.isfinite(arr[t]), arr[t], prev)
        out[t] = alpha * curr + (1.0 - alpha) * prev
    return out


def _ewm_vol(arr: NDArray[np.float32 | np.float64], span: int) -> NDArray[np.float64]:
    sq = arr.astype(np.float64, copy=False) ** 2
    mean_sq = _ewm_2d(sq, span)
    mean = _ewm_2d(arr, span)
    var = mean_sq - mean ** 2
    var = np.maximum(var, 0.0)
    return np.sqrt(var)


def _log_return(close: NDArray[np.float32], lb_bars: int) -> NDArray[np.float64]:
    ret = np.full((close.shape[0], close.shape[1]), np.nan, dtype=np.float64)
    if close.shape[0] <= lb_bars:
        return ret
    prev = close[:-lb_bars].astype(np.float64)
    curr = close[lb_bars:].astype(np.float64)
    mask = (prev > 0) & (curr > 0)
    if np.any(mask):
        ret[lb_bars:] = np.where(mask, np.log(curr / prev), np.nan)
    return ret


def _rolling_mad_z_numpy(arr: NDArray[np.float64], window: int, min_periods: int) -> NDArray[np.float64]:
    n_t, _ = arr.shape
    z = np.full_like(arr, np.nan, dtype=np.float64)
    for t in range(min_periods - 1, n_t):
        start = max(0, t - window + 1)
        chunk = arr[start:t + 1]
        med = np.nanmedian(chunk, axis=0)
        mad = np.nanmedian(np.abs(chunk - med), axis=0)
        mad = np.where(mad < 1e-12, np.nan, mad)
        z[t] = (arr[t] - med) / (1.4826 * mad)
    return z


def _rolling_mad_z(arr: NDArray[np.float64], window: int, min_periods: int) -> NDArray[np.float64]:
    global _numba_fallback_count
    try:
        arr_contig = np.ascontiguousarray(arr, dtype=np.float64)
        return cast(
            NDArray[np.float64],
            _rolling_mad_z_numba_kernel(arr_contig, window, min_periods),
        )
    except Exception:
        _numba_fallback_count += 1
        _logger.warning("[SYS] _rolling_mad_z numba kernel failed, falling back to numpy")
        return _rolling_mad_z_numpy(arr, window, min_periods)


def _compute_trend_ema(
    close: NDArray[np.float32], high: NDArray[np.float32], low: NDArray[np.float32],
    lb_bars: int,
) -> NDArray[np.float64]:
    close_f64 = close.astype(np.float64)
    high_f64 = high.astype(np.float64)
    low_f64 = low.astype(np.float64)

    ema_fast = _ewm_2d(close_f64, lb_bars)
    ema_slow = _ewm_2d(close_f64, lb_bars * 3)

    prev_close = np.roll(close_f64, 1, axis=0)
    prev_close[0] = close_f64[0]
    tr = np.maximum(
        high_f64 - low_f64,
        np.maximum(
            np.abs(high_f64 - prev_close),
            np.abs(low_f64 - prev_close),
        ),
    )
    atr = _ewm_2d(tr, lb_bars)
    atr = np.where(atr < 1e-12, 1e-12, atr)
    return (ema_fast - ema_slow) / atr


def _compute_momentum_ts(close: NDArray[np.float32], lb_bars: int) -> NDArray[np.float64]:
    log_ret = _log_return(close, lb_bars)
    vol = _ewm_vol(log_ret, max(lb_bars, 42))
    vol = np.maximum(vol, 1e-6)
    result: NDArray[np.float64] = log_ret / (vol * np.sqrt(lb_bars))
    return result


def _compute_breakout_donchian(
    high: NDArray[np.float32], low: NDArray[np.float32], close: NDArray[np.float32],
    lb_bars: int,
) -> NDArray[np.float64]:
    n_t = close.shape[0]
    raw = np.full((n_t, close.shape[1]), np.nan, dtype=np.float64)
    for t in range(lb_bars - 1, n_t):
        start = max(0, t - lb_bars + 1)
        hi = np.max(high[start:t + 1], axis=0)
        lo = np.min(low[start:t + 1], axis=0)
        mid = (hi + lo) / 2.0
        half_range = 0.5 * (hi - lo) + 1e-12
        raw[t] = (close[t].astype(np.float64) - mid) / half_range
    return raw


def _compute_reversal_st(close: NDArray[np.float32], lb_bars: int) -> NDArray[np.float64]:
    log_ret = _log_return(close, lb_bars)
    vol = _ewm_vol(log_ret, max(lb_bars, 42))
    vol = np.maximum(vol, 1e-6)
    return -log_ret / vol


def _subsample_to_4h(arr_1h: NDArray[np.float32 | np.float64], n_4h: int) -> NDArray[np.float64]:
    close_idx = np.arange(3, 4 * n_4h, 4)
    if close_idx[-1] >= arr_1h.shape[0]:
        n_avail = arr_1h.shape[0] // 4
        close_idx = np.arange(3, 4 * n_avail, 4)
    result: NDArray[np.float64] = arr_1h[close_idx].astype(np.float64, copy=False)
    return result


def _compute_carry_funding(
    funding: NDArray[np.float32], premium: NDArray[np.float32],
    lookback_hours: int,
) -> NDArray[np.float64]:
    combined = (funding + premium).astype(np.float64)
    ewm_combined = _ewm_2d(combined, max(lookback_hours, 42))
    return ewm_combined.astype(np.float64, copy=False)


def _compute_basis_gap(
    mark: NDArray[np.float32], index_arr: NDArray[np.float32], lookback_hours: int,
) -> NDArray[np.float64]:
    idx_safe = np.where(index_arr > 0, index_arr.astype(np.float64), 1.0)
    basis = mark.astype(np.float64) / idx_safe - 1.0
    return _ewm_2d(basis, max(lookback_hours, 42))


def _compute_smart_money_divergence(
    top_trader_ratio: NDArray[np.float32], retail_ratio: NDArray[np.float32], lb_bars: int,
) -> NDArray[np.float64]:
    mask = (top_trader_ratio > 0) & np.isfinite(top_trader_ratio) & (retail_ratio > 0) & np.isfinite(retail_ratio)
    raw = np.full(top_trader_ratio.shape, np.nan, dtype=np.float64)
    if np.any(mask):
        tt = top_trader_ratio.astype(np.float64)
        rt = retail_ratio.astype(np.float64)
        lt = np.empty_like(tt)
        lr = np.empty_like(rt)
        lt[:] = np.nan
        lr[:] = np.nan
        lt[mask] = np.log(tt[mask])
        lr[mask] = np.log(rt[mask])
        raw[mask] = -(lt[mask] - lr[mask])
    return _ewm_2d(raw, max(lb_bars, 42))


def _compute_flow_taker(
    taker_buy_quote: NDArray[np.float32], quote_volume: NDArray[np.float32],
    lookback_hours: int,
) -> NDArray[np.float64]:
    vol_safe = np.where(quote_volume > 0, quote_volume.astype(np.float64), 1.0)
    imbalance = (2.0 * taker_buy_quote.astype(np.float64) - quote_volume.astype(np.float64)) / vol_safe
    ewm_imb = _ewm_2d(imbalance, max(lookback_hours, 42))
    return ewm_imb

def _compute_volatility_squeeze_keltner(
    high: NDArray[np.float32], low: NDArray[np.float32], close: NDArray[np.float32],
    lb_bars: int,
) -> NDArray[np.float64]:
    close_f64 = close.astype(np.float64)
    high_f64 = high.astype(np.float64)
    low_f64 = low.astype(np.float64)
    sma = _ewm_2d(close_f64, lb_bars)
    std = np.sqrt(_ewm_2d(close_f64 ** 2, lb_bars) - sma ** 2)
    std = np.maximum(std, 1e-12)
    bb_width = 4.0 * std / np.maximum(sma, 1e-12)
    prev_close = np.roll(close_f64, 1, axis=0)
    prev_close[0] = close_f64[0]
    tr = np.maximum(
        high_f64 - low_f64,
        np.maximum(np.abs(high_f64 - prev_close), np.abs(low_f64 - prev_close)),
    )
    atr = _ewm_2d(tr, lb_bars)
    atr = np.maximum(atr, 1e-12)
    kc_width = 2.0 * atr / np.maximum(sma, 1e-12)
    return bb_width / kc_width - 1.0


def _compute_funding_carry_reversion(
    funding: NDArray[np.float32], premium: NDArray[np.float32],
    lookback_hours: int,
) -> NDArray[np.float64]:
    combined = (funding + premium).astype(np.float64)
    ewm_combined = _ewm_2d(combined, max(lookback_hours, 42))
    span = max(lookback_hours * 2, 84)
    z = _rolling_mad_z(ewm_combined, window=span, min_periods=max(lookback_hours, 42))
    return -z


def _compute_flow_imbalance_taker(
    taker_buy_quote: NDArray[np.float32], quote_volume: NDArray[np.float32],
    lookback_hours: int,
) -> NDArray[np.float64]:
    vol_safe = np.where(quote_volume > 0, quote_volume.astype(np.float64), 1.0)
    imbalance = (2.0 * taker_buy_quote.astype(np.float64) - quote_volume.astype(np.float64)) / vol_safe
    ewm_imb = _ewm_2d(imbalance, max(lookback_hours, 42))
    span = max(lookback_hours * 2, 84)
    return _rolling_mad_z(ewm_imb, window=span, min_periods=max(lookback_hours, 42))


def _compute_open_interest_confirmation(
    open_interest: NDArray[np.float32], quote_volume: NDArray[np.float32],
    lookback_hours: int,
) -> NDArray[np.float64]:
    oi_f64 = open_interest.astype(np.float64)
    vol_safe = np.where(quote_volume > 0, quote_volume.astype(np.float64), 1.0)
    oi_change = np.zeros_like(oi_f64)
    oi_change[1:] = np.diff(oi_f64, axis=0) / vol_safe[1:]
    ewm_oi = _ewm_2d(oi_change, max(lookback_hours, 42))
    span = max(lookback_hours * 2, 84)
    return _rolling_mad_z(ewm_oi, window=span, min_periods=max(lookback_hours, 42))


def _compute_xs_rank_signal(
    close: NDArray[np.float32], lb_bars: int, eligible_2d: NDArray[np.bool_], sign: float,
) -> NDArray[np.float64]:
    from scipy.stats import rankdata

    mom = _log_return(close, lb_bars)
    n_t, n_s = mom.shape
    result = np.full((n_t, n_s), np.nan, dtype=np.float64)
    for t in range(n_t):
        row = mom[t]
        elig = eligible_2d[t]
        valid = np.isfinite(row) & elig
        n_valid = np.sum(valid)
        if n_valid < 10:
            continue
        ranks = rankdata(row[valid])
        pct = (ranks - 1.0) / (n_valid - 1.0) - 0.5
        result[t, valid] = sign * pct * 6.0
    return result

def _compute_xs_reversal(
    close: NDArray[np.float32], lb_bars: int, eligible_2d: NDArray[np.bool_],
) -> NDArray[np.float64]:
    return _compute_xs_rank_signal(close, lb_bars, eligible_2d, sign=-1.0)


_FAMILY_NATIVE_TF: dict[str, str] = {
    "trend_ema": "4h",
    "momentum_ts": "4h",
    "breakout_donchian": "4h",
    "reversal_st": "4h",
    "carry_funding": "1h",
    "basis_gap": "1h",
    "flow_taker": "1h",
    "xs_reversal": "4h",
    "xs_momentum_slow": "4h",
    "smart_money_divergence": "1h",
    "funding_carry_reversion": "1h",
    "flow_imbalance_taker": "1h",
    "volatility_squeeze_keltner": "4h",
    "open_interest_confirmation": "1h",
}


def _default_catalog() -> tuple[SignalDescriptor, ...]:
    catalog = build_canonical_alpha_catalog()  # noqa: F841
    _ = catalog
    descriptors: list[SignalDescriptor] = []
    for family in ("trend_ema", "momentum_ts", "breakout_donchian", "reversal_st", "funding_carry_reversion", "flow_imbalance_taker"):
        for speed, lb_hours in _SPEED_LADDER_8:
            descriptors.append(SignalDescriptor(
                signal_id=f"{family}:{speed}",
                family=family,
                speed=speed,
                lookback_hours=lb_hours,
                native_timeframe=_FAMILY_NATIVE_TF[family],
                target_horizon_hours=lb_hours,
            ))
    for family in ("volatility_squeeze_keltner", "open_interest_confirmation"):
        for speed, lb_hours in _SPEED_LADDER_6:
            descriptors.append(SignalDescriptor(
                signal_id=f"{family}:{speed}",
                family=family,
                speed=speed,
                lookback_hours=lb_hours,
                native_timeframe=_FAMILY_NATIVE_TF[family],
                target_horizon_hours=lb_hours,
            ))
    return tuple(descriptors)


def _normalize_return_type(
    raw: NDArray[np.float64], lb_bars: int, min_span_bars: int = 42,
) -> NDArray[np.float64]:
    span = max(lb_bars, min_span_bars)
    vol = _ewm_vol(raw, span)
    vol = np.maximum(vol, 1e-6)
    return raw / vol


def _normalize_mad_z(raw: NDArray[np.float64]) -> NDArray[np.float64]:
    return _rolling_mad_z(raw, window=540, min_periods=180)


def _compute_raw_signal(
    desc: SignalDescriptor,
    bars: MultiTimeframeBars,
    eligible_2d: NDArray[np.bool_],
) -> NDArray[np.float64] | None:
    family = desc.family
    cube_4h = bars.cubes["4h"]
    close_4h = cube_4h.close_2d
    high_4h = cube_4h.high_2d
    low_4h = cube_4h.low_2d

    try:
        if family == "trend_ema":
            if desc.native_timeframe != "4h":
                return None
            lb_bars_4h = desc.lookback_hours // 4
            raw = _compute_trend_ema(close_4h, high_4h, low_4h, lb_bars_4h)
            return _normalize_return_type(raw, lb_bars_4h)

        if family == "momentum_ts":
            if desc.native_timeframe != "4h":
                return None
            lb_bars_4h = desc.lookback_hours // 4
            return _compute_momentum_ts(close_4h, lb_bars_4h)

        if family == "breakout_donchian":
            if desc.native_timeframe != "4h":
                return None
            lb_bars_4h = desc.lookback_hours // 4
            raw = _compute_breakout_donchian(high_4h, low_4h, close_4h, lb_bars_4h)
            return _normalize_mad_z(raw)

        if family == "reversal_st":
            if desc.native_timeframe != "4h":
                return None
            lb_bars_4h = desc.lookback_hours // 4
            return _compute_reversal_st(close_4h, lb_bars_4h)

        if family == "carry_funding":
            funding = bars.aux_1h_fields.get("funding")
            premium = bars.aux_1h_fields.get("premium")
            if funding is None or premium is None:
                _logger.warning("[DATA] carry_funding: missing funding/premium in aux_1h_fields")
                return None
            raw_1h = _compute_carry_funding(funding, premium, desc.lookback_hours)
            n_t = bars.decision_timestamps_ns.size
            raw = _subsample_to_4h(raw_1h, n_t)
            return _normalize_mad_z(raw)

        if family == "basis_gap":
            mark = bars.aux_1h_fields.get("mark")
            index_arr = bars.aux_1h_fields.get("index")
            if mark is None or index_arr is None:
                _logger.warning("[DATA] basis_gap: missing mark/index in aux_1h_fields")
                return None
            raw_1h = _compute_basis_gap(mark, index_arr, desc.lookback_hours)
            n_t = bars.decision_timestamps_ns.size
            raw = _subsample_to_4h(raw_1h, n_t)
            return _normalize_mad_z(raw)

        if family == "flow_taker":
            taker_buy = bars.aux_1h_fields.get("taker_buy_quote")
            volume = bars.aux_1h_fields.get("quote_volume")
            if taker_buy is None or volume is None:
                _logger.warning("[DATA] flow_taker: missing taker_buy_quote/quote_volume in aux_1h_fields")
                return None
            raw_1h = _compute_flow_taker(taker_buy, volume, desc.lookback_hours)
            n_t = bars.decision_timestamps_ns.size
            raw = _subsample_to_4h(raw_1h, n_t)
            return _normalize_mad_z(raw)

        if family == "xs_reversal":
            if desc.native_timeframe != "4h":
                return None
            lb_bars_xs = desc.lookback_hours // 4
            return _compute_xs_reversal(close_4h, lb_bars_xs, eligible_2d)

        if family == "xs_momentum_slow":
            if desc.native_timeframe != "4h":
                return None
            lb_bars_xs = desc.lookback_hours // 4
            return _compute_xs_rank_signal(close_4h, lb_bars_xs, eligible_2d, sign=+1.0)

        if family == "smart_money_divergence":
            top_trader = bars.aux_1h_fields.get("top_trader_long_short_ratio")
            retail = bars.aux_1h_fields.get("long_short_ratio")
            if top_trader is None or retail is None:
                _logger.warning(
                    "[DATA] smart_money_divergence: missing top_trader_long_short_ratio"
                    "/long_short_ratio in aux_1h_fields",
                )
                return None
            raw_1h = _compute_smart_money_divergence(top_trader, retail, desc.lookback_hours)
            n_t = bars.decision_timestamps_ns.size
            raw = _subsample_to_4h(raw_1h, n_t)
            return _normalize_mad_z(raw)

        if family == "volatility_squeeze_keltner":
            if desc.native_timeframe != "4h":
                return None
            lb_bars_4h = desc.lookback_hours // 4
            raw = _compute_volatility_squeeze_keltner(high_4h, low_4h, close_4h, lb_bars_4h)
            return _normalize_mad_z(raw)

        if family == "funding_carry_reversion":
            funding = bars.aux_1h_fields.get("funding")
            premium = bars.aux_1h_fields.get("premium")
            if funding is None or premium is None:
                _logger.warning("[DATA] funding_carry_reversion: missing funding/premium in aux_1h_fields")
                return None
            raw_1h = _compute_funding_carry_reversion(funding, premium, desc.lookback_hours)
            n_t = bars.decision_timestamps_ns.size
            raw = _subsample_to_4h(raw_1h, n_t)
            return _normalize_mad_z(raw)

        if family == "flow_imbalance_taker":
            taker_buy = bars.aux_1h_fields.get("taker_buy_quote")
            volume = bars.aux_1h_fields.get("quote_volume")
            if taker_buy is None or volume is None:
                _logger.warning("[DATA] flow_imbalance_taker: missing taker_buy_quote/quote_volume in aux_1h_fields")
                return None
            raw_1h = _compute_flow_imbalance_taker(taker_buy, volume, desc.lookback_hours)
            n_t = bars.decision_timestamps_ns.size
            raw = _subsample_to_4h(raw_1h, n_t)
            return _normalize_mad_z(raw)

        if family == "open_interest_confirmation":
            open_interest = bars.aux_1h_fields.get("open_interest")
            volume = bars.aux_1h_fields.get("quote_volume")
            if volume is None and "1h" in bars.cubes:
                volume = bars.cubes["1h"].quote_volume_2d
            if open_interest is None or volume is None:
                _logger.warning("[DATA] open_interest_confirmation: missing open_interest/quote_volume in aux_1h_fields")
                return None
            raw_1h = _compute_open_interest_confirmation(open_interest, volume, desc.lookback_hours)
            n_t = bars.decision_timestamps_ns.size
            raw = _subsample_to_4h(raw_1h, n_t)
            return _normalize_mad_z(raw)
    except Exception:
        _logger.exception("[DATA] failed computing family=%s signal_id=%s", family, desc.signal_id)
        return None

    return None


def estimate_signal_panel_peak_bytes(
    *,
    current_rss_bytes: int,
    n_bars: int,
    n_symbols: int,
    n_recipes: int,
    max_native_rows: int,
    numba_threads: int,
    mad_window: int = 540,
) -> int:
    if current_rss_bytes < 0 or n_bars < 0 or n_symbols < 0 or n_recipes < 0 or max_native_rows < 0:
        raise ValueError("dimensions must be non-negative")
    if numba_threads < 1 or mad_window < 1:
        raise ValueError("numba_threads and mad_window must be positive")

    panel_bytes = n_bars * n_symbols * n_recipes * (4 + 1)
    sigma_bytes = n_bars * n_symbols * 8 * 3
    max_recipe_rows = max(n_bars, max_native_rows)
    recipe_bytes = max_recipe_rows * n_symbols * 8
    numba_bytes = numba_threads * mad_window * n_symbols * 8 * 2

    total = current_rss_bytes + panel_bytes + sigma_bytes + recipe_bytes + numba_bytes
    return math.ceil(1.15 * total)


def build_raw_signal_panel(
    bars: MultiTimeframeBars,
    eligible_2d: NDArray[np.bool_],
    catalog: tuple[SignalDescriptor, ...] | None = None,
    *,
    numba_threads: int = 6,
    max_rss_mb: int = 12_000,
) -> RawSignalPanel:
    if catalog is None:
        catalog = _default_catalog()

    if numba_threads < 1 or numba_threads > 6:
        raise ValueError(f"numba_threads must be 1..6, got {numba_threads}")

    n_cat = len(catalog)
    n_t = bars.decision_timestamps_ns.size
    symbols = bars.cubes["4h"].symbols
    n_syms = len(symbols)

    if eligible_2d.shape != (n_t, n_syms):
        raise ValueError(f"eligible_2d shape {eligible_2d.shape} != ({n_t}, {n_syms})")

    max_native_rows = max(
        (v.shape[0] for v in bars.aux_1h_fields.values()),
        default=n_t * 4,
    )
    max_rss_bytes = max_rss_mb * 1024 * 1024
    current_rss = psutil.Process().memory_info().rss
    estimated_peak = estimate_signal_panel_peak_bytes(
        current_rss_bytes=current_rss,
        n_bars=n_t,
        n_symbols=n_syms,
        n_recipes=n_cat,
        max_native_rows=max_native_rows,
        numba_threads=numba_threads,
    )
    if estimated_peak >= max_rss_bytes:
        raise MemoryError(
            f"preflight RSS estimate {estimated_peak} >= max {max_rss_bytes}"
        )

    cube_4h = bars.cubes["4h"]
    complete_4h = cube_4h.complete_2d

    log_ret_4h = _log_return(cube_4h.close_2d, 1)
    sigma_2d = _ewm_vol(log_ret_4h, span=42)
    sigma_2d = np.where(sigma_2d < 1e-6, 1e-6, sigma_2d).astype(np.float32)

    z_3d = np.full((n_t, n_syms, n_cat), np.nan, dtype=np.float32)
    valid_3d = np.zeros((n_t, n_syms, n_cat), dtype=np.bool_)

    global _numba_fallback_count
    _numba_fallback_count = 0
    prior_numba_threads = _numba_get_num_threads()
    effective_threads = min(numba_threads, _numba_get_num_threads())
    _numba_set_num_threads(effective_threads)

    started = time.perf_counter()
    observed_peak_rss = current_rss

    n_workers = min(4, os.cpu_count() or 1)
    use_tpe = n_workers > 1 and n_cat >= n_workers

    if use_tpe:
        _numba_set_num_threads(1)

    def _run_one(k: int, desc: SignalDescriptor) -> tuple[int, NDArray[np.float64] | None]:
        try:
            raw = _compute_raw_signal(desc, bars, eligible_2d)
        except Exception:
            _logger.exception("[SYS] recipe %d failed", k)
            raw = None
        return k, raw

    try:
        if use_tpe:
            catalog_list = list(enumerate(catalog))
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_run_one, k, desc): k for k, desc in catalog_list}
                for future in as_completed(futures):
                    k, raw = future.result()
                    _write_recipe_result(z_3d, valid_3d, k, raw, eligible_2d, complete_4h)
                    del raw
                    obs = psutil.Process().memory_info().rss
                    if obs > observed_peak_rss:
                        observed_peak_rss = obs
                    if obs >= max_rss_bytes:
                        _logger.error(
                            "[SYS][L1] RSS exceeded limit: observed_rss_mb=%.1f max_rss_mb=%d",
                            obs / 1048576, max_rss_mb,
                        )
                        raise MemoryError(f"runtime RSS {obs} >= max {max_rss_bytes}")
        else:
            for k, desc in enumerate(catalog):
                k, raw = _run_one(k, desc)
                _write_recipe_result(z_3d, valid_3d, k, raw, eligible_2d, complete_4h)
                del raw
                obs = psutil.Process().memory_info().rss
                if obs > observed_peak_rss:
                    observed_peak_rss = obs
                if obs >= max_rss_bytes:
                    _logger.error(
                        "[SYS][L1] RSS exceeded limit: observed_rss_mb=%.1f max_rss_mb=%d",
                        obs / 1048576, max_rss_mb,
                    )
                    raise MemoryError(f"runtime RSS {obs} >= max {max_rss_bytes}")
    finally:
        _numba_set_num_threads(prior_numba_threads)

    elapsed = time.perf_counter() - started
    total_valid = int(np.sum(valid_3d))
    total_cells = valid_3d.size
    valid_ratio = total_valid / max(total_cells, 1)
    executed_mad_recipes = sum(1 for d in catalog if d.family in ("breakout_donchian", "basis_gap", "smart_money_divergence", "carry_funding", "flow_taker", "volatility_squeeze_keltner", "funding_carry_reversion", "flow_imbalance_taker", "open_interest_confirmation"))
    _logger.info(
        "[PERF][L1] signal_panel elapsed_s=%.4f recipes=%d executed_mad_recipes=%d numba_threads=%d numba_fallbacks=%d valid_ratio=%.4f signal_count=%d field_count=%d estimated_peak_mb=%.0f observed_peak_mb=%.0f max_rss_mb=%d",
        elapsed, n_cat, executed_mad_recipes, effective_threads, _numba_fallback_count,
        valid_ratio, n_cat, n_syms, estimated_peak / 1048576, observed_peak_rss / 1048576, max_rss_mb,
    )

    if valid_ratio < 0.05:
        raise InsufficientCoverageError(
            f"valid ratio {valid_ratio:.4f} < 0.05",
        )

    return RawSignalPanel(
        decision_timestamps_ns=bars.decision_timestamps_ns,
        symbols=symbols,
        descriptors=catalog,
        z_3d=z_3d,
        valid_3d=valid_3d,
        sigma_2d=sigma_2d.astype(np.float32),
    )


def _write_recipe_result(
    z_3d: NDArray[np.float32],
    valid_3d: NDArray[np.bool_],
    k: int,
    raw: NDArray[np.float64] | None,
    eligible_2d: NDArray[np.bool_],
    complete_4h: NDArray[np.bool_],
) -> None:
    if raw is None:
        valid_3d[:, :, k] = False
        z_3d[:, :, k] = np.nan
        return
    z_slice = np.clip(raw, -3.0, 3.0)
    valid = eligible_2d & complete_4h & np.isfinite(z_slice)
    z_3d[:, :, k] = z_slice.astype(np.float32)
    valid_3d[:, :, k] = valid
