from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.contracts import (
    InsufficientCoverageError,
    MultiTimeframeBars,
    RawSignalPanel,
    SignalDescriptor,
)

_logger = logging.getLogger(__name__)

# P3 wiring: bars = build_multi_timeframe_bars(market); panel = build_raw_signal_panel(bars, eligible_2d=eligible_4h)

SPEED_LADDER: tuple[tuple[str, int], ...] = (
    ("fast", 24),
    ("medium", 72),
    ("slow", 216),
    ("very_slow", 648),
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


def _rolling_mad_z(arr: NDArray[np.float64], window: int, min_periods: int) -> NDArray[np.float64]:
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


def _compute_flow_taker(
    taker_buy_quote: NDArray[np.float32], quote_volume: NDArray[np.float32],
    lookback_hours: int,
) -> NDArray[np.float64]:
    vol_safe = np.where(quote_volume > 0, quote_volume.astype(np.float64), 1.0)
    imbalance = (2.0 * taker_buy_quote.astype(np.float64) - quote_volume.astype(np.float64)) / vol_safe
    ewm_imb = _ewm_2d(imbalance, max(lookback_hours, 42))
    return ewm_imb

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
}


def _default_catalog() -> tuple[SignalDescriptor, ...]:
    descriptors: list[SignalDescriptor] = []
    for family in ("trend_ema", "momentum_ts", "breakout_donchian", "carry_funding", "basis_gap"):
        for speed, lb_hours in SPEED_LADDER:
            desc = SignalDescriptor(
                signal_id=f"{family}:{speed}",
                family=family,
                speed=speed,
                lookback_hours=lb_hours,
                native_timeframe=_FAMILY_NATIVE_TF[family],
            )
            descriptors.append(desc)
    descriptors.append(SignalDescriptor(
        signal_id="reversal_st:fast",
        family="reversal_st",
        speed="fast",
        lookback_hours=24,
        native_timeframe="4h",
    ))
    descriptors.append(SignalDescriptor(
        signal_id="xs_reversal:fast",
        family="xs_reversal",
        speed="fast",
        lookback_hours=8,
        native_timeframe="4h",
    ))
    descriptors.append(SignalDescriptor(
        signal_id="xs_reversal:medium",
        family="xs_reversal",
        speed="medium",
        lookback_hours=24,
        native_timeframe="4h",
    ))
    descriptors.append(SignalDescriptor(
        signal_id="xs_momentum_slow:slow",
        family="xs_momentum_slow",
        speed="slow",
        lookback_hours=216,
        native_timeframe="4h",
        target_horizon_hours=216,
    ))
    descriptors.append(SignalDescriptor(
        signal_id="xs_momentum_slow:very_slow",
        family="xs_momentum_slow",
        speed="very_slow",
        lookback_hours=648,
        native_timeframe="4h",
        target_horizon_hours=648,
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


def build_raw_signal_panel(
    bars: MultiTimeframeBars,
    eligible_2d: NDArray[np.bool_],
    catalog: tuple[SignalDescriptor, ...] | None = None,
) -> RawSignalPanel:
    if catalog is None:
        catalog = _default_catalog()

    n_cat = len(catalog)
    n_t = bars.decision_timestamps_ns.size
    symbols = bars.cubes["4h"].symbols
    n_syms = len(symbols)

    if eligible_2d.shape != (n_t, n_syms):
        raise ValueError(f"eligible_2d shape {eligible_2d.shape} != ({n_t}, {n_syms})")

    cube_4h = bars.cubes["4h"]
    close_4h = cube_4h.close_2d
    high_4h = cube_4h.high_2d
    low_4h = cube_4h.low_2d
    complete_4h = cube_4h.complete_2d

    log_ret_4h = _log_return(close_4h, 1)
    sigma_2d = _ewm_vol(log_ret_4h, span=42)
    sigma_2d = np.where(sigma_2d < 1e-6, 1e-6, sigma_2d).astype(np.float32)

    z_3d = np.full((n_t, n_syms, n_cat), np.nan, dtype=np.float32)
    valid_3d = np.zeros((n_t, n_syms, n_cat), dtype=np.bool_)

    for k, desc in enumerate(catalog):
        recipe_ok = True
        family = desc.family
        lb_bars_4h = desc.lookback_hours // 4 if desc.native_timeframe == "4h" else 0

        try:
            if family == "trend_ema":
                if desc.native_timeframe != "4h":
                    recipe_ok = False
                else:
                    raw = _compute_trend_ema(close_4h, high_4h, low_4h, lb_bars_4h)
                    raw = _normalize_return_type(raw, lb_bars_4h)
            elif family == "momentum_ts":
                if desc.native_timeframe != "4h":
                    recipe_ok = False
                else:
                    raw = _compute_momentum_ts(close_4h, lb_bars_4h)
            elif family == "breakout_donchian":
                if desc.native_timeframe != "4h":
                    recipe_ok = False
                else:
                    raw = _compute_breakout_donchian(high_4h, low_4h, close_4h, lb_bars_4h)
                    raw = _normalize_mad_z(raw)
            elif family == "reversal_st":
                if desc.native_timeframe != "4h":
                    recipe_ok = False
                else:
                    raw = _compute_reversal_st(close_4h, lb_bars_4h)
            elif family == "carry_funding":
                funding = bars.aux_1h_fields.get("funding")
                premium = bars.aux_1h_fields.get("premium")
                if funding is None or premium is None:
                    _logger.warning("[DATA] carry_funding: missing funding/premium in aux_1h_fields")
                    recipe_ok = False
                else:
                    raw_1h = _compute_carry_funding(funding, premium, desc.lookback_hours)
                    raw = _subsample_to_4h(raw_1h, n_t)
                    raw = _normalize_mad_z(raw)
            elif family == "basis_gap":
                mark = bars.aux_1h_fields.get("mark")
                index_arr = bars.aux_1h_fields.get("index")
                if mark is None or index_arr is None:
                    _logger.warning("[DATA] basis_gap: missing mark/index in aux_1h_fields")
                    recipe_ok = False
                else:
                    raw_1h = _compute_basis_gap(mark, index_arr, desc.lookback_hours)
                    raw = _subsample_to_4h(raw_1h, n_t)
                    raw = _normalize_mad_z(raw)
            elif family == "flow_taker":
                taker_buy = bars.aux_1h_fields.get("taker_buy_quote")
                volume = bars.aux_1h_fields.get("quote_volume")
                if taker_buy is None or volume is None:
                    _logger.warning("[DATA] flow_taker: missing taker_buy_quote/quote_volume in aux_1h_fields")
                    recipe_ok = False
                else:
                    raw_1h = _compute_flow_taker(taker_buy, volume, desc.lookback_hours)
                    raw = _subsample_to_4h(raw_1h, n_t)
                    raw = _normalize_mad_z(raw)
            elif family == "xs_reversal":
                if desc.native_timeframe != "4h":
                    recipe_ok = False
                else:
                    lb_bars_xs = desc.lookback_hours // 4
                    raw = _compute_xs_reversal(close_4h, lb_bars_xs, eligible_2d)
            elif family == "xs_momentum_slow":
                if desc.native_timeframe != "4h":
                    recipe_ok = False
                else:
                    lb_bars_xs = desc.lookback_hours // 4
                    raw = _compute_xs_rank_signal(close_4h, lb_bars_xs, eligible_2d, sign=+1.0)
            else:
                recipe_ok = False
        except Exception:
            _logger.exception("[DATA] failed computing family=%s signal_id=%s", family, desc.signal_id)
            recipe_ok = False

        if not recipe_ok:
            valid_3d[:, :, k] = False
            z_3d[:, :, k] = np.nan
            continue

        z_slice = np.clip(raw, -3.0, 3.0)
        valid = eligible_2d & complete_4h & np.isfinite(z_slice)

        z_3d[:, :, k] = z_slice.astype(np.float32)
        valid_3d[:, :, k] = valid

    total_valid = np.sum(valid_3d)
    total_cells = valid_3d.size
    valid_ratio = total_valid / max(total_cells, 1)
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
