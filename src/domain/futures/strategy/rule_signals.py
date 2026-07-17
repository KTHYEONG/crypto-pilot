"""Strategy-side mirror of the signal family registry.

[ADR_20260709_L0_TREND_PULLBACK_HARDENING_SYNC]
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.strategy.candidate_contracts import CandidateSignalPanel, SignalExitPolicy
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig, apply_per_family_params
from src.domain.futures.strategy.exit_policies import build_exit_policies_for_panel
from src.domain.futures.strategy.market_regime import MarketRegimeContext, compute_market_regime_context
from src.domain.futures.strategy.timeframe_contracts import hours_per_bar, scale_bar_count

_logger = logging.getLogger(__name__)
_ROBUST_Z_EPS = 1e-9
_ROBUST_Z_CLIP = 3.0

# [ADR_20260706_L0_SIGNAL_FAMILY_DIVERSITY] Must stay identical to signals/rules.py's copy.
ALL_SIGNAL_FAMILIES: tuple[str, ...] = (
    "trend_ma",
    "trend_donchian",
    "vol_breakout",
    "btc_regime_pullback",
    "trend_pullback_continuation",
    "dual_momentum",
    "residual_reversion",
    "xs_momentum",
    "xs_flow",
    "xs_oi_skew",
    "mtf_trend_pullback",
    "mtf_breakout_retest",
    "taker_imbalance_momentum",
    "funding_flow_carry",
    "funding_extreme_reversal",
    "lsr_oi_regime_filter",
    "vol_term_structure_gate",
    "macd_4h",
    "supertrend",
    "ichimoku_trend",
    "trend_pullback_quality_v2",
    "residual_momentum_xs",
    "funding_flow_exhaustion_sparse",
    "oi_lsr_unwind",
    "vol_contraction_breakout",
    "xs_residual_rebalance",
    "carry_net_of_funding",
    "liquidity_participation_breakout",
    "btc_neutral_residual_reversal",
    "price_band_reversion",
    "mtf_fusion",
)


def candidate_variant_key(family: str, variant: str) -> str:
    """Return a stable candidate variant key."""
    return f"{family}:{variant}"


@numba.njit  # type: ignore[untyped-decorator]
def _robust_zscore_numba(
    raw_scores: np.ndarray,
    groups: np.ndarray,
    clip: float,
    eps: float,
) -> np.ndarray:
    """Numba-accelerated per-group robust z-score using median/MAD normalization."""
    n = raw_scores.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(groups)
    sorted_scores = raw_scores[order]
    sorted_groups = groups[order]
    sorted_result = np.zeros(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_groups[j] == sorted_groups[i]:
            j += 1
        vals = sorted_scores[i:j]
        finite_count = 0
        for k in range(vals.shape[0]):
            if np.isfinite(vals[k]):
                finite_count += 1
        if finite_count == 0:
            i = j
            continue
        finite_vals = np.empty(finite_count, dtype=np.float64)
        idx = 0
        for k in range(vals.shape[0]):
            if np.isfinite(vals[k]):
                finite_vals[idx] = vals[k]
                idx += 1
        m = np.median(finite_vals)
        mad = np.median(np.abs(finite_vals - m)) * 1.4826
        for k in range(i, j):
            if mad > eps and np.isfinite(sorted_scores[k]):
                sorted_result[k] = (sorted_scores[k] - m) / mad
        i = j

    for k in range(n):
        sorted_result[k] = max(-clip, min(clip, sorted_result[k]))

    result = np.zeros(n, dtype=np.float64)
    for k in range(n):
        result[order[k]] = sorted_result[k]
    return result


def _cross_sectional_robust_zscore(raw_scores: NDArray[np.float64], groups: NDArray[np.int64]) -> NDArray[np.float64]:
    """Return per-group robust z-scores using median/MAD normalization."""
    result = _robust_zscore_numba(raw_scores, groups, _ROBUST_Z_CLIP, _ROBUST_Z_EPS)
    assert isinstance(result, np.ndarray)
    return result


def _normalize_linear_score(
    raw: NDArray[np.float64],
    *,
    scale: float,
    positive_only: bool = False,
) -> NDArray[np.float64]:
    """Normalize raw amplitudes into bounded scores."""
    normalized = raw / max(scale, 1e-12)
    if positive_only:
        return np.clip(normalized, 0.0, 1.0)
    return np.clip(normalized, -1.0, 1.0)


def _candidate_variant_set(values: tuple[str, ...]) -> set[str]:
    return {value for value in values if value}


def filter_rule_signal_panels(
    panels: tuple[CandidateSignalPanel, ...],
    *,
    cfg: CandidateStrategyConfig,
) -> tuple[CandidateSignalPanel, ...]:
    """Filter panels by configured family and variant allowlists."""
    before = len(panels)
    family_allowlist = _candidate_variant_set(cfg.candidate_families)
    enabled_variants = _candidate_variant_set(cfg.enabled_candidate_variants)

    filtered: list[CandidateSignalPanel] = []
    for panel in panels:
        if family_allowlist and panel.family not in family_allowlist:
            continue
        key = candidate_variant_key(panel.family, panel.variant)
        if enabled_variants and key not in enabled_variants:
            continue
        filtered.append(panel)

    _logger.debug(
        "[DIAG][RULE_PANEL_FILTER] before=%d after=%d families=%s variants=%s",
        before,
        len(filtered),
        ",".join(cfg.candidate_families) if cfg.candidate_families else "",
        ",".join(cfg.enabled_candidate_variants) if cfg.enabled_candidate_variants else "",
    )
    return tuple(filtered)


# --- Vectorized Technical Indicator Helpers ---


@numba.njit(cache=True, fastmath=True)  # type: ignore
def _ema_2d_jit(arr: np.ndarray, span: int) -> np.ndarray:
    t_len, n_sym = arr.shape
    out = np.empty((t_len, n_sym), dtype=np.float64)
    out.fill(np.nan)
    alpha = 2.0 / (float(span) + 1.0)

    last = np.zeros(n_sym, dtype=np.float64)
    initialized = np.zeros(n_sym, dtype=np.bool_)

    for t in range(t_len):
        for c in range(n_sym):
            val = arr[t, c]
            if np.isfinite(val):
                if initialized[c]:
                    last[c] = (1.0 - alpha) * last[c] + alpha * val
                else:
                    last[c] = val
                    initialized[c] = True
            if initialized[c]:
                out[t, c] = last[c]
    return out


def _ema_2d(arr: NDArray[np.float64], span: int) -> NDArray[np.float64]:
    if span <= 1:
        return arr.copy()
    return np.asarray(_ema_2d_jit(arr, span), dtype=np.float64)


def _rolling_mean_2d(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    # Time: O(T*N) via pandas C-layer; Space: O(T*N)
    if window <= 1:
        return arr.copy()
    df = pd.DataFrame(arr)
    result: NDArray[np.float64] = df.rolling(window=window, min_periods=1).mean().shift(1).to_numpy(dtype=np.float64)
    return result


def _rolling_std_2d(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    # Time: O(T*N) via pandas C-layer; Space: O(T*N)
    if window <= 1:
        return np.zeros_like(arr, dtype=np.float64)
    df = pd.DataFrame(arr)
    raw: NDArray[np.float64] = df.rolling(window=window, min_periods=2).std().shift(1).to_numpy(dtype=np.float64)
    result: NDArray[np.float64] = np.where(np.isfinite(raw), raw, 1e-12)
    return result


def _log_return_2d(close: NDArray[np.float64], lag: int) -> NDArray[np.float64]:
    shifted = np.vstack([np.tile(close[:1], (lag, 1)), close[:-lag]])
    out = np.log(np.maximum(close, 1e-12) / np.maximum(shifted, 1e-12))
    out[:lag] = 0.0
    return out.astype(np.float64, copy=False)


def _zscore_2d(arr: NDArray[np.float64], window: int, eps: float = 1e-12) -> NDArray[np.float64]:
    mean = _rolling_mean_2d(arr, window=window)
    std = _rolling_std_2d(arr, window=window)
    return (arr - mean) / np.maximum(std, eps)


def _safe_taker_imbalance_2d(
    taker_buy: NDArray[np.float64] | None,
    volume: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return bounded taker imbalance and a validity mask."""
    if taker_buy is None:
        return np.zeros_like(volume, dtype=np.float64), np.zeros_like(volume, dtype=np.bool_)
    if taker_buy.shape != volume.shape:
        raise ValueError("taker_buy and volume shapes must match")

    valid = (
        np.isfinite(taker_buy)
        & np.isfinite(volume)
        & (volume > 0.0)
        & (taker_buy >= 0.0)
        & (taker_buy <= volume * 1.01)
    )
    ratio = np.zeros_like(volume, dtype=np.float64)
    ratio[valid] = taker_buy[valid] / volume[valid]
    imbalance = np.clip(2.0 * ratio - 1.0, -1.0, 1.0)
    imbalance[~valid] = 0.0
    return imbalance, valid


def _rolling_max_2d(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    # shift(1): exclude current bar to produce trailing-exclusive channel (Donchian semantics).
    # close[t] > prior_high[t] triggers breakout; without shift, roll_max[t]==high[t] → signal impossible.
    df = pd.DataFrame(arr)
    result: NDArray[np.float64] = df.rolling(window=window, min_periods=1).max().shift(1).to_numpy(dtype=np.float64)
    return result


def _rolling_min_2d(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    df = pd.DataFrame(arr)
    result: NDArray[np.float64] = df.rolling(window=window, min_periods=1).min().shift(1).to_numpy(dtype=np.float64)
    return result


def _rolling_corr_with_col(arr: NDArray[np.float64], ref_col: int, window: int) -> NDArray[np.float64]:
    """Rolling Pearson corr of each column with arr[:, ref_col].

    Loops over N symbols (not T time steps) — acceptable for N~20.
    """
    t_len, n_sym = arr.shape
    out = np.full((t_len, n_sym), np.nan, dtype=np.float64)
    ref = pd.Series(arr[:, ref_col])
    for col in range(n_sym):
        corr = pd.Series(arr[:, col]).rolling(window, min_periods=window).corr(ref)
        out[:, col] = corr.to_numpy(dtype=np.float64)
    return out


def _entry_rising_edge_2d(condition: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Return True only on False->True transitions for each [t, symbol].

    Args:
        condition: Boolean array of shape [T, N].

    Returns:
        Boolean array of shape [T, N] where True marks the first bar of each
        consecutive True-run. The first row is always False (no prior state).

    Time: O(T*N)  Space: O(T*N) — one vstack + elementwise AND/NOT
    """
    # prev[t] = condition[t-1]; prev[0] = False (no prior bar)
    prev: NDArray[np.bool_] = np.vstack([np.zeros((1, condition.shape[1]), dtype=bool), condition[:-1]])
    result: NDArray[np.bool_] = condition & ~prev
    # Row 0 has no prior state — suppress any apparent transition
    result[0, :] = False
    return result


def _atr_2d(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int,
) -> NDArray[np.float64]:
    prev_close = np.vstack([close[:1], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return _rolling_mean_2d(tr, window=period)


def _rsi_2d(close: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    delta = np.diff(close, axis=0, prepend=close[:1])
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)
    avg_gain = _rolling_mean_2d(gain, window=period)
    avg_loss = _rolling_mean_2d(loss, window=period)
    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return np.clip(rsi, 0.0, 100.0)


def project_higher_tf_to_grid(
    *,
    feature_higher: NDArray[np.float64],
    dt_higher: NDArray[np.datetime64],
    dt_grid: NDArray[np.datetime64],
) -> NDArray[np.float64]:
    """Release-timestamp based backward-asof projection.

    Accepts 1D (T_i,) or 2D (T_i, N) feature arrays.
    Returns same dimensionality along grid dimension: (T_base,) or (T_base, N).
    Prevents look-ahead leak by only allowing grid[t] to see higher TF values
    that were closed/released on or before grid[t].
    """
    dt_higher_int = dt_higher.astype(np.int64)
    dt_grid_int = dt_grid.astype(np.int64)
    indices = np.searchsorted(dt_higher_int, dt_grid_int, side="right") - 1
    valid_mask = indices >= 0
    clipped_indices = np.clip(indices, 0, len(dt_higher) - 1)

    out = feature_higher[clipped_indices]
    out[~valid_mask] = np.nan
    return out


def _resample_to_htf_and_project(
    *,
    datetimes_4h: NDArray[np.datetime64],
    values_4h: NDArray[np.float64],
    htf: str,
    agg_method: str,
    compute_feature_fn: Callable[[pd.DataFrame], pd.DataFrame | NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Resamples 4h series to HTF, computes feature, and projects back causally."""
    _, n_cols = values_4h.shape
    df_4h = pd.DataFrame(values_4h, index=pd.to_datetime(datetimes_4h))

    resampler = df_4h.resample(htf, closed="left", label="left")
    if agg_method == "last":
        df_htf = resampler.last()
    elif agg_method == "max":
        df_htf = resampler.max()
    elif agg_method == "min":
        df_htf = resampler.min()
    elif agg_method == "first":
        df_htf = resampler.first()
    elif agg_method == "sum":
        df_htf = resampler.sum()
    else:
        raise ValueError(f"unknown agg_method: {agg_method}")

    df_feature_htf = compute_feature_fn(df_htf)
    if isinstance(df_feature_htf, np.ndarray):
        df_feature_htf = pd.DataFrame(df_feature_htf, index=df_htf.index)

    # Release timestamp: start of HTF bar + duration
    delta = pd.Timedelta(htf)
    dt_higher = (df_feature_htf.index + delta).to_numpy()

    feature_higher = df_feature_htf.to_numpy(dtype=np.float64)
    out_4h = np.zeros_like(values_4h)
    for col in range(n_cols):
        out_4h[:, col] = project_higher_tf_to_grid(
            feature_higher=feature_higher[:, col], dt_higher=dt_higher, dt_grid=datetimes_4h
        )
    return out_4h


def _resample_ohlc_to_htf_and_project(
    *,
    datetimes_4h: NDArray[np.datetime64],
    high_4h: NDArray[np.float64],
    low_4h: NDArray[np.float64],
    close_4h: NDArray[np.float64],
    htf: str,
    compute_feature_fn: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame],
) -> NDArray[np.float64]:
    """Resample OHLC to HTF once (all symbols), compute feature once, project back with ONE searchsorted call.

    [ADR_20260713_L0_MTF_FUSION_PERF_OPT] Replaces the per-symbol loop (build df_sym x N, call filter_fn x N,
    call project_higher_tf_to_grid x N). searchsorted's indices/valid_mask depend only on timestamps,
    not per-symbol values -- computing them once and reusing via fancy-indexing is exact, not approximate.
    """
    idx_4h = pd.to_datetime(datetimes_4h)
    res_high = pd.DataFrame(high_4h, index=idx_4h).resample(htf, closed="left", label="left").max()
    res_low = pd.DataFrame(low_4h, index=idx_4h).resample(htf, closed="left", label="left").min()
    res_close = pd.DataFrame(close_4h, index=idx_4h).resample(htf, closed="left", label="left").last()

    feature_htf = compute_feature_fn(res_high, res_low, res_close)
    feature_arr = feature_htf.to_numpy(dtype=np.float64)

    delta = pd.Timedelta(htf)
    dt_higher = (res_close.index + delta).to_numpy()
    indices = np.searchsorted(dt_higher.astype(np.int64), datetimes_4h.astype(np.int64), side="right") - 1
    valid_mask = indices >= 0
    clipped_indices = np.clip(indices, 0, len(dt_higher) - 1)
    out: NDArray[np.float64] = feature_arr[clipped_indices, :]
    out[~valid_mask, :] = np.nan
    return out


_MTF_FUSION_HTF_CANDIDATES: tuple[str, ...] = ("1D", "12h", "8h")


def _tf_hours(tf: str) -> float:
    if tf.endswith("h"):
        return float(tf[:-1])
    if tf.endswith("D") or tf.endswith("d"):
        return float(tf[:-1]) * 24
    return hours_per_bar(tf)


def resolve_valid_htf_choices(native_tf: str) -> tuple[str, ...]:
    """Return HTF candidates strictly coarser than native_tf. [ADR_20260713_L0_MTF_FUSION_FACTORY]"""
    native_hours = _tf_hours(native_tf)
    return tuple(htf for htf in _MTF_FUSION_HTF_CANDIDATES if _tf_hours(htf) > native_hours)


def _weighted_moving_average_2d(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    weights = np.arange(1, window + 1, dtype=np.float64)
    weights_sum = weights.sum()
    out = np.full_like(values, np.nan)
    for t in range(window - 1, values.shape[0]):
        window_slice = values[t - window + 1 : t + 1]
        out[t] = np.tensordot(weights, window_slice, axes=(0, 0)) / weights_sum
    return out


def _htf_ema_slope_filter(df_htf: pd.DataFrame, span: int) -> pd.DataFrame:
    ema = df_htf.ewm(span=span, adjust=False).mean()
    return np.sign(ema.diff()).fillna(0.0)


def _htf_macd_cross_filter(df_htf: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = df_htf.ewm(span=fast, adjust=False).mean()
    ema_slow = df_htf.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return np.sign(macd_line - macd_signal).fillna(0.0)


def _htf_hma_slope_filter(df_htf: pd.DataFrame, window: int) -> pd.DataFrame:
    half = max(1, window // 2)
    root = max(1, round(np.sqrt(window)))
    values = df_htf.to_numpy(dtype=np.float64)
    wma_half = _weighted_moving_average_2d(values, half)
    wma_full = _weighted_moving_average_2d(values, window)
    raw_hma = 2.0 * wma_half - wma_full
    hma = _weighted_moving_average_2d(raw_hma, root)
    slope = np.diff(hma, axis=0, prepend=hma[:1])
    return pd.DataFrame(np.sign(np.nan_to_num(slope, nan=0.0)), index=df_htf.index)


def _htf_ichimoku_cloud_filter(
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    tenkan: int,
    kijun: int,
    senkou_b: int,
) -> pd.DataFrame:
    """Multi-column (all symbols at once) Ichimoku cloud direction. [ADR_20260713_L0_MTF_FUSION_PERF_OPT]"""
    tenkan_line = (high_df.rolling(tenkan).max() + low_df.rolling(tenkan).min()) / 2.0
    kijun_line = (high_df.rolling(kijun).max() + low_df.rolling(kijun).min()) / 2.0
    senkou_a = (tenkan_line + kijun_line) / 2.0
    senkou_b_line = (high_df.rolling(senkou_b).max() + low_df.rolling(senkou_b).min()) / 2.0
    # np.fmax/np.fmin (not np.maximum/np.minimum) to match pandas .max(axis=1)/.min(axis=1)'s
    # skipna=True semantics from the pre-vectorization implementation: during the partial-warmup
    # window where senkou_b_line is still NaN (senkou_b > kijun periods) but senkou_a is already
    # valid, the cloud boundary must fall back to senkou_a, not propagate NaN. [ADR_20260713_L0_MTF_FUSION_PERF_OPT]
    cloud_top = np.fmax(senkou_a, senkou_b_line)
    cloud_bottom = np.fmin(senkou_a, senkou_b_line)
    direction = np.where(close_df > cloud_top, 1.0, np.where(close_df < cloud_bottom, -1.0, 0.0))
    return pd.DataFrame(direction, index=close_df.index, columns=close_df.columns)


def _htf_adx_dmi_filter(
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    period: int,
    threshold: float,
) -> pd.DataFrame:
    """Multi-column (all symbols at once) ADX/DMI direction, strength-gated. [ADR_20260713_L0_MTF_FUSION_PERF_OPT]

    16.7x measured speedup vs. the per-symbol loop (217.5ms -> 13.0ms, N=126), np.allclose-verified.
    """
    prev_close = close_df.shift(1)
    up_move = high_df.diff()
    down_move = -low_df.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = (
        pd.concat(
            [high_df - low_df, (high_df - prev_close).abs(), (low_df - prev_close).abs()],
            keys=["a", "b", "c"],
        )
        .groupby(level=1)
        .max()
    )
    alpha = 1.0 / period
    smoothed_tr = tr.ewm(alpha=alpha, adjust=False).mean()
    smoothed_plus = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    smoothed_minus = minus_dm.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100.0 * smoothed_plus / smoothed_tr.replace(0.0, np.nan)
    minus_di = 100.0 * smoothed_minus / smoothed_tr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean().fillna(0.0)
    raw_direction = np.sign((plus_di - minus_di).fillna(0.0))
    return raw_direction.where(adx >= threshold, 0.0)


def _ltf_rsi_band_trigger(
    *,
    close: NDArray[np.float64],
    period: int,
    lo: float,
    hi: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float64]]:
    rsi = _rsi_2d(close, period)
    rsi_prev = np.vstack([rsi[:1], rsi[:-1]])
    long_mask = (rsi_prev < lo) & (rsi >= lo)
    short_mask = (rsi_prev > hi) & (rsi <= hi)
    score = np.tanh((rsi - 50.0) / 10.0)
    return long_mask, short_mask, score


def _ltf_macd_cross_trigger(
    *,
    close: NDArray[np.float64],
    atr: NDArray[np.float64],
    fast: int,
    slow: int,
    signal: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float64]]:
    ema_fast = _ema_2d(close, span=fast)
    ema_slow = _ema_2d(close, span=slow)
    macd_line = ema_fast - ema_slow
    macd_signal = _ema_2d(macd_line, span=signal)
    hist = macd_line - macd_signal
    hist_prev = np.vstack([hist[:1], hist[:-1]])
    long_mask = (hist > 0) & (hist_prev <= 0)
    short_mask = (hist < 0) & (hist_prev >= 0)
    score = np.tanh(hist / atr)
    return long_mask, short_mask, score


def _ltf_donchian_retest_trigger(
    *,
    close: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    atr: NDArray[np.float64],
    lookback: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float64]]:
    upper = _rolling_max_2d(high, window=lookback)
    lower = _rolling_min_2d(low, window=lookback)
    prev_close = np.vstack([close[:1], close[:-1]])
    long_mask = (prev_close < upper) & (close >= upper)
    short_mask = (prev_close > lower) & (close <= lower)
    score = np.where(long_mask, (close - upper) / atr, np.where(short_mask, (close - lower) / atr, 0.0))
    return long_mask, short_mask, score


def _ltf_stochastic_cross_trigger(
    *,
    close: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    k_period: int,
    d_period: int,
    lo: float,
    hi: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float64]]:
    lowest_low = _rolling_min_2d(low, window=k_period)
    highest_high = _rolling_max_2d(high, window=k_period)
    denom = np.maximum(highest_high - lowest_low, 1e-12)
    pct_k = 100.0 * (close - lowest_low) / denom
    pct_d = _rolling_mean_2d(pct_k, window=d_period)
    k_prev = np.vstack([pct_k[:1], pct_k[:-1]])
    d_prev = np.vstack([pct_d[:1], pct_d[:-1]])
    cross_up = (k_prev <= d_prev) & (pct_k > pct_d)
    cross_down = (k_prev >= d_prev) & (pct_k < pct_d)
    long_mask = cross_up & (pct_d < lo)
    short_mask = cross_down & (pct_d > hi)
    score = np.tanh((pct_k - 50.0) / 25.0)
    return long_mask, short_mask, score


def _resolve_panel_archetype(panel: Any) -> str:
    """Resolve fallback archetype from family when panel.metadata lacks one.

    [ADR_20260707_L1_BACKTEST_FIDELITY_FIXES]
    """
    archetype = str(panel.metadata.get("archetype", panel.archetype or "")).strip()
    family = panel.family
    if archetype:
        return archetype
    if family in {
        "trend_ma",
        "trend_donchian",
        "vol_breakout",
        "trend_pullback_continuation",
        "mtf_trend_pullback",
        "mtf_breakout_retest",
        "mtf_fusion",
        "vol_term_structure_gate",
        "btc_regime_pullback",
        "sparse_breakout_retest_v2",
        "trend_pullback_quality_v2",
        "sparse_breakout_retest_liquidity",
    }:
        return "trend"
    if family in {"dual_momentum", "taker_imbalance_momentum"}:
        return "ts_mom"
    if family in {"funding_zscore_carry", "funding_flow_carry"}:
        return "carry_rev"
    if family in {"residual_reversion", "residual_momentum_xs"}:
        return "beta_neut"
    if family in {"funding_extreme_reversal", "funding_flow_exhaustion_sparse"}:
        return "unwind"
    if family in {"oi_lsr_unwind"}:
        return "unwind"
    if family in {"vol_contraction_breakout"}:
        return "mean_rev"
    if family in {"xs_residual_rebalance"}:
        return "xs_alpha"
    if family in {"carry_net_of_funding"}:
        return "carry_rev"
    return "mean_rev"


def _allowed_regimes_for_archetype(archetype: str) -> tuple[str, ...]:
    if archetype == "xs_alpha":
        return ()
    if archetype in {"trend", "ts_mom"}:
        return ("bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile")
    if archetype in {"flow_rev", "unwind"}:
        return ("bull_volatile", "bear_volatile", "crash")
    if archetype == "carry_rev":
        return ("bull_quiet", "bear_quiet", "transition")
    if archetype == "beta_neut":
        return ("bull_quiet",)
    return ("bull_quiet", "bear_quiet", "transition")


def _legacy_exit_policy(panel: CandidateSignalPanel) -> SignalExitPolicy:
    return SignalExitPolicy(
        policy_id="legacy",
        archetype="mean_rev",
        stop_atr_mult=float(panel.stop_atr_mult),
        take_profit_atr_mult=float(panel.take_profit_atr_mult),
        expected_holding_bars=int(panel.expected_holding_bars),
        min_holding_bars=int(panel.min_holding_bars),
        description="Backward-compatible panel policy.",
    )


def _entry_regime_fields(
    panel: CandidateSignalPanel,
    t_idx: NDArray[np.int64] | NDArray[np.int32],
) -> tuple[NDArray[np.int8], NDArray[np.object_]]:
    regime_code_1d = panel.regime_code_1d
    regime_names = panel.regime_name_by_code
    if regime_code_1d is None or not regime_names:
        return (
            np.full(t_idx.shape[0], 4, dtype=np.int8),
            np.full(t_idx.shape[0], "transition", dtype=object),
        )
    entry_regime_codes = regime_code_1d[t_idx].astype(np.int8, copy=False)
    entry_regimes = np.asarray([regime_names[int(code)] for code in entry_regime_codes], dtype=object)
    return entry_regime_codes, entry_regimes


def _attach_signal_context(
    panels: tuple[CandidateSignalPanel, ...],
    *,
    cfg: CandidateStrategyConfig,
    regime_ctx: MarketRegimeContext,
) -> tuple[CandidateSignalPanel, ...]:
    out: list[CandidateSignalPanel] = []
    regime_names = np.asarray(regime_ctx.name_by_code, dtype=object)
    for panel in panels:
        archetype = _resolve_panel_archetype(panel)
        allowed_regimes = _allowed_regimes_for_archetype(archetype)
        side_hint_2d = np.asarray(panel.side_hint_2d, dtype=np.int8).copy()
        if (
            cfg.regime_signal_gating_enabled
            or (cfg.mean_rev_gating_enabled and archetype == "mean_rev")
            or (cfg.beta_neut_gating_enabled and archetype == "beta_neut")
        ):
            allowed_mask = np.isin(regime_names[regime_ctx.code_1d], np.asarray(allowed_regimes, dtype=object))
            side_hint_2d[~allowed_mask, :] = 0
        exit_policies = build_exit_policies_for_panel(
            archetype=archetype,
            regime_name=allowed_regimes[0] if allowed_regimes else "transition",
            base_expected_holding_bars=int(panel.expected_holding_bars),
            base_min_holding_bars=int(panel.min_holding_bars),
            max_policies=cfg.max_exit_policy_variants_per_signal,
            fallback_stop_atr_mult=float(panel.stop_atr_mult),
            fallback_take_profit_atr_mult=float(panel.take_profit_atr_mult),
        )
        out.append(
            CandidateSignalPanel(
                family=panel.family,
                variant=panel.variant,
                params=panel.params,
                datetimes=panel.datetimes,
                symbols=panel.symbols,
                signed_score_2d=panel.signed_score_2d,
                side_hint_2d=side_hint_2d,
                expected_holding_bars=panel.expected_holding_bars,
                min_holding_bars=panel.min_holding_bars,
                stop_atr_mult=panel.stop_atr_mult,
                take_profit_atr_mult=panel.take_profit_atr_mult,
                turnover_proxy_2d=panel.turnover_proxy_2d,
                valid_mask_2d=panel.valid_mask_2d,
                metadata=panel.metadata,
                archetype=archetype,
                allowed_regimes=allowed_regimes,
                exit_policies=exit_policies,
                regime_code_1d=regime_ctx.code_1d,
                regime_name_by_code=regime_ctx.name_by_code,
            )
        )
    return tuple(out)


# --- 8 Vectorized Rule Families ---


def _cross_sectional_rank_signed_2d(
    raw_2d: NDArray[np.float64],
    valid_2d: NDArray[np.bool_],
    *,
    min_cross_section: int,
    top_q: float = 0.70,
    bot_q: float = 0.30,
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    masked = np.where(valid_2d, raw_2d, np.nan)
    count = valid_2d.sum(axis=1)
    pct = np.asarray(pd.DataFrame(masked).rank(axis=1, pct=True).to_numpy().copy(), dtype=np.float64)
    signed = np.nan_to_num(2.0 * pct - 1.0, nan=0.0)
    side = np.zeros_like(raw_2d, dtype=np.int8)
    side[pct >= top_q] = 1
    side[pct <= bot_q] = -1
    row_block = count < min_cross_section
    signed[row_block, :] = 0.0
    side[row_block, :] = 0
    return signed.astype(np.float64), side


def _beta_residual_return_2d(
    close: NDArray[np.float64],
    btc_idx: int,
    lookback: int,
) -> NDArray[np.float64]:
    r = np.diff(np.log(np.maximum(close, 1e-12)), axis=0, prepend=0.0)
    rb = r[:, btc_idx : btc_idx + 1]
    beta = _rolling_mean_2d(r * rb, lookback) / np.maximum(_rolling_mean_2d(rb * rb, lookback), 1e-12)
    resid = r - beta * rb
    warm = np.ones_like(r, dtype=np.bool_)
    warm[:lookback] = False
    _out = pd.DataFrame(np.where(warm, resid, 0.0)).rolling(lookback, min_periods=1).sum().to_numpy(np.float64).copy()
    out = np.asarray(_out, dtype=np.float64)
    out[:lookback] = 0.0
    return out


def _supertrend_2d(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 10,
    multiplier: float = 2.5,
) -> NDArray[np.int8]:
    """Simplified SuperTrend: binary trend direction +1/-1 per bar [T, N]."""
    n_bar = high.shape[0]
    n_sym = high.shape[1]
    atr_st = _atr_2d(high, low, close, period=period)
    hl_avg = (high + low) / 2.0
    upper = hl_avg + multiplier * atr_st
    lower = hl_avg - multiplier * atr_st
    trend = np.zeros((n_bar, n_sym), dtype=np.int8)
    for t in range(1, n_bar):
        prev = trend[t - 1]
        upper[t] = np.where(
            (upper[t] < upper[t - 1]) | (close[t - 1] > upper[t - 1]),
            upper[t],
            upper[t - 1],
        )
        lower[t] = np.where(
            (lower[t] > lower[t - 1]) | (close[t - 1] < lower[t - 1]),
            lower[t],
            lower[t - 1],
        )
        trend[t] = np.where(
            prev <= 0,
            np.where(close[t] > upper[t], 1, -1),
            np.where(close[t] < lower[t], -1, 1),
        )
    return trend


@dataclass(slots=True, frozen=True)
class _SignalIndicatorCache:
    """Precomputed shared indicators reused within a single TF build to avoid recomputation.

    The content covers all indicators computed before the per-family loop in
    ``build_rule_signal_panels``.  When passed as ``precomputed`` to that function,
    the long indicator-preparation block is skipped and these cached arrays are
    used directly.
    """

    atr: NDArray[np.float64]
    flow_imbalance: NDArray[np.float64]
    flow_mean_6: NDArray[np.float64]
    flow_z_24: NDArray[np.float64]
    funding_z_96: NDArray[np.float64]
    funding_z_168: NDArray[np.float64]
    oi_build_z_42: NDArray[np.float64]
    lsr_log_z_42: NDArray[np.float64]


def _precompute_shared_indicators(
    *,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    normalize_time_horizon: bool = False,
    horizon_base_tf: str = "4h",
) -> _SignalIndicatorCache:
    """Compute the set of shared rolling indicators once for reuse by all families.

    This is the exact same computation that ``build_rule_signal_panels`` runs in
    its indicator-preparation block.  Extracted so callers that
    build panels for multiple TFs can share the work.
    """
    close = aligned.close_2d
    high = aligned.high_2d
    low = aligned.low_2d
    vol = aligned.volume_2d
    funding = aligned.funding_2d

    def scale_window(base_bars: int, minimum: int = 1) -> int:
        if not normalize_time_horizon:
            return max(minimum, base_bars)
        from src.domain.futures.strategy.timeframe_contracts import scale_bar_count

        return scale_bar_count(base_bars, cfg.timeframe, horizon_base_tf, minimum=minimum)

    atr_period = scale_window(14)
    flow_mean_6_window = scale_window(6)
    flow_z_24_window = scale_window(24)
    funding_z_96_window = scale_window(96)
    funding_z_168_window = scale_window(168)
    oi_build_z_42_window = scale_window(42)
    lsr_log_z_42_window = scale_window(42)

    atr = _atr_2d(high, low, close, period=atr_period)
    atr = np.maximum(atr, 1e-12)
    flow_imbalance, _ = _safe_taker_imbalance_2d(aligned.taker_buy_2d, vol)
    flow_mean_6 = _rolling_mean_2d(flow_imbalance, window=flow_mean_6_window)
    flow_z_24 = _zscore_2d(flow_imbalance, window=flow_z_24_window)
    funding_z_96 = _zscore_2d(funding, window=funding_z_96_window, eps=1e-6)
    funding_z_168 = _zscore_2d(funding, window=funding_z_168_window, eps=1e-6)

    oi = aligned.oi_2d if aligned.oi_2d is not None else np.full_like(close, np.nan, dtype=np.float64)
    lsr = aligned.lsr_2d if aligned.lsr_2d is not None else np.full_like(close, np.nan, dtype=np.float64)
    oi_valid = np.isfinite(oi) & (oi > 0.0)
    lsr_valid = np.isfinite(lsr) & (lsr > 0.0)
    oi_log = np.where(oi_valid, np.log(oi), np.nan)
    lsr_log = np.where(lsr_valid, np.log(lsr), np.nan)
    oi_log_change_6 = oi_log - np.roll(oi_log, flow_mean_6_window, axis=0)
    oi_log_change_6[:flow_mean_6_window] = np.nan
    oi_build_z_42 = _zscore_2d(oi_log_change_6, window=oi_build_z_42_window)
    lsr_log_z_42 = _zscore_2d(lsr_log, window=lsr_log_z_42_window)

    return _SignalIndicatorCache(
        atr=atr,
        flow_imbalance=flow_imbalance,
        flow_mean_6=flow_mean_6,
        flow_z_24=flow_z_24,
        funding_z_96=funding_z_96,
        funding_z_168=funding_z_168,
        oi_build_z_42=oi_build_z_42,
        lsr_log_z_42=lsr_log_z_42,
    )


def build_rule_signal_panels(
    *,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    normalize_time_horizon: bool = False,
    horizon_base_tf: str = "4h",
    family_filter: tuple[str, ...] | None = None,
    precomputed: _SignalIndicatorCache | None = None,
) -> tuple[CandidateSignalPanel, ...]:
    """Build trailing-only rule candidates for all symbols.

    When *precomputed* is provided the costly rolling-indicator
    preparation block is skipped — the caller is responsible for computing
    ``_precompute_shared_indicators`` on the **same** aligned object.

    [ADR_20260714_L1_MEMORY_EXECUTION]
    """

    def scale_window(base_bars: int, minimum: int = 1) -> int:
        if not normalize_time_horizon:
            return max(minimum, base_bars)
        return scale_bar_count(base_bars, cfg.timeframe, horizon_base_tf, minimum=minimum)

    regime_ctx = compute_market_regime_context(aligned=aligned)
    close = aligned.close_2d
    btc_idx = 0
    for idx, sym in enumerate(aligned.symbols):
        if "BTC" in sym:
            btc_idx = idx
            break
    high = aligned.high_2d
    low = aligned.low_2d
    vol = aligned.volume_2d
    funding = aligned.funding_2d
    signal_active_mask = (
        aligned.inference_active_mask if aligned.inference_active_mask is not None else aligned.active_mask
    )
    entry_warm_mask = (
        aligned.inference_entry_warm_mask if aligned.inference_entry_warm_mask is not None else aligned.warm_mask
    )
    execution_eligibility_mask = (
        aligned.execution_eligibility_mask
        if aligned.execution_eligibility_mask is not None
        else np.ones_like(signal_active_mask, dtype=bool)
    )
    strategy_readiness_mask = (
        aligned.strategy_readiness_mask
        if aligned.strategy_readiness_mask is not None
        else np.ones_like(signal_active_mask, dtype=bool)
    )
    promotion_active_mask = (
        aligned.promotion_active_mask
        if aligned.promotion_active_mask is not None
        else np.ones_like(signal_active_mask, dtype=bool)
    )
    valid_mask = (
        signal_active_mask
        & entry_warm_mask
        & execution_eligibility_mask
        & strategy_readiness_mask
        & promotion_active_mask
        & ~aligned.entry_block_mask
        & ~aligned.kill_mask
        & np.isfinite(close)
        & np.isfinite(high)
        & np.isfinite(low)
    )

    flow_mean_6_window = scale_window(6)
    flow_z_24_window = scale_window(24)
    atr_period = scale_window(14)
    funding_z_96_window = scale_window(96)
    funding_z_168_window = scale_window(168)
    oi_build_z_42_window = scale_window(42)
    lsr_log_z_42_window = scale_window(42)
    positioning_warm_bars = scale_window(168)
    rsi_14_window = scale_window(14)
    btc_fast_window = scale_window(20)
    btc_slow_window = scale_window(100)
    alt_mean_window = scale_window(50)
    funding_flow_window = scale_window(96)

    oi = aligned.oi_2d if aligned.oi_2d is not None else np.full_like(close, np.nan, dtype=np.float64)
    lsr = aligned.lsr_2d if aligned.lsr_2d is not None else np.full_like(close, np.nan, dtype=np.float64)
    oi_valid = np.isfinite(oi) & (oi > 0.0)
    lsr_valid = np.isfinite(lsr) & (lsr > 0.0)
    oi_log = np.where(oi_valid, np.log(oi), np.nan)
    lsr_log = np.where(lsr_valid, np.log(lsr), np.nan)

    if precomputed is not None:
        atr = precomputed.atr
        flow_imbalance = precomputed.flow_imbalance
        flow_mean_6 = precomputed.flow_mean_6
        flow_z_24 = precomputed.flow_z_24
        funding_z_96 = precomputed.funding_z_96
        funding_z_168 = precomputed.funding_z_168
        oi_build_z_42 = precomputed.oi_build_z_42
        lsr_log_z_42 = precomputed.lsr_log_z_42
        flow_valid = np.isfinite(flow_imbalance) & np.isfinite(
            aligned.taker_buy_2d if aligned.taker_buy_2d is not None else flow_imbalance
        )  # noqa: E501
    else:
        atr = _atr_2d(high, low, close, period=atr_period)
        atr = np.maximum(atr, 1e-12)
        flow_imbalance, flow_valid = _safe_taker_imbalance_2d(aligned.taker_buy_2d, vol)
        flow_mean_6 = _rolling_mean_2d(flow_imbalance, window=flow_mean_6_window)
        flow_z_24 = _zscore_2d(flow_imbalance, window=flow_z_24_window)
        funding_z_96 = _zscore_2d(funding, window=funding_z_96_window, eps=1e-6)
        funding_z_168 = _zscore_2d(funding, window=funding_z_168_window, eps=1e-6)
        oi_log_change_6 = oi_log - np.roll(oi_log, flow_mean_6_window, axis=0)
        oi_log_change_6[:flow_mean_6_window] = np.nan
        oi_build_z_42 = _zscore_2d(oi_log_change_6, window=oi_build_z_42_window)
        lsr_log_z_42 = _zscore_2d(lsr_log, window=lsr_log_z_42_window)

    shared_valid = valid_mask & flow_valid & np.isfinite(funding)
    fxr_valid = valid_mask & flow_valid
    # UNW warm-up: require 168 bars of continuous valid data for z-score stability
    positioning_warm = np.ones_like(valid_mask, dtype=np.bool_)
    positioning_warm[:positioning_warm_bars] = False
    positioning_valid = valid_mask & flow_valid & np.isfinite(funding) & oi_valid & lsr_valid & positioning_warm

    def _build_single_family(fam: str) -> list[CandidateSignalPanel]:
        fam_panels: list[CandidateSignalPanel] = []

        if fam == "trend_ma":
            # 1. Trend MA Cross
            ema_fast = _ema_2d(close, span=scale_window(12))
            ema_slow = _ema_2d(close, span=scale_window(72))
            ma_diff = (ema_fast - ema_slow) / atr
            signed_score_ma = np.tanh(ma_diff)
            side_hint_ma = np.zeros_like(signed_score_ma, dtype=np.int8)
            side_hint_ma[_entry_rising_edge_2d(ma_diff > 0.5)] = 1
            side_hint_ma[_entry_rising_edge_2d(ma_diff < -0.5)] = -1
            fam_panels.append(
                CandidateSignalPanel(
                    family="trend_ma",
                    variant="ema_12_72",
                    params={"ema_fast": scale_window(12), "ema_slow": scale_window(72), "atr_period": atr_period},
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=signed_score_ma,
                    side_hint_2d=side_hint_ma,
                    expected_holding_bars=scale_window(18),
                    min_holding_bars=scale_window(6),
                    stop_atr_mult=2.0,
                    take_profit_atr_mult=4.0,
                    turnover_proxy_2d=np.abs(np.diff(signed_score_ma, axis=0, prepend=0.0)),
                    valid_mask_2d=valid_mask,
                )
            )

            # 1c. Trend MA Cross — ema_18_108
            ema_fast_18 = _ema_2d(close, span=scale_window(18))
            ema_slow_108 = _ema_2d(close, span=scale_window(108))
            ma_diff_18_108 = (ema_fast_18 - ema_slow_108) / atr
            signed_score_ma_18_108 = np.tanh(ma_diff_18_108)
            side_hint_ma_18_108 = np.zeros_like(signed_score_ma_18_108, dtype=np.int8)
            side_hint_ma_18_108[_entry_rising_edge_2d(ma_diff_18_108 > 0.5)] = 1
            side_hint_ma_18_108[_entry_rising_edge_2d(ma_diff_18_108 < -0.5)] = -1
            fam_panels.append(
                CandidateSignalPanel(
                    family="trend_ma",
                    variant="ema_18_108",
                    params={"ema_fast": scale_window(18), "ema_slow": scale_window(108), "atr_period": atr_period},
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=signed_score_ma_18_108,
                    side_hint_2d=side_hint_ma_18_108,
                    expected_holding_bars=scale_window(24),
                    min_holding_bars=scale_window(8),
                    stop_atr_mult=2.5,
                    take_profit_atr_mult=5.0,
                    turnover_proxy_2d=np.abs(np.diff(signed_score_ma_18_108, axis=0, prepend=0.0)),
                    valid_mask_2d=valid_mask,
                )
            )

        elif fam == "trend_donchian":
            # 2. Trend Donchian — donchian_72 only (donchian_18/36 removed: p>0.34, possym<0.50)
            d72_high = _rolling_max_2d(high, window=scale_window(72))
            d72_low = _rolling_min_2d(low, window=scale_window(72))
            d72_side = np.zeros_like(close, dtype=np.int8)
            d72_side[_entry_rising_edge_2d(close > d72_high)] = 1
            d72_side[_entry_rising_edge_2d(close < d72_low)] = -1
            d72_score = np.zeros_like(close)
            above_72 = close > d72_high
            below_72 = close < d72_low
            d72_score[above_72] = (close[above_72] - d72_high[above_72]) / atr[above_72]
            d72_score[below_72] = (close[below_72] - d72_low[below_72]) / atr[below_72]
            fam_panels.append(
                CandidateSignalPanel(
                    family="trend_donchian",
                    variant="donchian_72",
                    params={"lookback": scale_window(72)},
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=np.clip(d72_score, -1.0, 1.0),
                    side_hint_2d=d72_side,
                    expected_holding_bars=scale_window(36),
                    min_holding_bars=scale_window(12),
                    stop_atr_mult=2.5,
                    take_profit_atr_mult=5.0,
                    turnover_proxy_2d=np.abs(np.diff(d72_score, axis=0, prepend=0.0)),
                    valid_mask_2d=valid_mask,
                )
            )

        elif fam == "vol_breakout":
            # 3. Vol Breakout
            bb_mean = _rolling_mean_2d(close, window=scale_window(20))
            bb_std = _rolling_std_2d(close, window=scale_window(20))
            bandwidth = (bb_std * 4.0) / np.maximum(bb_mean, 1e-12)
            bw_mean_120 = _rolling_mean_2d(bandwidth, window=scale_window(120))
            bw_std_120 = _rolling_std_2d(bandwidth, window=scale_window(120))
            bw_z = (bandwidth - bw_mean_120) / np.maximum(bw_std_120, 1e-12)
            compressed = bw_z < -1.0
            vol_side = np.zeros_like(close, dtype=np.int8)
            vol_side[_entry_rising_edge_2d(compressed & (close > bb_mean + bb_std * 2.0))] = 1
            vol_side[_entry_rising_edge_2d(compressed & (close < bb_mean - bb_std * 2.0))] = -1
            vol_score = np.where(vol_side != 0, (close - bb_mean) / atr, 0.0)
            fam_panels.append(
                CandidateSignalPanel(
                    family="vol_breakout",
                    variant="bb_compress_20",
                    params={"bb_window": scale_window(20), "compression_window": scale_window(120)},
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=np.clip(vol_score, -1.0, 1.0),
                    side_hint_2d=vol_side,
                    expected_holding_bars=scale_window(18),
                    min_holding_bars=scale_window(6),
                    stop_atr_mult=1.5,
                    take_profit_atr_mult=3.0,
                    turnover_proxy_2d=np.abs(np.diff(vol_score, axis=0, prepend=0.0)),
                    valid_mask_2d=valid_mask,
                )
            )

        elif fam == "btc_regime_pullback":
            # 8. BTC Regime Pullback (+ slow-lookback control + RSI-oscillator variant)
            for _variant, _btc_fast, _btc_slow, _alt_window, _oscillator in [
                ("btc_pullback_50", btc_fast_window, btc_slow_window, alt_mean_window, "zscore"),
                ("btc_pullback_100_slow", scale_window(50), scale_window(200), scale_window(100), "zscore"),
                ("btc_pullback_50_rsi", btc_fast_window, btc_slow_window, alt_mean_window, "rsi"),
            ]:
                btc_close = close[:, btc_idx : btc_idx + 1]
                btc_ema_fast = _ema_2d(btc_close, span=_btc_fast)
                btc_ema_slow = _ema_2d(btc_close, span=_btc_slow)
                btc_trend_up = btc_ema_fast > btc_ema_slow

                if _oscillator == "rsi":
                    _alt_rsi = _rsi_2d(close, period=rsi_14_window)
                    _osc_long = _alt_rsi < 30.0
                    _osc_short = _alt_rsi > 70.0
                    _osc_score = (50.0 - _alt_rsi) / 20.0
                elif _oscillator == "zscore":
                    _alt_mean = _rolling_mean_2d(close, window=_alt_window)
                    _alt_std = _rolling_std_2d(close, window=_alt_window)
                    _alt_pullback_z = (close - _alt_mean) / np.maximum(_alt_std, 1e-12)
                    _osc_long = _alt_pullback_z < -1.5
                    _osc_short = _alt_pullback_z > 1.5
                    _osc_score = -_alt_pullback_z / 2.0
                else:
                    raise ValueError(f"unsupported oscillator type: {_oscillator}")

                btc_side = np.zeros_like(close, dtype=np.int8)
                btc_side[_entry_rising_edge_2d(btc_trend_up & _osc_long)] = 1
                btc_side[_entry_rising_edge_2d(~btc_trend_up & _osc_short)] = -1
                btc_score = np.clip(_osc_score, -1.0, 1.0)
                _holding = scale_window(36) if _variant == "btc_pullback_100_slow" else scale_window(18)
                _min_holding = scale_window(12) if _variant == "btc_pullback_100_slow" else scale_window(6)
                fam_panels.append(
                    CandidateSignalPanel(
                        family="btc_regime_pullback",
                        variant=_variant,
                        params={
                            "window": _alt_window,
                            "btc_fast": _btc_fast,
                            "btc_slow": _btc_slow,
                            "oscillator": _oscillator,
                        },
                        datetimes=aligned.datetimes,
                        symbols=aligned.symbols,
                        signed_score_2d=btc_score,
                        side_hint_2d=btc_side,
                        expected_holding_bars=_holding,
                        min_holding_bars=_min_holding,
                        stop_atr_mult=2.0,
                        take_profit_atr_mult=3.0,
                        turnover_proxy_2d=np.abs(np.diff(btc_score, axis=0, prepend=0.0)),
                        valid_mask_2d=valid_mask,
                        metadata={
                            "archetype": "trend",
                            "regime": "btc_regime_pullback",
                            "edge_hypothesis": {
                                "btc_pullback_50": "baseline: BTC trend regime gates alt-coin z-score pullback entries",
                                "btc_pullback_100_slow": (
                                    "[LIMIT-03] slow-lookback control: turnover_per_year should drop "
                                    "materially vs baseline if the archetype's survival is turnover-driven"
                                ),
                                "btc_pullback_50_rsi": (
                                    "[LIMIT-04] oscillator substitution: tests whether edge is tied to the "
                                    "regime-filter structure or specifically to the z-score trigger formulation"
                                ),
                            }[_variant],
                        },
                    )
                )

        elif fam == "trend_pullback_continuation":
            # 16. Trend Pullback Continuation
            for _fast, _slow, _rsi_lo, _rsi_hi in [
                (scale_window(20), scale_window(100), 40.0, 65.0),
                (scale_window(50), scale_window(200), 40.0, 65.0),
                (scale_window(100), scale_window(400), 40.0, 65.0),
            ]:
                _ema_fast = _ema_2d(close, span=_fast)
                _ema_slow = _ema_2d(close, span=_slow)
                _close_prev = np.vstack([close[:1], close[:-1]])
                _ema_fast_prev = np.vstack([_ema_fast[:1], _ema_fast[:-1]])
                _rsi_trend = _rsi_2d(close, period=rsi_14_window)
                _trend_up = _ema_fast > _ema_slow
                _trend_dn = _ema_fast < _ema_slow
                _long = (
                    _trend_up
                    & (_close_prev <= _ema_fast_prev)
                    & (close > _ema_fast)
                    & (_rsi_trend >= _rsi_lo)
                    & (_rsi_trend <= _rsi_hi)
                )
                _short = (
                    _trend_dn
                    & (_close_prev >= _ema_fast_prev)
                    & (close < _ema_fast)
                    & (_rsi_trend >= (100.0 - _rsi_hi))
                    & (_rsi_trend <= (100.0 - _rsi_lo))
                )
                _pullback_dist = (close - _ema_fast) / atr
                _tpc_score = np.where(
                    _long,
                    np.clip(_pullback_dist, 0.0, 2.0),
                    np.where(_short, np.clip(_pullback_dist, -2.0, 0.0), 0.0),
                )
                _tpc_side = np.zeros_like(close, dtype=np.int8)
                _tpc_side[_long] = 1
                _tpc_side[_short] = -1
                fam_panels.append(
                    CandidateSignalPanel(
                        family="trend_pullback_continuation",
                        variant=f"tpc_{_fast}_{_slow}",
                        params={"ema_fast": _fast, "ema_slow": _slow, "rsi_lo": int(_rsi_lo), "rsi_hi": int(_rsi_hi)},
                        datetimes=aligned.datetimes,
                        symbols=aligned.symbols,
                        signed_score_2d=_normalize_linear_score(_tpc_score, scale=2.0),
                        side_hint_2d=_tpc_side,
                        expected_holding_bars=max(scale_window(8), _fast // 2),
                        min_holding_bars=max(scale_window(3), _fast // 8),
                        stop_atr_mult=1.5,
                        take_profit_atr_mult=3.0,
                        turnover_proxy_2d=np.abs(
                            np.diff(_normalize_linear_score(_tpc_score, scale=2.0), axis=0, prepend=0.0)
                        ),
                        valid_mask_2d=valid_mask,
                        metadata={
                            "archetype": "trend",
                            "regime": "established_trend_pullback",
                            "edge_hypothesis": (
                                "[LIMIT-03] slow-lookback control: turnover_per_year should drop "
                                "materially vs baseline if the archetype's survival is turnover-driven"
                                if (_fast, _slow) == (scale_window(100), scale_window(400))
                                else (
                                    "pullback recovery inside an established trend improves continuation entry quality"
                                )
                            ),
                        },
                    )
                )

        elif fam == "dual_momentum":
            # 17. Dual Momentum
            for _short_lb, _long_lb in [
                (scale_window(12), scale_window(48)),
                (scale_window(24), scale_window(96)),
            ]:
                _ret_short = _log_return_2d(close, lag=_short_lb)
                _ret_long = _log_return_2d(close, lag=_long_lb)
                _ret_short_z = _zscore_2d(_ret_short, window=_long_lb)
                _ret_long_z = _zscore_2d(_ret_long, window=_long_lb)
                _dm_score = np.tanh((_ret_short_z + _ret_long_z) / 2.0)
                _dm_side = np.zeros_like(close, dtype=np.int8)
                _dm_side[_entry_rising_edge_2d((_ret_short_z > 0.5) & (_ret_long_z > 0.5))] = 1
                _dm_side[_entry_rising_edge_2d((_ret_short_z < -0.5) & (_ret_long_z < -0.5))] = -1
                _dm_score[:_long_lb] = 0.0
                _dm_side[:_long_lb] = 0
                fam_panels.append(
                    CandidateSignalPanel(
                        family="dual_momentum",
                        variant=f"dm_{_short_lb}_{_long_lb}",
                        params={"short_lookback": _short_lb, "long_lookback": _long_lb},
                        datetimes=aligned.datetimes,
                        symbols=aligned.symbols,
                        signed_score_2d=np.clip(_dm_score, -1.0, 1.0),
                        side_hint_2d=_dm_side,
                        expected_holding_bars=max(scale_window(8), _short_lb),
                        min_holding_bars=max(scale_window(3), _short_lb // 3),
                        stop_atr_mult=1.5,
                        take_profit_atr_mult=3.0,
                        turnover_proxy_2d=np.abs(np.diff(np.clip(_dm_score, -1.0, 1.0), axis=0, prepend=0.0)),
                        valid_mask_2d=valid_mask,
                        metadata={
                            "archetype": "ts_mom",
                            "regime": "multi_horizon_trend_agreement",
                            "edge_hypothesis": (
                                "aligned short and long horizon momentum produces more durable "
                                "continuation than single-horizon trend rules"
                            ),
                        },
                    )
                )

        elif fam == "residual_reversion":
            # 20. Residual Reversion
            _log_ret_rr = np.diff(np.log(np.maximum(close, 1e-12)), axis=0, prepend=0.0)
            _btc_ret_rr = _log_ret_rr[:, btc_idx : btc_idx + 1]
            for _rr_win in [scale_window(24), scale_window(48)]:
                _btc_var_rr = _rolling_mean_2d(_btc_ret_rr**2, window=_rr_win)
                _cov_alt_btc_rr = _rolling_mean_2d(_log_ret_rr * _btc_ret_rr, window=_rr_win)
                _beta_hat_rr = _cov_alt_btc_rr / np.maximum(_btc_var_rr, 1e-12)
                _resid_ret_rr = _log_ret_rr - _beta_hat_rr * _btc_ret_rr
                _resid_mean_rr = _rolling_mean_2d(_resid_ret_rr, window=_rr_win)
                _resid_std_rr = _rolling_std_2d(_resid_ret_rr, window=_rr_win)
                _resid_z_rr = (_resid_ret_rr - _resid_mean_rr) / np.maximum(_resid_std_rr, 1e-12)
                _rr_side = np.zeros_like(close, dtype=np.int8)
                _rr_side[_entry_rising_edge_2d(_resid_z_rr <= -2.0)] = 1
                _rr_side[_entry_rising_edge_2d(_resid_z_rr >= 2.0)] = -1
                _rr_score = np.where(_rr_side != 0, -_normalize_linear_score(_resid_z_rr, scale=3.0), 0.0)
                fam_panels.append(
                    CandidateSignalPanel(
                        family="residual_reversion",
                        variant=f"rr_{_rr_win}",
                        params={"window": _rr_win},
                        datetimes=aligned.datetimes,
                        symbols=aligned.symbols,
                        signed_score_2d=_rr_score.astype(np.float64, copy=False),
                        side_hint_2d=_rr_side,
                        expected_holding_bars=max(scale_window(4), _rr_win // 6),
                        min_holding_bars=scale_window(2),
                        stop_atr_mult=1.25,
                        take_profit_atr_mult=2.0,
                        turnover_proxy_2d=np.abs(
                            np.diff(_rr_score.astype(np.float64, copy=False), axis=0, prepend=0.0)
                        ),
                        valid_mask_2d=valid_mask,
                        metadata={
                            "archetype": "beta_neut",
                            "regime": "btc_adjusted_overextension",
                            "edge_hypothesis": (
                                "large BTC-adjusted residual moves mean revert when they reflect "
                                "idiosyncratic overextension rather than broad market beta"
                            ),
                        },
                    )
                )

            # =========================================================================
            # XS ALPHA FAMILIES (cross-sectional, beta-neutral)
            # =========================================================================

        elif fam == "xs_momentum":
            _min_xs = cfg.l1_min_cross_section
            for _lb in [scale_window(12), scale_window(48)]:
                _raw = _beta_residual_return_2d(close, btc_idx, _lb)
                _sc, _sd = _cross_sectional_rank_signed_2d(
                    _raw,
                    valid_mask,
                    min_cross_section=_min_xs,
                )
                fam_panels.append(
                    CandidateSignalPanel(
                        family="xs_momentum",
                        variant=f"xs_mom_{_lb}",
                        params={"lookback": _lb},
                        datetimes=aligned.datetimes,
                        symbols=aligned.symbols,
                        signed_score_2d=_sc,
                        side_hint_2d=_sd,
                        expected_holding_bars=scale_window(18) if _lb <= 12 else scale_window(24),
                        min_holding_bars=scale_window(6) if _lb <= 12 else scale_window(8),
                        stop_atr_mult=1.5,
                        take_profit_atr_mult=2.5,
                        turnover_proxy_2d=np.abs(np.diff(_sc, axis=0, prepend=0.0)),
                        valid_mask_2d=valid_mask,
                        metadata={
                            "archetype": "xs_alpha",
                            "edge_hypothesis": ("cross-sectional relative momentum, beta-neutral by construction"),
                        },
                    )
                )

        elif fam == "xs_flow":
            _min_xs = cfg.l1_min_cross_section
            _sc, _sd = _cross_sectional_rank_signed_2d(
                flow_z_24,
                fxr_valid,
                min_cross_section=_min_xs,
            )
            fam_panels.append(
                CandidateSignalPanel(
                    family="xs_flow",
                    variant="xs_flow_24",
                    params={"flow_z_window": 24},
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=_sc,
                    side_hint_2d=_sd,
                    expected_holding_bars=scale_window(12),
                    min_holding_bars=scale_window(4),
                    stop_atr_mult=1.5,
                    take_profit_atr_mult=2.5,
                    turnover_proxy_2d=np.abs(np.diff(_sc, axis=0, prepend=0.0)),
                    valid_mask_2d=fxr_valid,
                    metadata={
                        "archetype": "xs_alpha",
                        "edge_hypothesis": ("cross-sectional taker flow: long relative buying, short selling"),
                    },
                )
            )

        elif fam == "xs_oi_skew":
            _min_xs = cfg.l1_min_cross_section
            _raw = -(oi_build_z_42 * np.sign(lsr_log_z_42))
            _sc, _sd = _cross_sectional_rank_signed_2d(
                _raw,
                positioning_valid,
                min_cross_section=_min_xs,
            )
            fam_panels.append(
                CandidateSignalPanel(
                    family="xs_oi_skew",
                    variant="xs_oi_42",
                    params={"oi_build_window": 42, "lsr_window": 42},
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=_sc,
                    side_hint_2d=_sd,
                    expected_holding_bars=scale_window(18),
                    min_holding_bars=scale_window(6),
                    stop_atr_mult=1.5,
                    take_profit_atr_mult=2.5,
                    turnover_proxy_2d=np.abs(np.diff(_sc, axis=0, prepend=0.0)),
                    valid_mask_2d=positioning_valid,
                    metadata={
                        "archetype": "xs_alpha",
                        "edge_hypothesis": ("cross-sectional OI build with LSR skew: short crowded longs"),
                    },
                )
            )

            # =========================================================================
            # NEW G1 ~ G10 FAMILIES
            # =========================================================================

        elif fam == "mtf_trend_pullback":
            # G1. mtf_trend_pullback
            for _n_htf in [20, 50, 100]:

                def _compute_g1_htf(
                    df_htf: pd.DataFrame,
                    span: int = _n_htf,
                ) -> pd.DataFrame:
                    ema = df_htf.ewm(span=span, adjust=False).mean()
                    slope = ema.diff()
                    return np.sign(slope).fillna(0.0)

                _proj_htf_slope = _resample_to_htf_and_project(
                    datetimes_4h=aligned.datetimes,
                    values_4h=close,
                    htf="1D",
                    agg_method="last",
                    compute_feature_fn=_compute_g1_htf,
                )

                _rsi_4h = _rsi_2d(close, rsi_14_window)
                _rsi_prev = np.vstack([_rsi_4h[:1], _rsi_4h[:-1]])
                _rsi_lo, _rsi_hi = 30.0, 70.0

                _long_trig = (_rsi_prev < _rsi_lo) & (_rsi_4h >= _rsi_lo)
                _short_trig = (_rsi_prev > _rsi_hi) & (_rsi_4h <= _rsi_hi)

                _g1_side = np.zeros_like(close, dtype=np.int8)
                _g1_side[(_proj_htf_slope > 0) & _long_trig] = 1
                _g1_side[(_proj_htf_slope < 0) & _short_trig] = -1
                _g1_score = np.tanh((_rsi_4h - 50.0) / 10.0)

                fam_panels.append(
                    CandidateSignalPanel(
                        family="mtf_trend_pullback",
                        variant=f"mtf_tpb_{_n_htf}_30",
                        params={"n_htf": _n_htf, "rsi_lo": _rsi_lo, "rsi_hi": _rsi_hi},
                        datetimes=aligned.datetimes,
                        symbols=aligned.symbols,
                        signed_score_2d=np.clip(_g1_score, -1.0, 1.0),
                        side_hint_2d=_g1_side,
                        expected_holding_bars=scale_window(18),
                        min_holding_bars=scale_window(6),
                        stop_atr_mult=2.0,
                        take_profit_atr_mult=3.0,
                        turnover_proxy_2d=np.abs(np.diff(np.clip(_g1_score, -1.0, 1.0), axis=0, prepend=0.0)),
                        valid_mask_2d=valid_mask,
                        metadata={
                            "archetype": "trend",
                            "regime": "top_down_trend_pullback",
                            "edge_hypothesis": (
                                "[LIMIT-03] slow-lookback control: turnover_per_year should drop "
                                "materially vs baseline if the archetype's survival is turnover-driven"
                                if _n_htf == 100
                                else ("1d trend direction filters 4h rsi oversold/overbought pullback trigger entries")
                            ),
                        },
                    )
                )

        elif fam == "mtf_breakout_retest":
            # G2. mtf_breakout_retest
            for _n_htf in [20, 40]:

                def _compute_g2_high_htf(
                    df_htf: pd.DataFrame,
                    window: int = _n_htf,
                ) -> pd.DataFrame:
                    return df_htf.rolling(window=window).max().shift(1)

                def _compute_g2_low_htf(
                    df_htf: pd.DataFrame,
                    window: int = _n_htf,
                ) -> pd.DataFrame:
                    return df_htf.rolling(window=window).min().shift(1)

                _proj_don_high = _resample_to_htf_and_project(
                    datetimes_4h=aligned.datetimes,
                    values_4h=high,
                    htf="1D",
                    agg_method="max",
                    compute_feature_fn=_compute_g2_high_htf,
                )
                _proj_don_low = _resample_to_htf_and_project(
                    datetimes_4h=aligned.datetimes,
                    values_4h=low,
                    htf="1D",
                    agg_method="min",
                    compute_feature_fn=_compute_g2_low_htf,
                )

                _prev_close = np.vstack([close[:1], close[:-1]])
                _long_trig = (_prev_close < _proj_don_high) & (close >= _proj_don_high)
                _short_trig = (_prev_close > _proj_don_low) & (close <= _proj_don_low)

                _g2_side = np.zeros_like(close, dtype=np.int8)
                _g2_side[_long_trig] = 1
                _g2_side[_short_trig] = -1
                _g2_score = np.where(
                    _g2_side != 0,
                    np.where(_g2_side > 0, (close - _proj_don_high) / atr, (close - _proj_don_low) / atr),
                    0.0,
                )

                fam_panels.append(
                    CandidateSignalPanel(
                        family="mtf_breakout_retest",
                        variant=f"mtf_bor_{_n_htf}",
                        params={"n_htf": _n_htf},
                        datetimes=aligned.datetimes,
                        symbols=aligned.symbols,
                        signed_score_2d=np.clip(_g2_score, -1.0, 1.0),
                        side_hint_2d=_g2_side,
                        expected_holding_bars=scale_window(12),
                        min_holding_bars=scale_window(4),
                        stop_atr_mult=2.0,
                        take_profit_atr_mult=4.0,
                        turnover_proxy_2d=np.abs(np.diff(np.clip(_g2_score, -1.0, 1.0), axis=0, prepend=0.0)),
                        valid_mask_2d=valid_mask,
                        metadata={
                            "archetype": "trend",
                            "regime": "retest_breakout",
                            "edge_hypothesis": (
                                "1d breakout channel level retested on 4h grid with subsequent continuation bounce"
                            ),
                        },
                    )
                )

        elif fam == "mtf_fusion":
            # [ADR_20260713_L0_MTF_FUSION_FACTORY] HTF filter x LTF trigger recipe factory.
            _htf_filters: tuple[tuple[str, Callable[..., pd.DataFrame], dict[str, Any]], ...] = (
                ("ema_slope", _htf_ema_slope_filter, {"span": 50}),
                ("macd_cross", _htf_macd_cross_filter, {"fast": 12, "slow": 26, "signal": 9}),
                ("hma_slope", _htf_hma_slope_filter, {"window": 20}),
                ("ichimoku_cloud", _htf_ichimoku_cloud_filter, {"tenkan": 9, "kijun": 26, "senkou_b": 52}),
                ("adx_dmi", _htf_adx_dmi_filter, {"period": 14, "threshold": 25.0}),
            )
            _ltf_trig_fn = Callable[..., tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float64]]]
            _ltf_triggers: tuple[tuple[str, _ltf_trig_fn, dict[str, Any]], ...] = (
                ("rsi_band", _ltf_rsi_band_trigger, {"period": rsi_14_window, "lo": 30.0, "hi": 70.0}),
                ("macd_cross", _ltf_macd_cross_trigger, {"atr": atr, "fast": 12, "slow": 26, "signal": 9}),
                (
                    "donchian_retest",
                    _ltf_donchian_retest_trigger,
                    {
                        "high": high,
                        "low": low,
                        "atr": atr,
                        "lookback": scale_window(20),
                    },
                ),
                (
                    "stochastic_cross",
                    _ltf_stochastic_cross_trigger,
                    {
                        "high": high,
                        "low": low,
                        "k_period": scale_window(14),
                        "d_period": 3,
                        "lo": 20.0,
                        "hi": 80.0,
                    },
                ),
            )
            for _htf in resolve_valid_htf_choices(cfg.timeframe):
                for _filter_name, _filter_fn, _filter_params in _htf_filters:
                    if _filter_name in ("ichimoku_cloud", "adx_dmi"):

                        def _mtf_fusion_ohlc_fn(
                            h: pd.DataFrame,
                            l: pd.DataFrame,
                            c: pd.DataFrame,
                            _fn: Callable[..., pd.DataFrame] = _filter_fn,
                            _p: dict[str, Any] = _filter_params,
                        ) -> pd.DataFrame:
                            return _fn(h, l, c, **_p)

                        _proj_htf_dir = _resample_ohlc_to_htf_and_project(
                            datetimes_4h=aligned.datetimes,
                            high_4h=high,
                            low_4h=low,
                            close_4h=close,
                            htf=_htf,
                            compute_feature_fn=_mtf_fusion_ohlc_fn,
                        )
                    else:

                        def _mtf_fusion_close_fn(
                            df: pd.DataFrame,
                            _fn: Callable[..., Any] = _filter_fn,
                            _p: dict[str, Any] = _filter_params,
                        ) -> pd.DataFrame:
                            return _fn(df, **_p)

                        _proj_htf_dir = _resample_to_htf_and_project(
                            datetimes_4h=aligned.datetimes,
                            values_4h=close,
                            htf=_htf,
                            agg_method="last",
                            compute_feature_fn=_mtf_fusion_close_fn,
                        )
                    for _trigger_name, _trigger_fn, _trigger_params in _ltf_triggers:
                        _long_trig, _short_trig, _mf_score = _trigger_fn(close=close, **_trigger_params)
                        _mf_side = np.zeros_like(close, dtype=np.int8)
                        _mf_side[(_proj_htf_dir > 0) & _long_trig] = 1
                        _mf_side[(_proj_htf_dir < 0) & _short_trig] = -1
                        _variant = f"mtf_fusion_{_htf}_{_filter_name}_{_trigger_name}"
                        fam_panels.append(
                            CandidateSignalPanel(
                                family="mtf_fusion",
                                variant=_variant,
                                params=apply_per_family_params(
                                    cfg,
                                    "mtf_fusion",
                                    _variant,
                                    {"htf": _htf, "filter": _filter_name, "trigger": _trigger_name},
                                ),
                                datetimes=aligned.datetimes,
                                symbols=aligned.symbols,
                                signed_score_2d=np.clip(_mf_score, -1.0, 1.0),
                                side_hint_2d=_mf_side,
                                expected_holding_bars=scale_window(16),
                                min_holding_bars=scale_window(5),
                                stop_atr_mult=2.0,
                                take_profit_atr_mult=3.0,
                                turnover_proxy_2d=np.abs(np.diff(np.clip(_mf_score, -1.0, 1.0), axis=0, prepend=0.0)),
                                valid_mask_2d=valid_mask,
                                metadata={
                                    "archetype": "trend",
                                    "regime": "mtf_fusion_factory",
                                    "edge_hypothesis": (
                                        f"{_htf} {_filter_name} regime filter gates native-TF "
                                        f"{_trigger_name} trigger entries"
                                    ),
                                },
                            )
                        )

        elif fam == "taker_imbalance_momentum":
            # G7. taker_imbalance_momentum
            for _tim_win in [scale_window(12), flow_z_24_window]:
                _cvd_z = flow_z_24 if _tim_win == flow_z_24_window else _zscore_2d(flow_imbalance, window=_tim_win)

                _g7_side = np.zeros_like(close, dtype=np.int8)
                _g7_side[_entry_rising_edge_2d(_cvd_z >= 1.5)] = 1
                _g7_side[_entry_rising_edge_2d(_cvd_z <= -1.5)] = -1
                _g7_score = np.tanh(_cvd_z / 1.5)

                fam_panels.append(
                    CandidateSignalPanel(
                        family="taker_imbalance_momentum",
                        variant=f"tim_{_tim_win}",
                        params={"window": _tim_win},
                        datetimes=aligned.datetimes,
                        symbols=aligned.symbols,
                        signed_score_2d=_g7_score,
                        side_hint_2d=_g7_side,
                        expected_holding_bars=scale_window(12),
                        min_holding_bars=scale_window(4),
                        stop_atr_mult=1.5,
                        take_profit_atr_mult=3.0,
                        turnover_proxy_2d=np.abs(np.diff(_g7_score, axis=0, prepend=0.0)),
                        valid_mask_2d=valid_mask & flow_valid,
                        metadata={
                            "archetype": "ts_mom",
                            "regime": "orderflow_momentum",
                            "edge_hypothesis": "order-flow imbalance persistence drives near-term price momentum",
                            "causal_inputs": "trailing taker-buy imbalance z-score",
                        },
                    )
                )

        elif fam == "funding_flow_carry":
            # G8. funding_flow_carry
            _ffc_side_raw = -np.sign(funding_z_96).astype(np.int8)
            _ffc_condition = (
                (np.abs(funding_z_96) >= 1.5)
                & ((_ffc_side_raw.astype(np.float64) * flow_mean_6) >= 0.10)
                & shared_valid
            )
            _ffc_entry = _entry_rising_edge_2d(_ffc_condition)
            _ffc_side = np.where(_ffc_entry, _ffc_side_raw, 0).astype(np.int8, copy=False)
            _ffc_score_mag = np.clip((np.abs(funding_z_96) - 1.5) / 1.5 + np.abs(flow_mean_6), 0.0, 1.0)
            _ffc_score = _ffc_side.astype(np.float64) * _ffc_score_mag
            _ffc_side[:funding_flow_window] = 0
            _ffc_score[:funding_flow_window] = 0.0
            fam_panels.append(
                CandidateSignalPanel(
                    family="funding_flow_carry",
                    variant="ffc_96",
                    params={
                        "funding_window": funding_flow_window,
                        "funding_z_threshold": 1.5,
                        "flow_window": flow_mean_6_window,
                    },
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=_ffc_score,
                    side_hint_2d=_ffc_side,
                    expected_holding_bars=scale_window(18),
                    min_holding_bars=scale_window(6),
                    stop_atr_mult=1.75,
                    take_profit_atr_mult=3.0,
                    turnover_proxy_2d=np.abs(np.diff(_ffc_score, axis=0, prepend=0.0)),
                    valid_mask_2d=shared_valid,
                    metadata={
                        "archetype": "carry_rev",
                        "regime": "funding_flow_confirmation",
                        "edge_hypothesis": (
                            "extreme funding mean-reverts more cleanly when taker flow "
                            "already confirms the crowded side"
                        ),
                        "causal_inputs": "trailing 96-bar funding z-score and 6-bar taker imbalance mean",
                    },
                )
            )

        elif fam == "funding_extreme_reversal":
            # G9. funding_extreme_reversal
            for _fer_win in [funding_z_168_window]:
                _f_z = funding_z_168 if _fer_win == 168 else _zscore_2d(funding, window=_fer_win, eps=1e-6)

                _g9_side = np.zeros_like(close, dtype=np.int8)
                _g9_side[_entry_rising_edge_2d(_f_z >= 1.645)] = -1
                _g9_side[_entry_rising_edge_2d(_f_z <= -1.645)] = 1
                _g9_score = np.clip(-_f_z / 1.645, -1.0, 1.0)

                fam_panels.append(
                    CandidateSignalPanel(
                        family="funding_extreme_reversal",
                        variant=f"fer_{_fer_win}",
                        params={"window": _fer_win},
                        datetimes=aligned.datetimes,
                        symbols=aligned.symbols,
                        signed_score_2d=_g9_score,
                        side_hint_2d=_g9_side,
                        expected_holding_bars=scale_window(16),
                        min_holding_bars=scale_window(4),
                        stop_atr_mult=1.5,
                        take_profit_atr_mult=2.5,
                        turnover_proxy_2d=np.abs(np.diff(_g9_score, axis=0, prepend=0.0)),
                        valid_mask_2d=valid_mask,
                        metadata={
                            "archetype": "unwind",
                            "regime": "funding_extreme",
                            "edge_hypothesis": (
                                "extreme funding rates indicate overcrowded positioning "
                                "and trigger rapid liquidation/unwind reversion"
                            ),
                        },
                    )
                )

        elif fam == "lsr_oi_regime_filter":
            # G9g. lsr_oi_regime_filter (BTN conditioning gate)
            # Blocks mean-reversion entries when LSR extreme + OI building (positioning-dominated regime)
            _loi_regime_oi_rising = oi_build_z_42 >= 0.5
            _loi_regime_lsr_extreme = np.abs(lsr_log_z_42) >= 1.0
            _loi_regime_active = _loi_regime_oi_rising & _loi_regime_lsr_extreme & positioning_valid
            _loi_regime_entry = _entry_rising_edge_2d(_loi_regime_active)
            _loi_score = np.where(
                _loi_regime_entry,
                np.clip(
                    (np.abs(lsr_log_z_42) - 1.0) / 2.0 + (oi_build_z_42 - 0.5) / 2.0,
                    0.0,
                    1.0,
                ),
                0.0,
            )

            fam_panels.append(
                CandidateSignalPanel(
                    family="lsr_oi_regime_filter",
                    variant="lsr_oi_gate_42",
                    params={"oi_window": oi_build_z_42_window, "lsr_window": lsr_log_z_42_window},
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=_loi_score,
                    side_hint_2d=np.where(
                        _loi_regime_entry,
                        -np.sign(lsr_log_z_42).astype(np.int8),  # fade the crowded side
                        0,
                    ),
                    expected_holding_bars=scale_window(24),
                    min_holding_bars=scale_window(12),
                    stop_atr_mult=1.5,
                    take_profit_atr_mult=2.0,
                    turnover_proxy_2d=np.abs(np.diff(_loi_score, axis=0, prepend=0.0)),
                    valid_mask_2d=positioning_valid,
                    metadata={
                        "archetype": "beta_neut",
                        "regime": "lsr_oi_positioning_regime",
                        "edge_hypothesis": (
                            "extreme LSR with rising OI identifies positioning-dominated regimes; "
                            "conditioning score gates trend-following vs mean-reversion allocation"
                        ),
                    },
                )
            )

        elif fam == "vol_term_structure_gate":
            # G10. vol_term_structure_gate
            for _vts_win in [scale_window(20)]:

                def _compute_g10_htf(
                    df_htf: pd.DataFrame,
                    window: int = _vts_win,
                ) -> pd.DataFrame:
                    ret = np.log(df_htf / df_htf.shift(1)).fillna(0.0)
                    return ret.rolling(window=window).std()

                _proj_htf_vol = _resample_to_htf_and_project(
                    datetimes_4h=aligned.datetimes,
                    values_4h=close,
                    htf="1D",
                    agg_method="last",
                    compute_feature_fn=_compute_g10_htf,
                )

                _ret_4h = np.diff(np.log(np.maximum(close, 1e-12)), axis=0, prepend=0.0)
                _ltf_vol = _rolling_std_2d(_ret_4h, window=_vts_win)

                _vol_ratio = _proj_htf_vol / np.maximum(_ltf_vol, 1e-12)
                _gate_active = _vol_ratio >= 1.2

                _don_high = _rolling_max_2d(high, window=_vts_win)
                _don_low = _rolling_min_2d(low, window=_vts_win)
                _trig_up = close > _don_high
                _trig_dn = close < _don_low

                _g10_side = np.zeros_like(close, dtype=np.int8)
                _g10_side[_entry_rising_edge_2d(_gate_active & _trig_up)] = 1
                _g10_side[_entry_rising_edge_2d(_gate_active & _trig_dn)] = -1
                _g10_score = np.where(_trig_up, 1.0, np.where(_trig_dn, -1.0, 0.0))

                fam_panels.append(
                    CandidateSignalPanel(
                        family="vol_term_structure_gate",
                        variant=f"vts_gate_{_vts_win}",
                        params={"window": _vts_win},
                        datetimes=aligned.datetimes,
                        symbols=aligned.symbols,
                        signed_score_2d=_g10_score,
                        side_hint_2d=_g10_side,
                        expected_holding_bars=scale_window(24),
                        min_holding_bars=scale_window(6),
                        stop_atr_mult=2.0,
                        take_profit_atr_mult=4.0,
                        turnover_proxy_2d=np.abs(np.diff(_g10_score, axis=0, prepend=0.0)),
                        valid_mask_2d=valid_mask,
                        metadata={
                            "archetype": "trend",
                            "regime": "vol_ratio_gated_trend",
                            "edge_hypothesis": (
                                "breakouts are filtered to execute only during high HTF "
                                "realized vol vs LTF realized vol to avoid whipsaws"
                            ),
                        },
                    )
                )

            # ── NEW: 6 signal families ───────────────────────────────────────────────

        elif fam == "macd_4h":
            # D. macd_4h
            _ema_12_m = _ema_2d(close, span=scale_window(12))
            _ema_26_m = _ema_2d(close, span=scale_window(26))
            _macd_line = _ema_12_m - _ema_26_m
            _macd_signal = _ema_2d(_macd_line, span=scale_window(9))
            _macd_hist = _macd_line - _macd_signal
            _macd_hist_prev = np.vstack([_macd_hist[:1], _macd_hist[:-1]])
            _macd_cross_up = (_macd_hist > 0) & (_macd_hist_prev <= 0)
            _macd_cross_down = (_macd_hist < 0) & (_macd_hist_prev >= 0)
            _macd_side = np.zeros_like(close, dtype=np.int8)
            _macd_side[_macd_cross_up] = 1
            _macd_side[_macd_cross_down] = -1
            _macd_score = np.tanh(_macd_hist / atr)
            fam_panels.append(
                CandidateSignalPanel(
                    family="macd_4h",
                    variant="macd_12_26_9",
                    params=apply_per_family_params(
                        cfg,
                        "macd_4h",
                        "macd_12_26_9",
                        {
                            "fast": 12,
                            "slow": 26,
                            "signal": 9,
                        },
                    ),
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=np.clip(_macd_score, -1.0, 1.0),
                    side_hint_2d=_macd_side,
                    expected_holding_bars=scale_window(12),
                    min_holding_bars=scale_window(4),
                    stop_atr_mult=2.0,
                    take_profit_atr_mult=3.0,
                    turnover_proxy_2d=np.abs(np.diff(_macd_score, axis=0, prepend=0.0)),
                    valid_mask_2d=valid_mask,
                    metadata={
                        "archetype": "trend",
                        "regime": "macd_crossover",
                        "edge_hypothesis": "MACD histogram zero crossover with ATR-normalized signal strength",
                    },
                )
            )

        elif fam == "supertrend":
            # E. supertrend
            _st_period = scale_window(10)
            _st_trend = _supertrend_2d(high, low, close, period=_st_period, multiplier=2.5)
            _st_prev = np.vstack([np.zeros((1, _st_trend.shape[1]), dtype=_st_trend.dtype), _st_trend[:-1]])
            _st_flip = (_st_trend != _st_prev) & (_st_prev != 0)
            _st_side = np.where(_st_flip, _st_trend, 0).astype(np.int8)
            _st_score = _st_side.astype(np.float64)
            fam_panels.append(
                CandidateSignalPanel(
                    family="supertrend",
                    variant=f"st_{_st_period}",
                    params=apply_per_family_params(
                        cfg,
                        "supertrend",
                        f"st_{_st_period}",
                        {
                            "period": _st_period,
                            "multiplier": 2.5,
                        },
                    ),
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=_st_score,
                    side_hint_2d=_st_side,
                    expected_holding_bars=scale_window(18),
                    min_holding_bars=scale_window(6),
                    stop_atr_mult=2.5,
                    take_profit_atr_mult=4.0,
                    turnover_proxy_2d=np.abs(np.diff(_st_score, axis=0, prepend=0.0)),
                    valid_mask_2d=valid_mask,
                    metadata={
                        "archetype": "trend",
                        "regime": "supertrend",
                        "edge_hypothesis": (
                            "SuperTrend ATR-based trailing stop provides clean trend signals on higher TFs"
                        ),
                    },
                )
            )

        elif fam == "ichimoku_trend":
            # F. ichimoku_trend
            _tenkan = (_rolling_max_2d(high, scale_window(9)) + _rolling_min_2d(low, scale_window(9))) / 2
            _kijun = (_rolling_max_2d(high, scale_window(26)) + _rolling_min_2d(low, scale_window(26))) / 2
            _cloud_top = (_tenkan + _kijun) / 2
            _ichi_bull = (_tenkan > _kijun) & (close > _cloud_top)
            _ichi_bear = (_tenkan < _kijun) & (close < _cloud_top)
            _ichi_side = np.zeros_like(close, dtype=np.int8)
            _ichi_side[_entry_rising_edge_2d(_ichi_bull)] = 1
            _ichi_side[_entry_rising_edge_2d(_ichi_bear)] = -1
            _ichi_score = (_tenkan - _kijun) / np.maximum(atr, 1e-12)
            fam_panels.append(
                CandidateSignalPanel(
                    family="ichimoku_trend",
                    variant="ichi_9_26",
                    params=apply_per_family_params(
                        cfg,
                        "ichimoku_trend",
                        "ichi_9_26",
                        {
                            "tenkan": 9,
                            "kijun": 26,
                        },
                    ),
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=np.clip(np.tanh(_ichi_score), -1.0, 1.0),
                    side_hint_2d=_ichi_side,
                    expected_holding_bars=scale_window(24),
                    min_holding_bars=scale_window(8),
                    stop_atr_mult=3.0,
                    take_profit_atr_mult=5.0,
                    turnover_proxy_2d=np.abs(np.diff(np.tanh(_ichi_score), axis=0, prepend=0.0)),
                    valid_mask_2d=_ichi_side.astype(bool) & valid_mask,
                    metadata={
                        "archetype": "trend",
                        "regime": "ichimoku_confirmed",
                        "edge_hypothesis": (
                            "Ichimoku 3-confirmation (TK cross + cloud) filters false breakouts on 12h TF"
                        ),
                    },
                )
            )

        elif fam == "sparse_breakout_retest_v2":
            for _sbr_channel in [scale_window(20), scale_window(40)]:
                _sbr_high = _rolling_max_2d(high, window=_sbr_channel)
                _sbr_low = _rolling_min_2d(low, window=_sbr_channel)
                _sbr_prev_close = np.vstack([close[:1], close[:-1]])
                _sbr_breakout_up = (_sbr_prev_close <= _sbr_high) & (close > _sbr_high)
                _sbr_breakout_dn = (_sbr_prev_close >= _sbr_low) & (close < _sbr_low)
                _sbr_prev_high = np.vstack([_sbr_high[:1], _sbr_high[:-1]])
                _sbr_prev_low = np.vstack([_sbr_low[:1], _sbr_low[:-1]])
                _sbr_retest_up = _sbr_breakout_up & (close <= _sbr_prev_high * 1.01)
                _sbr_retest_dn = _sbr_breakout_dn & (close >= _sbr_prev_low * 0.99)
                _sbr_side = np.zeros_like(close, dtype=np.int8)
                _sbr_side[_sbr_retest_up] = 1
                _sbr_side[_sbr_retest_dn] = -1
                _sbr_score = np.where(
                    _sbr_side > 0,
                    (close - _sbr_high) / atr,
                    np.where(_sbr_side < 0, (close - _sbr_low) / atr, 0.0),
                )
                fam_panels.append(
                    CandidateSignalPanel(
                        family="sparse_breakout_retest_v2",
                        variant=f"bor_v2_{_sbr_channel}",
                        params={"channel": _sbr_channel, "retest": 3},
                        datetimes=aligned.datetimes,
                        symbols=aligned.symbols,
                        signed_score_2d=np.clip(_sbr_score, -1.0, 1.0),
                        side_hint_2d=_sbr_side,
                        expected_holding_bars=max(scale_window(8), _sbr_channel // 4),
                        min_holding_bars=scale_window(3),
                        stop_atr_mult=2.0,
                        take_profit_atr_mult=4.0,
                        turnover_proxy_2d=np.abs(np.diff(_sbr_score, axis=0, prepend=0.0)),
                        valid_mask_2d=valid_mask,
                        metadata={
                            "archetype": "trend",
                            "regime": "sparse_breakout_retest",
                            "edge_hypothesis": (
                                "channel breakout confirmed by retest within 1% produces higher quality sparse entries"
                            ),
                        },
                    )
                )

        elif fam == "trend_pullback_quality_v2":
            for _tpq_fast, _tpq_slow, _tpq_rsi_lo, _tpq_rsi_hi in [
                (scale_window(20), scale_window(100), 40.0, 65.0),
                (scale_window(50), scale_window(200), 35.0, 70.0),
                (scale_window(100), scale_window(400), 35.0, 70.0),
            ]:
                _tpq_ema_fast = _ema_2d(close, span=_tpq_fast)
                _tpq_ema_slow = _ema_2d(close, span=_tpq_slow)
                _tpq_trend_up = _tpq_ema_fast > _tpq_ema_slow
                _tpq_trend_dn = _tpq_ema_fast < _tpq_ema_slow
                _tpq_rsi = _rsi_2d(close, period=rsi_14_window)
                _tpq_rsi_prev = np.vstack([_tpq_rsi[:1], _tpq_rsi[:-1]])
                _tpq_pullback_up = _tpq_trend_up & (close < _tpq_ema_fast) & (close >= _tpq_ema_fast * 0.97)
                _tpq_pullback_dn = _tpq_trend_dn & (close > _tpq_ema_fast) & (close <= _tpq_ema_fast * 1.03)
                _tpq_rsi_ok_long = (_tpq_rsi_prev < _tpq_rsi_lo) & (_tpq_rsi >= _tpq_rsi_lo)
                _tpq_rsi_ok_short = (_tpq_rsi_prev > _tpq_rsi_hi) & (_tpq_rsi <= _tpq_rsi_hi)
                _tpq_side = np.zeros_like(close, dtype=np.int8)
                _tpq_side[_tpq_pullback_up & _tpq_rsi_ok_long] = 1
                _tpq_side[_tpq_pullback_dn & _tpq_rsi_ok_short] = -1
                _tpq_score = np.where(
                    _tpq_side > 0,
                    np.clip((_tpq_ema_fast - close) / atr, 0.0, 2.0),
                    np.where(_tpq_side < 0, np.clip((close - _tpq_ema_fast) / atr, -2.0, 0.0), 0.0),
                )
                fam_panels.append(
                    CandidateSignalPanel(
                        family="trend_pullback_quality_v2",
                        variant=f"tpq_v2_{_tpq_fast}_{_tpq_slow}",
                        params={
                            "fast": _tpq_fast,
                            "slow": _tpq_slow,
                            "rsi_lo": int(_tpq_rsi_lo),
                            "rsi_hi": int(_tpq_rsi_hi),
                        },
                        datetimes=aligned.datetimes,
                        symbols=aligned.symbols,
                        signed_score_2d=_normalize_linear_score(_tpq_score, scale=2.0),
                        side_hint_2d=_tpq_side,
                        expected_holding_bars=max(scale_window(8), _tpq_fast // 2),
                        min_holding_bars=max(scale_window(3), _tpq_fast // 8),
                        stop_atr_mult=1.5,
                        take_profit_atr_mult=3.0,
                        turnover_proxy_2d=np.abs(
                            np.diff(_normalize_linear_score(_tpq_score, scale=2.0), axis=0, prepend=0.0)
                        ),
                        valid_mask_2d=valid_mask,
                        metadata={
                            "archetype": "trend",
                            "regime": "quality_trend_pullback",
                            "edge_hypothesis": (
                                "[LIMIT-03] slow-lookback control: turnover_per_year should drop "
                                "materially vs baseline if the archetype's survival is turnover-driven"
                                if (_tpq_fast, _tpq_slow) == (scale_window(100), scale_window(400))
                                else (
                                    "pullback entries filtered by RSI regime shift "
                                    "and EMA proximity produce higher quality trend entries"
                                )
                            ),
                        },
                    )
                )

        elif fam == "residual_momentum_xs":
            _min_xs = cfg.l1_min_cross_section
            for _rm_lb in [scale_window(12), scale_window(24)]:
                _raw = _beta_residual_return_2d(close, btc_idx, _rm_lb)
                _sc, _sd = _cross_sectional_rank_signed_2d(
                    _raw,
                    valid_mask,
                    min_cross_section=_min_xs,
                )
                fam_panels.append(
                    CandidateSignalPanel(
                        family="residual_momentum_xs",
                        variant=f"rm_xs_{_rm_lb}",
                        params={"lookback": _rm_lb, "btc_beta_cap": 0.80},
                        datetimes=aligned.datetimes,
                        symbols=aligned.symbols,
                        signed_score_2d=_sc,
                        side_hint_2d=_sd,
                        expected_holding_bars=scale_window(18) if _rm_lb <= 12 else scale_window(24),
                        min_holding_bars=scale_window(6),
                        stop_atr_mult=1.5,
                        take_profit_atr_mult=2.5,
                        turnover_proxy_2d=np.abs(np.diff(_sc, axis=0, prepend=0.0)),
                        valid_mask_2d=valid_mask,
                        metadata={
                            "archetype": "xs_alpha",
                            "edge_hypothesis": (
                                "cross-sectional residual momentum captures relative strength after removing BTC beta"
                            ),
                            "max_abs_btc_beta": 0.80,
                        },
                    )
                )

            # ── NEW: 6 alpha signal families ─────────────────────────────────────

        elif fam == "sparse_breakout_retest_liquidity":
            for _sbrl_channel in [scale_window(20), scale_window(40)]:
                _sbrl_high = _rolling_max_2d(high, window=_sbrl_channel)
                _sbrl_low = _rolling_min_2d(low, window=_sbrl_channel)
                _sbrl_prev_close = np.vstack([close[:1], close[:-1]])
                _sbrl_breakout_up = (_sbrl_prev_close <= _sbrl_high) & (close > _sbrl_high)
                _sbrl_breakout_dn = (_sbrl_prev_close >= _sbrl_low) & (close < _sbrl_low)
                _sbrl_retest_up = _sbrl_breakout_up & (close <= _sbrl_high * 1.01)
                _sbrl_retest_dn = _sbrl_breakout_dn & (close >= _sbrl_low * 0.99)
                _sbrl_side = np.zeros_like(close, dtype=np.int8)
                _sbrl_side[_sbrl_retest_up] = 1
                _sbrl_side[_sbrl_retest_dn] = -1
                _sbrl_score = np.where(
                    _sbrl_side > 0,
                    (close - _sbrl_high) / atr,
                    np.where(_sbrl_side < 0, (close - _sbrl_low) / atr, 0.0),
                )
                # Liquidity data availability gate -> valid_mask
                _liq_valid = valid_mask.copy()
                if aligned.execution_cost_bps_2d is None or aligned.adv_usdt_2d is None:
                    _liq_valid[:] = False
                fam_panels.append(
                    CandidateSignalPanel(
                        family="sparse_breakout_retest_liquidity",
                        variant=f"sbrl_{_sbrl_channel}_3",
                        params={"channel": _sbrl_channel, "retest": 3},
                        datetimes=aligned.datetimes,
                        symbols=aligned.symbols,
                        signed_score_2d=np.clip(_sbrl_score, -1.0, 1.0),
                        side_hint_2d=_sbrl_side,
                        expected_holding_bars=max(scale_window(6), _sbrl_channel // 4),
                        min_holding_bars=scale_window(2),
                        stop_atr_mult=2.0,
                        take_profit_atr_mult=4.0,
                        turnover_proxy_2d=np.abs(np.diff(_sbrl_score, axis=0, prepend=0.0)),
                        valid_mask_2d=_liq_valid,
                        metadata={
                            "archetype": "trend",
                            "regime": "sparse_breakout_retest_liquidity",
                            "edge_hypothesis": "breakout+retest w/ spread filter for liq-aware sparse entries",
                        },
                    )
                )

        elif fam == "funding_flow_exhaustion_sparse":
            _ffes_funding_z = _zscore_2d(funding, window=scale_window(96), eps=1e-6)
            _ffes_imbalance, _ffes_valid = _safe_taker_imbalance_2d(aligned.taker_buy_2d, vol)
            _ffes_imbalance_mean = _rolling_mean_2d(_ffes_imbalance, window=scale_window(12))
            _ffes_extreme = np.abs(_ffes_funding_z) >= 1.5
            _ffes_side_raw = -np.sign(_ffes_funding_z).astype(np.int8)
            _ffes_crowded = (_ffes_side_raw.astype(np.float64) * _ffes_imbalance_mean) >= 0.10
            _ffes_condition = _ffes_extreme & _ffes_crowded & valid_mask & _ffes_valid & np.isfinite(funding)
            _ffes_entry = _entry_rising_edge_2d(_ffes_condition)
            _ffes_side = np.where(_ffes_entry, -np.sign(_ffes_funding_z).astype(np.int8), 0).astype(np.int8, copy=False)
            _ffes_score = np.where(_ffes_side != 0, np.clip((np.abs(_ffes_funding_z) - 1.5) / 1.5, 0.0, 1.0), 0.0)
            _ffes_side[: scale_window(96)] = 0
            _ffes_score[: scale_window(96)] = 0.0
            fam_panels.append(
                CandidateSignalPanel(
                    family="funding_flow_exhaustion_sparse",
                    variant="ffes_96",
                    params={"funding_window": 96, "funding_z_threshold": 1.5, "imbalance_window": 12},
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=_ffes_score.astype(np.float64, copy=False),
                    side_hint_2d=_ffes_side,
                    expected_holding_bars=scale_window(12),
                    min_holding_bars=scale_window(4),
                    stop_atr_mult=1.5,
                    take_profit_atr_mult=2.5,
                    turnover_proxy_2d=np.abs(np.diff(_ffes_score, axis=0, prepend=0.0)),
                    valid_mask_2d=valid_mask & _ffes_valid & np.isfinite(funding),
                    metadata={
                        "archetype": "flow",
                        "regime": "funding_flow_exhaustion",
                        "edge_hypothesis": "funding extreme + taker crowding + OI fade for sparse flow exhaustion",
                    },
                )
            )

        elif fam == "oi_lsr_unwind":
            _oiu_log = np.where(oi_valid, np.log(oi), np.nan)
            _oiu_log_change = _oiu_log - np.roll(_oiu_log, scale_window(21), axis=0)
            _oiu_log_change[: scale_window(21)] = np.nan
            _oiu_oi_z = _zscore_2d(_oiu_log_change, window=scale_window(42))
            _oiu_lsr_z = _zscore_2d(lsr_log, window=lsr_log_z_42_window)
            _oiu_crowding = (np.abs(_oiu_lsr_z) >= 1.0) & (_oiu_oi_z > 0.5)
            _oiu_unwind = _oiu_crowding & (np.abs(_oiu_lsr_z) < np.abs(np.roll(_oiu_lsr_z, 1, axis=0)))
            _oiu_entry = _entry_rising_edge_2d(_oiu_unwind & positioning_valid)
            _oiu_side = np.where(_oiu_entry, -np.sign(_oiu_lsr_z).astype(np.int8), 0).astype(np.int8, copy=False)
            _oiu_score = np.where(_oiu_side != 0, np.clip(np.abs(_oiu_lsr_z) / 2.0, 0.0, 1.0), 0.0)
            _oiu_side[:positioning_warm_bars] = 0
            _oiu_score[:positioning_warm_bars] = 0.0
            fam_panels.append(
                CandidateSignalPanel(
                    family="oi_lsr_unwind",
                    variant="oiu_42",
                    params={"oi_window": 42, "lsr_window": 21, "z_exit": 0.5},
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=_oiu_score.astype(np.float64, copy=False),
                    side_hint_2d=_oiu_side,
                    expected_holding_bars=scale_window(18),
                    min_holding_bars=scale_window(6),
                    stop_atr_mult=1.5,
                    take_profit_atr_mult=2.5,
                    turnover_proxy_2d=np.abs(np.diff(_oiu_score, axis=0, prepend=0.0)),
                    valid_mask_2d=positioning_valid,
                    metadata={
                        "archetype": "flow",
                        "regime": "oi_lsr_unwind",
                        "edge_hypothesis": "OI/LSR crowding unwind sparse reversal entries on positioning exhaustion",
                    },
                )
            )

        elif fam == "vol_contraction_breakout":
            _vcb_bb_mean = _rolling_mean_2d(close, window=scale_window(20))
            _vcb_bb_std = _rolling_std_2d(close, window=scale_window(20))
            _vcb_bandwidth = (_vcb_bb_std * 4.0) / np.maximum(_vcb_bb_mean, 1e-12)
            _vcb_bw_mean = _rolling_mean_2d(_vcb_bandwidth, window=scale_window(120))
            _vcb_bw_std = _rolling_std_2d(_vcb_bandwidth, window=scale_window(120))
            _vcb_bw_z = (_vcb_bandwidth - _vcb_bw_mean) / np.maximum(_vcb_bw_std, 1e-12)
            _vcb_contracted = _vcb_bw_z < -1.0
            _vcb_expansion = _vcb_bandwidth > _vcb_bw_mean * 1.5
            _vcb_trig_up = _vcb_contracted & _vcb_expansion & (close > _vcb_bb_mean + _vcb_bb_std * 2.0)
            _vcb_trig_dn = _vcb_contracted & _vcb_expansion & (close < _vcb_bb_mean - _vcb_bb_std * 2.0)
            _vcb_side = np.zeros_like(close, dtype=np.int8)
            _vcb_side[_vcb_trig_up] = 1
            _vcb_side[_vcb_trig_dn] = -1
            _vcb_score = np.where(_vcb_side != 0, (close - _vcb_bb_mean) / atr, 0.0)
            fam_panels.append(
                CandidateSignalPanel(
                    family="vol_contraction_breakout",
                    variant="vcb_20_120",
                    params={"bb_window": 20, "vol_window": 120, "expansion_ratio": 1.5},
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=np.clip(_vcb_score, -1.0, 1.0),
                    side_hint_2d=_vcb_side,
                    expected_holding_bars=scale_window(12),
                    min_holding_bars=scale_window(4),
                    stop_atr_mult=1.5,
                    take_profit_atr_mult=3.0,
                    turnover_proxy_2d=np.abs(np.diff(_vcb_score, axis=0, prepend=0.0)),
                    valid_mask_2d=valid_mask,
                    metadata={
                        "archetype": "trend",
                        "regime": "vol_contraction_breakout",
                        "edge_hypothesis": "low vol squeeze + expansion breakout -> mean-reverting reversals",
                    },
                )
            )

        elif fam == "xs_residual_rebalance":
            _min_xs = cfg.l1_min_cross_section
            for _xsrr_lb in [scale_window(12), scale_window(24)]:
                _raw = _beta_residual_return_2d(close, btc_idx, _xsrr_lb)
                _pct = np.asarray(
                    pd.DataFrame(np.where(valid_mask, _raw, np.nan)).rank(axis=1, pct=True).to_numpy().copy(),
                    dtype=np.float64,
                )
                _bucket = np.floor(_pct / 0.20).astype(np.int8)
                _bucket_prev = np.vstack([_bucket[:1], _bucket[:-1]])
                _bucket_cross = _bucket != _bucket_prev
                _sign = np.sign(_raw)
                _sign_prev = np.vstack([_sign[:1], _sign[:-1]])
                _sign_flip = (
                    np.isfinite(_sign)
                    & np.isfinite(_sign_prev)
                    & (_sign != 0)
                    & (_sign_prev != 0)
                    & (_sign != _sign_prev)
                )
                _xsrr_entry = (_bucket_cross | _sign_flip) & np.isfinite(_raw)
                _count = valid_mask.sum(axis=1)
                _row_block = _count < _min_xs
                _xsrr_entry[_row_block, :] = False
                _xsrr_score = np.where(_xsrr_entry, np.clip(_pct * 2.0 - 1.0, -1.0, 1.0), 0.0)
                _xsrr_side = np.zeros_like(close, dtype=np.int8)
                _xsrr_side[_xsrr_entry & (_pct >= 0.70)] = 1
                _xsrr_side[_xsrr_entry & (_pct <= 0.30)] = -1
                _xsrr_side[_row_block, :] = 0
                fam_panels.append(
                    CandidateSignalPanel(
                        family="xs_residual_rebalance",
                        variant=f"xsrr_{_xsrr_lb}",
                        params={"rank_window": _xsrr_lb, "bucket_threshold": 0.20},
                        datetimes=aligned.datetimes,
                        symbols=aligned.symbols,
                        signed_score_2d=_xsrr_score.astype(np.float64, copy=False),
                        side_hint_2d=_xsrr_side,
                        expected_holding_bars=scale_window(18) if _xsrr_lb <= 12 else scale_window(24),
                        min_holding_bars=scale_window(6) if _xsrr_lb <= 12 else scale_window(8),
                        stop_atr_mult=1.5,
                        take_profit_atr_mult=2.5,
                        turnover_proxy_2d=np.abs(
                            np.diff(_xsrr_score.astype(np.float64, copy=False), axis=0, prepend=0.0)
                        ),
                        valid_mask_2d=valid_mask,
                        metadata={
                            "archetype": "cross_sectional",
                            "regime": "xs_residual_rebalance",
                            "edge_hypothesis": "bucket crossing or sign flip triggers xs residual rebalance",
                        },
                    )
                )

        elif fam == "carry_net_of_funding":
            _cnf_funding_z = _zscore_2d(funding, window=scale_window(96), eps=1e-6)
            _cnf_funding_ma = _rolling_mean_2d(funding, window=scale_window(24))
            _cnf_favourable = (_cnf_funding_ma < 0) & (_cnf_funding_z < 0)
            _cnf_favourable |= (_cnf_funding_ma > 0) & (_cnf_funding_z > 0)
            _cnf_carry = _log_return_2d(close, lag=scale_window(24))
            _cnf_carry_z = _zscore_2d(_cnf_carry, window=scale_window(96))
            _cnf_condition = _cnf_favourable & (np.abs(_cnf_carry_z) >= 0.5) & valid_mask & np.isfinite(funding)
            _cnf_entry = _entry_rising_edge_2d(_cnf_condition)
            _cnf_side = np.zeros_like(close, dtype=np.int8)
            _cnf_side[_cnf_entry & (_cnf_carry_z > 0)] = -1
            _cnf_side[_cnf_entry & (_cnf_carry_z < 0)] = 1
            _cnf_score = np.where(
                _cnf_side != 0,
                np.clip(np.abs(_cnf_carry_z) / 2.0, 0.0, 1.0) * np.where(_cnf_favourable, 1.0, 0.5),
                0.0,
            )
            _cnf_side[: scale_window(96)] = 0
            _cnf_score[: scale_window(96)] = 0.0
            fam_panels.append(
                CandidateSignalPanel(
                    family="carry_net_of_funding",
                    variant="cnf_96",
                    params={"funding_window": 96, "z_threshold": 0.5, "carry_window": 24},
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=_cnf_score.astype(np.float64, copy=False),
                    side_hint_2d=_cnf_side,
                    expected_holding_bars=scale_window(24),
                    min_holding_bars=scale_window(8),
                    stop_atr_mult=1.5,
                    take_profit_atr_mult=3.0,
                    turnover_proxy_2d=np.abs(np.diff(_cnf_score, axis=0, prepend=0.0)),
                    valid_mask_2d=valid_mask & np.isfinite(funding),
                    metadata={
                        "archetype": "carry",
                        "regime": "carry_net_of_funding",
                        "edge_hypothesis": "funding direction aligned with carry return filters carry/reversal entries",
                    },
                )
            )

        elif fam == "price_band_reversion":
            _pbr_mean = _rolling_mean_2d(close, window=scale_window(20))
            _pbr_std = _rolling_std_2d(close, window=scale_window(20))
            _pbr_upper = _pbr_mean + _pbr_std * 2.0
            _pbr_lower = _pbr_mean - _pbr_std * 2.0
            _prev_close = np.vstack([close[:1], close[:-1]])
            _pbr_reclaim_long = (_prev_close <= _pbr_lower) & (close > _pbr_lower)
            _pbr_reclaim_short = (_prev_close >= _pbr_upper) & (close < _pbr_upper)
            _pbr_side_std = np.zeros_like(close, dtype=np.int8)
            _pbr_side_std[_entry_rising_edge_2d(_pbr_reclaim_long)] = 1
            _pbr_side_std[_entry_rising_edge_2d(_pbr_reclaim_short)] = -1
            _pbr_score_std = np.where(
                _pbr_side_std != 0,
                np.clip((close - _pbr_mean) / np.maximum(_pbr_std, 1e-12), -2.0, 2.0) / 2.0,
                0.0,
            )
            fam_panels.append(
                CandidateSignalPanel(
                    family="price_band_reversion",
                    variant="pbr_std_20",
                    params={"band_period": 20, "band_std": 2.0},
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=_pbr_score_std.astype(np.float64, copy=False),
                    side_hint_2d=_pbr_side_std,
                    expected_holding_bars=scale_window(8),
                    min_holding_bars=scale_window(3),
                    stop_atr_mult=0.90,
                    take_profit_atr_mult=1.60,
                    turnover_proxy_2d=np.abs(np.diff(_pbr_score_std, axis=0, prepend=0.0)),
                    valid_mask_2d=valid_mask,
                    metadata={
                        "archetype": "mean_rev",
                        "regime": "price_band_reversion",
                        "edge_hypothesis": (
                            "price reclaim of trailing-exclusive std band signals faded mean-reversion entry"
                        ),
                    },
                )
            )
            _pbr_atr_lower = _pbr_mean - atr * 2.0
            _pbr_atr_upper = _pbr_mean + atr * 2.0
            _pbr_atr_reclaim_long = (_prev_close <= _pbr_atr_lower) & (close > _pbr_atr_lower)
            _pbr_atr_reclaim_short = (_prev_close >= _pbr_atr_upper) & (close < _pbr_atr_upper)
            _pbr_side_atr = np.zeros_like(close, dtype=np.int8)
            _pbr_side_atr[_entry_rising_edge_2d(_pbr_atr_reclaim_long)] = 1
            _pbr_side_atr[_entry_rising_edge_2d(_pbr_atr_reclaim_short)] = -1
            _pbr_score_atr = np.where(
                _pbr_side_atr != 0,
                np.clip((_pbr_atr_upper - close) / np.maximum(atr, 1e-12), -2.0, 2.0) / 2.0,
                0.0,
            )
            fam_panels.append(
                CandidateSignalPanel(
                    family="price_band_reversion",
                    variant="pbr_atr_20",
                    params={"band_period": 20, "atr_mult": 2.0},
                    datetimes=aligned.datetimes,
                    symbols=aligned.symbols,
                    signed_score_2d=_pbr_score_atr.astype(np.float64, copy=False),
                    side_hint_2d=_pbr_side_atr,
                    expected_holding_bars=scale_window(8),
                    min_holding_bars=scale_window(3),
                    stop_atr_mult=0.90,
                    take_profit_atr_mult=1.60,
                    turnover_proxy_2d=np.abs(np.diff(_pbr_score_atr, axis=0, prepend=0.0)),
                    valid_mask_2d=valid_mask,
                    metadata={
                        "archetype": "mean_rev",
                        "regime": "price_band_reversion",
                        "edge_hypothesis": ("price reclaim of ATR band signals faded mean-reversion entry"),
                    },
                )
            )

        return fam_panels

    active_families = list(ALL_SIGNAL_FAMILIES)
    if family_filter is not None:
        active_families = [f for f in active_families if f in family_filter]

    # A family keeps several full [time, symbol] intermediates alive while it
    # builds its panels. Four concurrent families multiply that transient peak
    # before L0 can reject any candidate, exceeding the 18 GiB runtime budget
    # on the production universe. Family ordering is deterministic and does not
    # affect signal values, so build one family at a time.
    panels: list[CandidateSignalPanel] = []
    for family in active_families:
        panels.extend(_build_single_family(family))

    from src.domain.futures.signals.causal_diversified_candidates import (
        build_causal_diversified_signal_panels,
    )

    if family_filter is None or {
        "liquidity_participation_breakout",
        "btc_neutral_residual_reversal",
    }.intersection(family_filter):
        panels.extend(
            build_causal_diversified_signal_panels(  # type: ignore[arg-type]
                aligned=aligned,
                cfg=cfg,
                valid_mask_2d=valid_mask,
                atr_2d=atr,
                btc_index=btc_idx,
            )
        )

    panels.sort(key=lambda p: (p.family, p.variant))

    # ── family_filter: keep only matching families ──
    if family_filter is not None:
        panels = [p for p in panels if p.family in family_filter]

    # ── per_family_params: apply param overrides ──
    if cfg.per_family_params is not None:
        from dataclasses import replace

        panels = [replace(p, params=apply_per_family_params(cfg, p.family, p.variant, p.params)) for p in panels]

    panels_with_context = _attach_signal_context(tuple(panels), cfg=cfg, regime_ctx=regime_ctx)
    return filter_rule_signal_panels(panels_with_context, cfg=cfg)


def candidate_panels_to_events(
    panels: Sequence[Any],
    *,
    min_abs_score: float,
    side_flip_variants: tuple[str, ...] = (),
    cost_floor_bps: float = 24.0,
    execution_cost_bps_2d: NDArray[np.float64] | None = None,
    n_workers: int = 4,
) -> pd.DataFrame:
    """Convert dense [T,N] panels into sparse candidate event rows.

    If panel.metadata["l0_event_mask_2d"] exists, it must be a bool array with the
    same shape as panel.signed_score_2d and is ANDed into the event mask.
    """
    _l0_cols = [
        "l0_discovery_unit_id",
        "l0_parent_recipe_id",
        "l0_execution_style",
        "l0_horizon_bars",
    ]
    _base_cols = [
        "datetime",
        "symbol",
        "family",
        "variant",
        "side",
        "raw_score",
        "score_z",
        "expected_holding_bars",
        "min_holding_bars",
        "stop_atr_mult",
        "take_profit_atr_mult",
        "turnover_proxy",
        "cost_floor_bps",
        "entry_idx",
        "side_flipped",
        "exit_policy_id",
        "signal_cell",
        "archetype",
        "entry_regime",
        "entry_regime_code",
    ]
    _identity_cols = [
        "l0_recipe_id",
    ]
    _all_cols = _base_cols + _l0_cols + _identity_cols
    if not panels:
        return pd.DataFrame(columns=_all_cols)

    def _convert_single_panel(panel: CandidateSignalPanel) -> list[pd.DataFrame]:
        panel_events: list[pd.DataFrame] = []
        side_flip_allowlist = _candidate_variant_set(side_flip_variants)
        scores = panel.signed_score_2d
        sides = panel.side_hint_2d
        l0_mask = panel.metadata.get("l0_event_mask_2d")
        if l0_mask is not None:
            if l0_mask.shape != scores.shape:
                raise ValueError(f"l0_event_mask_2d shape {l0_mask.shape} != signed_score_2d shape {scores.shape}")
            if l0_mask.dtype != np.bool_:
                raise ValueError("event_mask_2d must be bool")
        mask = panel.valid_mask_2d & (np.abs(scores) >= min_abs_score) & (sides != 0)
        if l0_mask is not None:
            mask = mask & l0_mask
        if not np.any(mask):
            return panel_events

        t_idx, s_idx = np.where(mask)
        if t_idx.size == 0:
            return panel_events

        event_datetimes = panel.datetimes[t_idx]
        event_symbols = np.array([panel.symbols[s] for s in s_idx], dtype=object)
        variant_key = candidate_variant_key(panel.family, panel.variant)
        side_flipped = variant_key in side_flip_allowlist
        raw_scores = scores[t_idx, s_idx]
        score_z = _cross_sectional_robust_zscore(raw_scores, t_idx.astype(np.int64, copy=False))
        event_sides = sides[t_idx, s_idx].copy()
        if side_flipped:
            raw_scores = -raw_scores
            score_z = -score_z
            event_sides = -event_sides
        event_cost = np.full(t_idx.shape[0], float(cost_floor_bps), dtype=np.float64)
        if execution_cost_bps_2d is not None:
            physical_cost = execution_cost_bps_2d[t_idx, s_idx].astype(np.float64, copy=False)
            physical_cost = np.nan_to_num(physical_cost, nan=0.0, posinf=0.0, neginf=0.0)
            event_cost = np.maximum(event_cost, physical_cost)
        turnover_proxy = panel.turnover_proxy_2d[t_idx, s_idx]
        entry_regime_codes, entry_regimes = _entry_regime_fields(panel, t_idx)
        unique_entry_regimes = tuple(dict.fromkeys(str(regime) for regime in entry_regimes))
        for regime_name in unique_entry_regimes:
            regime_mask = entry_regimes == regime_name
            if not bool(np.any(regime_mask)):
                continue
            # ── Pre-extract base arrays ONCE per regime ──
            r_datetimes = event_datetimes[regime_mask]
            r_symbols = event_symbols[regime_mask]
            r_sides = event_sides[regime_mask]
            r_raw_scores = raw_scores[regime_mask]
            r_score_z = score_z[regime_mask]
            r_turnover = turnover_proxy[regime_mask]
            r_cost = event_cost[regime_mask]
            r_entry_idx = t_idx[regime_mask] + 1
            r_entry_regime = entry_regimes[regime_mask]
            r_entry_code = entry_regime_codes[regime_mask]

            regime_policies = build_exit_policies_for_panel(
                archetype=str(panel.archetype),
                regime_name=regime_name,
                base_expected_holding_bars=int(panel.expected_holding_bars),
                base_min_holding_bars=int(panel.min_holding_bars),
                max_policies=max(1, len(panel.exit_policies)) if panel.exit_policies else 1,
                fallback_stop_atr_mult=float(panel.stop_atr_mult),
                fallback_take_profit_atr_mult=float(panel.take_profit_atr_mult),
            )
            for policy in regime_policies:
                signal_cells = np.asarray(
                    [f"{panel.family}:{panel.variant}:{policy.policy_id}:{name}" for name in r_entry_regime],
                    dtype=object,
                )
                if side_flipped:
                    signal_cells = np.asarray([f"{cell}:flip" for cell in signal_cells], dtype=object)
                df_dict: dict[str, Any] = {
                    "datetime": r_datetimes,
                    "symbol": r_symbols,
                    "family": panel.family,
                    "variant": panel.variant,
                    "side": r_sides,
                    "raw_score": r_raw_scores,
                    "score_z": r_score_z,
                    "expected_holding_bars": int(policy.expected_holding_bars),
                    "min_holding_bars": int(policy.min_holding_bars),
                    "stop_atr_mult": float(policy.stop_atr_mult),
                    "take_profit_atr_mult": float(policy.take_profit_atr_mult),
                    "turnover_proxy": r_turnover,
                    "cost_floor_bps": r_cost,
                    "entry_idx": r_entry_idx,
                    "side_flipped": side_flipped,
                    "exit_policy_id": policy.policy_id,
                    "signal_cell": signal_cells,
                    "archetype": panel.archetype,
                    "entry_regime": r_entry_regime,
                    "entry_regime_code": r_entry_code,
                }
                df_dict["l0_recipe_id"] = str(panel.metadata.get("recipe_id", ""))
                if l0_mask is not None:
                    df_dict["l0_discovery_unit_id"] = panel.metadata.get("l0_discovery_unit_id", "")
                    df_dict["l0_parent_recipe_id"] = panel.metadata.get("l0_parent_recipe_id", "")
                    df_dict["l0_execution_style"] = panel.metadata.get("l0_execution_style", "")
                    df_dict["l0_horizon_bars"] = panel.metadata.get("l0_horizon_bars", 0)
                df = pd.DataFrame(df_dict)
                panel_events.append(df)
        return panel_events

    all_events: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_convert_single_panel, panel) for panel in panels]
        for fut in futures:
            all_events.extend(fut.result())

    if not all_events:
        return pd.DataFrame(columns=_all_cols)

    return pd.concat(all_events, axis=0, ignore_index=True)
