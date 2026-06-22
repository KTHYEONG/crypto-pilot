from __future__ import annotations

import logging
from collections.abc import Callable

import numba
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.strategy.candidate_contracts import CandidateSignalPanel, SignalExitPolicy
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig, apply_per_family_params
from src.domain.futures.strategy.exit_policies import build_exit_policies_for_panel
from src.domain.futures.strategy.market_regime import MarketRegimeContext, compute_market_regime_context
from src.domain.futures.strategy.timeframe_contracts import scale_bar_count

_logger = logging.getLogger(__name__)
_ROBUST_Z_EPS = 1e-9
_ROBUST_Z_CLIP = 3.0


def candidate_variant_key(family: str, variant: str) -> str:
    """Return a stable candidate variant key."""
    return f"{family}:{variant}"


def _cross_sectional_robust_zscore(raw_scores: NDArray[np.float64], groups: NDArray[np.int64]) -> NDArray[np.float64]:
    """Return per-group robust z-scores using median/MAD normalization."""
    score_z = np.zeros(raw_scores.shape[0], dtype=np.float64)
    if raw_scores.size == 0:
        return score_z
    for group in np.unique(groups):
        group_mask = groups == group
        values = raw_scores[group_mask]
        finite_mask = np.isfinite(values)
        if not bool(finite_mask.any()):
            continue
        finite_values = values[finite_mask]
        median = float(np.median(finite_values))
        mad = float(np.median(np.abs(finite_values - median)) * 1.4826)
        normalized = np.zeros(values.shape[0], dtype=np.float64)
        if mad > _ROBUST_Z_EPS:
            normalized[finite_mask] = (finite_values - median) / mad
        score_z[group_mask] = np.clip(normalized, -_ROBUST_Z_CLIP, _ROBUST_Z_CLIP)
    return score_z


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
    result: NDArray[np.float64] = df.rolling(window=window, min_periods=1).mean().to_numpy(dtype=np.float64)
    return result


def _rolling_std_2d(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    # Time: O(T*N) via pandas C-layer; Space: O(T*N)
    if window <= 1:
        return np.zeros_like(arr, dtype=np.float64)
    df = pd.DataFrame(arr)
    raw: NDArray[np.float64] = df.rolling(window=window, min_periods=2).std().to_numpy(dtype=np.float64)
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
    prev: NDArray[np.bool_] = np.vstack(
        [np.zeros((1, condition.shape[1]), dtype=bool), condition[:-1]]
    )
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
    
    Prevents look-ahead leak by only allowing grid[t] to see higher TF values
    that were closed/released on or before grid[t].
    """
    indices = np.searchsorted(dt_higher.astype(np.int64), dt_grid.astype(np.int64), side="right") - 1
    valid_mask = indices >= 0
    clipped_indices = np.clip(indices, 0, len(dt_higher) - 1)
    
    out = feature_higher[clipped_indices]
    # Set pre-warmup/before earliest release bars to NaN
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
            feature_higher=feature_higher[:, col],
            dt_higher=dt_higher,
            dt_grid=datetimes_4h
        )
    return out_4h


def _resolve_panel_archetype(panel: CandidateSignalPanel) -> str:
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
        "vol_term_structure_gate",
    }:
        return "trend"
    if family in {"dual_momentum", "taker_imbalance_momentum"}:
        return "ts_mom"
    if family in {"funding_carry", "funding_zscore_carry", "funding_flow_carry"}:
        return "carry_rev"
    if family in {"residual_reversion"}:
        return "beta_neut"
    if family in {"funding_extreme_reversal", "funding_flow_unwind"}:
        return "unwind"
    if family in {"flow_exhaustion_reversal"}:
        return "flow_rev"
    return "mean_rev"


def _allowed_regimes_for_archetype(archetype: str) -> tuple[str, ...]:
    if archetype in {"trend", "ts_mom"}:
        return ("bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile")
    if archetype in {"flow_rev", "unwind"}:
        return ("bull_volatile", "bear_volatile", "crash")
    if archetype == "carry_rev":
        return ("bull_quiet", "bear_quiet", "transition")
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
        if cfg.regime_signal_gating_enabled or (
            cfg.mean_rev_gating_enabled and archetype == "mean_rev"
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


def _vwap_2d(close: NDArray[np.float64], volume: NDArray[np.float64], window: int = 24) -> NDArray[np.float64]:
    """Rolling VWAP for 2D arrays [T, N]."""
    n_sym = close.shape[1]
    pv = close * volume
    pv_cumsum = np.cumsum(pv, axis=0)
    v_cumsum = np.cumsum(volume, axis=0)
    pv_shifted = np.vstack([np.zeros((window, n_sym)), pv_cumsum[:-window]])
    v_shifted = np.vstack([np.zeros((window, n_sym)), v_cumsum[:-window]])
    window_pv = pv_cumsum - pv_shifted
    window_v = v_cumsum - v_shifted
    return window_pv / np.maximum(window_v, 1e-12)


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
            upper[t], upper[t - 1],
        )
        lower[t] = np.where(
            (lower[t] > lower[t - 1]) | (close[t - 1] < lower[t - 1]),
            lower[t], lower[t - 1],
        )
        trend[t] = np.where(
            prev <= 0,
            np.where(close[t] > upper[t], 1, -1),
            np.where(close[t] < lower[t], -1, 1),
        )
    return trend


def build_rule_signal_panels(
    *,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    normalize_time_horizon: bool = False,
    horizon_base_tf: str = "4h",
    family_filter: tuple[str, ...] | None = None,
) -> tuple[CandidateSignalPanel, ...]:
    """Build trailing-only rule candidates for all symbols."""
    def scale_window(base_bars: int, minimum: int = 1) -> int:
        if not normalize_time_horizon:
            return max(minimum, base_bars)
        return scale_bar_count(base_bars, cfg.timeframe, horizon_base_tf, minimum=minimum)

    regime_ctx = compute_market_regime_context(aligned=aligned)
    close = aligned.close_2d
    high = aligned.high_2d
    low = aligned.low_2d
    open = aligned.open_2d
    vol = aligned.volume_2d
    funding = aligned.funding_2d
    signal_active_mask = (
        aligned.inference_active_mask
        if aligned.inference_active_mask is not None
        else aligned.active_mask
    )
    entry_warm_mask = (
        aligned.inference_entry_warm_mask
        if aligned.inference_entry_warm_mask is not None
        else aligned.warm_mask
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
    ret_1_window = scale_window(1)
    ret_12_window = scale_window(12)
    ret_z_48_window = scale_window(48)
    oi_build_z_42_window = scale_window(42)
    lsr_log_z_42_window = scale_window(42)
    positioning_warm_bars = scale_window(168)
    rsi_14_window = scale_window(14)
    rsi_6_window = scale_window(6)
    funding_mean_window = scale_window(24)
    btc_fast_window = scale_window(20)
    btc_slow_window = scale_window(100)
    alt_mean_window = scale_window(50)
    funding_flow_window = scale_window(96)
    atr = _atr_2d(high, low, close, period=atr_period)
    atr = np.maximum(atr, 1e-12)
    flow_imbalance, flow_valid = _safe_taker_imbalance_2d(aligned.taker_buy_2d, vol)
    flow_mean_6 = _rolling_mean_2d(flow_imbalance, window=flow_mean_6_window)
    flow_z_24 = _zscore_2d(flow_imbalance, window=flow_z_24_window)
    funding_z_96 = _zscore_2d(funding, window=funding_z_96_window, eps=1e-6)
    funding_z_168 = _zscore_2d(funding, window=funding_z_168_window, eps=1e-6)
    ret_1 = _log_return_2d(close, lag=ret_1_window)
    ret_12 = _log_return_2d(close, lag=ret_12_window)
    ret_z_48 = _zscore_2d(ret_12, window=ret_z_48_window)
    shared_valid = valid_mask & flow_valid & np.isfinite(funding)
    fxr_valid = valid_mask & flow_valid
    oi = (
        aligned.oi_2d
        if aligned.oi_2d is not None
        else np.full_like(close, np.nan, dtype=np.float64)
    )
    lsr = (
        aligned.lsr_2d
        if aligned.lsr_2d is not None
        else np.full_like(close, np.nan, dtype=np.float64)
    )
    oi_valid = np.isfinite(oi) & (oi > 0.0)
    lsr_valid = np.isfinite(lsr) & (lsr > 0.0)
    oi_log = np.where(oi_valid, np.log(oi), np.nan)
    lsr_log = np.where(lsr_valid, np.log(lsr), np.nan)
    oi_log_change_6 = oi_log - np.roll(oi_log, flow_mean_6_window, axis=0)
    oi_log_change_6[:flow_mean_6_window] = np.nan
    oi_build_z_42 = _zscore_2d(oi_log_change_6, window=oi_build_z_42_window)
    lsr_log_z_42 = _zscore_2d(lsr_log, window=lsr_log_z_42_window)
    # UNW warm-up: require 168 bars of continuous valid data for z-score stability
    positioning_warm = np.ones_like(valid_mask, dtype=np.bool_)
    positioning_warm[:positioning_warm_bars] = False
    positioning_valid = (valid_mask & flow_valid & np.isfinite(funding) & oi_valid & lsr_valid
                         & positioning_warm)
    # Shared derived features for new panels
    funding_ts_slope = funding_z_96 - funding_z_168  # positive = short-term more extreme

    panels: list[CandidateSignalPanel] = []

    # 1. Trend MA Cross
    ema_fast = _ema_2d(close, span=scale_window(12))
    ema_slow = _ema_2d(close, span=scale_window(72))
    ma_diff = (ema_fast - ema_slow) / atr
    signed_score_ma = np.tanh(ma_diff)
    side_hint_ma = np.zeros_like(signed_score_ma, dtype=np.int8)
    side_hint_ma[ma_diff > 0.5] = 1
    side_hint_ma[ma_diff < -0.5] = -1
    panels.append(
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
    side_hint_ma_18_108[ma_diff_18_108 > 0.5] = 1
    side_hint_ma_18_108[ma_diff_18_108 < -0.5] = -1
    panels.append(
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

    # 2. Trend Donchian — donchian_72 only (donchian_18/36 removed: p>0.34, possym<0.50)
    d72_high = _rolling_max_2d(high, window=scale_window(72))
    d72_low = _rolling_min_2d(low, window=scale_window(72))
    d72_side = np.zeros_like(close, dtype=np.int8)
    d72_side[close > d72_high] = 1
    d72_side[close < d72_low] = -1
    d72_score = np.zeros_like(close)
    above_72 = close > d72_high
    below_72 = close < d72_low
    d72_score[above_72] = (close[above_72] - d72_high[above_72]) / atr[above_72]
    d72_score[below_72] = (close[below_72] - d72_low[below_72]) / atr[below_72]
    panels.append(
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

    # 3. Vol Breakout
    bb_mean = _rolling_mean_2d(close, window=scale_window(20))
    bb_std = _rolling_std_2d(close, window=scale_window(20))
    bandwidth = (bb_std * 4.0) / np.maximum(bb_mean, 1e-12)
    bw_mean_120 = _rolling_mean_2d(bandwidth, window=scale_window(120))
    bw_std_120 = _rolling_std_2d(bandwidth, window=scale_window(120))
    bw_z = (bandwidth - bw_mean_120) / np.maximum(bw_std_120, 1e-12)
    compressed = bw_z < -1.0
    vol_side = np.zeros_like(close, dtype=np.int8)
    vol_side[compressed & (close > bb_mean + bb_std * 2.0)] = 1
    vol_side[compressed & (close < bb_mean - bb_std * 2.0)] = -1
    vol_score = np.where(vol_side != 0, (close - bb_mean) / atr, 0.0)
    panels.append(
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

    # 4. Bollinger Reversion
    bb_mean_rev = _rolling_mean_2d(close, window=scale_window(20))
    bb_std_rev = _rolling_std_2d(close, window=scale_window(20))
    bb_z_rev = (close - bb_mean_rev) / np.maximum(bb_std_rev, 1e-12)
    rev_side = np.zeros_like(close, dtype=np.int8)
    rev_side[_entry_rising_edge_2d(bb_z_rev < -2.0)] = 1
    rev_side[_entry_rising_edge_2d(bb_z_rev > 2.0)] = -1
    rev_score = -bb_z_rev / 3.0
    panels.append(
        CandidateSignalPanel(
            family="bollinger_reversion",
            variant="bollinger_20",
            params={"window": scale_window(20), "entry_z": 2.0},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(rev_score, -1.0, 1.0),
            side_hint_2d=rev_side,
            expected_holding_bars=scale_window(12),
            min_holding_bars=scale_window(4),
            stop_atr_mult=2.5,
            take_profit_atr_mult=3.0,
            turnover_proxy_2d=np.abs(np.diff(rev_score, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
            metadata={
                "edge_hypothesis": (
                    "first breach of extreme z-score produces mean-reversion entry"
                    " with higher expected edge than subsequent persistent-state bars"
                ),
                "causal_inputs": "trailing 20-bar rolling mean and std of close price",
                "expected_failure_mode": (
                    "trending markets where z-score remains extreme without reverting"
                ),
            },
        )
    )

    # 5. RSI Reversion
    rsi = _rsi_2d(close, period=rsi_14_window)
    rsi_prev = np.vstack([rsi[:1], rsi[:-1]])
    rsi_side = np.zeros_like(close, dtype=np.int8)
    rsi_side[(rsi_prev < 30) & (rsi > rsi_prev)] = 1
    rsi_side[(rsi_prev > 70) & (rsi < rsi_prev)] = -1
    rsi_score = (50.0 - rsi) / 20.0
    panels.append(
        CandidateSignalPanel(
            family="rsi_reversion",
            variant="rsi_14",
            params={"rsi_period": rsi_14_window, "oversold": 30.0, "overbought": 70.0},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(rsi_score, -1.0, 1.0),
            side_hint_2d=rsi_side,
            expected_holding_bars=scale_window(12),
            min_holding_bars=scale_window(4),
            stop_atr_mult=2.0,
            take_profit_atr_mult=3.0,
            turnover_proxy_2d=np.abs(np.diff(rsi_score, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
        )
    )

    # 5b. RSI Reversion — rsi_6
    rsi6 = _rsi_2d(close, period=rsi_6_window)
    rsi6_prev = np.vstack([rsi6[:1], rsi6[:-1]])
    rsi6_side = np.zeros_like(close, dtype=np.int8)
    rsi6_side[(rsi6_prev < 20) & (rsi6 > rsi6_prev)] = 1
    rsi6_side[(rsi6_prev > 80) & (rsi6 < rsi6_prev)] = -1
    rsi6_score = (50.0 - rsi6) / 20.0
    panels.append(
        CandidateSignalPanel(
            family="rsi_reversion",
            variant="rsi_6",
            params={"rsi_period": rsi_6_window, "oversold": 20.0, "overbought": 80.0},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(rsi6_score, -1.0, 1.0),
            side_hint_2d=rsi6_side,
            expected_holding_bars=scale_window(8),
            min_holding_bars=scale_window(2),
            stop_atr_mult=1.5,
            take_profit_atr_mult=2.5,
            turnover_proxy_2d=np.abs(np.diff(rsi6_score, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
        )
    )

    # 6. Funding Carry
    funding_mean = _rolling_mean_2d(funding, window=funding_mean_window)
    funding_std = _rolling_std_2d(funding, window=funding_mean_window)
    funding_z = (funding - funding_mean) / np.maximum(funding_std, 1e-6)
    carry_side = np.zeros_like(close, dtype=np.int8)
    carry_side[funding_z < -1.5] = 1
    carry_side[funding_z > 1.5] = -1
    carry_score = -funding_z / 2.0
    panels.append(
        CandidateSignalPanel(
            family="funding_carry",
            variant="funding_24",
            params={"window": funding_mean_window, "entry_z": 1.5},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(carry_score, -1.0, 1.0),
            side_hint_2d=carry_side,
            expected_holding_bars=scale_window(24),
            min_holding_bars=scale_window(8),
            stop_atr_mult=2.0,
            take_profit_atr_mult=3.0,
            turnover_proxy_2d=np.abs(np.diff(carry_score, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
            metadata={"archetype": "carry_rev"},
        )
    )

    # 8. BTC Regime Pullback
    btc_idx = 0
    for idx, sym in enumerate(aligned.symbols):
        if "BTC" in sym:
            btc_idx = idx
            break
    btc_close = close[:, btc_idx : btc_idx + 1]
    btc_ema_fast = _ema_2d(btc_close, span=btc_fast_window)
    btc_ema_slow = _ema_2d(btc_close, span=btc_slow_window)
    btc_trend_up = btc_ema_fast > btc_ema_slow

    alt_mean = _rolling_mean_2d(close, window=alt_mean_window)
    alt_std = _rolling_std_2d(close, window=alt_mean_window)
    alt_pullback_z = (close - alt_mean) / np.maximum(alt_std, 1e-12)

    btc_side = np.zeros_like(close, dtype=np.int8)
    btc_side[btc_trend_up & (alt_pullback_z < -1.5)] = 1
    btc_side[~btc_trend_up & (alt_pullback_z > 1.5)] = -1
    btc_score = -alt_pullback_z / 2.0
    panels.append(
        CandidateSignalPanel(
            family="btc_regime_pullback",
            variant="btc_pullback_50",
            params={"window": alt_mean_window, "btc_fast": btc_fast_window, "btc_slow": btc_slow_window},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(btc_score, -1.0, 1.0),
            side_hint_2d=btc_side,
            expected_holding_bars=scale_window(18),
            min_holding_bars=scale_window(6),
            stop_atr_mult=2.0,
            take_profit_atr_mult=3.0,
            turnover_proxy_2d=np.abs(np.diff(btc_score, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
        )
    )


    # 10. Funding Z-Score Carry (F2)
    for _fz_win, _fz_thr in [
        (scale_window(48), 2.0),
        (scale_window(96), 2.0),
        (scale_window(168), 1.5),
    ]:
        if _fz_win == funding_z_96_window:
            _f_z = funding_z_96
        elif _fz_win == funding_z_168_window:
            _f_z = funding_z_168
        else:
            _f_z = _zscore_2d(funding, window=_fz_win, eps=1e-6)
        _fz_side = np.zeros_like(close, dtype=np.int8)
        _fz_side[_f_z >= _fz_thr] = -1   # extreme positive funding → mean reversion short
        _fz_side[_f_z <= -_fz_thr] = 1   # extreme negative funding → long
        _fz_score = np.clip(-_f_z / _fz_thr, -1.0, 1.0)
        panels.append(
            CandidateSignalPanel(
                family="funding_zscore_carry",
                variant=f"fzs_{_fz_win}",
                params={"z_window": _fz_win, "z_threshold": _fz_thr},
                datetimes=aligned.datetimes,
                symbols=aligned.symbols,
                signed_score_2d=_fz_score,
                side_hint_2d=_fz_side,
                expected_holding_bars=max(1, _fz_win // 2),
                min_holding_bars=scale_window(8),
                stop_atr_mult=2.0,
                take_profit_atr_mult=3.0,
                turnover_proxy_2d=np.abs(np.diff(_fz_score, axis=0, prepend=0.0)),
                valid_mask_2d=valid_mask,
                metadata={"archetype": "carry_rev"},
            )
        )

    # 11. Vol Regime Reversion (F3)
    _price_dir = np.sign(np.diff(close, axis=0, prepend=close[:1]))
    for _vr_win, _vr_thr in [(scale_window(20), 2.0), (scale_window(40), 1.5)]:
        _atr_mean = _rolling_mean_2d(atr, window=_vr_win)
        _atr_std = _rolling_std_2d(atr, window=_vr_win)
        _vol_z = (atr - _atr_mean) / np.maximum(_atr_std, 1e-12)
        _high_vol = _vol_z >= _vr_thr
        _vr_side = np.zeros_like(close, dtype=np.int8)
        _vr_side[_entry_rising_edge_2d(_high_vol & (_price_dir > 0))] = -1  # high vol + up move → fade (short)
        _vr_side[_entry_rising_edge_2d(_high_vol & (_price_dir < 0))] = 1   # high vol + down move → fade (long)
        _vr_score = np.clip(-_vol_z / _vr_thr * _price_dir, -1.0, 1.0)
        panels.append(
            CandidateSignalPanel(
                family="vol_regime_reversion",
                variant=f"vrr_{_vr_win}",
                params={"vol_window": _vr_win, "vol_z_threshold": _vr_thr},
                datetimes=aligned.datetimes,
                symbols=aligned.symbols,
                signed_score_2d=_vr_score,
                side_hint_2d=_vr_side,
                expected_holding_bars=_vr_win,
                min_holding_bars=scale_window(4),
                stop_atr_mult=2.0,
                take_profit_atr_mult=3.0,
                turnover_proxy_2d=np.abs(np.diff(_vr_score, axis=0, prepend=0.0)),
                valid_mask_2d=valid_mask,
                metadata={
                    "edge_hypothesis": (
                        "first appearance of high-vol spike with directional move produces"
                        " mean-reversion entry; subsequent bars carry diminishing edge"
                    ),
                    "causal_inputs": "trailing ATR z-score and single-bar price direction",
                    "expected_failure_mode": (
                        "sustained volatility regimes where initial spike does not revert"
                    ),
                },
            )
        )



    # 16. Trend Pullback Continuation
    for _fast, _slow, _rsi_lo, _rsi_hi in [
        (scale_window(20), scale_window(100), 40.0, 65.0),
        (scale_window(50), scale_window(200), 40.0, 65.0),
    ]:
        _ema_fast = _ema_2d(close, span=_fast)
        _ema_slow = _ema_2d(close, span=_slow)
        _close_prev = np.vstack([close[:1], close[:-1]])
        _ema_fast_prev = np.vstack([_ema_fast[:1], _ema_fast[:-1]])
        _rsi_trend = _rsi_2d(close, period=rsi_14_window)
        _trend_up = _ema_fast > _ema_slow
        _trend_dn = _ema_fast < _ema_slow
        _long = _trend_up & (_close_prev <= _ema_fast_prev) & (close > _ema_fast) & (_rsi_trend >= _rsi_lo) & (
            _rsi_trend <= _rsi_hi
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
        panels.append(
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
                        "pullback recovery inside an established trend improves continuation "
                        "entry quality"
                    ),
                },
            )
        )

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
        panels.append(
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



    # 20. Residual Reversion
    _log_ret_rr = np.diff(np.log(np.maximum(close, 1e-12)), axis=0, prepend=0.0)
    _btc_ret_rr = _log_ret_rr[:, btc_idx : btc_idx + 1]
    for _rr_win in [scale_window(24), scale_window(48)]:
        _btc_var_rr = _rolling_mean_2d(_btc_ret_rr ** 2, window=_rr_win)
        _cov_alt_btc_rr = _rolling_mean_2d(_log_ret_rr * _btc_ret_rr, window=_rr_win)
        _beta_hat_rr = _cov_alt_btc_rr / np.maximum(_btc_var_rr, 1e-12)
        _resid_ret_rr = _log_ret_rr - _beta_hat_rr * _btc_ret_rr
        _resid_mean_rr = _rolling_mean_2d(_resid_ret_rr, window=_rr_win)
        _resid_std_rr = _rolling_std_2d(_resid_ret_rr, window=_rr_win)
        _resid_z_rr = (_resid_ret_rr - _resid_mean_rr) / np.maximum(_resid_std_rr, 1e-12)
        _rr_side = np.zeros_like(close, dtype=np.int8)
        _rr_side[_resid_z_rr <= -2.0] = 1
        _rr_side[_resid_z_rr >= 2.0] = -1
        _rr_score = np.where(_rr_side != 0, -_normalize_linear_score(_resid_z_rr, scale=3.0), 0.0)
        panels.append(
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
                turnover_proxy_2d=np.abs(np.diff(_rr_score.astype(np.float64, copy=False), axis=0, prepend=0.0)),
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
    # NEW G1 ~ G10 FAMILIES
    # =========================================================================

    # G1. mtf_trend_pullback
    for _n_htf in [20, 50]:
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
            compute_feature_fn=_compute_g1_htf
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
        
        panels.append(
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
                        "1d trend direction filters 4h rsi oversold/overbought "
                        "pullback trigger entries"
                    ),
                },
            )
        )

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
        
        panels.append(
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
                        "1d breakout channel level retested on 4h grid with "
                        "subsequent continuation bounce"
                    ),
                },
            )
        )

    # G7. taker_imbalance_momentum
    for _tim_win in [scale_window(12), flow_z_24_window]:
        _cvd_z = flow_z_24 if _tim_win == flow_z_24_window else _zscore_2d(flow_imbalance, window=_tim_win)
        
        _g7_side = np.zeros_like(close, dtype=np.int8)
        _g7_side[_cvd_z >= 1.5] = 1
        _g7_side[_cvd_z <= -1.5] = -1
        _g7_score = np.tanh(_cvd_z / 1.5)
        
        panels.append(
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
    panels.append(
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

    # G9. funding_extreme_reversal
    for _fer_win in [funding_z_168_window]:
        _f_z = funding_z_168 if _fer_win == 168 else _zscore_2d(funding, window=_fer_win, eps=1e-6)
        
        _g9_side = np.zeros_like(close, dtype=np.int8)
        _g9_side[_f_z >= 1.645] = -1
        _g9_side[_f_z <= -1.645] = 1
        _g9_score = np.clip(-_f_z / 1.645, -1.0, 1.0)
        
        panels.append(
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

    # G9b. funding_flow_unwind
    _ffu_crowded_side = np.sign(funding_z_168).astype(np.int8)
    _ffu_crowded = (np.abs(funding_z_168) >= 1.645) & ((_ffu_crowded_side.astype(np.float64) * ret_z_48) >= 1.0)
    _ffu_reversal = ((_ffu_crowded_side.astype(np.float64) * flow_z_24) <= -1.0) & (
        (_ffu_crowded_side.astype(np.float64) * ret_1) < 0.0
    )
    _ffu_condition = _ffu_crowded & _ffu_reversal & shared_valid
    _ffu_entry = _entry_rising_edge_2d(_ffu_condition)
    _ffu_side = np.where(_ffu_entry, -_ffu_crowded_side, 0).astype(np.int8, copy=False)
    _ffu_score_mag = np.clip((np.abs(funding_z_168) - 1.645) / 1.645 + np.abs(flow_z_24) / 2.0, 0.0, 1.0)
    _ffu_score = _ffu_side.astype(np.float64) * _ffu_score_mag
    _ffu_side[:funding_z_168_window] = 0
    _ffu_score[:funding_z_168_window] = 0.0
    panels.append(
        CandidateSignalPanel(
            family="funding_flow_unwind",
            variant="ffu_168",
            params={
                "funding_window": funding_z_168_window,
                "funding_z_threshold": 1.645,
                "flow_window": flow_z_24_window,
                "ret_window": ret_z_48_window,
            },
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=_ffu_score,
            side_hint_2d=_ffu_side,
            expected_holding_bars=scale_window(10),
            min_holding_bars=scale_window(3),
            stop_atr_mult=1.5,
            take_profit_atr_mult=2.5,
            turnover_proxy_2d=np.abs(np.diff(_ffu_score, axis=0, prepend=0.0)),
            valid_mask_2d=shared_valid,
            metadata={
                "archetype": "unwind",
                "regime": "funding_flow_reversal",
                "edge_hypothesis": (
                    "crowded funding regimes unwind when order flow and one-bar "
                    "returns flip together"
                ),
                "causal_inputs": (
                    "trailing 168-bar funding z-score, 24-bar flow z-score, "
                    "12-bar return z-score, and 1-bar return"
                ),
            },
        )
    )

    # G9c. flow_exhaustion_reversal
    _fxr_shock_side = np.sign(ret_z_48).astype(np.int8)
    _fxr_exhausted = (np.abs(ret_z_48) >= 1.0) & ((_fxr_shock_side.astype(np.float64) * flow_z_24) >= 2.0)
    _fxr_price_reversal = (_fxr_shock_side.astype(np.float64) * ret_1) < 0.0
    _fxr_condition = _fxr_exhausted & _fxr_price_reversal & fxr_valid
    _fxr_entry = _entry_rising_edge_2d(_fxr_condition)
    _fxr_side = np.where(_fxr_entry, -_fxr_shock_side, 0).astype(np.int8, copy=False)
    _fxr_ret_excess = np.clip(np.abs(ret_z_48) - 1.0, 0.0, 1.0)
    _fxr_flow_excess = np.clip((np.abs(flow_z_24) - 2.0) / 2.0, 0.0, 1.0)
    _fxr_score_mag = 0.5 * _fxr_ret_excess + 0.5 * _fxr_flow_excess
    _fxr_score = _fxr_side.astype(np.float64) * _fxr_score_mag
    _fxr_side[:ret_z_48_window] = 0
    _fxr_score[:ret_z_48_window] = 0.0
    panels.append(
        CandidateSignalPanel(
            family="flow_exhaustion_reversal",
            variant="fxr_24",
            params={"flow_window": flow_z_24_window, "shock_window": ret_z_48_window, "shock_z_threshold": 1.0},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=_fxr_score,
            side_hint_2d=_fxr_side,
            expected_holding_bars=scale_window(8),
            min_holding_bars=scale_window(2),
            stop_atr_mult=1.25,
            take_profit_atr_mult=2.0,
            turnover_proxy_2d=np.abs(np.diff(_fxr_score, axis=0, prepend=0.0)),
            valid_mask_2d=fxr_valid,
            metadata={
                "archetype": "flow_rev",
                "regime": "flow_exhaustion_reversal",
                "edge_hypothesis": (
                    "price shocks with simultaneous order-flow extremes tend "
                    "to snap back on immediate reversal bars"
                ),
                "causal_inputs": (
                    "trailing 12-bar return z-score, 24-bar flow z-score, "
                    "and 1-bar return"
                ),
            },
        )
    )

    # G9d. positioning_unwind
    _pu_crowded_side = np.sign(funding_z_168).astype(np.int8)
    _pu_crowding = (
        (np.abs(funding_z_168) >= 1.0)
        & ((_pu_crowded_side.astype(np.float64) * lsr_log_z_42) >= 0.75)
        & (oi_build_z_42 >= 0.75)
    )
    _pu_reversal = (
        (_pu_crowded_side.astype(np.float64) * flow_z_24 <= -1.0)
        & (_pu_crowded_side.astype(np.float64) * ret_1 < 0.0)
    )
    _pu_condition = _pu_crowding & _pu_reversal & positioning_valid
    _pu_entry = _entry_rising_edge_2d(_pu_condition)
    _pu_side = np.where(_pu_entry, -_pu_crowded_side, 0).astype(np.int8, copy=False)
    _pu_funding_excess = np.clip(np.abs(funding_z_168) - 1.0, 0.0, 1.0)
    _pu_lsr_excess = np.clip(np.abs(lsr_log_z_42) - 0.75, 0.0, 1.0)
    _pu_oi_excess = np.clip(oi_build_z_42 - 0.75, 0.0, 1.0)
    _pu_flow_excess = np.clip((-_pu_crowded_side.astype(np.float64) * flow_z_24) - 1.0, 0.0, 1.0)
    _pu_score_mag = 0.25 * (
        _pu_funding_excess + _pu_lsr_excess + _pu_oi_excess + _pu_flow_excess
    )
    _pu_score = _pu_side.astype(np.float64) * _pu_score_mag
    _pu_side[:funding_z_168_window] = 0
    _pu_score[:funding_z_168_window] = 0.0
    panels.append(
        CandidateSignalPanel(
            family="positioning_unwind",
            variant="pu_42",
            params={
                "funding_window": funding_z_168_window,
                "positioning_window": oi_build_z_42_window,
                "oi_lag": 6,
            },
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=_pu_score,
            side_hint_2d=_pu_side,
            expected_holding_bars=scale_window(10),
            min_holding_bars=scale_window(3),
            stop_atr_mult=1.5,
            take_profit_atr_mult=2.5,
            turnover_proxy_2d=np.abs(np.diff(_pu_score, axis=0, prepend=0.0)),
            valid_mask_2d=positioning_valid,
            metadata={
                "archetype": "unwind",
                "regime": "positioning_unwind",
                "edge_hypothesis": (
                    "crowded positioning with rising open interest and long-short skew "
                    "unwinds when flow and price reverse together"
                ),
            },
        )
    )

    # G9e. funding_term_structure_carry
    # Entry when funding acceleration is same-direction (short-term z more extreme than long-term)
    _fts_crowded = np.abs(funding_z_168) >= 0.5  # meaningful funding level
    _fts_accel = np.abs(funding_ts_slope) >= 0.75  # acceleration above noise
    _fts_same_dir = (funding_z_96 * funding_z_168) > 0.0  # same direction
    _fts_entry_cond = _fts_crowded & _fts_accel & _fts_same_dir & shared_valid
    _fts_entry = _entry_rising_edge_2d(_fts_entry_cond)
    _fts_side = np.where(_fts_entry, np.sign(funding_z_168).astype(np.int8), 0)
    _fts_score = _fts_side.astype(np.float64) * np.clip(
        np.abs(funding_ts_slope) - 0.75, 0.0, 1.0
    )

    panels.append(
        CandidateSignalPanel(
            family="funding_term_structure_carry",
            variant="fts_carry_96168",
            params={"short_window": funding_z_96_window, "long_window": funding_z_168_window},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=_fts_score,
            side_hint_2d=_fts_side,
            expected_holding_bars=scale_window(12),
            min_holding_bars=scale_window(4),
            stop_atr_mult=1.5,
            take_profit_atr_mult=2.0,
            turnover_proxy_2d=np.abs(np.diff(_fts_score, axis=0, prepend=0.0)),
            valid_mask_2d=shared_valid,
            metadata={
                "archetype": "carry_rev",
                "regime": "funding_ts_carry",
                "edge_hypothesis": (
                    "funding acceleration (short-term z > long-term z) "
                    "signals inflow continuing before mean-reversion"
                ),
            },
        )
    )

    # G9f. flow_trend_continuation
    # Entry when flow supports ongoing trend (not reversal but continuation)
    _flo_cont_flow_ok = flow_z_24 >= 1.0    # strong flow
    _flo_cont_ret_trend = ret_12 > 0.0       # positive 12-bar trend
    _flo_cont_ret_inertia = ret_1 > 0.0      # same-bar continuation
    _flo_cont_cond = (_flo_cont_flow_ok & _flo_cont_ret_trend & _flo_cont_ret_inertia
                      & shared_valid)
    _flo_cont_entry = _entry_rising_edge_2d(_flo_cont_cond)
    _flo_cont_side = np.where(_flo_cont_entry, 1, 0).astype(np.int8)  # long-only continuation
    _flo_cont_score = np.where(
        _flo_cont_entry,
        np.clip(flow_z_24 / 3.0, 0.0, 1.0) * np.clip(ret_12, 0.0, 0.03) / 0.03,
        0.0,
    )

    panels.append(
        CandidateSignalPanel(
            family="flow_trend_continuation",
            variant="flo_cont_24",
            params={"flow_window": flow_z_24_window, "ret_window": ret_12_window},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=_flo_cont_score,
            side_hint_2d=_flo_cont_side,
            expected_holding_bars=scale_window(8),
            min_holding_bars=scale_window(3),
            stop_atr_mult=1.5,
            take_profit_atr_mult=2.0,
            turnover_proxy_2d=np.abs(np.diff(_flo_cont_score, axis=0, prepend=0.0)),
            valid_mask_2d=shared_valid,
            metadata={
                "archetype": "ts_mom",
                "regime": "flow_momentum_continuation",
                "edge_hypothesis": (
                    "strong flow supporting ongoing price trend signals "
                    "continuation before exhaustion"
                ),
            },
        )
    )

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
            0.0, 1.0,
        ),
        0.0,
    )

    panels.append(
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
        _g10_side[_gate_active & _trig_up] = 1
        _g10_side[_gate_active & _trig_dn] = -1
        _g10_score = np.where(_trig_up, 1.0, np.where(_trig_dn, -1.0, 0.0))
        
        panels.append(
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

    # A. gap_fade_1h
    _open_minus_close = open - np.vstack([close[:1], close[:-1]])
    _gap = _open_minus_close / np.maximum(atr, 1e-12)
    _gap_extreme = np.abs(_gap) > 2.0
    _gap_side = np.zeros_like(close, dtype=np.int8)
    _gap_side[_gap > 2.0] = -1
    _gap_side[_gap < -2.0] = 1
    _gap_score = -np.tanh(_gap / 2.0)
    panels.append(
        CandidateSignalPanel(
            family="gap_fade_1h",
            variant="gf_1h",
            params=apply_per_family_params(cfg, "gap_fade_1h", "gf_1h", {
                "entry_z": 2.0,
            }),
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(_gap_score, -1.0, 1.0),
            side_hint_2d=_gap_side,
            expected_holding_bars=scale_window(4),
            min_holding_bars=scale_window(1),
            stop_atr_mult=1.0,
            take_profit_atr_mult=2.0,
            turnover_proxy_2d=np.abs(np.diff(_gap_score, axis=0, prepend=0.0)),
            valid_mask_2d=_gap_extreme & valid_mask,
            metadata={
                "archetype": "mean_rev",
                "regime": "gap_reversion",
                "edge_hypothesis": "extreme open-close gap mean-reverts within 4 bars in 1h timeframe",
            },
        )
    )

    # B. vwap_reversion_1h
    _vwap = _vwap_2d(close, vol, window=scale_window(24))
    _vwap_dev = (close - _vwap) / np.maximum(_rolling_std_2d(close, scale_window(24)), 1e-12)
    _vwap_extreme = np.abs(_vwap_dev) > 2.0
    _vwap_side = np.zeros_like(close, dtype=np.int8)
    _vwap_side[_vwap_dev > 2.0] = -1
    _vwap_side[_vwap_dev < -2.0] = 1
    _vwap_score = -np.tanh(_vwap_dev / 2.0)
    panels.append(
        CandidateSignalPanel(
            family="vwap_reversion_1h",
            variant="vwap_24",
            params=apply_per_family_params(cfg, "vwap_reversion_1h", "vwap_24", {
                "vwap_window": 24,
            }),
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(_vwap_score, -1.0, 1.0),
            side_hint_2d=_vwap_side,
            expected_holding_bars=scale_window(6),
            min_holding_bars=scale_window(2),
            stop_atr_mult=1.5,
            take_profit_atr_mult=2.5,
            turnover_proxy_2d=np.abs(np.diff(_vwap_score, axis=0, prepend=0.0)),
            valid_mask_2d=_vwap_extreme & valid_mask,
            metadata={
                "archetype": "mean_rev",
                "regime": "vwap_gap_reversion",
                "edge_hypothesis": "24h VWAP deviation above 2 sigma mean-reverts in 1h crypto market",
            },
        )
    )

    # C. volume_climax_1h
    _vol_ma20 = _rolling_mean_2d(vol, window=scale_window(20))
    _vol_std20 = _rolling_std_2d(vol, window=scale_window(20))
    _vol_z = (vol - _vol_ma20) / np.maximum(_vol_std20, 1e-12)
    _ret_abs = np.abs(_log_return_2d(close, lag=scale_window(1)))
    _atr_rel = atr / np.maximum(close, 1e-12)
    _climax = (_vol_z > 3.0) & (_ret_abs < 0.5 * _atr_rel)
    _ma20_close = _rolling_mean_2d(close, window=scale_window(20))
    _climax_side = np.zeros_like(close, dtype=np.int8)
    _climax_side[_climax & (close > _ma20_close)] = -1
    _climax_side[_climax & (close < _ma20_close)] = 1
    _climax_score = _climax_side.astype(np.float64)
    panels.append(
        CandidateSignalPanel(
            family="volume_climax_1h",
            variant="vc_1h",
            params=apply_per_family_params(cfg, "volume_climax_1h", "vc_1h", {
                "vol_window": 20,
                "vol_z_threshold": 3.0,
            }),
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=_climax_score,
            side_hint_2d=_climax_side,
            expected_holding_bars=scale_window(6),
            min_holding_bars=scale_window(2),
            stop_atr_mult=1.5,
            take_profit_atr_mult=2.0,
            turnover_proxy_2d=_climax.astype(np.float64),
            valid_mask_2d=valid_mask,
            metadata={
                "archetype": "mean_rev",
                "regime": "exhaustion_climax",
                "edge_hypothesis": "extreme volume with stalled price signals exhaustion (Wyckoff distribution)",
            },
        )
    )

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
    panels.append(
        CandidateSignalPanel(
            family="macd_4h",
            variant="macd_12_26_9",
            params=apply_per_family_params(cfg, "macd_4h", "macd_12_26_9", {
                "fast": 12, "slow": 26, "signal": 9,
            }),
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

    # E. supertrend
    _st_period = scale_window(10)
    _st_trend = _supertrend_2d(high, low, close, period=_st_period, multiplier=2.5)
    _st_score = _st_trend.astype(np.float64)
    panels.append(
        CandidateSignalPanel(
            family="supertrend",
            variant=f"st_{_st_period}",
            params=apply_per_family_params(cfg, "supertrend", f"st_{_st_period}", {
                "period": _st_period, "multiplier": 2.5,
            }),
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=_st_score,
            side_hint_2d=_st_trend,
            expected_holding_bars=scale_window(18),
            min_holding_bars=scale_window(6),
            stop_atr_mult=2.5,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.abs(np.diff(_st_score, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
            metadata={
                "archetype": "trend",
                "regime": "supertrend",
                "edge_hypothesis": "SuperTrend ATR-based trailing stop provides clean trend signals on higher TFs",
            },
        )
    )

    # F. ichimoku_trend
    _tenkan = (_rolling_max_2d(high, scale_window(9)) + _rolling_min_2d(low, scale_window(9))) / 2
    _kijun = (_rolling_max_2d(high, scale_window(26)) + _rolling_min_2d(low, scale_window(26))) / 2
    _cloud_top = (_tenkan + _kijun) / 2
    _ichi_bull = (_tenkan > _kijun) & (close > _cloud_top)
    _ichi_bear = (_tenkan < _kijun) & (close < _cloud_top)
    _ichi_side = np.zeros_like(close, dtype=np.int8)
    _ichi_side[_ichi_bull] = 1
    _ichi_side[_ichi_bear] = -1
    _ichi_score = (_tenkan - _kijun) / np.maximum(atr, 1e-12)
    panels.append(
        CandidateSignalPanel(
            family="ichimoku_trend",
            variant="ichi_9_26",
            params=apply_per_family_params(cfg, "ichimoku_trend", "ichi_9_26", {
                "tenkan": 9, "kijun": 26,
            }),
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
                "edge_hypothesis": "Ichimoku 3-confirmation (TK cross + cloud) filters false breakouts on 12h TF",
            },
        )
    )

    # ── family_filter: keep only matching families ──
    if family_filter is not None:
        panels = [p for p in panels if p.family in family_filter]

    # ── per_family_params: apply param overrides ──
    if cfg.per_family_params is not None:
        from dataclasses import replace
        panels = [
            replace(p, params=apply_per_family_params(cfg, p.family, p.variant, p.params))
            for p in panels
        ]

    panels_with_context = _attach_signal_context(tuple(panels), cfg=cfg, regime_ctx=regime_ctx)
    return filter_rule_signal_panels(panels_with_context, cfg=cfg)


def candidate_panels_to_events(
    panels: tuple[CandidateSignalPanel, ...],
    *,
    min_abs_score: float,
    side_flip_variants: tuple[str, ...] = (),
    cost_floor_bps: float = 24.0,
    execution_cost_bps_2d: NDArray[np.float64] | None = None,
) -> pd.DataFrame:
    """Convert dense [T,N] panels into sparse candidate event rows."""
    all_events: list[pd.DataFrame] = []
    side_flip_allowlist = _candidate_variant_set(side_flip_variants)
    for panel in panels:
        scores = panel.signed_score_2d
        sides = panel.side_hint_2d
        mask = panel.valid_mask_2d & (np.abs(scores) >= min_abs_score) & (sides != 0)
        if not np.any(mask):
            continue

        t_idx, s_idx = np.where(mask)
        if t_idx.size == 0:
            continue

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
                    [
                        f"{panel.family}:{panel.variant}:{policy.policy_id}:{entry_regime}"
                        for entry_regime in entry_regimes[regime_mask]
                    ],
                    dtype=object,
                )
                if side_flipped:
                    signal_cells = np.asarray([f"{cell}:flip" for cell in signal_cells], dtype=object)
                df = pd.DataFrame({
                    "datetime": event_datetimes[regime_mask],
                    "symbol": event_symbols[regime_mask],
                    "family": panel.family,
                    "variant": panel.variant,
                    "side": event_sides[regime_mask],
                    "raw_score": raw_scores[regime_mask],
                    "score_z": score_z[regime_mask],
                    "expected_holding_bars": int(policy.expected_holding_bars),
                    "min_holding_bars": int(policy.min_holding_bars),
                    "stop_atr_mult": float(policy.stop_atr_mult),
                    "take_profit_atr_mult": float(policy.take_profit_atr_mult),
                    "turnover_proxy": turnover_proxy[regime_mask],
                    "cost_floor_bps": event_cost[regime_mask],
                    "entry_idx": t_idx[regime_mask] + 1,
                    "side_flipped": side_flipped,
                    "exit_policy_id": policy.policy_id,
                    "signal_cell": signal_cells,
                    "archetype": panel.archetype,
                    "entry_regime": entry_regimes[regime_mask],
                    "entry_regime_code": entry_regime_codes[regime_mask],
                })
                all_events.append(df)

    if not all_events:
        return pd.DataFrame(columns=[
            "datetime", "symbol", "family", "variant", "side",
            "raw_score", "score_z", "expected_holding_bars", "min_holding_bars",
            "stop_atr_mult", "take_profit_atr_mult", "turnover_proxy", "cost_floor_bps", "entry_idx",
            "side_flipped", "exit_policy_id", "signal_cell", "archetype", "entry_regime", "entry_regime_code",
        ])

    return pd.concat(all_events, axis=0, ignore_index=True).sort_values("datetime").reset_index(drop=True)
