from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.strategy.candidate_contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig

_logger = logging.getLogger(__name__)


def candidate_variant_key(family: str, variant: str) -> str:
    """Return a stable candidate variant key."""
    return f"{family}:{variant}"


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

def _ema_2d(arr: NDArray[np.float64], span: int) -> NDArray[np.float64]:
    if span <= 1:
        return arr.copy()
    alpha = 2.0 / (float(span) + 1.0)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    last = np.zeros(arr.shape[1], dtype=np.float64)
    initialized = np.zeros(arr.shape[1], dtype=bool)
    for t in range(arr.shape[0]):
        row = arr[t]
        finite = np.isfinite(row)
        upd = finite & initialized
        init = finite & ~initialized
        last[upd] = (1.0 - alpha) * last[upd] + alpha * row[upd]
        last[init] = row[init]
        initialized[init] = True
        out[t, initialized] = last[initialized]
    return out


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


# --- 8 Vectorized Rule Families ---

def build_rule_signal_panels(
    *,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
) -> tuple[CandidateSignalPanel, ...]:
    """Build trailing-only rule candidates for all symbols."""
    close = aligned.close_2d
    high = aligned.high_2d
    low = aligned.low_2d
    vol = aligned.volume_2d
    funding = aligned.funding_2d
    oi = aligned.oi_2d if aligned.oi_2d is not None else np.zeros_like(close)
    entry_warm_mask = (
        aligned.inference_entry_warm_mask
        if aligned.inference_entry_warm_mask is not None
        else aligned.warm_mask
    )
    valid_mask = (
        aligned.active_mask
        & entry_warm_mask
        & ~aligned.entry_block_mask
        & ~aligned.kill_mask
        & np.isfinite(close)
        & np.isfinite(high)
        & np.isfinite(low)
    )

    atr = _atr_2d(high, low, close, period=14)
    atr = np.maximum(atr, 1e-12)

    panels: list[CandidateSignalPanel] = []

    # 1. Trend MA Cross
    ema_fast = _ema_2d(close, span=12)
    ema_slow = _ema_2d(close, span=72)
    ma_diff = (ema_fast - ema_slow) / atr
    signed_score_ma = np.tanh(ma_diff)
    side_hint_ma = np.zeros_like(signed_score_ma, dtype=np.int8)
    side_hint_ma[ma_diff > 0.5] = 1
    side_hint_ma[ma_diff < -0.5] = -1
    panels.append(
        CandidateSignalPanel(
            family="trend_ma",
            variant="ema_12_72",
            params={"ema_fast": 12, "ema_slow": 72, "atr_period": 14},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=signed_score_ma,
            side_hint_2d=side_hint_ma,
            expected_holding_bars=18,
            min_holding_bars=6,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.abs(np.diff(signed_score_ma, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
        )
    )

    # 1b. Trend MA Cross — ema_6_36
    ema_fast_6 = _ema_2d(close, span=6)
    ema_slow_36 = _ema_2d(close, span=36)
    ma_diff_6_36 = (ema_fast_6 - ema_slow_36) / atr
    signed_score_ma_6_36 = np.tanh(ma_diff_6_36)
    side_hint_ma_6_36 = np.zeros_like(signed_score_ma_6_36, dtype=np.int8)
    side_hint_ma_6_36[ma_diff_6_36 > 0.5] = 1
    side_hint_ma_6_36[ma_diff_6_36 < -0.5] = -1
    panels.append(
        CandidateSignalPanel(
            family="trend_ma",
            variant="ema_6_36",
            params={"ema_fast": 6, "ema_slow": 36, "atr_period": 14},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=signed_score_ma_6_36,
            side_hint_2d=side_hint_ma_6_36,
            expected_holding_bars=12,
            min_holding_bars=4,
            stop_atr_mult=1.5,
            take_profit_atr_mult=3.0,
            turnover_proxy_2d=np.abs(np.diff(signed_score_ma_6_36, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
        )
    )

    # 1c. Trend MA Cross — ema_18_108
    ema_fast_18 = _ema_2d(close, span=18)
    ema_slow_108 = _ema_2d(close, span=108)
    ma_diff_18_108 = (ema_fast_18 - ema_slow_108) / atr
    signed_score_ma_18_108 = np.tanh(ma_diff_18_108)
    side_hint_ma_18_108 = np.zeros_like(signed_score_ma_18_108, dtype=np.int8)
    side_hint_ma_18_108[ma_diff_18_108 > 0.5] = 1
    side_hint_ma_18_108[ma_diff_18_108 < -0.5] = -1
    panels.append(
        CandidateSignalPanel(
            family="trend_ma",
            variant="ema_18_108",
            params={"ema_fast": 18, "ema_slow": 108, "atr_period": 14},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=signed_score_ma_18_108,
            side_hint_2d=side_hint_ma_18_108,
            expected_holding_bars=24,
            min_holding_bars=8,
            stop_atr_mult=2.5,
            take_profit_atr_mult=5.0,
            turnover_proxy_2d=np.abs(np.diff(signed_score_ma_18_108, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
        )
    )

    # 2. Trend Donchian
    donchian_high = _rolling_max_2d(high, window=36)
    donchian_low = _rolling_min_2d(low, window=36)
    donchian_score = np.zeros_like(close)
    donchian_side = np.zeros_like(close, dtype=np.int8)
    donchian_side[close > donchian_high] = 1
    donchian_side[close < donchian_low] = -1
    above = close > donchian_high
    below = close < donchian_low
    donchian_score[above] = (close[above] - donchian_high[above]) / atr[above]
    donchian_score[below] = (close[below] - donchian_low[below]) / atr[below]
    panels.append(
        CandidateSignalPanel(
            family="trend_donchian",
            variant="donchian_36",
            params={"lookback": 36},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(donchian_score, -1.0, 1.0),
            side_hint_2d=donchian_side,
            expected_holding_bars=24,
            min_holding_bars=8,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.abs(np.diff(donchian_score, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
        )
    )

    # 2b. Trend Donchian — donchian_18
    d18_high = _rolling_max_2d(high, window=18)
    d18_low = _rolling_min_2d(low, window=18)
    d18_side = np.zeros_like(close, dtype=np.int8)
    d18_side[close > d18_high] = 1
    d18_side[close < d18_low] = -1
    d18_score = np.zeros_like(close)
    above_18 = close > d18_high
    below_18 = close < d18_low
    d18_score[above_18] = (close[above_18] - d18_high[above_18]) / atr[above_18]
    d18_score[below_18] = (close[below_18] - d18_low[below_18]) / atr[below_18]
    panels.append(
        CandidateSignalPanel(
            family="trend_donchian",
            variant="donchian_18",
            params={"lookback": 18},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(d18_score, -1.0, 1.0),
            side_hint_2d=d18_side,
            expected_holding_bars=12,
            min_holding_bars=4,
            stop_atr_mult=1.5,
            take_profit_atr_mult=3.0,
            turnover_proxy_2d=np.abs(np.diff(d18_score, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
        )
    )

    # 2c. Trend Donchian — donchian_72
    d72_high = _rolling_max_2d(high, window=72)
    d72_low = _rolling_min_2d(low, window=72)
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
            params={"lookback": 72},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(d72_score, -1.0, 1.0),
            side_hint_2d=d72_side,
            expected_holding_bars=36,
            min_holding_bars=12,
            stop_atr_mult=2.5,
            take_profit_atr_mult=5.0,
            turnover_proxy_2d=np.abs(np.diff(d72_score, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
        )
    )

    # 3. Vol Breakout
    bb_mean = _rolling_mean_2d(close, window=20)
    bb_std = _rolling_std_2d(close, window=20)
    bandwidth = (bb_std * 4.0) / np.maximum(bb_mean, 1e-12)
    bw_mean_120 = _rolling_mean_2d(bandwidth, window=120)
    bw_std_120 = _rolling_std_2d(bandwidth, window=120)
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
            params={"bb_window": 20, "compression_window": 120},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(vol_score, -1.0, 1.0),
            side_hint_2d=vol_side,
            expected_holding_bars=18,
            min_holding_bars=6,
            stop_atr_mult=1.5,
            take_profit_atr_mult=3.0,
            turnover_proxy_2d=np.abs(np.diff(vol_score, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
        )
    )

    # 4. Bollinger Reversion
    bb_mean_rev = _rolling_mean_2d(close, window=20)
    bb_std_rev = _rolling_std_2d(close, window=20)
    bb_z_rev = (close - bb_mean_rev) / np.maximum(bb_std_rev, 1e-12)
    rev_side = np.zeros_like(close, dtype=np.int8)
    rev_side[bb_z_rev < -2.0] = 1
    rev_side[bb_z_rev > 2.0] = -1
    rev_score = -bb_z_rev / 3.0
    panels.append(
        CandidateSignalPanel(
            family="bollinger_reversion",
            variant="bollinger_20",
            params={"window": 20, "entry_z": 2.0},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(rev_score, -1.0, 1.0),
            side_hint_2d=rev_side,
            expected_holding_bars=12,
            min_holding_bars=4,
            stop_atr_mult=2.5,
            take_profit_atr_mult=3.0,
            turnover_proxy_2d=np.abs(np.diff(rev_score, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
        )
    )

    # 5. RSI Reversion
    rsi = _rsi_2d(close, period=14)
    rsi_prev = np.vstack([rsi[:1], rsi[:-1]])
    rsi_side = np.zeros_like(close, dtype=np.int8)
    rsi_side[(rsi_prev < 30) & (rsi > rsi_prev)] = 1
    rsi_side[(rsi_prev > 70) & (rsi < rsi_prev)] = -1
    rsi_score = (50.0 - rsi) / 20.0
    panels.append(
        CandidateSignalPanel(
            family="rsi_reversion",
            variant="rsi_14",
            params={"rsi_period": 14, "oversold": 30.0, "overbought": 70.0},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(rsi_score, -1.0, 1.0),
            side_hint_2d=rsi_side,
            expected_holding_bars=12,
            min_holding_bars=4,
            stop_atr_mult=2.0,
            take_profit_atr_mult=3.0,
            turnover_proxy_2d=np.abs(np.diff(rsi_score, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
        )
    )

    # 5b. RSI Reversion — rsi_6
    rsi6 = _rsi_2d(close, period=6)
    rsi6_prev = np.vstack([rsi6[:1], rsi6[:-1]])
    rsi6_side = np.zeros_like(close, dtype=np.int8)
    rsi6_side[(rsi6_prev < 20) & (rsi6 > rsi6_prev)] = 1
    rsi6_side[(rsi6_prev > 80) & (rsi6 < rsi6_prev)] = -1
    rsi6_score = (50.0 - rsi6) / 20.0
    panels.append(
        CandidateSignalPanel(
            family="rsi_reversion",
            variant="rsi_6",
            params={"rsi_period": 6, "oversold": 20.0, "overbought": 80.0},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(rsi6_score, -1.0, 1.0),
            side_hint_2d=rsi6_side,
            expected_holding_bars=8,
            min_holding_bars=2,
            stop_atr_mult=1.5,
            take_profit_atr_mult=2.5,
            turnover_proxy_2d=np.abs(np.diff(rsi6_score, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
        )
    )

    # 6. Funding Carry
    funding_mean = _rolling_mean_2d(funding, window=24)
    funding_std = _rolling_std_2d(funding, window=24)
    funding_z = (funding - funding_mean) / np.maximum(funding_std, 1e-6)
    carry_side = np.zeros_like(close, dtype=np.int8)
    carry_side[funding_z < -1.5] = 1
    carry_side[funding_z > 1.5] = -1
    carry_score = -funding_z / 2.0
    panels.append(
        CandidateSignalPanel(
            family="funding_carry",
            variant="funding_24",
            params={"window": 24, "entry_z": 1.5},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(carry_score, -1.0, 1.0),
            side_hint_2d=carry_side,
            expected_holding_bars=24,
            min_holding_bars=8,
            stop_atr_mult=2.0,
            take_profit_atr_mult=3.0,
            turnover_proxy_2d=np.abs(np.diff(carry_score, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
        )
    )

    # 7. OI Volume Impulse
    vol_mean = _rolling_mean_2d(vol, window=20)
    vol_std = _rolling_std_2d(vol, window=20)
    vol_z = (vol - vol_mean) / np.maximum(vol_std, 1e-12)
    price_ret = np.diff(close, axis=0, prepend=close[:1]) / np.maximum(close, 1e-12)
    oi_ret = np.diff(oi, axis=0, prepend=oi[:1]) / np.maximum(oi, 1e-12)
    impulse_side = np.zeros_like(close, dtype=np.int8)
    impulse_side[(vol_z > 1.5) & (price_ret > 0.0) & (oi_ret > 0.0)] = 1
    impulse_side[(vol_z > 1.5) & (price_ret < 0.0) & (oi_ret > 0.0)] = -1
    impulse_score = vol_z / 3.0 * np.sign(price_ret)
    panels.append(
        CandidateSignalPanel(
            family="oi_volume_impulse",
            variant="oi_impulse_20",
            params={"window": 20, "volume_z_entry": 1.5},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(impulse_score, -1.0, 1.0),
            side_hint_2d=impulse_side,
            expected_holding_bars=18,
            min_holding_bars=6,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.abs(np.diff(impulse_score, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
        )
    )

    # 8. BTC Regime Pullback
    btc_idx = 0
    for idx, sym in enumerate(aligned.symbols):
        if "BTC" in sym:
            btc_idx = idx
            break
    btc_close = close[:, btc_idx : btc_idx + 1]
    btc_ema_fast = _ema_2d(btc_close, span=20)
    btc_ema_slow = _ema_2d(btc_close, span=100)
    btc_trend_up = btc_ema_fast > btc_ema_slow

    alt_mean = _rolling_mean_2d(close, window=50)
    alt_std = _rolling_std_2d(close, window=50)
    alt_pullback_z = (close - alt_mean) / np.maximum(alt_std, 1e-12)

    btc_side = np.zeros_like(close, dtype=np.int8)
    btc_side[btc_trend_up & (alt_pullback_z < -1.5)] = 1
    btc_side[~btc_trend_up & (alt_pullback_z > 1.5)] = -1
    btc_score = -alt_pullback_z / 2.0
    panels.append(
        CandidateSignalPanel(
            family="btc_regime_pullback",
            variant="btc_pullback_50",
            params={"window": 50, "btc_fast": 20, "btc_slow": 100},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.clip(btc_score, -1.0, 1.0),
            side_hint_2d=btc_side,
            expected_holding_bars=18,
            min_holding_bars=6,
            stop_atr_mult=2.0,
            take_profit_atr_mult=3.0,
            turnover_proxy_2d=np.abs(np.diff(btc_score, axis=0, prepend=0.0)),
            valid_mask_2d=valid_mask,
        )
    )

    # 9. Cross-Sectional Momentum (F1)
    # Look-ahead guard: close_shifted[t] = close[t-lb] for t>=lb, else close[0]
    for _lb, _q in [(5, 0.2), (10, 0.2), (20, 0.3)]:
        _close_shifted = np.vstack([
            np.tile(close[0:1], (_lb, 1)),
            close[:-_lb],
        ])
        _ret_lb = close / np.maximum(_close_shifted, 1e-12) - 1.0
        _rank_pct: NDArray[np.float64] = (
            pd.DataFrame(_ret_lb).rank(axis=1, pct=True).to_numpy(dtype=np.float64)
        )
        _cs_score = np.tanh((_rank_pct - 0.5) * 4.0)
        _cs_side = np.zeros_like(_cs_score, dtype=np.int8)
        _cs_side[_rank_pct >= 1.0 - _q] = 1
        _cs_side[_rank_pct <= _q] = -1
        # Zero out warmup bars (no valid lookback)
        _cs_score[:_lb] = 0.0
        _cs_side[:_lb] = 0
        panels.append(
            CandidateSignalPanel(
                family="cross_sectional_momentum",
                variant=f"cs_mom_{_lb}",
                params={"lookback": _lb, "quantile": _q},
                datetimes=aligned.datetimes,
                symbols=aligned.symbols,
                signed_score_2d=np.clip(_cs_score, -1.0, 1.0),
                side_hint_2d=_cs_side,
                expected_holding_bars=_lb * 2,
                min_holding_bars=max(2, _lb // 2),
                stop_atr_mult=2.0,
                take_profit_atr_mult=3.0,
                turnover_proxy_2d=np.abs(np.diff(np.clip(_cs_score, -1.0, 1.0), axis=0, prepend=0.0)),
                valid_mask_2d=valid_mask,
            )
        )

    # 10. Funding Z-Score Carry (F2)
    for _fz_win, _fz_thr in [(48, 2.0), (96, 2.0), (168, 1.5)]:
        _f_mean = _rolling_mean_2d(funding, window=_fz_win)
        _f_std = _rolling_std_2d(funding, window=_fz_win)
        _f_z = (funding - _f_mean) / np.maximum(_f_std, 1e-6)
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
                expected_holding_bars=_fz_win // 2,
                min_holding_bars=8,
                stop_atr_mult=2.0,
                take_profit_atr_mult=3.0,
                turnover_proxy_2d=np.abs(np.diff(_fz_score, axis=0, prepend=0.0)),
                valid_mask_2d=valid_mask,
            )
        )

    # 11. Vol Regime Reversion (F3)
    _price_dir = np.sign(np.diff(close, axis=0, prepend=close[:1]))
    for _vr_win, _vr_thr in [(20, 2.0), (40, 1.5)]:
        _atr_mean = _rolling_mean_2d(atr, window=_vr_win)
        _atr_std = _rolling_std_2d(atr, window=_vr_win)
        _vol_z = (atr - _atr_mean) / np.maximum(_atr_std, 1e-12)
        _high_vol = _vol_z >= _vr_thr
        _vr_side = np.zeros_like(close, dtype=np.int8)
        _vr_side[_high_vol & (_price_dir > 0)] = -1  # high vol + up move → fade (short)
        _vr_side[_high_vol & (_price_dir < 0)] = 1   # high vol + down move → fade (long)
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
                min_holding_bars=4,
                stop_atr_mult=2.0,
                take_profit_atr_mult=3.0,
                turnover_proxy_2d=np.abs(np.diff(_vr_score, axis=0, prepend=0.0)),
                valid_mask_2d=valid_mask,
            )
        )

    # 12. BTC Correlation Regime (F4)
    _close_ret = np.diff(np.log(np.maximum(close, 1e-12)), axis=0, prepend=0.0)
    for _cr_win, _cr_thr in [(24, 0.6), (48, 0.5), (96, 0.5)]:
        _corr_2d = _rolling_corr_with_col(_close_ret, btc_idx, _cr_win)
        _btc_dir = np.sign(_close_ret[:, btc_idx : btc_idx + 1])  # BTC return direction [T,1]
        # High-corr regime: follow BTC direction; score = corr * btc_direction
        _cr_score = np.where(
            np.isfinite(_corr_2d) & (_corr_2d >= _cr_thr),
            _corr_2d * _btc_dir,
            0.0,
        )
        _cr_side = np.zeros_like(close, dtype=np.int8)
        _cr_valid = np.isfinite(_corr_2d) & (_corr_2d >= _cr_thr)
        _cr_side[_cr_valid & (_btc_dir[:, 0:1].repeat(close.shape[1], axis=1) > 0)] = 1
        _cr_side[_cr_valid & (_btc_dir[:, 0:1].repeat(close.shape[1], axis=1) < 0)] = -1
        panels.append(
            CandidateSignalPanel(
                family="btc_corr_regime",
                variant=f"bcr_{_cr_win}",
                params={"corr_window": _cr_win, "corr_threshold": _cr_thr},
                datetimes=aligned.datetimes,
                symbols=aligned.symbols,
                signed_score_2d=np.clip(_cr_score, -1.0, 1.0),
                side_hint_2d=_cr_side,
                expected_holding_bars=_cr_win // 4,
                min_holding_bars=2,
                stop_atr_mult=1.5,
                take_profit_atr_mult=2.5,
                turnover_proxy_2d=np.abs(np.diff(np.clip(_cr_score, -1.0, 1.0), axis=0, prepend=0.0)),
                valid_mask_2d=valid_mask,
            )
        )

    # 13. Funding Acceleration Carry (F5)
    # Signal: funding z-score slope (acceleration) * persistence proxy.
    # Long when funding z-score is falling from high (carry unwind expected);
    # Short when funding z-score is rising steeply (negative funding carry building).
    # Look-ahead guard: all rolling ops are trailing-only.
    _funding_finite = np.where(np.isfinite(funding), funding, 0.0)
    for _fa_win, _fa_slope_win in [(48, 6), (168, 12)]:
        _f_mean = _rolling_mean_2d(_funding_finite, window=_fa_win)
        _f_std = _rolling_std_2d(_funding_finite, window=_fa_win)
        _f_z = (_funding_finite - _f_mean) / np.maximum(_f_std, 1e-9)
        # Slope of z-score over short window (acceleration)
        _f_z_lag = np.vstack([_f_z[:_fa_slope_win], _f_z[:-_fa_slope_win]])
        _f_slope = _f_z - _f_z_lag
        # Persistence: exponential decay weight of recent z-scores
        _f_persist = _rolling_mean_2d(np.abs(_f_z), window=_fa_slope_win)
        _f_signal = -np.tanh(_f_slope * _f_persist)  # fade the slope direction
        _fa_side = np.zeros_like(close, dtype=np.int8)
        _fa_side[_f_signal > 0.3] = 1
        _fa_side[_f_signal < -0.3] = -1
        panels.append(
            CandidateSignalPanel(
                family="funding_acceleration_carry",
                variant=f"fac_{_fa_win}",
                params={"funding_window": _fa_win, "slope_window": _fa_slope_win},
                datetimes=aligned.datetimes,
                symbols=aligned.symbols,
                signed_score_2d=np.clip(_f_signal, -1.0, 1.0),
                side_hint_2d=_fa_side,
                expected_holding_bars=_fa_win // 6,
                min_holding_bars=4,
                stop_atr_mult=2.0,
                take_profit_atr_mult=3.0,
                turnover_proxy_2d=np.abs(np.diff(_f_signal, axis=0, prepend=0.0)),
                valid_mask_2d=valid_mask,
            )
        )

    # 14. BTC Residual Momentum (F6)
    # Signal: alt return residual after BTC beta adjustment.
    # Measures alt-specific alpha: excess return vs expected BTC-driven move.
    # Look-ahead guard: beta estimated from trailing window only.
    _log_ret = np.diff(np.log(np.maximum(close, 1e-12)), axis=0, prepend=0.0)
    _btc_ret = _log_ret[:, btc_idx : btc_idx + 1]
    for _br_win in [24, 48]:
        _btc_var = _rolling_mean_2d(_btc_ret ** 2, window=_br_win)
        _cov_alt_btc = _rolling_mean_2d(_log_ret * _btc_ret, window=_br_win)
        _beta_hat = _cov_alt_btc / np.maximum(_btc_var, 1e-12)
        _resid_ret = _log_ret - _beta_hat * _btc_ret
        _resid_mean = _rolling_mean_2d(_resid_ret, window=_br_win)
        _resid_std = _rolling_std_2d(_resid_ret, window=_br_win)
        _resid_z = _resid_mean / np.maximum(_resid_std / np.sqrt(float(_br_win)), 1e-12)
        _resid_z = np.clip(_resid_z, -3.0, 3.0)
        _br_side = np.zeros_like(close, dtype=np.int8)
        _br_side[_resid_z > 1.5] = 1
        _br_side[_resid_z < -1.5] = -1
        panels.append(
            CandidateSignalPanel(
                family="btc_residual_momentum",
                variant=f"brm_{_br_win}",
                params={"window": _br_win},
                datetimes=aligned.datetimes,
                symbols=aligned.symbols,
                signed_score_2d=np.tanh(_resid_z / 2.0),
                side_hint_2d=_br_side,
                expected_holding_bars=_br_win // 4,
                min_holding_bars=3,
                stop_atr_mult=1.5,
                take_profit_atr_mult=3.0,
                turnover_proxy_2d=np.abs(np.diff(_resid_z, axis=0, prepend=0.0)),
                valid_mask_2d=valid_mask,
            )
        )

    # 15. OI-Volume Confirmed Breakout (F7)
    # Signal: price range breakout confirmed by concurrent OI impulse + volume z-score.
    # All three conditions must align: prevents noise-driven breakouts.
    # Look-ahead guard: donchian channels and volume/OI z-scores are trailing-only.
    _vol_finite = np.where(np.isfinite(vol), vol, 0.0)
    _oi_finite = np.where(np.isfinite(oi), oi, 0.0)
    for _ob_win in [20, 40]:
        _don_high = _rolling_max_2d(high, window=_ob_win)
        _don_low = _rolling_min_2d(low, window=_ob_win)
        _don_mid = (_don_high + _don_low) / 2.0
        _breakout_up = close >= _don_high * 0.998
        _breakout_dn = close <= _don_low * 1.002
        _vol_mean = _rolling_mean_2d(_vol_finite, window=_ob_win)
        _vol_std = _rolling_std_2d(_vol_finite, window=_ob_win)
        _vol_z_ob = (_vol_finite - _vol_mean) / np.maximum(_vol_std, 1e-12)
        _oi_mean = _rolling_mean_2d(_oi_finite, window=_ob_win)
        _oi_std = _rolling_std_2d(_oi_finite, window=_ob_win)
        _oi_z_ob = (_oi_finite - _oi_mean) / np.maximum(_oi_std, 1e-12)
        _confirmed = (_vol_z_ob >= 1.0) & (_oi_z_ob >= 0.5)
        _ob_score = np.where(
            _breakout_up & _confirmed, np.clip(_vol_z_ob / 3.0, 0.0, 1.0),
            np.where(_breakout_dn & _confirmed, -np.clip(_vol_z_ob / 3.0, 0.0, 1.0), 0.0),
        )
        _ob_side = np.zeros_like(close, dtype=np.int8)
        _ob_side[_breakout_up & _confirmed] = 1
        _ob_side[_breakout_dn & _confirmed] = -1
        panels.append(
            CandidateSignalPanel(
                family="oi_volume_confirmed_breakout",
                variant=f"oib_{_ob_win}",
                params={"window": _ob_win},
                datetimes=aligned.datetimes,
                symbols=aligned.symbols,
                signed_score_2d=_ob_score.astype(np.float64),
                side_hint_2d=_ob_side,
                expected_holding_bars=_ob_win // 4,
                min_holding_bars=3,
                stop_atr_mult=1.5,
                take_profit_atr_mult=3.5,
                turnover_proxy_2d=np.abs(np.diff(_ob_score.astype(np.float64), axis=0, prepend=0.0)),
                valid_mask_2d=valid_mask,
            )
        )

    return filter_rule_signal_panels(tuple(panels), cfg=cfg)


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
        score_z = raw_scores.copy()
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

        df = pd.DataFrame({
            "datetime": event_datetimes,
            "symbol": event_symbols,
            "family": panel.family,
            "variant": panel.variant,
            "side": event_sides,
            "raw_score": raw_scores,
            "score_z": score_z,  # proxy
            "expected_holding_bars": panel.expected_holding_bars,
            "min_holding_bars": panel.min_holding_bars,
            "stop_atr_mult": panel.stop_atr_mult,
            "take_profit_atr_mult": panel.take_profit_atr_mult,
            "turnover_proxy": panel.turnover_proxy_2d[t_idx, s_idx],
            "cost_floor_bps": event_cost,
            "entry_idx": t_idx + 1,
            "side_flipped": side_flipped,
        })
        all_events.append(df)

    if not all_events:
        return pd.DataFrame(columns=[
            "datetime", "symbol", "family", "variant", "side",
            "raw_score", "score_z", "expected_holding_bars", "min_holding_bars",
            "stop_atr_mult", "take_profit_atr_mult", "turnover_proxy", "cost_floor_bps", "entry_idx",
            "side_flipped",
        ])

    return pd.concat(all_events, axis=0, ignore_index=True).sort_values("datetime").reset_index(drop=True)
