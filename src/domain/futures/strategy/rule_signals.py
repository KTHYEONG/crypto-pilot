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
    valid_mask = aligned.active_mask & np.isfinite(close) & np.isfinite(high) & np.isfinite(low)

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

    return filter_rule_signal_panels(tuple(panels), cfg=cfg)


def candidate_panels_to_events(
    panels: tuple[CandidateSignalPanel, ...],
    *,
    min_abs_score: float,
    side_flip_variants: tuple[str, ...] = (),
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
            "cost_floor_bps": 24.0,  # default floor
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
