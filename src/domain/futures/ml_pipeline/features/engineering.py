"""GP / HMM input feature builders (1h OHLCV + funding)."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from numba import njit

from src.domain.futures.ml_pipeline.regime.regime_contracts import REGIME_PROB_COLUMNS

# Columns produced by build_gp_input_features (for CS-rank / imputation in pipeline).
# Systemic HMM: TVTP uses macro_trend_168h (0) + microstructure (4–8). Return-path emissions (9): level.
# Reduced to 11 features for Tiered Universe Architecture.
SYSTEMIC_HMM_FEATURE_COLUMNS: tuple[str, ...] = (
    "macro_trend_168h",
    "macro_trend_24h",
    "macro_vol_24h",
    "macro_downside_vol_24h",
    "macro_cs_dispersion_24h",
    "macro_oi_delta_24h",
    "macro_funding_mom_24h",
    "macro_liq_proxy_24h",
    "macro_lsr_delta_24h",
    "macro_ret_1h",
    "macro_breadth_168h",
    # Tier 3: Universe Breadth (Phase C)
    "macro_breadth_ma20",
    "macro_alt_btc_rs_24h",
)

# Posterior columns aligned to stable semantic labels (order for MetaLabeler).
# Keep a local alias for backward compatibility with existing imports.
HMM_SEMANTIC_PROB_COLUMNS: tuple[str, ...] = REGIME_PROB_COLUMNS

# Bump when ALPHA feature semantics change without renaming columns so raw GP
# caches are invalidated and Tier 2 retraining actually happens.
GP_FEATURE_SCHEMA_VERSION: str = "v22"

def _tf_to_hours(tf: str) -> float:
    """Convert timeframe string to decimal hours."""
    if tf.endswith("h"):
        return float(tf[:-1])
    if tf.endswith("m"):
        return float(tf[:-1]) / 60.0
    if tf.endswith("d"):
        return float(tf[:-1]) * 24.0
    return 1.0


def _get_window(hours: int, tf: str) -> int:
    """Calculate bar count for a given hour-horizon in a specific timeframe."""
    tf_h = _tf_to_hours(tf)
    res = round(hours / tf_h)
    return max(1, res)


def _macro_risk_adj_ret_1h(close: pd.Series, w24: int) -> pd.Series:
    """Risk-adjusted 1-bar return: pct_change(1) / rolling stdev of pct returns (Sharpe-like, causal)."""
    r = close.astype(np.float64).pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    min_p = max(2, w24 // 4)
    vol_24h = r.rolling(w24, min_periods=min(w24, min_p)).std().fillna(0.0)
    return cast(pd.Series, r / (vol_24h + 1e-8))


ALPHA_ENGINEERED_FEATURE_NAMES: tuple[str, ...] = (
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "ret_24",
    "ma_dist_24",
    "ma_dist_168",
    "vol_ratio_24",
    "vol_ratio_168",
    "hl_spread",
    "funding_rate",
    "buy_sell_ratio",
    "funding_z_72",
    "funding_chg_8",
    "taker_imbalance_z_24",
    "realized_vol_yz_24",
    "ret_vol_adj_6",
    "ret_vol_adj_24",
    "liq_proxy_6",
    "vol_skew_24",
    "mom_proxy_12",
    "vwap_dist_24",
    "vol_surface_24_168",
    "funding_mom_24",
    "acceleration_24",
    "tail_risk_24",
    "amihud_illiq_24",
    "range_pos_24",
    "frac_diff_04",
    "hurst_24",
    "vol_of_vol_24",
    "tail_rejection_24",
    "dist_from_high_24",
    "corr_btc_24",
    "btc_beta",
    "vpin_proxy_12",
    "funding_trap_24",
    "downside_jump_24",
    "oi_momentum_4h",
    "oi_momentum_24h",
    "oi_price_divergence_24h",
    "oi_funding_trap_24h",
    "top_trader_lsr_z_24h",
    "global_lsr_z_24h",
    "lsr_spread_12h",
    "taker_buy_sell_ratio_12h",
    "cvd_divergence_24h",
    "taker_acceleration_24h",
    "funding_intensity_24h",
    "absorption_ratio_12h",
    "motif_crowded_long_unwind",
    "motif_funding_short_squeeze",
    "motif_taker_absorption",
    "motif_oi_price_dislocation",
    "motif_liq_pressure",
    # F-A: OI Delta + Acceleration
    "oi_delta_4h",
    "oi_accel_24h",
    # F-B: Funding x Realized-Vol Interaction
    "funding_vol_interaction",
    # F-C: Perp-Spot Basis Proxy (rolling mean deviation)
    "perp_basis_proxy",
    # F-D: High-Order Interactions (Quant Institutional Sniper Edition)
    "orderflow_price_divergence",
    "beta_neutral_momentum",
    "vol_structural_squeeze",
    # [NEW] Alpha Miner Refactor
    "dist_from_weekly_vwap",
    "macro_vol_regime_shift",
    "taker_absorption_score",
    "liq_intensity_proxy",
    "capitulation_proxy",
    "idiosyncratic_return_24h",
    "exhaustion_cascade_score",
    "price_impact_asymmetry",
    "session_seasonality_sin",
    "session_seasonality_cos",
)


def _rolling_robust_z(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling median/MAD Z-score (causal).

    When MAD ≈ 0 (constant-like series, e.g. 8h funding repeated on 1h bars),
    falls back to rolling std to prevent divide-by-near-zero explosion.

    Args:
        series: Input price or feature series.
        window: Rolling window size.

    Returns:
        Z-scored series.

    """
    min_p = max(1, window // 10)
    med = series.rolling(window=window, min_periods=min(window, min_p)).median()
    mad = (series - med).abs().rolling(window=window, min_periods=min(window, min_p)).median()
    # Fallback denominator: use rolling std when MAD is near-zero (repeated values)
    std = series.rolling(window=window, min_periods=min(window, min_p)).std().fillna(0.0)
    denom = np.where(mad * 1.4826 > 1e-8, mad * 1.4826, std + 1e-12)
    z: pd.Series = (series - med) / pd.Series(denom, index=series.index)
    return z


def _log_modulus(z: pd.Series) -> pd.Series:
    """Apply log-modulus transformation to preserve sign while compressing tails.

    Args:
        z: Input series.

    Returns:
        Transformed series.

    """
    s = np.sign(z.to_numpy(dtype=np.float64))
    a = np.log(np.abs(z.to_numpy(dtype=np.float64)) + 1.0)
    return pd.Series(s * a, index=z.index)


def _yang_zhang_vol_24(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 24
) -> pd.Series:
    """Calculate rolling Yang-Zhang volatility (OHLC).

    Args:
        open_: Open price series.
        high: High price series.
        low: Low price series.
        close: Close price series.
        window: Rolling window size.

    Returns:
        Volatility series (sqrt variance per bar).

    """
    o = open_.astype(np.float64)
    h = high.astype(np.float64)
    low_s = low.astype(np.float64)
    c = close.astype(np.float64)
    log_ho = np.log((h / o).clip(lower=1e-12))
    log_lo = np.log((low_s / o).clip(lower=1e-12))
    log_co = np.log((c / o).clip(lower=1e-12))
    log_cc = np.log((c / c.shift(1)).clip(lower=1e-12))
    log_oo = np.log((o / c.shift(1)).clip(lower=1e-12))
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    w = float(window)
    k = 0.34 / (1.34 + (w + 1.0) / max(w - 1.0, 1.0))
    min_p = max(2, int(window) // 4)
    close_vol = log_cc.rolling(int(window), min_periods=min(int(window), min_p)).var()
    overnight_vol = log_oo.rolling(int(window), min_periods=min(int(window), min_p)).var()
    rs_mean = rs.rolling(int(window), min_periods=min(int(window), min_p)).mean()
    var_yz: pd.Series = overnight_vol + k * close_vol + (1.0 - k) * rs_mean
    return cast(pd.Series, np.sqrt(var_yz.clip(lower=0.0)))


@njit(cache=True, fastmath=True)  # type: ignore[untyped-decorator]
def _get_frac_weights(d: float, size: int) -> np.ndarray:
    """Generate weights for fractional differentiation.

    Args:
        d: Differentiation order.
        size: Number of weights to generate.

    Returns:
        Array of weights.

    """
    w = np.empty(size, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, size):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return w


@njit(cache=True, fastmath=True)  # type: ignore[untyped-decorator]
def _fast_frac_diff(series: np.ndarray, w: np.ndarray, tau: float) -> np.ndarray:
    """Apply fractional differentiation using Numba for speed.

    Args:
        series: Input price array.
        w: Weight array.
        tau: Weight threshold for truncation.

    Returns:
        Differentiated array.

    """
    n = len(series)
    out = np.full(n, np.nan, dtype=np.float64)
    w_abs = np.abs(w)
    active_len = len(w)
    for i in range(len(w)):
        if w_abs[i] < tau:
            active_len = i
            break

    w_active = w[:active_len]

    for i in range(active_len - 1, n):
        val = 0.0
        for j in range(active_len):
            val += w_active[j] * series[i - j]
        out[i] = val
    return out


def fractional_differentiation(series: pd.Series, d: float, tau: float = 1e-5) -> pd.Series:
    """Apply fractional differentiation to achieve stationarity while preserving memory.

    Uses a fixed-window approach to balance memory preservation and data loss.

    Args:
        series: Input price series.
        d: Differentiation order.
        tau: Weight threshold.

    Returns:
        Stationary series.

    """
    if d == 0.0:
        return series
    if d == 1.0:
        return series.diff()

    arr = series.to_numpy(dtype=np.float64)
    # Limit weights to max 250 bars to preserve more data
    n_weights = min(len(arr), 250)
    w = _get_frac_weights(d, n_weights)
    diff_arr = _fast_frac_diff(arr, w, tau)
    return pd.Series(diff_arr, index=series.index)


@njit(cache=True, fastmath=True)  # type: ignore[untyped-decorator]
def _rolling_hurst_rs(prices: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling Hurst exponent using R/S approximation.

    Args:
        prices: Input price array.
        window: Rolling window size.

    Returns:
        Hurst exponent array.

    """
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 4:
        return out

    # Fast localized Hurst approximation using variance ratio
    for i in range(window, n):
        m1 = 0.0
        for j in range(window):
            v1 = prices[i - window + j + 1] - prices[i - window + j]
            m1 += v1
        m1 /= window

        var1 = 0.0
        for j in range(window):
            v1 = prices[i - window + j + 1] - prices[i - window + j]
            var1 += (v1 - m1) ** 2
        var1 /= window

        m_2 = 0.0
        for j in range(window - 1):
            v2 = prices[i - window + j + 2] - prices[i - window + j]
            m_2 += v2
        m_2 /= window - 1

        var2 = 0.0
        for j in range(window - 1):
            v2 = prices[i - window + j + 2] - prices[i - window + j]
            var2 += (v2 - m_2) ** 2
        var2 /= window - 1

        if var1 > 1e-12 and var2 > 1e-12:
            h = 0.5 * np.log(var2 / var1) / np.log(2.0)
            out[i] = h
        else:
            out[i] = 0.5

    return out


def build_gp_input_features(df: pd.DataFrame, tf: str = "1h") -> pd.DataFrame:
    """Build features for GP SymbolicTransformer input.

    Enhanced with momentum, MA distances, volatility markers, and microstructure.

    Args:
        df: Input OHLCV dataframe.
        tf: Timeframe of the input dataframe.

    Returns:
        Dataframe containing engineered ALPHA features.

    """
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    vol = df["volume"].astype(np.float64)
    open_ = df["open"].astype(np.float64) if "open" in df.columns else close.shift(1).fillna(close)

    # 1. Price Momentum (log-modulus of log returns)
    # Using fixed hour horizons: 1h, 3h, 6h, 12h, 24h, 48h, 72h, 168h
    for h_hours in [1, 3, 6, 12, 24, 48, 72, 168]:
        w = _get_window(h_hours, tf)
        raw_ret = pd.Series(
            np.log(close / close.shift(w).clip(lower=1e-12)),
            index=out.index,
        )
        out[f"ret_{h_hours}"] = _log_modulus(raw_ret)

    # 2. Moving Average Distance (Mean Reversion markers)
    for h_hours in [24, 168]:
        w = _get_window(h_hours, tf)
        ma = close.rolling(window=w, min_periods=min(w, max(1, w // 4))).mean()
        out[f"ma_dist_{h_hours}"] = (close / (ma + 1e-12)) - 1.0

    # 3. Volume Intensity
    for h_hours in [24, 168]:
        w = _get_window(h_hours, tf)
        vma = vol.rolling(window=w, min_periods=min(w, max(1, w // 4))).mean()
        out[f"vol_ratio_{h_hours}"] = (vol / (vma + 1e-12)).fillna(1.0)

    # 4. HL Spread (Volatility proxy) — long-tail compression
    raw_hl = (high - low) / (close + 1e-9)
    out["hl_spread"] = np.log1p(raw_hl.clip(lower=0.0))

    # 5. Funding Rate
    if "funding_rate" in df.columns:
        out["funding_rate"] = df["funding_rate"].astype(np.float64)
    else:
        out["funding_rate"] = np.nan

    w72 = _get_window(72, tf)
    w8 = _get_window(8, tf)
    out["funding_z_72"] = _rolling_robust_z(out["funding_rate"], w72)
    out["funding_chg_8"] = out["funding_rate"].diff(w8)

    # 6. Buy/Sell Ratio
    if "quote_volume" in df.columns:
        qv = df["quote_volume"].astype(np.float64)
    else:
        qv = close * vol

    tbq = (
        df["taker_buy_quote_volume"].astype(np.float64)
        if "taker_buy_quote_volume" in df.columns
        else qv * 0.5
    )
    tsq = qv - tbq
    out["buy_sell_ratio"] = (tbq / (tsq + 1e-9)).fillna(1.0)
    imb = out["buy_sell_ratio"] - 1.0
    
    w24 = _get_window(24, tf)
    out["taker_imbalance_z_24"] = _rolling_robust_z(imb, w24)

    out["realized_vol_yz_24"] = _yang_zhang_vol_24(open_, high, low, close, w24)

    log_ret_1 = np.log(close / close.shift(1).clip(lower=1e-12))
    sig_24 = log_ret_1.rolling(w24, min_periods=min(w24, max(2, w24 // 2))).std() + 1e-12
    
    w6 = _get_window(6, tf)
    out["ret_vol_adj_6"] = out["ret_6"] / (sig_24 * np.sqrt(w6) + 1e-12)
    out["ret_vol_adj_24"] = out["ret_24"] / (sig_24 * np.sqrt(w24) + 1e-12)

    out["liq_proxy_6"] = (low.rolling(w6, min_periods=min(w6, max(1, w6 // 2))).min() - close) / (close + 1e-12)

    out["vol_skew_24"] = log_ret_1.rolling(w24, min_periods=min(w24, max(2, w24 // 2))).skew().fillna(0.0)
    
    w12 = _get_window(12, tf)
    ma_12 = close.rolling(w12, min_periods=min(w12, max(1, w12 // 2))).mean()
    out["mom_proxy_12"] = (close / (ma_12 + 1e-12)) - 1.0

    # VWAP Distance 24h
    typ_price = (high + low + close) / 3.0
    pv = typ_price * vol
    vwap_24 = pv.rolling(w24, min_periods=min(w24, max(1, w24 // 4))).sum() / (vol.rolling(w24, min_periods=min(w24, max(1, w24 // 4))).sum() + 1e-12)
    out["vwap_dist_24"] = (close / (vwap_24 + 1e-12)) - 1.0

    # Volatility Surface 24h vs 168h
    w168 = _get_window(168, tf)
    vol_yz_168 = _yang_zhang_vol_24(open_, high, low, close, w168)
    out["vol_surface_24_168"] = out["realized_vol_yz_24"] / (vol_yz_168 + 1e-12)

    # Funding Momentum 24h
    if "funding_rate" in df.columns:
        out["funding_mom_24"] = out["funding_rate"].diff(w24)
    else:
        out["funding_mom_24"] = np.nan

    # Acceleration 24h
    out["acceleration_24"] = out["ret_24"] - out["ret_24"].shift(w24)

    # Tail Risk 24h
    neg_ret = log_ret_1.clip(upper=0.0)
    down_var = neg_ret.rolling(w24, min_periods=min(w24, max(2, w24 // 2))).var()
    tot_var = log_ret_1.rolling(w24, min_periods=min(w24, max(2, w24 // 2))).var()
    out["tail_risk_24"] = (down_var / (tot_var + 1e-12)).fillna(0.5)

    # microstructure + range
    dollar_vol = (close * vol).clip(lower=1.0)
    out["amihud_illiq_24"] = (
        (log_ret_1.abs() / dollar_vol).rolling(w24, min_periods=min(w24, max(1, w24 // 2))).mean()
    )
    high_24 = high.rolling(w24, min_periods=min(w24, max(1, w24 // 2))).max()
    low_24 = low.rolling(w24, min_periods=min(w24, max(1, w24 // 2))).min()
    out["range_pos_24"] = (close - low_24) / (high_24 - low_24 + 1e-12)

    # fractional differentiation and hurst exponent
    out["frac_diff_04"] = fractional_differentiation(close, 0.4)
    out["hurst_24"] = pd.Series(
        _rolling_hurst_rs(close.to_numpy(dtype=np.float64), w24), index=out.index
    )

    # vol-of-vol, tail rejection, session distance, btc correlation, vpin
    out["vol_of_vol_24"] = out["realized_vol_yz_24"].rolling(w24, min_periods=min(w24, max(1, w24 // 2))).std()
    
    shadows = (high - np.maximum(open_, close)) + (np.minimum(open_, close) - low)
    out["tail_rejection_24"] = (
        (shadows / (high - low + 1e-12)).rolling(w24, min_periods=min(w24, max(1, w24 // 2))).mean().fillna(0.0)
    )
    
    out["dist_from_high_24"] = (high_24 - close) / (high_24 - low_24 + 1e-12)

    if "btc_close" in df.columns:
        btc_log_ret = np.log(df["btc_close"] / df["btc_close"].shift(1).clip(lower=1e-12))
        out["corr_btc_24"] = log_ret_1.rolling(w24, min_periods=min(w24, max(2, w24 // 2))).corr(btc_log_ret).fillna(0.0)
        
        # Calculate Rolling Beta: cov(asset, btc) / var(btc)
        cov_btc = log_ret_1.rolling(w24, min_periods=min(w24, max(2, w24 // 2))).cov(btc_log_ret)
        var_btc = btc_log_ret.rolling(w24, min_periods=min(w24, max(2, w24 // 2))).var()
        out["btc_beta"] = (cov_btc / (var_btc + 1e-12)).fillna(0.0)
    else:
        out["corr_btc_24"] = 0.0
        out["btc_beta"] = 0.0

    buy_vol = tbq / (close + 1e-9)
    sell_vol = tsq / (close + 1e-9)
    out["vpin_proxy_12"] = (
        (buy_vol - sell_vol).abs().rolling(w12, min_periods=min(w12, max(1, w12 // 2))).sum() / 
        (vol.rolling(w12, min_periods=min(w12, max(1, w12 // 2))).sum() + 1e-12)
    ).fillna(0.0)

    # Structural Alpha Features
    if "funding_rate" in df.columns:
        f_rate = df["funding_rate"].astype(np.float64)
        out["funding_trap_24"] = (
            ((close > close.shift(w24)) & (f_rate < f_rate.shift(w24)))
            .astype(np.float64)
        )
    else:
        out["funding_trap_24"] = 0.0

    # downside_jump_24: capture extreme negative movements
    out["downside_jump_24"] = (
        (log_ret_1 / (sig_24 + 1e-12)).clip(upper=0.0).rolling(w24).min().abs()
    )

    # OI Based
    if "sum_open_interest" in df.columns:
        oi = df["sum_open_interest"].astype(np.float64)
        w4 = _get_window(4, tf)
        oi_mom_4 = np.log(oi / oi.shift(w4).clip(lower=1e-12))
        oi_mom_24 = np.log(oi / oi.shift(w24).clip(lower=1e-12))
        out["oi_momentum_4h"] = _log_modulus(oi_mom_4)
        out["oi_momentum_24h"] = _log_modulus(oi_mom_24)
        out["oi_price_divergence_24h"] = out["ret_24"] - out["oi_momentum_24h"]
        
        f_rate = df["funding_rate"].astype(np.float64) if "funding_rate" in df.columns else 0.0
        out["oi_funding_trap_24h"] = (
            ((out["ret_24"] < -0.01) & (oi_mom_24 > 0.02) & (f_rate < 0))
            .astype(np.float64)
        )
    else:
        for c in ["oi_momentum_4h", "oi_momentum_24h", "oi_price_divergence_24h", "oi_funding_trap_24h"]:
            out[c] = 0.0

    if "top_trader_long_short_ratio" in df.columns:
        out["top_trader_lsr_z_24h"] = _rolling_robust_z(df["top_trader_long_short_ratio"], w24)
    else:
        out["top_trader_lsr_z_24h"] = 0.0

    if "long_short_ratio" in df.columns:
        out["global_lsr_z_24h"] = _rolling_robust_z(df["long_short_ratio"], w24)
    else:
        out["global_lsr_z_24h"] = 0.0

    if "top_trader_long_short_ratio" in df.columns and "long_short_ratio" in df.columns:
        out["lsr_spread_12h"] = (
            (df["top_trader_long_short_ratio"] - df["long_short_ratio"])
            .rolling(w12, min_periods=min(w12, max(1, w12 // 2))).mean().fillna(0.0)
        )
    else:
        out["lsr_spread_12h"] = 0.0

    out["taker_buy_sell_ratio_12h"] = (
        tbq.rolling(w12, min_periods=min(w12, max(1, w12 // 2))).sum() / 
        (tsq.rolling(w12, min_periods=min(w12, max(1, w12 // 2))).sum() + 1e-9)
    ).fillna(1.0)
    
    cvd_rolling = (tbq - tsq).rolling(w24, min_periods=min(w24, max(2, w24 // 2))).sum()
    cvd_norm = cvd_rolling / (vol.rolling(w24, min_periods=min(w24, max(2, w24 // 2))).sum() + 1e-9)
    out["cvd_divergence_24h"] = out["ret_24"] - _log_modulus(cvd_norm)
    
    out["taker_acceleration_24h"] = ((tbq - tsq) / (vol + 1e-9)).diff(w24).fillna(0.0)

    if "funding_rate" in df.columns:
        out["funding_intensity_24h"] = (
            df["funding_rate"].abs() * out["realized_vol_yz_24"]
        ).rolling(w24, min_periods=min(w24, max(2, w24 // 2))).mean().fillna(0.0)
    else:
        out["funding_intensity_24h"] = 0.0
        
    out["absorption_ratio_12h"] = (
        out["ret_12"].abs() / (out["realized_vol_yz_24"] * out["vol_ratio_24"] + 1e-9)
    ).fillna(0.0)

    out["motif_crowded_long_unwind"] = (out["global_lsr_z_24h"] + out["top_trader_lsr_z_24h"] + out["oi_momentum_24h"] - out["taker_imbalance_z_24"]).fillna(0.0)
    out["motif_funding_short_squeeze"] = (-out["funding_z_72"] + out["oi_momentum_24h"] - out["ret_12"]).fillna(0.0)
    out["motif_taker_absorption"] = (_log_modulus(out["taker_buy_sell_ratio_12h"] - 1.0) - out["ret_vol_adj_6"]).fillna(0.0)
    out["motif_oi_price_dislocation"] = (out["oi_momentum_24h"] - out["ret_24"]).fillna(0.0)
    out["motif_liq_pressure"] = (out["downside_jump_24"] + out["tail_risk_24"] - out["range_pos_24"]).fillna(0.0)

    # F-A: OI Delta + OI Acceleration
    # oi_delta_4h: 4-bar log-diff of OI, normalized by rolling std (stationarity 보장)
    # oi_accel_24h: oi_delta_4h의 24h-window 추가 diff (position build/unwind 가속도)
    if "sum_open_interest" in df.columns:
        oi = df["sum_open_interest"].astype(np.float64)
        w4 = _get_window(4, tf)
        oi_log_diff = np.log(oi / oi.shift(1).clip(lower=1e-12))
        oi_delta_raw = oi_log_diff.rolling(w4, min_periods=min(w4, max(1, w4 // 2))).sum()
        out["oi_delta_4h"] = _rolling_robust_z(oi_delta_raw, w24)
        out["oi_accel_24h"] = (out["oi_delta_4h"] - out["oi_delta_4h"].shift(w24)).fillna(0.0)
    else:
        out["oi_delta_4h"] = 0.0
        out["oi_accel_24h"] = 0.0

    # F-B: Funding x Realized-Vol Interaction
    # funding_z_72 (기존) x realized_vol_yz_24 z-score: high funding + high vol = extremum
    if "funding_rate" in df.columns:
        realized_vol_z = _rolling_robust_z(out["realized_vol_yz_24"], w24)
        out["funding_vol_interaction"] = (out["funding_z_72"] * realized_vol_z).fillna(0.0)
    else:
        out["funding_vol_interaction"] = 0.0

    # F-C: Perp-Spot Basis Proxy
    # Spot 데이터 미제공 → close의 168h rolling mean 대비 deviation을 carry proxy로 사용
    # perp_basis_proxy = close / rolling_mean(close, 168h) - 1 (non-stationary 방지: ratio 형태)
    w168_basis = _get_window(168, tf)
    rolling_mean_168 = close.rolling(w168_basis, min_periods=min(w168_basis, max(1, w168_basis // 4))).mean()
    out["perp_basis_proxy"] = ((close / (rolling_mean_168 + 1e-12)) - 1.0).fillna(0.0)

    # [NEW] F-D: High-Order Interactions
    # 1. Orderflow-Price Divergence: VPIN * Return Vol Adj
    # If VPIN is high (toxicity) and return is negative, it indicates high sell pressure absorption.
    out["orderflow_price_divergence"] = (out["vpin_proxy_12"] * out["ret_vol_adj_24"]).fillna(0.0)

    # 2. Beta-Neutral Momentum (Initial raw return, will be cross-sectionally neutralized in Step 2)
    # Using 24h as base
    out["beta_neutral_momentum"] = out["ret_24"]

    # [NEW] Alpha Miner Refactor Features
    # 1. Macro Structural Features
    # dist_from_weekly_vwap: 168h rolling average price distance
    vwap_168 = pv.rolling(w168, min_periods=min(w168, max(1, w168 // 4))).sum() / (vol.rolling(w168, min_periods=min(w168, max(1, w168 // 4))).sum() + 1e-12)
    out["dist_from_weekly_vwap"] = (close / (vwap_168 + 1e-12)) - 1.0
    
    # macro_vol_regime_shift: realized_vol_yz_24 / realized_vol_yz_168
    out["macro_vol_regime_shift"] = (out["realized_vol_yz_24"] / (vol_yz_168 + 1e-12)).fillna(1.0)

    # 2. Taker Aggression Refinement
    # taker_absorption_score: _rolling_robust_z(taker_buy_quote_volume - taker_sell_quote_volume, 24) - _rolling_robust_z(log_ret_1, 24)
    taker_net_qv = tbq - tsq
    out["taker_absorption_score"] = _rolling_robust_z(taker_net_qv, w24) - _rolling_robust_z(log_ret_1, w24)

    # 3. Liquidation Proxy (Simulation)
    out["liq_intensity_proxy"] = (high - np.maximum(open_, close)) / (high - low + 1e-9)
    out["capitulation_proxy"] = (np.minimum(open_, close) - low) / (high - low + 1e-9)

    # 4. Advanced Alpha Features (v22)
    # idio_ret: R_asset - (beta * R_btc)
    if "btc_close" in df.columns:
        btc_log_ret_24 = np.log(df["btc_close"] / df["btc_close"].shift(w24).clip(lower=1e-12))
        raw_ret_24 = np.log(close / close.shift(w24).clip(lower=1e-12))
        out["idiosyncratic_return_24h"] = raw_ret_24 - (out["btc_beta"] * btc_log_ret_24)
    else:
        out["idiosyncratic_return_24h"] = 0.0

    # exhaustion_cascade_score: 3-bar sum of capitulation + ret drop + oi drop
    if "sum_open_interest" in df.columns:
        oi_tmp = df["sum_open_interest"].astype(np.float64)
        oi_log_diff = np.log(oi_tmp / oi_tmp.shift(1).clip(lower=1e-12))
    else:
        oi_log_diff = pd.Series(0.0, index=out.index)
    
    exhaustion_signal = (
        (out["capitulation_proxy"] > 0.5) & 
        (log_ret_1 < 0) & 
        (oi_log_diff < 0)
    ).astype(np.float64)
    out["exhaustion_cascade_score"] = exhaustion_signal.rolling(3).sum().fillna(0.0)

    # price_impact_asymmetry: (AbsRet/BuyVol) / (AbsRet/SellVol)
    abs_ret_1 = log_ret_1.abs()
    impact_buy = (abs_ret_1 / (tbq + 1e-9)).rolling(w24).mean()
    impact_sell = (abs_ret_1 / (tsq + 1e-9)).rolling(w24).mean()
    out["price_impact_asymmetry"] = (impact_buy / (impact_sell + 1e-9)).fillna(1.0)

    # session_seasonality: Sine/Cosine of hour
    if isinstance(df.index, pd.DatetimeIndex):
        hours = df.index.hour
        out["session_seasonality_sin"] = np.sin(2 * np.pi * hours / 24)
        out["session_seasonality_cos"] = np.cos(2 * np.pi * hours / 24)
    else:
        out["session_seasonality_sin"] = 0.0
        out["session_seasonality_cos"] = 0.0

    # [NEW] Macro Decoupling Features (Spec v15)
    w168 = _get_window(168, tf)
    # macro_trend_168h: BTC-like trend proxy (if btc_close not in df, use own close)
    ref_close = df["btc_close"] if "btc_close" in df.columns else close
    out["macro_trend_168h"] = np.log(ref_close / ref_close.shift(w168).clip(lower=1e-12))

    out["macro_ret_1h"] = _macro_risk_adj_ret_1h(ref_close, w24)

    # macro_vol_24h: Yang-Zhang Volatility
    out["macro_vol_24h"] = _yang_zhang_vol_24(open_, high, low, close, w24)
    
    # macro_liq_24h: Volume Momentum (Liquidity Proxy)
    vma_24 = vol.rolling(w24).mean()
    vma_168 = vol.rolling(w168).mean()
    out["macro_liq_24h"] = (vma_24 / (vma_168 + 1e-12)) - 1.0
    
    # macro_cost_168h: Rolling Funding Mean
    if "funding_rate" in df.columns:
        out["macro_cost_168h"] = df["funding_rate"].rolling(w168).mean()
    else:
        out["macro_cost_168h"] = 0.0

    # [Mixed Normalization - Causal]
    # Vol-like features: log1p + rolling robust z-score
    # Other features: rolling robust z-score

    _vol_macro_cols = [
        c for c in ["macro_vol_24h", "macro_downside_vol_24h", "macro_cs_dispersion_24h"]
        if c in out.columns
    ]
    if _vol_macro_cols:
        from src.domain.futures.ml_pipeline.regime.causal_transformers import (
            causal_log_robust_zscore,
        )
        _scaled_vol = causal_log_robust_zscore(
            out[_vol_macro_cols].fillna(0.0),
            window=w168,
            min_periods=max(8, w24 // 2),
            clip=5.0,
        )
        out[_vol_macro_cols] = _scaled_vol

    _other_macro_cols = [c for c in SYSTEMIC_HMM_FEATURE_COLUMNS if c not in _vol_macro_cols and c in out.columns]
    if _other_macro_cols:
        from src.domain.futures.ml_pipeline.regime.causal_transformers import (
            causal_robust_zscore,
        )
        _scaled_other = causal_robust_zscore(
            out[_other_macro_cols].fillna(0.0),
            window=w168,
            min_periods=max(8, w24 // 2),
            clip=5.0,
        )
        out[_other_macro_cols] = _scaled_other

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_hmm_input_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build GaussianHMM features designed for regime structural detection.

    Design (HMM improvement):
    - Frac_Diff_04: fractional_differentiation(close, d=0.4) for stationarity + memory.
    - Vol_Ratio: short-term/long-term Parkinson volatility ratio.
    - Volume_Momentum: 24h vs 168h MA ratio.
    - Funding_Inversion_Intensity: Intensity of negative funding during price drops.

    Args:
        df: Input OHLCV dataframe.

    Returns:
        Dataframe containing HMM input features.

    """
    close = df["close"].astype(np.float64)
    log_ret = np.log(close / close.shift(1).clip(lower=1e-12))

    out = pd.DataFrame(index=df.index)

    # 1. Fractional Differentiation (d=0.4) - Stationarity + Memory
    out["Frac_Diff_04"] = fractional_differentiation(close, d=0.4)

    # 2. Short/long volatility ratio (Parkinson estimator)
    h_hi = df["high"].astype(np.float64)
    h_lo = df["low"].astype(np.float64) + 1e-12
    hl = np.log((h_hi / h_lo).clip(lower=1e-12))
    park = np.sqrt(np.maximum(1.0 / (4.0 * np.log(2.0)) * (hl**2), 0.0))
    pser = pd.Series(park, index=df.index)
    short_vol = pser.rolling(window=24, min_periods=min(24, 5)).mean()
    long_vol = pser.rolling(window=168, min_periods=min(168, 20)).mean()
    # Vol ratio > 1 = volatility expanding (trending/crisis), < 1 = compressing (range)
    out["Vol_Ratio"] = (short_vol / (long_vol + 1e-12)).replace([np.inf, -np.inf], 1.0).fillna(1.0)

    # 3. Volume momentum: 24h vs 168h MA (buying pressure trend)
    vol = df["volume"].astype(np.float64)
    vol_ma_s = vol.rolling(24, min_periods=min(24, 5)).mean()
    vol_ma_l = vol.rolling(168, min_periods=min(168, 20)).mean()
    out["Volume_Momentum"] = (vol_ma_s / (vol_ma_l + 1e-12)) - 1.0

    # 4. Funding Inversion Intensity: capture bearish extreme/stress
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].astype(np.float64).ffill().fillna(0.0)
        # Intensity = abs(funding) * abs(return) only when both are negative
        # Captures strength of shorts paying longs during active price decline
        inversion = np.where((fr < 0) & (log_ret < 0), np.abs(fr * log_ret), 0.0)
        # Use EWM to smooth intensity for HMM stability
        intensity_ser = pd.Series(inversion, index=df.index)
        out["Funding_Inversion_Intensity"] = intensity_ser.ewm(span=12).mean()
    else:
        out["Funding_Inversion_Intensity"] = 0.0

    return out


def build_systemic_hmm_features(
    panel_df: pd.DataFrame, alpha_panel: pd.DataFrame | None = None, tf: str = "1h"
) -> pd.DataFrame:
    """Build systemic macro features for HMM (Tiered Universe Architecture).

    Tier 1: Macro Trend/Vol/Funding (Anchor Symbols: BTC, ETH)
    Tier 2: Cross-Sectional Dynamics (Index Symbols: Top 10)
    Tier 3: Dynamic Universe (Filtered out to avoid noise)

    Args:
        panel_df: Input panel dataframe.
        alpha_panel: Optional alpha panel dataframe.
        tf: Timeframe of the input dataframe.

    Returns:
        Dataframe indexed by datetime with SYSTEMIC_HMM_FEATURE_COLUMNS.

    """
    from config.opt_config import FUTURES_ANCHOR_SYMBOLS, FUTURES_MACRO_INDEX_SYMBOLS

    _ = alpha_panel

    if panel_df.empty:
        return pd.DataFrame(columns=list(SYSTEMIC_HMM_FEATURE_COLUMNS))

    idx = panel_df.index.get_level_values("datetime").unique().sort_values()
    syms = panel_df.index.get_level_values("symbol").unique()
    
    # Tier 1 Selection (Anchors)
    available_anchors = [s for s in FUTURES_ANCHOR_SYMBOLS if s in syms]
    tier1_sym = available_anchors[0] if available_anchors else next((s for s in syms if "BTC" in s), None)

    # Tier 2 Selection (Macro Index)
    available_indices = [s for s in FUTURES_MACRO_INDEX_SYMBOLS if s in syms]
    tier2_syms = available_indices if available_indices else list(syms)

    out = pd.DataFrame(index=idx)
    w24 = _get_window(24, tf)
    w168 = _get_window(168, tf)
    
    if tier1_sym:
        t1_df = panel_df.xs(tier1_sym, level="symbol")
        t1_close = t1_df["close"].astype(np.float64)
        t1_high = t1_df["high"].astype(np.float64)
        t1_low = t1_df["low"].astype(np.float64)
        t1_open = t1_df["open"].astype(np.float64) if "open" in t1_df.columns else t1_close.shift(1).fillna(t1_close)
        t1_vol = t1_df["volume"].astype(np.float64)

        # 1. macro_trend_168h & macro_trend_24h
        out["macro_trend_168h"] = np.log(t1_close / t1_close.shift(w168).clip(lower=1e-12))
        out["macro_trend_24h"] = np.log(t1_close / t1_close.shift(w24).clip(lower=1e-12))

        t1_log_ret_1h = np.log(t1_close / t1_close.shift(1).clip(lower=1e-12)).fillna(0.0)
        out["macro_ret_1h"] = _macro_risk_adj_ret_1h(t1_close, w24)

        # 2. macro_vol_24h (Yang-Zhang)
        out["macro_vol_24h"] = _yang_zhang_vol_24(t1_open, t1_high, t1_low, t1_close, w24)

        # 4. macro_funding_mom_24h
        if "funding_rate" in t1_df.columns:
            fr = t1_df["funding_rate"].astype(np.float64).ffill().fillna(0.0)
            out["macro_funding_mom_24h"] = fr.diff(w24).fillna(0.0)
        else:
            out["macro_funding_mom_24h"] = 0.0

        # 5. macro_downside_vol_24h
        neg_ret = t1_log_ret_1h.where(t1_log_ret_1h < 0, 0.0)
        out["macro_downside_vol_24h"] = neg_ret.rolling(
            w24, min_periods=min(w24, max(5, w24 // 4))
        ).std().fillna(0.0)

        # 6. macro_oi_delta_24h
        if "sum_open_interest" in t1_df.columns:
            oi = t1_df["sum_open_interest"].astype(np.float64).replace(0, np.nan).ffill().fillna(0.0)
            oi_chg = oi.pct_change(w24).fillna(0.0)
            out["macro_oi_delta_24h"] = oi_chg.clip(-1.0, 1.0)
        else:
            out["macro_oi_delta_24h"] = 0.0

        # 7. macro_liq_proxy_24h
        neg_vol = t1_vol.where(t1_log_ret_1h < 0, 0.0)
        neg_vol_ma = neg_vol.rolling(w24, min_periods=min(w24, max(5, w24 // 4))).mean()
        total_vol_ma = t1_vol.rolling(w24, min_periods=min(w24, max(5, w24 // 4))).mean()
        out["macro_liq_proxy_24h"] = (neg_vol_ma / (total_vol_ma + 1e-12)).fillna(0.5)

        # 8. macro_lsr_delta_24h
        if "long_short_ratio" in t1_df.columns:
            lsr = t1_df["long_short_ratio"].astype(np.float64).ffill().fillna(1.0)
            out["macro_lsr_delta_24h"] = lsr.pct_change(w24).fillna(0.0)
        else:
            out["macro_lsr_delta_24h"] = 0.0
    else:
        # Extreme fallback
        for col in ["macro_trend_168h", "macro_trend_24h", "macro_ret_1h", "macro_vol_24h", "macro_funding_mom_24h", "macro_downside_vol_24h", "macro_oi_delta_24h", "macro_liq_proxy_24h", "macro_lsr_delta_24h"]:
            out[col] = 0.0

    # Tier 2 Features: Cross-Sectional Dynamics
    panel_t2 = panel_df[panel_df.index.get_level_values("symbol").isin(tier2_syms)]
    
    if not panel_t2.empty:
        close_panel_t2 = panel_t2["close"].unstack(level="symbol")
        
        # 9. macro_cs_dispersion_24h
        cs_log_ret = np.log(close_panel_t2 / close_panel_t2.shift(1).clip(lower=1e-12))
        cs_disp = (
            cs_log_ret.rolling(w24, min_periods=min(w24, max(2, w24 // 4)))
            .std(ddof=0)
            .mean(axis=1)
        )
        out["macro_cs_dispersion_24h"] = cs_disp.reindex(idx).fillna(0.0)

        # 10. macro_breadth_168h
        ma_168_t2 = close_panel_t2.rolling(w168, min_periods=min(w168, max(20, w168 // 4))).mean()
        breadth = (close_panel_t2 > ma_168_t2).mean(axis=1)
        out["macro_breadth_168h"] = breadth.reindex(idx).fillna(0.5)
    else:
        out["macro_cs_dispersion_24h"] = 0.0
        out["macro_breadth_168h"] = 0.5

    # Tier 3: Universe Breadth (Phase C)
    # 11. macro_breadth_ma20: fraction of universe symbols with close > MA(20).
    #     shift(1) applied to prevent look-ahead bias.
    #     Formula: breadth_i,t = mean_j(close_{j,t-1} > MA20_{j,t-1})
    # 12. macro_alt_btc_rs_24h: median(alt 24h return) - btc 24h return.
    #     Positive = alt-season, Negative = BTC dominance.
    #     shift(1) applied to both numerator and denominator.
    eps = 1e-12
    panel_t3_syms = [s for s in syms if s in (tier2_syms or list(syms))]
    panel_t3 = panel_df[panel_df.index.get_level_values("symbol").isin(panel_t3_syms)]
    if not panel_t3.empty:
        close_t3 = panel_t3["close"].unstack(level="symbol")
        # MA(20) breadth -- shift(1) for causality
        w20 = max(2, _get_window(20, tf))
        ma20 = close_t3.rolling(w20, min_periods=min(w20, 2)).mean()
        above_ma20_lagged = (close_t3.shift(1) > ma20.shift(1)).astype(np.float64)
        out["macro_breadth_ma20"] = above_ma20_lagged.mean(axis=1).reindex(idx).fillna(0.5)

        # Alt-coin vs BTC relative strength (24h return spread)
        ret_24h = np.log(
            close_t3.shift(1).clip(lower=eps)
            / close_t3.shift(1 + w24).clip(lower=eps)
        )
        btc_col = next((c for c in ret_24h.columns if "BTC" in str(c)), None)
        if btc_col is not None and ret_24h.shape[1] > 1:
            alt_cols = [c for c in ret_24h.columns if c != btc_col]
            alt_median_ret = ret_24h[alt_cols].median(axis=1)
            btc_ret = ret_24h[btc_col]
            out["macro_alt_btc_rs_24h"] = (alt_median_ret - btc_ret).reindex(idx).fillna(0.0)
        else:
            out["macro_alt_btc_rs_24h"] = 0.0
    else:
        out["macro_breadth_ma20"] = 0.5
        out["macro_alt_btc_rs_24h"] = 0.0

    # Mixed normalization [T1-A] (causal):
    #   Vol-like features: log1p + rolling robust z-score
    #   Other features: rolling robust z-score

    vol_feat_cols = [
        c for c in ["macro_vol_24h", "macro_downside_vol_24h", "macro_cs_dispersion_24h"]
        if c in out.columns
    ]
    if vol_feat_cols:
        from src.domain.futures.ml_pipeline.regime.causal_transformers import (
            causal_log_robust_zscore,
        )
        out[vol_feat_cols] = causal_log_robust_zscore(
            out[vol_feat_cols].fillna(0.0),
            window=w168,
            min_periods=max(8, w24 // 2),
            clip=5.0,
        )

    other_cols = [c for c in SYSTEMIC_HMM_FEATURE_COLUMNS if c not in vol_feat_cols and c in out.columns]
    if other_cols:
        from src.domain.futures.ml_pipeline.regime.causal_transformers import (
            causal_robust_zscore,
        )
        clean_other = out[other_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[other_cols] = causal_robust_zscore(
            clean_other,
            window=w168,
            min_periods=max(8, w24 // 2),
            clip=5.0,
        )

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_macro_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between macro HMM probabilities and individual asset metrics.

    Interactions:
    - Trend Sensitivity: btc_beta * hmm_prob_bull_trend
    - Volatility State: realized_vol_24h * hmm_prob_crisis
    - Funding Pressure: funding_level * hmm_prob_bear_trend

    Args:
        df: Panel DataFrame with HMM probabilities and asset features.

    Returns:
        DataFrame with added interaction features.

    """
    # [Optimization ④] df.copy() 제거 → df.assign() 사용 (CoW-friendly, 신규 컬럼만 shallow-copy)
    # 1. Bull Trend Probability (sum calm and vol_up if they exist)
    if "hmm_prob_bull_trend" in df.columns:
        bull_prob = df["hmm_prob_bull_trend"]
    elif "hmm_prob_bull_calm" in df.columns and "hmm_prob_bull_vol_up" in df.columns:
        bull_prob = df["hmm_prob_bull_calm"] + df["hmm_prob_bull_vol_up"]
    else:
        bull_prob = pd.Series(0.0, index=df.index)

    # 2. Asset Metrics
    beta = df.get("btc_beta", df.get("corr_btc_24", pd.Series(0.0, index=df.index)))
    vol = df.get("realized_vol_yz_24", pd.Series(0.0, index=df.index))

    # Use funding_rate as funding_level proxy
    funding = df.get("funding_rate", pd.Series(0.0, index=df.index))

    crisis_prob = df.get("hmm_prob_crisis", pd.Series(0.0, index=df.index))
    bear_prob = df.get("hmm_prob_bear_trend", pd.Series(0.0, index=df.index))

    # 3. Interactions — assign() returns new DataFrame without full copy of existing columns
    return df.assign(
        btc_beta_x_bull_trend=(beta * bull_prob).fillna(0.0),
        realized_vol_x_crisis=(vol * crisis_prob).fillna(0.0),
        funding_x_bear_trend=(funding * bear_prob).fillna(0.0),
    )
