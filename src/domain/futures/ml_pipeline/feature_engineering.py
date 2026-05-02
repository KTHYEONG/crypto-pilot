"""GP / HMM input feature builders (1h OHLCV + funding)."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from numba import njit

# Columns produced by build_gp_input_features (for CS-rank / imputation in pipeline).
# Systemic HMM input (10-dim: Trend, Risk, Energy, Breadth, Funding, Dispersion, RS).
SYSTEMIC_HMM_FEATURE_COLUMNS: tuple[str, ...] = (
    "btc_trend_vol_adj_24h",
    "btc_trend_vol_adj_168h",
    "realized_vol_regime",
    "downside_vol_ratio",
    "btc_ma_dist_168h",
    "volume_momentum_24h",
    "market_breadth",
    "funding_level",
    "cs_dispersion",
    "eth_btc_rs",
)

# Posterior columns aligned to stable semantic labels (order for MetaLabeler).
HMM_SEMANTIC_PROB_COLUMNS: tuple[str, ...] = (
    "hmm_prob_bull_trend",
    "hmm_prob_bear_trend",
    "hmm_prob_chop",
    "hmm_prob_crisis",
)

# Bump when GP feature semantics change without renaming columns so raw GP
# caches are invalidated and Tier 2 retraining actually happens.
GP_FEATURE_SCHEMA_VERSION: str = "v11"

GP_ENGINEERED_FEATURE_NAMES: tuple[str, ...] = (
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
)


def _rolling_robust_z(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling median/MAD Z-score (causal).

    Args:
        series: Input price or feature series.
        window: Rolling window size.

    Returns:
        Z-scored series.

    """
    med = series.rolling(window=window, min_periods=max(10, window // 10)).median()
    mad = (series - med).abs().rolling(window=window, min_periods=max(10, window // 10)).median()
    z: pd.Series = (series - med) / (mad * 1.4826 + 1e-12)
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
    close_vol = log_cc.rolling(int(window), min_periods=max(12, int(window) // 4)).var()
    overnight_vol = log_oo.rolling(int(window), min_periods=max(12, int(window) // 4)).var()
    rs_mean = rs.rolling(int(window), min_periods=max(12, int(window) // 4)).mean()
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


def build_gp_input_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features for GP SymbolicTransformer input.

    Enhanced with momentum, MA distances, volatility markers, and microstructure.

    Args:
        df: Input OHLCV dataframe.

    Returns:
        Dataframe containing engineered GP features.

    """
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    vol = df["volume"].astype(np.float64)
    open_ = df["open"].astype(np.float64) if "open" in df.columns else close.shift(1).fillna(close)

    # 1. Price Momentum (log-modulus of log returns; tmp.md 1-D stationarity)
    for h in [1, 3, 6, 12, 24, 48, 72, 168]:
        raw_ret = pd.Series(
            np.log(close / close.shift(int(h)).clip(lower=1e-12)),
            index=out.index,
        )
        out[f"ret_{h}"] = _log_modulus(raw_ret)

    # 2. Moving Average Distance (Mean Reversion markers)
    for w in [24, 168]:
        ma = close.rolling(window=w, min_periods=w // 4).mean()
        out[f"ma_dist_{w}"] = (close / (ma + 1e-12)) - 1.0

    # 3. Volume Intensity
    for w_vol in [24, 168]:
        vma = vol.rolling(window=w_vol, min_periods=w_vol // 4).mean()
        out[f"vol_ratio_{w_vol}"] = (vol / (vma + 1e-12)).fillna(1.0)

    # 4. HL Spread (Volatility proxy) — long-tail compression
    raw_hl = (high - low) / (close + 1e-9)
    out["hl_spread"] = np.log1p(raw_hl.clip(lower=0.0))

    # 5. Funding Rate (no ffill: stale funding avoided; panel CS median impute later)
    if "funding_rate" in df.columns:
        out["funding_rate"] = df["funding_rate"].astype(np.float64)
    else:
        out["funding_rate"] = np.nan

    out["funding_z_72"] = _rolling_robust_z(out["funding_rate"], 72)
    out["funding_chg_8"] = out["funding_rate"].diff(8)

    # 6. Buy/Sell Ratio (Order Flow Imbalance proxy)
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
    out["taker_imbalance_z_24"] = _rolling_robust_z(imb, 24)

    out["realized_vol_yz_24"] = _yang_zhang_vol_24(open_, high, low, close, 24)

    log_ret_1 = np.log(close / close.shift(1).clip(lower=1e-12))
    sig_24 = log_ret_1.rolling(24, min_periods=12).std() + 1e-12
    out["ret_vol_adj_6"] = out["ret_6"] / (sig_24 * np.sqrt(6.0))
    out["ret_vol_adj_24"] = out["ret_24"] / (sig_24 * np.sqrt(24.0))

    roll_low_min = low.rolling(6, min_periods=2).min()
    out["liq_proxy_6"] = (roll_low_min - close) / (close + 1e-12)

    out["vol_skew_24"] = log_ret_1.rolling(24, min_periods=12).skew().fillna(0.0)
    ma_12 = close.rolling(12, min_periods=6).mean()
    out["mom_proxy_12"] = (close / (ma_12 + 1e-12)) - 1.0

    # --- New Alpha Injections ---
    # 1. VWAP Distance 24h
    typ_price = (high + low + close) / 3.0
    pv = typ_price * vol
    vwap_24 = pv.rolling(24, min_periods=6).sum() / (vol.rolling(24, min_periods=6).sum() + 1e-12)
    out["vwap_dist_24"] = (close / (vwap_24 + 1e-12)) - 1.0

    # 2. Volatility Surface 24h vs 168h
    vol_yz_168 = _yang_zhang_vol_24(open_, high, low, close, 168)
    out["vol_surface_24_168"] = out["realized_vol_yz_24"] / (vol_yz_168 + 1e-12)

    # 3. Funding Momentum 24h
    if "funding_rate" in df.columns:
        out["funding_mom_24"] = out["funding_rate"].diff(24)
    else:
        out["funding_mom_24"] = np.nan

    # 4. Acceleration 24h (Momentum of momentum)
    out["acceleration_24"] = out["ret_24"] - out["ret_24"].shift(24)

    # 5. Tail Risk 24h (Downside semivariance / Total variance)
    neg_ret = log_ret_1.clip(upper=0.0)
    down_var = neg_ret.rolling(24, min_periods=12).var()
    tot_var = log_ret_1.rolling(24, min_periods=12).var()
    out["tail_risk_24"] = (down_var / (tot_var + 1e-12)).fillna(0.5)

    # v5: microstructure + range (low overlap with v4 hl_yz / funding vol paths)
    dollar_vol = (close * vol).clip(lower=1.0)
    out["amihud_illiq_24"] = (
        (log_ret_1.abs() / dollar_vol).rolling(24, min_periods=12).mean()
    )
    high_24 = high.rolling(24, min_periods=12).max()
    low_24 = low.rolling(24, min_periods=12).min()
    out["range_pos_24"] = (close - low_24) / (high_24 - low_24 + 1e-12)

    # v6: fractional differentiation and hurst exponent
    out["frac_diff_04"] = fractional_differentiation(close, 0.4)
    out["hurst_24"] = pd.Series(
        _rolling_hurst_rs(close.to_numpy(dtype=np.float64), 24), index=out.index
    )

    # v7: new alpha injections (vol-of-vol, tail rejection, session distance, btc correlation, vpin)
    out["vol_of_vol_24"] = out["realized_vol_yz_24"].rolling(24, min_periods=12).std()
    
    shadows = (high - np.maximum(open_, close)) + (np.minimum(open_, close) - low)
    out["tail_rejection_24"] = (
        (shadows / (high - low + 1e-12)).rolling(24, min_periods=12).mean().fillna(0.0)
    )
    
    high_24 = high.rolling(24, min_periods=12).max()
    low_24 = low.rolling(24, min_periods=12).min()
    out["dist_from_high_24"] = (high_24 - close) / (high_24 - low_24 + 1e-12)

    if "btc_close" in df.columns:
        btc_log_ret = np.log(df["btc_close"] / df["btc_close"].shift(1).clip(lower=1e-12))
        out["corr_btc_24"] = log_ret_1.rolling(24, min_periods=12).corr(btc_log_ret).fillna(0.0)
    else:
        out["corr_btc_24"] = 0.0

    buy_vol = tbq / (close + 1e-9)
    sell_vol = tsq / (close + 1e-9)
    out["vpin_proxy_12"] = (
        (buy_vol - sell_vol).abs().rolling(12, min_periods=6).sum() / 
        (vol.rolling(12, min_periods=6).sum() + 1e-12)
    ).fillna(0.0)

    # v9: Structural Alpha Features (funding_trap_24 only; funding_squeeze_24 removed — IC≈0)
    if "funding_rate" in df.columns:
        f_rate = df["funding_rate"].astype(np.float64)
        out["funding_trap_24"] = (
            ((close > close.shift(24)) & (f_rate < f_rate.shift(24)))
            .astype(np.float64)
        )
    else:
        out["funding_trap_24"] = 0.0

    # v10: Jump / Tail Features (Pragmatic Alternative 2)
    # downside_jump_24: capture extreme negative movements (z-score < -3)
    out["downside_jump_24"] = (
        (log_ret_1 / (sig_24 + 1e-12)).clip(upper=0.0).rolling(24).min().abs()
    )

    # v11: Institutional Metrics (OI, LSR, Microstructure, Stress)
    # OI Based
    if "sum_open_interest" in df.columns:
        oi = df["sum_open_interest"].astype(np.float64)
        oi_mom_4 = np.log(oi / oi.shift(4).clip(lower=1e-12))
        oi_mom_24 = np.log(oi / oi.shift(24).clip(lower=1e-12))
        out["oi_momentum_4h"] = _log_modulus(oi_mom_4)
        out["oi_momentum_24h"] = _log_modulus(oi_mom_24)
        out["oi_price_divergence_24h"] = out["ret_24"] - out["oi_momentum_24h"]
        
        # oi_funding_trap_24h: Price down, OI up, Funding negative -> potential short trap
        f_rate = df["funding_rate"].astype(np.float64) if "funding_rate" in df.columns else 0.0
        out["oi_funding_trap_24h"] = (
            ((out["ret_24"] < -0.01) & (oi_mom_24 > 0.02) & (f_rate < 0))
            .astype(np.float64)
        )
    else:
        oi_cols = [
            "oi_momentum_4h", "oi_momentum_24h", "oi_price_divergence_24h", "oi_funding_trap_24h"
        ]
        for c in oi_cols:
            out[c] = 0.0

    # LSR Based
    if "top_trader_long_short_ratio" in df.columns:
        tt_lsr = df["top_trader_long_short_ratio"].astype(np.float64)
        out["top_trader_lsr_z_24h"] = _rolling_robust_z(tt_lsr, 24)
    else:
        out["top_trader_lsr_z_24h"] = 0.0

    if "long_short_ratio" in df.columns:
        g_lsr = df["long_short_ratio"].astype(np.float64)
        out["global_lsr_z_24h"] = _rolling_robust_z(g_lsr, 24)
    else:
        out["global_lsr_z_24h"] = 0.0

    if "top_trader_long_short_ratio" in df.columns and "long_short_ratio" in df.columns:
        out["lsr_spread_12h"] = (
            (df["top_trader_long_short_ratio"] - df["long_short_ratio"])
            .rolling(12, min_periods=6).mean().fillna(0.0)
        )
    else:
        out["lsr_spread_12h"] = 0.0

    # Microstructure
    tbq_12 = tbq.rolling(12, min_periods=6).sum()
    tsq_12 = tsq.rolling(12, min_periods=6).sum()
    out["taker_buy_sell_ratio_12h"] = (tbq_12 / (tsq_12 + 1e-9)).fillna(1.0)
    
    cvd_rolling = (tbq - tsq).rolling(24, min_periods=12).sum()
    cvd_norm = cvd_rolling / (vol.rolling(24, min_periods=12).sum() + 1e-9)
    out["cvd_divergence_24h"] = out["ret_24"] - _log_modulus(cvd_norm)
    
    imb_24 = (tbq - tsq) / (vol + 1e-9)
    out["taker_acceleration_24h"] = imb_24.diff(24).fillna(0.0)

    # Stress
    if "funding_rate" in df.columns:
        out["funding_intensity_24h"] = (
            df["funding_rate"].abs() * out["realized_vol_yz_24"]
        ).rolling(24, min_periods=12).mean().fillna(0.0)
    else:
        out["funding_intensity_24h"] = 0.0
        
    # absorption_ratio_12h: proxy for price discovery efficiency (Return / Volatility / Volume)
    out["absorption_ratio_12h"] = (
        out["ret_12"].abs() / (out["realized_vol_yz_24"] * out["vol_ratio_24"] + 1e-9)
    ).fillna(0.0)

    # v12: low-DoF structural motif features from existing v11 primitives
    out["motif_crowded_long_unwind"] = (
        out["global_lsr_z_24h"]
        + out["top_trader_lsr_z_24h"]
        + out["oi_momentum_24h"]
        - out["taker_imbalance_z_24"]
    ).fillna(0.0)
    out["motif_funding_short_squeeze"] = (
        -out["funding_z_72"] + out["oi_momentum_24h"] - out["ret_12"]
    ).fillna(0.0)
    out["motif_taker_absorption"] = (
        _log_modulus(out["taker_buy_sell_ratio_12h"] - 1.0) - out["ret_vol_adj_6"]
    ).fillna(0.0)
    out["motif_oi_price_dislocation"] = (out["oi_momentum_24h"] - out["ret_24"]).fillna(0.0)
    out["motif_liq_pressure"] = (
        out["downside_jump_24"] + out["tail_risk_24"] - out["range_pos_24"]
    ).fillna(0.0)


    # vol_ratio / buy_sell_ratio: keep finite defaults; ret/ma_dist/funding left NaN for CS impute
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
    short_vol = pser.rolling(window=24, min_periods=5).mean()
    long_vol = pser.rolling(window=168, min_periods=20).mean()
    # Vol ratio > 1 = volatility expanding (trending/crisis), < 1 = compressing (range)
    out["Vol_Ratio"] = (short_vol / (long_vol + 1e-12)).replace([np.inf, -np.inf], 1.0).fillna(1.0)

    # 3. Volume momentum: 24h vs 168h MA (buying pressure trend)
    vol = df["volume"].astype(np.float64)
    vol_ma_s = vol.rolling(24, min_periods=5).mean()
    vol_ma_l = vol.rolling(168, min_periods=20).mean()
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
    panel_df: pd.DataFrame, alpha_panel: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build single-timeseries features for Macro Market HMM using systemic aggregates.

    8-dim: Trend (24h/168h), Vol Regime, Downside Risk, Distance, Energy, Breadth, Funding.

    Args:
        panel_df: Input panel dataframe.
        alpha_panel: Optional alpha panel dataframe.

    Returns:
        Dataframe containing systemic HMM features.

    """
    _ = alpha_panel

    if panel_df.empty:
        # Return empty dataframe with expected columns
        cols = [
            "btc_trend_24h", "btc_trend_168h", "vol_ratio", "downside_vol_ratio",
            "btc_ma_dist_168h", "volume_momentum_24h", "market_breadth", "funding_mean"
        ]
        # Return a dataframe with datetime index to avoid KeyError later
        idx = pd.DatetimeIndex([], name="datetime")
        return pd.DataFrame(columns=cols, index=idx)

    market_feats = panel_df.groupby(level="datetime").agg(
        {
            "cs_dispersion": "first",
            "market_breadth": "first",
        }
    )
    if "funding_rate" in panel_df.columns:
        funding_mean = panel_df["funding_rate"].groupby(level="datetime").mean()
    else:
        funding_mean = pd.Series(0.0, index=market_feats.index)

    idx = market_feats.index
    syms = panel_df.index.get_level_values("symbol").unique()
    btc_sym = next((s for s in syms if "BTC" in s), None)

    if btc_sym:
        btc_df = panel_df.xs(btc_sym, level="symbol")
        btc_close = btc_df["close"].astype(np.float64)
        log_ret_1 = np.log(btc_close / btc_close.shift(1).clip(lower=1e-12))
        sig_24 = log_ret_1.rolling(24, min_periods=12).std() + 1e-12

        # 1. Trend (Short/Long)
        btc_trend_24h = (np.log(btc_close / btc_close.shift(24).clip(lower=1e-12))) / sig_24
        btc_trend_168h = (np.log(btc_close / btc_close.shift(168).clip(lower=1e-12))) / (
            sig_24 * np.sqrt(7.0)
        )

        # 2. Volatility Regime
        h_hi = btc_df["high"].astype(np.float64)
        h_lo = btc_df["low"].astype(np.float64) + 1e-12
        hl = np.log((h_hi / h_lo).clip(lower=1e-12))
        park = pd.Series(
            np.sqrt(np.maximum(1.0 / (4.0 * np.log(2.0)) * (hl**2), 0.0)),
            index=btc_df.index,
        )
        rv_regime = park.rolling(24).mean() / (park.rolling(168).mean() + 1e-12)

        # 3. Downside Risk
        neg_ret = log_ret_1.clip(upper=0.0)
        downside_vol_ratio = (
            neg_ret.rolling(24).std() / (log_ret_1.rolling(24).std() + 1e-12)
        ).fillna(0.5)

        # 4. Structural Distance (Capitulation)
        ma_168 = btc_close.rolling(168, min_periods=24).mean()
        btc_ma_dist_168h = (btc_close / (ma_168 + 1e-12)) - 1.0

        # 5. Energy (Squeeze Precursor)
        btc_vol = btc_df["volume"].astype(np.float64)
        vol_energy = btc_vol.rolling(24).mean() / (btc_vol.rolling(168).mean() + 1e-12) - 1.0
    else:
        btc_trend_24h = pd.Series(0.0, index=idx)
        btc_trend_168h = pd.Series(0.0, index=idx)
        rv_regime = pd.Series(1.0, index=idx)
        downside_vol_ratio = pd.Series(0.5, index=idx)
        btc_ma_dist_168h = pd.Series(0.0, index=idx)
        vol_energy = pd.Series(0.0, index=idx)

    out = pd.DataFrame(index=idx)
    out["btc_trend_vol_adj_24h"] = btc_trend_24h.reindex(idx).fillna(0.0)
    out["btc_trend_vol_adj_168h"] = btc_trend_168h.reindex(idx).fillna(0.0)
    out["realized_vol_regime"] = rv_regime.reindex(idx).fillna(1.0)
    out["downside_vol_ratio"] = downside_vol_ratio.reindex(idx).fillna(0.5)
    out["btc_ma_dist_168h"] = btc_ma_dist_168h.reindex(idx).fillna(0.0)
    out["volume_momentum_24h"] = vol_energy.reindex(idx).fillna(0.0)
    out["market_breadth"] = market_feats["market_breadth"].reindex(idx).fillna(0.0)
    out["funding_level"] = funding_mean.reindex(idx).fillna(0.0)
    
    # 6. Cross-Sectional Dispersion
    out["cs_dispersion"] = market_feats["cs_dispersion"].reindex(idx).fillna(0.0)
    
    # 7. ETH/BTC Relative Strength (Risk-on/off proxy)
    eth_sym = next((s for s in syms if "ETH" in s), None)
    if btc_sym and eth_sym:
        eth_close = panel_df.xs(eth_sym, level="symbol")["close"].astype(np.float64)
        rs = eth_close / btc_close
        rs_ema = rs.ewm(span=24).mean()
        out["eth_btc_rs"] = (rs / (rs_ema + 1e-12)) - 1.0
    else:
        out["eth_btc_rs"] = 0.0

    out = out[list(SYSTEMIC_HMM_FEATURE_COLUMNS)]
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
