"""GP / HMM input feature builders (1h OHLCV + funding)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns produced by build_gp_input_features (for CS-rank / imputation in pipeline).
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
)


def _rolling_robust_z(series: pd.Series, window: int) -> pd.Series:
    """Rolling median/MAD Z-score (causal)."""
    med = series.rolling(window=window, min_periods=max(10, window // 10)).median()
    mad = (series - med).abs().rolling(window=window, min_periods=max(10, window // 10)).median()
    z = (series - med) / (mad * 1.4826 + 1e-12)
    return z


def _log_modulus(z: pd.Series) -> pd.Series:
    s = np.sign(z.to_numpy(dtype=np.float64))
    a = np.log(np.abs(z.to_numpy(dtype=np.float64)) + 1.0)
    return pd.Series(s * a, index=z.index)


def _yang_zhang_vol_24(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 24
) -> pd.Series:
    """Rolling Yang–Zhang-style volatility (OHLC), sqrt variance per bar."""
    o = open_.astype(np.float64)
    h = high.astype(np.float64)
    l = low.astype(np.float64)
    c = close.astype(np.float64)
    log_ho = np.log((h / o).clip(lower=1e-12))
    log_lo = np.log((l / o).clip(lower=1e-12))
    log_co = np.log((c / o).clip(lower=1e-12))
    log_cc = np.log((c / c.shift(1)).clip(lower=1e-12))
    log_oo = np.log((o / c.shift(1)).clip(lower=1e-12))
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    w = float(window)
    k = 0.34 / (1.34 + (w + 1.0) / max(w - 1.0, 1.0))
    close_vol = log_cc.rolling(int(window), min_periods=max(12, int(window) // 4)).var()
    overnight_vol = log_oo.rolling(int(window), min_periods=max(12, int(window) // 4)).var()
    rs_mean = rs.rolling(int(window), min_periods=max(12, int(window) // 4)).mean()
    var_yz = overnight_vol + k * close_vol + (1.0 - k) * rs_mean
    return np.sqrt(var_yz.clip(lower=0.0))


def build_gp_input_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for GP SymbolicTransformer input.
    Enhanced with momentum, MA distances, volatility markers, and microstructure.
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    vol = df["volume"].astype(np.float64)
    open_ = df["open"].astype(np.float64) if "open" in df.columns else close.shift(1).fillna(close)

    # 1. Price Momentum (log-modulus of log returns; tmp.md 1-D stationarity)
    for h in [1, 3, 6, 12, 24]:
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
    for w in [24, 168]:
        vma = vol.rolling(window=w, min_periods=w // 4).mean()
        out[f"vol_ratio_{w}"] = (vol / (vma + 1e-12)).fillna(1.0)

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

    # vol_ratio / buy_sell_ratio: keep finite defaults; ret/ma_dist/funding left NaN for CS impute
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_hmm_input_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    GaussianHMM features designed for regime structural detection.

    Design (HMM improvement):
    - Log_Return → EMA-smoothed (removes high-frequency noise that collapses transition probs)
    - Vol_Ratio: short-term/long-term Parkinson volatility ratio (expansion vs contraction)
    - Volume_Momentum: 24h vs 168h MA ratio (trend vs mean-reversion regime marker)
    - Funding_Cum_Dev: cumulative deviation from rolling mean (macro regime shift indicator)

    Fat tails mitigated via winsorize in inferrer.
    """
    close = df["close"].astype(np.float64)
    log_ret = np.log(close / close.shift(1))

    out = pd.DataFrame(index=df.index)

    # 1. EMA-smoothed log return (span=12h removes tick noise)
    out["Log_Return_Smooth"] = log_ret.ewm(span=12, min_periods=5).mean()

    # 2. Short/long volatility ratio (Parkinson estimator)
    h_hi = df["high"].astype(np.float64)
    h_lo = df["low"].astype(np.float64) + 1e-12
    hl = np.log((h_hi / h_lo).clip(lower=1e-12))
    park = np.sqrt(np.maximum(1.0 / (4.0 * np.log(2.0)) * (hl**2), 0.0))
    pser = pd.Series(park, index=df.index)
    short_vol = pser.rolling(window=24, min_periods=5).mean()
    long_vol = pser.rolling(window=168, min_periods=20).mean()
    # Vol ratio > 1 = volatility expanding (trending/crisis), < 1 = compressing (range)
    # Numerical safety: replace Inf/NaN with 1.0 (neutral) to suppress warnings
    out["Vol_Ratio"] = (short_vol / (long_vol + 1e-12)).replace([np.inf, -np.inf], 1.0).fillna(1.0)

    # 3. Volume momentum: 24h vs 168h MA (buying pressure trend)
    vol = df["volume"].astype(np.float64)
    vol_ma_s = vol.rolling(24, min_periods=5).mean()
    vol_ma_l = vol.rolling(168, min_periods=20).mean()
    out["Volume_Momentum"] = (vol_ma_s / (vol_ma_l + 1e-12)) - 1.0

    # 4. Funding rate spread (Short vs Long mean) - Stationary
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].astype(np.float64).ffill().fillna(0.0)
        fr_short = fr.rolling(24, min_periods=5).mean().fillna(0.0)
        fr_long = fr.rolling(168, min_periods=20).mean().fillna(0.0)
        fr_spread = fr_short - fr_long
        roll_std = fr_spread.rolling(168, min_periods=20).std().fillna(1e-6)
        out["Funding_Spread_Z"] = (fr_spread / roll_std).clip(-3.0, 3.0)
    else:
        out["Funding_Spread_Z"] = 0.0

    return out


def build_systemic_hmm_features(panel_df: pd.DataFrame, alpha_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Builds single-timeseries features for Market HMM using panel aggregates and GP performance.
    """
    # 1. GP Alpha Long-Short Spread (Rolling IC proxy)
    # Ensure time-series integrity by unstacking before shift
    close_grid = panel_df["close"].unstack(level="symbol")
    # Force UTC aware index for grid safety
    if close_grid.index.tz is None:
        close_grid.index = close_grid.index.tz_localize("UTC")
    else:
        close_grid.index = close_grid.index.tz_convert("UTC")

    # 24h lag return (past-only): log(P_t / P_{t-24}) — no forward shift (avoid look-ahead)
    ret_lag_grid = np.log(close_grid / close_grid.shift(24).clip(lower=1e-12))
    
    alpha_grid = alpha_panel["gp_alpha_00"].unstack(level="symbol")
    if alpha_grid.index.tz is None:
        alpha_grid.index = alpha_grid.index.tz_localize("UTC")
    else:
        alpha_grid.index = alpha_grid.index.tz_convert("UTC")
    
    # Align both grids to the same index to avoid failures
    common_idx = alpha_grid.index.intersection(ret_lag_grid.index)
    alpha_grid = alpha_grid.reindex(common_idx)
    ret_lag_grid = ret_lag_grid.reindex(common_idx)
    
    # --- Vectorized LS Spread Calculation ---
    # 1. Rank symbols cross-sectionally at each time step
    ranks_grid = alpha_grid.rank(axis=1, pct=True, method='first')
    
    # 2. Identify Top and Bottom symbols
    # [FIX] For small universes, 20% (0.8/0.2) is too restrictive. 
    # Use 33% (0.67/0.33) to ensure we have at least one symbol in each bucket for 3-symbol test.
    n_syms = alpha_grid.shape[1]
    thr = 0.8 if n_syms >= 10 else 0.65
    bot_thr = 0.2 if n_syms >= 10 else 0.35
    
    top_mask = ranks_grid >= thr
    bot_mask = ranks_grid <= bot_thr
    
    top_mean_ret = ret_lag_grid[top_mask].mean(axis=1)
    bot_mean_ret = ret_lag_grid[bot_mask].mean(axis=1)
    
    ls_spread = (top_mean_ret - bot_mean_ret).fillna(0.0)
    
    # [DEBUG] Data Integrity Check
    import logging
    debug_logger = logging.getLogger("ml_pipeline")
    
    debug_logger.info(" [DEBUG] Grid Alignment: common_idx_len=%d", len(common_idx))
    debug_logger.info(" [DEBUG] LS Spread Stats: min=%.6f, max=%.6f, mean=%.6f, non_zero=%d", 
                 ls_spread.min(), ls_spread.max(), ls_spread.mean(), (ls_spread != 0).sum())
    
    # 2. Aggregated Market Features from panel_df
    # (already calculated by CrossSectionalPipelineUtils)
    # Since these are duplicated for all symbols, we just take the first per datetime
    market_feats = (
        panel_df[["cs_dispersion", "market_breadth"]]
        .groupby(level="datetime")
        .first()
    )
    
    # 3. BTC Volatility Ratio (Anchor Vol)
    has_btc = "BTCUSDT" in panel_df.index.get_level_values("symbol")
    btc_df = panel_df.xs("BTCUSDT", level="symbol") if has_btc else None
    if btc_df is not None:
        h_hi = btc_df["high"].astype(np.float64)
        h_lo = btc_df["low"].astype(np.float64) + 1e-12
        hl = np.log((h_hi / h_lo).clip(lower=1e-12))
        park = np.sqrt(np.maximum(1.0 / (4.0 * np.log(2.0)) * (hl**2), 0.0))
        btc_vol_ratio = (park.rolling(24).mean() / (park.rolling(168).mean() + 1e-12)).fillna(1.0)
    else:
        btc_vol_ratio = pd.Series(1.0, index=market_feats.index)
        
    out = pd.DataFrame(index=market_feats.index)
    # ewm might cause the entire series to be 0 if the first few are 0. Use rolling mean instead.
    out["GP_LS_Spread"] = ls_spread.rolling(12, min_periods=1).mean().fillna(0.0)
    out["CS_Dispersion"] = market_feats["cs_dispersion"]
    out["Market_Breadth"] = market_feats["market_breadth"]
    out["BTC_Vol_Ratio"] = btc_vol_ratio
    
    return out.fillna(0.0)
