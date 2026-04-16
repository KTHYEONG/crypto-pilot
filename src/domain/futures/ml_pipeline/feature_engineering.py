"""GP / HMM input feature builders (1h OHLCV + funding)."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def build_gp_input_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for GP SymbolicTransformer input.
    Expects columns: close, volume, high, low, funding_rate (optional),
    taker_buy_quote_volume, quote_volume (optional).
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(np.float64)
    out["Close_Return_1h"] = close.pct_change(1)
    out["Volume_Log"] = np.log1p(df["volume"].astype(np.float64).clip(lower=0.0))
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].astype(np.float64)
    else:
        fr = pd.Series(0.0, index=df.index)
    out["Funding_Rate"] = fr.ffill().fillna(0.0)
    if "quote_volume" in df.columns:
        qv = df["quote_volume"].astype(np.float64)
    else:
        qv = close * df["volume"].astype(np.float64)
    tbq = (
        df["taker_buy_quote_volume"].astype(np.float64)
        if "taker_buy_quote_volume" in df.columns
        else qv * 0.5
    )
    out["Taker_Buy_Sell_Ratio"] = tbq / (qv + 1e-9)
    hl_spread = df["high"].astype(np.float64) - df["low"].astype(np.float64)
    out["High_Low_Spread_Norm"] = hl_spread / (close + 1e-9)
    z = _rolling_robust_z(close, window=2000)
    out["Close_RobustZ_LogMod"] = _log_modulus(z)
    return out


def build_hmm_input_features(df: pd.DataFrame) -> pd.DataFrame:
    """GaussianHMM features; fat tails mitigated via winsorize in inferrer."""
    close = df["close"].astype(np.float64)
    log_ret = np.log(close / close.shift(1))
    h_hi = df["high"].astype(np.float64)
    h_lo = df["low"].astype(np.float64) + 1e-12
    hl = np.log((h_hi / h_lo).clip(lower=1e-12))
    park = np.sqrt(np.maximum(1.0 / (4.0 * np.log(2.0)) * (hl**2), 0.0))
    pser = pd.Series(park, index=df.index)
    out = pd.DataFrame(index=df.index)
    out["Log_Return"] = log_ret
    out["Parkinson_Vol"] = pser.rolling(window=24, min_periods=5).mean()
    vol_z = _rolling_robust_z(df["volume"].astype(np.float64), window=500)
    out["Volume_Shock"] = vol_z
    if "funding_rate" in df.columns:
        fr2 = df["funding_rate"].astype(np.float64)
    else:
        fr2 = pd.Series(0.0, index=df.index)
    out["Funding_Rate_Raw"] = fr2.ffill().fillna(0.0)
    return out
