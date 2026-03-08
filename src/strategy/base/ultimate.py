from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from .core import StrategyBase


@njit(cache=True)
def _rolling_last_rank_numba(values: np.ndarray, window: int) -> np.ndarray:
    out = np.empty(values.shape[0], dtype=np.float64)
    out[:] = np.nan
    if window <= 0:
        return out
    for i in range(window - 1, values.shape[0]):
        last_val = values[i]
        if np.isnan(last_val):
            continue
        count = 0
        valid = 0
        start = i - window + 1
        for j in range(start, i + 1):
            v = values[j]
            if np.isnan(v):
                continue
            valid += 1
            if last_val >= v:
                count += 1
        if valid > 0:
            out[i] = count / valid
    return out


@njit(cache=True)
def _rolling_r2_numba(values: np.ndarray, window: int) -> np.ndarray:
    out = np.empty(values.shape[0], dtype=np.float64)
    out[:] = np.nan
    if window < 5: return out
    
    x = np.arange(window, dtype=np.float64)
    x_mean = (window - 1) / 2.0
    ss_xx = np.sum((x - x_mean)**2)
    
    for i in range(window - 1, values.shape[0]):
        y = values[i - window + 1 : i + 1]
        if np.isnan(y).any(): continue
        
        y_mean = np.mean(y)
        ss_yy = np.sum((y - y_mean)**2)
        if ss_yy <= 0:
            out[i] = 0.0
            continue
            
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        r2 = (ss_xy**2) / (ss_xx * ss_yy)
        out[i] = r2
    return out


@njit(cache=True)
def _rolling_t_stat_numba(values: np.ndarray, window: int) -> np.ndarray:
    out = np.empty(values.shape[0], dtype=np.float64)
    out[:] = np.nan
    if window < 3:
        return out

    x_sum = window * (window - 1) / 2.0
    sxx = window * (window**2 - 1) / 12.0
    sqrt_term = np.sqrt(window - 2.0)

    for i in range(window - 1, values.shape[0]):
        sum_y = 0.0
        sum_y2 = 0.0
        sum_xy = 0.0
        start = i - window + 1
        valid = True
        for k in range(window):
            y = values[start + k]
            if np.isnan(y):
                valid = False
                break
            sum_y += y
            sum_y2 += y * y
            sum_xy += k * y
        if not valid:
            continue

        sxy = sum_xy - (x_sum * sum_y) / window
        syy = sum_y2 - (sum_y * sum_y) / window
        denom = sxx * syy - (sxy * sxy)
        if syy == 0.0 or denom <= 0.0:
            out[i] = 0.0
        else:
            out[i] = sxy * sqrt_term / np.sqrt(denom)

    return out


class UltimateStrategyBase(StrategyBase):
    """
    Shared signal-generation pipeline for Spot/Futures.
    Pure TSMOM (Time-Series Momentum) Z-Score Strategy.
    """

    INDICATORS: SimpleNamespace | None = None
    ENTRY_SHIFT: bool = False  # Engine already uses prev_i (i-1), so shift here causes 2-bar delay

    def _ind(self) -> SimpleNamespace:
        if self.INDICATORS is None:
            raise RuntimeError("INDICATORS bindings are not configured.")
        return self.INDICATORS

    def _shift_if_needed(self, series: pd.Series) -> pd.Series:
        return series.shift(1) if self.ENTRY_SHIFT else series

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # [ROBUSTNESS] Ensure all price columns are float64 for TA-Lib compatibility
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)
                
        ind = self._ind()
        
        # --- 1. RSM-VT Parameters ---
        # 4h parameters directly mapped
        macro_ema_period = int(self.params.get("MACRO_EMA_PERIOD", 200))
        momentum_period = int(self.params.get("MOMENTUM_PERIOD", 20))
        kc_mult = float(self.params.get("KC_MULT", 1.5))
        atr_period = int(self.params.get("ATR_PERIOD", 20))
        
        # --- 2. Core Indicators Calculation ---
        df["atr"] = ind.calculate_atr(df, window=atr_period)
        
        # Macro Trend
        df["macro_ema"] = ind.calculate_ema(df["close"], window=macro_ema_period)
        
        # Bollinger Bands (20, 2.0)
        df["bb_mid"] = ind.calculate_sma(df["close"], window=20)
        std_dev = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_mid"] + (std_dev * 2.0)
        df["bb_lower"] = df["bb_mid"] - (std_dev * 2.0)
        
        # Keltner Channels (20, KC_MULT)
        df["kc_mid"] = ind.calculate_ema(df["close"], window=20)
        df["kc_upper"] = df["kc_mid"] + (df["atr"] * kc_mult)
        df["kc_lower"] = df["kc_mid"] - (df["atr"] * kc_mult)
        
        # Squeeze Condition: BB is entirely inside KC
        df["is_squeezing"] = (df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])
        # Has it squeezed recently? (In the last 5 bars)
        df["recent_squeeze"] = df["is_squeezing"].rolling(window=5).sum() > 0
        
        # Volume Spike
        df["vol_sma"] = df["volume"].rolling(window=20).mean()
        vol_spike = df["volume"] > df["vol_sma"] * 1.2
        
        # Breakout Channels
        df["dc_upper"] = df["high"].rolling(window=momentum_period).max()
        df["dc_lower"] = df["low"].rolling(window=momentum_period).min()
        
        # --- 3. Signal Generation (Squeeze Breakout) ---
        macro_uptrend = df["close"] > df["macro_ema"]
        macro_downtrend = df["close"] < df["macro_ema"]
        
        # Bull: Squeezed recently, Macro is UP, Price closes above KC Upper, Vol Spike
        bull_breakout = df["recent_squeeze"] & macro_uptrend & (df["close"] > df["kc_upper"]) & vol_spike
        
        # Bear: Squeezed recently, Macro is DOWN, Price closes below KC Lower, Vol Spike
        bear_breakout = df["recent_squeeze"] & macro_downtrend & (df["close"] < df["kc_lower"]) & vol_spike
        
        # For Numba engine execution:
        df["strength_filter"] = np.where(bull_breakout | bear_breakout, 1, 0)
        df["trend_direction"] = np.where(bull_breakout, 1, np.where(bear_breakout, -1, 0))
        
        # [HACK FOR ENGINE] 
        # By setting entry_upper to 0.0 when confirmed, `c_high > 0.0` triggers Open entry.
        df["entry_upper"] = np.where(bull_breakout, 0.0, 999999.0)
        df["entry_lower"] = np.where(bear_breakout, 999999.0, 0.0)
        
        # Shift so the engine reads the signal from the previous closed bar
        df["entry_upper"] = self._shift_if_needed(df["entry_upper"])
        df["entry_lower"] = self._shift_if_needed(df["entry_lower"])
        
        df["atr"] = df["atr"].ffill().fillna(df["close"] * 0.01)
        return df
