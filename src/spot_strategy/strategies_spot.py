import numpy as np
import pandas as pd
from types import SimpleNamespace
from src.strategy.base.ultimate import UltimateStrategyBase
from src.futures_strategy.strategies_futures import _FUTURES_INDICATORS as _SPOT_INDICATORS

class UltimateSpotStrategy(UltimateStrategyBase):
    """
    Spot-Specific Trend-Following Strategy.
    Uses Dual EMA + ADX to filter out bear market fakeouts.
    Long-Only execution.
    """
    INDICATORS = _SPOT_INDICATORS
    ENTRY_SHIFT = False

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)
                
        ind = self._ind()
        
        # Spot Parameters
        macro_ema_period = int(self.params.get("MACRO_EMA_PERIOD", 200))
        fast_ema_period = int(self.params.get("FAST_EMA_PERIOD", 50))
        adx_period = int(self.params.get("ADX_PERIOD", 14))
        adx_threshold = float(self.params.get("ADX_THRESHOLD", 25.0))
        momentum_period = int(self.params.get("MOMENTUM_PERIOD", 20))
        kc_mult = float(self.params.get("KC_MULT", 1.5))
        atr_period = int(self.params.get("ATR_PERIOD", 20))
        
        # Core
        df["atr"] = ind.calculate_atr(df, window=atr_period)
        
        # Macro Dual-Trend
        df["macro_ema"] = ind.calculate_ema(df["close"], window=macro_ema_period)
        df["fast_ema"] = ind.calculate_ema(df["close"], window=fast_ema_period)
        
        # ADX for Trend Strength
        df["adx"] = ind.calculate_adx(df, window=adx_period)
        
        # Breakout (Donchian Channel - more sensitive than KC)
        # shift(1) is used because we want to breakout of the *previous* N bars high
        df["dc_upper"] = df["high"].rolling(window=momentum_period).max().shift(1)
        
        # Dual Alignment & ADX filter (Long Only)
        bull_alignment = (df["fast_ema"] > df["macro_ema"]) & (df["close"] > df["fast_ema"])
        strong_trend = df["adx"] > adx_threshold
        
        # Spot Entry: Relaxed. No Squeeze required.
        # If we are in a strong bull alignment, and price breaks recent momentum high.
        bull_breakout = bull_alignment & strong_trend & (df["close"] > df["dc_upper"])
        
        df["strength_filter"] = np.where(bull_breakout, 1, 0)
        df["trend_direction"] = np.where(bull_breakout, 1, 0)
        
        # entry_upper triggers when high > entry_upper. Setting to 0.0 forces immediate entry check on next open.
        df["entry_upper"] = np.where(bull_breakout, 0.0, 999999.0)
        df["entry_lower"] = 999999.0 # Short is impossible in spot
        
        df["entry_upper"] = self._shift_if_needed(df["entry_upper"])
        df["entry_lower"] = self._shift_if_needed(df["entry_lower"])
        
        df["atr"] = df["atr"].ffill().fillna(df["close"] * 0.01)
        return df
