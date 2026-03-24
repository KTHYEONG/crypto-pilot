from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.base.ultimate import UltimateStrategyBase
from src.futures_strategy.strategies_futures import _FUTURES_INDICATORS as _SPOT_INDICATORS


class UltimateSpotStrategy(UltimateStrategyBase):
    """
    Spot trend-following: EMA trend filter + ADX whipsaw filter + Donchian breakout.
    No squeeze/KC/volume-z regime layer (reduced search dimensionality).
    """

    INDICATORS = _SPOT_INDICATORS
    ENTRY_SHIFT = False

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = df[col].astype(np.float64)

        if "btc_close" not in df.columns:
            df["btc_close"] = df["close"].astype(np.float64)

        ind = self._ind()

        macro_ema_period = int(self.params.get("MACRO_EMA_PERIOD", 200))
        adx_period = int(self.params.get("ADX_PERIOD", 14))
        adx_threshold = float(self.params.get("ADX_THRESHOLD", 25.0))
        momentum_period = int(self.params.get("MOMENTUM_PERIOD", 20))
        atr_period = int(self.params.get("ATR_PERIOD", 20))

        df["atr"] = ind.calculate_atr(df, window=atr_period)
        df["macro_ema"] = ind.calculate_ema(df["close"], window=macro_ema_period)
        df["adx"] = ind.calculate_adx(df, window=adx_period)

        # Prior N-bar high (exclude current bar): no look-ahead
        df["dc_upper"] = df["high"].rolling(window=momentum_period).max().shift(1)

        bull_alignment = df["close"] > df["macro_ema"]
        strong_trend = df["adx"] >= adx_threshold
        donchian_break = df["close"] > df["dc_upper"]
        bull_breakout = bull_alignment & strong_trend & donchian_break

        df["long_entry_signal"] = np.where(bull_breakout, 1.0, 0.0)
        df["sig_long_entry_signal"] = df["long_entry_signal"]

        df["entry_upper"] = np.where(bull_breakout, df["close"], 999999.0)
        df["entry_lower"] = 999999.0

        df["strength_filter"] = np.where(bull_breakout, 1, 0)
        df["trend_direction"] = np.where(bull_breakout, 1, 0)

        rs_num = df["close"] / df["close"].shift(20).replace(0.0, np.nan)
        if "btc_close" in df.columns:
            rs_den = (
                df["btc_close"] / df["btc_close"].shift(20).replace(0.0, np.nan)
            ).replace(0.0, np.nan)
            rs_ratio = (rs_num / rs_den).replace([np.inf, -np.inf], np.nan)
        else:
            rs_ratio = rs_num
        rs_score = np.log(rs_ratio.clip(lower=1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rs_score = np.clip(rs_score, -3.0, 3.0)

        break_score = ((df["close"] - df["dc_upper"]) / (df["atr"] + 1e-9)).clip(-6.0, 6.0)

        df["slot_rank_score"] = (0.5 * rs_score + 0.5 * break_score).astype(np.float64)

        df["entry_upper"] = self._shift_if_needed(df["entry_upper"])
        df["entry_lower"] = self._shift_if_needed(df["entry_lower"])

        df["atr"] = df["atr"].ffill().fillna(df["close"] * 0.01)
        return df
