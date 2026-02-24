from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class StrategyBase(ABC):
    def __init__(self, name: str, params: dict[str, Any]) -> None:
        self.name = name
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def get_required_warmup(self, freq: str = "daily") -> int:
        """
        Return required warmup bar count in the requested timeframe.
        freq='daily': daily-bar count (for strategies that run on daily data).
        freq='hourly': hourly-bar count so that warmup duration matches daily (daily_bars * 24).
        """
        daily_bars = self._compute_warmup_bars()
        if freq == "hourly":
            return daily_bars * 24
        return daily_bars

    def _compute_warmup_bars(self) -> int:
        """Daily-bar warmup from strategy params (used by get_required_warmup)."""
        return calculate_required_warmup_bars(self.params)


def calculate_required_warmup_bars(
    params: dict[str, Any],
    *,
    min_bars: int = 50,
    safety_factor: int = 3,
) -> int:
    period_keys = (
        "ENTRY_PERIOD",
        "MA_PERIOD",
        "ATR_PERIOD",
        "SUPERTREND_PERIOD",
        "MACD_SLOW",
        "ICHIMOKU_SENKOU_B",
        "STRENGTH_FILTER_PERIOD",
        "VOLUME_MA_PERIOD",
        "CMF_PERIOD",
        "HURST_PERIOD",
    )
    max_period = 0
    for key in period_keys:
        value = params.get(key)
        if isinstance(value, (int, float)):
            max_period = max(max_period, int(value))
    return max(int(max_period * safety_factor), int(min_bars))


class MasterStrategyBase(StrategyBase):
    """
    Legacy compatibility strategy.
    - Builds regime line
    - Builds ADX pass/fail filter
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        filter_type = self.params.get("REGIME_FILTER", "EMA")

        if filter_type == "HMA" and "hma" in df.columns:
            df["regime_line"] = df["hma"]
        elif "ema_trend" in df.columns:
            df["regime_line"] = df["ema_trend"]
        else:
            df["regime_line"] = df["ma50"]

        if self.params.get("USE_ADX", False) and "adx" in df.columns:
            adx_threshold = self.params.get("ADX_THRESHOLD", 20)
            df["adx_filter"] = np.where(df["adx"] > adx_threshold, 1, 0)
        else:
            df["adx_filter"] = 1

        return df
