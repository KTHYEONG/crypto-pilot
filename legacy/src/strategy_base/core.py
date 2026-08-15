from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class StrategyBase(ABC):
    def __init__(self, name: str, params: dict[str, Any]) -> None:
        self.name = name
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def get_required_warmup(self, freq: str = "daily") -> int:
        """Return required warmup bar count in the requested timeframe.
        freq='daily': daily-bar count (for strategies that run on daily data).
        freq='1h' or 'hourly': hourly-bar count (daily_bars * 24).
        freq='4h': 4h-bar count (daily_bars * 6).
        """
        daily_bars = self._compute_warmup_bars()
        if freq in ("hourly", "1h"):
            return daily_bars * 24
        elif freq == "4h":
            return daily_bars * 6
        return daily_bars

    @abstractmethod
    def _compute_warmup_bars(self) -> int:
        """Daily-bar warmup requirement. Must be implemented by subclasses."""
        raise NotImplementedError


def calculate_required_warmup_bars(
    params: dict[str, Any],
    *,
    min_bars: int = 300,
    safety_factor: int = 3,
) -> int:
    """Generic helper to estimate required warmup bars based on any key ending in '_PERIOD' or '_WINDOW'.
    This removes hardcoded dependencies on specific indicator names (ADX, HMA, etc.).
    """
    max_period = 0
    for key, value in params.items():
        if not isinstance(value, (int, float)):
            continue
        
        # Heuristic: any parameter ending in PERIOD, WINDOW, or SLOW/FAST is likely a lookback
        k_upper = key.upper()
        if any(suffix in k_upper for suffix in ("_PERIOD", "_WINDOW", "_SLOW", "_FAST", "LOOKBACK")):
            max_period = max(max_period, int(value))
            
    return max(int(max_period * safety_factor), int(min_bars))


class MasterStrategyBase(StrategyBase):
    """Abstract orchestration-tier strategy marker (Spot & Futures).

    Subclasses define all signal/indicate logic via ``generate_signals``;
    no default HMA/ADX or other rule presets are injected here.
    """

    pass
