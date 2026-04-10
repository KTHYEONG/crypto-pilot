from __future__ import annotations

from types import SimpleNamespace
import pandas as pd

from .core import StrategyBase
from src.core.indicators.indicators import IndicatorEngine


class PipelineStrategyBase(StrategyBase):
    """
    Base class for plugin-based strategy pipelines.
    Domain-specific strategies (Spot/Futures) must override generate_signals.
    """

    INDICATORS: SimpleNamespace | IndicatorEngine | None = None
    ENTRY_SHIFT: bool = False

    def _ind(self) -> SimpleNamespace | IndicatorEngine:
        if self.INDICATORS is None:
            raise RuntimeError("INDICATORS bindings are not configured.")
        return self.INDICATORS

    def _shift_if_needed(self, series: pd.Series) -> pd.Series:
        return series.shift(1) if self.ENTRY_SHIFT else series

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Plugin-based signal generation pipeline.
        Must be implemented by subclasses using REGISTRY objects.
        """
        raise NotImplementedError("Subclasses must implement generate_signals using their respective registries.")
