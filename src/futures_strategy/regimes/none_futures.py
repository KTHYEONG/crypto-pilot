"""No regime adjustment: unit multipliers."""
from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.futures_strategy.regimes.registry import register_regime


@register_regime
class NoneFuturesRegime:
    name: ClassVar[str] = "NONE"
    param_space: ClassVar[Dict[str, Any]] = {}

    def compute_long_short_mult(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(df)
        ones = np.ones(n, dtype=np.float64)
        return ones, ones
