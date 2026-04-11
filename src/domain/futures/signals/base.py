"""Futures signal plugins: bidirectional entries + kill strengths + rank score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FuturesSignalOutput:
    long_entry: np.ndarray
    short_entry: np.ndarray
    kill_long: np.ndarray
    kill_short: np.ndarray
    rank_score: np.ndarray


class IFuturesSignal(Protocol):
    name: ClassVar[str]
    param_space: ClassVar[Dict[str, Any]]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FuturesSignalOutput: ...
