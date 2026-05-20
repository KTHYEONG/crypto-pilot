from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalOutput:
    entry_signal: np.ndarray
    kill_signal: np.ndarray
    # Higher rank_score => higher priority among concurrent entry candidates (shared-cash).
    rank_score: np.ndarray


class ISignal(Protocol):
    name: ClassVar[str]
    param_space: ClassVar[dict[str, Any]]

    def compute(self, df: pd.DataFrame, params: dict[str, Any]) -> SignalOutput: ...
