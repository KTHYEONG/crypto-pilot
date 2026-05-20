from __future__ import annotations

from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd


class ISizing(Protocol):
    name: ClassVar[str]
    param_space: ClassVar[dict[str, Any]]

    def compute(self, df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray: ...
