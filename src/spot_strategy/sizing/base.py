from __future__ import annotations

from typing import Any, ClassVar, Dict, Protocol

import numpy as np
import pandas as pd


class ISizing(Protocol):
    name: ClassVar[str]
    param_space: ClassVar[Dict[str, Any]]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray: ...
