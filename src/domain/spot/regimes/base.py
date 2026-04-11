from __future__ import annotations

from typing import Any, ClassVar, Dict, Protocol

import numpy as np


class IRegime(Protocol):
    name: ClassVar[str]
    param_space: ClassVar[Dict[str, Any]]

    def compute(
        self, data_maps: Dict[str, Dict[str, Any]], params: Dict[str, Any]
    ) -> np.ndarray: ...
