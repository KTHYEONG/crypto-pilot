from __future__ import annotations

from typing import Any, ClassVar, Protocol

import numpy as np


class IRegime(Protocol):
    name: ClassVar[str]
    param_space: ClassVar[dict[str, Any]]

    def compute(
        self, data_maps: dict[str, dict[str, Any]], params: dict[str, Any]
    ) -> np.ndarray: ...
