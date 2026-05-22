from __future__ import annotations

from typing import Protocol
import numpy as np


class Sleeve(Protocol):
    """Sleeve protocol that all strategy sleeves must implement."""

    name: str

    def compute_raw(
        self,
        close_2d: np.ndarray,
        aux: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Computes raw directional signals of shape [T, N].

        Args:
            close_2d: [T, N] price panel.
            aux: Auxiliary panels including volume_2d, funding_2d, etc.

        Returns:
            [T, N] directional signals. Positive is bullish, negative is bearish.
            Uninitialized / warmup periods should return NaN.
        """
        ...
