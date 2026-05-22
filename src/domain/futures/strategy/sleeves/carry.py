from __future__ import annotations

import numpy as np


class CarrySleeve:
    """Funding rate carry trade sleeve.

    Captures the carry edge in crypto markets by going long when funding rate
    is negative (receives funding) and shorting when funding rate is highly positive.
    """

    def __init__(self, smooth_bars: int = 6) -> None:
        """Initializes the sleeve with smooth span."""
        if smooth_bars < 1:
            raise ValueError("smooth_bars must be >= 1")
        self.name: str = f"carry_{smooth_bars}"
        self.smooth_bars: int = smooth_bars

    def compute_raw(
        self,
        close_2d: np.ndarray,
        aux: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Computes carry signal: sig[t, i] = -EWMA(funding[t, i], smooth_bars).

        Args:
            close_2d: [T, N] prices array.
            aux: Dictionary containing 'funding_2d' of shape [T, N].

        Returns:
            [T, N] signals. Warmup period has no special NaN mask except standard EWMA initialization.

        """
        if "funding_2d" not in aux:
            raise KeyError("aux dictionary must contain 'funding_2d' for CarrySleeve")

        funding = aux["funding_2d"]
        if funding.shape != close_2d.shape:
            raise ValueError("funding_2d and close_2d must have the same shape")

        t_len, n_syms = funding.shape
        sig = np.zeros((t_len, n_syms), dtype=np.float64)

        if t_len == 0:
            return sig

        # Compute EWMA on funding panel
        alpha = 2.0 / (self.smooth_bars + 1.0)

        # Initialize the first row
        init_row = funding[0].copy()
        init_row[~np.isfinite(init_row)] = 0.0
        sig[0] = init_row

        for t in range(1, t_len):
            val = funding[t]
            prev = sig[t - 1]
            mask = ~np.isfinite(val)
            # If NaN/Inf, carry forward the previous EWMA state, otherwise apply EWMA step
            sig[t] = np.where(mask, prev, alpha * val + (1.0 - alpha) * prev)

        # Carry signal is -EWMA(funding)
        return -sig
