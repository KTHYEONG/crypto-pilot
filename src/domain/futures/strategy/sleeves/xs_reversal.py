from __future__ import annotations

from typing import Any
import numpy as np


class XSReversalSleeve:
    """Short-term cross-sectional reversal sleeve.

    Captures short-term mean-reversion by shorting short-term outperformers
    and buying short-term underperformers.
    """

    def __init__(self, lookback_bars: int = 6) -> None:
        """Initializes the sleeve with lookback parameters."""
        if lookback_bars < 1:
            raise ValueError("lookback_bars must be >= 1")
        self.name: str = f"xs_reversal_{lookback_bars}"
        self.lookback_bars: int = lookback_bars

    def compute_raw(
        self,
        close_2d: np.ndarray,
        aux: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Computes reversal signal: sig[t, i] = -log(close[t, i] / close[t-lookback, i]).

        Args:
            close_2d: [T, N] prices array.
            aux: Auxiliary dictionary (unused here).

        Returns:
            [T, N] signals. Warmup rows are NaN.
        """
        del aux
        if close_2d.ndim != 2:
            raise ValueError("close_2d must be 2D")

        t_len, n_syms = close_2d.shape
        sig = np.full((t_len, n_syms), np.nan, dtype=np.float64)

        if t_len <= self.lookback_bars:
            return sig

        prev = close_2d[:-self.lookback_bars]
        curr = close_2d[self.lookback_bars:]

        with np.errstate(divide="ignore", invalid="ignore"):
            # Prevent zero or negative prices
            safe_prev = np.maximum(prev, 1e-12)
            safe_curr = np.maximum(curr, 1e-12)
            raw_ret = np.log(safe_curr / safe_prev)

        # Reversal: sell recent gainers, buy recent losers
        sig[self.lookback_bars :] = -raw_ret
        return sig
