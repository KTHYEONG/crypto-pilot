from __future__ import annotations

import numpy as np


class TSMomentumSleeve:
    """Time-series momentum sleeve.

    Captures trend following behaviour by checking an asset's own log return over
    a lookback window, skipping recent bars to avoid short-term mean-reversion.
    """

    def __init__(self, lookback_bars: int = 18, skip_bars: int = 1) -> None:
        """Initializes the sleeve with lookback and gap skip parameters."""
        if lookback_bars < 1:
            raise ValueError("lookback_bars must be >= 1")
        if skip_bars < 0:
            raise ValueError("skip_bars must be >= 0")
        self.name: str = f"ts_momentum_{lookback_bars}_skip_{skip_bars}"
        self.lookback_bars: int = lookback_bars
        self.skip_bars: int = skip_bars

    def compute_raw(
        self,
        close_2d: np.ndarray,
        aux: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Computes momentum signal: sig[t, i] = log(close[t-skip, i] / close[t-skip-lookback, i]).

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

        total_lag = self.lookback_bars + self.skip_bars
        if t_len <= total_lag:
            return sig

        prev = close_2d[:-total_lag]
        curr = close_2d[self.lookback_bars : -self.skip_bars] if self.skip_bars > 0 else close_2d[self.lookback_bars :]

        with np.errstate(divide="ignore", invalid="ignore"):
            safe_prev = np.maximum(prev, 1e-12)
            safe_curr = np.maximum(curr, 1e-12)
            mom = np.log(safe_curr / safe_prev)

        sig[total_lag:] = mom
        return sig
