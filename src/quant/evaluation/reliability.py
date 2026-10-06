from __future__ import annotations

import numpy as np


def derive_block_size(returns: np.ndarray) -> int:
    """Data-driven block length from the run's own trade-return autocorrelation.

    White noise (no significant lag) returns 1, reducing to a plain bootstrap.
    Never raises; tiny samples (n<10) return 1 rather than an unreliable ACF.
    """
    n = len(returns)
    if n < 10:
        return 1
    x = returns - returns.mean()
    denom = float(np.sum(x**2))
    if denom <= 0.0:
        return 1
    max_lag = min(20, n // 4)
    band = 1.96 / np.sqrt(n)
    block = 1
    for lag in range(1, max_lag + 1):
        acf = float(np.sum(x[:-lag] * x[lag:])) / denom
        if abs(acf) > band:
            block = lag
    return min(block, max(1, n // 5))
