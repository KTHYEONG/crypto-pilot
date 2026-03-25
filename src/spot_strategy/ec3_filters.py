"""EC-3 asymmetric persistence: immediate exit (M=1), entry after M=2 consecutive confirmations."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def apply_ec3_persistence(raw_long: np.ndarray, m_entry: int = 2) -> np.ndarray:
    """
    raw_long: 1.0 = raw bullish signal, 0.0 = no signal.
    Exit: first bar where raw is false -> output 0 immediately.
    Entry: require m_entry consecutive raw True bars before output becomes 1.
    """
    n = len(raw_long)
    out = np.zeros(n, dtype=np.float64)
    streak = 0
    m = max(1, int(m_entry))
    for i in range(n):
        if raw_long[i] > 0.5:
            streak += 1
            if streak >= m:
                out[i] = 1.0
            else:
                out[i] = 0.0
        else:
            streak = 0
            out[i] = 0.0
    return out
