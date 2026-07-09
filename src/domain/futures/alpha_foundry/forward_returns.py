"""Shared causal forward-return SSOT for Alpha Foundry L0 gating.

[ADR_20260709_L0_TREND_PULLBACK_HARDENING_SYNC]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_causal_forward_returns_bps(
    *,
    close: NDArray[np.float64],
    side: NDArray[np.int8],
    causal_lag_bars: int,
    holding_bars: int,
) -> NDArray[np.float64]:
    """Compute causal forward returns in bps from the actual entry bar.

    [ADR_20260709_L0_TREND_PULLBACK_HARDENING_SYNC]
    """
    if causal_lag_bars < 0:
        raise ValueError(f"causal_lag_bars must be >= 0, got {causal_lag_bars}")
    if holding_bars <= 0:
        raise ValueError(f"holding_bars must be > 0, got {holding_bars}")

    t, n = close.shape
    fwd_ret = np.full((t, n), np.nan, dtype=np.float64)
    idx_end = t - causal_lag_bars - holding_bars
    if idx_end <= 0:
        return fwd_ret

    for i in range(idx_end):
        entry = i + causal_lag_bars
        exit_bar = entry + holding_bars
        fwd_ret[i, :] = (
            side[i, :].astype(np.float64)
            * np.log(close[exit_bar, :] / close[entry, :])
            * 10000.0
        )
    return fwd_ret
