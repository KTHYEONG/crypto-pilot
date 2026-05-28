"""Regime-conditional alpha exposure scaling (trailing BTC regime, no look-ahead)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.strategy.alpha_evaluation import _compute_regime_labels
from src.domain.futures.strategy.config import StrategyMLConfig

_logger = logging.getLogger(__name__)


def apply_regime_gate(
    alpha_long: NDArray[np.float32],
    alpha_short: NDArray[np.float32],
    datetimes: NDArray[np.datetime64],
    btc_close: pd.Series,
    cfg: StrategyMLConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Scale alpha by trailing BTC regime. Returns (scaled_long, scaled_short).

    Time Complexity: O(T) for regime label map + O(T*N) for scalar broadcast.
    Space Complexity: O(T) scalar array; original arrays are not mutated.

    Args:
        alpha_long: [T, N] float32 long alpha.
        alpha_short: [T, N] float32 short alpha.
        datetimes: [T] datetime64 array (rows of alpha arrays).
        btc_close: BTC close prices indexed by datetime (timezone-aware or naive).
        cfg: ML strategy config with regime_gate_enabled and exposure scalars.

    Returns:
        Regime-scaled copies; originals unchanged if gate disabled or data insufficient.

    """
    if not cfg.regime_gate_enabled:
        return alpha_long, alpha_short

    t_len: int = alpha_long.shape[0]

    if btc_close.empty:
        _logger.warning("[REGIME-GATE] BTC series is empty — gate bypassed")
        return alpha_long, alpha_short

    dt_idx = pd.to_datetime(datetimes)
    if btc_close.index.tz is not None:
        if dt_idx.tz is None:
            dt_idx = dt_idx.tz_localize("UTC")
        else:
            dt_idx = dt_idx.tz_convert(btc_close.index.tz)
    else:
        if dt_idx.tz is not None:
            dt_idx = dt_idx.tz_localize(None)

    btc_aligned = btc_close.reindex(dt_idx, method="ffill").bfill()
    btc_arr: NDArray[np.float64] = btc_aligned.to_numpy(dtype=np.float64)

    finite_count = int(np.sum(np.isfinite(btc_arr)))
    if finite_count < 30:
        _logger.warning(
            "[REGIME-GATE] insufficient BTC data (finite=%d < 30) — gate bypassed",
            finite_count,
        )
        return alpha_long, alpha_short

    labels: list[str | None] = _compute_regime_labels(btc_arr, trend_window=30)

    scalar_map: dict[str | None, float] = {
        "bull": float(cfg.regime_exposure_bull),
        "bear": float(cfg.regime_exposure_bear),
        "chop": float(cfg.regime_exposure_chop),
        None: 1.0,
    }

    # [T] — scalar per bar; fromiter avoids intermediate list allocation
    scalars: NDArray[np.float32] = np.fromiter(
        (scalar_map.get(lbl, 1.0) for lbl in labels),
        dtype=np.float32,
        count=t_len,
    )

    counts: dict[str | None, int] = {
        r: sum(1 for lbl in labels if lbl == r) for r in ("bull", "bear", "chop", None)
    }
    _logger.info(
        "[REGIME-GATE] applied: bull=%d bear=%d chop=%d unlabeled=%d",
        counts["bull"],
        counts["bear"],
        counts["chop"],
        counts[None],
    )

    # [T, 1] broadcast → [T, N]; copy=False reuses buffer when dtype already matches
    col_view: NDArray[np.float32] = scalars[:, np.newaxis]
    return (
        (alpha_long * col_view).astype(np.float32, copy=False),
        (alpha_short * col_view).astype(np.float32, copy=False),
    )
