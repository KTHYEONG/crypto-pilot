from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.strategy.common.alignment import AlignedMarketData

_REGIME_NAMES = (
    "bull_quiet",
    "bull_volatile",
    "bear_quiet",
    "bear_volatile",
    "transition",
    "crash",
)

# 4-state regime names: coarser, more robust for sizing multipliers.
_REGIME_NAMES_4STATE = (
    "trend_up",
    "trend_up",   # bull_volatile -> trend_up
    "trend_down",
    "trend_down",  # bear_volatile -> trend_down
    "chop",        # transition -> chop
    "crisis",      # crash -> crisis
)

# Mapping from 6-state code to 4-state name (index matches _REGIME_NAMES).
_6STATE_TO_4STATE: dict[str, str] = {
    "bull_quiet": "trend_up",
    "bull_volatile": "trend_up",
    "bear_quiet": "trend_down",
    "bear_volatile": "trend_down",
    "transition": "chop",
    "crash": "crisis",
}


def _ema_1d(values: NDArray[np.float64], span: int) -> NDArray[np.float64]:
    alpha = 2.0 / (float(span) + 1.0)
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for idx in range(1, values.shape[0]):
        cur = values[idx]
        prev = out[idx - 1]
        out[idx] = cur if not np.isfinite(prev) else (alpha * cur) + ((1.0 - alpha) * prev)
    return out


def _rolling_mean_1d(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for idx in range(values.shape[0]):
        start = max(0, idx - window + 1)
        window_values = values[start : idx + 1]
        finite = window_values[np.isfinite(window_values)]
        if finite.size > 0:
            out[idx] = float(np.mean(finite))
    return out


def _rolling_std_1d(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for idx in range(values.shape[0]):
        start = max(0, idx - window + 1)
        window_values = values[start : idx + 1]
        finite = window_values[np.isfinite(window_values)]
        if finite.size > 0:
            out[idx] = float(np.std(finite, ddof=0))
    return out


def _zscore_1d(values: NDArray[np.float64], window: int, eps: float = 1e-12) -> NDArray[np.float64]:
    mean = _rolling_mean_1d(values, window)
    std = _rolling_std_1d(values, window)
    return (values - mean) / np.maximum(std, eps)


@dataclass(slots=True, frozen=True)
class MarketRegimeContext:
    code_1d: NDArray[np.int8]
    name_by_code: tuple[str, ...]
    trend_score_1d: NDArray[np.float64]
    vol_z_1d: NDArray[np.float64]
    dispersion_z_1d: NDArray[np.float64]

    def names(self) -> NDArray[np.object_]:
        return np.asarray([self.name_by_code[int(code)] for code in self.code_1d], dtype=object)


def compute_market_regime_context(*, aligned: AlignedMarketData) -> MarketRegimeContext:
    close = np.asarray(aligned.close_2d, dtype=np.float64)
    if close.ndim != 2 or close.shape[0] == 0:
        raise ValueError("aligned.close_2d must be non-empty 2D array")

    btc_idx = 0
    for idx, symbol in enumerate(aligned.symbols):
        if "BTC" in symbol.upper():
            btc_idx = idx
            break

    btc_close = np.maximum(close[:, btc_idx], 1e-12)
    ema_fast = _ema_1d(btc_close, span=20)
    ema_slow = _ema_1d(btc_close, span=100)
    trend_score = (ema_fast / np.maximum(ema_slow, 1e-12)) - 1.0

    log_ret = np.zeros_like(close, dtype=np.float64)
    log_ret[1:] = np.diff(np.log(np.maximum(close, 1e-12)), axis=0)
    mean_log_ret = np.nanmean(log_ret, axis=1)
    dispersion = np.nanstd(log_ret, axis=1, ddof=0)
    vol_20 = _rolling_std_1d(mean_log_ret, window=20)
    vol_z = _zscore_1d(vol_20, window=120)
    dispersion_z = _zscore_1d(dispersion, window=120)

    code = np.full(close.shape[0], 0, dtype=np.int8)
    bull_mask = trend_score >= 0.0
    high_vol_mask = vol_z > 0.5

    code[~bull_mask & ~high_vol_mask] = 2
    code[~bull_mask & high_vol_mask] = 3
    code[bull_mask & high_vol_mask] = 1

    transition_mask = (~np.isfinite(trend_score)) | (np.abs(trend_score) < 0.002)
    crash_mask = (vol_z > 2.0) & (dispersion_z > 1.0)
    code[transition_mask] = 4
    code[crash_mask] = 5

    return MarketRegimeContext(
        code_1d=code,
        name_by_code=_REGIME_NAMES,
        trend_score_1d=trend_score,
        vol_z_1d=vol_z,
        dispersion_z_1d=dispersion_z,
    )


def compute_market_regime_context_4state(*, aligned: AlignedMarketData) -> MarketRegimeContext:
    """Return a 4-state coarsened regime context.

    Maps the 6-state output of :func:`compute_market_regime_context` to 4 states:
    - ``trend_up``  : bull_quiet | bull_volatile
    - ``trend_down``: bear_quiet | bear_volatile
    - ``chop``      : transition
    - ``crisis``    : crash

    The coarser representation improves robustness and is intended for use as a
    sizing multiplier layer rather than a signal gate.

    Args:
        aligned: Aligned market data with at least a BTC symbol.

    Returns:
        :class:`MarketRegimeContext` with 4-state codes.  ``name_by_code`` is
        a length-6 tuple that maps the same 0-5 integer codes to 4-state names,
        so downstream consumers can keep the same indexing logic.
    """
    ctx_6 = compute_market_regime_context(aligned=aligned)
    return MarketRegimeContext(
        code_1d=ctx_6.code_1d,
        name_by_code=_REGIME_NAMES_4STATE,
        trend_score_1d=ctx_6.trend_score_1d,
        vol_z_1d=ctx_6.vol_z_1d,
        dispersion_z_1d=ctx_6.dispersion_z_1d,
    )
