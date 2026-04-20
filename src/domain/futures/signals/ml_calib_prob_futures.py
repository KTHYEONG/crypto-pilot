"""ML calibrated probability gate: dual Long/Short directional signals.

Design (Q3 decision):
  - Phase 2: rolling quantile gate (window=ENTRY_QUANTILE_WINDOW, q=ENTRY_THRESHOLD ∈ [0.8,0.98])
  - long_entry / short_entry use past-only quantile vs raw probabilities.
  - Portfolio engine uses ml_calib_prob = gate(max(long, short)); numba compares to ~0.5.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, cast

import numba
import numpy as np
import pandas as pd

from src.domain.futures.signals.base import FuturesSignalOutput
from src.domain.futures.signals.registry import register_futures_signal


@numba.njit(cache=True)
def _numba_rolling_quantile_gate(raw: np.ndarray, q: float, window: int) -> np.ndarray:
    n = raw.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    w = max(3, int(window))
    mp = max(3, min(max(20, w // 6), w - 1))
    buf = np.empty(w, dtype=np.float64)

    for i in range(n):
        if i == 0:
            out[i] = 0.0
            continue

        start_idx = max(0, i - w)
        count = i - start_idx

        if count < mp:
            out[i] = 0.0
            continue

        valid_count = 0
        for j in range(count):
            v = raw[start_idx + j]
            if not np.isnan(v):
                buf[valid_count] = v
                valid_count += 1

        if valid_count < mp:
            out[i] = 0.0
            continue

        valid_buf = buf[:valid_count]
        valid_buf.sort()

        idx = (valid_count - 1) * q
        idx_lower = int(idx)
        idx_upper = min(idx_lower + 1, valid_count - 1)
        weight = idx - idx_lower

        thr = valid_buf[idx_lower] * (1.0 - weight) + valid_buf[idx_upper] * weight

        # past-only quantile from [start_idx: i], compare with current raw[i]
        curr_val = raw[i]
        if not np.isnan(curr_val) and curr_val >= thr:
            out[i] = 1.0
        else:
            out[i] = 0.0

    return out

@numba.njit(cache=True)
def gate_ml_calib_prob_matrix(ml_raw_2d: np.ndarray, q: float, window: int) -> np.ndarray:
    """Per-symbol rolling quantile gate for aligned (bars, symbols) strength."""
    _, cols = ml_raw_2d.shape
    out = np.empty_like(ml_raw_2d, dtype=np.float64)
    for j in range(cols):
        out[:, j] = _numba_rolling_quantile_gate(ml_raw_2d[:, j], q, window)
    return out

def rolling_quantile_gate_score(raw: np.ndarray, q: float, window: int) -> np.ndarray:
    """1.0 where raw >= rolling_quantile(raw, q).shift(1), else 0.0 (past-only, no lookahead)."""
    return _numba_rolling_quantile_gate(raw, q, window)


def apply_ml_calib_gate_column(df: pd.DataFrame, params: Dict[str, Any]) -> None:
    """Sets ml_calib_prob for PortfolioBacktestEngineFast (binary gate vs ENTRY_NUMBA_THRESHOLD)."""
    q = float(params.get("ENTRY_THRESHOLD", 0.90))
    window = int(params.get("ENTRY_QUANTILE_WINDOW", 240))
    if "ml_calib_prob_long" in df.columns:
        p_long = df["ml_calib_prob_long"].to_numpy(dtype=np.float64)
    elif "ml_calib_prob" in df.columns:
        p_long = df["ml_calib_prob"].to_numpy(dtype=np.float64)
    else:
        p_long = np.full(len(df), 0.5, dtype=np.float64)
    p_short = (
        df["ml_calib_prob_short"].to_numpy(dtype=np.float64)
        if "ml_calib_prob_short" in df.columns
        else np.full(len(df), 0.5, dtype=np.float64)
    )
    raw_max = np.maximum(p_long, p_short)
    df["ml_calib_prob"] = rolling_quantile_gate_score(raw_max, q, window)


def _rolling_pass_mask(p: np.ndarray, q: float, window: int) -> np.ndarray:
    out = _numba_rolling_quantile_gate(p, q, window)
    return cast(np.ndarray, np.asarray(out > 0.5, dtype=np.bool_))


@register_futures_signal
class MlCalibProbFuturesSignal:
    name: ClassVar[str] = "ML_CALIB_PROB"
    param_space: ClassVar[Dict[str, Any]] = {}

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FuturesSignalOutput:
        q = float(params.get("ENTRY_THRESHOLD", 0.90))
        window = int(params.get("ENTRY_QUANTILE_WINDOW", 240))

        # [Q3] Independent directional probabilities from dual MetaLabeler
        p_long = (
            df["ml_calib_prob_long"].to_numpy(dtype=np.float64)
            if "ml_calib_prob_long" in df.columns
            else (
                # Legacy fallback: single calib prob → Long only
                df["ml_calib_prob"].to_numpy(dtype=np.float64)
                if "ml_calib_prob" in df.columns
                else np.full(len(df), 0.5)
            )
        )
        p_short = (
            df["ml_calib_prob_short"].to_numpy(dtype=np.float64)
            if "ml_calib_prob_short" in df.columns
            else np.full(len(df), 0.5)
        )

        gate_long = _rolling_pass_mask(p_long, q, window)
        gate_short = _rolling_pass_mask(p_short, q, window)
        direction = p_long >= p_short
        long_e = direction & gate_long
        short_e = (~direction) & gate_short

        n = len(df)
        kill_l: np.ndarray = np.zeros(n, dtype=np.float64)
        kill_s: np.ndarray = np.zeros(n, dtype=np.float64)

        # rank_score: long prob - short prob (positive = long bias, negative = short bias)
        rank = np.clip(p_long - p_short, -1.0, 1.0)

        return FuturesSignalOutput(
            long_entry=np.asarray(long_e, dtype=np.bool_),
            short_entry=np.asarray(short_e, dtype=np.bool_),
            kill_long=kill_l,
            kill_short=kill_s,
            rank_score=rank,
        )
