from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from filelock import FileLock
except ImportError:
    FileLock = None  # type: ignore[misc, assignment]


_logger: logging.Logger = logging.getLogger("opt_spot")

# Optuna TPE `constraints_func`: each value <= 0 means satisfied (Gardner-style soft constraints).


def _segment_with_context(
    full_signal_df: pd.DataFrame,
    exec_start_idx: int,
    exec_end_idx: int,
) -> Tuple[pd.DataFrame, int]:
    slice_start = max(0, int(exec_start_idx) - 1)
    slice_end = max(slice_start, int(exec_end_idx))
    segment = full_signal_df.iloc[slice_start:slice_end].copy()
    execution_start_idx = int(exec_start_idx) - slice_start
    if execution_start_idx == 0 and len(segment) > 1:
        execution_start_idx = 1
    return segment, execution_start_idx


def _dataframe_to_symbol_arrays(sig_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    # volume: required for shared-cash Numba ADV anchor (concurrency slippage scaling).
    required = ("open", "high", "low", "close", "volume", "atr", "long_entry_signal", "entry_upper")
    for c in required:
        if c not in sig_df.columns:
            raise ValueError(f"Missing column {c} for shared-cash segment.")
    out: Dict[str, np.ndarray] = {}
    for c in required:
        out[c] = sig_df[c].to_numpy(dtype=np.float64)
    if "regime_risk_mult" in sig_df.columns:
        out["regime_risk_mult"] = sig_df["regime_risk_mult"].to_numpy(dtype=np.float64)
    if "regime_entry_gate" in sig_df.columns:
        out["regime_entry_gate"] = sig_df["regime_entry_gate"].to_numpy(dtype=np.float64)
    if "regime_state" in sig_df.columns:
        out["regime_state"] = sig_df["regime_state"].to_numpy(dtype=np.float64)
    if "garch_kelly_f" in sig_df.columns:
        out["garch_kelly_f"] = sig_df["garch_kelly_f"].to_numpy(dtype=np.float64)
    if "kill_signal" in sig_df.columns:
        out["kill_signal"] = sig_df["kill_signal"].to_numpy(dtype=np.float64)
    if "bb_upper" in sig_df.columns:
        out["bb_upper"] = sig_df["bb_upper"].to_numpy(dtype=np.float64)
    if "trail_tighten_flag" in sig_df.columns:
        out["trail_tighten_flag"] = sig_df["trail_tighten_flag"].to_numpy(dtype=np.float64)
    return out


def _dataframe_to_symbol_arrays_extended(sig_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Full IS arrays for shared-cash + optional slot_rank_score (views used in CPCV slices)."""
    base = _dataframe_to_symbol_arrays(sig_df)
    if "slot_rank_score" in sig_df.columns:
        base["slot_rank_score"] = sig_df["slot_rank_score"].to_numpy(dtype=np.float64)
    return base


def _slice_symbol_arrays_view(
    full: Dict[str, np.ndarray],
    slice_start: int,
    slice_end: int,
) -> Dict[str, np.ndarray]:
    return {k: v[slice_start:slice_end] for k, v in full.items()}


def _segment_span_days(sig_df: pd.DataFrame, execution_start_idx: int) -> float:
    if "datetime" not in sig_df.columns or sig_df.empty:
        return 1.0
    i0 = min(max(0, int(execution_start_idx)), len(sig_df) - 1)
    span_seconds = float(
        (sig_df["datetime"].iloc[-1] - sig_df["datetime"].iloc[i0]).total_seconds()
    )
    return max(span_seconds / 86400.0, 1.0)


def _span_days_ref_slice(ref_df: pd.DataFrame, abs_start: int, abs_end: int) -> float:
    """Calendar span (days) for CPCV segment [abs_start, abs_end) on aligned reference OHLCV."""
    if ref_df.empty:
        return 1.0
    if "datetime" not in ref_df.columns:
        return max(float(abs_end - abs_start), 1.0) * (4.0 / 24.0)
    i0 = int(np.clip(abs_start, 0, len(ref_df) - 1))
    i1 = int(np.clip(abs_end - 1, 0, len(ref_df) - 1))
    if i1 < i0:
        return 1.0 / 24.0
    dt0 = ref_df["datetime"].iloc[i0]
    dt1 = ref_df["datetime"].iloc[i1]
    sec = float((dt1 - dt0).total_seconds())
    return max(sec / 86400.0, 1.0 / 24.0)
