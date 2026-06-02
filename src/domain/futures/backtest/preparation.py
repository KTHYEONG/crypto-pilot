"""Common preparation helpers for futures backtest execution inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_SUPPORTED_EXECUTION_MODES: frozenset[str] = frozenset({"coarse", "intrabar_1m"})


@dataclass(slots=True)
class PreparedBacktestInputs:
    """Prepared aligned arrays and execution metadata."""

    aligned_data: dict[str, np.ndarray]
    execution_mode: str
    exec_bar_start_1m_idx: np.ndarray | None = None
    exec_bar_end_1m_idx: np.ndarray | None = None
    mark_price_1m: np.ndarray | None = None


def _aggregate_1h_to_4h_block(arr: np.ndarray, mode: str) -> np.ndarray:
    """Aggregate 2D [bars, symbols] 1h arrays to 4h blocks with no look-ahead."""
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got ndim={arr.ndim}")
    n_bars, n_syms = arr.shape
    factor = 4
    if n_bars < factor:
        raise ValueError("insufficient 1h bars for 4h aggregation")
    usable = (n_bars // factor) * factor
    trimmed = arr[n_bars - usable :, :]
    block = trimmed.reshape(usable // factor, factor, n_syms)
    if mode == "open":
        return np.asarray(block[:, 0, :], dtype=np.float64)
    if mode == "high":
        return np.asarray(np.nanmax(block, axis=1), dtype=np.float64)
    if mode == "low":
        return np.asarray(np.nanmin(block, axis=1), dtype=np.float64)
    if mode == "close":
        return np.asarray(block[:, -1, :], dtype=np.float64)
    if mode == "sum":
        return np.asarray(np.nansum(block, axis=1), dtype=np.float64)
    if mode == "max":
        return np.asarray(np.nanmax(block, axis=1), dtype=np.float64)
    if mode == "last":
        return np.asarray(block[:, -1, :], dtype=np.float64)
    if mode == "finite_last":
        finite_mask = np.isfinite(block)
        rev_mask = finite_mask[:, ::-1, :]
        has_any = np.any(rev_mask, axis=1)
        idx_from_end = np.argmax(rev_mask, axis=1)
        take_idx = (factor - 1) - idx_from_end
        gathered = np.take_along_axis(block, take_idx[:, None, :], axis=1)[:, 0, :]
        return np.where(has_any, gathered, np.nan)
    raise ValueError(f"unsupported aggregation mode: {mode}")


def _should_aggregate_from_1h(params: dict[str, Any]) -> bool:
    target_tf = str(params.get("TIMEFRAME", "4h")).lower()
    data_tf = str(params.get("DATA_TIMEFRAME", target_tf)).lower()
    return target_tf == "4h" and data_tf == "1h"


def _aggregate_aligned_data_1h_to_4h(
    aligned_data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return a shallow-copied aligned_data aggregated from 1h bars to 4h bars."""
    out: dict[str, np.ndarray] = dict(aligned_data)
    mode_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "sum",
        "funding_rate_sum": "sum",
        "kill_signal": "max",
        "atr": "finite_last",
        "dyn_leverage": "last",
        "target_weights": "last",
        "candidate_stop_atr_mult": "last",
        "candidate_take_profit_atr_mult": "last",
    }
    mode_map["composer_sigma_bar"] = "last"

    for key, mode in mode_map.items():
        raw = out.get(key)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim != 2:
            continue
        out[key] = _aggregate_1h_to_4h(arr, mode)
    return out


def _aggregate_1h_to_4h(arr: np.ndarray, mode: str) -> np.ndarray:
    return _aggregate_1h_to_4h_block(arr, mode)


def _normalize_execution_mode(params: dict[str, Any]) -> str:
    raw = str(params.get("FUTURES_EXECUTION_MODE", "coarse")).strip().lower()
    if raw in _SUPPORTED_EXECUTION_MODES:
        return raw
    return "coarse"


def _as_1d_float_array(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.ndim != 1:
        return None
    return arr.astype(np.float64, copy=False)


def _build_optional_intrabar_window_mapping(
    aligned_data: dict[str, np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build optional decision bar -> 1m execution window mapping.

    Mapping is only built when datetime-like indices exist:
    - decision bars: ``dt_index`` (1D)
    - execution bars: ``exec_dt_index_1m`` (1D)
    """
    decision_dt = _as_1d_float_array(aligned_data.get("dt_index"))
    exec_dt_1m = _as_1d_float_array(aligned_data.get("exec_dt_index_1m"))
    if decision_dt is None or exec_dt_1m is None:
        return None, None
    if decision_dt.size == 0 or exec_dt_1m.size == 0:
        return None, None

    starts = np.searchsorted(exec_dt_1m, decision_dt, side="left").astype(np.int64, copy=False)
    ends = np.empty_like(starts)
    if starts.size > 1:
        ends[:-1] = np.maximum(starts[1:] - 1, starts[:-1])
    ends[-1] = exec_dt_1m.size - 1
    valid = (starts >= 0) & (starts < exec_dt_1m.size) & (ends >= starts)
    if not np.any(valid):
        return None, None
    starts = np.where(valid, starts, -1)
    ends = np.where(valid, ends, -1)
    return starts, ends


def prepare_backtest_inputs(
    aligned_data: dict[str, np.ndarray],
    params: dict[str, Any],
    mark_price_1m_raw: np.ndarray | None = None,
) -> PreparedBacktestInputs:
    """Prepare aligned arrays for unified backtest execution paths."""
    out = dict(aligned_data)
    if _should_aggregate_from_1h(params):
        out = _aggregate_aligned_data_1h_to_4h(out)
    execution_mode = _normalize_execution_mode(params)
    start_idx, end_idx = _build_optional_intrabar_window_mapping(out)
    if start_idx is not None and end_idx is not None:
        out["exec_bar_start_1m_idx"] = start_idx
        out["exec_bar_end_1m_idx"] = end_idx

    mark_price_1m = None
    if mark_price_1m_raw is not None:
        exec_open_1m = out.get("exec_open_1m")
        if exec_open_1m is not None and mark_price_1m_raw.shape != exec_open_1m.shape:
            raise ValueError(
                f"mark_price_1m_raw shape {mark_price_1m_raw.shape} "
                f"does not match exec_open_1m shape {exec_open_1m.shape}"
            )
        mark_price_1m = mark_price_1m_raw

    return PreparedBacktestInputs(
        aligned_data=out,
        execution_mode=execution_mode,
        exec_bar_start_1m_idx=start_idx,
        exec_bar_end_1m_idx=end_idx,
        mark_price_1m=mark_price_1m,
    )
