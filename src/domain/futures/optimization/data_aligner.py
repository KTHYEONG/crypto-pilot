"""Futures Optuna objective.

Anchored walk-forward (AWF) legs, Kelly-CVaR scalar, disk+memory signal cache.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

_FUTURES_2D_REQUIRED_COLS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "entry_upper",
    "entry_lower",
    "trend_direction",
    "strength_filter",
    "atr",
    "garch_kelly_f",
    "funding_rate_sum",
    "kill_signal",
    "membership_kill_signal",
    "entry_block_mask",
    "slot_rank_score",
    "ml_calib_prob",
    "alpha_long",
    "alpha_short",
    "dyn_leverage",
    "xs_score_long",
    "xs_score_short",
    "composer_sigma_bar",
    "btc_trend_vol_adj_24h",
)

_EXEC_1M_VALUE_COLS: tuple[str, ...] = (
    "exec_open_1m",
    "exec_high_1m",
    "exec_low_1m",
    "exec_close_1m",
    "exec_volume_1m",
    "funding_event_mask_1m",   # intrabar 엔진 funding event 판정용
    "funding_rate_event_1m",   # intrabar 엔진 funding rate 적용용
)
_EXEC_1M_DT_COL = "exec_dt_index_1m"
_DECISION_DT_COL = "dt_index"


def merge_effective_membership_constraints(
    aligned_data: dict[str, np.ndarray],
    *,
    clamp_target_weights: bool = False,
) -> dict[str, Any]:
    """Merge membership-derived constraints into aligned arrays in-place.

    Returns compact stats to support diagnostics/persistence.
    """
    close_2d = np.asarray(aligned_data.get("close"), dtype=np.float64)
    if close_2d.ndim != 2:
        return {"rows": []}
    n_bars, n_syms = close_2d.shape

    raw_kill = aligned_data.get("kill_signal")
    kill_2d = (
        np.asarray(raw_kill, dtype=np.float64)
        if raw_kill is not None and np.asarray(raw_kill).shape == close_2d.shape
        else np.zeros_like(close_2d, dtype=np.float64)
    )
    membership_kill = aligned_data.get("membership_kill_signal")
    if membership_kill is not None and np.asarray(membership_kill).shape == close_2d.shape:
        kill_2d = np.maximum(kill_2d, np.asarray(membership_kill, dtype=np.float64))

    active = aligned_data.get("universe_active_mask")
    warm = aligned_data.get("universe_entry_warm_mask")
    entry_block = aligned_data.get("entry_block_mask")
    if entry_block is not None and np.asarray(entry_block).shape == close_2d.shape:
        entry_block_2d = np.asarray(entry_block, dtype=np.float64)
    else:
        entry_block_2d = np.zeros_like(close_2d, dtype=np.float64)
    if active is not None and np.asarray(active).shape == close_2d.shape:
        entry_block_2d = np.maximum(
            entry_block_2d,
            np.where(np.asarray(active, dtype=np.float64) > 0.0, 0.0, 1.0),
        )
    if warm is not None and np.asarray(warm).shape == close_2d.shape:
        entry_block_2d = np.maximum(
            entry_block_2d,
            np.where(np.asarray(warm, dtype=np.float64) > 0.0, 0.0, 1.0),
        )

    aligned_data["kill_signal"] = np.ascontiguousarray(kill_2d)
    aligned_data["effective_kill_signal"] = np.ascontiguousarray(kill_2d)
    aligned_data["entry_block_mask"] = np.ascontiguousarray(entry_block_2d)

    target_weights = aligned_data.get("target_weights")
    if clamp_target_weights and target_weights is not None:
        tw = np.asarray(target_weights, dtype=np.float64)
        if tw.shape == close_2d.shape:
            aligned_data["target_weights"] = np.where(entry_block_2d > 0.0, 0.0, tw)

    symbol_names = aligned_data.get("symbol_names")
    rows: list[dict[str, Any]] = []
    if (
        isinstance(symbol_names, np.ndarray)
        and symbol_names.ndim == 1
        and symbol_names.shape[0] == n_syms
    ):
        for s_idx, symbol in enumerate(symbol_names.tolist()):
            rows.append(
                {
                    "symbol": str(symbol),
                    "active_ratio": float(
                        np.mean(
                            (
                                np.asarray(active, dtype=np.float64)[:, s_idx]
                                if active is not None and np.asarray(active).shape == close_2d.shape
                                else np.ones(n_bars, dtype=np.float64)
                            )
                            > 0.0
                        )
                    ),
                    "warm_ratio": float(
                        np.mean(
                            (
                                np.asarray(warm, dtype=np.float64)[:, s_idx]
                                if warm is not None and np.asarray(warm).shape == close_2d.shape
                                else np.ones(n_bars, dtype=np.float64)
                            )
                            > 0.0
                        )
                    ),
                    "forced_exit_count": int(np.count_nonzero(kill_2d[:, s_idx] > 0.0)),
                    "blocked_entry_count": int(np.count_nonzero(entry_block_2d[:, s_idx] > 0.0)),
                }
            )
    return {"rows": rows}


def _dataframe_to_symbol_arrays(sig_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convert a signal DataFrame to a dictionary of numpy arrays.

    Optimized to minimize allocations and redundant filling.
    """
    out: dict[str, np.ndarray] = {}
    
    # 1. Base OHLCV (Direct to numpy, no filling needed for these usually)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in sig_df.columns:
            out[col] = sig_df[col].to_numpy(dtype=np.float64, copy=False)
        else:
            out[col] = np.ones(len(sig_df), dtype=np.float64)
    
    # 2. ATR (Fast ffill then fallback)
    atr = sig_df["atr"].to_numpy(dtype=np.float64, copy=True)
    mask = np.isnan(atr)
    if mask.any():
        # Simple forward fill on numpy array
        idx = np.where(~mask, np.arange(mask.size), 0)
        np.maximum.accumulate(idx, out=idx)
        atr = atr[idx]
        # Final fallback for leading NaNs
        np.nan_to_num(atr, copy=False, nan=out["close"][0] * 0.01)
    out["atr"] = atr

    # 3. Remaining signal columns (Fast constant fill)
    fill_map = {
        "strength_filter": 0.0,
        "trend_direction": 0.0,
        "entry_upper": 999999.0,
        "entry_lower": 0.0,
        "garch_kelly_f": 1.0,
        "funding_rate_sum": 0.0,
        "kill_signal": 0.0,
        "membership_kill_signal": 0.0,
        "entry_block_mask": 0.0,
        "slot_rank_score": 0.0,
        "ml_calib_prob": 1.0,
        "alpha_long": 0.0,
        "alpha_short": 0.0,
        "dyn_leverage": 5.0,
        "xs_score_long": 0.0,
        "xs_score_short": 0.0,
        "composer_sigma_bar": 0.01,
        "btc_trend_vol_adj_24h": 0.0,
    }
    
    for col, fill_val in fill_map.items():
        if col in sig_df.columns:
            arr = sig_df[col].to_numpy(dtype=np.float64, copy=True)
            np.nan_to_num(arr, copy=False, nan=fill_val)
            out[col] = arr
        else:
            # Create zeros if missing
            out[col] = np.full(out["close"].shape, fill_val, dtype=np.float64)
            
    return out


def _build_aligned_2d_from_prebuilt(
    prebuilt_arrays: dict[str, dict[str, np.ndarray]],
    symbols: list[str],
    slice_start: int,
    slice_end: int,
    sigma_3d_full: np.ndarray | None = None,
) -> dict[str, np.ndarray] | None:
    if slice_end - slice_start < 2:
        return None
    aligned_data: dict[str, np.ndarray] = {}
    aligned_data["symbol_names"] = np.asarray(symbols, dtype=object)
    
    # Slice the precomputed 3D covariance if provided
    if sigma_3d_full is not None:
        if slice_end > sigma_3d_full.shape[0]:
            return None
        aligned_data["sigma_3d"] = sigma_3d_full[slice_start:slice_end]
        
    for col in _FUTURES_2D_REQUIRED_COLS:
        col_views: list[np.ndarray] = []
        for sym in symbols:
            sym_arrs = prebuilt_arrays.get(sym)
            if sym_arrs is None:
                return None
            arr = sym_arrs.get(col)
            if arr is None or slice_end > int(arr.shape[0]):
                return None
            col_views.append(arr[slice_start:slice_end])
        try:
            merged = np.column_stack(col_views).astype(np.float64, copy=False)
        except ValueError:
            return None
        aligned_data[col] = np.ascontiguousarray(merged)

    decision_dt = _extract_decision_dt_index(
        prebuilt_arrays=prebuilt_arrays,
        symbols=symbols,
        slice_start=slice_start,
        slice_end=slice_end,
    )
    if decision_dt is not None:
        aligned_data[_DECISION_DT_COL] = np.ascontiguousarray(decision_dt)

    exec_payload = _build_optional_exec_1m_payload(
        prebuilt_arrays=prebuilt_arrays,
        symbols=symbols,
    )
    if exec_payload is not None:
        aligned_data.update(exec_payload)
    return aligned_data


def _extract_decision_dt_index(
    prebuilt_arrays: dict[str, dict[str, np.ndarray]],
    symbols: list[str],
    slice_start: int,
    slice_end: int,
) -> np.ndarray | None:
    """Extract decision timeframe datetime index for current slice."""
    for sym in symbols:
        sym_arrs = prebuilt_arrays.get(sym)
        if sym_arrs is None:
            continue
        raw_dt = sym_arrs.get(_DECISION_DT_COL)
        if raw_dt is None:
            continue
        dt_arr = np.asarray(raw_dt, dtype=np.float64)
        if dt_arr.ndim != 1 or slice_end > int(dt_arr.shape[0]):
            continue
        return dt_arr[slice_start:slice_end]
    return None


def _build_optional_exec_1m_payload(
    prebuilt_arrays: dict[str, dict[str, np.ndarray]],
    symbols: list[str],
) -> dict[str, np.ndarray] | None:
    """Build optional 1m execution payload aligned on deterministic master index.

    Alignment rule: master union sorted index from all available symbol-level
    ``exec_dt_index_1m`` arrays. Missing timestamps for a symbol remain NaN.
    """
    exec_dt_candidates: list[np.ndarray] = []
    symbol_exec_dt: dict[str, np.ndarray] = {}
    for sym in symbols:
        sym_arrs = prebuilt_arrays.get(sym)
        if sym_arrs is None:
            continue
        raw_dt = sym_arrs.get(_EXEC_1M_DT_COL)
        if raw_dt is None:
            continue
        dt_arr = np.asarray(raw_dt, dtype=np.float64)
        if dt_arr.ndim != 1 or dt_arr.size == 0:
            continue
        exec_dt_candidates.append(dt_arr)
        symbol_exec_dt[sym] = dt_arr
    if not exec_dt_candidates:
        return None

    master_exec_dt = np.unique(np.concatenate(exec_dt_candidates).astype(np.float64, copy=False))
    if master_exec_dt.size == 0:
        return None

    n_exec = int(master_exec_dt.size)
    n_syms = len(symbols)
    payload: dict[str, np.ndarray] = {
        _EXEC_1M_DT_COL: np.ascontiguousarray(master_exec_dt),
    }
    for col in _EXEC_1M_VALUE_COLS:
        payload[col] = np.full((n_exec, n_syms), np.nan, dtype=np.float64)

    for s_idx, sym in enumerate(symbols):
        sym_arrs = prebuilt_arrays.get(sym)
        sym_dt = symbol_exec_dt.get(sym)
        if sym_arrs is None or sym_dt is None:
            continue
        idx = np.searchsorted(master_exec_dt, sym_dt, side="left")
        in_range = idx < n_exec
        if not np.any(in_range):
            continue
        idx = idx[in_range]
        src_mask = master_exec_dt[idx] == sym_dt[in_range]
        if not np.any(src_mask):
            continue
        row_idx = idx[src_mask]

        for col in _EXEC_1M_VALUE_COLS:
            raw_val = sym_arrs.get(col)
            if raw_val is None:
                continue
            val_arr = np.asarray(raw_val, dtype=np.float64)
            if val_arr.ndim != 1 or val_arr.shape[0] != sym_dt.shape[0]:
                continue
            payload[col][row_idx, s_idx] = val_arr[in_range][src_mask]

    # [ML-UPGRADE] 1M Price/Volume/Funding Imputation & Data Audit Guard
    # High-speed JIT-equivalent NumPy imputation per symbol column
    for col in _EXEC_1M_VALUE_COLS:
        arr_2d = payload[col]
        col_total_size = arr_2d.size
        col_nan_count_pre = np.count_nonzero(np.isnan(arr_2d))
        is_price_col = col in {
            "exec_open_1m",
            "exec_high_1m",
            "exec_low_1m",
            "exec_close_1m",
        }

        for s_idx in range(n_syms):
            col_arr = arr_2d[:, s_idx]
            nan_mask = np.isnan(col_arr)
            if not np.any(nan_mask):
                continue

            if is_price_col:
                # 1. Forward Fill: Propagate previous valid pricing
                non_nan_indices = np.flatnonzero(~nan_mask)
                if non_nan_indices.size > 0:
                    idx = np.where(~nan_mask, np.arange(col_arr.size), 0)
                    np.maximum.accumulate(idx, out=idx)
                    col_arr[nan_mask] = col_arr[idx[nan_mask]]

                    # 2. Backward Fill: Propagate earliest price to leading NaNs
                    nan_mask_post = np.isnan(col_arr)
                    if np.any(nan_mask_post):
                        col_arr[nan_mask_post] = col_arr[non_nan_indices[0]]
                else:
                    # Fallback to 0.0 only if the entire series is empty
                    col_arr[nan_mask] = 0.0
            else:
                # Flat fill volume and funding event rates to 0.0
                col_arr[nan_mask] = 0.0

        col_nan_count_post = np.count_nonzero(np.isnan(arr_2d))
        nan_pct_pre = (col_nan_count_pre / max(1, col_total_size)) * 100.0
        nan_pct_post = (col_nan_count_post / max(1, col_total_size)) * 100.0
        _logger.info(
            " 📊 [DATA INTEGRITY AUDIT: 1M EXECUTION] col=%s nan_pct=%.2f%% -> %.2f%%",
            col,
            nan_pct_pre,
            nan_pct_post,
        )

    return payload


def align_data_for_2d_engine(
    signal_dfs: dict[str, pd.DataFrame],
    symbols: list[str],
) -> tuple[dict[str, np.ndarray], pd.Series]:
    all_dates: list[pd.Series] = []
    for sym in symbols:
        df = signal_dfs.get(sym)
        if df is not None and "datetime" in df.columns:
            all_dates.append(df["datetime"])
    if not all_dates:
        empty: dict[str, np.ndarray] = {}
        return empty, pd.Series(dtype="datetime64[ns]")

    master_index = (
        pd.concat(all_dates, ignore_index=True)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    master_df = pd.DataFrame({"datetime": master_index})
    n_bars = len(master_index)
    n_syms = len(symbols)

    target_cols = [
        "open",
        "high",
        "low",
        "close",
        "entry_upper",
        "entry_lower",
        "trend_direction",
        "strength_filter",
        "atr",
        "garch_kelly_f",
        "funding_rate_sum",
        "kill_signal",
        "membership_kill_signal",
        "entry_block_mask",
        "slot_rank_score",
        "ml_calib_prob",
        "dyn_leverage",
        "xs_score_long",
        "xs_score_short",
        "alpha_long",
        "alpha_short",
    ]
    aligned_data: dict[str, np.ndarray] = {
        col: np.full((n_bars, n_syms), np.nan, dtype=np.float64) for col in target_cols
    }

    for s_idx, sym in enumerate(symbols):
        df = signal_dfs.get(sym)
        if df is None:
            continue
        merged = pd.merge(master_df, df, on="datetime", how="left")
        for col in ["open", "high", "low", "close", "atr"]:
            if col in merged.columns:
                aligned_data[col][:, s_idx] = merged[col].ffill().values
        # ATR fallback: atr이 없거나 모두 0/NaN이면 close의 2%로 채움
        atr_col = aligned_data["atr"][:, s_idx]
        if not np.any(np.isfinite(atr_col) & (atr_col > 0)):
            close_col = aligned_data["close"][:, s_idx]
            close_finite = np.where(np.isfinite(close_col) & (close_col > 0), close_col, 1.0)
            aligned_data["atr"][:, s_idx] = close_finite * 0.02
        for col in [
            "strength_filter",
            "trend_direction",
            "garch_kelly_f",
            "funding_rate_sum",
            "kill_signal",
            "membership_kill_signal",
            "entry_block_mask",
            "slot_rank_score",
            "ml_calib_prob",
            "xs_score_long",
            "xs_score_short",
            "alpha_long",
            "alpha_short",
        ]:
            if col in merged.columns:
                val = merged[col].fillna(0).values
                if col.startswith("hmm_modulator"):
                    val = merged[col].fillna(1.0).values
                aligned_data[col][:, s_idx] = val
            else:
                aligned_data[col][:, s_idx] = 1.0 if col.startswith("hmm_modulator") else 0.0
        if "dyn_leverage" in merged.columns:
            aligned_data["dyn_leverage"][:, s_idx] = merged["dyn_leverage"].fillna(5.0).values
        else:
            aligned_data["dyn_leverage"][:, s_idx] = 5.0
        for col in ["entry_upper", "entry_lower"]:
            if col in merged.columns:
                default_val = 999999.0 if col == "entry_upper" else 0.0
                aligned_data[col][:, s_idx] = merged[col].fillna(default_val).values

    return aligned_data, master_index


def _segment_with_context(
    full_signal_df: pd.DataFrame,
    exec_start_idx: int,
    exec_end_idx: int,
) -> tuple[pd.DataFrame, int]:
    slice_start = max(0, int(exec_start_idx) - 1)
    slice_end = max(slice_start, int(exec_end_idx))
    segment = full_signal_df.iloc[slice_start:slice_end].copy(deep=False)
    execution_start_idx = int(exec_start_idx) - slice_start
    if execution_start_idx == 0 and len(segment) > 1:
        execution_start_idx = 1
    return segment, execution_start_idx
