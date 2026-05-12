"""Futures Optuna objective: anchored walk-forward (AWF) legs, Kelly-CVaR scalar,
disk+memory signal cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    "slot_rank_score",
    "ml_calib_prob",
    "dyn_leverage",
    "xs_score_long",
    "xs_score_short",
    "composer_sigma_bar",
    "hmm_prob_bull_calm",
    "hmm_prob_bull_vol_up",
    "hmm_prob_bear_trend",
    "hmm_prob_chop",
    "hmm_prob_crisis",
    "hmm_hard_state",
    "hmm_modulator_long",
    "hmm_modulator_short",
    "hmm_modulator_base_long",
    "btc_trend_vol_adj_24h",
)


def _dataframe_to_symbol_arrays(sig_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Converts a signal DataFrame to a dictionary of numpy arrays.
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
        "slot_rank_score": 0.0,
        "ml_calib_prob": 1.0,
        "dyn_leverage": 5.0,
        "xs_score_long": 0.0,
        "xs_score_short": 0.0,
        "composer_sigma_bar": 0.01,
        "hmm_prob_bull_calm": 0.0,
        "hmm_prob_bull_vol_up": 0.0,
        "hmm_prob_bear_trend": 0.0,
        "hmm_prob_chop": 0.0,
        "hmm_prob_crisis": 0.0,
        "hmm_hard_state": 0.0,
        "hmm_modulator_long": 1.0,
        "hmm_modulator_short": 1.0,
        "hmm_modulator_base_long": 1.0,
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
    return aligned_data


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
        "slot_rank_score",
        "ml_calib_prob",
        "dyn_leverage",
        "xs_score_long",
        "xs_score_short",
        "hmm_prob_crisis",
        "hmm_hard_state",
        "hmm_modulator_long",
        "hmm_modulator_short",
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
        for col in [
            "strength_filter",
            "trend_direction",
            "garch_kelly_f",
            "funding_rate_sum",
            "slot_rank_score",
            "ml_calib_prob",
            "xs_score_long",
            "xs_score_short",
            "hmm_prob_crisis",
            "hmm_hard_state",
            "hmm_modulator_long",
            "hmm_modulator_short",
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
