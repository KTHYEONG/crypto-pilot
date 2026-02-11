import argparse
import pandas as pd
import os
import sys
import optuna
import logging
import sqlite3
import numpy as np
from pathlib import Path
import threading
import time
import re
from typing import Dict, List, Optional, Sequence, Tuple

# Project Root Setup
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    sys.path.append(os.getcwd())

from config.settings import (
    BACKTEST_START_DATE,
    BACKTEST_END_DATE,
    TRAIN_CUTOFF_DATE,
    FUTURES_INITIAL_BALANCE,
)
from config.optimization_config_modes import GET_SEARCH_SPACE, BASE_SEARCH_SPACE
from config.optimization_config_ultimate import (
    COMMON_SEARCH_SPACE,
)  # Keep for potential shared usage
from src.data.collector import DataCollector
from src.futures_strategy.strategies_futures import UltimateStrategy
from src.futures_strategy.engine_fast_futures import (
    BacktestEngineFast,
    backtest_loop_numba,
)
from src.optimization.opt_utils import suggest_params, calculate_score, OBJECTIVE_CFG

# Shared constants
DAILY_BUFFER_DAYS = 200
GC_TRIAL_INTERVAL = int(os.getenv("OPTUNA_GC_TRIAL_INTERVAL", "25"))
_gc_trial_counter = 0
_gc_trial_lock = threading.Lock()

WARMUP_BUFFER_BARS = {
    "5m": 500,
    "15m": 400,
    "30m": 350,
    "1h": 300,
    "2h": 250,
    "4h": 200,
    "1d": 150,
    "3d": 100,
}

# AWFO defaults (pragmatic runtime vs robustness)
AWFO_DEFAULTS = {
    "enabled_modes": {"UNIFIED", "ALL"},
    "folds": 3,
    "min_trades_per_fold": 40,
    "min_test_bars": {
        "15m": 1200,
        "30m": 900,
        "1h": 600,
        "2h": 420,
        "4h": 240,
        "1d": 120,
        "3d": 60,
    },
    "embargo_bars": {
        "15m": 32,
        "30m": 24,
        "1h": 24,
        "2h": 16,
        "4h": 12,
        "1d": 5,
        "3d": 2,
    },
}

STRUCTURE_PARAM_KEYS = [
    "ENTRY_TYPE",
    "TREND_FILTER_TYPE",
    "STRENGTH_FILTER_TYPE",
    "EXIT_TYPE",
    "STOP_LOSS_TYPE",
    "USE_TAKE_PROFIT",
    "USE_VOLUME_FILTER",
    "TIMEFRAME",
    "USE_DYNAMIC_RISK",
]

TWO_STAGE_UNIFIED_DEFAULTS = {
    "stage1_total_trials": 1200,
    "stage1_fidelity_steps": [
        {"name": "low", "ratio": 0.55, "symbols": 1, "data_ratio": 0.45, "folds": 2, "min_trades": 25, "startup_ratio": 0.35},
        {"name": "mid", "ratio": 0.30, "symbols": 2, "data_ratio": 0.70, "folds": 3, "min_trades": 35, "startup_ratio": 0.28},
        {"name": "high", "ratio": 0.15, "symbols": 2, "data_ratio": 1.00, "folds": 3, "min_trades": 40, "startup_ratio": 0.22},
    ],
    "promotion_ratio": 0.35,
    "stage2_top_structures": 6,
    "stage2_trials_per_structure": 140,
    "stage2_folds": 3,
    "stage2_min_trades": 40,
    "stage2_startup_ratio": 0.18,
    "stage2_refine_ratio": 0.45,
    "stage2_refine_top_quantile": 0.22,
    "stage2_refine_min_width_ratio": 0.22,
    "stage2_refine_min_samples": 28,
    "stage2_refine_step_span": 5,
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


ROBUST_OBJECTIVE_DEFAULTS = {
    "fold_score_w_avg": _env_float("OPTUNA_FOLD_W_AVG", 0.30),
    "fold_score_w_p25": _env_float("OPTUNA_FOLD_W_P25", 0.40),
    "fold_score_w_worst": _env_float("OPTUNA_FOLD_W_WORST", 0.30),
    "fold_consistency_target": _env_float("OPTUNA_FOLD_CONSISTENCY_TARGET", 0.55),
    "fold_consistency_penalty": _env_float("OPTUNA_FOLD_CONSISTENCY_PENALTY", 70.0),
    "side_min_ratio_target": _env_float("OPTUNA_SIDE_MIN_RATIO_TARGET", 0.07),
    "side_imbalance_penalty_mult": _env_float("OPTUNA_SIDE_IMBALANCE_PENALTY_MULT", 140.0),
    "side_single_penalty": _env_float("OPTUNA_SIDE_SINGLE_PENALTY", 15.0),
    "cost_stress_per_trade_pct": _env_float("OPTUNA_COST_STRESS_PER_TRADE_PCT", 0.02),
    "cost_stress_weight": _env_float("OPTUNA_COST_STRESS_WEIGHT", 0.08),
    "fold_ret_p25_weight": _env_float("OPTUNA_FOLD_RET_P25_WEIGHT", 0.25),
    "fold_ret_p25_clip": _env_float("OPTUNA_FOLD_RET_P25_CLIP", 80.0),
    "cross_ret_p25_weight": _env_float("OPTUNA_CROSS_RET_P25_WEIGHT", 10.0),
    "cross_pf_p25_weight": _env_float("OPTUNA_CROSS_PF_P25_WEIGHT", 4.0),
    "cross_score_p25_weight": _env_float("OPTUNA_CROSS_SCORE_P25_WEIGHT", 0.08),
    "seed_min_trials": _env_int("OPTUNA_SEED_MIN_TRIALS", 80),
}


def _parse_seed_list(seed_arg: Optional[str]) -> List[int]:
    if seed_arg is None:
        return []
    raw = [s.strip() for s in str(seed_arg).split(",") if s.strip()]
    seeds: List[int] = []
    for item in raw:
        try:
            seeds.append(int(item))
        except ValueError:
            continue
    unique: List[int] = []
    seen = set()
    for s in seeds:
        if s in seen:
            continue
        seen.add(s)
        unique.append(s)
    return unique


def _allocate_seed_trials(
    total_trials: int,
    seeds: Sequence[int],
    min_trials_per_seed: int,
) -> List[Tuple[int, int]]:
    total_trials = int(max(1, total_trials))
    if not seeds:
        return [(13, total_trials)]
    seeds = list(seeds)
    max_seed_count = max(1, total_trials // max(1, min_trials_per_seed))
    active = seeds[:max_seed_count]
    per_seed = total_trials // len(active)
    remainder = total_trials % len(active)
    alloc: List[Tuple[int, int]] = []
    for idx, seed in enumerate(active):
        n_trials = per_seed + (1 if idx < remainder else 0)
        if n_trials > 0:
            alloc.append((int(seed), int(n_trials)))
    return alloc or [(int(seeds[0]), total_trials)]


def _study_complete_trials(study: optuna.study.Study) -> List[optuna.trial.FrozenTrial]:
    completed = [tr for tr in study.trials if tr.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda tr: float(tr.value), reverse=True)
    return completed


def _robust_value_from_trials(completed: Sequence[optuna.trial.FrozenTrial]) -> float:
    if not completed:
        return -float("inf")
    values = np.array([float(tr.value) for tr in completed], dtype=np.float64)
    top_k = values[: max(3, min(12, len(values)))]
    top_mean = float(np.mean(top_k))
    top_p25 = float(np.percentile(top_k, 25))
    return (0.65 * top_mean) + (0.35 * top_p25)


def _robust_value_from_study(study: optuna.study.Study) -> float:
    return _robust_value_from_trials(_study_complete_trials(study))


def build_anchored_splits(n_bars, n_folds, embargo_bars=0, min_test_bars=120):
    """
    Build anchored walk-forward splits:
    Fold i uses [0..train_end_i] as anchor context and next block as OOS.
    """
    if n_folds < 1 or n_bars < (n_folds + 1):
        return []

    block = n_bars // (n_folds + 1)
    if block < 2:
        return []

    splits = []
    for i in range(1, n_folds + 1):
        train_end = block * i
        test_start = train_end + max(embargo_bars, 0)
        test_end = (block * (i + 1)) if i < n_folds else n_bars

        if test_start >= test_end:
            continue
        if (test_end - test_start) < min_test_bars:
            continue

        splits.append((test_start, test_end))

    return splits


def compute_segment_merge_index(hourly_df, daily_df):
    """
    Build merge index for a sliced segment so engine can use fast index mapping.
    """
    hourly_days = pd.to_datetime(hourly_df["datetime"]).dt.normalize().values.astype("datetime64[ns]")
    daily_days = pd.to_datetime(daily_df["datetime"]).dt.normalize().values.astype("datetime64[ns]")
    if len(daily_days) == 0:
        return np.zeros(len(hourly_days), dtype=np.int32)

    # As-of backward mapping: use the most recent available daily bar for each intraday bar.
    # This is safer than forcing missing keys to index 0.
    pos = np.searchsorted(daily_days, hourly_days, side="right") - 1
    pos = np.clip(pos, 0, len(daily_days) - 1).astype(np.int32)
    return pos


def calculate_oos_mdd_pct(pnl_series, initial_balance):
    """
    Compute MDD (%) from realized trade PnL sequence in OOS window.
    """
    if pnl_series.empty:
        return 0.0

    equity = initial_balance + pnl_series.cumsum().values
    running_max = np.maximum.accumulate(equity)
    running_max[running_max == 0] = 1e-9
    drawdown = (equity - running_max) / running_max * 100.0
    return float(np.min(drawdown)) if len(drawdown) else 0.0


def build_awfo_plan(data_maps, timeframes, folds, min_trades):
    """Build AWFO split plan for the provided data maps."""
    awfo_plan = {"enabled": True, "splits": {}, "min_trades_per_fold": int(min_trades)}
    for sym, tf_map in data_maps.items():
        awfo_plan["splits"][sym] = {}
        for tf in timeframes:
            if tf not in tf_map or tf_map[tf].empty:
                awfo_plan["splits"][sym][tf] = []
                continue

            n_bars = len(tf_map[tf])
            min_test_bars = AWFO_DEFAULTS["min_test_bars"].get(tf, 120)
            embargo_bars = AWFO_DEFAULTS["embargo_bars"].get(tf, 0)
            splits = build_anchored_splits(
                n_bars=n_bars,
                n_folds=int(folds),
                embargo_bars=embargo_bars,
                min_test_bars=min_test_bars,
            )
            awfo_plan["splits"][sym][tf] = splits
    return awfo_plan


def build_awfo_runtime_cache(data_maps, timeframes, awfo_plan):
    """
    Precompute AWFO fold runtime artifacts (slices + merge indices) once.
    """
    if not awfo_plan or not awfo_plan.get("enabled", False):
        return {}

    cached = {}
    splits_by_symbol = awfo_plan.get("splits", {})

    for symbol, tf_map in data_maps.items():
        cached[symbol] = {}
        daily_df = tf_map.get("1d")
        if daily_df is None or daily_df.empty:
            for tf in timeframes:
                cached[symbol][tf] = []
            continue

        for tf in timeframes:
            hourly_df = tf_map.get(tf)
            if hourly_df is None or hourly_df.empty:
                cached[symbol][tf] = []
                continue

            warmup_buffer = WARMUP_BUFFER_BARS.get(tf, 200)
            fold_cache = []
            splits = splits_by_symbol.get(symbol, {}).get(tf, [])

            for test_start, test_end in splits:
                seg_start = max(0, test_start - warmup_buffer)
                segment_hourly = hourly_df.iloc[seg_start:test_end].copy()
                if len(segment_hourly) < 100:
                    continue

                segment_hourly.attrs["warmup_bars"] = test_start - seg_start
                actual_start_time = hourly_df.iloc[test_start]["datetime"]
                actual_end_time = hourly_df.iloc[test_end - 1]["datetime"]

                start_time_buffered = segment_hourly["datetime"].iloc[0]
                end_time = segment_hourly["datetime"].iloc[-1]
                daily_buffer_start = start_time_buffered - pd.Timedelta(days=DAILY_BUFFER_DAYS)
                segment_daily = daily_df[
                    (daily_df["datetime"] >= daily_buffer_start)
                    & (daily_df["datetime"] <= end_time)
                ].copy()
                if segment_daily.empty:
                    continue

                segment_merge_index = compute_segment_merge_index(segment_hourly, segment_daily)
                fold_cache.append(
                    {
                        "hourly": segment_hourly,
                        "daily": segment_daily,
                        "merge_index": segment_merge_index,
                        "actual_start_time": actual_start_time,
                        "actual_end_time": actual_end_time,
                    }
                )

            cached[symbol][tf] = fold_cache

    return cached


def maybe_collect_gc(force=False):
    import gc

    if force:
        gc.collect()
        return
    if GC_TRIAL_INTERVAL <= 0:
        return

    global _gc_trial_counter
    with _gc_trial_lock:
        _gc_trial_counter += 1
        should_collect = (_gc_trial_counter % GC_TRIAL_INTERVAL) == 0
    if should_collect:
        gc.collect()


def subset_data_maps(base_data_maps, ordered_symbols, n_symbols, data_ratio):
    """
    Create a smaller fidelity dataset:
    - only first N symbols
    - optionally use tail portion of each timeframe (recent data)
    """
    n_symbols = max(1, min(int(n_symbols), len(ordered_symbols)))
    selected_symbols = ordered_symbols[:n_symbols]
    ratio = float(max(0.1, min(data_ratio, 1.0)))

    subset = {}
    for sym in selected_symbols:
        subset[sym] = {}
        for tf, df in base_data_maps[sym].items():
            if ratio >= 0.999:
                sliced = df.copy()
            else:
                take_n = max(200, int(len(df) * ratio))
                sliced = df.iloc[-take_n:].copy()

            warm = getattr(df, "attrs", {}).get("warmup_bars", 0)
            sliced.attrs["warmup_bars"] = min(int(warm), len(sliced))
            subset[sym][tf] = sliced
    return subset


def build_stage1_search_space(full_search_space):
    """
    Stage-1 only explores structural/categorical choices.
    """
    stage1 = {}
    for key in STRUCTURE_PARAM_KEYS:
        if key in full_search_space:
            stage1[key] = full_search_space[key].copy()
    return stage1


def freeze_structure_in_space(full_search_space, structure_params):
    """
    Build Stage-2 search space by freezing structure keys to one value.
    """
    stage2 = {}
    for key, spec in full_search_space.items():
        stage2[key] = spec.copy()
        if key in structure_params and spec.get("type") == "categorical":
            stage2[key]["choices"] = [structure_params[key]]
    return stage2


def extract_structure_signature(params):
    """Extract structure-only params from an Optuna params dict."""
    sig = {}
    for key in STRUCTURE_PARAM_KEYS:
        if key in params:
            sig[key] = params[key]
    return sig


def restrict_stage1_space_by_candidates(stage1_space, candidate_structures):
    """
    Restrict categorical choices to values observed in promoted structures.
    """
    if not candidate_structures:
        return stage1_space

    restricted = {}
    for key, spec in stage1_space.items():
        restricted[key] = spec.copy()
        if spec.get("type") == "categorical":
            values = sorted({c[key] for c in candidate_structures if key in c})
            if values:
                restricted[key]["choices"] = values
    return restricted


def _align_to_step(value: float, base: float, step: float, mode: str = "round") -> float:
    if step <= 0:
        return float(value)
    pos = (value - base) / step
    if mode == "ceil":
        snapped = np.ceil(pos) * step + base
    elif mode == "floor":
        snapped = np.floor(pos) * step + base
    else:
        snapped = np.round(pos) * step + base
    return float(snapped)


def build_adaptive_numeric_space(
    base_space: Dict[str, dict],
    completed_trials: Sequence[optuna.trial.FrozenTrial],
    top_quantile: float = 0.22,
    min_width_ratio: float = 0.22,
    min_samples: int = 28,
    min_step_span: int = 5,
) -> Dict[str, dict]:
    """
    Narrow numeric search bounds from elite trial distribution with minimum guard width.
    """
    if not completed_trials:
        return base_space

    narrowed: Dict[str, dict] = {k: v.copy() for k, v in base_space.items()}
    top_quantile = float(np.clip(top_quantile, 0.05, 0.50))
    min_width_ratio = float(np.clip(min_width_ratio, 0.05, 0.80))
    min_samples = int(max(10, min_samples))
    min_step_span = int(max(2, min_step_span))

    top_n = max(1, int(len(completed_trials) * top_quantile))
    elite = list(completed_trials[:top_n])

    for key, spec in narrowed.items():
        if spec.get("type") not in {"float", "int"}:
            continue
        if "low" not in spec or "high" not in spec:
            continue

        vals: List[float] = []
        for tr in elite:
            if key in tr.params:
                try:
                    vals.append(float(tr.params[key]))
                except (TypeError, ValueError):
                    continue
        if len(vals) < min_samples:
            continue

        base_low = float(spec["low"])
        base_high = float(spec["high"])
        if base_high <= base_low:
            continue

        q20, q50, q80 = np.percentile(vals, [20, 50, 80])
        elite_span = max(float(q80 - q20), 1e-12)
        base_span = base_high - base_low
        min_span = base_span * min_width_ratio
        target_span = max(min_span, elite_span * 1.6)

        if spec.get("type") == "int":
            target_span = max(target_span, float(min_step_span))
        elif "step" in spec:
            step = float(spec["step"])
            target_span = max(target_span, step * float(min_step_span))

        new_low = max(base_low, float(q50) - (target_span / 2.0))
        new_high = min(base_high, float(q50) + (target_span / 2.0))

        if new_high <= new_low:
            continue

        if "step" in spec:
            step = float(spec["step"])
            new_low = _align_to_step(new_low, base_low, step, mode="ceil")
            new_high = _align_to_step(new_high, base_low, step, mode="floor")
            if new_high <= new_low:
                continue
            span_steps = int(np.floor((new_high - new_low) / max(step, 1e-12)))
            if span_steps < min_step_span:
                continue

        if spec.get("type") == "int":
            new_low = int(np.floor(new_low))
            new_high = int(np.ceil(new_high))
            if new_high <= new_low:
                continue
            if (new_high - new_low) < min_step_span:
                continue
            spec["low"] = int(max(int(base_low), new_low))
            spec["high"] = int(min(int(base_high), new_high))
        else:
            spec["low"] = float(max(base_low, new_low))
            spec["high"] = float(min(base_high, new_high))

    return narrowed


def run_seeded_studies(
    study_name_prefix: str,
    storage: str,
    n_trials: int,
    n_jobs: int,
    startup_ratio: float,
    objective_fn,
    seeds: Sequence[int],
) -> List[Tuple[int, int, optuna.study.Study, int]]:
    """
    Run multiple studies with different TPE seeds and return (seed, trials, study, startup).
    """
    allocations = _allocate_seed_trials(
        total_trials=int(max(1, n_trials)),
        seeds=seeds,
        min_trials_per_seed=int(ROBUST_OBJECTIVE_DEFAULTS["seed_min_trials"]),
    )
    seed_runs: List[Tuple[int, int, optuna.study.Study, int]] = []
    for seed, seed_trials in allocations:
        seed_study_name = f"{study_name_prefix}__seed_{seed}_{int(time.time()*1000)}"
        try:
            optuna.delete_study(study_name=seed_study_name, storage=storage)
        except Exception:
            pass

        study, startup = run_optuna_study(
            study_name=seed_study_name,
            storage=storage,
            n_trials=seed_trials,
            n_jobs=n_jobs,
            startup_ratio=startup_ratio,
            objective_fn=objective_fn,
            seed=seed,
        )
        seed_runs.append((seed, seed_trials, study, startup))
    return seed_runs


def load_all_timeframes(symbol, start_date, end_date, timeframes):
    """Load all necessary timeframe data into memory"""
    data_map = {}
    collector = DataCollector()

    # Daily Data (Required for Indicators)
    # Even for SCALP mode, daily context is often useful (e.g., trend alignment)
    try:
        df = collector.ensure_data(symbol, "1d", start_date, end_date)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["date_key"] = df["datetime"].dt.strftime("%Y-%m-%d")
        data_map["1d"] = df
    except Exception as e:
        print(f"[ERROR]: Failed to load {symbol}-1d data: {e}")
        sys.exit(1)

    for tf in timeframes:
        try:
            df = collector.ensure_data(symbol, tf, start_date, end_date)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["date_key"] = df["datetime"].dt.strftime("%Y-%m-%d")
            data_map[tf] = df
        except Exception as e:
            print(f"[ERROR]: Failed to load {symbol}-{tf} data: {e}")
            sys.exit(1)

    return data_map


def compute_merge_indices(data_maps):
    """
    Pre-compute merge index mappings for all symbols and timeframes.
    
    This eliminates the need for pd.merge on every trial by creating a lookup table
    that maps each hourly bar to its corresponding daily bar index.
    
    Returns:
        dict: {symbol: {timeframe: merge_index_array}}
    """
    merge_indices = {}
    
    for symbol, data_map in data_maps.items():
        merge_indices[symbol] = {}
        
        # Get normalized daily timestamps for fast as-of mapping
        daily_df = data_map["1d"]
        daily_days = pd.to_datetime(daily_df["datetime"]).dt.normalize().values.astype("datetime64[ns]")
        if len(daily_days) == 0:
            print(f"[WARN]  Warning: {symbol}-1d is empty. Using fallback index 0 for all timeframes.")
            for tf, tf_df in data_map.items():
                if tf == "1d":
                    continue
                merge_indices[symbol][tf] = np.zeros(len(tf_df), dtype=np.int32)
            continue
        
        # For each timeframe, create the merge index
        for tf, tf_df in data_map.items():
            if tf == '1d':
                continue  # Skip daily itself
            
            # As-of backward mapping (hourly day -> latest daily day <= hourly day)
            hourly_days = pd.to_datetime(tf_df["datetime"]).dt.normalize().values.astype("datetime64[ns]")
            pos = np.searchsorted(daily_days, hourly_days, side="right") - 1
            before_first = int(np.sum(pos < 0))
            after_last = int(np.sum(hourly_days > daily_days[-1]))
            if before_first > 0 or after_last > 0:
                print(
                    f"[INFO]  Info: {symbol}-{tf} as-of mapped out-of-range bars "
                    f"(before_first={before_first}, after_last={after_last})."
                )
            merge_index = np.clip(pos, 0, len(daily_days) - 1).astype(np.int32)
            merge_indices[symbol][tf] = merge_index
    
    return merge_indices









def objective(
    trial,
    strategy_cls,
    strategy_name,
    data_maps,
    search_space,
    common_search_space,
    merge_indices=None,
    awfo_plan=None,
):
    """
    Multi-symbol objective function.

    Args:
        merge_indices: Pre-computed merge index mappings for full-run path.
        awfo_plan: Dict containing AWFO settings and pre-built splits.
    """
    # 1. Generate Params
    strategy_params = suggest_params(trial, search_space)
    common_params = suggest_params(trial, common_search_space)
    full_params = {**strategy_params, **common_params}

    # [VALIDATION] Enforce Logical Constraints
    if full_params.get("TREND_FILTER_TYPE") == "MACD":
        if full_params.get("MACD_FAST", 12) >= full_params.get("MACD_SLOW", 26):
            return -10000

    selected_tf = full_params.get("TIMEFRAME", "1h")
    mode_str = strategy_name.split("_")[-1]

    awfo_enabled = bool(awfo_plan and awfo_plan.get("enabled", False))
    awfo_splits_by_symbol = awfo_plan.get("splits", {}) if awfo_enabled else {}
    awfo_cache_by_symbol = awfo_plan.get("cache", {}) if awfo_enabled else {}
    awfo_min_trades = awfo_plan.get("min_trades_per_fold", 40) if awfo_enabled else None

    symbol_scores = []
    symbol_results = {}
    report_step = 0

    def append_symbol_fallback(sym, reason):
        # Keep trial alive with a conservative finite score instead of global collapse.
        print(f"[WARN] Symbol fallback applied: {sym} ({reason})")
        symbol_scores.append(-180.0)
        symbol_results[sym] = {
            "return": -20.0,
            "mdd": -55.0,
            "trades": 0,
            "win_rate": 0.0,
            "pf": 0.0,
        }

    for symbol, data_map in data_maps.items():
        if selected_tf not in data_map:
            append_symbol_fallback(symbol, f"missing timeframe={selected_tf}")
            trial.set_user_attr(f"ret_{symbol.replace('/', '_')}", float(symbol_results[symbol]["return"]))
            trial.set_user_attr(f"mdd_{symbol.replace('/', '_')}", float(symbol_results[symbol]["mdd"]))
            trial.set_user_attr(f"pf_{symbol.replace('/', '_')}", float(symbol_results[symbol]["pf"]))
            continue

        hourly_df = data_map[selected_tf]
        daily_df = data_map["1d"]

        # ===== Path A: AWFO (anchored OOS folds) =====
        if awfo_enabled:
            awfo_splits = awfo_splits_by_symbol.get(symbol, {}).get(selected_tf, [])
            awfo_cached_folds = awfo_cache_by_symbol.get(symbol, {}).get(selected_tf, [])
            if len(awfo_splits) < 2 or len(awfo_cached_folds) < 2:
                append_symbol_fallback(symbol, "insufficient awfo folds")
                trial.set_user_attr(f"ret_{symbol.replace('/', '_')}", float(symbol_results[symbol]["return"]))
                trial.set_user_attr(f"mdd_{symbol.replace('/', '_')}", float(symbol_results[symbol]["mdd"]))
                trial.set_user_attr(f"pf_{symbol.replace('/', '_')}", float(symbol_results[symbol]["pf"]))
                continue

            fold_scores = []
            fold_returns = []
            fold_mdds = []
            fold_pfs = []
            fold_stress_returns = []
            fold_trade_counts = []
            long_trades = 0
            short_trades = 0
            invalid_fold_count = 0

            for fold_idx, fold_ctx in enumerate(awfo_cached_folds):
                segment_hourly = fold_ctx["hourly"]
                segment_daily = fold_ctx["daily"]
                segment_merge_index = fold_ctx["merge_index"]
                actual_start_time = fold_ctx["actual_start_time"]
                actual_end_time = fold_ctx["actual_end_time"]
                fold_score = None
                try:
                    strategy = strategy_cls(f"{strategy_name}_{symbol}_F{fold_idx+1}", full_params)
                    engine = BacktestEngineFast(
                        segment_hourly,
                        segment_daily,
                        strategy,
                        initial_balance=FUTURES_INITIAL_BALANCE,
                        merge_index_map=segment_merge_index,
                    )
                    engine.leverage = full_params.get("LEVERAGE", 1)
                    engine.risk_per_trade = full_params.get("RISK_PER_TRADE", 0.02)
                    result = engine.run()
                except Exception as e:
                    print(f"[WARN] AWFO fold backtest failed for {symbol} (fold {fold_idx+1}): {e}")
                    import traceback
                    traceback.print_exc()
                    if "engine" in locals():
                        del engine
                    if "strategy" in locals():
                        del strategy
                    maybe_collect_gc()
                    invalid_fold_count += 1
                    continue

                trades_df = result["trades_df"]

                if not trades_df.empty and "exit_time" in trades_df.columns:
                    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
                    oos_trades = trades_df[
                        (trades_df["exit_time"] >= actual_start_time)
                        & (trades_df["exit_time"] <= actual_end_time)
                    ].copy()

                    if not oos_trades.empty and "pnl" in oos_trades.columns:
                        pnl_cumsum = oos_trades["pnl"].cumsum().shift(1).fillna(0)
                        oos_trades["balance_before"] = FUTURES_INITIAL_BALANCE + pnl_cumsum
                        oos_trades["balance_before"] = oos_trades["balance_before"].replace(0, 1e-9)
                        oos_trades["pnl_pct"] = (
                            oos_trades["pnl"] / oos_trades["balance_before"]
                        ) * 100.0
                        oos_trades["pnl_pct"] = oos_trades["pnl_pct"].replace(
                            [np.inf, -np.inf], 0
                        ).fillna(0)

                        fold_ret = (oos_trades["pnl"].sum() / FUTURES_INITIAL_BALANCE) * 100.0
                        fold_mdd = calculate_oos_mdd_pct(oos_trades["pnl"], FUTURES_INITIAL_BALANCE)
                        fold_score = calculate_score(
                            fold_ret,
                            fold_mdd,
                            oos_trades,
                            mode=mode_str,
                            market_type="futures",
                            timeframe=selected_tf,
                            min_trades_override=awfo_min_trades,
                        )

                        if np.isfinite(fold_score) and fold_score > -9000:
                            fold_scores.append(float(fold_score))
                            fold_returns.append(float(fold_ret))
                            fold_mdds.append(float(fold_mdd))
                            fold_trade_count = int(len(oos_trades))
                            fold_trade_counts.append(fold_trade_count)
                            stress_ret = float(
                                fold_ret - (
                                    fold_trade_count
                                    * ROBUST_OBJECTIVE_DEFAULTS["cost_stress_per_trade_pct"]
                                )
                            )
                            fold_stress_returns.append(stress_ret)
                            if "side" in oos_trades.columns:
                                side_counts = oos_trades["side"].value_counts()
                                long_trades += int(side_counts.get("LONG", 0))
                                short_trades += int(side_counts.get("SHORT", 0))
                            gross_profit = oos_trades[oos_trades["pnl"] > 0]["pnl"].sum()
                            gross_loss = abs(oos_trades[oos_trades["pnl"] < 0]["pnl"].sum())
                            fold_pf = (
                                gross_profit / gross_loss
                                if gross_loss > 0
                                else (gross_profit if gross_profit > 0 else 0.0)
                            )
                            fold_pfs.append(float(fold_pf))
                        else:
                            invalid_fold_count += 1
                else:
                    invalid_fold_count += 1

                del engine, result, strategy, trades_df
                report_step += 1
                running_fold_score = (
                    float(np.percentile(fold_scores, 25)) - (invalid_fold_count * 20.0)
                    if fold_scores
                    else (-220.0 - (invalid_fold_count * 10.0))
                )
                trial.report(running_fold_score, report_step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            required_valid_folds = max(2, int(np.ceil(len(awfo_cached_folds) * 0.6)))
            if len(fold_scores) < required_valid_folds:
                append_symbol_fallback(
                    symbol,
                    f"valid_folds={len(fold_scores)} < required={required_valid_folds}",
                )
                trial.set_user_attr(f"ret_{symbol.replace('/', '_')}", float(symbol_results[symbol]["return"]))
                trial.set_user_attr(f"mdd_{symbol.replace('/', '_')}", float(symbol_results[symbol]["mdd"]))
                trial.set_user_attr(f"pf_{symbol.replace('/', '_')}", float(symbol_results[symbol]["pf"]))
                continue

            avg_fold_score = float(np.mean(fold_scores))
            p25_fold_score = float(np.percentile(fold_scores, 25))
            worst_fold_score = float(np.min(fold_scores))
            p25_fold_ret = float(np.percentile(fold_returns, 25)) if fold_returns else -100.0
            p25_stress_ret = (
                float(np.percentile(fold_stress_returns, 25))
                if fold_stress_returns
                else p25_fold_ret
            )
            consistency = (
                sum(score > 0 for score in fold_scores) / len(fold_scores)
                if fold_scores
                else 0.0
            )
            total_side_trades = max(0, long_trades + short_trades)
            side_min_ratio = (
                min(long_trades, short_trades) / float(total_side_trades)
                if total_side_trades > 0
                else 0.0
            )

            final_symbol_score = (
                (ROBUST_OBJECTIVE_DEFAULTS["fold_score_w_avg"] * avg_fold_score)
                + (ROBUST_OBJECTIVE_DEFAULTS["fold_score_w_p25"] * p25_fold_score)
                + (ROBUST_OBJECTIVE_DEFAULTS["fold_score_w_worst"] * worst_fold_score)
            )
            final_symbol_score += (
                ROBUST_OBJECTIVE_DEFAULTS["fold_ret_p25_weight"]
                * np.clip(
                    p25_fold_ret,
                    -ROBUST_OBJECTIVE_DEFAULTS["fold_ret_p25_clip"],
                    ROBUST_OBJECTIVE_DEFAULTS["fold_ret_p25_clip"],
                )
            )
            final_symbol_score += (
                ROBUST_OBJECTIVE_DEFAULTS["cost_stress_weight"]
                * np.clip(p25_stress_ret, -60.0, 60.0)
            )
            consistency_target = ROBUST_OBJECTIVE_DEFAULTS["fold_consistency_target"]
            if consistency < consistency_target:
                final_symbol_score -= (
                    (consistency_target - consistency)
                    * ROBUST_OBJECTIVE_DEFAULTS["fold_consistency_penalty"]
                )
            if fold_returns and min(fold_returns) < -20.0:
                final_symbol_score -= 150.0
            side_shortfall = max(
                0.0,
                ROBUST_OBJECTIVE_DEFAULTS["side_min_ratio_target"] - side_min_ratio,
            )
            final_symbol_score -= (
                side_shortfall * ROBUST_OBJECTIVE_DEFAULTS["side_imbalance_penalty_mult"]
            )
            if long_trades == 0 or short_trades == 0:
                final_symbol_score -= ROBUST_OBJECTIVE_DEFAULTS["side_single_penalty"]
            final_symbol_score -= invalid_fold_count * 15.0

            symbol_scores.append(final_symbol_score)
            symbol_results[symbol] = {
                "return": float(np.mean(fold_returns)) if fold_returns else -100.0,
                "mdd": float(np.mean(fold_mdds)) if fold_mdds else 0.0,
                "trades": int(np.mean(fold_trade_counts)) if fold_trade_counts else 0,
                "win_rate": 0.0,
                "pf": float(np.mean(fold_pfs)) if fold_pfs else 0.0,
                "ret_p25": p25_fold_ret,
                "stress_ret_p25": p25_stress_ret,
                "long_trades": int(long_trades),
                "short_trades": int(short_trades),
                "side_min_ratio": float(side_min_ratio),
            }

        # ===== Path B: Single full-run (legacy / non-AWFO) =====
        else:
            try:
                strategy = strategy_cls(f"{strategy_name}_{symbol}", full_params)
                current_merge_index = None
                if merge_indices and symbol in merge_indices and selected_tf in merge_indices[symbol]:
                    current_merge_index = merge_indices[symbol][selected_tf]

                engine = BacktestEngineFast(
                    hourly_df,
                    daily_df,
                    strategy,
                    initial_balance=FUTURES_INITIAL_BALANCE,
                    merge_index_map=current_merge_index,
                )
                engine.leverage = full_params.get("LEVERAGE", 1)
                engine.risk_per_trade = full_params.get("RISK_PER_TRADE", 0.02)
                result = engine.run()
            except Exception as e:
                print(f"[WARN] Backtest failed for {symbol}: {e}")
                import traceback
                traceback.print_exc()
                if "engine" in locals():
                    del engine
                if "strategy" in locals():
                    del strategy
                maybe_collect_gc(force=True)
                append_symbol_fallback(symbol, "non-awfo backtest exception")
                trial.set_user_attr(f"ret_{symbol.replace('/', '_')}", float(symbol_results[symbol]["return"]))
                trial.set_user_attr(f"mdd_{symbol.replace('/', '_')}", float(symbol_results[symbol]["mdd"]))
                trial.set_user_attr(f"pf_{symbol.replace('/', '_')}", float(symbol_results[symbol]["pf"]))
                continue

            ret = result["total_return_pct"]
            mdd = result["mdd_pct"]
            trades = result["total_trades"]
            win_rate = result["win_rate"]
            trades_df = result["trades_df"]

            pf = 0.0
            if not trades_df.empty and "pnl" in trades_df.columns:
                gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
                gross_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
                pf = (
                    gross_profit / gross_loss
                    if gross_loss > 0
                    else (gross_profit if gross_profit > 0 else 0.0)
                )

            score = calculate_score(
                ret, mdd, trades_df, mode=mode_str, market_type="futures", timeframe=selected_tf
            )
            if not np.isfinite(score) or score <= -9000:
                score = -220.0

            symbol_scores.append(score)
            long_trades = 0
            short_trades = 0
            if not trades_df.empty and "side" in trades_df.columns:
                side_counts = trades_df["side"].value_counts()
                long_trades = int(side_counts.get("LONG", 0))
                short_trades = int(side_counts.get("SHORT", 0))
            side_total = max(1, long_trades + short_trades)
            side_min_ratio = min(long_trades, short_trades) / float(side_total)
            symbol_results[symbol] = {
                "return": ret,
                "mdd": mdd,
                "trades": trades,
                "win_rate": win_rate,
                "pf": pf,
                "ret_p25": ret,
                "stress_ret_p25": ret - (trades * ROBUST_OBJECTIVE_DEFAULTS["cost_stress_per_trade_pct"]),
                "long_trades": int(long_trades),
                "short_trades": int(short_trades),
                "side_min_ratio": float(side_min_ratio),
            }

            del engine, result, strategy, trades_df

        trial.set_user_attr(f"ret_{symbol.replace('/', '_')}", float(symbol_results[symbol]["return"]))
        trial.set_user_attr(f"mdd_{symbol.replace('/', '_')}", float(symbol_results[symbol]["mdd"]))
        trial.set_user_attr(f"pf_{symbol.replace('/', '_')}", float(symbol_results[symbol]["pf"]))

    # Combine scores across symbols:
    # - Keep safety via hard MDD guard
    # - Use geometric-dominant blend for better return-efficiency exploration
    # - Penalize high cross-symbol dispersion for long-term stability
    if not symbol_scores:
        maybe_collect_gc()
        return -10000

    ret_values = [float(r["return"]) for r in symbol_results.values()]
    mdd_values_raw = [float(r["mdd"]) for r in symbol_results.values()]
    mdd_values_abs = [abs(v) for v in mdd_values_raw]
    pf_values = [float(r["pf"]) for r in symbol_results.values()]
    stress_ret_p25_values = [
        float(r.get("stress_ret_p25", r["return"])) for r in symbol_results.values()
    ]
    side_min_ratios = [float(r.get("side_min_ratio", 0.0)) for r in symbol_results.values()]

    avg_ret = float(np.mean(ret_values))
    ret_p25 = float(np.percentile(ret_values, 25))
    stress_ret_p25 = float(np.percentile(stress_ret_p25_values, 25))
    avg_mdd_raw = float(np.mean(mdd_values_raw))
    avg_mdd_abs = float(np.mean(mdd_values_abs))
    max_mdd_abs = float(np.max(mdd_values_abs))
    avg_pf = float(np.mean(pf_values))
    pf_p25 = float(np.percentile(pf_values, 25))
    score_p25 = float(np.percentile(symbol_scores, 25))
    side_ratio_p25 = float(np.percentile(side_min_ratios, 25)) if side_min_ratios else 0.0

    # Hard safety guard: reject pathological drawdown profiles.
    if max_mdd_abs > 65.0 or avg_mdd_abs > 52.0:
        maybe_collect_gc()
        return -10000

    offset = 240.0
    raw_shifted_scores = np.array(symbol_scores, dtype=np.float64) + offset
    collapsed_symbol_count = int(np.sum(raw_shifted_scores <= 1e-9))
    shifted_scores = np.clip(raw_shifted_scores, 1e-6, None)
    harmonic_mean = float(len(shifted_scores) / np.sum(1.0 / shifted_scores))
    geometric_mean = float(np.exp(np.mean(np.log(shifted_scores))))
    log_dispersion = float(np.std(np.log(shifted_scores)))
    dispersion_penalty = float(np.exp(-0.22 * log_dispersion))

    # Geometric-first blend: keeps downside discipline while reducing over-conservatism.
    blended_shifted = (0.35 * harmonic_mean) + (0.65 * geometric_mean)

    # Mild efficiency encouragement (asinh to reduce early saturation).
    efficiency_boost = (
        10.0 * np.clip(np.arcsinh(avg_ret / 45.0), -2.8, 2.8)
        + 4.0 * np.clip(np.arcsinh((avg_pf - 1.1) / 0.9), -2.4, 2.4)
    )

    # Soft drawdown penalty on top of hard guard.
    soft_mdd_penalty = (
        max(0.0, avg_mdd_abs - 32.0) * 1.4
        + max(0.0, max_mdd_abs - 45.0) * 2.2
    )

    final_score = (blended_shifted * dispersion_penalty) - offset
    final_score += efficiency_boost
    final_score += (
        ROBUST_OBJECTIVE_DEFAULTS["cross_ret_p25_weight"]
        * np.clip(np.arcsinh(ret_p25 / 35.0), -2.4, 2.4)
    )
    final_score += (
        ROBUST_OBJECTIVE_DEFAULTS["cross_pf_p25_weight"]
        * np.clip(np.arcsinh((pf_p25 - 1.0) / 0.8), -2.0, 2.0)
    )
    final_score += ROBUST_OBJECTIVE_DEFAULTS["cross_score_p25_weight"] * score_p25
    final_score += 0.08 * np.clip(stress_ret_p25, -80.0, 80.0)
    final_score -= soft_mdd_penalty
    if side_ratio_p25 < ROBUST_OBJECTIVE_DEFAULTS["side_min_ratio_target"]:
        final_score -= (
            ROBUST_OBJECTIVE_DEFAULTS["side_min_ratio_target"] - side_ratio_p25
        ) * 160.0
    final_score -= collapsed_symbol_count * 120.0
    if not np.isfinite(final_score):
        final_score = -10000

    trial.set_user_attr("return_avg", avg_ret)
    trial.set_user_attr("return_p25", ret_p25)
    trial.set_user_attr("stress_return_p25", stress_ret_p25)
    trial.set_user_attr("mdd_avg", avg_mdd_raw)
    trial.set_user_attr("mdd_avg_abs", avg_mdd_abs)
    trial.set_user_attr("pf_avg", avg_pf)
    trial.set_user_attr("pf_p25", pf_p25)
    trial.set_user_attr("side_ratio_p25", side_ratio_p25)

    maybe_collect_gc()
    return final_score


def run_optuna_study(
    study_name,
    storage,
    n_trials,
    n_jobs,
    startup_ratio,
    objective_fn,
    seed=None,
):
    """
    Utility runner with dynamic sampler/pruner settings.
    """
    n_trials = int(max(1, n_trials))
    n_startup_trials = int(max(50, round(n_trials * startup_ratio)))
    n_startup_trials = min(n_startup_trials, max(1, n_trials - 1))

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=n_startup_trials,
        multivariate=True,
        constant_liar=True,
        warn_independent_sampling=False,
        seed=seed,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(30, n_startup_trials // 2),
        n_warmup_steps=2,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )
    return study, n_startup_trials


def publish_best_trial_alias(storage, target_study_name, source_study, source_label=""):
    """Publish top trials from source study into canonical target study."""
    if source_study is None:
        return None

    complete_trials = [
        tr for tr in source_study.trials if tr.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete_trials:
        return None

    try:
        optuna.delete_study(study_name=target_study_name, storage=storage)
    except Exception:
        pass

    published = optuna.create_study(
        study_name=target_study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=1),
    )

    # Keep only top-N from the latest run source, so canonical study
    # always represents latest execution winners (not historical mixed trials).
    TOP_K = 10
    ranked = sorted(complete_trials, key=lambda tr: float(tr.value), reverse=True)[:TOP_K]
    for rank_idx, tr in enumerate(ranked, start=1):
        user_attrs = dict(tr.user_attrs) if tr.user_attrs else {}
        if source_label:
            user_attrs["published_from"] = source_label
        user_attrs["published_rank"] = rank_idx
        frozen_trial = optuna.trial.create_trial(
            params=tr.params,
            distributions=tr.distributions,
            value=tr.value,
            user_attrs=user_attrs,
        )
        published.add_trial(frozen_trial)
    return published


def cleanup_old_stage_studies(storage, base_study_name, keep_recent=30):
    """
    Delete old 2-stage temporary studies and keep only recent ones.
    """
    keep_recent = int(max(0, keep_recent))
    prefix = f"{base_study_name}__s"

    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage)
    except Exception as e:
        print(f"[WARN] Stage cleanup skipped (list failed): {e}")
        return

    stage_studies = []
    ts_re = re.compile(r".*_(\d{6,})$")
    for s in summaries:
        name = s.study_name
        if not name.startswith(prefix):
            continue
        m = ts_re.match(name)
        ts = int(m.group(1)) if m else 0
        stage_studies.append((name, ts))

    total = len(stage_studies)
    if total <= keep_recent:
        print(f"[CLEANUP] Stage cleanup: nothing to delete (found={total}, keep={keep_recent})")
        return

    stage_studies.sort(key=lambda x: (x[1], x[0]), reverse=True)
    to_delete = [name for name, _ in stage_studies[keep_recent:]]

    deleted = 0
    for study_name in to_delete:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            deleted += 1
        except Exception:
            pass

    print(
        f"[CLEANUP] Stage cleanup complete: deleted={deleted}, "
        f"kept={total - deleted}, total_before={total}"
    )


def main():
    exit_code = 0
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of optimization trials (default: auto-set by mode)",
    )
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        default="UNIFIED",
        choices=["SCALP", "DAY", "SWING", "UNIFIED", "ALL"],
        help="Trading Mode: SCALP, DAY, SWING, or UNIFIED (recommended - auto-selects best timeframe)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated Optuna sampler seeds for multi-seed robustness (e.g., 13,37,73).",
    )
    args = parser.parse_args()

    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",")]
    mode = args.mode.upper()
    awfo_enabled = mode in AWFO_DEFAULTS["enabled_modes"]
    is_two_stage_mode = mode in {"UNIFIED", "ALL"}
    stage_keep_recent = int(os.getenv("OPTUNA_STAGE_KEEP_RECENT", "30"))
    seed_arg = args.seeds if args.seeds is not None else os.getenv("OPTUNA_SEEDS")
    if seed_arg is None:
        seed_arg = "13,37,73" if is_two_stage_mode else "13"
    seed_list = _parse_seed_list(seed_arg)
    if not seed_list:
        seed_list = [13]

    # Auto-set trials based on mode if not specified
    # AWFO-enabled UNIFIED defaults are reduced to keep runtime practical.
    MODE_TRIALS_MAP = {
        "SCALP": 3600,  # 3 timeframes (5m,15m,30m), high data volume but narrow param range
        "DAY": 4200,    # 3 timeframes (1h,2h,4h), balanced - most commonly used mode
        "SWING": 5000,  # 3 timeframes (4h,1d,3d), wide param range + low data volume (overfitting risk)
        "UNIFIED": 5600, # Multi-seed(3) + 2-stage robust search baseline
        "ALL": 5600,     # Alias for UNIFIED
    }

    if args.trials is None:
        trials = MODE_TRIALS_MAP.get(mode, 2500)
        print(f"[INFO]  Auto-setting trials for {mode} mode: {trials}")
    else:
        trials = args.trials
        print(f"[INFO]  Using custom trials: {trials}")

    # Adjust Logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("src.futures_strategy.engine_fast_futures").setLevel(
        logging.WARNING
    )

    # Get Search Space & Timeframes
    try:
        search_space = GET_SEARCH_SPACE(mode, market_type="futures")
        timeframes = search_space["TIMEFRAME"]["choices"]
    except Exception as e:
        print(f"[ERROR]: Failed to load search space for mode '{mode}'")
        print(f"   Details: {e}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"[MODE]: {mode} OPRIMIZATION")
    print(f"[TARGET] Timeframes: {timeframes}")
    # print(f"[INFO] Search Space Size: {len(search_space)} parameters")
    print(f"{'='*70}\n")

    data_maps = {}
    print(f"[INFO] Loading data for symbols: {', '.join(symbols)}")

    # [CRITICAL] Ensure '1d' is loaded for HTF Trend Filter, even if not optimizing on it
    loading_timeframes = list(set(timeframes + ['1d']))
    
    for symbol in symbols:
        print(f"Loading {symbol}...")
        data_maps[symbol] = load_all_timeframes(
            symbol, BACKTEST_START_DATE, BACKTEST_END_DATE, loading_timeframes
        )
        
        # [VALIDATION] Ensure all required data is loaded
        if not data_maps[symbol] or '1d' not in data_maps[symbol]:
            print(f"[ERROR]: Failed to load data for {symbol}")
            sys.exit(1)
        for tf in timeframes:
            if tf not in data_maps[symbol] or data_maps[symbol][tf].empty:
                print(f"[ERROR]: Failed to load {symbol}-{tf} data")
                sys.exit(1)
    print(f"[INFO] Data loaded successfully for all symbols")

    # [CRITICAL] Slice Data for Optimization (Train Set) with Warmup Buffer
    print(f"[INFO] Trimming Data for Optimization (Train Period: ~ {TRAIN_CUTOFF_DATE})")
    cutoff_ts = pd.Timestamp(TRAIN_CUTOFF_DATE)

    for sym in data_maps:
        for tf in data_maps[sym]:
            df = data_maps[sym][tf]
            original_len = len(df)
            
            # Find cutoff index (end of training period)
            cutoff_mask = df["datetime"] < cutoff_ts
            train_end_idx = cutoff_mask.sum()
            
            if train_end_idx == 0:
                print(f"[WARN]  Warning: {sym}-{tf} has no data before cutoff date.")
                continue
            
            # Desired warmup period
            desired_warmup = WARMUP_BUFFER_BARS.get(tf, 200)
            
            # Slice from start to cutoff (entire training period)
            data_maps[sym][tf] = df.iloc[:train_end_idx].copy()
            
            # Set warmup: first N bars are for indicator warmup, trading starts after
            data_maps[sym][tf].attrs['warmup_bars'] = min(desired_warmup, train_end_idx)
            
            new_len = len(data_maps[sym][tf])
            # if tf == timeframes[0]:
            #     print(f"  [{sym}] Train Size: {new_len} (Original: {original_len}, Warmup: {desired_warmup})")

    # [AWFO] Build baseline plan
    awfo_plan = {"enabled": False, "splits": {}, "min_trades_per_fold": None, "cache": {}}
    if awfo_enabled:
        awfo_plan = build_awfo_plan(
            data_maps,
            timeframes,
            folds=AWFO_DEFAULTS["folds"],
            min_trades=AWFO_DEFAULTS["min_trades_per_fold"],
        )
        awfo_plan["cache"] = build_awfo_runtime_cache(data_maps, timeframes, awfo_plan)
        total_cached_folds = sum(
            len(tf_folds)
            for sym_map in awfo_plan["cache"].values()
            for tf_folds in sym_map.values()
        )
        print(
            f"[AWFO] AWFO enabled: {AWFO_DEFAULTS['folds']} anchored folds "
            f"(min trades per fold={AWFO_DEFAULTS['min_trades_per_fold']})"
        )
        print(f"[INFO] AWFO runtime cache prepared: {total_cached_folds} fold segments")

    # [OPTIMIZATION] Pre-compute merge indices to eliminate pd.merge overhead
    print(f"[PREP] Pre-computing merge indices for fast data alignment...")
    merge_indices = compute_merge_indices(data_maps)
    print(f"[INFO] Merge indices computed for {len(merge_indices)} symbols")

    # DB Setup (MySQL)
    from dotenv import load_dotenv
    from urllib.parse import quote_plus

    load_dotenv()

    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "trading_optuna")

    if not all([db_user, db_pass, db_name]):
        print("[ERROR]: Missing DB credentials in .env (DB_USER, DB_PASS, DB_NAME)")
        sys.exit(1)

    study_name = f"futures_{mode.lower()}_strategy"
    # [CRITICAL] Encode password to handle special characters like '@'
    safe_pass = quote_plus(db_pass)
    storage_url = f"mysql+pymysql://{db_user}:{safe_pass}@{db_host}:{db_port}/{db_name}"

    # [Clean Start] Intead of deleting file, delete study from DB
    print(f"[STUDY] Preparing study: {study_name}")
    
    # [VALIDATION] Test DB connection before proceeding
    try:
        test_storage = optuna.storages.RDBStorage(url=storage_url)
        print(f"[INFO] DB connection successful")
    except Exception as e:
        print(f"[ERROR]: Failed to connect to MySQL database")
        print(f"   Details: {e}")
        print(f"   Please check your .env credentials and ensure MySQL is running")
        sys.exit(1)
    
    if is_two_stage_mode:
        # 2-stage mode uses timestamped stage studies and publishes canonical alias later.
        # Skipping upfront delete avoids long DB lock when legacy study has many trials.
        print(f"[INFO]  Skip upfront delete for 2-stage mode: {study_name}")
    else:
        print(f"[CLEANUP] Cleaning old study (single-stage): {study_name}")
        try:
            optuna.delete_study(study_name=study_name, storage=storage_url)
            print(f"[OK] Deleted old study: {study_name}")
        except Exception:
            pass  # Study might not exist

    # [Performance] Optimize for parallel MySQL access
    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={
            "pool_size": max(30, args.jobs * 2),  # Scale with jobs
            "max_overflow": 10,  # Allow burst connections
            "pool_recycle": 3600,
            "pool_pre_ping": True,  # Validate connections
        },
    )

    print(f"\n{'='*70}")
    print(f"[RUN] STARTING OPTIMIZATION for {study_name}")
    print(f"[DB] Storage: MySQL ({db_host}/{db_name})")
    print(f"[TRIALS] Total Trials: {trials}")
    print(f"[AWFO] AWFO: {'ON' if awfo_enabled else 'OFF'}")
    print(f"[JOBS] Parallel Jobs: {args.jobs}")
    print(f"[SEEDS] {seed_list}")
    print(
        "[OBJECTIVE] "
        f"gate_floor={OBJECTIVE_CFG.gate_floor:.2f}, "
        f"min_trades_fut={OBJECTIVE_CFG.min_trades_futures}, "
        f"side_ratio_fut={OBJECTIVE_CFG.min_side_ratio_futures:.2f}, "
        f"asinh_clip={OBJECTIVE_CFG.asinh_clip:.2f}"
    )
    print(
        "[ROBUST] "
        f"fold_consistency_target={ROBUST_OBJECTIVE_DEFAULTS['fold_consistency_target']:.2f}, "
        f"side_target={ROBUST_OBJECTIVE_DEFAULTS['side_min_ratio_target']:.2f}, "
        f"cost_stress_per_trade={ROBUST_OBJECTIVE_DEFAULTS['cost_stress_per_trade_pct']:.3f}%"
    )
    print(f"{'='*70}\n")

    # [Performance] Numba JIT Warmup
    print("[RUN] Warming up Numba JIT...", end="", flush=True)
    dummy_len = 10
    _dummy_arr = np.ones(dummy_len, dtype=np.float64)
    _dummy_int = np.zeros(dummy_len, dtype=np.int64)
    _dummy_ts = np.zeros(dummy_len, dtype=np.int64)  # Timestamps
    try:
        backtest_loop_numba(
            _dummy_arr,
            _dummy_arr,
            _dummy_arr,  # OHLC (close, high, low)
            _dummy_arr,  # Open Prices [NEW]
            _dummy_arr,  # Volume Ratio
            _dummy_arr,
            _dummy_arr,  # Entry Upper/Lower
            _dummy_int,
            _dummy_int,
            _dummy_arr,  # Trend, Strength, ATR
            _dummy_arr,  # Parabolic SAR
            _dummy_arr,  # RSI
            _dummy_arr,  # [NEW] Hurst
            _dummy_arr,  # [NEW] NATR
            10000.0,
            1.0,
            0.001,
            0.001,  # Bal, Lev, Fee, Slip
            0,  # Exit Type (0=Trailing, 1=SAR)
            0,
            0.01,
            1.5,  # SL Type, Pct, Mult
            3.0,  # ATR Mult
            0.02,  # Risk
            False,
            1.0,  # Vol Filter
            False,
            3.0,  # TP
            _dummy_ts,
            0.0001,
            8,  # Funding params
            1000,
            0.0,  # Max Hold, Trailing Act
            0.5,  # Time-based Exit Profit Threshold
            80.0, # RSI Exit Threshold
            False, # [NEW] Use Dynamic Risk
            0.6,   # Strong Hurst
            1.5,   # Strong NATR
            1.5,   # Strong Multiplier
            0.55,  # Weak Hurst
            0.5,   # Weak Multiplier
            4.0,   # Panic NATR
            0.25,  # Panic Multiplier
            0,     # Warmup bars
            False, # [NEW] use_compounding
            1000000.0 # [NEW] max_capital_usage
        )
        print(" Done!")
    except Exception as e:
        print(f"\n[WARN]  Warning: Numba warmup failed: {e}")
        print(f"   First trial will be slower due to JIT compilation")

    study = None
    best_candidate_study = None
    best_candidate_value = -float("inf")
    best_candidate_label = ""

    try:
        if is_two_stage_mode:
            print("[2STAGE] 2-Stage + Multi-Fidelity mode enabled (UNIFIED)")
            cfg = TWO_STAGE_UNIFIED_DEFAULTS.copy()
            scale = max(0.4, trials / 2800.0)

            stage1_total = int(cfg["stage1_total_trials"] * scale)
            stage1_total = max(300, min(stage1_total, max(400, int(trials * 0.6))))
            stage2_total_budget = max(200, trials - stage1_total)

            promoted_structures = []
            stage1_space_base = build_stage1_search_space(search_space)
            top_pool_limit = max(12, cfg["stage2_top_structures"] * 3)

            print(
                f"   Stage1 trials: {stage1_total}, Stage2 budget: {stage2_total_budget}, "
                f"Target structures: {cfg['stage2_top_structures']}"
            )

            for fidelity in cfg["stage1_fidelity_steps"]:
                step_trials = int(stage1_total * fidelity["ratio"])
                if step_trials < 50:
                    continue

                step_symbols = max(1, min(fidelity["symbols"], len(symbols)))
                step_data = subset_data_maps(
                    data_maps,
                    symbols,
                    n_symbols=step_symbols,
                    data_ratio=fidelity["data_ratio"],
                )
                step_merge_indices = compute_merge_indices(step_data)
                step_awfo = build_awfo_plan(
                    step_data,
                    timeframes,
                    folds=fidelity["folds"],
                    min_trades=fidelity["min_trades"],
                )
                step_awfo["cache"] = build_awfo_runtime_cache(step_data, timeframes, step_awfo)

                step_space = restrict_stage1_space_by_candidates(
                    stage1_space_base, promoted_structures
                )
                step_study_name = (
                    f"{study_name}__s1_{fidelity['name']}_{int(time.time())}"
                )
                try:
                    optuna.delete_study(study_name=step_study_name, storage=storage)
                except Exception:
                    pass

                print(
                    f"\n[STAGE1] {fidelity['name'].upper()} | "
                    f"trials={step_trials}, symbols={step_symbols}, "
                    f"data_ratio={fidelity['data_ratio']}, folds={fidelity['folds']}, seeds={seed_list}"
                )
                seed_runs = run_seeded_studies(
                    study_name_prefix=step_study_name,
                    storage=storage,
                    n_trials=step_trials,
                    n_jobs=args.jobs,
                    startup_ratio=fidelity["startup_ratio"],
                    objective_fn=lambda t, _data=step_data, _space=step_space, _merge=step_merge_indices, _awfo=step_awfo: objective(
                        t,
                        UltimateStrategy,
                        f"Ultimate_{mode}",
                        _data,
                        _space,
                        {},
                        _merge,
                        _awfo,
                    ),
                    seeds=seed_list,
                )
                completed = []
                for seed, seed_trials, seed_study, step_startup in seed_runs:
                    seed_best = (
                        float(seed_study.best_value)
                        if len(seed_study.trials) > 0
                        else -float("inf")
                    )
                    seed_robust = _robust_value_from_study(seed_study)
                    print(
                        f"   [INFO] Stage1-{fidelity['name']} seed={seed} "
                        f"| trials={seed_trials} | startup={step_startup} "
                        f"| best={seed_best:.2f} | robust={seed_robust:.2f}"
                    )
                    if np.isfinite(seed_robust) and seed_robust > best_candidate_value:
                        best_candidate_value = float(seed_robust)
                        best_candidate_study = seed_study
                        best_candidate_label = f"{step_study_name}:seed{seed}"
                    completed.extend(_study_complete_trials(seed_study))
                if not completed:
                    continue
                completed.sort(key=lambda tr: float(tr.value), reverse=True)
                top_k = max(
                    cfg["stage2_top_structures"],
                    int(len(completed) * cfg["promotion_ratio"]),
                )
                top_k = min(top_k, top_pool_limit, len(completed))

                promoted = []
                seen = set()
                for tr in completed[:top_k]:
                    sig = extract_structure_signature(tr.params)
                    sig_key = tuple((k, sig.get(k)) for k in STRUCTURE_PARAM_KEYS)
                    if sig_key in seen or not sig:
                        continue
                    seen.add(sig_key)
                    promoted.append(sig)

                promoted_structures = promoted
                print(
                    f"   [PROMOTE] Promoted structures: {len(promoted_structures)} "
                    f"(top {top_k} trials)"
                )

            if not promoted_structures:
                print("[WARN] Stage1 produced no promoted structures; falling back to global search space.")
                promoted_structures = [extract_structure_signature({k: v["choices"][0] for k, v in stage1_space_base.items()})]

            promoted_structures = promoted_structures[:cfg["stage2_top_structures"]]

            per_structure_trials = max(
                80,
                min(
                    cfg["stage2_trials_per_structure"],
                    int(stage2_total_budget / max(1, len(promoted_structures))),
                ),
            )
            stage2_awfo = build_awfo_plan(
                data_maps,
                timeframes,
                folds=cfg["stage2_folds"],
                min_trades=cfg["stage2_min_trades"],
            )
            stage2_awfo["cache"] = build_awfo_runtime_cache(data_maps, timeframes, stage2_awfo)

            best_stage2_study = None
            best_stage2_value = -float("inf")

            for i, struct_sig in enumerate(promoted_structures, start=1):
                stage2_space = freeze_structure_in_space(search_space, struct_sig)
                step_study_name = f"{study_name}__s2_{i}_{int(time.time())}"

                print(
                    f"\n[STAGE2] Stage2-{i}/{len(promoted_structures)} | total_trials={per_structure_trials} | "
                    f"structure={struct_sig}"
                )
                pass1_trials = max(40, int(per_structure_trials * cfg["stage2_refine_ratio"]))
                pass2_trials = max(0, per_structure_trials - pass1_trials)

                stage2_seed_runs_p1 = run_seeded_studies(
                    study_name_prefix=f"{step_study_name}_p1",
                    storage=storage,
                    n_trials=pass1_trials,
                    n_jobs=args.jobs,
                    startup_ratio=cfg["stage2_startup_ratio"],
                    objective_fn=lambda t, _space=stage2_space: objective(
                        t,
                        UltimateStrategy,
                        f"Ultimate_{mode}",
                        data_maps,
                        _space,
                        {},
                        merge_indices,
                        stage2_awfo,
                    ),
                    seeds=seed_list,
                )
                stage2_completed = []
                struct_best_study = None
                struct_best_robust = -float("inf")
                for seed, seed_trials, seed_study, seed_startup in stage2_seed_runs_p1:
                    seed_best = (
                        float(seed_study.best_value)
                        if len(seed_study.trials) > 0
                        else -float("inf")
                    )
                    seed_robust = _robust_value_from_study(seed_study)
                    print(
                        f"   [INFO] Stage2-{i} pass1 seed={seed} "
                        f"| trials={seed_trials} | startup={seed_startup} "
                        f"| best={seed_best:.2f} | robust={seed_robust:.2f}"
                    )
                    stage2_completed.extend(_study_complete_trials(seed_study))
                    if np.isfinite(seed_robust) and seed_robust > struct_best_robust:
                        struct_best_robust = float(seed_robust)
                        struct_best_study = seed_study

                refined_space = stage2_space
                if stage2_completed and pass2_trials >= 30:
                    refined_space = build_adaptive_numeric_space(
                        stage2_space,
                        stage2_completed,
                        top_quantile=cfg["stage2_refine_top_quantile"],
                        min_width_ratio=cfg["stage2_refine_min_width_ratio"],
                        min_samples=cfg["stage2_refine_min_samples"],
                        min_step_span=cfg["stage2_refine_step_span"],
                    )
                    stage2_seed_runs_p2 = run_seeded_studies(
                        study_name_prefix=f"{step_study_name}_p2",
                        storage=storage,
                        n_trials=pass2_trials,
                        n_jobs=args.jobs,
                        startup_ratio=cfg["stage2_startup_ratio"],
                        objective_fn=lambda t, _space=refined_space: objective(
                            t,
                            UltimateStrategy,
                            f"Ultimate_{mode}",
                            data_maps,
                            _space,
                            {},
                            merge_indices,
                            stage2_awfo,
                        ),
                        seeds=seed_list,
                    )
                    for seed, seed_trials, seed_study, seed_startup in stage2_seed_runs_p2:
                        seed_best = (
                            float(seed_study.best_value)
                            if len(seed_study.trials) > 0
                            else -float("inf")
                        )
                        seed_robust = _robust_value_from_study(seed_study)
                        print(
                            f"   [INFO] Stage2-{i} pass2 seed={seed} "
                            f"| trials={seed_trials} | startup={seed_startup} "
                            f"| best={seed_best:.2f} | robust={seed_robust:.2f}"
                        )
                        stage2_completed.extend(_study_complete_trials(seed_study))
                        if np.isfinite(seed_robust) and seed_robust > struct_best_robust:
                            struct_best_robust = float(seed_robust)
                            struct_best_study = seed_study

                if not stage2_completed or struct_best_study is None:
                    continue

                structure_robust = _robust_value_from_trials(stage2_completed)
                print(
                    f"   [INFO] Stage2-{i} summary | robust={structure_robust:.2f} "
                    f"| complete_trials={len(stage2_completed)}"
                )
                if np.isfinite(structure_robust) and structure_robust > best_candidate_value:
                    best_candidate_value = float(structure_robust)
                    best_candidate_study = struct_best_study
                    best_candidate_label = f"{step_study_name}:robust"

                if structure_robust > best_stage2_value:
                    best_stage2_value = float(structure_robust)
                    best_stage2_study = struct_best_study
            if best_stage2_study is None:
                raise RuntimeError("2-Stage optimization failed to produce any complete Stage2 study.")

            # Publish final winner into standard study name for downstream compatibility.
            study = publish_best_trial_alias(
                storage=storage,
                target_study_name=study_name,
                source_study=best_stage2_study,
                source_label="stage2_winner",
            )
            if study is None:
                raise RuntimeError("Failed to publish final stage2 winner to canonical study.")

        else:
            startup_ratio = 0.22 if awfo_enabled else 0.20
            print(f"[SINGLE] Startup ratio: {startup_ratio:.2f}")
            seed_runs = run_seeded_studies(
                study_name_prefix=f"{study_name}__single",
                storage=storage,
                n_trials=trials,
                n_jobs=args.jobs,
                startup_ratio=startup_ratio,
                objective_fn=lambda t: objective(
                    t,
                    UltimateStrategy,
                    f"Ultimate_{mode}",
                    data_maps,
                    search_space,
                    {},
                    merge_indices,
                    awfo_plan,
                ),
                seeds=seed_list,
            )
            single_best_study = None
            single_best_seed = None
            single_best_robust = -float("inf")
            for seed, seed_trials, seed_study, seed_startup in seed_runs:
                seed_best = (
                    float(seed_study.best_value)
                    if len(seed_study.trials) > 0
                    else -float("inf")
                )
                seed_robust = _robust_value_from_study(seed_study)
                print(
                    f"   [INFO] Single-stage seed={seed} | trials={seed_trials} "
                    f"| startup={seed_startup} | best={seed_best:.2f} | robust={seed_robust:.2f}"
                )
                if np.isfinite(seed_robust) and seed_robust > single_best_robust:
                    single_best_robust = float(seed_robust)
                    single_best_study = seed_study
                    single_best_seed = seed
            if single_best_study is None:
                raise RuntimeError("Single-stage optimization failed to produce any complete seed study.")

            if single_best_robust > best_candidate_value:
                best_candidate_value = single_best_robust
                best_candidate_study = single_best_study
                best_candidate_label = f"{study_name}:single_seed_{single_best_seed}"

            study = publish_best_trial_alias(
                storage=storage,
                target_study_name=study_name,
                source_study=single_best_study,
                source_label=f"single_seed_{single_best_seed}",
            )
            if study is None:
                raise RuntimeError("Failed to publish single-stage winner to canonical study.")
            print(
                f"[INFO] Single-stage done | seed={single_best_seed} "
                f"| robust={single_best_robust:.2f} | best={study.best_value:.2f}"
            )

    except KeyboardInterrupt:
        print("\n[STOP] Optimization Interrupted by User")
        exit_code = 130
        if study is not None:
            print(f"[INFO] Progress saved: {len(study.trials)} trials completed")
    except Exception as e:
        print(f"\n[ERROR] Optimization failed with error: {e}")
        exit_code = 1
        # Safety net for 2-stage runs: publish best completed intermediate study
        # to canonical name so verify can still find a usable result.
        if study is None and best_candidate_study is not None:
            try:
                published = publish_best_trial_alias(
                    storage=storage,
                    target_study_name=study_name,
                    source_study=best_candidate_study,
                    source_label=f"fallback:{best_candidate_label}",
                )
                if published is not None:
                    study = published
                    print(
                        f"[FALLBACK] Fallback published to '{study_name}' from "
                        f"'{best_candidate_label}' (best={study.best_value:.2f})"
                    )
            except Exception as pub_e:
                print(f"[WARN] Fallback publish failed: {pub_e}")
        if study is not None:
            print(f"[INFO] Progress saved: {len(study.trials)} trials completed before failure")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    if exit_code == 0:
        print(f"[INFO] {mode} Optimization Complete!")
    elif exit_code == 130:
        print(f"[INTERRUPTED] {mode} Optimization Interrupted.")
    else:
        print(f"[INFO] {mode} Optimization Failed.")

    if study is not None and len(study.trials) > 0:
        print(f"[BEST] Best Score: {study.best_value:.2f}")
        print(f"[INFO] Best Params: {study.best_params}")
        
        # [NEW] Detailed Report for TRAIN Period
        print(f"\n{'='*70}")
        print(f"[TRAIN] TRAIN PERIOD PERFORMANCE (Best Strategy)")
        print(f"{'='*70}")
        
        best_params = study.best_params
        # Merge fixed params if any (handled inside objective but good to be safe)
        
        selected_tf = best_params.get('TIMEFRAME', '1h')
        
        for symbol in symbols:
            if selected_tf not in data_maps[symbol]:
                continue
                
            hourly_df = data_maps[symbol][selected_tf]
            daily_df = data_maps[symbol]['1d']
            
            # Re-create Strategy & Engine
            strategy = UltimateStrategy(f"Best_{symbol}", best_params)
            current_merge_index = None
            if merge_indices and symbol in merge_indices and selected_tf in merge_indices[symbol]:
                current_merge_index = merge_indices[symbol][selected_tf]
            engine = BacktestEngineFast(
                hourly_df,
                daily_df,
                strategy,
                initial_balance=FUTURES_INITIAL_BALANCE,
                merge_index_map=current_merge_index,
            )
                
            engine.leverage = best_params.get('LEVERAGE', 1)
            engine.risk_per_trade = best_params.get('RISK_PER_TRADE', 0.02)
            
            try:
                res = engine.run()
                
                ret = res['total_return_pct']
                mdd = res['mdd_pct']
                cnt = res['total_trades']
                win = res['win_rate']
                
                trades_df = res['trades_df']
                pf = 0.0
                if not trades_df.empty and 'pnl' in trades_df.columns:
                    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
                    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
                    pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
                
                print(f"   - {symbol:<9} : Return {ret:>7.2f}% | MDD {mdd:>6.2f}% | Trades {cnt:>3} | Win {win:>5.1f}% | PF {pf:.2f}")
                
            except Exception as e:
                print(f"   - {symbol:<9} : Error calculating performance: {e}")

    # Housekeeping: keep DB compact by pruning old 2-stage temporary studies.
    if is_two_stage_mode:
        cleanup_old_stage_studies(
            storage=storage,
            base_study_name=study_name,
            keep_recent=stage_keep_recent,
        )

    print(f"{'='*70}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

