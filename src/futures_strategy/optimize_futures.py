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
from src.optimization.opt_utils import suggest_params, calculate_score

# Shared constants
DAILY_BUFFER_DAYS = 200

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
}


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
    if "date_key" not in hourly_df.columns:
        hourly_keys = pd.to_datetime(hourly_df["datetime"]).dt.strftime("%Y-%m-%d").values
    else:
        hourly_keys = hourly_df["date_key"].values

    if "date_key" not in daily_df.columns:
        daily_keys = pd.to_datetime(daily_df["datetime"]).dt.strftime("%Y-%m-%d").values
    else:
        daily_keys = daily_df["date_key"].values

    date_to_daily_idx = {date_key: idx for idx, date_key in enumerate(daily_keys)}
    merge_index = np.array(
        [date_to_daily_idx.get(date_key, -1) for date_key in hourly_keys],
        dtype=np.int32,
    )
    if np.any(merge_index == -1):
        merge_index[merge_index == -1] = 0
    return merge_index


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
        print(f"❌ Error: Failed to load {symbol}-1d data: {e}")
        sys.exit(1)

    for tf in timeframes:
        try:
            df = collector.ensure_data(symbol, tf, start_date, end_date)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["date_key"] = df["datetime"].dt.strftime("%Y-%m-%d")
            data_map[tf] = df
        except Exception as e:
            print(f"❌ Error: Failed to load {symbol}-{tf} data: {e}")
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
        
        # Get daily date_key mapping
        daily_df = data_map['1d']
        daily_date_keys = daily_df['date_key'].values
        
        # Create a lookup dict: date_key -> daily_index
        date_to_daily_idx = {date_key: idx for idx, date_key in enumerate(daily_date_keys)}
        
        # For each timeframe, create the merge index
        for tf, tf_df in data_map.items():
            if tf == '1d':
                continue  # Skip daily itself
            
            # Map each hourly bar's date_key to its daily index
            hourly_date_keys = tf_df['date_key'].values
            merge_index = np.array([
                date_to_daily_idx.get(date_key, -1) 
                for date_key in hourly_date_keys
            ], dtype=np.int32)
            
            # Handle missing mappings (shouldn't happen with proper data)
            # Replace -1 with 0 and issue warning if found
            if np.any(merge_index == -1):
                print(f"⚠️  Warning: {symbol}-{tf} has unmapped date_keys. Using fallback index 0.")
                merge_index[merge_index == -1] = 0
            
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
    import gc

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
    awfo_min_trades = awfo_plan.get("min_trades_per_fold", 40) if awfo_enabled else None

    symbol_scores = []
    symbol_results = {}
    report_step = 0

    for symbol, data_map in data_maps.items():
        if selected_tf not in data_map:
            return -10000

        hourly_df = data_map[selected_tf]
        daily_df = data_map["1d"]

        # ===== Path A: AWFO (anchored OOS folds) =====
        if awfo_enabled:
            awfo_splits = awfo_splits_by_symbol.get(symbol, {}).get(selected_tf, [])
            if len(awfo_splits) < 2:
                return -10000

            fold_scores = []
            fold_returns = []
            fold_mdds = []
            fold_pfs = []
            warmup_buffer = WARMUP_BUFFER_BARS.get(selected_tf, 200)

            for fold_idx, (test_start, test_end) in enumerate(awfo_splits):
                seg_start = max(0, test_start - warmup_buffer)
                segment_hourly = hourly_df.iloc[seg_start:test_end].copy()
                if len(segment_hourly) < 100:
                    continue

                warmup_bars = test_start - seg_start
                segment_hourly.attrs["warmup_bars"] = warmup_bars

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

                strategy = strategy_cls(f"{strategy_name}_{symbol}_F{fold_idx+1}", full_params)
                segment_merge_index = compute_segment_merge_index(segment_hourly, segment_daily)
                engine = BacktestEngineFast(
                    segment_hourly,
                    segment_daily,
                    strategy,
                    initial_balance=FUTURES_INITIAL_BALANCE,
                    merge_index_map=segment_merge_index,
                )
                engine.leverage = full_params.get("LEVERAGE", 1)
                engine.risk_per_trade = full_params.get("RISK_PER_TRADE", 0.02)

                try:
                    result = engine.run()
                except Exception as e:
                    print(f"⚠️ AWFO fold backtest failed for {symbol} (fold {fold_idx+1}): {e}")
                    import traceback
                    traceback.print_exc()
                    del engine, strategy
                    gc.collect()
                    return -10000

                trades_df = result["trades_df"]
                fold_score = -10000.0

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

                        if np.isfinite(fold_score):
                            fold_returns.append(float(fold_ret))
                            fold_mdds.append(float(fold_mdd))
                            gross_profit = oos_trades[oos_trades["pnl"] > 0]["pnl"].sum()
                            gross_loss = abs(oos_trades[oos_trades["pnl"] < 0]["pnl"].sum())
                            fold_pf = (
                                gross_profit / gross_loss
                                if gross_loss > 0
                                else (gross_profit if gross_profit > 0 else 0.0)
                            )
                            fold_pfs.append(float(fold_pf))

                fold_scores.append(float(fold_score) if np.isfinite(fold_score) else -10000.0)

                del engine, result, strategy, trades_df
                report_step += 1
                running_fold_score = float(np.mean(fold_scores))
                trial.report(running_fold_score, report_step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if len(fold_scores) < 2:
                return -10000

            avg_fold_score = float(np.mean(fold_scores))
            worst_fold_score = float(np.min(fold_scores))
            consistency = (
                sum(score > 0 for score in fold_scores) / len(fold_scores)
                if fold_scores
                else 0.0
            )

            final_symbol_score = (avg_fold_score * 0.7) + (worst_fold_score * 0.3)
            if consistency < 0.5:
                final_symbol_score -= (0.5 - consistency) * 80.0
            if fold_returns and min(fold_returns) < -20.0:
                final_symbol_score -= 150.0

            symbol_scores.append(final_symbol_score)
            symbol_results[symbol] = {
                "return": float(np.mean(fold_returns)) if fold_returns else -100.0,
                "mdd": float(np.mean(fold_mdds)) if fold_mdds else 0.0,
                "trades": 0,
                "win_rate": 0.0,
                "pf": float(np.mean(fold_pfs)) if fold_pfs else 0.0,
            }

        # ===== Path B: Single full-run (legacy / non-AWFO) =====
        else:
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

            try:
                result = engine.run()
            except Exception as e:
                print(f"⚠️ Backtest failed for {symbol}: {e}")
                import traceback
                traceback.print_exc()
                del engine, strategy
                gc.collect()
                return -10000

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
            if score < -50:
                del engine, result, strategy, trades_df
                gc.collect()
                return -10000

            symbol_scores.append(score)
            symbol_results[symbol] = {
                "return": ret,
                "mdd": mdd,
                "trades": trades,
                "win_rate": win_rate,
                "pf": pf,
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
        gc.collect()
        return -10000

    ret_values = [float(r["return"]) for r in symbol_results.values()]
    mdd_values_raw = [float(r["mdd"]) for r in symbol_results.values()]
    mdd_values_abs = [abs(v) for v in mdd_values_raw]
    pf_values = [float(r["pf"]) for r in symbol_results.values()]

    avg_ret = float(np.mean(ret_values))
    avg_mdd_raw = float(np.mean(mdd_values_raw))
    avg_mdd_abs = float(np.mean(mdd_values_abs))
    max_mdd_abs = float(np.max(mdd_values_abs))
    avg_pf = float(np.mean(pf_values))

    # Hard safety guard: reject pathological drawdown profiles.
    if max_mdd_abs > 65.0 or avg_mdd_abs > 52.0:
        gc.collect()
        return -10000

    offset = 240.0
    shifted_scores = np.array(symbol_scores, dtype=np.float64) + offset
    if np.any(shifted_scores <= 1e-9):
        final_score = -10000
    else:
        harmonic_mean = float(len(shifted_scores) / np.sum(1.0 / shifted_scores))
        geometric_mean = float(np.exp(np.mean(np.log(shifted_scores))))
        log_dispersion = float(np.std(np.log(shifted_scores)))
        dispersion_penalty = float(np.exp(-0.22 * log_dispersion))

        # Geometric-first blend: keeps downside discipline while reducing over-conservatism.
        blended_shifted = (0.35 * harmonic_mean) + (0.65 * geometric_mean)

        # Mild efficiency encouragement with bounded transforms.
        efficiency_boost = (
            10.0 * np.tanh(avg_ret / 45.0)
            + 4.0 * np.tanh((avg_pf - 1.1) / 0.9)
        )

        # Soft drawdown penalty on top of hard guard.
        soft_mdd_penalty = (
            max(0.0, avg_mdd_abs - 32.0) * 1.4
            + max(0.0, max_mdd_abs - 45.0) * 2.2
        )

        final_score = (blended_shifted * dispersion_penalty) - offset
        final_score += efficiency_boost
        final_score -= soft_mdd_penalty
        if not np.isfinite(final_score):
            final_score = -10000

    trial.set_user_attr("return_avg", avg_ret)
    trial.set_user_attr("mdd_avg", avg_mdd_raw)
    trial.set_user_attr("mdd_avg_abs", avg_mdd_abs)
    trial.set_user_attr("pf_avg", avg_pf)

    gc.collect()
    return final_score


def run_optuna_study(
    study_name,
    storage,
    n_trials,
    n_jobs,
    startup_ratio,
    objective_fn,
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


def main():
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
    args = parser.parse_args()

    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",")]
    mode = args.mode.upper()
    awfo_enabled = mode in AWFO_DEFAULTS["enabled_modes"]

    # Auto-set trials based on mode if not specified
    # AWFO-enabled UNIFIED defaults are reduced to keep runtime practical.
    MODE_TRIALS_MAP = {
        "SCALP": 3600,  # 3 timeframes (5m,15m,30m), high data volume but narrow param range
        "DAY": 4200,    # 3 timeframes (1h,2h,4h), balanced - most commonly used mode
        "SWING": 5000,  # 3 timeframes (4h,1d,3d), wide param range + low data volume (overfitting risk)
        "UNIFIED": 2800, # AWFO(3 folds) 기준 시간/탐색 균형
        "ALL": 2800,     # Alias for UNIFIED
    }

    if args.trials is None:
        trials = MODE_TRIALS_MAP.get(mode, 2500)
        print(f"ℹ️  Auto-setting trials for {mode} mode: {trials}")
    else:
        trials = args.trials
        print(f"ℹ️  Using custom trials: {trials}")

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
        print(f"❌ Error: Failed to load search space for mode '{mode}'")
        print(f"   Details: {e}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"🚀 MODE: {mode} OPRIMIZATION")
    print(f"⏰ Target Timeframes: {timeframes}")
    # print(f"🔍 Search Space Size: {len(search_space)} parameters")
    print(f"{'='*70}\n")

    data_maps = {}
    print(f"📡 Loading data for symbols: {', '.join(symbols)}")

    # [CRITICAL] Ensure '1d' is loaded for HTF Trend Filter, even if not optimizing on it
    loading_timeframes = list(set(timeframes + ['1d']))
    
    for symbol in symbols:
        print(f"Loading {symbol}...")
        data_maps[symbol] = load_all_timeframes(
            symbol, BACKTEST_START_DATE, BACKTEST_END_DATE, loading_timeframes
        )
        
        # [VALIDATION] Ensure all required data is loaded
        if not data_maps[symbol] or '1d' not in data_maps[symbol]:
            print(f"❌ Error: Failed to load data for {symbol}")
            sys.exit(1)
        for tf in timeframes:
            if tf not in data_maps[symbol] or data_maps[symbol][tf].empty:
                print(f"❌ Error: Failed to load {symbol}-{tf} data")
                sys.exit(1)
    print(f"✅ Data loaded successfully for all symbols")

    # [CRITICAL] Slice Data for Optimization (Train Set) with Warmup Buffer
    print(f"✂️  Trimming Data for Optimization (Train Period: ~ {TRAIN_CUTOFF_DATE})")
    cutoff_ts = pd.Timestamp(TRAIN_CUTOFF_DATE)

    for sym in data_maps:
        for tf in data_maps[sym]:
            df = data_maps[sym][tf]
            original_len = len(df)
            
            # Find cutoff index (end of training period)
            cutoff_mask = df["datetime"] < cutoff_ts
            train_end_idx = cutoff_mask.sum()
            
            if train_end_idx == 0:
                print(f"⚠️  Warning: {sym}-{tf} has no data before cutoff date.")
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
    awfo_plan = {"enabled": False, "splits": {}, "min_trades_per_fold": None}
    if awfo_enabled:
        awfo_plan = build_awfo_plan(
            data_maps,
            timeframes,
            folds=AWFO_DEFAULTS["folds"],
            min_trades=AWFO_DEFAULTS["min_trades_per_fold"],
        )
        print(
            f"🧭 AWFO enabled: {AWFO_DEFAULTS['folds']} anchored folds "
            f"(min trades per fold={AWFO_DEFAULTS['min_trades_per_fold']})"
        )

    # [OPTIMIZATION] Pre-compute merge indices to eliminate pd.merge overhead
    print(f"🔗 Pre-computing merge indices for fast data alignment...")
    merge_indices = compute_merge_indices(data_maps)
    print(f"✅ Merge indices computed for {len(merge_indices)} symbols")

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
        print("❌ Error: Missing DB credentials in .env (DB_USER, DB_PASS, DB_NAME)")
        sys.exit(1)

    study_name = f"futures_{mode.lower()}_strategy"
    # [CRITICAL] Encode password to handle special characters like '@'
    safe_pass = quote_plus(db_pass)
    storage_url = f"mysql+pymysql://{db_user}:{safe_pass}@{db_host}:{db_port}/{db_name}"

    # [Clean Start] Intead of deleting file, delete study from DB
    print(f"🔄 Preparing study: {study_name}")
    
    # [VALIDATION] Test DB connection before proceeding
    try:
        test_storage = optuna.storages.RDBStorage(url=storage_url)
        print(f"✅ DB connection successful")
    except Exception as e:
        print(f"❌ Error: Failed to connect to MySQL database")
        print(f"   Details: {e}")
        print(f"   Please check your .env credentials and ensure MySQL is running")
        sys.exit(1)
    
    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
        print(f"🗑️  Deleted old study: {study_name}")
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
    print(f"🔥 STARTING OPTIMIZATION for {study_name}")
    print(f"🛢️  Storage: MySQL ({db_host}/{db_name})")
    print(f"📈 Total Trials: {trials}")
    print(f"🧭 AWFO: {'ON' if awfo_enabled else 'OFF'}")
    print(f"💻 Parallel Jobs: {args.jobs}")
    print(f"{'='*70}\n")

    # [Performance] Numba JIT Warmup
    print("🔥 Warming up Numba JIT...", end="", flush=True)
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
        print(f"\n⚠️  Warning: Numba warmup failed: {e}")
        print(f"   First trial will be slower due to JIT compilation")

    study = None
    is_two_stage_mode = mode in {"UNIFIED", "ALL"}

    try:
        if is_two_stage_mode:
            print("🧠 2-Stage + Multi-Fidelity mode enabled (UNIFIED)")
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
                    f"\n🧪 Stage1-{fidelity['name'].upper()} | "
                    f"trials={step_trials}, symbols={step_symbols}, "
                    f"data_ratio={fidelity['data_ratio']}, folds={fidelity['folds']}"
                )
                step_study, step_startup = run_optuna_study(
                    study_name=step_study_name,
                    storage=storage,
                    n_trials=step_trials,
                    n_jobs=args.jobs,
                    startup_ratio=fidelity["startup_ratio"],
                    objective_fn=lambda t: objective(
                        t,
                        UltimateStrategy,
                        f"Ultimate_{mode}",
                        step_data,
                        step_space,
                        {},
                        step_merge_indices,
                        step_awfo,
                    ),
                )
                print(
                    f"   ✅ Stage1-{fidelity['name']} done | startup={step_startup} "
                    f"| best={step_study.best_value:.2f}"
                )

                completed = [
                    tr for tr in step_study.trials
                    if tr.state == optuna.trial.TrialState.COMPLETE
                ]
                if not completed:
                    continue

                completed.sort(key=lambda tr: tr.value, reverse=True)
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
                    f"   🔼 Promoted structures: {len(promoted_structures)} "
                    f"(top {top_k} trials)"
                )

            if not promoted_structures:
                print("⚠️ Stage1 produced no promoted structures; falling back to global search space.")
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

            best_stage2_study = None
            best_stage2_value = -float("inf")

            for i, struct_sig in enumerate(promoted_structures, start=1):
                stage2_space = freeze_structure_in_space(search_space, struct_sig)
                step_study_name = f"{study_name}__s2_{i}_{int(time.time())}"
                try:
                    optuna.delete_study(study_name=step_study_name, storage=storage)
                except Exception:
                    pass

                print(
                    f"\n🎯 Stage2-{i}/{len(promoted_structures)} | trials={per_structure_trials} | "
                    f"structure={struct_sig}"
                )
                s2_study, s2_startup = run_optuna_study(
                    study_name=step_study_name,
                    storage=storage,
                    n_trials=per_structure_trials,
                    n_jobs=args.jobs,
                    startup_ratio=cfg["stage2_startup_ratio"],
                    objective_fn=lambda t: objective(
                        t,
                        UltimateStrategy,
                        f"Ultimate_{mode}",
                        data_maps,
                        stage2_space,
                        {},
                        merge_indices,
                        stage2_awfo,
                    ),
                )
                print(
                    f"   ✅ Stage2-{i} done | startup={s2_startup} | best={s2_study.best_value:.2f}"
                )

                if s2_study.best_value > best_stage2_value:
                    best_stage2_value = s2_study.best_value
                    best_stage2_study = s2_study

            if best_stage2_study is None:
                raise RuntimeError("2-Stage optimization failed to produce any complete Stage2 study.")

            # Publish final winner into standard study name for deployment compatibility.
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
            except Exception:
                pass
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(n_startup_trials=1),
            )
            best_trial = best_stage2_study.best_trial
            frozen_trial = optuna.trial.create_trial(
                params=best_trial.params,
                distributions=best_trial.distributions,
                value=best_trial.value,
                user_attrs=best_trial.user_attrs,
            )
            study.add_trial(frozen_trial)

        else:
            startup_ratio = 0.22 if awfo_enabled else 0.20
            print(f"🧪 Single-stage startup ratio: {startup_ratio:.2f}")
            study, n_startup_trials = run_optuna_study(
                study_name=study_name,
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
            )
            print(f"✅ Single-stage done | startup={n_startup_trials} | best={study.best_value:.2f}")

    except KeyboardInterrupt:
        print("\n🛑 Optimization Interrupted by User")
        if study is not None:
            print(f"💾 Progress saved: {len(study.trials)} trials completed")
    except Exception as e:
        print(f"\n❌ Optimization failed with error: {e}")
        if study is not None:
            print(f"💾 Progress saved: {len(study.trials)} trials completed before failure")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"✅ {mode} Optimization Complete!")

    if study is not None and len(study.trials) > 0:
        print(f"🏆 Best Score: {study.best_value:.2f}")
        print(f"✨ Best Params: {study.best_params}")
        
        # [NEW] Detailed Report for TRAIN Period
        print(f"\n{'='*70}")
        print(f"📊 TRAIN PERIOD PERFORMANCE (Best Strategy)")
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

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
