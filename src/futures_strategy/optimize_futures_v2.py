import argparse
import logging
import os
import sys
import gc
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Project Root Setup
project_root: str = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.futures_strategy.data_collector import DataCollector
from src.futures_strategy.strategies_futures import UltimateStrategy
from src.futures_strategy.engine_fast_futures import BacktestEngineFast, backtest_loop_numba
from config.settings import (
    FUTURES_INITIAL_BALANCE,
    TRADING_FEE_RATE,
    SLIPPAGE_RATE,
    DATA_DIR,
)
from config.opt_config import OPT_V2_CONFIG, SEARCH_SPACE_V2

# Reuse merge index util from v1 (no cached signals: v2 uses full daily signals once)
from src.futures_strategy.optimize_futures import compute_segment_merge_index
from src.futures_strategy.funding_utils import merge_funding_into_ohlcv

# Importing from the new modular structure
from src.futures_strategy.opt_v2_utils.metrics import calc_romad, calc_romad_from_metrics
from src.futures_strategy.opt_v2_utils.cv_utils import build_anchored_folds
from src.futures_strategy.opt_v2_utils.opt_params import suggest_params_v2
from src.futures_strategy.opt_v2_utils.db_utils import save_study_to_sqlite, fast_reset_study
from src.futures_strategy.opt_v2_utils.evaluator import objective_v2, evaluate_symbol_fold

import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Logger setup
# --------------------------------------------------------------------------
# Set Optuna logging level to WARNING to hide default per-trial logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
_logger: logging.Logger = logging.getLogger("opt_v2")

# Constants
START_DATE: str = "2022-01-01"
END_DATE: str = "2025-11-01"


# --------------------------------------------------------------------------
# Execution Main
# --------------------------------------------------------------------------
def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT")
    parser.add_argument("--trials", type=int, default=OPT_V2_CONFIG["total_trials"])
    parser.add_argument("--jobs", type=int, default=OPT_V2_CONFIG["n_jobs"])
    parser.add_argument("--test", action="store_true", help="Load best study from DB and evaluate without optimizing")
    args: argparse.Namespace = parser.parse_args()

    symbols: List[str] = [s.strip() for s in args.symbols.split(",")]
    oos_symbols: List[str] = ["ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]

    collector: DataCollector = DataCollector()
    data_maps: Dict[str, Dict[str, Any]] = {}

    _logger.info("Loading target futures database (%s to %s)...", START_DATE, END_DATE)
    for sym in symbols:
        data_maps[sym] = {}
        for tf in ["1d", "4h"]:
            df: pd.DataFrame = collector.collect_and_save(sym, tf, START_DATE, END_DATE)
            if df.empty:
                _logger.error("Failed to load %s %s data", sym, tf)
                sys.exit(1)
            df = merge_funding_into_ohlcv(sym, df, DATA_DIR)
            data_maps[sym][tf] = df
            
        data_maps[sym]["merge_idx_1d"] = compute_segment_merge_index(data_maps[sym]["1d"], data_maps[sym]["1d"])
        data_maps[sym]["merge_idx_4h"] = compute_segment_merge_index(data_maps[sym]["4h"], data_maps[sym]["1d"])
    _logger.info("Target Data load complete.")

    study_name: str = "futures_v2_romad_opt"
    
    db_user: str = os.getenv("DB_USER", "root")
    db_pass: str = os.getenv("DB_PASS", "1234")
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: str = os.getenv("DB_PORT", "3306")
    db_name: str = os.getenv("DB_NAME", "trading_optuna")
    from urllib.parse import quote_plus
    safe_pass: str = quote_plus(db_pass)
    storage_url: str = f"mysql+pymysql://{db_user}:{safe_pass}@{db_host}:{db_port}/{db_name}"

    seeds: List[int] = OPT_V2_CONFIG["seeds"]
    n_trials: int = args.trials

    q_seeds: List[int]
    base_trials: int
    if n_trials <= len(seeds):
        q_seeds = seeds[:n_trials]
        base_trials = n_trials
    else:
        q_seeds = seeds
        base_trials = n_trials

    _logger.info("Starting V2 Optimization. Total Trials: %d, Seeds: %s, Workers: %d", base_trials, q_seeds, args.jobs)

    sampler: optuna.samplers.TPESampler = optuna.samplers.TPESampler(
        n_startup_trials=OPT_V2_CONFIG["n_startup_trials"],
        multivariate=True,
        constant_liar=True,
        warn_independent_sampling=False,
        seed=q_seeds[0],
    )

    if args.test:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            _logger.info(f"Loaded existing study '{study_name}' for testing.")
        except Exception as e:
            _logger.error(f"Failed to load study '{study_name}': {e}")
            sys.exit(1)
    else:
        # Fast-path: bypass Optuna's slow ORM cascade deletion via direct raw SQL.
        # Falls back to optuna.delete_study() automatically if direct SQL fails.
        deleted_fast: bool = fast_reset_study(
            study_name=study_name,
            db_user=db_user,
            db_pass=db_pass,
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
        )
        if not deleted_fast:
            try:
                optuna.delete_study(study_name=study_name, storage=storage_url)
                _logger.info(f"Deleted existing study '{study_name}' for a fresh start (ORM fallback).")
            except KeyError:
                pass

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="maximize",
            sampler=sampler,
            load_if_exists=False,
        )

        study.optimize(
            lambda t: objective_v2(t, data_maps, symbols),
            n_trials=base_trials,
            n_jobs=args.jobs,
            catch=(Exception,),
            show_progress_bar=True,  
        )

    SEP_WIDTH: int = 70
    _logger.info("=" * SEP_WIDTH)
    _logger.info("  🚀 Optimization Complete")
    _logger.info("=" * SEP_WIDTH)
    best_trial: optuna.trial.FrozenTrial = study.best_trial
    # 1. Best Parameters
    _logger.info("  [Best Parameters]")
    best_params: Dict[str, Any] = best_trial.params.copy()
    tf_val: str = str(best_params.get("TIMEFRAME", "4h"))
    if "MAX_HOLDING_BARS_4H" in best_params and "MAX_HOLDING_BARS_1D" in best_params:
        best_params["MAX_HOLDING_BARS"] = best_params["MAX_HOLDING_BARS_4H"] if tf_val == "4h" else best_params["MAX_HOLDING_BARS_1D"]
        best_params.pop("MAX_HOLDING_BARS_4H", None)
        best_params.pop("MAX_HOLDING_BARS_1D", None)

    for k, v in best_params.items():
        if isinstance(v, float):
            _logger.info(f"  - {k:26s}: {v:.6f}")
        else:
            _logger.info(f"  - {k:26s}: {v}")
    
    _logger.info("-" * SEP_WIDTH)
    _logger.info(f"  ⭐ Best Score: {best_trial.value:.4f}")
    _logger.info("-" * SEP_WIDTH)
    
    # 2. Target Symbol Performance
    _logger.info("  [Target Performance: %s]", ", ".join(symbols))
    header: str = f"| {'Symbol':<10} | {'Ret(%)':>8} | {'MDD(%)':>8} | {'Trd':>5} | {'Win(%)':>8} | {'Score':>7} |"
    _logger.info(header)
    _logger.info("|" + "-" * (len(header)-2) + "|")
    
    for sym in symbols:
        r: float = float(best_trial.user_attrs.get(f"{sym}_ret", 0.0))
        m: float = float(best_trial.user_attrs.get(f"{sym}_mdd", 0.0))
        s_val: float = float(best_trial.user_attrs.get(f"{sym}_score", 0.0))
        t: float = float(best_trial.user_attrs.get(f"{sym}_trades", 0.0))
        w: float = float(best_trial.user_attrs.get(f"{sym}_win_rate", 0.0))
        _logger.info(f"| {sym:<10} | {r:8.2f} | {m:8.2f} | {int(t):5d} | {w:8.2f} | {s_val:7.2f} |")
    _logger.info("-" * SEP_WIDTH)

    # 3. OOS Verification
    _logger.info("  🔍 Running OOS Cross-Symbol Verification (CV-Fold Mode)...")
    from src.futures_strategy.opt_v2_utils.evaluator import EMBARGO_BARS

    for sym in oos_symbols:
        if sym in data_maps:
            continue
        data_maps[sym] = {}
        tf_list: List[str] = list(set(["1d", tf_val]))
        for tf in tf_list:
            oos_df: pd.DataFrame = collector.collect_and_save(sym, tf, START_DATE, END_DATE)
            if not oos_df.empty:
                oos_df = merge_funding_into_ohlcv(sym, oos_df, DATA_DIR)
                data_maps[sym][tf] = oos_df
        if "1d" in data_maps[sym] and tf_val in data_maps[sym]:
            data_maps[sym][f"merge_idx_{tf_val}"] = compute_segment_merge_index(data_maps[sym][tf_val], data_maps[sym]["1d"])

    _logger.info("  [OOS Verification Summary (Averaged Over Folds)]")
    strategy_oos: UltimateStrategy = UltimateStrategy(name="FuturesV2_OOS", params=best_params)
    
    _logger.info(header.replace("Score", "RoMaD"))
    _logger.info("|" + "-" * (len(header)-2) + "|")

    for sym in oos_symbols:
        target_df: Any = data_maps[sym].get(tf_val)
        daily_df: Any = data_maps[sym].get("1d")
        full_merge_idx: Any = data_maps[sym].get(f"merge_idx_{tf_val}")
        
        if target_df is None or daily_df is None or full_merge_idx is None:
            _logger.info(f"| {sym:<10} | {'N/A':>8} | {'N/A':>8} | {'N/A':>5} | {'N/A':>8} | {'N/A':>7} |")
            continue
            
        try:
            folds_oos: List[Tuple[int, int, int]] = build_anchored_folds(target_df, n_folds=3, embargo=EMBARGO_BARS.get(tf_val, 0))
            if not folds_oos:
                _logger.info(f"| {sym:<10} | {'NOFOLD':>8} | {'NOFOLD':>8} | {'NOFOLD':>5} | {'NOFOLD':>8} | {'NOFOLD':>7} |")
                continue

            f_rets: List[float] = []
            f_mdds: List[float] = []
            f_trds: List[float] = []
            f_wins: List[float] = []
            f_scrs: List[float] = []

            for train_end, test_start, test_end in folds_oos:
                # Truncate daily_df for current fold
                tf_idx_test_end_minus_1: int = test_end - 1
                daily_end_idx: int = int(full_merge_idx[tf_idx_test_end_minus_1]) if tf_idx_test_end_minus_1 < len(full_merge_idx) else len(daily_df) - 1
                daily_trunc: pd.DataFrame = daily_df.iloc[:daily_end_idx + 1].copy()
                
                precomputed_daily: pd.DataFrame = strategy_oos.generate_signals(daily_trunc)
                s_f, r_f, m_f, t_f, w_f = evaluate_symbol_fold(
                    strategy_oos, best_params, sym, tf_val, target_df, daily_trunc,
                    full_merge_idx, precomputed_daily, test_start, test_end
                )
                f_scrs.append(s_f)
                f_rets.append(r_f)
                f_mdds.append(m_f)
                f_trds.append(float(t_f))
                f_wins.append(w_f)

            _logger.info(
                f"| {sym:<10} | {np.mean(f_rets):8.2f} | {np.mean(f_mdds):8.2f} | "
                f"{int(np.sum(f_trds)):5d} | {np.mean(f_wins):8.2f} | {np.mean(f_scrs):7.2f} |"
            )
        except Exception as e:
            _logger.info(f"| {sym:<10} | {'ERR':>8} | {'ERR':>8} | {'ERR':>5} | {'ERR':>8} | {'ERR':>7} |")
            
    _logger.info("=" * SEP_WIDTH)

    # 4. Save to SQLite (Only during optimization)
    if not args.test:
        save_study_to_sqlite(study, project_root)

if __name__ == "__main__":
    main()
