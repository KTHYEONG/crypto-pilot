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
project_root = str(Path(__file__).resolve().parents[2])
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

# --------------------------------------------------------------------------
# Logger setup
# --------------------------------------------------------------------------
# Set Optuna logging level to WARNING to hide default per-trial logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
_logger = logging.getLogger("opt_v2")

# Constants
START_DATE = "2022-01-01"
END_DATE = "2025-11-01"
WARMUP_BARS = {"4h": 400, "1d": 400}
EMBARGO_BARS = {"4h": 6, "1d": 2}


# --------------------------------------------------------------------------
# Score Grade Definitions (Theoretic)
# - S (Platinum) [8.0+]: Extreme performance with consistent equity growth across all folds.
# - A (Gold) [4.0~8.0]: Strong risk-adjusted returns, production-ready robustness.
# - B (Silver) [1.5~4.0]: Reliable strategy with good market-beating potential.
# - C (Bronze) [0.5~1.5]: Profitable but low efficiency or high volatility sensitivity.
# - D (Fail) [<0.5]: Inconsistent performance or excessive risk (MDD > 40%).
# --------------------------------------------------------------------------
# Metric Calculation
# --------------------------------------------------------------------------
def calc_romad(pnl_series: pd.Series, n_trades: int, tf: str) -> Tuple[float, float, float]:
    """
    Calculate Return on Max Drawdown (RoMaD) as the primary risk-adjusted metric.
    Returns: (romad_score, return_pct, mdd_pct)
    """
    if pnl_series.empty or n_trades == 0:
        return -20.0, 0.0, 0.0

    # 1. Equity curve & Return
    equity = FUTURES_INITIAL_BALANCE + pnl_series.cumsum()
    end_equity = equity.iloc[-1]
    ret_pct = ((end_equity / FUTURES_INITIAL_BALANCE) - 1.0) * 100.0

    # 2. Maximum Drawdown (MDD)
    running_max = np.maximum.accumulate(equity.values)
    running_max[running_max == 0] = 1e-9
    drawdown = (equity.values - running_max) / running_max * 100.0
    mdd_pct = abs(float(np.min(drawdown))) if len(drawdown) > 0 else 0.0

    # 3. Annualized Return
    # Calculate span in days
    span_days = (pnl_series.index[-1] - pnl_series.index[0]).total_seconds() / 86400.0
    if span_days < 1.0:
        span_days = 1.0
    annual_return = ret_pct * (365.0 / span_days)

    # 4. RoMaD Core (Floor MDD at 5% to avoid div-by-zero or excessive score on low-trade flat curves)
    romad = annual_return / max(mdd_pct, 5.0)

    # 5. Penalties (Soft quadratic penalty for low trade counts)
    min_trades_target: int = 60 if tf == "4h" else 30
    trade_ratio: float = float(n_trades) / min_trades_target
    penalty_multiplier: float = min(trade_ratio ** 0.5, 1.2)

    final_score = (romad * penalty_multiplier) - max(0, mdd_pct - 40.0) * 0.3
    return final_score, ret_pct, mdd_pct


def calc_romad_from_metrics(
    ret_pct: float,
    mdd_pct: float,
    n_trades: int,
    tf: str,
    span_days: float,
) -> Tuple[float, float, float]:
    """
    RoMaD score from precomputed return and MDD (e.g. engine bar-level metrics).
    Returns: (romad_score, return_pct, mdd_pct). Use when MDD/return come from
    bar-level equity to avoid risk understatement vs trade-exit-only.
    """
    if n_trades == 0:
        return -20.0, ret_pct, mdd_pct
    mdd_abs = abs(mdd_pct)
    days = max(float(span_days), 1.0)
    annual_return = ret_pct * (365.0 / days)
    romad = annual_return / max(mdd_abs, 5.0)
    min_trades_target: int = 60 if tf == "4h" else 30
    trade_ratio: float = float(n_trades) / min_trades_target
    penalty_multiplier: float = min(trade_ratio ** 0.5, 1.2)
    final_score = (romad * penalty_multiplier) - max(0, mdd_abs - 40.0) * 0.3
    return final_score, ret_pct, mdd_abs


# --------------------------------------------------------------------------
# Validation Folds Setup
# --------------------------------------------------------------------------
def build_anchored_folds(df: pd.DataFrame, n_folds: int = 3, embargo: int = 0) -> List[Tuple[int, int, int]]:
    """Build train(0~idx) -> OOS(idx~idx_next) splits."""
    n_bars = len(df)
    if n_bars < 500:
        return []

    block = n_bars // (n_folds + 1)
    splits = []
    for i in range(1, n_folds + 1):
        train_end = block * i
        test_start = train_end + embargo
        test_end = (block * (i + 1)) if i < n_folds else n_bars

        if test_start < test_end:
            splits.append((train_end, test_start, test_end))

    return splits


# --------------------------------------------------------------------------
# Optuna Suggestion Logic
# --------------------------------------------------------------------------
def suggest_params_v2(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    params = {}
    for k, spec in space.items():
        if spec["type"] == "categorical":
            params[k] = trial.suggest_categorical(k, spec["choices"])
        elif spec["type"] == "int":
            log = spec.get("log", False)
            step = spec.get("step", 1)
            params[k] = trial.suggest_int(k, spec["low"], spec["high"], step=step, log=log)
        elif spec["type"] == "float":
            log = spec.get("log", False)
            step = spec.get("step", None)
            params[k] = trial.suggest_float(k, spec["low"], spec["high"], step=step, log=log)

    # Dimensionality conditional pruning: If parameter isn't used by the structure, remove it
    entry = params["ENTRY_TYPE"]
    if entry != "BOLLINGER":
        params.pop("BB_STD", None)
    if entry != "KELTNER":
        params.pop("KELTNER_ATR_MULT", None)

    trend = params["TREND_FILTER_TYPE"]
    if trend != "SUPERTREND":
        params.pop("SUPERTREND_MULT", None)
        params.pop("SUPERTREND_PERIOD", None)
    if trend != "DMI":
        params.pop("DMI_PERIOD", None)
    if trend != "VWAP":
        params.pop("VWAP_STD_MULT", None)

    strength = params["STRENGTH_FILTER_TYPE"]
    if strength == "NONE":
        params.pop("STRENGTH_FILTER_PERIOD", None)
    if strength != "ADX":
        params.pop("ADX_THRESHOLD", None)
    if strength != "NATR":
        params.pop("NATR_THRESHOLD", None)
    if strength != "ER":
        params.pop("ER_THRESHOLD", None)

    # Sanity constraints
    # (MACD constraint removed as indicator was swapped for DMI)
    if not params.get("USE_VOLUME_FILTER", False):
        params.pop("VOLUME_MA_PERIOD", None)
        params.pop("VOLUME_Z_THRESHOLD", None)

    return params


# --------------------------------------------------------------------------
# Objective Evaluation
# --------------------------------------------------------------------------
def evaluate_symbol_fold(
    strategy: UltimateStrategy,
    params: Dict[str, Any],
    symbol: str,
    tf: str,
    target_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    full_merge_idx: np.ndarray,
    precomputed_daily_df: pd.DataFrame,
    test_start: int,
    test_end: int,
) -> Tuple[float, float, float, int, float]:
    sig_oos = target_df.iloc[test_start:test_end].copy()
    sig_oos.attrs = {"warmup_bars": 0}
    merge_oos = full_merge_idx[test_start:test_end]

    engine = BacktestEngineFast(
        hourly_df=sig_oos,
        daily_df=daily_df,
        strategy=strategy,
        initial_balance=FUTURES_INITIAL_BALANCE,
        merge_index_map=merge_oos,
        precomputed_daily_df=precomputed_daily_df,
    )
    engine.leverage = params.get("LEVERAGE", 1)
    engine.risk_per_trade = params.get("RISK_PER_TRADE", 0.02)
    engine.funding_events_per_bar = 3 if params.get("TIMEFRAME") == "1d" else 1

    try:
        result = engine.run()
        trades_df = result.get("trades_df")
    except Exception as e:
        _logger.warning("Backtest engine error: %s", e)
        return -100.0, 0.0, 0.0, 0, 0.0

    if trades_df is None or trades_df.empty:
        return -100.0, 0.0, 0.0, 0, 0.0

    long_count = len(trades_df[trades_df["side"] == "LONG"])
    short_count = len(trades_df[trades_df["side"] == "SHORT"])

    # Use engine bar-level MDD and return (includes unrealized P&L) to avoid risk understatement
    mdd_pct = abs(result.get("mdd_pct", 0.0))
    ret_pct = result.get("total_return_pct", 0.0)

    # Span for annualization: fixed length of OOS
    span_days = (sig_oos["datetime"].iloc[-1] - sig_oos["datetime"].iloc[0]).total_seconds() / 86400.0 if "datetime" in sig_oos.columns else 1.0
    span_days = max(float(span_days), 1.0)

    true_pnl = trades_df["pnl"] - trades_df["entry_fee"]
    win_rate = (len(trades_df[true_pnl > 0]) / len(trades_df)) * 100 if len(trades_df) > 0 else 0.0
    score, ret_pct, mdd_pct = calc_romad_from_metrics(ret_pct, mdd_pct, len(trades_df), tf, span_days)

    if long_count == 0 or short_count == 0:
        score -= 5.0

    return score, ret_pct, mdd_pct, len(trades_df), win_rate


def objective_v2(trial: optuna.Trial, data_maps: Dict[str, Dict[str, pd.DataFrame]]) -> float:
    params = suggest_params_v2(trial, SEARCH_SPACE_V2)
    if params.pop("_INVALID_CONSTRAINT", False):
        return -100.0

    tf = params["TIMEFRAME"]
    symbols = list(data_maps.keys())

    strategy = UltimateStrategy(name="FuturesV2", params=params)

    # Build folds using min length across all symbols so every symbol has valid OOS segments
    lengths = [len(data_maps[sym][tf]) for sym in symbols if tf in data_maps.get(sym, {})]
    if not lengths:
        return -10000.0
    min_len = min(lengths)
    base_df_for_folds = pd.DataFrame(index=range(min_len))
    folds = build_anchored_folds(base_df_for_folds, n_folds=3, embargo=EMBARGO_BARS.get(tf, 0))

    if not folds:
        return -10000.0

    # Evaluate fold by fold to allow Hyperband Pruning
    fold_scores = []
    fold_rets = []
    fold_mdds = []
    fold_trades = []
    fold_wins = []
    
    # Track per-symbol cumulative metrics across folds
    sym_total_rets = {sym: [] for sym in symbols}
    sym_total_mdds = {sym: [] for sym in symbols}
    sym_total_scores = {sym: [] for sym in symbols}
    sym_total_trades = {sym: [] for sym in symbols}
    sym_total_wins = {sym: [] for sym in symbols}
    
    for f_idx, (train_end, test_start, test_end) in enumerate(folds):
        sym_scores = []
        sym_rets = []
        sym_mdds = []
        sym_trades_fold = []
        sym_wins_fold = []
        for sym in symbols:
            target_df = data_maps[sym].get(tf)
            daily_df = data_maps[sym].get("1d")
            full_merge_idx = data_maps[sym].get(f"merge_idx_{tf}")
            if target_df is None or daily_df is None or full_merge_idx is None:
                continue

            # Prevent look-ahead bias by truncating data up to test_end
            tf_idx_test_end_minus_1 = test_end - 1
            daily_end_idx = full_merge_idx[tf_idx_test_end_minus_1] if tf_idx_test_end_minus_1 < len(full_merge_idx) else len(daily_df) - 1
            daily_df_trunc = daily_df.iloc[:daily_end_idx + 1].copy()

            try:
                precomputed_daily_df = strategy.generate_signals(daily_df_trunc)
            except Exception as e:
                continue
            
            s, r, m, t, w = evaluate_symbol_fold(
                strategy, params, sym, tf, target_df, daily_df_trunc, 
                full_merge_idx, precomputed_daily_df, test_start, test_end
            )
            sym_scores.append(s)
            sym_rets.append(r)
            sym_mdds.append(m)
            sym_trades_fold.append(t)
            sym_wins_fold.append(w)
            
            # Store per-symbol metrics
            sym_total_scores[sym].append(s)
            sym_total_rets[sym].append(r)
            sym_total_mdds[sym].append(m)
            sym_total_trades[sym].append(t)
            sym_total_wins[sym].append(w)

        if not sym_scores:
            return -10000.0

        # Mean across symbols with penalty for worst performer
        mean_score = float(np.mean(sym_scores))
        min_score = float(np.min(sym_scores))
        penalty = max(0.0, 0.0 - min_score) * 1.5
        avg_fold_sym_score = mean_score - penalty

        fold_scores.append(avg_fold_sym_score)
        fold_rets.append(float(np.mean(sym_rets)))
        fold_mdds.append(float(np.mean(sym_mdds)))
        fold_trades.append(float(np.mean(sym_trades_fold)))
        fold_wins.append(float(np.mean(sym_wins_fold)))


    # Aggregate Fold Scores using Harmonic-like mean to punish the worst fold
    # Shift by 10 to handle negatives up to -9
    shifted = [s + 10.0 for s in fold_scores]
    if any(s <= 0 for s in shifted):
        final_score = np.mean(fold_scores) - 20.0 # Heavy penalty if any fold strongly negative
    else:
        hm = len(shifted) / sum(1.0 / s for s in shifted)
        final_score = hm - 10.0

    trial.set_user_attr("avg_score", float(np.mean(fold_scores)))
    trial.set_user_attr("avg_ret", float(np.mean(fold_rets)))
    trial.set_user_attr("avg_mdd", float(np.mean(fold_mdds)))
    trial.set_user_attr("avg_trades", float(np.mean(fold_trades)))
    trial.set_user_attr("avg_win_rate", float(np.mean(fold_wins)))
    
    # Store per-symbol averages in user_attrs
    sym_log_msgs = []
    for sym in symbols:
        if len(sym_total_scores[sym]) > 0:
            s_score = float(np.mean(sym_total_scores[sym]))
            s_ret = float(np.mean(sym_total_rets[sym]))
            s_mdd = float(np.mean(sym_total_mdds[sym]))
            s_trades = float(np.mean(sym_total_trades[sym]))
            s_wins = float(np.mean(sym_total_wins[sym]))
        else:
            s_score, s_ret, s_mdd, s_trades, s_wins = -100.0, 0.0, 0.0, 0, 0.0
            
        trial.set_user_attr(f"{sym}_score", s_score)
        trial.set_user_attr(f"{sym}_ret", s_ret)
        trial.set_user_attr(f"{sym}_mdd", s_mdd)
        trial.set_user_attr(f"{sym}_trades", s_trades)
        trial.set_user_attr(f"{sym}_win_rate", s_wins)
        sym_log_msgs.append(f"[{sym} R:{s_ret:5.1f}% M:{s_mdd:4.1f}%]")
        
    
    # Progress handled by Optuna's UI internally now
    
    return float(final_score)


# --------------------------------------------------------------------------
# Execution Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--trials", type=int, default=OPT_V2_CONFIG["total_trials"])
    parser.add_argument("--jobs", type=int, default=OPT_V2_CONFIG["n_jobs"])
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    collector = DataCollector()
    data_maps = {}

    _logger.info("Loading futures database (%s to %s)...", START_DATE, END_DATE)
    for sym in symbols:
        data_maps[sym] = {}
        for tf in ["1d", "4h"]:
            df = collector.collect_and_save(sym, tf, START_DATE, END_DATE)
            if df.empty:
                _logger.error("Failed to load %s %s data", sym, tf)
                sys.exit(1)
            df = merge_funding_into_ohlcv(sym, df, DATA_DIR)
            data_maps[sym][tf] = df
            
        data_maps[sym]["merge_idx_1d"] = compute_segment_merge_index(data_maps[sym]["1d"], data_maps[sym]["1d"])
        data_maps[sym]["merge_idx_4h"] = compute_segment_merge_index(data_maps[sym]["4h"], data_maps[sym]["1d"])
    _logger.info("Data load complete.")

    study_name = "futures_v2_romad_opt"
    
    # DB connection setup matching v1
    db_user = os.getenv("DB_USER", "root")
    db_pass = os.getenv("DB_PASS", "1234")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "trading_optuna")
    from urllib.parse import quote_plus
    safe_pass = quote_plus(db_pass)
    storage_url = f"mysql+pymysql://{db_user}:{safe_pass}@{db_host}:{db_port}/{db_name}"

    # Setup 2-seed queue logic
    seeds = OPT_V2_CONFIG["seeds"]
    n_trials = args.trials

    # Custom sampler with deterministic seeds split
    if n_trials <= len(seeds):
        q_seeds = seeds[:n_trials]
        base_trials = n_trials
    else:
        q_seeds = seeds
        base_trials = n_trials

    _logger.info("Starting V2 Optimization. Total Trials: %d, Seeds: %s, Workers: %d", base_trials, q_seeds, args.jobs)

    # Note: ConstantLiar is handled implicitly by optuna when running concurrent jobs
    # But for a clear mathematical approach, we use TPESampler with a fixed seed across the board
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=OPT_V2_CONFIG["n_startup_trials"],
        multivariate=True,
        constant_liar=True,
        warn_independent_sampling=False,
        seed=q_seeds[0], # Use primary seed
    )



    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
        _logger.info(f"Deleted existing study '{study_name}' for a fresh start.")
    except KeyError:
        pass  # Study does not exist yet

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        load_if_exists=False,
    )

    study.optimize(
        lambda t: objective_v2(t, data_maps),
        n_trials=base_trials,
        n_jobs=args.jobs,
        catch=(Exception,),
        show_progress_bar=True,  # Enables tqdm progress bar natively
    )

    _logger.info("=" * 60)
    _logger.info("Optimization Complete.")
    best_trial = study.best_trial
    _logger.info(f"Best Score: {best_trial.value:.4f}")
    _logger.info(f"  - Avg Return (FoldxSym): {best_trial.user_attrs.get('avg_ret', 0):.2f}%")
    _logger.info(f"  - Avg MDD    (FoldxSym): {best_trial.user_attrs.get('avg_mdd', 0):.2f}%")
    _logger.info(f"  - Avg Trades (FoldxSym): {best_trial.user_attrs.get('avg_trades', 0):.1f}")
    _logger.info(f"  - Avg WinRate(FoldxSym): {best_trial.user_attrs.get('avg_win_rate', 0):.2f}%")
    
    _logger.info("  [Per-Symbol Performance]")
    for sym in symbols:
        r = best_trial.user_attrs.get(f"{sym}_ret", 0)
        m = best_trial.user_attrs.get(f"{sym}_mdd", 0)
        s = best_trial.user_attrs.get(f"{sym}_score", 0)
        t = best_trial.user_attrs.get(f"{sym}_trades", 0)
        w = best_trial.user_attrs.get(f"{sym}_win_rate", 0)
        _logger.info(f"    - {sym:10s} | Score: {s:7.2f} | Return: {r:6.2f}% | MDD: {m:5.2f}% | Trades: {t:4.1f} | WinRate: {w:5.2f}%")
        
    _logger.info("Best Params:")
    for k, v in best_trial.params.items():
        _logger.info(f"  - {k:25s}: {v}")
    _logger.info("=" * 60)


if __name__ == "__main__":
    main()
