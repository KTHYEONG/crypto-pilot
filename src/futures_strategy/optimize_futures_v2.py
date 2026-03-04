from __future__ import annotations

import argparse
import logging
import os
import sys
import gc
import threading
import traceback
import optuna
from optuna.trial import TrialState
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import concurrent.futures
from multiprocessing import Manager
from functools import partial
from tqdm import tqdm

# Project Root Setup
project_root: str = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.futures_strategy.data_collector import DataCollector
from src.futures_strategy.strategies_futures import UltimateStrategy
from config.settings import (
    FUTURES_INITIAL_BALANCE,
    TRADING_FEE_RATE,
    SLIPPAGE_RATE,
    DATA_DIR,
)
from config.opt_config import OPT_V2_CONFIG, get_search_space_v2, get_quarterly_window

# Reuse merge index util from v1 (no cached signals: v2 uses full daily signals once)
from src.futures_strategy.optimize_futures import compute_segment_merge_index
from src.futures_strategy.funding_utils import merge_funding_into_ohlcv

# Importing from the new modular structure
from src.futures_strategy.opt_v2_utils.db_utils import save_study_to_sqlite
from src.futures_strategy.opt_v2_utils.evaluator import objective_v2, evaluate_symbol_fold
from src.futures_strategy.opt_v2_utils.go_nogo import run_go_nogo_check, GoNoGoResult

import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Logger setup
# --------------------------------------------------------------------------
# Set Optuna logging level to WARNING to hide default per-trial logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
_logger: logging.Logger = logging.getLogger("opt_v2")

SEP_WIDTH: int = 60


class _ThreadSafeJournalStorageWrapper:
    """
    Serializes storage access within a single process (thread lock). On Windows,
    cross-process serialization is handled by WindowsNamedMutexJournalLock.
    """

    def __init__(self, storage: Any) -> None:
        self._storage = storage
        self._lock: threading.Lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._storage, name)
        if callable(attr):
            def _locked(*args: Any, **kwargs: Any) -> Any:
                with self._lock:
                    return attr(*args, **kwargs)
            return _locked
        return attr


def _ensure_fresh_journal(journal_path: Path) -> None:
    """
    Remove existing journal file so optimization runs with no prior trials.
    If the file is missing, no-op. Uses Path.unlink for atomic removal.
    """
    if journal_path.exists():
        try:
            journal_path.unlink()
            _logger.info("Removed existing journal file for fresh optimization: %s", journal_path)
        except OSError as e:
            _logger.warning("Could not remove journal file %s: %s. Truncating instead.", journal_path, e)
            try:
                journal_path.write_text("")
            except OSError:
                pass


@dataclass
class _TfOptimizationContext:
    """Picklable context for per-timeframe optimization in ProcessPoolExecutor."""
    clean_symbol: str
    seeds: List[int]
    n_trials: int
    n_jobs: int
    data_maps: Dict[str, Dict[str, Any]]
    symbols: List[str]
    project_root: str
    progress_queue: Any  # multiprocessing.managers.QueueProxy; optional for pickling


def _run_tf_optimization(
    tf: str, ctx: _TfOptimizationContext
) -> Tuple[str, List[optuna.trial.FrozenTrial]]:
    """Module-level worker for ProcessPoolExecutor. Returns only (tf, best_trial); study is not picklable (contains Lock)."""
    tf_study_name: str = f"FuturesV2_{ctx.clean_symbol}_{tf}"
    journal_path: Path = Path(ctx.project_root) / f"optuna_journal_{tf}.log"
    _ensure_fresh_journal(journal_path)

    lock_obj: Any = None
    if sys.platform == "win32":
        from src.futures_strategy.opt_v2_utils.win_journal_lock import WindowsNamedMutexJournalLock
        lock_obj = WindowsNamedMutexJournalLock(str(journal_path))

    backend = JournalFileBackend(str(journal_path), lock_obj=lock_obj)
    journal_storage = JournalStorage(backend)
    storage = _ThreadSafeJournalStorageWrapper(journal_storage)

    population_size: int = int(OPT_V2_CONFIG.get("n_startup_trials", 40))
    def _constraints(trial: optuna.trial.FrozenTrial) -> Tuple[float, ...]:
        avg_trades = trial.user_attrs.get("avg_trades", 0.0)
        total_trades = avg_trades * len(ctx.symbols)
        trade_violation = float(150.0 - total_trades)

        worst_sym_sortino = min(
            float(trial.user_attrs.get(f"{sym}_cv_mean", -100.0))
            for sym in ctx.symbols
        )
        sym_violation = float(-1.0 - worst_sym_sortino)

        return (trade_violation, sym_violation)

    sampler = optuna.samplers.NSGAIISampler(
        population_size=population_size, mutation_prob=0.1, crossover_prob=0.9, seed=ctx.seeds[0],
        constraints_func=_constraints
    )

    study = optuna.create_study(
        study_name=tf_study_name,
        storage=storage,
        directions=["maximize", "minimize"],
        sampler=sampler,
    )

    def _progress_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # Avoid O(N) access to Study.trials / Study.best_trials; callback must stay O(1).
        _unused_study: optuna.Study = study
        n_done: int = int(trial.number) + 1
        if trial.values:
            best_num: int = trial.number
            best_val: float = float(trial.values[0])
        else:
            best_num = trial.number
            best_val = 0.0
        ctx.progress_queue.put((tf, n_done, ctx.n_trials, best_num, best_val))

    # NEW: Fetch TF-specific search space (Option B)
    tf_space = get_search_space_v2(tf)

    def _objective_with_logging(trial: optuna.Trial) -> float:
        """Log full traceback on objective failure; store on trial for visibility when 0 completed."""
        try:
            return objective_v2(
                trial,
                data_maps=ctx.data_maps,
                symbols=ctx.symbols,
                tf_target=tf,
                space=tf_space,
                project_root=ctx.project_root,
            )
        except optuna.TrialPruned:
            raise
        except Exception as e:
            tb_str: str = traceback.format_exc()
            _logger.exception("[%s] Trial %d failed (root cause above).", tf, trial.number)
            try:
                trial.set_user_attr("_fail_traceback", tb_str)
                trial.set_user_attr("_fail_message", str(e))
            except Exception:
                pass
            raise

    _logger.info("[%s] Starting optimization (%s trials, %s workers)...", tf, ctx.n_trials, ctx.n_jobs)
    study.optimize(
        _objective_with_logging,
        n_trials=ctx.n_trials,
        n_jobs=ctx.n_jobs,
        catch=(Exception,),
        callbacks=[_progress_cb],
    )
    n_actual: int = len(study.trials)
    if n_actual < ctx.n_trials:
        _logger.warning(
            "[%s] Optimization stopped early: %d/%d trials run. Check for timeouts or worker failures.",
            tf, n_actual, ctx.n_trials,
        )
    completed_trials: List[optuna.trial.FrozenTrial] = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed_trials:
        failed_trials: List[optuna.trial.FrozenTrial] = [t for t in study.trials if t.state == TrialState.FAIL]
        pruned_trials: List[optuna.trial.FrozenTrial] = [t for t in study.trials if t.state == TrialState.PRUNED]
        _logger.warning("[%s] No completed trials: %d failed, %d pruned.", tf, len(failed_trials), len(pruned_trials))
        for i, t in enumerate(failed_trials[:3]):
            attrs: Dict[str, Any] = getattr(t, "system_attrs", {}) or {}
            uattrs: Dict[str, Any] = getattr(t, "user_attrs", {}) or {}
            fail_msg: Optional[str] = uattrs.get("_fail_message") or attrs.get("fail_reason")
            fail_tb: Optional[str] = uattrs.get("_fail_traceback")
            _logger.warning("[%s] Failed trial %d: %s", tf, t.number, fail_msg or "(no message)")
            if fail_tb:
                _logger.warning("[%s] Trial %d traceback:\n%s", tf, t.number, fail_tb)
            if not fail_msg and not fail_tb:
                _logger.warning("[%s] Failed trial %d system_attrs: %s", tf, t.number, attrs)
        raise ValueError(
            f"No trials are completed yet (all {len(study.trials)} trials failed or pruned). "
            "See exception traceback(s) above for root cause."
        )
    best_val = float(study.best_trials[0].values[0]) if study.best_trials and study.best_trials[0].values else 0.0
    ctx.progress_queue.put(("done", tf, best_val, n_actual))
    _logger.info("[%s] Optimization complete. Pareto Front Size: %d (%d/%d trials)", tf, len(study.best_trials), n_actual, ctx.n_trials)
    return tf, study.best_trials


# --------------------------------------------------------------------------
# Execution Main
# --------------------------------------------------------------------------
def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,SOL/USDT,DOGE/USDT,LINK/USDT")
    parser.add_argument("--trials", type=int, default=OPT_V2_CONFIG["total_trials"])
    parser.add_argument("--jobs", type=int, default=OPT_V2_CONFIG["n_jobs"], help="Optuna workers per timeframe study (Max 10)")
    parser.add_argument("--test", action="store_true", help="Load best study from DB and evaluate without optimizing")
    parser.add_argument("--reference-date", type=str, default=None, help="Reference date for quarterly window (YYYY-MM-DD)")
    args: argparse.Namespace = parser.parse_args()

    FETCH_START_DATE, START_DATE, IS_END_DATE, END_DATE = get_quarterly_window(args.reference_date)

    symbols: List[str] = [s.strip() for s in args.symbols.split(",")]

    collector: DataCollector = DataCollector()
    data_maps: Dict[str, Dict[str, Any]] = {}
    oos_data_maps: Dict[str, Dict[str, Any]] = {}

    _logger.info("Loading target futures database (Fetch: %s, IS: %s to %s, OOS: %s to %s)...", FETCH_START_DATE, START_DATE, IS_END_DATE, IS_END_DATE, END_DATE)
    target_tfs: List[str] = OPT_V2_CONFIG.get("TARGET_TIMEFRAMES", ["1h"])
    for sym in symbols:
        data_maps[sym] = {}
        oos_data_maps[sym] = {}
        for tf in target_tfs + ["1d"]:
            # Load full data first to ensure technical indicators (like EMA) have enough history, 
            # but we will slice it immediately for engine usage.
            full_df: pd.DataFrame = collector.collect_and_save(sym, tf, FETCH_START_DATE, END_DATE)
            if full_df.empty:
                _logger.error("Failed to load %s %s data", sym, tf)
                sys.exit(1)
            full_df = merge_funding_into_ohlcv(sym, full_df, DATA_DIR)
            
            # Split into IS and OOS
            if full_df["datetime"].dt.tz is None:
                is_start_dt = pd.to_datetime(START_DATE)
                is_end_dt = pd.to_datetime(IS_END_DATE)
            else:
                is_start_dt = pd.to_datetime(START_DATE).tz_localize(full_df["datetime"].dt.tz)
                is_end_dt = pd.to_datetime(IS_END_DATE).tz_localize(full_df["datetime"].dt.tz)

            is_mask = full_df["datetime"] < is_end_dt

            # Store full_df up to IS_END in data_maps to keep historical data for indicator warmup
            is_df: pd.DataFrame = full_df[is_mask].reset_index(drop=True)
            data_maps[sym][tf] = is_df

            # Compute IS start index as POSITION within the sliced IS dataframe
            is_mask_for_idx: pd.Series = is_df["datetime"] >= is_start_dt
            if is_mask_for_idx.any():
                data_maps[sym][f"is_start_idx_{tf}"] = int(is_mask_for_idx.to_numpy().argmax())
            else:
                data_maps[sym][f"is_start_idx_{tf}"] = len(is_df)

            # Store full_df in oos_data_maps to keep historical data for indicator warmup
            oos_data_maps[sym][tf] = full_df

            # Compute OOS start index as POSITION for iloc slicing
            oos_mask_for_idx: pd.Series = full_df["datetime"] >= is_end_dt
            if oos_mask_for_idx.any():
                oos_data_maps[sym][f"oos_start_idx_{tf}"] = int(oos_mask_for_idx.to_numpy().argmax())
            else:
                oos_data_maps[sym][f"oos_start_idx_{tf}"] = len(full_df)
            
        for tf in target_tfs:
             data_maps[sym][f"merge_idx_{tf}"] = compute_segment_merge_index(data_maps[sym][tf], data_maps[sym]["1d"])
             oos_data_maps[sym][f"merge_idx_{tf}"] = compute_segment_merge_index(oos_data_maps[sym][tf], oos_data_maps[sym]["1d"])

        data_maps[sym]["merge_idx_1d"] = compute_segment_merge_index(data_maps[sym]["1d"], data_maps[sym]["1d"])
        oos_data_maps[sym]["merge_idx_1d"] = compute_segment_merge_index(oos_data_maps[sym]["1d"], oos_data_maps[sym]["1d"])
    _logger.info("Target Data load complete (IS and OOS).")

    if "," in args.symbols:
        clean_symbol = "multi"
    else:
        clean_symbol = args.symbols.replace("/", "").replace(" ", "")
        
    study_name: str = f"futures_strategy_{clean_symbol}"
    
    db_user: str = os.getenv("DB_USER", "root")
    db_pass: str = os.getenv("DB_PASS", "1234")
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: str = os.getenv("DB_PORT", "3306")
    db_name: str = os.getenv("DB_NAME", "optuna_crypto")
    from urllib.parse import quote_plus
    safe_pass: str = quote_plus(db_pass)
    storage_url: str = f"mysql+pymysql://{db_user}:{safe_pass}@{db_host}:{db_port}/{db_name}"

    seeds: List[int] = OPT_V2_CONFIG["seeds"]
    n_trials: int = args.trials

    _logger.info("Starting Multi-TF Parallel Optimization. Target TFs: %s", target_tfs)

    manager: Manager = Manager()
    progress_queue: Any = manager.Queue()
    opt_ctx: _TfOptimizationContext = _TfOptimizationContext(
        clean_symbol=clean_symbol,
        seeds=seeds,
        n_trials=n_trials,
        n_jobs=args.jobs,
        data_maps=data_maps,
        symbols=symbols,
        project_root=project_root,
        progress_queue=progress_queue,
    )

    tf_bars: Dict[str, tqdm] = {}
    for i, tf in enumerate(target_tfs):
        tf_bars[tf] = tqdm(
            total=n_trials,
            desc=f"[{tf}] Best trial: 0. Best value: 0.000000",
            position=i,
            leave=True,
            dynamic_ncols=True,
        )

    def _progress_listener() -> None:
        done_count: int = 0
        while True:
            msg: Any = progress_queue.get()
            if msg is None:
                break
            if msg[0] == "done":
                _, done_tf, best_val, n_actual = msg
                bar = tf_bars[done_tf]
                bar.n = n_actual
                bar.total = bar.total  # keep requested total for display
                bar.set_description(
                    f"[{done_tf}] Best: {best_val:.6f} ({n_actual}/{bar.total})" if best_val is not None else f"[{done_tf}] Done ({n_actual})"
                )
                bar.close()
                done_count += 1
            else:
                tf, current, total, best_num, best_val = msg
                bar = tf_bars[tf]
                bar.n = current
                bar.total = total
                bar.set_description(f"[{tf}] Best trial: {best_num}. Best value: {best_val:.6f}")
                bar.refresh()

    progress_thread: threading.Thread = threading.Thread(target=_progress_listener, daemon=False)
    progress_thread.start()

    best_results: Dict[str, List[optuna.trial.FrozenTrial]] = {}
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(target_tfs)) as executor:
            future_to_tf = {executor.submit(_run_tf_optimization, tf, opt_ctx): tf for tf in target_tfs}
            for future in concurrent.futures.as_completed(future_to_tf):
                tf = future_to_tf[future]
                try:
                    res_tf, res_trials = future.result()
                    best_results[res_tf] = res_trials
                except Exception as e:
                    _logger.error("[%s] Parallel optimization failed: %s", tf, e)
    finally:
        progress_queue.put(None)
        progress_thread.join(timeout=5.0)

    if not best_results:
        _logger.error("All timeframe optimizations failed.")
        return

    # Per-TF summary for DB selection: (tf, go_nogo_passed, mean_oos_romad)
    tf_summary: List[Tuple[str, bool, float, List[optuna.trial.FrozenTrial]]] = []

    # Evaluate each TF in fixed order (target_tfs) so logs are scannable per timeframe
    for tf_val in target_tfs:
        if tf_val not in best_results:
            continue
            
        pareto_trials = best_results[tf_val]
        if not pareto_trials:
            continue
            
        # Select the trial with the highest Sortino Ratio from the Pareto Front to run Go/No-Go and true OOS
        best_trial = max(pareto_trials, key=lambda t: t.values[0] if t.values else -100)

        _logger.info("")
        _logger.info("=" * SEP_WIDTH)
        _logger.info("  [TF: %s] Results for Timeframe: %s (Pareto Front Size: %d)", tf_val, tf_val, len(pareto_trials))
        _logger.info("=" * SEP_WIDTH)

        # 1. Best Parameters (from highest Sortino trial)
        _logger.info("  [TF: %s] Highest Sortino Parameters", tf_val)
        best_params: Dict[str, Any] = best_trial.params.copy()
        best_params["TIMEFRAME"] = tf_val
        for k, v in best_params.items():
            if isinstance(v, float):
                _logger.info("  - %s: %.6f", k, v)
            else:
                _logger.info("  - %s: %s", k, v)
        _logger.info("-" * SEP_WIDTH)
        _logger.info("  [TF: %s] Best CV Sortino: %.4f, MDD: %.2f%%", tf_val, best_trial.values[0], best_trial.values[1])
        _logger.info("-" * SEP_WIDTH)

        # 2. IS (Purged Walk-Forward) metrics
        _logger.info("  [TF: %s] IS — Purged Walk-Forward Metrics", tf_val)
        header: str = f"| {'Symbol':<10} | {'Ret%':>8} | {'MDD%':>7} | {'Trades':>6} | {'Win%':>6} | {'PF':>6} | {'Score':>7} |"
        _logger.info(header)
        _logger.info("|" + "-" * (len(header) - 2) + "|")
        for sym in symbols:
            ret_sum = float(best_trial.user_attrs.get(f"{sym}_ret_sum", 0.0))
            m = float(best_trial.user_attrs.get(f"{sym}_mdd", 0.0))
            t = int(best_trial.user_attrs.get(f"{sym}_trades", 0))
            win = float(best_trial.user_attrs.get(f"{sym}_win", 0.0))
            pf = float(best_trial.user_attrs.get(f"{sym}_pf", 1.0))
            cv_mean = float(best_trial.user_attrs.get(f"{sym}_cv_mean", 0.0))
            _logger.info("| %s | %8.2f | %7.2f | %6d | %6.2f | %6.2f | %7.2f |", sym, ret_sum, m, t, win, pf, cv_mean)
        _logger.info("-" * SEP_WIDTH)

        # 3. True OOS Verification
        _logger.info("  [TF: %s] True OOS Forward-Testing", tf_val)
        header_oos: str = f"| {'Symbol':<10} | {'Ret%':>8} | {'MDD%':>7} | {'Trades':>6} | {'Win%':>6} | {'PF':>6} | {'Score':>7} |"
        _logger.info(header_oos)
        _logger.info("|" + "-" * (len(header_oos) - 2) + "|")
        strategy_oos: UltimateStrategy = UltimateStrategy(name=f"FuturesV2_OOS_{tf_val}", params=best_params)
        oos_romad_scores: List[float] = []
        oos_pfs: List[float] = []
        oos_mdds: List[float] = []
        oos_longs: int = 0
        oos_shorts: int = 0

        for sym in symbols:
            target_df_oos = oos_data_maps.get(sym, {}).get(tf_val)
            daily_df_oos = oos_data_maps.get(sym, {}).get("1d")
            full_merge_idx_oos = oos_data_maps.get(sym, {}).get(f"merge_idx_{tf_val}")
            if target_df_oos is None or target_df_oos.empty:
                continue
            try:
                n_oos_bars = len(target_df_oos)
                oos_start_idx = oos_data_maps.get(sym, {}).get(f"oos_start_idx_{tf_val}", 0)
                s_f, r_f, m_f, t_f, w_f, pf_f, lc_f, sc_f, _ = evaluate_symbol_fold(
                    strategy_oos, best_params, sym, tf_val, target_df_oos, daily_df_oos,
                    full_merge_idx_oos, None, oos_start_idx, n_oos_bars
                )
                oos_romad_scores.append(s_f)
                oos_pfs.append(pf_f)
                oos_mdds.append(m_f)
                oos_longs += int(lc_f)
                oos_shorts += int(sc_f)
                _logger.info("| %s | %8.2f | %7.2f | %6d | %6.2f | %6.2f | %7.2f |", sym, r_f, m_f, int(t_f), w_f, pf_f, s_f)
            except Exception as e:
                _logger.error("  [TF: %s] OOS error for %s: %s", tf_val, sym, e)
        _logger.info("-" * SEP_WIDTH)

        # 4. Go/No-Go Checklist
        _logger.info("  [TF: %s] Go/No-Go Checklist", tf_val)
        cv_scores = best_trial.user_attrs.get("cv_scores", [])
        holdout_score = best_trial.user_attrs.get("holdout_score", 0.0)
        true_max_mdd = float(np.max(oos_mdds)) if oos_mdds else 99.9
        avg_pf = float(np.mean(oos_pfs)) if oos_pfs else 0.0
        go_nogo: GoNoGoResult = run_go_nogo_check(
            cv_fold_scores=cv_scores,
            holdout_score=holdout_score,
            oos_romad_scores=oos_romad_scores,
            max_mdd_pct=true_max_mdd,
            profit_factor=avg_pf,
            long_count=oos_longs,
            short_count=oos_shorts,
            tf=tf_val,
        )
        _logger.info("%s", go_nogo.summary)
        mean_oos_romad: float = float(np.mean(oos_romad_scores)) if oos_romad_scores else 0.0
        _logger.info("  [TF: %s] Go/No-Go: %s | Mean OOS RoMaD: %.4f", tf_val, "PASS" if go_nogo.passed else "FAIL", mean_oos_romad)

        best_trial.set_user_attr("go_nogo_passed", go_nogo.passed)
        tf_summary.append((tf_val, go_nogo.passed, mean_oos_romad, pareto_trials))

    # Save to DB: among Go/No-Go passed TFs, the one with highest mean OOS RoMaD (build minimal study from best_trial)
    passed_tfs: List[Tuple[str, List[optuna.trial.FrozenTrial], float]] = [
        (tf, pareto_trials, score) for tf, passed, score, pareto_trials in tf_summary if passed
    ]
    if passed_tfs:
        best_tf, best_pareto_trials, best_oos = max(passed_tfs, key=lambda x: x[2])
        minimal_study: optuna.Study = optuna.create_study(
            study_name=study_name,
            directions=["maximize", "minimize"],
        )
        for t in best_pareto_trials:
            minimal_study.add_trial(t)
        save_study_to_sqlite(minimal_study, project_root, target_study_name=study_name)
        _logger.info("")
        _logger.info("  💾 Saved to DB: study '%s' with %d Pareto strategies (TF: %s, OOS RoMaD: %.4f)", study_name, len(best_pareto_trials), best_tf, best_oos)
    else:
        _logger.warning("  ❌ No TF passed Go/No-Go. SQLite DB not updated.")

    # Summary table: TF | Go/No-Go | OOS Score
    _logger.info("")
    _logger.info("  [Summary] Per-Timeframe")
    _logger.info("  | %s | %s | %s |", "TF".ljust(6), "Go/No-Go".ljust(8), "Mean OOS RoMaD")
    _logger.info("  |%s|%s|%s|", "-" * 8, "-" * 10, "-" * 16)
    for tf, passed, score, _ in tf_summary:
        _logger.info("  | %s | %s | %14.4f |", tf.ljust(6), "PASS" if passed else "FAIL", score)

if __name__ == "__main__":
    main()
