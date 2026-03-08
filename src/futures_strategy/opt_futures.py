from __future__ import annotations

import argparse
import logging
import math
import os
import queue
import sys
import gc
import threading
import traceback
import optuna
from optuna.trial import TrialState
from optuna.storages import InMemoryStorage
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
from config.opt_config import OPT_FUTURES_CONFIG, get_search_space_futures, get_quarterly_window

from src.optimization.opt_utils import compute_segment_merge_index
from src.futures_strategy.funding_utils import merge_funding_into_ohlcv

from src.futures_strategy.opt_futures_utils.db_utils import save_study_to_sqlite
from src.futures_strategy.opt_futures_utils.evaluator import objective_futures, evaluate_symbol_fold
from src.futures_strategy.opt_futures_utils.go_nogo import run_go_nogo_check, GoNoGoResult

import warnings
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
_logger: logging.Logger = logging.getLogger("opt_futures")

SEP_WIDTH: int = 60

class _ThreadSafeJournalStorageWrapper:
    def __init__(self, storage: Any) -> None:
        self._storage = storage
        self._lock: threading.Lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._storage, name)
        if callable(attr):
            def _locked(*args: Any, **kwargs: Any) -> Any:
                with self._lock: return attr(*args, **kwargs)
            return _locked
        return attr

def _ensure_fresh_journal(journal_path: Path) -> None:
    if journal_path.exists():
        try: journal_path.unlink()
        except OSError: journal_path.write_text("")

@dataclass
class _TfOptimizationContext:
    clean_symbol: str
    seeds: List[int]
    n_trials: int
    n_jobs: int
    data_maps: Dict[str, Dict[str, Any]]
    symbols: List[str]
    project_root: str
    progress_queue: Any 
    mode: str = "single"
    use_journal_storage: bool = True

@dataclass(frozen=True)
class _ParallelExecutionPlan:
    task_workers: int
    jobs_per_task: int
    cpu_budget: int
    logical_cpus: int
    task_count: int

    @property
    def total_active_workers(self) -> int: return self.task_workers * self.jobs_per_task
    @property
    def batch_count(self) -> int: return max(1, math.ceil(self.task_count / max(1, self.task_workers)))

def _resolve_parallel_plan(task_count: int, requested_jobs: int, requested_task_workers: int, requested_cpu_budget: int) -> _ParallelExecutionPlan:
    logical_cpus = max(1, os.cpu_count() or 1)
    cpu_budget = max(1, requested_cpu_budget or (4 if logical_cpus > 2 else 1))
    jobs_per_task = max(1, requested_jobs)
    task_workers = min(task_count, requested_task_workers) if requested_task_workers > 0 else min(task_count, max(1, cpu_budget // jobs_per_task))
    return _ParallelExecutionPlan(task_workers=max(1, task_workers), jobs_per_task=jobs_per_task, cpu_budget=cpu_budget, logical_cpus=logical_cpus, task_count=task_count)

def _run_tf_optimization(task: Tuple[Any, str], ctx: _TfOptimizationContext) -> Tuple[Tuple[Any, str], List[optuna.trial.FrozenTrial]]:
    target_obj, tf = task
    target_str = "_".join(target_obj) if isinstance(target_obj, (list, tuple)) else target_obj
    tf_study_name: str = f"OptFutures_{target_str.replace('/', '')}_{tf}_{ctx.mode}"
    
    storage: Any
    if ctx.use_journal_storage:
        journal_path = Path(ctx.project_root) / f"optuna_journal_{target_str.replace('/', '')}_{tf}.log"
        _ensure_fresh_journal(journal_path)
        lock_obj = None
        if sys.platform == "win32":
            from src.futures_strategy.opt_futures_utils.win_journal_lock import WindowsNamedMutexJournalLock
            lock_obj = WindowsNamedMutexJournalLock(str(journal_path))
        backend = JournalFileBackend(str(journal_path), lock_obj=lock_obj)
        storage = _ThreadSafeJournalStorageWrapper(JournalStorage(backend))
    else:
        storage = InMemoryStorage()

    sampler = optuna.samplers.NSGAIISampler(
        seed=ctx.seeds[0],
        population_size=OPT_FUTURES_CONFIG.get("n_startup_trials", 500)
    )
    
    study = optuna.create_study(
        study_name=tf_study_name, 
        storage=storage, 
        directions=["maximize", "minimize"],
        sampler=sampler
    )

    def _progress_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        try:
            b_val = float(trial.values[0]) if trial.values else 0.0
        except:
            b_val = 0.0
        ctx.progress_queue.put((f"{target_str}_{tf}", trial.number + 1, ctx.n_trials, b_val))

    tf_space = get_search_space_futures(tf)

    def _objective_with_logging(trial: optuna.Trial) -> Tuple[float, float]:
        try:
            return objective_futures(trial, data_maps=ctx.data_maps, symbols=ctx.symbols, tf_target=tf, space=tf_space, mode=ctx.mode, project_root=ctx.project_root)
        except optuna.TrialPruned: raise
        except Exception:
            _logger.exception("[%s/%s] Trial %d failed.", target_str, tf, trial.number)
            raise

    _logger.info("[%s/%s/%s] Starting NSGA-II optimization...", target_str, tf, ctx.mode)
    study.optimize(_objective_with_logging, n_trials=ctx.n_trials, n_jobs=ctx.n_jobs, catch=(Exception,), callbacks=[_progress_cb])
    return (target_obj, tf), study.best_trials

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="ETH/USDT,SOL/USDT")
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="multi")
    parser.add_argument("--trials", type=int, default=OPT_FUTURES_CONFIG["total_trials"])
    parser.add_argument("--jobs", type=int, default=int(OPT_FUTURES_CONFIG.get("n_jobs", 10)))
    parser.add_argument("--task-workers", type=int, default=int(OPT_FUTURES_CONFIG.get("task_workers", 1)))
    parser.add_argument("--tf", type=str, choices=["1h", "4h"], default=OPT_FUTURES_CONFIG.get("TARGET_TIMEFRAMES", ["4h"])[0])
    parser.add_argument("--reference-date", type=str, default=None)
    args = parser.parse_args()

    FETCH_START_DATE, START_DATE, IS_END_DATE, END_DATE = get_quarterly_window(args.reference_date)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    collector = DataCollector(); data_maps = {}; oos_data_maps = {}
    _logger.info("Loading data for %d symbols (Mode: %s)...", len(symbols), args.mode)
    for sym in symbols:
        data_maps[sym] = {}; oos_data_maps[sym] = {}
        for tf in [args.tf, "1d"]:
            full_df = collector.collect_and_save(sym, tf, FETCH_START_DATE, END_DATE)
            full_df = merge_funding_into_ohlcv(sym, full_df, DATA_DIR)
            tz = full_df["datetime"].dt.tz
            is_start_dt = pd.to_datetime(START_DATE).tz_localize(tz) if tz else pd.to_datetime(START_DATE)
            is_end_dt = pd.to_datetime(IS_END_DATE).tz_localize(tz) if tz else pd.to_datetime(IS_END_DATE)
            data_maps[sym][tf] = full_df[full_df["datetime"] < is_end_dt].reset_index(drop=True)
            m = data_maps[sym][tf]["datetime"] >= is_start_dt
            data_maps[sym][f"is_start_idx_{tf}"] = int(m.to_numpy().argmax()) if m.any() else 0
            oos_data_maps[sym][tf] = full_df
            m_oos = full_df["datetime"] >= is_end_dt
            oos_data_maps[sym][f"oos_start_idx_{tf}"] = int(m_oos.to_numpy().argmax()) if m_oos.any() else len(full_df)
        data_maps[sym][f"merge_idx_{args.tf}"] = compute_segment_merge_index(data_maps[sym][args.tf], data_maps[sym]["1d"])
        oos_data_maps[sym][f"merge_idx_{args.tf}"] = compute_segment_merge_index(oos_data_maps[sym][args.tf], oos_data_maps[sym]["1d"])

    tasks = [(tuple(symbols), args.tf)] if args.mode == "multi" else [(s, args.tf) for s in symbols]
    plan = _resolve_parallel_plan(len(tasks), args.jobs, args.task_workers, 0)
    use_mp = len(tasks) > 1 and plan.task_workers > 1
    manager = Manager() if use_mp else None; progress_queue = manager.Queue() if manager else queue.Queue()

    tf_bars = {}
    for i, (target, tf) in enumerate(tasks):
        key = "_".join(target) if isinstance(target, tuple) else target
        tf_bars[f"{key}_{tf}"] = tqdm(total=args.trials, desc=f"[{key}] Waiting...", position=i, leave=True)

    def _progress_listener():
        while True:
            msg = progress_queue.get()
            if msg is None: break
            k, cur, tot, b_val = msg; bar = tf_bars[k]; bar.n = cur; bar.set_description(f"[{k}] Best CAGR: {b_val:.2f}%"); bar.refresh()

    progress_thread = threading.Thread(target=_progress_listener, daemon=True); progress_thread.start()

    best_results = {}
    try:
        if use_mp:
            with concurrent.futures.ProcessPoolExecutor(max_workers=plan.task_workers) as exec:
                futures = {exec.submit(_run_tf_optimization, t, _TfOptimizationContext(
                    clean_symbol=("_".join(t[0]) if isinstance(t[0], tuple) else t[0]).replace("/", ""),
                    seeds=OPT_FUTURES_CONFIG["seeds"], n_trials=args.trials, n_jobs=plan.jobs_per_task,
                    data_maps={s: data_maps[s] for s in (list(t[0]) if isinstance(t[0], tuple) else [t[0]])},
                    symbols=(list(t[0]) if isinstance(t[0], tuple) else [t[0]]), project_root=project_root,
                    progress_queue=progress_queue, mode=args.mode
                )): t for t in tasks}
                for f in concurrent.futures.as_completed(futures): t_res, trials = f.result(); best_results[t_res] = trials
        else:
            for t in tasks:
                t_res, trials = _run_tf_optimization(t, _TfOptimizationContext(
                    clean_symbol=("_".join(t[0]) if isinstance(t[0], tuple) else t[0]).replace("/", ""),
                    seeds=OPT_FUTURES_CONFIG["seeds"], n_trials=args.trials, n_jobs=plan.jobs_per_task,
                    data_maps=data_maps, symbols=(list(t[0]) if isinstance(t[0], tuple) else [t[0]]),
                    project_root=project_root, progress_queue=progress_queue, mode=args.mode, use_journal_storage=False
                ))
                best_results[t_res] = trials
    finally:
        progress_queue.put(None); progress_thread.join(timeout=2.0)
        if manager: manager.shutdown()

    final_summaries = []
    for (target, tf_eval), trials in best_results.items():
        if not trials: continue
        
        # Select best trial from Pareto front: Maximize CAGR (values[0]), while MDD (values[1]) <= 35.0%
        # Embrace optimal Kelly compounding over strict institutional limits.
        valid_trials = [t for t in trials if t.values is not None and len(t.values) == 2 and t.values[1] <= 35.0]
        if valid_trials:
            best_trial = max(valid_trials, key=lambda x: x.values[0])
        else:
            valid_any = [t for t in trials if t.values is not None and len(t.values) == 2]
            if valid_any:
                best_trial = max(valid_any, key=lambda x: x.values[0])
            else:
                continue

        params = best_trial.params.copy()
        params["TIMEFRAME"] = tf_eval
        # [CRITICAL FIX] Restore hardcoded institutional params that Optuna doesn't store in trial.params
        params["LEVERAGE"] = 20
        params["USE_COMPOUNDING"] = True
        
        target_symbols = list(target) if isinstance(target, tuple) else [target]
        
        if args.mode == "multi":
            _logger.info("\n" + "═" * 60)
            _logger.info("  [BEST TRIAL PORTFOLIO INTERNAL BREAKDOWN (IS)]")
            _logger.info("-" * 60)
            _logger.info(f"  Target Score (CAGR/MDD): {best_trial.values[0]:.2f}% / {best_trial.values[1]:.2f}%")
            _logger.info(f"  Avg Portfolio CAGR: {best_trial.user_attrs.get('avg_cagr', 0):.2f}%")
            _logger.info(f"  Avg Portfolio MDD : {best_trial.user_attrs.get('avg_mdd', 0):.2f}%")
            _logger.info(f"  Avg Portfolio PF  : {best_trial.user_attrs.get('avg_pf', 0):.2f}")
            
            for s_eval in target_symbols:
                s_cagr = best_trial.user_attrs.get(f"{s_eval}_cv_cagr", -100)
                s_mdd = best_trial.user_attrs.get(f"{s_eval}_mdd", 0)
                _logger.info(f"  > {s_eval:<10}: {s_cagr:>7.2f}% CAGR | {s_mdd:>5.2f}% MDD")
            _logger.info("═" * 60)

        is_all_passed = True
        target_symbols = sorted(target_symbols) 
        for s_eval in target_symbols:
            s_is, r_is, m_is, t_is, _, pf_is, _, _, _ = evaluate_symbol_fold(UltimateStrategy(name=f"IS_{s_eval}", params=params), params, s_eval, tf_eval, data_maps[s_eval][tf_eval], data_maps[s_eval]["1d"], data_maps[s_eval][f"merge_idx_{tf_eval}"], None, data_maps[s_eval][f"is_start_idx_{tf_eval}"], len(data_maps[s_eval][tf_eval]))
            s_oos, r_oos, m_oos, t_oos, _, pf_oos, lc_oos, sc_oos, _ = evaluate_symbol_fold(UltimateStrategy(name=f"OOS_{s_eval}", params=params), params, s_eval, tf_eval, oos_data_maps[s_eval][tf_eval], oos_data_maps[s_eval]["1d"], oos_data_maps[s_eval][f"merge_idx_{tf_eval}"], None, oos_data_maps[s_eval][f"oos_start_idx_{tf_eval}"], len(oos_data_maps[s_eval][tf_eval]))
            go_nogo = run_go_nogo_check([], 0.0, [s_oos], m_oos, pf_oos, int(lc_oos), int(sc_oos), tf_eval)
            if not go_nogo.passed: is_all_passed = False
            final_summaries.append({"sym": s_eval, "tf": tf_eval, "is": (s_is, r_is, m_is, t_is, pf_is), "oos": (s_oos, r_oos, m_oos, t_oos, pf_oos), "passed": go_nogo.passed})
        
        best_score_final = best_trial.values[0] if best_trial.values else -100
        if best_score_final > 0 and is_all_passed:
            import json
            s_name = f"best_params_{('_'.join(target) if isinstance(target, tuple) else target).replace('/', '')}_{tf_eval}"
            
            # [UPGRADE] Save to 'results' directory
            results_dir = os.path.join(project_root, "results")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                
            json_path = os.path.join(results_dir, f"{s_name}.json")
            
            # Save only the optimal parameters for the live bot
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=4)
            
            _logger.info("  ⚡ Saved Live Bot Config: results/%s.json", s_name)
        else:
            _logger.info("  🚫 JSON Save Skipped: Institutional criteria not met.")

    if final_summaries:
        table_w = 120
        _logger.info("\n" + "═" * table_w)
        _logger.info(f"{'  FINAL OPTIMIZATION SUMMARY (IS vs OOS)':^{table_w}}")
        _logger.info("─" * table_w)
        header = f"{'Symbol':<12} | {'TF':<3} | {'IS (Score / Ret% / MDD% / Trd / PF)':^45} | {'OOS (Score / Ret% / MDD% / Trd / PF)':^45} | Status"
        _logger.info(header)
        _logger.info("─" * table_w)
        
        is_scores, is_rets, is_mdds, is_trds, is_pfs = [], [], [], [], []
        oos_scores, oos_rets, oos_mdds, oos_trds, oos_pfs = [], [], [], [], []

        for r in final_summaries:
            is_v = f"{r['is'][0]:>7.2f} / {r['is'][1]:>5.1f}% / {r['is'][2]:>4.1f}% / {int(r['is'][3]):>4} / {r['is'][4]:>4.2f}"
            oos_v = f"{r['oos'][0]:>7.2f} / {r['oos'][1]:>5.1f}% / {r['oos'][2]:>4.1f}% / {int(r['oos'][3]):>4} / {r['oos'][4]:>4.2f}"
            stat = "✅ PASS" if r['passed'] else "❌ FAIL"
            _logger.info(f"{r['sym']:<12} | {r['tf']:<3} | {is_v} | {oos_v} | {stat}")
            
            is_scores.append(r['is'][0]); is_rets.append(r['is'][1]); is_mdds.append(r['is'][2]); is_trds.append(r['is'][3]); is_pfs.append(r['is'][4])
            oos_scores.append(r['oos'][0]); oos_rets.append(r['oos'][1]); oos_mdds.append(r['oos'][2]); oos_trds.append(r['oos'][3]); oos_pfs.append(r['oos'][4])

        if args.mode == "multi":
            _logger.info("─" * table_w)
            port_is = f"{np.mean(is_scores):>7.2f} / {np.mean(is_rets):>5.1f}% / {np.mean(is_mdds):>4.1f}% / {int(np.sum(is_trds)):>4} / {np.mean(is_pfs):>4.2f}"
            port_oos = f"{np.mean(oos_scores):>7.2f} / {np.mean(oos_rets):>5.1f}% / {np.mean(oos_mdds):>4.1f}% / {int(np.sum(oos_trds)):>4} / {np.mean(oos_pfs):>4.2f}"
            _logger.info(f"{'PORTFOLIO':<12} | {args.tf:<3} | {port_is} | {port_oos} | {'---'}")
        _logger.info("═" * table_w + "\n")

if __name__ == "__main__":
    main()