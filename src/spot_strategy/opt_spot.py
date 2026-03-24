from __future__ import annotations

import argparse
import logging
import math
import os
import queue
import sys
import threading
import optuna
from optuna.trial import TrialState
from optuna.storages import InMemoryStorage
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import concurrent.futures
from multiprocessing import Manager
from tqdm import tqdm

project_root: str = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.spot_strategy.data_collector_spot import DataCollectorSpot
from src.spot_strategy.strategies_spot import UltimateSpotStrategy
from config.settings import DATA_DIR, SPOT_INITIAL_BALANCE
from config.opt_config import OPT_SPOT_CONFIG, get_search_space_spot, get_quarterly_window
from src.optimization.opt_utils import compute_segment_merge_index
from src.spot_strategy.opt_spot_utils.evaluator import (
    evaluate_symbol_fold,
    objective_spot,
    run_holdout_shared_cash_portfolio,
)
from src.spot_strategy.opt_spot_utils.go_nogo import (
    FinalDeploymentReportInput,
    SymbolGateRow,
    run_final_deployment_report,
    run_go_nogo_check,
    run_holdout_portfolio_shared_cash,
    run_holdout_portfolio_trade_floor,
    run_portfolio_discovery_veto,
)

import warnings
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
_logger: logging.Logger = logging.getLogger("opt_spot")

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

def _task_progress_key(target_obj: Any, tf: str) -> str:
    """Short tqdm key: portfolio -> SPOT_{tf}; single symbol -> SPOT_{tf}_ETH."""
    if isinstance(target_obj, (list, tuple)) and len(target_obj) > 1:
        return f"SPOT_{tf}"
    sym = target_obj[0] if isinstance(target_obj, (list, tuple)) else target_obj
    short = str(sym).replace("KRW-", "").replace("-", "")
    return f"SPOT_{tf}_{short}"


def _rebuild_is_data_maps_from_aligned_oos(
    data_maps: Dict[str, Dict[str, Any]],
    oos_data_maps: Dict[str, Dict[str, Any]],
    symbols: Sequence[str],
    tf: str,
    is_start_dt: pd.Timestamp,
    is_end_dt: pd.Timestamp,
) -> None:
    """After full-timeline alignment, derive IS-only `data_maps` so IS rows match OOS prefixes."""
    for sym in symbols:
        full = oos_data_maps[sym][tf]
        is_df = full[full["datetime"] < is_end_dt].copy().reset_index(drop=True)
        data_maps[sym][tf] = is_df
        m = is_df["datetime"] >= is_start_dt
        data_maps[sym][f"is_start_idx_{tf}"] = int(m.to_numpy().argmax()) if bool(m.any()) else 0
        data_maps[sym][f"merge_idx_{tf}"] = compute_segment_merge_index(
            is_df, data_maps[sym]["1d"]
        )


def _align_oos_dataframes_on_common_datetimes(
    oos_data_maps: Dict[str, Dict[str, Any]],
    symbols: Sequence[str],
    tf: str,
    is_end_dt: pd.Timestamp,
) -> None:
    """
    Same bar alignment as IS, for full OHLCV (IS+OOS) used in holdout shared-cash.
    """
    sym_list = list(symbols)
    if len(sym_list) < 2:
        return

    common = (
        oos_data_maps[sym_list[0]][tf][["datetime"]]
        .drop_duplicates(subset=["datetime"])
        .copy()
    )
    for sym in sym_list[1:]:
        right = (
            oos_data_maps[sym][tf][["datetime"]]
            .drop_duplicates(subset=["datetime"])
            .rename(columns={"datetime": "_dt_r"})
        )
        common = common.merge(right, left_on="datetime", right_on="_dt_r", how="inner")
        common = common[["datetime"]]

    if len(common) < 200:
        raise ValueError(
            f"Insufficient overlapping {tf} bars in OOS maps after alignment ({len(common)} < 200)."
        )

    common_order = common["datetime"].sort_values()
    for sym in sym_list:
        df = oos_data_maps[sym][tf]
        filtered = (
            df[df["datetime"].isin(common_order)]
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        oos_data_maps[sym][tf] = filtered
        m_oos = filtered["datetime"] >= is_end_dt
        oos_data_maps[sym][f"oos_start_idx_{tf}"] = (
            int(m_oos.to_numpy().argmax()) if bool(m_oos.any()) else len(filtered)
        )
        oos_data_maps[sym][f"merge_idx_{tf}"] = compute_segment_merge_index(
            filtered, oos_data_maps[sym]["1d"]
        )


def _resolve_parallel_plan(task_count: int, requested_jobs: int, requested_task_workers: int, requested_cpu_budget: int) -> _ParallelExecutionPlan:
    logical_cpus = max(1, os.cpu_count() or 1)
    cpu_budget = max(1, requested_cpu_budget or (4 if logical_cpus > 2 else 1))
    jobs_per_task = max(1, requested_jobs)
    task_workers = min(task_count, requested_task_workers) if requested_task_workers > 0 else min(task_count, max(1, cpu_budget // jobs_per_task))
    return _ParallelExecutionPlan(task_workers=max(1, task_workers), jobs_per_task=jobs_per_task, cpu_budget=cpu_budget, logical_cpus=logical_cpus, task_count=task_count)

def _run_tf_optimization(task: Tuple[Any, str], ctx: _TfOptimizationContext) -> Tuple[Tuple[Any, str], optuna.Study]:
    target_obj, tf = task
    target_str = "_".join(target_obj) if isinstance(target_obj, (list, tuple)) else target_obj
    progress_key = _task_progress_key(target_obj, tf)
    tf_study_name: str = f"OptSpot_{target_str.replace('/', '').replace('-', '')}_{tf}_{ctx.mode}"
    
    storage: Any
    if ctx.use_journal_storage:
        journal_path = Path(ctx.project_root) / f"optuna_spot_journal_{target_str.replace('/', '').replace('-', '')}_{tf}.log"
        _ensure_fresh_journal(journal_path)
        lock_obj = None
        if sys.platform == "win32":
            from src.futures_strategy.opt_futures_utils.win_journal_lock import WindowsNamedMutexJournalLock
            lock_obj = WindowsNamedMutexJournalLock(str(journal_path))
        backend = JournalFileBackend(str(journal_path), lock_obj=lock_obj)
        storage = _ThreadSafeJournalStorageWrapper(JournalStorage(backend))
    else:
        storage = InMemoryStorage()

    sampler = optuna.samplers.TPESampler(
        seed=ctx.seeds[0],
        n_startup_trials=int(OPT_SPOT_CONFIG.get("tpe_n_startup_trials", 96)),
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(OPT_SPOT_CONFIG.get("tpe_pruner_n_startup_trials", 10)),
        n_warmup_steps=int(OPT_SPOT_CONFIG.get("tpe_pruner_n_warmup_steps", 2)),
    )

    study = optuna.create_study(
        study_name=tf_study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    best_so_far: float = float("-inf")

    def _progress_cb(study_inner: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal best_so_far
        cur_val: float = 0.0
        if trial.value is not None:
            try:
                cur_val = float(trial.value)
            except Exception:
                cur_val = 0.0
        if cur_val > best_so_far:
            best_so_far = cur_val
        ctx.progress_queue.put(
            (progress_key, trial.number + 1, ctx.n_trials, 0.0 if best_so_far == float("-inf") else best_so_far)
        )

    tf_space = get_search_space_spot(tf)

    def _objective_with_logging(trial: optuna.Trial) -> float:
        try:
            return objective_spot(
                trial,
                data_maps=ctx.data_maps,
                symbols=ctx.symbols,
                tf_target=tf,
                space=tf_space,
                mode=ctx.mode,
                project_root=ctx.project_root,
            )
        except optuna.TrialPruned:
            raise
        except Exception:
            _logger.exception("[%s/%s] Trial %d failed.", target_str, tf, trial.number)
            raise

    _logger.info("[%s/%s] Starting Spot TPE (CPCV discovery, growth objective)...", target_str, tf)
    study.optimize(_objective_with_logging, n_trials=ctx.n_trials, n_jobs=ctx.n_jobs, catch=(Exception,), callbacks=[_progress_cb])
    return (target_obj, tf), study

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="KRW-BTC,KRW-XRP,KRW-ETH,KRW-LINK,KRW-ADA,KRW-DOT")
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="multi")
    parser.add_argument("--trials", type=int, default=OPT_SPOT_CONFIG["total_trials"])
    parser.add_argument("--jobs", type=int, default=int(OPT_SPOT_CONFIG.get("n_jobs", 8)))
    parser.add_argument("--task-workers", type=int, default=int(OPT_SPOT_CONFIG.get("task_workers", 1)))
    parser.add_argument("--tf", type=str, choices=["4h"], default=OPT_SPOT_CONFIG.get("TARGET_TIMEFRAMES", ["4h"])[0])
    parser.add_argument("--reference-date", type=str, default=None)
    args = parser.parse_args()

    FETCH_START_DATE, START_DATE, IS_END_DATE, END_DATE = get_quarterly_window(args.reference_date)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    collector = DataCollectorSpot()
    data_maps = {}
    oos_data_maps = {}
    _logger.info("Loading Spot data for %d symbols (discovery + holdout)...", len(symbols))
    valid_symbols = []
    for sym in symbols:
        data_maps[sym] = {}; oos_data_maps[sym] = {}
        skip_symbol = False
        for tf in [args.tf, "1d"]:
            full_df = collector.collect_and_save(sym, tf, FETCH_START_DATE, END_DATE)
            
            if full_df is None or full_df.empty or "datetime" not in full_df.columns:
                _logger.warning(f"⚠️ [{sym}] No data available for {tf}. Skipping this symbol.")
                skip_symbol = True
                break
                
            # Spot doesn't use funding fees, skipping merge_funding_into_ohlcv
            tz = full_df["datetime"].dt.tz
            is_start_dt = pd.to_datetime(START_DATE).tz_localize(tz) if tz else pd.to_datetime(START_DATE)
            is_end_dt = pd.to_datetime(IS_END_DATE).tz_localize(tz) if tz else pd.to_datetime(IS_END_DATE)
            data_maps[sym][tf] = full_df[full_df["datetime"] < is_end_dt].reset_index(drop=True)
            m = data_maps[sym][tf]["datetime"] >= is_start_dt
            data_maps[sym][f"is_start_idx_{tf}"] = int(m.to_numpy().argmax()) if m.any() else 0
            oos_data_maps[sym][tf] = full_df
            m_oos = full_df["datetime"] >= is_end_dt
            oos_data_maps[sym][f"oos_start_idx_{tf}"] = int(m_oos.to_numpy().argmax()) if m_oos.any() else len(full_df)
        
        if skip_symbol:
            continue

        data_maps[sym][f"merge_idx_{args.tf}"] = compute_segment_merge_index(data_maps[sym][args.tf], data_maps[sym]["1d"])
        oos_data_maps[sym][f"merge_idx_{args.tf}"] = compute_segment_merge_index(oos_data_maps[sym][args.tf], oos_data_maps[sym]["1d"])
        valid_symbols.append(sym)

    if "KRW-BTC" in valid_symbols:
        btc_4h = data_maps["KRW-BTC"][args.tf][["datetime", "close"]].copy()
        btc_4h = btc_4h.rename(columns={"close": "btc_close"})
        btc_oos = oos_data_maps["KRW-BTC"][args.tf][["datetime", "close"]].copy()
        btc_oos = btc_oos.rename(columns={"close": "btc_close"})
        for sym in valid_symbols:
            if sym == "KRW-BTC":
                continue
            data_maps[sym][args.tf] = data_maps[sym][args.tf].merge(btc_4h, on="datetime", how="left")
            oos_data_maps[sym][args.tf] = oos_data_maps[sym][args.tf].merge(btc_oos, on="datetime", how="left")

    if len(valid_symbols) >= 2:
        ref_tz = oos_data_maps[valid_symbols[0]][args.tf]["datetime"].dt.tz
        is_start_align = pd.to_datetime(START_DATE)
        is_end_align = pd.to_datetime(IS_END_DATE)
        if ref_tz is not None:
            is_start_align = is_start_align.tz_localize(ref_tz)
            is_end_align = is_end_align.tz_localize(ref_tz)
        _align_oos_dataframes_on_common_datetimes(
            oos_data_maps, valid_symbols, args.tf, is_end_align
        )
        _rebuild_is_data_maps_from_aligned_oos(
            data_maps, oos_data_maps, valid_symbols, args.tf, is_start_align, is_end_align
        )
        _logger.info(
            "Aligned full %s on common datetimes: IS=%d bars, OOS full=%d bars (%d symbols).",
            args.tf,
            len(data_maps[valid_symbols[0]][args.tf]),
            len(oos_data_maps[valid_symbols[0]][args.tf]),
            len(valid_symbols),
        )

    if not valid_symbols:
        _logger.error("❌ No valid symbols with data found. Aborting optimization.")
        return

    tasks = [(tuple(valid_symbols), args.tf)] if args.mode == "multi" else [(s, args.tf) for s in valid_symbols]
    plan = _resolve_parallel_plan(len(tasks), args.jobs, args.task_workers, 0)
    use_mp = len(tasks) > 1 and plan.task_workers > 1
    manager = Manager() if use_mp else None; progress_queue = manager.Queue() if manager else queue.Queue()

    tf_bars = {}
    for i, (target, tf) in enumerate(tasks):
        progress_key = _task_progress_key(target, tf)
        tf_bars[progress_key] = tqdm(
            total=args.trials,
            desc=f"{progress_key} …",
            position=i,
            leave=True,
        )

    def _progress_listener():
        while True:
            msg = progress_queue.get()
            if msg is None: break
            k, cur, tot, b_val = msg
            bar = tf_bars[k]
            bar.n = cur
            bar.set_description(f"{k} | Best: {b_val:.2f}")
            bar.refresh()

    progress_thread = threading.Thread(target=_progress_listener, daemon=True); progress_thread.start()

    best_results = {}
    pending_json_writes: List[Tuple[str, Dict[str, Any], float, bool]] = []
    try:
        if use_mp:
            with concurrent.futures.ProcessPoolExecutor(max_workers=plan.task_workers) as exec:
                futures = {exec.submit(_run_tf_optimization, t, _TfOptimizationContext(
                    clean_symbol=("_".join(t[0]) if isinstance(t[0], tuple) else t[0]).replace("/", ""),
                    seeds=OPT_SPOT_CONFIG["seeds"], n_trials=args.trials, n_jobs=plan.jobs_per_task,
                    data_maps={s: data_maps[s] for s in (list(t[0]) if isinstance(t[0], tuple) else [t[0]])},
                    symbols=(list(t[0]) if isinstance(t[0], tuple) else [t[0]]), project_root=project_root,
                    progress_queue=progress_queue, mode=args.mode
                )): t for t in tasks}
                for f in concurrent.futures.as_completed(futures):
                    t_res, study = f.result()
                    best_results[t_res] = study
        else:
            for t in tasks:
                t_res, study = _run_tf_optimization(
                    t,
                    _TfOptimizationContext(
                        clean_symbol=("_".join(t[0]) if isinstance(t[0], tuple) else t[0]).replace("/", ""),
                        seeds=OPT_SPOT_CONFIG["seeds"],
                        n_trials=args.trials,
                        n_jobs=plan.jobs_per_task,
                        data_maps=data_maps,
                        symbols=(list(t[0]) if isinstance(t[0], tuple) else [t[0]]),
                        project_root=project_root,
                        progress_queue=progress_queue,
                        mode=args.mode,
                        use_journal_storage=False,
                    ),
                )
                best_results[t_res] = study
    finally:
        progress_queue.put(None); progress_thread.join(timeout=2.0)
        if manager: manager.shutdown()

    final_summaries = []
    top_k = int(OPT_SPOT_CONFIG.get("SPOT_SHORTLIST_TOP_K", 50))
    max_ho_cvar = float(OPT_SPOT_CONFIG.get("SPOT_HOLDOUT_MAX_CVAR_PCT", 25.0))
    min_pf_trades = int(OPT_SPOT_CONFIG.get("SPOT_HOLDOUT_MIN_PORTFOLIO_LONG_TRADES", 8))
    holdout_min_tail = float(OPT_SPOT_CONFIG.get("SPOT_HOLDOUT_MIN_TAIL_RATIO", 2.0))
    holdout_min_cagr = float(OPT_SPOT_CONFIG.get("SPOT_HOLDOUT_MIN_CAGR_PCT", 30.0))
    holdout_mdd_limit = float(OPT_SPOT_CONFIG.get("SPOT_HOLDOUT_MDD_LIMIT_PCT", 45.0))
    holdout_hwm_max_days = float(OPT_SPOT_CONFIG.get("SPOT_HOLDOUT_HWM_RECOVERY_MAX_DAYS", 300.0))
    holdout_alpha_floor = float(OPT_SPOT_CONFIG.get("SPOT_HOLDOUT_ALPHA_DECAY_FLOOR_PCT", -50.0))
    gate1_sqn_min = float(OPT_SPOT_CONFIG.get("SPOT_GATE1_SQN_MIN", 3.0))
    gate1_psort_min = float(OPT_SPOT_CONFIG.get("SPOT_GATE1_PATH_SORTINO_MIN", 2.5))
    gate1_tr_min = float(OPT_SPOT_CONFIG.get("SPOT_GATE1_TAIL_RATIO_MIN", 3.0))
    discovery_dsr_min = float(OPT_SPOT_CONFIG.get("SPOT_DISCOVERY_DSR_MIN", -1.0))

    for (target, tf_eval), study in best_results.items():
        if study is None:
            continue
        completed = [
            t
            for t in study.trials
            if t.state == TrialState.COMPLETE and t.value is not None
        ]
        if not completed:
            continue
        ranked = sorted(completed, key=lambda tr: float(tr.value), reverse=True)[:top_k]
        best_trial = max(
            ranked,
            key=lambda tr: float(tr.user_attrs.get("min_path_terminal_wealth_ratio", 0.0)),
        )

        params = best_trial.params.copy()
        params["TIMEFRAME"] = tf_eval
        params["LEVERAGE"] = 1
        params["USE_COMPOUNDING"] = True

        target_symbols = sorted(list(target) if isinstance(target, tuple) else [target])

        psr_v = float(best_trial.user_attrs.get("psr_paths", 0.0))
        dsr_v = float(best_trial.user_attrs.get("dsr_paths", 0.0))
        veto = run_portfolio_discovery_veto(
            psr=psr_v,
            dsr=dsr_v,
            dsr_min=discovery_dsr_min,
        )
        veto_ok = bool(veto.passed)

        port_ho = run_holdout_shared_cash_portfolio(params, target_symbols, tf_eval, oos_data_maps)
        oos_dd_days = float(port_ho["dd_bars"]) / 6.0

        symbol_fold_payloads: List[Dict[str, Any]] = []
        is_cagr_vals: List[float] = []
        for s_eval in target_symbols:
            s_is, r_is, m_is, t_is, wr_is, pf_is, lc_is, _, tr_is = evaluate_symbol_fold(
                UltimateSpotStrategy(name=f"IS_{s_eval}", params=params),
                params,
                s_eval,
                tf_eval,
                data_maps[s_eval][tf_eval],
                data_maps[s_eval]["1d"],
                data_maps[s_eval][f"merge_idx_{tf_eval}"],
                None,
                data_maps[s_eval][f"is_start_idx_{tf_eval}"],
                len(data_maps[s_eval][tf_eval]),
            )
            s_oos, r_oos, m_oos, trd_oos, wr_oos, pf_oos, lc_oos, _, tail_oos = evaluate_symbol_fold(
                UltimateSpotStrategy(name=f"OOS_{s_eval}", params=params),
                params,
                s_eval,
                tf_eval,
                oos_data_maps[s_eval][tf_eval],
                oos_data_maps[s_eval]["1d"],
                oos_data_maps[s_eval][f"merge_idx_{tf_eval}"],
                None,
                oos_data_maps[s_eval][f"oos_start_idx_{tf_eval}"],
                len(oos_data_maps[s_eval][tf_eval]),
            )
            is_cagr_vals.append(s_is)
            symbol_fold_payloads.append(
                {
                    "sym": s_eval,
                    "is_row": (s_is, r_is, m_is, t_is, pf_is),
                    "oos": {
                        "cagr": s_oos,
                        "ret": r_oos,
                        "mdd": m_oos,
                        "trd": trd_oos,
                        "wr": wr_oos,
                        "pf": pf_oos,
                        "lc": lc_oos,
                        "tail": tail_oos,
                    },
                }
            )
        is_mean_cagr = float(np.mean(is_cagr_vals)) if is_cagr_vals else 0.0

        trade_floor = run_holdout_portfolio_trade_floor(
            portfolio_long_trades=int(port_ho["long_trades"]),
            min_portfolio_trades=min_pf_trades,
        )
        shared_cash_gate = run_holdout_portfolio_shared_cash(
            portfolio_cagr_pct=float(port_ho["portfolio_cagr_pct"]),
            portfolio_mdd_pct=float(port_ho["mdd_pct"]),
            portfolio_cvar_pct=float(port_ho["cvar_pct"]),
            portfolio_tail_ratio=float(port_ho["tail_ratio"]),
            min_path_terminal_wealth_ratio=float(port_ho["min_path_tw"]),
            max_cvar_pct=max_ho_cvar,
            tail_ratio_min=holdout_min_tail,
            cagr_min_pct=holdout_min_cagr,
            mdd_limit_pct=holdout_mdd_limit,
            oos_dd_days=oos_dd_days,
            hw_recovery_days_max=holdout_hwm_max_days,
            is_cagr_pct=is_mean_cagr,
            alpha_decay_floor_pct=holdout_alpha_floor,
        )
        is_all_passed = bool(veto_ok and trade_floor.passed and shared_cash_gate.passed)

        n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        _logger.info(
            "Post-study: complete=%d shortlist=%d (top_k=%d)",
            n_complete,
            len(ranked),
            top_k,
        )
        _logger.info(
            "Post-study: selected trial=%d (max min_path_terminal_wealth_ratio within top-%d by objective)",
            int(best_trial.number),
            len(ranked),
        )
        _logger.info("%s", veto.summary)
        _logger.info("%s", trade_floor.summary)
        _logger.info("%s", shared_cash_gate.summary)
        dd_bars = float(port_ho.get("dd_bars", 0.0))
        _logger.info(
            "Holdout shared-cash: CAGR=%.2f%% MDD=%.2f%% CVaR=%.2f%% trades=%d tw=%.4f dd_bars=%d tail=%.2f",
            float(port_ho["portfolio_cagr_pct"]),
            float(port_ho["mdd_pct"]),
            float(port_ho["cvar_pct"]),
            int(port_ho["long_trades"]),
            float(port_ho["min_path_tw"]),
            int(dd_bars),
            float(port_ho["tail_ratio"]),
        )

        symbol_gate_rows: List[SymbolGateRow] = []
        for pl in symbol_fold_payloads:
            s_eval = pl["sym"]
            s_is, r_is, m_is, t_is, pf_is = pl["is_row"]
            o = pl["oos"]
            s_oos = float(o["cagr"])
            r_oos = float(o["ret"])
            m_oos = float(o["mdd"])
            trd_oos = int(o["trd"])
            wr_oos = float(o["wr"])
            pf_oos = float(o["pf"])
            lc_oos = int(o["lc"])
            tail_oos = float(o["tail"])
            go_nogo = run_go_nogo_check(
                [],
                0.0,
                [s_oos],
                m_oos,
                tail_oos,
                lc_oos,
                tf_eval,
                mdd_limit_pct=holdout_mdd_limit,
                tail_ratio_min=holdout_min_tail,
            )
            symbol_gate_rows.append(
                SymbolGateRow(
                    symbol=s_eval,
                    net_cagr_pct=s_oos,
                    max_mdd_pct=m_oos,
                    tail_ratio=tail_oos,
                    win_rate_pct=wr_oos,
                    trade_count=trd_oos,
                )
            )
            final_summaries.append(
                {
                    "sym": s_eval,
                    "tf": tf_eval,
                    "is": (s_is, r_is, m_is, t_is, pf_is),
                    "oos": (s_oos, r_oos, m_oos, trd_oos, pf_oos),
                    "passed": go_nogo.passed,
                }
            )

        oos_cagrs = [float(r.net_cagr_pct) for r in symbol_gate_rows]
        pos_sum = float(sum(max(0.0, x) for x in oos_cagrs))
        if pos_sum > 1e-9 and oos_cagrs:
            max_share = float(max(oos_cagrs)) / pos_sum
            loso_warning = (
                f"경고 (단일 심볼 OOS CAGR 비중 {max_share:.0%} >= 40%)"
                if max_share >= 0.4
                else f"안전 (특정 심볼 의존도 {max_share:.0%} < 40%)"
            )
        else:
            loso_warning = "N/A (OOS CAGR 비중 산출 불가)"

        alpha_decay_pct = float(shared_cash_gate.advisory.get("alpha_decay_pct", -100.0))

        hard_passed = (
            sum(1 for v in veto.details.values() if v)
            + (1 if trade_floor.passed else 0)
            + sum(1 for v in shared_cash_gate.details.values() if v)
        )
        hard_total = len(veto.details) + 1 + len(shared_cash_gate.details)

        report = run_final_deployment_report(
            FinalDeploymentReportInput(
                gate1_sqn=float(best_trial.user_attrs.get("gate1_sqn", 0.0)),
                gate1_path_sortino=float(best_trial.user_attrs.get("gate1_path_sortino", 0.0)),
                gate1_tail_ratio=float(best_trial.user_attrs.get("cpcv_path_tail_ratio", 0.0)),
                cpcv_mean_path_return_pct=float(best_trial.user_attrs.get("cpcv_mean_path_return_pct", 0.0)),
                cpcv_worst_segment_mdd_pct=float(best_trial.user_attrs.get("cpcv_worst_segment_mdd_pct", 0.0)),
                sqn_target=gate1_sqn_min,
                path_sortino_target=gate1_psort_min,
                tail_ratio_target=gate1_tr_min,
                moic=float(port_ho["moic"]),
                initial_capital_krw=float(SPOT_INITIAL_BALANCE),
                oos_net_cagr_pct=float(port_ho["portfolio_cagr_pct"]),
                oos_mdd_pct=float(port_ho["mdd_pct"]),
                hw_recovery_days=oos_dd_days,
                alpha_decay_pct=alpha_decay_pct,
                oos_cagr_target_pct=holdout_min_cagr,
                oos_mdd_limit_pct=holdout_mdd_limit,
                hw_recovery_max_days=holdout_hwm_max_days,
                alpha_decay_floor_pct=holdout_alpha_floor,
                symbol_rows=symbol_gate_rows,
                loso_warning=loso_warning,
                hard_passed=hard_passed,
                hard_total=hard_total,
                final_decision_go=is_all_passed,
            )
        )
        _logger.info(
            "\n%s",
            report,
        )

        best_score_final = float(best_trial.value) if best_trial.value is not None else -100.0

        mean_log = float(best_trial.user_attrs.get("mean_log_terminal_wealth", 0.0))
        should_save = bool(mean_log > 0.0 and is_all_passed)
        # File name is required by downstream loaders; overwrite risk is limited to multi-mode.
        clean_sym = str(target).replace("/", "").replace("-", "") if not isinstance(target, tuple) else ""
        json_filename = f"spot_params_{tf_eval}.json" if args.mode == "multi" else f"spot_params_{tf_eval}_{clean_sym}.json"
        pending_json_writes.append((json_filename, params, best_score_final, should_save))

    # JSON save logs last
    if pending_json_writes:
        import json
        results_dir = Path(project_root) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        for json_filename, params, best_score_final, should_save in pending_json_writes:
            json_path = results_dir / json_filename
            if should_save:
                json_path.write_text(json.dumps(params, indent=4), encoding="utf-8")
                _logger.info("Saved config: %s", json_path.resolve())
            else:
                _logger.info(
                    "JSON save skipped: criteria not met (growth_score / gates). objective=%.4f",
                    best_score_final,
                )

if __name__ == "__main__":
    main()
