from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import queue
import tempfile
import time
import sys
import threading
import traceback
import sqlite3

import optuna
from optuna.pruners import MedianPruner, PatientPruner
from optuna.samplers import QMCSampler, TPESampler
from optuna.storages import InMemoryStorage
from optuna.trial import TrialState
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import concurrent.futures
from collections import Counter
from multiprocessing import Manager
from tqdm import tqdm

# Project Root Setup
project_root: str = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.data_collector import DataCollector
from src.domain.futures.strategies_futures import UltimateStrategy
from config.settings import (
    FUTURES_CACHE_DIR,
    FUTURES_DATA_DIR,
    FUTURES_INITIAL_BALANCE,
    SLIPPAGE_RATE,
    TRADING_FEE_RATE,
)
from config.opt_config import (
    OPT_FUTURES_CONFIG,
    FUTURES_ANCHOR_SYMBOLS,
    FUTURES_DYNAMIC_CANDIDATE_POOL,
    FUTURES_SCREENER_CONFIG,
    get_quarterly_window,
    get_search_space_futures,
)

from src.core.optimization.opt_utils import compute_segment_merge_index
from src.domain.futures.funding_utils import merge_funding_into_ohlcv

from src.domain.futures.opt_futures_utils.db_utils import save_study_to_sqlite
from src.domain.futures.opt_futures_utils.cv_utils import build_cpcv_test_paths_with_fallback
from src.domain.futures.opt_futures_utils.evaluator import (
    evaluate_symbol_fold,
    objective_futures,
    run_oos_margin_shared_portfolio,
    run_multi_window_oos_holdout,
    compute_regime_conditional_oos_metrics,
)
from src.domain.futures.opt_futures_utils.go_nogo import run_go_nogo_check
from src.domain.futures.opt_futures_utils.go_nogo_futures import (
    FuturesDeploymentReportInput,
    FuturesSymbolGateRow,
    run_futures_deployment_report,
    run_multi_window_oos_gate,
    format_regime_oos_diagnostic_block,
)
from src.domain.spot.opt_spot_utils.go_nogo import (
    run_portfolio_discovery_veto,
    run_holdout_portfolio_trade_floor,
)
from src.domain.futures.opt_futures_utils.universe_screener_futures import (
    screen_futures_symbol_refinement,
    screen_futures_universe,
)
from src.domain.futures.opt_futures_utils.combination_screener_futures import (
    CombinationScoreFutures,
    run_combination_screening_futures,
)
from src.domain.futures.opt_futures_utils.opt_params import (
    build_combined_param_space_futures,
    build_multi_combo_param_space_futures,
)

import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
_logger: logging.Logger = logging.getLogger("opt_futures")

SEP_WIDTH: int = 60
PROGRESS_MIN_INTERVAL: float = 0.2
MODE_MULTI: str = "multi"
BEST_PARAMS_FUTURES_JSON_STEM: str = "best_futures_4h"

EMBARGO_BARS: Dict[str, int] = {
    "1h": 24,
    "4h": 6,
}


def _ensure_fresh_sqlite(db_path: Path) -> None:
    for suffix in ["", "-wal", "-shm"]:
        p = Path(str(db_path) + suffix)
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass


def _build_sqlite_storage(db_path: Path) -> optuna.storages.RDBStorage:
    url: str = f"sqlite:///{db_path.absolute()}"
    storage: optuna.storages.RDBStorage = optuna.storages.RDBStorage(
        url=url,
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    return storage


def futures_frozen_trial_constraints(trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
    pf = float(trial.user_attrs.get("avg_pf", 0.0) or 0.0)
    win_rate = float(trial.user_attrs.get("avg_win_rate", 0.0) or 0.0)
    trades = float(trial.user_attrs.get("avg_trades", 0.0) or 0.0)
    min_sym_pf = float(trial.user_attrs.get("min_sym_pf", 0.0) or 0.0)
    avg_mdd = float(trial.user_attrs.get("avg_mdd", 100.0) or 100.0)
    ls_ratio = float(trial.user_attrs.get("long_short_ratio", 0.0) or 0.0)
    return (
        1.2 - pf,
        25.0 - win_rate,
        20.0 - trades,
        1.1 - min_sym_pf,
        avg_mdd - 25.0,
        0.15 - ls_ratio,
    )


def _futures_tpe_sampler(seed: int) -> TPESampler:
    return TPESampler(
        seed=seed,
        n_startup_trials=0,
        multivariate=True,
        group=True,
        constant_liar=True,
        constraints_func=futures_frozen_trial_constraints,
    )


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
    mode: str = MODE_MULTI
    use_journal_storage: bool = True
    parallel_tpe_workers: int = 1
    disable_inner_tpe_pool: bool = False
    fresh_journal: bool = True
    progress_min_interval: float = PROGRESS_MIN_INTERVAL
    signal_cache_dir: str = ""
    locked_param_space: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class _FuturesExecutionPlan:
    outer_task_workers: int
    jobs_per_task: int
    parallel_tpe_workers: int
    use_outer_process_pool: bool
    logical_cpus: int
    task_count: int

def _split_tpe_trials(total: int, n_workers: int) -> List[int]:
    if total <= 0:
        return []
    n_workers = max(1, min(int(n_workers), int(total)))
    base = total // n_workers
    rem = total % n_workers
    return [base + (1 if i < rem else 0) for i in range(n_workers)]


def _resolve_futures_execution_plan(
    task_count: int,
    mode: str,
    requested_jobs: int,
    requested_task_workers: int,
) -> _FuturesExecutionPlan:
    worker_cap = 3
    logical_cpus = max(1, os.cpu_count() or 1)
    if mode == MODE_MULTI and task_count == 1:
        tw = requested_task_workers if requested_task_workers > 0 else requested_jobs
        tw = max(1, min(int(tw), logical_cpus, worker_cap))
        return _FuturesExecutionPlan(
            outer_task_workers=1,
            jobs_per_task=1,
            parallel_tpe_workers=tw,
            use_outer_process_pool=False,
            logical_cpus=logical_cpus,
            task_count=task_count,
        )

    jobs = max(1, min(int(requested_jobs), logical_cpus, worker_cap))
    if requested_task_workers <= 0:
        cpu_budget = max(1, 4 if logical_cpus > 2 else 1)
        outer_tw = min(task_count, max(1, cpu_budget // jobs))
    else:
        outer_tw = min(task_count, int(requested_task_workers))
    outer_tw = max(1, min(outer_tw, worker_cap))
    use_outer = task_count > 1 and outer_tw > 1
    return _FuturesExecutionPlan(
        outer_task_workers=outer_tw,
        jobs_per_task=jobs,
        parallel_tpe_workers=1,
        use_outer_process_pool=use_outer,
        logical_cpus=logical_cpus,
        task_count=task_count,
    )


def _task_progress_key(target_obj: Any, tf: str) -> str:
    if isinstance(target_obj, (list, tuple)):
        base = "_".join(str(x) for x in target_obj)
    else:
        base = str(target_obj)
    return f"{base}_{tf}"


def _rebuild_is_data_maps_from_aligned_oos(
    data_maps: Dict[str, Dict[str, Any]],
    oos_data_maps: Dict[str, Dict[str, Any]],
    symbols: Sequence[str],
    tf: str,
    is_start_dt: pd.Timestamp,
    is_end_dt: pd.Timestamp,
) -> None:
    for sym in symbols:
        full = oos_data_maps[sym][tf]
        is_df = full[full["datetime"] < is_end_dt].copy().reset_index(drop=True)
        data_maps[sym][tf] = is_df
        m = is_df["datetime"] >= is_start_dt
        data_maps[sym][f"is_start_idx_{tf}"] = int(m.to_numpy().argmax()) if bool(m.any()) else 0
        data_maps[sym][f"merge_idx_{tf}"] = compute_segment_merge_index(is_df, data_maps[sym]["1d"])


def _align_oos_dataframes_on_common_datetimes(
    oos_data_maps: Dict[str, Dict[str, Any]],
    symbols: Sequence[str],
    tf: str,
    is_end_dt: pd.Timestamp,
) -> None:
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


def _best_trial_from_study(study: optuna.Study) -> Optional[optuna.trial.FrozenTrial]:
    try:
        return study.best_trial
    except Exception:
        complete_trials = [
            t
            for t in study.get_trials(deepcopy=True)
            if t.state == TrialState.COMPLETE and t.value is not None
        ]
        if complete_trials:
            return max(complete_trials, key=lambda t: float(t.value) if t.value is not None else -1e9)
        return None


def _select_best_trial_from_shortlist(
    ranked: List[optuna.trial.FrozenTrial],
) -> optuna.trial.FrozenTrial:
    """
    Within ~2% of top objective value, prefer higher CPCV robustness (Spot parity).
    """
    if not ranked:
        raise ValueError("ranked trials empty")
    top_val = float(ranked[0].value if ranked[0].value is not None else -1e9)
    band = max(abs(top_val) * 0.02, 0.02)
    close_trials = [
        t
        for t in ranked
        if t.value is not None and abs(float(t.value) - top_val) <= band
    ]
    pool = close_trials if close_trials else ranked[:1]

    def robust_key(t: optuna.trial.FrozenTrial) -> Tuple[float, float, float, float]:
        ua = t.user_attrs
        return (
            float(ua.get("psr_paths", ua.get("gate1_psr", 0.0))),
            float(ua.get("gate1_sqn", 0.0)),
            float(ua.get("gate1_dsr", 0.0)),
            float(ua.get("gate1_tail_ratio", 0.0)),
        )

    return max(pool, key=robust_key)


def _futures_tpe_worker_run(payload: Dict[str, Any]) -> None:
    import gc

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")

    db_path = Path(str(payload["journal_path"]))
    tf_study_name = str(payload["study_name"])
    tf = str(payload["tf"])
    target_str = str(payload["target_str"])
    symbols = list(payload["symbols"])
    mode = str(payload["mode"])
    seed = int(payload["seed"])
    n_trials = int(payload["n_trials"])
    n_trials_total = int(payload["n_trials_total"])
    prebuilt_cpcv_bundle = payload.get("prebuilt_cpcv_bundle")
    data_maps_file = str(payload.get("data_maps_file", ""))
    if data_maps_file and Path(data_maps_file).is_file():
        with Path(data_maps_file).open("rb") as f:
            data_maps: Dict[str, Dict[str, Any]] = pickle.load(f)
    else:
        data_maps = payload.get("data_maps", {})
    progress_queue = payload.get("progress_queue")
    project_root = str(payload["project_root"])
    signal_cache_dir = str(payload.get("signal_cache_dir", ""))
    progress_key = str(payload["progress_key"])
    raw_space = payload.get("locked_param_space")
    tf_space: Dict[str, Any] = (
        raw_space if isinstance(raw_space, dict) else get_search_space_futures(tf)
    )

    base_pruner = MedianPruner(
        n_startup_trials=int(OPT_FUTURES_CONFIG.get("tpe_pruner_n_startup_trials", 10)),
        n_warmup_steps=int(OPT_FUTURES_CONFIG.get("tpe_pruner_n_warmup_steps", 8)),
    )
    pruner = PatientPruner(
        base_pruner,
        patience=int(OPT_FUTURES_CONFIG.get("tpe_pruner_patience", 2)),
    )
    storage = _build_sqlite_storage(db_path)
    study = optuna.load_study(
        study_name=tf_study_name,
        storage=storage,
        sampler=_futures_tpe_sampler(seed),
        pruner=pruner,
    )

    try:
        best_so_far = float(study.best_value)
    except ValueError:
        best_so_far = float("-inf")

    def _progress_cb(_study_inner: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal best_so_far
        if progress_queue is None:
            return
        if trial.value is not None:
            try:
                cur_val = float(trial.value)
            except Exception:
                cur_val = 0.0
            if cur_val > best_so_far:
                best_so_far = cur_val
        progress_queue.put(
            (
                progress_key,
                trial.number + 1,
                n_trials_total,
                float(trial.user_attrs.get("kelly_score_pct", best_so_far * 100.0))
                if best_so_far != float("-inf")
                else 0.0,
            )
        )

    cache_root = Path(signal_cache_dir) if signal_cache_dir else FUTURES_CACHE_DIR

    def _objective_with_logging(trial: optuna.Trial) -> float:
        try:
            return float(
                objective_futures(
                    trial,
                    data_maps=data_maps,
                    symbols=symbols,
                    tf_target=tf,
                    space=tf_space,
                    mode=mode,
                    project_root=project_root,
                    prebuilt_cpcv_bundle=prebuilt_cpcv_bundle,
                    signal_disk_cache_root=cache_root,
                )
            )
        except optuna.TrialPruned:
            raise
        except Exception:
            _logger.exception("[%s/%s] Trial %d failed.", target_str, tf, trial.number)
            raise

    try:
        study.optimize(
            _objective_with_logging,
            n_trials=n_trials,
            n_jobs=1,
            catch=(Exception,),
            callbacks=[_progress_cb],
        )
    finally:
        del data_maps
        gc.collect()

def _run_tf_optimization(
    task: Tuple[Any, str], ctx: _TfOptimizationContext
) -> Tuple[Tuple[Any, str], Optional[optuna.Study]]:
    target_obj, tf = task
    target_str = "_".join(target_obj) if isinstance(target_obj, (list, tuple)) else target_obj
    progress_key = _task_progress_key(target_obj, tf)
    tf_study_name: str = f"OptFutures_{target_str.replace('/', '')}_{tf}_{ctx.mode}"

    db_path: Path | None = None
    storage: optuna.storages.RDBStorage | InMemoryStorage
    if ctx.use_journal_storage:
        db_filename = f"opt_futures_{target_str.replace('/', '')}_{tf}_{ctx.mode}.db"
        shm_p = Path("/dev/shm")
        if shm_p.exists() and os.access(shm_p, os.W_OK):
            db_path = shm_p / db_filename
        else:
            results_dir = Path(ctx.project_root) / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            db_path = results_dir / db_filename
        if ctx.fresh_journal:
            _ensure_fresh_sqlite(db_path)
        storage = _build_sqlite_storage(db_path)
    else:
        storage = InMemoryStorage()

    seed = int(ctx.seeds[0])
    n_qmc = min(int(OPT_FUTURES_CONFIG.get("tpe_n_startup_trials", 256)), int(ctx.n_trials))
    qmc_sampler = QMCSampler(qmc_type="sobol", scramble=True, seed=seed, warn_independent_sampling=False)
    base_pruner = MedianPruner(
        n_startup_trials=int(OPT_FUTURES_CONFIG.get("tpe_pruner_n_startup_trials", 10)),
        n_warmup_steps=int(OPT_FUTURES_CONFIG.get("tpe_pruner_n_warmup_steps", 8)),
    )
    pruner = PatientPruner(
        base_pruner,
        patience=int(OPT_FUTURES_CONFIG.get("tpe_pruner_patience", 2)),
    )

    ref_sym0 = ctx.symbols[0]
    is_off0 = int(ctx.data_maps[ref_sym0].get(f"is_start_idx_{tf}", 0))
    ref_df0 = ctx.data_maps[ref_sym0][tf]
    prebuilt_cpcv_bundle = None
    if ref_df0 is not None and not ref_df0.empty:
        ref_len0 = len(ref_df0) - is_off0
        if ref_len0 >= 200:
            prebuilt_cpcv_bundle = build_cpcv_test_paths_with_fallback(
                ref_len0, embargo=int(EMBARGO_BARS.get(tf, 0))
            )

    cache_root = Path(ctx.signal_cache_dir) if ctx.signal_cache_dir else FUTURES_CACHE_DIR

    tf_space = ctx.locked_param_space if ctx.locked_param_space is not None else get_search_space_futures(tf)

    def _objective_with_logging(trial: optuna.Trial) -> float:
        try:
            return float(
                objective_futures(
                    trial,
                    data_maps=ctx.data_maps,
                    symbols=ctx.symbols,
                    tf_target=tf,
                    space=tf_space,
                    mode=ctx.mode,
                    project_root=ctx.project_root,
                    prebuilt_cpcv_bundle=prebuilt_cpcv_bundle,
                    signal_disk_cache_root=cache_root,
                )
            )
        except optuna.TrialPruned:
            raise
        except Exception:
            _logger.exception("[%s/%s] Trial %d failed.", target_str, tf, trial.number)
            raise

    last_progress_ts: float = 0.0
    last_progress_trial: int = -1

    def _progress_cb(_study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal last_progress_ts, last_progress_trial
        if ctx.progress_queue is None:
            return
        cur = trial.number + 1
        if cur == last_progress_trial:
            return
        if ctx.progress_min_interval > 0.0 and cur < ctx.n_trials:
            now = time.monotonic()
            if (now - last_progress_ts) < ctx.progress_min_interval:
                return
            last_progress_ts = now
        else:
            last_progress_ts = time.monotonic()
        last_progress_trial = cur
        kelly_pct = 0.0
        if trial.value is not None:
            try:
                kelly_pct = float(trial.user_attrs.get("kelly_score_pct", float(trial.value) * 100.0))
            except Exception:
                kelly_pct = 0.0
        ctx.progress_queue.put((progress_key, cur, ctx.n_trials, kelly_pct))

    study = optuna.create_study(
        study_name=tf_study_name,
        storage=storage,
        direction="maximize",
        sampler=qmc_sampler,
        pruner=pruner,
        load_if_exists=False,
    )
    _logger.info("[%s/%s/%s] QMC (Sobol) startup then TPE (CPCV)...", target_str, tf, ctx.mode)
    callbacks = [_progress_cb] if ctx.progress_queue is not None else []
    study.optimize(
        _objective_with_logging,
        n_trials=n_qmc,
        n_jobs=ctx.n_jobs,
        catch=(Exception,),
        callbacks=callbacks,
    )

    remaining = int(ctx.n_trials) - n_qmc
    if remaining <= 0:
        return (target_obj, tf), study

    study = optuna.load_study(
        study_name=tf_study_name,
        storage=storage,
        sampler=_futures_tpe_sampler(seed),
        pruner=pruner,
    )
    use_inner_pool = (
        ctx.parallel_tpe_workers > 1
        and remaining > 0
        and not ctx.disable_inner_tpe_pool
        and ctx.use_journal_storage
    )
    if use_inner_pool:
        assert db_path is not None
        n_workers = min(int(ctx.parallel_tpe_workers), int(remaining))
        splits = _split_tpe_trials(remaining, n_workers)
        data_maps_file = ""
        with tempfile.NamedTemporaryFile(
            suffix=".pkl", delete=False, dir=str(db_path.parent)
        ) as dm_tmp:
            data_maps_file = dm_tmp.name
            pickle.dump(ctx.data_maps, dm_tmp, protocol=pickle.HIGHEST_PROTOCOL)

        payloads: List[Dict[str, Any]] = []
        try:
            for i, nt in enumerate(splits):
                if nt <= 0:
                    continue
                payloads.append(
                    {
                        "journal_path": str(db_path),
                        "study_name": tf_study_name,
                        "tf": tf,
                        "target_str": target_str,
                        "symbols": list(ctx.symbols),
                        "mode": ctx.mode,
                        "seed": seed + i,
                        "n_trials": nt,
                        "n_trials_total": ctx.n_trials,
                        "prebuilt_cpcv_bundle": prebuilt_cpcv_bundle,
                        "data_maps_file": data_maps_file,
                        "progress_queue": ctx.progress_queue,
                        "project_root": ctx.project_root,
                        "signal_cache_dir": ctx.signal_cache_dir,
                        "progress_key": progress_key,
                        "locked_param_space": ctx.locked_param_space,
                    }
                )
            with concurrent.futures.ProcessPoolExecutor(max_workers=len(payloads)) as pool:
                futures = [pool.submit(_futures_tpe_worker_run, p) for p in payloads]
                for f in concurrent.futures.as_completed(futures):
                    f.result()
        finally:
            if data_maps_file:
                try:
                    Path(data_maps_file).unlink()
                except OSError:
                    pass
        study_final = optuna.load_study(
            study_name=tf_study_name,
            storage=_build_sqlite_storage(db_path),
        )
        return (target_obj, tf), study_final

    study.optimize(
        _objective_with_logging,
        n_trials=remaining,
        n_jobs=ctx.n_jobs,
        catch=(Exception,),
        callbacks=callbacks,
    )
    return (target_obj, tf), study

def main() -> None:
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-universe",
        action="store_true",
        help="Skip Phase A/B screening; use anchors + first 5 dynamic pool symbols.",
    )
    parser.add_argument("--trials", type=int, default=OPT_FUTURES_CONFIG["total_trials"])
    parser.add_argument("--jobs", type=int, default=int(OPT_FUTURES_CONFIG.get("n_jobs", 10)))
    parser.add_argument("--task-workers", type=int, default=int(OPT_FUTURES_CONFIG.get("task_workers", 1)))
    parser.add_argument("--tf", type=str, choices=["1h", "4h"], default=OPT_FUTURES_CONFIG.get("TARGET_TIMEFRAMES", ["4h"])[0])
    parser.add_argument("--reference-date", type=str, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    # [INFRA] Signal Cache & Memory Management (Alignment with Spot strategy)
    os.environ.setdefault("OPT_FUTURES_SIGNAL_CACHE_MAX_GB", "24")
    os.environ.setdefault("OPT_FUTURES_SIGNAL_MEM_CACHE_MAX", "96")
    os.environ.setdefault("OPT_FUTURES_SIGNAL_CACHE_CLEANUP_INTERVAL_SEC", "300")
    
    FETCH_START_DATE, START_DATE, IS_END_DATE, END_DATE = get_quarterly_window(args.reference_date)
    collector = DataCollector()

    broad_candidates: List[str] = []
    if not args.skip_universe:
        # Pass empty pool to trigger dynamic all-market discovery
        broad_candidates, _n_raw = screen_futures_universe(
            collector,
            [], 
            args.tf,
            FUTURES_SCREENER_CONFIG,
            FETCH_START_DATE,
            END_DATE,
            data_dir=FUTURES_DATA_DIR,
        )
        anchors_to_add = [s for s in FUTURES_ANCHOR_SYMBOLS if s not in broad_candidates]
        symbols = list(broad_candidates) + anchors_to_add # Initial symbols for data loading
    else:
        from config.opt_config import FUTURES_SYMBOLS
        symbols = list(dict.fromkeys(FUTURES_SYMBOLS))

    _logger.info(
        "Loading Futures data for %d symbols (Mode: %s)...",
        len(symbols),
        MODE_MULTI,
    )
    if not args.skip_universe:
        _logger.info(
            "Universe screening: %d candidates (ADV+ATR, sorted by ADV desc)",
            len(broad_candidates),
        )
    else:
        _logger.info("Fallback symbols: %s", symbols)

    data_maps: Dict[str, Dict[str, Any]] = {}
    oos_data_maps: Dict[str, Dict[str, Any]] = {}
    for sym in symbols:
        data_maps[sym] = {}
        oos_data_maps[sym] = {}
        collector.ensure_funding_data(sym, FETCH_START_DATE, END_DATE)
        for tf in [args.tf, "1d"]:
            full_df = collector.collect_and_save(sym, tf, FETCH_START_DATE, END_DATE)
            full_df = merge_funding_into_ohlcv(sym, full_df, FUTURES_DATA_DIR)
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
        oos_data_maps[sym][f"merge_idx_{args.tf}"] = compute_segment_merge_index(
            oos_data_maps[sym][args.tf], oos_data_maps[sym]["1d"]
        )

    locked_param_space: Optional[Dict[str, Any]] = None
    _logger.info("Combination screening (signal × regime × sizing)...")
    phase1_no_edge = False

    stage1_k = int(FUTURES_SCREENER_CONFIG.get("COMBO_SAMPLE_K", 12))
    stage1_syms = symbols[:stage1_k] if not args.skip_universe else symbols
    stage1_data_maps = {s: data_maps[s] for s in stage1_syms}
    _logger.info("Phase B: combination screening on N=%d broad candidates", len(stage1_syms))

    screening = run_combination_screening_futures(
        data_maps=stage1_data_maps,
        symbols=stage1_syms,
        tf=args.tf,
        project_root=project_root,
        signal_cache_dir=str(FUTURES_CACHE_DIR),
    )
    top_combos = screening.combos
    phase1_no_edge = screening.phase1_no_edge

    if top_combos and not top_combos[0].disqualified:
        # Ported from Spot: Explore top-5 combos; cap duplicate signal AND sizing types (union space noise).
        max_combos = 5
        max_per_dim = int(OPT_FUTURES_CONFIG.get("FUTURES_STAGE2_MAX_PER_SIGNAL_TYPE", 2))
        sig_ct: Counter[str] = Counter()
        siz_ct: Counter[str] = Counter()
        filtered_tops: List[CombinationScoreFutures] = []
        for combo in top_combos:
            if combo.disqualified:
                continue
            if (
                sig_ct[combo.signal] < max_per_dim
                and siz_ct[combo.sizing] < max_per_dim
            ):
                filtered_tops.append(combo)
                sig_ct[combo.signal] += 1
                siz_ct[combo.sizing] += 1
            if len(filtered_tops) >= max_combos:
                break
        valid_tops = filtered_tops if filtered_tops else top_combos[:3]
        locked_param_space = build_multi_combo_param_space_futures(valid_tops)
        
        if not args.skip_universe:
            # Phase C: Refinement mini-backtest with winning combo (uses #1 for symbol refinement)
            from src.domain.futures.opt_futures_utils.combination_screener_futures import _mid_params_from_space
            t0_space = build_combined_param_space_futures(valid_tops[0].signal, valid_tops[0].regime, valid_tops[0].sizing)
            winning_params = _mid_params_from_space(t0_space, args.tf)
            
            refined_symbols = screen_futures_symbol_refinement(
                broad_candidates,
                FUTURES_ANCHOR_SYMBOLS,
                FUTURES_SCREENER_CONFIG,
                data_maps=data_maps,
                winning_params=winning_params
            )
            
            import importlib
            import config.opt_config
            importlib.reload(config.opt_config)
            from config.opt_config import FUTURES_SYMBOLS
            
            # Re-filter data maps to only include refined symbols for Optuna
            symbols = [s for s in FUTURES_SYMBOLS if s in data_maps]
            data_maps = {s: data_maps[s] for s in symbols}
            oos_data_maps = {s: oos_data_maps[s] for s in symbols}
            _logger.info("Phase C: Selection finalized. Proceeding to Phase D (Optuna) with %d symbols.", len(symbols))
        _logger.info(
            "   Locked combos: SIGNAL choices=%s REGIME choices=%s SIZING choices=%s",
            list(dict.fromkeys(c.signal for c in valid_tops)),
            list(dict.fromkeys(c.regime for c in valid_tops)),
            list(dict.fromkeys(c.sizing for c in valid_tops)),
        )
    else:
        _logger.warning("Phase B returned no valid combo; Phase C uses full discovery space.")

    if len(symbols) < 2:
        _logger.error("Need at least 2 symbols for futures multi discovery. Aborting.")
        return

    # Spot parity: align full OHLCV timelines once, then rebuild IS slices from aligned full maps.
    ref_tz = oos_data_maps[symbols[0]][args.tf]["datetime"].dt.tz
    is_start_align = pd.to_datetime(START_DATE)
    is_end_align = pd.to_datetime(IS_END_DATE)
    if ref_tz is not None:
        is_start_align = is_start_align.tz_localize(ref_tz)
        is_end_align = is_end_align.tz_localize(ref_tz)
    _align_oos_dataframes_on_common_datetimes(oos_data_maps, symbols, args.tf, is_end_align)
    _rebuild_is_data_maps_from_aligned_oos(
        data_maps, oos_data_maps, symbols, args.tf, is_start_align, is_end_align
    )
    _logger.info(
        "Aligned full %s on common datetimes: IS=%d bars, OOS full=%d bars (%d symbols).",
        args.tf,
        len(data_maps[symbols[0]][args.tf]),
        len(oos_data_maps[symbols[0]][args.tf]),
        len(symbols),
    )

    if phase1_no_edge:
        _obj_floor = float(OPT_FUTURES_CONFIG.get("FUTURES_OBJECTIVE_FLOOR_WHEN_NO_EDGE", -2.0))
        for s_sym in symbols:
            if s_sym in data_maps:
                data_maps[s_sym]["_futures_opt_meta"] = {"objective_floor_strict": _obj_floor}

    tasks: List[Tuple[Any, str]] = [(tuple(symbols), args.tf)]
    plan = _resolve_futures_execution_plan(len(tasks), MODE_MULTI, args.jobs, args.task_workers)
    use_mp = plan.use_outer_process_pool
    enable_progress = not args.no_progress
    _logger.info(
        "Starting CPCV discovery (n_trials=%d, n_jobs=%d, tpe_workers=%d)...",
        args.trials,
        plan.jobs_per_task,
        plan.parallel_tpe_workers,
    )
    need_multiprocess_queue = use_mp or (plan.parallel_tpe_workers > 1)
    manager = Manager() if need_multiprocess_queue and enable_progress else None
    progress_queue = manager.Queue() if manager else (queue.Queue() if enable_progress else None)

    tf_bars: Dict[str, Any] = {}
    if enable_progress:
        for i, (target, tf) in enumerate(tasks):
            key = _task_progress_key(target, tf)
            tf_bars[key] = tqdm(total=args.trials, desc=f"[{key}] Waiting...", position=i, leave=True)

    _multi_short_label = (
        f"Multi({len(tasks[0][0])} syms)_{tasks[0][1]}" if tasks and isinstance(tasks[0][0], tuple) else None
    )

    progress_thread = None
    if enable_progress:
        def _progress_listener() -> None:
            while True:
                msg = progress_queue.get()
                if msg is None:
                    break
                k, cur, tot, kelly_pct = msg
                bar = tf_bars.get(k)
                if bar is None:
                    continue
                bar.n = cur
                desc = (
                    f"[{_multi_short_label}] Best Kelly: {kelly_pct:.2f}%"
                    if _multi_short_label
                    else f"[{k}] Best Kelly: {kelly_pct:.2f}%"
                )
                bar.set_description(desc)
                bar.refresh()

        progress_thread = threading.Thread(target=_progress_listener, daemon=True)
        progress_thread.start()

    best_results: Dict[Tuple[Any, str], Optional[optuna.Study]] = {}
    try:
        if use_mp:
            with concurrent.futures.ProcessPoolExecutor(max_workers=plan.outer_task_workers) as exec:
                futures = {
                    exec.submit(
                        _run_tf_optimization,
                        t,
                        _TfOptimizationContext(
                            clean_symbol=("_".join(t[0]) if isinstance(t[0], tuple) else t[0]).replace(
                                "/", ""
                            ),
                            seeds=OPT_FUTURES_CONFIG["seeds"],
                            n_trials=args.trials,
                            n_jobs=plan.jobs_per_task,
                            data_maps={s: data_maps[s] for s in (list(t[0]) if isinstance(t[0], tuple) else [t[0]])},
                            symbols=(list(t[0]) if isinstance(t[0], tuple) else [t[0]]),
                            project_root=project_root,
                            progress_queue=progress_queue,
                            mode=MODE_MULTI,
                            use_journal_storage=False,
                            parallel_tpe_workers=1,
                            disable_inner_tpe_pool=True,
                            fresh_journal=True,
                            signal_cache_dir=str(FUTURES_CACHE_DIR),
                            locked_param_space=locked_param_space,
                        ),
                    ): t
                    for t in tasks
                }
                for f in concurrent.futures.as_completed(futures):
                    t_res, study = f.result()
                    best_results[t_res] = study
        else:
            for t in tasks:
                t_res, study = _run_tf_optimization(
                    t,
                    _TfOptimizationContext(
                        clean_symbol=("_".join(t[0]) if isinstance(t[0], tuple) else t[0]).replace("/", ""),
                        seeds=OPT_FUTURES_CONFIG["seeds"],
                        n_trials=args.trials,
                        n_jobs=plan.jobs_per_task,
                        data_maps=data_maps,
                        symbols=(list(t[0]) if isinstance(t[0], tuple) else [t[0]]),
                        project_root=project_root,
                        progress_queue=progress_queue,
                        mode=MODE_MULTI,
                        use_journal_storage=True,
                        parallel_tpe_workers=plan.parallel_tpe_workers,
                        disable_inner_tpe_pool=plan.use_outer_process_pool,
                        fresh_journal=True,
                        signal_cache_dir=str(FUTURES_CACHE_DIR),
                        locked_param_space=locked_param_space,
                    ),
                )
                best_results[t_res] = study
    finally:
        if progress_queue is not None:
            progress_queue.put(None)
        if progress_thread is not None:
            progress_thread.join(timeout=2.0)
        if manager: manager.shutdown()

    pending_json_writes: List[Tuple[str, Dict[str, Any], float, bool]] = []

    # Threshold Adjustment for Futures
    OOS_CAGR_TARGET = float(OPT_FUTURES_CONFIG.get("FUTURES_MIN_CAGR_PCT", 30.0))
    OOS_MDD_LIMIT = float(OPT_FUTURES_CONFIG.get("FUTURES_MAX_MDD", 25.0))
    OOS_CALMAR_TARGET = 1.2
    OOS_PF_TARGET = 1.35
    ALPHA_DECAY_FLOOR = -50.0
    MIN_TRADES_TOTAL = max(40, len(symbols) * 8)
    TW_TARGET = 1.0 - (OOS_MDD_LIMIT / 100.0)

    for (target, tf_eval), study in best_results.items():
        if study is None:
            continue

        completed = [t for t in study.get_trials(deepcopy=True) if t.state == TrialState.COMPLETE and t.value is not None]
        if not completed:
            continue
        ranked = sorted(completed, key=lambda tr: float(tr.value), reverse=True)[:50]
        viable = [
            tr
            for tr in ranked
            if float(tr.user_attrs.get("min_path_terminal_wealth_ratio", 0.0)) >= 1.0
        ]
        shortlist_pool = viable if viable else ranked
        best_trial = _select_best_trial_from_shortlist(shortlist_pool)

        params = best_trial.params.copy()
        params["TIMEFRAME"] = tf_eval
        _lev_def = int(OPT_FUTURES_CONFIG.get("FUTURES_DISCOVERY_LEVERAGE", 8))
        params["LEVERAGE"] = int(os.getenv("FUTURES_DISCOVERY_LEVERAGE", str(_lev_def)))
        params["USE_COMPOUNDING"] = True

        target_symbols = sorted(list(target) if isinstance(target, tuple) else [target])

        # Tier 1 Veto: Portfolio Discovery Rigor
        ua = best_trial.user_attrs
        psr_v = float(ua.get("gate1_psr", 0.0))
        dsr_v = float(ua.get("gate1_dsr", 0.0))
        gmgr_v = float(ua.get("gate1_p10_gmgr", 0.0))
        veto = run_portfolio_discovery_veto(
            psr=psr_v,
            dsr=dsr_v,
            p10_gmgr=gmgr_v,
            psr_min=0.5,
            dsr_min=0.25,
        )

        # 1. IS Margin-Shared Portfolio Run (Alpha Decay Rigor)
        is_holdout_maps: Dict[str, Dict[str, Any]] = {}
        for s_eval in target_symbols:
            is_holdout_maps[s_eval] = dict(data_maps[s_eval])
            is_holdout_maps[s_eval][f"oos_start_idx_{tf_eval}"] = data_maps[s_eval][f"is_start_idx_{tf_eval}"]

        port_is = run_oos_margin_shared_portfolio(
            target_symbols,
            tf_eval,
            params,
            is_holdout_maps,
            cache_root=FUTURES_CACHE_DIR,
            return_signal_dfs=False,
        )
        is_portfolio_cagr: float = float(port_is.get("cagr_pct", -100.0))
        is_cagr_floor = float(OPT_FUTURES_CONFIG.get("FUTURES_IS_CAGR_FLOOR", 0.0))
        is_cagr_ok = is_portfolio_cagr >= is_cagr_floor
        if not is_cagr_ok:
            _logger.warning(
                "IS portfolio CAGR=%.2f%% below floor %.2f%% (CPCV vs full IS mismatch risk).",
                is_portfolio_cagr,
                is_cagr_floor,
            )

        # 2. OOS Margin-Shared Portfolio Run (Signal Cache & Integrity)
        oos_port = run_oos_margin_shared_portfolio(
            target_symbols,
            tf_eval,
            params,
            oos_data_maps,
            cache_root=FUTURES_CACHE_DIR,
            return_signal_dfs=True,
        )
        post_full_signal_dfs: Dict[str, pd.DataFrame] = oos_port.get("full_signal_dfs", {})

        passed_count = 0
        symbol_gate_rows: List[FuturesSymbolGateRow] = []
        total_funding_oos = 0.0
        total_gross_oos = 0.0
        is_cagrs: List[float] = []

        from src.domain.spot.opt_spot_utils.evaluator import _segment_with_context

        for s_eval in target_symbols:
            # IS Evaluation (Consistency check)
            s_is, r_is, m_is, t_is, wr_is, pf_is, lc_is, sc_is, _, _, _ = evaluate_symbol_fold(
                UltimateStrategy(name=f"IS_{s_eval}", params=params),
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
            
            # OOS Evaluation (Using cached portfolio signals)
            pre_oos_sig_full = post_full_signal_dfs.get(s_eval)
            oos_start_idx = int(oos_data_maps[s_eval][f"oos_start_idx_{tf_eval}"])
            oos_local_end = len(oos_data_maps[s_eval][tf_eval])
            
            if pre_oos_sig_full is not None:
                pre_oos_sig, exec_start_oos = _segment_with_context(
                    pre_oos_sig_full, oos_start_idx, oos_local_end
                )
            else:
                pre_oos_sig, exec_start_oos = None, 0

            s_oos, r_oos, m_oos, t_oos, wr_oos, pf_oos, lc_oos, sc_oos, _, fp_oos, gr_oos = evaluate_symbol_fold(
                UltimateStrategy(name=f"OOS_{s_eval}", params=params),
                params,
                s_eval,
                tf_eval,
                oos_data_maps[s_eval][tf_eval],
                oos_data_maps[s_eval]["1d"],
                oos_data_maps[s_eval][f"merge_idx_{tf_eval}"],
                None,
                oos_start_idx,
                oos_local_end,
                precomputed_signal_df=pre_oos_sig,
                execution_start_idx=exec_start_oos,
            )
            
            total_funding_oos += float(fp_oos)
            total_gross_oos += float(abs(gr_oos))
            is_cagrs.append(float(s_is))
            tot_ls = int(lc_oos) + int(sc_oos)
            ls_ratio_oos = float(min(int(lc_oos), int(sc_oos)) / max(tot_ls, 1))
            go_nogo = run_go_nogo_check(
                [],
                0.0,
                [s_oos],
                m_oos,
                pf_oos,
                int(lc_oos),
                int(sc_oos),
                tf_eval,
                long_short_ratio_oos=ls_ratio_oos,
            )
            if go_nogo.passed:
                passed_count += 1

            symbol_gate_rows.append(
                FuturesSymbolGateRow(
                    symbol=s_eval,
                    net_cagr_pct=float(s_oos),
                    max_mdd_pct=float(m_oos),
                    win_rate_pct=float(wr_oos),
                    trade_count=int(t_oos),
                )
            )

        best_score_final = float(best_trial.value) if best_trial.value is not None else -100.0

        # Multi-Window OOS Logic (Ported from Spot)
        mw_enabled = True
        mw_gate = None
        mw_summary = ""
        if mw_enabled:
            mw_subs = int(OPT_FUTURES_CONFIG.get("FUTURES_MULTI_WINDOW_OOS_SUBS", 3))
            mw_res = run_multi_window_oos_holdout(
                params, target_symbols, tf_eval, oos_data_maps, 
                n_sub_windows=mw_subs, cache_root=FUTURES_CACHE_DIR, full_holdout_result=oos_port
            )
            mw_min_pos = int(OPT_FUTURES_CONFIG.get("FUTURES_MULTI_WINDOW_MIN_POSITIVE", 3))
            mw_gate = run_multi_window_oos_gate(
                window_results=mw_res["windows"],
                min_positive_windows=mw_min_pos,
                min_median_cagr_pct=OOS_CAGR_TARGET,
                max_worst_mdd_pct=OOS_MDD_LIMIT
            )
            mw_summary = mw_gate.summary

        # PBO Evaluation (Tier 1 advisory/hard)
        pbo_val: float = float("nan")
        rho_val: float = float("nan")
        pbo_n_paths = 0
        oos_log_tw = ua.get("cpcv_path_oos_log_tw")
        if isinstance(oos_log_tw, list) and oos_log_tw:
            ref_pbo_sym = target_symbols[0]
            is_off_pbo = int(data_maps[ref_pbo_sym].get(f"is_start_idx_{tf_eval}", 0))
            ref_len_pbo = len(data_maps[ref_pbo_sym][tf_eval]) - is_off_pbo
            emb_pbo = int(EMBARGO_BARS.get(tf_eval, 0))
            from src.domain.futures.opt_futures_utils.cv_utils import list_cpcv_block_ranges
            prebuilt_pbo = build_cpcv_test_paths_with_fallback(ref_len_pbo, embargo=emb_pbo)
            cpcv_paths_pbo, nb_pbo, _k_pbo = prebuilt_pbo
            pbo_n_paths = len(cpcv_paths_pbo)
            all_blocks_pbo = list_cpcv_block_ranges(ref_len_pbo, nb_pbo, emb_pbo)
            if len(oos_log_tw) == pbo_n_paths and all_blocks_pbo:
                from src.domain.futures.opt_futures_utils.evaluator import run_cpcv_complement_evaluation
                pbo_val, rho_val = run_cpcv_complement_evaluation(
                    params,
                    target_symbols,
                    tf_eval,
                    data_maps,
                    cpcv_paths_pbo,
                    all_blocks_pbo,
                    oos_path_scores=oos_log_tw,
                    signal_disk_cache_root=FUTURES_CACHE_DIR,
                    project_root=project_root,
                    concurrency_penalty_scale=1.0,
                )

        # Regime Diagnostic Logic (Tier 4 advisory)
        regime_block = ""
        full_sigs_diag = oos_port.get("full_signal_dfs", {})
        if full_sigs_diag:
            oos_start_reg = int(oos_data_maps[target_symbols[0]].get(f"oos_start_idx_{tf_eval}", 0))
            regime_metrics = compute_regime_conditional_oos_metrics(
                full_sigs_diag, np.asarray(oos_port["equity_curve"]), oos_start_reg, target_symbols
            )
            regime_block = format_regime_oos_diagnostic_block(regime_metrics, stress_mdd_warn_pct=30.0)

        funding_drag_pct_oos = (
            (total_funding_oos / max(total_gross_oos, 1e-9)) * 100.0 if total_gross_oos > 0 else 0.0
        )
        # Ported from Spot: Accurate alpha decay using margin-shared portfolio CAGR
        oos_portfolio_cagr = float(oos_port["cagr_pct"])
        alpha_decay_pct = (
            (
                (oos_portfolio_cagr - is_portfolio_cagr)
                / max(abs(is_portfolio_cagr), abs(oos_portfolio_cagr), 1e-6)
            )
            * 100.0
            if (
                abs(is_portfolio_cagr) > 1e-12 or abs(oos_portfolio_cagr) > 1e-12
            )
            else 0.0
        )

        # Ported from Spot: Synchronization and Friction risk diagnostics
        portfolio_mdd_pct = float(oos_port["mdd_pct"])
        if symbol_gate_rows:
            max_sym_mdd = max(float(r.max_mdd_pct) for r in symbol_gate_rows)
            if max_sym_mdd > 0.0 and abs(portfolio_mdd_pct) > 3.0 * max_sym_mdd:
                 _logger.warning(
                    "  ⚠ [SYNC RISK] Portfolio MDD(%.1f%%) > 3x max Symbol MDD(%.1f%%)",
                    abs(portfolio_mdd_pct), max_sym_mdd
                )

        for row_fr in symbol_gate_rows:
            if 0.0 < float(row_fr.net_cagr_pct) < 1.0 and float(row_fr.win_rate_pct) > 55.0:
                 _logger.warning(
                    "  ⚠ [FRICTION RISK] %s: WinRate %.1f%% vs CAGR %.1f%% (Noise dominated)",
                    row_fr.symbol, row_fr.win_rate_pct, row_fr.net_cagr_pct
                )

        oos_cagrs_pos = sorted([float(r.net_cagr_pct) for r in symbol_gate_rows if r.net_cagr_pct > 0.0], reverse=True)
        if oos_cagrs_pos:
            pos_sum = float(sum(oos_cagrs_pos)) + 1e-9
            max_share = oos_cagrs_pos[0] / pos_sum
            top2_share = (
                sum(oos_cagrs_pos[:2]) / pos_sum if len(oos_cagrs_pos) >= 2 else max_share
            )
            if max_share >= 0.4:
                loso_warning = f"경고 (단일 심볼 OOS CAGR 비중 {max_share:.0%} >= 40%)"
            elif top2_share >= 0.65:
                loso_warning = f"주의 (상위 2개 심볼 집중도 {top2_share:.0%} >= 65%)"
            else:
                loso_warning = f"안전 (최대 {max_share:.0%} / 상위2 {top2_share:.0%})"
        else:
            loso_warning = "N/A (OOS CAGR 비중 산출 불가)"

        # Hard Gates Check
        pbo_max_cfg = float(OPT_FUTURES_CONFIG.get("FUTURES_PBO_MAX", 0.45))
        pbo_hard = bool(OPT_FUTURES_CONFIG.get("FUTURES_PBO_GATE_HARD", False))
        
        trade_floor = run_holdout_portfolio_trade_floor(
            portfolio_long_trades=int(oos_port["total_trades"]),
            min_portfolio_trades=MIN_TRADES_TOTAL
        )

        pbo_gate_ok = bool(math.isfinite(pbo_val) and pbo_val <= pbo_max_cfg)
        core_gate_checks = [
            veto.passed,
            is_cagr_ok,
            float(ua.get("gate1_sqn", 0.0)) >= 2.0,
            float(ua.get("gate1_path_sortino", 0.0)) >= 1.5,
            float(ua.get("gate1_tail_ratio", 0.0)) >= 1.5,
            trade_floor.passed,
            abs(float(oos_port["mdd_pct"])) <= OOS_MDD_LIMIT,
            float(oos_port["cvar_pct"]) <= 12.0,
            float(oos_port["hw_recovery_days"]) <= 180.0,
            float(oos_port["calmar_ratio"]) >= OOS_CALMAR_TARGET,
            funding_drag_pct_oos <= 15.0,
            float(oos_port["cagr_pct"]) >= OOS_CAGR_TARGET,
            float(oos_port["profit_factor"]) >= OOS_PF_TARGET,
            alpha_decay_pct >= ALPHA_DECAY_FLOOR,
            float(oos_port["terminal_wealth_ratio"]) >= TW_TARGET,
            (mw_gate.passed if mw_gate else True),
        ]
        core_passed_count = int(sum(1 for c in core_gate_checks if c))
        if pbo_hard:
            hard_total = len(core_gate_checks) + 1
            hard_passed = core_passed_count + (1 if pbo_gate_ok else 0)
            is_all_passed = all(core_gate_checks) and pbo_gate_ok
        else:
            hard_total = len(core_gate_checks)
            hard_passed = core_passed_count
            is_all_passed = all(core_gate_checks)

        report = run_futures_deployment_report(
            FuturesDeploymentReportInput(
                gate1_sqn=float(ua.get("gate1_sqn", 0.0)),
                gate1_path_sortino=float(ua.get("gate1_path_sortino", 0.0)),
                gate1_tail_ratio=float(ua.get("gate1_tail_ratio", 0.0)),
                gate1_p10_gmgr=float(ua.get("gate1_p10_gmgr", 0.0)),
                gate1_psr=float(ua.get("gate1_psr", 0.0)),
                gate1_dsr=float(ua.get("gate1_dsr", 0.0)),
                cpcv_mean_path_return_pct=float(ua.get("cpcv_mean_path_return_pct", 0.0)),
                cpcv_worst_segment_mdd_pct=float(ua.get("cpcv_worst_segment_mdd_pct", 0.0)),
                sqn_target=2.0,
                path_sortino_target=1.5,
                tail_ratio_target=1.5,
                psr_target=0.50,
                dsr_target=0.25,
                moic=float(oos_port["moic"]),
                initial_capital_usdt=float(FUTURES_INITIAL_BALANCE),
                oos_net_cagr_pct=float(oos_port["cagr_pct"]),
                oos_mdd_pct=float(oos_port["mdd_pct"]),
                hw_recovery_days=float(oos_port["hw_recovery_days"]),
                alpha_decay_pct=float(alpha_decay_pct),
                oos_cagr_target_pct=OOS_CAGR_TARGET,
                oos_mdd_limit_pct=OOS_MDD_LIMIT,
                hw_recovery_max_days=180.0,
                alpha_decay_floor_pct=ALPHA_DECAY_FLOOR,
                oos_cvar_pct=float(oos_port["cvar_pct"]),
                cvar_limit_pct=12.0,
                funding_drag_pct=float(funding_drag_pct_oos),
                funding_drag_limit_pct=15.0,
                terminal_wealth_ratio=float(oos_port["terminal_wealth_ratio"]),
                tw_target=TW_TARGET,
                oos_total_trades=int(oos_port["total_trades"]),
                oos_pf=float(oos_port["profit_factor"]),
                pf_target=OOS_PF_TARGET,
                oos_calmar=float(oos_port["calmar_ratio"]),
                calmar_target=OOS_CALMAR_TARGET,
                oos_win_rate_pct=float(oos_port["win_rate_pct"]),
                oos_long_short_minority_pct=float(oos_port["oos_long_short_minority_pct"]),
                symbol_rows=symbol_gate_rows,
                loso_warning=loso_warning,
                hard_passed=hard_passed,
                hard_total=hard_total,
                final_decision_go=is_all_passed,
                pbo=float(pbo_val),
                spearman_rho=float(rho_val),
                pbo_n_paths=int(pbo_n_paths),
                pbo_gate_passed=bool(pbo_val <= pbo_max_cfg),
                pbo_hard_gate=pbo_hard,
                multi_window_passed=mw_gate.passed if mw_gate else True,
                multi_window_summary=mw_summary,
                regime_diagnostic_block=regime_block,
                oos_long_trades=int(oos_port["long_trades"]),
                oos_short_trades=int(oos_port["short_trades"]),
                funding_cost_total_usdt=float(total_funding_oos),
                gross_pnl_abs_usdt=float(total_gross_oos),
            )
        )
        _logger.info("\n%s", report)

        growth_save = float(ua.get("growth_score", 0.0))
        should_save = bool(growth_save > 0.0 and is_all_passed)
        json_filename = f"{BEST_PARAMS_FUTURES_JSON_STEM}.json"
        pending_json_writes.append((json_filename, params, best_score_final, should_save))

    if pending_json_writes:
        import json
        from src.core.utils.secure_config import encrypt_config, get_strategy_secret
        results_dir_p = Path(project_root) / "results"
        results_dir_p.mkdir(parents=True, exist_ok=True)
        secret = get_strategy_secret()

        for json_filename, params, best_score_final, should_save in pending_json_writes:
            json_path = results_dir_p / json_filename
            enc_path = json_path.with_suffix(".enc")
            
            if should_save:
                json_path.write_text(json.dumps(params, indent=4), encoding="utf-8")
                _logger.info("Saved config: %s", json_path.resolve())
                if secret:
                    enc_path.write_bytes(encrypt_config(params, secret))
                    _logger.info("Saved encrypted config: %s", enc_path.resolve())
                else:
                    _logger.warning("STRATEGY_SECRET_KEY not set; skipping encrypted save.")
            else:
                _logger.info(
                    "JSON save skipped: criteria not met (growth_score / gates). objective=%.4f",
                    best_score_final,
                )

if __name__ == "__main__":
    main()
