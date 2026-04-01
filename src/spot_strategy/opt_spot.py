from __future__ import annotations

import argparse
import logging
import os
import pickle
import queue
import sys
import tempfile
import threading
import warnings
import optuna
from optuna.pruners import MedianPruner, PatientPruner
from optuna.samplers import QMCSampler, TPESampler
from optuna.trial import TrialState
from optuna.storages import InMemoryStorage
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import concurrent.futures
from multiprocessing import Manager
from tqdm import tqdm

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

project_root: str = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.spot_strategy.data_collector_spot import DataCollectorSpot
from src.spot_strategy.strategies_spot import UltimateSpotStrategy
from config.settings import SPOT_INITIAL_BALANCE
from config.opt_config import OPT_SPOT_CONFIG, SPOT_SYMBOLS, get_search_space_spot, get_quarterly_window
from src.optimization.opt_utils import compute_segment_merge_index
from src.spot_strategy.opt_spot_utils.cv_utils import CPCVPath, build_cpcv_test_paths_with_fallback
from src.spot_strategy.opt_spot_utils.evaluator import (
    EMBARGO_BARS,
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
_logger: logging.Logger = logging.getLogger("opt_spot")

def _ensure_fresh_sqlite(db_path: Path) -> None:
    """RAM Disk 내 기존 DB 및 관련 저널 파일들을 안전하게 제거함."""
    for suffix in ["", "-wal", "-shm"]:
        p = Path(str(db_path) + suffix)
        if p.exists():
            try: p.unlink()
            except OSError: pass

def _build_sqlite_storage(db_path: Path) -> optuna.storages.RDBStorage:
    """WSL2 환경 최적화: RAM Disk 경로에 SQLite WAL 모드를 활성화하여 RDBStorage 생성함."""
    import sqlite3
    url: str = f"sqlite:///{db_path.absolute()}"
    storage: optuna.storages.RDBStorage = optuna.storages.RDBStorage(
        url=url,
        engine_kwargs={"connect_args": {"timeout": 60}}
    )
    # WAL 모드 강제 활성화 (병렬 성능 극대화)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    return storage

def _get_effective_search_space_spot(
    tf: str,
    narrowed_override: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    base = get_search_space_spot(tf)
    if not narrowed_override:
        return dict(base)
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in base.items():
        out[k] = dict(v) if isinstance(v, dict) else v
    for k, spec in narrowed_override.items():
        out[k] = dict(spec) if isinstance(spec, dict) else spec
    return out


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
    parallel_tpe_workers: int = 1
    disable_inner_tpe_pool: bool = False
    fresh_journal: bool = True
    signal_cache_dir: str = ""
    narrowed_space: Optional[Dict[str, Dict[str, Any]]] = None

@dataclass(frozen=True)
class _SpotExecutionPlan:
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


def _resolve_spot_execution_plan(
    task_count: int,
    mode: str,
    requested_jobs: int,
    requested_task_workers: int,
) -> _SpotExecutionPlan:
    logical_cpus = max(1, os.cpu_count() or 1)
    if mode == "multi" and task_count == 1:
        tw = requested_task_workers if requested_task_workers > 0 else requested_jobs
        tw = max(1, min(int(tw), logical_cpus))
        return _SpotExecutionPlan(
            outer_task_workers=1,
            jobs_per_task=1,
            parallel_tpe_workers=tw,
            use_outer_process_pool=False,
            logical_cpus=logical_cpus,
            task_count=task_count,
        )
    jobs = max(1, min(int(requested_jobs), logical_cpus))
    if requested_task_workers <= 0:
        cpu_budget = max(1, 4 if logical_cpus > 2 else 1)
        outer_tw = min(task_count, max(1, cpu_budget // jobs))
    else:
        outer_tw = min(task_count, int(requested_task_workers))
    outer_tw = max(1, outer_tw)
    use_outer = task_count > 1 and outer_tw > 1
    return _SpotExecutionPlan(
        outer_task_workers=outer_tw,
        jobs_per_task=jobs,
        parallel_tpe_workers=1,
        use_outer_process_pool=use_outer,
        logical_cpus=logical_cpus,
        task_count=task_count,
    )


def _task_progress_key(target_obj: Any, tf: str) -> str:
    """Short tqdm key: portfolio -> SPOT_{tf}; single symbol -> SPOT_{tf}_ETH."""
    if isinstance(target_obj, (list, tuple)) and len(target_obj) > 1:
        return f"SPOT_{tf}"
    sym = target_obj[0] if isinstance(target_obj, (list, tuple)) else target_obj
    short = str(sym).replace("KRW-", "").replace("-", "")
    return f"SPOT_{tf}_{short}"


def _build_precomputed_segment(
    full_signal_df: Any,
    exec_start_idx: int,
    exec_end_idx: int,
) -> Tuple[pd.DataFrame | None, int]:
    if not isinstance(full_signal_df, pd.DataFrame) or full_signal_df.empty:
        return None, 0
    n = len(full_signal_df)
    start = max(0, min(int(exec_start_idx), n))
    end = max(start, min(int(exec_end_idx), n))
    if end - start < 2:
        return None, 0
    slice_start = max(0, start - 1)
    segment = full_signal_df.iloc[slice_start:end].copy()
    execution_start_idx = start - slice_start
    if execution_start_idx == 0 and len(segment) > 1:
        execution_start_idx = 1
    return segment, execution_start_idx


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


def _spot_tpe_worker_run(payload: Dict[str, Any]) -> None:
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
    prebuilt_cpcv_bundle = payload["prebuilt_cpcv_bundle"]
    data_maps_file = str(payload.get("data_maps_file", ""))
    if data_maps_file and Path(data_maps_file).is_file():
        with Path(data_maps_file).open("rb") as f:
            data_maps: Dict[str, Dict[str, Any]] = pickle.load(f)
    else:
        data_maps = payload.get("data_maps", {})
    progress_queue = payload["progress_queue"]
    project_root = str(payload["project_root"])
    signal_cache_dir = str(payload.get("signal_cache_dir", ""))
    if signal_cache_dir:
        os.environ["OPT_SPOT_SIGNAL_CACHE_DIR"] = signal_cache_dir
    progress_key = str(payload["progress_key"])
    raw_narrow = payload.get("narrowed_space")
    narrowed_w: Optional[Dict[str, Dict[str, Any]]] = raw_narrow if isinstance(raw_narrow, dict) else None
    tf_space = _get_effective_search_space_spot(tf, narrowed_w)

    sampler = TPESampler(
        seed=seed,
        n_startup_trials=0,
        multivariate=True,
        group=True,
        constant_liar=True,
    )
    base_pruner = MedianPruner(
        n_startup_trials=int(OPT_SPOT_CONFIG.get("tpe_pruner_n_startup_trials", 10)),
        n_warmup_steps=int(OPT_SPOT_CONFIG.get("tpe_pruner_n_warmup_steps", 8)),
    )
    pruner = PatientPruner(
        base_pruner,
        patience=int(OPT_SPOT_CONFIG.get("tpe_pruner_patience", 2)),
    )

    storage: optuna.storages.RDBStorage = _build_sqlite_storage(db_path)

    study = optuna.load_study(
        study_name=tf_study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
    )

    try:
        best_so_far: float = float(study.best_value)
    except ValueError:
        best_so_far = float("-inf")

    def _progress_cb(_study_inner: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal best_so_far
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
                0.0 if best_so_far == float("-inf") else best_so_far,
            )
        )

    cache_root_opt = Path(signal_cache_dir) if signal_cache_dir else None

    def _objective_with_logging(trial: optuna.Trial) -> float:
        try:
            return objective_spot(
                trial,
                data_maps=data_maps,
                symbols=symbols,
                tf_target=tf,
                space=tf_space,
                mode=mode,
                project_root=project_root,
                prebuilt_cpcv_bundle=prebuilt_cpcv_bundle,
                signal_disk_cache_root=cache_root_opt,
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


def _run_tf_optimization(task: Tuple[Any, str], ctx: _TfOptimizationContext) -> Tuple[Tuple[Any, str], optuna.Study]:
    target_obj, tf = task
    target_str = "_".join(target_obj) if isinstance(target_obj, (list, tuple)) else target_obj
    progress_key = _task_progress_key(target_obj, tf)
    tf_study_name: str = f"OptSpot_{target_str.replace('/', '').replace('-', '')}_{tf}_{ctx.mode}"

    db_path: Path | None = None
    if ctx.use_journal_storage:
        # WSL2 성능 극대화: RAM Disk 경로에 SQLite DB 생성
        db_path = Path("/dev/shm") / f"optuna_spot_{target_str.replace('/', '').replace('-', '')}_{tf}.db"
        if ctx.fresh_journal:
            _ensure_fresh_sqlite(db_path)

    n_qmc = min(int(OPT_SPOT_CONFIG.get("tpe_n_startup_trials", 96)), int(ctx.n_trials))
    seed = int(ctx.seeds[0])

    storage: optuna.storages.RDBStorage | InMemoryStorage
    if ctx.use_journal_storage:
        assert db_path is not None
        storage = _build_sqlite_storage(db_path)
    else:
        storage = InMemoryStorage()

    qmc_sampler = QMCSampler(qmc_type="sobol", scramble=True, seed=seed, warn_independent_sampling=False)
    base_pruner = MedianPruner(
        n_startup_trials=int(OPT_SPOT_CONFIG.get("tpe_pruner_n_startup_trials", 10)),
        n_warmup_steps=int(OPT_SPOT_CONFIG.get("tpe_pruner_n_warmup_steps", 8)),
    )
    pruner = PatientPruner(
        base_pruner,
        patience=int(OPT_SPOT_CONFIG.get("tpe_pruner_patience", 2)),
    )

    study = optuna.create_study(
        study_name=tf_study_name,
        storage=storage,
        direction="maximize",
        sampler=qmc_sampler,
        pruner=pruner,
        load_if_exists=False,
    )

    best_so_far: float = float("-inf")

    def _progress_cb(_study_inner: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal best_so_far
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

    tf_space = _get_effective_search_space_spot(tf, ctx.narrowed_space)

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

    cache_root_opt = Path(ctx.signal_cache_dir) if ctx.signal_cache_dir else None

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
                prebuilt_cpcv_bundle=prebuilt_cpcv_bundle,
                signal_disk_cache_root=cache_root_opt,
            )
        except optuna.TrialPruned:
            raise
        except Exception:
            _logger.exception("[%s/%s] Trial %d failed.", target_str, tf, trial.number)
            raise

    _logger.info("[%s/%s] Spot QMC startup (Sobol) then TPE (CPCV discovery)...", target_str, tf)
    study.optimize(_objective_with_logging, n_trials=n_qmc, n_jobs=ctx.n_jobs, catch=(Exception,), callbacks=[_progress_cb])

    remaining = int(ctx.n_trials) - n_qmc
    if remaining <= 0:
        return (target_obj, tf), study

    tpe_sampler = TPESampler(
        seed=seed,
        n_startup_trials=0,
        multivariate=True,
        group=True,
        constant_liar=True,
    )

    if ctx.use_journal_storage:
        assert db_path is not None
        storage_tpe = _build_sqlite_storage(db_path)
    else:
        storage_tpe = storage

    study = optuna.load_study(
        study_name=tf_study_name,
        storage=storage_tpe,
        sampler=tpe_sampler,
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
        # DB 파일이 있는 RAM Disk 경로에 데이터 맵 임시 파일 생성
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
                        "narrowed_space": ctx.narrowed_space,
                    }
                )
            with concurrent.futures.ProcessPoolExecutor(max_workers=len(payloads)) as pool:
                futures = [pool.submit(_spot_tpe_worker_run, p) for p in payloads]
                for f in concurrent.futures.as_completed(futures):
                    f.result()
        finally:
            if data_maps_file:
                try:
                    Path(data_maps_file).unlink()
                except OSError:
                    pass
        storage_final = _build_sqlite_storage(db_path)
        study = optuna.load_study(study_name=tf_study_name, storage=storage_final)
        return (target_obj, tf), study

    study.optimize(
        _objective_with_logging,
        n_trials=remaining,
        n_jobs=ctx.n_jobs,
        catch=(Exception,),
        callbacks=[_progress_cb],
    )
    return (target_obj, tf), study


def _merge_narrowed_space_dicts(
    base: Optional[Dict[str, Dict[str, Any]]],
    override: Optional[Dict[str, Dict[str, Any]]],
) -> Optional[Dict[str, Dict[str, Any]]]:
    if not base and not override:
        return None
    if not base:
        return {k: dict(v) if isinstance(v, dict) else v for k, v in (override or {}).items()}
    if not override:
        return {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    out: Dict[str, Dict[str, Any]] = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    for k, v in override.items():
        out[k] = dict(v) if isinstance(v, dict) else v
    return out


def _build_narrowed_space_for_signal_types(
    winning_types: tuple[str, ...],
    tf: str,
) -> Dict[str, Dict[str, Any]]:
    base = get_search_space_spot(tf)
    out: Dict[str, Dict[str, Any]] = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    st_spec = out.get("SIGNAL_TYPE")
    if isinstance(st_spec, dict) and st_spec.get("type") == "categorical":
        out["SIGNAL_TYPE"] = {**st_spec, "choices": tuple(winning_types)}
    return out


def _run_stage1_structure_discovery(
    *,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf: str,
    trials_per_signal: int,
    top_k: int,
    min_p10_gmgr: float,
    project_root: str,
    signal_cache_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """Per-SIGNAL_TYPE in-memory Optuna studies; narrow categorical to top performers."""
    all_types = tuple(get_search_space_spot(tf)["SIGNAL_TYPE"]["choices"])
    ref_sym = symbols[0]
    is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
    ref_df = data_maps[ref_sym][tf]
    prebuilt: Optional[Tuple[List[CPCVPath], int, int]] = None
    if ref_df is not None and not ref_df.empty:
        ref_len = len(ref_df) - is_off
        if ref_len >= 200:
            prebuilt = build_cpcv_test_paths_with_fallback(ref_len, embargo=int(EMBARGO_BARS.get(tf, 0)))
    cache_root = Path(signal_cache_dir) if signal_cache_dir else None
    results: List[tuple[str, float]] = []
    n_tri = max(5, int(trials_per_signal))
    for sig_type in all_types:
        stage_space = _build_narrowed_space_for_signal_types((sig_type,), tf)
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(
                n_startup_trials=min(20, max(5, n_tri // 4)),
                seed=42,
            ),
            storage=InMemoryStorage(),
        )

        def _obj(trial: optuna.Trial) -> float:
            return objective_spot(
                trial,
                data_maps,
                symbols,
                tf,
                space=stage_space,
                mode="multi",
                project_root=project_root,
                prebuilt_cpcv_bundle=prebuilt,
                signal_disk_cache_root=cache_root,
            )

        study.optimize(_obj, n_trials=n_tri, n_jobs=1, show_progress_bar=False)
        completed = [
            t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None
        ]
        if not completed:
            results.append((sig_type, -1e9))
            _logger.warning("Stage1 [%s]: no complete trials.", sig_type)
            continue
        completed.sort(key=lambda tr: float(tr.value or -1e9), reverse=True)
        ktop = max(1, n_tri // 10)
        top_trials = completed[:ktop]
        mean_obj = float(np.mean([float(t.value) for t in top_trials]))
        results.append((sig_type, mean_obj))
        _logger.info("Stage1 [%s]: top-%d mean objective=%.6f", sig_type, ktop, mean_obj)
    results.sort(key=lambda x: x[1], reverse=True)
    winning = [s for s, m in results[: max(1, top_k)] if m > min_p10_gmgr]
    if not winning:
        winning = [results[0][0]]
    _logger.info("Stage1 narrowed SIGNAL_TYPE choices: %s", winning)
    return _build_narrowed_space_for_signal_types(tuple(winning), tf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=",".join(SPOT_SYMBOLS))
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="multi")
    parser.add_argument("--trials", type=int, default=OPT_SPOT_CONFIG["total_trials"])
    parser.add_argument("--jobs", type=int, default=int(OPT_SPOT_CONFIG.get("n_jobs", 6)))
    parser.add_argument("--task-workers", type=int, default=int(OPT_SPOT_CONFIG.get("task_workers", 0)))
    parser.add_argument("--tf", type=str, choices=["4h"], default=OPT_SPOT_CONFIG.get("TARGET_TIMEFRAMES", ["4h"])[0])
    parser.add_argument("--reference-date", type=str, default=None)
    parser.add_argument(
        "--narrowed-space-json",
        type=str,
        default=None,
        help="Path to JSON dict merged into spot search space (overrides combination-screener output).",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip structure-discovery tournament; use --narrowed-space-json and/or full space.",
    )
    parser.add_argument(
        "--signal-type",
        type=str,
        default=None,
        help="Force a single SIGNAL_TYPE (narrows categorical); merged with --narrowed-space-json if set.",
    )
    parser.add_argument(
        "--stage1-trials",
        type=int,
        default=int(OPT_SPOT_CONFIG.get("SPOT_STAGE1_TRIALS_PER_SIGNAL", 80)),
        help="Trials per SIGNAL_TYPE during Stage-1 structure discovery.",
    )
    args = parser.parse_args()

    narrowed_space_file: Optional[Dict[str, Dict[str, Any]]] = None
    if args.narrowed_space_json:
        np_path = Path(args.narrowed_space_json)
        if np_path.is_file():
            import json

            narrowed_space_file = json.loads(np_path.read_text(encoding="utf-8"))
            if not isinstance(narrowed_space_file, dict):
                narrowed_space_file = None
                _logger.warning("narrowed-space-json root must be an object; ignoring.")
        else:
            _logger.warning("narrowed-space-json path not found: %s", np_path)

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

    for sym in valid_symbols:
        tf_k = str(args.tf)
        oos_df = oos_data_maps[sym][tf_k]
        oos_ix = int(oos_data_maps[sym].get(f"oos_start_idx_{tf_k}", len(oos_df)))
        oos_df.attrs["nu_fit_end"] = oos_ix

    if not valid_symbols:
        _logger.error("❌ No valid symbols with data found. Aborting optimization.")
        return

    tasks = [(tuple(valid_symbols), args.tf)] if args.mode == "multi" else [(s, args.tf) for s in valid_symbols]
    plan = _resolve_spot_execution_plan(len(tasks), args.mode, args.jobs, args.task_workers)
    
    # [수정] 메모리 절약을 위한 디스크 캐시 활성화
    signal_cache_dir = str(Path(project_root) / "data" / "cache_spot")
    Path(signal_cache_dir).mkdir(parents=True, exist_ok=True)
    _logger.info(f"Signal Disk Cache Enabled: {signal_cache_dir}")

    narrowed_space_effective: Optional[Dict[str, Dict[str, Any]]] = (
        {k: dict(v) if isinstance(v, dict) else v for k, v in narrowed_space_file.items()}
        if narrowed_space_file
        else None
    )
    if args.signal_type:
        forced = _build_narrowed_space_for_signal_types((str(args.signal_type).upper(),), args.tf)
        narrowed_space_effective = _merge_narrowed_space_dicts(forced, narrowed_space_effective)
        _logger.info("Applied --signal-type narrowed space: %s", str(args.signal_type).upper())
    elif not args.skip_stage1 and args.mode == "multi":
        from src.spot_strategy.opt_spot_utils.combination_screener import run_combination_screening
        from src.spot_strategy.opt_spot_utils.opt_params import build_combined_param_space

        tops = run_combination_screening(
            data_maps={s: data_maps[s] for s in valid_symbols},
            symbols=valid_symbols,
            tf=args.tf,
            project_root=project_root,
            signal_cache_dir=signal_cache_dir,
        )
        if not tops:
            _logger.error(
                "Stage1 combination screening found no viable combo (screen score <= threshold). "
                "No edge detected in current data/symbols. Aborting optimization."
            )
            return
        _logger.info(
            "Stage1 combination screening top combos: %s",
            [
                (x.signal, x.regime, x.sizing, round(x.p10_gmgr, 6), round(x.mean_signal_rate, 5), x.reason or "ok")
                for x in tops
            ],
        )
        narrowed_space_effective = _merge_narrowed_space_dicts(
            build_combined_param_space(tops[0].signal, tops[0].regime, tops[0].sizing),
            narrowed_space_effective,
        )
    elif not args.skip_stage1:
        stage1_space = _run_stage1_structure_discovery(
            data_maps=data_maps,
            symbols=valid_symbols,
            tf=args.tf,
            trials_per_signal=int(args.stage1_trials),
            top_k=int(OPT_SPOT_CONFIG.get("SPOT_STAGE1_TOP_K", 2)),
            min_p10_gmgr=float(OPT_SPOT_CONFIG.get("SPOT_STAGE1_MIN_P10_GMGR", -0.5)),
            project_root=project_root,
            signal_cache_dir=signal_cache_dir,
        )
        narrowed_space_effective = _merge_narrowed_space_dicts(stage1_space, narrowed_space_effective)

    need_multiprocess_queue = plan.use_outer_process_pool or (
        args.mode == "multi" and plan.parallel_tpe_workers > 1
    )
    manager = Manager() if need_multiprocess_queue else None
    progress_queue = manager.Queue() if manager else queue.Queue()

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
        if plan.use_outer_process_pool:
            with concurrent.futures.ProcessPoolExecutor(max_workers=plan.outer_task_workers) as exec:
                futures = {
                    exec.submit(
                        _run_tf_optimization,
                        t,
                        _TfOptimizationContext(
                            clean_symbol=("_".join(t[0]) if isinstance(t[0], tuple) else t[0]).replace("/", ""),
                            seeds=OPT_SPOT_CONFIG["seeds"],
                            n_trials=args.trials,
                            n_jobs=plan.jobs_per_task,
                            data_maps={s: data_maps[s] for s in (list(t[0]) if isinstance(t[0], tuple) else [t[0]])},
                            symbols=(list(t[0]) if isinstance(t[0], tuple) else [t[0]]),
                            project_root=project_root,
                            progress_queue=progress_queue,
                            mode=args.mode,
                            use_journal_storage=False,
                            parallel_tpe_workers=1,
                            disable_inner_tpe_pool=True,
                            fresh_journal=True,
                            signal_cache_dir=signal_cache_dir,
                            narrowed_space=narrowed_space_effective,
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
                        seeds=OPT_SPOT_CONFIG["seeds"],
                        n_trials=args.trials,
                        n_jobs=plan.jobs_per_task,
                        data_maps=data_maps,
                        symbols=(list(t[0]) if isinstance(t[0], tuple) else [t[0]]),
                        project_root=project_root,
                        progress_queue=progress_queue,
                        mode=args.mode,
                        use_journal_storage=(args.mode == "multi"),
                        parallel_tpe_workers=plan.parallel_tpe_workers if args.mode == "multi" else 1,
                        disable_inner_tpe_pool=plan.use_outer_process_pool,
                        fresh_journal=True,
                        signal_cache_dir=signal_cache_dir,
                        narrowed_space=narrowed_space_effective,
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
    gate1_tr_min = float(OPT_SPOT_CONFIG.get("SPOT_GATE1_TAIL_RATIO_MIN", 2.0))
    discovery_dsr_min = float(OPT_SPOT_CONFIG.get("SPOT_DISCOVERY_DSR_MIN", 0.25))
    holdout_min_pf = float(OPT_SPOT_CONFIG.get("SPOT_HOLDOUT_MIN_PROFIT_FACTOR", 1.3))
    holdout_min_calmar = float(OPT_SPOT_CONFIG.get("SPOT_HOLDOUT_MIN_CALMAR_RATIO", 1.5))

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
        gmgr_v = float(best_trial.user_attrs.get("p10_gmgr", 0.0))
        veto = run_portfolio_discovery_veto(
            psr=psr_v,
            dsr=dsr_v,
            p10_gmgr=gmgr_v,
            dsr_min=discovery_dsr_min,
        )
        veto_ok = bool(veto.passed)

        min_pf_trades_dynamic = max(50, len(target_symbols) * 8)

        port_ho = run_holdout_shared_cash_portfolio(
            params,
            target_symbols,
            tf_eval,
            oos_data_maps,
            signal_disk_cache_root=Path(signal_cache_dir),
            return_signal_dfs=True,
            concurrency_penalty_scale=1.0,
        )
        post_full_signal_dfs: Dict[str, pd.DataFrame] = port_ho.get("full_signal_dfs", {})
        oos_dd_days = float(port_ho["dd_bars"]) / 6.0

        symbol_fold_payloads: List[Dict[str, Any]] = []
        is_cagr_vals: List[float] = []
        for s_eval in target_symbols:
            is_start_idx = int(data_maps[s_eval][f"is_start_idx_{tf_eval}"])
            is_end_idx = len(data_maps[s_eval][tf_eval])
            oos_start_idx = int(oos_data_maps[s_eval][f"oos_start_idx_{tf_eval}"])
            oos_end_idx = len(oos_data_maps[s_eval][tf_eval])
            pre_sig_full = post_full_signal_dfs.get(s_eval)
            pre_is_df, pre_is_exec = _build_precomputed_segment(pre_sig_full, is_start_idx, is_end_idx)
            pre_oos_df, pre_oos_exec = _build_precomputed_segment(pre_sig_full, oos_start_idx, oos_end_idx)

            s_is, r_is, m_is, t_is, wr_is, pf_is, lc_is, _, tr_is = evaluate_symbol_fold(
                UltimateSpotStrategy(name=f"IS_{s_eval}", params=params),
                params,
                s_eval,
                tf_eval,
                data_maps[s_eval][tf_eval],
                data_maps[s_eval]["1d"],
                data_maps[s_eval][f"merge_idx_{tf_eval}"],
                None,
                is_start_idx,
                is_end_idx,
                precomputed_signal_df=pre_is_df,
                execution_start_idx=pre_is_exec,
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
                oos_start_idx,
                oos_end_idx,
                precomputed_signal_df=pre_oos_df,
                execution_start_idx=pre_oos_exec,
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
        is_holdout_maps: Dict[str, Dict[str, Any]] = {}
        for s_eval in target_symbols:
            is_holdout_maps[s_eval] = dict(data_maps[s_eval])
            is_holdout_maps[s_eval][f"oos_start_idx_{tf_eval}"] = data_maps[s_eval][f"is_start_idx_{tf_eval}"]

        port_is = run_holdout_shared_cash_portfolio(
            params,
            target_symbols,
            tf_eval,
            is_holdout_maps,
            signal_disk_cache_root=Path(signal_cache_dir),
            return_signal_dfs=False,
            concurrency_penalty_scale=1.0,
        )
        is_portfolio_cagr: float = float(best_trial.user_attrs.get("cpcv_mean_path_return_pct", 0.0))

        trade_floor = run_holdout_portfolio_trade_floor(
            portfolio_long_trades=int(port_ho["long_trades"]),
            min_portfolio_trades=min_pf_trades_dynamic,
        )
        shared_cash_gate = run_holdout_portfolio_shared_cash(
            portfolio_cagr_pct=float(port_ho["portfolio_cagr_pct"]),
            portfolio_mdd_pct=float(port_ho["mdd_pct"]),
            portfolio_cvar_pct=float(port_ho["cvar_pct"]),
            portfolio_tail_ratio=float(port_ho["tail_ratio"]),
            min_path_terminal_wealth_ratio=float(port_ho["min_path_tw"]),
            portfolio_profit_factor=float(port_ho["profit_factor"]),
            portfolio_calmar_ratio=float(port_ho["calmar_ratio"]),
            max_cvar_pct=max_ho_cvar,
            tail_ratio_min=holdout_min_tail,
            cagr_min_pct=holdout_min_cagr,
            mdd_limit_pct=holdout_mdd_limit,
            oos_dd_days=oos_dd_days,
            hw_recovery_days_max=holdout_hwm_max_days,
            is_cagr_pct=is_portfolio_cagr,
            alpha_decay_floor_pct=holdout_alpha_floor,
            pf_min=holdout_min_pf,
            calmar_min=holdout_min_calmar,
        )
        is_all_passed = bool(veto_ok and trade_floor.passed and shared_cash_gate.passed)
        if not is_all_passed:
            _logger.info("❌ Gate check failed. Diagnostic details:")
            _logger.info("\n%s", veto.summary)
            _logger.info("\n%s", trade_floor.summary)
            _logger.info("\n%s", shared_cash_gate.summary)

        symbol_gate_rows: List[SymbolGateRow] = []
        for pl in symbol_fold_payloads:
            s_eval = pl["sym"]
            o = pl["oos"]
            s_oos = float(o["cagr"])
            m_oos = float(o["mdd"])
            trd_oos = int(o["trd"])
            wr_oos = float(o["wr"])
            tail_oos = float(o["tail"])
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
                gate1_p10_gmgr=float(best_trial.user_attrs.get("p10_gmgr", 0.0)),
                gate1_max_ui=float(best_trial.user_attrs.get("max_ulcer_index", 0.0)),
                gate1_psr=psr_v,
                gate1_dsr=dsr_v,
                cpcv_mean_path_return_pct=float(best_trial.user_attrs.get("cpcv_mean_path_return_pct", 0.0)),
                cpcv_worst_segment_mdd_pct=float(best_trial.user_attrs.get("cpcv_worst_segment_mdd_pct", 0.0)),
                sqn_target=gate1_sqn_min,
                path_sortino_target=gate1_psort_min,
                tail_ratio_target=gate1_tr_min,
                psr_target=0.5,
                dsr_target=discovery_dsr_min,
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
                oos_cvar_pct=float(port_ho["cvar_pct"]),
                cvar_limit_pct=max_ho_cvar,
                terminal_wealth_ratio=float(port_ho["min_path_tw"]),
                tw_target=1.0,
                oos_total_trades=int(port_ho.get("long_trades", 0)),
                oos_pf=float(port_ho["profit_factor"]),
                pf_target=holdout_min_pf,
                oos_calmar=float(port_ho["calmar_ratio"]),
                calmar_target=holdout_min_calmar,
                oos_win_rate_pct=float(port_ho["win_rate_pct"]),
                symbol_rows=symbol_gate_rows,
                loso_warning=loso_warning,
                hard_passed=hard_passed,
                hard_total=hard_total,
                final_decision_go=is_all_passed,
            )
        )
        _logger.info("\n%s", report)

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
