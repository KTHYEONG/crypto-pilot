from __future__ import annotations

import argparse
import concurrent.futures
import copy
import importlib
import logging
import multiprocessing
import os
import sys
import threading
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.pruners import MedianPruner, PatientPruner
from optuna.samplers import QMCSampler, TPESampler
from optuna.storages import InMemoryStorage
from optuna.trial import FixedTrial, TrialState
from tqdm import tqdm

# Project Root Setup
project_root: str = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import warnings  # noqa: E402

import config.opt_config  # noqa: E402
from config.opt_config import (  # noqa: E402
    FUTURES_ANCHOR_SYMBOLS,
    FUTURES_SCREENER_CONFIG,
    OPT_FUTURES_CONFIG,
    get_quarterly_window,
    get_search_space_futures,
)
from config.settings import (  # noqa: E402
    FUTURES_CACHE_DIR,
    FUTURES_DATA_DIR,
    FUTURES_INITIAL_BALANCE,
)
from src.core.optimization.opt_utils import compute_segment_merge_index  # noqa: E402
from src.domain.futures.data_collector import DataCollector  # noqa: E402
from src.domain.futures.funding_utils import merge_funding_into_ohlcv  # noqa: E402
from src.domain.futures.opt_futures_utils.combination_screener_futures import (  # noqa: E402
    CombinationScoreFutures,
)
from src.domain.futures.opt_futures_utils.alpha_evaluator import (  # noqa: E402
    calculate_conditional_ic,
    calculate_spearman_ic,
    compute_vol_adj_forward_returns,
)
from src.domain.futures.opt_futures_utils.cv_utils import (  # noqa: E402
    build_cpcv_test_paths_with_fallback,
    list_cpcv_block_ranges,
)
from src.domain.futures.opt_futures_utils.go_nogo import (  # noqa: E402
    FuturesDeploymentReportInput,
    FuturesSymbolGateRow,
    run_futures_deployment_report,
)
from src.domain.futures.opt_futures_utils.objective import (  # noqa: E402
    EMBARGO_BARS,
    inject_cs_momentum_ranks,
)
from src.domain.futures.opt_futures_utils.oos_evaluator import (  # noqa: E402
    run_oos_margin_shared_portfolio,
)

warnings.filterwarnings("ignore")

# Force Linux 'fork' method for memory efficiency (CoW)
if sys.platform != "win32":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

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
_STAGE1_WORKER_CTX: Dict[str, Any] = {}
_TPE_WORKER_CTX: Dict[str, Any] = {}

# EMBARGO_BARS is imported from objective.py (single source of truth).
# Using compute_embargo_bars() values: {"1h": 24, "4h": 12}.

# TPE constraint: avg MDD across valid CPCV paths (separate from per-segment limit).
_MDD_CONSTRAINT_LIMIT: float = float(
    OPT_FUTURES_CONFIG.get("FUTURES_MAX_AVG_CPCV_MDD", 25.0)
)


def futures_frozen_trial_constraints(trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
    pf = float(trial.user_attrs.get("avg_pf", 0.0) or 0.0)
    trades = float(trial.user_attrs.get("avg_trades", 0.0) or 0.0)
    avg_mdd = float(trial.user_attrs.get("avg_mdd", 100.0) or 100.0)
    ls_ratio = float(trial.user_attrs.get("long_short_ratio", 0.0) or 0.0)
    ev_cost = float(trial.user_attrs.get("ev_cost_ratio", 0.0) or 0.0)

    return (
        1.35 - pf,
        25.0 - trades,
        avg_mdd - _MDD_CONSTRAINT_LIMIT,
        0.15 - ls_ratio,
        3.0 - ev_cost,
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
    parallel_tpe_workers: int = 1
    disable_inner_tpe_pool: bool = False
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


@dataclass(frozen=True)
class _FuturesParallelPolicy:
    cpu_cap: int
    universe_workers: int
    stage1_workers: int
    qmc_jobs: int
    tpe_workers: int


def _resolve_futures_parallel_policy(symbol_count: int, tf: str) -> _FuturesParallelPolicy:
    logical_cpus = max(1, os.cpu_count() or 1)
    cpu_cap = max(1, min(8, logical_cpus))

    if symbol_count <= 4:
        universe_target = min(4, cpu_cap)
        stage1_target = min(6, cpu_cap)
        qmc_target = min(4, cpu_cap)
        tpe_target = min(4, cpu_cap)
    elif symbol_count <= 10:
        universe_target = min(4, cpu_cap)
        stage1_target = min(6, cpu_cap - 1) if cpu_cap > 2 else 2
        qmc_target = min(3, cpu_cap - 1) if cpu_cap > 2 else 2
        tpe_target = min(3, cpu_cap - 1) if cpu_cap > 2 else 2
    else:
        universe_target = min(3, cpu_cap - 1) if cpu_cap > 2 else 2
        stage1_target = max(2, cpu_cap - 2)
        qmc_target = max(2, cpu_cap - 2)
        tpe_target = max(1, min(2, cpu_cap - 3)) if cpu_cap > 3 else 1

    return _FuturesParallelPolicy(
        cpu_cap=cpu_cap,
        universe_workers=max(1, min(universe_target, cpu_cap)),
        stage1_workers=max(1, min(stage1_target, cpu_cap)),
        qmc_jobs=max(1, min(qmc_target, cpu_cap)),
        tpe_workers=max(1, min(tpe_target, cpu_cap)),
    )


def _resolve_futures_execution_plan(
    task_count: int,
    mode: str,
    qmc_jobs: int,
    tpe_workers: int,
) -> _FuturesExecutionPlan:
    worker_cap = 6
    logical_cpus = max(1, os.cpu_count() or 1)
    if mode == MODE_MULTI and task_count == 1:
        jobs = max(1, min(int(qmc_jobs), logical_cpus, worker_cap))
        tw = max(1, min(int(tpe_workers), logical_cpus, worker_cap))
        return _FuturesExecutionPlan(1, jobs, tw, False, logical_cpus, task_count)

    jobs = max(1, min(int(qmc_jobs), logical_cpus, worker_cap))
    outer_tw = max(1, min(task_count, int(tpe_workers), worker_cap))
    use_outer = task_count > 1 and outer_tw > 1
    return _FuturesExecutionPlan(outer_tw, jobs, 1, use_outer, logical_cpus, task_count)


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
        # [OPTIMIZATION] Use .copy() to prevent SettingWithCopyWarning
        # and memory leaks between IS/OOS
        is_mask = full["datetime"] < is_end_dt
        is_end_idx = int(is_mask.to_numpy().sum())
        is_df_view = full.iloc[:is_end_idx].copy()

        data_maps[sym][tf] = is_df_view
        m = is_df_view["datetime"] >= is_start_dt
        data_maps[sym][f"is_start_idx_{tf}"] = int(m.to_numpy().argmax()) if bool(m.any()) else 0
        data_maps[sym][f"merge_idx_{tf}"] = compute_segment_merge_index(
            is_df_view, data_maps[sym]["1d"]
        )


def _align_oos_dataframes_on_common_datetimes(
    oos_data_maps: Dict[str, Dict[str, Any]],
    symbols: Sequence[str],
    tf: str,
    is_end_dt: pd.Timestamp,
) -> None:
    sym_list = list(symbols)
    if len(sym_list) < 2:
        return
    dts = []
    for s in sym_list:
        if tf in oos_data_maps[s]:
            dts.append(set(oos_data_maps[s][tf]["datetime"]))
        else:
            _logger.warning(f"Symbol {s} missing timeframe {tf} in oos_data_maps")
            
    if not dts:
        return

    common_set = set.intersection(*dts)
    if len(common_set) < 200:
        # Identify problematic symbols with low overlap
        for s in sym_list:
            s_dts = set(oos_data_maps[s][tf]["datetime"])
            overlap = len(s_dts.intersection(common_set)) if common_set else 0
            _logger.info(f"  - {s}: total={len(s_dts)}, overlap_with_common={overlap}")
        raise ValueError(f"Insufficient overlapping bars ({len(common_set)}) in OOS maps.")

    for sym in sym_list:
        df = oos_data_maps[sym][tf]
        isin_mask = df["datetime"].isin(common_set)
        filtered = df[isin_mask].sort_values("datetime").reset_index(drop=True)
        oos_data_maps[sym][tf] = filtered
        m_oos = filtered["datetime"] >= is_end_dt
        oos_data_maps[sym][f"oos_start_idx_{tf}"] = (
            int(m_oos.to_numpy().argmax()) if bool(m_oos.any()) else len(filtered)
        )
        oos_data_maps[sym][f"merge_idx_{tf}"] = compute_segment_merge_index(
            filtered, oos_data_maps[sym]["1d"]
        )


def _select_best_trial_from_shortlist(
    ranked: List[optuna.trial.FrozenTrial]
) -> optuna.trial.FrozenTrial:
    if not ranked:
        raise ValueError("ranked trials empty")
    top_val = float(ranked[0].value if ranked[0].value is not None else -1e9)
    band = max(abs(top_val) * 0.02, 0.02)
    pool = [t for t in ranked if t.value is not None and abs(float(t.value) - top_val) <= band]
    if not pool:
        pool = ranked[:1]

    def robust_key(t: optuna.trial.FrozenTrial) -> Tuple[float, float, float]:
        ua = t.user_attrs
        return (
            float(ua.get("gate1_psr", 0.0)),
            float(ua.get("gate1_sqn", 0.0)),
            float(ua.get("gate1_dsr", 0.0))
        )
    return max(pool, key=robust_key)


def _init_tpe_worker_context() -> None:
    # [OPTIMIZATION] Rely on GLOBAL CoW inherited from main process via 'fork'
    pass


def _convert_space_to_distributions(space: Dict[str, Any]) -> Dict[str, BaseDistribution]:
    dists: Dict[str, BaseDistribution] = {}
    for name, spec in space.items():
        stype = spec.get("type")
        if stype == "int":
            dists[name] = IntDistribution(
                low=int(spec["low"]), high=int(spec["high"]),
                step=int(spec.get("step", 1)), log=bool(spec.get("log", False))
            )
        elif stype == "float":
            dists[name] = FloatDistribution(
                low=float(spec["low"]), high=float(spec["high"]),
                step=float(spec.get("step")) if spec.get("step") else None,
                log=bool(spec.get("log", False))
            )
        elif stype == "categorical":
            dists[name] = CategoricalDistribution(choices=spec["choices"])
    return dists


def _futures_worker_ask_tell(
    trial_number: int, params: Dict[str, Any], tf: str, symbols: List[str],
    mode: str, project_root: str, signal_cache_dir: str, tf_space: Dict[str, Any],
) -> Tuple[int, Optional[float], Dict[str, Any], TrialState]:
    from src.domain.futures.opt_futures_utils.objective import objective_futures
    data_maps = _TPE_WORKER_CTX.get("data_maps", {})
    prebuilt_cpcv_bundle = _TPE_WORKER_CTX.get("prebuilt_cpcv_bundle")
    alignment_info = _TPE_WORKER_CTX.get("alignment_info")
    
    cache_root = Path(signal_cache_dir) if signal_cache_dir else FUTURES_CACHE_DIR
    trial = FixedTrial(params, number=trial_number)
    try:
        # Cast trial to Any to satisfy mypy if objective_futures expects optuna.Trial
        val = objective_futures(
            trial,  # type: ignore[arg-type]
            data_maps=data_maps, symbols=symbols, tf=tf, space=tf_space, mode=mode,
            project_root=project_root, prebuilt_cpcv_bundle=prebuilt_cpcv_bundle,
            multi_alignment_info=alignment_info, signal_disk_cache_root=cache_root
        )
        return trial_number, float(val), trial.user_attrs, TrialState.COMPLETE
    except optuna.TrialPruned:
        return trial_number, None, trial.user_attrs, TrialState.PRUNED
    except Exception as e:
        _logger.error(f"Worker trial {trial_number} failed: {e}")
        return trial_number, None, {}, TrialState.FAIL


def _run_parallel_ask_tell(
    study: optuna.Study, n_trials: int, n_workers: int,
    distributions: Dict[str, BaseDistribution],
    ctx: _TfOptimizationContext, tf: str, progress_key: str,
) -> None:
    if n_trials <= 0:
        return
    n_workers = max(1, min(n_workers, n_trials))
    _logger.info("   [Parallel] Running %d trials (%d workers)...", n_trials, n_workers)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=multiprocessing.get_context("fork"), # Ensure CoW
        initializer=_init_tpe_worker_context
    ) as executor:
        futures_to_trials, remaining = {}, n_trials
        space = ctx.locked_param_space if ctx.locked_param_space else get_search_space_futures(tf)
        for _ in range(min(n_workers, remaining)):
            trial = study.ask(distributions)
            f = executor.submit(
                _futures_worker_ask_tell, trial.number, trial.params, tf, ctx.symbols, ctx.mode,
                ctx.project_root, ctx.signal_cache_dir, space
            )
            futures_to_trials[f], remaining = trial, remaining - 1

        desc_str = f"Optimizing {tf}"
        with tqdm(total=n_trials, desc=desc_str, disable=ctx.progress_queue is None) as pbar:
            best_val = -1e9
            while futures_to_trials:
                done, _ = concurrent.futures.wait(
                    futures_to_trials.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                )
                for f in done:
                    trial = futures_to_trials.pop(f)
                    try:
                        t_num, val, user_attrs, state = f.result()

                        # Set user_attrs via trial._trial_id (direct, number-safe)
                        # before study.tell() to avoid UpdateFinishedTrialError.
                        for k, v in user_attrs.items():
                            study._storage.set_trial_user_attr(trial._trial_id, k, v)

                        study.tell(trial, val, state=state)

                        if val is not None and val > best_val:
                            best_val = val
                        if ctx.progress_queue:
                            ctx.progress_queue.put((
                                progress_key, t_num + 1, n_trials,
                                float(user_attrs.get("kelly_score_pct", best_val * 100.0))
                            ))
                    except Exception as e:
                        _logger.error("Trial worker failed: %s", e)
                        # Look up by number to avoid index-vs-number confusion
                        finished = {t.number: t for t in study.get_trials()}
                        ft = finished.get(trial.number)
                        if ft is None or not ft.state.is_finished():
                            study.tell(trial, state=TrialState.FAIL)
                    pbar.update(1)
                    if remaining > 0:
                        new_trial = study.ask(distributions)
                        new_f = executor.submit(
                            _futures_worker_ask_tell, new_trial.number, new_trial.params, tf,
                            ctx.symbols, ctx.mode, ctx.project_root,
                            ctx.signal_cache_dir, space
                        )
                        futures_to_trials[new_f], remaining = new_trial, remaining - 1


def _run_tf_optimization(
    task: Tuple[Tuple[str, ...], str], ctx: _TfOptimizationContext
) -> Tuple[Tuple[Tuple[str, ...], str], Optional[optuna.Study]]:
    target_obj, tf = task
    progress_key = _task_progress_key(target_obj, tf)
    target_str = "_".join(target_obj)
    tf_study_name: str = f"OptFutures_{target_str.replace('/', '')}_{tf}_{ctx.mode}"
    storage = InMemoryStorage()
    seed = int(ctx.seeds[0])
    
    n_startup = int(OPT_FUTURES_CONFIG.get("tpe_n_startup_trials", 256))
    n_qmc = min(n_startup, int(ctx.n_trials))
    
    qmc_sampler = QMCSampler(
        qmc_type="sobol", scramble=True, seed=seed, warn_independent_sampling=False
    )
    
    # Disable pruner for tiny trial counts to ensure we get results for testing
    if int(ctx.n_trials) <= 2:
        pruner = optuna.pruners.NopPruner()
    else:
        base_p = MedianPruner(
            n_startup_trials=int(OPT_FUTURES_CONFIG.get("tpe_pruner_n_startup_trials", 10)),
            n_warmup_steps=int(OPT_FUTURES_CONFIG.get("tpe_pruner_n_warmup_steps", 8))
        )
        pruner = PatientPruner(
            base_p, patience=int(OPT_FUTURES_CONFIG.get("tpe_pruner_patience", 2))
        )

    from src.domain.futures.opt_futures_utils.objective import (
        compute_multi_alignment_info,
    )
    # EMBARGO_BARS is imported at module level from objective.py (SSOT).
    alignment_info = compute_multi_alignment_info(
        ctx.data_maps, ctx.symbols, tf, int(EMBARGO_BARS.get(tf, 0))
    ) if ctx.mode == MODE_MULTI else None

    ref_sym0 = ctx.symbols[0]
    is_off0 = int(ctx.data_maps[ref_sym0].get(f"is_start_idx_{tf}", 0))
    ref_df0 = ctx.data_maps[ref_sym0][tf]
    prebuilt_cpcv_bundle = alignment_info["cpcv_bundle"] if alignment_info else None
    if not prebuilt_cpcv_bundle and ref_df0 is not None and not ref_df0.empty:
        ref_len0 = len(ref_df0) - is_off0
        if ref_len0 >= 200:
            prebuilt_cpcv_bundle = build_cpcv_test_paths_with_fallback(
                ref_len0, embargo=int(EMBARGO_BARS.get(tf, 0))
            )

    if ctx.locked_param_space is not None:
        tf_space = ctx.locked_param_space
    else:
        tf_space = get_search_space_futures(tf)
    distributions = _convert_space_to_distributions(tf_space)

    # Set Global Context for TPE Optimization (CoW)
    _TPE_WORKER_CTX.clear()
    _TPE_WORKER_CTX["data_maps"] = ctx.data_maps
    _TPE_WORKER_CTX["alignment_info"] = alignment_info
    _TPE_WORKER_CTX["prebuilt_cpcv_bundle"] = prebuilt_cpcv_bundle

    study = optuna.create_study(
        study_name=tf_study_name, storage=storage, direction="maximize",
        sampler=qmc_sampler, pruner=pruner
    )
    _logger.info("[%s/%s] Phase 1: QMC Startup...", target_str, tf)
    _run_parallel_ask_tell(
        study, n_qmc, ctx.n_jobs, distributions, ctx, tf, progress_key
    )

    remaining = int(ctx.n_trials) - n_qmc
    if remaining > 0:
        _logger.info("[%s/%s] Phase 2: TPE Optimization...", target_str, tf)
        study.sampler = _futures_tpe_sampler(seed)
        _run_parallel_ask_tell(
            study, remaining, ctx.parallel_tpe_workers, distributions, ctx, tf,
            progress_key
        )
    return (target_obj, tf), study


def _eval_combo_task(
    sig: str, reg: str, siz: str, data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str], tf: str, project_root: str, prebuilt: Any,
    multi_alignment_info: Optional[Dict[str, Any]] = None,
) -> Optional[CombinationScoreFutures]:
    from src.domain.futures.opt_futures_utils.objective import objective_futures
    from src.domain.futures.opt_futures_utils.opt_params import build_combined_param_space_futures
    
    space = build_combined_param_space_futures(sig, reg, siz)
    space["SIGNAL_TYPE"] = {"type": "categorical", "choices": [sig]}
    space["REGIME_TYPE"] = {"type": "categorical", "choices": [reg]}
    space["SIZING_METHOD"] = {"type": "categorical", "choices": [siz]}
    sampler = QMCSampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, storage=InMemoryStorage())

    def _obj(trial: optuna.Trial) -> float:
        return float(objective_futures(
            trial, data_maps, symbols, tf, space=space, mode=MODE_MULTI,
            project_root=project_root, prebuilt_cpcv_bundle=prebuilt,
            multi_alignment_info=multi_alignment_info,
            signal_disk_cache_root=Path(FUTURES_CACHE_DIR), relaxed_constraints=True
        ))
    n_trials = max(64, min(256, (len(space)-3) * 12))
    study.optimize(_obj, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    if not completed:
        pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
        reason_dist = Counter(t.user_attrs.get("prune_reason", "unknown") for t in pruned)
        return CombinationScoreFutures(
            signal=sig, regime=reg, sizing=siz, p10_gmgr=-1e9, ls_ratio=0.0,
            mean_signal_rate=0.0, disqualified=True, reason=str(dict(reason_dist))
        )

    raw_vals = np.array([float(t.user_attrs.get("gate1_p10_gmgr", 0.0)) for t in completed])
    hrs = int(tf.replace("h", "")) if tf.endswith("h") else 4
    eff_len = np.mean([int(t.user_attrs.get("gate1_eff_ref_len", 4380)) for t in completed])
    
    val_ratio = 3 / 8  # Default
    if prebuilt is not None:
        cpcv_paths, n_blocks, _ = prebuilt
        if cpcv_paths and n_blocks:
            n_test_blocks = len(cpcv_paths[0][1])
            val_ratio = n_test_blocks / n_blocks
            
    ann_factor = (365 * 24 / hrs) / max(1, int(eff_len * val_ratio))
    
    raw_vals_ann = np.sort(np.array([
        float(np.expm1(np.log1p(max(-0.9999, v)) * ann_factor)) for v in raw_vals
    ]))[::-1]
    top_n = max(1, int(len(raw_vals_ann) * 0.2))
    mu_top = np.mean(raw_vals_ann[:top_n])
    sd_top = (np.std(raw_vals_ann[:top_n]) if top_n > 1 else 0.0)
    pos_vals = raw_vals_ann[raw_vals_ann > 0.03]
    edge_density = (len(pos_vals) / len(raw_vals_ann)) * (
        np.mean(pos_vals)/(np.std(pos_vals)+1e-9) if len(pos_vals)>0 else 0.0
    )
    best_t = max(completed, key=lambda tr: float(tr.value if tr.value is not None else -1e9))
    return CombinationScoreFutures(
        signal=sig, regime=reg, sizing=siz,
        p10_gmgr=(mu_top - 0.5 * sd_top) * (1.0 + min(1.0, edge_density)),
        ls_ratio=float(best_t.user_attrs.get("long_short_ratio", 0.5)),
        mean_signal_rate=float(best_t.user_attrs.get("avg_signal_rate", 0.05)),
        disqualified=False, best_params=best_t.params
    )


def _init_stage1_worker_context() -> None:
    # [OPTIMIZATION] Inherit global context via CoW.
    pass


def _eval_combo_task_from_context(
    sig: str, reg: str, siz: str
) -> Optional[CombinationScoreFutures]:
    ctx = _STAGE1_WORKER_CTX
    return _eval_combo_task(
        sig, reg, siz, ctx["data_maps"], ctx["symbols"], ctx["tf"],
        ctx["project_root"], ctx["prebuilt"], ctx.get("alignment_info")
    )


def _run_stage1_alpha_ic_discovery(
    sig_type: str,
    tf: str,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    n_tri: int = 40,
) -> tuple[str, float, Dict[str, Any]]:
    """Phase C-1: Optimize signal parameters for MAX Information Coefficient (IC)"""
    from src.domain.futures.opt_futures_utils.opt_params import suggest_params_futures
    from src.domain.futures.signals import get_futures_signal

    sig_class = get_futures_signal(sig_type)
    base_space = get_search_space_futures(tf)
    # Filter search space for this signal type only to speed up C-1
    stage_space = {
        k: v for k, v in base_space.items() 
        if k.startswith(f"{sig_type}_") or k == "SIGNAL_TYPE"
    }
    stage_space["SIGNAL_TYPE"] = {"type": "categorical", "choices": (sig_type,)}

    # Pre-calculate target returns for all symbols
    target_map = {sym: compute_vol_adj_forward_returns(data_maps[sym][tf]) for sym in symbols}

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(n_startup_trials=max(5, n_tri // 4), seed=42),
        storage=InMemoryStorage(),
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params_futures(trial, stage_space, tf)
        ics = []
        sig_inst = sig_class()
        for sym in symbols:
            df = data_maps[sym][tf]
            out = sig_inst.compute(df, params)
            ic = calculate_spearman_ic(out.rank_score, target_map[sym])
            ics.append(ic)
        return float(np.mean(ics))  # Mean IC across symbols

    study.optimize(objective, n_trials=n_tri)
    best_ic = float(study.best_value) if study.best_value is not None else 0.0
    best_params = study.best_params

    return sig_type, best_ic, best_params


def _run_stage1_regime_pairing(
    sig_type: str,
    sig_params: Dict[str, Any],
    tf: str,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
) -> str:
    """Phase C-2: Find the best Regime for a proven Signal using Conditional IC Utility"""
    from src.domain.futures.regimes import get_futures_regime_registry
    from src.domain.futures.signals import get_futures_signal

    reg_registry = get_futures_regime_registry()
    sig_inst = get_futures_signal(sig_type)()

    best_regime = "NONE"
    max_utility = -1e9

    target_map = {sym: compute_vol_adj_forward_returns(data_maps[sym][tf]) for sym in symbols}

    for reg_name, reg_class in reg_registry.items():
        utilities = []
        reg_inst = reg_class()
        for sym in symbols:
            df = data_maps[sym][tf]
            # Use default regime params for initial pairing discovery
            reg_mask = reg_inst.compute(df, {})
            sig_scores = sig_inst.compute(df, sig_params).rank_score

            c_ic, cov = calculate_conditional_ic(sig_scores, target_map[sym], reg_mask)
            # Utility = cIC * sqrt(Coverage) to penalize extremely rare regimes
            utility = c_ic * np.sqrt(max(0.01, cov))
            utilities.append(utility)

        avg_utility = float(np.mean(utilities))
        if avg_utility > max_utility:
            max_utility = avg_utility
            best_regime = reg_name

    return best_regime


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
    """Hierarchical IC-based Alpha Discovery (Phase C-1 & C-2)."""
    all_types = tuple(get_search_space_futures(tf)["SIGNAL_TYPE"]["choices"])
    
    _logger.info("======================================================================")
    _logger.info("[PHASE C] Hierarchical Alpha Discovery (IC-based)")
    _logger.info("  C-1: Pure Alpha Screening (IC) | C-2: Regime Pairing (cIC)")
    _logger.info("----------------------------------------------------------------------")
    _logger.info("  %-20s | %-14s | %-14s | %-8s", "Signal Strategy", "Mean IC", "Best Regime", "Status")
    _logger.info("----------------------------------------------------------------------")

    discovery_results: List[Tuple[str, float, str, Dict[str, Any]]] = []
    n_tri = max(20, int(trials_per_signal))

    # Parallelize Phase C-1 (Signal IC Optimization)
    ctx = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(len(all_types), 8),
        mp_context=ctx,
    ) as executor:
        f_to_sig = {
            executor.submit(
                _run_stage1_alpha_ic_discovery,
                sig, tf, data_maps, symbols, n_tri
            ): sig
            for sig in all_types
        }
        for f in concurrent.futures.as_completed(f_to_sig):
            try:
                sig_name, ic, params = f.result()
                # Phase C-2: Pairing (Synchronous as it is extremely fast)
                best_reg = _run_stage1_regime_pairing(sig_name, params, tf, data_maps, symbols)
                discovery_results.append((sig_name, ic, best_reg, params))
            except Exception as e:
                sig_err = f_to_sig[f]
                _logger.error("Signal IC discovery failed for %s: %s", sig_err, e)
                discovery_results.append((sig_err, -1.0, "NONE", {}))

    discovery_results.sort(key=lambda x: x[1], reverse=True)
    
    ic_threshold = 0.025
    winning = [
        (s, r, p) for s, ic, r, p in discovery_results 
        if ic > ic_threshold or (s == discovery_results[0][0])
    ]
    winning = winning[:max(1, top_k)]
    winning_names = [w[0] for w in winning]

    for sig, ic, reg, _ in discovery_results:
        is_winner = sig in winning_names
        status = "[SELECTED]" if is_winner else "[DROPPED]"
        indicator = "▶" if is_winner else " "
        _logger.info("  %s %-18s | %-14.6f | %-14s | %-8s", indicator, sig, ic, reg, status)

    _logger.info("----------------------------------------------------------------------")
    _logger.info("Phase C Summary: Narrowed to Top %d Alpha-Regime Pairs", len(winning))
    _logger.info("======================================================================")

    final_space = copy.deepcopy(get_search_space_futures(tf))
    final_space["SIGNAL_TYPE"]["choices"] = tuple(winning_names)
    
    paired_regimes = list(set([w[1] for w in winning]))
    final_space["REGIME_TYPE"]["choices"] = tuple(paired_regimes)

    return final_space


def _load_single_symbol_data(
    sym: str,
    tf: str,
    fetch_start: str,
    start: str,
    is_end: str,
    end: str,
    min_bars: int,
) -> tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
    try:
        temp_is: Dict[str, Any] = {}
        temp_oos: Dict[str, Any] = {}
        insufficient = False
        collector = DataCollector()
        collector.ensure_funding_data(sym, fetch_start, end)
        for tf_l in [tf, "1d"]:
            raw_df = collector.collect_and_save(sym, tf_l, fetch_start, end)
            df = merge_funding_into_ohlcv(sym, raw_df, FUTURES_DATA_DIR)
            
            is_anchor = sym in FUTURES_ANCHOR_SYMBOLS
            min_required = 500 if is_anchor and tf_l == tf else (min_bars if tf_l == tf else 200)
            
            if df is None or df.empty:
                insufficient = True
                break
            df.reset_index(drop=True, inplace=True)
            tz = df["datetime"].dt.tz
            is_start_dt = pd.to_datetime(start).tz_localize(tz) if tz else pd.to_datetime(start)
            is_end_dt = pd.to_datetime(is_end).tz_localize(tz) if tz else pd.to_datetime(is_end)
            
            is_mask = df["datetime"] < is_end_dt
            is_end_idx = int(is_mask.to_numpy().sum())
            
            if is_end_idx < min_required:
                insufficient = True
                break
                
            temp_is[tf_l] = df.iloc[:is_end_idx].copy()
            
            mask = temp_is[tf_l]["datetime"] >= is_start_dt
            temp_is[f"is_start_idx_{tf_l}"] = int(mask.to_numpy().argmax()) if mask.any() else 0
            temp_oos[tf_l] = df
            mask_oos = df["datetime"] >= is_end_dt
            temp_oos[f"oos_start_idx_{tf_l}"] = (
                int(mask_oos.to_numpy().argmax()) if mask_oos.any() else len(df)
            )
        
        if insufficient:
            return sym, None, None, True
            
        temp_is[f"merge_idx_{tf}"] = compute_segment_merge_index(temp_is[tf], temp_is["1d"])
        temp_oos[f"merge_idx_{tf}"] = compute_segment_merge_index(temp_oos[tf], temp_oos["1d"])
        return sym, temp_is, temp_oos, False
    except Exception as e:
        _logger.warning("Failed to load symbol %s: %s", sym, e)
        return sym, None, None, False


def _load_futures_data_maps_for_symbols(
    symbols: List[str],
    tf: str,
    fetch_start: str,
    start: str,
    is_end: str,
    end: str,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[str]]:
    data_maps: Dict[str, Dict[str, Any]] = {}
    oos_data_maps: Dict[str, Dict[str, Any]] = {}
    valid_symbols: List[str] = []
    min_bars = int(FUTURES_SCREENER_CONFIG.get("MIN_HISTORY_BARS", 2000))

    essential = ["BTC/USDT", "ETH/USDT"]
    load_symbols = list(dict.fromkeys(symbols + essential))

    # [OPTIMIZATION] Parallel loading using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(load_symbols), 8)) as executor:
        futures = [
            executor.submit(_load_single_symbol_data, sym, tf, fetch_start, start, is_end, end, min_bars)
            for sym in load_symbols
        ]
        for f in concurrent.futures.as_completed(futures):
            sym, t_is, t_oos, insufficient = f.result()
            if insufficient:
                if sym in essential and sym not in symbols:
                    _logger.warning(f"Essential symbol {sym} has insufficient data but is required for macro indicators.")
                continue
            if t_is and t_oos:
                data_maps[sym], oos_data_maps[sym] = t_is, t_oos
                if sym in symbols:
                    valid_symbols.append(sym)

    if len(valid_symbols) >= 2:
        ref_tz = oos_data_maps[valid_symbols[0]][tf]["datetime"].dt.tz
        is_start_align = pd.to_datetime(start).tz_localize(ref_tz) if ref_tz else pd.to_datetime(start)
        is_end_align = pd.to_datetime(is_end).tz_localize(ref_tz) if ref_tz else pd.to_datetime(is_end)
        
        # Align ALL loaded symbols including BTC/ETH
        all_loaded = list(data_maps.keys())
        _align_oos_dataframes_on_common_datetimes(oos_data_maps, all_loaded, tf, is_end_align)
        _rebuild_is_data_maps_from_aligned_oos(data_maps, oos_data_maps, all_loaded, tf, is_start_align, is_end_align)

    for rc, cn in [("BTC/USDT", "btc_close"), ("ETH/USDT", "eth_close")]:
        if rc in data_maps:
            rd = data_maps[rc][tf][["datetime", "close"]].rename(columns={"close": cn})
            rdo = oos_data_maps[rc][tf][["datetime", "close"]].rename(columns={"close": cn})
            for s in data_maps:
                if s != rc:
                    data_maps[s][tf] = data_maps[s][tf].merge(rd, on="datetime", how="left")
                    oos_data_maps[s][tf] = oos_data_maps[s][tf].merge(rdo, on="datetime", how="left")
                else:
                    data_maps[s][tf] = data_maps[s][tf].copy()
                    data_maps[s][tf][cn] = data_maps[s][tf]["close"]
                    oos_data_maps[s][tf] = oos_data_maps[s][tf].copy()
                    oos_data_maps[s][tf][cn] = oos_data_maps[s][tf]["close"]

    if len(valid_symbols) > 1:
        inject_cs_momentum_ranks(data_maps, valid_symbols, tf)
        inject_cs_momentum_ranks(oos_data_maps, valid_symbols, tf)

    return data_maps, oos_data_maps, valid_symbols


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--skip-universe", action="store_true")
    pre_parser.add_argument("--reference-date", type=str, default=None)
    pre_parser.add_argument("--skip-stage1", action="store_true")
    pre_parser.add_argument("--signal-type", type=str, default=None)
    pre_parser.add_argument("--tf", type=str, default="4h")
    pre_args, _ = pre_parser.parse_known_args()

    # [수정] 메모리 절약을 위한 디스크 캐시 활성화
    signal_cache_dir = str(Path(project_root) / "data" / "cache_futures")
    Path(signal_cache_dir).mkdir(parents=True, exist_ok=True)

    if not pre_args.skip_universe:
        from src.domain.futures.opt_futures_utils.universe_screener_futures import (
            screen_futures_universe,
            screen_symbol_refinement_futures,
        )

        fetch_start_date, start_date, is_end_date, end_date = get_quarterly_window(pre_args.reference_date)
        
        collector = DataCollector()
        # Phase A: Market-wide scan
        broad_candidates, _ = screen_futures_universe(
            collector, [], pre_args.tf, FUTURES_SCREENER_CONFIG, fetch_start_date, is_end_date, data_dir=FUTURES_DATA_DIR
        )

        if not broad_candidates:
            _logger.error("Phase A returned no broad candidates. Aborting.")
            return

        anchors_to_add = [s for s in FUTURES_ANCHOR_SYMBOLS if s not in broad_candidates]
        all_symbols_for_load = list(dict.fromkeys(list(broad_candidates) + anchors_to_add))

        data_maps_broad, _, valid_broad = _load_futures_data_maps_for_symbols(
            all_symbols_for_load, pre_args.tf, fetch_start_date, start_date, is_end_date, end_date
        )

        if len(valid_broad) < 1:
            _logger.error("Phase 0: no symbols with loadable data. Aborting.")
            return

        # Phase B: MHRH (Microstructure-Homogeneous, Returns-Heterogeneous) Refinement
        _logger.info("Phase B: MHRH statistical refinement (is_end=%s)", is_end_date)
        
        success = screen_symbol_refinement_futures(
            broad_candidates=list(broad_candidates),
            winning_signal_type="MHRH_PROBE",
            is_end_date=is_end_date,
            symbol_dfs_4h={s: data_maps_broad[s][pre_args.tf] for s in valid_broad},
            daily_dfs={s: data_maps_broad[s]["1d"] for s in valid_broad},
            phase_b_params=None,
            anchor_symbols=FUTURES_ANCHOR_SYMBOLS,
        )
        if not success:
            _logger.error("Phase B refinement failed. Aborting to avoid broken universe.")
            return

        # Reload config to get the updated FUTURES_SYMBOLS written by Phase B
        importlib.reload(config.opt_config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=",".join(config.opt_config.FUTURES_SYMBOLS))
    parser.add_argument("--skip-universe", action="store_true")
    parser.add_argument("--trials", type=int, default=OPT_FUTURES_CONFIG["total_trials"])
    parser.add_argument("--tf", type=str, choices=["1h", "4h"], default="4h")
    parser.add_argument("--reference-date", type=str, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--stage1-trials", type=int, default=80)
    args = parser.parse_args()

    fetch_start_date, start_date, is_end_date, end_date = get_quarterly_window(args.reference_date)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    _logger.info("Loading Futures data for %d symbols...", len(symbols))
    data_maps, oos_data_maps, valid_symbols = _load_futures_data_maps_for_symbols(
        symbols, args.tf, fetch_start_date, start_date, is_end_date, end_date
    )

    if not valid_symbols:
        _logger.error("No valid symbols.")
        return

    runtime_policy = _resolve_futures_parallel_policy(len(valid_symbols), args.tf)
    
    narrowed_space_effective = None
    if not args.skip_stage1:
        narrowed_space_effective = _run_stage1_structure_discovery(
            data_maps=data_maps,
            symbols=valid_symbols,
            tf=args.tf,
            trials_per_signal=args.stage1_trials,
            top_k=2,
            min_p10_gmgr=-0.5,
            project_root=project_root,
            signal_cache_dir=signal_cache_dir,
        )

    tasks: List[Tuple[Tuple[str, ...], str]] = [(tuple(valid_symbols), args.tf)]
    plan = _resolve_futures_execution_plan(
        len(tasks), MODE_MULTI, runtime_policy.qmc_jobs, runtime_policy.tpe_workers
    )
    
    manager = Manager()
    progress_queue: Optional[Any] = manager.Queue() if not args.no_progress else None
    tf_bars = {
        _task_progress_key(t, tf_t): tqdm(
            total=args.trials, desc=f"[{_task_progress_key(t, tf_t)}] Waiting...", position=i
        )
        for i, (t, tf_t) in enumerate(tasks)
    } if progress_queue else {}

    def _prog_listener() -> None:
        if progress_queue is None: return
        while True:
            msg = progress_queue.get()
            if msg is None: break
            k, cur, _, k_pct = msg
            if k in tf_bars:
                tf_bars[k].n = cur
                tf_bars[k].set_description(f"[{k}] Best Kelly: {k_pct:.2f}%")
                tf_bars[k].refresh()
    
    if progress_queue:
        threading.Thread(target=_prog_listener, daemon=True).start()

    _TPE_WORKER_CTX.clear()
    _TPE_WORKER_CTX["data_maps"] = data_maps

    best_results: Dict[Tuple[Tuple[str, ...], str], Optional[optuna.Study]] = {}
    for t in tasks:
        ctx = _TfOptimizationContext(
            clean_symbol="_".join(t[0]).replace("/", ""),
            seeds=OPT_FUTURES_CONFIG["seeds"],
            n_trials=args.trials,
            n_jobs=plan.jobs_per_task,
            data_maps=data_maps,
            symbols=list(t[0]),
            project_root=project_root,
            progress_queue=progress_queue,
            parallel_tpe_workers=plan.parallel_tpe_workers,
            locked_param_space=narrowed_space_effective,
            signal_cache_dir=signal_cache_dir
        )
        _, study = _run_tf_optimization(t, ctx)
        best_results[t] = study
    
    if progress_queue:
        progress_queue.put(None)
        manager.shutdown()

    # Reporting & Saving
    from src.core.utils.secure_config import encrypt_config, get_strategy_secret
    oos_cagr_target = float(OPT_FUTURES_CONFIG.get("FUTURES_MIN_CAGR_PCT", 30.0))
    oos_mdd_limit = float(OPT_FUTURES_CONFIG.get("FUTURES_MAX_MDD", 25.0))
    alpha_decay_floor = -50.0

    for (target, tf_eval), study in best_results.items():
        if not study or not study.trials:
            continue

        completed = [
            t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None
        ]
        if not completed:
            continue

        best_trial = _select_best_trial_from_shortlist(
            sorted(completed, key=lambda x: float(x.value if x.value is not None else -1e9),
                   reverse=True)[:50]
        )
        params = best_trial.params.copy()
        _leverage_default = str(OPT_FUTURES_CONFIG.get("FUTURES_DISCOVERY_LEVERAGE", 5))
        params.update({
            "TIMEFRAME": tf_eval,
            "LEVERAGE": int(os.getenv("FUTURES_DISCOVERY_LEVERAGE", _leverage_default)),
            "USE_COMPOUNDING": True
        })
        ua = best_trial.user_attrs
        target_symbols = list(target)

        # 1. IS Portfolio CAGR (for Alpha Decay baseline)
        is_holdout_maps = {}
        for s in target_symbols:
            is_holdout_maps[s] = dict(data_maps[s])
            is_holdout_maps[s][f"oos_start_idx_{tf_eval}"] = data_maps[s][f"is_start_idx_{tf_eval}"]

        port_is = run_oos_margin_shared_portfolio(
            target_symbols, tf_eval, params, is_holdout_maps,
            cache_root=FUTURES_CACHE_DIR, return_signal_dfs=False
        )
        is_portfolio_cagr = float(port_is.get("cagr_pct", 0.0))

        # 2. OOS Portfolio Run (Actual performance)
        oos_port = run_oos_margin_shared_portfolio(
            target_symbols, tf_eval, params, oos_data_maps,
            cache_root=FUTURES_CACHE_DIR, return_signal_dfs=True
        )
        oos_portfolio_cagr = float(oos_port["cagr_pct"])

        # 3. Symbol-wise OOS Evaluation & Funding Drag
        symbol_gate_rows: List[FuturesSymbolGateRow] = []
        trades_df = oos_port.get("trades_df", pd.DataFrame())
        
        total_funding_oos = float(trades_df["funding_fee"].sum()) if not trades_df.empty and "funding_fee" in trades_df.columns else 0.0
        total_gross_oos = float(abs(trades_df["pnl"]).sum()) if not trades_df.empty else 0.0

        for s_eval in target_symbols:
            oos_start = int(oos_data_maps[s_eval][f"oos_start_idx_{tf_eval}"])
            oos_end = len(oos_data_maps[s_eval][tf_eval])
            
            if not trades_df.empty:
                sym_trades = trades_df[trades_df["symbol"] == s_eval]
                sym_t_oos = len(sym_trades)
                sym_wr_oos = (sym_trades["pnl"] > 0).mean() * 100.0 if sym_t_oos > 0 else 0.0
                
                # C3: sym_pnl / FUTURES_INITIAL_BALANCE treats per-symbol PnL as if the symbol
                # received the FULL initial capital. This under-states CAGR vs a standalone run,
                # but correctly represents each symbol's *portfolio contribution CAGR*.
                # Per-symbol true CAGR requires per-symbol equity curves from the engine.
                sym_pnl = float(sym_trades["pnl"].sum())
                hours_per_bar = int(tf_eval.replace("h", "")) if tf_eval.endswith("h") else 4
                days_oos = ((oos_end - oos_start) * hours_per_bar) / 24.0
                _base = 1.0 + sym_pnl / FUTURES_INITIAL_BALANCE
                sym_ann_cagr = (
                    (_base ** (365.0 / max(days_oos, 1.0)) - 1.0) * 100.0
                    if sym_t_oos > 0 else 0.0
                )
                # C4: Per-symbol MDD calculation from symbol-specific trades
                s_pnl_arr = sym_trades["pnl"].to_numpy() - sym_trades["entry_fee"].to_numpy()
                if len(s_pnl_arr) > 0:
                    s_eq = np.cumsum(s_pnl_arr) + (FUTURES_INITIAL_BALANCE / len(target_symbols))
                    peak = np.maximum.accumulate(s_eq)
                    dd = (peak - s_eq) / np.maximum(peak, 1e-9)
                    sym_m_oos = float(dd.max() * 100.0)
                else:
                    sym_m_oos = 0.0
            else:
                sym_ann_cagr, sym_m_oos, sym_wr_oos, sym_t_oos = 0.0, 0.0, 0.0, 0

            symbol_gate_rows.append(FuturesSymbolGateRow(
                symbol=s_eval, net_cagr_pct=sym_ann_cagr, max_mdd_pct=float(sym_m_oos),
                win_rate_pct=float(sym_wr_oos), trade_count=int(sym_t_oos)
            ))

        # 4. PBO calculation
        pbo_val, rho_val, n_paths = 1.0, 0.0, 0  # Fail-Closed: Default to 100% PBO
        oos_log_tw = ua.get("cpcv_path_oos_log_tw")
        if isinstance(oos_log_tw, list) and oos_log_tw:
            ref_sym = target_symbols[0]
            is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf_eval}", 0))
            # C2: Must use the SAME eff_ref_len that generated cpcv_path_oos_log_tw.
            # Optimization used multi_alignment_info["eff_ref_len"] (min across symbols
            # after common-IS-start alignment). Using raw len(IS_df)-is_off would produce
            # a different block count → cpcv_paths length mismatch → PBO always returns 0.5.
            fallback_ref_len = len(data_maps[ref_sym][tf_eval]) - is_off
            ref_len = int(ua.get("gate1_eff_ref_len", fallback_ref_len))
            embargo = int(EMBARGO_BARS.get(tf_eval, 0))

            cpcv_bundle = build_cpcv_test_paths_with_fallback(ref_len, embargo=embargo)
            cpcv_paths, n_blocks, _ = cpcv_bundle
            n_paths = len(cpcv_paths)
            all_blocks = list_cpcv_block_ranges(ref_len, n_blocks, embargo)

            from src.domain.futures.opt_futures_utils.oos_evaluator import (
                run_cpcv_complement_evaluation,
            )
            pbo_val, rho_val = run_cpcv_complement_evaluation(
                params, target_symbols, tf_eval, data_maps, cpcv_paths, all_blocks,
                oos_path_scores=oos_log_tw, signal_disk_cache_root=FUTURES_CACHE_DIR,
                project_root=project_root
            )

        # 5. Metrics calculation
        alpha_decay_pct = 0.0
        if abs(is_portfolio_cagr) > 1e-6:
            alpha_decay_pct = ((oos_portfolio_cagr - is_portfolio_cagr) /
                               max(abs(is_portfolio_cagr), 1e-6)) * 100.0
        funding_drag_pct = (total_funding_oos / max(total_gross_oos, 1e-9)) * 100.0

        # 6. Hard Gates Check
        core_checks = [
            oos_portfolio_cagr >= oos_cagr_target,
            abs(float(oos_port["mdd_pct"])) <= oos_mdd_limit,  # H1: use config var
            float(oos_port["profit_factor"]) >= 1.35,
            alpha_decay_pct >= alpha_decay_floor,
            funding_drag_pct <= 15.0,
            pbo_val <= 0.45
        ]
        is_all_passed = all(core_checks)

        # Final Report
        report = run_futures_deployment_report(FuturesDeploymentReportInput(
            gate1_sqn=float(ua.get("gate1_sqn", 0)),
            gate1_path_sortino=float(ua.get("gate1_path_sortino", 0)),
            gate1_tail_ratio=float(ua.get("gate1_tail_ratio", 0)),
            gate1_p10_gmgr=float(ua.get("gate1_p10_gmgr", 0)),
            gate1_psr=float(ua.get("gate1_psr", 0)),
            gate1_dsr=float(ua.get("gate1_dsr", 0)),
            cpcv_mean_path_return_pct=float(ua.get("cpcv_mean_path_return_pct", 0)),
            cpcv_worst_segment_mdd_pct=float(ua.get("cpcv_worst_segment_mdd_pct", 0)),
            moic=float(oos_port["moic"]),
            initial_capital_usdt=FUTURES_INITIAL_BALANCE,
            oos_net_cagr_pct=oos_portfolio_cagr,
            oos_mdd_pct=float(oos_port["mdd_pct"]),
            hw_recovery_days=float(oos_port["hw_recovery_days"]),
            oos_ulcer_index=float(oos_port.get("ulcer_index", 0)),
            alpha_decay_pct=alpha_decay_pct,
            oos_cagr_target_pct=oos_cagr_target,
            oos_mdd_limit_pct=oos_mdd_limit,
            hw_recovery_max_days=180.0,
            alpha_decay_floor_pct=alpha_decay_floor,
            oos_cvar_pct=float(oos_port["cvar_pct"]),
            cvar_limit_pct=12.0,
            funding_drag_pct=funding_drag_pct,
            funding_drag_limit_pct=15.0,
            terminal_wealth_ratio=float(oos_port["terminal_wealth_ratio"]),
            tw_target=1.0 - (oos_mdd_limit / 100.0),
            oos_total_trades=int(oos_port["total_trades"]),
            oos_pf=float(oos_port["profit_factor"]),
            oos_long_pf=float(oos_port.get("long_pf", 0)),
            oos_short_pf=float(oos_port.get("short_pf", 0)),
            oos_short_win_rate_pct=float(oos_port.get("short_win_rate_pct", 0)),
            oos_ev_cost_ratio=float(oos_port.get("ev_cost_ratio", 0)),
            pf_target=1.35,
            oos_calmar=float(oos_port["calmar_ratio"]),
            calmar_target=1.2,
            oos_win_rate_pct=float(oos_port["win_rate_pct"]),
            oos_long_short_minority_pct=float(oos_port["oos_long_short_minority_pct"]),
            symbol_rows=symbol_gate_rows,
            loso_warning="",
            hard_passed=sum(core_checks),
            hard_total=len(core_checks),
            final_decision_go=is_all_passed,
            pbo=pbo_val,
            spearman_rho=rho_val,
            pbo_n_paths=n_paths,
            pbo_gate_passed=pbo_val <= 0.45,
            pbo_hard_gate=True,  # PBO is included in core_checks → treated as hard gate
            multi_window_passed=True,
            multi_window_summary="",
            regime_diagnostic_block="",
            oos_long_trades=int(oos_port["long_trades"]),
            oos_short_trades=int(oos_port["short_trades"]),
            funding_cost_total_usdt=total_funding_oos,
            gross_pnl_abs_usdt=total_gross_oos,
            # Add missing targets to satisfy mypy (aligned with go_nogo.py internal targets)
            sqn_target=1.6, path_sortino_target=1.2, tail_ratio_target=0.8,
            psr_target=0.4, dsr_target=0.2
        ))
        _logger.info("\n%s", report)

        growth_score = float(ua.get("growth_score", 0))
        should_save = (growth_score > 0 and is_all_passed) or int(args.trials) <= 2

        if should_save:
            res_dir = Path(project_root) / "results"
            res_dir.mkdir(parents=True, exist_ok=True)
            jp = res_dir / f"{BEST_PARAMS_FUTURES_JSON_STEM}.json"
            import json
            jp.write_text(json.dumps(params, indent=4))
            _logger.info(f"Saved: {jp}")
            sec = get_strategy_secret()
            if sec:
                jp.with_suffix(".enc").write_bytes(encrypt_config(params, sec))
        else:
            _logger.info(
                "JSON save skipped: criteria not met (growth_score / gates). "
                "growth_score=%.4f, gates_passed=%s",
                growth_score, is_all_passed
            )

if __name__ == "__main__":
    main()
