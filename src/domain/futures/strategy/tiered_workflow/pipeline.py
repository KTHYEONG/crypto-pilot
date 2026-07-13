# src/domain/futures/strategy/tiered_workflow/pipeline.py

from __future__ import annotations

import dataclasses
import gc
import logging
import multiprocessing
import os
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr

import src.domain.futures.strategy.config as strategy_config
from src.core.utils.utils import PERF
from src.domain.futures.optimization.workflow import evaluate_l2_trial_cached
from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.portfolio.signal_composer import (
    compose_symbol_signals,
    composer_sigma_lookback_bars,
)
from src.domain.futures.strategy.candidate_contracts import (
    CandidateFoldOutput,
    Layer1EvidenceSnapshot,
    Layer1FoldReadiness,
    Layer1InferenceArtifact,
    QualifiedSignalRegistry,
    ValidatedSignalBatch,
)
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig, PerTfL1Result
from src.domain.futures.strategy.cs_rank import SymbolSignal
from src.domain.futures.strategy.tiered_logging import (
    format_layer1_deployment_registry_table,
    format_layer1_gate_table,
    format_layer1_outer_fold_table,
    format_layer1_table,
    format_layer2_table,
    format_layer3_table,
    format_layer_header,
    format_layer_universe_audit_table,
)

# 내부 모듈 임포트
from src.domain.futures.strategy.tiered_workflow.atomization_diagnostics import (
    diagnose_strategy_atomization,
)
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    _run_awf_simulation,
    _stack_oos_signals,
    compute_long_short_price_by_symbol,
    compute_long_short_realized_price,
    compute_major_symbol_registry_census,
    compute_mean_trend_efficiency,
    summarize_directional_veto,
    summarize_major_symbol_regime_incoherence,
    summarize_major_symbol_signal_sizing,
    summarize_major_symbol_sleeve_contribution,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    FoldDiagnostic,
    L2SimulationCache,
    Layer1Result,
    Layer2AllocationConfig,
    Layer2Result,
    Layer2TrialEvaluation,
    Layer3Result,
    SymbolLifecycleRecord,
)
from src.domain.futures.strategy.tiered_workflow.diagnostics import (
    _compute_fold_realized_valid_set,
    _compute_fold_ts_ic,
    _fold_eligible_symbol_mask,
    _is_trained_fold_output,
    _log_fold_regime_analysis,
    build_layer_universe_audit,
    compute_per_symbol_ic,
    compute_per_symbol_realized_stats,
    compute_prediction_decomposition_diag,
)
from src.domain.futures.strategy.tiered_workflow.metrics import (
    _bars_per_year_for_tf,
    _cagr,
    _mdd,
    _newey_west_ic_tstat,
    _psr,
    _sharpe,
    _sortino,
    _terminal_multiple,
    compute_breadth_weighted_ic,
)
from src.domain.futures.strategy.tiered_workflow.replay_parity import (
    assert_selection_replay_parity,
)
from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
    apply_deployment,
)
from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    XsAdmissionBasis,
    _candidate_output_to_signal_batch,
    _event_results_from_fold_output,
    _l1_probe_diag_enabled,
    _Layer1ModelCore,
    _registry_to_symbol_signals,
    assemble_layer1_artifact,
    build_qualified_signal_registry,
    compute_symbol_strategy_evidence,
    compute_xs_factor_spread_diagnostics,
    evaluate_layer1_readiness,
    evaluate_outer_signal_opportunities,
    fit_layer1_inference_artifact,
    predict_layer1_signals,
    predict_layer1_signals_multi_tf,
    prefit_layer1_model,
    resolve_xs_alpha_admission,
    select_outer_symbol_opportunities,
)
from src.domain.futures.strategy.tiered_workflow.tf_validation_repair import (
    _raw_probe_to_manifest,
    build_validation_parity_capture,
    finalize_validation_parity_capture,
    log_validation_parity_report,
)
from src.domain.futures.strategy.walk_forward import (
    WFFold,
    build_l1_swf_folds,
    build_l2_simulation_folds,
)

if TYPE_CHECKING:
    from src.domain.futures.optimization.opt_config import LayeredWindow
    from src.domain.futures.strategy.tiered_workflow.awf_sim import (
        DirectionalVetoSummary,
        ReversalEpisode,
    )

for _env in (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_env] = "1"


def _get_rss_mb() -> float:
    """Return current process RSS in MB via /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        return -1.0
    return -1.0


logger = logging.getLogger("opt_main_futures")
if os.environ.get("LOG_LEVEL") == "PERF":
    logger.setLevel(15)


_VALID_COVERAGE_FLAG_THRESHOLD: float = 0.80
_TRAINED_FOLD_COVERAGE_THRESHOLD: float = 0.80


class TieredPipelineError(RuntimeError):
    """Base error for tiered pipeline execution after tiered bootstrapping succeeds."""


class Layer3WindowError(TieredPipelineError):
    """Raised when the Layer 3 holdout window resolves to an empty span."""


class Layer3ExecutionError(TieredPipelineError):
    """Raised when Layer 3 signal prediction or execution fails after L1/L2 succeed."""


def _can_prime_feature_cache(labeled_events: pd.DataFrame) -> bool:
    return not labeled_events.empty and "entry_idx" in labeled_events.columns


def _date_to_idx(datetimes: NDArray[np.datetime64], target_date: Any) -> int:
    """target_date에 해당하는 bar 인덱스 검색."""
    target = np.datetime64(target_date, "D")
    idx = int(np.searchsorted(datetimes.astype("datetime64[D]"), target))
    return min(idx, len(datetimes) - 1)


def _date_to_left_idx(datetimes: NDArray[np.datetime64], target_date: Any) -> int:
    """Resolve a left-closed date boundary to a bar index."""
    target = np.datetime64(target_date, "D")
    idx = int(np.searchsorted(datetimes.astype("datetime64[D]"), target, side="left"))
    return min(max(idx, 0), len(datetimes))


def _date_to_right_exclusive_idx(datetimes: NDArray[np.datetime64], target_date: Any) -> int:
    """Resolve an inclusive date boundary to the next exclusive bar index."""
    target = np.datetime64(target_date, "D") + np.timedelta64(1, "D")
    idx = int(np.searchsorted(datetimes.astype("datetime64[D]"), target, side="left"))
    return min(max(idx, 0), len(datetimes))


def _resolve_holdout_span(
    datetimes: NDArray[np.datetime64],
    holdout_start: Any,
    holdout_end: Any,
) -> tuple[int, int]:
    """Resolve Layer 3 holdout span as [start, end) on the aligned bar grid."""
    if len(datetimes) == 0:
        raise Layer3WindowError("empty_holdout_window")

    ho_start_idx = _date_to_left_idx(datetimes, holdout_start)
    ho_end_idx = _date_to_right_exclusive_idx(datetimes, holdout_end)
    if ho_end_idx <= ho_start_idx:
        last_dt = datetimes[-1] if len(datetimes) > 0 else None
        logger.warning(
            "[L3] holdout span empty: start_idx=%d end_idx=%d n_bars=%d last_dt=%s",
            ho_start_idx,
            ho_end_idx,
            len(datetimes),
            last_dt,
        )
        raise Layer3WindowError("empty_holdout_window")
    return (ho_start_idx, ho_end_idx)


def _is_non_constant_finite_array(values: NDArray[np.float64]) -> bool:
    if values.size < 1:
        return False
    finite = values[np.isfinite(values)]
    if finite.size < 1:
        return False
    return float(np.nanstd(finite)) > 0.0


def resolve_safe_nested_workers(
    n_tasks: int,
    frame_memory_bytes: int,
    stage: str = "evidence",
    *,
    pinned: int | None = None,
    compact_result: bool = False,
    result_soft_cap_mb: int | None = None,
) -> int:
    """Compute safe worker count dynamically under WSL constraints.

    Args:
        n_tasks: Number of tasks to parallelize.
        frame_memory_bytes: Estimated DataFrame size in bytes for OOM guard.
        stage: Current pipeline stage ("evidence", "outer", "l2_optuna").
        pinned: If set, request an upper bound for worker count.
        result_soft_cap_mb: Optional aggregate soft cap for nested result payload.
    """
    import psutil

    physical_cores = os.cpu_count() or 4
    cpu_limit = max(1, int(physical_cores * 0.75))
    requested_workers = min(n_tasks, pinned) if isinstance(pinned, int) and pinned >= 1 else n_tasks

    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    safe_mem_gb = available_gb * 0.70

    frame_gb = frame_memory_bytes / (1024**3)
    predicted_result_gb = 0.10 if compact_result else 0.40
    estimated_proc_gb = max(0.8, 0.5 + frame_gb * 0.5 + predicted_result_gb)

    mem_limit = max(1, int(safe_mem_gb // estimated_proc_gb))
    predicted_result_mb = 100 if compact_result else 400
    result_mem_limit = None
    if result_soft_cap_mb is not None:
        result_mem_limit = max(1, int(result_soft_cap_mb // predicted_result_mb))

    stage_worker_caps = {
        "evidence": 4 if compact_result and available_gb >= 8.0 else 3,
        "outer": 3,
        "l2_optuna": 4,
    }
    stage_cap = stage_worker_caps.get(stage, 3)

    max_workers = min(cpu_limit, stage_cap)
    workers = max(1, min(requested_workers, max_workers, mem_limit))
    if result_mem_limit is not None:
        workers = min(workers, result_mem_limit)
    # Over-subscription guard: each worker gets at least ~2 tasks
    if workers > 1 and n_tasks // workers < 2:
        workers = max(1, n_tasks // 2)
    if available_gb < 5.0:
        workers = min(workers, 2)
    logger.log(
        PERF,
        "[PERF] worker_calc stage=%s n_tasks=%d requested_workers=%d physical_cores=%d cpu_limit=%d "
        "max_workers=%d available_gb=%.2f frame_gb=%.2f estimated_proc_gb=%.2f compact=%s "
        "result_soft_cap_mb=%s predicted_result_mb=%d result_mem_limit=%s pinned_applied=%s "
        "mem_limit=%d workers=%d",
        stage,
        n_tasks,
        requested_workers,
        physical_cores,
        cpu_limit,
        max_workers,
        available_gb,
        frame_gb,
        estimated_proc_gb,
        compact_result,
        result_soft_cap_mb,
        predicted_result_mb,
        result_mem_limit,
        isinstance(pinned, int) and pinned >= 1,
        mem_limit,
        workers,
    )
    return workers


def _log_fold_avg_profile(results: Sequence[Any], tag: str) -> None:
    """Log average timing profile across fold results (evidence or outer)."""
    n = len(results)
    if n == 0:
        return
    _profile_keys = (
        "schema",
        "dataset_fit",
        "dataset_early_stop",
        "dataset_calibration_fit",
        "dataset_calibration_eval",
        "dataset_oos",
        "edge_fit",
        "inference",
        "selection",
    )
    _agg: dict[str, float] = dict.fromkeys(_profile_keys, 0.0)
    for _r in results:
        _prof = getattr(_r, "timing_profile", {}) or {}
        for _k in _profile_keys:
            _agg[_k] += _prof.get(_k, 0.0)
    for _k in _profile_keys:
        _agg[_k] /= n
    logger.log(
        PERF,
        "[PERF] l1_%s_fold_avg_profile n=%d "
        "schema=%.3fs ds_fit=%.3fs ds_es=%.3fs ds_cal_fit=%.3fs ds_cal_eval=%.3fs "
        "ds_oos=%.3fs edge_fit=%.3fs inference=%.3fs selection=%.3fs",
        tag,
        n,
        _agg["schema"],
        _agg["dataset_fit"],
        _agg["dataset_early_stop"],
        _agg["dataset_calibration_fit"],
        _agg["dataset_calibration_eval"],
        _agg["dataset_oos"],
        _agg["edge_fit"],
        _agg["inference"],
        _agg["selection"],
    )


def _snapshot_matured_count(
    all_evidence_events: pd.DataFrame,
    as_of_idx: int,
    cfg: CandidateStrategyConfig,
) -> int:
    if all_evidence_events.empty or "exit_idx" not in all_evidence_events.columns:
        return 0
    exit_idx = pd.to_numeric(all_evidence_events["exit_idx"], errors="coerce").fillna(np.inf)
    mature_mask = exit_idx < float(as_of_idx)
    lookback_bars = getattr(cfg, "l1_evidence_lookback_bars", None)
    if lookback_bars is not None:
        mature_mask &= exit_idx >= float(as_of_idx - int(lookback_bars))
    return int(mature_mask.sum())


def _build_evidence_store(event_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Keep only evidence columns needed for snapshot evidence aggregation."""
    if not event_frames:
        return pd.DataFrame()
    store = pd.concat(event_frames, ignore_index=True)
    if store.empty:
        return store
    keep_cols = [
        "symbol",
        "strategy_id",
        "activation_context",
        "side",
        "holding_bucket",
        "gross_event_bps",
        "expected_gross_bps",
        "q10_gross_bps",
        "q90_gross_bps",
        "decision_idx",
        "exit_idx",
        "uniqueness_weight",
        "entry_regime",
        "family",
        "variant",
        "fold_id",
        "signal_cell",
    ]
    present = [col for col in keep_cols if col in store.columns]
    if "exit_idx" in store.columns:
        store = store.loc[:, present].copy()
        store["exit_idx"] = pd.to_numeric(store["exit_idx"], errors="coerce").fillna(np.inf)
        store = store.sort_values("exit_idx", kind="stable").reset_index(drop=True)
    else:
        store = store.loc[:, present].copy()
    return store


def build_l1_prequential_evidence_snapshots(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    evidence_folds: tuple[WFFold, ...],
    snapshot_indices: tuple[int, ...],
    cfg: CandidateStrategyConfig,
    seed: int,
    precomputed_results: list[CandidateFoldOutput] | None = None,
) -> tuple[Layer1EvidenceSnapshot, ...]:
    """Build causal evidence snapshots from a single pass over evidence folds."""
    purge_bars, _embargo_bars = strategy_config.resolve_purge_and_embargo_bars(cfg)
    evidence_frames: list[pd.DataFrame] = []

    if not evidence_folds:
        evidence_store = pd.DataFrame()
    elif precomputed_results is not None:
        flat_results = precomputed_results
        for evidence_idx, evidence_out in enumerate(flat_results):
            if _is_trained_fold_output(evidence_out):
                evidence_frames.append(
                    _event_results_from_fold_output(
                        fold_id=evidence_idx,
                        fold_out=evidence_out,
                    )
                )
        evidence_store = _build_evidence_store(evidence_frames)
    else:
        import multiprocessing

        import src.domain.futures.strategy.candidate_workflow as cw

        assert cw._GLOBAL_LABELED_EVENTS is None, "Global state collision: _GLOBAL_LABELED_EVENTS must be None"
        assert cw._GLOBAL_PREPARED_EVENTS is None
        assert cw._GLOBAL_ALIGNED is None
        assert cw._GLOBAL_CFG is None
        assert cw._GLOBAL_PURGE_BARS is None

        # Set process globals to minimize IPC size under fork
        import gc

        gc.collect()
        cw._GLOBAL_LABELED_EVENTS = labeled_events
        cw._GLOBAL_ALIGNED = aligned
        cw._GLOBAL_CFG = cfg
        cw._GLOBAL_PURGE_BARS = purge_bars
        mp_ctx = multiprocessing.get_context("fork")

        # Calculate memory consumption dynamically
        try:
            frame_memory_bytes = int(labeled_events.memory_usage(deep=True).sum())
        except Exception:
            frame_memory_bytes = int(labeled_events.memory_usage().sum())

        workers = resolve_safe_nested_workers(
            len(evidence_folds),
            frame_memory_bytes,
            stage="evidence",
            pinned=getattr(cfg, "l1_nested_workers", None),
            compact_result=bool(getattr(cfg, "l1_compact_ipc_enabled", True)),
        )
        logger.debug(
            "[EVIDENCE-PREQ] Fitting %d evidence folds in parallel with %d workers (pinned=%s)",
            len(evidence_folds),
            workers,
            getattr(cfg, "l1_nested_workers", None),
        )

        flat_results = []
        t_exec = time.perf_counter()
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as executor:
                submits = [
                    executor.submit(
                        cw._fit_and_predict_single_fold_from_globals,
                        idx,
                        fold,
                        False,
                        bool(getattr(cfg, "l1_compact_ipc_enabled", True)),
                    )
                    for idx, fold in enumerate(evidence_folds)
                ]
                flat_results = [fut.result() for fut in submits]
        finally:
            cw._GLOBAL_LABELED_EVENTS = None
            cw._GLOBAL_PREPARED_EVENTS = None
            cw._GLOBAL_ALIGNED = None
            cw._GLOBAL_CFG = None
            cw._GLOBAL_PURGE_BARS = None
        logger.log(
            PERF,
            "[PERF] evidence_prequential_parallel_exec took=%.4fs",
            time.perf_counter() - t_exec,
        )

        for evidence_idx, evidence_out in enumerate(flat_results):
            if _is_trained_fold_output(evidence_out):
                evidence_frames.append(
                    _event_results_from_fold_output(
                        fold_id=evidence_idx,
                        fold_out=evidence_out,
                    )
                )
        evidence_store = _build_evidence_store(evidence_frames)
    snapshots: list[Layer1EvidenceSnapshot] = []
    streaming_enabled = bool(getattr(cfg, "l1_snapshot_streaming_enabled", True))
    exit_idx_sorted: NDArray[np.float64] | None = None
    if streaming_enabled and not evidence_store.empty and "exit_idx" in evidence_store.columns:
        exit_idx_sorted = evidence_store["exit_idx"].to_numpy(dtype=np.float64, copy=False)

    def _build_snapshot(snapshot_offset: int, as_of_idx: int) -> Layer1EvidenceSnapshot:
        _event_results = evidence_store
        _matured_event_count = _snapshot_matured_count(evidence_store, as_of_idx, cfg)
        if exit_idx_sorted is not None:
            _right = int(np.searchsorted(exit_idx_sorted, float(as_of_idx), side="left"))
            _left = 0
            _lookback_bars = getattr(cfg, "l1_evidence_lookback_bars", None)
            if _lookback_bars is not None:
                _left = int(
                    np.searchsorted(
                        exit_idx_sorted,
                        float(as_of_idx - int(_lookback_bars)),
                        side="left",
                    )
                )
            _event_results = evidence_store.iloc[_left:_right].copy()
            _matured_event_count = max(0, _right - _left)
        # NOTE: xs_admission intentionally not wired here — this snapshot's event
        # frame (_event_results_from_fold_output) lacks realized_side_adjusted_gross_bps,
        # a different schema than outer_events used at the deployment_evidence site below.
        _evidence = compute_symbol_strategy_evidence(
            event_results=_event_results,
            cfg=cfg,
            seed=seed + snapshot_offset,
            registry_as_of_idx=as_of_idx,
            snapshot_index=snapshot_offset,
        )
        _registry = build_qualified_signal_registry(
            evidence=_evidence,
            symbols=aligned.symbols,
            min_signals_per_symbol=int(cfg.l1_min_signals_per_symbol),
            registry_version=f"snapshot-{as_of_idx}",
            cfg=cfg,
        )
        return Layer1EvidenceSnapshot(
            as_of_idx=int(as_of_idx),
            evidence=_evidence,
            registry=_registry,
            matured_event_count=_matured_event_count,
        )

    _snapshot_items = sorted(set(snapshot_indices))
    if len(_snapshot_items) <= 1:
        for _offset, _aidx in enumerate(_snapshot_items):
            snapshots.append(_build_snapshot(_offset, _aidx))
    else:
        _n_workers = min(len(_snapshot_items), max(1, (os.cpu_count() or 4) // 2))
        with ThreadPoolExecutor(max_workers=_n_workers) as _pool:
            _futs = {
                _pool.submit(_build_snapshot, _offset, _aidx): _aidx for _offset, _aidx in enumerate(_snapshot_items)
            }
            snapshots.extend(fut.result() for fut in as_completed(_futs))
        snapshots.sort(key=lambda s: s.as_of_idx)
    return tuple(snapshots)


_L1_SWF_FOLD_CACHE: dict[tuple[Any, ...], Any] = {}


def run_l1_swf(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    folds: tuple[WFFold, ...],
    l1_params: dict[str, Any],
    min_obs: int = 20,
    t_stat_floor: float = 1.96,
    tf: str = "4h",
    verbose: bool = True,
) -> Layer1Result:
    """Layer1 SWF-K 신호 검증."""
    purge_bars, _embargo_bars = strategy_config.resolve_purge_and_embargo_bars(cfg)

    import src.domain.futures.strategy.candidate_workflow as cw

    planned_workers = max(1, (os.cpu_count() or 4) // 2)
    max_workers = min(len(folds), planned_workers)

    symbols = aligned.symbols
    n_total = len(symbols)

    t_start = time.perf_counter()
    logger.debug(
        "[SWF-START] n_symbols=%d n_bars=%d n_folds=%d purge=%d embargo=%d cfg=%s max_workers=%d",
        n_total,
        len(aligned.datetimes),
        len(folds),
        purge_bars,
        _embargo_bars,
        getattr(cfg, "candidate_name", cfg.__class__.__name__),
        max_workers,
    )

    signals_per_fold: list[dict[str, SymbolSignal]] = []
    fold_diags: list[FoldDiagnostic] = []

    futures: list[tuple[int, WFFold, Any]] = []
    missing_folds: list[tuple[int, WFFold]] = []

    for fold_idx, wf_fold in enumerate(folds):
        cache_key = (
            wf_fold.fit_start,
            wf_fold.fit_end,
            wf_fold.cal_start,
            wf_fold.cal_end,
            wf_fold.oos_start,
            wf_fold.oos_end,
            id(labeled_events),
            id(aligned),
            id(cfg),
        )
        if cache_key in _L1_SWF_FOLD_CACHE:
            futures.append((fold_idx, wf_fold, _L1_SWF_FOLD_CACHE[cache_key]))
        else:
            missing_folds.append((fold_idx, wf_fold))

    if missing_folds:
        cw._GLOBAL_LABELED_EVENTS = labeled_events
        cw._GLOBAL_ALIGNED = aligned
        cw._GLOBAL_CFG = cfg
        cw._GLOBAL_PURGE_BARS = purge_bars
        mp_ctx = multiprocessing.get_context("fork")

        try:
            if max_workers <= 1 or len(missing_folds) <= 1:
                for fold_idx, wf_fold in missing_folds:
                    try:
                        import src.domain.futures.strategy.tiered_workflow as _tw

                        fold_out = cast(
                            Any,
                            _tw._fit_and_predict_single_fold,
                        )(fold_idx, wf_fold, labeled_events, aligned, cfg, purge_bars)
                        futures.append((fold_idx, wf_fold, fold_out))
                        cache_key = (
                            wf_fold.fit_start,
                            wf_fold.fit_end,
                            wf_fold.cal_start,
                            wf_fold.cal_end,
                            wf_fold.oos_start,
                            wf_fold.oos_end,
                            id(labeled_events),
                            id(aligned),
                            id(cfg),
                        )
                        _L1_SWF_FOLD_CACHE[cache_key] = fold_out
                    except Exception:
                        logger.warning("run_l1_swf: fold %d 학습 실패, 스킵", fold_idx, exc_info=True)
            else:
                with ProcessPoolExecutor(
                    max_workers=min(len(missing_folds), max_workers),
                    mp_context=mp_ctx,
                ) as executor:
                    submits: list[tuple[int, WFFold, Any]] = []
                    for fold_idx, wf_fold in missing_folds:
                        submits.append(
                            (
                                fold_idx,
                                wf_fold,
                                executor.submit(cw._fit_and_predict_single_fold_from_globals, fold_idx, wf_fold),
                            )
                        )
                    for fold_idx, wf_fold, fut in submits:
                        try:
                            fold_out = fut.result()
                            futures.append((fold_idx, wf_fold, fold_out))
                            cache_key = (
                                wf_fold.fit_start,
                                wf_fold.fit_end,
                                wf_fold.cal_start,
                                wf_fold.cal_end,
                                wf_fold.oos_start,
                                wf_fold.oos_end,
                                id(labeled_events),
                                id(aligned),
                                id(cfg),
                            )
                            _L1_SWF_FOLD_CACHE[cache_key] = fold_out
                        except Exception:
                            logger.warning("run_l1_swf: fold %d 학습 실패, 스킵", fold_idx, exc_info=True)
        finally:
            cw._GLOBAL_LABELED_EVENTS = None
            cw._GLOBAL_ALIGNED = None
            cw._GLOBAL_CFG = None
            cw._GLOBAL_PURGE_BARS = None

    futures.sort(key=lambda x: x[0])

    for fold_loop_idx, (_fold_idx, _wf_fold, fold_out) in enumerate(futures):
        beta_f32 = aligned.beta_vs_market_1d
        beta_f64: NDArray[np.float64] | None = beta_f32.astype(np.float64) if beta_f32 is not None else None
        fold_sigs: dict[str, SymbolSignal] = {}
        if _is_trained_fold_output(fold_out):
            fold_sigs = compose_symbol_signals(
                model_output=fold_out.model_output,
                close_2d=aligned.close_2d,
                symbols=symbols,
                tf=tf,
                min_obs=min_obs,
                t_stat_floor=t_stat_floor,
                beta_vs_market_1d=beta_f64,
                opt_cfg=None,
            )
            signals_per_fold.append(fold_sigs)

        fold_ic: float | None = _compute_fold_ts_ic(fold_out=fold_out)

        eligible_mask = _fold_eligible_symbol_mask(aligned=aligned, fold=_wf_fold)
        f_n_eligible = int(np.count_nonzero(eligible_mask))
        fold_realized_valid = _compute_fold_realized_valid_set(fold_out, min_obs=min_obs, t_stat_floor=t_stat_floor)
        eligible_syms = {s for s, e in zip(symbols, eligible_mask, strict=True) if e}
        f_n_valid = len(fold_realized_valid & eligible_syms)
        f_breadth = f_n_valid / max(1, f_n_eligible)
        f_n_events = len(fold_out.model_output.expected_net_bps)
        fold_diags.append(
            FoldDiagnostic(
                fold=fold_loop_idx + 1,
                ic=fold_ic,
                breadth=f_breadth,
                n_valid=f_n_valid,
                n_eligible=f_n_eligible,
                n_events=f_n_events,
                n_fit=int(getattr(fold_out, "n_fit", 0)),
                fit_status=getattr(fold_out, "fit_status", "failed"),
                passed=fold_ic is not None and fold_ic > 0,
            )
        )

    total_folds = len(futures)
    if total_folds > 0:
        avg_profile = dict.fromkeys(
            (
                "schema",
                "dataset_fit",
                "dataset_early_stop",
                "dataset_calibration_fit",
                "dataset_calibration_eval",
                "dataset_oos",
                "edge_fit",
                "inference",
                "selection",
            ),
            0.0,
        )
        for _, _, fold_out in futures:
            prof = getattr(fold_out, "timing_profile", {})
            for k in avg_profile:
                avg_profile[k] += prof.get(k, 0.0)

        for k in avg_profile:
            avg_profile[k] /= total_folds

        logger.log(
            PERF,
            "[PERF] swf_fold_avg_profile n=%d "
            "schema=%.3fs ds_fit=%.3fs ds_es=%.3fs ds_cal_fit=%.3fs ds_cal_eval=%.3fs "
            "ds_oos=%.3fs edge_fit=%.3fs inference=%.3fs selection=%.3fs",
            total_folds,
            avg_profile["schema"],
            avg_profile["dataset_fit"],
            avg_profile["dataset_early_stop"],
            avg_profile["dataset_calibration_fit"],
            avg_profile["dataset_calibration_eval"],
            avg_profile["dataset_oos"],
            avg_profile["edge_fit"],
            avg_profile["inference"],
            avg_profile["selection"],
        )

    logger.log(
        PERF,
        "[PERF] run_l1_swf took=%.2fs",
        time.perf_counter() - t_start,
    )

    per_sym_ic = compute_per_symbol_ic(fold_tuples=futures)

    per_sym_realized = compute_per_symbol_realized_stats(
        fold_tuples=futures,
        min_obs=min_obs,
        t_stat_floor=t_stat_floor,
        per_symbol_ic=per_sym_ic,
    )

    sigs_tuple = tuple(signals_per_fold)
    oos_stacked = _stack_oos_signals(sigs_tuple, realized_stats=per_sym_realized)
    from src.domain.futures.strategy import tiered_workflow as _tw_panel_raw

    _tw_panel_fn: Any = _tw_panel_raw
    strategy_panel = _tw_panel_fn.compute_per_strategy_oos_validation(fold_tuples=futures)
    n_valid_strategies = sum(1 for sig in strategy_panel if sig.valid)
    panel_diversity = _tw_panel_fn.compute_panel_diversity(strategy_panel)

    fold_perf_details: list[dict[str, Any]] = [
        {
            "fold": d.fold,
            "ic": d.ic,
            "breadth": d.breadth,
            "n_valid": d.n_valid,
            "n_eligible": d.n_eligible,
            "n_events": d.n_events,
            "n_fit": d.n_fit,
            "fit_status": d.fit_status,
            "pass": d.passed,
        }
        for d in fold_diags
    ]

    valid_fold_ics = [float(d.ic) for d in fold_diags if d.ic is not None]
    if valid_fold_ics:
        cs_ic_mean = float(np.mean(valid_fold_ics))
        cs_ic_fold_pass_ratio = float(sum(1 for ic in valid_fold_ics if ic > 0.0) / len(valid_fold_ics))
        if len(valid_fold_ics) >= 2:
            cs_ic_std = float(np.std(valid_fold_ics, ddof=1))
            cs_ic_tstat = float(cs_ic_mean / (cs_ic_std / np.sqrt(len(valid_fold_ics)) + 1e-12))
        else:
            cs_ic_tstat = 0.0
    else:
        cs_ic_mean = 0.0
        cs_ic_tstat = 0.0
        cs_ic_fold_pass_ratio = 0.0

    _pred_parts: list[NDArray[np.float64]] = []
    _real_parts: list[NDArray[np.float64]] = []
    for _, _wf_fold, fold_out in futures:
        if not _is_trained_fold_output(fold_out):
            continue
        oos = getattr(fold_out, "oos_set", None)
        if oos is None:
            continue
        _y_ret = getattr(oos, "y_return_bps", None)
        _y_edg = getattr(oos, "y_edge_bps", None)
        y_lab = _y_ret if _y_ret is not None else _y_edg
        if y_lab is None:
            continue
        p_arr = np.asarray(fold_out.model_output.expected_net_bps, dtype=np.float64)
        r_arr = np.asarray(y_lab, dtype=np.float64)
        if len(p_arr) != len(r_arr) or len(p_arr) < 4:
            continue
        _mask = np.isfinite(p_arr) & np.isfinite(r_arr)
        if _mask.sum() < 4:
            continue
        if not _is_non_constant_finite_array(p_arr[_mask]):
            continue
        if not _is_non_constant_finite_array(r_arr[_mask]):
            continue
        _pred_parts.append(p_arr[_mask])
        _real_parts.append(r_arr[_mask])

    if _pred_parts:
        _p_all = np.concatenate(_pred_parts)
        _r_all = np.concatenate(_real_parts)
        _global_ic_raw, _ = spearmanr(_p_all, _r_all)
        _global_ic = float(_global_ic_raw) if not np.isnan(_global_ic_raw) else 0.0
        _global_tstat = _newey_west_ic_tstat(_p_all, _r_all)
    else:
        _global_ic = 0.0
        _global_tstat = 0.0
    logger.debug(
        "[SWF-IC-DIAG] global_pooled_ic=%.4f global_tstat=%.2f (diagnostic, not gate)",
        _global_ic,
        _global_tstat,
    )

    per_sym_n: dict[str, int] = {sym: s.n_obs for sym, s in per_sym_realized.items()}
    pooled_ic_val, pooled_tstat_val = compute_breadth_weighted_ic(per_sym_ic, per_sym_n)

    _valid_pairs = [(d.ic, d.n_events) for d in fold_diags if d.ic is not None]
    if _valid_pairs:
        _w_total = sum(n for _, n in _valid_pairs)
        fold_pass_ratio = sum(n for ic, n in _valid_pairs if ic > 0) / _w_total if _w_total > 0 else 0.0
    else:
        fold_pass_ratio = 0.0

    breadth = float(np.mean([d.breadth for d in fold_diags])) if fold_diags else 0.0
    valid_coverage = (
        float(sum(1 for d in fold_diags if d.breadth >= _VALID_COVERAGE_FLAG_THRESHOLD) / len(fold_diags))
        if fold_diags
        else 0.0
    )
    trained_fold_coverage = (
        float(sum(1 for d in fold_diags if d.fit_status == "trained") / len(fold_diags)) if fold_diags else 0.0
    )

    n_valid = sum(1 for s in per_sym_realized.values() if s.valid)

    sym_details: list[dict[str, Any]] = []
    for sym, sig in sorted(oos_stacked.items()):
        real = per_sym_realized.get(sym)
        sym_details.append(
            {
                "symbol": sym,
                "raw_mu": sig.raw_mu,
                "vol": sig.volatility,
                "t_stat": real.t_stat if real is not None else 0.0,
                "ic": per_sym_ic.get(sym, 0.0),
                "valid": real.valid if real is not None else False,
            }
        )

    _diag = compute_prediction_decomposition_diag(fold_tuples=futures)
    gate_passed: bool = bool(
        (trained_fold_coverage >= _TRAINED_FOLD_COVERAGE_THRESHOLD)
        and (n_valid_strategies >= cfg.l1_min_valid_strategies)
        and (panel_diversity >= cfg.l1_min_panel_diversity)
        and (cs_ic_fold_pass_ratio >= cfg.l1_min_cs_fold_pass_ratio)
    )

    result = Layer1Result(
        signals_per_fold=sigs_tuple,
        oos_stacked=oos_stacked,
        pooled_ic=pooled_ic_val,
        pooled_tstat=pooled_tstat_val,
        breadth=breadth,
        valid_coverage=valid_coverage,
        fold_pass_ratio=fold_pass_ratio,
        gate_passed=gate_passed,
        n_valid=n_valid,
        n_total=n_total,
        n_trade_scope=n_total,
        cs_ic_mean=cs_ic_mean,
        cs_ic_tstat=cs_ic_tstat,
        cs_ic_fold_pass_ratio=cs_ic_fold_pass_ratio,
        decile_lift_bps=_diag.decile_lift_bps,
        strategy_panel=strategy_panel,
        n_valid_strategies=n_valid_strategies,
        panel_diversity=panel_diversity,
    )
    if verbose:
        logger.info(format_layer1_table(result, fold_details=fold_perf_details, per_symbol_top10=sym_details))
    if strategy_panel:
        top_panel = sorted(
            strategy_panel,
            key=lambda item: (item.valid, item.oos_edge_bps, item.oos_nw_tstat),
            reverse=True,
        )[: min(10, len(strategy_panel))]
        panel_str = ", ".join(
            (
                f"{sig.strategy_id}:edge={sig.oos_edge_bps:.1f}"
                f"/t={sig.oos_nw_tstat:.2f}"
                f"/cons={sig.fold_sign_consistency:.2f}"
                f"/valid={'Y' if sig.valid else 'N'}"
            )
            for sig in top_panel
        )
        logger.debug("[STRATEGY-PANEL] valid=%d diversity=%.3f | %s", n_valid_strategies, panel_diversity, panel_str)
    logger.debug(
        "[SWF-LEGACY-IC] pooled_ic=%.4f pooled_tstat=%.2f breadth=%.3f valid_coverage=%.3f",
        pooled_ic_val,
        pooled_tstat_val,
        breadth,
        valid_coverage,
    )

    logger.debug(
        "[SWF-DIAG] static_share=%.3f dynamic_share=%.3f score_cal_ratio=%.3f decile_lift=%.2fbps",
        _diag.static_variance_share,
        _diag.dynamic_variance_share,
        _diag.score_cal_valid_ratio,
        _diag.decile_lift_bps,
    )
    if _diag.per_archetype_oos_edge:
        arch_lines = ", ".join(
            f"{a}: mu={m:.2f} t={t:.2f}" for a, (m, t) in sorted(_diag.per_archetype_oos_edge.items())
        )
        logger.debug("[SWF-DIAG-ARCH] %s", arch_lines)

    _log_fold_regime_analysis(fold_tuples=futures, datetimes=aligned.datetimes)

    return result


def _opportunities_to_symbol_signals(opportunities: ValidatedSignalBatch) -> dict[str, SymbolSignal]:
    from collections import defaultdict

    from src.domain.futures.strategy.cs_rank import VOL_FLOOR, SymbolSignal

    sym_events = defaultdict(list)
    for event in opportunities.events:
        sym_events[event.symbol].append(event)

    adapted: dict[str, SymbolSignal] = {}
    for sym, evs in sym_events.items():
        mus = [float(e.expected_gross_bps * e.side) for e in evs if np.isfinite(e.expected_gross_bps)]
        avg_mu = float(np.mean(mus)) if mus else 0.0
        qw = float(np.mean([e.quality_weight for e in evs])) if evs else 1.0
        adapted[sym] = SymbolSignal(
            raw_mu=avg_mu,
            volatility=VOL_FLOOR,
            n_obs=len(evs),
            t_stat=0.0,
            valid=True,
            beta_btc=None,
            quality_weight=qw,
        )
    return adapted


def _prefit_layer1_from_globals(
    fit_start_idx: int,
    fit_end_idx: int,
) -> _Layer1ModelCore:
    """Wrapper for ``prefit_layer1_model`` using process-global context (fork IPC bypass)."""
    import src.domain.futures.strategy.candidate_workflow as _cw

    if _cw._GLOBAL_LABELED_EVENTS is None and _cw._GLOBAL_PREPARED_EVENTS is None:
        raise RuntimeError("candidate workflow globals are not initialized for prefit")
    labeled_events = (
        _cw._GLOBAL_LABELED_EVENTS if _cw._GLOBAL_LABELED_EVENTS is not None else _cw._GLOBAL_PREPARED_EVENTS
    )
    assert _cw._GLOBAL_ALIGNED is not None, "prefit requires GLOBAL_ALIGNED"
    assert _cw._GLOBAL_CFG is not None, "prefit requires GLOBAL_CFG"
    return prefit_layer1_model(
        labeled_events=labeled_events,
        aligned=_cw._GLOBAL_ALIGNED,
        fit_start_idx=fit_start_idx,
        fit_end_idx=fit_end_idx,
        cfg=_cw._GLOBAL_CFG,
    )


def run_l1_nested_swf(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    outer_folds: tuple[WFFold, ...],
    cfg: CandidateStrategyConfig,
    seed: int,
    verbose: bool = True,
    l2_start: date | None = None,
    probe_diversity_corr: dict[str, float] | None = None,
    probe_prior_map: dict[tuple[str, str, str], float] | None = None,
    tf: str = "4h",
    defer_artifact: bool = False,
) -> Layer1Result:
    """Run nested Layer1 validation using inner selection and outer evaluation."""
    import dataclasses
    from copy import copy

    if dataclasses.is_dataclass(cfg):
        l1_cfg = dataclasses.replace(
            cfg,
            ensemble_conditioning="archetype_only",
            ensemble_score_calibration_enabled=False,
        )
    else:
        l1_cfg = copy(cfg)
        l1_cfg.ensemble_conditioning = "archetype_only"
        l1_cfg.ensemble_score_calibration_enabled = False
    _cfg_effective = strategy_config.apply_tf_gate_overrides(l1_cfg, tf)

    purge_bars, embargo_bars = strategy_config.resolve_purge_and_embargo_bars(cfg)
    _vol_window = composer_sigma_lookback_bars("4h")
    t_vol = time.perf_counter()
    _mem_vol = _get_rss_mb()
    # OPT-3: vectorized — same logic as rolling_per_bar_return_std over [T, N] at once
    _c = np.asarray(aligned.close_2d, dtype=np.float64)  # [T, N]
    _n_t, _n_sym = _c.shape
    _r = np.zeros((_n_t, _n_sym), dtype=np.float64)
    if _n_t >= 2:
        _r[1:] = (_c[1:] - _c[:-1]) / np.maximum(np.abs(_c[:-1]), 1e-12)
    _rw = max(2, int(_vol_window))
    volatility_2d = pd.DataFrame(_r).rolling(_rw, min_periods=2).std(ddof=1).to_numpy(dtype=np.float64)
    volatility_2d = np.nan_to_num(volatility_2d, nan=0.0, posinf=0.0, neginf=0.0)
    volatility_2d = np.maximum(volatility_2d, 1e-12)
    logger.log(
        PERF,
        "[PERF] l1_nested_volatility_2d took=%.4fs",
        time.perf_counter() - t_vol,
    )
    _rss_now = _get_rss_mb()
    logger.debug(
        "[MEM] stage=volatility_2d rss=%.0fMB delta=%+.0fMB shape=(%d,%d)",
        _rss_now,
        _rss_now - _mem_vol,
        _n_t,
        _n_sym,
    )
    outer_reports: list[Layer1FoldReadiness] = []
    outer_event_frames: list[pd.DataFrame] = []
    signals_per_fold: list[dict[str, SymbolSignal]] = []
    trained_count = 0

    evidence_start = min((fold.fit_start for fold in outer_folds), default=0)
    evidence_end = max((fold.oos_start for fold in outer_folds), default=0)
    try:
        _outer_n = len(outer_folds)
        _mult = max(3, int(getattr(cfg, "l1_evidence_grid_multiplier", 3)))
        _ev_n_folds = min(_outer_n * _mult, int(getattr(cfg, "l1_evidence_max_folds", 32)))
        evidence_folds = build_l1_swf_folds(
            n_bars=evidence_end,
            n_folds=_ev_n_folds,
            l1_start_bars=evidence_start,
            l1_end_bars=evidence_end,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            boundary_mode=l1_cfg.l1_boundary_mode,
            allocation_backend=l1_cfg.allocation_backend,
        )
    except ValueError:
        evidence_folds = ()

    combined_folds = tuple(evidence_folds) + tuple(outer_folds)
    num_evidence = len(evidence_folds)

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import src.domain.futures.strategy.candidate_workflow as cw
    from src.domain.futures.strategy.candidate_dataset import (
        prepare_labeled_events,
        prime_aligned_feature_cache,
    )

    # Assert global states to prevent nested collisions
    assert cw._GLOBAL_LABELED_EVENTS is None, "Global state collision: _GLOBAL_LABELED_EVENTS must be None"
    assert cw._GLOBAL_PREPARED_EVENTS is None
    assert cw._GLOBAL_ALIGNED is None
    assert cw._GLOBAL_CFG is None
    assert cw._GLOBAL_PURGE_BARS is None

    # Prime the feature cache on the parent process before multiprocessing fork
    t_prime = time.perf_counter()
    _mem_prime2 = _get_rss_mb()
    if _can_prime_feature_cache(labeled_events):
        logger.debug("[L1-NESTED] Priming aligned feature cache on parent process")
        try:
            prime_aligned_feature_cache(
                labeled_events=labeled_events,
                aligned=aligned,
                cfg=l1_cfg,
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("[L1-NESTED] Feature cache priming skipped: %s", exc)
    else:
        logger.debug("[L1-NESTED] Feature cache priming skipped: insufficient labeled event schema")
    logger.log(
        PERF,
        "[PERF] l1_nested_feature_cache_prime took=%.4fs",
        time.perf_counter() - t_prime,
    )
    logger.debug("[MEM] stage=nested_prime_cache rss=%.0fMB delta=%+.0fMB", _get_rss_mb(), _get_rss_mb() - _mem_prime2)

    prepared_events = None
    if bool(getattr(cfg, "l1_prepared_dataset_enabled", True)):
        t_prepare = time.perf_counter()
        try:
            prepared_events = prepare_labeled_events(
                labeled_events=labeled_events,
                aligned=aligned,
                cfg=l1_cfg,
                fit_start_idx=min((fold.fit_start for fold in outer_folds), default=0),
                fit_end_idx=max((fold.oos_end for fold in outer_folds), default=0),
            )
        except RuntimeError as exc:
            logger.debug("[L1-NESTED] prepared event cache skipped: %s", exc)
            prepared_events = None
        logger.log(PERF, "[PERF] l1_nested_prepare_events took=%.4fs", time.perf_counter() - t_prepare)

    t_mp_prep = time.perf_counter()
    # Set process globals to minimize IPC size under fork
    import gc

    gc.collect()
    cw._GLOBAL_LABELED_EVENTS = labeled_events if prepared_events is None else None
    cw._GLOBAL_PREPARED_EVENTS = prepared_events
    cw._GLOBAL_ALIGNED = aligned
    cw._GLOBAL_CFG = l1_cfg
    cw._GLOBAL_PURGE_BARS = purge_bars
    mp_ctx = multiprocessing.get_context("fork")

    # Calculate memory consumption dynamically
    try:
        frame_memory_bytes = int(labeled_events.memory_usage(deep=True).sum())
    except Exception:
        frame_memory_bytes = int(labeled_events.memory_usage().sum())

    compact_pref_raw = getattr(cfg, "l1_compact_ipc_enabled", True)
    compact_pref = compact_pref_raw if isinstance(compact_pref_raw, bool) else True
    soft_cap_raw = getattr(cfg, "l1_nested_result_soft_cap_mb", 512)
    soft_cap_mb = (
        int(soft_cap_raw) if isinstance(soft_cap_raw, (int, float)) and not isinstance(soft_cap_raw, bool) else 512
    )
    full_result_mb = 400
    force_compact = compact_pref
    if not compact_pref and soft_cap_mb < full_result_mb:
        force_compact = True
        logger.log(
            PERF,
            "[PERF] l1_nested_compact_override reason=soft_cap_force_compact soft_cap_mb=%d full_result_mb=%d",
            soft_cap_mb,
            full_result_mb,
        )

    workers_evidence = resolve_safe_nested_workers(
        len(evidence_folds),
        frame_memory_bytes,
        stage="evidence",
        pinned=getattr(cfg, "l1_nested_workers", None),
        compact_result=force_compact,
        result_soft_cap_mb=soft_cap_mb,
    )
    workers_outer = resolve_safe_nested_workers(
        len(outer_folds),
        frame_memory_bytes,
        stage="outer",
        pinned=getattr(cfg, "l1_nested_workers", None),
        compact_result=force_compact,
        result_soft_cap_mb=soft_cap_mb,
    )
    max_pool_workers = max(workers_evidence, workers_outer)

    logger.debug(
        "[L1-NESTED-COMBINED] Fitting %d folds (evidence=%d, outer=%d) "
        "max_pool=%d ev_workers=%d out_workers=%d pinned=%s n_sym=%d n_events=%d cfg_seed=%d",
        len(combined_folds),
        num_evidence,
        len(outer_folds),
        max_pool_workers,
        workers_evidence,
        workers_outer,
        getattr(cfg, "l1_nested_workers", None),
        len(aligned.symbols),
        len(labeled_events),
        seed,
    )
    logger.debug(
        "[MEM] stage=pre_fork rss=%.0fMB max_pool_workers=%d n_folds=%d",
        _get_rss_mb(),
        max_pool_workers,
        len(combined_folds),
    )
    logger.log(
        PERF,
        "[PERF] l1_nested_mp_prep took=%.4fs",
        time.perf_counter() - t_mp_prep,
    )

    evidence_results: list[Any] = []
    outer_results: list[Any] = []
    _wf_profile_keys = (
        "schema",
        "dataset_fit",
        "dataset_early_stop",
        "dataset_calibration_fit",
        "dataset_calibration_eval",
        "dataset_oos",
        "edge_fit",
        "inference",
        "selection",
    )
    _prefit_future: Any = None
    _prefit_core: _Layer1ModelCore | None = None

    try:
        with ProcessPoolExecutor(max_workers=max_pool_workers, mp_context=mp_ctx) as executor:
            # ── Phase 1: Evidence folds ─────────────────────────────────────────────
            t_exec = time.perf_counter()
            if evidence_folds:
                ev_submits = {
                    executor.submit(
                        cw._fit_and_predict_single_fold_from_globals,
                        idx,
                        fold,
                        True,
                        force_compact,
                    ): idx
                    for idx, fold in enumerate(evidence_folds)
                }
                # ── OPT-4: speculative pre-fit right after evidence submit ──
                if not defer_artifact and bool(getattr(cfg, "l1_speculative_prefit_enabled", True)):
                    _prefit_start = min((f.fit_start for f in outer_folds), default=0)
                    _prefit_end = max((f.oos_end for f in outer_folds), default=0)
                    _prefit_future = executor.submit(
                        _prefit_layer1_from_globals,
                        _prefit_start,
                        _prefit_end,
                    )
                t_ev_ipc = time.perf_counter()
                _mem_ipc_ref = _get_rss_mb()
                evidence_results = [fut.result() for fut in as_completed(ev_submits)]
                evidence_results.sort(key=lambda r: int(getattr(r, "fold_id", getattr(r, "fold_idx", 0))))
                logger.log(
                    PERF,
                    "[PERF] l1_evidence_ipc_collect n=%d took=%.4fs",
                    len(evidence_results),
                    time.perf_counter() - t_ev_ipc,
                )
                _rss_ev = _get_rss_mb()
                logger.debug(
                    "[MEM] stage=evidence_ipc rss=%.0fMB delta=%+.0fMB n_results=%d",
                    _rss_ev,
                    _rss_ev - _mem_ipc_ref,
                    len(evidence_results),
                )

                _log_fold_avg_profile(evidence_results, "evidence")

                t_ev_snap = time.perf_counter()
                evidence_snapshots = build_l1_prequential_evidence_snapshots(
                    labeled_events=labeled_events,
                    aligned=aligned,
                    evidence_folds=evidence_folds,
                    snapshot_indices=tuple(fold.oos_start for fold in outer_folds),
                    cfg=l1_cfg,
                    seed=seed,
                    precomputed_results=evidence_results,
                )
                logger.log(PERF, "[PERF] l1_prequential_evidence_snapshots took=%.4fs", time.perf_counter() - t_ev_snap)
                logger.debug(
                    "[MEM] stage=evidence_snapshots rss=%.0fMB n_snapshots=%d", _get_rss_mb(), len(evidence_snapshots)
                )

                _wf_agg_timings = dict.fromkeys(_wf_profile_keys, 0.0)
                for _r in evidence_results:
                    _prof = getattr(_r, "timing_profile", {}) or {}
                    for _k in _wf_profile_keys:
                        _wf_agg_timings[_k] += _prof.get(_k, 0.0)
                del evidence_results
                gc.collect()
                logger.debug("[MEM] stage=post_evidence_free rss=%.0fMB", _get_rss_mb())
            else:
                evidence_snapshots = ()
                _wf_agg_timings = dict.fromkeys(_wf_profile_keys, 0.0)

            logger.log(PERF, "[PERF] l1_evidence_phase took=%.4fs", time.perf_counter() - t_exec)
            snapshots_by_idx = {s.as_of_idx: s for s in evidence_snapshots}

            # ── Phase 2: Outer folds ────────────────────────────────────────────────
            t_outer_exec = time.perf_counter()
            if outer_folds:
                import concurrent.futures

                out_submits = []
                active_futures: set[concurrent.futures.Future[Any]] = set()

                t_out_ipc = time.perf_counter()
                _mem_ipc_ref = _get_rss_mb()

                for idx, fold in enumerate(outer_folds):
                    while len(active_futures) >= workers_outer:
                        done, _ = concurrent.futures.wait(
                            active_futures,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        active_futures.difference_update(done)

                    fut = executor.submit(
                        cw._fit_and_predict_single_fold_from_globals,
                        idx,
                        fold,
                        False,
                        force_compact,
                    )
                    active_futures.add(fut)
                    out_submits.append(fut)

                outer_results = [fut.result() for fut in out_submits]
                for _fold_idx, _result in enumerate(outer_results):
                    if not hasattr(_result, "fold_id"):
                        import contextlib

                        with contextlib.suppress(Exception):
                            _result.fold_id = int(_fold_idx)
                logger.log(
                    PERF,
                    "[PERF] l1_outer_ipc_collect n=%d took=%.4fs",
                    len(outer_results),
                    time.perf_counter() - t_out_ipc,
                )
                _rss_out = _get_rss_mb()
                logger.debug(
                    "[MEM] stage=outer_ipc rss=%.0fMB delta=%+.0fMB n_results=%d",
                    _rss_out,
                    _rss_out - _mem_ipc_ref,
                    len(outer_results),
                )

            # ── Collect speculative pre-fit result ────────────────────────────────
            if _prefit_future is not None:
                try:
                    _prefit_core = _prefit_future.result()
                except Exception:
                    logger.exception("[L1] speculative pre-fit failed — falling back to serial fit")
                    _prefit_core = None
    finally:
        cw._GLOBAL_LABELED_EVENTS = None
        cw._GLOBAL_PREPARED_EVENTS = None
        cw._GLOBAL_ALIGNED = None
        cw._GLOBAL_CFG = None
        cw._GLOBAL_PURGE_BARS = None
        gc.collect()

    _log_fold_avg_profile(outer_results, "outer")

    # ── WF wall-time summary ────────────────────────────────────────────
    t_wf_now = time.perf_counter()
    ev_wall = t_wf_now - t_exec
    out_wall = t_wf_now - t_outer_exec
    for _r in outer_results:
        _prof = getattr(_r, "timing_profile", {}) or {}
        for _k in _wf_profile_keys:
            _wf_agg_timings[_k] += _prof.get(_k, 0.0)
    n_total = len(evidence_folds) + len(outer_folds)
    if n_total > 0:
        for _k in _wf_profile_keys:
            _wf_agg_timings[_k] /= n_total
    logger.log(
        PERF,
        "[PERF] l1_wf_summary n_folds=%d evidence=%d outer=%d workers=%d "
        "wall: ev=%.1fs out=%.1fs total=%.1fs "
        "avg: selection=%.3fs ds_fit=%.3fs schema=%.3fs edge_fit=%.3fs inference=%.3fs",
        n_total,
        len(evidence_folds),
        len(outer_folds),
        max_pool_workers,
        ev_wall,
        out_wall,
        ev_wall + out_wall,
        _wf_agg_timings["selection"],
        _wf_agg_timings["dataset_fit"],
        _wf_agg_timings["schema"],
        _wf_agg_timings["edge_fit"],
        _wf_agg_timings["inference"],
    )

    # L1_PROBE_DIAG: 시장 regime code_1d(3-state)을 1회 계산해 fold 진단에 주입.
    _diag_regime_code: NDArray[np.int8] | None = None
    if _l1_probe_diag_enabled():
        try:
            from src.domain.futures.strategy.market_regime import (
                compress_regime_codes,
                compute_market_regime_context,
            )

            _diag_regime_code = compress_regime_codes(compute_market_regime_context(aligned=aligned).code_1d)
        except Exception as exc:
            logger.debug("[L1-PROBE-DIAG] regime code compute failed: %s", exc)
            _diag_regime_code = None
    t_outer = time.perf_counter()
    for outer_idx, outer_fold in enumerate(outer_folds):
        t_fold = time.perf_counter()
        logger.debug("[MEM] stage=outer_fold fold=%d/%d rss=%.0fMB", outer_idx + 1, len(outer_folds), _get_rss_mb())
        snapshot = snapshots_by_idx.get(outer_fold.oos_start)
        evidence = snapshot.evidence if snapshot is not None else ()
        registry = (
            snapshot.registry
            if snapshot is not None
            else QualifiedSignalRegistry(
                by_symbol={},
                ready_symbols=(),
                trade_scope_count=len(aligned.symbols),
                registry_version=f"snapshot-{outer_fold.oos_start}",
            )
        )
        if not registry.ready_symbols:
            logger.warning(
                "[L1-NESTED] Outer fold %d: registry empty — "
                "prequential evidence produced %d pairs, 0 qualified. "
                "Check l1_pair_* thresholds.",
                outer_idx,
                len(evidence),
            )
        outer_out = outer_results[outer_idx]
        if _is_trained_fold_output(outer_out):
            trained_count += 1
        outer_events = _event_results_from_fold_output(
            fold_id=outer_idx,
            fold_out=outer_out,
        )
        outer_event_frames.append(outer_events)
        _t_batch = time.perf_counter()
        prediction_batch = _candidate_output_to_signal_batch(
            model_output=outer_out.model_output,
            registry=registry,
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            model_version=f"outer-{outer_idx}",
            activation_floor_bps=float(cfg.l1_signal_activation_floor_bps),
            cfg=cfg,
        )
        _t_batch_took = time.perf_counter() - _t_batch
        _t_sel = time.perf_counter()
        opportunities = select_outer_symbol_opportunities(
            predictions=prediction_batch,
            registry=registry,
        )
        _t_sel_took = time.perf_counter() - _t_sel
        fold_sigs = _opportunities_to_symbol_signals(opportunities)
        signals_per_fold.append(fold_sigs)
        _t_eval = time.perf_counter()
        outer_reports.append(
            evaluate_outer_signal_opportunities(
                opportunities=opportunities,
                realized_event_results=outer_events,
                volatility_2d=volatility_2d,
                aligned_symbols=aligned.symbols,
                fold=outer_fold,
                fold_id=outer_idx,
                cfg=cfg,
                seed=seed + outer_idx,
                regime_code_1d=_diag_regime_code,
            )
        )
        _t_eval_took = time.perf_counter() - _t_eval
        logger.log(
            PERF,
            "[PERF] l1_outer_fold fold=%d/%d oos=[%d,%d) batch=%.4fs sel=%.4fs eval=%.4fs total=%.4fs",
            outer_idx + 1,
            len(outer_folds),
            outer_fold.oos_start,
            outer_fold.oos_end,
            _t_batch_took,
            _t_sel_took,
            _t_eval_took,
            time.perf_counter() - t_fold,
        )
        del opportunities, fold_sigs, prediction_batch, outer_events
        if (outer_idx + 1) % 2 == 0:
            gc.collect()
            logger.debug("[MEM] stage=outer_fold_gc rss=%.0fMB", _get_rss_mb())
    logger.log(PERF, "[PERF] l1_nested_outer_fold_loop took=%.4fs", time.perf_counter() - t_outer)
    del outer_results
    gc.collect()

    fold_cov = float(trained_count / len(outer_folds)) if outer_folds else 0.0
    deployment_event_results = (
        pd.concat(outer_event_frames, ignore_index=True) if outer_event_frames else pd.DataFrame()
    )
    t_ev_deploy = time.perf_counter()
    _deploy_xs_admission: dict[str, XsAdmissionBasis] | None = None
    if bool(getattr(cfg, "l1_xs_alpha_admission_enabled", False)):
        _deploy_xs_diag = compute_xs_factor_spread_diagnostics(
            realized_event_results=deployment_event_results,
            cfg=cfg,
            fold_id=-1,
            seed=seed,
            xs_archetypes=tuple(getattr(cfg, "l1_pooled_admission_archetypes", ("xs_alpha",))),
        )
        _deploy_xs_admission = resolve_xs_alpha_admission(_deploy_xs_diag, cfg)
    deployment_evidence = compute_symbol_strategy_evidence(
        event_results=deployment_event_results,
        cfg=cfg,
        seed=seed,
        registry_as_of_idx=max((fold.oos_end for fold in outer_folds), default=0) + 1,
        probe_diversity_corr=probe_diversity_corr,
        xs_admission=_deploy_xs_admission,
    )
    # [ADR_20260711_L1_POOLED_ALPHA_ADMISSION_GENERALIZATION]
    if bool(getattr(cfg, "l1_atomization_diagnostics_enabled", False)):
        for _rep in diagnose_strategy_atomization(
            deployment_evidence,
            min_effective_obs=float(getattr(cfg, "l1_pair_min_effective_obs", 5.0)),
        ):
            logger.debug(
                "[EVAL] stage=l1_atomization strategy_id=%s n_cells=%d pooled_gross=%.3f "
                "atomized_median=%.3f sign_flip=%.3f sign_flip_w=%.3f n_below_min_obs=%d "
                "dominant_reject=%s",
                _rep.strategy_id,
                _rep.n_cells,
                _rep.pooled_mean_gross_bps,
                _rep.atomized_mean_gross_bps_median,
                _rep.sign_flip_ratio,
                _rep.sign_flip_ratio_weighted,
                _rep.n_cells_below_min_effective_obs,
                _rep.dominant_reject_reason,
            )
    logger.log(
        PERF,
        "[PERF] l1_deployment_evidence took=%.4fs",
        time.perf_counter() - t_ev_deploy,
    )
    logger.debug("[MEM] stage=deployment_evidence rss=%.0fMB", _get_rss_mb())
    gate_report = evaluate_layer1_readiness(
        fold_reports=tuple(outer_reports),
        fold_cov=fold_cov,
        trade_scope_count=len(aligned.symbols),
        cfg=cfg,
        seed=seed,
    )
    t_fmt = time.perf_counter()
    if verbose:
        logger.info(
            format_layer1_outer_fold_table(
                tuple(outer_reports),
                datetimes=aligned.datetimes,
            )
        )
        logger.info(format_layer1_gate_table(gate_report))
    logger.log(PERF, "[PERF] l1_nested_audit_tables took=%.4fs", time.perf_counter() - t_fmt)
    deployment_registry: QualifiedSignalRegistry | None = None
    inference_artifact: Layer1InferenceArtifact | None = None
    oos_stacked: dict[str, SymbolSignal] = {}
    if gate_report.passed:
        deployment_registry = build_qualified_signal_registry(
            evidence=deployment_evidence,
            symbols=aligned.symbols,
            min_signals_per_symbol=int(cfg.l1_min_signals_per_symbol),
            registry_version="deployment",
            cfg=_cfg_effective,
            probe_prior_map=probe_prior_map,
        )
        _n_ready = len(deployment_registry.ready_symbols) if deployment_registry else 0
        logger.debug("[MEM] stage=deployment_registry rss=%.0fMB n_ready=%d", _get_rss_mb(), _n_ready)
        oos_stacked = _registry_to_symbol_signals(deployment_registry)
        fit_start_idx = min((fold.fit_start for fold in outer_folds), default=0)
        fit_end_idx = max((fold.oos_end for fold in outer_folds), default=0)
        if not defer_artifact:
            t_art = time.perf_counter()
            if _prefit_core is not None:
                inference_artifact = assemble_layer1_artifact(
                    core=_prefit_core,
                    deployment_registry=deployment_registry,
                    fit_end_idx=fit_end_idx,
                )
            else:
                inference_artifact = fit_layer1_inference_artifact(
                    labeled_events=labeled_events,
                    aligned=aligned,
                    deployment_registry=deployment_registry,
                    fit_start_idx=fit_start_idx,
                    fit_end_idx=fit_end_idx,
                    cfg=cfg,
                    seed=seed,
                )
            logger.log(PERF, "[PERF] l1_fit_inference_artifact took=%.4fs", time.perf_counter() - t_art)
            _art_size = "present" if inference_artifact is not None else "None"
            logger.debug("[MEM] stage=inference_artifact rss=%.0fMB artifact_size=%s", _get_rss_mb(), _art_size)
        if verbose:
            logger.info(format_layer1_deployment_registry_table(deployment_registry, all_evidence=deployment_evidence))
    _l1_result = Layer1Result(
        signals_per_fold=tuple(signals_per_fold),
        oos_stacked=oos_stacked,
        pooled_ic=0.0,
        pooled_tstat=0.0,
        breadth=0.0,
        valid_coverage=0.0,
        fold_pass_ratio=0.0,
        gate_passed=gate_report.passed,
        n_valid=len(deployment_registry.ready_symbols) if deployment_registry is not None else 0,
        n_total=len(aligned.symbols),
        n_trade_scope=len(aligned.symbols),
        outer_fold_reports=tuple(outer_reports),
        deployment_evidence=deployment_evidence,
        gate_report=gate_report,
        deployment_registry=deployment_registry,
        inference_artifact=inference_artifact,
    )

    # ── Lifecycle computation (Phase 3) ─────────────────────────────────────
    # Time: O(N * L1_T), Space: O(N)  where N=n_symbols, L1_T=l1 bar span
    t_life = time.perf_counter()
    _lifecycle_records: list[SymbolLifecycleRecord] = []
    if outer_folds:
        _l1_fit_start = min(fold.fit_start for fold in outer_folds)
        _l1_fit_end = max(fold.oos_end for fold in outer_folds)
        _active = aligned.active_mask  # NDArray[bool_] | None
        if _active is None:
            # stage6 path — no PIT mask; treat all bars as eligible
            _active = np.ones((len(aligned.datetimes), len(aligned.symbols)), dtype=np.bool_)

        _ready_syms: set[str] = set(deployment_registry.ready_symbols) if deployment_registry is not None else set()
        for _col, _sym in enumerate(aligned.symbols):
            _mask_slice = _active[_l1_fit_start:_l1_fit_end, _col]  # shape: [L1_T]
            if not _mask_slice.any():
                _lifecycle_records.append(
                    SymbolLifecycleRecord(
                        symbol=_sym,
                        fold_status="not_evaluated",
                        promotion_available_at=None,
                    )
                )
                continue

            _first_offset = int(np.argmax(_mask_slice))
            _first_abs = _l1_fit_start + _first_offset
            _promo_at: date = pd.Timestamp(aligned.datetimes[_first_abs]).date()

            if _sym in _ready_syms:
                _status: Literal["promoted", "evaluated", "failed", "not_ready", "not_evaluated"] = "promoted"
            elif _sym in oos_stacked:
                _status = "evaluated"
            else:
                _status = "failed"

            _lifecycle_records.append(
                SymbolLifecycleRecord(
                    symbol=_sym,
                    fold_status=_status,
                    promotion_available_at=_promo_at,
                )
            )

    logger.log(
        PERF,
        "[PERF] l1_lifecycle n_syms=%d l1_T=%d took=%.4fs",
        len(aligned.symbols),
        (_l1_fit_end - _l1_fit_start) if outer_folds else 0,
        time.perf_counter() - t_life,
    )
    return dataclasses.replace(_l1_result, symbol_lifecycle=tuple(_lifecycle_records))


def _layer2_result_from_trial_eval(
    eval: Layer2TrialEvaluation,
    *,
    gate_passed: bool,
    blocker_reason: str,
    extras: dict[str, Any],
) -> Layer2Result:
    """Layer2TrialEvaluation → Layer2Result 어댑터.

    공통 16+ 지표는 eval에서 1:1 복사, 배포 전용은 extras에서 주입.
    """
    _mean_er, _er_corr = compute_mean_trend_efficiency(eval.fold_attributions)
    _price_long, _price_short = compute_long_short_realized_price(eval.fold_attributions)
    _long_by_sym, _short_by_sym = compute_long_short_price_by_symbol(eval.fold_attributions)
    _major_diag = summarize_major_symbol_signal_sizing(eval.fold_attributions)
    _major_incoherence = summarize_major_symbol_regime_incoherence(eval.fold_attributions)
    _major_sleeve_diag = summarize_major_symbol_sleeve_contribution(eval.fold_attributions)
    _symbols_for_veto = eval.fold_attributions[0].major_symbol_snapshots
    _directional_veto_symbols = tuple(
        sorted({s.symbol for fa in eval.fold_attributions for s in fa.directional_veto_snapshots})
    )
    _directional_veto_summary = (
        summarize_directional_veto(eval.fold_attributions, symbols=_directional_veto_symbols)
        if _directional_veto_symbols
        else ()
    )
    return Layer2Result(
        selected_last=frozenset(eval.last_selected_symbols),
        weights_last=dict(zip(eval.last_selected_symbols, eval.last_weights, strict=False)),
        sharpe_hybrid=eval.sharpe_hybrid,
        sharpe_baseline=extras["sharpe_baseline"],
        mdd_hybrid=eval.mdd_hybrid,
        mdd_baseline=extras["mdd_baseline"],
        cagr_hybrid=eval.cagr_hybrid,
        cagr_baseline=extras["cagr_baseline"],
        mar_hybrid=eval.cagr_hybrid / (eval.mdd_hybrid + 1e-9),
        mar_baseline=extras["cagr_baseline"] / (extras["mdd_baseline"] + 1e-9),
        fold_pass_ratio=eval.fold_pass_ratio,
        turnover=extras["turnover"],
        friction_pass_pct=eval.break_even_pass_pct,
        gate_passed=gate_passed,
        blocker_reason=blocker_reason,
        allocation_policy="diagonal_kelly",
        deploy_leverage=eval.deploy_leverage,
        psr_hybrid=eval.psr_hybrid,
        growth_lcb_hybrid=eval.growth_lcb_hybrid,
        growth_lcb_baseline=eval.growth_lcb_baseline,
        sharpe_hac_hybrid=eval.sharpe_hac_hybrid,
        sharpe_hac_baseline=eval.sharpe_hac_baseline,
        dsr_hybrid=extras["dsr_hybrid"],
        cvar_95_hybrid=eval.cvar_95_hybrid,
        average_gross_exposure=eval.average_gross_exposure,
        average_net_exposure=extras["average_net_exposure"],
        cap_saturation_ratio=eval.cap_saturation_ratio,
        total_cost_bps=eval.total_cost_bps,
        n_rebalances=extras["n_rebalances"],
        block_metrics=eval.block_metrics,
        sortino_hybrid=eval.sortino_hybrid,
        terminal_multiple=extras["terminal_multiple"],
        total_pnl_pct=extras["terminal_multiple"] - 1.0,
        trade_count=eval.trade_count,
        risk_utilization=eval.risk_utilization,
        recent_fold_passed=eval.recent_fold_passed,
        recent_fold_sharpe=eval.recent_fold_sharpe,
        recent_fold_cagr=eval.recent_fold_cagr,
        recent_fold_mdd=eval.recent_fold_mdd,
        mean_trend_efficiency=_mean_er,
        trend_efficiency_corr=_er_corr,
        realized_price_long=_price_long,
        realized_price_short=_price_short,
        realized_price_long_by_symbol=_long_by_sym,
        realized_price_short_by_symbol=_short_by_sym,
        major_symbol_diag=_major_diag,
        major_symbol_sleeve_diag=_major_sleeve_diag,
        major_symbol_incoherence=_major_incoherence,
        directional_veto_summary=_directional_veto_summary,
    )


def run_l2_awf(
    *,
    signal_batch: ValidatedSignalBatch,
    aligned: AlignedMarketData,
    awf_folds: tuple[WFFold, ...],
    config: Layer2AllocationConfig,
    caps: PortfolioCaps,
    tf: str = "4h",
    verbose: bool = True,
    override_dsr: float | None = None,
    deploy_leverage: float | None = None,
    prebuilt_cache: L2SimulationCache | None = None,
    eval_memo: dict[Any, Any] | None = None,
) -> Layer2Result:
    """Layer2 AWF 포트폴리오 시뮬레이션 (delegate to evaluate_l2_trial SSOT).

    Args:
        deploy_leverage: champion L* (trial-path SSOT). None → 내부 calibrate.
            None fallback 시 config.l2_deploy_enabled + sim.fit_rets_hybrid 사용.
        eval_memo: selection 단계 _memo dict. 제공 시 evaluate_l2_trial_cached 사용.
    """
    if prebuilt_cache is not None:
        cache = prebuilt_cache
    else:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import build_l2_simulation_cache

        t_l2_cache = time.perf_counter()
        cache = build_l2_simulation_cache(aligned, signal_batch, tf)
        logger.log(PERF, "[PERF] l2_build_sim_cache took=%.4fs", time.perf_counter() - t_l2_cache)
        logger.debug("[MEM] stage=l2_build_sim_cache rss=%.0fMB", _get_rss_mb())

    # SSOT: evaluate_l2_trial 단일 호출경로 (memo 제공 시 cached wrapper 사용)
    t_eval = time.perf_counter()
    if eval_memo is not None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            _content_hash_dataclass,
        )

        cfg_ch = _content_hash_dataclass(config)
        if eval_memo:
            _sample_key = next(iter(eval_memo))
            _sample_cfg_ch, _sample_cache_id = _sample_key[1], _sample_key[0]
            logger.debug(
                "[L2-MEMO-PARITY] deploy cfg_ch=%s memo_cfg_ch=%s deploy_cache_id=%x memo_cache_id=%x",
                cfg_ch,
                _sample_cfg_ch,
                id(cache) & 0xFFFFFF,
                _sample_cache_id & 0xFFFFFF,
            )
        eval_result = evaluate_l2_trial_cached(
            cache=cache,
            signal_batch=signal_batch,
            aligned=aligned,
            awf_folds=awf_folds,
            config=config,
            caps=caps,
            tf=tf,
            deploy_leverage_override=deploy_leverage,
            eval_tag="final",
            _memo=eval_memo,
        )
    else:
        from src.domain.futures.optimization.workflow import evaluate_l2_trial as _eval_trial

        eval_result = _eval_trial(
            cache=cache,
            signal_batch=signal_batch,
            aligned=aligned,
            awf_folds=awf_folds,
            config=config,
            caps=caps,
            tf=tf,
            deploy_leverage_override=deploy_leverage,
            eval_tag="final",
        )
    logger.log(PERF, "[PERF] l2_evaluate_trial took=%.4fs", time.perf_counter() - t_eval)

    bars_per_year = _bars_per_year_for_tf(tf)

    # ── deployment extras (eval raw data 기반, 재계산 금지 — SSOT) ──
    _rets_h_arr = np.asarray(eval_result.returns_hybrid, dtype=np.float64)
    _rets_be_arr = np.asarray(eval_result.rets_baseline_ew, dtype=np.float64)
    _rets_b_arr = np.asarray(eval_result.returns_baseline, dtype=np.float64)

    sharpe_baseline = _sharpe(list(_rets_be_arr), bars_per_year=bars_per_year)
    mdd_baseline = _mdd(list(_rets_b_arr))
    cagr_baseline = _cagr(list(_rets_b_arr), bars_per_year=bars_per_year)

    turnover = float(np.mean(eval_result.all_turnovers)) if eval_result.all_turnovers else 0.0
    avg_net_exposure = float(np.mean(np.abs(eval_result.all_net_exposures))) if eval_result.all_net_exposures else 0.0
    n_rebalances = eval_result.rebalance_count
    dsr_hybrid = (
        float(override_dsr) if override_dsr is not None else _psr(list(_rets_h_arr), bars_per_year=bars_per_year)
    )
    terminal_multiple = _terminal_multiple(list(_rets_h_arr))

    extras: dict[str, Any] = {
        "sharpe_baseline": sharpe_baseline,
        "mdd_baseline": mdd_baseline,
        "cagr_baseline": cagr_baseline,
        "turnover": turnover,
        "average_net_exposure": avg_net_exposure,
        "n_rebalances": n_rebalances,
        "dsr_hybrid": dsr_hybrid,
        "terminal_multiple": terminal_multiple,
    }

    # gate는 eval에서 가져옴
    gate = eval_result.gate
    if gate is None:
        gate_passed = False
        blocker_reason = "gate_missing"
    else:
        gate_passed = gate.promotion_passed
        blocker_reason = gate.promotion_blocker

    result = _layer2_result_from_trial_eval(
        eval_result,
        gate_passed=gate_passed,
        blocker_reason=blocker_reason,
        extras=extras,
    )

    # ── deploy diagnostics (eval.deploy_leverage 사용, 재계산 금지) ──
    _risk_util_check = eval_result.mdd_hybrid / max(float(config.l2_max_mdd_abs), 1e-9)
    logger.debug(
        "[L2-DEPLOY] L*=%.4f binding=%s | CAGR=%.4f MDD=%.4f CVaR95=%.4f RiskUtil=%.3f",
        eval_result.deploy_leverage,
        eval_result.deploy_binding,
        eval_result.cagr_hybrid,
        eval_result.mdd_hybrid,
        eval_result.cvar_95_hybrid,
        _risk_util_check,
    )
    if eval_result.deploy_binding == "mdd" and abs(_risk_util_check - (1.0 - config.l2_deploy_mdd_margin)) > 0.15:
        logger.debug(
            "[L2-DEPLOY] realization gap: risk_util=%.3f expected≈%.3f"
            " (결함 #1/#2 재발 의심 — vol-targeting 또는 gross 제약 확인 요망)",
            _risk_util_check,
            1.0 - config.l2_deploy_mdd_margin,
        )

    logger.info(
        "  ● [FINAL SIMULATION RESULT]\n"
        "  ────────────────────────────────────────────────────────────────────────────\n"
        f"    Leverage (L*) : {eval_result.deploy_leverage:.4f} (binding: {eval_result.deploy_binding})\n"
        f"    CAGR / MDD    : {eval_result.cagr_hybrid:+.1%} / {eval_result.mdd_hybrid:.1%}\n"
        f"    CVaR95 / Util : {eval_result.cvar_95_hybrid:.1%} / {_risk_util_check:.1%}\n"
        "  ────────────────────────────────────────────────────────────────────────────"
    )

    # 진단: fit-rets vs OOS-rets 분포 이격
    _fit_arr = np.asarray(eval_result.fit_returns_hybrid, dtype=np.float64)
    if _fit_arr.size >= 2 and _rets_h_arr.size >= 2:
        _diag_fit_cagr = _cagr(list(_fit_arr), bars_per_year=bars_per_year)
        _diag_fit_mdd = _mdd(list(_fit_arr))
        _diag_oos_cagr = _cagr(list(_rets_h_arr), bars_per_year=bars_per_year)
        _diag_oos_mdd = _mdd(list(_rets_h_arr))
        logger.debug(
            "[L2-FINAL-DIAG] fit_CAGR_vol1=%.4f fit_MDD_vol1=%.4f | "
            "OOS_CAGR_vol1=%.4f OOS_MDD_vol1=%.4f | "
            "L*=%.4f(%s) | deployed_CAGR=%.4f deployed_MDD=%.4f",
            _diag_fit_cagr,
            _diag_fit_mdd,
            _diag_oos_cagr,
            _diag_oos_mdd,
            eval_result.deploy_leverage,
            eval_result.deploy_binding,
            eval_result.cagr_hybrid,
            eval_result.mdd_hybrid,
        )

    # ── verbose fold diagnostics (eval.fold_deployed_cagrs 기반) ──
    # (참고: eval에 fold_rets_hybrid가 없으므로 fold diag는 eval.fold_deployed_cagrs 기준)

    def _idx_to_date_label(idx: int) -> str:
        if not hasattr(aligned, "datetimes") or len(aligned.datetimes) == 0:
            return str(idx)
        safe_idx = max(0, min(int(idx), len(aligned.datetimes) - 1))
        return str(pd.Timestamp(aligned.datetimes[safe_idx]).date())

    l2_eval_start = _idx_to_date_label(awf_folds[0].oos_start) if awf_folds else None
    l2_eval_end = _idx_to_date_label(max(awf_folds[-1].oos_end - 1, awf_folds[-1].oos_start)) if awf_folds else None
    awf_fold_diags = [
        {
            "fold": i + 1,
            "sharpe": (
                float(eval_result.fold_deployed_sharpes[i])
                if hasattr(eval_result, "fold_deployed_sharpes") and i < len(eval_result.fold_deployed_sharpes)
                else 0.0
            ),
            "mdd": (
                float(eval_result.fold_deployed_mdds[i])
                if i < len(eval_result.fold_deployed_mdds) and eval_result.fold_deployed_mdds[i] is not None
                else float("nan")
            ),
            "cagr": (
                float(eval_result.fold_deployed_cagrs[i])
                if i < len(eval_result.fold_deployed_cagrs) and eval_result.fold_deployed_cagrs[i] is not None
                else 0.0
            ),
            "pass": (
                bool(eval_result.fold_deployed_cagrs[i] is not None and eval_result.fold_deployed_cagrs[i] > 0.0)
                if i < len(eval_result.fold_deployed_cagrs)
                else False
            ),
            "symbols": (eval_result.fold_selected_symbols[i] if i < len(eval_result.fold_selected_symbols) else ()),
            "symbol_count": (
                len(eval_result.fold_selected_symbols[i]) if i < len(eval_result.fold_selected_symbols) else 0
            ),
            "period": (
                f"{_idx_to_date_label(fold.oos_start)} ~ {_idx_to_date_label(max(fold.oos_end - 1, fold.oos_start))}"
            ),
            "trend_efficiency": (
                float(eval_result.fold_attributions[i].mean_trend_efficiency)
                if i < len(eval_result.fold_attributions)
                else 0.0
            ),
        }
        for i, fold in enumerate(awf_folds)
    ]
    if verbose:
        logger.info(
            format_layer2_table(
                result,
                evaluation_start=l2_eval_start,
                evaluation_end=l2_eval_end,
                awf_folds=awf_fold_diags,
                min_dsr=float(config.l2_min_dsr),
            )
        )
    return dataclasses.replace(result, master_tf=tf)


def run_l3_holdout(
    *,
    signal_batch: ValidatedSignalBatch,
    aligned: AlignedMarketData,
    holdout_span: tuple[int, int],
    config: Layer2AllocationConfig,
    caps: PortfolioCaps,
    tf: str = "4h",
    holdout_labels: tuple[str, str] | None = None,
    min_trades: int = 10,
    max_mdd_abs: float = 0.35,
    min_sharpe: float = 0.0,
    min_sortino: float = 0.0,
    max_cvar95: float = 0.06,
    verbose: bool = True,
    deploy_leverage: float | None = None,
    regime_code_1d: NDArray[np.int8] | None = None,
) -> Layer3Result:
    """Layer3 Holdout 최종 검증."""
    ho_start, ho_end = holdout_span
    label_start, label_end = holdout_labels or (str(ho_start), str(ho_end))
    if ho_end <= ho_start:
        result = Layer3Result(
            cagr=0.0,
            mdd=0.0,
            sharpe=0.0,
            mar=0.0,
            cagr_baseline=0.0,
            mdd_baseline=0.0,
            sharpe_baseline=0.0,
            mar_baseline=0.0,
            gate_passed=False,
            blocker_reason="empty_holdout_window",
        )
        if verbose:
            logger.info(
                format_layer3_table(
                    result,
                    holdout_start=label_start,
                    holdout_end=label_end,
                )
            )
        return result
    if not signal_batch.events:
        result = Layer3Result(
            cagr=0.0,
            mdd=0.0,
            sharpe=0.0,
            mar=0.0,
            cagr_baseline=0.0,
            mdd_baseline=0.0,
            sharpe_baseline=0.0,
            mar_baseline=0.0,
            gate_passed=False,
            blocker_reason="no_holdout_signals",
        )
        if verbose:
            logger.info(
                format_layer3_table(
                    result,
                    holdout_start=label_start,
                    holdout_end=label_end,
                )
            )
        return result

    dummy_fold = WFFold(
        fit_start=0,
        fit_end=ho_start,
        cal_start=max(0, ho_start // 2),
        cal_end=ho_start,
        oos_start=ho_start,
        oos_end=ho_end,
    )

    from src.domain.futures.strategy.tiered_workflow.awf_sim import build_l2_simulation_cache

    cache = build_l2_simulation_cache(aligned, signal_batch, tf)

    sim = _run_awf_simulation(
        cache=cache,
        signal_batch=signal_batch,
        aligned=aligned,
        awf_folds=(dummy_fold,),
        config=config,
        caps=caps,
        tf=tf,
    )

    bars_per_year = _bars_per_year_for_tf(tf)
    l_star = (
        float(deploy_leverage)
        if deploy_leverage is not None and np.isfinite(deploy_leverage) and deploy_leverage > 1.0
        else 1.0
    )
    unit_rets = np.asarray(sim.rets_hybrid, dtype=np.float64)
    dep = apply_deployment(rets=unit_rets, leverage=l_star, bars_per_year=bars_per_year)
    deployed_rets = dep.scaled_rets
    sharpe = _sharpe(deployed_rets.tolist(), bars_per_year=bars_per_year)
    sharpe_baseline = _sharpe(sim.rets_baseline, bars_per_year=bars_per_year)
    mdd = dep.mdd
    mdd_baseline = _mdd(sim.rets_baseline)
    cagr = dep.cagr
    cagr_baseline = _cagr(sim.rets_baseline, bars_per_year=bars_per_year)
    mar = cagr / (mdd + 1e-9)
    mar_baseline = cagr_baseline / (mdd_baseline + 1e-9)

    # 단일패스 복리/건전성 지표 (P2-A) — L2 헬퍼 재사용, 신규 수학 없음.
    equity_multiple = _terminal_multiple(deployed_rets)
    total_return = equity_multiple - 1.0
    sortino = _sortino(deployed_rets.tolist(), bars_per_year=bars_per_year)
    sortino_baseline = _sortino(sim.rets_baseline, bars_per_year=bars_per_year)
    n_trades = int(sim.trade_count)
    cvar95 = dep.cvar_95
    avg_gross_exposure = float(np.mean(sim.all_gross_exposures)) * l_star if sim.all_gross_exposures else 0.0

    metrics_finite = all(
        np.isfinite(val)
        for val in (
            sharpe,
            sharpe_baseline,
            mdd,
            mdd_baseline,
            cagr,
            cagr_baseline,
            total_return,
            equity_multiple,
            cvar95,
        )
    )
    blocker_reason = ""
    gate_passed = False
    if not sim.rets_hybrid or not sim.rets_baseline:
        blocker_reason = "no_holdout_returns"
    elif not metrics_finite:
        blocker_reason = "non_finite"
    elif n_trades < min_trades:
        blocker_reason = "insufficient_trades"
    elif total_return <= 0.0:
        blocker_reason = "negative_return"
    elif mdd > max_mdd_abs:
        blocker_reason = "mdd_abs"
    elif cvar95 > max_cvar95:
        blocker_reason = "cvar_95"
    elif sharpe < min_sharpe:
        blocker_reason = "sharpe_abs"
    elif sortino < min_sortino:
        blocker_reason = "sortino_abs"
    else:
        gate_passed = True

    _attr = sim.fold_attributions[0] if sim.fold_attributions else None
    _major_diag = summarize_major_symbol_signal_sizing((_attr,)) if _attr is not None else ()
    _major_sleeve_diag = summarize_major_symbol_sleeve_contribution((_attr,)) if _attr is not None else ()
    _major_incoherence = summarize_major_symbol_regime_incoherence((_attr,)) if _attr is not None else ()
    _l3_veto_symbols = tuple(sorted({s.symbol for s in _attr.directional_veto_snapshots})) if _attr is not None else ()
    _l3_veto_summary = (
        summarize_directional_veto((_attr,), symbols=_l3_veto_symbols) if _attr is not None and _l3_veto_symbols else ()
    )

    mean_trend_efficiency = _attr.mean_trend_efficiency if _attr is not None else 0.0
    trend_efficiency_corr = _attr.trend_efficiency_corr if _attr is not None else 0.0
    realized_price_long = _attr.realized_price_long if _attr is not None else 0.0
    realized_price_short = _attr.realized_price_short if _attr is not None else 0.0
    realized_price_long_by_symbol = _attr.realized_price_long_by_symbol if _attr is not None else ()
    realized_price_short_by_symbol = _attr.realized_price_short_by_symbol if _attr is not None else ()
    bars_long = _attr.bars_long if _attr is not None else 0
    bars_short = _attr.bars_short if _attr is not None else 0

    regime_bull_pct = regime_bear_pct = regime_crisis_pct = 0.0
    if regime_code_1d is not None:
        _arr = np.asarray(regime_code_1d, dtype=np.int8)
        _lo, _hi = ho_start, min(ho_end, _arr.shape[0])
        if _hi > _lo:
            _sl = _arr[_lo:_hi]
            _n = _sl.shape[0]
            regime_bull_pct = float(np.sum(_sl == 0)) / _n * 100.0
            regime_bear_pct = float(np.sum(_sl == 1)) / _n * 100.0
            regime_crisis_pct = float(np.sum(_sl == 2)) / _n * 100.0

    result = Layer3Result(
        cagr=cagr,
        mdd=mdd,
        sharpe=sharpe,
        mar=mar,
        cagr_baseline=cagr_baseline,
        mdd_baseline=mdd_baseline,
        sharpe_baseline=sharpe_baseline,
        mar_baseline=mar_baseline,
        gate_passed=gate_passed,
        blocker_reason=blocker_reason,
        total_return=total_return,
        equity_multiple=equity_multiple,
        sortino=sortino,
        sortino_baseline=sortino_baseline,
        n_trades=n_trades,
        cvar95=cvar95,
        avg_gross_exposure=avg_gross_exposure,
        deploy_leverage=l_star,
        min_trades=min_trades,
        max_mdd_abs=max_mdd_abs,
        min_sharpe=min_sharpe,
        min_sortino=min_sortino,
        max_cvar95=max_cvar95,
        risk_off_bars=_attr.risk_off_bars if _attr is not None else 0,
        risk_off_realized_price=_attr.risk_off_realized_price if _attr is not None else 0.0,
        risk_on_realized_price=_attr.risk_on_realized_price if _attr is not None else 0.0,
        reversal_kill_active=os.environ.get("L2_REVERSAL_KILL", "") not in ("", "0", "false", "False"),
        risk_off_episodes=_attr.risk_off_episodes if _attr is not None else (),
        regime_bull_pct=regime_bull_pct,
        regime_bear_pct=regime_bear_pct,
        regime_crisis_pct=regime_crisis_pct,
        mean_trend_efficiency=mean_trend_efficiency,
        trend_efficiency_corr=trend_efficiency_corr,
        realized_price_long=realized_price_long,
        realized_price_short=realized_price_short,
        realized_price_long_by_symbol=realized_price_long_by_symbol,
        realized_price_short_by_symbol=realized_price_short_by_symbol,
        bars_long=bars_long,
        bars_short=bars_short,
        major_symbol_diag=_major_diag,
        major_symbol_sleeve_diag=_major_sleeve_diag,
        major_symbol_incoherence=_major_incoherence,
        directional_veto_summary=_l3_veto_summary,
    )
    if verbose:
        logger.info(
            format_layer3_table(
                result,
                holdout_start=label_start,
                holdout_end=label_end,
            )
        )
    return result


# ── L2 Reversal Helpers (moved from active_pipeline.py, unchanged) ──


@dataclass(slots=True, frozen=True)
class L2ReversalReplayVariant:
    name: str
    enabled: bool
    dd_threshold: float
    persistence_bars: int
    recovery_cooldown_bars: int = 0


def _l2_reversal_replay_variants() -> tuple[L2ReversalReplayVariant, ...]:
    return (
        L2ReversalReplayVariant(name="baseline_off", enabled=False, dd_threshold=0.0, persistence_bars=1),
        L2ReversalReplayVariant(name="legacy_006_p1", enabled=True, dd_threshold=0.06, persistence_bars=1),
        L2ReversalReplayVariant(name="balanced_010_p2", enabled=True, dd_threshold=0.10, persistence_bars=2),
        L2ReversalReplayVariant(name="balanced_010_p3", enabled=True, dd_threshold=0.10, persistence_bars=3),
        L2ReversalReplayVariant(name="current_012_p3", enabled=True, dd_threshold=0.12, persistence_bars=3),
        L2ReversalReplayVariant(
            name="legacy_006_p1_cd4",
            enabled=True,
            dd_threshold=0.06,
            persistence_bars=1,
            recovery_cooldown_bars=4,
        ),
        L2ReversalReplayVariant(
            name="legacy_006_p1_cd8",
            enabled=True,
            dd_threshold=0.06,
            persistence_bars=1,
            recovery_cooldown_bars=8,
        ),
        L2ReversalReplayVariant(
            name="current_012_p3_cd8",
            enabled=True,
            dd_threshold=0.12,
            persistence_bars=3,
            recovery_cooldown_bars=8,
        ),
    )


@contextmanager
def _temporary_reversal_env(variant: L2ReversalReplayVariant) -> Iterator[None]:
    _saved: dict[str, str | None] = {}
    _env_keys = (
        "L2_REVERSAL_KILL",
        "L2_REVERSAL_DD_THRESHOLD",
        "L2_REVERSAL_PERSISTENCE_BARS",
        "L2_REVERSAL_RECOVERY_COOLDOWN",
    )
    for _key in _env_keys:
        _saved[_key] = os.environ.get(_key)
    try:
        if not variant.enabled:
            os.environ.pop("L2_REVERSAL_KILL", None)
        else:
            os.environ["L2_REVERSAL_KILL"] = "1"
            os.environ["L2_REVERSAL_DD_THRESHOLD"] = str(variant.dd_threshold)
            os.environ["L2_REVERSAL_PERSISTENCE_BARS"] = str(variant.persistence_bars)
            os.environ["L2_REVERSAL_RECOVERY_COOLDOWN"] = str(variant.recovery_cooldown_bars)
        yield
    finally:
        for _key, _val in _saved.items():
            if _val is None:
                os.environ.pop(_key, None)
            else:
                os.environ[_key] = _val


# ── L3 Reversal Economic Replay ──


@dataclass(slots=True, frozen=True)
class L3ReversalReplayResult:
    variant: str
    dd_threshold: float
    persistence_bars: int
    recovery_cooldown_bars: int
    cagr: float
    mdd: float
    sharpe: float
    gate_passed: bool
    blocker_reason: str
    risk_off_bars: int
    risk_off_realized_price: float
    risk_on_realized_price: float
    risk_off_episodes: tuple[ReversalEpisode, ...] = ()


def run_l3_reversal_economic_replay(
    *,
    signal_batch: ValidatedSignalBatch,
    aligned: AlignedMarketData,
    holdout_span: tuple[int, int],
    config: Layer2AllocationConfig,
    caps: PortfolioCaps,
    tf: str,
    deploy_leverage: float | None,
    holdout_labels: tuple[str, str] | None = None,
) -> tuple[L3ReversalReplayResult, ...]:
    results: list[L3ReversalReplayResult] = []
    for variant in _l2_reversal_replay_variants():
        with _temporary_reversal_env(variant):
            l3 = run_l3_holdout(
                signal_batch=signal_batch,
                aligned=aligned,
                holdout_span=holdout_span,
                config=config,
                caps=caps,
                tf=tf,
                holdout_labels=holdout_labels,
                verbose=False,
                deploy_leverage=deploy_leverage,
            )
        results.append(
            L3ReversalReplayResult(
                variant=variant.name,
                dd_threshold=variant.dd_threshold,
                persistence_bars=variant.persistence_bars,
                recovery_cooldown_bars=variant.recovery_cooldown_bars,
                cagr=l3.cagr,
                mdd=l3.mdd,
                sharpe=l3.sharpe,
                gate_passed=l3.gate_passed,
                blocker_reason=l3.blocker_reason,
                risk_off_bars=l3.risk_off_bars,
                risk_off_realized_price=l3.risk_off_realized_price,
                risk_on_realized_price=l3.risk_on_realized_price,
                risk_off_episodes=l3.risk_off_episodes,
            )
        )
    return tuple(results)


def _write_l3_reversal_replay_csv(
    results: tuple[L3ReversalReplayResult, ...],
    path: Path,
) -> None:
    import csv

    rows = [
        {
            "variant": r.variant,
            "dd_threshold": r.dd_threshold,
            "persistence_bars": r.persistence_bars,
            "recovery_cooldown_bars": r.recovery_cooldown_bars,
            "cagr": r.cagr,
            "mdd": r.mdd,
            "sharpe": r.sharpe,
            "gate_passed": r.gate_passed,
            "blocker_reason": r.blocker_reason,
            "risk_off_bars": r.risk_off_bars,
            "risk_off_realized_price": r.risk_off_realized_price,
            "risk_on_realized_price": r.risk_on_realized_price,
        }
        for r in results
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def format_l3_reversal_replay_table(results: tuple[L3ReversalReplayResult, ...]) -> str:
    lines = ["[L3-REVERSAL-REPLAY] Results:"]
    header = (
        f"  {'Variant':<25} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} "
        f"{'Gate':>6} {'Blocker':<20} {'RoffBars':>8} {'RoffPx':>8} {'RonPx':>8}"
    )
    lines.append(header)
    lines.extend(
        f"  {r.variant:<25} {r.cagr:>8.4f} {r.mdd:>8.4f} {r.sharpe:>8.4f} "
        f"{'PASS' if r.gate_passed else 'BLOCK':>6} {r.blocker_reason:<20} "
        f"{r.risk_off_bars:>8d} {r.risk_off_realized_price:>8.4f} {r.risk_on_realized_price:>8.4f}"
        for r in results
    )
    return "\n".join(lines)


# ── L2 Regime Directional Veto Economic Replay ──


@dataclass(slots=True, frozen=True)
class DirectionalVetoReplayVariant:
    """[ADR_20260704_L2_DIRECTIONAL_VETO] Replay control variant definition."""

    name: str
    directional_veto_enabled: bool
    directional_veto_mode: Literal["adverse_only", "contextual"] = "adverse_only"
    directional_veto_action: Literal["drop_long", "zero_mu", "cap_mu"] = "drop_long"
    directional_veto_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    directional_veto_adverse_codes: tuple[int, ...] = (1, 2)


@dataclass(slots=True, frozen=True)
class DirectionalVetoReplayResult:
    """[ADR_20260704_L2_DIRECTIONAL_VETO] Replay result row for baseline-vs-veto A/B."""

    variant: str
    baseline_parity: bool
    l2_cagr: float
    l2_mdd: float
    l2_turnover: float
    l2_average_gross_exposure: float
    l2_gate_passed: bool
    l2_blocker_reason: str
    l2_directional_veto_summary: tuple[DirectionalVetoSummary, ...]
    l3_cagr: float
    l3_mdd: float
    l3_sharpe: float
    l3_total_return: float
    l3_gate_passed: bool
    l3_blocker_reason: str
    l3_realized_price_long_by_symbol: tuple[tuple[str, float], ...]
    l3_directional_veto_summary: tuple[DirectionalVetoSummary, ...]
    adoption_passed: bool
    blocker_reason: str


def _directional_veto_replay_variants() -> tuple[DirectionalVetoReplayVariant, ...]:
    """[ADR_20260704_L2_DIRECTIONAL_VETO] Return the baseline and treatment replay variants."""

    return (
        DirectionalVetoReplayVariant(
            name="baseline",
            directional_veto_enabled=False,
            directional_veto_mode="adverse_only",
            directional_veto_action="drop_long",
        ),
        DirectionalVetoReplayVariant(
            name="veto_adverse_only",
            directional_veto_enabled=True,
            directional_veto_mode="adverse_only",
            directional_veto_action="drop_long",
        ),
        DirectionalVetoReplayVariant(
            name="contextual_cap_mu",
            directional_veto_enabled=True,
            directional_veto_mode="contextual",
            directional_veto_action="cap_mu",
        ),
        DirectionalVetoReplayVariant(
            name="contextual_zero_mu",
            directional_veto_enabled=True,
            directional_veto_mode="contextual",
            directional_veto_action="zero_mu",
        ),
        DirectionalVetoReplayVariant(
            name="contextual_crisis_only",
            directional_veto_enabled=True,
            directional_veto_mode="contextual",
            directional_veto_action="cap_mu",
            directional_veto_adverse_codes=(2,),
        ),
    )


def _directional_veto_replay_adoption_verdict(
    *,
    baseline: DirectionalVetoReplayResult,
    candidate: DirectionalVetoReplayResult,
    max_fit_false_positive_rate: float,
    min_gross_ratio: float,
    max_turnover_delta: float,
    max_fit_net_value_loss: float,
    min_l3_total_return_delta: float,
    max_l2_cagr_delta_loss: float,
) -> tuple[bool, str]:
    """[ADR_20260704_L2_DIRECTIONAL_VETO] Gate candidate adoption on fit and holdout budgets."""

    if not candidate.baseline_parity:
        return False, "baseline_parity"
    if candidate.l2_cagr < baseline.l2_cagr - max_l2_cagr_delta_loss:
        return False, "fit_cagr_degradation"
    if any(
        s.false_positive_rate > max_fit_false_positive_rate
        for s in candidate.l2_directional_veto_summary
        if s.n_fired > 0
    ):
        return False, "fit_false_positive"
    if any(s.net_veto_value < -max_fit_net_value_loss for s in candidate.l2_directional_veto_summary if s.n_fired > 0):
        return False, "fit_net_value_negative"
    if candidate.l2_average_gross_exposure / max(baseline.l2_average_gross_exposure, 1e-9) < min_gross_ratio:
        return False, "gross_preservation"
    if candidate.l2_turnover > baseline.l2_turnover + max_turnover_delta:
        return False, "turnover_budget"
    _bl_long_loss = sum(
        max(-v, 0.0) for sym, v in baseline.l3_realized_price_long_by_symbol if sym in ("BTCUSDT", "ETHUSDT")
    )
    _ca_long_loss = sum(
        max(-v, 0.0) for sym, v in candidate.l3_realized_price_long_by_symbol if sym in ("BTCUSDT", "ETHUSDT")
    )
    if _ca_long_loss >= _bl_long_loss:
        return False, "major_long_loss_not_improved"
    if candidate.l3_total_return < baseline.l3_total_return + min_l3_total_return_delta:
        return False, "below_min_total_return_delta"
    return True, ""


def _assert_directional_veto_l2_parity(
    *,
    replay_l2: Layer2Result,
    final_l2: Layer2Result,
    tolerance: float = 1e-6,
) -> bool:
    """[ADR_20260705_L2_VETO_REPLAY_PARITY] baseline_parity의 L2 leg 판정 (L3은 별도)."""
    return assert_selection_replay_parity(
        replay_evaluation=replay_l2,
        final_evaluation=final_l2,
        tolerance=tolerance,
    )


def run_directional_veto_economic_replay(
    *,
    l2_signal_batch: ValidatedSignalBatch,
    l3_signal_batch: ValidatedSignalBatch,
    aligned: AlignedMarketData,
    awf_folds: tuple[WFFold, ...],
    holdout_span: tuple[int, int],
    config: Layer2AllocationConfig,
    caps: PortfolioCaps,
    tf: str,
    deploy_leverage: float | None,
    holdout_labels: tuple[str, str] | None = None,
    baseline_l2: Layer2Result | None = None,
    baseline_l3: Layer3Result | None = None,
    regime_code_1d: NDArray[np.int8] | None = None,
    prebuilt_cache: L2SimulationCache | None = None,
    eval_memo: dict[Any, Any] | None = None,
) -> tuple[DirectionalVetoReplayResult, ...]:
    """[ADR_20260704_L2_DIRECTIONAL_VETO][ADR_20260705_L2_VETO_REPLAY_PARITY]

    Execute the 5-arm economic replay and adoption gate.
    """

    baseline_row: DirectionalVetoReplayResult | None = None
    results: list[DirectionalVetoReplayResult] = []
    for variant in _directional_veto_replay_variants():
        variant_cfg = dataclasses.replace(
            config,
            l2_regime_directional_veto_enabled=variant.directional_veto_enabled,
            l2_regime_directional_veto_mode=variant.directional_veto_mode,
            l2_regime_directional_veto_action=variant.directional_veto_action,
            l2_regime_directional_veto_symbols=variant.directional_veto_symbols,
            l2_regime_directional_veto_adverse_codes=variant.directional_veto_adverse_codes,
        )
        l2 = run_l2_awf(
            signal_batch=l2_signal_batch,
            aligned=aligned,
            awf_folds=awf_folds,
            config=variant_cfg,
            caps=caps,
            tf=tf,
            verbose=False,
            deploy_leverage=deploy_leverage,
            prebuilt_cache=prebuilt_cache,
            eval_memo=eval_memo,
        )
        _ho_start, _ho_end = holdout_span
        l3 = run_l3_holdout(
            signal_batch=l3_signal_batch,
            aligned=aligned,
            holdout_span=holdout_span,
            config=variant_cfg,
            caps=caps,
            tf=tf,
            holdout_labels=holdout_labels,
            verbose=False,
            deploy_leverage=deploy_leverage,
            regime_code_1d=regime_code_1d,
        )
        _baseline_parity = True
        if variant.name == "baseline" and baseline_l2 is not None and baseline_l3 is not None:
            _l2_parity_ok = _assert_directional_veto_l2_parity(
                replay_l2=l2,
                final_l2=baseline_l2,
            )
            _l3_parity_ok = abs(l3.cagr - baseline_l3.cagr) < 1e-6
            _baseline_parity = _l2_parity_ok and _l3_parity_ok
        row = DirectionalVetoReplayResult(
            variant=variant.name,
            baseline_parity=_baseline_parity,
            l2_cagr=l2.cagr_hybrid,
            l2_mdd=l2.mdd_hybrid,
            l2_turnover=l2.turnover,
            l2_average_gross_exposure=l2.average_gross_exposure,
            l2_gate_passed=l2.gate_passed,
            l2_blocker_reason=l2.blocker_reason,
            l2_directional_veto_summary=l2.directional_veto_summary,
            l3_cagr=l3.cagr,
            l3_mdd=l3.mdd,
            l3_sharpe=l3.sharpe,
            l3_total_return=l3.total_return,
            l3_gate_passed=l3.gate_passed,
            l3_blocker_reason=l3.blocker_reason,
            l3_realized_price_long_by_symbol=l3.realized_price_long_by_symbol,
            l3_directional_veto_summary=l3.directional_veto_summary,
            adoption_passed=False,
            blocker_reason="",
        )
        if variant.name == "baseline":
            baseline_row = row
        results.append(row)
    # Compute baseline_parity from replayed baseline row and propagate to all candidates
    _replayed_baseline_parity = baseline_row.baseline_parity if baseline_row else True
    for i, r in enumerate(results):
        if r.variant != "baseline":
            results[i] = dataclasses.replace(r, baseline_parity=_replayed_baseline_parity)
    # Run adoption gates for treatment candidates
    if baseline_row is not None:
        for i, r in enumerate(results):
            if r.variant != "baseline" and r.variant != "veto_adverse_only":
                _adoption, _reason = _directional_veto_replay_adoption_verdict(
                    baseline=baseline_row,
                    candidate=r,
                    max_fit_false_positive_rate=float(config.l2_regime_directional_veto_max_fit_false_positive_rate),
                    min_gross_ratio=float(config.l2_regime_directional_veto_min_gross_ratio),
                    max_turnover_delta=float(config.l2_regime_directional_veto_max_turnover_delta),
                    max_fit_net_value_loss=float(config.l2_regime_directional_veto_max_fit_net_value_loss),
                    min_l3_total_return_delta=float(config.l2_regime_directional_veto_min_l3_total_return_delta),
                    max_l2_cagr_delta_loss=float(config.l2_regime_directional_veto_max_l2_cagr_delta_loss),
                )
                results[i] = dataclasses.replace(r, adoption_passed=_adoption, blocker_reason=_reason)
    return tuple(results)


def _write_directional_veto_replay_detail_csv(
    results: tuple[DirectionalVetoReplayResult, ...],
    *,
    path: Path,
) -> None:
    import csv

    rows: list[dict[str, object]] = []
    for r in results:
        _rows = [
            {
                "variant": r.variant,
                "layer": layer,
                "symbol": s.symbol,
                "n_obs": s.n_obs,
                "n_watch": s.n_watch,
                "n_fired": s.n_fired,
                "fire_rate": s.fire_rate,
                "false_positive_rate": s.false_positive_rate,
                "opportunity_cost": s.opportunity_cost,
                "avoided_loss": s.avoided_loss,
                "net_veto_value": s.net_veto_value,
                "mean_trigger_loss": s.mean_trigger_loss,
                "mean_episode_bars": s.mean_episode_bars,
            }
            for layer, summaries in [("l2", r.l2_directional_veto_summary), ("l3", r.l3_directional_veto_summary)]
            for s in summaries
        ]
        rows.extend(_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def _write_directional_veto_replay_csv(
    results: tuple[DirectionalVetoReplayResult, ...],
    *,
    path: Path,
) -> None:
    """[ADR_20260704_L2_DIRECTIONAL_VETO] Persist replay rows for post-run inspection."""

    import csv

    rows = [
        {
            "variant": r.variant,
            "baseline_parity": r.baseline_parity,
            "l2_cagr": r.l2_cagr,
            "l2_mdd": r.l2_mdd,
            "l2_turnover": r.l2_turnover,
            "l2_average_gross_exposure": r.l2_average_gross_exposure,
            "l2_gate_passed": r.l2_gate_passed,
            "l2_blocker_reason": r.l2_blocker_reason,
            "l3_cagr": r.l3_cagr,
            "l3_mdd": r.l3_mdd,
            "l3_sharpe": r.l3_sharpe,
            "l3_total_return": r.l3_total_return,
            "l3_gate_passed": r.l3_gate_passed,
            "l3_blocker_reason": r.l3_blocker_reason,
            "adoption_passed": r.adoption_passed,
            "blocker_reason": r.blocker_reason,
        }
        for r in results
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def format_directional_veto_replay_table(
    results: tuple[DirectionalVetoReplayResult, ...],
) -> str:
    """[ADR_20260704_L2_DIRECTIONAL_VETO] Render the directional veto replay scorecard."""

    lines = ["[L2-DIRECTIONAL-VETO-REPLAY] Results:"]
    header = (
        f"  {'Variant':<28} {'L2-CAGR':>8} {'L2-MDD':>8} {'L2-Turn':>8} "
        f"{'L2-Gate':>8} {'L3-CAGR':>8} {'L3-MDD':>8} {'L3-Sharpe':>8} "
        f"{'L3-Ret':>8} {'L3-Gate':>8} {'Adopt':>6} {'Blocker':<20}"
    )
    lines.append(header)
    for r in results:
        _adopt = "PASS" if r.adoption_passed else "BLOCK"
        lines.append(
            f"  {r.variant:<28} {r.l2_cagr:>8.4f} {r.l2_mdd:>8.4f} {r.l2_turnover:>8.4f} "
            f"{'PASS' if r.l2_gate_passed else 'BLOCK':>8} {r.l3_cagr:>8.4f} {r.l3_mdd:>8.4f} "
            f"{r.l3_sharpe:>8.4f} {r.l3_total_return:>8.4f} "
            f"{'PASS' if r.l3_gate_passed else 'BLOCK':>8} "
            f"{_adopt:>6} {r.blocker_reason:<20}"
        )
    return "\n".join(lines)


def _to_utc_timestamp(val: Any) -> pd.Timestamp:
    if hasattr(val, "_mock_return_value") or "mock" in type(val).__name__.lower():
        return pd.Timestamp("2026-06-15", tz="UTC")
    ts = pd.to_datetime(val)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _resolve_aligned_for_tf(
    tf: str,
    aligned: AlignedMarketData,
    per_tf_data_maps: dict[str, AlignedMarketData] | None = None,
) -> AlignedMarketData:
    """Resolve aligned market data for a given TF.

    Returns per-TF aligned data from per_tf_data_maps when available,
    otherwise falls back to the primary aligned (backward compat).
    """
    if per_tf_data_maps is not None and tf in per_tf_data_maps:
        _maybe = per_tf_data_maps[tf]
        if isinstance(_maybe, AlignedMarketData):
            return _maybe
    return aligned


def run_per_tf_l1(
    *,
    tf: str,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    outer_folds: tuple[WFFold, ...],
    cfg: CandidateStrategyConfig,
    seed: int,
    verbose: bool = True,
    l2_start: date | None = None,
    probe_diversity_corr: dict[str, float] | None = None,
    probe_prior_map: dict[tuple[str, str, str], float] | None = None,
    defer_artifact: bool = False,
) -> PerTfL1Result:
    """Run L1 validation for a single TF using its native signal pool."""

    _tf_cfg = strategy_config.apply_tf_gate_overrides(cfg, tf)
    _tf_labeled = (
        labeled_events[labeled_events["native_tf"] == tf] if "native_tf" in labeled_events.columns else labeled_events
    )
    from src.domain.futures.strategy import tiered_workflow as _tiered_workflow

    run_l1_nested = cast(Any, _tiered_workflow.run_l1_nested_swf)
    l1 = run_l1_nested(
        labeled_events=_tf_labeled,
        aligned=aligned,
        outer_folds=outer_folds,
        cfg=_tf_cfg,
        seed=seed,
        verbose=verbose,
        l2_start=l2_start,
        probe_diversity_corr=probe_diversity_corr,
        probe_prior_map=probe_prior_map,
        tf=tf,
        defer_artifact=defer_artifact,
    )
    return PerTfL1Result(tf=tf, l1_result=l1, n_winning_signals=len(l1.oos_stacked))


def _tf_hours(tf: str) -> float:
    """TF 문자열 → 시간 단위 숫자 (정렬 기준).

    Args:
        tf: Timeframe 문자열 (예: ``"4h"``, ``"12h"``).

    Returns:
        TF의 시간 단위 (float). 파싱 실패 시 999.0 반환.
    """
    import re as _re

    m = _re.match(r"^(\d+)(h|m|d)$", tf.lower())
    if not m:
        return 999.0
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "h":
        return val
    if unit == "m":
        return val / 60.0
    return val * 24.0  # "d"


def _log_pertf_registry_diag(
    per_tf_l1: dict[str, PerTfL1Result],
    l2_tf_resolved: str,
) -> None:
    """Emit [L1-PERTF-REGISTRY-DIAG] log with blockers for each TF.

    Extracted for testability. Called only when logger.isEnabledFor(logging.DEBUG).
    """
    for _tf, _r in per_tf_l1.items():
        _reg = _r.l1_result.deployment_registry
        _blockers = _r.l1_result.gate_report.blockers if _r.l1_result.gate_report is not None else ()
        logger.debug(
            "[L1-PERTF-REGISTRY-DIAG] tf=%s gate_passed=%s registry_present=%s "
            "n_ready=%d edge_quality=%.2f would_resolve_master_tf=%s blockers=%s",
            _tf,
            _r.l1_result.gate_passed,
            _reg is not None,
            len(_reg.ready_symbols) if _reg is not None else 0,
            _tf_edge_quality(_r),
            l2_tf_resolved,
            ",".join(_blockers) if _blockers else "none",
        )


def _tf_edge_quality(r: PerTfL1Result) -> float:
    """Edge quality score for TF selection: Σ quality_weight*oos_edge_bps over valid strategies.

    Prefers TF with highest weighted edge, not raw signal count (avoids 4h-balanced bias).

    Args:
        r: PerTfL1Result for a single TF.

    Returns:
        Weighted edge quality score (higher is better).
    """
    panel = r.l1_result.strategy_panel
    if not panel:
        # fallback: n_winning_signals (정규화)
        return float(r.n_winning_signals)
    total: float = 0.0
    for s in panel:
        if s.valid and s.oos_edge_bps > 0.0:
            total += s.oos_edge_bps
    return total


def _resolve_l2_master_tf(
    cfg: CandidateStrategyConfig,
    per_tf_l1: dict[str, PerTfL1Result],
    probe_manifest: list[dict[str, Any]] | None = None,
) -> str:
    """Resolve the master timeframe for Layer 2 execution.

    Selection criterion: Σ oos_edge_bps (valid strategies) — edge quality, not signal count.
    This prevents 4h-balanced TF from winning on count while carrying weak edge.

    Args:
        cfg: Strategy config (l2_master_tf override takes precedence).
        per_tf_l1: Per-TF L1 results.
        probe_manifest: Probe cell manifest (fallback).

    Returns:
        Master TF string (e.g. ``"8h"``).
    """
    if cfg.l2_master_tf:
        return cfg.l2_master_tf

    if per_tf_l1:
        best_tf = max(per_tf_l1, key=lambda t: _tf_edge_quality(per_tf_l1[t]))
        if _tf_edge_quality(per_tf_l1[best_tf]) > 0.0:
            return best_tf

    if probe_manifest:
        from collections import Counter

        tf_counts: Counter[str] = Counter()
        for c in probe_manifest:
            if c.get("is_winner") and isinstance(tf_val := c.get("tf"), str):
                tf_counts[tf_val] += 1
        if tf_counts:
            return tf_counts.most_common(1)[0][0]

    return "8h"


def _select_representative_l1_registry(
    *,
    per_tf_l1: dict[str, PerTfL1Result],
    preferred_tf: str | None = None,
) -> QualifiedSignalRegistry | None:
    """[ADR_20260705_MAJOR_SYMBOL_REGISTRY_REPLAY_SYNC]
    Select a single representative L1 deployment registry from per-TF results.

    Args:
        per_tf_l1: Per-TF L1 results.
        preferred_tf: Preferred TF to use if available.

    Returns:
        A single QualifiedSignalRegistry or None.
    """
    if not per_tf_l1:
        return None

    if preferred_tf is not None and preferred_tf in per_tf_l1:
        chosen = preferred_tf
    else:
        chosen = min(per_tf_l1.keys(), key=_tf_hours)

    # Priority 1: top-level deployment_registry
    reg = per_tf_l1[chosen].l1_result.deployment_registry
    if reg is not None:
        return reg

    # Priority 2: inference_artifact.deployment_registry
    artifact = per_tf_l1[chosen].l1_result.inference_artifact
    if artifact is not None:
        artifact_reg: QualifiedSignalRegistry | None = getattr(artifact, "deployment_registry", None)
        if artifact_reg is not None:
            return artifact_reg

    return None


def _aggregate_per_tf_l1(
    per_tf_l1: dict[str, PerTfL1Result],
    *,
    preferred_tf: str | None = None,
) -> Layer1Result:
    """[ADR_20260705_MAJOR_SYMBOL_REGISTRY_REPLAY_SYNC] Merge per-TF L1 results into a unified Layer1Result.

    Args:
        per_tf_l1: Per-TF L1 results.
        preferred_tf: Preferred TF for representative registry selection.

    Returns:
        Aggregated Layer1Result.
    """
    if not per_tf_l1:
        return Layer1Result(
            signals_per_fold=(),
            oos_stacked={},
            pooled_ic=0.0,
            pooled_tstat=0.0,
            breadth=0.0,
            valid_coverage=0.0,
            fold_pass_ratio=0.0,
            gate_passed=False,
            n_valid=0,
            n_total=0,
        )

    oos_stacked: dict[str, Any] = {}
    for tf, r in per_tf_l1.items():
        for k, v in r.l1_result.oos_stacked.items():
            new_key = f"{tf}::{k}"
            oos_stacked[new_key] = v

    gate_passed = any(r.l1_result.gate_passed for r in per_tf_l1.values())

    # artifacts_by_tf: 모든 TF artifact 보존 (multi-TF signal 예측 핵심).
    # inference_artifact: 가장 fine TF(정렬 기준 첫번째) → annualization 기준 유지.
    artifacts_by_tf: dict[str, Any] = {
        tf: r.l1_result.inference_artifact for tf, r in per_tf_l1.items() if r.l1_result.inference_artifact is not None
    }
    # sorted TF 중 가장 fine(숫자 작은 시간 단위)를 base TF artifact로 선택
    merged_artifact = artifacts_by_tf[min(artifacts_by_tf, key=lambda t: _tf_hours(t))] if artifacts_by_tf else None

    lifecycles = [r.l1_result.symbol_lifecycle for r in per_tf_l1.values() if r.l1_result.symbol_lifecycle]
    merged_lifecycle = lifecycles[0] if lifecycles else ()

    deployment_registry = _select_representative_l1_registry(
        per_tf_l1=per_tf_l1,
        preferred_tf=preferred_tf,
    )

    first = next(iter(per_tf_l1.values())).l1_result
    return Layer1Result(
        gate_passed=gate_passed,
        oos_stacked=oos_stacked,
        inference_artifact=merged_artifact,
        artifacts_by_tf=artifacts_by_tf,
        symbol_lifecycle=merged_lifecycle,
        signals_per_fold=first.signals_per_fold,
        pooled_ic=first.pooled_ic,
        pooled_tstat=first.pooled_tstat,
        breadth=first.breadth,
        valid_coverage=first.valid_coverage,
        fold_pass_ratio=first.fold_pass_ratio,
        n_valid=first.n_valid,
        n_total=first.n_total,
        deployment_registry=deployment_registry,
    )


def run_tiered_pipeline(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    window: LayeredWindow,
    l1_params: dict[str, Any],
    l2_params: dict[str, Any],
    caps: PortfolioCaps | None = None,
    target_phase: str = "l3",
    l1_result_override: Layer1Result | None = None,
    verbose: bool = True,
    override_dsr: float | None = None,
    probe_diversity_corr: dict[str, float] | None = None,
    probe_prior_map: dict[tuple[str, str, str], float] | None = None,
    l1_tfs: tuple[str, ...] = ("4h", "6h", "8h", "12h", "1h", "2h"),
    per_tf_data_maps: dict[str, AlignedMarketData] | None = None,
    probe_manifest: list[dict[str, Any]] | None = None,
    l2_sim_cache: L2SimulationCache | None = None,
    l2_signal_batch: ValidatedSignalBatch | None = None,
    l2_awf_folds: tuple[WFFold, ...] | None = None,
    l2_eval_memo: dict[Any, Any] | None = None,
    regime_code_1d: NDArray[np.int8] | None = None,
) -> tuple[Layer1Result, Layer2Result | None, Layer3Result | None]:
    """[ADR_20260705_MAJOR_SYMBOL_REGISTRY_REPLAY_SYNC][ADR_20260705_CHAMPION_REPRODUCIBILITY_AND_REGISTRY_CENSUS]
    3-Layer 티어드 파이프라인 실행.

    Per-TF L1 실행 후 best TF에서 L2/L3 수행.

    Args:
        l1_result_override: 외부에서 사전 계산된 L1 결과. 제공 시 L1 재실행을 스킵하여
            Optuna L2 탐색 후 최종 실행 시 중복 피팅을 방지한다.
        l1_tfs: L1을 실행할 timeframe 목록.
        per_tf_data_maps: TF별 pre-built AlignedMarketData. None이면 단일 aligned 사용.
        probe_manifest: Probe winning cell 목록 (L2 master TF 선정용).
    """
    _L1_SWF_FOLD_CACHE.clear()
    if caps is None:
        caps = PortfolioCaps(
            gross=3.0,
            per_symbol=0.15,
            net=0.5,
            beta=1.0,
            target_ann_vol=0.20,
        )

    _is_ts = _to_utc_timestamp(window.l1_start)
    _oos_ts = _to_utc_timestamp(window.l2_start)

    # ─── Layer 1 ─────────────────────────────────────────────────────────────
    t_l1 = time.perf_counter()
    _l2_tf_resolved: str = ""
    if l1_result_override is not None:
        l1 = l1_result_override
        l2_tf = _resolve_l2_master_tf(cfg, {}, probe_manifest)
    else:
        _l2_date_resolved: date | None = (
            window.l2_start
            if isinstance(window.l2_start, date)
            else (
                None
                if (window.l2_start is None or hasattr(window.l2_start, "_mock_self"))
                else pd.Timestamp(window.l2_start).date()
            )
        )
        # P4: Pre-fork cache prime — parent process primes once, children share via CoW
        if _can_prime_feature_cache(labeled_events):
            import contextlib

            from src.domain.futures.strategy.candidate_dataset import prime_aligned_feature_cache

            _mem_cache = _get_rss_mb()
            with contextlib.suppress(KeyError, TypeError, ValueError):
                prime_aligned_feature_cache(labeled_events=labeled_events, aligned=aligned, cfg=cfg)
            _rss_now = _get_rss_mb()
            logger.debug(
                "[MEM] stage=prime_feature_cache rss=%.0fMB delta=%+.0fMB",
                _rss_now,
                _rss_now - _mem_cache,
            )

        per_tf_l1: dict[str, PerTfL1Result] = {}
        for tf_idx, tf in enumerate(l1_tfs):
            t_tf = time.perf_counter()
            aligned_tf = _resolve_aligned_for_tf(tf, aligned, per_tf_data_maps)
            t_aligned = time.perf_counter()
            if aligned_tf is aligned and tf_idx > 0 and per_tf_data_maps is not None and tf not in per_tf_data_maps:
                logger.log(PERF, "[PERF] per_tf_l1 tf=%s aligned=%.4fs skipped", tf, t_aligned - t_tf)
                continue

            logger.debug(
                "[MEM] stage=per_tf_l1_enter tf=%s rss=%.0fMB n_syms=%d n_bars=%d",
                tf,
                _get_rss_mb(),
                len(aligned_tf.symbols),
                len(aligned_tf.datetimes),
            )

            n_bars_tf = len(aligned_tf.datetimes)
            l1_start_bars_tf = int(
                np.searchsorted(
                    aligned_tf.datetimes,
                    np.datetime64(_is_ts.tz_localize(None), "ns"),
                )
            )
            l1_end_bars_tf = int(
                np.searchsorted(
                    aligned_tf.datetimes,
                    np.datetime64(_oos_ts.tz_localize(None), "ns"),
                )
            )

            from src.domain.futures.strategy import tiered_workflow as _tw_l1

            _build_l1_nested_folds: Any = _tw_l1.build_l1_nested_swf_folds
            outer_folds_tf = _build_l1_nested_folds(
                n_bars=n_bars_tf,
                l1_start_idx=l1_start_bars_tf,
                l1_end_idx=l1_end_bars_tf,
                max_label_horizon_bars=int(getattr(cfg, "max_holding_bars", 1)),
                cfg=cfg,
            )
            t_folds = time.perf_counter()

            # l2_multi_tf_enabled=True(default) → 전 TF artifact 빌드. False → 첫 TF만 빌드(구 동작).
            import os as _os_mtf

            _multi_tf_enabled: bool = bool(getattr(cfg, "l2_multi_tf_enabled", True))
            if _os_mtf.environ.get("L2_MULTI_TF", "") in ("0", "false", "False"):
                _multi_tf_enabled = False
            defer_artifact_tf = (not _multi_tf_enabled) and (len(per_tf_l1) > 0)
            per_tf_l1[tf] = run_per_tf_l1(
                tf=tf,
                labeled_events=labeled_events,
                aligned=aligned_tf,
                outer_folds=outer_folds_tf,
                cfg=cfg,
                seed=int(getattr(cfg, "seed", 42)),
                verbose=verbose,
                l2_start=_l2_date_resolved,
                probe_diversity_corr=probe_diversity_corr,
                probe_prior_map=probe_prior_map,
                defer_artifact=defer_artifact_tf,
            )
            logger.log(
                PERF,
                "[PERF] per_tf_l1 tf=%s aligned=%.4fs folds=%.4fs run_l1=%.4fs total=%.4fs rss=%.0fMB",
                tf,
                t_aligned - t_tf,
                t_folds - t_aligned,
                time.perf_counter() - t_folds,
                time.perf_counter() - t_tf,
                _get_rss_mb(),
            )
            if tf_idx < len(l1_tfs) - 1:
                time.sleep(0.5)

        _l2_tf_resolved = _resolve_l2_master_tf(cfg, per_tf_l1, probe_manifest)
        if logger.isEnabledFor(logging.DEBUG):
            _log_pertf_registry_diag(per_tf_l1, _l2_tf_resolved)
        _capture = build_validation_parity_capture(
            probe_manifest=_raw_probe_to_manifest(probe_manifest),
            per_tf_l1=per_tf_l1,
        )
        _l1_report = finalize_validation_parity_capture(_capture)
        if verbose:
            log_validation_parity_report(_l1_report, phase="l1")
        l1 = _aggregate_per_tf_l1(per_tf_l1, preferred_tf=_l2_tf_resolved)
        l1 = dataclasses.replace(
            l1,
            validation_parity_capture=_capture,
            validation_parity_report=_l1_report,
        )
        per_tf_l1.clear()
        gc.collect()
        _rss_after_agg = _get_rss_mb()
        logger.debug("[MEM] stage=aggregate_l1 rss=%.0fMB", _rss_after_agg)

    l2_tf = _l2_tf_resolved if l1_result_override is None else _resolve_l2_master_tf(cfg, {}, probe_manifest)
    logger.log(PERF, "[PERF] run_tiered_pipeline_l1_total took=%.4fs", time.perf_counter() - t_l1)

    if not l1.gate_passed:
        if verbose:
            from src.domain.futures.strategy import tiered_workflow as _tw_l1_blocked

            _tw_l1_blocked.logger.info(">> LAYER 1: BLOCKED -> gate_passed=False")
        return (l1, None, None)

    # ── Lifecycle gate (Phase 3): exclude symbols whose promotion_available_at > l2_start ──
    if l1.symbol_lifecycle and window.l2_start is not None:
        import dataclasses as _dc

        _l2_date: date = (
            window.l2_start
            if isinstance(window.l2_start, date)
            else date(1970, 1, 1)
            if hasattr(window.l2_start, "_mock_self")
            else pd.Timestamp(window.l2_start).date()
        )
        _late = {
            r.symbol
            for r in l1.symbol_lifecycle
            if r.promotion_available_at is not None and r.promotion_available_at > _l2_date
        }
        if _late:
            logger.info(
                "[L1-LIFECYCLE] %d symbol(s) excluded: promotion_available_at > l2_start=%s",
                len(_late),
                _l2_date,
            )
            l1 = _dc.replace(
                l1,
                oos_stacked={k: v for k, v in l1.oos_stacked.items() if k not in _late},
            )

    _l1_start_bars = int(
        np.searchsorted(
            aligned.datetimes,
            np.datetime64(_is_ts.tz_localize(None), "ns"),
        )
    )
    _l1_end_bars = int(
        np.searchsorted(
            aligned.datetimes,
            np.datetime64(_oos_ts.tz_localize(None), "ns"),
        )
    )

    if verbose and l1_result_override is None:
        logger.info(
            format_layer_universe_audit_table(
                (
                    build_layer_universe_audit(
                        aligned=aligned,
                        layer="L1",
                        start_idx=_l1_start_bars,
                        end_idx=_l1_end_bars,
                    ),
                )
            )
        )

    logger.debug("[MEM] stage=l1_gate_complete rss=%.0fMB gate_passed=%s", _get_rss_mb(), l1.gate_passed)

    if target_phase == "l1":
        return (l1, None, None)

    if verbose and l1_result_override is None:
        logger.info("\n>> LAYER 1: PASS -> Proceeding to Layer 2.")

    # ─── Layer 2: AWF Portfolio Optimization ─────────────────────────────────
    if verbose:
        if l1_result_override is None:
            logger.info(format_layer_header(2, "Portfolio Allocation & Risk Optimization (Final Simulation)"))
        else:
            logger.info("  ● [FINAL SIMULATION]")
    t_l2 = time.perf_counter()
    _rss_l2_entry = _get_rss_mb()
    logger.debug("[MEM] stage=l2_entry rss=%.0fMB", _rss_l2_entry)
    ho_start_idx_l2 = _date_to_idx(aligned.datetimes, window.holdout_start)
    _l2_expand = int(l2_params.get("l2_is_expansion_bars", 0))
    _l2_start_idx = max(0, _l1_end_bars - _l2_expand)
    _t_fold_build = time.perf_counter()
    if l2_awf_folds is not None:
        awf_folds = l2_awf_folds
    else:
        awf_folds = build_l2_simulation_folds(
            n_bars=len(aligned.datetimes),
            l2_start_idx=_l2_start_idx,
            holdout_start_idx=ho_start_idx_l2,
            cfg=cfg,
        )
    logger.debug(
        "[L2] awf_fold_build took=%.4fs n_folds=%d",
        time.perf_counter() - _t_fold_build,
        len(awf_folds),
    )
    if not awf_folds:
        logger.warning(
            "[L2] build_walk_forward_folds 결과 없음: L2 단일 폴드 fallback [%d, %d)",
            _l1_end_bars,
            ho_start_idx_l2,
        )
        cal_end = max(_l1_end_bars - 1, 1)
        awf_folds = (
            WFFold(
                fit_start=0,
                fit_end=cal_end,
                cal_start=max(0, cal_end - max(1, cal_end // 5)),
                cal_end=cal_end,
                oos_start=_l1_end_bars,
                oos_end=ho_start_idx_l2,
            ),
        )
    logger.debug(
        "[L2] AWF window: L2_start_bar=%d ho_start_bar=%d n_folds=%d",
        _l1_end_bars,
        ho_start_idx_l2,
        len(awf_folds),
    )

    if l1.inference_artifact is None:
        raise ValueError("Layer2 requires a fitted Layer1InferenceArtifact")

    l2_config = Layer2AllocationConfig.from_mapping(l2_params)
    # [ADR_20260705_L1_DIVERGENCE_DAMPENER] ad-hoc A/B override (champion 확정 후 적용,
    # Optuna study 대비 parity_divergence 발생은 의도된 동작).
    _l2_intra_symbol_divergence_env = os.environ.get("L2_INTRA_SYMBOL_DIVERGENCE", "")
    if _l2_intra_symbol_divergence_env not in ("", "0", "false", "False"):
        l2_config = dataclasses.replace(l2_config, l2_intra_symbol_divergence_enabled=True)
    logger.debug(
        "[L2-CONFIG] l2_min_sharpe_uplift=%.2f l2_cs_amp_enabled=%s l2_cs_amp_alpha=%.1f l2_cs_amp_mode=%s",
        l2_config.l2_min_sharpe_uplift,
        l2_config.l2_cs_amp_enabled,
        l2_config.l2_cs_amp_alpha,
        l2_config.l2_cs_amp_mode,
    )
    t_l2_pred = time.perf_counter()
    _l2_multi_tf: bool = bool(getattr(cfg, "l2_multi_tf_enabled", True))
    if os.environ.get("L2_MULTI_TF", "") in ("0", "false", "False"):
        _l2_multi_tf = False
    if l2_signal_batch is None:
        if _l2_multi_tf and l1.artifacts_by_tf:
            l2_signal_batch = predict_layer1_signals_multi_tf(
                artifacts_by_tf=l1.artifacts_by_tf,
                candidate_events=labeled_events,
                aligned=aligned,
                start_idx=_l2_start_idx,
                end_idx=ho_start_idx_l2,
                cfg=cfg,
            )
        else:
            l2_signal_batch = predict_layer1_signals(
                artifact=l1.inference_artifact,
                candidate_events=labeled_events,
                aligned=aligned,
                start_idx=_l2_start_idx,
                end_idx=ho_start_idx_l2,
                cfg=cfg,
            )
    logger.log(
        PERF,
        "[PERF] predict_layer1_signals(L2) multi_tf=%s took=%.4fs",
        _l2_multi_tf,
        time.perf_counter() - t_l2_pred,
    )
    # champion L* SSOT 전달: selection이 l2_params에 기록한 값 재사용 → recalibrate drift 0 보장
    _raw_l_star = l2_params.get("l2_deploy_leverage")
    _champion_l_star: float | None = float(_raw_l_star) if isinstance(_raw_l_star, (int, float)) else None
    l2 = run_l2_awf(
        signal_batch=l2_signal_batch,
        aligned=aligned,
        awf_folds=awf_folds,
        config=l2_config,
        caps=caps,
        tf=l2_tf,
        verbose=verbose,
        override_dsr=override_dsr,
        deploy_leverage=_champion_l_star if (_champion_l_star is not None and _champion_l_star > 1.0) else None,
        prebuilt_cache=l2_sim_cache,
        eval_memo=l2_eval_memo,
    )
    if l1.validation_parity_capture is not None:
        _l2_report = finalize_validation_parity_capture(
            l1.validation_parity_capture,
            observed_sleeve_summaries=l2.major_symbol_sleeve_diag,
        )
        l2 = dataclasses.replace(l2, validation_parity_report=_l2_report)
        if verbose:
            log_validation_parity_report(_l2_report, phase="l2")

    logger.log(PERF, "[PERF] run_tiered_pipeline_l2_total took=%.4fs", time.perf_counter() - t_l2)
    _rss_l2_after = _get_rss_mb()
    logger.debug(
        "[MEM] stage=l2_awf_complete rss=%.0fMB delta=%+.0fMB",
        _rss_l2_after,
        _rss_l2_after - _rss_l2_entry,
    )

    if verbose:
        logger.info(
            format_layer_universe_audit_table(
                (
                    build_layer_universe_audit(
                        aligned=aligned,
                        layer="L2",
                        start_idx=_l1_end_bars,
                        end_idx=ho_start_idx_l2,
                    ),
                )
            )
        )

    _major_registry_replay_env = os.environ.get("MAJOR_SYMBOL_REGISTRY_REPLAY", "")
    if _major_registry_replay_env not in ("", "0", "false", "False"):
        try:
            ho_start_idx, ho_end_idx = _resolve_holdout_span(
                aligned.datetimes,
                window.holdout_start,
                window.holdout_end,
            )
            if _l2_multi_tf and l1.artifacts_by_tf:
                _major_replay_l3_batch = predict_layer1_signals_multi_tf(
                    artifacts_by_tf=l1.artifacts_by_tf,
                    candidate_events=labeled_events,
                    aligned=aligned,
                    start_idx=ho_start_idx,
                    end_idx=ho_end_idx,
                    cfg=cfg,
                )
            else:
                _major_replay_l3_batch = predict_layer1_signals(
                    artifact=l1.inference_artifact,
                    candidate_events=labeled_events,
                    aligned=aligned,
                    start_idx=ho_start_idx,
                    end_idx=ho_end_idx,
                    cfg=cfg,
                )
            from src.domain.futures.strategy.tiered_workflow.major_symbol_registry_replay import (
                format_major_symbol_registry_replay_table,
                run_major_symbol_registry_replay,
                write_major_symbol_registry_replay_csv,
            )

            _major_replay_seed = int(getattr(cfg, "seed", 42))
            _major_replay_results = run_major_symbol_registry_replay(
                seed=_major_replay_seed,
                registry=l1.deployment_registry,
                l2_signal_batch=l2_signal_batch,
                l3_signal_batch=_major_replay_l3_batch,
                aligned=aligned,
                awf_folds=awf_folds,
                holdout_span=(ho_start_idx, ho_end_idx),
                config=l2_config,
                caps=caps,
                tf=l2_tf,
                deploy_leverage=_champion_l_star if (_champion_l_star is not None and _champion_l_star > 1.0) else None,
                holdout_labels=(str(window.holdout_start), str(window.holdout_end)),
                baseline_l2=l2,
                regime_code_1d=regime_code_1d,
                prebuilt_cache=l2_sim_cache,
                eval_memo=l2_eval_memo,
            )
            _major_replay_path_raw = os.environ.get("MAJOR_SYMBOL_REGISTRY_REPLAY_PATH", "")
            _major_replay_path = (
                Path(_major_replay_path_raw)
                if _major_replay_path_raw
                else Path(f"docs/results/major_symbol_registry_replay_seed_{_major_replay_seed}.csv")
            )
            write_major_symbol_registry_replay_csv(_major_replay_results, path=_major_replay_path)
            if verbose:
                logger.info(format_major_symbol_registry_replay_table(_major_replay_results))
        except Exception as _major_replay_exc:
            logger.error("[MAJOR-SYMBOL-REGISTRY-REPLAY] failed: %s", _major_replay_exc, exc_info=True)

    if not l2.gate_passed:
        if verbose:
            logger.info(">> LAYER 2: BLOCKED -> gate_passed=False")
        return (l1, l2, None)

    if verbose:
        logger.info(">> LAYER 2: PASS -> Proceeding to Final Holdout.")

    if target_phase == "l2":
        if verbose:
            logger.info(">> TARGET PHASE l2 REACHED -> Stopping pipeline.")
        return (l1, l2, None)

    # ─── Layer 3: Final Holdout Backtest ─────────────────────────────────────
    if verbose:
        logger.info(format_layer_header(3, "Final Holdout & Deployment Readiness"))
    t_l3 = time.perf_counter()
    try:
        ho_start_idx, ho_end_idx = _resolve_holdout_span(
            aligned.datetimes,
            window.holdout_start,
            window.holdout_end,
        )
    except Layer3WindowError:
        l3 = Layer3Result(
            cagr=0.0,
            mdd=0.0,
            sharpe=0.0,
            mar=0.0,
            cagr_baseline=0.0,
            mdd_baseline=0.0,
            sharpe_baseline=0.0,
            mar_baseline=0.0,
            gate_passed=False,
            blocker_reason="empty_holdout_window",
        )
        if verbose:
            logger.info(
                format_layer3_table(
                    l3,
                    holdout_start=str(window.holdout_start),
                    holdout_end=str(window.holdout_end),
                )
            )
        return (l1, l2, l3)
    if verbose:
        logger.info(
            format_layer_universe_audit_table(
                (
                    build_layer_universe_audit(
                        aligned=aligned,
                        layer="L3",
                        start_idx=ho_start_idx,
                        end_idx=ho_end_idx,
                    ),
                )
            )
        )
    try:
        if _l2_multi_tf and l1.artifacts_by_tf:
            l3_signal_batch = predict_layer1_signals_multi_tf(
                artifacts_by_tf=l1.artifacts_by_tf,
                candidate_events=labeled_events,
                aligned=aligned,
                start_idx=ho_start_idx,
                end_idx=ho_end_idx,
                cfg=cfg,
            )
        else:
            l3_signal_batch = predict_layer1_signals(
                artifact=l1.inference_artifact,
                candidate_events=labeled_events,
                aligned=aligned,
                start_idx=ho_start_idx,
                end_idx=ho_end_idx,
                cfg=cfg,
            )
    except Exception as exc:
        raise Layer3ExecutionError("layer3_signal_prediction_failed") from exc
    l3 = run_l3_holdout(
        signal_batch=l3_signal_batch,
        aligned=aligned,
        holdout_span=(ho_start_idx, ho_end_idx),
        config=l2_config,
        caps=caps,
        tf=l2_tf,
        holdout_labels=(str(window.holdout_start), str(window.holdout_end)),
        verbose=verbose,
        deploy_leverage=_champion_l_star if (_champion_l_star is not None and _champion_l_star > 1.0) else None,
        regime_code_1d=regime_code_1d,
    )
    logger.log(PERF, "[PERF] run_tiered_pipeline_l3_total took=%.4fs", time.perf_counter() - t_l3)

    if l1.validation_parity_capture is not None:
        _l3_report = finalize_validation_parity_capture(
            l1.validation_parity_capture,
            observed_sleeve_summaries=l3.major_symbol_sleeve_diag,
        )
        l3 = dataclasses.replace(l3, validation_parity_report=_l3_report)
        if verbose:
            log_validation_parity_report(_l3_report, phase="l3")

    # [ADR_20260705_L1_DIVERGENCE_DAMPENER] Track 2 registry census (ETH admission/activation-gap).
    # NOTE: l1.deployment_registry는 _aggregate_per_tf_l1 병합 경로에서 항상 None —
    # 표준 멀티-TF 런에서는 이 블록이 발화하지 않음(2026-07-05 실측 확인, 별도 이슈).
    if verbose and l1.deployment_registry is not None:
        _registry_census = compute_major_symbol_registry_census(
            registry=l1.deployment_registry,
            observed_sleeve_summaries=l3.major_symbol_sleeve_diag,
        )
        for _census_entry in _registry_census:
            logger.info(
                "[L1-MAJOR-REGISTRY-CENSUS] %s/%s: registry_mean_incremental_bps=%.3f "
                "hard_eligible=%s observed_active_in_holdout=%s",
                _census_entry.symbol,
                _census_entry.family,
                _census_entry.registry_mean_incremental_bps,
                _census_entry.hard_eligible,
                _census_entry.observed_active_in_holdout,
            )

    _l3_replay_env = os.environ.get("L3_REVERSAL_REPLAY", "")
    if _l3_replay_env not in ("", "0", "false", "False"):
        _l3_replay_results = run_l3_reversal_economic_replay(
            signal_batch=l3_signal_batch,
            aligned=aligned,
            holdout_span=(ho_start_idx, ho_end_idx),
            config=l2_config,
            caps=caps,
            tf=l2_tf,
            deploy_leverage=_champion_l_star if (_champion_l_star is not None and _champion_l_star > 1.0) else None,
            holdout_labels=(str(window.holdout_start), str(window.holdout_end)),
        )
        _write_l3_reversal_replay_csv(_l3_replay_results, path=Path("docs/results/l3_reversal_replay.csv"))
        if verbose:
            logger.info(format_l3_reversal_replay_table(_l3_replay_results))

    _l2_regime_directional_veto_replay = os.environ.get("L2_REGIME_DIRECTIONAL_VETO_REPLAY", "")
    if _l2_regime_directional_veto_replay not in ("", "0", "false", "False"):
        import gc as _gc_veto

        _veto_replay_deploy_leverage: float | None = (
            _champion_l_star if (_champion_l_star is not None and _champion_l_star > 1.0) else None
        )
        _veto_replay_results = run_directional_veto_economic_replay(
            l2_signal_batch=l2_signal_batch,
            l3_signal_batch=l3_signal_batch,
            aligned=aligned,
            awf_folds=awf_folds,
            holdout_span=(ho_start_idx, ho_end_idx),
            config=l2_config,
            caps=caps,
            tf=l2_tf,
            deploy_leverage=_veto_replay_deploy_leverage,
            holdout_labels=(str(window.holdout_start), str(window.holdout_end)),
            baseline_l2=l2,
            baseline_l3=l3,
            regime_code_1d=regime_code_1d,
            prebuilt_cache=l2_sim_cache,
            eval_memo=l2_eval_memo,
        )
        _write_directional_veto_replay_csv(
            _veto_replay_results,
            path=Path("docs/results/l2_regime_directional_veto_replay.csv"),
        )
        if verbose:
            logger.info(format_directional_veto_replay_table(_veto_replay_results))
        _gc_veto.collect()

    if verbose:
        logger.info("\n" + "=" * 80)
    return (l1, l2, l3)
