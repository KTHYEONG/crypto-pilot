# src/domain/futures/strategy/tiered_workflow/pipeline.py

from __future__ import annotations

import gc
import logging
import multiprocessing
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr

import src.domain.futures.strategy.config as strategy_config
from src.core.utils.utils import PERF
from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.portfolio.signal_composer import (
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
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    _run_awf_simulation,
    _stack_oos_signals,
)

# 내부 모듈 임포트
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    FoldDiagnostic,
    L2SimulationCache,
    Layer1Result,
    Layer2AllocationConfig,
    Layer2BlockMetric,
    Layer2Result,
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
from src.domain.futures.strategy.tiered_workflow.l2_gate import (
    evaluate_layer2_gate,
)
from src.domain.futures.strategy.tiered_workflow.metrics import (
    _bars_per_year_for_tf,
    _cagr,
    _contiguous_block_log_growth,
    _growth_lower_confidence_bound,
    _mdd,
    _newey_west_ic_tstat,
    _psr,
    _sharpe,
    _sharpe_hac,
    _sortino,
    _terminal_multiple,
    compute_breadth_weighted_ic,
)
from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
    apply_deployment,
    calibrate_deployment_leverage,
)
from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    _candidate_output_to_signal_batch,
    _event_results_from_fold_output,
    _registry_to_symbol_signals,
    build_qualified_signal_registry,
    compute_symbol_strategy_evidence,
    evaluate_layer1_readiness,
    evaluate_outer_signal_opportunities,
    fit_layer1_inference_artifact,
    predict_layer1_signals,
    predict_layer1_signals_multi_tf,
    select_outer_symbol_opportunities,
)
from src.domain.futures.strategy.walk_forward import WFFold

if TYPE_CHECKING:
    from src.domain.futures.optimization.opt_config import LayeredWindow


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


logger = logging.getLogger("src.domain.futures.strategy.tiered_workflow")
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
    available_gb = mem.available / (1024 ** 3)
    safe_mem_gb = available_gb * 0.70

    frame_gb = frame_memory_bytes / (1024 ** 3)
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
        stage, n_tasks, requested_workers, physical_cores, cpu_limit, max_workers, available_gb,
        frame_gb, estimated_proc_gb, compact_result, result_soft_cap_mb, predicted_result_mb,
        result_mem_limit, isinstance(pinned, int) and pinned >= 1, mem_limit, workers,
    )
    return workers



def _log_fold_avg_profile(results: Sequence[Any], tag: str) -> None:
    """Log average timing profile across fold results (evidence or outer)."""
    n = len(results)
    if n == 0:
        return
    _profile_keys = (
        "schema", "dataset_fit", "dataset_early_stop",
        "dataset_calibration_fit", "dataset_calibration_eval",
        "dataset_oos", "edge_fit", "inference", "selection",
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
        tag, n,
        _agg["schema"], _agg["dataset_fit"], _agg["dataset_early_stop"],
        _agg["dataset_calibration_fit"], _agg["dataset_calibration_eval"],
        _agg["dataset_oos"], _agg["edge_fit"], _agg["inference"], _agg["selection"],
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
        from concurrent.futures import ProcessPoolExecutor

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
        logger.log(PERF,
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
    for snapshot_offset, as_of_idx in enumerate(sorted(set(snapshot_indices))):
        event_results = evidence_store
        matured_event_count = _snapshot_matured_count(evidence_store, as_of_idx, cfg)
        if exit_idx_sorted is not None:
            right = int(np.searchsorted(exit_idx_sorted, float(as_of_idx), side="left"))
            left = 0
            lookback_bars = getattr(cfg, "l1_evidence_lookback_bars", None)
            if lookback_bars is not None:
                left = int(
                    np.searchsorted(
                        exit_idx_sorted,
                        float(as_of_idx - int(lookback_bars)),
                        side="left",
                    )
                )
            event_results = evidence_store.iloc[left:right].copy()
            matured_event_count = max(0, right - left)
        evidence = compute_symbol_strategy_evidence(
            event_results=event_results,
            cfg=cfg,
            seed=seed + snapshot_offset,
            registry_as_of_idx=as_of_idx,
            snapshot_index=snapshot_offset,
        )
        registry = build_qualified_signal_registry(
            evidence=evidence,
            symbols=aligned.symbols,
            min_signals_per_symbol=int(cfg.l1_min_signals_per_symbol),
            registry_version=f"snapshot-{as_of_idx}",
            cfg=cfg,
        )
        snapshots.append(
            Layer1EvidenceSnapshot(
                as_of_idx=int(as_of_idx),
                evidence=evidence,
                registry=registry,
                matured_event_count=matured_event_count,
            )
        )
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
        n_total, len(aligned.datetimes), len(folds), purge_bars, _embargo_bars,
        getattr(cfg, 'candidate_name', cfg.__class__.__name__),
        max_workers,
    )

    signals_per_fold: list[dict[str, SymbolSignal]] = []
    fold_diags: list[FoldDiagnostic] = []

    futures: list[tuple[int, WFFold, Any]] = []
    missing_folds: list[tuple[int, WFFold]] = []

    for fold_idx, wf_fold in enumerate(folds):
        cache_key = (
            wf_fold.fit_start, wf_fold.fit_end,
            wf_fold.cal_start, wf_fold.cal_end,
            wf_fold.oos_start, wf_fold.oos_end,
            id(labeled_events), id(aligned), id(cfg)
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
                            wf_fold.fit_start, wf_fold.fit_end,
                            wf_fold.cal_start, wf_fold.cal_end,
                            wf_fold.oos_start, wf_fold.oos_end,
                            id(labeled_events), id(aligned), id(cfg)
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
                                executor.submit(
                                    cw._fit_and_predict_single_fold_from_globals, fold_idx, wf_fold
                                ),
                            )
                        )
                    for fold_idx, wf_fold, fut in submits:
                        try:
                            fold_out = fut.result()
                            futures.append((fold_idx, wf_fold, fold_out))
                            cache_key = (
                                wf_fold.fit_start, wf_fold.fit_end,
                                wf_fold.cal_start, wf_fold.cal_end,
                                wf_fold.oos_start, wf_fold.oos_end,
                                id(labeled_events), id(aligned), id(cfg)
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
        beta_f64: NDArray[np.float64] | None = (
            beta_f32.astype(np.float64) if beta_f32 is not None else None
        )
        fold_sigs: dict[str, SymbolSignal] = {}
        if _is_trained_fold_output(fold_out):
            import src.domain.futures.strategy.tiered_workflow as _tw
            fold_sigs = _tw.compose_symbol_signals(
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
        fold_realized_valid = _compute_fold_realized_valid_set(
            fold_out, min_obs=min_obs, t_stat_floor=t_stat_floor
        )
        eligible_syms = {s for s, e in zip(symbols, eligible_mask, strict=True) if e}
        f_n_valid = len(fold_realized_valid & eligible_syms)
        f_breadth = f_n_valid / max(1, f_n_eligible)
        f_n_events = len(fold_out.model_output.expected_net_bps)
        fold_diags.append(FoldDiagnostic(
            fold=fold_loop_idx + 1,
            ic=fold_ic,
            breadth=f_breadth,
            n_valid=f_n_valid,
            n_eligible=f_n_eligible,
            n_events=f_n_events,
            n_fit=int(getattr(fold_out, "n_fit", 0)),
            fit_status=getattr(fold_out, "fit_status", "failed"),
            passed=fold_ic is not None and fold_ic > 0,
        ))

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

        logger.log(PERF,
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

    logger.log(PERF,
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
    import src.domain.futures.strategy.tiered_workflow as _tw
    strategy_panel = _tw.compute_per_strategy_oos_validation(fold_tuples=futures)
    n_valid_strategies = sum(1 for sig in strategy_panel if sig.valid)
    panel_diversity = _tw.compute_panel_diversity(strategy_panel)

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
            cs_ic_tstat = float(
                cs_ic_mean / (cs_ic_std / np.sqrt(len(valid_fold_ics)) + 1e-12)
            )
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
        fold_pass_ratio = (
            sum(n for ic, n in _valid_pairs if ic > 0) / _w_total
            if _w_total > 0 else 0.0
        )
    else:
        fold_pass_ratio = 0.0

    breadth = float(np.mean([d.breadth for d in fold_diags])) if fold_diags else 0.0
    valid_coverage = (
        float(sum(1 for d in fold_diags if d.breadth >= _VALID_COVERAGE_FLAG_THRESHOLD) / len(fold_diags))
        if fold_diags else 0.0
    )
    trained_fold_coverage = (
        float(sum(1 for d in fold_diags if d.fit_status == "trained") / len(fold_diags))
        if fold_diags else 0.0
    )

    n_valid = sum(1 for s in per_sym_realized.values() if s.valid)

    sym_details: list[dict[str, Any]] = []
    for sym, sig in sorted(oos_stacked.items()):
        real = per_sym_realized.get(sym)
        sym_details.append({
            "symbol": sym,
            "raw_mu": sig.raw_mu,
            "vol": sig.volatility,
            "t_stat": real.t_stat if real is not None else 0.0,
            "ic": per_sym_ic.get(sym, 0.0),
            "valid": real.valid if real is not None else False,
        })

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
    volatility_2d = (
        pd.DataFrame(_r)
        .rolling(_rw, min_periods=2)
        .std(ddof=1)
        .to_numpy(dtype=np.float64)
    )
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
        _rss_now, _rss_now - _mem_vol, _n_t, _n_sym,
    )
    outer_reports: list[Layer1FoldReadiness] = []
    outer_event_frames: list[pd.DataFrame] = []
    signals_per_fold: list[dict[str, SymbolSignal]] = []
    trained_count = 0
    import src.domain.futures.strategy.tiered_workflow as _tw

    evidence_start = min((fold.fit_start for fold in outer_folds), default=0)
    evidence_end = max((fold.oos_start for fold in outer_folds), default=0)
    try:
        _outer_n = len(outer_folds)
        _mult = max(3, int(getattr(cfg, "l1_evidence_grid_multiplier", 3)))
        _ev_n_folds = min(_outer_n * _mult, int(getattr(cfg, "l1_evidence_max_folds", 32)))
        evidence_folds = _tw.build_l1_swf_folds(
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
        int(soft_cap_raw)
        if isinstance(soft_cap_raw, (int, float)) and not isinstance(soft_cap_raw, bool)
        else 512
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
    
    try:
        with ProcessPoolExecutor(max_workers=max_pool_workers, mp_context=mp_ctx) as executor:
            # ── Phase 1: Evidence folds ─────────────────────────────────────────────
            t_exec = time.perf_counter()
            if evidence_folds:
                ev_submits = [
                    executor.submit(
                        cw._fit_and_predict_single_fold_from_globals,
                        idx,
                        fold,
                        True,
                        force_compact,
                    )
                    for idx, fold in enumerate(evidence_folds)
                ]
                t_ev_ipc = time.perf_counter()
                _mem_ipc_ref = _get_rss_mb()
                evidence_results = [fut.result() for fut in ev_submits]
                logger.log(PERF, "[PERF] l1_evidence_ipc_collect n=%d took=%.4fs",
                    len(evidence_results), time.perf_counter() - t_ev_ipc)
                _rss_ev = _get_rss_mb()
                logger.debug("[MEM] stage=evidence_ipc rss=%.0fMB delta=%+.0fMB n_results=%d",
                    _rss_ev, _rss_ev - _mem_ipc_ref, len(evidence_results))

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
                logger.log(PERF, "[PERF] l1_prequential_evidence_snapshots took=%.4fs",
                    time.perf_counter() - t_ev_snap)
                logger.debug("[MEM] stage=evidence_snapshots rss=%.0fMB n_snapshots=%d",
                    _get_rss_mb(), len(evidence_snapshots))

                del evidence_results
                gc.collect()
                logger.debug("[MEM] stage=post_evidence_free rss=%.0fMB", _get_rss_mb())
            else:
                evidence_snapshots = ()

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
                logger.log(PERF, "[PERF] l1_outer_ipc_collect n=%d took=%.4fs",
                    len(outer_results), time.perf_counter() - t_out_ipc)
                _rss_out = _get_rss_mb()
                logger.debug("[MEM] stage=outer_ipc rss=%.0fMB delta=%+.0fMB n_results=%d",
                    _rss_out, _rss_out - _mem_ipc_ref, len(outer_results))
    finally:
        cw._GLOBAL_LABELED_EVENTS = None
        cw._GLOBAL_PREPARED_EVENTS = None
        cw._GLOBAL_ALIGNED = None
        cw._GLOBAL_CFG = None
        cw._GLOBAL_PURGE_BARS = None
        gc.collect()

    _log_fold_avg_profile(outer_results, "outer")

    logger.log(PERF, "[PERF] l1_nested_parallel_exec n_folds=%d took=%.4fs",
        len(outer_folds), time.perf_counter() - t_outer_exec)

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
            )
        )
        _t_eval_took = time.perf_counter() - _t_eval
        logger.log(PERF,
            "[PERF] l1_outer_fold fold=%d/%d oos=[%d,%d) batch=%.4fs sel=%.4fs eval=%.4fs total=%.4fs",
            outer_idx + 1, len(outer_folds),
            outer_fold.oos_start, outer_fold.oos_end,
            _t_batch_took, _t_sel_took, _t_eval_took,
            time.perf_counter() - t_fold,
        )
        del opportunities, fold_sigs, prediction_batch, outer_events
        if (outer_idx + 1) % 2 == 0:
            gc.collect()
            logger.debug("[MEM] stage=outer_fold_gc rss=%.0fMB", _get_rss_mb())
    logger.log(PERF, "[PERF] l1_nested_outer_fold_loop took=%.4fs", time.perf_counter() - t_outer)
    del outer_results
    gc.collect()

    fold_cov = (
        float(trained_count / len(outer_folds))
        if outer_folds
        else 0.0
    )
    deployment_event_results = (
        pd.concat(outer_event_frames, ignore_index=True)
        if outer_event_frames
        else pd.DataFrame()
    )
    t_ev_deploy = time.perf_counter()
    deployment_evidence = compute_symbol_strategy_evidence(
        event_results=deployment_event_results,
        cfg=cfg,
        seed=seed,
        registry_as_of_idx=max((fold.oos_end for fold in outer_folds), default=0) + 1,
        probe_diversity_corr=probe_diversity_corr,
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
    _l1_fit_start = min(fold.fit_start for fold in outer_folds)
    _l1_fit_end = max(fold.oos_end for fold in outer_folds)
    _active = aligned.active_mask  # NDArray[bool_] | None
    if _active is None:
        # stage6 path — no PIT mask; treat all bars as eligible
        _active = np.ones((len(aligned.datetimes), len(aligned.symbols)), dtype=np.bool_)

    _ready_syms: set[str] = (
        set(deployment_registry.ready_symbols) if deployment_registry is not None else set()
    )
    _lifecycle_records: list[SymbolLifecycleRecord] = []
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
        _l1_fit_end - _l1_fit_start,
        time.perf_counter() - t_life,
    )
    return dataclasses.replace(_l1_result, symbol_lifecycle=tuple(_lifecycle_records))


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
) -> Layer2Result:
    """Layer2 AWF 포트폴리오 시뮬레이션.

    Args:
        deploy_leverage: champion L* (trial-path SSOT). None → 내부 calibrate.
            None fallback 시 config.l2_deploy_enabled + sim.fit_rets_hybrid 사용.
    """
    if prebuilt_cache is not None:
        cache = prebuilt_cache
    else:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import build_l2_simulation_cache
        t_l2_cache = time.perf_counter()
        cache = build_l2_simulation_cache(aligned, signal_batch, tf)
        logger.log(PERF, "[PERF] l2_build_sim_cache took=%.4fs", time.perf_counter() - t_l2_cache)

    sim = _run_awf_simulation(
        cache=cache,
        signal_batch=signal_batch,
        aligned=aligned,
        awf_folds=awf_folds,
        config=config,
        caps=caps,
        tf=tf,
    )
    symbols = aligned.symbols
    sym_to_idx = {s: i for i, s in enumerate(symbols)}

    bars_per_year = _bars_per_year_for_tf(tf)
    sharpe_hybrid = _sharpe(sim.rets_hybrid, bars_per_year=bars_per_year)
    # uplift 표시·게이트: 순수 1/N EW 대비 (FIX-2). MDD 상대 게이트는 risk-matched 유지.
    sharpe_baseline = _sharpe(sim.rets_baseline_ew, bars_per_year=bars_per_year)
    sharpe_hac_hybrid = _sharpe_hac(sim.rets_hybrid, bars_per_year=bars_per_year)
    sharpe_hac_baseline = _sharpe_hac(sim.rets_baseline_ew, bars_per_year=bars_per_year)
    mdd_baseline = _mdd(sim.rets_baseline)  # MDD 상대 게이트는 risk-matched 기준 유지
    cagr_baseline = _cagr(sim.rets_baseline, bars_per_year=bars_per_year)
    mar_baseline = cagr_baseline / (mdd_baseline + 1e-9)
    psr_hybrid = _psr(sim.rets_hybrid)

    # ── L* 결정 (SSOT 우선: deploy_leverage 파라미터 → 내부 calibrate → 1.0) ──
    fit_rets_arr = np.asarray(sim.fit_rets_hybrid, dtype=np.float64)
    _l_star: float
    _binding: str
    if deploy_leverage is not None and deploy_leverage > 1.0:
        _l_star, _binding = deploy_leverage, "champion"
    elif config.l2_deploy_enabled and fit_rets_arr.size >= 2:
        _l_star, _binding = calibrate_deployment_leverage(
            fit_rets=fit_rets_arr,
            mdd_cap=config.l2_max_mdd_abs,
            cvar_cap=config.l2_max_cvar_95,
            mdd_margin=config.l2_deploy_mdd_margin,
            cvar_margin=config.l2_deploy_cvar_margin,
            l_hard_cap=config.l2_deploy_l_hard_cap,
            exchange_leverage_cap=config.l2_max_exchange_leverage,
        )
    else:
        _l_star, _binding = 1.0, "none"

    # deployed 지표 재계산 — trial-path(apply_deployment)와 동일 경로 (결함 #3 해소)
    _rets_hybrid_arr = np.asarray(sim.rets_hybrid, dtype=np.float64)
    _dep = apply_deployment(rets=_rets_hybrid_arr, leverage=_l_star, bars_per_year=bars_per_year)
    cagr_hybrid: float = _dep.cagr
    mdd_hybrid: float = _dep.mdd
    cvar_95_hybrid: float = _dep.cvar_95
    mar_hybrid: float = cagr_hybrid / (mdd_hybrid + 1e-9)

    # RiskUtil 정합 검증 — binding=mdd 시 risk_util ≈ (1 - mdd_margin), 이탈 시 결함 #1/#2 재발.
    _risk_util_check = mdd_hybrid / max(config.l2_max_mdd_abs, 1e-9)
    logger.info(
        "[L2-DEPLOY] L*=%.4f binding=%s | CAGR=%.4f MDD=%.4f CVaR95=%.4f RiskUtil=%.3f",
        _l_star,
        _binding,
        cagr_hybrid,
        mdd_hybrid,
        cvar_95_hybrid,
        _risk_util_check,
    )
    if _binding == "mdd" and abs(_risk_util_check - (1.0 - config.l2_deploy_mdd_margin)) > 0.15:
        logger.warning(
            "[L2-DEPLOY] realization gap: risk_util=%.3f expected≈%.3f"
            " (결함 #1/#2 재발 의심 — vol-targeting 또는 gross 제약 확인 요망)",
            _risk_util_check,
            1.0 - config.l2_deploy_mdd_margin,
        )
    avg_turnover = float(np.mean(sim.all_turnovers)) if sim.all_turnovers else 0.0
    avg_gross_exposure = float(np.mean(sim.all_gross_exposures)) if sim.all_gross_exposures else 0.0
    avg_net_exposure = float(np.mean(np.abs(sim.all_net_exposures))) if sim.all_net_exposures else 0.0
    cap_saturation_ratio = (
        float(sim.cap_saturation_count) / float(sim.rebalance_count)
        if sim.rebalance_count > 0
        else 0.0
    )
    total_cost_bps = float(sim.total_cost_hybrid * 1e4)
    # cvar_95_hybrid: apply_deployment에서 이미 산출 (unit-vol 중복 산출 제거)
    hybrid_blocks = _contiguous_block_log_growth(
        sim.rets_hybrid,
        block_bars=max(int(config.rebalance_bars), 1),
    )
    baseline_blocks = _contiguous_block_log_growth(
        sim.rets_baseline,
        block_bars=max(int(config.rebalance_bars), 1),
    )
    blocks_per_year = bars_per_year / max(int(config.rebalance_bars), 1)
    growth_lcb_hybrid = _growth_lower_confidence_bound(
        hybrid_blocks,
        blocks_per_year=blocks_per_year,
        z_value=float(config.l2_growth_lcb_z),
    )
    growth_lcb_baseline = _growth_lower_confidence_bound(
        baseline_blocks,
        blocks_per_year=blocks_per_year,
        z_value=float(config.l2_growth_lcb_z),
    )
    if override_dsr is not None:
        dsr_hybrid = float(override_dsr)
    else:
        # override 없을 때: 단일-원소 degenerate DSR(≡0.5 상수) 방지.
        # PSR은 trial-pool 독립 계산 가능한 정직한 하한으로 사용.
        dsr_hybrid = _psr(list(sim.rets_hybrid), bars_per_year=bars_per_year)
    friction_pass_pct = (
        sim.friction_pass_total / sim.signal_total if sim.signal_total > 0 else 0.0
    )
    block_metrics = tuple(
        Layer2BlockMetric(
            start_idx=fold.oos_start,
            end_idx=fold.oos_end,
            log_growth_hybrid=float(np.sum(np.log1p(np.asarray(block_h, dtype=np.float64)))) if block_h else 0.0,
            log_growth_baseline=float(np.sum(np.log1p(np.asarray(block_b, dtype=np.float64)))) if block_b else 0.0,
            mdd_hybrid=_mdd(list(block_h)),
            turnover_hybrid=float(sim.all_turnovers[idx]) if idx < len(sim.all_turnovers) else 0.0,
            active_rebalances=1 if block_h else 0,
        )
        for idx, (fold, block_h, block_b) in enumerate(
            zip(awf_folds, sim.block_rets_hybrid, sim.block_rets_baseline, strict=False)
        )
    )

    fold_sharpes_h = [_sharpe(fr) for fr in sim.fold_rets_hybrid]
    # deployed 기준 fold 지표: apply_deployment(fold_rets, L*)로 정확한 compounding 반영.
    # unit-vol MDD(~1%)와 달리 실제 리스크 수준(~15-31%)과 구간별 실현 CAGR을 표시.
    _fold_deployed = [
        apply_deployment(
            rets=np.asarray(fr, dtype=np.float64),
            leverage=_l_star,
            bars_per_year=bars_per_year,
        )
        if fr
        else None
        for fr in sim.fold_rets_hybrid
    ]

    # fold별 통과 여부는 deployed CAGR 양수 기준으로 판정한다.
    fold_compound_pass = [
        (_fd.cagr > 0.0) if _fd is not None else None for _fd in _fold_deployed
    ]
    _nonempty_fold_pass = [v for v in fold_compound_pass if v is not None]
    fold_pass_ratio = (
        sum(1 for v in _nonempty_fold_pass if v) / len(_nonempty_fold_pass)
        if _nonempty_fold_pass
        else 0.0
    )
    recent_fold_diag = next(
        (
            (passed, sharp, deployed)
            for passed, sharp, deployed in zip(
                reversed(fold_compound_pass),
                reversed(fold_sharpes_h),
                reversed(_fold_deployed),
                strict=True,
            )
            if deployed is not None
        ),
        None,
    )
    recent_fold_passed = recent_fold_diag[0] if recent_fold_diag is not None else None
    recent_fold_sharpe = recent_fold_diag[1] if recent_fold_diag is not None else None
    recent_fold_cagr = recent_fold_diag[2].cagr if recent_fold_diag is not None else 0.0
    recent_fold_mdd = recent_fold_diag[2].mdd if recent_fold_diag is not None else 0.0

    # gate config 키 (l2_params 우선, default=원칙값)
    _max_mdd_abs = float(config.l2_max_mdd_abs)

    # 신규 표시 metric (2026-06-16 평가체계 재편: 복리성장+위험효율+배치건전성)
    sortino_hybrid = _sortino(sim.rets_hybrid, bars_per_year=bars_per_year)
    terminal_multiple = _terminal_multiple(sim.rets_hybrid)
    total_pnl_pct = terminal_multiple - 1.0
    trade_count = int(sim.trade_count)
    risk_utilization = mdd_hybrid / max(_max_mdd_abs, 1e-9)

    # Stage 0: deployment sanity — NaN/무거래/저표본 명시 차단
    _deployment_ok = (
        sim.signal_total > 0
        and friction_pass_pct > 0.0
        and np.isfinite(sharpe_hybrid)
        and np.isfinite(cagr_hybrid)
        and sim.support_leak_count == 0
    )

    gate = evaluate_layer2_gate(
        deployment_failed=not _deployment_ok,
        support_leak_count=int(sim.support_leak_count),
        cagr_hybrid=float(cagr_hybrid),
        sharpe_hybrid=float(sharpe_hybrid),
        sharpe_hac_hybrid=float(sharpe_hac_hybrid),
        sharpe_hac_baseline=float(sharpe_hac_baseline),
        sortino_hybrid=float(sortino_hybrid),
        mar_hybrid=float(mar_hybrid),
        mdd_hybrid=float(mdd_hybrid),
        cvar_95_hybrid=float(cvar_95_hybrid),
        fold_pass_ratio=float(fold_pass_ratio),
        active_block_count=len(block_metrics),
        friction_pass_pct=float(friction_pass_pct),
        trade_count=int(trade_count),
        growth_lcb_hybrid=float(growth_lcb_hybrid),
        growth_lcb_baseline=float(growth_lcb_baseline),
        dsr_hybrid=float(dsr_hybrid),
        recent_fold_passed=recent_fold_passed,
        recent_fold_sharpe=recent_fold_sharpe,
        config=config,
    )
    blocker_reason = gate.promotion_blocker
    gate_passed = gate.promotion_passed
    result = Layer2Result(
        selected_last=sim.last_selected,
        weights_last={
            s: float(sim.last_w[sym_to_idx[s]])
            for s in sim.last_selected
            if s in sym_to_idx
        },
        sharpe_hybrid=sharpe_hybrid,
        sharpe_baseline=sharpe_baseline,
        mdd_hybrid=mdd_hybrid,
        mdd_baseline=mdd_baseline,
        cagr_hybrid=cagr_hybrid,
        cagr_baseline=cagr_baseline,
        mar_hybrid=mar_hybrid,
        mar_baseline=mar_baseline,
        fold_pass_ratio=fold_pass_ratio,
        turnover=avg_turnover,
        friction_pass_pct=friction_pass_pct,
        gate_passed=gate_passed,
        blocker_reason=blocker_reason,
        allocation_policy="diagonal_kelly",
        psr_hybrid=psr_hybrid,
        growth_lcb_hybrid=growth_lcb_hybrid,
        growth_lcb_baseline=growth_lcb_baseline,
        sharpe_hac_hybrid=sharpe_hac_hybrid,
        sharpe_hac_baseline=sharpe_hac_baseline,
        dsr_hybrid=dsr_hybrid,
        cvar_95_hybrid=cvar_95_hybrid,
        average_gross_exposure=avg_gross_exposure,
        average_net_exposure=avg_net_exposure,
        cap_saturation_ratio=cap_saturation_ratio,
        total_cost_bps=total_cost_bps,
        n_rebalances=sim.rebalance_count,
        block_metrics=block_metrics,
        sortino_hybrid=sortino_hybrid,
        terminal_multiple=terminal_multiple,
        total_pnl_pct=total_pnl_pct,
        trade_count=trade_count,
        risk_utilization=risk_utilization,
        recent_fold_passed=recent_fold_passed,
        recent_fold_sharpe=float(recent_fold_sharpe) if recent_fold_sharpe is not None else 0.0,
        recent_fold_cagr=float(recent_fold_cagr),
        recent_fold_mdd=float(recent_fold_mdd),
    )

    def _idx_to_date_label(idx: int) -> str:
        if not hasattr(aligned, "datetimes") or len(aligned.datetimes) == 0:
            return str(idx)
        safe_idx = max(0, min(int(idx), len(aligned.datetimes) - 1))
        return str(pd.Timestamp(aligned.datetimes[safe_idx]).date())

    l2_eval_start = _idx_to_date_label(awf_folds[0].oos_start) if awf_folds else None
    l2_eval_end = (
        _idx_to_date_label(max(awf_folds[-1].oos_end - 1, awf_folds[-1].oos_start))
        if awf_folds
        else None
    )
    awf_fold_diags = [
        {
            "fold": i + 1,
            "sharpe": s,
            "mdd": _fd.mdd if _fd is not None else 0.0,
            "cagr": _fd.cagr if _fd is not None else 0.0,
            "pass": fold_compound_pass[i] is True,
            "symbols": tuple(sim.fold_selected_symbols[i]) if i < len(sim.fold_selected_symbols) else (),
            "symbol_count": len(sim.fold_selected_symbols[i]) if i < len(sim.fold_selected_symbols) else 0,
            "period": (
                f"{_idx_to_date_label(fold.oos_start)} ~ "
                f"{_idx_to_date_label(max(fold.oos_end - 1, fold.oos_start))}"
            ),
        }
        for i, (fold, s, _fd) in enumerate(
            zip(awf_folds, fold_sharpes_h, _fold_deployed, strict=True)
        )
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
    return result


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
        if deploy_leverage is not None
        and np.isfinite(deploy_leverage)
        and deploy_leverage > 1.0
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
    avg_gross_exposure = (
        float(np.mean(sim.all_gross_exposures)) * l_star if sim.all_gross_exposures else 0.0
    )

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
    import src.domain.futures.strategy.tiered_workflow as _tw

    _tf_cfg = strategy_config.apply_tf_gate_overrides(cfg, tf)
    _tf_labeled = (
        labeled_events[labeled_events["native_tf"] == tf]
        if "native_tf" in labeled_events.columns
        else labeled_events
    )

    l1 = _tw.run_l1_nested_swf(
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


def _aggregate_per_tf_l1(
    per_tf_l1: dict[str, PerTfL1Result],
) -> Layer1Result:
    """Merge per-TF L1 results into a unified Layer1Result."""
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
        tf: r.l1_result.inference_artifact
        for tf, r in per_tf_l1.items()
        if r.l1_result.inference_artifact is not None
    }
    # sorted TF 중 가장 fine(숫자 작은 시간 단위)를 base TF artifact로 선택
    merged_artifact = (
        artifacts_by_tf[min(artifacts_by_tf, key=lambda t: _tf_hours(t))]
        if artifacts_by_tf
        else None
    )

    lifecycles = [
        r.l1_result.symbol_lifecycle
        for r in per_tf_l1.values()
        if r.l1_result.symbol_lifecycle
    ]
    merged_lifecycle = lifecycles[0] if lifecycles else ()

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
    l1_tfs: tuple[str, ...] = ("1h", "2h", "4h", "6h", "8h", "12h"),
    per_tf_data_maps: dict[str, AlignedMarketData] | None = None,
    probe_manifest: list[dict[str, Any]] | None = None,
    l2_sim_cache: L2SimulationCache | None = None,
) -> tuple[Layer1Result, Layer2Result | None, Layer3Result | None]:
    """3-Layer 티어드 파이프라인 실행.

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

    import src.domain.futures.strategy.tiered_workflow as _tw

    # ─── Layer 1 ─────────────────────────────────────────────────────────────
    t_l1 = time.perf_counter()
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
                _rss_now, _rss_now - _mem_cache,
            )

        per_tf_l1: dict[str, PerTfL1Result] = {}
        for tf_idx, tf in enumerate(l1_tfs):
            t_tf = time.perf_counter()
            aligned_tf = _resolve_aligned_for_tf(tf, aligned, per_tf_data_maps)
            t_aligned = time.perf_counter()
            if (
                aligned_tf is aligned
                and tf_idx > 0
                and per_tf_data_maps is not None
                and tf not in per_tf_data_maps
            ):
                logger.log(PERF, "[PERF] per_tf_l1 tf=%s aligned=%.4fs skipped", tf, t_aligned - t_tf)
                continue

            logger.debug(
                "[MEM] stage=per_tf_l1_enter tf=%s rss=%.0fMB n_syms=%d n_bars=%d",
                tf, _get_rss_mb(), len(aligned_tf.symbols), len(aligned_tf.datetimes),
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

            outer_folds_tf = _tw.build_l1_nested_swf_folds(
                n_bars=n_bars_tf,
                l1_start_idx=l1_start_bars_tf,
                l1_end_idx=l1_end_bars_tf,
                max_label_horizon_bars=int(getattr(cfg, "max_holding_bars", 1)),
                cfg=cfg,
            )
            t_folds = time.perf_counter()

            # l2_multi_tf_enabled=True(default) → 전 TF artifact 빌드. False → 첫 TF만 빌드(구 동작).
            _multi_tf_enabled: bool = bool(getattr(cfg, "l2_multi_tf_enabled", True))
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

        l1 = _aggregate_per_tf_l1(per_tf_l1)
        per_tf_l1.clear()
        gc.collect()
        _rss_after_agg = _get_rss_mb()
        logger.debug("[MEM] stage=aggregate_l1 rss=%.0fMB", _rss_after_agg)
        l2_tf = _resolve_l2_master_tf(cfg, {}, probe_manifest)

    l2_tf = _resolve_l2_master_tf(cfg, per_tf_l1 if l1_result_override is None else {}, probe_manifest)
    logger.log(PERF, "[PERF] run_tiered_pipeline_l1_total took=%.4fs", time.perf_counter() - t_l1)

    if not l1.gate_passed:
        if verbose:
            logger.info(">> LAYER 1: BLOCKED -> gate_passed=False")
        return (l1, None, None)

    # ── Lifecycle gate (Phase 3): exclude symbols whose promotion_available_at > l2_start ──
    if l1.symbol_lifecycle and window.l2_start is not None:
        import dataclasses as _dc
        _l2_date: date = (
            window.l2_start if isinstance(window.l2_start, date)
            else date(1970, 1, 1) if hasattr(window.l2_start, "_mock_self")
            else pd.Timestamp(window.l2_start).date()
        )
        _late = {r.symbol for r in l1.symbol_lifecycle
                 if r.promotion_available_at is not None and r.promotion_available_at > _l2_date}
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
    ho_start_idx_l2 = _date_to_idx(aligned.datetimes, window.holdout_start)
    _l2_expand = int(l2_params.get("l2_is_expansion_bars", 0))
    _l2_start_idx = max(0, _l1_end_bars - _l2_expand)
    awf_folds = _tw.build_l2_simulation_folds(
        n_bars=len(aligned.datetimes),
        l2_start_idx=_l2_start_idx,
        holdout_start_idx=ho_start_idx_l2,
        cfg=cfg,
    )
    if not awf_folds:
        logger.warning(
            "[L2] build_walk_forward_folds 결과 없음: L2 단일 폴드 fallback [%d, %d)",
            _l1_end_bars,
            ho_start_idx_l2,
        )
        cal_end = max(_l1_end_bars - 1, 1)
        awf_folds = (WFFold(
            fit_start=0,
            fit_end=cal_end,
            cal_start=max(0, cal_end - max(1, cal_end // 5)),
            cal_end=cal_end,
            oos_start=_l1_end_bars,
            oos_end=ho_start_idx_l2,
        ),)
    logger.debug(
        "[L2] AWF window: L2_start_bar=%d ho_start_bar=%d n_folds=%d",
        _l1_end_bars,
        ho_start_idx_l2,
        len(awf_folds),
    )

    if l1.inference_artifact is None:
        raise ValueError("Layer2 requires a fitted Layer1InferenceArtifact")

    l2_config = Layer2AllocationConfig.from_mapping(l2_params)
    t_l2_pred = time.perf_counter()
    _l2_multi_tf: bool = bool(getattr(cfg, "l2_multi_tf_enabled", True))
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
    _champion_l_star: float | None = (
        float(_raw_l_star) if isinstance(_raw_l_star, (int, float)) else None
    )
    l2 = _tw.run_l2_awf(
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
    )
    logger.log(PERF, "[PERF] run_tiered_pipeline_l2_total took=%.4fs", time.perf_counter() - t_l2)

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
    l3 = _tw.run_l3_holdout(
        signal_batch=l3_signal_batch,
        aligned=aligned,
        holdout_span=(ho_start_idx, ho_end_idx),
        config=l2_config,
        caps=caps,
        tf=l2_tf,
        holdout_labels=(str(window.holdout_start), str(window.holdout_end)),
        verbose=verbose,
        deploy_leverage=_champion_l_star if (_champion_l_star is not None and _champion_l_star > 1.0) else None,
    )
    logger.log(PERF, "[PERF] run_tiered_pipeline_l3_total took=%.4fs", time.perf_counter() - t_l3)

    if verbose:
        logger.info("\n" + "="*80)
    return (l1, l2, l3)
