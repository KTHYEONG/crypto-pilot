from __future__ import annotations

import gc
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.strategy.candidate_contracts import Layer1EvidenceSnapshot
from src.domain.futures.strategy.tiered_workflow.memory import (
    GIB,
    L1MemoryPlan,
    L1PilotMeasurement,
    resolve_post_pilot_memory_plan,
    snapshot_process_tree_memory,
)

if TYPE_CHECKING:
    from src.domain.futures.strategy.config import CandidateStrategyConfig

logger = logging.getLogger(__name__)

PERF = 15

# Module globals for forked worker IPC (minimize serialization under fork)
_GLOBAL_EVIDENCE_STORE: pd.DataFrame | None = None
_GLOBAL_CFG: CandidateStrategyConfig | None = None
_GLOBAL_SYMBOLS: tuple[str, ...] | None = None
_GLOBAL_SEED: int = 0
_GLOBAL_PROBE_DIVERSITY_CORR: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class L1SnapshotTask:
    snapshot_offset: int
    as_of_idx: int


@dataclass(frozen=True, slots=True)
class L1SnapshotExecutionReport:
    pilot_task: L1SnapshotTask | None
    pilot_worker_private_bytes: int | None
    resolved_workers: int
    mode: str
    fallback_reason: str = ""


def _clear_globals() -> None:
    global _GLOBAL_EVIDENCE_STORE, _GLOBAL_CFG, _GLOBAL_SYMBOLS, _GLOBAL_SEED, _GLOBAL_PROBE_DIVERSITY_CORR
    _GLOBAL_EVIDENCE_STORE = None
    _GLOBAL_CFG = None
    _GLOBAL_SYMBOLS = None
    _GLOBAL_SEED = 0
    _GLOBAL_PROBE_DIVERSITY_CORR = None


def _execute_snapshot_task(task: L1SnapshotTask) -> Layer1EvidenceSnapshot:
    from src.domain.futures.strategy.tiered_workflow.signal_selection import (
        build_qualified_signal_registry,
        compute_symbol_strategy_evidence,
    )

    evidence_store = _GLOBAL_EVIDENCE_STORE
    cfg = _GLOBAL_CFG
    symbols = _GLOBAL_SYMBOLS
    seed_val = _GLOBAL_SEED
    probe_diversity_corr = _GLOBAL_PROBE_DIVERSITY_CORR

    if evidence_store is None or cfg is None or symbols is None:
        raise ValueError("snapshot globals not set")

    as_of_idx = task.as_of_idx
    snapshot_offset = task.snapshot_offset

    exit_idx_sorted: NDArray[np.float64] | None = None
    if not evidence_store.empty and "exit_idx" in evidence_store.columns:
        exit_idx_sorted = evidence_store["exit_idx"].to_numpy(dtype=np.float64, copy=False)

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
        seed=seed_val + snapshot_offset,
        registry_as_of_idx=as_of_idx,
        snapshot_index=snapshot_offset,
        fdr_hard_reject_override=False,
        probe_diversity_corr=probe_diversity_corr,
    )
    registry = build_qualified_signal_registry(
        evidence=evidence,
        symbols=symbols,
        min_signals_per_symbol=int(cfg.l1_min_signals_per_symbol),
        registry_version=f"snapshot-{as_of_idx}",
        cfg=cfg,
    )
    return Layer1EvidenceSnapshot(
        as_of_idx=int(as_of_idx),
        evidence=evidence,
        registry=registry,
        matured_event_count=matured_event_count,
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


def execute_l1_snapshot_batch(
    *,
    evidence_store: pd.DataFrame,
    tasks: tuple[L1SnapshotTask, ...],
    cfg: CandidateStrategyConfig,
    symbols: tuple[str, ...],
    seed: int,
    probe_diversity_corr: dict[str, float] | None = None,
) -> tuple[tuple[Layer1EvidenceSnapshot, ...], L1SnapshotExecutionReport]:
    if not tasks:
        return (), L1SnapshotExecutionReport(
            pilot_task=None,
            pilot_worker_private_bytes=None,
            resolved_workers=0,
            mode="no_tasks",
        )

    sorted_tasks = tuple(sorted(tasks, key=lambda t: t.as_of_idx))
    pilot = sorted_tasks[-1]
    n_tasks = len(sorted_tasks)

    gc.collect()

    if n_tasks < 2:
        serial_results: list[Layer1EvidenceSnapshot] = [_execute_snapshot_task(t) for t in sorted_tasks]
        return tuple(serial_results), L1SnapshotExecutionReport(
            pilot_task=pilot,
            pilot_worker_private_bytes=None,
            resolved_workers=1,
            mode="serial_few_tasks",
        )

    # Phase 1: Fork pilot and measure memory
    import multiprocessing

    mp_ctx = multiprocessing.get_context("fork")

    # Set globals before fork so children inherit them
    global _GLOBAL_EVIDENCE_STORE, _GLOBAL_CFG, _GLOBAL_SYMBOLS, _GLOBAL_SEED, _GLOBAL_PROBE_DIVERSITY_CORR
    _GLOBAL_EVIDENCE_STORE = evidence_store
    _GLOBAL_CFG = cfg
    _GLOBAL_SYMBOLS = symbols
    _GLOBAL_SEED = seed
    _GLOBAL_PROBE_DIVERSITY_CORR = probe_diversity_corr

    pilot_result: Layer1EvidenceSnapshot | None = None
    pilot_measurement: L1PilotMeasurement | None = None
    pilot_failed_infrastructure = False
    pilot_error_msg = ""

    try:
        tree_before_pilot = snapshot_process_tree_memory(os.getpid())
        t_pilot = time.perf_counter()

        with ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx) as pilot_executor:
            pilot_future = pilot_executor.submit(_execute_snapshot_task, pilot)
            pilot_result = pilot_future.result(timeout=600)

        tree_after_pilot = snapshot_process_tree_memory(os.getpid())
        elapsed = time.perf_counter() - t_pilot

        measured_private: int | None = None
        if tree_before_pilot.tree_pss_bytes is not None and tree_after_pilot.tree_pss_bytes is not None:
            measured_private = max(0, tree_after_pilot.tree_pss_bytes - tree_before_pilot.tree_pss_bytes)

        if measured_private is not None and measured_private > 0:
            pilot_measurement = L1PilotMeasurement(
                stage="evidence_snapshot",
                fold_id=pilot.as_of_idx,
                shared_input_bytes=int(evidence_store.memory_usage(deep=True).sum()) if not evidence_store.empty else 0,
                worker_private_bytes=measured_private,
                result_bytes=0,
                elapsed_seconds=elapsed,
            )
            logger.log(
                PERF,
                "[SNAP-PILOT] stage=l1_snapshot_pilot as_of_idx=%d pss_delta=%.1fMB elapsed=%.3fs",
                pilot.as_of_idx,
                measured_private / (1024 * 1024),
                elapsed,
            )
    except (ValueError, BrokenProcessPool, OSError) as exc:
        logger.warning("[SNAP-PILOT] infrastructure failure: %s — running all serially", exc)
        pilot_failed_infrastructure = True
        pilot_error_msg = str(exc)
    except Exception as exc:
        logger.error("[SNAP-PILOT] unexpected pilot error: %s", exc)
        raise

    remaining_tasks = tuple(t for t in sorted_tasks if t != pilot)
    workers = 1
    mode = "serial_fallback"
    fallback_reason = ""

    if pilot_failed_infrastructure:
        all_serial: list[Layer1EvidenceSnapshot] = [_execute_snapshot_task(t) for t in sorted_tasks]
        _clear_globals()
        return tuple(all_serial), L1SnapshotExecutionReport(
            pilot_task=pilot,
            pilot_worker_private_bytes=None,
            resolved_workers=1,
            mode="serial_all_infrastructure_failure",
            fallback_reason=pilot_error_msg,
        )

    # Phase 2: Resolve post-pilot memory plan
    stage_cap = 2
    cpu_cap = max(1, min(6, os.cpu_count() or 4))
    tree_pss_cap_bytes = 10 * GIB
    reserve_bytes = 1 * GIB
    safety_margin = 1.3

    if pilot_measurement is not None:
        try:
            tree_pss = snapshot_process_tree_memory(os.getpid())
            plan: L1MemoryPlan = resolve_post_pilot_memory_plan(
                n_remaining=len(remaining_tasks),
                pilot=pilot_measurement,
                snapshot=tree_pss,
                stage_cap=stage_cap,
                cpu_cap=cpu_cap,
                tree_pss_cap_bytes=tree_pss_cap_bytes,
                reserve_bytes=reserve_bytes,
                safety_margin=safety_margin,
            )
            workers = plan.workers
            mode = "parallel" if workers > 1 else "serial_remaining"
            logger.log(
                PERF,
                "[SNAP-PLAN] stage=l1_snapshot_plan n_remaining=%d resolved_workers=%d "
                "binding=%s projected_tree_mb=%.0f",
                len(remaining_tasks),
                workers,
                plan.binding_constraint,
                plan.projected_tree_bytes / (1024 * 1024),
            )
        except Exception as exc:
            logger.warning("[SNAP-PLAN] memory plan failed: %s — serial remaining", exc)
            workers = 1
            mode = "serial_remaining"
            fallback_reason = str(exc)
    else:
        logger.warning("[SNAP-PLAN] PSS metrics unavailable — serial remaining")
        workers = 1
        mode = "serial_remaining_no_pss"
        fallback_reason = "pss_unavailable"

    remaining_results: list[Layer1EvidenceSnapshot] = []
    if workers > 1 and len(remaining_tasks) > 0:
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as executor:
                rem_futs = {executor.submit(_execute_snapshot_task, t): t for t in remaining_tasks}
                remaining_results = [fut.result() for fut in as_completed(rem_futs)]
            remaining_results.sort(key=lambda s: s.as_of_idx)
        except (BrokenProcessPool, OSError) as exc:
            logger.warning("[SNAP-REM] pool failure: %s — serial fallback remaining", exc)
            mode = "serial_fallback_post_pool"
            fallback_reason = str(exc)
            remaining_results = [_execute_snapshot_task(t) for t in remaining_tasks]
    else:
        for t in remaining_tasks:
            remaining_results.append(_execute_snapshot_task(t))

    all_results: list[Layer1EvidenceSnapshot] = (
        [pilot_result, *remaining_results] if pilot_result is not None else remaining_results
    )
    all_results.sort(key=lambda s: s.as_of_idx)

    _clear_globals()
    gc.collect()

    report = L1SnapshotExecutionReport(
        pilot_task=pilot,
        pilot_worker_private_bytes=measured_private if pilot_measurement is not None else None,
        resolved_workers=workers,
        mode=mode,
        fallback_reason=fallback_reason,
    )
    return tuple(all_results), report
