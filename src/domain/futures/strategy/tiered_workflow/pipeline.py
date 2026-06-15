# src/domain/futures/strategy/tiered_workflow/pipeline.py

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.portfolio.signal_composer import (
    composer_sigma_lookback_bars,
    rolling_per_bar_return_std,
)
from src.domain.futures.strategy.candidate_contracts import (
    CandidateFoldOutput,
    Layer1EvidenceSnapshot,
    Layer1FoldReadiness,
    Layer1InferenceArtifact,
    QualifiedSignalRegistry,
    ValidatedSignalBatch,
)
from src.domain.futures.strategy.cs_rank import SymbolSignal
from src.domain.futures.strategy.tiered_logging import (
    format_layer1_deployment_registry_table,
    format_layer1_gate_table,
    format_layer1_outer_fold_table,
    format_layer1_table,
    format_layer2_table,
    format_layer3_table,
    format_layer_header,
)
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    _run_awf_simulation,
    _stack_oos_signals,
)

# 내부 모듈 임포트
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    FoldDiagnostic,
    Layer1Result,
    Layer2AllocationConfig,
    Layer2Result,
    Layer3Result,
)
from src.domain.futures.strategy.tiered_workflow.diagnostics import (
    _compute_fold_realized_valid_set,
    _compute_fold_ts_ic,
    _fold_eligible_symbol_mask,
    _is_trained_fold_output,
    _log_fold_regime_analysis,
    compute_per_symbol_ic,
    compute_per_symbol_realized_stats,
    compute_prediction_decomposition_diag,
)
from src.domain.futures.strategy.tiered_workflow.metrics import (
    _cagr,
    _mdd,
    _newey_west_ic_tstat,
    _psr,
    _sharpe,
    compute_breadth_weighted_ic,
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
    select_outer_symbol_opportunities,
)
from src.domain.futures.strategy.walk_forward import WFFold

if TYPE_CHECKING:
    from src.domain.futures.optimization.opt_config import LayeredWindow
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.config import CandidateStrategyConfig

logger = logging.getLogger("src.domain.futures.strategy.tiered_workflow")

_VALID_COVERAGE_FLAG_THRESHOLD: float = 0.80
_TRAINED_FOLD_COVERAGE_THRESHOLD: float = 0.80


def _can_prime_feature_cache(labeled_events: pd.DataFrame) -> bool:
    return not labeled_events.empty and "entry_idx" in labeled_events.columns


def _date_to_idx(datetimes: NDArray[np.datetime64], target_date: Any) -> int:
    """target_date에 해당하는 bar 인덱스 검색."""
    target = np.datetime64(target_date, "D")
    idx = int(np.searchsorted(datetimes.astype("datetime64[D]"), target))
    return min(idx, len(datetimes) - 1)


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
    *,
    pinned: int | None = None,
) -> int:
    """Compute safe worker count dynamically under WSL constraints.

    Args:
        n_tasks: Number of tasks to parallelize.
        frame_memory_bytes: Estimated DataFrame size in bytes for OOM guard.
        pinned: If set, fix worker count to this value (reproducibility mode).
    """
    if isinstance(pinned, int) and pinned >= 1:
        return max(1, min(n_tasks, pinned))

    import psutil

    physical_cores = os.cpu_count() or 4
    cpu_limit = max(1, int(physical_cores * 0.75))

    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    safe_mem_gb = available_gb * 0.70

    frame_gb = frame_memory_bytes / (1024 ** 3)
    estimated_proc_gb = 0.3 + max(0.1, frame_gb * 3.0)

    mem_limit = max(1, int(safe_mem_gb // estimated_proc_gb))

    return max(1, min(n_tasks, cpu_limit, mem_limit, 6))


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
    from src.domain.futures.strategy.config import resolve_purge_and_embargo_bars

    purge_bars, _embargo_bars = resolve_purge_and_embargo_bars(cfg)
    evidence_frames: list[pd.DataFrame] = []

    if not evidence_folds:
        all_evidence_events = pd.DataFrame()
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
        all_evidence_events = (
            pd.concat(evidence_frames, ignore_index=True)
            if evidence_frames
            else pd.DataFrame()
        )
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        import src.domain.futures.strategy.candidate_workflow as cw

        assert cw._GLOBAL_LABELED_EVENTS is None, "Global state collision: _GLOBAL_LABELED_EVENTS must be None"
        assert cw._GLOBAL_ALIGNED is None
        assert cw._GLOBAL_CFG is None
        assert cw._GLOBAL_PURGE_BARS is None

        # Set process globals to minimize IPC size under fork
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
            pinned=getattr(cfg, "l1_nested_workers", None),
        )
        logger.debug(
            "[EVIDENCE-PREQ] Fitting %d evidence folds in parallel with %d workers (WSL OOM Guard, pinned=%s)",
            len(evidence_folds),
            workers,
            getattr(cfg, "l1_nested_workers", None),
        )

        flat_results = []
        t_exec = time.perf_counter()
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as executor:
                submits = [
                    executor.submit(cw._fit_and_predict_single_fold_from_globals, idx, fold)
                    for idx, fold in enumerate(evidence_folds)
                ]
                flat_results = [fut.result() for fut in submits]
        finally:
            cw._GLOBAL_LABELED_EVENTS = None
            cw._GLOBAL_ALIGNED = None
            cw._GLOBAL_CFG = None
            cw._GLOBAL_PURGE_BARS = None
        logger.debug(
            "[perf-tiered] build_l1_prequential_evidence_snapshots parallel execution took %.4fs",
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
        all_evidence_events = (
            pd.concat(evidence_frames, ignore_index=True)
            if evidence_frames
            else pd.DataFrame()
        )
    snapshots: list[Layer1EvidenceSnapshot] = []
    for snapshot_offset, as_of_idx in enumerate(sorted(set(snapshot_indices))):
        evidence = compute_symbol_strategy_evidence(
            event_results=all_evidence_events,
            cfg=cfg,
            seed=seed + snapshot_offset,
            registry_as_of_idx=as_of_idx,
        )
        registry = build_qualified_signal_registry(
            evidence=evidence,
            symbols=aligned.symbols,
            min_signals_per_symbol=int(cfg.l1_min_signals_per_symbol),
            registry_version=f"snapshot-{as_of_idx}",
        )
        matured_event_count = 0
        if not all_evidence_events.empty and "exit_idx" in all_evidence_events.columns:
            exit_idx = pd.to_numeric(all_evidence_events["exit_idx"], errors="coerce").fillna(np.inf)
            mature_mask = exit_idx < float(as_of_idx)
            lookback_bars = getattr(cfg, "l1_evidence_lookback_bars", None)
            if lookback_bars is not None:
                mature_mask &= exit_idx >= float(as_of_idx - int(lookback_bars))
            matured_event_count = int(mature_mask.sum())
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
) -> Layer1Result:
    """Layer1 SWF-K 신호 검증."""
    from src.domain.futures.strategy.config import resolve_purge_and_embargo_bars

    purge_bars, _embargo_bars = resolve_purge_and_embargo_bars(cfg)

    import src.domain.futures.strategy.candidate_workflow as cw

    planned_workers = max(1, (os.cpu_count() or 4) // 2)
    max_workers = min(len(folds), planned_workers)

    t_start = time.perf_counter()
    logger.debug(
        "[SWF-START] Starting SWF-K L1 signal validation with %d folds (max_workers=%d)",
        len(folds),
        max_workers,
    )

    signals_per_fold: list[dict[str, SymbolSignal]] = []
    fold_diags: list[FoldDiagnostic] = []
    symbols = aligned.symbols
    n_total = len(symbols)

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
                        fold_out = _tw._fit_and_predict_single_fold(
                            fold_idx, wf_fold, labeled_events, aligned, cfg, purge_bars
                        )
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

        logger.debug(
            "[SWF-PROFILE] Average sub-fold execution breakdown: "
            "schema=%.3fs, ds_fit=%.3fs, ds_es=%.3fs, ds_cal_fit=%.3fs, ds_cal_eval=%.3fs, "
            "ds_oos=%.3fs, edge_fit=%.3fs, inference=%.3fs, selection=%.3fs",
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

    logger.debug(
        "[SWF-END] SWF-K L1 signal validation completed in %.2fs",
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
) -> Layer1Result:
    """Run nested Layer1 validation using inner selection and outer evaluation."""
    import dataclasses
    from copy import copy

    from src.domain.futures.strategy.config import resolve_purge_and_embargo_bars

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

    purge_bars, embargo_bars = resolve_purge_and_embargo_bars(cfg)
    _vol_window = composer_sigma_lookback_bars("4h")
    volatility_2d = np.column_stack(
        [rolling_per_bar_return_std(aligned.close_2d[:, i], _vol_window) for i in range(aligned.close_2d.shape[1])]
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
        )
    except ValueError:
        evidence_folds = ()

    combined_folds = tuple(evidence_folds) + tuple(outer_folds)
    num_evidence = len(evidence_folds)

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import src.domain.futures.strategy.candidate_workflow as cw
    from src.domain.futures.strategy.candidate_dataset import prime_aligned_feature_cache

    # Assert global states to prevent nested collisions
    assert cw._GLOBAL_LABELED_EVENTS is None, "Global state collision: _GLOBAL_LABELED_EVENTS must be None"
    assert cw._GLOBAL_ALIGNED is None
    assert cw._GLOBAL_CFG is None
    assert cw._GLOBAL_PURGE_BARS is None

    # Prime the feature cache on the parent process before multiprocessing fork
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

    # Set process globals to minimize IPC size under fork
    cw._GLOBAL_LABELED_EVENTS = labeled_events
    cw._GLOBAL_ALIGNED = aligned
    cw._GLOBAL_CFG = l1_cfg
    cw._GLOBAL_PURGE_BARS = purge_bars
    mp_ctx = multiprocessing.get_context("fork")

    # Calculate memory consumption dynamically
    try:
        frame_memory_bytes = int(labeled_events.memory_usage(deep=True).sum())
    except Exception:
        frame_memory_bytes = int(labeled_events.memory_usage().sum())

    workers = resolve_safe_nested_workers(
        len(combined_folds),
        frame_memory_bytes,
        pinned=getattr(cfg, "l1_nested_workers", None),
    )
    logger.debug(
        "[L1-NESTED-COMBINED] Fitting %d folds (evidence=%d, outer=%d) in parallel with %d workers (pinned=%s)",
        len(combined_folds),
        num_evidence,
        len(outer_folds),
        workers,
        getattr(cfg, "l1_nested_workers", None),
    )

    combined_results = []
    t_exec = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as executor:
            submits = [
                executor.submit(cw._fit_and_predict_single_fold_from_globals, idx, fold)
                for idx, fold in enumerate(combined_folds)
            ]
            combined_results = [fut.result() for fut in submits]
    finally:
        cw._GLOBAL_LABELED_EVENTS = None
        cw._GLOBAL_ALIGNED = None
        cw._GLOBAL_CFG = None
        cw._GLOBAL_PURGE_BARS = None
    logger.debug(
        "[perf-tiered] run_l1_nested_swf combined parallel execution took %.4fs",
        time.perf_counter() - t_exec,
    )

    evidence_results = combined_results[:num_evidence]
    outer_results = combined_results[num_evidence:]

    evidence_snapshots = build_l1_prequential_evidence_snapshots(
        labeled_events=labeled_events,
        aligned=aligned,
        evidence_folds=evidence_folds,
        snapshot_indices=tuple(fold.oos_start for fold in outer_folds),
        cfg=l1_cfg,
        seed=seed,
        precomputed_results=evidence_results,
    )
    snapshots_by_idx = {snapshot.as_of_idx: snapshot for snapshot in evidence_snapshots}

    for outer_idx, outer_fold in enumerate(outer_folds):
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
        prediction_batch = _candidate_output_to_signal_batch(
            model_output=outer_out.model_output,
            registry=registry,
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            model_version=f"outer-{outer_idx}",
            activation_floor_bps=float(cfg.l1_signal_activation_floor_bps),
            cfg=cfg,
        )
        opportunities = select_outer_symbol_opportunities(
            predictions=prediction_batch,
            registry=registry,
        )
        fold_sigs = _opportunities_to_symbol_signals(opportunities)
        signals_per_fold.append(fold_sigs)
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
    deployment_evidence = compute_symbol_strategy_evidence(
        event_results=deployment_event_results,
        cfg=cfg,
        seed=seed,
        registry_as_of_idx=max((fold.oos_end for fold in outer_folds), default=0) + 1,
    )
    gate_report = evaluate_layer1_readiness(
        fold_reports=tuple(outer_reports),
        fold_cov=fold_cov,
        trade_scope_count=len(aligned.symbols),
        cfg=cfg,
        seed=seed,
    )
    logger.info(format_layer1_outer_fold_table(tuple(outer_reports)))
    logger.info(format_layer1_gate_table(gate_report))
    deployment_registry: QualifiedSignalRegistry | None = None
    inference_artifact: Layer1InferenceArtifact | None = None
    oos_stacked: dict[str, SymbolSignal] = {}
    if gate_report.passed:
        deployment_registry = build_qualified_signal_registry(
            evidence=deployment_evidence,
            symbols=aligned.symbols,
            min_signals_per_symbol=int(cfg.l1_min_signals_per_symbol),
            registry_version="deployment",
        )
        oos_stacked = _registry_to_symbol_signals(deployment_registry)
        fit_start_idx = min((fold.fit_start for fold in outer_folds), default=0)
        fit_end_idx = max((fold.oos_end for fold in outer_folds), default=0)
        inference_artifact = fit_layer1_inference_artifact(
            labeled_events=labeled_events,
            aligned=aligned,
            deployment_registry=deployment_registry,
            fit_start_idx=fit_start_idx,
            fit_end_idx=fit_end_idx,
            cfg=cfg,
            seed=seed,
        )
        logger.info(format_layer1_deployment_registry_table(deployment_registry))
    return Layer1Result(
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


def run_l2_awf(
    *,
    signal_batch: ValidatedSignalBatch,
    aligned: AlignedMarketData,
    awf_folds: tuple[WFFold, ...],
    config: Layer2AllocationConfig,
    caps: PortfolioCaps,
    tf: str = "4h",
) -> Layer2Result:
    """Layer2 AWF 포트폴리오 시뮬레이션."""
    sim = _run_awf_simulation(
        signal_batch=signal_batch,
        aligned=aligned,
        awf_folds=awf_folds,
        config=config,
        caps=caps,
        tf=tf,
    )
    symbols = aligned.symbols
    sym_to_idx = {s: i for i, s in enumerate(symbols)}

    sharpe_hybrid = _sharpe(sim.rets_hybrid)
    sharpe_baseline = _sharpe(sim.rets_baseline)
    mdd_hybrid = _mdd(sim.rets_hybrid)
    mdd_baseline = _mdd(sim.rets_baseline)
    cagr_hybrid = _cagr(sim.rets_hybrid)
    cagr_baseline = _cagr(sim.rets_baseline)
    mar_hybrid = cagr_hybrid / (mdd_hybrid + 1e-9)
    mar_baseline = cagr_baseline / (mdd_baseline + 1e-9)
    psr_hybrid = _psr(sim.rets_hybrid)
    avg_turnover = float(np.mean(sim.all_turnovers)) if sim.all_turnovers else 0.0
    friction_pass_pct = (
        sim.friction_pass_total / sim.signal_total if sim.signal_total > 0 else 0.0
    )

    # fold별 복리 수익 여부: prod(1+r)>1.0 기준 (Sharpe>0보다 엄격).
    # 전체 fold 정렬 유지 (빈 fold 제외, 분모=nonempty). zip(strict=True) 길이 정합.
    fold_compound_pass = [
        float(np.prod(1.0 + np.asarray(fr, dtype=np.float64))) > 1.0
        if fr else None
        for fr in sim.fold_rets_hybrid
    ]
    _nonempty_fold_pass = [v for v in fold_compound_pass if v is not None]
    fold_pass_ratio = (
        sum(1 for v in _nonempty_fold_pass if v) / len(_nonempty_fold_pass)
        if _nonempty_fold_pass
        else 0.0
    )

    # gate config 키 (l2_params 우선, default=원칙값)
    _min_cagr = float(config.l2_min_cagr)
    _min_mar = float(config.l2_min_mar)
    _min_sharpe_abs = float(config.l2_min_sharpe_abs)
    _max_mdd_abs = float(config.l2_max_mdd_abs)
    _min_fold_pass = float(config.l2_min_fold_pass_ratio)
    _min_uplift = float(config.l2_min_sharpe_uplift)
    _min_psr = float(config.l2_min_psr)
    _min_friction_pass = float(config.l2_min_friction_pass)

    # Stage 0: deployment sanity — NaN/무거래 명시 차단
    _deployment_ok = (
        sim.signal_total > 0
        and friction_pass_pct > 0.0
        and np.isfinite(sharpe_hybrid)
        and np.isfinite(cagr_hybrid)
        and sim.support_leak_count == 0
    )

    blocker_reason = ""
    gate_passed = False
    if not _deployment_ok:
        blocker_reason = "no_deployment"
    elif cagr_hybrid <= _min_cagr:
        blocker_reason = "cagr"
    elif mar_hybrid < _min_mar:
        blocker_reason = "mar"
    elif sharpe_hybrid < _min_sharpe_abs:
        blocker_reason = "sharpe_abs"
    elif mdd_hybrid > mdd_baseline:
        blocker_reason = "mdd_rel"
    elif mdd_hybrid > _max_mdd_abs:
        blocker_reason = "mdd_abs"
    elif fold_pass_ratio < _min_fold_pass:
        blocker_reason = "fold"
    elif psr_hybrid < _min_psr:
        blocker_reason = "psr"
    elif friction_pass_pct < _min_friction_pass:
        blocker_reason = "friction"
    elif sharpe_hybrid < sharpe_baseline + _min_uplift:
        blocker_reason = "uplift"
    else:
        gate_passed = True

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
        psr_hybrid=psr_hybrid,
    )
    fold_sharpes_h = [_sharpe(fr) for fr in sim.fold_rets_hybrid]
    awf_fold_diags = [
        {
            "fold": i + 1,
            "sharpe": s,
            "mdd": _mdd(fr),
            "pass": fold_compound_pass[i] is True,
        }
        for i, (s, fr) in enumerate(zip(fold_sharpes_h, sim.fold_rets_hybrid, strict=True))
    ]
    logger.info(format_layer2_table(result, awf_folds=awf_fold_diags))
    return result


def run_l3_holdout(
    *,
    signal_batch: ValidatedSignalBatch,
    aligned: AlignedMarketData,
    holdout_span: tuple[int, int],
    config: Layer2AllocationConfig,
    caps: PortfolioCaps,
    tf: str = "4h",
) -> Layer3Result:
    """Layer3 Holdout 최종 검증."""
    ho_start, ho_end = holdout_span
    dummy_fold = WFFold(
        fit_start=0,
        fit_end=ho_start,
        cal_start=max(0, ho_start // 2),
        cal_end=ho_start,
        oos_start=ho_start,
        oos_end=ho_end,
    )

    sim = _run_awf_simulation(
        signal_batch=signal_batch,
        aligned=aligned,
        awf_folds=(dummy_fold,),
        config=config,
        caps=caps,
        tf=tf,
    )

    sharpe = _sharpe(sim.rets_hybrid)
    sharpe_baseline = _sharpe(sim.rets_baseline)
    mdd = _mdd(sim.rets_hybrid)
    mdd_baseline = _mdd(sim.rets_baseline)
    cagr = _cagr(sim.rets_hybrid)
    cagr_baseline = _cagr(sim.rets_baseline)
    mar = cagr / (mdd + 1e-9)
    mar_baseline = cagr_baseline / (mdd_baseline + 1e-9)

    gate_passed: bool = bool(
        (sharpe >= sharpe_baseline) and (mdd <= mdd_baseline)
    )

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
    )
    logger.info(format_layer3_table(result, ho_start=str(ho_start), ho_end=str(ho_end)))
    return result


def run_tiered_pipeline(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    window: LayeredWindow,
    l1_params: dict[str, Any],
    l2_params: dict[str, Any],
    caps: PortfolioCaps | None = None,
    tf: str = "4h",
    target_phase: str = "l3",
) -> tuple[Layer1Result, Layer2Result | None, Layer3Result | None]:
    """3-Layer 티어드 파이프라인 실행."""
    from src.domain.futures.strategy.config import resolve_purge_and_embargo_bars

    if caps is None:
        caps = PortfolioCaps(
            gross=3.0,
            per_symbol=0.15,
            net=0.5,
            beta=1.0,
            target_ann_vol=0.20,
        )

    purge_bars, embargo_bars = resolve_purge_and_embargo_bars(cfg)
    n_bars = len(aligned.datetimes)

    _is_ts = pd.Timestamp(window.l1_start, tz="UTC")
    _oos_ts = pd.Timestamp(window.l2_start, tz="UTC")
    l1_start_bars = int(np.searchsorted(aligned.datetimes, np.datetime64(_is_ts.replace(tzinfo=None), "ns")))
    l1_end_bars = int(np.searchsorted(aligned.datetimes, np.datetime64(_oos_ts.replace(tzinfo=None), "ns")))

    import src.domain.futures.strategy.tiered_workflow as _tw
    
    # Layer 1 Header is already printed in opt_main_futures _run_strategy_stage
    
    t_l1 = time.perf_counter()
    outer_folds = _tw.build_l1_nested_swf_folds(
        n_bars=n_bars,
        l1_start_idx=l1_start_bars,
        l1_end_idx=l1_end_bars,
        max_label_horizon_bars=max(int(getattr(cfg, "max_holding_bars", 1)), purge_bars + embargo_bars),
        cfg=cfg,
    )
    l1 = _tw.run_l1_nested_swf(
        labeled_events=labeled_events,
        aligned=aligned,
        outer_folds=outer_folds,
        cfg=cfg,
        seed=int(getattr(cfg, "seed", 42)),
    )
    logger.debug("[perf-tiered] run_tiered_pipeline Layer 1 total took %.4fs", time.perf_counter() - t_l1)

    if not l1.gate_passed:
        logger.info("\n>> LAYER 1 RESULT: [BLOCKED] -> gate_passed=False")
        return (l1, None, None)
    
    logger.info("\n>> LAYER 1 RESULT: [PASS] -> Proceeding to Layer 2.")
    
    if target_phase == "l1":
        logger.info(">> TARGET PHASE l1 REACHED -> Stopping pipeline.")
        return (l1, None, None)

    # ─── Layer 2: AWF Portfolio Optimization ─────────────────────────────────
    logger.info(format_layer_header(2, "Portfolio Allocation & Risk Optimization"))
    t_l2 = time.perf_counter()
    awf_folds = _tw.build_walk_forward_folds(n_bars=n_bars, cfg=cfg)

    # L2 window 경계 필터링: OOS 구간이 [l2_start, holdout_start) 내로 제한
    # → l1_end_bars == l2_start bar index (Line 1068 참조)
    ho_start_idx_l2 = _date_to_idx(aligned.datetimes, window.holdout_start)
    awf_folds = tuple(
        f for f in awf_folds
        if f.oos_start >= l1_end_bars and f.oos_end <= ho_start_idx_l2
    )
    if not awf_folds:
        logger.warning(
            "[L2] build_walk_forward_folds 결과 없음: L2 단일 폴드 fallback [%d, %d)",
            l1_end_bars,
            ho_start_idx_l2,
        )
        cal_end = max(l1_end_bars - 1, 1)
        awf_folds = (WFFold(
            fit_start=0,
            fit_end=cal_end,
            cal_start=max(0, cal_end - max(1, cal_end // 5)),
            cal_end=cal_end,
            oos_start=l1_end_bars,
            oos_end=ho_start_idx_l2,
        ),)
    logger.debug(
        "[L2] AWF window: L2_start_bar=%d, ho_start_bar=%d, n_folds=%d",
        l1_end_bars,
        ho_start_idx_l2,
        len(awf_folds),
    )

    if l1.inference_artifact is None:
        raise ValueError("Layer2 requires a fitted Layer1InferenceArtifact")

    l2_config = Layer2AllocationConfig.from_mapping(l2_params)
    l2_signal_batch = predict_layer1_signals(
        artifact=l1.inference_artifact,
        candidate_events=labeled_events,
        aligned=aligned,
        start_idx=l1_end_bars,
        end_idx=ho_start_idx_l2,
        cfg=cfg,
    )
    l2 = _tw.run_l2_awf(
        signal_batch=l2_signal_batch,
        aligned=aligned,
        awf_folds=awf_folds,
        config=l2_config,
        caps=caps,
        tf=tf,
    )
    logger.debug("[perf-tiered] run_tiered_pipeline Layer 2 total took %.4fs", time.perf_counter() - t_l2)

    if not l2.gate_passed:
        logger.info("\n>> LAYER 2 RESULT: [BLOCKED] -> gate_passed=False")
        return (l1, l2, None)
    
    logger.info("\n>> LAYER 2 RESULT: [PASS] -> Proceeding to Final Holdout.")
    
    if target_phase == "l2":
        logger.info(">> TARGET PHASE l2 REACHED -> Stopping pipeline.")
        return (l1, l2, None)

    # ─── Layer 3: Final Holdout Backtest ─────────────────────────────────────
    logger.info(format_layer_header(3, "Final Holdout & Deployment Readiness"))
    t_l3 = time.perf_counter()
    ho_start_idx = _date_to_idx(aligned.datetimes, window.holdout_start)
    ho_end_idx = _date_to_idx(aligned.datetimes, window.holdout_end)
    l3_signal_batch = predict_layer1_signals(
        artifact=l1.inference_artifact,
        candidate_events=labeled_events,
        aligned=aligned,
        start_idx=ho_start_idx,
        end_idx=ho_end_idx,
        cfg=cfg,
    )
    l3 = _tw.run_l3_holdout(
        signal_batch=l3_signal_batch,
        aligned=aligned,
        holdout_span=(ho_start_idx, ho_end_idx),
        config=l2_config,
        caps=caps,
        tf=tf,
    )
    logger.debug("[perf-tiered] run_tiered_pipeline Layer 3 total took %.4fs", time.perf_counter() - t_l3)
    
    logger.info("\n" + "="*80)
    logger.info("[FINAL PIPELINE STATUS]")
    logger.info(f">> ROUTING: L1({'PASS' if l1.gate_passed else 'FAIL'}) -> "
                f"L2({'PASS' if l2.gate_passed else 'FAIL'}) -> "
                f"L3({'PASS' if l3.gate_passed else 'FAIL'})")
    if l3.gate_passed:
        logger.info(">> ACTION:  DEPLOYMENT ELIGIBLE 🚀")
    else:
        logger.info(">> ACTION:  REJECTED - Fails Final Holdout Gate")
    logger.info("="*80)

    return (l1, l2, l3)
