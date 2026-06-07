from __future__ import annotations

import logging
import multiprocessing
import os
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.domain.futures.strategy.candidate_contracts import (
    CandidateFoldOutput,
    CandidateModelOutput,
    EdgeSource,
    EdgeValidationReport,
    GateValidationReport,
)
from src.domain.futures.strategy.candidate_dataset import build_candidate_dataset, fit_candidate_feature_schema
from src.domain.futures.strategy.candidate_edge import fit_candidate_edge_models, predict_candidate_edges
from src.domain.futures.strategy.candidate_gate import fit_candidate_gate, predict_candidate_gate
from src.domain.futures.strategy.candidate_portfolio import select_candidate_events_for_portfolio
from src.domain.futures.strategy.config import resolve_purge_and_embargo_bars, with_max_holding_bars

if TYPE_CHECKING:
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.strategy.walk_forward import WFFold

_logger = logging.getLogger(__name__)
_GLOBAL_LABELED_EVENTS: pd.DataFrame | None = None
_GLOBAL_ALIGNED: AlignedMarketData | None = None
_GLOBAL_CFG: CandidateStrategyConfig | None = None
_GLOBAL_PURGE_BARS: int | None = None


def _dataset_timing_total(timing_profile: dict[str, float] | None) -> float:
    """Return combined dataset build time from a timing profile."""
    if timing_profile is None:
        return 0.0
    return float(
        timing_profile.get("dataset_fit", 0.0)
        + timing_profile.get("dataset_early_stop", 0.0)
        + timing_profile.get("dataset_calibration_fit", 0.0)
        + timing_profile.get("dataset_calibration_eval", 0.0)
        + timing_profile.get("dataset_oos", 0.0)
    )


def _fit_and_predict_single_fold_from_globals(
    fold_idx: int,
    fold: WFFold,
) -> CandidateFoldOutput:
    """Run a fold using process-global context to avoid large IPC payloads."""
    if (
        _GLOBAL_LABELED_EVENTS is None
        or _GLOBAL_ALIGNED is None
        or _GLOBAL_CFG is None
        or _GLOBAL_PURGE_BARS is None
    ):
        raise RuntimeError("candidate workflow globals are not initialized")
    return _fit_and_predict_single_fold(
        fold_idx,
        fold,
        _GLOBAL_LABELED_EVENTS,
        _GLOBAL_ALIGNED,
        _GLOBAL_CFG,
        _GLOBAL_PURGE_BARS,
    )


def _fit_and_predict_single_fold(
    fold_idx: int,
    fold: WFFold,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    purge_bars: int,
) -> CandidateFoldOutput:
    """Run model fitting and out-of-sample prediction for a single fold."""
    t_total = time.perf_counter()
    timing_profile: dict[str, float] = {
        "schema": 0.0,
        "dataset_fit": 0.0,
        "dataset_early_stop": 0.0,
        "dataset_calibration_fit": 0.0,
        "dataset_calibration_eval": 0.0,
        "dataset_oos": 0.0,
        "gate_fit": 0.0,
        "edge_fit": 0.0,
        "inference": 0.0,
        "selection": 0.0,
        "total": 0.0,
    }
    fit_span = max(0, fold.fit_end - fold.fit_start)
    early_stop_len = max(1, int(fit_span * cfg.model_early_stop_fraction))
    early_stop_start = max(fold.fit_start + 1, fold.fit_end - early_stop_len)
    train_end = max(fold.fit_start + 1, early_stop_start - purge_bars)

    # 1. Feature Schema
    t_step = time.perf_counter()
    schema = fit_candidate_feature_schema(
        labeled_events=labeled_events,
        cfg=cfg,
        split_start=fold.fit_start,
        split_end=train_end,
    )
    timing_profile["schema"] = time.perf_counter() - t_step

    # 2. Split Datasets
    t_step = time.perf_counter()
    fit_set = build_candidate_dataset(
        labeled_events=labeled_events,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=fold.fit_start,
        split_end=train_end,
        is_fit_split=True,
    )
    timing_profile["dataset_fit"] = time.perf_counter() - t_step
    t_step = time.perf_counter()
    early_stop_set = build_candidate_dataset(
        labeled_events=labeled_events,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=early_stop_start,
        split_end=fold.fit_end,
    )
    timing_profile["dataset_early_stop"] = time.perf_counter() - t_step
    cal_fit_end = max(
        fold.cal_start + 1,
        fold.cal_start + int(
            max(1, (fold.cal_end - fold.cal_start) * cfg.calibration_fit_fraction)
        ),
    )
    t_step = time.perf_counter()
    calibration_fit_set = build_candidate_dataset(
        labeled_events=labeled_events,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=fold.cal_start,
        split_end=min(cal_fit_end, fold.cal_end),
    )
    timing_profile["dataset_calibration_fit"] = time.perf_counter() - t_step
    t_step = time.perf_counter()
    calibration_eval_set = build_candidate_dataset(
        labeled_events=labeled_events,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=min(cal_fit_end, fold.cal_end),
        split_end=fold.cal_end,
    )
    timing_profile["dataset_calibration_eval"] = time.perf_counter() - t_step
    t_step = time.perf_counter()
    oos_set = build_candidate_dataset(
        labeled_events=labeled_events,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=fold.oos_start,
        split_end=fold.oos_end,
    )
    timing_profile["dataset_oos"] = time.perf_counter() - t_step

    # ESS and Minimum sample size checks for LGBM models fitting
    n_fit = fit_set.X.shape[0] if fit_set.X is not None else 0

    if n_fit < cfg.min_fit_obs or n_fit < 2:
        _logger.warning(
            "[WORKFLOW] Fold %d skipped ML (fit=%d < 2)",
            fold_idx, n_fit
        )
        # Prior-only outputs fallback
        n_oos = oos_set.X.shape[0] if oos_set.X is not None else 0
        gate_rep = GateValidationReport(
            enabled=False, threshold=0.5, raw_brier=0.25, calibrated_brier=0.25,
            base_brier=0.25, brier_skill=0.0, roc_auc=0.5, average_precision=0.5,
            decile_lift=0.0, incremental_log_growth_lcb=0.0, reason="insufficient_observations"
        )
        edge_rep = EdgeValidationReport(
            source=EdgeSource.DISABLED, prior_rank_ic=0.0, residual_rank_ic=0.0,
            incremental_log_growth_mean=0.0, incremental_log_growth_lcb=0.0,
            selected=False, reason="insufficient_observations"
        )
        ml_out = CandidateModelOutput(
            events=oos_set.event_index,
            p_pass=np.full(n_oos, 0.5, dtype=np.float64),
            gate_enabled=False,
            gate_threshold=0.5,
            edge_source=EdgeSource.DISABLED,
            expected_return_r=np.zeros(n_oos, dtype=np.float64),
            expected_net_bps=np.zeros(n_oos, dtype=np.float64),
            q10_return_r=np.zeros(n_oos, dtype=np.float64),
            q10_net_bps=np.zeros(n_oos, dtype=np.float64),
            q90_return_r=np.zeros(n_oos, dtype=np.float64),
            q90_net_bps=np.zeros(n_oos, dtype=np.float64),
            selection_score=np.zeros(n_oos, dtype=np.float64),
            kelly_fraction=np.zeros(n_oos, dtype=np.float64),
            validation_diagnostics={}
        )
        t_step = time.perf_counter()
        selected_events = select_candidate_events_for_portfolio(model_output=ml_out, cfg=cfg)
        timing_profile["selection"] = time.perf_counter() - t_step
        timing_profile["total"] = time.perf_counter() - t_total
        return CandidateFoldOutput(
            fold_id=fold_idx,
            oos_start=fold.oos_start,
            oos_end=fold.oos_end,
            model_output=ml_out,
            selected_events=selected_events,
            gate_report=gate_rep,
            edge_report=edge_rep,
            gate_model=None,
            edge_models=None,
            fit_set=fit_set,
            calibration_set=calibration_eval_set,
            oos_set=oos_set,
            timing_profile=timing_profile,
        )

    # 3. Fit Gate & Edge models
    t_step = time.perf_counter()
    gate_model = fit_candidate_gate(
        train=fit_set,
        early_stop=early_stop_set,
        calibration=calibration_fit_set,
        calibration_eval=calibration_eval_set,
        cfg=cfg,
    )
    timing_profile["gate_fit"] = time.perf_counter() - t_step
    t_step = time.perf_counter()
    edge_models = fit_candidate_edge_models(
        train=fit_set,
        valid=early_stop_set,
        calibration_eval=calibration_eval_set,
        cfg=cfg,
    )
    timing_profile["edge_fit"] = time.perf_counter() - t_step

    # 4. Calibration Acceptance & Validation Reports
    validation = getattr(gate_model, "validation", None)
    gate_enabled = validation.enabled if validation is not None else False
    gate_threshold = validation.threshold if validation is not None else 0.5

    gate_rep = GateValidationReport(
        enabled=gate_enabled,
        threshold=gate_threshold,
        raw_brier=float(getattr(validation, "raw_brier", 0.25)),
        calibrated_brier=float(getattr(validation, "calibrated_brier", 0.25)),
        base_brier=float(getattr(validation, "base_brier", 0.25)),
        brier_skill=float(getattr(validation, "brier_skill", 0.0)),
        roc_auc=float(getattr(validation, "roc_auc", 0.5)),
        average_precision=float(getattr(validation, "average_precision", 0.5)),
        decile_lift=float(getattr(validation, "decile_lift", 0.0)),
        incremental_log_growth_lcb=float(getattr(validation, "incremental_log_growth_lcb", 0.0)),
        reason=getattr(validation, "reason", "none")
    )

    prediction_mode = edge_models.prediction_mode if edge_models is not None else "disabled"
    edge_source = {
        "disabled": EdgeSource.DISABLED,
        "direct": EdgeSource.DIRECT_MODEL,
        "prior_only": EdgeSource.PRIOR_ONLY,
        "prior_residual": EdgeSource.PRIOR_RESIDUAL,
    }[prediction_mode]
    edge_val = getattr(edge_models, "validation", None)
    edge_rep = EdgeValidationReport(
        source=edge_source,
        prior_rank_ic=float(getattr(edge_val, "prior_rank_ic", 0.0)),
        residual_rank_ic=float(getattr(edge_val, "residual_rank_ic", 0.0)),
        incremental_log_growth_mean=float(getattr(edge_val, "incremental_log_growth_mean", 0.0)),
        incremental_log_growth_lcb=float(getattr(edge_val, "incremental_log_growth_lcb", 0.0)),
        selected=bool(edge_source in {EdgeSource.DIRECT_MODEL, EdgeSource.PRIOR_RESIDUAL}),
        reason=getattr(edge_val, "reason", "none")
    )

    # 5. Inference
    t_step = time.perf_counter()
    p_pass = predict_candidate_gate(model=gate_model, dataset=oos_set, cfg=cfg)
    ml_out = predict_candidate_edges(
        models=edge_models,
        dataset=oos_set,
        p_pass=p_pass,
        cfg=cfg,
        gate_enabled=gate_enabled,
        gate_threshold=gate_threshold,
        edge_source=edge_source,
    )
    timing_profile["inference"] = time.perf_counter() - t_step

    t_step = time.perf_counter()
    selected_events = select_candidate_events_for_portfolio(model_output=ml_out, cfg=cfg)
    timing_profile["selection"] = time.perf_counter() - t_step
    timing_profile["total"] = time.perf_counter() - t_total

    return CandidateFoldOutput(
        fold_id=fold_idx,
        oos_start=fold.oos_start,
        oos_end=fold.oos_end,
        model_output=ml_out,
        selected_events=selected_events,
        gate_report=gate_rep,
        edge_report=edge_rep,
        gate_model=gate_model,
        edge_models=edge_models,
        fit_set=fit_set,
        calibration_set=calibration_eval_set,
        oos_set=oos_set,
        timing_profile=timing_profile,
    )


def run_candidate_walk_forward(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    folds: tuple[WFFold, ...],
) -> tuple[CandidateFoldOutput, ...]:
    """Orchestrate training, validation, gate/edge calibration/acceptance, and OOS prediction per fold.

    Prevents redundant model fitting between bridge and ablation.
    """
    from concurrent.futures import ProcessPoolExecutor

    max_holding_bars = (
        int(pd.to_numeric(labeled_events["expected_holding_bars"], errors="coerce").max())
        if not labeled_events.empty and "expected_holding_bars" in labeled_events.columns
        else None
    )
    resolved_cfg = with_max_holding_bars(cfg, max_holding_bars=max_holding_bars)
    purge_bars, _ = resolve_purge_and_embargo_bars(resolved_cfg)
    planned_workers = max(1, (os.cpu_count() or 4) // 2)
    max_workers = min(len(folds), planned_workers)

    if max_workers <= 1 or len(folds) <= 1:
        mode = "sequential"
        outputs = []
        for fold_idx, fold in enumerate(folds):
            outputs.append(
                _fit_and_predict_single_fold(
                    fold_idx, fold, labeled_events, aligned, cfg, purge_bars
                )
            )
    else:
        mode = "process_pool"
        global _GLOBAL_LABELED_EVENTS, _GLOBAL_ALIGNED, _GLOBAL_CFG, _GLOBAL_PURGE_BARS
        _GLOBAL_LABELED_EVENTS = labeled_events
        _GLOBAL_ALIGNED = aligned
        _GLOBAL_CFG = cfg
        _GLOBAL_PURGE_BARS = purge_bars
        mp_ctx = multiprocessing.get_context("fork")
        try:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
                futures = [
                    executor.submit(
                        _fit_and_predict_single_fold_from_globals,
                        fold_idx,
                        fold,
                    )
                    for fold_idx, fold in enumerate(folds)
                ]
                outputs = [f.result() for f in futures]
        finally:
            _GLOBAL_LABELED_EVENTS = None
            _GLOBAL_ALIGNED = None
            _GLOBAL_CFG = None
            _GLOBAL_PURGE_BARS = None

    total_vals = [float(out.timing_profile.get("total", 0.0)) for out in outputs if out.timing_profile]
    gate_vals = [float(out.timing_profile.get("gate_fit", 0.0)) for out in outputs if out.timing_profile]
    edge_vals = [float(out.timing_profile.get("edge_fit", 0.0)) for out in outputs if out.timing_profile]
    dataset_vals = [_dataset_timing_total(out.timing_profile) for out in outputs if out.timing_profile]
    _logger.info(
        (
            "[WF-PROF] mode=%s workers=%d folds=%d total_mean=%.4fs total_max=%.4fs "
            "gate_mean=%.4fs gate_max=%.4fs edge_mean=%.4fs edge_max=%.4fs "
            "dataset_mean=%.4fs dataset_max=%.4fs"
        ),
        mode,
        max_workers,
        len(folds),
        float(np.mean(total_vals)) if total_vals else 0.0,
        float(np.max(total_vals)) if total_vals else 0.0,
        float(np.mean(gate_vals)) if gate_vals else 0.0,
        float(np.max(gate_vals)) if gate_vals else 0.0,
        float(np.mean(edge_vals)) if edge_vals else 0.0,
        float(np.max(edge_vals)) if edge_vals else 0.0,
        float(np.mean(dataset_vals)) if dataset_vals else 0.0,
        float(np.max(dataset_vals)) if dataset_vals else 0.0,
    )

    return tuple(outputs)
