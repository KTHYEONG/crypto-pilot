from __future__ import annotations

import logging
import multiprocessing
import os
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from src.core.utils.utils import PERF
from src.domain.futures.strategy.candidate_contracts import (
    CandidateFoldOutput,
    CandidateModelOutput,
    EdgeSource,
    EdgeValidationReport,
    FoldFitStatus,
    GateValidationReport,
)
from src.domain.futures.strategy.candidate_dataset import (
    CandidateDataset,
    CandidateFeatureSchema,
    PreparedLabeledEvents,
    build_candidate_dataset,
    fit_candidate_feature_schema,
)
from src.domain.futures.strategy.candidate_ensemble import (
    fit_regime_conditional_ensemble,
    predict_regime_conditional_ensemble,
)
from src.domain.futures.strategy.candidate_portfolio import select_candidate_events_for_portfolio
from src.domain.futures.strategy.config import resolve_purge_and_embargo_bars, with_max_holding_bars

if TYPE_CHECKING:
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.strategy.walk_forward import WFFold

_logger = logging.getLogger(__name__)
_GLOBAL_LABELED_EVENTS: pd.DataFrame | None = None
_GLOBAL_PREPARED_EVENTS: PreparedLabeledEvents | None = None
_GLOBAL_ALIGNED: AlignedMarketData | None = None
_GLOBAL_CFG: CandidateStrategyConfig | None = None
_GLOBAL_PURGE_BARS: int | None = None


def _resolve_fold_fit_status(
    *,
    n_fit: int,
    min_fit_obs: int,
    n_oos: int,
    prediction: NDArray[np.float64] | None,
) -> tuple[FoldFitStatus, str | None]:
    """Classify fold fit/prediction viability for downstream filtering."""
    if n_fit < max(2, min_fit_obs):
        return ("insufficient_fit", "insufficient_observations")
    if n_oos < 1:
        return ("empty_oos", "empty_oos")
    if prediction is None or prediction.size < 1:
        return ("empty_oos", "empty_oos")
    if float(np.nanstd(prediction.astype(np.float64))) <= 0.0:
        return ("constant_prediction", "constant_prediction")
    return ("trained", None)


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


def _empty_candidate_dataset(schema: CandidateFeatureSchema | object) -> CandidateDataset:
    raw_feature_names = getattr(schema, "feature_names", ())
    feature_names = tuple(str(name) for name in raw_feature_names)
    schema_version = str(getattr(schema, "version", "candidate_v6"))
    return CandidateDataset(
        X=np.zeros((0, len(feature_names)), dtype=np.float32),
        y_gate=np.zeros((0,), dtype=np.int8),
        y_edge_bps=np.zeros((0,), dtype=np.float32),
        y_q10_bps=np.zeros((0,), dtype=np.float32),
        y_mfe_bps=np.zeros((0,), dtype=np.float32),
        gate_weight=np.zeros((0,), dtype=np.float32),
        edge_weight=np.zeros((0,), dtype=np.float32),
        groups=np.zeros((0,), dtype=np.int32),
        event_index=pd.DataFrame(),
        feature_names=feature_names,
        effective_sample_size=0.0,
        feature_schema_version=schema_version,
        y_return_r=np.zeros((0,), dtype=np.float32),
        y_return_bps=np.zeros((0,), dtype=np.float32),
        y_gross_return_bps=np.zeros((0,), dtype=np.float32),
        y_gross_return_r=np.zeros((0,), dtype=np.float32),
        y_mae_r=np.zeros((0,), dtype=np.float32),
        risk_unit_bps=np.zeros((0,), dtype=np.float32),
    )


def compact_candidate_fold_output(
    fold_out: CandidateFoldOutput,
    *,
    drop_models: bool = True,
    drop_datasets: bool = True,
) -> CandidateFoldOutput:
    """Return a lightweight fold output for nested L1 IPC handoff."""
    oos_set = fold_out.oos_set
    events = fold_out.model_output.events.copy()
    if not events.empty:
        size = len(events)
        if oos_set is not None:
            y_return = getattr(oos_set, "y_return_bps", None)
            if y_return is not None:
                arr = np.asarray(y_return, dtype=np.float64)
                if arr.size >= 1:
                    size = min(size, int(arr.size))
                    events = events.iloc[:size].copy()
                    events["gross_event_bps"] = arr[:size]
            edge_weight = getattr(oos_set, "edge_weight", None)
            if edge_weight is not None:
                arr = np.asarray(edge_weight, dtype=np.float64)
                if arr.size >= len(events):
                    events["uniqueness_weight"] = arr[: len(events)]

    model_output = fold_out.model_output
    if drop_models:
        model_output = CandidateModelOutput(
            events=events,
            p_pass=np.asarray(fold_out.model_output.p_pass, dtype=np.float64),
            gate_enabled=bool(fold_out.model_output.gate_enabled),
            gate_threshold=float(fold_out.model_output.gate_threshold),
            edge_source=fold_out.model_output.edge_source,
            expected_return_r=np.asarray(fold_out.model_output.expected_return_r, dtype=np.float64),
            expected_net_bps=np.asarray(fold_out.model_output.expected_net_bps, dtype=np.float64),
            expected_gross_bps=np.asarray(fold_out.model_output.expected_gross_bps, dtype=np.float64),
            q10_return_r=np.asarray(fold_out.model_output.q10_return_r, dtype=np.float64),
            q10_net_bps=np.asarray(fold_out.model_output.q10_net_bps, dtype=np.float64),
            q10_gross_bps=np.asarray(fold_out.model_output.q10_gross_bps, dtype=np.float64),
            q90_return_r=np.asarray(fold_out.model_output.q90_return_r, dtype=np.float64),
            q90_net_bps=np.asarray(fold_out.model_output.q90_net_bps, dtype=np.float64),
            q90_gross_bps=np.asarray(fold_out.model_output.q90_gross_bps, dtype=np.float64),
            selection_score=np.asarray(fold_out.model_output.selection_score, dtype=np.float64),
            kelly_fraction=np.asarray(fold_out.model_output.kelly_fraction, dtype=np.float64),
            prediction_scale_bps=np.asarray(fold_out.model_output.prediction_scale_bps, dtype=np.float64),
            validation_diagnostics=fold_out.model_output.validation_diagnostics,
        )

    return CandidateFoldOutput(
        fold_id=fold_out.fold_id,
        oos_start=fold_out.oos_start,
        oos_end=fold_out.oos_end,
        model_output=model_output,
        selected_events=fold_out.selected_events,
        gate_report=fold_out.gate_report,
        edge_report=fold_out.edge_report,
        fit_status=fold_out.fit_status,
        n_fit=fold_out.n_fit,
        skip_reason=fold_out.skip_reason,
        gate_model=None if drop_models else fold_out.gate_model,
        edge_models=None if drop_models else fold_out.edge_models,
        fit_set=None if drop_datasets else fold_out.fit_set,
        calibration_set=None if drop_datasets else fold_out.calibration_set,
        oos_set=None if drop_datasets else fold_out.oos_set,
        timing_profile=fold_out.timing_profile,
    )


def _fit_and_predict_single_fold_from_globals(
    fold_idx: int,
    fold: WFFold,
    is_evidence_fold: bool = False,
    compact_result: bool = False,
) -> CandidateFoldOutput:
    """Run a fold using process-global context to avoid large IPC payloads."""
    if (
        (_GLOBAL_LABELED_EVENTS is None and _GLOBAL_PREPARED_EVENTS is None)
        or _GLOBAL_ALIGNED is None
        or _GLOBAL_CFG is None
        or _GLOBAL_PURGE_BARS is None
    ):
        raise RuntimeError("candidate workflow globals are not initialized")

    import gc
    gc.disable()
    try:
        res = _fit_and_predict_single_fold(
            fold_idx,
            fold,
            _GLOBAL_PREPARED_EVENTS if _GLOBAL_PREPARED_EVENTS is not None else _GLOBAL_LABELED_EVENTS,
            _GLOBAL_ALIGNED,
            _GLOBAL_CFG,
            _GLOBAL_PURGE_BARS,
            is_evidence_fold=is_evidence_fold,
            compact_result=compact_result,
        )
        return res
    finally:
        gc.enable()
        gc.collect()


def _fit_and_predict_single_fold(
    fold_idx: int,
    fold: WFFold,
    labeled_events: pd.DataFrame | PreparedLabeledEvents,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    purge_bars: int,
    is_evidence_fold: bool = False,
    compact_result: bool = False,
) -> CandidateFoldOutput:
    """Run model fitting and out-of-sample prediction for a single fold."""
    with threadpool_limits(limits=1, user_api="blas"):
        return _fit_and_predict_single_fold_inner(
            fold_idx, fold, labeled_events, aligned, cfg, purge_bars,
            is_evidence_fold=is_evidence_fold,
            compact_result=compact_result,
        )


def _fit_and_predict_single_fold_inner(
    fold_idx: int,
    fold: WFFold,
    labeled_events: pd.DataFrame | PreparedLabeledEvents,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    purge_bars: int,
    is_evidence_fold: bool = False,
    compact_result: bool = False,
) -> CandidateFoldOutput:
    """Inner fold execution under BLAS single-thread context (called from _fit_and_predict_single_fold)."""
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
    boundary_mode = str(getattr(cfg, "l1_boundary_mode", "exact_label_interval"))
    boundary_buffer = max(0, int(getattr(cfg, "l1_boundary_buffer_bars", 0)))

    def _label_end(boundary_end: int) -> int | None:
        if boundary_mode != "exact_label_interval":
            return None
        return max(boundary_end - boundary_buffer, 0)

    fit_span = max(0, fold.fit_end - fold.fit_start)
    early_stop_len = max(1, int(fit_span * cfg.model_early_stop_fraction))
    if boundary_mode == "exact_label_interval":
        if cfg.allocation_backend == "ensemble_b0":
            early_stop_start = fold.fit_end
            train_end = fold.fit_end
        else:
            early_stop_start = max(fold.fit_start + 1, fold.fit_end - early_stop_len)
            train_end = early_stop_start
    else:
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

    skip_feat = (cfg.allocation_backend == "ensemble_b0")

    def _build_window(
        *,
        split_start: int,
        split_end: int,
        label_end_exclusive: int | None,
        is_fit_split: bool = False,
    ) -> CandidateDataset:
        if split_end <= split_start:
            return _empty_candidate_dataset(schema)
        return build_candidate_dataset(
            labeled_events=labeled_events,
            aligned=aligned,
            cfg=cfg,
            schema=schema,
            split_start=split_start,
            split_end=split_end,
            label_end_exclusive=label_end_exclusive,
            is_fit_split=is_fit_split,
            skip_features=skip_feat,
        )

    # 2. Split Datasets
    t_step = time.perf_counter()
    fit_set = _build_window(
        split_start=fold.fit_start,
        split_end=train_end,
        label_end_exclusive=_label_end(train_end),
        is_fit_split=True,
    )
    timing_profile["dataset_fit"] = time.perf_counter() - t_step
    t_step = time.perf_counter()
    early_stop_set = _build_window(
        split_start=early_stop_start,
        split_end=fold.fit_end,
        label_end_exclusive=_label_end(fold.fit_end),
    )
    timing_profile["dataset_early_stop"] = time.perf_counter() - t_step
    cal_fit_end = max(
        fold.cal_start + 1,
        fold.cal_start + int(
            max(1, (fold.cal_end - fold.cal_start) * cfg.calibration_fit_fraction)
        ),
    )
    t_step = time.perf_counter()
    calibration_fit_set = _build_window(
        split_start=fold.cal_start,
        split_end=min(cal_fit_end, fold.cal_end),
        label_end_exclusive=_label_end(min(cal_fit_end, fold.cal_end)),
    )
    timing_profile["dataset_calibration_fit"] = time.perf_counter() - t_step
    t_step = time.perf_counter()
    calibration_eval_set = _build_window(
        split_start=min(cal_fit_end, fold.cal_end),
        split_end=fold.cal_end,
        label_end_exclusive=_label_end(fold.cal_end),
    )
    timing_profile["dataset_calibration_eval"] = time.perf_counter() - t_step
    t_step = time.perf_counter()
    oos_set = _build_window(
        split_start=fold.oos_start,
        split_end=fold.oos_end,
        label_end_exclusive=_label_end(fold.oos_end),
    )
    timing_profile["dataset_oos"] = time.perf_counter() - t_step

    # ESS and Minimum sample size checks for LGBM models fitting
    n_fit = fit_set.X.shape[0] if fit_set.X is not None else 0

    if n_fit < cfg.min_fit_obs or n_fit < 2:
        if not is_evidence_fold:
            _logger.warning(
                "[WORKFLOW] Fold %d skipped Ensemble (fit=%d < 2)",
                fold_idx, n_fit,
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
            expected_gross_bps=np.zeros(n_oos, dtype=np.float64),
            q10_return_r=np.zeros(n_oos, dtype=np.float64),
            q10_net_bps=np.zeros(n_oos, dtype=np.float64),
            q10_gross_bps=np.zeros(n_oos, dtype=np.float64),
            q90_return_r=np.zeros(n_oos, dtype=np.float64),
            q90_net_bps=np.zeros(n_oos, dtype=np.float64),
            q90_gross_bps=np.zeros(n_oos, dtype=np.float64),
            selection_score=np.zeros(n_oos, dtype=np.float64),
            kelly_fraction=np.zeros(n_oos, dtype=np.float64),
            validation_diagnostics={}
        )
        t_step = time.perf_counter()
        selected_events = select_candidate_events_for_portfolio(model_output=ml_out, cfg=cfg)
        timing_profile["selection"] = time.perf_counter() - t_step
        timing_profile["total"] = time.perf_counter() - t_total
        fold_out = CandidateFoldOutput(
            fold_id=fold_idx,
            oos_start=fold.oos_start,
            oos_end=fold.oos_end,
            model_output=ml_out,
            selected_events=selected_events,
            gate_report=gate_rep,
            edge_report=edge_rep,
            fit_status="insufficient_fit",
            n_fit=n_fit,
            skip_reason="insufficient_observations",
            gate_model=None,
            edge_models=None,
            fit_set=fit_set,
            calibration_set=calibration_eval_set,
            oos_set=oos_set,
            timing_profile=timing_profile,
        )
        return compact_candidate_fold_output(fold_out) if compact_result else fold_out

    gate_model = None
    edge_models = None
    if cfg.allocation_backend == "ensemble_b0":
        train_events = fit_set.event_index.copy()
        train_events["net_return_bps"] = (
            fit_set.y_return_bps
            if fit_set.y_return_bps is not None
            else fit_set.y_edge_bps
        )
        proof_events: pd.DataFrame | None = None
        proof_fold_ids: np.ndarray | None = None
        if not calibration_eval_set.event_index.empty:
            proof_events = calibration_eval_set.event_index.copy()
            proof_events["net_return_bps"] = (
                calibration_eval_set.y_return_bps
                if calibration_eval_set.y_return_bps is not None
                else calibration_eval_set.y_edge_bps
            )
            proof_fold_ids = np.full(proof_events.shape[0], fold_idx, dtype=np.int32)
        t_step = time.perf_counter()
        ensemble_model = fit_regime_conditional_ensemble(
            train_events=train_events,
            cfg=cfg,
            oos_proof_events=proof_events,
            fold_ids=proof_fold_ids,
        )
        timing_profile["edge_fit"] = time.perf_counter() - t_step

        t_step = time.perf_counter()
        ml_out = predict_regime_conditional_ensemble(model=ensemble_model, oos_events=oos_set.event_index, cfg=cfg)
        timing_profile["inference"] = time.perf_counter() - t_step

        # --- Capture Ensemble Diagnostics for aggregation ---
        diag_data = getattr(ensemble_model, "ensemble_diagnostics", {})
        if diag_data:
            ml_out.validation_diagnostics["ensemble_diagnostics"] = diag_data

        # --- Calculate Rank IC for ensemble_b0 on OOS ---
        from src.domain.futures.strategy.candidate_edge import _rank_ic
        pred_oos = ml_out.expected_net_bps
        realized_oos = np.asarray(
            oos_set.y_return_bps
            if oos_set.y_return_bps is not None
            else (oos_set.y_edge_bps if oos_set.y_edge_bps is not None else np.zeros_like(pred_oos)),
            dtype=np.float64,
        )
        rank_ic_val = _rank_ic(pred_oos, realized_oos) if pred_oos.size >= 2 else 0.0
        if not np.isfinite(rank_ic_val):
            rank_ic_val = 0.0

        # Update model output validation diagnostics for logging parity
        ml_out.validation_diagnostics["prediction_mode"] = "ensemble_b0"
        ml_out.validation_diagnostics["prior_component_p90_bps"] = (
            float(np.percentile(pred_oos, 90)) if pred_oos.size > 0 else 0.0
        )
        _conditioning = getattr(ensemble_model, "conditioning", "ensemble_b0")
        _val_ic = float(getattr(ensemble_model, "validation_rank_ic", 0.0))
        _lam_value = ml_out.validation_diagnostics.get("mu_shrinkage_lambda", 1.0)
        _lam = (
            float(_lam_value)
            if isinstance(_lam_value, (int, float, np.integer, np.floating))
            else 1.0
        )
        ml_out.validation_diagnostics["conditioning"] = _conditioning
        ml_out.validation_diagnostics["val_rank_ic"] = _val_ic
        ml_out.validation_diagnostics["oos_rank_ic"] = rank_ic_val
        ml_out.validation_diagnostics["mu_shrinkage_lambda"] = _lam

        gate_rep = GateValidationReport(
            enabled=False,
            threshold=0.0,
            raw_brier=0.25,
            calibrated_brier=0.25,
            base_brier=0.25,
            brier_skill=0.0,
            roc_auc=0.5,
            average_precision=0.5,
            decile_lift=0.0,
            incremental_log_growth_lcb=0.0,
            reason="ensemble_b0",
        )
        edge_rep = EdgeValidationReport(
            source=EdgeSource.PRIOR_ONLY,
            prior_rank_ic=float(rank_ic_val),
            residual_rank_ic=0.0,
            incremental_log_growth_mean=0.0,
            incremental_log_growth_lcb=0.0,
            selected=True,
            reason=f"ensemble_b0:{_conditioning}",
        )
    else:
        from src.domain.futures.strategy.candidate_edge import fit_candidate_edge_models, predict_candidate_edges
        from src.domain.futures.strategy.candidate_gate import fit_candidate_gate, predict_candidate_gate

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
            reason=getattr(validation, "reason", "none"),
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
            reason=getattr(edge_val, "reason", "none"),
        )

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

    fit_status, skip_reason = _resolve_fold_fit_status(
        n_fit=n_fit,
        min_fit_obs=cfg.min_fit_obs,
        n_oos=len(ml_out.expected_net_bps),
        prediction=np.asarray(ml_out.expected_net_bps, dtype=np.float64),
    )

    _logger.log(PERF,
        "[CANDIDATE-FOLD] fold=%d fit_status=%s total=%.3fs schema=%.3fs ds_fit=%.3fs ds_es=%.3fs "
        "ds_cal_fit=%.3fs ds_cal_eval=%.3fs ds_oos=%.3fs edge_fit=%.3fs inference=%.3fs selection=%.3fs",
        fold_idx, fit_status,
        timing_profile.get("total", 0),
        timing_profile.get("schema", 0),
        timing_profile.get("dataset_fit", 0),
        timing_profile.get("dataset_early_stop", 0),
        timing_profile.get("dataset_calibration_fit", 0),
        timing_profile.get("dataset_calibration_eval", 0),
        timing_profile.get("dataset_oos", 0),
        timing_profile.get("edge_fit", 0),
        timing_profile.get("inference", 0),
        timing_profile.get("selection", 0),
    )

    fold_out = CandidateFoldOutput(
        fold_id=fold_idx,
        oos_start=fold.oos_start,
        oos_end=fold.oos_end,
        model_output=ml_out,
        selected_events=selected_events,
        gate_report=gate_rep,
        edge_report=edge_rep,
        fit_status=fit_status,
        n_fit=n_fit,
        skip_reason=skip_reason,
        gate_model=gate_model,
        edge_models=edge_models,
        fit_set=fit_set,
        calibration_set=calibration_eval_set,
        oos_set=oos_set,
        timing_profile=timing_profile,
    )
    return compact_candidate_fold_output(fold_out) if compact_result else fold_out


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
