from __future__ import annotations

import logging
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

if TYPE_CHECKING:
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.strategy.walk_forward import WFFold

_logger = logging.getLogger(__name__)


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
    outputs: list[CandidateFoldOutput] = []

    for fold_idx, fold in enumerate(folds):
        fit_span = max(0, fold.fit_end - fold.fit_start)
        early_stop_len = max(1, int(fit_span * cfg.model_early_stop_fraction))
        early_stop_start = max(fold.fit_start + 1, fold.fit_end - early_stop_len)
        train_end = max(fold.fit_start + 1, early_stop_start - cfg.purge_bars)

        # 1. Feature Schema
        schema = fit_candidate_feature_schema(
            labeled_events=labeled_events,
            cfg=cfg,
            split_start=fold.fit_start,
            split_end=train_end,
        )

        # 2. Split Datasets
        fit_set = build_candidate_dataset(
            labeled_events=labeled_events,
            aligned=aligned,
            cfg=cfg,
            schema=schema,
            split_start=fold.fit_start,
            split_end=train_end,
            is_fit_split=True,
        )
        early_stop_set = build_candidate_dataset(
            labeled_events=labeled_events,
            aligned=aligned,
            cfg=cfg,
            schema=schema,
            split_start=early_stop_start,
            split_end=fold.fit_end,
        )
        cal_fit_end = max(
            fold.cal_start + 1,
            fold.cal_start + int(
                max(1, (fold.cal_end - fold.cal_start) * cfg.calibration_fit_fraction)
            ),
        )
        calibration_fit_set = build_candidate_dataset(
            labeled_events=labeled_events,
            aligned=aligned,
            cfg=cfg,
            schema=schema,
            split_start=fold.cal_start,
            split_end=min(cal_fit_end, fold.cal_end),
        )
        calibration_eval_set = build_candidate_dataset(
            labeled_events=labeled_events,
            aligned=aligned,
            cfg=cfg,
            schema=schema,
            split_start=min(cal_fit_end, fold.cal_end),
            split_end=fold.cal_end,
        )
        oos_set = build_candidate_dataset(
            labeled_events=labeled_events,
            aligned=aligned,
            cfg=cfg,
            schema=schema,
            split_start=fold.oos_start,
            split_end=fold.oos_end,
        )

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
                source=EdgeSource.PRIOR_ONLY, prior_rank_ic=0.0, residual_rank_ic=0.0,
                incremental_log_growth_mean=0.0, incremental_log_growth_lcb=0.0,
                selected=False, reason="insufficient_observations"
            )
            ml_out = CandidateModelOutput(
                events=oos_set.event_index,
                p_pass=np.full(n_oos, 0.5, dtype=np.float64),
                gate_enabled=False,
                gate_threshold=0.5,
                edge_source=EdgeSource.PRIOR_ONLY,
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
            selected_events = select_candidate_events_for_portfolio(model_output=ml_out, cfg=cfg)
            outputs.append(CandidateFoldOutput(
                fold_id=fold_idx,
                oos_start=fold.oos_start,
                oos_end=fold.oos_end,
                model_output=ml_out,
                selected_events=selected_events,
                gate_report=gate_rep,
                edge_report=edge_rep
            ))
            continue

        # 3. Fit Gate & Edge models
        gate_model = fit_candidate_gate(
            train=fit_set,
            early_stop=early_stop_set,
            calibration=calibration_fit_set,
            calibration_eval=calibration_eval_set,
            cfg=cfg,
        )
        edge_models = fit_candidate_edge_models(
            train=fit_set,
            valid=early_stop_set,
            calibration_eval=calibration_eval_set,
            cfg=cfg,
        )

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

        edge_source = (
            EdgeSource.PRIOR_RESIDUAL
            if edge_models is not None and edge_models.target_mode == "prior_residual"
            else EdgeSource.PRIOR_ONLY
        )
        edge_val = getattr(edge_models, "validation", None)
        edge_rep = EdgeValidationReport(
            source=edge_source,
            prior_rank_ic=float(getattr(edge_val, "prior_rank_ic", 0.0)),
            residual_rank_ic=float(getattr(edge_val, "residual_rank_ic", 0.0)),
            incremental_log_growth_mean=float(getattr(edge_val, "incremental_log_growth_mean", 0.0)),
            incremental_log_growth_lcb=float(getattr(edge_val, "incremental_log_growth_lcb", 0.0)),
            selected=bool(edge_source == EdgeSource.PRIOR_RESIDUAL),
            reason=getattr(edge_val, "reason", "none")
        )

        # 5. Inference
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

        selected_events = select_candidate_events_for_portfolio(model_output=ml_out, cfg=cfg)

        outputs.append(
            CandidateFoldOutput(
                fold_id=fold_idx,
                oos_start=fold.oos_start,
                oos_end=fold.oos_end,
                model_output=ml_out,
                selected_events=selected_events,
                gate_report=gate_rep,
                edge_report=edge_rep
            )
        )

    return tuple(outputs)
