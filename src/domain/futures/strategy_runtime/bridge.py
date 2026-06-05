from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.domain.futures.strategy.config import StrategyConfig

_logger = logging.getLogger(__name__)


def _candidate_ml_split_indices(
    *,
    n_bars: int,
    fit_fraction: float,
    calibration_fraction: float,
    purge_bars: int,
    embargo_bars: int,
) -> tuple[int, int, int, int, int, int]:
    """Return fit, calibration, and OOS index ranges."""
    fit_start = 0
    fit_end = int(n_bars * fit_fraction)
    calibration_start = fit_end + purge_bars
    calibration_end = int(n_bars * (fit_fraction + calibration_fraction))
    oos_start = calibration_end + embargo_bars
    oos_end = n_bars
    if not (fit_start < fit_end <= n_bars):
        raise ValueError("fit split is empty or invalid")
    if not (calibration_start < calibration_end <= n_bars):
        raise ValueError("calibration split is empty or invalid")
    if not (oos_start < oos_end <= n_bars):
        raise ValueError("oos split is empty or invalid")
    return fit_start, fit_end, calibration_start, calibration_end, oos_start, oos_end


def _finite_summary(values: np.ndarray) -> dict[str, float]:
    """Return finite mean/median/p90/min/max statistics for a numeric array."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _threshold_rate(values: np.ndarray, threshold: float) -> float:
    """Return fraction of finite values above a threshold."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float((finite >= threshold).mean())


def _recommendation_window_indices(
    *,
    fit_start: int,
    fit_end: int,
    calibration_start: int,
    calibration_end: int,
    cfg: Any,
) -> tuple[int, int]:
    """Return the contiguous recommendation window to evaluate for promotion."""
    basis = str(getattr(cfg, "promotion_decision_split", "fit_calibration"))
    if basis == "fit":
        return fit_start, fit_end
    if basis == "calibration":
        return calibration_start, calibration_end
    if basis == "fit_calibration":
        return fit_start, calibration_end
    raise ValueError(f"unsupported promotion_decision_split: {basis}")


@dataclass(slots=True)
class CandidatePipelineOutput:
    """Candidate strategy bridge output."""

    alpha_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_weights: np.ndarray | None = None
    rule_report: dict[str, Any] | None = None


def run_candidate_strategy_for_universe(
    symbols: list[str],
    tf: str,
    *,
    strategy_cfg: StrategyConfig | None = None,
    preloaded_data_maps: dict[str, dict[str, Any]] | None = None,
    silent: bool = False,
) -> CandidatePipelineOutput:
    """Run candidate strategy pipeline and return candidate output."""
    if strategy_cfg is None or preloaded_data_maps is None:
        return CandidatePipelineOutput()

    from dataclasses import replace

    from src.domain.futures.strategy.ablation import apply_variant_promotions, validate_candidate_signals
    from src.domain.futures.strategy.candidate_dataset import build_candidate_dataset
    from src.domain.futures.strategy.candidate_edge import fit_candidate_edge_models, predict_candidate_edges
    from src.domain.futures.strategy.candidate_gate import fit_candidate_gate, predict_candidate_gate
    from src.domain.futures.strategy.candidate_labels import label_candidate_events
    from src.domain.futures.strategy.candidate_portfolio import (
        build_candidate_alpha_panel,
        build_candidate_target_weights,
        select_candidate_events_for_portfolio,
    )
    from src.domain.futures.strategy.common.alignment import align_data_maps
    from src.domain.futures.strategy.rule_diagnostics import compute_rule_diagnostics
    from src.domain.futures.strategy.rule_signals import (
        build_rule_signal_panels,
        candidate_panels_to_events,
    )
    from src.domain.futures.strategy.walk_forward import build_walk_forward_folds

    aligned = align_data_maps(preloaded_data_maps, symbols, tf)
    n_bars = aligned.close_2d.shape[0]

    panels = build_rule_signal_panels(aligned=aligned, cfg=strategy_cfg.candidate)
    raw_events = candidate_panels_to_events(
        panels,
        min_abs_score=strategy_cfg.candidate.min_rule_net_bps * 1e-4,
        side_flip_variants=strategy_cfg.candidate.side_flip_candidate_variants,
        cost_floor_bps=strategy_cfg.candidate.cost_floor_bps,
        execution_cost_bps_2d=aligned.execution_cost_bps_2d,
    )

    if raw_events.empty:
        alpha_panel = build_candidate_alpha_panel(
            selected_events=raw_events,
            target_weights_2d=np.zeros_like(aligned.close_2d),
            datetimes=aligned.datetimes,
            symbols=tuple(symbols),
        )
        return CandidatePipelineOutput(
            alpha_panel=alpha_panel,
            target_weights=np.zeros_like(aligned.close_2d),
            rule_report={
                "events_total": 0,
                "labeled_total": 0,
                "promoted_total": 0,
                "fit_total": 0,
                "calibration_total": 0,
                "oos_total": 0,
                "selected_pre_group": 0,
                "selected_total": 0,
                "eligible": 0,
                "n_keep": 0,
                "policy": strategy_cfg.candidate.selection_policy,
                "zero_reason": "no_events",
                "gate_calibration_used": False,
                "gate_calibration_reason": "no_events",
                "recommended_keep_variants": (),
                "recommended_flip_variants": (),
            },
        )

    labeled = label_candidate_events(events=raw_events, aligned=aligned, cfg=strategy_cfg.candidate)
    fit_start, fit_end, calibration_start, calibration_end, oos_start, oos_end = _candidate_ml_split_indices(
        n_bars=n_bars,
        fit_fraction=strategy_cfg.candidate.ml_fit_fraction,
        calibration_fraction=strategy_cfg.candidate.ml_calibration_fraction,
        purge_bars=strategy_cfg.candidate.purge_bars,
        embargo_bars=strategy_cfg.candidate.embargo_bars,
    )
    recommendation_start, recommendation_end = _recommendation_window_indices(
        fit_start=fit_start,
        fit_end=fit_end,
        calibration_start=calibration_start,
        calibration_end=calibration_end,
        cfg=strategy_cfg.candidate,
    )
    if strategy_cfg.candidate.promotion_decision_split == "fit_calibration":
        entry_idx = pd.to_numeric(labeled["entry_idx"], errors="coerce")
        labeled_for_diag = labeled.loc[(entry_idx < fit_end) | (entry_idx >= calibration_start)].copy()
    else:
        labeled_for_diag = labeled

    diag = compute_rule_diagnostics(
        labeled_events=labeled_for_diag,
        aligned=aligned,
        cfg=strategy_cfg.candidate,
        min_obs=max(strategy_cfg.candidate.min_candidate_obs, 100),
        silent=silent,
        recommendation_start=recommendation_start,
        recommendation_end=recommendation_end,
        report_start=oos_start,
        report_end=oos_end,
    )

    if strategy_cfg.candidate.promotion_filter_enabled:
        labeled = apply_variant_promotions(
            labeled=labeled,
            keep_variants=diag.recommended_keep_variants,
            flip_variants=diag.recommended_flip_variants,
        )
        if labeled.empty:
            _logger.debug(
                "[BRIDGE] all candidate variants blocked by promotion filter; producing zero weights"
            )
            alpha_panel = build_candidate_alpha_panel(
                selected_events=pd.DataFrame(),
                target_weights_2d=np.zeros_like(aligned.close_2d),
                datetimes=aligned.datetimes,
                symbols=tuple(symbols),
            )
            return CandidatePipelineOutput(
                alpha_panel=alpha_panel,
                target_weights=np.zeros_like(aligned.close_2d),
                rule_report={
                    "events_total": len(raw_events),
                    "labeled_total": len(labeled),
                    "promoted_total": 0,
                    "fit_total": 0,
                    "calibration_total": 0,
                    "oos_total": 0,
                    "selected_pre_group": 0,
                    "selected_total": 0,
                    "eligible": 0,
                    "n_keep": 0,
                    "policy": strategy_cfg.candidate.selection_policy,
                    "zero_reason": "promotion_filter_empty",
                    "gate_calibration_used": False,
                    "gate_calibration_reason": "promotion_filter_empty",
                    "recommended_keep_variants": diag.recommended_keep_variants,
                    "recommended_flip_variants": diag.recommended_flip_variants,
                },
            )
    promoted_total = len(labeled)

    # Compute split indices needed for signal_only + WF (done once for OOS window bounds)
    if strategy_cfg.candidate.wf_enabled and strategy_cfg.candidate.wf_scheme != "single":
        _folds = build_walk_forward_folds(n_bars=n_bars, cfg=strategy_cfg.candidate)
        _oos_start_ref = _folds[0].oos_start if _folds else 0
        _oos_end_ref = _folds[-1].oos_end if _folds else n_bars
    else:
        _s = _candidate_ml_split_indices(
            n_bars=n_bars,
            fit_fraction=strategy_cfg.candidate.ml_fit_fraction,
            calibration_fraction=strategy_cfg.candidate.ml_calibration_fraction,
            purge_bars=strategy_cfg.candidate.purge_bars,
            embargo_bars=strategy_cfg.candidate.embargo_bars,
        )
        _oos_start_ref, _oos_end_ref = _s[4], _s[5]
        _folds = None  # single-fold path handled below

    # signal_only: validate rule signals, skip ML training
    if strategy_cfg.candidate.signal_only:
        signal_reports = validate_candidate_signals(
            labeled=labeled,
            diag=diag,
            cfg=strategy_cfg.candidate,
            oos_start=_oos_start_ref,
            oos_end=_oos_end_ref,
        )
        any_passes = any(r.survives_cost for r in signal_reports)
        _logger.info(
            "[SIGNAL-VALIDATION] variants=%d any_passes=%s",
            len(signal_reports),
            any_passes,
        )
        for rpt in signal_reports:
            _logger.info(
                "[SIGNAL-VALIDATION] variant=%s n=%d net_p50=%.1f stress_p50=%.1f "
                "hit=%.3f t=%.2f survives=%s",
                rpt.variant, rpt.n_events, rpt.net_edge_bps_p50,
                rpt.net_edge_bps_stress_p50, rpt.hit_rate, rpt.ir_t_stat, rpt.survives_cost,
            )
        alpha_panel_sv = build_candidate_alpha_panel(
            selected_events=pd.DataFrame(),
            target_weights_2d=np.zeros_like(aligned.close_2d),
            datetimes=aligned.datetimes,
            symbols=tuple(symbols),
        )
        return CandidatePipelineOutput(
            alpha_panel=alpha_panel_sv,
            target_weights=np.zeros_like(aligned.close_2d),
            rule_report={
                "events_total": len(raw_events),
                "labeled_total": len(labeled),
                "promoted_total": promoted_total,
                "fit_total": 0,
                "calibration_total": 0,
                "oos_total": 0,
                "selected_pre_group": 0,
                "selected_total": 0,
                "eligible": 0,
                "n_keep": 0,
                "policy": strategy_cfg.candidate.selection_policy,
                "zero_reason": "signal_only_mode",
                "gate_calibration_used": False,
                "gate_calibration_reason": "signal_only_mode",
                "recommended_keep_variants": diag.recommended_keep_variants,
                "recommended_flip_variants": diag.recommended_flip_variants,
                "signal_validation": [
                    {
                        "variant": r.variant,
                        "n_events": r.n_events,
                        "net_edge_bps_p50": r.net_edge_bps_p50,
                        "net_edge_bps_stress_p50": r.net_edge_bps_stress_p50,
                        "hit_rate": r.hit_rate,
                        "ir_t_stat": r.ir_t_stat,
                        "survives_cost": r.survives_cost,
                        "deployment_count": r.deployment_count,
                    }
                    for r in signal_reports
                ],
                "signal_validation_pass": any_passes,
            },
        )

    # Build WF folds (multi-fold or single)
    wf_folds = build_walk_forward_folds(n_bars=n_bars, cfg=strategy_cfg.candidate) if _folds is None else _folds

    # --- WF fold loop: train per fold, concat OOS predictions ---
    fold_p_pass_parts: list[np.ndarray] = []
    fold_mu_parts: list[np.ndarray] = []
    fold_q10_parts: list[np.ndarray] = []
    fold_q90_parts: list[np.ndarray] = []
    fold_utility_parts: list[np.ndarray] = []
    fold_event_parts: list[Any] = []
    fold_gate_model = None
    fold_edge_models = None
    fold_calibration_used = False
    fold_calibration_reason = "not_fit"
    total_fit = total_cal = total_oos = 0
    fold_cost_survival: list[bool] = []  # per-fold cost survival for min_wf_fold_pass_ratio gate

    for fold in wf_folds:
        fit_set = build_candidate_dataset(
            labeled_events=labeled,
            aligned=aligned,
            cfg=strategy_cfg.candidate,
            split_start=fold.fit_start,
            split_end=fold.fit_end,
        )
        calibration_set = build_candidate_dataset(
            labeled_events=labeled,
            aligned=aligned,
            cfg=strategy_cfg.candidate,
            split_start=fold.cal_start,
            split_end=fold.cal_end,
        )
        oos_set = build_candidate_dataset(
            labeled_events=labeled,
            aligned=aligned,
            cfg=strategy_cfg.candidate,
            split_start=fold.oos_start,
            split_end=fold.oos_end,
        )
        total_fit += int(fit_set.X.shape[0])
        total_cal += int(calibration_set.X.shape[0])
        total_oos += int(oos_set.X.shape[0])

        # Prior-only fallback when fit set is too small
        if fit_set.X.shape[0] < strategy_cfg.candidate.min_fit_obs:
            _logger.warning(
                "[BRIDGE][WF] fold oos=[%d,%d) fit_obs=%d < min_fit_obs=%d; prior-only",
                fold.oos_start, fold.oos_end,
                int(fit_set.X.shape[0]), strategy_cfg.candidate.min_fit_obs,
            )
            fold_cost_survival.append(False)  # prior-only = no cost survival
            if oos_set.X.shape[0] > 0:
                n_oos = oos_set.X.shape[0]
                fold_p_pass_parts.append(np.full(n_oos, 0.5, dtype=np.float64))
                fold_mu_parts.append(np.zeros(n_oos, dtype=np.float64))
                fold_q10_parts.append(np.zeros(n_oos, dtype=np.float64))
                fold_q90_parts.append(np.zeros(n_oos, dtype=np.float64))
                fold_utility_parts.append(np.zeros(n_oos, dtype=np.float64))
                fold_event_parts.append(oos_set.event_index)
            continue

        fold_gate_model = fit_candidate_gate(train=fit_set, valid=calibration_set, cfg=strategy_cfg.candidate)
        fold_edge_models = fit_candidate_edge_models(train=fit_set, valid=calibration_set, cfg=strategy_cfg.candidate)
        fold_calibration_used = bool(fold_gate_model.calibration_used) if fold_gate_model is not None else False
        fold_calibration_reason = fold_gate_model.calibration_reason if fold_gate_model is not None else "not_fit"

        fold_p = predict_candidate_gate(model=fold_gate_model, dataset=oos_set)
        fold_ml = predict_candidate_edges(
            models=fold_edge_models, dataset=oos_set, p_pass=fold_p, cfg=strategy_cfg.candidate
        )
        # cost survival: fold mean net edge > 0 (mu_net_decision_bps is already cost/hurdle-deducted)
        _fold_mu_finite = fold_ml.mu_net_decision_bps[np.isfinite(fold_ml.mu_net_decision_bps)]
        fold_cost_survival.append(
            float(np.mean(_fold_mu_finite)) > 0.0
            if _fold_mu_finite.size > 0 else False
        )
        fold_p_pass_parts.append(fold_p)
        fold_mu_parts.append(fold_ml.mu_net_decision_bps)
        fold_q10_parts.append(fold_ml.q10_net_bps)
        fold_q90_parts.append(fold_ml.q90_net_bps)
        fold_utility_parts.append(fold_ml.utility_score)
        fold_event_parts.append(oos_set.event_index)

    # Combine fold OOS outputs (time-ordered concat)
    if fold_event_parts:
        combined_events = (
            pd.concat(fold_event_parts, ignore_index=True) if len(fold_event_parts) > 1 else fold_event_parts[0]
        )
        p_pass = np.concatenate(fold_p_pass_parts) if fold_p_pass_parts else np.array([], dtype=np.float64)
        _combined_mu = np.concatenate(fold_mu_parts) if fold_mu_parts else np.array([], dtype=np.float64)
        _combined_q10 = np.concatenate(fold_q10_parts) if fold_q10_parts else np.array([], dtype=np.float64)
        _combined_q90 = np.concatenate(fold_q90_parts) if fold_q90_parts else np.array([], dtype=np.float64)
        _combined_utility = np.concatenate(fold_utility_parts) if fold_utility_parts else np.array([], dtype=np.float64)
    else:
        # Fallback: use last fold's full OOS as single-fold behavior
        oos_set_fallback = build_candidate_dataset(
            labeled_events=labeled,
            aligned=aligned,
            cfg=strategy_cfg.candidate,
            split_start=_oos_start_ref,
            split_end=_oos_end_ref,
        )
        combined_events = oos_set_fallback.event_index
        p_pass = predict_candidate_gate(model=fold_gate_model, dataset=oos_set_fallback)
        _fallback_ml = predict_candidate_edges(
            models=fold_edge_models, dataset=oos_set_fallback, p_pass=p_pass, cfg=strategy_cfg.candidate
        )
        _combined_mu = _fallback_ml.mu_net_decision_bps
        _combined_q10 = _fallback_ml.q10_net_bps
        _combined_q90 = _fallback_ml.q90_net_bps
        _combined_utility = _fallback_ml.utility_score
        total_oos = int(oos_set_fallback.X.shape[0])

    # Use last fold's models for selection_thresholds (best fit available)
    _last_oos_set = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=strategy_cfg.candidate,
        split_start=wf_folds[-1].oos_start,
        split_end=wf_folds[-1].oos_end,
    )
    _ref_ml = predict_candidate_edges(
        models=fold_edge_models, dataset=_last_oos_set,
        p_pass=predict_candidate_gate(model=fold_gate_model, dataset=_last_oos_set),
        cfg=strategy_cfg.candidate,
    )
    from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput
    ml_out = CandidateModelOutput(
        events=combined_events,
        p_pass=p_pass.astype(np.float64, copy=False) if p_pass.size > 0 else p_pass,
        mu_gross_bps=_combined_mu,
        mu_net_decision_bps=_combined_mu,
        q10_net_bps=_combined_q10,
        q90_net_bps=_combined_q90,
        utility_score=_combined_utility,
        selection_thresholds=_ref_ml.selection_thresholds,
    )
    ml_out = replace(ml_out, events=combined_events)

    # --- Cross-fold consistency gate (min_wf_fold_pass_ratio) ---
    _n_folds_total = len(fold_cost_survival)
    _n_folds_pass = sum(fold_cost_survival)
    _fold_pass_ratio = _n_folds_pass / max(_n_folds_total, 1)
    _logger.info(
        "[BRIDGE][WF] fold_cost_survival=%s pass_ratio=%.2f min_required=%.2f",
        fold_cost_survival,
        _fold_pass_ratio,
        strategy_cfg.candidate.min_wf_fold_pass_ratio,
    )
    if _fold_pass_ratio < strategy_cfg.candidate.min_wf_fold_pass_ratio:
        _logger.warning(
            "[BRIDGE][WF] fold_pass_ratio=%.2f < min_wf_fold_pass_ratio=%.2f → fail-closed",
            _fold_pass_ratio,
            strategy_cfg.candidate.min_wf_fold_pass_ratio,
        )
        _wf_fail_panel = build_candidate_alpha_panel(
            selected_events=pd.DataFrame(),
            target_weights_2d=np.zeros_like(aligned.close_2d),
            datetimes=aligned.datetimes,
            symbols=tuple(symbols),
        )
        return CandidatePipelineOutput(
            alpha_panel=_wf_fail_panel,
            target_weights=np.zeros_like(aligned.close_2d),
            rule_report={
                "events_total": len(raw_events),
                "labeled_total": len(labeled),
                "promoted_total": promoted_total,
                "fit_total": total_fit,
                "calibration_total": total_cal,
                "oos_total": total_oos,
                "selected_pre_group": 0,
                "selected_total": 0,
                "eligible": 0,
                "n_keep": 0,
                "policy": strategy_cfg.candidate.selection_policy,
                "zero_reason": "wf_fold_pass_ratio_fail",
                "gate_calibration_used": fold_calibration_used,
                "gate_calibration_reason": fold_calibration_reason,
                "wf_fold_pass_ratio": _fold_pass_ratio,
                "wf_n_folds": _n_folds_total,
                "wf_scheme": strategy_cfg.candidate.wf_scheme,
                "recommended_keep_variants": diag.recommended_keep_variants,
                "recommended_flip_variants": diag.recommended_flip_variants,
            },
        )

    # For downstream logging, expose last-fold model state
    gate_model = fold_gate_model
    # Reconstruct oos_set for report counts (use combined)
    oos_set = _last_oos_set

    _logger.info(
        (
            "[DIAG][PIPELINE] raw=%d labeled=%d promoted=%d fit=%d cal=%d oos=%d "
            "n_folds=%d wf_scheme=%s"
        ),
        len(raw_events),
        len(labeled),
        promoted_total,
        total_fit,
        total_cal,
        total_oos,
        len(wf_folds),
        strategy_cfg.candidate.wf_scheme,
    )
    gate_summary = _finite_summary(p_pass)
    edge_summary = _finite_summary(ml_out.mu_net_decision_bps)
    q10_summary = _finite_summary(ml_out.q10_net_bps)
    utility_summary = _finite_summary(ml_out.utility_score)
    _logger.info(
        (
            "[DIAG][PIPELINE_GATE] calibrated=%s reason=%s mean=%.4f median=%.4f p90=%.4f max=%.4f "
            "pct_ge40=%.3f pct_ge45=%.3f pct_ge50=%.3f pct_ge55=%.3f"
        ),
        bool(gate_model.calibration_used) if gate_model is not None else False,
        gate_model.calibration_reason if gate_model is not None else "not_fit",
        gate_summary["mean"],
        gate_summary["median"],
        gate_summary["p90"],
        gate_summary["max"],
        _threshold_rate(p_pass, 0.40),
        _threshold_rate(p_pass, 0.45),
        _threshold_rate(p_pass, 0.50),
        _threshold_rate(p_pass, 0.55),
    )
    _logger.info(
        (
            "[DIAG][PIPELINE_EDGE] mu_mean=%.1f mu_median=%.1f mu_p90=%.1f mu_max=%.1f "
            "q10_mean=%.1f q10_p10=%.1f q10_median=%.1f q10_min=%.1f "
            "utility_mean=%.3f utility_median=%.3f utility_p90=%.3f utility_max=%.3f"
        ),
        edge_summary["mean"],
        edge_summary["median"],
        edge_summary["p90"],
        edge_summary["max"],
        q10_summary["mean"],
        q10_summary["p10"],
        q10_summary["median"],
        q10_summary["min"],
        utility_summary["mean"],
        utility_summary["median"],
        utility_summary["p90"],
        utility_summary["max"],
    )

    selected = select_candidate_events_for_portfolio(model_output=ml_out, cfg=strategy_cfg.candidate)
    selection_diag = dict(getattr(selected, "attrs", {}).get("candidate_selection_diagnostics", {}))
    _logger.info(
        (
            "[DIAG][PIPELINE_SELECT] policy=%s zero_reason=%s eligible=%s selected_pre_group=%s "
            "selected=%s n_keep=%s breakeven_floor=%.1f"
        ),
        selection_diag.get("policy", strategy_cfg.candidate.selection_policy),
        selection_diag.get("zero_reason", "unknown"),
        selection_diag.get("eligible", 0),
        selection_diag.get("selected_pre_group", 0),
        selection_diag.get("selected_total", len(selected)),
        selection_diag.get("n_keep", 0),
        float(selection_diag.get("breakeven_floor_bps", strategy_cfg.candidate.cost_floor_bps)),
    )
    target_weights = build_candidate_target_weights(
        selected_events=selected,
        close_2d=aligned.close_2d,
        symbols=tuple(symbols),
        beta_2d=None,
        sigma_3d=None,
        cfg=strategy_cfg.candidate,
    )
    alpha_panel = build_candidate_alpha_panel(
        selected_events=selected,
        target_weights_2d=target_weights,
        datetimes=aligned.datetimes,
        symbols=tuple(symbols),
    )

    if strategy_cfg.candidate.exit_policy_mode == "label_only":
        # label_only: suppress per-event TP/SL; engine uses global ATR_MULT only
        _logger.debug("[BRIDGE] exit_policy_mode=label_only; zeroing per-event TP/SL columns")
        alpha_panel = alpha_panel.copy()
        alpha_panel["candidate_stop_atr_mult"] = 0.0
        alpha_panel["candidate_take_profit_atr_mult"] = 0.0

    return CandidatePipelineOutput(
        alpha_panel=alpha_panel,
        target_weights=target_weights,
        rule_report={
            "events_total": len(raw_events),
            "labeled_total": len(labeled),
            "promoted_total": promoted_total,
            "fit_total": total_fit,
            "calibration_total": total_cal,
            "oos_total": total_oos,
            "fit_start": wf_folds[0].fit_start,
            "fit_end": wf_folds[-1].fit_end,
            "calibration_start": wf_folds[0].cal_start,
            "calibration_end": wf_folds[-1].cal_end,
            "oos_start": wf_folds[0].oos_start,
            "oos_end": wf_folds[-1].oos_end,
            "wf_n_folds": len(wf_folds),
            "wf_scheme": strategy_cfg.candidate.wf_scheme,
            "y_gate_oos_pos_rate": float(np.mean(p_pass > 0.5)) if p_pass.size > 0 else 0.0,
            "gate_calibration_used": fold_calibration_used,
            "gate_calibration_reason": fold_calibration_reason,
            "gate_p_mean": gate_summary["mean"],
            "gate_p_median": gate_summary["median"],
            "gate_p_p90": gate_summary["p90"],
            "gate_p_max": gate_summary["max"],
            "gate_pct_ge40": _threshold_rate(p_pass, 0.40),
            "gate_pct_ge45": _threshold_rate(p_pass, 0.45),
            "gate_pct_ge50": _threshold_rate(p_pass, 0.50),
            "gate_pct_ge55": _threshold_rate(p_pass, 0.55),
            "mu_mean_bps": edge_summary["mean"],
            "mu_median_bps": edge_summary["median"],
            "mu_p90_bps": edge_summary["p90"],
            "mu_max_bps": edge_summary["max"],
            "q10_mean_bps": q10_summary["mean"],
            "q10_p10_bps": q10_summary["p10"],
            "q10_median_bps": q10_summary["median"],
            "q10_min_bps": q10_summary["min"],
            "utility_mean": utility_summary["mean"],
            "utility_median": utility_summary["median"],
            "utility_p90": utility_summary["p90"],
            "utility_max": utility_summary["max"],
            "selected_pre_group": int(selection_diag.get("selected_pre_group", len(selected))),
            "selected_total": int(selection_diag.get("selected_total", len(selected))),
            "eligible": int(selection_diag.get("eligible", 0)),
            "n_keep": int(selection_diag.get("n_keep", 0)),
            "policy": str(selection_diag.get("policy", strategy_cfg.candidate.selection_policy)),
            "zero_reason": str(selection_diag.get("zero_reason", "unknown")),
            "breakeven_floor_bps": float(
                selection_diag.get("breakeven_floor_bps", strategy_cfg.candidate.cost_floor_bps)
            ),
            "recommended_keep_variants": diag.recommended_keep_variants,
            "recommended_flip_variants": diag.recommended_flip_variants,
            "recommendation_basis": diag.recommendation_basis,
            "recommendation_start": int(diag.recommendation_split[0]),
            "recommendation_end": int(diag.recommendation_split[1]),
            "report_start": int(diag.report_split[0]),
            "report_end": int(diag.report_split[1]),
        },
    )


def merge_candidate_output_into_data_maps(
    candidate_out: CandidatePipelineOutput,
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    log_tag: str = "",
) -> None:
    """Merge candidate output payload into data maps."""
    panel = getattr(candidate_out, "alpha_panel", None)
    if panel is None or panel.empty:
        return
    required = {
        "alpha_long", "alpha_short", "target_weight", "candidate_family",
        "candidate_variant", "p_pass", "mu_net_decision_bps", "q10_net_bps", "utility_score",
        "candidate_stop_atr_mult", "candidate_take_profit_atr_mult",
    }
    if not required.issubset(panel.columns):
        _logger.warning("[%s] candidate panel missing required columns; skip merge", log_tag)
        return
    by_sym = panel.reset_index().groupby("symbol", sort=False)
    for sym in symbols:
        if sym not in data_maps or tf not in data_maps[sym]:
            continue
        df = data_maps[sym][tf]
        for col, default in (
            ("alpha_long", 0.0),
            ("alpha_short", 0.0),
            ("target_weight", 0.0),
            ("p_pass", 0.0),
            ("mu_net_decision_bps", 0.0),
            ("q10_net_bps", 0.0),
            ("utility_score", 0.0),
            ("candidate_stop_atr_mult", 0.0),
            ("candidate_take_profit_atr_mult", 0.0),
        ):
            if col not in df.columns:
                df[col] = np.full(len(df), default, dtype=np.float64)
        if "candidate_family" not in df.columns:
            df["candidate_family"] = np.full(len(df), "", dtype=object)
        if "candidate_variant" not in df.columns:
            df["candidate_variant"] = np.full(len(df), "", dtype=object)

        try:
            sym_rows = by_sym.get_group(sym)
        except KeyError:
            continue
        left = df[["datetime"]].copy()
        right = sym_rows[["datetime", *list(required)]].copy()
        left["_merge_datetime"] = pd.to_datetime(left["datetime"], utc=True).dt.tz_localize(None)
        right["_merge_datetime"] = pd.to_datetime(right["datetime"], utc=True).dt.tz_localize(None)
        merged = left.merge(right[["_merge_datetime", *list(required)]], on="_merge_datetime", how="left")

        df["alpha_long"] = merged["alpha_long"].fillna(0.0).to_numpy(dtype=np.float64)
        df["alpha_short"] = merged["alpha_short"].fillna(0.0).to_numpy(dtype=np.float64)
        df["target_weight"] = merged["target_weight"].fillna(0.0).to_numpy(dtype=np.float64)
        df["candidate_family"] = merged["candidate_family"].fillna("").to_numpy(dtype=object)
        df["candidate_variant"] = merged["candidate_variant"].fillna("").to_numpy(dtype=object)
        df["p_pass"] = merged["p_pass"].fillna(0.0).to_numpy(dtype=np.float64)
        df["mu_net_decision_bps"] = merged["mu_net_decision_bps"].fillna(0.0).to_numpy(dtype=np.float64)
        df["q10_net_bps"] = merged["q10_net_bps"].fillna(0.0).to_numpy(dtype=np.float64)
        df["utility_score"] = merged["utility_score"].fillna(0.0).to_numpy(dtype=np.float64)
        df["candidate_stop_atr_mult"] = merged["candidate_stop_atr_mult"].fillna(0.0).to_numpy(dtype=np.float64)
        df["candidate_take_profit_atr_mult"] = (
            merged["candidate_take_profit_atr_mult"].fillna(0.0).to_numpy(dtype=np.float64)
        )


def merge_candidate_output_into_is_and_oos(
    candidate_out: CandidatePipelineOutput,
    is_maps: dict[str, dict[str, Any]],
    oos_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
) -> None:
    """Merge candidate output into both IS and OOS maps."""
    merge_candidate_output_into_data_maps(candidate_out, is_maps, valid_symbols, tf, log_tag="is")
    merge_candidate_output_into_data_maps(candidate_out, oos_maps, valid_symbols, tf, log_tag="oos")


def copy_data_maps_tf_clone(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for sym in symbols:
        out[sym] = dict(data_maps.get(sym, {}))
        frame = out[sym].get(tf)
        if isinstance(frame, pd.DataFrame):
            out[sym][tf] = frame.copy()
    return out
