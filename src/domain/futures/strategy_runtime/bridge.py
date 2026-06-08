from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.config import StrategyConfig

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _RuntimeBreakdown:
    total: float
    steps: Mapping[str, float]

    @property
    def accounted(self) -> float:
        return float(sum(max(float(value), 0.0) for value in self.steps.values()))

    @property
    def unaccounted(self) -> float:
        return max(float(self.total) - self.accounted, 0.0)


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


def _log_universe_volatility_deciles(
    *,
    events: pd.DataFrame,
    selected: pd.DataFrame,
    mu_net_decision_bps: np.ndarray,
    q10_net_bps: np.ndarray,
) -> None:
    if events.empty or "vol_30d" not in events.columns:
        return
    vol = pd.to_numeric(events["vol_30d"], errors="coerce")
    valid = vol.notna()
    if int(valid.sum()) < 10:
        return
    diag = events.loc[valid].copy()
    diag["_mu_net_decision_bps"] = np.asarray(mu_net_decision_bps, dtype=np.float64)[valid.to_numpy()]
    diag["_q10_net_bps"] = np.asarray(q10_net_bps, dtype=np.float64)[valid.to_numpy()]
    diag["_selected"] = False
    if not selected.empty:
        selected_keys = {
            (
                pd.Timestamp(dt).tz_localize(None) if pd.Timestamp(dt).tzinfo is not None else pd.Timestamp(dt),
                str(sym),
                int(entry_idx),
                str(family),
                str(variant),
            )
            for dt, sym, entry_idx, family, variant in selected.loc[
                :, ["datetime", "symbol", "entry_idx", "family", "variant"]
            ].itertuples(index=False, name=None)
        }
        diag["_selected"] = [
            (
                pd.Timestamp(dt).tz_localize(None) if pd.Timestamp(dt).tzinfo is not None else pd.Timestamp(dt),
                str(sym),
                int(entry_idx),
                str(family),
                str(variant),
            )
            in selected_keys
            for dt, sym, entry_idx, family, variant in diag.loc[
                :, ["datetime", "symbol", "entry_idx", "family", "variant"]
            ].itertuples(index=False, name=None)
        ]
    diag["_vol_decile"] = pd.qcut(vol.loc[valid], q=10, labels=False, duplicates="drop")
    grouped = diag.groupby("_vol_decile", sort=True, dropna=True)
    for decile, group in grouped:
        _logger.debug(
            "[DIAG][VOL_DECILE] decile=%s events=%d mu_mean=%.1f q10_median=%.1f selected_pass_rate=%.3f",
            int(decile) + 1,
            int(group.shape[0]),
            float(pd.to_numeric(group["_mu_net_decision_bps"], errors="coerce").mean()),
            float(pd.to_numeric(group["_q10_net_bps"], errors="coerce").median()),
            float(pd.to_numeric(group["_selected"], errors="coerce").mean()),
        )


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
    aligned: AlignedMarketData | None = None
    labeled: pd.DataFrame | None = None
    labeled_unfiltered: pd.DataFrame | None = None
    fit_set: Any | None = None
    calibration_set: Any | None = None
    oos_set: Any | None = None
    gate_model: Any | None = None
    edge_models: Any | None = None
    fit_start: int | None = None
    fit_end: int | None = None
    calibration_start: int | None = None
    calibration_end: int | None = None
    oos_start: int | None = None
    oos_end: int | None = None
    fold_oos_boundaries: tuple[tuple[int, int], ...] | None = None


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

    import time
    from dataclasses import replace

    from src.domain.futures.strategy.ablation import apply_variant_promotions, validate_candidate_signals
    from src.domain.futures.strategy.candidate_dataset import build_candidate_dataset
    from src.domain.futures.strategy.candidate_edge import predict_candidate_edges
    from src.domain.futures.strategy.candidate_gate import predict_candidate_gate
    from src.domain.futures.strategy.candidate_labels import label_candidate_events
    from src.domain.futures.strategy.candidate_portfolio import (
        build_candidate_alpha_panel,
        build_candidate_target_weights,
        select_candidate_events_for_portfolio,
    )
    from src.domain.futures.strategy.common.alignment import align_data_maps
    from src.domain.futures.strategy.config import resolve_purge_and_embargo_bars, with_max_holding_bars
    from src.domain.futures.strategy.rule_diagnostics import compute_rule_diagnostics
    from src.domain.futures.strategy.rule_signals import (
        build_rule_signal_panels,
        candidate_panels_to_events,
    )
    from src.domain.futures.strategy.walk_forward import build_walk_forward_folds

    bridge_t0 = time.perf_counter()
    bridge_prof: dict[str, float] = {
        "align": 0.0,
        "rules": 0.0,
        "events": 0.0,
        "label": 0.0,
        "diagnostics": 0.0,
        "promotions": 0.0,
        "walk_forward": 0.0,
        "post_wf": 0.0,
        "selection": 0.0,
        "weights": 0.0,
        "alpha_panel": 0.0,
    }

    def _emit_bridge_profile() -> None:
        breakdown = _RuntimeBreakdown(total=time.perf_counter() - bridge_t0, steps=bridge_prof)
        _logger.info(
            (
                "[BRIDGE-PROF] total=%.4fs align=%.4fs rules=%.4fs events=%.4fs "
                "label=%.4fs diagnostics=%.4fs promotions=%.4fs walk_forward=%.4fs "
                "post_wf=%.4fs selection=%.4fs weights=%.4fs alpha_panel=%.4fs "
                "accounted=%.4fs unaccounted=%.4fs"
            ),
            breakdown.total,
            bridge_prof["align"],
            bridge_prof["rules"],
            bridge_prof["events"],
            bridge_prof["label"],
            bridge_prof["diagnostics"],
            bridge_prof["promotions"],
            bridge_prof["walk_forward"],
            bridge_prof["post_wf"],
            bridge_prof["selection"],
            bridge_prof["weights"],
            bridge_prof["alpha_panel"],
            breakdown.accounted,
            breakdown.unaccounted,
        )

    t_step = time.perf_counter()
    aligned = align_data_maps(preloaded_data_maps, symbols, tf)
    bridge_prof["align"] = time.perf_counter() - t_step
    n_bars = aligned.close_2d.shape[0]

    t_step = time.perf_counter()
    panels = build_rule_signal_panels(aligned=aligned, cfg=strategy_cfg.candidate)
    bridge_prof["rules"] = time.perf_counter() - t_step
    t_step = time.perf_counter()
    raw_events = candidate_panels_to_events(
        panels,
        min_abs_score=strategy_cfg.candidate.min_rule_net_bps * 1e-4,
        side_flip_variants=strategy_cfg.candidate.side_flip_candidate_variants,
        cost_floor_bps=strategy_cfg.candidate.cost_floor_bps,
        execution_cost_bps_2d=aligned.execution_cost_bps_2d,
    )
    bridge_prof["events"] = time.perf_counter() - t_step
    max_holding_bars = (
        int(pd.to_numeric(raw_events["expected_holding_bars"], errors="coerce").max())
        if not raw_events.empty and "expected_holding_bars" in raw_events.columns
        else None
    )
    candidate_cfg = with_max_holding_bars(
        strategy_cfg.candidate,
        max_holding_bars=max_holding_bars,
    )
    purge_bars, embargo_bars = resolve_purge_and_embargo_bars(candidate_cfg)

    if raw_events.empty:
        t_step = time.perf_counter()
        alpha_panel = build_candidate_alpha_panel(
            selected_events=raw_events,
            target_weights_2d=np.zeros_like(aligned.close_2d),
            datetimes=aligned.datetimes,
            symbols=tuple(symbols),
            cfg=strategy_cfg.candidate,
        )
        bridge_prof["alpha_panel"] = time.perf_counter() - t_step
        _emit_bridge_profile()
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
                "policy": candidate_cfg.selection_policy,
                "zero_reason": "no_events",
                "gate_calibration_used": False,
                "gate_calibration_reason": "no_events",
                "recommended_keep_variants": (),
                "recommended_flip_variants": (),
                "recommended_keep_signal_cells": (),
                "recommended_flip_signal_cells": (),
            },
        )

    t_step = time.perf_counter()
    labeled = label_candidate_events(events=raw_events, aligned=aligned, cfg=candidate_cfg)
    bridge_prof["label"] = time.perf_counter() - t_step
    labeled_all = labeled.copy()
    fit_start, fit_end, calibration_start, calibration_end, oos_start, oos_end = _candidate_ml_split_indices(
        n_bars=n_bars,
        fit_fraction=candidate_cfg.ml_fit_fraction,
        calibration_fraction=candidate_cfg.ml_calibration_fraction,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
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

    t_step = time.perf_counter()
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
    bridge_prof["diagnostics"] = time.perf_counter() - t_step

    if strategy_cfg.candidate.promotion_filter_enabled:
        t_step = time.perf_counter()
        labeled = apply_variant_promotions(
            labeled=labeled,
            keep_variants=diag.recommended_keep_variants,
            flip_variants=diag.recommended_flip_variants,
            keep_signal_cells=diag.recommended_keep_signal_cells,
            flip_signal_cells=diag.recommended_flip_signal_cells,
        )
        bridge_prof["promotions"] = time.perf_counter() - t_step
        if labeled.empty:
            _logger.debug(
                "[BRIDGE] all candidate variants blocked by promotion filter; producing zero weights"
            )
            t_step = time.perf_counter()
            alpha_panel = build_candidate_alpha_panel(
                selected_events=pd.DataFrame(),
                target_weights_2d=np.zeros_like(aligned.close_2d),
                datetimes=aligned.datetimes,
                symbols=tuple(symbols),
                cfg=strategy_cfg.candidate,
            )
            bridge_prof["alpha_panel"] = time.perf_counter() - t_step
            _emit_bridge_profile()
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
                    "recommended_keep_signal_cells": diag.recommended_keep_signal_cells,
                    "recommended_flip_signal_cells": diag.recommended_flip_signal_cells,
                },
            )
    promoted_total = len(labeled)

    # Compute split indices needed for signal_only + WF (done once for OOS window bounds)
    if strategy_cfg.candidate.wf_enabled and strategy_cfg.candidate.wf_scheme != "single":
        _folds = build_walk_forward_folds(n_bars=n_bars, cfg=candidate_cfg, max_holding_bars=max_holding_bars)
        _oos_start_ref = _folds[0].oos_start if _folds else 0
        _oos_end_ref = _folds[-1].oos_end if _folds else n_bars
    else:
        _s = _candidate_ml_split_indices(
            n_bars=n_bars,
            fit_fraction=candidate_cfg.ml_fit_fraction,
            calibration_fraction=candidate_cfg.ml_calibration_fraction,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )
        _oos_start_ref, _oos_end_ref = _s[4], _s[5]
        _folds = None  # single-fold path handled below

    # signal_only: validate rule signals, skip ML training
    if strategy_cfg.candidate.signal_only:
        signal_reports = validate_candidate_signals(
            labeled_all=labeled_all,
            labeled_promoted=labeled,
            cfg=candidate_cfg,
            oos_start=_oos_start_ref,
            oos_end=_oos_end_ref,
        )
        if strategy_cfg.candidate.blend_survival_require_promoted:
            promoted_rpt = next((r for r in signal_reports if r.variant == "rule_promo_no_leak"), None)
            any_passes = bool(promoted_rpt is not None and promoted_rpt.survives_cost)
        else:
            any_passes = any(r.survives_cost for r in signal_reports)
        _logger.debug(
            "[SIGNAL-VALIDATION] variants=%d any_passes=%s",
            len(signal_reports),
            any_passes,
        )
        for rpt in signal_reports:
            _logger.debug(
                "[SIGNAL-VALIDATION] variant=%s n=%d net_p50=%.1f stress_p50=%.1f "
                "mean=%.1f stress_mean=%.1f hit=%.3f hac_t=%.2f decision_bars=%d survives=%s",
                rpt.variant, rpt.n_events, rpt.net_edge_bps_p50,
                rpt.net_edge_bps_stress_p50, rpt.net_edge_bps_mean,
                rpt.net_edge_bps_stress_mean, rpt.hit_rate, rpt.hac_t_stat, rpt.decision_bar_count, rpt.survives_cost,
            )
        t_step = time.perf_counter()
        alpha_panel_sv = build_candidate_alpha_panel(
            selected_events=pd.DataFrame(),
            target_weights_2d=np.zeros_like(aligned.close_2d),
            datetimes=aligned.datetimes,
            symbols=tuple(symbols),
            cfg=strategy_cfg.candidate,
        )
        bridge_prof["alpha_panel"] = time.perf_counter() - t_step
        _emit_bridge_profile()
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
                "failure_report": diag.recommendation_failure_report,
                "signal_validation": [
                    {
                        "variant": r.variant,
                        "n_events": r.n_events,
                        "net_edge_bps_p50": r.net_edge_bps_p50,
                        "net_edge_bps_stress_p50": r.net_edge_bps_stress_p50,
                        "net_edge_bps_mean": r.net_edge_bps_mean,
                        "net_edge_bps_stress_mean": r.net_edge_bps_stress_mean,
                        "hit_rate": r.hit_rate,
                        "hac_t_stat": r.hac_t_stat,
                        "survives_cost": r.survives_cost,
                        "deployment_count": r.deployment_count,
                        "decision_bar_count": r.decision_bar_count,
                    }
                    for r in signal_reports
                ],
                "signal_validation_pass": any_passes,
            },
        )

    # Build WF folds (multi-fold or single)
    wf_folds = (
        build_walk_forward_folds(n_bars=n_bars, cfg=candidate_cfg, max_holding_bars=max_holding_bars)
        if _folds is None
        else _folds
    )

    # --- WF fold loop: train per fold using shared workflow ---
    from src.domain.futures.strategy.candidate_workflow import run_candidate_walk_forward
    t_step = time.perf_counter()
    wf_outputs = run_candidate_walk_forward(
        labeled_events=labeled,
        aligned=aligned,
        cfg=candidate_cfg,
        folds=wf_folds,
    )
    bridge_prof["walk_forward"] = time.perf_counter() - t_step
    t_post_wf = time.perf_counter()

    fold_p_pass_parts: list[np.ndarray] = []
    fold_mu_parts: list[np.ndarray] = []
    fold_q10_parts: list[np.ndarray] = []
    fold_q90_parts: list[np.ndarray] = []
    fold_utility_parts: list[np.ndarray] = []
    fold_expected_return_r_parts: list[np.ndarray] = []
    fold_q10_return_r_parts: list[np.ndarray] = []
    fold_q90_return_r_parts: list[np.ndarray] = []
    fold_kelly_fraction_parts: list[np.ndarray] = []
    fold_event_parts: list[Any] = []
    fold_gate_model = None  # We don't expose private models directly anymore, fallback used below
    fold_edge_models = None
    fold_calibration_used = False
    fold_calibration_reason = "not_fit"
    total_fit = total_cal = total_oos = 0
    fold_cost_survival: list[bool] = []
    fold_selection_reports: list[dict[str, Any]] = []

    wf_fold_details: list[dict[str, Any]] = []

    # Map validation status from workflow outputs
    for fold_out in wf_outputs:
        fold = wf_folds[fold_out.fold_id]
        ml_out = fold_out.model_output
        selected_fold = fold_out.selected_events
        selection_diag_fold = dict(getattr(selected_fold, "attrs", {}).get("candidate_selection_diagnostics", {}))

        if "edge_after_hurdle_bps" in selected_fold.columns:
            realized_edge = pd.to_numeric(selected_fold["edge_after_hurdle_bps"], errors="coerce")
        else:
            realized_edge = pd.Series(dtype="float64")
        selected_count = int(selected_fold.shape[0])
        realized_mean = float(realized_edge.mean()) if realized_edge.notna().any() else float("nan")
        realized_hit_rate = float((realized_edge > 0.0).mean()) if realized_edge.notna().any() else 0.0
        
        if realized_edge.notna().any():
            log_growth_proxy = float(
                np.mean(
                    np.log1p(
                        np.clip(realized_edge.to_numpy(dtype=np.float64, copy=False) * 1e-4, -0.99, None)
                    )
                )
            )
        else:
            log_growth_proxy = float("-inf")

        # Default values for lift variables (only computed in realized_selected_edge branch)
        ml_lift_bps: float = float("nan")
        pass_lift: bool = False

        survival_metric = strategy_cfg.candidate.fold_survival_metric
        if survival_metric == "predicted_mu_tstat":
            fold_mu_finite = ml_out.mu_net_decision_bps[np.isfinite(ml_out.mu_net_decision_bps)]
            if fold_mu_finite.size >= 10:
                mean_edge = float(np.mean(fold_mu_finite))
                std_edge = float(np.std(fold_mu_finite)) + 1e-12
                n_obs = fold_mu_finite.size
                t_stat = mean_edge / (std_edge / np.sqrt(n_obs))
                pass_survival = bool(mean_edge > 0.0 and t_stat > 1.645)
                survival_reason = "predicted_mu_tstat_pass" if pass_survival else "predicted_mu_tstat_fail"
            else:
                pass_survival = False
                survival_reason = "predicted_mu_tstat_insufficient_obs"
        elif survival_metric == "realized_log_growth":
            pass_survival = (
                selected_count >= strategy_cfg.candidate.min_fold_selected_events
                and np.isfinite(log_growth_proxy)
                and log_growth_proxy >= strategy_cfg.candidate.min_fold_log_growth
            )
            survival_reason = "realized_log_growth_pass" if pass_survival else "realized_log_growth_fail"
        else:
            # Compute ML selection lift: mean(selected_edge) - mean(all_fold_oos_edge)
            fold_oos_events = labeled[
                (labeled["entry_idx"] >= fold.oos_start) & (labeled["entry_idx"] < fold.oos_end)
            ] if "entry_idx" in labeled.columns else pd.DataFrame()
            if not fold_oos_events.empty and "edge_after_hurdle_bps" in fold_oos_events.columns:
                baseline_mean = float(
                    pd.to_numeric(fold_oos_events["edge_after_hurdle_bps"], errors="coerce").mean()
                )
            else:
                baseline_mean = float("nan")
            ml_lift_bps = (
                (realized_mean - baseline_mean)
                if np.isfinite(realized_mean) and np.isfinite(baseline_mean)
                else float("nan")
            )
            pass_lift = bool(np.isfinite(ml_lift_bps) and ml_lift_bps > 0.0)
            pass_survival = (
                selected_count >= strategy_cfg.candidate.min_fold_selected_events
                and np.isfinite(realized_mean)
                and realized_mean >= strategy_cfg.candidate.min_fold_realized_edge_bps
                and pass_lift
            )
            survival_reason = "realized_selected_edge_pass" if pass_survival else "realized_selected_edge_fail"
        
        fold_cost_survival.append(bool(pass_survival))
        fold_selection_reports.append(
            {
                "eligible": int(selection_diag_fold.get("eligible", 0)),
                "selected_total": selected_count,
                "realized_mean_bps": realized_mean,
                "realized_status": "empty" if selected_count == 0 else "observed",
                "log_growth_proxy": log_growth_proxy,
                "waterfall_expected_utility_adj_p90_bps": selection_diag_fold.get(
                    "waterfall_expected_utility_adj_p90_bps",
                    float("nan"),
                ),
                "waterfall_downside_drag_p90_bps": selection_diag_fold.get(
                    "waterfall_downside_drag_p90_bps",
                    float("nan"),
                ),
                "waterfall_breakeven_floor_bps": selection_diag_fold.get(
                    "waterfall_breakeven_floor_bps",
                    float("nan"),
                ),
                "shadow_profile_count": int(selection_diag_fold.get("shadow_profile_count", 0)),
                "shadow_max_selected_total": int(selection_diag_fold.get("shadow_max_selected_total", 0)),
                "shadow_max_eligible": int(selection_diag_fold.get("shadow_max_eligible", 0)),
            }
        )

        # Collect for summary table
        _vdiag = getattr(ml_out, "validation_diagnostics", {}) or {}
        _edge_rep = fold_out.edge_report
        _mode = str(_vdiag.get("prediction_mode", "n/a"))
        _rank_ic_val = (
            _edge_rep.residual_rank_ic
            if _mode == "prior_residual"
            else _edge_rep.prior_rank_ic
        )
        wf_fold_details.append({
            "fold_id": len(fold_selection_reports),
            "inference_mode": _mode,
            "rank_ic": float(_rank_ic_val),
            "n_events": int(ml_out.events.shape[0]) if ml_out.events is not None else 0,
            "prior_bps": float(_vdiag.get("prior_component_p90_bps", 0.0)),
            "eu_p90": float(selection_diag_fold.get("waterfall_expected_utility_adj_p90_bps", 0.0)),
            "pass_cost": bool(pass_survival),
        })

        _logger.debug(
            (
                "[BRIDGE][WF_REALIZED] metric=%s oos=[%d,%d) selected=%d realized_mean=%.3f "
                "status=%s hit_rate=%.3f log_growth=%.6f lift=%.3f pass_lift=%s pass=%s reason=%s"
            ),
            survival_metric,
            fold.oos_start,
            fold.oos_end,
            selected_count,
            realized_mean,
            "empty" if selected_count == 0 else "observed",
            realized_hit_rate,
            log_growth_proxy,
            ml_lift_bps if survival_metric not in ("predicted_mu_tstat", "realized_log_growth") else float("nan"),
            pass_lift if survival_metric not in ("predicted_mu_tstat", "realized_log_growth") else False,
            pass_survival,
            survival_reason,
        )
        _logger.debug(
            (
                "[BRIDGE][WF_DIAG] fold=%d eligible=%s selected=%d eu_p90=%.3f downside_p90=%.3f "
                "breakeven=%.1f shadow_profiles=%d shadow_max_selected=%d shadow_max_eligible=%d"
            ),
            len(fold_selection_reports),
            selection_diag_fold.get("eligible", 0),
            selected_count,
            float(selection_diag_fold.get("waterfall_expected_utility_adj_p90_bps", float("nan"))),
            float(selection_diag_fold.get("waterfall_downside_drag_p90_bps", float("nan"))),
            float(selection_diag_fold.get("waterfall_breakeven_floor_bps", float("nan"))),
            int(selection_diag_fold.get("shadow_profile_count", 0)),
            int(selection_diag_fold.get("shadow_max_selected_total", 0)),
            int(selection_diag_fold.get("shadow_max_eligible", 0)),
        )

        fold_p_pass_parts.append(ml_out.p_pass)
        fold_mu_parts.append(ml_out.expected_net_bps)
        fold_q10_parts.append(ml_out.q10_net_bps)
        fold_q90_parts.append(ml_out.q90_net_bps)
        fold_utility_parts.append(ml_out.selection_score)
        fold_expected_return_r_parts.append(ml_out.expected_return_r)
        fold_q10_return_r_parts.append(ml_out.q10_return_r)
        fold_q90_return_r_parts.append(ml_out.q90_return_r)
        fold_kelly_fraction_parts.append(ml_out.kelly_fraction)
        fold_event_parts.append(ml_out.events)

    # Note: fallback behavior down below is preserved using oos_set_fallback
    # Reconstruct counts from fold outputs
    p_pass = np.concatenate(fold_p_pass_parts) if fold_p_pass_parts else np.array([], dtype=np.float64)
    _combined_mu = np.concatenate(fold_mu_parts) if fold_mu_parts else np.array([], dtype=np.float64)
    _combined_q10 = np.concatenate(fold_q10_parts) if fold_q10_parts else np.array([], dtype=np.float64)
    _combined_q90 = np.concatenate(fold_q90_parts) if fold_q90_parts else np.array([], dtype=np.float64)
    _combined_utility = np.concatenate(fold_utility_parts) if fold_utility_parts else np.array([], dtype=np.float64)
    _combined_expected_return_r = (
        np.concatenate(fold_expected_return_r_parts)
        if fold_expected_return_r_parts
        else np.array([], dtype=np.float64)
    )
    _combined_q10_return_r = (
        np.concatenate(fold_q10_return_r_parts)
        if fold_q10_return_r_parts
        else np.array([], dtype=np.float64)
    )
    _combined_q90_return_r = (
        np.concatenate(fold_q90_return_r_parts)
        if fold_q90_return_r_parts
        else np.array([], dtype=np.float64)
    )
    _combined_kelly_fraction = (
        np.concatenate(fold_kelly_fraction_parts)
        if fold_kelly_fraction_parts
        else np.array([], dtype=np.float64)
    )
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
        _combined_expected_return_r = (
            np.concatenate(fold_expected_return_r_parts)
            if fold_expected_return_r_parts
            else np.array([], dtype=np.float64)
        )
        _combined_q10_return_r = (
            np.concatenate(fold_q10_return_r_parts)
            if fold_q10_return_r_parts
            else np.array([], dtype=np.float64)
        )
        _combined_q90_return_r = (
            np.concatenate(fold_q90_return_r_parts)
            if fold_q90_return_r_parts
            else np.array([], dtype=np.float64)
        )
        _combined_kelly_fraction = (
            np.concatenate(fold_kelly_fraction_parts)
            if fold_kelly_fraction_parts
            else np.array([], dtype=np.float64)
        )
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
        # Note: We need a trained gate model for fallback prediction. We can extract it
        # from wf_outputs if available. But fold_gate_model is now managed by the workflow outputs.
        # We fallback to the last output fold's models if available.
        # However, if wf_outputs is empty, we create degenerate predictions.
        if len(wf_outputs) > 0:
            # Re-run inference using last fold as fallback reference.
            # (In practice, wf_outputs is rarely empty if n_bars is valid).
            p_pass = np.full(oos_set_fallback.X.shape[0] if oos_set_fallback.X is not None else 0, 0.5)
            _combined_mu = np.zeros_like(p_pass)
            _combined_q10 = np.zeros_like(p_pass)
            _combined_q90 = np.zeros_like(p_pass)
            _combined_utility = np.zeros_like(p_pass)
            _combined_expected_return_r = np.zeros_like(p_pass)
            _combined_q10_return_r = np.zeros_like(p_pass)
            _combined_q90_return_r = np.zeros_like(p_pass)
            _combined_kelly_fraction = np.zeros_like(p_pass)
        else:
            p_pass = np.array([], dtype=np.float64)
            _combined_mu = np.array([], dtype=np.float64)
            _combined_q10 = np.array([], dtype=np.float64)
            _combined_q90 = np.array([], dtype=np.float64)
            _combined_utility = np.array([], dtype=np.float64)
            _combined_expected_return_r = np.array([], dtype=np.float64)
            _combined_q10_return_r = np.array([], dtype=np.float64)
            _combined_q90_return_r = np.array([], dtype=np.float64)
            _combined_kelly_fraction = np.array([], dtype=np.float64)
        total_oos = int(oos_set_fallback.X.shape[0] if oos_set_fallback.X is not None else 0)

    # Use last fold's models for selection_thresholds (best fit available)
    _last_oos_set = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=strategy_cfg.candidate,
        split_start=wf_folds[-1].oos_start,
        split_end=wf_folds[-1].oos_end,
    )
    
    from src.domain.futures.strategy.candidate_contracts import (
        CandidateModelOutput,
        CandidateWorkflowStatus,
        EdgeSource,
    )
    
    validation = getattr(fold_gate_model, "validation", None) if fold_gate_model is not None else None
    gate_enabled = validation.enabled if validation is not None else False
    gate_threshold = validation.threshold if validation is not None else 0.5
    edge_source = (
        EdgeSource.PRIOR_RESIDUAL
        if fold_edge_models is not None and fold_edge_models.prediction_mode == "prior_residual"
        else EdgeSource.PRIOR_ONLY
    )

    if fold_gate_model is not None:
        p_pass_ref = predict_candidate_gate(
            model=fold_gate_model, dataset=_last_oos_set, cfg=strategy_cfg.candidate
        )
    else:
        p_pass_ref = np.zeros(_last_oos_set.X.shape[0] if _last_oos_set.X is not None else 0)

    _ref_ml = predict_candidate_edges(
        models=fold_edge_models,
        dataset=_last_oos_set,
        p_pass=p_pass_ref,
        cfg=strategy_cfg.candidate,
        gate_enabled=gate_enabled,
        gate_threshold=gate_threshold,
        edge_source=edge_source,
    )
    ml_out = CandidateModelOutput(
        events=combined_events,
        p_pass=p_pass.astype(np.float64, copy=False) if p_pass.size > 0 else p_pass,
        gate_enabled=gate_enabled,
        gate_threshold=gate_threshold,
        edge_source=edge_source,
        expected_return_r=_combined_expected_return_r,
        expected_net_bps=_combined_mu,
        q10_return_r=_combined_q10_return_r,
        q10_net_bps=_combined_q10,
        q90_return_r=_combined_q90_return_r,
        q90_net_bps=_combined_q90,
        selection_score=_combined_utility,
        kelly_fraction=_combined_kelly_fraction,
        validation_diagnostics=_ref_ml.validation_diagnostics,
    )
    ml_out = replace(ml_out, events=combined_events)

    # --- Cross-fold consistency gate (min_wf_fold_pass_ratio) ---
    _n_folds_total = len(fold_cost_survival)
    _n_folds_pass = sum(fold_cost_survival)
    _fold_pass_ratio = _n_folds_pass / max(_n_folds_total, 1)
    _logger.debug(
        "[BRIDGE][WF] fold_cost_survival=%s pass_ratio=%.2f min_required=%.2f",
        fold_cost_survival,
        _fold_pass_ratio,
        strategy_cfg.candidate.min_wf_fold_pass_ratio,
    )
    wf_selected_total = int(sum(int(r.get("selected_total", 0)) for r in fold_selection_reports))
    wf_eligible_total = int(sum(int(r.get("eligible", 0)) for r in fold_selection_reports))
    realized_mean_values = [
        float(r["realized_mean_bps"])
        for r in fold_selection_reports
        if np.isfinite(float(r.get("realized_mean_bps", float("nan"))))
    ]
    log_growth_values = [
        float(r["log_growth_proxy"])
        for r in fold_selection_reports
        if np.isfinite(float(r.get("log_growth_proxy", float("nan"))))
    ]
    wf_fold_realized_mean_bps = (
        float(np.mean(realized_mean_values)) if realized_mean_values else float("nan")
    )
    wf_fold_log_growth_mean = (
        float(np.mean(log_growth_values)) if log_growth_values else float("nan")
    )
    wf_waterfall_expected_utility_p90_bps = float(
        np.mean(
            [
                float(r["waterfall_expected_utility_adj_p90_bps"])
                for r in fold_selection_reports
                if np.isfinite(float(r.get("waterfall_expected_utility_adj_p90_bps", float("nan"))))
            ]
        )
    ) if any(
        np.isfinite(float(r.get("waterfall_expected_utility_adj_p90_bps", float("nan"))))
        for r in fold_selection_reports
    ) else float("nan")
    wf_waterfall_downside_drag_p90_bps = float(
        np.mean(
            [
                float(r["waterfall_downside_drag_p90_bps"])
                for r in fold_selection_reports
                if np.isfinite(float(r.get("waterfall_downside_drag_p90_bps", float("nan"))))
            ]
        )
    ) if any(
        np.isfinite(float(r.get("waterfall_downside_drag_p90_bps", float("nan"))))
        for r in fold_selection_reports
    ) else float("nan")
    wf_shadow_profile_count = max((int(r.get("shadow_profile_count", 0)) for r in fold_selection_reports), default=0)
    wf_shadow_max_selected_total = max(
        (int(r.get("shadow_max_selected_total", 0)) for r in fold_selection_reports),
        default=0,
    )
    wf_shadow_max_eligible = max(
        (int(r.get("shadow_max_eligible", 0)) for r in fold_selection_reports),
        default=0,
    )
    _fold_oos_boundaries = tuple((f.oos_start, f.oos_end) for f in wf_folds)
    last_fold_out = wf_outputs[-1] if wf_outputs else None
    if _fold_pass_ratio < strategy_cfg.candidate.min_wf_fold_pass_ratio:
        _logger.debug(
            "[BRIDGE][WF] fold_pass_ratio=%.2f < min_wf_fold_pass_ratio=%.2f → fail-closed",
            _fold_pass_ratio,
            strategy_cfg.candidate.min_wf_fold_pass_ratio,
        )
        bridge_prof["post_wf"] = time.perf_counter() - t_post_wf
        t_step = time.perf_counter()
        _wf_fail_panel = build_candidate_alpha_panel(
            selected_events=pd.DataFrame(),
            target_weights_2d=np.zeros_like(aligned.close_2d),
            datetimes=aligned.datetimes,
            symbols=tuple(symbols),
            cfg=strategy_cfg.candidate,
        )
        bridge_prof["alpha_panel"] = time.perf_counter() - t_step
        out = CandidatePipelineOutput(
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
                "workflow_status": CandidateWorkflowStatus.BLOCKED.value,
                "gate_calibration_used": fold_calibration_used,
                "gate_calibration_reason": fold_calibration_reason,
                "wf_fold_pass_ratio": _fold_pass_ratio,
                "wf_n_folds": _n_folds_total,
                "wf_scheme": strategy_cfg.candidate.wf_scheme,
                "wf_selected_total": wf_selected_total,
                "wf_eligible_total": wf_eligible_total,
                "wf_fold_realized_mean_bps": wf_fold_realized_mean_bps,
                "wf_fold_log_growth_mean": wf_fold_log_growth_mean,
                "wf_shadow_profile_count": wf_shadow_profile_count,
                "wf_shadow_max_selected_total": wf_shadow_max_selected_total,
                "wf_shadow_max_eligible": wf_shadow_max_eligible,
                "wf_waterfall_expected_utility_p90_bps": wf_waterfall_expected_utility_p90_bps,
                "wf_waterfall_downside_drag_p90_bps": wf_waterfall_downside_drag_p90_bps,
                "wf_fold_details": wf_fold_details,
                "recommended_keep_variants": diag.recommended_keep_variants,
                "recommended_flip_variants": diag.recommended_flip_variants,
                "recommended_keep_signal_cells": diag.recommended_keep_signal_cells,
                "recommended_flip_signal_cells": diag.recommended_flip_signal_cells,
            },
            aligned=aligned,
            labeled=labeled,
            labeled_unfiltered=labeled_all,
            fit_set=last_fold_out.fit_set if last_fold_out else None,
            calibration_set=last_fold_out.calibration_set if last_fold_out else None,
            oos_set=last_fold_out.oos_set if last_fold_out else None,
            gate_model=last_fold_out.gate_model if last_fold_out else None,
            edge_models=last_fold_out.edge_models if last_fold_out else None,
            fit_start=wf_folds[0].fit_start,
            fit_end=wf_folds[-1].fit_end,
            calibration_start=wf_folds[0].cal_start,
            calibration_end=wf_folds[-1].cal_end,
            oos_start=wf_folds[0].oos_start,
            oos_end=wf_folds[-1].oos_end,
            fold_oos_boundaries=_fold_oos_boundaries,
        )
        _emit_bridge_profile()
        return out

    # For downstream logging, expose last-fold model state
    gate_model = fold_gate_model
    # Reconstruct oos_set for report counts (use combined)

    _logger.debug(
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
    _logger.debug(
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
    _logger.debug(
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

    bridge_prof["post_wf"] = time.perf_counter() - t_post_wf
    t_step = time.perf_counter()
    selected = select_candidate_events_for_portfolio(model_output=ml_out, cfg=strategy_cfg.candidate)
    bridge_prof["selection"] = time.perf_counter() - t_step
    selection_diag = dict(getattr(selected, "attrs", {}).get("candidate_selection_diagnostics", {}))
    _logger.debug(
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
    _log_universe_volatility_deciles(
        events=combined_events,
        selected=selected,
        mu_net_decision_bps=ml_out.mu_net_decision_bps,
        q10_net_bps=ml_out.q10_net_bps,
    )
    t_step = time.perf_counter()
    target_weights = build_candidate_target_weights(
        selected_events=selected,
        close_2d=aligned.close_2d,
        symbols=tuple(symbols),
        beta_2d=None,
        sigma_3d=None,
        cfg=strategy_cfg.candidate,
    )
    bridge_prof["weights"] = time.perf_counter() - t_step
    t_step = time.perf_counter()
    alpha_panel = build_candidate_alpha_panel(
        selected_events=selected,
        target_weights_2d=target_weights,
        datetimes=aligned.datetimes,
        symbols=tuple(symbols),
        cfg=strategy_cfg.candidate,
    )
    bridge_prof["alpha_panel"] = time.perf_counter() - t_step

    if strategy_cfg.candidate.exit_policy_mode == "label_only":
        # label_only: suppress per-event TP/SL; engine uses global ATR_MULT only
        _logger.debug("[BRIDGE] exit_policy_mode=label_only; zeroing per-event TP/SL columns")
        alpha_panel = alpha_panel.copy()
        alpha_panel["candidate_stop_atr_mult"] = 0.0
        alpha_panel["candidate_take_profit_atr_mult"] = 0.0

    _emit_bridge_profile()

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
            "wf_selected_total": wf_selected_total,
            "wf_eligible_total": wf_eligible_total,
            "wf_fold_realized_mean_bps": wf_fold_realized_mean_bps,
            "wf_fold_log_growth_mean": wf_fold_log_growth_mean,
            "wf_shadow_profile_count": wf_shadow_profile_count,
            "wf_shadow_max_selected_total": wf_shadow_max_selected_total,
            "wf_shadow_max_eligible": wf_shadow_max_eligible,
            "wf_waterfall_expected_utility_p90_bps": wf_waterfall_expected_utility_p90_bps,
            "wf_waterfall_downside_drag_p90_bps": wf_waterfall_downside_drag_p90_bps,
            "wf_fold_details": wf_fold_details,
            "selected_pre_group": int(selection_diag.get("selected_pre_group", len(selected))),
            "selected_total": int(selection_diag.get("selected_total", len(selected))),
            "eligible": int(selection_diag.get("eligible", 0)),
            "n_keep": int(selection_diag.get("n_keep", 0)),
            "policy": str(selection_diag.get("policy", strategy_cfg.candidate.selection_policy)),
            "zero_reason": str(selection_diag.get("zero_reason", "unknown")),
            "workflow_status": (
                CandidateWorkflowStatus.WF_ELIGIBLE.value
                if int(selection_diag.get("selected_total", len(selected))) > 0
                else CandidateWorkflowStatus.BLOCKED.value
            ),
            "breakeven_floor_bps": float(
                selection_diag.get("breakeven_floor_bps", strategy_cfg.candidate.cost_floor_bps)
            ),
            "recommended_keep_variants": diag.recommended_keep_variants,
            "recommended_flip_variants": diag.recommended_flip_variants,
            "recommended_keep_signal_cells": diag.recommended_keep_signal_cells,
            "recommended_flip_signal_cells": diag.recommended_flip_signal_cells,
            "failure_report": diag.recommendation_failure_report,
            "recommendation_basis": diag.recommendation_basis,
            "recommendation_start": int(diag.recommendation_split[0]),
            "recommendation_end": int(diag.recommendation_split[1]),
            "report_start": int(diag.report_split[0]),
            "report_end": int(diag.report_split[1]),
        },
        aligned=aligned,
        labeled=labeled,
        labeled_unfiltered=labeled_all,
        fit_set=last_fold_out.fit_set if last_fold_out else None,
        calibration_set=last_fold_out.calibration_set if last_fold_out else None,
        oos_set=last_fold_out.oos_set if last_fold_out else None,
        gate_model=last_fold_out.gate_model if last_fold_out else None,
        edge_models=last_fold_out.edge_models if last_fold_out else None,
        fit_start=wf_folds[0].fit_start,
        fit_end=wf_folds[-1].fit_end,
        calibration_start=wf_folds[0].cal_start,
        calibration_end=wf_folds[-1].cal_end,
        oos_start=wf_folds[0].oos_start,
        oos_end=wf_folds[-1].oos_end,
        fold_oos_boundaries=_fold_oos_boundaries,
    )


def merge_candidate_output_into_data_maps(
    candidate_out: CandidatePipelineOutput,
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    log_tag: str = "",
) -> None:
    """Merge candidate output payload into data maps."""
    import time
    t_merge_start_all = time.perf_counter()
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

    # Hoist pd.to_datetime out of the loop for right dataframe
    panel_df = panel.reset_index() if "symbol" not in panel.columns else panel.copy()
    panel_df["_merge_datetime"] = pd.to_datetime(panel_df["datetime"], utc=True).dt.tz_localize(None)
    by_sym = panel_df.groupby("symbol", sort=False)

    for sym in symbols:
        t_sym = time.perf_counter()
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

        # Skip pd.to_datetime inside the loop if already datetime64 type
        df_dt = df["datetime"]
        if pd.api.types.is_datetime64_any_dtype(df_dt):
            left_merge_dt = df_dt.dt.tz_convert(None) if isinstance(df_dt.dtype, pd.DatetimeTZDtype) else df_dt
        else:
            left_merge_dt = pd.to_datetime(df_dt, utc=True).dt.tz_localize(None)

        left = pd.DataFrame({"_merge_datetime": left_merge_dt})
        right = sym_rows[["_merge_datetime", *list(required)]]
        merged = left.merge(right, on="_merge_datetime", how="left")

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
        # Log fast sym merge details under debug to prevent verbose log flood
        _logger.debug("[PROFILE][MERGE] sym %s took %.4fs", sym, time.perf_counter() - t_sym)
    _logger.info("[PROFILE][MERGE] Total merge %s took %.4fs", log_tag, time.perf_counter() - t_merge_start_all)


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
