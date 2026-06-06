from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.backtest.engine import FuturesBacktestEngine
from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput, EdgeSource
from src.domain.futures.strategy.candidate_dataset import build_candidate_dataset
from src.domain.futures.strategy.candidate_edge import fit_candidate_edge_models, predict_candidate_edges
from src.domain.futures.strategy.candidate_evaluation import evaluate_compound_backtest
from src.domain.futures.strategy.candidate_gate import fit_candidate_gate, predict_candidate_gate
from src.domain.futures.strategy.candidate_labels import label_candidate_events
from src.domain.futures.strategy.candidate_portfolio import (
    build_candidate_target_weights,
    select_candidate_events_for_portfolio,
)
from src.domain.futures.strategy.common.alignment import AlignedMarketData, align_data_maps
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.rule_diagnostics import RuleDiagnosticsResult, compute_rule_diagnostics
from src.domain.futures.strategy.rule_signals import build_rule_signal_panels, candidate_panels_to_events
from src.domain.futures.strategy_runtime.bridge import _candidate_ml_split_indices, _recommendation_window_indices

_logger = logging.getLogger(__name__)


def _is_nan(v: float) -> bool:
    return v != v  # NaN-safe check without math import


@dataclass(slots=True, frozen=True)
class SignalValidationReport:
    """Signal-only validation result (produced before ML training)."""

    variant: str
    n_events: int
    net_edge_bps_p50: float
    net_edge_bps_stress_p50: float
    net_edge_bps_mean: float
    net_edge_bps_stress_mean: float
    hit_rate: float
    hac_t_stat: float
    survives_cost: bool
    deployment_count: int
    decision_bar_count: int


@dataclass(slots=True, frozen=True)
class EdgeAttributionReport:
    """Per-variant predicted vs realised edge attribution."""

    variant: str
    trade_count: int
    deployed_bar_fraction: float
    pred_edge_bps_p50: float
    real_edge_bps_p50: float
    edge_capture_ratio: float
    gross_cost_bps: float
    turnover_total: float


@dataclass(slots=True, frozen=True)
class AblationRow:
    """Represents a row in the ablation comparison study."""

    variant: str
    mean_log_growth: float
    cagr: float
    max_drawdown: float
    mar: float
    turnover: float
    final_equity: float
    pass_compound_gate: bool
    # Attribution fields (default 0/False for backward-compat)
    trade_count: int = 0
    deployed_bar_fraction: float = 0.0
    pred_edge_bps_p50: float = float("nan")
    real_edge_bps_p50: float = float("nan")
    edge_capture_ratio: float = float("nan")
    gross_cost_bps: float = float("nan")
    pass_deployment_gate: bool = False


def _variant_key(frame: pd.DataFrame) -> pd.Series:
    return frame["family"].astype(str).str.cat(frame["variant"].astype(str), sep=":")


def _diagnostic_labeled_events(
    *,
    labeled: pd.DataFrame,
    cfg: CandidateStrategyConfig,
    fit_end: int,
    calibration_start: int,
) -> pd.DataFrame:
    """Drop purge-gap rows from diagnostics when using fit+calibration recommendations."""
    if cfg.promotion_decision_split != "fit_calibration":
        return labeled
    entry_idx = pd.to_numeric(labeled["entry_idx"], errors="coerce")
    return labeled.loc[(entry_idx < fit_end) | (entry_idx >= calibration_start)].copy()


def _oos_only_events(*, labeled: pd.DataFrame, oos_start: int, oos_end: int) -> pd.DataFrame:
    """Return events whose entry falls inside the OOS report window only."""
    if labeled.empty:
        return labeled
    entry_idx = pd.to_numeric(labeled["entry_idx"], errors="coerce")
    return labeled.loc[(entry_idx >= oos_start) & (entry_idx < oos_end)].copy()


def _stress_edge_bps(
    *,
    edge_bps: NDArray[np.float64],
    base_cost_bps: NDArray[np.float64],
    stress_multiplier: float,
) -> NDArray[np.float64]:
    stress_cost_bps = base_cost_bps * stress_multiplier
    return edge_bps - (stress_cost_bps - base_cost_bps)


def _decision_bar_edge_series(events: pd.DataFrame, edge_column: str) -> NDArray[np.float64]:
    if events.empty or edge_column not in events.columns or "entry_idx" not in events.columns:
        return np.zeros((0,), dtype=np.float64)
    frame = events.copy()
    frame["_edge"] = pd.to_numeric(frame[edge_column], errors="coerce")
    frame["_entry_idx"] = pd.to_numeric(frame["entry_idx"], errors="coerce")
    if "symbol" in frame.columns:
        grouped = (
            frame.dropna(subset=["_edge", "_entry_idx"])
            .groupby(["_entry_idx", "symbol"], sort=True, as_index=False)["_edge"]
            .mean()
            .groupby("_entry_idx", sort=True)["_edge"]
            .mean()
        )
    else:
        grouped = (
            frame.dropna(subset=["_edge", "_entry_idx"])
            .groupby("_entry_idx", sort=True)["_edge"]
            .mean()
        )
    return np.asarray(grouped.to_numpy(dtype=np.float64, copy=False), dtype=np.float64)


def _newey_west_t_stat(series: NDArray[np.float64], max_lag: int) -> float:
    finite = np.asarray(series, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    n_obs = finite.size
    if n_obs < 2:
        return 0.0
    mean_val = float(np.mean(finite))
    centered = finite - mean_val
    lag = min(max_lag, n_obs - 1)
    gamma0 = float(np.dot(centered, centered) / n_obs)
    long_run_var = gamma0
    for idx in range(1, lag + 1):
        gamma = float(np.dot(centered[idx:], centered[:-idx]) / n_obs)
        weight = 1.0 - (idx / (lag + 1))
        long_run_var += 2.0 * weight * gamma
    stderr = float(np.sqrt(max(long_run_var, 0.0) / n_obs))
    return mean_val / (stderr + 1e-12)


def _build_calibration_variant_priors(
    *,
    calibration_set: Any,
    cfg: CandidateStrategyConfig,
) -> tuple[dict[str, float], float]:
    """Return calibration-set variant priors and the fallback global prior."""
    if calibration_set.X.shape[0] == 0 or calibration_set.event_index.empty:
        return {}, 0.0

    keys = (
        calibration_set.event_index["family"].astype(str).str.cat(
            calibration_set.event_index["variant"].astype(str), sep=":"
        )
        if {"family", "variant"}.issubset(calibration_set.event_index.columns)
        else pd.Series(["__global__"] * calibration_set.X.shape[0], dtype=object)
    )
    y_edge = np.asarray(calibration_set.y_edge_bps, dtype=np.float64)
    weights = np.asarray(calibration_set.edge_weight, dtype=np.float64)
    global_mask = np.isfinite(y_edge) & np.isfinite(weights) & (weights > 0.0)
    if bool(global_mask.any()):
        global_prior = float(np.average(y_edge[global_mask], weights=weights[global_mask]))
    else:
        finite_values = y_edge[np.isfinite(y_edge)]
        global_prior = float(np.mean(finite_values)) if finite_values.size > 0 else 0.0
    variant_prior_bps: dict[str, float] = {}
    key_to_indices: dict[str, list[int]] = {}
    for idx, key in enumerate(keys):
        key_to_indices.setdefault(str(key), []).append(idx)

    shrinkage_obs = float(cfg.edge_prior_shrinkage_obs)
    min_obs = int(cfg.edge_prior_min_obs)
    for key, indices in key_to_indices.items():
        indexer = np.asarray(indices, dtype=np.int32)
        obs = int(indexer.shape[0])
        if obs < min_obs:
            continue
        variant_values = y_edge[indexer]
        variant_weights = weights[indexer]
        finite_mask = np.isfinite(variant_values) & np.isfinite(variant_weights) & (variant_weights > 0.0)
        if bool(finite_mask.any()):
            variant_mean = float(np.average(variant_values[finite_mask], weights=variant_weights[finite_mask]))
        else:
            finite_values = variant_values[np.isfinite(variant_values)]
            variant_mean = float(np.mean(finite_values)) if finite_values.size > 0 else global_prior
        shrink = obs / (obs + shrinkage_obs)
        variant_prior_bps[key] = float(shrink * variant_mean + (1.0 - shrink) * global_prior)
    return variant_prior_bps, global_prior


def _utility_min_threshold(*, utility_score: np.ndarray, cfg: CandidateStrategyConfig) -> float:
    """Return a utility threshold computed from the provided utility scores."""
    finite = np.asarray(utility_score, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("-inf")
    quantile = max(0.0, min(1.0, 1.0 - float(cfg.selection_top_quantile)))
    return float(np.quantile(finite, quantile))


def _compute_rule_diagnostics_for_ablation(
    *,
    labeled: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    fit_start: int,
    fit_end: int,
    calibration_start: int,
    calibration_end: int,
    oos_start: int,
    oos_end: int,
) -> tuple[RuleDiagnosticsResult, RuleDiagnosticsResult]:
    """Return no-leak and oracle rule diagnostics for ablation."""
    recommendation_start, recommendation_end = _recommendation_window_indices(
        fit_start=fit_start,
        fit_end=fit_end,
        calibration_start=calibration_start,
        calibration_end=calibration_end,
        cfg=cfg,
    )
    labeled_for_diag = _diagnostic_labeled_events(
        labeled=labeled,
        cfg=cfg,
        fit_end=fit_end,
        calibration_start=calibration_start,
    )
    diag_no_leak = compute_rule_diagnostics(
        labeled_events=labeled_for_diag,
        aligned=aligned,
        cfg=cfg,
        min_obs=max(cfg.min_candidate_obs, 100),
        recommendation_start=recommendation_start,
        recommendation_end=recommendation_end,
        report_start=oos_start,
        report_end=oos_end,
        silent=True,
    )
    diag_oracle = compute_rule_diagnostics(
        labeled_events=labeled_for_diag,
        aligned=aligned,
        cfg=cfg,
        min_obs=max(cfg.min_candidate_obs, 100),
        recommendation_start=oos_start,
        recommendation_end=oos_end,
        report_start=oos_start,
        report_end=oos_end,
        silent=True,
    )
    return diag_no_leak, diag_oracle


def apply_variant_promotions(
    *,
    labeled: pd.DataFrame,
    keep_variants: tuple[str, ...],
    flip_variants: tuple[str, ...],
    keep_signal_cells: tuple[str, ...] = (),
    flip_signal_cells: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Filter labeled events to only recommended variants.

    Returns empty DataFrame (fail-closed) when no variants are recommended,
    so callers must guard for empty result.
    """
    if labeled.empty:
        return labeled

    allowed_signal_cells = set(keep_signal_cells) | set(flip_signal_cells)
    allowed_variants = set(keep_variants) | set(flip_variants)
    allowed = allowed_signal_cells | allowed_variants
    if not allowed:
        _logger.warning("[PROMO_FILTER] no variants recommended by diagnostics; blocking all candidates (fail-closed)")
        return labeled.iloc[0:0].copy()

    if allowed_signal_cells and "signal_cell" in labeled.columns:
        out = labeled.loc[labeled["signal_cell"].astype(str).isin(allowed_signal_cells)].copy()
    else:
        out = labeled.loc[_variant_key(labeled).isin(allowed_variants)].copy()
    if out.empty:
        _logger.warning("[PROMO_FILTER] all candidates removed after variant filter; returning empty")
        return labeled.iloc[0:0].copy()

    flip_mask = pd.Series(False, index=out.index)
    if flip_signal_cells and "signal_cell" in out.columns:
        flip_mask = out["signal_cell"].astype(str).isin(set(flip_signal_cells))
    elif flip_variants:
        flip_mask = _variant_key(out).isin(set(flip_variants))
    if bool(flip_mask.any()):
        out.loc[flip_mask, "side"] = -pd.to_numeric(out.loc[flip_mask, "side"], errors="coerce")
        if "raw_score" in out.columns:
            out.loc[flip_mask, "raw_score"] = -pd.to_numeric(out.loc[flip_mask, "raw_score"], errors="coerce")
        if "score_z" in out.columns:
            out.loc[flip_mask, "score_z"] = -pd.to_numeric(out.loc[flip_mask, "score_z"], errors="coerce")
        out.loc[flip_mask, "side_flipped"] = True
        if "signal_cell" in out.columns:
            out.loc[flip_mask, "signal_cell"] = out.loc[flip_mask, "signal_cell"].astype(str)
    return out.reset_index(drop=True)


def _build_rule_equal_size_weights(
    *,
    raw_events: pd.DataFrame,
    close_2d: np.ndarray,
    symbols: tuple[str, ...],
    max_symbol_weight: float,
) -> np.ndarray:
    raw_w = np.zeros_like(close_2d)
    n_bars = close_2d.shape[0]
    for row in raw_events.itertuples(index=False):
        t = int(row.entry_idx)
        for s_idx, sym in enumerate(symbols):
            if sym == row.symbol and 0 <= t < n_bars:
                raw_w[t, s_idx] = float(row.side) * max_symbol_weight
    return raw_w


def _build_uncapped_kelly_edge_weights(
    *,
    selected_events: pd.DataFrame,
    close_2d: np.ndarray,
    symbols: tuple[str, ...],
    kelly_fraction: float,
) -> np.ndarray:
    n_times, n_symbols = close_2d.shape
    raw_kelly_edge_w = np.zeros((n_times, n_symbols), dtype=np.float64)
    sym_to_idx = {sym: idx for idx, sym in enumerate(symbols)}
    for row in selected_events.itertuples(index=False):
        sym = str(row.symbol)
        if sym not in sym_to_idx:
            continue
        s_idx = sym_to_idx[sym]
        t = int(row.entry_idx)
        if not (0 <= t < n_times):
            continue

        side = float(row.side)
        holding_bars = max(int(getattr(row, "expected_holding_bars", 1)), 1)
        mu_i_per_bar = float(row.mu_net_decision_bps) * 1e-4 / holding_bars

        st = max(0, t - 20)
        variance_i = 1e-4
        if t > st:
            ret = np.diff(close_2d[st : t + 1, s_idx]) / np.maximum(close_2d[st:t, s_idx], 1e-12)
            v = float(np.var(ret))
            if np.isfinite(v) and v > 1e-12:
                variance_i = v
        raw_w_val = kelly_fraction * mu_i_per_bar / max(variance_i, 1e-12)
        raw_kelly_edge_w[t, s_idx] = raw_w_val * np.sign(side)
    return raw_kelly_edge_w


def _build_variant_prior_output(
    *,
    edge_models: Any,
    calibration_set: Any,
    oos_set: Any,
    p_pass: np.ndarray,
    cfg: CandidateStrategyConfig,
) -> CandidateModelOutput:
    """Construct a prior-only edge output without residual center predictions."""
    base_out = predict_candidate_edges(models=edge_models, dataset=oos_set, p_pass=p_pass, cfg=cfg)
    variant_prior_bps, global_prior_bps = _build_calibration_variant_priors(
        calibration_set=calibration_set,
        cfg=cfg,
    )
    keys = (
        oos_set.event_index["family"].astype(str).str.cat(oos_set.event_index["variant"].astype(str), sep=":")
        if not oos_set.event_index.empty
        else pd.Series(["__global__"] * oos_set.X.shape[0], dtype=object)
    )
    prior_mu = np.asarray(
        [variant_prior_bps.get(str(key), global_prior_bps) for key in keys],
        dtype=np.float64,
    )
    turnover_proxy = np.ones_like(prior_mu)
    if "turnover_proxy" in oos_set.feature_names:
        t_idx = oos_set.feature_names.index("turnover_proxy")
        turnover_proxy = oos_set.X[:, t_idx].astype(np.float64, copy=False)
    utility_score = (
        p_pass * prior_mu
        - float(cfg.downside_penalty) * np.abs(np.minimum(base_out.q10_net_bps, 0.0))
        - float(cfg.turnover_penalty) * turnover_proxy
        - float(cfg.concentration_penalty)
    )
    validation_diagnostics = dict(base_out.validation_diagnostics)
    validation_diagnostics["utility_min"] = _utility_min_threshold(
        utility_score=utility_score,
        cfg=cfg,
    )

    risk_unit_raw = getattr(oos_set, "risk_unit_bps", None)
    risk_unit_bps = (
        risk_unit_raw.astype(np.float64, copy=False)
        if risk_unit_raw is not None
        else np.full(oos_set.X.shape[0], float(getattr(cfg, "min_risk_unit_bps", 25.0)), dtype=np.float64)
    )
    expected_return_r = prior_mu / risk_unit_bps

    return CandidateModelOutput(
        events=oos_set.event_index,
        p_pass=p_pass.astype(np.float64, copy=False),
        gate_enabled=base_out.gate_enabled,
        gate_threshold=base_out.gate_threshold,
        edge_source=EdgeSource.PRIOR_ONLY,
        expected_return_r=expected_return_r,
        expected_net_bps=prior_mu,
        q10_return_r=base_out.q10_return_r,
        q10_net_bps=base_out.q10_net_bps,
        q90_return_r=base_out.q90_return_r,
        q90_net_bps=base_out.q90_net_bps,
        selection_score=utility_score,
        kelly_fraction=base_out.kelly_fraction,
        validation_diagnostics=validation_diagnostics,
    )


def run_candidate_ablation(
    *,
    data_maps: dict[str, dict[str, Any]],
    symbols: tuple[str, ...],
    tf: str,
    cfg: CandidateStrategyConfig,
) -> pd.DataFrame:
    """Run ablation variants to prove each complexity layer adds compounding value."""
    # 1. Align market data
    aligned = align_data_maps(data_maps, list(symbols), tf)
    
    # 2. Build hypothesis rule candidates
    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)
    raw_events = candidate_panels_to_events(
        panels,
        min_abs_score=cfg.min_rule_net_bps * 1e-4,
        side_flip_variants=cfg.side_flip_candidate_variants,
        cost_floor_bps=cfg.cost_floor_bps,
        execution_cost_bps_2d=aligned.execution_cost_bps_2d,
    )

    if raw_events.empty:
        return pd.DataFrame(columns=[
            "variant", "mean_log_growth", "cagr", "max_drawdown", "mar",
            "turnover", "final_equity", "pass_compound_gate",
            "trade_count", "deployed_bar_fraction", "pred_edge_bps_p50",
            "real_edge_bps_p50", "edge_capture_ratio", "gross_cost_bps",
            "pass_deployment_gate",
        ])

    # 3. Label events and split dataset
    labeled = label_candidate_events(events=raw_events, aligned=aligned, cfg=cfg)
    n_bars = aligned.close_2d.shape[0]
    fit_start, fit_end, calibration_start, calibration_end, oos_start, oos_end = _candidate_ml_split_indices(
        n_bars=n_bars,
        fit_fraction=cfg.ml_fit_fraction,
        calibration_fraction=cfg.ml_calibration_fraction,
        purge_bars=cfg.purge_bars,
        embargo_bars=cfg.embargo_bars,
    )
    # Compute WF folds for fold_oos_boundaries (used by evaluate_compound_backtest DSR/PBO)
    from src.domain.futures.strategy.walk_forward import build_walk_forward_folds
    _wf_folds = build_walk_forward_folds(n_bars=n_bars, cfg=cfg)
    _fold_oos_boundaries: tuple[tuple[int, int], ...] = tuple(
        (f.oos_start, f.oos_end) for f in _wf_folds
    )
    diag, _ = _compute_rule_diagnostics_for_ablation(
        labeled=labeled,
        aligned=aligned,
        cfg=cfg,
        fit_start=fit_start,
        fit_end=fit_end,
        calibration_start=calibration_start,
        calibration_end=calibration_end,
        oos_start=oos_start,
        oos_end=oos_end,
    )
    _logger.info(
        "[DIAG][RULE_RECOMMEND_ABLATION] keep=%s flip=%s",
        ",".join(diag.recommended_keep_variants) if diag.recommended_keep_variants else "",
        ",".join(diag.recommended_flip_variants) if diag.recommended_flip_variants else "",
    )
    labeled_unfiltered = labeled  # save before promotion filter for ablation row 7

    if cfg.promotion_filter_enabled:
        labeled = apply_variant_promotions(
            labeled=labeled,
            keep_variants=diag.recommended_keep_variants,
            flip_variants=diag.recommended_flip_variants,
            keep_signal_cells=diag.recommended_keep_signal_cells,
            flip_signal_cells=diag.recommended_flip_signal_cells,
        )
        if labeled.empty:
            _logger.warning(
                "[ABLATION][PROMO_FILTER] all candidates blocked; "
                "falling back to unfiltered events for ML training"
            )

    # When promo filter blocks all variants, fall back to unfiltered data so ML
    # variants still produce informative (non-crash) results for comparison.
    _labeled_for_ml = labeled if not labeled.empty else labeled_unfiltered
    fit_set = build_candidate_dataset(
        labeled_events=_labeled_for_ml, aligned=aligned, cfg=cfg, split_start=fit_start, split_end=fit_end
    )
    calibration_set = build_candidate_dataset(
        labeled_events=_labeled_for_ml,
        aligned=aligned,
        cfg=cfg,
        split_start=calibration_start,
        split_end=calibration_end,
    )
    oos_set = build_candidate_dataset(
        labeled_events=_labeled_for_ml, aligned=aligned, cfg=cfg, split_start=oos_start, split_end=oos_end
    )

    # 4. Train ML Models
    gate_model = fit_candidate_gate(train=fit_set, early_stop=calibration_set, calibration=calibration_set, cfg=cfg)
    edge_models = fit_candidate_edge_models(
        train=fit_set, valid=calibration_set, calibration_eval=calibration_set, cfg=cfg
    )

    # 5. Predict outcomes for OOS sample only
    p_pass = predict_candidate_gate(model=gate_model, dataset=oos_set)
    ml_out = predict_candidate_edges(models=edge_models, dataset=oos_set, p_pass=p_pass, cfg=cfg)
    ml_out = replace(ml_out, events=oos_set.event_index)

    rows: list[AblationRow] = []

    # 1. rule_stop_risk (Raw trigger rules with stop-risk sizing and no ML components)
    cfg_rule = replace(
        cfg,
        sizing_mode="stop_risk",
        gross_cap=999.0,
        net_cap=999.0,
        beta_cap=999.0,
        target_ann_vol=999.0,
    )
    rule_events = _oos_only_events(labeled=labeled_unfiltered, oos_start=oos_start, oos_end=oos_end)
    rule_w = build_candidate_target_weights(
        selected_events=rule_events,
        close_2d=aligned.close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=cfg_rule,
    )
    rows.append(
        _run_backtest_and_evaluate(
            rule_w,
            aligned,
            "rule_stop_risk",
            cfg,
            start_idx=oos_start,
            end_idx=oos_end,
            barrier_events=rule_events,
            fold_oos_boundaries=_fold_oos_boundaries,
        )
    )

    # 2. prior_rank_stop_risk (Rule candidates filtered by prior variant-level rank selection only)
    p_pass_ones = np.ones(oos_set.X.shape[0], dtype=np.float64)
    prior_out = _build_variant_prior_output(
        edge_models=edge_models,
        calibration_set=calibration_set,
        oos_set=oos_set,
        p_pass=p_pass_ones,
        cfg=cfg,
    )
    prior_selected = select_candidate_events_for_portfolio(model_output=prior_out, cfg=cfg)
    prior_w = build_candidate_target_weights(
        selected_events=prior_selected,
        close_2d=aligned.close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=replace(cfg, sizing_mode="stop_risk", gross_cap=999.0, net_cap=999.0, beta_cap=999.0, target_ann_vol=999.0),
    )
    rows.append(
        _run_backtest_and_evaluate(
            prior_w,
            aligned,
            "prior_rank_stop_risk",
            cfg,
            start_idx=oos_start,
            end_idx=oos_end,
            selected_events=prior_selected,
            fold_oos_boundaries=_fold_oos_boundaries,
        )
    )

    # 3. prior_residual_rank_stop_risk (Adds ML residual model to rank selection, but no gate veto)
    edge_out_nogate = predict_candidate_edges(models=edge_models, dataset=oos_set, p_pass=p_pass_ones, cfg=cfg)
    edge_out_nogate = replace(edge_out_nogate, events=oos_set.event_index)
    residual_selected = select_candidate_events_for_portfolio(model_output=edge_out_nogate, cfg=cfg)
    residual_w = build_candidate_target_weights(
        selected_events=residual_selected,
        close_2d=aligned.close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=replace(cfg, sizing_mode="stop_risk", gross_cap=999.0, net_cap=999.0, beta_cap=999.0, target_ann_vol=999.0),
    )
    rows.append(
        _run_backtest_and_evaluate(
            residual_w,
            aligned,
            "prior_residual_rank_stop_risk",
            cfg,
            start_idx=oos_start,
            end_idx=oos_end,
            selected_events=residual_selected,
            fold_oos_boundaries=_fold_oos_boundaries,
        )
    )

    # 4. edge_plus_validated_gate_stop_risk (Adds ML gate veto to selection, retains stop-risk sizing)
    edge_out_gate = predict_candidate_edges(models=edge_models, dataset=oos_set, p_pass=p_pass, cfg=cfg)
    edge_out_gate = replace(edge_out_gate, events=oos_set.event_index)
    gate_selected = select_candidate_events_for_portfolio(model_output=edge_out_gate, cfg=cfg)
    gate_w = build_candidate_target_weights(
        selected_events=gate_selected,
        close_2d=aligned.close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=replace(cfg, sizing_mode="stop_risk", gross_cap=999.0, net_cap=999.0, beta_cap=999.0, target_ann_vol=999.0),
    )
    rows.append(
        _run_backtest_and_evaluate(
            gate_w,
            aligned,
            "edge_plus_validated_gate_stop_risk",
            cfg,
            start_idx=oos_start,
            end_idx=oos_end,
            selected_events=gate_selected,
            fold_oos_boundaries=_fold_oos_boundaries,
        )
    )

    # 5. edge_plus_gate_event_kelly (Replaces stop-risk sizing with calibrated event Kelly, portfolio caps bypassed)
    kelly_w = build_candidate_target_weights(
        selected_events=gate_selected,
        close_2d=aligned.close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=replace(
            cfg,
            sizing_mode="calibrated_event_kelly",
            gross_cap=999.0,
            net_cap=999.0,
            beta_cap=999.0,
            target_ann_vol=999.0,
        ),
    )
    rows.append(
        _run_backtest_and_evaluate(
            kelly_w,
            aligned,
            "edge_plus_gate_event_kelly",
            cfg,
            start_idx=oos_start,
            end_idx=oos_end,
            selected_events=gate_selected,
            fold_oos_boundaries=_fold_oos_boundaries,
        )
    )

    # 6. full_portfolio_caps (Applies final portfolio constraints/caps projection to Kelly weights)
    full_caps_w = build_candidate_target_weights(
        selected_events=gate_selected,
        close_2d=aligned.close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=replace(cfg, sizing_mode="calibrated_event_kelly"),
    )
    rows.append(
        _run_backtest_and_evaluate(
            full_caps_w,
            aligned,
            "full_portfolio_caps",
            cfg,
            start_idx=oos_start,
            end_idx=oos_end,
            selected_events=gate_selected,
            fold_oos_boundaries=_fold_oos_boundaries,
        )
    )

    # Convert results to DataFrame
    df_results = pd.DataFrame([
        {
            "variant": r.variant,
            "mean_log_growth": r.mean_log_growth,
            "cagr": r.cagr,
            "max_drawdown": r.max_drawdown,
            "mar": r.mar,
            "turnover": r.turnover,
            "final_equity": r.final_equity,
            "pass_compound_gate": r.pass_compound_gate,
            "trade_count": r.trade_count,
            "deployed_bar_fraction": r.deployed_bar_fraction,
            "pred_edge_bps_p50": r.pred_edge_bps_p50,
            "real_edge_bps_p50": r.real_edge_bps_p50,
            "edge_capture_ratio": r.edge_capture_ratio,
            "gross_cost_bps": r.gross_cost_bps,
            "pass_deployment_gate": r.pass_deployment_gate,
        }
        for r in rows
    ])

    return df_results



def _build_barrier_arrays(
    *,
    selected_events: pd.DataFrame,
    n_times: int,
    n_symbols: int,
    symbols: tuple[str, ...],
    start_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build [T, N] stop/TP barrier multiplier arrays for engine consumption.

    Fills the entry bar and forward-fills over the holding window (same pattern
    as target_weights) so the engine can apply the same barrier semantics that
    were used during triple-barrier labeling.
    """
    stop_2d = np.zeros((n_times, n_symbols), dtype=np.float64)
    tp_2d = np.zeros((n_times, n_symbols), dtype=np.float64)
    if selected_events.empty:
        return stop_2d, tp_2d
    # Defensive sort: oldest entry first so the first event fills each slot.
    # selected_events is normally already sorted but this guards against
    # out-of-order inputs (e.g., unsorted rule event frames).
    if "entry_idx" in selected_events.columns:
        selected_events = selected_events.sort_values("entry_idx", kind="stable").reset_index(drop=True)
    sym_to_idx = {sym: idx for idx, sym in enumerate(symbols)}
    for row in selected_events.itertuples(index=False):
        sym = str(row.symbol)
        if sym not in sym_to_idx:
            continue
        s_idx = sym_to_idx[sym]
        local_t = int(row.entry_idx) - start_idx
        if local_t < 0 or local_t >= n_times:
            continue
        stop_val = float(getattr(row, "stop_atr_mult", 0.0))
        tp_val = float(getattr(row, "take_profit_atr_mult", 0.0))
        holding = max(int(getattr(row, "expected_holding_bars", 1)), 1)
        fill_end = min(local_t + holding, n_times)
        for fill_t in range(local_t, fill_end):
            if stop_2d[fill_t, s_idx] == 0.0:
                stop_2d[fill_t, s_idx] = stop_val
            if tp_2d[fill_t, s_idx] == 0.0:
                tp_2d[fill_t, s_idx] = tp_val
    return stop_2d, tp_2d


def _compute_realized_edge(trades: pd.DataFrame) -> float:
    """Return median realized net return in bps from engine trade records.

    Uses net edge formula:
    net_trade_bps = (trade.pnl - trade.entry_fee) / (trade.entry_price * trade.amount) * 10_000
    """
    if trades.empty:
        return float("nan")
    required = {"pnl", "entry_fee", "entry_price", "amount"}
    if not required.issubset(trades.columns):
        return float("nan")
    pnl = trades["pnl"].to_numpy(dtype=np.float64)
    entry_fee = trades["entry_fee"].to_numpy(dtype=np.float64)
    entry_px = trades["entry_price"].to_numpy(dtype=np.float64)
    amount = trades["amount"].to_numpy(dtype=np.float64)

    denominator = entry_px * amount
    valid = (
        np.isfinite(pnl)
        & np.isfinite(entry_fee)
        & np.isfinite(entry_px)
        & np.isfinite(amount)
        & (denominator > 1e-12)
    )
    if not bool(valid.any()):
        return float("nan")
    net_bps = (pnl[valid] - entry_fee[valid]) / denominator[valid] * 10_000
    return float(np.median(net_bps))


def _compute_attribution(
    *,
    target_weights_eval: np.ndarray,
    selected_events: pd.DataFrame | None,
    trades: pd.DataFrame,
    cfg: CandidateStrategyConfig,
) -> tuple[int, float, float, float]:
    """Return (trade_count, deployed_bar_fraction, pred_edge_p50, gross_cost_bps)."""
    n_bars = target_weights_eval.shape[0]
    gross = np.abs(target_weights_eval).sum(axis=1) if target_weights_eval.ndim == 2 else np.abs(target_weights_eval)
    trade_count = len(trades)
    deployed_bar_fraction = float((gross > 1e-9).mean()) if n_bars > 0 else 0.0

    pred_edge_p50 = float("nan")
    if selected_events is not None and not selected_events.empty and "mu_net_decision_bps" in selected_events.columns:
        mu_vals = pd.to_numeric(selected_events["mu_net_decision_bps"], errors="coerce").dropna().to_numpy()
        if mu_vals.size > 0:
            pred_edge_p50 = float(np.median(mu_vals))

    cost_2d = cfg.cost_floor_bps * 1e-4
    if target_weights_eval.ndim == 2:
        delta_w = np.abs(np.diff(target_weights_eval, axis=0, prepend=0.0))
    else:
        delta_w = np.abs(np.diff(target_weights_eval, prepend=0.0))
    gross_cost_bps = float(delta_w.sum() * cost_2d * 1e4) if delta_w.size > 0 else float("nan")

    return trade_count, deployed_bar_fraction, pred_edge_p50, gross_cost_bps


def _run_backtest_and_evaluate(
    target_weights: np.ndarray,
    aligned: AlignedMarketData,
    variant_name: str,
    cfg: CandidateStrategyConfig,
    *,
    start_idx: int | None = None,
    end_idx: int | None = None,
    selected_events: pd.DataFrame | None = None,
    barrier_events: pd.DataFrame | None = None,
    fold_oos_boundaries: tuple[tuple[int, int], ...] | None = None,
) -> AblationRow:
    """Helper to inject target_weights into data_maps and run backtest simulation.

    Args:
        selected_events: Used for edge attribution metrics.  Also the barrier
            source when ``barrier_events`` is not provided.
        barrier_events: Explicit barrier source for variants where
            ``selected_events`` is None (e.g., rule-only variants that carry
            ``stop_atr_mult``/``take_profit_atr_mult`` columns but whose events
            are not used for attribution).  When both are provided,
            ``barrier_events`` wins for barrier construction.
    """
    from src.domain.futures.strategy.rule_signals import _atr_2d
    if start_idx is None and end_idx is None:
        aligned_eval = aligned
        target_weights_eval = target_weights
    else:
        st = 0 if start_idx is None else max(0, int(start_idx))
        ed = aligned.close_2d.shape[0] if end_idx is None else min(int(end_idx), aligned.close_2d.shape[0])
        aligned_eval = replace(
            aligned,
            datetimes=aligned.datetimes[st:ed],
            open_2d=aligned.open_2d[st:ed],
            high_2d=aligned.high_2d[st:ed],
            low_2d=aligned.low_2d[st:ed],
            close_2d=aligned.close_2d[st:ed],
            volume_2d=aligned.volume_2d[st:ed],
            funding_2d=aligned.funding_2d[st:ed],
            active_mask=aligned.active_mask[st:ed],
            warm_mask=aligned.warm_mask[st:ed],
            entry_block_mask=aligned.entry_block_mask[st:ed],
            kill_mask=aligned.kill_mask[st:ed],
            basis_2d=None if aligned.basis_2d is None else aligned.basis_2d[st:ed],
            oi_2d=None if aligned.oi_2d is None else aligned.oi_2d[st:ed],
            adv_usdt_2d=None if aligned.adv_usdt_2d is None else aligned.adv_usdt_2d[st:ed],
            execution_cost_bps_2d=(
                None if aligned.execution_cost_bps_2d is None else aligned.execution_cost_bps_2d[st:ed]
            ),
            inference_active_mask=(
                None if aligned.inference_active_mask is None else aligned.inference_active_mask[st:ed]
            ),
            inference_entry_warm_mask=(
                None if aligned.inference_entry_warm_mask is None else aligned.inference_entry_warm_mask[st:ed]
            ),
            cluster_id_1d=aligned.cluster_id_1d,
            beta_vs_market_1d=aligned.beta_vs_market_1d,
            cluster_size_1d=aligned.cluster_size_1d,
            anchor_cluster_1d=aligned.anchor_cluster_1d,
            symbol_meta=aligned.symbol_meta,
        )
        target_weights_eval = target_weights[st:ed]
    atr_2d = _atr_2d(aligned_eval.high_2d, aligned_eval.low_2d, aligned_eval.close_2d, period=14)

    # Phase 1 (RC1): track eval window start for global→local index remapping
    _eval_start = 0 if start_idx is None else max(0, int(start_idx))

    aligned_data = {
        "close": aligned_eval.close_2d,
        "high": aligned_eval.high_2d,
        "low": aligned_eval.low_2d,
        "open": aligned_eval.open_2d,
        "volume": aligned_eval.volume_2d,
        "atr": atr_2d,
        "target_weights": target_weights_eval,
    }
    # Phase 1 (RC1): wire TP/SL barriers so evaluation matches label semantics.
    # Without this, the engine holds every position for the full horizon with no
    # profit-take or stop, which diverges from the triple-barrier labeling used
    # to train the gate/edge models.
    # Fix 2 (audit): barrier_events is the explicit source for rule-only variants
    # that carry stop/tp columns but use None for selected_events (attribution).
    _barrier_source = barrier_events if barrier_events is not None else selected_events
    if cfg.eval_apply_candidate_barriers and _barrier_source is not None and not _barrier_source.empty:
        _n_times_eval, _n_syms_eval = target_weights_eval.shape
        _stop_2d, _tp_2d = _build_barrier_arrays(
            selected_events=_barrier_source,
            n_times=_n_times_eval,
            n_symbols=_n_syms_eval,
            symbols=aligned_eval.symbols,
            start_idx=_eval_start,
        )
        aligned_data["candidate_stop_atr_mult"] = _stop_2d
        aligned_data["candidate_take_profit_atr_mult"] = _tp_2d

    # Execute backtest engine
    trades, equity_curve, _, _ = FuturesBacktestEngine.run_multi(
        aligned_data=aligned_data,
        symbol_names=list(aligned_eval.symbols),
        strategy_params={},
    )

    # Phase 0 (diagnostic): compute realized net edge from engine trade records.
    # edge_capture_ratio = real / pred; values << 1 confirm RC1 is the primary cause.
    _real_edge_p50 = _compute_realized_edge(trades)

    # Phase 2 (RC2): compute attribution BEFORE evaluation so deployment can gate passing.
    trade_count, deployed_bar_fraction, pred_edge_p50, gross_cost_bps = _compute_attribution(
        target_weights_eval=target_weights_eval,
        selected_events=selected_events,
        trades=trades,
        cfg=cfg,
    )

    # Evaluate compounding growth
    # Remap fold_oos_boundaries to local (sliced) index space when start_idx is applied
    _local_boundaries: tuple[tuple[int, int], ...] | None = None
    if fold_oos_boundaries and start_idx is not None:
        _offset = int(start_idx)
        _local_boundaries = tuple(
            (max(0, s - _offset), max(0, e - _offset))
            for s, e in fold_oos_boundaries
            if e > _offset
        ) or None
    elif fold_oos_boundaries:
        _local_boundaries = fold_oos_boundaries
    report = evaluate_compound_backtest(
        trades=trades,
        equity_curve=equity_curve,
        cfg=cfg,
        fold_oos_boundaries=_local_boundaries,
        deployed_bar_fraction=deployed_bar_fraction,
        trade_count=trade_count,
    )

    # Deployment integrity gate
    pass_deployment_gate = (
        trade_count >= cfg.min_deployment_trade_count
        and deployed_bar_fraction >= cfg.min_deployment_capital_fraction
    )

    _capture_ratio = (
        float(_real_edge_p50 / pred_edge_p50)
        if (not _is_nan(_real_edge_p50) and not _is_nan(pred_edge_p50) and abs(pred_edge_p50) > 1e-9)
        else float("nan")
    )

    if cfg.edge_attribution_enabled:
        _logger.info(
            "[DIAG][EDGE_ATTRIB] variant=%s trades=%d deployed=%.3f "
            "pred_p50=%.1fbps real_p50=%.1fbps capture=%.3f "
            "cost=%.1fbps pass_deploy=%s",
            variant_name,
            trade_count,
            deployed_bar_fraction,
            pred_edge_p50 if not _is_nan(pred_edge_p50) else float("nan"),
            _real_edge_p50 if not _is_nan(_real_edge_p50) else float("nan"),
            _capture_ratio if not _is_nan(_capture_ratio) else float("nan"),
            gross_cost_bps if not _is_nan(gross_cost_bps) else float("nan"),
            pass_deployment_gate,
        )

    return AblationRow(
        variant=variant_name,
        mean_log_growth=report.mean_log_growth,
        cagr=report.cagr,
        max_drawdown=report.max_drawdown,
        mar=report.mar,
        turnover=report.turnover,
        final_equity=report.final_equity,
        pass_compound_gate=report.pass_compound_gate,
        trade_count=trade_count,
        deployed_bar_fraction=deployed_bar_fraction,
        pred_edge_bps_p50=pred_edge_p50,
        real_edge_bps_p50=_real_edge_p50,
        edge_capture_ratio=_capture_ratio,
        gross_cost_bps=gross_cost_bps,
        pass_deployment_gate=pass_deployment_gate,
    )


def validate_candidate_signals(
    *,
    labeled_all: pd.DataFrame,
    labeled_promoted: pd.DataFrame,
    cfg: CandidateStrategyConfig,
    oos_start: int,
    oos_end: int,
) -> list[SignalValidationReport]:
    """Produce signal-only validation reports without ML training.

    Runs variant 1 (rule_only_equal_size) and 1b (rule_promo_no_leak) only.
    Used by bridge when cfg.signal_only=True to cut off before ML training.
    """
    from src.domain.futures.strategy.execution_cost import ExecutionCostModel

    cost_model = ExecutionCostModel(
        maker_fee_bps=cfg.maker_fee_bps,
        taker_fee_bps=cfg.taker_fee_bps,
        maker_ratio=cfg.maker_ratio,
        slippage_bps=cfg.slippage_bps,
        impact_coeff_bps=cfg.impact_coeff_bps,
        stress_multiplier=cfg.cost_stress_multiplier,
    )
    reports: list[SignalValidationReport] = []

    for variant_name, events_df in [
        ("rule_only_equal_size", labeled_all),
        ("rule_promo_no_leak", labeled_promoted),
    ]:
        oos_events = _oos_only_events(labeled=events_df, oos_start=oos_start, oos_end=oos_end)
        n_events = len(oos_events)
        if n_events == 0:
            reports.append(SignalValidationReport(
                variant=variant_name,
                n_events=0,
                net_edge_bps_p50=float("nan"),
                net_edge_bps_stress_p50=float("nan"),
                net_edge_bps_mean=float("nan"),
                net_edge_bps_stress_mean=float("nan"),
                hit_rate=0.0,
                hac_t_stat=0.0,
                survives_cost=False,
                deployment_count=0,
                decision_bar_count=0,
            ))
            continue

        edge_col = "edge_after_hurdle_bps"
        edge_arr = (
            oos_events[edge_col].to_numpy(dtype=np.float64)
            if edge_col in oos_events.columns
            else np.zeros(n_events, dtype=np.float64)
        )
        # base 비용 복원 (이중차감 방지): edge_after_hurdle은 이미 base RT 차감됨
        if "ex_ante_cost_bps" in oos_events.columns:
            base_cost_arr = oos_events["ex_ante_cost_bps"].to_numpy(dtype=np.float64)
        else:
            base_cost_arr = np.full(n_events, cost_model.taker_round_trip_bps(), dtype=np.float64)

        finite_mask = np.isfinite(edge_arr) & np.isfinite(base_cost_arr) & (base_cost_arr >= 0.0)
        finite_edge = edge_arr[finite_mask]
        net_stress_arr = _stress_edge_bps(
            edge_bps=edge_arr[finite_mask],
            base_cost_bps=base_cost_arr[finite_mask],
            stress_multiplier=cfg.cost_stress_multiplier,
        )

        mean_net = float(np.mean(finite_edge)) if finite_edge.size > 0 else float("nan")
        mean_net_stress = float(np.mean(net_stress_arr)) if net_stress_arr.size > 0 else float("nan")
        net_p50 = float(np.median(finite_edge)) if finite_edge.size > 0 else float("nan")
        net_stress_p50 = float(np.median(net_stress_arr)) if net_stress_arr.size > 0 else float("nan")

        hit_rate = float((finite_edge > 0).mean()) if finite_edge.size > 0 else 0.0
        stress_events = oos_events.loc[finite_mask].copy()
        stress_events["_stress_edge_bps"] = net_stress_arr
        decision_bar_edge = _decision_bar_edge_series(stress_events, "_stress_edge_bps")
        hac_t = _newey_west_t_stat(decision_bar_edge, max_lag=max(0, min(cfg.purge_bars, decision_bar_edge.size - 1)))
        decision_bar_count = int(decision_bar_edge.size)

        if cfg.blend_survival_use_mean:
            survives = (
                np.isfinite(mean_net_stress)
                and mean_net_stress > cfg.blend_survival_min_net_stress_bps
                and hac_t >= cfg.min_rule_ir_t
            )
        else:
            # legacy median path (회귀 대비 보존)
            survives = np.isfinite(net_stress_p50) and net_stress_p50 > 0.0 and hac_t >= cfg.min_rule_ir_t

        reports.append(SignalValidationReport(
            variant=variant_name,
            n_events=n_events,
            net_edge_bps_p50=net_p50,
            net_edge_bps_stress_p50=net_stress_p50,
            net_edge_bps_mean=mean_net,
            net_edge_bps_stress_mean=mean_net_stress,
            hit_rate=hit_rate,
            hac_t_stat=hac_t,
            survives_cost=survives,
            deployment_count=int(n_events),
            decision_bar_count=decision_bar_count,
        ))

    return reports
