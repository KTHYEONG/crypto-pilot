from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from src.domain.futures.backtest.engine import FuturesBacktestEngine
from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput
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
    weights = np.asarray(calibration_set.sample_weight, dtype=np.float64)
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
) -> pd.DataFrame:
    """Filter labeled events to only recommended variants.

    Returns empty DataFrame (fail-closed) when no variants are recommended,
    so callers must guard for empty result.
    """
    if labeled.empty:
        return labeled

    allowed = set(keep_variants) | set(flip_variants)
    if not allowed:
        _logger.warning("[PROMO_FILTER] no variants recommended by diagnostics; blocking all candidates (fail-closed)")
        return labeled.iloc[0:0].copy()

    out = labeled.loc[_variant_key(labeled).isin(allowed)].copy()
    if out.empty:
        _logger.warning("[PROMO_FILTER] all candidates removed after variant filter; returning empty")
        return labeled.iloc[0:0].copy()

    flip_mask = _variant_key(out).isin(set(flip_variants))
    if bool(flip_mask.any()):
        out.loc[flip_mask, "side"] = -pd.to_numeric(out.loc[flip_mask, "side"], errors="coerce")
        if "raw_score" in out.columns:
            out.loc[flip_mask, "raw_score"] = -pd.to_numeric(out.loc[flip_mask, "raw_score"], errors="coerce")
        if "score_z" in out.columns:
            out.loc[flip_mask, "score_z"] = -pd.to_numeric(out.loc[flip_mask, "score_z"], errors="coerce")
        out.loc[flip_mask, "side_flipped"] = True
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
    selection_thresholds = dict(base_out.selection_thresholds)
    selection_thresholds["utility_min"] = _utility_min_threshold(
        utility_score=utility_score,
        cfg=cfg,
    )
    return CandidateModelOutput(
        events=oos_set.event_index,
        p_pass=p_pass.astype(np.float64, copy=False),
        mu_gross_bps=prior_mu,
        mu_net_decision_bps=prior_mu,
        q10_net_bps=base_out.q10_net_bps,
        q90_net_bps=base_out.q90_net_bps,
        utility_score=utility_score,
        selection_thresholds=selection_thresholds,
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
            "turnover", "final_equity", "pass_compound_gate"
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
    diag, diag_oracle = _compute_rule_diagnostics_for_ablation(
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
    gate_model = fit_candidate_gate(train=fit_set, valid=calibration_set, cfg=cfg)
    edge_models = fit_candidate_edge_models(train=fit_set, valid=calibration_set, cfg=cfg)

    # 5. Predict outcomes for OOS sample only
    p_pass = predict_candidate_gate(model=gate_model, dataset=oos_set)
    ml_out = predict_candidate_edges(models=edge_models, dataset=oos_set, p_pass=p_pass, cfg=cfg)
    ml_out = replace(ml_out, events=oos_set.event_index)

    rows: list[AblationRow] = []

    # Variant 1: rule_only_equal_size (Simple benchmark)
    # equal weight assigned to any rule trigger
    raw_w = _build_rule_equal_size_weights(
        raw_events=raw_events,
        close_2d=aligned.close_2d,
        symbols=symbols,
        max_symbol_weight=cfg.max_symbol_weight,
    )

    rows.append(_run_backtest_and_evaluate(raw_w, aligned, "rule_only_equal_size", cfg))

    # Variant 1b: no-leak rule promotion only
    promoted_rule_events = apply_variant_promotions(
        labeled=labeled_unfiltered,
        keep_variants=diag.recommended_keep_variants,
        flip_variants=diag.recommended_flip_variants,
    )
    promoted_rule_events = _oos_only_events(
        labeled=promoted_rule_events,
        oos_start=oos_start,
        oos_end=oos_end,
    )
    promoted_rule_w = _build_rule_equal_size_weights(
        raw_events=promoted_rule_events,
        close_2d=aligned.close_2d,
        symbols=symbols,
        max_symbol_weight=cfg.max_symbol_weight,
    )
    rows.append(
        _run_backtest_and_evaluate(
            promoted_rule_w,
            aligned,
            "rule_promo_no_leak",
            cfg,
            start_idx=oos_start,
            end_idx=oos_end,
        )
    )

    # Variant 1c: OOS oracle rule promotion for comparison only
    oracle_rule_events = apply_variant_promotions(
        labeled=labeled_unfiltered,
        keep_variants=diag_oracle.recommended_keep_variants,
        flip_variants=diag_oracle.recommended_flip_variants,
    )
    oracle_rule_events = _oos_only_events(
        labeled=oracle_rule_events,
        oos_start=oos_start,
        oos_end=oos_end,
    )
    oracle_rule_w = _build_rule_equal_size_weights(
        raw_events=oracle_rule_events,
        close_2d=aligned.close_2d,
        symbols=symbols,
        max_symbol_weight=cfg.max_symbol_weight,
    )
    rows.append(
        _run_backtest_and_evaluate(
            oracle_rule_w,
            aligned,
            "rule_promo_oos_oracle",
            cfg,
            start_idx=oos_start,
            end_idx=oos_end,
        )
    )

    # Variant 2: rule_only_fractional_kelly (Kelly Sizing but no ML)
    # create artificial mock edge output using constant score
    mock_events = raw_events.copy()
    mock_events["p_pass"] = 1.0
    mock_events["mu_net_decision_bps"] = 50.0  # Constant expectation
    mock_events["q10_net_bps"] = -10.0
    mock_events["utility_score"] = 1.0
    raw_kelly_w = build_candidate_target_weights(
        selected_events=mock_events,
        close_2d=aligned.close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=cfg,
    )
    rows.append(_run_backtest_and_evaluate(raw_kelly_w, aligned, "rule_only_fractional_kelly", cfg))

    # Variant 3: rule_plus_ml_gate (Gate filtering only)
    # ML gate filters events, but sizes them using constant ex-ante edge dummy values
    gate_events_only = select_candidate_events_for_portfolio(model_output=ml_out, cfg=cfg)
    # Override mu to ex-ante constant proxy for Variant 3
    gate_events_only_mock = gate_events_only.copy()
    gate_events_only_mock["mu_net_decision_bps"] = 50.0
    gate_only_w = build_candidate_target_weights(
        selected_events=gate_events_only_mock,
        close_2d=aligned.close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=cfg,
    )
    rows.append(
        _run_backtest_and_evaluate(gate_only_w, aligned, "rule_plus_ml_gate", cfg)
    )

    # Variant 4: rule_plus_ml_gate_plus_edge (Gate + Edge, but uncapped/uncapped Kelly)
    # Sized dynamically using predicted expected edge mu, but bypasses the cap projection loop (raw fractional Kelly)
    # Calculate raw Kelly weights manually to bypass project_all_caps
    raw_kelly_edge_w = _build_uncapped_kelly_edge_weights(
        selected_events=gate_events_only,
        close_2d=aligned.close_2d,
        symbols=symbols,
        kelly_fraction=cfg.kelly_fraction,
    )

    rows.append(
        _run_backtest_and_evaluate(raw_kelly_edge_w, aligned, "rule_plus_ml_gate_plus_edge", cfg)
    )

    # Variant 5: rule_plus_ml_gate_plus_edge_plus_portfolio_caps (Full sizing caps applied)
    # Full constraint projection on Kelly weights
    gate_plus_edge_plus_caps_w = build_candidate_target_weights(
        selected_events=gate_events_only,
        close_2d=aligned.close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=cfg,
    )
    rows.append(
        _run_backtest_and_evaluate(
            gate_plus_edge_plus_caps_w,
            aligned,
            "rule_plus_ml_gate_plus_edge_plus_portfolio_caps",
            cfg,
        )
    )

    # Variant 6: candidate_ml_full (OOS-only signal — production-equivalent split)
    gate_events_oos = gate_events_only
    full_ml_w = build_candidate_target_weights(
        selected_events=gate_events_oos,
        close_2d=aligned.close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=cfg,
    )
    rows.append(
        _run_backtest_and_evaluate(
            full_ml_w,
            aligned,
            "candidate_ml_full",
            cfg,
            start_idx=oos_start,
            end_idx=oos_end,
        )
    )

    # Variant 6b: direct edge model without prior-residual decomposition
    rows.append(
        _run_oos_only_ablation_variant(
            labeled=_labeled_for_ml,
            aligned=aligned,
            cfg=replace(cfg, edge_prior_enabled=False, edge_residual_model_enabled=False),
            variant_name="candidate_ml_direct_edge",
        )
    )

    # Variant 6c: prior-only edge using shrunk variant means
    prior_only_out = _build_variant_prior_output(
        edge_models=edge_models,
        calibration_set=calibration_set,
        oos_set=oos_set,
        p_pass=p_pass,
        cfg=cfg,
    )
    prior_selected = select_candidate_events_for_portfolio(model_output=prior_only_out, cfg=cfg)
    prior_only_w = build_candidate_target_weights(
        selected_events=prior_selected,
        close_2d=aligned.close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=cfg,
    )
    rows.append(
        _run_backtest_and_evaluate(
            prior_only_w,
            aligned,
            "candidate_ml_variant_prior",
            cfg,
            start_idx=oos_start,
            end_idx=oos_end,
        )
    )

    # ── New OOS-only ablation rows (7-10): each isolates one added layer ──────
    future_to_idx = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Row 7: without promotion filter
        future_to_idx[executor.submit(
            _run_oos_only_ablation_variant,
            labeled=labeled_unfiltered,
            aligned=aligned,
            cfg=cfg,
            variant_name="candidate_ml_promotion_filter",
        )] = 0

        # Row 8: with hard selection
        future_to_idx[executor.submit(
            _run_oos_only_ablation_variant,
            labeled=labeled,
            aligned=aligned,
            cfg=replace(cfg, selection_policy="hard"),
            variant_name="candidate_ml_validation_quantile_selection",
        )] = 1

        # Row 9: without identity features
        future_to_idx[executor.submit(
            _run_oos_only_ablation_variant,
            labeled=labeled,
            aligned=aligned,
            cfg=replace(cfg, candidate_identity_features_enabled=False),
            variant_name="candidate_ml_identity_features",
        )] = 2

        # Row 10: without market-state features
        future_to_idx[executor.submit(
            _run_oos_only_ablation_variant,
            labeled=labeled,
            aligned=aligned,
            cfg=replace(cfg, market_state_features_enabled=False),
            variant_name="candidate_ml_market_state_features",
        )] = 3

        ablation_results = [
            (future_to_idx[future], future.result()) for future in as_completed(future_to_idx)
        ]

        ablation_results.sort(key=lambda x: x[0])
        for _, result in ablation_results:
            rows.append(result)

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
        }
        for r in rows
    ])

    return df_results


def _run_oos_only_ablation_variant(
    *,
    labeled: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    variant_name: str,
) -> AblationRow:
    """Retrain models with modified cfg/labeled and evaluate on OOS split only."""
    n_bars = aligned.close_2d.shape[0]
    zero_w = np.zeros_like(aligned.close_2d)

    if labeled.empty:
        _logger.warning("[ABLATION][%s] empty labeled events; returning zero weights", variant_name)
        return _run_backtest_and_evaluate(zero_w, aligned, variant_name, cfg)

    fit_start, fit_end, calibration_start, calibration_end, oos_start, oos_end = _candidate_ml_split_indices(
        n_bars=n_bars,
        fit_fraction=cfg.ml_fit_fraction,
        calibration_fraction=cfg.ml_calibration_fraction,
        purge_bars=cfg.purge_bars,
        embargo_bars=cfg.embargo_bars,
    )
    fit_set = build_candidate_dataset(
        labeled_events=labeled, aligned=aligned, cfg=cfg, split_start=fit_start, split_end=fit_end
    )
    calibration_set = build_candidate_dataset(
        labeled_events=labeled, aligned=aligned, cfg=cfg, split_start=calibration_start, split_end=calibration_end
    )
    oos_set = build_candidate_dataset(
        labeled_events=labeled, aligned=aligned, cfg=cfg, split_start=oos_start, split_end=oos_end
    )

    gate_model = fit_candidate_gate(train=fit_set, valid=calibration_set, cfg=cfg)
    edge_models = fit_candidate_edge_models(train=fit_set, valid=calibration_set, cfg=cfg)

    p_pass = predict_candidate_gate(model=gate_model, dataset=oos_set)
    ml_out = predict_candidate_edges(models=edge_models, dataset=oos_set, p_pass=p_pass, cfg=cfg)
    ml_out = replace(ml_out, events=oos_set.event_index)

    selected = select_candidate_events_for_portfolio(model_output=ml_out, cfg=cfg)
    w = build_candidate_target_weights(
        selected_events=selected,
        close_2d=aligned.close_2d,
        symbols=aligned.symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=cfg,
    )
    return _run_backtest_and_evaluate(
        w,
        aligned,
        variant_name,
        cfg,
        start_idx=oos_start,
        end_idx=oos_end,
    )



def _run_backtest_and_evaluate(
    target_weights: np.ndarray,
    aligned: AlignedMarketData,
    variant_name: str,
    cfg: CandidateStrategyConfig,
    *,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> AblationRow:
    """Helper to inject target_weights into data_maps and run backtest simulation."""
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

    aligned_data = {
        "close": aligned_eval.close_2d,
        "high": aligned_eval.high_2d,
        "low": aligned_eval.low_2d,
        "open": aligned_eval.open_2d,
        "volume": aligned_eval.volume_2d,
        "atr": atr_2d,
        "target_weights": target_weights_eval,
    }

    # Execute backtest engine
    trades, equity_curve, _, _ = FuturesBacktestEngine.run_multi(
        aligned_data=aligned_data,
        symbol_names=list(aligned_eval.symbols),
        strategy_params={},
    )

    # Evaluate compounding growth
    report = evaluate_compound_backtest(
        trades=trades,
        equity_curve=equity_curve,
        cfg=cfg,
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
    )
