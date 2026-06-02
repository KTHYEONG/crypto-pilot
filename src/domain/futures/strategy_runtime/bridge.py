from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.domain.futures.strategy.config import StrategyConfig

_logger = logging.getLogger(__name__)


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
) -> CandidatePipelineOutput:
    """Run candidate strategy pipeline and return candidate output."""
    if strategy_cfg is None or preloaded_data_maps is None:
        return CandidatePipelineOutput()

    from dataclasses import replace

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
    from src.domain.futures.strategy.rule_signals import build_rule_signal_panels, candidate_panels_to_events

    aligned = align_data_maps(preloaded_data_maps, symbols, tf)
    n_bars = aligned.close_2d.shape[0]

    panels = build_rule_signal_panels(aligned=aligned, cfg=strategy_cfg.candidate)
    raw_events = candidate_panels_to_events(panels, min_abs_score=strategy_cfg.candidate.min_rule_net_bps * 1e-4)

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
            rule_report={"events_total": 0},
        )

    labeled = label_candidate_events(events=raw_events, aligned=aligned, cfg=strategy_cfg.candidate)

    split_val = int(n_bars * 0.8)
    train_set = build_candidate_dataset(
        labeled_events=labeled, aligned=aligned, cfg=strategy_cfg.candidate, split_start=0, split_end=split_val
    )
    valid_set = build_candidate_dataset(
        labeled_events=labeled, aligned=aligned, cfg=strategy_cfg.candidate, split_start=split_val, split_end=n_bars
    )

    gate_model = fit_candidate_gate(train=train_set, valid=valid_set, cfg=strategy_cfg.candidate)
    edge_models = fit_candidate_edge_models(train=train_set, valid=valid_set, cfg=strategy_cfg.candidate)

    # OOS-only predict: IS 구간(0~80%)은 target_weight=0으로 유지하여 look-ahead 누수 방지
    p_pass = predict_candidate_gate(model=gate_model, dataset=valid_set)
    ml_out = predict_candidate_edges(models=edge_models, dataset=valid_set, p_pass=p_pass, cfg=strategy_cfg.candidate)
    ml_out = replace(ml_out, events=valid_set.event_index)

    selected = select_candidate_events_for_portfolio(model_output=ml_out, cfg=strategy_cfg.candidate)
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

    return CandidatePipelineOutput(
        alpha_panel=alpha_panel,
        target_weights=target_weights,
        rule_report={"events_total": len(raw_events), "selected_total": len(selected)},
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
        "candidate_variant", "p_pass", "mu_net_decision_bps", "q10_net_bps", "utility_score"
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
