from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.domain.futures.strategy.config import StrategyConfig

_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MLPipelineOutput:
    """Neutral bridge output for runtime compatibility."""

    alpha_panel: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(slots=True)
class CandidatePipelineOutput:
    """Candidate strategy bridge output."""

    alpha_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_weights: pd.DataFrame | None = None
    rule_report: dict[str, Any] | None = None


def run_candidate_strategy_for_universe(
    symbols: list[str],
    tf: str,
    *,
    strategy_cfg: StrategyConfig | None = None,
    preloaded_data_maps: dict[str, dict[str, Any]] | None = None,
) -> CandidatePipelineOutput:
    """Run candidate strategy pipeline and return bridge-neutral output."""
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

    # 1. Alignment
    aligned = align_data_maps(preloaded_data_maps, symbols, tf)
    n_bars = aligned.close_2d.shape[0]

    # 2. Rule panel generation
    panels = build_rule_signal_panels(aligned=aligned, cfg=strategy_cfg.candidate)
    raw_events = candidate_panels_to_events(panels, min_abs_score=strategy_cfg.candidate.min_rule_net_bps * 1e-4)

    if raw_events.empty:
        # Fallback to empty panel
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

    # 3. Label events
    labeled = label_candidate_events(events=raw_events, aligned=aligned, cfg=strategy_cfg.candidate)

    # 4. Dataset building
    # Split train/validation (80/20)
    split_val = int(n_bars * 0.8)
    train_set = build_candidate_dataset(
        labeled_events=labeled, aligned=aligned, cfg=strategy_cfg.candidate, split_start=0, split_end=split_val
    )
    valid_set = build_candidate_dataset(
        labeled_events=labeled, aligned=aligned, cfg=strategy_cfg.candidate, split_start=split_val, split_end=n_bars
    )
    full_set = build_candidate_dataset(
        labeled_events=labeled, aligned=aligned, cfg=strategy_cfg.candidate, split_start=0, split_end=n_bars
    )

    # 5. ML Models Training
    gate_model = fit_candidate_gate(train=train_set, valid=valid_set, cfg=strategy_cfg.candidate)
    edge_models = fit_candidate_edge_models(train=train_set, valid=valid_set, cfg=strategy_cfg.candidate)

    # 6. Prediction
    p_pass = predict_candidate_gate(model=gate_model, dataset=full_set)
    ml_out = predict_candidate_edges(models=edge_models, dataset=full_set, p_pass=p_pass, cfg=strategy_cfg.candidate)
    ml_out = replace(ml_out, events=full_set.event_index)

    _logger.info(
        f"[DEBUG] raw_events={len(raw_events)} train_set={train_set.X.shape} valid_set={valid_set.X.shape} "
        f"p_pass_mean={np.nanmean(p_pass):.4f} mu_mean={np.nanmean(ml_out.mu_net_decision_bps):.4f} "
        f"q10_mean={np.nanmean(ml_out.q10_net_bps):.4f}"
    )

    # 7. Portfolio Sizing and Alpha Panel formatting
    selected = select_candidate_events_for_portfolio(model_output=ml_out, cfg=strategy_cfg.candidate)
    _logger.info(f"[DEBUG] selected_after_portfolio={len(selected)}")

    target_weights = build_candidate_target_weights(
        selected_events=selected,
        close_2d=aligned.close_2d,
        symbols=tuple(symbols),
        beta_2d=None,  # Fallback to trailing btc beta inside builder if None
        sigma_3d=None,
        cfg=strategy_cfg.candidate,
    )
    
    # Log nonzero target weights count
    _logger.info(f"[DEBUG] nonzero_weights={np.count_nonzero(target_weights)}")

    alpha_panel = build_candidate_alpha_panel(
        selected_events=selected,
        target_weights_2d=target_weights,
        datetimes=aligned.datetimes,
        symbols=tuple(symbols),
    )

    rule_report = {
        "events_total": len(raw_events),
        "selected_total": len(selected),
    }

    return CandidatePipelineOutput(
        alpha_panel=alpha_panel,
        target_weights=target_weights,
        rule_report=rule_report,
    )


def run_ml_pipeline_for_universe(
    symbols: list[str],
    tf: str,
    fetch_start: str | None,
    end_date: str | None,
    opt_config: dict[str, Any],
    *,
    strategy_cfg: StrategyConfig | None = None,
    preloaded_data_maps: dict[str, dict[str, Any]] | None = None,
    **kwargs: Any,
) -> MLPipelineOutput:
    t0 = time.perf_counter()
    del fetch_start, end_date, opt_config, kwargs
    if strategy_cfg is None or preloaded_data_maps is None:
        return MLPipelineOutput()

    if strategy_cfg.name in {"candidate_ml", "rule_baseline"}:
        candidate_out = run_candidate_strategy_for_universe(
            symbols=symbols,
            tf=tf,
            strategy_cfg=strategy_cfg,
            preloaded_data_maps=preloaded_data_maps,
        )
        return MLPipelineOutput(alpha_panel=candidate_out.alpha_panel)

    from src.domain.futures.strategy.builder import build_strategy_alpha

    alpha_panel = build_strategy_alpha(
        data_maps=preloaded_data_maps,
        symbols=symbols,
        tf=tf,
        cfg=strategy_cfg,
    )

    out = MLPipelineOutput(alpha_panel=alpha_panel)
    alpha_rows = len(alpha_panel)
    _logger.info(
        "[ML-PIPE-PROF] symbols=%d tf=%s elapsed=%.2fs alpha_rows=%d",
        len(symbols),
        tf,
        time.perf_counter() - t0,
        alpha_rows,
    )
    return out


def merge_candidate_output_into_data_maps(
    candidate_out: CandidatePipelineOutput,
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    log_tag: str = "",
) -> None:
    """Merge target_weights and candidate diagnostics into data maps."""
    panel = getattr(candidate_out, "alpha_panel", None)
    if panel is None or panel.empty:
        return
    required = {
        "alpha_long", "alpha_short", "target_weight", "candidate_family",
        "candidate_variant", "p_pass", "mu_net_decision_bps", "q10_net_bps", "utility_score"
    }
    by_sym = panel.reset_index().groupby("symbol", sort=False)
    for sym in symbols:
        if sym not in data_maps or tf not in data_maps[sym]:
            continue
        try:
            sym_rows = by_sym.get_group(sym)
        except KeyError:
            continue
        df = data_maps[sym][tf]
        left = df[["datetime"]].copy()
        merge_cols = ["datetime", *list(required)]
        right = sym_rows[merge_cols].copy()
        left["_merge_datetime"] = pd.to_datetime(left["datetime"], utc=True).dt.tz_localize(None)
        right["_merge_datetime"] = pd.to_datetime(right["datetime"], utc=True).dt.tz_localize(None)
        merge_value_cols = ["_merge_datetime", *list(required)]
        merged = left.merge(
            right[merge_value_cols],
            on="_merge_datetime",
            how="left",
        )
        df["alpha_long"] = merged["alpha_long"].fillna(0.0).to_numpy(dtype=np.float64)
        df["alpha_short"] = merged["alpha_short"].fillna(0.0).to_numpy(dtype=np.float64)
        df["target_weight"] = merged["target_weight"].fillna(0.0).to_numpy(dtype=np.float64)
        df["candidate_family"] = merged["candidate_family"].fillna("").to_numpy(dtype=object)
        df["candidate_variant"] = merged["candidate_variant"].fillna("").to_numpy(dtype=object)
        df["p_pass"] = merged["p_pass"].fillna(0.0).to_numpy(dtype=np.float64)
        df["mu_net_decision_bps"] = merged["mu_net_decision_bps"].fillna(0.0).to_numpy(dtype=np.float64)
        df["q10_net_bps"] = merged["q10_net_bps"].fillna(0.0).to_numpy(dtype=np.float64)
        df["utility_score"] = merged["utility_score"].fillna(0.0).to_numpy(dtype=np.float64)


def merge_ml_output_into_is_and_oos(
    ml_out: MLPipelineOutput | CandidatePipelineOutput,
    is_maps: dict[str, dict[str, Any]],
    oos_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
) -> None:
    if isinstance(ml_out, CandidatePipelineOutput):
        merge_candidate_output_into_data_maps(ml_out, is_maps, valid_symbols, tf, log_tag="is")
        merge_candidate_output_into_data_maps(ml_out, oos_maps, valid_symbols, tf, log_tag="oos")
    else:
        merge_ml_output_into_data_maps(ml_out, is_maps, valid_symbols, tf, log_tag="is")
        merge_ml_output_into_data_maps(ml_out, oos_maps, valid_symbols, tf, log_tag="oos")


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


def merge_ml_output_into_data_maps(
    ml_out: MLPipelineOutput,
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    log_tag: str = "",
) -> None:
    panel = getattr(ml_out, "alpha_panel", None)
    if panel is None or panel.empty:
        return
    required = {"alpha_long", "alpha_short"}
    if not required.issubset(panel.columns):
        _logger.warning("[%s] alpha_panel missing required columns; skip merge", log_tag)
        return

    by_sym = panel.reset_index().groupby("symbol", sort=False)
    for sym in symbols:
        if sym not in data_maps or tf not in data_maps[sym]:
            continue
        try:
            sym_rows = by_sym.get_group(sym)
        except KeyError:
            continue
        df = data_maps[sym][tf]
        if "alpha_long" in df.columns or "alpha_short" in df.columns:
            _logger.warning("[%s] overwrite alpha columns for symbol=%s", log_tag, sym)
        left = df[["datetime"]].copy()
        merge_cols = ["datetime", "alpha_long", "alpha_short"]
        right = sym_rows[merge_cols].copy()
        left["_merge_datetime"] = pd.to_datetime(left["datetime"], utc=True).dt.tz_localize(None)
        right["_merge_datetime"] = pd.to_datetime(right["datetime"], utc=True).dt.tz_localize(None)
        merge_value_cols = ["_merge_datetime", "alpha_long", "alpha_short"]
        merged = left.merge(
            right[merge_value_cols],
            on="_merge_datetime",
            how="left",
        )

        df["alpha_long"] = merged["alpha_long"].fillna(0.0).to_numpy(dtype=np.float64)
        df["alpha_short"] = merged["alpha_short"].fillna(0.0).to_numpy(dtype=np.float64)

    # [ALPHA-MERGE] summary log
    merged_symbols = sum(
        1
        for sym in symbols
        if sym in data_maps
        and tf in data_maps[sym]
        and "alpha_long" in data_maps[sym][tf].columns
    )
    _long_nz_ratios: list[float] = []
    _short_nz_ratios: list[float] = []
    for sym in symbols:
        if sym not in data_maps or tf not in data_maps[sym]:
            continue
        _df = data_maps[sym][tf]
        if "alpha_long" in _df.columns:
            _long_nz_ratios.append(float((_df["alpha_long"] != 0).mean()))
        if "alpha_short" in _df.columns:
            _short_nz_ratios.append(float((_df["alpha_short"] != 0).mean()))
    
    _alpha_long_nz = float(np.mean(_long_nz_ratios)) if _long_nz_ratios else 0.0
    _alpha_short_nz = float(np.mean(_short_nz_ratios)) if _short_nz_ratios else 0.0
    _panel_start = "na"
    _panel_end = "na"
    try:
        _panel_dt = pd.to_datetime(panel.index.get_level_values("datetime"), utc=True).tz_localize(
            None
        )
        _panel_start = str(_panel_dt.min())
        _panel_end = str(_panel_dt.max())
    except Exception as exc:
        _logger.debug("[ALPHA-MERGE] panel datetime extraction failed: %s", exc)
    
    _logger.info(
        ".. ALPHA_MERGE: syms=%d span=%s ~ %s L_nz=%.3f S=%.3f",
        merged_symbols,
        str(_panel_start)[:10],
        str(_panel_end)[:10],
        _alpha_long_nz,
        _alpha_short_nz,
    )


class FuturesMLStrategy:
    """Neutral strategy stub for bridge contract."""

    def __init__(self, name: str, params: dict[str, Any]) -> None:
        """Store strategy identity and params."""
        self.name = name
        self.params = params

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return pass-through copy of input frame."""
        return df.copy()
