from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps, project_all_caps
from src.domain.futures.strategy.candidate_contracts import (
    CandidateModelOutput,
)
from src.domain.futures.strategy.config import CandidateStrategyConfig


@dataclass(frozen=True)
class SelectionWaterfall:
    total: int
    gate_floor: float
    breakeven_floor_bps: float
    utility_floor_bps: float
    gate_eligible: int
    catastrophic_eligible: int
    utility_eligible: int
    all_eligible: int
    mu_ge_floor: int
    expected_utility_ge_zero: int
    expected_utility_ge_floor: int
    prob_adjusted_mu_p50_bps: float | None
    prob_adjusted_mu_p90_bps: float | None
    downside_drag_p50_bps: float | None
    downside_drag_p90_bps: float | None
    turnover_drag_p50_bps: float | None
    soft_gate_penalty_p50_bps: float | None
    expected_utility_raw_p50_bps: float | None
    expected_utility_raw_p90_bps: float | None
    expected_utility_adj_p50_bps: float | None
    expected_utility_adj_p90_bps: float | None


def _candidate_variant_key(frame: pd.DataFrame) -> str:
    return f"{frame['family'].iloc[0]!s}:{frame['variant'].iloc[0]!s}"


def _series_or_default(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _shortfall_limit_bps(
    df: pd.DataFrame,
    cfg: CandidateStrategyConfig,
    *,
    catastrophic: bool,
) -> pd.Series:
    if cfg.shortfall_threshold_basis == "absolute_bps" or "sl_thr_bps" not in df.columns:
        value = cfg.catastrophic_shortfall_bps if catastrophic else cfg.max_expected_shortfall_bps
        return pd.Series(float(value), index=df.index, dtype="float64")
    sl = _series_or_default(df, "sl_thr_bps").clip(lower=0.0)
    cost = _series_or_default(df, "ex_ante_cost_bps").clip(lower=0.0)
    mult = cfg.catastrophic_shortfall_stop_mult if catastrophic else cfg.max_expected_shortfall_stop_mult
    absolute_floor = cfg.catastrophic_shortfall_bps if catastrophic else cfg.max_expected_shortfall_bps
    dynamic_limit = sl.mul(float(mult)).add(cost)
    return pd.Series(
        np.maximum(float(absolute_floor), dynamic_limit.to_numpy(dtype=np.float64, copy=False)),
        index=df.index,
        dtype="float64",
    )


def _q10_mask_for_mode(df: pd.DataFrame, cfg: CandidateStrategyConfig) -> pd.Series:
    if cfg.selection_shortfall_mode == "penalty_only":
        return pd.Series(True, index=df.index, dtype=bool)
    if cfg.selection_shortfall_mode == "catastrophic":
        return df["q10_net_bps"] >= -_shortfall_limit_bps(df, cfg, catastrophic=True)
    return df["q10_net_bps"] >= -_shortfall_limit_bps(df, cfg, catastrophic=False)


def _catastrophic_q10_mask(df: pd.DataFrame, cfg: CandidateStrategyConfig) -> pd.Series:
    return df["q10_net_bps"] >= -_shortfall_limit_bps(df, cfg, catastrophic=True)


def _utility_threshold(
    *,
    df: pd.DataFrame,
    cfg: CandidateStrategyConfig,
    model_output: CandidateModelOutput,
) -> float:
    threshold = model_output.selection_thresholds.get("utility_min")
    if threshold is not None and np.isfinite(threshold):
        return float(threshold)
    return float(getattr(cfg, "selection_min_expected_utility_bps", 0.0))


def _rank_ic(pred: pd.Series, target: pd.Series) -> float | None:
    valid = pd.DataFrame({"pred": pred, "target": target}).replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return None
    corr = valid["pred"].corr(valid["target"], method="spearman")
    if corr is None or not np.isfinite(corr):
        return None
    return float(corr)


def _finite_quantile_or_none(values: pd.Series | np.ndarray, q: float) -> float | None:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.quantile(arr, q))


def _resolve_breakeven_floor(frame: pd.DataFrame, cfg: CandidateStrategyConfig) -> float:
    """Compute the breakeven floor in bps.

    static: cfg.min_net_floor_cost_fraction * cfg.cost_floor_bps (original behaviour).
    fold_adaptive: derive floor from the fold-local cost_floor_bps distribution so that
    weak folds get a tighter floor and strong folds stay open.
    """
    if cfg.breakeven_floor_mode == "fold_adaptive" and "cost_floor_bps" in frame.columns:
        cost_series = pd.to_numeric(frame["cost_floor_bps"], errors="coerce").dropna()
        if cost_series.size > 0:
            fold_cost = float(np.quantile(cost_series.to_numpy(dtype=np.float64), cfg.breakeven_floor_cost_quantile))
            return float(cfg.min_net_floor_cost_fraction) * fold_cost
    return float(cfg.min_net_floor_cost_fraction) * float(cfg.cost_floor_bps)


def _selection_component_frame(
    *,
    events: pd.DataFrame,
    cfg: CandidateStrategyConfig,
    gate_mode: str,
    gate_floor: float,
    utility_mode: str | None = None,
) -> pd.DataFrame:
    frame = events.copy()
    resolved_utility_mode = utility_mode if utility_mode is not None else cfg.selection_utility_mode
    turnover_proxy = _series_or_default(frame, "turnover_proxy", 1.0)
    prob_adjusted_mu_bps = frame["p_pass"] * frame["mu_net_decision_bps"]
    downside_drag_bps = (1.0 - frame["p_pass"]) * frame["q10_net_bps"].clip(upper=0.0).abs()
    turnover_drag_bps = cfg.turnover_penalty * turnover_proxy
    gate_shortfall = np.maximum(0.0, gate_floor - frame["p_pass"].to_numpy(dtype=np.float64, copy=False))
    soft_gate_penalty_bps = gate_shortfall * np.maximum(
        pd.to_numeric(frame["mu_net_decision_bps"], errors="coerce").to_numpy(dtype=np.float64, copy=False),
        0.0,
    )
    expected_utility_raw_bps = prob_adjusted_mu_bps - downside_drag_bps - turnover_drag_bps
    expected_utility_bps = expected_utility_raw_bps.copy()
    if gate_mode == "soft_floor":
        expected_utility_bps = expected_utility_bps - soft_gate_penalty_bps
    frame["prob_adjusted_mu_bps"] = pd.to_numeric(prob_adjusted_mu_bps, errors="coerce")
    frame["downside_drag_bps"] = pd.to_numeric(downside_drag_bps, errors="coerce")
    frame["turnover_drag_bps"] = pd.to_numeric(turnover_drag_bps, errors="coerce")
    frame["soft_gate_penalty_bps"] = pd.to_numeric(soft_gate_penalty_bps, errors="coerce")
    frame["expected_utility_raw_bps"] = pd.to_numeric(expected_utility_raw_bps, errors="coerce")
    if resolved_utility_mode == "expected_edge_direct":
        # mu_net is the unconditional E[return] — already incorporates win/loss mix.
        # p_pass and q10 are demoted to sizing-only roles to avoid double-counting.
        frame["expected_utility_bps"] = pd.to_numeric(frame["mu_net_decision_bps"], errors="coerce")
    else:
        frame["expected_utility_bps"] = pd.to_numeric(expected_utility_bps, errors="coerce")
    return frame


def compute_selection_waterfall(
    *,
    events: pd.DataFrame,
    cfg: CandidateStrategyConfig,
) -> dict[str, int | float | None]:
    """Return component-level diagnostics for candidate selection eligibility."""
    gate_floor = float(cfg.selection_min_gate_probability_floor)
    breakeven_floor = _resolve_breakeven_floor(events, cfg)
    utility_floor = max(float(cfg.selection_min_expected_utility_bps), breakeven_floor)
    if events.empty:
        return asdict(
            SelectionWaterfall(
                total=0,
                gate_floor=gate_floor,
                breakeven_floor_bps=breakeven_floor,
                utility_floor_bps=utility_floor,
                gate_eligible=0,
                catastrophic_eligible=0,
                utility_eligible=0,
                all_eligible=0,
                mu_ge_floor=0,
                expected_utility_ge_zero=0,
                expected_utility_ge_floor=0,
                prob_adjusted_mu_p50_bps=None,
                prob_adjusted_mu_p90_bps=None,
                downside_drag_p50_bps=None,
                downside_drag_p90_bps=None,
                turnover_drag_p50_bps=None,
                soft_gate_penalty_p50_bps=None,
                expected_utility_raw_p50_bps=None,
                expected_utility_raw_p90_bps=None,
                expected_utility_adj_p50_bps=None,
                expected_utility_adj_p90_bps=None,
            )
        )

    frame = _selection_component_frame(
        events=events,
        cfg=cfg,
        gate_mode=cfg.selection_gate_mode,
        gate_floor=gate_floor,
    )
    catastrophic_mask = _catastrophic_q10_mask(frame, cfg)
    gate_eligible_mask = pd.Series(True, index=frame.index, dtype=bool)
    if cfg.selection_gate_mode == "hard_floor":
        gate_eligible_mask = frame["p_pass"] >= gate_floor
    utility_eligible_mask = frame["expected_utility_bps"] >= utility_floor
    waterfall = SelectionWaterfall(
        total=int(frame.shape[0]),
        gate_floor=gate_floor,
        breakeven_floor_bps=breakeven_floor,
        utility_floor_bps=utility_floor,
        gate_eligible=int(gate_eligible_mask.sum()),
        catastrophic_eligible=int(catastrophic_mask.sum()),
        utility_eligible=int(utility_eligible_mask.sum()),
        all_eligible=int((catastrophic_mask & gate_eligible_mask & utility_eligible_mask).sum()),
        mu_ge_floor=int((frame["mu_net_decision_bps"] >= breakeven_floor).sum()),
        expected_utility_ge_zero=int((frame["expected_utility_bps"] >= 0.0).sum()),
        expected_utility_ge_floor=int(utility_eligible_mask.sum()),
        prob_adjusted_mu_p50_bps=_finite_quantile_or_none(frame["prob_adjusted_mu_bps"], 0.50),
        prob_adjusted_mu_p90_bps=_finite_quantile_or_none(frame["prob_adjusted_mu_bps"], 0.90),
        downside_drag_p50_bps=_finite_quantile_or_none(frame["downside_drag_bps"], 0.50),
        downside_drag_p90_bps=_finite_quantile_or_none(frame["downside_drag_bps"], 0.90),
        turnover_drag_p50_bps=_finite_quantile_or_none(frame["turnover_drag_bps"], 0.50),
        soft_gate_penalty_p50_bps=_finite_quantile_or_none(frame["soft_gate_penalty_bps"], 0.50),
        expected_utility_raw_p50_bps=_finite_quantile_or_none(frame["expected_utility_raw_bps"], 0.50),
        expected_utility_raw_p90_bps=_finite_quantile_or_none(frame["expected_utility_raw_bps"], 0.90),
        expected_utility_adj_p50_bps=_finite_quantile_or_none(frame["expected_utility_bps"], 0.50),
        expected_utility_adj_p90_bps=_finite_quantile_or_none(frame["expected_utility_bps"], 0.90),
    )
    return asdict(waterfall)


def _evaluate_shadow_profile(
    *,
    events: pd.DataFrame,
    cfg: CandidateStrategyConfig,
    catastrophic_mask: pd.Series,
    gate_mode: str,
    gate_floor: float,
    utility_floor_base: float,
    breakeven_fraction: float,
    utility_mode: str,
) -> dict[str, object]:
    """Evaluate a single shadow profile and return a result record."""
    frame = _selection_component_frame(
        events=events,
        cfg=cfg,
        gate_mode=str(gate_mode),
        gate_floor=float(gate_floor),
        utility_mode=utility_mode,
    )
    gate_eligible_mask = pd.Series(True, index=frame.index, dtype=bool)
    if gate_mode == "hard_floor":
        gate_eligible_mask = frame["p_pass"] >= float(gate_floor)
    utility_floor = max(float(utility_floor_base), float(breakeven_fraction) * float(cfg.cost_floor_bps))
    eligible = catastrophic_mask & gate_eligible_mask & (frame["expected_utility_bps"] >= utility_floor)
    selected_pre_group = 0
    selected_total = 0
    realized_mean = float("nan")
    realized_hit_rate = float("nan")
    rank_ic_val = float("nan")
    log_growth_proxy = float("-inf")
    if int(eligible.sum()) > 0:
        n_keep = max(1, math.ceil(int(eligible.sum()) * float(cfg.selection_shadow_top_quantile)))
        top_idx = (
            frame.loc[eligible]
            .sort_values(
                ["expected_utility_bps", "p_pass", "mu_net_decision_bps", "q10_net_bps"],
                ascending=[False, False, False, False],
            )
            .head(n_keep)
            .index
        )
        selected_pre_group = len(top_idx)
        selected_frame = frame.loc[top_idx].copy()
        selected_frame["_merge_dt"] = pd.to_datetime(selected_frame["datetime"], utc=True).dt.tz_localize(None)
        selected_frame = (
            selected_frame.sort_values(
                ["_merge_dt", "symbol", "expected_utility_bps"],
                ascending=[True, True, False],
            )
            .groupby(["_merge_dt", "symbol"], as_index=False)
            .first()
        )
        selected_total = int(selected_frame.shape[0])
        if "edge_after_hurdle_bps" in selected_frame.columns:
            realized_edge = pd.to_numeric(selected_frame["edge_after_hurdle_bps"], errors="coerce")
            if realized_edge.notna().any():
                realized_mean = float(realized_edge.mean())
                realized_hit_rate = float((realized_edge > 0.0).mean())
                rank_val = _rank_ic(
                    pd.to_numeric(selected_frame["expected_utility_bps"], errors="coerce"),
                    realized_edge,
                )
                rank_ic_val = float(rank_val) if rank_val is not None else float("nan")
                log_growth_proxy = float(
                    np.mean(
                        np.log1p(
                            np.clip(realized_edge.to_numpy(dtype=np.float64, copy=False) * 1e-4, -0.99, None)
                        )
                    )
                )
    return {
        "profile_id": (
            f"{utility_mode}:{gate_mode}:{float(gate_floor):.2f}:"
            f"{float(utility_floor_base):.1f}:{float(breakeven_fraction):.2f}"
        ),
        "utility_mode": str(utility_mode),
        "gate_mode": str(gate_mode),
        "gate_floor": float(gate_floor),
        "utility_floor_bps": utility_floor,
        "breakeven_floor_fraction": float(breakeven_fraction),
        "eligible": int(eligible.sum()),
        "selected_pre_group": int(selected_pre_group),
        "selected_total": int(selected_total),
        "realized_mean_bps": realized_mean,
        "realized_hit_rate": realized_hit_rate,
        "rank_ic": rank_ic_val,
        "log_growth_proxy": log_growth_proxy,
    }


def compute_shadow_selection_profiles(
    *,
    events: pd.DataFrame,
    cfg: CandidateStrategyConfig,
) -> pd.DataFrame:
    """Evaluate relaxed/tightened selection profiles on the same fold predictions.

    Now includes utility_mode axis to A/B-test additive_drag vs expected_edge_direct
    without altering production selection.
    """
    columns = [
        "profile_id",
        "utility_mode",
        "gate_mode",
        "gate_floor",
        "utility_floor_bps",
        "breakeven_floor_fraction",
        "eligible",
        "selected_pre_group",
        "selected_total",
        "realized_mean_bps",
        "realized_hit_rate",
        "rank_ic",
        "log_growth_proxy",
    ]
    if not cfg.selection_shadow_profiles_enabled or events.empty:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, object]] = []
    catastrophic_mask = _catastrophic_q10_mask(events, cfg)
    shadow_utility_modes = list(cfg.selection_shadow_utility_modes) or ["additive_drag"]
    for utility_mode, gate_mode, gate_floor, utility_floor_base, breakeven_fraction in product(
        shadow_utility_modes,
        cfg.selection_shadow_gate_modes,
        cfg.selection_shadow_gate_floors,
        cfg.selection_shadow_utility_floors_bps,
        cfg.selection_shadow_breakeven_floor_fractions,
    ):
        if gate_mode == "off" and gate_floor > 0.0:
            continue
        records.append(
            _evaluate_shadow_profile(
                events=events,
                cfg=cfg,
                catastrophic_mask=catastrophic_mask,
                gate_mode=str(gate_mode),
                gate_floor=float(gate_floor),
                utility_floor_base=float(utility_floor_base),
                breakeven_fraction=float(breakeven_fraction),
                utility_mode=str(utility_mode),
            )
        )
    profiles = pd.DataFrame.from_records(records, columns=columns)
    if profiles.empty:
        return profiles
    profiles = profiles.sort_values(
        ["selected_total", "log_growth_proxy", "realized_mean_bps", "selected_total"],
        ascending=[False, False, False, False],
        key=lambda s: (s >= cfg.min_fold_selected_events) if s.name == "selected_total" else s,
    )
    return profiles.head(max(1, int(cfg.selection_shadow_max_profiles))).reset_index(drop=True)


def compute_selection_sensitivity(
    *,
    events: pd.DataFrame,
    gate_grid: tuple[float, ...],
    edge_grid_bps: tuple[float, ...],
    q10_grid_bps: tuple[float, ...],
    cfg: CandidateStrategyConfig | None = None,
) -> pd.DataFrame:
    """Return pass counts across gate, edge, and q10 threshold grids."""
    if events.empty:
        return pd.DataFrame(
            columns=[
                "gate_threshold",
                "edge_threshold_bps",
                "q10_shortfall_bps",
                "total",
                "gate_pass",
                "edge_pass",
                "q10_pass",
                "all_pass",
                "all_pass_rate",
                "top_variant",
                "top_variant_pass",
            ]
        )

    records: list[dict[str, float | int | str]] = []
    total = int(events.shape[0])
    variant_keys = (
        events["family"].astype(str).str.cat(events["variant"].astype(str), sep=":")
        if {"family", "variant"}.issubset(events.columns)
        else pd.Series([""] * total, index=events.index, dtype="object")
    )
    for gate_threshold in gate_grid:
        gate_mask = events["p_pass"] >= gate_threshold
        for edge_threshold in edge_grid_bps:
            edge_mask = events["mu_net_decision_bps"] >= edge_threshold
            for q10_threshold in q10_grid_bps:
                q10_mask = events["q10_net_bps"] >= -q10_threshold
                all_mask = gate_mask & edge_mask & q10_mask
                passed = variant_keys.loc[all_mask]
                if passed.empty:
                    top_variant = ""
                    top_variant_pass = 0
                else:
                    counts = passed.value_counts(sort=True)
                    top_variant = str(counts.index[0])
                    top_variant_pass = int(counts.iloc[0])
                records.append(
                    {
                        "shortfall_basis": (
                            "absolute_bps" if cfg is None else str(cfg.shortfall_threshold_basis)
                        ),
                        "gate_threshold": float(gate_threshold),
                        "edge_threshold_bps": float(edge_threshold),
                        "q10_shortfall_bps": float(q10_threshold),
                        "total": total,
                        "gate_pass": int(gate_mask.sum()),
                        "edge_pass": int(edge_mask.sum()),
                        "q10_pass": int(q10_mask.sum()),
                        "all_pass": int(all_mask.sum()),
                        "all_pass_rate": float(all_mask.mean()),
                        "top_variant": top_variant,
                        "top_variant_pass": top_variant_pass,
                    }
                )
    return pd.DataFrame.from_records(records)


def _log_selection_sensitivity(df: pd.DataFrame, *, cfg: CandidateStrategyConfig) -> None:
    if df.empty:
        return
    logger = logging.getLogger(__name__)
    top = df.sort_values(
        ["all_pass", "all_pass_rate", "gate_threshold", "edge_threshold_bps", "q10_shortfall_bps"],
        ascending=[False, False, True, True, True],
    ).head(max(1, int(cfg.diagnostic_top_k)))
    for row in top.itertuples(index=False):
        logger.debug(
            "[DIAG][SELECT_SENS] gate>=%.2f edge>=%.1f q10>=-%.1f passed=%d pass_rate=%.4f top_variant=%s top_pass=%d",
            float(row.gate_threshold),
            float(row.edge_threshold_bps),
            float(row.q10_shortfall_bps),
            int(row.all_pass),
            float(row.all_pass_rate),
            str(row.top_variant),
            int(row.top_variant_pass),
        )


def _log_selection_by_variant(
    *,
    df: pd.DataFrame,
    gate_mask: pd.Series,
    edge_mask: pd.Series,
    q10_mask: pd.Series,
    pass_mask: pd.Series,
    cfg: CandidateStrategyConfig,
) -> None:
    """Log grouped candidate selection failure reasons."""
    if df.empty:
        return

    logger = logging.getLogger(__name__)
    grouped = df.groupby(["family", "variant"], sort=False, dropna=False)
    rows: list[tuple[str, int, int, int, int, int, float, float, float]] = []
    for (family, variant), group in grouped:
        idx = group.index
        rows.append(
            (
                f"{family}:{variant}",
                int(group.shape[0]),
                int((~gate_mask.loc[idx]).sum()),
                int((~edge_mask.loc[idx]).sum()),
                int((~q10_mask.loc[idx]).sum()),
                int(pass_mask.loc[idx].sum()),
                float(pd.to_numeric(group["p_pass"], errors="coerce").mean()),
                float(pd.to_numeric(group["mu_net_decision_bps"], errors="coerce").max()),
                float(pd.to_numeric(group["q10_net_bps"], errors="coerce").max()),
            )
        )

    for key, total, gate_fail, edge_fail, q10_fail, passed, mean_p, max_mu, max_q10 in sorted(
        rows,
        key=lambda item: item[1],
        reverse=True,
    )[: max(1, int(getattr(cfg, "diagnostic_top_k", 10)))]:
        logger.info(
            (
                "[DIAG][SELECT_VARIANT] key=%s total=%d gate_fail=%d edge_fail=%d "
                "q10_fail=%d passed=%d mean_p=%.3f max_mu=%.1f max_q10=%.1f"
            ),
            key,
            total,
            gate_fail,
            edge_fail,
            q10_fail,
            passed,
            mean_p,
            max_mu,
            max_q10,
        )


def select_candidate_events_for_portfolio(
    *,
    model_output: CandidateModelOutput,
    cfg: CandidateStrategyConfig,
) -> pd.DataFrame:
    """Select at most one active candidate per symbol per timestamp.

    Filters candidates by gate and edge criteria, resolving long/short conflicts.
    """
    events = model_output.events
    if events is None or events.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "symbol",
                "family",
                "variant",
                "side",
                "raw_score",
                "score_z",
                "expected_holding_bars",
                "min_holding_bars",
                "stop_atr_mult",
                "take_profit_atr_mult",
                "turnover_proxy",
                "cost_floor_bps",
                "entry_idx",
                "side_flipped",
                "p_pass",
                "mu_net_decision_bps",
                "q10_net_bps",
                "utility_score",
            ]
        )

    df = events.copy()
    df["p_pass"] = np.asarray(model_output.p_pass, dtype=np.float64)
    df["mu_net_decision_bps"] = np.asarray(model_output.mu_net_decision_bps, dtype=np.float64)
    df["q10_net_bps"] = np.asarray(model_output.q10_net_bps, dtype=np.float64)
    df["utility_score"] = np.asarray(model_output.utility_score, dtype=np.float64)

    if cfg.selection_sensitivity_enabled:
        sensitivity = compute_selection_sensitivity(
            events=df,
            gate_grid=cfg.selection_gate_grid,
            edge_grid_bps=cfg.selection_edge_grid_bps,
            q10_grid_bps=cfg.selection_q10_grid_bps,
            cfg=cfg,
        )
        _log_selection_sensitivity(sensitivity, cfg=cfg)

    gate_floor = float(cfg.selection_min_gate_probability_floor)
    df = _selection_component_frame(
        events=df,
        cfg=cfg,
        gate_mode=cfg.selection_gate_mode,
        gate_floor=gate_floor,
    )
    gate_mask = df["p_pass"] >= cfg.min_gate_probability
    edge_mask = df["mu_net_decision_bps"] >= cfg.min_expected_net_bps
    q10_mask = _q10_mask_for_mode(df, cfg)
    catastrophic_mask = _catastrophic_q10_mask(df, cfg)
    catastrophic_limits = _shortfall_limit_bps(df, cfg, catastrophic=True)
    utility_threshold = _utility_threshold(df=df, cfg=cfg, model_output=model_output)
    utility_mask = df["utility_score"] >= utility_threshold
    breakeven_floor = _resolve_breakeven_floor(df, cfg)
    gate_eligible_mask = pd.Series(True, index=df.index, dtype=bool)
    if cfg.selection_gate_mode == "hard_floor":
        gate_eligible_mask = df["p_pass"] >= gate_floor
    utility_floor = max(float(cfg.selection_min_expected_utility_bps), breakeven_floor)
    utility_eligible_mask = df["expected_utility_bps"] >= utility_floor
    eligible = catastrophic_mask & gate_eligible_mask & utility_eligible_mask
    waterfall = compute_selection_waterfall(events=df, cfg=cfg)
    shadow_profiles = compute_shadow_selection_profiles(events=df, cfg=cfg)
    shadow_best = shadow_profiles.iloc[0].to_dict() if not shadow_profiles.empty else {}
    n_eligible = int(eligible.sum())
    n_keep = 0
    zero_reason = "selected_nonzero"

    if cfg.selection_policy == "hard":
        mask = gate_mask & edge_mask & q10_mask
        zero_reason = "policy_hard"
    elif cfg.selection_policy == "validation_quantile":
        mask = catastrophic_mask & (df["mu_net_decision_bps"] >= 0.0) & utility_mask
        zero_reason = "policy_validation_quantile"
    else:
        if n_eligible == 0:
            mask = eligible
            zero_reason = "no_eligible_after_breakeven_floor"
        else:
            primary_sort_col = (
                "expected_utility_bps"
                if cfg.selection_utility_mode == "expected_edge_direct"
                else "utility_score"
            )
            eligible_df = df.loc[eligible].copy()
            eligible_df["_merge_dt"] = pd.to_datetime(eligible_df["datetime"], utc=True).dt.tz_localize(None)
            top_idx: list[int] = []
            max_variant_fraction = float(getattr(cfg, "max_variant_selection_fraction", 1.0))
            max_events_per_bar = getattr(cfg, "selection_max_events_per_bar", None)
            for _, group in eligible_df.groupby("_merge_dt", sort=True):
                deduped = (
                    group.reset_index().sort_values(
                        [primary_sort_col, "p_pass", "mu_net_decision_bps", "q10_net_bps"],
                        ascending=[False, False, False, False],
                    )
                    .groupby("symbol", sort=False, as_index=False)
                    .first()
                )
                keep_for_bar = max(1, math.ceil(len(deduped) * cfg.selection_top_quantile))
                if max_events_per_bar is not None:
                    keep_for_bar = min(keep_for_bar, int(max_events_per_bar))
                n_keep += keep_for_bar
                if max_variant_fraction < 1.0 and keep_for_bar > 1 and {"family", "variant"}.issubset(deduped.columns):
                    max_per_variant = max(1, math.ceil(keep_for_bar * max_variant_fraction))
                    variant_counts: dict[tuple[str, str], int] = {}
                    selected_for_bar = 0
                    for row in deduped.itertuples(index=False):
                        key = (str(row.family), str(row.variant))
                        current = variant_counts.get(key, 0)
                        if current >= max_per_variant:
                            continue
                        top_idx.append(int(row.index))
                        variant_counts[key] = current + 1
                        selected_for_bar += 1
                        if selected_for_bar >= keep_for_bar:
                            break
                else:
                    top_idx.extend(int(idx) for idx in deduped.head(keep_for_bar)["index"].tolist())
            mask = pd.Series(False, index=df.index, dtype=bool)
            mask.loc[top_idx] = True
            zero_reason = "selected_nonzero" if int(mask.sum()) > 0 else "topk_selected_zero"

    _log_selection_by_variant(
        df=df,
        gate_mask=gate_mask,
        edge_mask=edge_mask,
        q10_mask=catastrophic_mask if cfg.selection_policy != "hard" else q10_mask,
        pass_mask=mask,
        cfg=cfg,
    )

    _sel_logger = logging.getLogger(__name__)
    filtered = df.loc[mask].copy()
    if filtered.empty:
        diagnostics = {
            "total": len(df),
            "gate_pass": int(gate_mask.sum()),
            "edge_pass": int(edge_mask.sum()),
            "q10_pass": int((catastrophic_mask if cfg.selection_policy != "hard" else q10_mask).sum()),
            "utility_pass": int(utility_mask.sum()),
            "gate_mode": cfg.selection_gate_mode,
            "gate_floor_used": gate_floor,
            "expected_utility_floor_bps": utility_floor,
            "gate_eligible": int(gate_eligible_mask.sum()),
            "soft_gate_penalty_mean_bps": (
                float(pd.to_numeric(df["soft_gate_penalty_bps"], errors="coerce").mean()) if len(df) > 0 else 0.0
            ),
            "eligible": n_eligible,
            "selected_pre_group": 0,
            "selected_total": 0,
            "n_keep": n_keep,
            "policy": cfg.selection_policy,
            "zero_reason": zero_reason,
            "breakeven_floor_bps": breakeven_floor,
            "utility_threshold": utility_threshold,
            "shortfall_basis": cfg.shortfall_threshold_basis,
            "shortfall_limit_mean_bps": float(catastrophic_limits.mean()),
            "shortfall_limit_p90_bps": float(catastrophic_limits.quantile(0.9)),
            "realized_selected_edge_mean_bps": None,
            "realized_selected_hit_rate": None,
            "selected_rank_ic": None,
        }
        diagnostics.update({f"waterfall_{key}": value for key, value in waterfall.items()})
        if shadow_best:
            diagnostics.update({f"shadow_{key}": value for key, value in shadow_best.items()})
        filtered.attrs["candidate_selection_diagnostics"] = diagnostics
        _sel_logger.warning(
            (
                "[DIAG][SELECT_ZERO] total=%d policy=%s zero_reason=%s "
                "gate_fail=%d edge_fail=%d q10_fail=%d utility_fail=%d eligible=%d"
            ),
            len(df),
            cfg.selection_policy,
            zero_reason,
            int((~gate_mask).sum()),
            int((~edge_mask).sum()),
            int((~(catastrophic_mask if cfg.selection_policy != "hard" else q10_mask)).sum()),
            int((~utility_mask).sum()),
            n_eligible,
        )
        return filtered

    # Ensure datetime format is uniform for grouping
    filtered["_merge_dt"] = pd.to_datetime(filtered["datetime"], utc=True).dt.tz_localize(None)

    # Sort to resolve conflicts: pick highest utility score first
    filtered = filtered.sort_values(["_merge_dt", "symbol", "utility_score"], ascending=[True, True, False])

    # Per (datetime, symbol), pick the variant with the highest utility
    selected = filtered.groupby(["_merge_dt", "symbol"], as_index=False).first()
    selected_total = int(selected.shape[0])
    diagnostics = {
        "total": len(df),
        "gate_pass": int(gate_mask.sum()),
        "edge_pass": int(edge_mask.sum()),
        "q10_pass": int((catastrophic_mask if cfg.selection_policy != "hard" else q10_mask).sum()),
        "utility_pass": int(utility_mask.sum()),
        "gate_mode": cfg.selection_gate_mode,
        "gate_floor_used": gate_floor,
        "expected_utility_floor_bps": utility_floor,
        "gate_eligible": int(gate_eligible_mask.sum()),
        "soft_gate_penalty_mean_bps": (
            float(pd.to_numeric(df["soft_gate_penalty_bps"], errors="coerce").mean()) if len(df) > 0 else 0.0
        ),
        "eligible": n_eligible,
        "selected_pre_group": int(mask.sum()),
        "selected_total": selected_total,
        "n_keep": n_keep,
        "policy": cfg.selection_policy,
        "zero_reason": zero_reason,
        "breakeven_floor_bps": breakeven_floor,
        "utility_threshold": utility_threshold,
        "shortfall_basis": cfg.shortfall_threshold_basis,
        "shortfall_limit_mean_bps": float(catastrophic_limits.mean()),
        "shortfall_limit_p90_bps": float(catastrophic_limits.quantile(0.9)),
        "realized_selected_edge_mean_bps": None,
        "realized_selected_hit_rate": None,
        "selected_rank_ic": None,
    }
    if "edge_after_hurdle_bps" in selected.columns:
        realized_edge = pd.to_numeric(selected["edge_after_hurdle_bps"], errors="coerce")
        if realized_edge.notna().any():
            _re_arr = realized_edge.to_numpy(dtype=np.float64, copy=False)
            _wins = _re_arr[_re_arr > 0.0]
            _losses = _re_arr[_re_arr < 0.0]
            _payoff = (
                (float(np.mean(_wins)) / abs(float(np.mean(_losses))))
                if _wins.size > 0 and _losses.size > 0
                else float("nan")
            )
            _mu_ic = _rank_ic(
                pd.to_numeric(selected["mu_net_decision_bps"], errors="coerce"),
                realized_edge,
            )
            _sel_logger.info(
                "[DIAG][EDGE_IC] n=%d mu_net_rank_ic=%.4f hit_rate=%.3f payoff_ratio=%.3f "
                "realized_mean=%.1f win_mean=%.1f loss_mean=%.1f",
                len(realized_edge),
                float(_mu_ic) if _mu_ic is not None else float("nan"),
                float((realized_edge > 0.0).mean()),
                _payoff,
                float(realized_edge.mean()),
                float(np.mean(_wins)) if _wins.size > 0 else float("nan"),
                float(np.mean(_losses)) if _losses.size > 0 else float("nan"),
            )
            diagnostics["mu_net_rank_ic"] = float(_mu_ic) if _mu_ic is not None else None
            diagnostics["payoff_ratio"] = _payoff
        diagnostics["realized_selected_edge_mean_bps"] = (
            float(realized_edge.mean()) if realized_edge.notna().any() else None
        )
        diagnostics["realized_selected_hit_rate"] = (
            float((realized_edge > 0.0).mean()) if realized_edge.notna().any() else None
        )
        diagnostics["selected_rank_ic"] = _rank_ic(
            pd.to_numeric(selected["expected_utility_bps"], errors="coerce"),
            realized_edge,
        )
    diagnostics.update({f"waterfall_{key}": value for key, value in waterfall.items()})
    if shadow_best:
        diagnostics.update({f"shadow_{key}": value for key, value in shadow_best.items()})
    selected.attrs["candidate_selection_diagnostics"] = diagnostics
    _sel_logger.info(
        (
            "[DIAG][SELECT] total=%d gate_pass=%d edge_pass=%d q10_pass=%d utility_pass=%d "
            "eligible=%d selected_pre_group=%d selected=%d n_keep=%d | policy=%s zero_reason=%s "
            "thresholds(gate>=%.2f edge_net>=%.1f q10>=-%.1f utility>=%.3f breakeven_floor=%.1f)"
        ),
        diagnostics["total"],
        diagnostics["gate_pass"],
        diagnostics["edge_pass"],
        diagnostics["q10_pass"],
        diagnostics["utility_pass"],
        diagnostics["eligible"],
        diagnostics["selected_pre_group"],
        diagnostics["selected_total"],
        diagnostics["n_keep"],
        cfg.selection_policy,
        zero_reason,
        gate_floor,
        cfg.min_expected_net_bps,
        cfg.catastrophic_shortfall_bps if cfg.selection_policy != "hard" else cfg.max_expected_shortfall_bps,
        utility_threshold,
        breakeven_floor,
    )
    result = selected.drop(columns=["_merge_dt"]).reset_index(drop=True)
    result.attrs["candidate_selection_diagnostics"] = diagnostics
    return result


def build_candidate_target_weights(
    *,
    selected_events: pd.DataFrame,
    close_2d: NDArray[np.float64],
    symbols: tuple[str, ...],
    beta_2d: NDArray[np.float64] | None,
    sigma_3d: NDArray[np.float64] | None,
    cfg: CandidateStrategyConfig,
) -> NDArray[np.float64]:
    """Build target_weights_2d for the backtest engine using Fractional Kelly & Caps."""
    n_times, n_symbols = close_2d.shape
    raw_weights = np.zeros((n_times, n_symbols), dtype=np.float64)

    if selected_events.empty:
        return raw_weights

    # Map symbols to index
    sym_to_idx = {sym: idx for idx, sym in enumerate(symbols)}

    # ---------- Pass 1: entry_idx bar에 raw_weight 기록 ----------
    # 각 이벤트의 (entry_idx, symbol) → raw_weight 를 먼저 매핑한다.
    # 타임스텝 정렬된 리스트도 함께 수집해 Pass 2에서 재사용한다.
    # event_records: list of (entry_idx, s_idx, raw_signed_weight, holding_bars)
    event_records: list[tuple[int, int, float, int]] = []

    for row in selected_events.itertuples(index=False):
        sym = str(row.symbol)
        if sym not in sym_to_idx:
            continue
        s_idx = sym_to_idx[sym]
        t = int(row.entry_idx)
        if t < 0 or t >= n_times:
            continue

        side = float(row.side)
        # Normalise expected edge to per-bar scale before Kelly calculation.
        # mu_net_decision_bps is a per-HORIZON figure; dividing by holding_bars
        # converts it to per-bar, matching the per-bar variance denominator.
        holding_bars = max(int(getattr(row, "expected_holding_bars", 1)), 1)
        mu_event_bps = float(row.mu_net_decision_bps)
        if bool(getattr(cfg, "kelly_use_probability_adjusted_mu", True)):
            mu_event_bps = float(getattr(row, "p_pass", 1.0)) * mu_event_bps
        mu_i_per_bar = mu_event_bps * 1e-4 / holding_bars

        # Trailing variance retrieval
        variance_i = 1e-4  # Default fallback
        if sigma_3d is not None:
            # sigma_3d shape is usually [T, N, N] covariance matrix
            variance_i = float(sigma_3d[t, s_idx, s_idx])
        else:
            # Fallback trailing close returns std
            st = max(0, t - 20)
            if t > st:
                ret = np.diff(close_2d[st : t + 1, s_idx]) / np.maximum(close_2d[st:t, s_idx], 1e-12)
                v = float(np.var(ret))
                if np.isfinite(v) and v > 1e-12:
                    variance_i = v

        if bool(getattr(cfg, "kelly_downside_variance_floor_enabled", True)):
            q10_per_bar = abs(min(float(getattr(row, "q10_net_bps", 0.0)), 0.0)) * 1e-4 / holding_bars
            downside_var_floor = q10_per_bar * q10_per_bar
            variance_i = max(variance_i, downside_var_floor)
        variance_i = max(variance_i, 1e-12)
        # Fractional Kelly: raw_weight = kelly_fraction * mu_i_per_bar / variance_i
        raw_w = cfg.kelly_fraction * mu_i_per_bar / variance_i

        # Phase 3: regime-as-size-multiplier.  Applies a continuous regime
        # multiplier at the Kelly weight level before cap projection, so the
        # signal itself is not masked — only position size is attenuated.
        regime_mult = 1.0
        if cfg.regime_as_size_multiplier:
            regime_name = str(getattr(row, "entry_regime", ""))
            regime_mult_map: dict[str, float] = dict(cfg.regime_size_multipliers)
            regime_mult = regime_mult_map.get(regime_name, 1.0)

        signed_w = raw_w * np.sign(side) * regime_mult
        raw_weights[t, s_idx] = signed_w
        event_records.append((t, s_idx, signed_w, holding_bars))

    # ---------- Pass 2: entry_idx → entry_idx + holding_bars - 1 구간 forward-fill ----------
    # 타임스텝 오름차순 정렬 후 순회: 이미 비영값(다른 이벤트로 채워진 구간)은 덮어쓰지 않는다.
    event_records.sort(key=lambda r: r[0])
    for entry_t, s_idx, signed_w, holding_bars in event_records:
        fill_end = min(entry_t + holding_bars, n_times)
        for fill_t in range(entry_t + 1, fill_end):
            if raw_weights[fill_t, s_idx] == 0.0:
                raw_weights[fill_t, s_idx] = signed_w

    # Apply 5-cap multi-cap projection per timestamp
    caps = PortfolioCaps(
        gross=cfg.gross_cap,
        per_symbol=cfg.max_symbol_weight,
        net=cfg.net_cap,
        beta=cfg.beta_cap,
        target_ann_vol=cfg.target_ann_vol,
    )

    target_weights = np.zeros_like(raw_weights)
    bars_per_year = 2190.0  # Default 4h bars per year (365 * 6)
    if cfg.timeframe == "1h":
        bars_per_year = 8760.0
    elif cfg.timeframe == "1d":
        bars_per_year = 365.0

    for t in range(n_times):
        w_pre = raw_weights[t]
        beta_t = beta_2d[t] if beta_2d is not None else np.zeros(n_symbols)
        
        # Portfolio standard deviation calculation
        sigma_port_t = 1e-3
        if sigma_3d is not None:
            cov = sigma_3d[t]
            var_port = float(np.dot(w_pre, np.dot(cov, w_pre)))
            if np.isfinite(var_port) and var_port > 0.0:
                sigma_port_t = math.sqrt(var_port)
        else:
            # Simple vol target fallback standard deviation
            sigma_port_t = float(np.nanstd(w_pre)) if np.any(w_pre) else 1e-3

        target_weights[t] = project_all_caps(
            w=w_pre,
            btc_beta=beta_t,
            sigma_port=sigma_port_t,
            bars_per_year=bars_per_year,
            caps=caps,
        )

    return target_weights


def build_candidate_alpha_panel(
    *,
    selected_events: pd.DataFrame,
    target_weights_2d: NDArray[np.float64],
    datetimes: NDArray[np.datetime64],
    symbols: tuple[str, ...],
    cfg: CandidateStrategyConfig | None = None,
) -> pd.DataFrame:
    """Build long-format panel for merge into data maps."""
    n_times, n_symbols = target_weights_2d.shape
    if n_times == 0 or n_symbols == 0:
        empty = pd.DataFrame(
            columns=[
                "alpha_long",
                "alpha_short",
                "target_weight",
                "candidate_family",
                "candidate_variant",
                "p_pass",
                "mu_net_decision_bps",
                "q10_net_bps",
                "utility_score",
                "candidate_stop_atr_mult",
                "candidate_take_profit_atr_mult",
            ]
        )
        empty.index = pd.MultiIndex.from_arrays(
            [pd.Index([], dtype="datetime64[ns]"), pd.Index([], dtype="object")],
            names=["datetime", "symbol"],
        )
        return empty

    rows: list[pd.DataFrame] = []

    # Map symbols to index
    sym_to_idx = {sym: idx for idx, sym in enumerate(symbols)}

    # Group by execution index so metadata aligns with the target weight row.
    df_selected = selected_events.copy()
    grouped: dict[int, pd.DataFrame] = {}
    if not df_selected.empty and "entry_idx" in df_selected.columns:
        if bool(getattr(cfg, "candidate_metadata_forward_fill", True)):
            expanded_rows: list[dict[str, object]] = []
            for row in df_selected.to_dict(orient="records"):
                entry_idx = int(row["entry_idx"])
                holding_bars = max(int(row.get("expected_holding_bars", 1)), 1)
                fill_end = min(entry_idx + holding_bars, n_times)
                for fill_idx in range(entry_idx, fill_end):
                    copied = dict(row)
                    copied["_entry_idx"] = fill_idx
                    expanded_rows.append(copied)
            expanded = (
                pd.DataFrame(expanded_rows)
                if expanded_rows
                else pd.DataFrame(columns=[*df_selected.columns, "_entry_idx"])
            )
            grouped = {int(key): group for key, group in expanded.groupby("_entry_idx")}
        else:
            df_selected["_entry_idx"] = df_selected["entry_idx"].astype(int)
            grouped = {int(key): group for key, group in df_selected.groupby("_entry_idx")}

    for t in range(n_times):
        # Default empty attributes
        alpha_long = np.zeros(n_symbols, dtype=np.float64)
        alpha_short = np.zeros(n_symbols, dtype=np.float64)
        target_w = target_weights_2d[t]

        # Extract direction components
        alpha_long[target_w > 0.0] = target_w[target_w > 0.0]
        alpha_short[target_w < 0.0] = -target_w[target_w < 0.0]

        families = [""] * n_symbols
        variants = [""] * n_symbols
        p_pass = np.zeros(n_symbols, dtype=np.float64)
        mu_bps = np.zeros(n_symbols, dtype=np.float64)
        q10_bps = np.zeros(n_symbols, dtype=np.float64)
        utility = np.zeros(n_symbols, dtype=np.float64)
        stop_atr_mult = np.zeros(n_symbols, dtype=np.float64)
        take_profit_atr_mult = np.zeros(n_symbols, dtype=np.float64)

        if t in grouped:
            dt_group = grouped[t]
            for row in dt_group.itertuples(index=False):
                sym = str(row.symbol)
                if sym in sym_to_idx:
                    s_idx = sym_to_idx[sym]
                    families[s_idx] = str(row.family)
                    variants[s_idx] = str(row.variant)
                    p_pass[s_idx] = float(row.p_pass)
                    mu_bps[s_idx] = float(row.mu_net_decision_bps)
                    q10_bps[s_idx] = float(row.q10_net_bps)
                    utility[s_idx] = float(row.utility_score)
                    stop_atr_mult[s_idx] = float(getattr(row, "stop_atr_mult", 0.0))
                    take_profit_atr_mult[s_idx] = float(getattr(row, "take_profit_atr_mult", 0.0))

        df_t = pd.DataFrame({
            "datetime": datetimes[t],
            "symbol": list(symbols),
            "alpha_long": alpha_long,
            "alpha_short": alpha_short,
            "target_weight": target_w,
            "candidate_family": families,
            "candidate_variant": variants,
            "p_pass": p_pass,
            "mu_net_decision_bps": mu_bps,
            "q10_net_bps": q10_bps,
            "utility_score": utility,
            "candidate_stop_atr_mult": stop_atr_mult,
            "candidate_take_profit_atr_mult": take_profit_atr_mult,
        })
        rows.append(df_t)

    if not rows:
        empty_df = pd.DataFrame(columns=[
            "alpha_long", "alpha_short", "target_weight", "candidate_family",
            "candidate_variant", "p_pass", "mu_net_decision_bps", "q10_net_bps", "utility_score",
            "candidate_stop_atr_mult", "candidate_take_profit_atr_mult",
        ])
        empty_df.index = pd.MultiIndex.from_arrays(
            [pd.Index([], dtype="datetime64[ns]"), pd.Index([], dtype="object")],
            names=["datetime", "symbol"]
        )
        return empty_df

    panel = (
        pd.concat(rows, axis=0, ignore_index=True)
        .set_index(["datetime", "symbol"])
        .sort_index()
    )
    return panel
