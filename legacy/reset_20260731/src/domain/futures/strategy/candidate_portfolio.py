from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import pandas as pd
from numba import njit
from numpy.typing import NDArray

from src.domain.futures.optimization.metrics import _bars_per_year_for_tf
from src.domain.futures.portfolio.covariance import (
    active_covariance,
    compute_log_returns_2d,
    solve_portfolio_kelly,
)
from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps, project_all_caps
from src.domain.futures.strategy.candidate_contracts import (
    CandidateModelOutput,
)
from src.domain.futures.strategy.config import CandidateStrategyConfig


@dataclass(frozen=True)
class SelectionWaterfall:
    total: int
    breakeven_floor_bps: float
    utility_floor_bps: float
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
    utility_mode: str | None = None,
) -> pd.DataFrame:
    frame = events.copy()
    resolved_utility_mode = utility_mode if utility_mode is not None else cfg.selection_utility_mode
    turnover_proxy = _series_or_default(frame, "turnover_proxy", 1.0)
    prob_adjusted_mu_bps = pd.to_numeric(frame["mu_net_decision_bps"], errors="coerce")
    downside_drag_bps = float(cfg.downside_penalty) * frame["q10_net_bps"].clip(upper=0.0).abs()
    turnover_drag_bps = cfg.turnover_penalty * turnover_proxy
    expected_utility_raw_bps = prob_adjusted_mu_bps - downside_drag_bps - turnover_drag_bps
    expected_utility_bps = expected_utility_raw_bps.copy()
    frame["prob_adjusted_mu_bps"] = pd.to_numeric(prob_adjusted_mu_bps, errors="coerce")
    frame["downside_drag_bps"] = pd.to_numeric(downside_drag_bps, errors="coerce")
    frame["turnover_drag_bps"] = pd.to_numeric(turnover_drag_bps, errors="coerce")
    frame["expected_utility_raw_bps"] = pd.to_numeric(expected_utility_raw_bps, errors="coerce")
    if resolved_utility_mode == "expected_edge_direct":
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
    breakeven_floor = _resolve_breakeven_floor(events, cfg)
    utility_floor = max(float(cfg.selection_min_expected_utility_bps), breakeven_floor)
    if events.empty:
        return asdict(
            SelectionWaterfall(
                total=0,
                breakeven_floor_bps=breakeven_floor,
                utility_floor_bps=utility_floor,
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
                expected_utility_raw_p50_bps=None,
                expected_utility_raw_p90_bps=None,
                expected_utility_adj_p50_bps=None,
                expected_utility_adj_p90_bps=None,
            )
        )

    frame = _selection_component_frame(
        events=events,
        cfg=cfg,
    )
    catastrophic_mask = _catastrophic_q10_mask(frame, cfg)
    utility_eligible_mask = frame["expected_utility_bps"] >= utility_floor
    waterfall = SelectionWaterfall(
        total=int(frame.shape[0]),
        breakeven_floor_bps=breakeven_floor,
        utility_floor_bps=utility_floor,
        catastrophic_eligible=int(catastrophic_mask.sum()),
        utility_eligible=int(utility_eligible_mask.sum()),
        all_eligible=int((catastrophic_mask & utility_eligible_mask).sum()),
        mu_ge_floor=int((frame["mu_net_decision_bps"] >= breakeven_floor).sum()),
        expected_utility_ge_zero=int((frame["expected_utility_bps"] >= 0.0).sum()),
        expected_utility_ge_floor=int(utility_eligible_mask.sum()),
        prob_adjusted_mu_p50_bps=_finite_quantile_or_none(frame["prob_adjusted_mu_bps"], 0.50),
        prob_adjusted_mu_p90_bps=_finite_quantile_or_none(frame["prob_adjusted_mu_bps"], 0.90),
        downside_drag_p50_bps=_finite_quantile_or_none(frame["downside_drag_bps"], 0.50),
        downside_drag_p90_bps=_finite_quantile_or_none(frame["downside_drag_bps"], 0.90),
        turnover_drag_p50_bps=_finite_quantile_or_none(frame["turnover_drag_bps"], 0.50),
        expected_utility_raw_p50_bps=_finite_quantile_or_none(frame["expected_utility_raw_bps"], 0.50),
        expected_utility_raw_p90_bps=_finite_quantile_or_none(frame["expected_utility_raw_bps"], 0.90),
        expected_utility_adj_p50_bps=_finite_quantile_or_none(frame["expected_utility_bps"], 0.50),
        expected_utility_adj_p90_bps=_finite_quantile_or_none(frame["expected_utility_bps"], 0.90),
    )
    return asdict(waterfall)


def _evaluate_shadow_profile(
    *,
    prebuilt_frame: pd.DataFrame,
    cfg: CandidateStrategyConfig,
    catastrophic_mask: pd.Series,
    utility_floor_base: float,
    breakeven_fraction: float,
    utility_mode: str,
) -> dict[str, object]:
    """Evaluate a single shadow profile and return a result record."""
    frame = prebuilt_frame
    utility_floor = max(float(utility_floor_base), float(breakeven_fraction) * float(cfg.cost_floor_bps))
    eligible = catastrophic_mask & (frame["expected_utility_bps"] >= utility_floor)
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
                ["expected_utility_bps", "mu_net_decision_bps", "q10_net_bps"],
                ascending=[False, False, False],
            )
            .head(n_keep)
            .index
        )
        selected_pre_group = len(top_idx)
        selected_frame = frame.loc[top_idx].copy()
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
                    np.mean(np.log1p(np.clip(realized_edge.to_numpy(dtype=np.float64, copy=False) * 1e-4, -0.99, None)))
                )
    return {
        "profile_id": (f"{utility_mode}:{float(utility_floor_base):.1f}:{float(breakeven_fraction):.2f}"),
        "utility_mode": str(utility_mode),
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

    events_with_dt = events.copy()
    events_with_dt["_merge_dt"] = pd.to_datetime(events_with_dt["datetime"], utc=True).dt.tz_localize(None)

    shadow_utility_modes = list(cfg.selection_shadow_utility_modes) or ["additive_drag"]
    prebuilt_frames = {}
    for mode in shadow_utility_modes:
        prebuilt_frames[mode] = _selection_component_frame(
            events=events_with_dt,
            cfg=cfg,
            utility_mode=mode,
        )

    records: list[dict[str, object]] = []
    catastrophic_mask = _catastrophic_q10_mask(events_with_dt, cfg)
    for utility_mode, utility_floor_base, breakeven_fraction in product(
        shadow_utility_modes,
        cfg.selection_shadow_utility_floors_bps,
        cfg.selection_shadow_breakeven_floor_fractions,
    ):
        records.append(
            _evaluate_shadow_profile(
                prebuilt_frame=prebuilt_frames[utility_mode],
                cfg=cfg,
                catastrophic_mask=catastrophic_mask,
                utility_floor_base=float(utility_floor_base),
                breakeven_fraction=float(breakeven_fraction),
                utility_mode=str(utility_mode),
            )
        )
    profiles = pd.DataFrame.from_records(records, columns=columns)
    if profiles.empty:
        return profiles
    profiles = profiles.sort_values(
        ["selected_total", "eligible", "utility_floor_bps", "breakeven_floor_fraction", "profile_id"],
        ascending=[False, False, True, True, True],
        key=lambda s: (s >= cfg.min_fold_selected_events) if s.name == "selected_total" else s,
    )
    return profiles.head(max(1, int(cfg.selection_shadow_max_profiles))).reset_index(drop=True)


@njit(cache=True)  # type: ignore
def _compute_sensitivity_grids_numba(
    mu_net: NDArray[np.float64],
    q10_net: NDArray[np.float64],
    variant_codes: NDArray[np.int64],
    num_variants: int,
    gate_grid: NDArray[np.float64],
    edge_grid: NDArray[np.float64],
    q10_grid: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    n_gate = gate_grid.shape[0]
    n_edge = edge_grid.shape[0]
    n_q10 = q10_grid.shape[0]
    n_events = mu_net.shape[0]

    all_pass_counts = np.zeros((n_gate, n_edge, n_q10), dtype=np.int64)
    top_variant_codes = np.full((n_gate, n_edge, n_q10), -1, dtype=np.int64)
    top_variant_counts = np.zeros((n_gate, n_edge, n_q10), dtype=np.int64)

    for g_idx in range(n_gate):
        for e_idx in range(n_edge):
            edge_val = edge_grid[e_idx]
            for q_idx in range(n_q10):
                q10_val = q10_grid[q_idx]

                pass_count = 0
                v_counts = np.zeros(num_variants, dtype=np.int64)

                for i in range(n_events):
                    if mu_net[i] >= edge_val and q10_net[i] >= -q10_val:
                        pass_count += 1
                        code = variant_codes[i]
                        if code >= 0:
                            v_counts[code] += 1

                all_pass_counts[g_idx, e_idx, q_idx] = pass_count

                max_count = 0
                best_code = -1
                for c in range(num_variants):
                    if v_counts[c] > max_count:
                        max_count = v_counts[c]
                        best_code = c

                top_variant_codes[g_idx, e_idx, q_idx] = best_code
                top_variant_counts[g_idx, e_idx, q_idx] = max_count

    return all_pass_counts, top_variant_codes, top_variant_counts


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

    mu_net = events["mu_net_decision_bps"].to_numpy(dtype=np.float64, copy=False)
    q10_net = events["q10_net_bps"].to_numpy(dtype=np.float64, copy=False)

    if {"family", "variant"}.issubset(events.columns):
        variant_keys = (events["family"].astype(str) + ":" + events["variant"].astype(str)).to_numpy(dtype=object)
        uniques, variant_codes = np.unique(variant_keys, return_inverse=True)
        num_variants = len(uniques)
    else:
        uniques = np.array([""])
        variant_codes = np.zeros(total, dtype=np.int64)
        num_variants = 1

    g_arr = np.array(gate_grid, dtype=np.float64)
    e_arr = np.array(edge_grid_bps, dtype=np.float64)
    q_arr = np.array(q10_grid_bps, dtype=np.float64)

    all_pass_counts, top_codes, top_counts = _compute_sensitivity_grids_numba(
        mu_net, q10_net, variant_codes.astype(np.int64), num_variants, g_arr, e_arr, q_arr
    )

    shortfall_basis = "absolute_bps" if cfg is None else str(cfg.shortfall_threshold_basis)
    for g_idx, gate_threshold in enumerate(gate_grid):
        for e_idx, edge_threshold in enumerate(edge_grid_bps):
            for q_idx, q10_threshold in enumerate(q10_grid_bps):
                code = top_codes[g_idx, e_idx, q_idx]
                top_variant = str(uniques[code]) if code >= 0 else ""
                records.append(
                    {
                        "shortfall_basis": shortfall_basis,
                        "gate_threshold": float(gate_threshold),
                        "edge_threshold_bps": float(edge_threshold),
                        "q10_shortfall_bps": float(q10_threshold),
                        "total": total,
                        "gate_pass": total,
                        "edge_pass": int(np.sum(mu_net >= edge_threshold)),
                        "q10_pass": int(np.sum(q10_net >= -q10_threshold)),
                        "all_pass": int(all_pass_counts[g_idx, e_idx, q_idx]),
                        "all_pass_rate": float(all_pass_counts[g_idx, e_idx, q_idx] / total),
                        "top_variant": top_variant,
                        "top_variant_pass": int(top_counts[g_idx, e_idx, q_idx]),
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
        logger.debug(
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


def _vectorized_topk_per_bar(
    eligible_df: pd.DataFrame,
    *,
    top_quantile: float,
    max_events_per_bar: int | None,
    max_variant_fraction: float,
    primary_sort_col: str,
) -> NDArray[np.intp]:
    eligible = eligible_df.sort_values(
        ["_merge_dt", primary_sort_col, "mu_net_decision_bps", "q10_net_bps"],
        ascending=[True, False, False, False],
    )
    deduped = eligible.drop_duplicates(["_merge_dt", "symbol"], keep="first")

    bar_size = deduped.groupby("_merge_dt", sort=False)["_merge_dt"].transform("size")
    keep = np.ceil(bar_size * top_quantile).clip(lower=1)
    if max_events_per_bar is not None:
        keep = np.minimum(keep, float(max_events_per_bar))

    if max_variant_fraction >= 1.0 or not {"family", "variant"}.issubset(deduped.columns):
        rank = deduped.groupby("_merge_dt", sort=False).cumcount()
        mask = rank < keep
        idx_arr = deduped.index.to_numpy(dtype=np.intp)
        return idx_arr[mask.to_numpy(dtype=bool)]  # type: ignore[no-any-return]

    max_per_variant = np.ceil(keep * max_variant_fraction).clip(lower=1)
    vr = deduped.groupby(["_merge_dt", "family", "variant"], sort=False).cumcount()
    variant_ok = vr < max_per_variant
    survived = deduped[variant_ok]
    if survived.empty:
        return np.array([], dtype=np.intp)
    re_rank = survived.groupby("_merge_dt", sort=False).cumcount()
    re_keep = keep.loc[survived.index]
    final_mask = re_rank < re_keep
    idx_arr = survived.index.to_numpy(dtype=np.intp)
    return idx_arr[final_mask.to_numpy(dtype=bool)]  # type: ignore[no-any-return]


def select_candidate_events_for_portfolio(
    *,
    model_output: CandidateModelOutput,
    cfg: CandidateStrategyConfig,
    enable_diagnostics: bool = True,
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
    df["q90_net_bps"] = np.asarray(model_output.q90_net_bps, dtype=np.float64)
    df["utility_score"] = np.asarray(model_output.utility_score, dtype=np.float64)

    if enable_diagnostics and cfg.selection_sensitivity_enabled:
        sensitivity = compute_selection_sensitivity(
            events=df,
            gate_grid=cfg.selection_gate_grid,
            edge_grid_bps=cfg.selection_edge_grid_bps,
            q10_grid_bps=cfg.selection_q10_grid_bps,
            cfg=cfg,
        )
        _log_selection_sensitivity(sensitivity, cfg=cfg)

    df = _selection_component_frame(
        events=df,
        cfg=cfg,
    )
    gate_mask = pd.Series(True, index=df.index, dtype=bool)
    edge_mask = df["mu_net_decision_bps"] >= cfg.min_expected_net_bps
    q10_mask = _q10_mask_for_mode(df, cfg)
    catastrophic_mask = _catastrophic_q10_mask(df, cfg)
    catastrophic_limits = _shortfall_limit_bps(df, cfg, catastrophic=True)
    utility_threshold = _utility_threshold(df=df, cfg=cfg, model_output=model_output)
    utility_mask = df["utility_score"] >= utility_threshold
    breakeven_floor = _resolve_breakeven_floor(df, cfg)
    utility_floor = max(float(cfg.selection_min_expected_utility_bps), breakeven_floor)
    utility_eligible_mask = df["expected_utility_bps"] >= utility_floor
    eligible = catastrophic_mask & utility_eligible_mask
    _diag_enabled = enable_diagnostics and cfg.l1_selection_diagnostics_enabled
    waterfall = compute_selection_waterfall(events=df, cfg=cfg) if _diag_enabled else {}
    shadow_profiles = (
        compute_shadow_selection_profiles(events=df, cfg=cfg)
        if enable_diagnostics and cfg.selection_shadow_profiles_enabled
        else pd.DataFrame()
    )
    n_eligible = int(eligible.sum())
    n_keep = 0
    zero_reason = "selected_nonzero"

    if cfg.selection_policy == "hard":
        mask = edge_mask & q10_mask
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
                "expected_utility_bps" if cfg.selection_utility_mode == "expected_edge_direct" else "utility_score"
            )
            eligible_df = df.loc[eligible].copy()
            eligible_df["_merge_dt"] = pd.to_datetime(eligible_df["datetime"], utc=True).dt.tz_localize(None)
            max_variant_fraction = float(getattr(cfg, "max_variant_selection_fraction", 1.0))
            max_events_per_bar = getattr(cfg, "selection_max_events_per_bar", None)
            top_idx = _vectorized_topk_per_bar(
                eligible_df,
                top_quantile=cfg.selection_top_quantile,
                max_events_per_bar=max_events_per_bar,
                max_variant_fraction=max_variant_fraction,
                primary_sort_col=primary_sort_col,
            )
            n_keep = len(top_idx)
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
            "gate_mode": "off",
            "gate_floor_used": 0.0,
            "expected_utility_floor_bps": utility_floor,
            "gate_eligible": len(df),
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
        diagnostics["shadow_profile_count"] = len(shadow_profiles)
        diagnostics["shadow_max_selected_total"] = (
            int(pd.to_numeric(shadow_profiles["selected_total"], errors="coerce").max())
            if not shadow_profiles.empty
            else 0
        )
        diagnostics["shadow_max_eligible"] = (
            int(pd.to_numeric(shadow_profiles["eligible"], errors="coerce").max()) if not shadow_profiles.empty else 0
        )
        filtered.attrs["candidate_selection_diagnostics"] = diagnostics
        _sel_logger.debug(
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
        "gate_mode": "off",
        "gate_floor_used": 0.0,
        "expected_utility_floor_bps": utility_floor,
        "gate_eligible": len(df),
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
            _sel_logger.debug(
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
    diagnostics["shadow_profile_count"] = len(shadow_profiles)
    diagnostics["shadow_max_selected_total"] = (
        int(pd.to_numeric(shadow_profiles["selected_total"], errors="coerce").max()) if not shadow_profiles.empty else 0
    )
    diagnostics["shadow_max_eligible"] = (
        int(pd.to_numeric(shadow_profiles["eligible"], errors="coerce").max()) if not shadow_profiles.empty else 0
    )
    selected.attrs["candidate_selection_diagnostics"] = diagnostics
    _sel_logger.debug(
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
        0.0,
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
    regime_code_1d: NDArray[np.int32] | None = None,
) -> NDArray[np.float64]:
    """Build target_weights_2d for the backtest engine using Fractional Kelly & Caps."""
    n_times, n_symbols = close_2d.shape
    raw_weights = np.zeros((n_times, n_symbols), dtype=np.float64)

    if selected_events.empty:
        return raw_weights

    # Map symbols to index
    sym_to_idx = {sym: idx for idx, sym in enumerate(symbols)}

    use_port_kelly: bool = bool(getattr(cfg, "use_portfolio_kelly", False))
    # mu_2d stores signed r-unit expected returns for portfolio Kelly; allocated only when needed
    mu_2d: NDArray[np.float64] | None = np.zeros((n_times, n_symbols), dtype=np.float64) if use_port_kelly else None

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
        holding_bars = max(int(getattr(row, "expected_holding_bars", 1)), 1)
        risk_unit_bps = max(
            float(getattr(row, "risk_unit_bps", getattr(cfg, "min_risk_unit_bps", 25.0))),
            float(getattr(cfg, "min_risk_unit_bps", 25.0)),
        )
        p_pass_val = float(getattr(row, "p_pass", 1.0))
        if getattr(cfg, "sizing_mode", "stop_risk") == "calibrated_event_kelly":
            mu_return_r = float(row.mu_net_decision_bps) / max(risk_unit_bps, 1e-12)
            q10_return_r = float(getattr(row, "q10_net_bps", 0.0)) / max(risk_unit_bps, 1e-12)
            q90_return_r = float(getattr(row, "q90_net_bps", 0.0)) / max(risk_unit_bps, 1e-12)
            sigma_r = np.maximum((q90_return_r - q10_return_r) / 2.563, 1e-6)
            second_moment = max(mu_return_r * mu_return_r + sigma_r * sigma_r, 1e-6)
            raw_abs_w = cfg.kelly_fraction * max(mu_return_r, 0.0) / second_moment
        else:
            mu_return_r = float(getattr(row, "mu_net_decision_bps", 0.0)) / max(risk_unit_bps, 1e-12)
            raw_abs_w = float(getattr(cfg, "event_risk_budget", 0.0025)) / max(risk_unit_bps * 1e-4, 1e-12)
        raw_abs_w = raw_abs_w * p_pass_val
        raw_w = min(float(cfg.max_symbol_weight), raw_abs_w)

        if mu_2d is not None:
            mu_2d[t, s_idx] = float(np.sign(side)) * abs(mu_return_r)

        overlay_mult = 1.0
        if bool(getattr(cfg, "overlay_sizing_enabled", True)):
            overlay_mult = float(getattr(row, "overlay_mult", 1.0))
        elif cfg.regime_as_size_multiplier:
            regime_name = str(getattr(row, "entry_regime", ""))
            regime_mult_map: dict[str, float] = dict(cfg.regime_size_multipliers)
            overlay_mult = regime_mult_map.get(regime_name, 1.0)

        signed_w = raw_w * np.sign(side) * overlay_mult
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
            if mu_2d is not None and mu_2d[fill_t, s_idx] == 0.0 and mu_2d[entry_t, s_idx] != 0.0:
                mu_2d[fill_t, s_idx] = mu_2d[entry_t, s_idx]

    # ---------- Precompute log-returns for portfolio Kelly ----------
    logret_2d: NDArray[np.float64] | None = None
    if use_port_kelly:
        logret_2d = compute_log_returns_2d(close_2d)

    target_weights = np.zeros_like(raw_weights)
    # [ADR_20260707_L1_BACKTEST_FIDELITY_FIXES] TF-generic annualization (was 4h/1h/1d-only elif chain)
    bars_per_year = _bars_per_year_for_tf(cfg.timeframe)

    # Extract portfolio Kelly config params once (avoid repeated getattr in inner loop)
    _cov_window: int = int(getattr(cfg, "cov_window", 180))
    _cov_min_obs: int = int(getattr(cfg, "cov_min_obs", 60))
    _cov_shrinkage_raw = getattr(cfg, "cov_shrinkage", "auto")
    _cov_shrinkage: float | None = None if _cov_shrinkage_raw == "auto" else float(_cov_shrinkage_raw)
    _cov_ridge_eps: float = float(getattr(cfg, "cov_ridge_eps", 1e-3))

    for t in range(n_times):
        w_pre = raw_weights[t].copy()

        # ---------- Portfolio Kelly covariance overlay ----------
        if use_port_kelly and logret_2d is not None and mu_2d is not None:
            active_idx_arr = np.nonzero(w_pre)[0].astype(np.int64)
            if len(active_idx_arr) >= 2:
                cov_s = active_covariance(logret_2d, t, active_idx_arr, _cov_window, _cov_shrinkage, _cov_min_obs)
                if cov_s is not None:
                    mu_s = mu_2d[t, active_idx_arr]
                    w_s = solve_portfolio_kelly(mu_s, cov_s, cfg.kelly_fraction, _cov_ridge_eps, cfg.max_symbol_weight)
                    w_pre_new = np.zeros(n_symbols, dtype=np.float64)
                    w_pre_new[active_idx_arr] = w_s
                    w_pre = w_pre_new

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

        # 1. 국면 코드 파악
        regime_code = int(regime_code_1d[t]) if regime_code_1d is not None else 1

        # 2. 국면별 Cap 조절 비율 획득
        gross_mult = cfg.regime_gross_multipliers.get(regime_code, 1.0)
        net_mult = cfg.regime_net_multipliers.get(regime_code, 1.0)

        # 3. Double Vol-Targeting Scaling Guard 적용
        use_double_scaling_guard = cfg.double_scaling_guard and (
            cfg.sizing_mode == "calibrated_event_kelly" or bool(getattr(cfg, "overlay_sizing_enabled", True))
        )
        target_vol = 0.0 if use_double_scaling_guard else cfg.target_ann_vol

        # 4. 동적 PortfolioCaps 인스턴스 생성
        caps = PortfolioCaps(
            gross=cfg.gross_cap * gross_mult,
            per_symbol=cfg.max_symbol_weight,
            net=cfg.net_cap * net_mult,
            beta=cfg.beta_cap,
            target_ann_vol=target_vol,
        )

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

    # -- Pre-allocate output arrays (n_times x n_symbols rows total) ----------
    total_rows = n_times * n_symbols
    _dt_arr = np.empty(total_rows, dtype="datetime64[ns]")
    _sym_arr = np.empty(total_rows, dtype=object)
    _al_arr = np.zeros(total_rows, dtype=np.float64)
    _as_arr = np.zeros(total_rows, dtype=np.float64)
    _tw_arr = np.zeros(total_rows, dtype=np.float64)
    _fam_arr = np.empty(total_rows, dtype=object)
    _var_arr = np.empty(total_rows, dtype=object)
    _pp_arr = np.zeros(total_rows, dtype=np.float64)
    _mu_arr = np.zeros(total_rows, dtype=np.float64)
    _q10_arr = np.zeros(total_rows, dtype=np.float64)
    _ut_arr = np.zeros(total_rows, dtype=np.float64)
    _satr_arr = np.zeros(total_rows, dtype=np.float64)
    _tpatr_arr = np.zeros(total_rows, dtype=np.float64)

    # Initialise string columns with empty string
    _fam_arr[:] = ""
    _var_arr[:] = ""

    sym_list = list(symbols)
    dt_arr_typed = datetimes.astype("datetime64[ns]")

    for t in range(n_times):
        base = t * n_symbols
        end = base + n_symbols
        target_w = target_weights_2d[t]

        _dt_arr[base:end] = dt_arr_typed[t]
        _sym_arr[base:end] = sym_list
        _tw_arr[base:end] = target_w

        long_mask = target_w > 0.0
        short_mask = target_w < 0.0
        _al_arr[base:end][long_mask] = target_w[long_mask]
        _as_arr[base:end][short_mask] = -target_w[short_mask]

        if t in grouped:
            dt_group = grouped[t]
            for row in dt_group.itertuples(index=False):
                sym = str(row.symbol)
                if sym in sym_to_idx:
                    s_idx = sym_to_idx[sym]
                    i = base + s_idx
                    _fam_arr[i] = str(row.family)
                    _var_arr[i] = str(row.variant)
                    _pp_arr[i] = float(row.p_pass)
                    _mu_arr[i] = float(row.mu_net_decision_bps)
                    _q10_arr[i] = float(row.q10_net_bps)
                    _ut_arr[i] = float(row.utility_score)
                    _satr_arr[i] = float(getattr(row, "stop_atr_mult", 0.0))
                    _tpatr_arr[i] = float(getattr(row, "take_profit_atr_mult", 0.0))

    if total_rows == 0:
        empty_df = pd.DataFrame(
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
        empty_df.index = pd.MultiIndex.from_arrays(
            [pd.Index([], dtype="datetime64[ns]"), pd.Index([], dtype="object")], names=["datetime", "symbol"]
        )
        return empty_df

    panel = (
        pd.DataFrame(
            {
                "datetime": pd.to_datetime(_dt_arr),
                "symbol": _sym_arr,
                "alpha_long": _al_arr,
                "alpha_short": _as_arr,
                "target_weight": _tw_arr,
                "candidate_family": _fam_arr,
                "candidate_variant": _var_arr,
                "p_pass": _pp_arr,
                "mu_net_decision_bps": _mu_arr,
                "q10_net_bps": _q10_arr,
                "utility_score": _ut_arr,
                "candidate_stop_atr_mult": _satr_arr,
                "candidate_take_profit_atr_mult": _tpatr_arr,
            }
        )
        .set_index(["datetime", "symbol"])
        .sort_index()
    )
    return panel
