from __future__ import annotations

import logging
import math
from typing import Literal, TypedDict

import numba
import numpy as np

_logger = logging.getLogger(__name__)

AlphaOutputUnit = Literal["return_fraction", "rank_weight"]
AlphaGateLayer = Literal[
    "mechanical_integrity",
    "rank_skill",
    "policy_economics",
    "execution_realism",
    "statistical_robustness",
]


class AlphaGateReason(TypedDict):
    reason: str
    layer: AlphaGateLayer
    metric: str
    observed: float
    threshold: float
    unit: str


def _nw_t_stat(series: np.ndarray, *, horizon_bars: int) -> float:
    """Compute Newey-West HAC t-stat for a 1D IC series."""
    valid = series[np.isfinite(series)]
    n_obs = int(valid.size)
    if n_obs < 2:
        return 0.0
    mean_ic = float(np.mean(valid))
    lag = min(max(int(horizon_bars), 1), n_obs // 4)
    demeaned = valid - mean_ic
    s0 = float(np.mean(demeaned**2))
    for j in range(1, lag + 1):
        cov_j = float(np.mean(demeaned[j:] * demeaned[:-j]))
        s0 += 2.0 * (1.0 - j / (lag + 1)) * cov_j
    se_nw = math.sqrt(max(s0, 1e-12) / n_obs)
    return float(mean_ic / max(se_nw, 1e-12))


@numba.njit(cache=True)  # type: ignore
def _fast_rank1d(x: np.ndarray) -> np.ndarray:
    """Numba-compatible 1D rank calculator (average rank handling for ties)."""
    n = len(x)
    temp = np.argsort(x)
    ranks = np.empty(n, dtype=np.float64)
    
    i = 0
    while i < n:
        j = i
        while j < n - 1 and x[temp[j]] == x[temp[j + 1]]:
            j += 1
        
        rank_val = (i + j + 2.0) / 2.0
        for k in range(i, j + 1):
            ranks[temp[k]] = rank_val
        i = j + 1
    return ranks


@numba.njit(cache=True)  # type: ignore
def _fast_pearson_core(x: np.ndarray, y: np.ndarray) -> float:
    """Numba-accelerated Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for i in range(n):
        dx = x[i] - x_mean
        dy = y[i] - y_mean
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
        
    den = np.sqrt(den_x * den_y)
    if den > 1e-12:
        return float(num / den)
    return 0.0


@numba.njit(cache=True)  # type: ignore
def _numba_rolling_ic_spearman(sig_2d: np.ndarray, fwd_ret_2d: np.ndarray) -> np.ndarray:
    """JIT compiled core Spearman rolling IC loop."""
    t_len, n_syms = sig_2d.shape
    ic_values = np.empty(t_len, dtype=np.float64)
    
    for t in range(t_len):
        row_sig = sig_2d[t]
        row_ret = fwd_ret_2d[t]
        
        # Collect non-NaN indices
        valid_count = 0
        for i in range(n_syms):
            if np.isfinite(row_sig[i]) and np.isfinite(row_ret[i]):
                valid_count += 1
                
        if valid_count < 5:
            ic_values[t] = np.nan
            continue
            
        s_vals = np.empty(valid_count, dtype=np.float64)
        r_vals = np.empty(valid_count, dtype=np.float64)
        curr = 0
        for i in range(n_syms):
            if np.isfinite(row_sig[i]) and np.isfinite(row_ret[i]):
                s_vals[curr] = row_sig[i]
                r_vals[curr] = row_ret[i]
                curr += 1
                
        is_const_s = True
        is_const_r = True
        s0 = s_vals[0]
        r0 = r_vals[0]
        for i in range(1, valid_count):
            if s_vals[i] != s0:
                is_const_s = False
            if r_vals[i] != r0:
                is_const_r = False
                
        if is_const_s or is_const_r:
            ic_values[t] = 0.0
            continue
            
        rx = _fast_rank1d(s_vals)
        ry = _fast_rank1d(r_vals)
        ic_values[t] = _fast_pearson_core(rx, ry)
    return ic_values


@numba.njit(cache=True)  # type: ignore
def _numba_rolling_ic_pearson(sig_2d: np.ndarray, fwd_ret_2d: np.ndarray) -> np.ndarray:
    """JIT compiled core Pearson rolling IC loop."""
    t_len, n_syms = sig_2d.shape
    ic_values = np.empty(t_len, dtype=np.float64)
    
    for t in range(t_len):
        row_sig = sig_2d[t]
        row_ret = fwd_ret_2d[t]
        
        valid_count = 0
        for i in range(n_syms):
            if np.isfinite(row_sig[i]) and np.isfinite(row_ret[i]):
                valid_count += 1
                
        if valid_count < 5:
            ic_values[t] = np.nan
            continue
            
        s_vals = np.empty(valid_count, dtype=np.float64)
        r_vals = np.empty(valid_count, dtype=np.float64)
        curr = 0
        for i in range(n_syms):
            if np.isfinite(row_sig[i]) and np.isfinite(row_ret[i]):
                s_vals[curr] = row_sig[i]
                r_vals[curr] = row_ret[i]
                curr += 1
                
        is_const_s = True
        is_const_r = True
        s0 = s_vals[0]
        r0 = r_vals[0]
        for i in range(1, valid_count):
            if s_vals[i] != s0:
                is_const_s = False
            if r_vals[i] != r0:
                is_const_r = False
                
        if is_const_s or is_const_r:
            ic_values[t] = 0.0
            continue
            
        ic_values[t] = _fast_pearson_core(s_vals, r_vals)
    return ic_values


def rolling_ic(
    sig_2d: np.ndarray,
    fwd_ret_2d: np.ndarray,
    *,
    method: str = "spearman",
) -> np.ndarray:
    """Compute cross-sectional IC over time utilizing Numba JIT acceleration."""
    if sig_2d.shape != fwd_ret_2d.shape:
        raise ValueError("sig_2d and fwd_ret_2d must have the same shape")
    
    s_2d = sig_2d.astype(np.float64)
    r_2d = fwd_ret_2d.astype(np.float64)
    
    if method == "spearman":
        return _numba_rolling_ic_spearman(s_2d, r_2d)  # type: ignore[no-any-return]
    elif method == "pearson":
        return _numba_rolling_ic_pearson(s_2d, r_2d)  # type: ignore[no-any-return]
    else:
        raise ValueError(f"Unknown method: {method}")


def feature_cs_ic_audit(
    feature_values_3d: np.ndarray,
    feature_names: tuple[str, ...],
    target_2d: np.ndarray,
    *,
    breakeven_ic: float,
    horizon_bars: int = 12,
    top_k: int = 15,
) -> list[dict[str, float | str]]:
    """Per-feature cross-sectional IC audit on OOS panel.

    Returns top-k rows sorted by |mean_ic| desc.
    """
    if feature_values_3d.ndim != 3:
        raise ValueError("feature_values_3d must be 3D [T, N, F]")
    t_len, n_len, f_len = feature_values_3d.shape
    if target_2d.shape != (t_len, n_len):
        raise ValueError("target_2d must have shape [T, N] matching feature_values_3d")
    if f_len != len(feature_names):
        raise ValueError("feature_names length must match feature dimension")
    if top_k <= 0:
        return []

    rows: list[dict[str, float | str]] = []
    for f_idx, name in enumerate(feature_names):
        ic_series = rolling_ic(feature_values_3d[:, :, f_idx], target_2d, method="spearman")
        stats = ic_summary(ic_series)
        mean_ic = float(stats["mean_ic"])
        gap = float(mean_ic - breakeven_ic)
        rows.append(
            {
                "name": name,
                "mean_ic": mean_ic,
                "t_stat_nw": _nw_t_stat(ic_series, horizon_bars=horizon_bars),
                "gap": gap,
                "hit": float(stats["hit_ratio"]),
            }
        )
    rows.sort(key=lambda item: abs(float(item["mean_ic"])), reverse=True)
    return rows[: min(top_k, len(rows))]


def ic_summary(ic_series: np.ndarray) -> dict[str, float]:
    """Summarize IC series with stability statistics."""
    valid = ic_series[np.isfinite(ic_series)]
    n_obs = len(valid)
    if n_obs == 0:
        return {
            "mean_ic": 0.0,
            "ic_std": 0.0,
            "icir": 0.0,
            "t_stat": 0.0,
            "n_obs": 0.0,
            "hit_ratio": 0.0,
        }
    mean_ic = float(np.mean(valid))
    ic_std = float(np.std(valid, ddof=1)) if n_obs > 1 else 0.0
    if ic_std < 1e-12:
        icir = 0.0
        t_stat = 0.0
    else:
        icir = mean_ic / ic_std
        t_stat = mean_ic / (ic_std / np.sqrt(n_obs))
    hit_ratio = float(np.mean(valid > 0.0))
    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "icir": icir,
        "t_stat": t_stat,
        "n_obs": float(n_obs),
        "hit_ratio": hit_ratio,
    }


def ic_lcb_hac(
    ic_series: np.ndarray,
    *,
    horizon_bars: int,
    z: float = 1.0,
) -> float:
    """Compute lower confidence bound of mean IC using HAC standard error."""
    valid = np.asarray(ic_series, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    n_obs = int(valid.size)
    if n_obs < 2:
        return 0.0
    mean_ic = float(np.mean(valid))
    t_nw = _nw_t_stat(valid, horizon_bars=horizon_bars)
    if not np.isfinite(t_nw) or abs(t_nw) < 1e-12:
        return mean_ic
    se_nw = abs(mean_ic / t_nw)
    return float(mean_ic - float(max(z, 0.0)) * se_nw)


def top_bottom_spread_bps(
    score_2d: np.ndarray,
    realized_2d: np.ndarray,
    eligible_2d: np.ndarray,
    *,
    quantile: float,
    cost_bps: float,
) -> dict[str, float]:
    """Top-bottom spread and cost-adjusted spread diagnostics in bps."""
    if score_2d.shape != realized_2d.shape or score_2d.shape != eligible_2d.shape:
        raise ValueError("score/realized/eligible shapes must match")
    q = float(np.clip(quantile, 0.01, 0.49))
    t_len, _n_len = score_2d.shape
    gross_spreads: list[float] = []
    turnover_proxy_rows: list[float] = []
    for t in range(t_len):
        row_mask = (
            np.asarray(eligible_2d[t], dtype=bool)
            & np.isfinite(score_2d[t])
            & np.isfinite(realized_2d[t])
        )
        idx = np.flatnonzero(row_mask)
        if idx.size < 6:
            continue
        keep = max(1, int(np.floor(idx.size * q)))
        row_score = score_2d[t, idx]
        order = np.argsort(row_score, kind="mergesort")
        short_idx = idx[order[:keep]]
        long_idx = idx[order[-keep:]]
        long_ret = float(np.nanmean(realized_2d[t, long_idx]))
        short_ret = float(np.nanmean(realized_2d[t, short_idx]))
        gross_spreads.append(long_ret - short_ret)
        turnover_proxy_rows.append((2.0 * keep) / float(idx.size))

    if not gross_spreads:
        return {
            "n_obs": 0.0,
            "gross_spread_bps": 0.0,
            "net_spread_bps": -float(cost_bps),
            "net_spread_lcb_bps": -float(cost_bps),
            "turnover_proxy": 0.0,
        }

    gross_arr = np.asarray(gross_spreads, dtype=np.float64)
    gross_mean_bps = float(np.mean(gross_arr) * 1e4)
    gross_std_bps = float(np.std(gross_arr, ddof=1) * 1e4) if gross_arr.size > 1 else 0.0
    se_bps = gross_std_bps / max(np.sqrt(gross_arr.size), 1.0)
    net_mean_bps = gross_mean_bps - float(cost_bps)
    net_lcb_bps = (gross_mean_bps - se_bps) - float(cost_bps)
    return {
        "n_obs": float(gross_arr.size),
        "gross_spread_bps": gross_mean_bps,
        "net_spread_bps": net_mean_bps,
        "net_spread_lcb_bps": net_lcb_bps,
        "turnover_proxy": float(np.mean(np.asarray(turnover_proxy_rows, dtype=np.float64))),
    }


def passes_ic_gate(
    summary: dict[str, float],
    *,
    min_mean_ic: float = 0.02,
    min_t_stat: float = 2.0,
    min_hit_ratio: float = 0.45,
) -> bool:
    """Return True when IC gate thresholds are satisfied.

    Accepts both ic_summary() output (keys: mean_ic, t_stat, hit_ratio)
    and build_quality_report() output (keys: spearman_rank_ic, ic_t_stat, ic_hit_ratio).

    Args:
        summary: IC statistics dict from ic_summary() or build_quality_report().
        min_mean_ic: Minimum mean IC threshold.
        min_t_stat: Minimum t-statistic threshold.
        min_hit_ratio: Minimum hit ratio threshold.

    Returns:
        True when all thresholds are satisfied.

    """
    mean_ic = summary.get("mean_ic", summary.get("spearman_rank_ic", 0.0))
    t_stat = summary.get("t_stat", summary.get("ic_t_stat", 0.0))
    hit_ratio = summary.get("hit_ratio", summary.get("ic_hit_ratio", 0.0))
    return bool(
        mean_ic >= min_mean_ic
        and t_stat >= min_t_stat
        and hit_ratio >= min_hit_ratio
    )


def ml_alpha_metrics(alpha_long: np.ndarray, alpha_short: np.ndarray) -> dict[str, float]:
    """Compute compact ML alpha diagnostics used by logs and tests."""
    al = alpha_long[np.isfinite(alpha_long)]
    ash = alpha_short[np.isfinite(alpha_short)]
    if al.size == 0 or ash.size == 0:
        return {"long_nz": 0.0, "short_nz": 0.0, "long_p95_bps": 0.0, "short_p95_bps": 0.0}
    return {
        "long_nz": float(np.mean(np.abs(al) > 1e-12)),
        "short_nz": float(np.mean(np.abs(ash) > 1e-12)),
        "alpha_p50_bps": float(np.nanpercentile(al - ash, 50) * 10000.0),
        "long_p95_bps": float(np.nanpercentile(al, 95) * 10000.0),
        "short_p95_bps": float(np.nanpercentile(ash, 95) * 10000.0),
    }


def nonzero_ratio(arr: np.ndarray, *, eps: float = 1e-12) -> float:
    """Return finite non-zero ratio for a numeric array."""
    finite = np.asarray(arr, dtype=np.float64)
    if finite.size == 0:
        return 0.0
    mask = np.isfinite(finite)
    if not np.any(mask):
        return 0.0
    vals = finite[mask]
    return float(np.count_nonzero(np.abs(vals) > eps) / vals.size)


def preservation_ratio(
    before: np.ndarray,
    after: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Return non-zero survival ratio after gating.

    Raises:
        ValueError: If shapes differ.

    """
    if before.shape != after.shape:
        raise ValueError("before and after must have the same shape")
    denom = nonzero_ratio(before, eps=eps)
    if denom <= 0.0:
        return 0.0
    ratio = float(nonzero_ratio(after, eps=eps) / denom)
    # Bounded contract: preservation ratio must stay within [0, 1].
    return float(np.clip(ratio, 0.0, 1.0))


def ndcg_proxy_at_k(score_2d: np.ndarray, rel_2d: np.ndarray, *, k: int = 5) -> float:
    """Compute simple NDCG proxy at K across timestamps."""
    if score_2d.shape != rel_2d.shape:
        raise ValueError("score_2d and rel_2d must have the same shape")
    if k <= 0:
        raise ValueError("k must be > 0")
    vals: list[float] = []
    log_denom = np.log2(np.arange(2, k + 2, dtype=np.float64))
    for t in range(score_2d.shape[0]):
        m = np.isfinite(score_2d[t]) & np.isfinite(rel_2d[t])
        if int(m.sum()) < k:
            continue
        s = score_2d[t, m]
        r = rel_2d[t, m]
        ord_pred = np.argsort(-s)[:k]
        ord_best = np.argsort(-r)[:k]
        dcg: float = float(np.sum((np.power(2.0, r[ord_pred]) - 1.0) / log_denom))
        idcg: float = float(np.sum((np.power(2.0, r[ord_best]) - 1.0) / log_denom))
        if idcg > 1e-12:
            vals.append(float(dcg / idcg))
    return float(np.mean(vals)) if vals else 0.0


def build_quality_report(
    *,
    feature_values: np.ndarray,
    feature_valid_mask: np.ndarray,
    label_eligible_mask: np.ndarray,
    score_2d: np.ndarray,
    signed_ret_2d: np.ndarray,
    relevance_2d: np.ndarray,
    q10_2d: np.ndarray | None = None,
    q50_2d: np.ndarray | None = None,
    q90_2d: np.ndarray | None = None,
    alpha_long_2d: np.ndarray | None = None,
    alpha_short_2d: np.ndarray | None = None,
    cost_2d: np.ndarray | None = None,
    ic_score_2d: np.ndarray | None = None,
) -> dict[str, float]:
    """Build structured diagnostics report used for quality gates."""
    ic_input = ic_score_2d if ic_score_2d is not None else score_2d
    if ic_score_2d is not None and ic_score_2d.shape != signed_ret_2d.shape:
        raise ValueError("ic_score_2d and signed_ret_2d must have the same shape")
    ic_series = rolling_ic(ic_input, signed_ret_2d, method="spearman")
    ic_stats = ic_summary(ic_series)
    report: dict[str, float] = {
        "feature_finite_ratio": (
            float(np.mean(np.isfinite(feature_values[feature_valid_mask])))
            if np.any(feature_valid_mask)
            else 0.0
        ),
        "label_valid_ratio": float(np.mean(label_eligible_mask)),
        "feature_valid_ratio": float(np.mean(feature_valid_mask)),
        "ranker_valid_ndcg_at_5": ndcg_proxy_at_k(score_2d, relevance_2d, k=5),
        "spearman_rank_ic": ic_stats["mean_ic"],
        "fold_oos_ic": ic_stats["mean_ic"],
        "ic_icir": ic_stats["icir"],
        "ic_t_stat": ic_stats["t_stat"],
        "ic_hit_ratio": ic_stats["hit_ratio"],
        "ic_n_obs": ic_stats["n_obs"],
    }
    if q10_2d is not None and q50_2d is not None and q90_2d is not None:
        hit_mask = np.isfinite(q50_2d) & np.isfinite(signed_ret_2d)
        report["q50_sign_hit"] = float(
            np.mean(
                (q50_2d[hit_mask] > 0.0) == (signed_ret_2d[hit_mask] > 0.0)
            )
        )
        report["q10_q50_q90_spread"] = float(
            np.nanmean((q90_2d - q50_2d) + (q50_2d - q10_2d))
        )
    if alpha_long_2d is not None and alpha_short_2d is not None:
        a_metrics = ml_alpha_metrics(alpha_long_2d, alpha_short_2d)
        report["alpha_long_non_zero_ratio"] = a_metrics["long_nz"]
        report["alpha_short_non_zero_ratio"] = a_metrics["short_nz"]
        report["alpha_p50_bps"] = a_metrics["alpha_p50_bps"]
        report["alpha_p95_bps"] = max(a_metrics["long_p95_bps"], a_metrics["short_p95_bps"])
    if cost_2d is not None:
        ev = np.abs(score_2d[np.isfinite(score_2d)])
        cost = np.abs(cost_2d[np.isfinite(cost_2d)])
        if cost.size > 0:
            report["ev_cost_ratio_proxy"] = float(np.mean(ev) / max(float(np.mean(cost)), 1e-12))
    return report


def passes_quality_gate(report: dict[str, float]) -> bool:
    """Return True when key quality metrics satisfy minimal thresholds."""
    return bool(
        report.get("feature_finite_ratio", 0.0) >= 0.990
        and report.get("label_valid_ratio", 0.0) > 0.0
        and report.get("ranker_valid_ndcg_at_5", 0.0) > 0.0
        and report.get("spearman_rank_ic", -1.0) >= 0.0
    )


def passes_directional_viability_gate(
    summary: dict[str, float],
    *,
    min_long_non_zero_ratio: float = 0.0,
    min_short_non_zero_ratio: float = 0.0,
) -> bool:
    """Check directional viability from alpha non-zero ratios only."""
    long_ratio = summary.get("alpha_long_non_zero_ratio", 0.0)
    short_ratio = summary.get("alpha_short_non_zero_ratio", 0.0)
    return bool(
        long_ratio >= min_long_non_zero_ratio
        and short_ratio >= min_short_non_zero_ratio
    )


def alpha_gate_diagnostics(
    *,
    alpha_p95_bps: float,
    friction_bps: float,
    hurdle_bps: float,
    long_nz: float,
    short_nz: float,
    xs_long_preservation_ratio: float,
    xs_short_preservation_ratio: float,
    min_long_nz: float,
    min_short_nz: float,
    min_xs_preservation: float,
    cost_wall_tolerance_bps: float = 0.0,
    active_alpha_p95_bps: float | None = None,
    tradable_long_nz: float = 0.0,
    tradable_short_nz: float = 0.0,
    min_tradable_long_nz: float = 0.0,
    min_tradable_short_nz: float = 0.0,
    alpha_output_unit: AlphaOutputUnit = "return_fraction",
    require_alpha_cost_wall: bool = True,
) -> dict[str, object]:
    """Evaluate alpha admission diagnostics without mixing rank weights and return bps."""
    floor_bps = float(friction_bps + hurdle_bps)
    metric_bps = (
        float(active_alpha_p95_bps)
        if active_alpha_p95_bps is not None
        else float(alpha_p95_bps)
    )
    metric_source = "active_alpha_p95_bps" if active_alpha_p95_bps is not None else "alpha_p95_bps"
    fail_reasons: list[str] = []
    reason_details: list[AlphaGateReason] = []
    cost_wall_required = bool(require_alpha_cost_wall and alpha_output_unit == "return_fraction")
    if cost_wall_required and metric_bps < (floor_bps - max(0.0, float(cost_wall_tolerance_bps))):
        fail_reasons.append("alpha_p95_below_cost_wall")
        reason_details.append(
            {
                "reason": "alpha_p95_below_cost_wall",
                "layer": "policy_economics",
                "metric": metric_source,
                "observed": metric_bps,
                "threshold": floor_bps,
                "unit": "bps",
            }
        )
    if long_nz < min_long_nz:
        fail_reasons.append("long_nz_below_threshold")
        reason_details.append(
            {
                "reason": "long_nz_below_threshold",
                "layer": "mechanical_integrity",
                "metric": "long_nz",
                "observed": float(long_nz),
                "threshold": float(min_long_nz),
                "unit": "ratio",
            }
        )
    if short_nz < min_short_nz:
        fail_reasons.append("short_nz_below_threshold")
        reason_details.append(
            {
                "reason": "short_nz_below_threshold",
                "layer": "mechanical_integrity",
                "metric": "short_nz",
                "observed": float(short_nz),
                "threshold": float(min_short_nz),
                "unit": "ratio",
            }
        )
    if xs_long_preservation_ratio < min_xs_preservation:
        fail_reasons.append("xs_long_preservation_below_threshold")
        reason_details.append(
            {
                "reason": "xs_long_preservation_below_threshold",
                "layer": "rank_skill",
                "metric": "xs_long_preservation_ratio",
                "observed": float(xs_long_preservation_ratio),
                "threshold": float(min_xs_preservation),
                "unit": "ratio",
            }
        )
    if xs_short_preservation_ratio < min_xs_preservation:
        fail_reasons.append("xs_short_preservation_below_threshold")
        reason_details.append(
            {
                "reason": "xs_short_preservation_below_threshold",
                "layer": "rank_skill",
                "metric": "xs_short_preservation_ratio",
                "observed": float(xs_short_preservation_ratio),
                "threshold": float(min_xs_preservation),
                "unit": "ratio",
            }
        )
    if alpha_output_unit != "rank_weight" and tradable_long_nz < min_tradable_long_nz:
        fail_reasons.append("tradable_long_nz_below_threshold")
        reason_details.append(
            {
                "reason": "tradable_long_nz_below_threshold",
                "layer": "execution_realism",
                "metric": "tradable_long_nz",
                "observed": float(tradable_long_nz),
                "threshold": float(min_tradable_long_nz),
                "unit": "ratio",
            }
        )
    if alpha_output_unit != "rank_weight" and tradable_short_nz < min_tradable_short_nz:
        fail_reasons.append("tradable_short_nz_below_threshold")
        reason_details.append(
            {
                "reason": "tradable_short_nz_below_threshold",
                "layer": "execution_realism",
                "metric": "tradable_short_nz",
                "observed": float(tradable_short_nz),
                "threshold": float(min_tradable_short_nz),
                "unit": "ratio",
            }
        )
    return {
        "alpha_gate_pass": len(fail_reasons) == 0,
        "alpha_gate_fail_reasons": fail_reasons,
        "alpha_gate_floor_bps": floor_bps,
        "alpha_gate_metric_bps": metric_bps,
        "alpha_gate_metric_source": metric_source,
        "alpha_output_unit": alpha_output_unit,
        "alpha_cost_wall_required": cost_wall_required,
        "alpha_gate_reason_details": reason_details,
    }


def side_alpha_tail_metrics(
    alpha_long: np.ndarray,
    alpha_short: np.ndarray,
    *,
    cost_floor: float,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Measure active-side magnitude and tradable density separately from full matrix sparsity."""
    if alpha_long.shape != alpha_short.shape:
        raise ValueError("alpha_long and alpha_short must have the same shape")
    long_vals = np.asarray(alpha_long, dtype=np.float64)
    short_vals = np.asarray(alpha_short, dtype=np.float64)
    long_finite = long_vals[np.isfinite(long_vals)]
    short_finite = short_vals[np.isfinite(short_vals)]
    long_active = long_finite[np.abs(long_finite) > eps]
    short_active = short_finite[np.abs(short_finite) > eps]
    long_p95 = float(np.nanpercentile(long_active, 95) * 10000.0) if long_active.size > 0 else 0.0
    short_p95 = (
        float(np.nanpercentile(short_active, 95) * 10000.0) if short_active.size > 0 else 0.0
    )
    long_full_p95 = (
        float(np.nanpercentile(long_finite, 95) * 10000.0) if long_finite.size > 0 else 0.0
    )
    short_full_p95 = (
        float(np.nanpercentile(short_finite, 95) * 10000.0) if short_finite.size > 0 else 0.0
    )
    long_tradable = float(np.mean(long_finite >= cost_floor)) if long_finite.size > 0 else 0.0
    short_tradable = float(np.mean(short_finite >= cost_floor)) if short_finite.size > 0 else 0.0
    return {
        "alpha_full_matrix_p95_bps": max(long_full_p95, short_full_p95),
        "alpha_active_p95_bps": max(long_p95, short_p95),
        "alpha_long_active_p95_bps": long_p95,
        "alpha_short_active_p95_bps": short_p95,
        "alpha_long_tradable_nz": long_tradable,
        "alpha_short_tradable_nz": short_tradable,
        "alpha_long_active_count": float(long_active.size),
        "alpha_short_active_count": float(short_active.size),
    }


def gross_return_diagnostics(
    gross_long_2d: np.ndarray,
    resid_long_2d: np.ndarray,
    eligible_2d: np.ndarray,
    *,
    min_symbols: int = 5,
) -> dict[str, float]:
    """Compare cross-sectional variance: raw vs beta-residualized returns.

    Computes variance retention ratio to quantify signal shrinkage from residualization.
    All computation is vectorized over the eligible mask; rows with fewer than
    ``min_symbols`` valid symbols are skipped.

    Args:
        gross_long_2d: Raw log returns before any transform, shape [T, N].
        resid_long_2d: Returns after beta-residualization (pre-CS-demean), shape [T, N].
        eligible_2d: Boolean eligibility mask, shape [T, N].
        min_symbols: Minimum number of valid symbols required per timestep.

    Returns:
        Dict with keys:
            raw_cs_std_mean: Mean cross-sectional std of gross_long_2d per timestep.
            resid_cs_std_mean: Mean cross-sectional std of resid_long_2d per timestep.
            variance_retention_ratio: resid_cs_std_mean / max(raw_cs_std_mean, 1e-12).
            n_timesteps: Number of eligible timesteps with >= min_symbols valid symbols.
            raw_nonzero_ratio: Fraction of eligible (t, i) where |gross_long| > 1e-8.
            resid_nonzero_ratio: Fraction of eligible (t, i) where |resid_long| > 1e-8.

    Time complexity: O(T * N). Space complexity: O(1) auxiliary beyond inputs.

    """
    # Shape: [T, N]
    t_len, _n_len = gross_long_2d.shape

    raw_cs_stds: list[float] = []
    resid_cs_stds: list[float] = []
    raw_nz_count: int = 0
    resid_nz_count: int = 0
    total_eligible: int = 0

    for t in range(t_len):
        row_mask = (
            eligible_2d[t]
            & np.isfinite(gross_long_2d[t])
            & np.isfinite(resid_long_2d[t])
        )
        n_valid = int(row_mask.sum())
        if n_valid < min_symbols:
            continue

        raw_vals = gross_long_2d[t, row_mask]
        resid_vals = resid_long_2d[t, row_mask]

        raw_cs_stds.append(float(np.std(raw_vals, ddof=0)))
        resid_cs_stds.append(float(np.std(resid_vals, ddof=0)))

        raw_nz_count += int(np.sum(np.abs(raw_vals) > 1e-8))
        resid_nz_count += int(np.sum(np.abs(resid_vals) > 1e-8))
        total_eligible += n_valid

    n_ts = len(raw_cs_stds)
    if n_ts == 0:
        return {
            "raw_cs_std_mean": 0.0,
            "resid_cs_std_mean": 0.0,
            "variance_retention_ratio": 0.0,
            "n_timesteps": 0.0,
            "raw_nonzero_ratio": 0.0,
            "resid_nonzero_ratio": 0.0,
        }

    raw_cs_std_mean = float(np.mean(raw_cs_stds))
    resid_cs_std_mean = float(np.mean(resid_cs_stds))
    var_retention = resid_cs_std_mean / max(raw_cs_std_mean, 1e-12)
    raw_nz_ratio = raw_nz_count / max(total_eligible, 1)
    resid_nz_ratio = resid_nz_count / max(total_eligible, 1)

    return {
        "raw_cs_std_mean": raw_cs_std_mean,
        "resid_cs_std_mean": resid_cs_std_mean,
        "variance_retention_ratio": var_retention,
        "n_timesteps": float(n_ts),
        "raw_nonzero_ratio": raw_nz_ratio,
        "resid_nonzero_ratio": resid_nz_ratio,
    }


def passes_signal_preservation_gate(
    summary: dict[str, float],
    *,
    min_long_preservation_ratio: float = 0.0,
    min_short_preservation_ratio: float = 0.0,
) -> bool:
    """Check alpha->xs composition preservation from preservation-ratio metrics."""
    long_ratio = summary.get("xs_long_preservation_ratio", 0.0)
    short_ratio = summary.get("xs_short_preservation_ratio", 0.0)
    return bool(
        long_ratio >= min_long_preservation_ratio
        and short_ratio >= min_short_preservation_ratio
    )
