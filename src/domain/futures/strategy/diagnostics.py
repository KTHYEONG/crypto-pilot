from __future__ import annotations

import numpy as np
import scipy.stats


def rolling_ic(
    sig_2d: np.ndarray,
    fwd_ret_2d: np.ndarray,
    *,
    method: str = "spearman",
) -> np.ndarray:
    """Compute cross-sectional IC over time."""
    if sig_2d.shape != fwd_ret_2d.shape:
        raise ValueError("sig_2d and fwd_ret_2d must have the same shape")
    t_len, _n_syms = sig_2d.shape
    ic = np.full(t_len, np.nan, dtype=np.float64)
    for t in range(t_len):
        row_sig = sig_2d[t]
        row_ret = fwd_ret_2d[t]
        m = np.isfinite(row_sig) & np.isfinite(row_ret)
        if int(m.sum()) < 5:
            continue
        s_vals = row_sig[m]
        r_vals = row_ret[m]
        if np.all(s_vals == s_vals[0]) or np.all(r_vals == r_vals[0]):
            ic[t] = 0.0
            continue
        if method == "spearman":
            stat = float(scipy.stats.spearmanr(s_vals, r_vals).statistic)
        elif method == "pearson":
            stat = float(scipy.stats.pearsonr(s_vals, r_vals).statistic)
        else:
            raise ValueError(f"Unknown method: {method}")
        ic[t] = stat if np.isfinite(stat) else 0.0
    return ic


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


def passes_ic_gate(
    summary: dict[str, float],
    *,
    min_mean_ic: float = 0.02,
    min_t_stat: float = 2.0,
    min_hit_ratio: float = 0.45,
) -> bool:
    """Return True when IC gate thresholds are satisfied."""
    return bool(
        summary["mean_ic"] >= min_mean_ic
        and summary["t_stat"] >= min_t_stat
        and summary["hit_ratio"] >= min_hit_ratio
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
        dcg = np.sum((np.power(2.0, r[ord_pred]) - 1.0) / log_denom)
        idcg = np.sum((np.power(2.0, r[ord_best]) - 1.0) / log_denom)
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
) -> dict[str, float]:
    """Build structured diagnostics report used for quality gates."""
    ic_series = rolling_ic(score_2d, signed_ret_2d, method="spearman")
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
