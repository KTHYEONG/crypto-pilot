from __future__ import annotations

import logging

import numpy as np
import scipy.stats

_logger = logging.getLogger(__name__)


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
    ic_values: list[float] = [float("nan")] * t_len
    for t in range(t_len):
        row_sig = sig_2d[t]
        row_ret = fwd_ret_2d[t]
        m = np.isfinite(row_sig) & np.isfinite(row_ret)
        if int(m.sum()) < 5:
            continue
        s_vals = row_sig[m]
        r_vals = row_ret[m]
        if np.all(s_vals == s_vals[0]) or np.all(r_vals == r_vals[0]):
            ic_values[t] = 0.0
            continue
        if method == "spearman":
            stat = float(scipy.stats.spearmanr(s_vals, r_vals).statistic)
        elif method == "pearson":
            stat = float(scipy.stats.pearsonr(s_vals, r_vals).statistic)
        else:
            raise ValueError(f"Unknown method: {method}")
        ic_values[t] = stat if np.isfinite(stat) else 0.0
    return np.array(ic_values, dtype=np.float64)


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
) -> dict[str, object]:
    """Evaluate alpha viability gate and expose fail reasons."""
    floor_bps = float(friction_bps + hurdle_bps)
    fail_reasons: list[str] = []
    if alpha_p95_bps < (floor_bps - max(0.0, float(cost_wall_tolerance_bps))):
        fail_reasons.append("alpha_p95_below_cost_wall")
    if long_nz < min_long_nz:
        fail_reasons.append("long_nz_below_threshold")
    if short_nz < min_short_nz:
        fail_reasons.append("short_nz_below_threshold")
    if xs_long_preservation_ratio < min_xs_preservation:
        fail_reasons.append("xs_long_preservation_below_threshold")
    if xs_short_preservation_ratio < min_xs_preservation:
        fail_reasons.append("xs_short_preservation_below_threshold")
    return {
        "alpha_gate_pass": len(fail_reasons) == 0,
        "alpha_gate_fail_reasons": fail_reasons,
        "alpha_gate_floor_bps": floor_bps,
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
