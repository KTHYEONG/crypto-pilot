from __future__ import annotations

import numpy as np
import scipy.stats


def rolling_ic(
    sig_2d: np.ndarray,
    fwd_ret_2d: np.ndarray,
    *,
    method: str = "spearman",
) -> np.ndarray:
    """Computes cross-sectional Information Coefficient (IC) chronologically for each bar.

    Args:
        sig_2d: Signal panel of shape [T, N].
        fwd_ret_2d: Forward 1-bar returns panel of shape [T, N]. (Typically close[t+1]/close[t] - 1).
        method: Correlation method, either 'spearman' or 'pearson'.

    Returns:
        [T] correlation values. Warmup or small sample bars are NaN.

    """
    if sig_2d.shape != fwd_ret_2d.shape:
        raise ValueError("sig_2d and fwd_ret_2d must have the same shape")

    t_len, n_syms = sig_2d.shape
    ic = np.full(t_len, np.nan, dtype=np.float64)

    for t in range(t_len):
        row_sig = sig_2d[t]
        row_ret = fwd_ret_2d[t]
        m = np.isfinite(row_sig) & np.isfinite(row_ret)
        n_valid = int(m.sum())

        if n_valid < 5:  # Minimum sample size for meaningful cross-sectional correlation
            continue

        s_vals = row_sig[m]
        r_vals = row_ret[m]

        if np.all(s_vals == s_vals[0]) or np.all(r_vals == r_vals[0]):
            ic[t] = 0.0
            continue

        try:
            if method == "spearman":
                res = scipy.stats.spearmanr(s_vals, r_vals)
                stat = float(res.statistic)
            elif method == "pearson":
                res = scipy.stats.pearsonr(s_vals, r_vals)
                stat = float(res.statistic)
            else:
                raise ValueError(f"Unknown correlation method: {method}")

            ic[t] = stat if np.isfinite(stat) else 0.0
        except Exception:
            ic[t] = 0.0

    return ic


def ic_summary(ic_series: np.ndarray) -> dict[str, float]:
    """Calculates summary statistics of the IC series.

    Args:
        ic_series: Information coefficient array of shape [T] (may contain NaNs).

    Returns:
        Dictionary containing:
        - 'mean_ic': Mean of valid ICs
        - 'ic_std': Standard deviation of valid ICs
        - 'icir': Information Ratio (mean_ic / ic_std)
        - 't_stat': T-statistic of the mean IC (mean_ic / standard_error)
        - 'n_obs': Count of valid ICs
        - 'hit_ratio': Ratio of positive valid ICs

    """
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
        se = ic_std / np.sqrt(n_obs)
        t_stat = mean_ic / se

    pos_count = np.sum(valid > 0.0)
    hit_ratio = float(pos_count) / n_obs

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
    """Evaluates whether the sleeve passes the selection gate.

    Args:
        summary: The dictionary returned by `ic_summary`.
        min_mean_ic: Minimum mean IC hurdle.
        min_t_stat: Minimum T-statistic hurdle.
        min_hit_ratio: Minimum hit ratio hurdle.

    Returns:
        True if all conditions are satisfied, else False.

    """
    mean_ic_ok = summary["mean_ic"] >= min_mean_ic
    t_stat_ok = summary["t_stat"] >= min_t_stat
    hit_ratio_ok = summary["hit_ratio"] >= min_hit_ratio
    return bool(mean_ic_ok and t_stat_ok and hit_ratio_ok)
