"""Liquidity-Stress Discriminative Diagnostics for Reversal-Kill Episodes.

Measure-first infrastructure: quantifies whether half-spread (order-book liquidity stress)
can discriminate between true-positive and false-positive reversal-kill episodes.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from src.domain.futures.strategy.tiered_workflow.awf_sim import ReversalEpisode


@dataclass(slots=True, frozen=True)
class LiquidityStressDiagnostic:
    """Half-spread discriminative power measurement for reversal-kill episodes.

    Attributes:
        n_episodes: Total number of episodes analyzed.
        n_true_positive: Episodes where realized_price < 0 (genuine defense).
        n_false_positive: Episodes where realized_price >= 0 (whipsaw).
        mean_stress_true_positive: Mean half-spread z-score during true-positive episodes.
        mean_stress_false_positive: Mean half-spread z-score during false-positive episodes.
        stress_gap: mean_tp - mean_fp (raw effect size direction).
        welch_t_stat: Welch t-test statistic (reference only).
        welch_p_value: Welch t-test p-value (reference only, NOT a gate).
        baseline_contaminated_episode_count: Episodes with insufficient calm baseline.
    """

    n_episodes: int
    n_true_positive: int
    n_false_positive: int
    mean_stress_true_positive: float
    mean_stress_false_positive: float
    stress_gap: float
    welch_t_stat: float
    welch_p_value: float
    baseline_contaminated_episode_count: int


def compute_liquidity_stress_discriminative_power(
    episodes: tuple[ReversalEpisode, ...],
    bar_datetimes: NDArray[np.datetime64],
    half_spread_bps: pd.Series,
    risk_off_mask: NDArray[np.bool_],
    baseline_window_bars: int = 180,
) -> LiquidityStressDiagnostic:
    """Compute half-spread discriminative power for reversal-kill episodes.

    Args:
        episodes: Tuple of ReversalEpisode from fold attribution.
        bar_datetimes: Per-bar datetime array aligned with risk_off_mask.
        half_spread_bps: Half-spread time series with datetime index.
        risk_off_mask: Per-bar risk-off boolean array.
        baseline_window_bars: Number of calm bars for z-score baseline.

    Returns:
        LiquidityStressDiagnostic with discriminative power metrics.

    Raises:
        ValueError: If risk_off_mask length mismatches bar_datetimes.
        ValueError: If half_spread_bps index is not monotonic/unique.
    """
    if risk_off_mask.shape[0] != bar_datetimes.shape[0]:
        raise ValueError("risk_off_mask length must match bar_datetimes")

    if not half_spread_bps.index.is_monotonic_increasing or not half_spread_bps.index.is_unique:
        raise ValueError("half_spread_bps index must be monotonic and unique")

    if not episodes:
        return LiquidityStressDiagnostic(
            n_episodes=0,
            n_true_positive=0,
            n_false_positive=0,
            mean_stress_true_positive=0.0,
            mean_stress_false_positive=0.0,
            stress_gap=0.0,
            welch_t_stat=0.0,
            welch_p_value=1.0,
            baseline_contaminated_episode_count=0,
        )

    calm_mask = ~risk_off_mask
    calm_half_spread = half_spread_bps.loc[bar_datetimes[calm_mask]]

    contaminated_count = 0
    tp_scores: list[float] = []
    fp_scores: list[float] = []

    for episode in episodes:
        start_idx = max(0, episode.start_idx)
        end_idx = min(episode.end_idx, bar_datetimes.shape[0])
        if start_idx >= end_idx:
            continue

        start_ts = bar_datetimes[start_idx]
        end_ts = bar_datetimes[end_idx - 1]

        baseline = calm_half_spread.loc[:start_ts].tail(baseline_window_bars)

        if len(baseline) < baseline_window_bars // 2:
            contaminated_count += 1
        else:
            mu = float(baseline.mean())
            sigma = float(baseline.std())
            episode_window = half_spread_bps.loc[start_ts:end_ts]
            episode_mean = float(episode_window.mean()) if len(episode_window) > 0 else 0.0
            z = (episode_mean - mu) / (sigma + 1e-12)

            if episode.realized_price < 0:
                tp_scores.append(z)
            else:
                fp_scores.append(z)

    n_true_positive = len(tp_scores)
    n_false_positive = len(fp_scores)

    mean_tp = float(np.mean(tp_scores)) if tp_scores else 0.0
    mean_fp = float(np.mean(fp_scores)) if fp_scores else 0.0
    stress_gap = mean_tp - mean_fp

    if len(tp_scores) >= 2 and len(fp_scores) >= 2:
        t_stat, p_value = stats.ttest_ind(tp_scores, fp_scores, equal_var=False)
        welch_t_stat = float(t_stat) if np.isfinite(t_stat) else 0.0
        welch_p_value = float(p_value) if np.isfinite(p_value) else 1.0
    else:
        welch_t_stat = 0.0
        welch_p_value = 1.0

    return LiquidityStressDiagnostic(
        n_episodes=len(episodes),
        n_true_positive=n_true_positive,
        n_false_positive=n_false_positive,
        mean_stress_true_positive=mean_tp,
        mean_stress_false_positive=mean_fp,
        stress_gap=stress_gap,
        welch_t_stat=welch_t_stat,
        welch_p_value=welch_p_value,
        baseline_contaminated_episode_count=contaminated_count,
    )
