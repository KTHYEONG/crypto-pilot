"""Tests for Liquidity-Stress Discriminative Diagnostics (Part 1)."""

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.tiered_workflow.awf_sim import ReversalEpisode
from src.domain.futures.strategy.tiered_workflow.liquidity_stress_diag import (
    LiquidityStressDiagnostic,
    compute_liquidity_stress_discriminative_power,
)


def _make_half_spread_series(
    n_bars: int, stress_bps: float, base_bps: float = 5.0
) -> pd.Series:
    """Create a half-spread series with uniform base and optional stress override."""
    idx = pd.date_range("2026-01-01", periods=n_bars, freq="4h")
    values = np.full(n_bars, base_bps)
    return pd.Series(values, index=idx)


def _make_bar_datetimes(n_bars: int) -> np.ndarray:
    """Create bar_datetimes array aligned with half_spread series."""
    idx = pd.date_range("2026-01-01", periods=n_bars, freq="4h")
    return np.asarray(idx.to_numpy(), dtype="datetime64[ns]")


# ── Scenario 1: Happy Path ──


def test_compute_liquidity_stress_discriminative_power_reports_positive_stress_gap() -> None:
    """Episode A (true positive) has high stress, Episode B (false positive) has low stress."""
    n_bars = 500
    bar_datetimes = _make_bar_datetimes(n_bars)
    risk_off_mask = np.zeros(n_bars, dtype=bool)

    half_spread = _make_half_spread_series(n_bars, stress_bps=5.0, base_bps=5.0)

    episode_a_start, episode_a_end = 100, 120
    episode_b_start, episode_b_end = 300, 320

    half_spread.iloc[episode_a_start:episode_a_end] = 15.0
    half_spread.iloc[episode_b_start:episode_b_end] = 5.0

    episodes = (
        ReversalEpisode(start_idx=episode_a_start, end_idx=episode_a_end, realized_price=-0.02),
        ReversalEpisode(start_idx=episode_b_start, end_idx=episode_b_end, realized_price=0.01),
    )

    result = compute_liquidity_stress_discriminative_power(
        episodes=episodes,
        bar_datetimes=bar_datetimes,
        half_spread_bps=half_spread,
        risk_off_mask=risk_off_mask,
        baseline_window_bars=180,
    )

    assert isinstance(result, LiquidityStressDiagnostic)
    assert result.n_episodes == 2
    assert result.n_true_positive == 1
    assert result.n_false_positive == 1
    assert result.stress_gap > 0
    assert result.mean_stress_true_positive > result.mean_stress_false_positive
    assert result.baseline_contaminated_episode_count == 0


# ── Scenario 2: Edge Cases ──


def test_compute_liquidity_stress_discriminative_power_zero_stress_gap_when_stress_uniform() -> None:
    """Two episodes with identical stress levels should yield stress_gap ≈ 0."""
    n_bars = 500
    bar_datetimes = _make_bar_datetimes(n_bars)
    risk_off_mask = np.zeros(n_bars, dtype=bool)

    half_spread = _make_half_spread_series(n_bars, stress_bps=5.0, base_bps=5.0)

    episodes = (
        ReversalEpisode(start_idx=100, end_idx=120, realized_price=-0.02),
        ReversalEpisode(start_idx=300, end_idx=320, realized_price=0.01),
    )

    result = compute_liquidity_stress_discriminative_power(
        episodes=episodes,
        bar_datetimes=bar_datetimes,
        half_spread_bps=half_spread,
        risk_off_mask=risk_off_mask,
        baseline_window_bars=180,
    )

    assert result.stress_gap == pytest.approx(0.0, abs=1e-6)
    assert result.welch_p_value >= 0.05


def test_compute_liquidity_stress_discriminative_power_empty_episodes_returns_zeroed_diagnostic() -> None:
    """Empty episodes tuple should return zeroed diagnostic without exception."""
    n_bars = 100
    bar_datetimes = _make_bar_datetimes(n_bars)
    risk_off_mask = np.zeros(n_bars, dtype=bool)
    half_spread = _make_half_spread_series(n_bars, stress_bps=5.0, base_bps=5.0)

    result = compute_liquidity_stress_discriminative_power(
        episodes=(),
        bar_datetimes=bar_datetimes,
        half_spread_bps=half_spread,
        risk_off_mask=risk_off_mask,
        baseline_window_bars=180,
    )

    assert result.n_episodes == 0
    assert result.n_true_positive == 0
    assert result.n_false_positive == 0
    assert result.mean_stress_true_positive == 0.0
    assert result.mean_stress_false_positive == 0.0
    assert result.stress_gap == 0.0
    assert result.welch_t_stat == 0.0
    assert result.welch_p_value == 1.0
    assert result.baseline_contaminated_episode_count == 0


def test_compute_liquidity_stress_discriminative_power_single_episode_reports_stress_gap_without_pvalue() -> None:
    """Single episode should report stress_gap but p-value = 1.0 (insufficient sample)."""
    n_bars = 500
    bar_datetimes = _make_bar_datetimes(n_bars)
    risk_off_mask = np.zeros(n_bars, dtype=bool)

    half_spread = _make_half_spread_series(n_bars, stress_bps=5.0, base_bps=5.0)
    half_spread.iloc[100:120] = 15.0

    episodes = (
        ReversalEpisode(start_idx=100, end_idx=120, realized_price=-0.02),
    )

    result = compute_liquidity_stress_discriminative_power(
        episodes=episodes,
        bar_datetimes=bar_datetimes,
        half_spread_bps=half_spread,
        risk_off_mask=risk_off_mask,
        baseline_window_bars=180,
    )

    assert result.n_episodes == 1
    assert result.n_true_positive == 1
    assert result.n_false_positive == 0
    assert result.welch_t_stat == 0.0
    assert result.welch_p_value == 1.0
    assert result.stress_gap != 0.0


def test_compute_liquidity_stress_discriminative_power_flags_baseline_contaminated_episode() -> None:
    """Episode with contaminated baseline should be counted and use raw mean instead of z-score."""
    n_bars = 500
    bar_datetimes = _make_bar_datetimes(n_bars)
    risk_off_mask = np.zeros(n_bars, dtype=bool)

    risk_off_mask[50:230] = True

    half_spread = _make_half_spread_series(n_bars, stress_bps=5.0, base_bps=5.0)
    half_spread.iloc[250:270] = 15.0

    episodes = (
        ReversalEpisode(start_idx=250, end_idx=270, realized_price=-0.02),
    )

    result = compute_liquidity_stress_discriminative_power(
        episodes=episodes,
        bar_datetimes=bar_datetimes,
        half_spread_bps=half_spread,
        risk_off_mask=risk_off_mask,
        baseline_window_bars=180,
    )

    assert result.baseline_contaminated_episode_count == 1


# ── Scenario 3: Error Handling ──


def test_compute_liquidity_stress_discriminative_power_raises_on_non_monotonic_half_spread_index() -> None:
    """Non-monotonic half_spread index should raise ValueError."""
    n_bars = 100
    bar_datetimes = _make_bar_datetimes(n_bars)
    risk_off_mask = np.zeros(n_bars, dtype=bool)

    idx = pd.date_range("2026-01-01", periods=n_bars, freq="4h")
    idx_list = list(idx)
    idx_list[5], idx_list[6] = idx_list[6], idx_list[5]
    shuffled_idx = pd.DatetimeIndex(idx_list)
    half_spread = pd.Series(np.full(n_bars, 5.0), index=shuffled_idx)

    episodes = (ReversalEpisode(start_idx=10, end_idx=20, realized_price=-0.02),)

    with pytest.raises(ValueError, match="half_spread_bps index must be monotonic and unique"):
        compute_liquidity_stress_discriminative_power(
            episodes=episodes,
            bar_datetimes=bar_datetimes,
            half_spread_bps=half_spread,
            risk_off_mask=risk_off_mask,
            baseline_window_bars=180,
        )


def test_compute_liquidity_stress_discriminative_power_clips_out_of_range_episode_bounds_gracefully() -> None:
    """Episode with end_idx beyond bar_datetimes length should be clipped, not crash."""
    n_bars = 100
    bar_datetimes = _make_bar_datetimes(n_bars)
    risk_off_mask = np.zeros(n_bars, dtype=bool)
    half_spread = _make_half_spread_series(n_bars, stress_bps=5.0, base_bps=5.0)

    episodes = (
        ReversalEpisode(start_idx=80, end_idx=150, realized_price=-0.02),
    )

    result = compute_liquidity_stress_discriminative_power(
        episodes=episodes,
        bar_datetimes=bar_datetimes,
        half_spread_bps=half_spread,
        risk_off_mask=risk_off_mask,
        baseline_window_bars=180,
    )

    assert result.n_episodes == 1


def test_compute_liquidity_stress_discriminative_power_raises_on_risk_off_mask_length_mismatch() -> None:
    """Mismatched risk_off_mask length should raise ValueError."""
    n_bars = 100
    bar_datetimes = _make_bar_datetimes(n_bars)
    risk_off_mask = np.zeros(50, dtype=bool)
    half_spread = _make_half_spread_series(n_bars, stress_bps=5.0, base_bps=5.0)

    episodes = (ReversalEpisode(start_idx=10, end_idx=20, realized_price=-0.02),)

    with pytest.raises(ValueError, match="risk_off_mask length must match bar_datetimes"):
        compute_liquidity_stress_discriminative_power(
            episodes=episodes,
            bar_datetimes=bar_datetimes,
            half_spread_bps=half_spread,
            risk_off_mask=risk_off_mask,
            baseline_window_bars=180,
        )
