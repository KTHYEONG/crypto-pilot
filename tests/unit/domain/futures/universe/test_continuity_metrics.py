"""Unit tests for compute_continuity_metrics in storage.py.

Scenarios (S1-S6) per spec docs/specs/universe-redesign-l1-ready.md - Test Scenario Design.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.universe.storage import compute_continuity_metrics

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_grid(
    start: date,
    end: date,
    freq: str = "4h",
) -> pd.DatetimeIndex:
    """Build a UTC 4h date_range for test scaffolding."""
    return pd.date_range(
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        freq=freq,
    )


def _grid_to_df(
    index: pd.DatetimeIndex,
    *,
    close_val: float = 100.0,
    volume_val: float = 1_000.0,
) -> pd.DataFrame:
    """Convert DatetimeIndex into minimal OHLCV kline DataFrame."""
    n = len(index)
    return pd.DataFrame(
        {
            "datetime": index,
            "open": np.full(n, close_val, dtype=np.float64),
            "high": np.full(n, close_val * 1.01, dtype=np.float64),
            "low": np.full(n, close_val * 0.99, dtype=np.float64),
            "close": np.full(n, close_val, dtype=np.float64),
            "volume": np.full(n, volume_val, dtype=np.float64),
            "quote_vol": np.full(n, volume_val * close_val, dtype=np.float64),
        }
    )


# ── S1: Happy path — perfect grid ────────────────────────────────────────────

def test_compute_continuity_metrics_s1_perfect_grid_no_gaps() -> None:
    # Arrange: 30-day perfect 4h grid with varying close to avoid frozen-bar false positive
    start = date(2024, 1, 1)
    end = date(2024, 1, 30)
    grid = _make_grid(start, end)
    klines = _grid_to_df(grid)
    klines = klines.copy()
    klines["close"] = np.linspace(100.0, 110.0, len(klines))

    # Act
    result = compute_continuity_metrics(klines, onboard_date=start, as_of_date=end, tf="4h")

    # Assert
    assert result["n_bar_gaps"] == 0
    assert result["max_gap_bars"] == 0
    assert result["coverage_ratio"] == pytest.approx(1.0, rel=1e-3)
    assert result["frozen_bars"] == 0
    assert result["has_nan"] is False
    assert result["has_inf"] is False
    assert result["has_timestamp_issues"] is False


# ── S2: 7-bar continuous gap in middle ───────────────────────────────────────

def test_compute_continuity_metrics_s2_single_7bar_gap() -> None:
    # Arrange: 30-day grid with 7 consecutive bars removed from the middle
    start = date(2024, 1, 1)
    end = date(2024, 1, 30)
    grid = _make_grid(start, end)
    # Remove bars at positions 20-26 (7 consecutive)
    mask = np.ones(len(grid), dtype=bool)
    mask[20:27] = False
    observed_grid = grid[mask]
    klines = _grid_to_df(observed_grid)

    # Act
    result = compute_continuity_metrics(klines, onboard_date=start, as_of_date=end, tf="4h")

    # Assert
    assert result["n_bar_gaps"] == 1, "exactly one contiguous gap run expected"
    assert result["max_gap_bars"] == 7, "longest gap should be exactly 7 bars"
    # coverage = (total - 7) / total
    expected_coverage = (len(grid) - 7) / len(grid)
    assert result["coverage_ratio"] == pytest.approx(expected_coverage, rel=1e-3)


# ── S3: PIT denominator — mid-window onboarding ──────────────────────────────

def test_compute_continuity_metrics_s3_pit_denominator_mid_window_onboard() -> None:
    """Symbol onboarded mid-window must not be penalised for pre-onboard absence.

    Spec invariant: coverage denominator = bars from onboard_date to as_of_date only.
    Coverage must be ≈1.0 if all post-onboard bars are present (no survival bias).
    """
    # Arrange: ledger window is 60 days but symbol onboarded after 30 days
    onboard = date(2024, 2, 1)   # mid-window (day 31 from 2024-01-01)
    end = date(2024, 3, 1)

    # Observe only post-onboard bars (perfect coverage from onboard onward)
    post_onboard_grid = _make_grid(onboard, end)
    klines = _grid_to_df(post_onboard_grid)

    # Act — pass onboard_date (not full_start) as PIT anchor
    result = compute_continuity_metrics(
        klines, onboard_date=onboard, as_of_date=end, tf="4h"
    )

    # Assert: coverage must be ≈1.0 (no pre-onboard penalty)
    assert result["coverage_ratio"] == pytest.approx(1.0, rel=1e-2), (
        "PIT-safe denominator must be post-onboard bars only — "
        "survival bias would show coverage < 1.0 if full window used"
    )
    assert result["n_bar_gaps"] == 0


# ── S4: Frozen bars — 8 consecutive identical close ──────────────────────────

def test_compute_continuity_metrics_s4_frozen_close_run_detected() -> None:
    # Arrange: 60-bar grid; bars 30-37 (8 bars) have identical close
    start = date(2024, 1, 1)
    end = date(2024, 1, 15)
    grid = _make_grid(start, end)
    klines = _grid_to_df(grid, close_val=200.0)

    # Freeze bars 30-37
    klines = klines.copy()
    klines.loc[30:37, "close"] = 199.0  # same value for 8 consecutive bars

    # Act
    result = compute_continuity_metrics(klines, onboard_date=start, as_of_date=end, tf="4h")

    # Assert
    assert result["frozen_bars"] >= 8, (
        f"Expected frozen_bars >= 8, got {result['frozen_bars']}"
    )


# ── S5: NaN in OHLCV → has_nan=True ─────────────────────────────────────────

def test_compute_continuity_metrics_s5_nan_in_ohlcv_detected() -> None:
    # Arrange: perfect grid with one NaN in 'high' column
    start = date(2024, 1, 1)
    end = date(2024, 1, 10)
    grid = _make_grid(start, end)
    klines = _grid_to_df(grid)
    klines = klines.copy()
    klines.loc[5, "high"] = float("nan")

    # Act
    result = compute_continuity_metrics(klines, onboard_date=start, as_of_date=end, tf="4h")

    # Assert
    assert result["has_nan"] is True
    # No gap — all timestamps present
    assert result["n_bar_gaps"] == 0


# ── S6: Duplicate timestamp → has_timestamp_issues=True ──────────────────────

def test_compute_continuity_metrics_s6_duplicate_timestamp_detected() -> None:
    # Arrange: 10-day grid with one duplicated timestamp
    start = date(2024, 1, 1)
    end = date(2024, 1, 10)
    grid = _make_grid(start, end)
    klines = _grid_to_df(grid)

    # Duplicate row at index 3
    dup_row = klines.iloc[[3]].copy()
    klines = pd.concat([klines.iloc[:5], dup_row, klines.iloc[5:]], ignore_index=True)

    # Act
    result = compute_continuity_metrics(klines, onboard_date=start, as_of_date=end, tf="4h")

    # Assert
    assert result["has_timestamp_issues"] is True


# ── Edge: empty DataFrame returns safe defaults ───────────────────────────────

def test_compute_continuity_metrics_empty_dataframe_returns_safe_defaults() -> None:
    # Arrange
    klines = pd.DataFrame(
        columns=["datetime", "open", "high", "low", "close", "volume", "quote_vol"]
    )
    start = date(2024, 1, 1)
    end = date(2024, 1, 10)

    # Act
    result = compute_continuity_metrics(klines, onboard_date=start, as_of_date=end, tf="4h")

    # Assert — all safe/zero defaults
    assert result["n_bar_gaps"] == 0
    assert result["max_gap_bars"] == 0
    assert result["coverage_ratio"] == pytest.approx(0.0)
    assert result["frozen_bars"] == 0
    assert result["has_nan"] is False
    assert result["has_inf"] is False
    assert result["has_timestamp_issues"] is False


# ── Edge: has_inf detection ────────────────────────────────────────────────────

def test_compute_continuity_metrics_inf_in_close_detected() -> None:
    # Arrange
    start = date(2024, 1, 1)
    end = date(2024, 1, 5)
    grid = _make_grid(start, end)
    klines = _grid_to_df(grid)
    klines = klines.copy()
    klines.loc[2, "close"] = float("inf")

    # Act
    result = compute_continuity_metrics(klines, onboard_date=start, as_of_date=end, tf="4h")

    # Assert
    assert result["has_inf"] is True


# ── Edge: zero-volume bars in 60d window ──────────────────────────────────────

def test_compute_continuity_metrics_zero_volume_bars_60d_counted() -> None:
    # Arrange: 70-day grid; last 30 days have 3 zero-volume bars
    start = date(2024, 1, 1)
    end = date(2024, 3, 10)  # ~70 days
    grid = _make_grid(start, end)
    klines = _grid_to_df(grid, volume_val=1000.0)

    # Set last 3 bars to zero quote_vol
    klines = klines.copy()
    klines.loc[len(klines) - 3 :, "quote_vol"] = 0.0

    # Act
    result = compute_continuity_metrics(klines, onboard_date=start, as_of_date=end, tf="4h")

    # Assert
    assert result["n_zero_volume_bars_60d"] == 3


def test_compute_continuity_metrics_zero_volume_bars_60d_falls_back_to_volume_when_quote_vol_nan() -> None:
    """Regression: quote_vol column present but NaN for recent rows (live-API gap) must
    fall back to the volume column instead of being misread as zero trading volume.
    """
    # Arrange: 70-day grid; last 30 days have quote_vol=NaN but volume is populated
    start = date(2024, 1, 1)
    end = date(2024, 3, 10)  # ~70 days
    grid = _make_grid(start, end)
    klines = _grid_to_df(grid, volume_val=1000.0)

    klines = klines.copy()
    klines.loc[len(klines) - 180 :, "quote_vol"] = np.nan

    # Act
    result = compute_continuity_metrics(klines, onboard_date=start, as_of_date=end, tf="4h")

    # Assert: volume fallback keeps these bars counted as non-zero
    assert result["n_zero_volume_bars_60d"] == 0
