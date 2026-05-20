"""Tests for No-Trade Buffer portfolio buffer filter."""

from __future__ import annotations

import numpy as np

from src.domain.futures.portfolio.risk_controls import apply_no_trade_buffer


def test_no_trade_buffer_below_threshold() -> None:
    """Test that tiny trades below threshold are filtered out."""
    target_weights = np.array([0.051, 0.03])
    current_weights = np.array([0.050, 0.03])
    cost_bps_per_symbol = np.array([10.0, 10.0])  # threshold = 2 * 10 = 20 bps

    # delta_w for symbol 0 = 0.001 (10 bps) < 20 bps -> kept at 0.050
    # delta_w for symbol 1 = 0.0 (0 bps) < 20 bps -> kept at 0.03
    adjusted = apply_no_trade_buffer(
        target_weights=target_weights,
        current_weights=current_weights,
        cost_bps_per_symbol=cost_bps_per_symbol,
        threshold_multiplier=2.0,
    )

    np.testing.assert_array_equal(adjusted, np.array([0.050, 0.03]))


def test_no_trade_buffer_above_threshold() -> None:
    """Test that trades above threshold are executed normally."""
    target_weights = np.array([0.055, 0.03])
    current_weights = np.array([0.050, 0.03])
    cost_bps_per_symbol = np.array([10.0, 10.0])  # threshold = 2 * 10 = 20 bps

    # delta_w for symbol 0 = 0.005 (50 bps) > 20 bps -> executed
    adjusted = apply_no_trade_buffer(
        target_weights=target_weights,
        current_weights=current_weights,
        cost_bps_per_symbol=cost_bps_per_symbol,
        threshold_multiplier=2.0,
    )

    np.testing.assert_array_equal(adjusted, np.array([0.055, 0.03]))


def test_no_trade_buffer_mixed_case() -> None:
    """Test mixed case where only some assets are filtered."""
    target_weights = np.array([0.055, 0.031, 0.010])
    current_weights = np.array([0.050, 0.030, 0.010])
    cost_bps_per_symbol = np.array([10.0, 10.0, 5.0])

    # Thres: 20 bps, 20 bps, 10 bps
    # Deltas: 50 bps, 10 bps, 0 bps
    # Expect: 0.055 (executed), 0.030 (filtered), 0.010 (filtered)
    adjusted = apply_no_trade_buffer(
        target_weights=target_weights,
        current_weights=current_weights,
        cost_bps_per_symbol=cost_bps_per_symbol,
        threshold_multiplier=2.0,
    )

    np.testing.assert_array_equal(adjusted, np.array([0.055, 0.030, 0.010]))


def test_no_trade_buffer_disabled() -> None:
    """Test that all trades are allowed when threshold multiplier is <= 0."""
    target_weights = np.array([0.051, 0.031])
    current_weights = np.array([0.050, 0.030])
    cost_bps_per_symbol = np.array([10.0, 10.0])

    adjusted = apply_no_trade_buffer(
        target_weights=target_weights,
        current_weights=current_weights,
        cost_bps_per_symbol=cost_bps_per_symbol,
        threshold_multiplier=0.0,
    )

    np.testing.assert_array_equal(adjusted, np.array([0.051, 0.031]))
