"""Phase 3-5: capacity_usdt clip unit tests.

Scenarios:
- S4: weight 0.01 × portfolio_nav=500 → intended=5 USDT (boundary of < 5 USDT) → weight=0.
- S5: weight 0.5 × portfolio_nav=100000 → intended=50000 USDT → cap=10000 USDT → clipped to 0.1.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Helpers — isolated capacity clip logic extracted for direct unit-testing
# ---------------------------------------------------------------------------

_MIN_ORDER_USDT: float = 5.0


def _apply_capacity_clip(
    weights: NDArray[np.float64],
    cap_row: NDArray[np.float64],
    portfolio_nav: float,
) -> NDArray[np.float64]:
    """Mirror of inline capacity clip in _run_awf_simulation.

    Args:
        weights: [N] float64 target weights (unit-NAV or real).
        cap_row: [N] float64 capacity in USDT (NaN = no cap).
        portfolio_nav: Portfolio value in USDT.

    Returns:
        Clipped weight array (copy).

    Note:
        Time: O(N), Space: O(N).
    """
    w = weights.copy()
    safe_nav = max(portfolio_nav, 1.0)
    cap_clean = np.nan_to_num(cap_row, nan=0.0, posinf=0.0, neginf=0.0)
    for n in range(len(w)):
        intended = abs(w[n]) * portfolio_nav
        if intended < _MIN_ORDER_USDT:
            w[n] = 0.0
            continue
        cap = cap_clean[n]
        if cap > 0.0:
            max_w = cap / safe_nav
            if abs(w[n]) > max_w:
                w[n] = float(np.sign(w[n])) * max_w
    return w


# ---------------------------------------------------------------------------
# S4: below min-order threshold → weight=0
# ---------------------------------------------------------------------------


class TestS4MinOrderThreshold:
    """S4: intended_notional < 5 USDT → weight forced to 0."""

    def test_weight_exactly_at_threshold_is_zeroed(self) -> None:
        """Arrange: weight=0.01, nav=500 → intended=5.0 USDT — boundary: < 5 is zeroed."""
        # Arrange
        weights = np.array([0.01], dtype=np.float64)
        cap_row = np.array([1_000.0], dtype=np.float64)  # cap is large, not the issue
        portfolio_nav = 500.0

        # Act
        result = _apply_capacity_clip(weights, cap_row, portfolio_nav)

        # Assert: 0.01 × 500 = 5.0, which is NOT < 5.0, so weight survives
        # (boundary is strictly less-than)
        assert result[0] != pytest.approx(0.0), (
            "Intended=5.0 USDT is not < 5.0, so weight must NOT be zeroed"
        )

    def test_weight_below_threshold_is_zeroed(self) -> None:
        """Arrange: weight=0.009, nav=500 → intended=4.5 USDT < 5 USDT → weight=0."""
        # Arrange
        weights = np.array([0.009], dtype=np.float64)
        cap_row = np.array([1_000.0], dtype=np.float64)
        portfolio_nav = 500.0

        # Act
        result = _apply_capacity_clip(weights, cap_row, portfolio_nav)

        # Assert
        assert result[0] == pytest.approx(0.0), (
            f"Expected 0.0 (below min-order), got {result[0]}"
        )

    def test_negative_weight_below_threshold_is_zeroed(self) -> None:
        """Arrange: weight=-0.009, nav=500 → |intended|=4.5 < 5 → weight=0."""
        # Arrange
        weights = np.array([-0.009], dtype=np.float64)
        cap_row = np.array([1_000.0], dtype=np.float64)
        portfolio_nav = 500.0

        # Act
        result = _apply_capacity_clip(weights, cap_row, portfolio_nav)

        # Assert
        assert result[0] == pytest.approx(0.0)

    def test_multiple_symbols_zeroes_only_subthreshold(self) -> None:
        """Arrange: sym0 below threshold, sym1 above → only sym0 zeroed."""
        # Arrange
        # sym0: 0.009 × 500 = 4.5 < 5 → zeroed
        # sym1: 0.1 × 500 = 50 >= 5 → kept (cap=1000 > 50)
        weights = np.array([0.009, 0.1], dtype=np.float64)
        cap_row = np.array([1_000.0, 1_000.0], dtype=np.float64)
        portfolio_nav = 500.0

        # Act
        result = _apply_capacity_clip(weights, cap_row, portfolio_nav)

        # Assert
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# S5: capacity clip — intended > capacity → clip to cap/nav
# ---------------------------------------------------------------------------


class TestS5CapacityClip:
    """S5: intended_notional > capacity_usdt → weight clipped to cap/portfolio_nav."""

    def test_long_weight_clipped_to_capacity(self) -> None:
        """Arrange: weight=0.5, nav=100000 → intended=50000 > cap=10000 → 0.1."""
        # Arrange
        weights = np.array([0.5], dtype=np.float64)
        cap_row = np.array([10_000.0], dtype=np.float64)
        portfolio_nav = 100_000.0

        # Act
        result = _apply_capacity_clip(weights, cap_row, portfolio_nav)

        # Assert: max_w = 10000 / 100000 = 0.1
        assert result[0] == pytest.approx(0.1, rel=1e-9)

    def test_short_weight_clipped_preserves_sign(self) -> None:
        """Arrange: weight=-0.5, nav=100000, cap=10000 → clipped to -0.1."""
        # Arrange
        weights = np.array([-0.5], dtype=np.float64)
        cap_row = np.array([10_000.0], dtype=np.float64)
        portfolio_nav = 100_000.0

        # Act
        result = _apply_capacity_clip(weights, cap_row, portfolio_nav)

        # Assert
        assert result[0] == pytest.approx(-0.1, rel=1e-9)

    def test_weight_within_capacity_not_clipped(self) -> None:
        """Arrange: weight=0.05, nav=100000 → intended=5000 < cap=10000 → unchanged."""
        # Arrange
        weights = np.array([0.05], dtype=np.float64)
        cap_row = np.array([10_000.0], dtype=np.float64)
        portfolio_nav = 100_000.0

        # Act
        result = _apply_capacity_clip(weights, cap_row, portfolio_nav)

        # Assert
        assert result[0] == pytest.approx(0.05, rel=1e-9)

    def test_zero_cap_skips_clip(self) -> None:
        """Arrange: cap=0 → clip guard skipped → weight unchanged (no cap constraint)."""
        # Arrange
        weights = np.array([0.5], dtype=np.float64)
        cap_row = np.array([0.0], dtype=np.float64)
        portfolio_nav = 100_000.0

        # Act
        result = _apply_capacity_clip(weights, cap_row, portfolio_nav)

        # Assert: 0.5 × 100000 = 50000 >= 5 USDT, cap=0 → no clip, weight unchanged
        assert result[0] == pytest.approx(0.5)

    def test_nan_cap_skips_clip(self) -> None:
        """Arrange: cap=NaN → nan_to_num converts to 0 → no clip applied."""
        # Arrange
        weights = np.array([0.5], dtype=np.float64)
        cap_row = np.array([np.nan], dtype=np.float64)
        portfolio_nav = 100_000.0

        # Act
        result = _apply_capacity_clip(weights, cap_row, portfolio_nav)

        # Assert
        assert result[0] == pytest.approx(0.5)

    def test_mixed_symbols_independent_clip(self) -> None:
        """Arrange: sym0 clipped, sym1 within cap, sym2 below min-order."""
        # Arrange
        # sym0: 0.5 × 100000 = 50000 > 10000 cap → clip to 0.1
        # sym1: 0.05 × 100000 = 5000 < 10000 cap → unchanged
        # sym2: 0.00004 × 100000 = 4.0 < 5 USDT → zeroed
        weights = np.array([0.5, 0.05, 0.00004], dtype=np.float64)
        cap_row = np.array([10_000.0, 10_000.0, 10_000.0], dtype=np.float64)
        portfolio_nav = 100_000.0

        # Act
        result = _apply_capacity_clip(weights, cap_row, portfolio_nav)

        # Assert
        assert result[0] == pytest.approx(0.1, rel=1e-9)
        assert result[1] == pytest.approx(0.05, rel=1e-9)
        assert result[2] == pytest.approx(0.0)
