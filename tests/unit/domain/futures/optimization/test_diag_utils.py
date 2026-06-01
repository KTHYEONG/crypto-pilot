"""Unit tests for src/domain/futures/optimization/diag_utils.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.optimization.diag_utils import preservation_ratio


class TestPreservationRatio:
    """Tests for preservation_ratio."""

    def test_before_all_zeros_returns_zero(self) -> None:
        # Arrange
        before = np.zeros(10, dtype=np.float64)
        after = np.zeros(10, dtype=np.float64)

        # Act
        result = preservation_ratio(before, after)

        # Assert
        assert result == pytest.approx(0.0)

    def test_half_nonzero_after_returns_half(self) -> None:
        # Arrange
        before = np.ones(10, dtype=np.float64)
        after = np.array([1.0, 0.0] * 5, dtype=np.float64)

        # Act
        result = preservation_ratio(before, after)

        # Assert
        assert result == pytest.approx(0.5, rel=1e-6)

    def test_shape_mismatch_raises_value_error(self) -> None:
        # Arrange
        before = np.ones(5, dtype=np.float64)
        after = np.ones(6, dtype=np.float64)

        # Act / Assert
        with pytest.raises(ValueError, match="same shape"):
            preservation_ratio(before, after)

    def test_before_and_after_identical_returns_one(self) -> None:
        # Arrange
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        # Act
        result = preservation_ratio(arr, arr.copy())

        # Assert
        assert result == pytest.approx(1.0, rel=1e-6)
