from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.legacy.strategy_sleev.combine import blend_sleeves


def test_blend_sleeves_basic() -> None:
    # 5 steps, 6 assets
    z1 = np.ones((5, 6)) * 1.0
    z2 = np.ones((5, 6)) * 2.0

    z_by_sleeve = {"s1": z1, "s2": z2}
    ic_weights = {"s1": 0.04, "s2": 0.08}

    # Weight sum = 0.12. Normalized w1 = 1/3, w2 = 2/3.
    # Blended before re-standardization = 1/3 * 1.0 + 2/3 * 2.0 = 1.6667
    # Since all assets in each row have the exact same value (constant row),
    # the robust z-score re-standardization will return all zeros.
    res = blend_sleeves(z_by_sleeve, ic_weights, min_symbols=5)

    assert res.shape == (5, 6)
    assert np.all(res == 0.0)


def test_blend_sleeves_fallback_equal_weight() -> None:
    # Use distinct values so that z-score is non-zero
    z1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
    z2 = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]], dtype=np.float64)

    z_by_sleeve = {"s1": z1, "s2": z2}
    ic_weights = {"s1": -0.02, "s2": 0.0}  # all <= 0

    # Equal weights fallback -> w1 = 0.5, w2 = 0.5
    # Blended row: [3.0, 3.0, 3.0, 3.0, 3.0]
    # Re-standardized: all identical -> all 0s.
    res = blend_sleeves(z_by_sleeve, ic_weights, min_symbols=5)
    assert np.all(res == 0.0)


def test_blend_sleeves_non_identical_z() -> None:
    # Test with varying values to make sure z-score returns non-zero when blended
    # 1 step, 6 assets
    z1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float64)
    z2 = np.array([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]], dtype=np.float64)

    z_by_sleeve = {"s1": z1, "s2": z2}
    ic_weights = {"s1": 0.1, "s2": 0.0}  # w1 = 1.0, w2 = 0.0

    res = blend_sleeves(z_by_sleeve, ic_weights, min_symbols=5)
    # Since w1=1.0 and w2=0.0, result should be re-standardized z1.
    from src.domain.futures.legacy.strategy_sleev.normalize import winsorized_cs_zscore

    expected = winsorized_cs_zscore(z1, min_symbols=5)
    assert np.allclose(res, expected)


def test_blend_sleeves_mismatch_shape() -> None:
    z1 = np.ones((5, 6))
    z2 = np.ones((5, 5))
    with pytest.raises(ValueError, match="does not match"):
        blend_sleeves({"s1": z1, "s2": z2}, {"s1": 0.1, "s2": 0.1})
