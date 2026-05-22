from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.normalize import to_return_units, winsorized_cs_zscore


def test_winsorized_cs_zscore_basic() -> None:
    # 10 steps, 6 assets
    np.random.seed(42)
    sig = np.random.normal(0, 1, (10, 6))
    sig[0, 0] = np.nan  # NaN in first row
    sig[1] = 5.0  # constant row (should return all 0s)
    sig[2, :4] = np.nan  # only 2 valid symbols (below min_symbols=5)

    z = winsorized_cs_zscore(sig, clip_z=3.0, min_symbols=5)

    assert z.shape == (10, 6)
    # Check NaN handling
    assert z[0, 0] == 0.0
    assert np.all(z[1] == 0.0)
    assert np.all(z[2] == 0.0)

    # Check winsorization clip
    for t in range(10):
        assert np.all(z[t] <= 3.0)
        assert np.all(z[t] >= -3.0)


def test_winsorized_cs_zscore_under_min_symbols() -> None:
    sig = np.ones((5, 4))  # 4 assets, min_symbols=5
    z = winsorized_cs_zscore(sig, min_symbols=5)
    assert np.all(z == 0.0)


def test_to_return_units_scalar_ic() -> None:
    z_2d = np.ones((5, 3))
    sigma_fwd_2d = np.full((5, 3), 0.02)
    ic_lagged = 0.05

    alpha_hat = to_return_units(z_2d, sigma_fwd_2d, ic_lagged)

    expected = 0.05 * 0.02 * 1.0
    assert np.allclose(alpha_hat, expected)


def test_to_return_units_array_ic() -> None:
    z_2d = np.ones((5, 2))
    sigma_fwd_2d = np.full((5, 2), 0.02)
    ic_lagged = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

    alpha_hat = to_return_units(z_2d, sigma_fwd_2d, ic_lagged)

    for t in range(5):
        expected = ic_lagged[t] * 0.02 * 1.0
        assert np.allclose(alpha_hat[t], expected)


def test_to_return_units_mismatch_shape() -> None:
    z_2d = np.ones((5, 2))
    sigma_fwd_2d = np.ones((5, 3))
    with pytest.raises(ValueError, match="same shape"):
        to_return_units(z_2d, sigma_fwd_2d, 0.05)
