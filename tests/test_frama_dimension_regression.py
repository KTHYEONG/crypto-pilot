from __future__ import annotations

import numpy as np

from src.spot_strategy.frama_evr_poc import compute_frama_series


def _make_price_arrays(scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 240
    x = np.linspace(0.0, 12.0 * np.pi, n, dtype=np.float64)
    base = scale * (1.0 + 0.02 * np.sin(x) + 0.01 * np.sin(3.0 * x))
    noise = scale * 0.002 * np.sin(7.0 * x)
    close = base + noise
    high = close + scale * 0.003
    low = close - scale * 0.003
    return high, low, close


def test_frama_is_scale_invariant_with_ehlers_dimension() -> None:
    high_small, low_small, close_small = _make_price_arrays(scale=100.0)
    high_large, low_large, close_large = _make_price_arrays(scale=5_000_000.0)

    frama_small = compute_frama_series(high_small, low_small, close_small, period=16)
    frama_large = compute_frama_series(high_large, low_large, close_large, period=16)

    rel_small = frama_small / np.maximum(close_small, 1e-12)
    rel_large = frama_large / np.maximum(close_large, 1e-12)
    assert np.allclose(rel_small, rel_large, rtol=1e-9, atol=1e-9)


def test_frama_not_identical_to_close_on_nontrivial_series() -> None:
    high, low, close = _make_price_arrays(scale=3_000_000.0)
    frama = compute_frama_series(high, low, close, period=16)

    n = 16
    delta = np.abs(frama[n - 1 :] - close[n - 1 :])
    assert float(np.max(delta)) > 0.0
