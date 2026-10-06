"""Invariant scenarios for ``src.quant.evaluation.reliability.derive_block_size``."""

from __future__ import annotations

import numpy as np

from src.quant.evaluation.reliability import derive_block_size


def test_block_size_is_one_for_samples_below_ten() -> None:
    assert derive_block_size(np.arange(9.0)) == 1


def test_block_size_is_one_for_zero_variance_input() -> None:
    result = derive_block_size(np.zeros(50))
    assert result == 1


def test_block_size_detects_strong_autocorrelation() -> None:
    rng = np.random.default_rng(0)
    n = 500
    phi = 0.9
    noise = rng.normal(0.0, 1.0, n)
    series = np.empty(n)
    series[0] = noise[0]
    for i in range(1, n):
        series[i] = phi * series[i - 1] + noise[i]
    result = derive_block_size(series)
    assert result > 1


def test_block_size_stays_within_lag_and_sample_caps() -> None:
    rng = np.random.default_rng(42)
    for n in (10, 40, 500):
        white = rng.normal(0.0, 1.0, n)
        ar = np.empty(n)
        ar[0] = 0.0
        for i in range(1, n):
            ar[i] = 0.5 * ar[i - 1] + rng.normal(0.0, 1.0)
        for series in (white, ar):
            result = derive_block_size(series)
            assert 1 <= result <= max(1, n // 5)
            assert result <= max(1, min(20, n // 4))


def test_block_size_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    series = rng.normal(0.0, 1.0, 200)
    assert derive_block_size(series) == derive_block_size(series)
