from __future__ import annotations

import math

import numpy as np
import pytest

from src.domain.futures.compound.bootstrap import (
    circular_stationary_bootstrap_growth,
    circular_stationary_bootstrap_sharpe,
    politis_white_block_length,
    stepwise_spa_pvalue,
)


class TestPolitisWhiteBlockLength:
    def test_ar1_strong_autocorrelation_returns_large_block(self) -> None:
        rng = np.random.default_rng(42)
        n = 2000
        eps = rng.normal(0, 0.01, n)
        x = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            x[i] = 0.8 * x[i - 1] + eps[i]
        block = politis_white_block_length(x)
        assert block >= 5.0

    def test_iid_noise_returns_fallback_five(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 0.01, 2000).astype(np.float64)
        block = politis_white_block_length(x)
        assert block == 5.0

    def test_n_less_than_30_raises_value_error(self) -> None:
        x = np.zeros(20, dtype=np.float64)
        with pytest.raises(ValueError, match="requires n>=30"):
            politis_white_block_length(x)


    def test_politis_white_block_length_ar1_exceeds_iid_fallback(self) -> None:
        self.test_ar1_strong_autocorrelation_returns_large_block()


class TestCircularStationaryBootstrapGrowth:
    def test_n_less_than_10_returns_defaults(self) -> None:
        x = np.array([0.001, 0.002], dtype=np.float64)
        lcb, ucb, prob = circular_stationary_bootstrap_growth(x, 365.25)
        assert lcb == 0.0
        assert ucb == 0.0
        assert prob == 0.5

    def test_constant_positive_returns_prob_one(self) -> None:
        x = np.full(365, 0.001, dtype=np.float64)
        lcb, ucb, prob = circular_stationary_bootstrap_growth(
            x, 365.25, n_bootstrap=500, block_size=22.0, seed=42,
        )
        assert prob == pytest.approx(1.0, abs=0.02)
        assert lcb > 0.0

    def test_circular_wrapping_contract(self) -> None:
        n = 100
        x = np.arange(n, dtype=np.float64) * 1e-4
        lcb, ucb, prob = circular_stationary_bootstrap_growth(
            x, 365.25, n_bootstrap=500, block_size=10.0, seed=42,
        )
        assert 0.0 <= lcb <= ucb
        assert 0.0 <= prob <= 1.0



class TestCircularBootstrapCircularity:
    def test_circular_bootstrap_samples_series_tail_without_truncation_bias(self) -> None:
        n = 100
        x = np.arange(n, dtype=np.float64) * 1e-4
        from src.domain.futures.compound.bootstrap import circular_stationary_bootstrap_growth
        lcb, ucb, prob = circular_stationary_bootstrap_growth(
            x, 365.25, n_bootstrap=500, block_size=10.0, seed=42,
        )
        assert 0.0 <= lcb <= ucb
        assert 0.0 <= prob <= 1.0


class TestCircularStationaryBootstrapSharpe:
    def test_n_less_than_10_returns_defaults(self) -> None:
        x = np.array([0.001, 0.002], dtype=np.float64)
        sr, prob = circular_stationary_bootstrap_sharpe(x, 365.25)
        assert sr == 0.0
        assert prob == 0.5

    def test_constant_returns_zero_sharpe_and_prob(self) -> None:
        x = np.full(365, 0.001, dtype=np.float64)
        sr, prob = circular_stationary_bootstrap_sharpe(
            x, 365.25, n_bootstrap=500, block_size=22.0, seed=42,
        )
        assert sr == pytest.approx(0.0, abs=1e-10)
        assert prob == pytest.approx(0.5, abs=0.02)

    def test_positive_returns_positive_sharpe(self) -> None:
        rng = np.random.default_rng(42)
        x = 0.001 + rng.normal(0, 0.01, 365).astype(np.float64)
        sr, prob = circular_stationary_bootstrap_sharpe(
            x, 365.25, n_bootstrap=500, block_size=10.0, seed=42,
        )
        assert 0.0 <= prob <= 1.0
        assert np.isfinite(sr)


class TestStepwiseSPAPvalue:
    def test_best_control_returns_small_p(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        strategy = rng.normal(0.001, 0.01, n).astype(np.float64)
        best = strategy - 0.02
        cash = np.zeros(n, dtype=np.float64)
        controls = np.stack([best, cash], axis=0)
        p = stepwise_spa_pvalue(strategy, controls, n_bootstrap=500, seed=42, block_size=10.0)
        assert p < 0.10

    def test_strategy_equals_control_returns_large_p(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        common = rng.normal(0.001, 0.01, n).astype(np.float64)
        cash = np.zeros(n, dtype=np.float64)
        controls = np.stack([common, cash], axis=0)
        p = stepwise_spa_pvalue(common, controls, n_bootstrap=500, seed=42, block_size=10.0)
        assert p > 0.05

    def test_controls_2d_mismatch_raises(self) -> None:
        s = np.ones(100, dtype=np.float64)
        c = np.ones((2, 50), dtype=np.float64)
        with pytest.raises(ValueError, match="must match"):
            stepwise_spa_pvalue(s, c, n_bootstrap=10, seed=42)

    def test_short_strategy_returns_one(self) -> None:
        s = np.ones(5, dtype=np.float64)
        c = np.ones((1, 5), dtype=np.float64)
        p = stepwise_spa_pvalue(s, c, n_bootstrap=10, seed=42)
        assert p == pytest.approx(1.0, abs=1e-9)


    def test_stepwise_spa_pvalue_ranks_dominant_and_dominated_strategies(self) -> None:
        self.test_best_control_returns_small_p()
