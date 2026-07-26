from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.multiplicity import (
    TrialMultiplicity,
    build_candidate_trial_returns,
    compute_trial_multiplicity,
    deflated_sharpe_probability,
)


class TestBuildCandidateTrialReturns:
    def test_build_candidate_trial_returns_zero_exposure_bar_yields_zero_return(self) -> None:
        T, N, K = 12, 3, 2
        z = np.zeros((T, N, K), dtype=np.float32)
        valid = np.ones((T, N, K), dtype=np.bool_)
        close = np.ones((T + 1, N), dtype=np.float32) * 100.0
        ts = np.arange(T + 1, dtype=np.int64) * np.int64(4 * 3600 * 10**9)
        result = build_candidate_trial_returns(z_3d=z, valid_3d=valid, close_2d=close, timestamps_ns=ts, start_idx=0, end_idx=T)
        assert result.shape == (K, T)
        assert np.all(result == 0.0)

    def test_single_symbol_z_one_matches_symbol_return(self) -> None:
        T, N, K = 6, 2, 1
        z = np.zeros((T, N, K), dtype=np.float32)
        z[:, 0, 0] = 1.0
        valid = np.ones((T, N, K), dtype=np.bool_)
        close = np.array([[100.0, 100.0], [102.0, 100.0], [104.0, 100.0], [106.0, 100.0], [108.0, 100.0], [110.0, 100.0], [112.0, 100.0]], dtype=np.float32)
        ts = np.arange(T + 1, dtype=np.int64) * np.int64(4 * 3600 * 10**9)
        result = build_candidate_trial_returns(z_3d=z, valid_3d=valid, close_2d=close, timestamps_ns=ts, start_idx=0, end_idx=T)
        for t in range(T):
            expected = close[t + 1, 0] / close[t, 0] - 1.0
            assert result[0, t] == pytest.approx(expected, rel=1e-5)

    def test_invalid_bounds_raises_value_error(self) -> None:
        T, N, K = 6, 2, 1
        z = np.zeros((T, N, K), dtype=np.float32)
        valid = np.ones((T, N, K), dtype=np.bool_)
        close = np.ones((T + 1, N), dtype=np.float32)
        ts = np.arange(T + 1, dtype=np.int64) * np.int64(4 * 3600 * 10**9)
        with pytest.raises(ValueError):
            build_candidate_trial_returns(z_3d=z, valid_3d=valid, close_2d=close, timestamps_ns=ts, start_idx=5, end_idx=2)


class TestComputeTrialMultiplicity:
    def test_compute_trial_multiplicity_identical_trials_collapse_to_one_effective(self) -> None:
        n_trial, n_day = 5, 100
        same_returns = np.random.default_rng(42).normal(0.001, 0.01, n_day)
        daily = np.tile(same_returns, (n_trial, 1))
        mult = compute_trial_multiplicity(daily)
        assert mult.effective_trials == pytest.approx(1.0, abs=0.1)

    def test_orthogonal_trials_effective_close_to_k(self) -> None:
        n_trial, n_day = 5, 200
        rng = np.random.default_rng(42)
        daily = rng.normal(0.001, 0.01, (n_trial, n_day))
        mult = compute_trial_multiplicity(daily)
        assert mult.effective_trials == pytest.approx(float(n_trial), rel=0.2)

    def test_zero_trials_returns_degenerate_multiplicity(self) -> None:
        mult = compute_trial_multiplicity(np.zeros((0, 40)))
        assert mult == TrialMultiplicity(0, 1.0, 0.0)

    def test_not_2d_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            compute_trial_multiplicity(np.zeros(10))


class TestDeflatedSharpeProbability:
    def test_deflated_sharpe_probability_matches_closed_form_and_degenerate_returns_half(self) -> None:
        rng = np.random.default_rng(42)
        excess = rng.normal(0.001, 0.01, 365).astype(np.float64)
        mult = TrialMultiplicity(10, 5.0, 0.5)
        result = deflated_sharpe_probability(observed_sharpe=2.0, multiplicity=mult, excess_returns=excess)
        assert 0.0 <= result <= 1.0
        assert np.isfinite(result)

    def test_keff_one_returns_half(self) -> None:
        excess = np.random.default_rng(42).normal(0.001, 0.01, 365).astype(np.float64)
        mult = TrialMultiplicity(5, 1.0, 0.5)
        result = deflated_sharpe_probability(observed_sharpe=2.0, multiplicity=mult, excess_returns=excess)
        assert result == 0.5

    def test_n_less_30_returns_half(self) -> None:
        excess = np.random.default_rng(42).normal(0.001, 0.01, 20).astype(np.float64)
        mult = TrialMultiplicity(5, 5.0, 0.5)
        result = deflated_sharpe_probability(observed_sharpe=2.0, multiplicity=mult, excess_returns=excess)
        assert result == 0.5

    def test_denominator_non_positive_returns_half(self) -> None:
        excess = np.random.default_rng(42).normal(0.001, 0.01, 365).astype(np.float64)
        excess[0] = 1e6
        mult = TrialMultiplicity(10, 5.0, 0.5)
        result = deflated_sharpe_probability(observed_sharpe=2.0, multiplicity=mult, excess_returns=excess)
        assert 0.0 <= result <= 1.0
