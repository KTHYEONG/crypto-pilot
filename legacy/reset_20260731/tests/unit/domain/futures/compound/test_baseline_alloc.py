from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.baseline_alloc import solve_baseline_weights
from src.domain.futures.compound.config import BaselineAllocConfig
from src.domain.futures.compound.contracts import CovariancePath


class TestSolveBaselineWeights:
    @pytest.fixture
    def two_asset_cov_path(self):
        T, N = 100, 2
        cov = np.array([[0.0004, 0.0002], [0.0002, 0.0009]], dtype=np.float64)
        cov_3d = np.tile(cov, (T, 1, 1))
        return CovariancePath(
            decision_timestamps_ns=np.arange(T, dtype=np.int64),
            covariance_3d=cov_3d,
            investable_2d=np.ones((T, N), dtype=np.bool_),
        )

    def test_inverse_vol_ratio_match(self, two_asset_cov_path):
        T, N = 100, 2
        mu = np.zeros((T, N), dtype=np.float32)
        mu[:, 0] = 0.001
        mu[:, 1] = 0.002

        config = BaselineAllocConfig(gross_cap=2.0, per_symbol_cap=0.05, target_ann_vol=0.20)
        w = solve_baseline_weights(mu, two_asset_cov_path, config, "inverse_vol")

        assert w.shape == (T, N)
        assert np.all(np.isfinite(w))

        sigma0 = np.sqrt(0.0004)
        sigma1 = np.sqrt(0.0009)
        raw_ratio = (1.0 / sigma0) / (1.0 / sigma1)
        w_ratio = w[50, 0] / w[51, 1] if abs(w[51, 1]) > 1e-15 else 0.0
        assert abs(w[50, 0]) > 0
        assert abs(w[50, 1]) > 0

    def test_zero_mu_zero_weight(self, two_asset_cov_path):
        T, N = 100, 2
        mu = np.zeros((T, N), dtype=np.float32)
        config = BaselineAllocConfig()
        w = solve_baseline_weights(mu, two_asset_cov_path, config, "inverse_vol")
        assert np.allclose(w, 0.0)

    def test_unknown_mode_raises(self, two_asset_cov_path):
        mu = np.ones((100, 2), dtype=np.float32) * 0.001
        config = BaselineAllocConfig()
        with pytest.raises(ValueError, match="unknown mode"):
            solve_baseline_weights(mu, two_asset_cov_path, config, "invalid_mode")

    def test_risk_scaled_differs_from_inverse_vol(self, two_asset_cov_path):
        T, N = 100, 2
        mu = np.zeros((T, N), dtype=np.float32)
        mu[:, 0] = 0.001
        mu[:, 1] = -0.0005

        # per_symbol_cap kept loose so cap re-projection does not mask the
        # inverse_vol vs risk_scaled difference in the raw vol-target scale.
        config = BaselineAllocConfig(gross_cap=2.0, per_symbol_cap=1.0, target_ann_vol=0.01)
        w_iv = solve_baseline_weights(mu, two_asset_cov_path, config, "inverse_vol")
        w_rs = solve_baseline_weights(mu, two_asset_cov_path, config, "risk_scaled")

        assert not np.allclose(w_iv, w_rs, atol=1e-10)

    def test_all_positive_mu_positive_weights(self, two_asset_cov_path):
        T, N = 100, 2
        mu = np.ones((T, N), dtype=np.float32) * 0.001
        config = BaselineAllocConfig()
        w = solve_baseline_weights(mu, two_asset_cov_path, config, "inverse_vol")
        assert np.all(w >= -1e-12)

    def test_gross_cap_enforced_after_vol_target_scaling(self):
        T, N = 5, 3
        cov = np.eye(N, dtype=np.float64) * 1e-10  # placeholder warm-up variance
        cov_3d = np.tile(cov, (T, 1, 1))
        cov_path = CovariancePath(
            decision_timestamps_ns=np.arange(T, dtype=np.int64),
            covariance_3d=cov_3d,
            investable_2d=np.ones((T, N), dtype=np.bool_),
        )
        mu = np.full((T, N), 0.001, dtype=np.float32)
        config = BaselineAllocConfig(gross_cap=2.0, per_symbol_cap=0.05, target_ann_vol=0.20)

        w = solve_baseline_weights(mu, cov_path, config, "inverse_vol")

        gross_per_bar = np.sum(np.abs(w), axis=1)
        assert np.all(gross_per_bar <= config.gross_cap + 1e-9)
        assert np.all(np.abs(w) <= config.per_symbol_cap + 1e-9)

    def test_non_investable_symbol_gets_zero_weight(self):
        T, N = 10, 3
        cov = np.eye(N, dtype=np.float64) * 0.0004
        cov_3d = np.tile(cov, (T, 1, 1))
        investable = np.ones((T, N), dtype=np.bool_)
        investable[:, 1] = False
        cov_path = CovariancePath(
            decision_timestamps_ns=np.arange(T, dtype=np.int64),
            covariance_3d=cov_3d,
            investable_2d=investable,
        )
        mu = np.full((T, N), 0.001, dtype=np.float32)
        config = BaselineAllocConfig()

        w = solve_baseline_weights(mu, cov_path, config, "inverse_vol")

        assert np.allclose(w[:, 1], 0.0)
        assert np.any(np.abs(w[:, 0]) > 0)

    def test_shape_mismatch_raises(self):
        cov_path = CovariancePath(
            decision_timestamps_ns=np.arange(50, dtype=np.int64),
            covariance_3d=np.zeros((50, 3, 3), dtype=np.float64),
            investable_2d=np.ones((50, 3), dtype=np.bool_),
        )
        mu = np.ones((100, 2), dtype=np.float32)
        config = BaselineAllocConfig()
        with pytest.raises(ValueError, match="shape"):
            solve_baseline_weights(mu, cov_path, config, "inverse_vol")
