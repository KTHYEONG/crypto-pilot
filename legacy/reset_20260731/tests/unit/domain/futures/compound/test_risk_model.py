from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.config import RiskModelConfig
from src.domain.futures.compound.contracts import CovariancePath
from src.domain.futures.compound.risk_model import estimate_covariance_path


class TestEstimateCovariancePath:
    def test_happy_two_asset_high_correlation(self):
        n_bars, n_syms = 300, 2
        rng = np.random.default_rng(42)
        common = rng.normal(0, 0.002, n_bars)
        ret = np.zeros((n_bars, n_syms), dtype=np.float32)
        ret[:, 0] = (common * 1.0 + rng.normal(0, 0.0005, n_bars)).astype(np.float32)
        ret[:, 1] = (common * 1.2 + rng.normal(0, 0.0005, n_bars)).astype(np.float32)
        valid = np.ones((n_bars, n_syms), dtype=np.bool_)

        config = RiskModelConfig(ewm_half_life_bars=60, min_history_bars=10)
        result = estimate_covariance_path(ret, valid, config)

        assert isinstance(result, CovariancePath)
        assert result.covariance_3d.shape == (n_bars, n_syms, n_syms)
        assert result.investable_2d.shape == (n_bars, n_syms)
        assert result.investable_2d[10:].all()

        cov_last = result.covariance_3d[-1]
        assert np.all(np.isfinite(cov_last))

        eigenvalues = np.linalg.eigvalsh(cov_last)
        assert np.all(eigenvalues > 0)

        var0 = cov_last[0, 0]
        var1 = cov_last[1, 1]
        corr = cov_last[0, 1] / np.sqrt(var0 * var1)
        assert corr > 0.5

    def test_non_finite_input_raises(self):
        ret = np.array([[1.0, np.nan]], dtype=np.float32)
        valid = np.array([[True, True]], dtype=np.bool_)
        config = RiskModelConfig()

        with pytest.raises(ValueError, match="non-finite"):
            estimate_covariance_path(ret, valid, config)

    def test_insufficient_history_investable_false(self):
        n_bars, n_syms = 10, 3
        ret = np.ones((n_bars, n_syms), dtype=np.float32) * 0.001
        valid = np.ones((n_bars, n_syms), dtype=np.bool_)
        config = RiskModelConfig(min_history_bars=60)

        result = estimate_covariance_path(ret, valid, config)
        assert not result.investable_2d.any()

    def test_psd_guarantee(self):
        n_bars, n_syms = 100, 5
        rng = np.random.default_rng(7)
        ret = rng.normal(0, 0.001, (n_bars, n_syms)).astype(np.float32)
        valid = np.ones((n_bars, n_syms), dtype=np.bool_)
        config = RiskModelConfig(min_history_bars=20)

        result = estimate_covariance_path(ret, valid, config)
        for t in range(20, n_bars):
            eig = np.linalg.eigvalsh(result.covariance_3d[t])
            assert np.all(eig > 0), f"non-PSD at t={t}: {eig}"

    def test_input_dimension_validation(self):
        config = RiskModelConfig()
        with pytest.raises(ValueError, match="2D"):
            estimate_covariance_path(
                np.ones(10, dtype=np.float32),
                np.ones(10, dtype=np.bool_),
                config,
            )
        with pytest.raises(ValueError, match="same shape"):
            estimate_covariance_path(
                np.ones((10, 3), dtype=np.float32),
                np.ones((10, 2), dtype=np.bool_),
                config,
            )

    def test_diagonal_variance_dominates(self):
        n_bars, n_syms = 150, 3
        rng = np.random.default_rng(1)
        ret = rng.normal(0, [0.001, 0.002, 0.003], (n_bars, n_syms)).astype(np.float32)
        valid = np.ones((n_bars, n_syms), dtype=np.bool_)
        config = RiskModelConfig(min_history_bars=20)

        result = estimate_covariance_path(ret, valid, config)
        cov_last = result.covariance_3d[-1]
        diag = np.diag(cov_last)
        assert diag[2] > diag[1] > diag[0]
