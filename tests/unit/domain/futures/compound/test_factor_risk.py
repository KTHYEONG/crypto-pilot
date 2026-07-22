from __future__ import annotations

import numpy as np

from src.domain.futures.compound.config import FactorRiskConfig
from src.domain.futures.compound.factor_risk import _ewm_cov, estimate_causal_factor_covariance


def test_ewm_covariance_supports_vector_and_matrix_inputs() -> None:
    vector = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    matrix = np.column_stack((vector, vector * 0.5))

    vector_cov = _ewm_cov(vector, half_life=2.0)
    matrix_cov = _ewm_cov(matrix, half_life=2.0)

    assert vector_cov.shape == (1, 1)
    assert matrix_cov.shape == (2, 2)
    assert np.all(np.isfinite(vector_cov))
    assert np.all(np.isfinite(matrix_cov))


class TestEstimateCausalFactorCovariance:
    def test_returns_square_covariance_matrix(self) -> None:
        n_days, n_syms = 200, 5
        returns = np.random.randn(n_days, n_syms).astype(np.float64) * 0.01
        cluster_ids = np.array([0, 0, 1, 1, 2], dtype=np.int16)
        config = FactorRiskConfig()
        cov = estimate_causal_factor_covariance(
            daily_returns_2d=returns, end_exclusive=180, cluster_ids_1d=cluster_ids, config=config,
        )
        assert cov.shape == (n_syms, n_syms)

    def test_excludes_current_day(self) -> None:
        n_days, n_syms = 100, 3
        returns = np.random.randn(n_days, n_syms).astype(np.float64) * 0.01
        cluster_ids = np.zeros(n_syms, dtype=np.int16)
        config = FactorRiskConfig()
        cov_90 = estimate_causal_factor_covariance(
            daily_returns_2d=returns, end_exclusive=90, cluster_ids_1d=cluster_ids, config=config,
        )
        cov_100 = estimate_causal_factor_covariance(
            daily_returns_2d=returns, end_exclusive=100, cluster_ids_1d=cluster_ids, config=config,
        )
        assert cov_90.shape == (3, 3)
        assert cov_100.shape == (3, 3)

    def test_when_non_psd_falls_back_to_diagonal(self) -> None:
        n_days, n_syms = 200, 10
        returns = np.zeros((n_days, n_syms), dtype=np.float64)
        cluster_ids = np.zeros(n_syms, dtype=np.int16)
        config = FactorRiskConfig()
        cov = estimate_causal_factor_covariance(
            daily_returns_2d=returns, end_exclusive=180, cluster_ids_1d=cluster_ids, config=config,
        )
        assert cov.shape == (n_syms, n_syms)
        off_diag = cov - np.diag(np.diag(cov))
        assert np.all(np.abs(off_diag) < 1e-12), "Fallback should be diagonal"

    def test_small_sample_falls_back(self) -> None:
        n_days, n_syms = 5, 3
        returns = np.random.randn(n_days, n_syms).astype(np.float64) * 0.01
        cluster_ids = np.zeros(n_syms, dtype=np.int16)
        config = FactorRiskConfig()
        cov = estimate_causal_factor_covariance(
            daily_returns_2d=returns, end_exclusive=5, cluster_ids_1d=cluster_ids, config=config,
        )
        assert cov.shape == (3, 3)


def test_factor_covariance_excludes_current_day() -> None:
    n_days, n_syms = 100, 3
    returns = np.random.randn(n_days, n_syms).astype(np.float64) * 0.01
    cluster_ids = np.zeros(n_syms, dtype=np.int16)
    config = FactorRiskConfig()
    cov_90 = estimate_causal_factor_covariance(
        daily_returns_2d=returns, end_exclusive=90, cluster_ids_1d=cluster_ids, config=config,
    )
    cov_100 = estimate_causal_factor_covariance(
        daily_returns_2d=returns, end_exclusive=100, cluster_ids_1d=cluster_ids, config=config,
    )
    assert cov_90.shape == (3, 3)
    assert cov_100.shape == (3, 3)


def test_factor_covariance_when_non_psd_falls_back_to_diagonal() -> None:
    n_days, n_syms = 200, 10
    returns = np.zeros((n_days, n_syms), dtype=np.float64)
    cluster_ids = np.zeros(n_syms, dtype=np.int16)
    config = FactorRiskConfig()
    cov = estimate_causal_factor_covariance(
        daily_returns_2d=returns, end_exclusive=180, cluster_ids_1d=cluster_ids, config=config,
    )
    assert cov.shape == (n_syms, n_syms)
    off_diag = cov - np.diag(np.diag(cov))
    assert np.all(np.abs(off_diag) < 1e-12)
