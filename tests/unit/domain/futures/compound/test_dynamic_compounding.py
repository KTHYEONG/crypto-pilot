from __future__ import annotations

import numpy as np
import pytest

from numpy.typing import NDArray

from src.domain.futures.compound.allocator import (
    compute_dynamic_compounding_path,
    compute_dynamic_compounding_weights,
)
from src.domain.futures.compound.config import DynamicCompoundingConfig
from src.domain.futures.compound.contracts import CalibratedForecastPanel, CombinedForecast


@pytest.fixture
def default_config() -> DynamicCompoundingConfig:
    return DynamicCompoundingConfig()


@pytest.fixture
def two_symbol_forecast() -> CombinedForecast:
    return CombinedForecast(
        mu_robust_1d=np.array([0.005, -0.003], dtype=np.float64),
        variance_1d=np.array([1e-4, 1e-4], dtype=np.float64),
        support_1d=np.array([True, True], dtype=np.bool_),
    )


class TestDynamicCompounding:
    def test_compute_dynamic_compounding_weights_happy_path(
        self, two_symbol_forecast: CombinedForecast, default_config: DynamicCompoundingConfig,
    ) -> None:
        funding = np.array([0.0001, -0.0001], dtype=np.float64)
        prev = np.zeros(2, dtype=np.float64)
        result = compute_dynamic_compounding_weights(
            forecast=two_symbol_forecast,
            funding_rates_1d=funding,
            previous_weights_1d=prev,
            config=default_config,
        )
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))
        assert result[0] > 0
        assert result[1] < 0
        gross = float(np.sum(np.abs(result)))
        assert gross <= default_config.max_gross_leverage
        assert gross > 0.25

    def test_asymmetric_leverage_caps(self, default_config: DynamicCompoundingConfig) -> None:
        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.1, 0.08, -0.06, -0.05], dtype=np.float64),
            variance_1d=np.array([1e-6, 1e-6, 1e-6, 1e-6], dtype=np.float64),
            support_1d=np.array([True, True, True, True], dtype=np.bool_),
        )
        funding = np.zeros(4, dtype=np.float64)
        prev = np.zeros(4, dtype=np.float64)
        result = compute_dynamic_compounding_weights(
            forecast=forecast,
            funding_rates_1d=funding,
            previous_weights_1d=prev,
            config=default_config,
        )
        long_exposure = float(np.sum(np.clip(result, 0, None)))
        short_exposure = float(np.sum(np.clip(result, None, 0)))
        gross = float(np.sum(np.abs(result)))
        assert long_exposure <= default_config.max_long_leverage + 1e-12
        assert abs(short_exposure) <= default_config.max_short_leverage + 1e-12
        assert gross <= default_config.max_gross_leverage + 1e-12

    def test_regime_fallback_and_nan_handling(self, default_config: DynamicCompoundingConfig) -> None:
        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.005, np.nan], dtype=np.float64),
            variance_1d=np.array([1e-4, 1e-4], dtype=np.float64),
            support_1d=np.array([True, True], dtype=np.bool_),
        )
        funding = np.array([0.0001, 0.0001], dtype=np.float64)
        prev = np.zeros(2, dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite"):
            compute_dynamic_compounding_weights(
                forecast=forecast,
                funding_rates_1d=funding,
                previous_weights_1d=prev,
                config=default_config,
            )

    def test_non_finite_mu_raises(self, default_config: DynamicCompoundingConfig) -> None:
        forecast = CombinedForecast(
            mu_robust_1d=np.array([np.inf, 0.003], dtype=np.float64),
            variance_1d=np.array([1e-4, 1e-4], dtype=np.float64),
            support_1d=np.array([True, True], dtype=np.bool_),
        )
        funding = np.array([0.0001, 0.0001], dtype=np.float64)
        prev = np.zeros(2, dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite"):
            compute_dynamic_compounding_weights(
                forecast=forecast,
                funding_rates_1d=funding,
                previous_weights_1d=prev,
                config=default_config,
            )

    def test_non_finite_variance_raises(self, default_config: DynamicCompoundingConfig) -> None:
        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.005, 0.003], dtype=np.float64),
            variance_1d=np.array([1e-4, np.nan], dtype=np.float64),
            support_1d=np.array([True, True], dtype=np.bool_),
        )
        funding = np.array([0.0001, 0.0001], dtype=np.float64)
        prev = np.zeros(2, dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite"):
            compute_dynamic_compounding_weights(
                forecast=forecast,
                funding_rates_1d=funding,
                previous_weights_1d=prev,
                config=default_config,
            )

    def test_non_finite_funding_rates_raises(self, default_config: DynamicCompoundingConfig) -> None:
        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.005, 0.003], dtype=np.float64),
            variance_1d=np.array([1e-4, 1e-4], dtype=np.float64),
            support_1d=np.array([True, True], dtype=np.bool_),
        )
        funding = np.array([np.inf, 0.0001], dtype=np.float64)
        prev = np.zeros(2, dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite"):
            compute_dynamic_compounding_weights(
                forecast=forecast,
                funding_rates_1d=funding,
                previous_weights_1d=prev,
                config=default_config,
            )

    def test_non_finite_previous_weights_raises(self, default_config: DynamicCompoundingConfig) -> None:
        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.005, 0.003], dtype=np.float64),
            variance_1d=np.array([1e-4, 1e-4], dtype=np.float64),
            support_1d=np.array([True, True], dtype=np.bool_),
        )
        funding = np.array([0.0001, 0.0001], dtype=np.float64)
        prev = np.array([0.1, np.nan], dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite"):
            compute_dynamic_compounding_weights(
                forecast=forecast,
                funding_rates_1d=funding,
                previous_weights_1d=prev,
                config=default_config,
            )

    def test_compound_engine_compounding_v6_wiring(self, default_config: DynamicCompoundingConfig) -> None:
        n_bars, n_syms = 4, 3
        mu_2d = np.tile(np.array([0.005, -0.003, 0.001], dtype=np.float32), (n_bars, 1))
        se_2d = np.full((n_bars, n_syms), 0.01, dtype=np.float32)
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
            symbols=("A", "B", "C"),
            mu_2d=mu_2d,
            se_2d=se_2d,
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=("f1",),
            admitted_signal_ids=("s1",),
            fold_manifest_hash="fh1",
        )
        funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
        result = compute_dynamic_compounding_path(
            forecast=forecast,
            funding_rates_1h_2d=funding,
            config=default_config,
        )
        assert result.shape == (n_bars, n_syms)
        assert np.all(np.isfinite(result))
        assert float(np.sum(np.abs(result[-1]))) <= default_config.max_gross_leverage

    def test_no_support_returns_zeros(self, default_config: DynamicCompoundingConfig) -> None:
        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.0, 0.0], dtype=np.float64),
            variance_1d=np.array([1e-4, 1e-4], dtype=np.float64),
            support_1d=np.array([False, False], dtype=np.bool_),
        )
        funding = np.zeros(2, dtype=np.float64)
        prev = np.zeros(2, dtype=np.float64)
        result = compute_dynamic_compounding_weights(
            forecast=forecast,
            funding_rates_1d=funding,
            previous_weights_1d=prev,
            config=default_config,
        )
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_funding_carry_increases_long_edge(self, default_config: DynamicCompoundingConfig) -> None:
        pos_mu = np.array([0.005], dtype=np.float64)
        forecast_no_carry = CombinedForecast(
            mu_robust_1d=pos_mu.copy(),
            variance_1d=np.array([1e-4], dtype=np.float64),
            support_1d=np.array([True], dtype=np.bool_),
        )
        config_no_carry = DynamicCompoundingConfig(funding_carry_enabled=False)
        prev = np.zeros(1, dtype=np.float64)
        result_no_carry = compute_dynamic_compounding_weights(
            forecast=forecast_no_carry,
            funding_rates_1d=np.array([0.0], dtype=np.float64),
            previous_weights_1d=prev,
            config=config_no_carry,
        )
        forecast_carry = CombinedForecast(
            mu_robust_1d=pos_mu.copy(),
            variance_1d=np.array([1e-4], dtype=np.float64),
            support_1d=np.array([True], dtype=np.bool_),
        )
        result_with_carry = compute_dynamic_compounding_weights(
            forecast=forecast_carry,
            funding_rates_1d=np.array([0.001], dtype=np.float64),
            previous_weights_1d=prev,
            config=default_config,
        )
        assert result_with_carry[0] >= result_no_carry[0]

    def test_path_produces_2d_weights(self, default_config: DynamicCompoundingConfig) -> None:
        n_bars, n_syms = 8, 3
        mu_2d = np.tile(np.array([0.005, -0.003, 0.001], dtype=np.float32), (n_bars, 1))
        se_2d = np.full((n_bars, n_syms), 0.01, dtype=np.float32)
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
            symbols=("A", "B", "C"),
            mu_2d=mu_2d,
            se_2d=se_2d,
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=("f1",),
            admitted_signal_ids=("s1",),
            fold_manifest_hash="fh1",
        )
        funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
        result = compute_dynamic_compounding_path(
            forecast=forecast,
            funding_rates_1h_2d=funding,
            config=default_config,
        )
        assert result.shape == (n_bars, n_syms)
        assert np.all(np.isfinite(result))

    def test_empty_funding_fills_zeros(self, default_config: DynamicCompoundingConfig) -> None:
        n_bars, n_syms = 4, 2
        mu_2d = np.tile(np.array([0.005, -0.003], dtype=np.float32), (n_bars, 1))
        se_2d = np.full((n_bars, n_syms), 0.01, dtype=np.float32)
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
            symbols=("X", "Y"),
            mu_2d=mu_2d,
            se_2d=se_2d,
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=("f1",),
            admitted_signal_ids=("s1",),
            fold_manifest_hash="fh1",
        )
        funding = np.zeros((0, n_syms), dtype=np.float32)
        result = compute_dynamic_compounding_path(
            forecast=forecast,
            funding_rates_1h_2d=funding,
            config=default_config,
        )
        assert result.shape == (n_bars, n_syms)

    def test_allocate_portfolio_step_wiring(self, default_config: DynamicCompoundingConfig) -> None:
        from src.domain.futures.compound.engine import allocate_portfolio_step

        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.005, -0.003], dtype=np.float64),
            variance_1d=np.array([1e-4, 1e-4], dtype=np.float64),
            support_1d=np.array([True, True], dtype=np.bool_),
        )
        funding: NDArray[np.float64] = np.array([0.0001, -0.0001])
        prev: NDArray[np.float64] = np.zeros(2)
        result = allocate_portfolio_step(
            forecast=forecast,
            funding_rates=funding,
            previous_weights=prev,
            config=default_config,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert result[0] > 0
        assert result[1] < 0
