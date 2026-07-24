from __future__ import annotations

import numpy as np
import pytest

from numpy.typing import NDArray

from src.domain.futures.compound.allocator import (
    _apply_portfolio_level_caps,
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


@pytest.fixture
def sigma_1d() -> NDArray[np.float64]:
    return np.array([0.01, 0.01], dtype=np.float64)


class TestDynamicCompounding:
    def test_compute_dynamic_compounding_weights_happy_path(
        self, two_symbol_forecast: CombinedForecast, sigma_1d: NDArray[np.float64],
        default_config: DynamicCompoundingConfig,
    ) -> None:
        funding = np.array([0.0001, -0.0001], dtype=np.float64)
        prev = np.zeros(2, dtype=np.float64)
        result = compute_dynamic_compounding_weights(
            forecast=two_symbol_forecast,
            sigma_1d=sigma_1d,
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
        assert gross > 0.001

    def test_asymmetric_leverage_caps(self, default_config: DynamicCompoundingConfig) -> None:
        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.1, 0.08, -0.06, -0.05], dtype=np.float64),
            variance_1d=np.array([1e-6, 1e-6, 1e-6, 1e-6], dtype=np.float64),
            support_1d=np.array([True, True, True, True], dtype=np.bool_),
        )
        sigma = np.array([0.005, 0.005, 0.005, 0.005], dtype=np.float64)
        funding = np.zeros(4, dtype=np.float64)
        prev = np.zeros(4, dtype=np.float64)
        result = compute_dynamic_compounding_weights(
            forecast=forecast,
            sigma_1d=sigma,
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
        sigma = np.array([0.01, 0.01], dtype=np.float64)
        funding = np.array([0.0001, 0.0001], dtype=np.float64)
        prev = np.zeros(2, dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite"):
            compute_dynamic_compounding_weights(
                forecast=forecast,
                sigma_1d=sigma,
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
        sigma = np.array([0.01, 0.01], dtype=np.float64)
        funding = np.array([0.0001, 0.0001], dtype=np.float64)
        prev = np.zeros(2, dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite"):
            compute_dynamic_compounding_weights(
                forecast=forecast,
                sigma_1d=sigma,
                funding_rates_1d=funding,
                previous_weights_1d=prev,
                config=default_config,
            )

    def test_non_finite_sigma_raises(self, default_config: DynamicCompoundingConfig) -> None:
        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.005, 0.003], dtype=np.float64),
            variance_1d=np.array([1e-4, 1e-4], dtype=np.float64),
            support_1d=np.array([True, True], dtype=np.bool_),
        )
        sigma = np.array([0.01, np.nan], dtype=np.float64)
        funding = np.array([0.0001, 0.0001], dtype=np.float64)
        prev = np.zeros(2, dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite"):
            compute_dynamic_compounding_weights(
                forecast=forecast,
                sigma_1d=sigma,
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
        sigma = np.array([0.01, 0.01], dtype=np.float64)
        funding = np.array([np.inf, 0.0001], dtype=np.float64)
        prev = np.zeros(2, dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite"):
            compute_dynamic_compounding_weights(
                forecast=forecast,
                sigma_1d=sigma,
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
        sigma = np.array([0.01, 0.01], dtype=np.float64)
        funding = np.array([0.0001, 0.0001], dtype=np.float64)
        prev = np.array([0.1, np.nan], dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite"):
            compute_dynamic_compounding_weights(
                forecast=forecast,
                sigma_1d=sigma,
                funding_rates_1d=funding,
                previous_weights_1d=prev,
                config=default_config,
            )

    def test_compound_engine_compounding_path_wiring(self, default_config: DynamicCompoundingConfig) -> None:
        n_bars, n_syms = 4, 3
        mu_2d = np.tile(np.array([0.005, -0.003, 0.001], dtype=np.float32), (n_bars, 1))
        sigma_2d = np.full((n_bars, n_syms), 0.01, dtype=np.float32)
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
            symbols=("A", "B", "C"),
            mu_2d=mu_2d,
            se_2d=np.full((n_bars, n_syms), 0.01, dtype=np.float32),
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=("f1",),
            admitted_signal_ids=("s1",),
            fold_manifest_hash="fh1",
        )
        funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
        close = np.ones((n_bars, n_syms), dtype=np.float32)
        result = compute_dynamic_compounding_path(
            forecast=forecast,
            sigma_2d=sigma_2d,
            funding_rates_1h_2d=funding,
            config=default_config,
            close_2d=close,
            cost_bps=0.0,
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
        sigma = np.array([0.01, 0.01], dtype=np.float64)
        funding = np.zeros(2, dtype=np.float64)
        prev = np.zeros(2, dtype=np.float64)
        result = compute_dynamic_compounding_weights(
            forecast=forecast,
            sigma_1d=sigma,
            funding_rates_1d=funding,
            previous_weights_1d=prev,
            config=default_config,
        )
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_funding_carry_increases_long_edge(self, default_config: DynamicCompoundingConfig) -> None:
        pos_mu = np.array([0.005], dtype=np.float64)
        sigma = np.array([0.01], dtype=np.float64)
        forecast_no_carry = CombinedForecast(
            mu_robust_1d=pos_mu.copy(),
            variance_1d=np.array([1e-4], dtype=np.float64),
            support_1d=np.array([True], dtype=np.bool_),
        )
        config_no_carry = DynamicCompoundingConfig(funding_carry_enabled=False)
        prev = np.zeros(1, dtype=np.float64)
        result_no_carry = compute_dynamic_compounding_weights(
            forecast=forecast_no_carry,
            sigma_1d=sigma,
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
            sigma_1d=sigma,
            funding_rates_1d=np.array([0.001], dtype=np.float64),
            previous_weights_1d=prev,
            config=default_config,
        )
        assert result_with_carry[0] >= result_no_carry[0]

    def test_path_produces_2d_weights(self, default_config: DynamicCompoundingConfig) -> None:
        n_bars, n_syms = 8, 3
        mu_2d = np.tile(np.array([0.005, -0.003, 0.001], dtype=np.float32), (n_bars, 1))
        sigma_2d = np.full((n_bars, n_syms), 0.01, dtype=np.float32)
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
            symbols=("A", "B", "C"),
            mu_2d=mu_2d,
            se_2d=np.full((n_bars, n_syms), 0.01, dtype=np.float32),
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=("f1",),
            admitted_signal_ids=("s1",),
            fold_manifest_hash="fh1",
        )
        funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
        close = np.ones((n_bars, n_syms), dtype=np.float32)
        result = compute_dynamic_compounding_path(
            forecast=forecast,
            sigma_2d=sigma_2d,
            funding_rates_1h_2d=funding,
            config=default_config,
            close_2d=close,
            cost_bps=0.0,
        )
        assert result.shape == (n_bars, n_syms)
        assert np.all(np.isfinite(result))

    def test_empty_funding_fills_zeros(self, default_config: DynamicCompoundingConfig) -> None:
        n_bars, n_syms = 4, 2
        mu_2d = np.tile(np.array([0.005, -0.003], dtype=np.float32), (n_bars, 1))
        sigma_2d = np.full((n_bars, n_syms), 0.01, dtype=np.float32)
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
            symbols=("X", "Y"),
            mu_2d=mu_2d,
            se_2d=np.full((n_bars, n_syms), 0.01, dtype=np.float32),
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=("f1",),
            admitted_signal_ids=("s1",),
            fold_manifest_hash="fh1",
        )
        funding = np.zeros((0, n_syms), dtype=np.float32)
        close = np.ones((n_bars, n_syms), dtype=np.float32)
        result = compute_dynamic_compounding_path(
            forecast=forecast,
            sigma_2d=sigma_2d,
            funding_rates_1h_2d=funding,
            config=default_config,
            close_2d=close,
            cost_bps=0.0,
        )
        assert result.shape == (n_bars, n_syms)

    def test_allocate_portfolio_step_wiring(self, default_config: DynamicCompoundingConfig) -> None:
        from src.domain.futures.compound.engine import allocate_portfolio_step

        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.005, -0.003], dtype=np.float64),
            variance_1d=np.array([1e-4, 1e-4], dtype=np.float64),
            support_1d=np.array([True, True], dtype=np.bool_),
        )
        sigma = np.array([0.01, 0.01], dtype=np.float64)
        funding: NDArray[np.float64] = np.array([0.0001, -0.0001])
        prev: NDArray[np.float64] = np.zeros(2)
        result = allocate_portfolio_step(
            forecast=forecast,
            sigma_1d=sigma,
            funding_rates=funding,
            previous_weights=prev,
            config=default_config,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert result[0] > 0
        assert result[1] < 0

    def test_existing_happy_path_regression(self, default_config: DynamicCompoundingConfig) -> None:
        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.005, -0.003], dtype=np.float64),
            variance_1d=np.array([1e-4, 1e-4], dtype=np.float64),
            support_1d=np.array([True, True], dtype=np.bool_),
        )
        sigma = np.array([0.01, 0.01], dtype=np.float64)
        funding = np.array([0.0001, -0.0001], dtype=np.float64)
        prev = np.zeros(2, dtype=np.float64)
        result = compute_dynamic_compounding_weights(
            forecast=forecast,
            sigma_1d=sigma,
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
        long_sum = float(np.sum(np.maximum(result, 0.0)))
        assert long_sum <= default_config.max_long_leverage + 1e-12

    # --- Audit Fix Scenarios (FIX-01 ~ FIX-04) ---

    def test_funding_rate_lag_no_lookahead(self, default_config: DynamicCompoundingConfig) -> None:
        n_bars, n_syms = 2, 1
        mu_2d = np.array([[0.001], [0.001]], dtype=np.float32)
        sigma_2d = np.full((n_bars, n_syms), 0.01, dtype=np.float32)
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
            symbols=("A",),
            mu_2d=mu_2d,
            se_2d=np.full((n_bars, n_syms), 0.01, dtype=np.float32),
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=("f1",),
            admitted_signal_ids=("s1",),
            fold_manifest_hash="fh1",
        )
        funding = np.zeros((8, n_syms), dtype=np.float32)
        funding[4] = 0.05
        close = np.ones((n_bars, n_syms), dtype=np.float32)
        result = compute_dynamic_compounding_path(
            forecast=forecast,
            sigma_2d=sigma_2d,
            funding_rates_1h_2d=funding,
            config=default_config,
            close_2d=close,
            cost_bps=0.0,
        )
        assert result.shape == (n_bars, n_syms)
        w0 = float(np.sum(np.abs(result[0])))
        w1 = float(np.sum(np.abs(result[1])))
        assert w1 < w0 * 3

    def test_drawdown_guard_hard_stop(self) -> None:
        n_bars, n_syms = 6, 1
        mu_2d = np.full((n_bars, n_syms), 1.0, dtype=np.float32)
        sigma_2d = np.full((n_bars, n_syms), 0.01, dtype=np.float32)
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
            symbols=("A",),
            mu_2d=mu_2d,
            se_2d=np.full((n_bars, n_syms), 0.01, dtype=np.float32),
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=("f1",),
            admitted_signal_ids=("s1",),
            fold_manifest_hash="fh1",
        )
        funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
        close = np.ones((n_bars, n_syms), dtype=np.float32)
        close[2:] = 0.50
        config = DynamicCompoundingConfig(
            soft_drawdown_limit=0.15, hard_drawdown_limit=0.25,
            max_gross_leverage=1.0, max_long_leverage=0.7, max_short_leverage=0.3,
        )
        result = compute_dynamic_compounding_path(
            forecast=forecast,
            sigma_2d=sigma_2d,
            funding_rates_1h_2d=funding,
            config=config,
            close_2d=close,
            cost_bps=0.0,
        )
        zero_mask = np.all(np.abs(result) < 1e-15, axis=1)
        first_zero = int(np.argmax(zero_mask)) if np.any(zero_mask) else n_bars
        assert first_zero <= 4, f"expected hard stop by bar 4, got first zero at bar {first_zero}"
        if first_zero < n_bars:
            assert np.all(np.abs(result[first_zero:]) < 1e-15)

    def test_drawdown_guard_soft_scaling(self) -> None:
        n_bars, n_syms = 4, 1
        mu_2d = np.full((n_bars, n_syms), 0.5, dtype=np.float32)
        sigma_2d = np.full((n_bars, n_syms), 0.01, dtype=np.float32)
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
            symbols=("A",),
            mu_2d=mu_2d,
            se_2d=np.full((n_bars, n_syms), 0.01, dtype=np.float32),
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=("f1",),
            admitted_signal_ids=("s1",),
            fold_manifest_hash="fh1",
        )
        funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
        close = np.ones((n_bars, n_syms), dtype=np.float32)
        close[2:] = 0.55
        config = DynamicCompoundingConfig(
            max_gross_leverage=1.0, max_long_leverage=0.7, max_short_leverage=0.3,
            soft_drawdown_limit=0.05, hard_drawdown_limit=0.40,
        )
        result = compute_dynamic_compounding_path(
            forecast=forecast,
            sigma_2d=sigma_2d,
            funding_rates_1h_2d=funding,
            config=config,
            close_2d=close,
            cost_bps=0.0,
        )
        w1_norm = float(np.sum(np.abs(result[1])))
        w2_norm = float(np.sum(np.abs(result[2])))
        assert w1_norm > 0, "bar 1 should have normal weights"
        assert 0 < w2_norm < w1_norm

    def test_portfolio_level_long_cap(self) -> None:
        w = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float64)
        result = _apply_portfolio_level_caps(w, max_long=1.5, max_short=0.5, max_gross=2.0)
        long_sum = float(np.sum(np.maximum(result, 0.0)))
        short_sum = float(np.sum(np.maximum(-result, 0.0)))
        gross = float(np.sum(np.abs(result)))
        assert long_sum <= 1.5 + 1e-12, f"long_sum={long_sum} exceeds 1.5"
        assert short_sum <= 0.5 + 1e-12, f"short_sum={short_sum} exceeds 0.5"
        assert gross <= 2.0 + 1e-12, f"gross={gross} exceeds 2.0"
        assert abs(long_sum - 1.5) < 1e-6, f"long_sum={long_sum} should bind at 1.5"

    def test_portfolio_level_short_cap(self) -> None:
        w = np.array([-0.4, -0.4, -0.4, -0.4], dtype=np.float64)
        result = _apply_portfolio_level_caps(w, max_long=1.5, max_short=0.5, max_gross=2.0)
        long_sum = float(np.sum(np.maximum(result, 0.0)))
        short_sum = float(np.sum(np.maximum(-result, 0.0)))
        gross = float(np.sum(np.abs(result)))
        assert long_sum <= 1.5 + 1e-12
        assert short_sum <= 0.5 + 1e-12, f"short_sum={short_sum} exceeds 0.5"
        assert gross <= 2.0 + 1e-12, f"gross={gross} exceeds 2.0"
        assert abs(short_sum - 0.5) < 1e-6, f"short_sum={short_sum} should bind at 0.5"

    def test_drawdown_cooldown_bars_zero_weights(self) -> None:
        n_bars, n_syms = 6, 1
        mu_2d = np.full((n_bars, n_syms), 1.0, dtype=np.float32)
        sigma_2d = np.full((n_bars, n_syms), 0.01, dtype=np.float32)
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
            symbols=("A",),
            mu_2d=mu_2d,
            se_2d=np.full((n_bars, n_syms), 0.01, dtype=np.float32),
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=("f1",),
            admitted_signal_ids=("s1",),
            fold_manifest_hash="fh1",
        )
        funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
        close = np.ones((n_bars, n_syms), dtype=np.float32)
        close[1:] = 0.50

        config = DynamicCompoundingConfig(
            soft_drawdown_limit=0.15, hard_drawdown_limit=0.25,
            max_gross_leverage=1.0, max_long_leverage=0.7, max_short_leverage=0.3,
        )
        result = compute_dynamic_compounding_path(
            forecast=forecast,
            sigma_2d=sigma_2d,
            funding_rates_1h_2d=funding,
            config=config,
            close_2d=close,
            cost_bps=0.0,
        )
        zero_mask = np.all(np.abs(result) < 1e-15, axis=1)
        first_zero = int(np.argmax(zero_mask)) if np.any(zero_mask) else n_bars
        assert first_zero < n_bars, "hard drawdown should zero out weights"
        if first_zero + 1 < n_bars:
            assert float(np.sum(np.abs(result[first_zero + 1]))) < 1e-15

    def test_portfolio_level_gross_cap_after_long_short(self) -> None:
        w = np.array([0.3, 0.3, 0.3, 0.3, 0.3, -0.1, -0.1, -0.1, -0.1, -0.1], dtype=np.float64)
        result = _apply_portfolio_level_caps(w, max_long=1.5, max_short=0.5, max_gross=1.5)
        gross = float(np.sum(np.abs(result)))
        assert gross <= 1.5 + 1e-12, f"gross={gross} exceeds max_gross=1.5"

    def test_engine_passes_sigma_and_close(self, default_config: DynamicCompoundingConfig) -> None:
        from src.domain.futures.compound.engine import allocate_portfolio_step

        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.005, -0.003], dtype=np.float64),
            variance_1d=np.array([1e-4, 1e-4], dtype=np.float64),
            support_1d=np.array([True, True], dtype=np.bool_),
        )
        sigma = np.array([0.01, 0.01], dtype=np.float64)
        funding = np.array([0.0001, -0.0001], dtype=np.float64)
        prev = np.zeros(2, dtype=np.float64)
        result = allocate_portfolio_step(forecast, sigma, funding, prev, default_config)
        assert isinstance(result, np.ndarray)
        assert result[0] > 0
        assert result[1] < 0
        w_capped = _apply_portfolio_level_caps(result, 1.5, 0.5, 2.0)
        np.testing.assert_array_almost_equal(result, w_capped)

    # === New Spec Scenarios (L1/L2 Price-Risk Sizing) ===

    def test_compute_weights_price_risk_sizing_exact_value(self) -> None:
        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.02], dtype=np.float64),
            variance_1d=np.array([1e-4], dtype=np.float64),
            support_1d=np.array([True], dtype=np.bool_),
        )
        sigma = np.array([0.01], dtype=np.float64)
        prev = np.zeros(1, dtype=np.float64)
        config = DynamicCompoundingConfig(kelly_fraction=0.20, sigma_floor=1e-4)
        result = compute_dynamic_compounding_weights(
            forecast=forecast,
            sigma_1d=sigma,
            funding_rates_1d=np.zeros(1, dtype=np.float64),
            previous_weights_1d=prev,
            config=config,
        )
        expected_raw = 0.20 * 0.02 / 0.01
        expected_smoothed = 0.03 * expected_raw + 0.97 * 0.0
        assert abs(float(result[0]) - expected_smoothed) < 1e-6

    def test_path_vol_target_caps_binding(self) -> None:
        n_bars, n_syms = 5, 3
        mu_2d = np.full((n_bars, n_syms), 0.1, dtype=np.float32)
        sigma_2d = np.full((n_bars, n_syms), 0.001, dtype=np.float32)
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
            symbols=("A", "B", "C"),
            mu_2d=mu_2d,
            se_2d=np.full((n_bars, n_syms), 0.01, dtype=np.float32),
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=("f1",),
            admitted_signal_ids=("s1",),
            fold_manifest_hash="fh1",
        )
        funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
        close = np.ones((n_bars, n_syms), dtype=np.float32)
        config = DynamicCompoundingConfig(
            kelly_fraction=0.20, sigma_floor=1e-4,
            max_gross_leverage=1.0, max_long_leverage=0.7, max_short_leverage=0.3,
            vol_scale_max=1.5,
        )
        result = compute_dynamic_compounding_path(
            forecast=forecast,
            sigma_2d=sigma_2d,
            funding_rates_1h_2d=funding,
            config=config,
            close_2d=close,
            cost_bps=0.0,
        )
        for t in range(n_bars):
            w = result[t]
            long_sum = float(np.sum(np.maximum(w, 0.0)))
            short_sum = float(np.sum(np.maximum(-w, 0.0)))
            gross = float(np.sum(np.abs(w)))
            assert long_sum <= 0.7 + 1e-10, f"bar {t}: long_sum={long_sum} > 0.7"
            assert short_sum <= 0.3 + 1e-10, f"bar {t}: short_sum={short_sum} > 0.3"
            assert gross <= 1.0 + 1e-10, f"bar {t}: gross={gross} > 1.0"

    def test_compute_weights_nonfinite_sigma_raises(self) -> None:
        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.02], dtype=np.float64),
            variance_1d=np.array([1e-4], dtype=np.float64),
            support_1d=np.array([True], dtype=np.bool_),
        )
        sigma = np.array([np.inf], dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite"):
            compute_dynamic_compounding_weights(
                forecast=forecast,
                sigma_1d=sigma,
                funding_rates_1d=np.zeros(1, dtype=np.float64),
                previous_weights_1d=np.zeros(1, dtype=np.float64),
                config=DynamicCompoundingConfig(),
            )

    def test_sigma_floor_applied_on_zero_sigma(self) -> None:
        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.02], dtype=np.float64),
            variance_1d=np.array([1e-4], dtype=np.float64),
            support_1d=np.array([True], dtype=np.bool_),
        )
        sigma = np.array([0.0], dtype=np.float64)
        prev = np.zeros(1, dtype=np.float64)
        config = DynamicCompoundingConfig(kelly_fraction=0.20, sigma_floor=1e-4)
        result = compute_dynamic_compounding_weights(
            forecast=forecast,
            sigma_1d=sigma,
            funding_rates_1d=np.zeros(1, dtype=np.float64),
            previous_weights_1d=prev,
            config=config,
        )
        # raw = 0.20 * 0.02 / 1e-4 = 40.0, smoothed = 1.2, cap(long=0.7) → 0.7
        assert abs(float(result[0]) - 0.7) < 1e-6

    def test_vol_scale_parameter_affects_weights(self, default_config: DynamicCompoundingConfig) -> None:
        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.01], dtype=np.float64),
            variance_1d=np.array([1e-4], dtype=np.float64),
            support_1d=np.array([True], dtype=np.bool_),
        )
        sigma = np.array([0.01], dtype=np.float64)
        prev = np.zeros(1, dtype=np.float64)
        funding = np.zeros(1, dtype=np.float64)
        w1 = compute_dynamic_compounding_weights(
            forecast=forecast, sigma_1d=sigma, funding_rates_1d=funding,
            previous_weights_1d=prev, config=default_config, vol_scale=1.0,
        )
        w2 = compute_dynamic_compounding_weights(
            forecast=forecast, sigma_1d=sigma, funding_rates_1d=funding,
            previous_weights_1d=prev, config=default_config, vol_scale=0.5,
        )
        assert abs(float(w2[0]) - 0.5 * float(w1[0])) < 1e-10
