from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.allocator import combine_alpha_forecasts, solve_growth_optimal_weights
from src.domain.futures.compound.config import AllocatorConfig
from src.domain.futures.compound.contracts import (
    AlphaForecastTape,
    AlphaLifecycle,
    CombinedForecast,
)


@pytest.fixture
def two_recipe_tape() -> AlphaForecastTape:
    n_bars, n_syms, n_recipes = 64, 3, 2
    mu = np.zeros((n_bars, n_syms, n_recipes), dtype=np.float32)
    mu[:, 0, 0] = 0.002
    mu[:, 1, 0] = 0.001
    mu[:, 0, 1] = 0.0015
    mu[:, 1, 1] = 0.0008
    variance = np.full((n_bars, n_syms, n_recipes), 1e-8, dtype=np.float32)
    return AlphaForecastTape(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
        symbols=("A", "B", "C"),
        recipe_ids=("trend_h4", "carry_h12"),
        gross_mu_3d=mu,
        forecast_var_3d=variance,
        reliability_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.float32),
        valid_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.bool_),
        horizon_bars_1d=np.array([4, 12], dtype=np.int16),
        lifecycle_by_recipe=(AlphaLifecycle.ACTIVE, AlphaLifecycle.ACTIVE),
        model_version="v1",
        data_manifest_hash="h1",
        fold_manifest_hash="fh1",
    )


class TestCombineAlphaForecasts:
    def test_when_signals_agree_reduces_variance(self, two_recipe_tape: AlphaForecastTape) -> None:
        config = AllocatorConfig(uncertainty_z=0.5)
        forecast = combine_alpha_forecasts(tape=two_recipe_tape, decision_idx=10, config=config)
        assert isinstance(forecast, CombinedForecast)
        assert forecast.mu_robust_1d.shape == (3,)
        assert np.isfinite(forecast.mu_robust_1d[0])
        assert forecast.mu_robust_1d[0] > forecast.mu_robust_1d[1]

    def test_when_uncertain_returns_zero_robust_mu(self) -> None:
        n_bars, n_syms, n_recipes = 64, 2, 1
        mu = np.zeros((n_bars, n_syms, n_recipes), dtype=np.float32)
        variance = np.full((n_bars, n_syms, n_recipes), 1.0, dtype=np.float32)
        tape = AlphaForecastTape(
            timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
            symbols=("A", "B"),
            recipe_ids=("bad",),
            gross_mu_3d=mu,
            forecast_var_3d=variance,
            reliability_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.float32),
            valid_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.bool_),
            horizon_bars_1d=np.array([4], dtype=np.int16),
            lifecycle_by_recipe=(AlphaLifecycle.ACTIVE,),
            model_version="v1",
            data_manifest_hash="h1",
            fold_manifest_hash="fh1",
        )
        config = AllocatorConfig(uncertainty_z=5.0)
        forecast = combine_alpha_forecasts(tape=tape, decision_idx=10, config=config)
        assert forecast.mu_robust_1d[0] == 0.0


class TestSolveGrowthOptimalWeights:
    def test_charges_turnover_once(self) -> None:
        n_syms = 5
        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.001, 0.002, 0.0, 0.0015, 0.0005], dtype=np.float64),
            variance_1d=np.ones(n_syms, dtype=np.float64) * 1e-8,
            support_1d=np.array([True, True, False, True, True], dtype=np.bool_),
        )
        cov = np.eye(n_syms, dtype=np.float64) * 1e-4
        prev_w = np.zeros(n_syms, dtype=np.float64)
        cost_bps = np.ones(n_syms, dtype=np.float64) * 12.0
        capacity = np.ones(n_syms, dtype=np.float64) * 0.1
        beta = np.zeros(n_syms, dtype=np.float64)
        config = AllocatorConfig(gross_cap=1.0, net_cap=0.3, per_symbol_cap=0.1)

        decision = solve_growth_optimal_weights(
            forecast=forecast, covariance_2d=cov, previous_weights_1d=prev_w,
            cost_bps_1d=cost_bps, capacity_weight_1d=capacity, beta_1d=beta, config=config,
        )
        assert decision.target_weights_1d.shape == (n_syms,)
        assert np.all(np.isfinite(decision.target_weights_1d))

    def test_respects_all_caps(self) -> None:
        n_syms = 20
        forecast = CombinedForecast(
            mu_robust_1d=np.ones(n_syms, dtype=np.float64) * 0.01,
            variance_1d=np.ones(n_syms, dtype=np.float64) * 1e-8,
            support_1d=np.ones(n_syms, dtype=np.bool_),
        )
        cov = np.eye(n_syms, dtype=np.float64) * 1e-4
        prev_w = np.zeros(n_syms, dtype=np.float64)
        cost_bps = np.ones(n_syms, dtype=np.float64) * 12.0
        capacity = np.ones(n_syms, dtype=np.float64) * 0.05
        beta = np.ones(n_syms, dtype=np.float64) * 0.5
        config = AllocatorConfig(gross_cap=1.0, net_cap=0.3, per_symbol_cap=0.1, beta_cap=0.25)

        decision = solve_growth_optimal_weights(
            forecast=forecast, covariance_2d=cov, previous_weights_1d=prev_w,
            cost_bps_1d=cost_bps, capacity_weight_1d=capacity, beta_1d=beta, config=config,
        )
        gross_exp = float(np.sum(np.abs(decision.target_weights_1d)))
        assert gross_exp <= config.gross_cap * 1.01
        net_exp = float(np.sum(decision.target_weights_1d))
        assert abs(net_exp) <= config.net_cap * 1.01
        assert np.all(np.abs(decision.target_weights_1d) <= config.per_symbol_cap * 1.01)


def test_combine_alpha_forecasts_when_signals_agree_reduces_variance(two_recipe_tape: AlphaForecastTape) -> None:
    config = AllocatorConfig(uncertainty_z=0.5)
    forecast = combine_alpha_forecasts(tape=two_recipe_tape, decision_idx=10, config=config)
    assert isinstance(forecast, CombinedForecast)
    assert np.isfinite(forecast.mu_robust_1d[0])


def test_combine_alpha_forecasts_when_uncertain_returns_zero_robust_mu() -> None:
    n_bars, n_syms, n_recipes = 64, 2, 1
    mu = np.zeros((n_bars, n_syms, n_recipes), dtype=np.float32)
    variance = np.full((n_bars, n_syms, n_recipes), 1.0, dtype=np.float32)
    tape = AlphaForecastTape(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
        symbols=("A", "B"), recipe_ids=("bad",),
        gross_mu_3d=mu, forecast_var_3d=variance,
        reliability_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.float32),
        valid_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.bool_),
        horizon_bars_1d=np.array([4], dtype=np.int16),
        lifecycle_by_recipe=(AlphaLifecycle.ACTIVE,),
        model_version="v1", data_manifest_hash="h1", fold_manifest_hash="fh1",
    )
    config = AllocatorConfig(uncertainty_z=5.0)
    forecast = combine_alpha_forecasts(tape=tape, decision_idx=10, config=config)
    assert forecast.mu_robust_1d[0] == 0.0


def test_growth_allocator_charges_turnover_once() -> None:
    n_syms = 5
    forecast = CombinedForecast(
        mu_robust_1d=np.array([0.001, 0.002, 0.0, 0.0015, 0.0005], dtype=np.float64),
        variance_1d=np.ones(n_syms, dtype=np.float64) * 1e-8,
        support_1d=np.array([True, True, False, True, True], dtype=np.bool_),
    )
    cov = np.eye(n_syms, dtype=np.float64) * 1e-4
    prev_w = np.zeros(n_syms, dtype=np.float64)
    cost_bps = np.ones(n_syms, dtype=np.float64) * 12.0
    capacity = np.ones(n_syms, dtype=np.float64) * 0.1
    beta = np.zeros(n_syms, dtype=np.float64)
    config = AllocatorConfig(gross_cap=1.0, net_cap=0.3, per_symbol_cap=0.1)
    decision = solve_growth_optimal_weights(
        forecast=forecast, covariance_2d=cov, previous_weights_1d=prev_w,
        cost_bps_1d=cost_bps, capacity_weight_1d=capacity, beta_1d=beta, config=config,
    )
    assert decision.target_weights_1d.shape == (n_syms,)
    assert np.all(np.isfinite(decision.target_weights_1d))


def test_growth_allocator_respects_all_caps() -> None:
    n_syms = 20
    forecast = CombinedForecast(
        mu_robust_1d=np.ones(n_syms, dtype=np.float64) * 0.01,
        variance_1d=np.ones(n_syms, dtype=np.float64) * 1e-8,
        support_1d=np.ones(n_syms, dtype=np.bool_),
    )
    cov = np.eye(n_syms, dtype=np.float64) * 1e-4
    prev_w = np.zeros(n_syms, dtype=np.float64)
    cost_bps = np.ones(n_syms, dtype=np.float64) * 12.0
    capacity = np.ones(n_syms, dtype=np.float64) * 0.05
    beta = np.ones(n_syms, dtype=np.float64) * 0.5
    config = AllocatorConfig(gross_cap=1.0, net_cap=0.3, per_symbol_cap=0.1, beta_cap=0.25)
    decision = solve_growth_optimal_weights(
        forecast=forecast, covariance_2d=cov, previous_weights_1d=prev_w,
        cost_bps_1d=cost_bps, capacity_weight_1d=capacity, beta_1d=beta, config=config,
    )
    assert np.sum(np.abs(decision.target_weights_1d)) <= config.gross_cap * 1.01
    assert abs(np.sum(decision.target_weights_1d)) <= config.net_cap * 1.01
    assert np.all(np.abs(decision.target_weights_1d) <= config.per_symbol_cap * 1.01)
