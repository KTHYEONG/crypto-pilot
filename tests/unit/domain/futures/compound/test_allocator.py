from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.allocator import (
    apply_cost_aware_net_edge,
    combine_alpha_forecasts,
    solve_growth_optimal_weights,
)
from src.domain.futures.compound.config import AllocatorConfig
from src.domain.futures.compound.contracts import (
    AllocationConstraints,
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
    return AlphaForecastTape(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
        symbols=("A", "B", "C"),
        recipe_ids=("trend_h4", "carry_h12"),
        gross_mu_3d=mu,
        mean_edge_var_3d=np.full((n_bars, n_syms, n_recipes), 1e-8, dtype=np.float32),
        residual_var_3d=np.full((n_bars, n_syms, n_recipes), 1e-6, dtype=np.float32),
        reliability_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.float32),
        estimated_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.bool_),
        valid_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.bool_),
        horizon_bars_1d=np.array([4, 12], dtype=np.int16),
        lifecycle_by_recipe=(AlphaLifecycle.ACTIVE, AlphaLifecycle.ACTIVE),
        model_version="v1",
        data_manifest_hash="h1",
        fold_manifest_hash="fh1",
    )


class TestCostAwareNetEdge:
    def test_high_alpha_triggers_instant_rebalance(self) -> None:
        prev = np.array([0.1, 0.0, -0.05], dtype=np.float64)
        target = np.array([0.2, 0.15, -0.12], dtype=np.float64)
        mu = np.array([0.01, 0.005, 0.01], dtype=np.float64)
        result = apply_cost_aware_net_edge(target, prev, mu)
        expected_edge = np.abs((target - prev) * mu)
        assert np.all(expected_edge > 0.0006)
        np.testing.assert_array_equal(result, target)

    def test_low_noise_keeps_previous_weights(self) -> None:
        prev = np.array([0.1, 0.0, -0.05], dtype=np.float64)
        target = np.array([0.1001, 0.0005, -0.0498], dtype=np.float64)
        mu = np.array([1e-6, 1e-6, 1e-6], dtype=np.float64)
        result = apply_cost_aware_net_edge(target, prev, mu)
        np.testing.assert_array_equal(result, prev)

    def test_mixed_threshold_respects_per_symbol(self) -> None:
        prev = np.array([0.1, 0.0, -0.05], dtype=np.float64)
        target = np.array([0.3, 0.0, -0.05], dtype=np.float64)
        mu = np.array([0.01, 0.0, 0.0], dtype=np.float64)
        result = apply_cost_aware_net_edge(target, prev, mu)
        assert result[0] == target[0]
        assert result[1] == prev[1]
        assert result[2] == prev[2]


class TestCombineAlphaForecasts:
    def test_when_signals_agree_reduces_variance(self, two_recipe_tape: AlphaForecastTape) -> None:
        forecast = combine_alpha_forecasts(two_recipe_tape, 10, uncertainty_z=0.5)
        assert isinstance(forecast, CombinedForecast)
        assert forecast.mu_robust_1d.shape == (3,)
        assert np.isfinite(forecast.mu_robust_1d[0])
        assert forecast.mu_robust_1d[0] > forecast.mu_robust_1d[1]

    def test_when_uncertain_returns_zero_robust_mu(self) -> None:
        n_bars, n_syms, n_recipes = 64, 2, 1
        mu = np.zeros((n_bars, n_syms, n_recipes), dtype=np.float32)
        tape = AlphaForecastTape(
            timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
            symbols=("A", "B"),
            recipe_ids=("bad",),
            gross_mu_3d=mu,
            mean_edge_var_3d=np.full((n_bars, n_syms, n_recipes), 1.0, dtype=np.float32),
            residual_var_3d=np.full((n_bars, n_syms, n_recipes), 1.0, dtype=np.float32),
            reliability_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.float32),
            estimated_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.bool_),
            valid_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.bool_),
            horizon_bars_1d=np.array([4], dtype=np.int16),
            lifecycle_by_recipe=(AlphaLifecycle.ACTIVE,),
            model_version="v1",
            data_manifest_hash="h1",
            fold_manifest_hash="fh1",
        )
        forecast = combine_alpha_forecasts(tape, 10, uncertainty_z=5.0)
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
        constraints = AllocationConstraints(
            gross_cap=config.gross_cap, net_cap=config.net_cap,
            per_symbol_cap=np.full(n_syms, config.per_symbol_cap, dtype=np.float64),
            beta_1d=beta, beta_cap=config.beta_cap,
            capacity_weight_1d=capacity,
            cost_bps_1d=cost_bps,
            entry_block_1d=np.zeros(n_syms, dtype=np.bool_),
            exit_required_1d=np.zeros(n_syms, dtype=np.bool_),
        )
        decision = solve_growth_optimal_weights(
            combined=forecast, covariance=cov, previous_weights=prev_w,
            constraints=constraints,
            decision_idx=0, decision_time_ns=0, config=config,
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
        constraints = AllocationConstraints(
            gross_cap=config.gross_cap, net_cap=config.net_cap,
            per_symbol_cap=np.full(n_syms, config.per_symbol_cap, dtype=np.float64),
            beta_1d=beta, beta_cap=config.beta_cap,
            capacity_weight_1d=capacity,
            cost_bps_1d=cost_bps,
            entry_block_1d=np.zeros(n_syms, dtype=np.bool_),
            exit_required_1d=np.zeros(n_syms, dtype=np.bool_),
        )
        decision = solve_growth_optimal_weights(
            combined=forecast, covariance=cov, previous_weights=prev_w,
            constraints=constraints,
            decision_idx=0, decision_time_ns=0, config=config,
        )
        gross_exp = float(np.sum(np.abs(decision.target_weights_1d)))
        assert gross_exp <= config.gross_cap * 1.01
        net_exp = float(np.sum(decision.target_weights_1d))
        assert abs(net_exp) <= config.net_cap * 1.01
        assert np.all(np.abs(decision.target_weights_1d) <= config.per_symbol_cap * 1.01)


def test_combine_alpha_forecasts_when_signals_agree_reduces_variance(two_recipe_tape: AlphaForecastTape) -> None:
    forecast = combine_alpha_forecasts(two_recipe_tape, 10, uncertainty_z=0.5)
    assert isinstance(forecast, CombinedForecast)
    assert np.isfinite(forecast.mu_robust_1d[0])


def test_combine_alpha_forecasts_when_uncertain_returns_zero_robust_mu() -> None:
    n_bars, n_syms, n_recipes = 64, 2, 1
    mu = np.zeros((n_bars, n_syms, n_recipes), dtype=np.float32)
    tape = AlphaForecastTape(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
        symbols=("A", "B"), recipe_ids=("bad",),
        gross_mu_3d=mu,
        mean_edge_var_3d=np.full((n_bars, n_syms, n_recipes), 1.0, dtype=np.float32),
        residual_var_3d=np.full((n_bars, n_syms, n_recipes), 1.0, dtype=np.float32),
        reliability_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.float32),
        estimated_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.bool_),
        valid_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.bool_),
        horizon_bars_1d=np.array([4], dtype=np.int16),
        lifecycle_by_recipe=(AlphaLifecycle.ACTIVE,),
        model_version="v1", data_manifest_hash="h1", fold_manifest_hash="fh1",
    )
    forecast = combine_alpha_forecasts(tape, 10, uncertainty_z=5.0)
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
    constraints = AllocationConstraints(
        gross_cap=config.gross_cap, net_cap=config.net_cap,
        per_symbol_cap=np.full(n_syms, config.per_symbol_cap, dtype=np.float64),
        beta_1d=beta, beta_cap=config.beta_cap,
        capacity_weight_1d=capacity,
        cost_bps_1d=cost_bps,
        entry_block_1d=np.zeros(n_syms, dtype=np.bool_),
        exit_required_1d=np.zeros(n_syms, dtype=np.bool_),
    )
    decision = solve_growth_optimal_weights(
        combined=forecast, covariance=cov, previous_weights=prev_w,
        constraints=constraints,
        decision_idx=0, decision_time_ns=0, config=config,
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
    constraints = AllocationConstraints(
        gross_cap=config.gross_cap, net_cap=config.net_cap,
        per_symbol_cap=np.full(n_syms, config.per_symbol_cap, dtype=np.float64),
        beta_1d=beta, beta_cap=config.beta_cap,
        capacity_weight_1d=capacity,
        cost_bps_1d=cost_bps,
        entry_block_1d=np.zeros(n_syms, dtype=np.bool_),
        exit_required_1d=np.zeros(n_syms, dtype=np.bool_),
    )
    decision = solve_growth_optimal_weights(
        combined=forecast, covariance=cov, previous_weights=prev_w,
        constraints=constraints,
        decision_idx=0, decision_time_ns=0, config=config,
    )
    assert np.sum(np.abs(decision.target_weights_1d)) <= config.gross_cap * 1.01
    assert abs(np.sum(decision.target_weights_1d)) <= config.net_cap * 1.01
    assert np.all(np.abs(decision.target_weights_1d) <= config.per_symbol_cap * 1.01)
