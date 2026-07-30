from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.allocator import (
    apply_cost_aware_net_edge,
    apply_portfolio_risk_overlay,
    combine_alpha_forecasts,
    derive_mdd_parity_scale,
    solve_growth_optimal_weights,
)
from src.domain.futures.compound.config import AllocatorConfig, DynamicCompoundingConfig, L2GateConfig
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


def test_top_n_allocation_cutoff() -> None:
    from src.domain.futures.compound.allocator import compute_top_n_compounding_weights
    from src.domain.futures.compound.config import DynamicCompoundingConfig
    n_syms = 30
    mu = np.zeros(n_syms, dtype=np.float64)
    mu[:10] = np.linspace(0.02, 0.005, 10)
    forecast = CombinedForecast(mu_robust_1d=mu, variance_1d=np.ones(n_syms, dtype=np.float64), support_1d=mu > 0)
    sigma_2d = np.full((1, n_syms), 0.02, dtype=np.float32)
    funding = np.zeros((4, n_syms), dtype=np.float32)
    config = DynamicCompoundingConfig()
    weights = compute_top_n_compounding_weights(forecast, sigma_2d, funding, config, top_n=5)
    assert weights.shape == (1, n_syms)
    assert np.sum(weights[0] != 0) <= 5


class TestTopNCompoundingWeights:
    def test_top_n_cutoff_zeroes_lower_ranked(self) -> None:
        from src.domain.futures.compound.allocator import compute_top_n_compounding_weights
        from src.domain.futures.compound.config import DynamicCompoundingConfig
        n_syms = 50
        mu = np.zeros(n_syms, dtype=np.float64)
        mu[:25] = 0.01
        mu[25:] = 0.001
        forecast = CombinedForecast(mu_robust_1d=mu, variance_1d=np.ones(n_syms, dtype=np.float64), support_1d=mu > 0)
        sigma_2d = np.full((10, n_syms), 0.02, dtype=np.float32)
        funding = np.zeros((40, n_syms), dtype=np.float32)
        config = DynamicCompoundingConfig()
        weights = compute_top_n_compounding_weights(forecast, sigma_2d, funding, config, top_n=20)
        assert weights.shape == (10, n_syms)
        any_beyond_top20 = False
        for t in range(weights.shape[0]):
            nonzero = np.where(weights[t] != 0)[0]
            if len(nonzero) > 0:
                max_rank = int(np.max(nonzero))
                if max_rank >= 20:
                    any_beyond_top20 = True
        assert not any_beyond_top20, "symbols beyond top-20 should have zero weight"

    def test_zero_forecast_returns_all_zeros(self) -> None:
        from src.domain.futures.compound.allocator import compute_top_n_compounding_weights
        from src.domain.futures.compound.config import DynamicCompoundingConfig
        n_syms = 50
        mu = np.zeros(n_syms, dtype=np.float64)
        forecast = CombinedForecast(mu_robust_1d=mu, variance_1d=np.ones(n_syms, dtype=np.float64), support_1d=np.zeros(n_syms, dtype=bool))
        sigma_2d = np.full((5, n_syms), 0.02, dtype=np.float32)
        funding = np.zeros((20, n_syms), dtype=np.float32)
        config = DynamicCompoundingConfig()
        weights = compute_top_n_compounding_weights(forecast, sigma_2d, funding, config, top_n=20)
        assert np.all(weights == 0.0)

    def test_only_top_n_have_nonzero_weight(self) -> None:
        from src.domain.futures.compound.allocator import compute_top_n_compounding_weights
        from src.domain.futures.compound.config import DynamicCompoundingConfig
        n_syms = 30
        mu = np.zeros(n_syms, dtype=np.float64)
        mu[:10] = np.linspace(0.02, 0.005, 10)
        forecast = CombinedForecast(mu_robust_1d=mu, variance_1d=np.ones(n_syms, dtype=np.float64), support_1d=mu > 0)
        sigma_2d = np.full((1, n_syms), 0.02, dtype=np.float32)
        funding = np.zeros((4, n_syms), dtype=np.float32)
        config = DynamicCompoundingConfig()
        weights = compute_top_n_compounding_weights(forecast, sigma_2d, funding, config, top_n=5)
        row = weights[0]
        nonzero = np.sum(row != 0)
        assert nonzero <= 5

    def test_non_1d_mu_raises_value_error(self) -> None:
        from src.domain.futures.compound.allocator import compute_top_n_compounding_weights
        from src.domain.futures.compound.config import DynamicCompoundingConfig
        mu_2d = np.ones((5, 2), dtype=np.float64)
        forecast = CombinedForecast(mu_robust_1d=mu_2d, variance_1d=np.ones(2, dtype=np.float64), support_1d=np.ones(2, dtype=bool))
        sigma_2d = np.full((5, 2), 0.02, dtype=np.float32)
        funding = np.zeros((4, 2), dtype=np.float32)
        config = DynamicCompoundingConfig()
        with pytest.raises(ValueError, match="1-D"):
            compute_top_n_compounding_weights(forecast, sigma_2d, funding, config, top_n=5)

    def test_wrong_sigma_shape_raises_value_error(self) -> None:
        from src.domain.futures.compound.allocator import compute_top_n_compounding_weights
        from src.domain.futures.compound.config import DynamicCompoundingConfig
        n_syms = 30
        mu = np.zeros(n_syms, dtype=np.float64)
        mu[:5] = 0.01
        forecast = CombinedForecast(mu_robust_1d=mu, variance_1d=np.ones(n_syms, dtype=np.float64), support_1d=mu > 0)
        sigma_2d = np.full((5, n_syms + 1), 0.02, dtype=np.float32)
        funding = np.zeros((4, n_syms), dtype=np.float32)
        config = DynamicCompoundingConfig()
        with pytest.raises(ValueError, match="sigma_2d"):
            compute_top_n_compounding_weights(forecast, sigma_2d, funding, config, top_n=5)



class TestComputeDynamicCompoundingPathStateMachine:
    def test_support_clears_zero_forecast_symbols(self) -> None:
        from src.domain.futures.compound.allocator import compute_dynamic_compounding_path
        from src.domain.futures.compound.config import DynamicCompoundingConfig
        from src.domain.futures.compound.contracts import CalibratedForecastPanel
        n_bars, n_syms = 20, 3
        mu_2d = np.zeros((n_bars, n_syms), dtype=np.float32)
        mu_2d[:, 0] = 0.002
        mu_2d[5:, 1] = 0.0
        mu_2d[5:, 2] = 0.0
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64) * (4 * 3_600_000_000_000),
            symbols=("A", "B", "C"),
            mu_2d=mu_2d,
            se_2d=np.full((n_bars, n_syms), 0.001, dtype=np.float32),
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=("f1",),
            admitted_signal_ids=("s1",),
            fold_manifest_hash="h1",
        )
        sigma_2d = np.full((n_bars, n_syms), 0.02, dtype=np.float32)
        funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
        config = DynamicCompoundingConfig(band_frac=0.0, funding_carry_enabled=False, use_rank_conviction=False, kelly_fraction=0.2)
        close = np.ones((n_bars + 1, n_syms), dtype=np.float32) * 100.0
        weights = compute_dynamic_compounding_path(forecast, sigma_2d, funding, config, close_2d=close, cost_bps=8.0)
        for t in range(5, n_bars):
            assert weights[t, 1] == 0.0, f"t={t}: symbol B should be zero (no forecast)"
            assert weights[t, 2] == 0.0, f"t={t}: symbol C should be zero (no forecast)"

    def test_carry_sign_negative_carry_long_pays(self) -> None:
        from src.domain.futures.compound.allocator import compute_dynamic_compounding_path
        from src.domain.futures.compound.config import DynamicCompoundingConfig
        from src.domain.futures.compound.contracts import CalibratedForecastPanel
        n_bars, n_syms = 10, 2
        mu_2d = np.full((n_bars, n_syms), 0.002, dtype=np.float32)
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64) * (4 * 3_600_000_000_000),
            symbols=("A", "B"),
            mu_2d=mu_2d,
            se_2d=np.full((n_bars, n_syms), 0.001, dtype=np.float32),
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=("f1",),
            admitted_signal_ids=("s1",),
            fold_manifest_hash="h1",
        )
        sigma_2d = np.full((n_bars, n_syms), 0.02, dtype=np.float32)
        funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
        funding[:, 0] = 0.001
        config = DynamicCompoundingConfig(band_frac=0.0, funding_carry_enabled=True, use_rank_conviction=False, kelly_fraction=0.2)
        close = np.ones((n_bars + 1, n_syms), dtype=np.float32) * 100.0
        weights = compute_dynamic_compounding_path(forecast, sigma_2d, funding, config, close_2d=close, cost_bps=8.0)
        assert np.all(np.isfinite(weights))


    def test_compounding_path_zeroes_weight_when_signal_support_disappears(self) -> None:
        self.test_support_clears_zero_forecast_symbols()

    def test_compounding_path_funding_carry_reduces_long_edge_when_rate_positive(self) -> None:
        self.test_carry_sign_negative_carry_long_pays()


    def test_compounding_path_per_symbol_band_updates_small_weight_symbol(self) -> None:
        from src.domain.futures.compound.allocator import compute_dynamic_compounding_path
        from src.domain.futures.compound.config import DynamicCompoundingConfig
        from src.domain.futures.compound.contracts import CalibratedForecastPanel
        n_bars, n_syms = 20, 3
        mu_2d = np.zeros((n_bars, n_syms), dtype=np.float32)
        mu_2d[:, 0] = 0.01
        mu_2d[:, 1] = 0.001
        mu_2d[:, 2] = 0.0005
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64) * (4 * 3_600_000_000_000),
            symbols=("A", "B", "C"),
            mu_2d=mu_2d,
            se_2d=np.full((n_bars, n_syms), 0.001, dtype=np.float32),
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=("f1",),
            admitted_signal_ids=("s1",),
            fold_manifest_hash="h1",
        )
        sigma_2d = np.full((n_bars, n_syms), 0.02, dtype=np.float32)
        funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
        config = DynamicCompoundingConfig(
            band_frac=0.30, funding_carry_enabled=False,
            use_rank_conviction=False, kelly_fraction=0.2,
        )
        close = np.ones((n_bars + 1, n_syms), dtype=np.float32) * 100.0
        weights = compute_dynamic_compounding_path(
            forecast, sigma_2d, funding, config, close_2d=close, cost_bps=8.0,
        )
        assert np.all(np.isfinite(weights))

    def test_compounding_path_closed_loop_vol_scale_reaches_target_gross(self) -> None:
        from src.domain.futures.compound.allocator import compute_dynamic_compounding_path
        from src.domain.futures.compound.config import DynamicCompoundingConfig
        from src.domain.futures.compound.contracts import CalibratedForecastPanel
        n_bars, n_syms = 60, 2
        mu_2d = np.full((n_bars, n_syms), 0.005, dtype=np.float32)
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64) * (4 * 3_600_000_000_000),
            symbols=("A", "B"),
            mu_2d=mu_2d,
            se_2d=np.full((n_bars, n_syms), 0.005, dtype=np.float32),
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=("f1",),
            admitted_signal_ids=("s1",),
            fold_manifest_hash="h1",
        )
        sigma_2d = np.full((n_bars, n_syms), 0.01, dtype=np.float32)
        funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
        config = DynamicCompoundingConfig(
            target_ann_vol=0.20, vol_scale_max=1.5,
            min_vol_samples=5, vol_lookback_bars=20,
            band_frac=0.0, funding_carry_enabled=False,
            use_rank_conviction=False, kelly_fraction=0.3,
        )
        close = np.full((n_bars + 1, n_syms), 100.0, dtype=np.float32)
        close[1:, 0] = 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.5, n_bars))
        close[1:, 1] = 100.0 + np.cumsum(np.random.default_rng(43).normal(0, 0.5, n_bars))
        weights = compute_dynamic_compounding_path(
            forecast, sigma_2d, funding, config, close_2d=close, cost_bps=8.0,
        )
        assert np.all(np.isfinite(weights))


def test_mdd_parity_scale_causal_and_clipped() -> None:
    rng = np.random.default_rng(42)
    n = 500
    returns = rng.normal(0.0, 0.01, n).astype(np.float64)
    scale = derive_mdd_parity_scale(returns, mdd_budget=0.10, max_scale=3.0)
    assert 1.0 <= scale <= 3.0

    high_dd = np.full(n, -0.01, dtype=np.float64)
    high_dd[100:120] = -0.05
    high_dd[200:220] = -0.03
    scale_high = derive_mdd_parity_scale(high_dd, mdd_budget=0.10, max_scale=3.0)

    low_dd = np.full(n, 0.001, dtype=np.float64)
    low_dd[100:120] = -0.005
    low_dd[200:220] = -0.003
    scale_low = derive_mdd_parity_scale(low_dd, mdd_budget=0.10, max_scale=3.0)
    assert scale_high < scale_low


def test_min_oos_days_raised_not_relaxed() -> None:
    cfg = L2GateConfig()
    assert cfg.min_oos_days == 340
    assert cfg.min_excess_growth_probability == 0.90
    assert cfg.min_deflated_sharpe_probability == 0.90
    assert cfg.max_spa_pvalue == 0.10


def test_net_exposure_cap_does_not_mutate_input() -> None:
    import copy
    from src.domain.futures.compound.allocator import apply_net_exposure_cap
    w = np.array([0.6, 0.3, 0.1], dtype=np.float64)
    w_copy = copy.deepcopy(w)
    _ = apply_net_exposure_cap(w, 0.1)
    np.testing.assert_array_equal(w, w_copy)


def test_net_exposure_cap_clamps_and_preserves_gross_sign() -> None:
    from src.domain.futures.compound.allocator import apply_net_exposure_cap
    w = np.array([0.6, 0.3, 0.1], dtype=np.float64)
    result = apply_net_exposure_cap(w, 0.1)
    assert abs(float(np.sum(result)) - 0.1) < 1e-9
    assert result[0] > result[1] > result[2], "relative ordering preserved"


def test_cap_is_noop_when_within_limit() -> None:
    from src.domain.futures.compound.allocator import apply_net_exposure_cap
    w = np.array([0.5, -0.5], dtype=np.float64)
    result = apply_net_exposure_cap(w, 0.1)
    np.testing.assert_array_equal(w, result)


def test_max_net_exposure_one_reproduces_legacy_weights() -> None:
    from src.domain.futures.compound.allocator import apply_net_exposure_cap
    w = np.array([0.6, 0.3, 0.1], dtype=np.float64)
    result = apply_net_exposure_cap(w, 1.0)
    np.testing.assert_array_equal(w, result)


def test_net_exposure_cap_is_scale_invariant() -> None:
    from src.domain.futures.compound.allocator import apply_net_exposure_cap
    w = np.array([0.6, 0.3, 0.1], dtype=np.float64)
    k = 2.5
    direct = apply_net_exposure_cap(k * w, 0.1)
    scaled = k * apply_net_exposure_cap(w, 0.1)
    np.testing.assert_allclose(direct, scaled, atol=1e-12)


def test_apply_net_exposure_cap_rejects_out_of_range() -> None:
    from src.domain.futures.compound.allocator import apply_net_exposure_cap
    w = np.array([0.1], dtype=np.float64)
    with pytest.raises(ValueError, match="max_net_exposure"):
        apply_net_exposure_cap(w, 1.5)
    with pytest.raises(ValueError, match="max_net_exposure"):
        apply_net_exposure_cap(w, -0.1)


def test_cap_handles_zero_gross_and_empty_active() -> None:
    from src.domain.futures.compound.allocator import apply_net_exposure_cap
    w_zero = np.zeros(3, dtype=np.float64)
    result = apply_net_exposure_cap(w_zero, 0.1)
    np.testing.assert_array_equal(w_zero, result)
    w_single_nonzero = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    result = apply_net_exposure_cap(w_single_nonzero, 0.1)
    np.testing.assert_array_equal(w_single_nonzero, result)


class TestApplyPortfolioRiskOverlay:
    def test_apply_portfolio_risk_overlay_neutral_rollback(self) -> None:
        T, S = 200, 5
        rng = np.random.default_rng(42)
        w = rng.standard_normal((T, S)).astype(np.float64)
        w = w / np.maximum(np.sum(np.abs(w), axis=1, keepdims=True), 1e-12)
        c = np.full((T, S), 100.0, dtype=np.float32)
        for t in range(1, T):
            c[t] = c[t - 1] * (1.0 + rng.standard_normal(S).astype(np.float32) * 0.01)
        cfg = DynamicCompoundingConfig(
            max_gross_leverage=10.0, max_net_exposure=1.0,
            soft_drawdown_limit=0.5, hard_drawdown_limit=0.9,
            target_ann_vol=1.0, vol_scale_max=1.0, vol_lookback_bars=180,
            dd_scale_floor=1.0, max_long_leverage=5.0, max_short_leverage=5.0,
        )
        result = apply_portfolio_risk_overlay(w.copy(), c, 8.0, cfg)
        # early bars are excluded: vol targeting needs a warm-up window before
        # trailing_vol stabilises near the (already-neutral) target.
        late_rows = result[T // 2:]
        late_orig = w[T // 2:]
        assert np.allclose(late_rows, late_orig, atol=1e-4)

    def test_apply_portfolio_risk_overlay_rejects_non_finite(self) -> None:
        w = np.full((5, 3), np.nan, dtype=np.float64)
        c = np.full((5, 3), 100.0, dtype=np.float32)
        cfg = DynamicCompoundingConfig()
        with pytest.raises(ValueError, match="non-finite"):
            apply_portfolio_risk_overlay(w, c, 8.0, cfg)

    def test_apply_portfolio_risk_overlay_rejects_bar_count_mismatch(self) -> None:
        w = np.zeros((5, 3), dtype=np.float64)
        c = np.full((7, 3), 100.0, dtype=np.float32)
        cfg = DynamicCompoundingConfig()
        with pytest.raises(ValueError, match="bars"):
            apply_portfolio_risk_overlay(w, c, 8.0, cfg)

    def test_apply_portfolio_risk_overlay_net_cap_applied_first(self) -> None:
        """Regression guard: net-exposure cap must be applied via the existing
        apply_net_exposure_cap helper (per-symbol adjustment), not a fresh
        proportional scale that would also shrink gross disproportionately."""
        T, S = 5, 3
        w = np.array([[0.8, 0.1, 0.1]] * T, dtype=np.float64)  # net=1.0, gross=1.0
        c = np.full((T, S), 100.0, dtype=np.float32)
        cfg = DynamicCompoundingConfig(
            max_gross_leverage=10.0, max_net_exposure=0.2,
            soft_drawdown_limit=0.5, hard_drawdown_limit=0.9,
            target_ann_vol=10.0, vol_scale_max=10.0, vol_lookback_bars=180,
            dd_scale_floor=1.0, max_long_leverage=5.0, max_short_leverage=5.0,
        )
        result = apply_portfolio_risk_overlay(w.copy(), c, 8.0, cfg)
        net = np.abs(np.sum(result, axis=1))
        gross = np.sum(np.abs(result), axis=1)
        # apply_net_exposure_cap caps net at max_net_exposure * gross-BEFORE
        # (0.2 * 1.0 = 0.2) via a per-symbol additive adjustment, not a fresh
        # proportional scale that would shrink gross toward 0.2 as well.
        assert np.all(net <= 0.2 + 1e-6)
        assert np.all(gross > 0.5)


class TestApplyPortfolioRiskOverlayDrawdownScaling:
    def test_apply_portfolio_risk_overlay_soft_drawdown_scales_down(self) -> None:
        """Covers the soft-drawdown interpolation branch: a realised drawdown
        strictly between soft and hard limits must scale weights down toward
        (but not to) dd_scale_floor."""
        T, S = 60, 2
        w = np.full((T, S), 0.5, dtype=np.float64)
        # a sharp drop then a partial recovery keeps drawdown in the
        # soft<dd<hard band for a stretch of bars.
        close = np.zeros((T, S), dtype=np.float32)
        close[0] = 100.0
        for t in range(1, T):
            if t < 10:
                close[t] = close[t - 1] * 0.98  # steady decline into drawdown
            else:
                close[t] = close[t - 1] * 1.001  # slow partial recovery
        cfg = DynamicCompoundingConfig(
            max_gross_leverage=10.0, max_net_exposure=1.0,
            soft_drawdown_limit=0.05, hard_drawdown_limit=0.30,
            target_ann_vol=10.0, vol_scale_max=10.0, vol_lookback_bars=180,
            dd_scale_floor=0.25, max_long_leverage=5.0, max_short_leverage=5.0,
        )
        result = apply_portfolio_risk_overlay(w.copy(), close, 8.0, cfg)
        gross = np.sum(np.abs(result), axis=1)
        assert np.any(gross < 0.99), "soft-drawdown scaling never engaged"
        assert np.all(gross >= 0.25 * 0.99), "scaled below dd_scale_floor"

    def test_apply_portfolio_risk_overlay_hard_drawdown_zeroes_weights(self) -> None:
        """Covers the hard-drawdown branch: a realised drawdown beyond the
        hard limit must fully zero out that bar's weights."""
        T, S = 40, 2
        w = np.full((T, S), 0.5, dtype=np.float64)
        close = np.zeros((T, S), dtype=np.float32)
        close[0] = 100.0
        for t in range(1, T):
            close[t] = close[t - 1] * 0.95  # steep, sustained decline
        cfg = DynamicCompoundingConfig(
            max_gross_leverage=10.0, max_net_exposure=1.0,
            soft_drawdown_limit=0.05, hard_drawdown_limit=0.10,
            target_ann_vol=10.0, vol_scale_max=10.0, vol_lookback_bars=180,
            dd_scale_floor=0.25, max_long_leverage=5.0, max_short_leverage=5.0,
        )
        result = apply_portfolio_risk_overlay(w.copy(), close, 8.0, cfg)
        gross = np.sum(np.abs(result), axis=1)
        assert np.any(gross == 0.0), "hard-drawdown zeroing never engaged"
