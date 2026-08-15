from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.allocator import solve_event_growth_weights
from src.domain.futures.compound.config import AllocatorConfig
from src.domain.futures.compound.contracts import (
    ActiveForecastState,
    AllocationConstraints,
)


@pytest.fixture
def allocator_inputs() -> dict:
    n_syms = 3
    config = AllocatorConfig(portfolio_nav_usdt=100_000.0)
    state = ActiveForecastState(
        decision_time_ns=1000,
        symbols=("A", "B", "C"),
        alpha_rate_1d=np.array([0.0005, -0.0003, 0.0001], dtype=np.float64),
        epistemic_variance_1d=np.array([1e-8, 1e-8, 1e-8], dtype=np.float64),
        active_event_ids=("ev1", "ev2"),
    )
    covariance_per_hour = np.eye(n_syms, dtype=np.float64) * 1e-4
    previous_weights = np.zeros(n_syms, dtype=np.float64)
    constraints = AllocationConstraints(
        gross_cap=1.0,
        net_cap=0.30,
        per_symbol_cap=np.full(n_syms, 0.10, dtype=np.float64),
        beta_1d=np.zeros(n_syms, dtype=np.float64),
        beta_cap=0.20,
        capacity_weight_1d=np.full(n_syms, 5_000.0, dtype=np.float64),
        cost_bps_1d=np.full(n_syms, 12.0, dtype=np.float64),
        entry_block_1d=np.zeros(n_syms, dtype=np.bool_),
        exit_required_1d=np.zeros(n_syms, dtype=np.bool_),
    )
    return {
        "state": state,
        "covariance_per_hour": covariance_per_hour,
        "previous_weights": previous_weights,
        "constraints": constraints,
        "config": config,
    }


class TestSolveEventGrowthWeights:
    def test_negative_alpha_can_allocate_short(self, allocator_inputs: dict) -> None:
        decision = solve_event_growth_weights(**allocator_inputs)
        assert decision.target_weights_1d[1] < 0.0

    def test_capacity_is_converted_from_usdt_to_weight(self, allocator_inputs: dict) -> None:
        decision = solve_event_growth_weights(**allocator_inputs)
        expected_cap = 5_000.0 / 100_000.0
        for w in decision.target_weights_1d:
            assert abs(w) <= expected_cap + 1e-12

    def test_no_timeframe_quota_enforced(self, allocator_inputs: dict) -> None:
        decision = solve_event_growth_weights(**allocator_inputs)
        assert len(decision.target_weights_1d) == 3

    def test_non_finite_covariance_raises(self, allocator_inputs: dict) -> None:
        inputs = dict(allocator_inputs)
        inputs["covariance_per_hour"] = np.full((3, 3), np.nan, dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite covariance"):
            solve_event_growth_weights(**inputs)
