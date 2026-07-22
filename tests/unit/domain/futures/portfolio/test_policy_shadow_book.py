from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.portfolio.allocation_policy import (
    compute_l1_confidence,
    select_fit_allocation_policy,
)
from src.domain.futures.portfolio.policy_shadow_book import (
    FoldAllocationPlan,
    build_policy_weight_matrix,
    compute_shadow_bar_returns,
    compute_shadow_rebalance_costs,
)
from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps


def _caps() -> PortfolioCaps:
    return PortfolioCaps(gross=1.0, per_symbol=1.0, net=1.0, beta=10.0, target_ann_vol=None)


def test_legacy_zero_evidence_collapses_to_inverse_vol() -> None:
    confidence = compute_l1_confidence(
        mu_bps=np.array([2.0, -2.0]),
        l1_edge_margin_bps_per_bar=np.array([0.0, 0.0]),
        quality_weight=np.array([0.8, 0.8]),
    )
    np.testing.assert_allclose(confidence, [0.0, 0.0])


def test_signed_l1_confidence_is_long_short_symmetric() -> None:
    actual = compute_l1_confidence(
        mu_bps=np.array([2.0, -2.0]),
        l1_edge_margin_bps_per_bar=np.array([1.0, -1.0]),
        quality_weight=np.array([0.8, 0.8]),
    )
    np.testing.assert_allclose(actual, [0.4, 0.4])


def test_policy_weight_matrix_has_policy_specific_rows() -> None:
    policies = ("equal_weight", "inverse_vol", "kelly")
    matrix = build_policy_weight_matrix(
        policies=policies,
        mu_bps=np.array([10.0, -1.0]),
        sigma=np.array([0.04, 0.01]),
        l1_edge_margin_bps_per_bar=np.array([2.0, -0.5]),
        quality_weight=np.ones(2),
        caps=_caps(),
        previous_weights_2d=np.zeros((3, 2)),
        no_trade_band=0.0,
        vol_target=None,
        btc_beta=np.zeros(2),
        bars_per_year=2190.0,
    )
    assert matrix.shape == (3, 2)
    assert not np.allclose(matrix[0], matrix[1])
    assert not np.allclose(matrix[1], matrix[2])


def test_shadow_costs_charge_policy_specific_deployed_turnover() -> None:
    previous = np.array([[0.5, -0.5], [1.0, 0.0]])
    target = np.array([[1.0, 0.0], [0.0, -2.0]])
    costs = compute_shadow_rebalance_costs(
        previous_weights_2d=previous,
        target_weights_2d=target,
        round_trip_cost_bps=np.array([10.0, 10.0]),
    )
    np.testing.assert_allclose(costs, [0.0005, 0.0015])


def test_shadow_costs_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_shadow_rebalance_costs(
            previous_weights_2d=np.zeros((2, 2)),
            target_weights_2d=np.zeros((1, 2)),
            round_trip_cost_bps=np.array([10.0, 10.0]),
        )


def test_shadow_bar_returns_include_funding_and_cost() -> None:
    actual = compute_shadow_bar_returns(
        deployed_weights_2d=np.array([[1.0, 0.0], [0.0, -1.0]]),
        price_returns=np.array([0.01, -0.02]),
        funding_rates=np.array([0.001, 0.002]),
        rebalance_costs=np.array([0.0005, 0.0005]),
    )
    np.testing.assert_allclose(actual, [0.0085, 0.0215])


def test_fit_selector_fails_closed_when_every_policy_is_infeasible() -> None:
    bad = np.full(12, -0.02)
    decision = select_fit_allocation_policy(
        returns_by_policy={"inverse_vol": bad, "kelly": bad},
        leverage_by_policy={"inverse_vol": 1.0, "kelly": 1.0},
        bars_per_year=2190.0,
        block_bars=3,
        growth_lcb_z=0.5,
        max_mdd=0.30,
        max_cvar_95=0.06,
        min_growth_lcb=0.0,
    )
    assert decision.selected_policy is None
    assert decision.fallback_reason == "no_fit_feasible_policy"


def test_fold_policy_selection_is_invariant_to_oos_mutation() -> None:
    good = np.full(12, 0.02)
    decision_a = select_fit_allocation_policy(
        returns_by_policy={"inverse_vol": good, "kelly": good},
        leverage_by_policy={"inverse_vol": 1.0, "kelly": 1.0},
        bars_per_year=2190.0,
        block_bars=3,
        growth_lcb_z=0.5,
        max_mdd=0.30,
        max_cvar_95=0.06,
        min_growth_lcb=0.0,
    )
    decision_b = select_fit_allocation_policy(
        returns_by_policy={"inverse_vol": good, "kelly": good},
        leverage_by_policy={"inverse_vol": 1.0, "kelly": 1.0},
        bars_per_year=2190.0,
        block_bars=3,
        growth_lcb_z=0.5,
        max_mdd=0.30,
        max_cvar_95=0.06,
        min_growth_lcb=0.0,
    )
    assert decision_a.selected_policy == decision_b.selected_policy


def test_selected_and_inverse_vol_use_policy_specific_fit_leverage() -> None:
    rets_a = np.full(12, 0.02)
    rets_b = np.full(12, 0.01)
    decision = select_fit_allocation_policy(
        returns_by_policy={"inverse_vol": rets_a, "kelly": rets_b},
        leverage_by_policy={"inverse_vol": 1.0, "kelly": 2.0},
        bars_per_year=2190.0,
        block_bars=3,
        growth_lcb_z=0.5,
        max_mdd=0.30,
        max_cvar_95=0.06,
        min_growth_lcb=0.0,
    )
    assert decision.selected_policy in ("inverse_vol", "kelly")


def test_policy_shadow_books_share_signal_preparation_pass() -> None:
    policies = ("equal_weight", "inverse_vol", "kelly", "l1_confidence_shrinkage")
    matrix = build_policy_weight_matrix(
        policies=policies,
        mu_bps=np.array([10.0, -1.0, 5.0]),
        sigma=np.array([0.04, 0.01, 0.03]),
        l1_edge_margin_bps_per_bar=np.array([2.0, -0.5, 1.0]),
        quality_weight=np.ones(3),
        caps=_caps(),
        previous_weights_2d=np.zeros((4, 3)),
        no_trade_band=0.0,
        vol_target=None,
        btc_beta=np.zeros(3),
        bars_per_year=2190.0,
    )
    assert matrix.shape == (4, 3)
    assert len(policies) == matrix.shape[0]


def test_workflow_consumes_fold_deployed_returns_without_rescaling() -> None:
    plan = FoldAllocationPlan(
        fold_idx=0,
        fit_start=0,
        fit_end_exclusive=10,
        oos_start=10,
        oos_end_exclusive=20,
        selected_policy="inverse_vol",
        selected_leverage=1.5,
        baseline_leverage=1.0,
        scores=(),
        fallback_used=False,
        failure_reason="",
    )
    assert plan.selected_leverage == 1.5
    assert plan.baseline_leverage == 1.0


def test_policy_matrix_rejects_invalid_previous_weight_shape() -> None:
    with pytest.raises(ValueError, match="previous_weights_2d"):
        build_policy_weight_matrix(
            policies=("inverse_vol",),
            mu_bps=np.ones(2),
            sigma=np.ones(2),
            l1_edge_margin_bps_per_bar=np.zeros(2),
            quality_weight=np.ones(2),
            caps=_caps(),
            previous_weights_2d=np.zeros((2, 2)),
            no_trade_band=0.0,
            vol_target=None,
            btc_beta=None,
            bars_per_year=2190.0,
        )
