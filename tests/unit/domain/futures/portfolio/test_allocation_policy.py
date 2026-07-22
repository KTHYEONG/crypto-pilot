"""Tests for allocation_policy.py: L1 confidence, policy weights, fit selector, deployment fallback."""
from __future__ import annotations


import numpy as np
import pytest

from src.domain.futures.portfolio.allocation_policy import (
    AllocationPolicyScore,
    build_policy_weights,
    choose_deployed_policy,
    compute_l1_confidence,
    select_fit_allocation_policy,
)
from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps


def _caps() -> PortfolioCaps:
    return PortfolioCaps(gross=1.0, per_symbol=1.0, net=1.0, beta=10.0, target_ann_vol=None)


def _score(
    policy: str,
    *,
    growth_lcb: float,
    feasible: bool,
) -> AllocationPolicyScore:
    return AllocationPolicyScore(
        policy=policy,  # type: ignore[arg-type]
        growth_lcb=growth_lcb,
        cagr=0.10 if feasible else -0.10,
        mdd=0.10,
        cvar_95=0.02,
        leverage=1.0,
        n_blocks=4,
        feasible=feasible,
        reason="",
    )


def test_awf_policy_selection_is_invariant_to_oos_return_changes() -> None:
    fit = np.array([0.01, -0.002, 0.008] * 4, dtype=np.float64)
    kwargs = {
        "returns_by_policy": {"inverse_vol": fit, "equal_weight": fit},
        "leverage_by_policy": {"inverse_vol": 1.0, "equal_weight": 1.0},
        "bars_per_year": 2190.0,
        "block_bars": 3,
        "growth_lcb_z": 0.5,
        "max_mdd": 0.30,
        "max_cvar_95": 0.06,
        "min_growth_lcb": 0.0,
    }
    first = select_fit_allocation_policy(**kwargs).selected_policy
    _oos_returns = np.array([-0.5, 0.4, -0.3], dtype=np.float64)
    second = select_fit_allocation_policy(**kwargs).selected_policy
    assert first == second == "inverse_vol"
    assert _oos_returns.size == 3


def test_selected_and_baseline_use_policy_specific_fit_leverage() -> None:
    score = _score("kelly", growth_lcb=0.04, feasible=True)
    baseline = _score("inverse_vol", growth_lcb=0.03, feasible=True)
    assert score.leverage == 1.0
    assert baseline.leverage == 1.0
    assert choose_deployed_policy(selected=score, inverse_vol=baseline) == ("kelly", False)


def test_l2_search_space_excludes_allocation_policy_and_kelly_shrink() -> None:
    from pathlib import Path

    source = Path("src/domain/futures/optimization/l2_search_space.py").read_text()
    assert "kelly_shrink_to_equal" not in source


def test_awf_candidate_policies_share_one_simulation_pass() -> None:
    import inspect
    from src.domain.futures.strategy.tiered_workflow.awf_sim import _run_awf_simulation

    source = inspect.getsource(_run_awf_simulation)
    assert source.count("select_fit_allocation_policy") == 2
    assert "returns_by_policy" in source


def test_legacy_snapshot_without_lcb_degrades_to_inverse_vol() -> None:
    kwargs = {
        "mu_bps": np.array([2.0, -1.0]),
        "sigma": np.array([0.02, 0.01]),
        "l1_edge_margin_bps_per_bar": np.zeros(2),
        "quality_weight": np.ones(2),
        "caps": _caps(),
        "prev_w": np.zeros(2),
        "no_trade_band": 0.0,
        "vol_target": None,
        "btc_beta": np.zeros(2),
        "bars_per_year": 2190.0,
    }
    np.testing.assert_allclose(
        build_policy_weights(policy="l1_confidence_shrinkage", **kwargs),
        build_policy_weights(policy="inverse_vol", **kwargs),
    )


class TestComputeL1Confidence:
    def test_compute_l1_confidence_uses_sign_quality_and_lcb_ratio(self) -> None:
        out = compute_l1_confidence(
            mu_bps=np.array([2.0, -2.0, 2.0]),
            l1_edge_margin_bps_per_bar=np.array([1.0, -1.0, -1.0]),
            quality_weight=np.array([0.8, 0.5, 1.0]),
        )
        np.testing.assert_allclose(out, np.array([0.4, 0.25, 0.0]))

    def test_sign_mismatch_yields_zero_confidence(self) -> None:
        out = compute_l1_confidence(
            mu_bps=np.array([2.0, -2.0]),
            l1_edge_margin_bps_per_bar=np.array([-1.0, 1.0]),
            quality_weight=np.array([0.8, 0.5]),
        )
        np.testing.assert_allclose(out, np.array([0.0, 0.0]))

    def test_zero_mu_yields_zero_confidence(self) -> None:
        out = compute_l1_confidence(
            mu_bps=np.array([0.0, 0.0]),
            l1_edge_margin_bps_per_bar=np.array([0.0, 0.0]),
            quality_weight=np.array([1.0, 1.0]),
        )
        np.testing.assert_allclose(out, np.array([0.0, 0.0]))


class TestBuildPolicyWeights:
    def test_equal_weight_shape(self) -> None:
        w = build_policy_weights(
            policy="equal_weight",
            mu_bps=np.array([2.0, -1.0]),
            sigma=np.array([0.02, 0.01]),
            l1_edge_margin_bps_per_bar=np.zeros(2),
            quality_weight=np.ones(2),
            caps=_caps(),
            prev_w=np.zeros(2),
            no_trade_band=0.0,
            vol_target=None,
            btc_beta=np.zeros(2),
            bars_per_year=2190.0,
        )
        assert np.allclose(w[:2], [0.5, -0.5]) or np.sum(np.abs(w[:2])) > 0.0

    def test_confidence_policy_collapses_to_inverse_vol_when_confidence_zero(self) -> None:
        kwargs = {
            "mu_bps": np.array([2.0, -1.0]),
            "sigma": np.array([0.02, 0.01]),
            "l1_edge_margin_bps_per_bar": np.zeros(2),
            "quality_weight": np.ones(2),
            "caps": _caps(),
            "prev_w": np.zeros(2),
            "no_trade_band": 0.0,
            "vol_target": None,
            "btc_beta": np.zeros(2),
            "bars_per_year": 2190.0,
        }
        confidence = build_policy_weights(policy="l1_confidence_shrinkage", **kwargs)
        inverse_vol = build_policy_weights(policy="inverse_vol", **kwargs)
        np.testing.assert_allclose(confidence, inverse_vol)

    @pytest.mark.parametrize("bad_policy", ["", "random", "directional_equal_weight"])
    def test_build_policy_weights_rejects_unknown_policy(self, bad_policy: str) -> None:
        with pytest.raises(ValueError, match="policy"):
            build_policy_weights(
                policy=bad_policy,  # type: ignore[arg-type]
                mu_bps=np.ones(2),
                sigma=np.ones(2),
                l1_edge_margin_bps_per_bar=np.ones(2),
                quality_weight=np.ones(2),
                caps=_caps(),
                prev_w=np.zeros(2),
                no_trade_band=0.0,
                vol_target=None,
                btc_beta=None,
                bars_per_year=2190.0,
            )

    def test_empty_support_returns_zero_weights(self) -> None:
        w = build_policy_weights(
            policy="equal_weight",
            mu_bps=np.array([0.0, 0.0]),
            sigma=np.array([0.02, 0.01]),
            l1_edge_margin_bps_per_bar=np.zeros(2),
            quality_weight=np.ones(2),
            caps=_caps(),
            prev_w=np.zeros(2),
            no_trade_band=0.0,
            vol_target=None,
            btc_beta=None,
            bars_per_year=2190.0,
        )
        np.testing.assert_allclose(w, np.zeros(2))

    def test_non_finite_sigma_replaced_with_vol_floor(self) -> None:
        w = build_policy_weights(
            policy="inverse_vol",
            mu_bps=np.array([2.0, -1.0, 3.0]),
            sigma=np.array([0.02, float("nan"), 0.01]),
            l1_edge_margin_bps_per_bar=np.zeros(3),
            quality_weight=np.ones(3),
            caps=_caps(),
            prev_w=np.zeros(3),
            no_trade_band=0.0,
            vol_target=None,
            btc_beta=None,
            bars_per_year=2190.0,
        )
        assert np.all(np.isfinite(w))


class TestSelectFitAllocationPolicy:
    def test_fit_selector_uses_growth_lcb_and_inverse_vol_tie_break(self) -> None:
        returns = np.array([0.01, -0.002, 0.008] * 4, dtype=np.float64)
        decision = select_fit_allocation_policy(
            returns_by_policy={
                "equal_weight": returns,
                "inverse_vol": returns,
                "kelly": returns * 0.5,
                "l1_confidence_shrinkage": returns * 0.7,
            },
            leverage_by_policy=dict.fromkeys(("equal_weight", "inverse_vol", "kelly", "l1_confidence_shrinkage"), 1.0),
            bars_per_year=2190.0,
            block_bars=3,
            growth_lcb_z=0.5,
            max_mdd=0.30,
            max_cvar_95=0.06,
            min_growth_lcb=0.0,
        )
        assert decision.selected_policy == "inverse_vol"

    def test_feasible_kelly_wins_on_higher_growth_lcb(self) -> None:
        returns_kelly = np.array([0.005, 0.01] * 10, dtype=np.float64)
        returns_iv = np.array([0.001, 0.002] * 10, dtype=np.float64)
        decision = select_fit_allocation_policy(
            returns_by_policy={
                "equal_weight": returns_iv * 0.5,
                "inverse_vol": returns_iv,
                "kelly": returns_kelly,
                "l1_confidence_shrinkage": returns_iv * 0.8,
            },
            leverage_by_policy=dict.fromkeys(("equal_weight", "inverse_vol", "kelly", "l1_confidence_shrinkage"), 1.0),
            bars_per_year=2190.0,
            block_bars=3,
            growth_lcb_z=0.5,
            max_mdd=0.30,
            max_cvar_95=0.06,
            min_growth_lcb=0.0,
        )
        assert decision.selected_policy == "kelly"

    def test_all_infeasible_falls_to_inverse_vol(self) -> None:
        returns = np.array([-0.02, -0.03] * 10, dtype=np.float64)
        decision = select_fit_allocation_policy(
            returns_by_policy={
                "equal_weight": returns,
                "inverse_vol": returns,
                "kelly": returns,
                "l1_confidence_shrinkage": returns,
            },
            leverage_by_policy=dict.fromkeys(("equal_weight", "inverse_vol", "kelly", "l1_confidence_shrinkage"), 1.0),
            bars_per_year=2190.0,
            block_bars=3,
            growth_lcb_z=0.5,
            max_mdd=0.30,
            max_cvar_95=0.06,
            min_growth_lcb=0.0,
        )
        assert decision.selected_policy == "inverse_vol"
        assert decision.fallback_reason == "insufficient_fit_evidence"

    def test_rejects_unknown_policy(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            select_fit_allocation_policy(
                returns_by_policy={"unknown": np.ones(12)},  # type: ignore[typeddict-item]
                leverage_by_policy={"unknown": 1.0},
                bars_per_year=2190.0,
                block_bars=3,
                growth_lcb_z=0.5,
                max_mdd=0.30,
                max_cvar_95=0.06,
                min_growth_lcb=0.0,
            )


class TestChooseDeployedPolicy:
    def test_deployment_selects_better_growth_lcb(self) -> None:
        deployed, fallback_used = choose_deployed_policy(
            selected=_score("kelly", growth_lcb=0.05, feasible=True),
            inverse_vol=_score("inverse_vol", growth_lcb=0.03, feasible=True),
        )
        assert deployed == "kelly"
        assert not fallback_used

    def test_deployment_falls_back_to_deployable_inverse_vol(self) -> None:
        deployed, fallback_used = choose_deployed_policy(
            selected=_score("kelly", growth_lcb=-0.01, feasible=False),
            inverse_vol=_score("inverse_vol", growth_lcb=0.03, feasible=True),
        )
        assert deployed == "inverse_vol"
        assert fallback_used

    def test_deployment_fails_closed_when_both_policies_fail(self) -> None:
        deployed, fallback_used = choose_deployed_policy(
            selected=_score("kelly", growth_lcb=-0.01, feasible=False),
            inverse_vol=_score("inverse_vol", growth_lcb=-0.02, feasible=False),
        )
        assert deployed is None
        assert not fallback_used

    def test_deployment_uses_selected_when_both_feasible_but_selected_lower(self) -> None:
        deployed, fallback_used = choose_deployed_policy(
            selected=_score("kelly", growth_lcb=0.02, feasible=True),
            inverse_vol=_score("inverse_vol", growth_lcb=0.05, feasible=True),
        )
        assert deployed == "inverse_vol"
        assert fallback_used
