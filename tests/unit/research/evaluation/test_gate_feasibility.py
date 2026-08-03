"""RGR: pre-backtest gate feasibility screen tests.

Covers the closed-form sizing feasibility screen and the breadth requirement
diagnostic against the measured panel (RGR-06 through RGR-08).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.research.evaluation.gate_feasibility import (
    BreadthRequirement,
    GateFeasibility,
    compute_breadth_requirement,
    compute_gate_feasibility,
)


class TestGateFeasibilityContract:
    def test_gate_feasibility_is_a_frozen_slots_dataclass(self) -> None:
        assert dataclasses.is_dataclass(GateFeasibility)
        assert dataclasses.is_dataclass(BreadthRequirement)


class TestFeasibilityMatchesMeasuredLeg:
    """RGR-06-FEASIBILITY-MATCHES-MEASURED-LEG"""

    def test_measured_ichimoku_leg_is_reproduced(self) -> None:
        r = compute_gate_feasibility(
            sharpe=1.158, vol=0.336, mdd_at_unit_leverage=-0.245, years=3.58,
        )
        assert abs(r.leverage_lcb_optimal - 1.43) < 0.15
        assert r.leverage_mdd_cap < 1.6
        assert r.max_lcb90_achievable < 0.15
        assert r.feasible is False

    def test_high_sharpe_low_vol_leg_is_feasible(self) -> None:
        r = compute_gate_feasibility(
            sharpe=2.0, vol=0.10, mdd_at_unit_leverage=-0.05, years=3.58,
        )
        assert r.feasible is True
        assert r.binding_constraint == "none"


class TestFeasibilityFailClosed:
    """RGR-07-FEASIBILITY-FAIL-CLOSED"""

    def test_compute_gate_feasibility_rejects_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="vol"):
            compute_gate_feasibility(1.0, 0.0, -0.1, 3.58)
        with pytest.raises(ValueError, match="years"):
            compute_gate_feasibility(1.0, 0.2, -0.1, 0.0)
        with pytest.raises(ValueError, match="mdd_at_unit_leverage"):
            compute_gate_feasibility(1.0, 0.2, 0.0, 3.58)
        with pytest.raises(ValueError, match="mdd_at_unit_leverage"):
            compute_gate_feasibility(1.0, 0.2, -1.0, 3.58)

    def test_compute_breadth_requirement_rejects_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="leg_sharpes"):
            compute_breadth_requirement([], 0.0, 3.58)
        with pytest.raises(ValueError, match="correlation"):
            compute_breadth_requirement([0.1, 0.2], -1.0, 3.58)


class TestBreadthRequirementMatchesMeasuredPanel:
    """RGR-08-BREADTH-REQUIREMENT-MATCHES-MEASURED-PANEL"""

    def test_measured_80_leg_panel_diagnosis_is_reproduced(self) -> None:
        leg_sharpes = list(np.full(80, -0.143))
        r = compute_breadth_requirement(leg_sharpes, 0.0433, 3.58)

        assert r.n_legs == 80
        assert abs(r.breadth_multiplier - 4.25) < 0.15
        assert abs(r.required_mean_leg_sharpe - 0.288) < 0.02
        assert r.achievable_portfolio_sharpe < 0.0
        assert r.sufficient is False
