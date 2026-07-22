"""Scenarios 1-2, 6: evaluate_layer2_gate crisis constraint & absolute growth gate."""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig
from src.domain.futures.strategy.tiered_workflow.l2_gate import _absolute_growth_constraint, evaluate_layer2_gate

_BASE_KWARGS = {
    "deployment_failed": False,
    "support_leak_count": 0,
    "cagr_hybrid": 0.30,
    "sharpe_hybrid": 1.0,
    "sharpe_hac_hybrid": 1.0,
    "sharpe_hac_baseline": 0.5,
    "sortino_hybrid": 1.5,
    "mar_hybrid": 2.0,
    "mdd_hybrid": 0.10,
    "cvar_95_hybrid": 0.03,
    "fold_pass_ratio": 0.80,
    "active_block_count": 10,
    "friction_pass_pct": 0.90,
    "trade_count": 50,
    "growth_lcb_hybrid": 0.20,
    "growth_lcb_baseline": 0.10,
    "growth_lcb_deployed": 0.20,
    "dsr_hybrid": None,
    "psr_hybrid": 0.95,
    "recent_fold_passed": True,
    "recent_fold_sharpe": 1.0,
    "worst_fold_cagr": -0.02,
    "positive_block_delta_ratio": 0.60,
    "fold_attributions": (),
    "config": Layer2AllocationConfig(),
}


class TestEvaluateLayer2GateCrisisConstraint:
    def test_evaluate_layer2_gate_crisis_constraint_violated_when_mdd_exceeds_budget(self) -> None:
        """[S1] crisis_mdd_hybrid > crisis_mdd_budget -> 10th slot 양수(위반)."""
        gate = evaluate_layer2_gate(**_BASE_KWARGS, crisis_mdd_hybrid=0.30, crisis_mdd_budget=0.21)

        assert len(gate.optuna_constraint_values) == 14
        assert gate.optuna_constraint_values[9] == pytest.approx(0.09, abs=1e-6)

    def test_evaluate_layer2_gate_crisis_constraint_absent_defaults_to_satisfied(self) -> None:
        """[S2] crisis_mdd_hybrid=None -> 10th slot -1.0(항상 만족), 기존 9개 제약 동일."""
        gate_with = evaluate_layer2_gate(**_BASE_KWARGS, crisis_mdd_hybrid=0.30, crisis_mdd_budget=0.21)
        gate_without = evaluate_layer2_gate(**_BASE_KWARGS, crisis_mdd_hybrid=None, crisis_mdd_budget=None)

        assert len(gate_without.optuna_constraint_values) == 14
        assert gate_without.optuna_constraint_values[9] > 0.0

    def test_evaluate_layer2_gate_optuna_constraints_include_growth_lcb(self) -> None:
        """[S1] growth_lcb >= 0 -> slot 5 <= 0."""
        kwargs = dict(_BASE_KWARGS, cagr_hybrid=0.35, growth_lcb_deployed=0.05, sharpe_hac_hybrid=1.2)
        gate = evaluate_layer2_gate(**kwargs)

        assert len(gate.optuna_constraint_values) == 14
        assert gate.optuna_constraint_values[5] <= 0.0

    def test_evaluate_layer2_gate_optuna_constraints_flag_growth_lcb_violation(self) -> None:
        """[S1] growth_lcb_deployed < min -> slot 5 > 0."""
        kwargs = dict(
            _BASE_KWARGS,
            cagr_hybrid=0.10,
            growth_lcb_deployed=-0.01,
            config=replace(Layer2AllocationConfig(), l2_min_absolute_cagr=0.0, l2_min_growth_lcb=0.0),
        )
        gate = evaluate_layer2_gate(**kwargs)

        assert len(gate.optuna_constraint_values) == 14
        assert gate.optuna_constraint_values[5] > 0.0

    def test_evaluate_layer2_gate_crisis_mdd_above_budget_marks_infeasible(self) -> None:
        """[S1] crisis_mdd > budget -> slot 9 > 0 (infeasible)."""
        gate = evaluate_layer2_gate(
            **_BASE_KWARGS,
            crisis_mdd_hybrid=0.25, crisis_mdd_budget=0.21,
            crisis_cagr_hybrid=-0.15, crisis_cagr_floor=-0.05,
        )

        assert len(gate.optuna_constraint_values) == 14
        assert gate.optuna_constraint_values[9] > 0.0

    def test_evaluate_layer2_gate_crisis_unmeasured_fails_closed(self) -> None:
        """[S2] crisis_mdd=None -> slot 9 fails closed (1.0)."""
        gate = evaluate_layer2_gate(
            **_BASE_KWARGS,
            crisis_mdd_hybrid=None, crisis_mdd_budget=None,
            crisis_cagr_hybrid=None, crisis_cagr_floor=None,
        )

        assert len(gate.optuna_constraint_values) == 14
        assert gate.optuna_constraint_values[9] > 0.0

    def test_evaluate_layer2_gate_crisis_mdd_within_budget_feasible(self) -> None:
        """[S1] crisis_mdd <= budget -> slot 9 <= 0 (feasible)."""
        gate = evaluate_layer2_gate(
            **_BASE_KWARGS,
            crisis_mdd_hybrid=0.15, crisis_mdd_budget=0.21,
            crisis_cagr_hybrid=-0.03, crisis_cagr_floor=-0.05,
        )

        assert len(gate.optuna_constraint_values) == 14
        assert gate.optuna_constraint_values[9] <= 0.0


class TestAbsoluteGrowthConstraint:
    def test_absolute_growth_passes_when_both_above_floor(self) -> None:
        value = _absolute_growth_constraint(
            cagr_hybrid=0.10, growth_lcb_hybrid=0.02,
            l2_min_absolute_cagr=0.0, l2_min_growth_lcb=0.0,
        )
        assert value <= 0.0

    def test_absolute_growth_fails_when_growth_lcb_below_floor(self) -> None:
        value = _absolute_growth_constraint(
            cagr_hybrid=0.10, growth_lcb_hybrid=-0.01,
            l2_min_absolute_cagr=0.0, l2_min_growth_lcb=0.0,
        )
        assert value > 0.0

    def test_absolute_growth_fails_when_cagr_below_floor(self) -> None:
        value = _absolute_growth_constraint(
            cagr_hybrid=-0.01, growth_lcb_hybrid=0.02,
            l2_min_absolute_cagr=0.0, l2_min_growth_lcb=0.0,
        )
        assert value > 0.0


class TestAbsoluteGrowthGate:
    def test_absolute_growth_gate_ignores_relative_uplift(self) -> None:
        config = replace(
            Layer2AllocationConfig(),
            l2_min_absolute_cagr=0.0,
            l2_min_growth_lcb=0.0,
            l2_min_cagr_uplift=0.50,
            l2_min_sharpe_uplift=2.0,
        )
        gate = evaluate_layer2_gate(
            deployment_failed=False,
            support_leak_count=0,
            cagr_hybrid=0.06,
            sharpe_hybrid=1.0,
            sharpe_hac_hybrid=1.0,
            sharpe_hac_baseline=1.2,
            sortino_hybrid=2.0,
            mar_hybrid=1.0,
            mdd_hybrid=0.06,
            cvar_95_hybrid=0.02,
            fold_pass_ratio=1.0,
            active_block_count=4,
            friction_pass_pct=1.0,
            trade_count=50,
            growth_lcb_hybrid=0.01,
            growth_lcb_baseline=0.02,
            growth_lcb_deployed=0.01,
            dsr_hybrid=None,
            psr_hybrid=0.99,
            recent_fold_passed=True,
            recent_fold_sharpe=1.0,
            worst_fold_cagr=0.0,
            positive_block_delta_ratio=0.5,
            fold_attributions=(),
            config=config,
        )
        assert gate.optuna_constraint_values[10] <= 0.0
        assert gate.optuna_constraint_values[11] <= 0.0


class TestLayer2AllocationConfigCagrGateMode:
    def test_layer2_allocation_config_rejects_invalid_cagr_gate_mode(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.dataclasses import (
            _validate_cagr_gate_mode,
        )
        with pytest.raises(ValueError, match="l2_cagr_gate_mode"):
            _validate_cagr_gate_mode("invalid")
