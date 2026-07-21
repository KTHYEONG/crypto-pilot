"""Scenarios 1-2: evaluate_layer2_gate crisis constraint."""
from __future__ import annotations

import pytest

from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig
from src.domain.futures.strategy.tiered_workflow.l2_gate import _cagr_gate_constraint, evaluate_layer2_gate

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
        assert gate_without.optuna_constraint_values[9] == pytest.approx(-1.0, abs=1e-6)
        assert gate_without.optuna_constraint_values[:9] == gate_with.optuna_constraint_values[:9]

    def test_evaluate_layer2_gate_optuna_constraints_include_cagr_and_uplift(self) -> None:
        """[S1] cagr>=0.30, sharpe_uplift>=0.05 -> optuna_constraint_values[10]<=0, [11]<=0."""
        kwargs = dict(_BASE_KWARGS, cagr_hybrid=0.35, sharpe_hac_hybrid=1.2, sharpe_hac_baseline=1.0)
        gate = evaluate_layer2_gate(**kwargs)

        assert len(gate.optuna_constraint_values) == 14
        assert gate.optuna_constraint_values[10] <= 0.0
        assert gate.optuna_constraint_values[11] <= 0.0

    def test_evaluate_layer2_gate_optuna_constraints_flag_cagr_violation(self) -> None:
        """[S1] cagr_hybrid=0.10 (<0.30) -> optuna_constraint_values[10] > 0.0 (infeasible)."""
        kwargs = dict(_BASE_KWARGS, cagr_hybrid=0.10, sharpe_hac_hybrid=1.2, sharpe_hac_baseline=1.0)
        gate = evaluate_layer2_gate(**kwargs)

        assert len(gate.optuna_constraint_values) == 14
        assert gate.optuna_constraint_values[10] > 0.0

    def test_evaluate_layer2_gate_crisis_cagr_below_floor_marks_infeasible(self) -> None:
        """[S1] crisis_cagr_hybrid < crisis_cagr_floor -> 13th slot > 0 (infeasible)."""
        gate = evaluate_layer2_gate(
            **_BASE_KWARGS,
            crisis_mdd_hybrid=0.15, crisis_mdd_budget=0.21,
            crisis_cagr_hybrid=-0.15, crisis_cagr_floor=-0.05,
        )

        assert len(gate.optuna_constraint_values) == 14
        assert gate.optuna_constraint_values[12] > 0.0

    def test_evaluate_layer2_gate_crisis_cagr_none_defaults_to_feasible(self) -> None:
        """[S2] crisis_cagr_hybrid=None -> 13th slot -1.0 (auto feasible)."""
        gate = evaluate_layer2_gate(
            **_BASE_KWARGS,
            crisis_mdd_hybrid=0.15, crisis_mdd_budget=0.21,
            crisis_cagr_hybrid=None, crisis_cagr_floor=None,
        )

        assert len(gate.optuna_constraint_values) == 14
        assert gate.optuna_constraint_values[12] == pytest.approx(-1.0, abs=1e-6)

    def test_evaluate_layer2_gate_crisis_cagr_above_floor_feasible(self) -> None:
        """[S1] crisis_cagr_hybrid >= crisis_cagr_floor -> 13th slot <= 0 (feasible)."""
        gate = evaluate_layer2_gate(
            **_BASE_KWARGS,
            crisis_mdd_hybrid=0.15, crisis_mdd_budget=0.21,
            crisis_cagr_hybrid=-0.03, crisis_cagr_floor=-0.05,
        )

        assert len(gate.optuna_constraint_values) == 14
        assert gate.optuna_constraint_values[12] <= 0.0


class TestCagrGateConstraint:
    @pytest.mark.parametrize(
        ("baseline", "cagr", "should_block"),
        [
            (0.10, 0.12, True),
            (0.10, 0.20, False),
            (-0.20, -0.01, True),
            (-0.20, 0.01, False),
        ],
    )
    def test_cagr_gate_constraint_relative_mode_uses_baseline_plus_uplift(self, baseline: float, cagr: float, should_block: bool) -> None:
        value = _cagr_gate_constraint(
            cagr_hybrid=cagr, cagr_baseline=baseline, mode="relative",
            l2_min_cagr=0.30, l2_min_cagr_uplift=0.05,
        )
        assert (value > 0.0) is should_block

    def test_cagr_gate_constraint_none_baseline_falls_back_to_absolute(self) -> None:
        value = _cagr_gate_constraint(
            cagr_hybrid=0.20, cagr_baseline=None, mode="relative",
            l2_min_cagr=0.30, l2_min_cagr_uplift=0.05,
        )
        assert value == pytest.approx(0.10)


class TestLayer2AllocationConfigCagrGateMode:
    def test_layer2_allocation_config_rejects_invalid_cagr_gate_mode(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.dataclasses import (
            _validate_cagr_gate_mode,
        )
        with pytest.raises(ValueError, match="l2_cagr_gate_mode"):
            _validate_cagr_gate_mode("invalid")


class TestEvaluateLayer2GateRelativeCagr:
    def test_evaluate_layer2_gate_relative_mode_wired_from_config(self) -> None:
        kwargs = dict(_BASE_KWARGS,
            cagr_hybrid=0.12,
            cagr_baseline=0.10,
            config=Layer2AllocationConfig(
                l2_cagr_gate_mode="relative",
                l2_min_cagr_uplift=0.05,
                l2_min_cagr=0.30,
            ),
        )
        gate = evaluate_layer2_gate(**kwargs)

        assert gate.optuna_constraint_values[10] > 0.0
        assert gate.promotion_constraint_values[2] > 0.0

    def test_evaluate_layer2_gate_relative_mode_high_cagr_passes(self) -> None:
        kwargs = dict(_BASE_KWARGS,
            cagr_hybrid=0.20,
            cagr_baseline=0.10,
            config=Layer2AllocationConfig(
                l2_cagr_gate_mode="relative",
                l2_min_cagr_uplift=0.05,
                l2_min_cagr=0.30,
            ),
        )
        gate = evaluate_layer2_gate(**kwargs)

        assert gate.optuna_constraint_values[10] <= 0.0
        assert gate.promotion_constraint_values[2] <= 0.0
