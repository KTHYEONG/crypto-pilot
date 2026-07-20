"""L2 Feasibility-first compound growth implementation tests."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import optuna
import pytest

from src.domain.futures.optimization.l2_search_space import L2_SEARCH_SPACE
from src.domain.futures.optimization.workflow import (
    compute_l2_compound_growth_objective,
    count_active_turnover_blocks,
    summarize_layer2_feasibility,
)
from src.domain.futures.strategy.tiered_workflow.l2_gate import (
    Layer2ConstraintVector,
    evaluate_layer2_gate,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
    Layer2TrialEvaluation,
)


class TestCountActiveTurnoverBlocks:
    """[LIMIT-01] Rebalance/bar axis integrity."""

    def test_sparse_late_turnover(self) -> None:
        """Scenario 1: 3 monthly blocks, active rebalances only in block 2/3."""
        n = count_active_turnover_blocks(
            turnovers=(0.0, 0.2, 0.3),
            turnover_return_indices=(0, 800, 1500),
            n_returns=1800,
            block_bars=720,
        )
        assert n == 2

    def test_counts_blocks_not_turnovers(self) -> None:
        """2 turnovers in same block should count as 1 active block."""
        n = count_active_turnover_blocks(
            turnovers=(0.2, 0.3, 0.5),
            turnover_return_indices=(0, 50, 100),
            n_returns=300,
            block_bars=60,
        )
        assert n == 2  # block 0 (idx 0//60=0 and 50//60=0) and block 1 (idx 100//60=1) = 2 unique blocks

    def test_zero_turnover_returns_zero_blocks(self) -> None:
        n = count_active_turnover_blocks(
            turnovers=(0.0, 0.0),
            turnover_return_indices=(0, 100),
            n_returns=200,
            block_bars=100,
        )
        assert n == 0

    def test_length_mismatch_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            count_active_turnover_blocks(
                turnovers=(0.1,),
                turnover_return_indices=(0, 1, 2),
                n_returns=100,
                block_bars=10,
            )

    def test_out_of_range_index_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            count_active_turnover_blocks(
                turnovers=(0.1, 0.2),
                turnover_return_indices=(50, -1),
                n_returns=100,
                block_bars=10,
            )

    def test_non_monotonic_indices_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="not monotonic"):
            count_active_turnover_blocks(
                turnovers=(0.1, 0.2),
                turnover_return_indices=(50, 30),
                n_returns=100,
                block_bars=10,
            )

    def test_empty_turnovers(self) -> None:
        n = count_active_turnover_blocks(
            turnovers=(),
            turnover_return_indices=(),
            n_returns=100,
            block_bars=10,
        )
        assert n == 0

    def test_zero_block_bars(self) -> None:
        n = count_active_turnover_blocks(
            turnovers=(0.1,),
            turnover_return_indices=(0,),
            n_returns=100,
            block_bars=0,
        )
        assert n == 0


class TestComputeL2CompoundGrowthObjective:
    """[LIMIT-04] Stable growth objective."""

    def test_finite_lcb_returns_as_is(self) -> None:
        obj = compute_l2_compound_growth_objective(growth_lcb_deployed=0.15)
        assert obj == pytest.approx(0.15)

    def test_nan_returns_negative_1e6(self) -> None:
        obj = compute_l2_compound_growth_objective(growth_lcb_deployed=float("nan"))
        assert obj == pytest.approx(-1e6)

    def test_inf_returns_negative_1e6(self) -> None:
        obj = compute_l2_compound_growth_objective(growth_lcb_deployed=float("inf"))
        assert obj == pytest.approx(-1e6)

    def test_negative_finite(self) -> None:
        obj = compute_l2_compound_growth_objective(growth_lcb_deployed=-0.05)
        assert obj == pytest.approx(-0.05)


class TestLayer2ConstraintVector:
    """[LIMIT-07] Named constraint vector."""

    def test_as_tuple_13_slots(self) -> None:
        cv = Layer2ConstraintVector(
            deployment=-1.0, support_leak=0.0, mdd=0.1, cvar_95=-0.05,
            fold=-0.2, recent_fold=-0.1, active_blocks=0.0, friction=-0.1,
            trades=-2.0, crisis_mdd=-1.0, cagr=-0.3, sharpe_uplift=-0.2,
            crisis_cagr=-1.0, crisis_measured=True,
        )
        t = cv.as_tuple()
        assert len(t) == 13
        assert t[2] == pytest.approx(0.1)

    def test_as_mapping_serializes_crisis_measurement_state(self) -> None:
        cv = Layer2ConstraintVector(
            deployment=-1.0, support_leak=0.0, mdd=0.1, cvar_95=-0.05,
            fold=-0.2, recent_fold=-0.1, active_blocks=0.0, friction=-0.1,
            trades=-2.0, crisis_mdd=-1.0, cagr=-0.3, sharpe_uplift=-0.2,
            crisis_cagr=-1.0, crisis_measured=True,
        )
        m = cv.as_mapping()
        assert "crisis_mdd" in m
        assert "crisis_cagr" in m
        assert m["crisis_measured"] is True
        assert len(m) == 14

    def test_non_crisis_feasible_true_when_all_non_crisis_slots_leq_zero(self) -> None:
        cv = Layer2ConstraintVector(
            deployment=-1.0, support_leak=0.0, mdd=0.0, cvar_95=-0.05,
            fold=-0.2, recent_fold=-0.1, active_blocks=-3.0, friction=-0.1,
            trades=-2.0, crisis_mdd=1.0, cagr=-0.3, sharpe_uplift=-0.2,
            crisis_cagr=1.0, crisis_measured=True,
        )
        assert cv.non_crisis_feasible()

    def test_non_crisis_feasible_false_when_crisis_slot_violates(self) -> None:
        cv = Layer2ConstraintVector(
            deployment=-1.0, support_leak=0.0, mdd=0.1, cvar_95=-0.05,
            fold=-0.2, recent_fold=-0.1, active_blocks=-3.0, friction=-0.1,
            trades=-2.0, crisis_mdd=-1.0, cagr=-0.3, sharpe_uplift=-0.2,
            crisis_cagr=-1.0, crisis_measured=True,
        )
        assert not cv.non_crisis_feasible()

    def test_jointly_feasible_requires_crisis_measured_and_all_slots_leq_zero(self) -> None:
        cv = Layer2ConstraintVector(
            deployment=-1.0, support_leak=0.0, mdd=-0.1, cvar_95=-0.05,
            fold=-0.2, recent_fold=-0.1, active_blocks=-3.0, friction=-0.1,
            trades=-2.0, crisis_mdd=-1.0, cagr=-0.3, sharpe_uplift=-0.2,
            crisis_cagr=-1.0, crisis_measured=True,
        )
        assert cv.jointly_feasible()

    def test_jointly_feasible_false_when_crisis_not_measured(self) -> None:
        cv = Layer2ConstraintVector(
            deployment=-1.0, support_leak=0.0, mdd=-0.1, cvar_95=-0.05,
            fold=-0.2, recent_fold=-0.1, active_blocks=-3.0, friction=-0.1,
            trades=-2.0, crisis_mdd=-1.0, cagr=-0.3, sharpe_uplift=-0.2,
            crisis_cagr=-1.0, crisis_measured=False,
        )
        assert not cv.jointly_feasible()


class TestGateEvaluationConstraintVector:
    """evaluate_layer2_gate returns constraint_vector with correct crisis_measured."""

    _BASE_KWARGS: ClassVar[dict[str, Any]] = {
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

    def test_gate_returns_constraint_vector(self) -> None:
        gate = evaluate_layer2_gate(**self._BASE_KWARGS)
        assert hasattr(gate, "constraint_vector")
        assert gate.constraint_vector is not None
        assert isinstance(gate.constraint_vector, Layer2ConstraintVector)

    def test_gate_crisis_measured_true_when_crisis_params_provided(self) -> None:
        gate = evaluate_layer2_gate(
            **self._BASE_KWARGS,
            crisis_mdd_hybrid=0.15, crisis_mdd_budget=0.21,
            crisis_cagr_hybrid=-0.03, crisis_cagr_floor=-0.05,
        )
        assert gate.constraint_vector is not None
        assert gate.constraint_vector.crisis_measured

    def test_gate_crisis_measured_false_when_crisis_params_none(self) -> None:
        gate = evaluate_layer2_gate(**self._BASE_KWARGS)
        assert gate.constraint_vector is not None
        assert not gate.constraint_vector.crisis_measured

    def test_gate_jointly_feasible_true_when_all_pass(self) -> None:
        gate = evaluate_layer2_gate(
            **self._BASE_KWARGS,
            crisis_mdd_hybrid=0.15, crisis_mdd_budget=0.21,
            crisis_cagr_hybrid=-0.03, crisis_cagr_floor=-0.05,
        )
        assert gate.constraint_vector is not None
        assert gate.constraint_vector.jointly_feasible()

    def test_constraint_vector_as_tuple_matches_optuna_constraint_values(self) -> None:
        gate = evaluate_layer2_gate(**self._BASE_KWARGS)
        assert gate.constraint_vector is not None
        assert len(gate.constraint_vector.as_tuple()) == len(gate.optuna_constraint_values)
        for a, b in zip(gate.constraint_vector.as_tuple(), gate.optuna_constraint_values, strict=True):
            assert a == pytest.approx(b, abs=1e-12)


class TestLayer2FeasibilityAudit:
    """[LIMIT-07] Named constraint audit."""

    def _make_trial(self, number: int, state: Any, constraint_map: dict | None = None, **attrs: Any) -> MagicMock:
        t = MagicMock()
        t.number = number
        t.state = state
        t.user_attrs = dict(attrs)
        t.value = attrs.get("value", 0.0)
        if constraint_map is not None:
            t.user_attrs["l2_constraint_map"] = constraint_map
        if "l2_constraint_values" not in t.user_attrs:
            t.user_attrs["l2_constraint_values"] = [-1.0] * 13
        if "l2_crisis_measured" not in t.user_attrs:
            t.user_attrs["l2_crisis_measured"] = True
        return t

    def test_audit_counts_active_block_failures(self) -> None:
        trials = []
        for i in range(10):
            constraints = [-1.0] * 13
            if i < 9:
                constraints[6] = 1.0  # active_blocks violation
            t = self._make_trial(i, optuna.trial.TrialState.COMPLETE,
                                 l2_constraint_values=list(constraints),
                                 l2_crisis_measured=i < 9)
            trials.append(t)

        audit = summarize_layer2_feasibility(trials=trials, requested_trials=10)
        failure_map = dict(audit.failure_counts)
        assert failure_map.get("active_blocks", 0) == 9
        assert audit.completed_trials == 10
        assert audit.crisis_measured_trials == 9

    def test_audit_zero_feasible_logs_failure_histogram(self) -> None:
        trials = []
        for i in range(5):
            constraints = [1.0] * 13  # all fail
            t = self._make_trial(i, optuna.trial.TrialState.COMPLETE,
                                 l2_constraint_values=list(constraints),
                                 l2_crisis_measured=True)
            trials.append(t)

        audit = summarize_layer2_feasibility(trials=trials, requested_trials=5)
        assert audit.joint_feasible_trials == 0
        assert len(audit.failure_counts) == 13

    def test_audit_uses_trial_crisis_measurement_for_legacy_constraint_map(self) -> None:
        constraint_map = Layer2ConstraintVector(
            deployment=-1.0, support_leak=-1.0, mdd=-1.0, cvar_95=-1.0,
            fold=-1.0, recent_fold=-1.0, active_blocks=-1.0, friction=-1.0,
            trades=-1.0, crisis_mdd=-1.0, cagr=-1.0, sharpe_uplift=-1.0,
            crisis_cagr=-1.0, crisis_measured=True,
        ).as_mapping()
        constraint_map.pop("crisis_measured")
        trial = self._make_trial(
            0,
            optuna.trial.TrialState.COMPLETE,
            constraint_map=constraint_map,
            l2_crisis_measured=True,
        )

        audit = summarize_layer2_feasibility(trials=[trial], requested_trials=1)

        assert audit.crisis_measured_trials == 1
        assert audit.joint_feasible_trials == 1

    def test_audit_diagnostic_frontier_includes_minimal_failure_candidates(self) -> None:
        trials = []
        for i in range(6):
            constraints = [-1.0] * 13
            if i > 0:
                constraints[0] = 1.0  # deployment failure
            if i > 3:
                constraints[6] = 1.0  # active_blocks failure
            t = self._make_trial(i, optuna.trial.TrialState.COMPLETE,
                                 l2_constraint_values=list(constraints),
                                 l2_crisis_measured=True)
            trials.append(t)

        audit = summarize_layer2_feasibility(trials=trials, requested_trials=6)
        # trial[0] has 0 failures → should be first in frontier
        assert len(audit.diagnostic_frontier_trial_numbers) > 0
        assert audit.diagnostic_frontier_trial_numbers[0] == 0


class TestFeasibilityEarlyStopCallback:
    """[LIMIT-05] Feasibility-aware early stop."""

    @pytest.fixture
    def callback(self) -> Any:
        from src.application.futures.runner.active_pipeline import L2FeasibilityEarlyStopCallback
        return L2FeasibilityEarlyStopCallback(
            no_improve_limit=30,
            min_trials=60,
            min_joint_feasible_trials=5,
            requested_trials=120,
        )

    def _make_frozen_trial(self, number: int, value: float, joint_feasible: bool,
                           state: Any = optuna.trial.TrialState.COMPLETE) -> MagicMock:
        t = MagicMock()
        t.number = number
        t.value = value
        t.state = state
        t.user_attrs = {"l2_joint_feasible": joint_feasible}
        return t

    def test_zero_feasible_never_stops(self, callback: Any) -> None:
        study = MagicMock(spec=[])
        study._stop_flag = False
        study.trials = []
        total = 120
        for i in range(total):
            t = self._make_frozen_trial(i, 0.05, False)
            study.trials.append(t)
            callback(study, t)
        assert not study._stop_flag

    def test_minimum_joint_feasible_before_stop_allowed(self, callback: Any) -> None:
        study = MagicMock(spec=[])
        study._stop_flag = False
        study.trials = []
        for i in range(6):
            t = self._make_frozen_trial(i, 0.05, True)
            study.trials.append(t)
            callback(study, t)
        assert not study._stop_flag

    def test_stops_after_min_trials_and_improvement_gap(self, callback: Any) -> None:
        study = MagicMock(spec=[])
        study._stop_flag = False
        study.trials = []
        for i in range(5):
            t = self._make_frozen_trial(i, 0.10, True)
            study.trials.append(t)
            callback(study, t)
        for i in range(5, 65):
            t = self._make_frozen_trial(i, 0.05, True)
            study.trials.append(t)
            callback(study, t)
        assert study._stop_flag


class TestChampionSnapshot:
    """[LIMIT-08] Durable anchor."""

    @pytest.fixture
    def search_space(self) -> dict[str, Any]:
        return L2_SEARCH_SPACE

    def test_save_and_load_round_trip(self) -> None:
        from src.domain.futures.optimization.observability.run_tracker import (
            L2ChampionSnapshot,
            load_l2_champion_snapshot,
            save_l2_champion_snapshot,
            _compute_search_space_hash,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            snapshot = L2ChampionSnapshot(
                schema_version=1,
                tf="4h",
                search_space_hash=_compute_search_space_hash(L2_SEARCH_SPACE),
                params={"K_RANK": 3},
                growth_lcb_deployed=0.15,
                constraints={"mdd": -0.1},
                replay_fingerprint="abc123",
            )
            save_l2_champion_snapshot(path=path, snapshot=snapshot)
            loaded = load_l2_champion_snapshot(path=path, tf="4h", search_space=L2_SEARCH_SPACE)
            assert loaded is not None
            assert loaded.schema_version == 1
            assert loaded.tf == "4h"
            assert loaded.growth_lcb_deployed == pytest.approx(0.15)
            assert loaded.params["K_RANK"] == 3
        finally:
            os.unlink(path)

    def test_corrupt_json_returns_none(self) -> None:
        from src.domain.futures.optimization.observability.run_tracker import (
            load_l2_champion_snapshot,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("not json")
            path = Path(f.name)
        try:
            loaded = load_l2_champion_snapshot(path=path, tf="4h", search_space=L2_SEARCH_SPACE)
            assert loaded is None
        finally:
            os.unlink(path)

    def test_tf_mismatch_returns_none(self) -> None:
        from src.domain.futures.optimization.observability.run_tracker import (
            L2ChampionSnapshot,
            load_l2_champion_snapshot,
            save_l2_champion_snapshot,
            _compute_search_space_hash,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            snapshot = L2ChampionSnapshot(
                schema_version=1,
                tf="4h",
                search_space_hash=_compute_search_space_hash(L2_SEARCH_SPACE),
                params={},
                growth_lcb_deployed=0.1,
                constraints={},
                replay_fingerprint="x",
            )
            save_l2_champion_snapshot(path=path, snapshot=snapshot)
            loaded = load_l2_champion_snapshot(path=path, tf="1h", search_space=L2_SEARCH_SPACE)
            assert loaded is None
        finally:
            os.unlink(path)

    def test_search_space_hash_mismatch_returns_none(self) -> None:
        from src.domain.futures.optimization.observability.run_tracker import (
            L2ChampionSnapshot,
            load_l2_champion_snapshot,
            save_l2_champion_snapshot,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            snapshot = L2ChampionSnapshot(
                schema_version=1,
                tf="4h",
                search_space_hash="different_hash",
                params={},
                growth_lcb_deployed=0.1,
                constraints={},
                replay_fingerprint="x",
            )
            save_l2_champion_snapshot(path=path, snapshot=snapshot)
            loaded = load_l2_champion_snapshot(path=path, tf="4h", search_space=L2_SEARCH_SPACE)
            assert loaded is None
        finally:
            os.unlink(path)


class TestSearchSpaceRemovedWeight:
    """[LIMIT-04] l2_objective_growth_lcb_weight removed from search space."""

    def test_weight_not_in_search_space(self) -> None:
        assert "l2_objective_growth_lcb_weight" not in L2_SEARCH_SPACE


class TestLayer2TrialEvaluationNewFields:
    """New fields on Layer2TrialEvaluation."""

    def test_default_values(self) -> None:
        ev = Layer2TrialEvaluation(
            objective_value=0.05,
            constraint_values=(),
            cagr_hybrid=0.1,
            cagr_baseline=0.05,
            growth_lcb_hybrid=0.08,
            growth_lcb_baseline=0.04,
            sharpe_hac_hybrid=1.0,
            sharpe_hac_baseline=0.5,
            psr_hybrid=0.9,
            mdd_hybrid=0.1,
            cvar_95_hybrid=0.03,
            fold_pass_ratio=0.7,
            break_even_pass_pct=0.6,
            average_gross_exposure=1.0,
            cap_saturation_ratio=0.0,
            total_cost_bps=10.0,
            block_metrics=(),
        )
        assert ev.active_block_count == 0
        assert ev.block_log_growth_signature == ()
        assert ev.growth_lcb_deployed == float("-inf")
        assert not ev.crisis_constraints_measured

    def test_custom_values(self) -> None:
        ev = Layer2TrialEvaluation(
            objective_value=0.05,
            constraint_values=(),
            cagr_hybrid=0.1,
            cagr_baseline=0.05,
            growth_lcb_hybrid=0.08,
            growth_lcb_baseline=0.04,
            sharpe_hac_hybrid=1.0,
            sharpe_hac_baseline=0.5,
            psr_hybrid=0.9,
            mdd_hybrid=0.1,
            cvar_95_hybrid=0.03,
            fold_pass_ratio=0.7,
            break_even_pass_pct=0.6,
            average_gross_exposure=1.0,
            cap_saturation_ratio=0.0,
            total_cost_bps=10.0,
            block_metrics=(),
            active_block_count=5,
            block_log_growth_signature=(0.01, 0.02),
            growth_lcb_deployed=0.12,
            crisis_constraints_measured=True,
        )
        assert ev.active_block_count == 5
        assert ev.block_log_growth_signature == (0.01, 0.02)
        assert ev.growth_lcb_deployed == pytest.approx(0.12)
        assert ev.crisis_constraints_measured
