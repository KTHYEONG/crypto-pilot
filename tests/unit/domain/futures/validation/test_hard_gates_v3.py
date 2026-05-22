"""Phase 3: V3HardGates 8-gate 체계 테스트.

사양서 §5.2 — 각 gate 독립 검증 (나머지는 통과, 해당 gate만 경계값 테스트).
"""

from __future__ import annotations

import numpy as np

from src.domain.futures.validation.gates import (
    evaluate_v3_hard_gates,
)


def _all_pass_inputs() -> dict:
    """모든 gate를 통과하는 기본 입력값."""
    return {
        "leg_log_tw": np.array([0.04, 0.06, 0.03, 0.05, 0.07, 0.04, 0.05, 0.06]),
        "worst_mdd": 0.20,
        "dsr": 0.65,
        "ev_cost": 4.0,
        "funding_drag_ratio": 0.20,
        "ergodicity_dev_pct": 10.0,
        "capacity_results": {50_000: True, 100_000: True, 250_000: True},
    }


class TestV3HardGates:
    """8-gate 체계 독립 검증."""

    def test_gate1_min_positive_leg_ratio_fail(self) -> None:
        """4/8 = 0.50 < 0.55 → FAIL."""
        inp = _all_pass_inputs()
        # 4개 양수, 4개 음수
        inp["leg_log_tw"] = np.array([0.04, -0.01, 0.03, -0.02, 0.05, -0.01, 0.04, -0.02])
        result = evaluate_v3_hard_gates(**inp)
        assert not result.passed
        assert "WF_POSITIVE_LEG_RATIO" in result.failures

    def test_gate1_min_positive_leg_ratio_pass(self) -> None:
        """5/8 = 0.625 > 0.55 → PASS."""
        inp = _all_pass_inputs()
        # 5개 양수, 3개 음수
        inp["leg_log_tw"] = np.array([0.04, -0.01, 0.03, 0.05, 0.07, -0.01, 0.05, -0.02])
        result = evaluate_v3_hard_gates(**inp)
        assert "WF_POSITIVE_LEG_RATIO" not in result.failures

    def test_gate2_worst_leg_tw_floor_fail(self) -> None:
        """Worst leg TW = exp(-0.17) ≈ 0.844 < 0.85 → FAIL."""
        inp = _all_pass_inputs()
        worst_log = np.log(0.844)
        inp["leg_log_tw"] = np.array([0.04, 0.06, 0.03, 0.05, worst_log, 0.04, 0.05, 0.06])
        result = evaluate_v3_hard_gates(**inp)
        assert not result.passed
        assert "WF_WORST_LEG_TW" in result.failures

    def test_gate2_worst_leg_tw_floor_pass(self) -> None:
        """Worst leg TW = exp(log(0.86)) ≈ 0.86 > 0.85 → PASS."""
        inp = _all_pass_inputs()
        worst_log = np.log(0.86)
        inp["leg_log_tw"] = np.array([0.04, 0.06, 0.03, 0.05, worst_log, 0.04, 0.05, 0.06])
        result = evaluate_v3_hard_gates(**inp)
        assert "WF_WORST_LEG_TW" not in result.failures

    def test_gate3_mean_leg_tw_floor_fail(self) -> None:
        """Mean TW = 1.014 < 1.015 → FAIL."""
        inp = _all_pass_inputs()
        # mean(exp(log_tw)) = 1.014 → mean_log = log(1.014) 근사치
        target_mean_log = np.log(1.014)
        inp["leg_log_tw"] = np.full(8, target_mean_log, dtype=np.float64)
        result = evaluate_v3_hard_gates(**inp)
        assert not result.passed
        assert "WF_MEAN_LEG_TW" in result.failures

    def test_gate3_mean_leg_tw_floor_pass(self) -> None:
        """Mean TW = 1.016 > 1.015 → PASS."""
        inp = _all_pass_inputs()
        target_mean_log = np.log(1.016)
        inp["leg_log_tw"] = np.full(8, target_mean_log, dtype=np.float64)
        result = evaluate_v3_hard_gates(**inp)
        assert "WF_MEAN_LEG_TW" not in result.failures

    def test_gate4_dsr_floor_fail(self) -> None:
        """DSR = 0.59 < 0.60 → FAIL."""
        inp = _all_pass_inputs()
        inp["dsr"] = 0.59
        result = evaluate_v3_hard_gates(**inp)
        assert not result.passed
        assert "DSR_FLOOR" in result.failures

    def test_gate4_dsr_floor_pass(self) -> None:
        """DSR = 0.61 > 0.60 → PASS."""
        inp = _all_pass_inputs()
        inp["dsr"] = 0.61
        result = evaluate_v3_hard_gates(**inp)
        assert "DSR_FLOOR" not in result.failures

    def test_gate5_funding_drag_ceiling_fail(self) -> None:
        """drag/return = 0.31 > 0.30 → FAIL."""
        inp = _all_pass_inputs()
        inp["funding_drag_ratio"] = 0.31
        result = evaluate_v3_hard_gates(**inp)
        assert not result.passed
        assert "FUNDING_DRAG" in result.failures

    def test_gate5_funding_drag_ceiling_pass(self) -> None:
        """drag/return = 0.29 < 0.30 → PASS."""
        inp = _all_pass_inputs()
        inp["funding_drag_ratio"] = 0.29
        result = evaluate_v3_hard_gates(**inp)
        assert "FUNDING_DRAG" not in result.failures

    def test_gate6_capacity_partial_fail(self) -> None:
        """50k pass, 100k FAIL → FAIL (3개 전부 필요)."""
        inp = _all_pass_inputs()
        inp["capacity_results"] = {50_000: True, 100_000: False, 250_000: False}
        result = evaluate_v3_hard_gates(**inp)
        assert not result.passed
        assert "CAPACITY" in result.failures

    def test_gate6_capacity_all_pass(self) -> None:
        """50k pass, 100k pass, 250k pass → PASS."""
        inp = _all_pass_inputs()
        inp["capacity_results"] = {50_000: True, 100_000: True, 250_000: True}
        result = evaluate_v3_hard_gates(**inp)
        assert "CAPACITY" not in result.failures

    def test_all_gates_pass_result_passed_true(self) -> None:
        """모든 gate 통과 시 GateResult.passed == True."""
        inp = _all_pass_inputs()
        result = evaluate_v3_hard_gates(**inp)
        assert result.passed
        assert result.failures == []

    def test_multiple_gate_failures_all_in_list(self) -> None:
        """복수 gate 실패 시 failures 리스트에 모두 포함."""
        inp = _all_pass_inputs()
        inp["dsr"] = 0.50  # DSR FAIL
        inp["funding_drag_ratio"] = 0.40  # FUNDING FAIL
        inp["capacity_results"] = {50_000: True, 100_000: False, 250_000: False}  # CAPACITY FAIL

        result = evaluate_v3_hard_gates(**inp)
        assert not result.passed
        assert len(result.failures) >= 2
        assert "DSR_FLOOR" in result.failures
        assert "FUNDING_DRAG" in result.failures

    def test_gate_result_metrics_populated(self) -> None:
        """GateResult.metrics에 주요 지표가 포함되어야 함."""
        inp = _all_pass_inputs()
        result = evaluate_v3_hard_gates(**inp)
        assert isinstance(result.metrics, dict)
        assert len(result.metrics) > 0
