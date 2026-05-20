"""Phase 2: ergodicity deviation gate 테스트.

기존 wf_path_ergodicity_deviation_pct 수식:
    deviation_pct = max(|TW_i - mean|) / mean * 100
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.domain.futures.optimization.validation import wf_path_ergodicity_deviation_pct


def _ergodicity_deviation_pct_reference(tw_legs: list[float]) -> float:
    """테스트 내부 검증용 — validation.py 실제 수식과 동일."""
    arr = np.asarray(tw_legs, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    m = float(np.mean(arr))
    if m < 1e-12:
        return 0.0
    return float(np.max(np.abs(arr - m)) / m * 100.0)


class TestErgodicityGate:
    """ergodicity deviation gate 검증."""

    def test_low_volatility_small_deviation_pass(self) -> None:
        """낮은 변동성 → deviation 작음 → 15% gate 통과."""
        tw_legs = [1.05, 1.04, 1.06, 1.05, 1.05, 1.04, 1.06, 1.05]
        dev_pct = _ergodicity_deviation_pct_reference(tw_legs)

        assert dev_pct < 15.0, f"낮은 변동성에서 deviation이 작아야 함: {dev_pct:.2f}%"

    def test_high_volatility_large_deviation_fail(self) -> None:
        """높은 변동성 (큰 손실 leg 포함) → deviation 큼 → gate 실패 (>15%).

        TW=[1.0, 0.50]: mean=0.75, max_dev=0.25, dev_pct=33.3%
        """
        tw_legs = [1.0, 0.50]  # mean=0.75, max_dev=0.25 → dev_pct=33.3%
        dev_pct = _ergodicity_deviation_pct_reference(tw_legs)

        assert dev_pct > 15.0, f"높은 변동성에서 deviation이 커야 함: {dev_pct:.2f}%"

    def test_deviation_formula_known_example(self) -> None:
        """deviation 정의 수식 검증 (known example).

        TW = [1.0, 0.5]: mean = 0.75
        max|TW_i - mean| = 0.25
        deviation_pct = 0.25 / 0.75 * 100 ≈ 33.33%
        """
        tw_legs = [1.0, 0.5]
        mean_tw = 0.75
        expected_dev = 0.25 / mean_tw * 100.0
        actual_dev = _ergodicity_deviation_pct_reference(tw_legs)
        assert abs(actual_dev - expected_dev) < 1e-9, (
            f"deviation 수식 불일치: actual={actual_dev:.4f}, expected={expected_dev:.4f}"
        )

    def test_wf_path_ergodicity_consistent_with_formula(self) -> None:
        """wf_path_ergodicity_deviation_pct가 내부 수식과 일치."""
        tw_legs_vals = [1.10, 0.85, 1.05, 1.08, 1.02, 0.90, 1.12, 1.07]
        expected_dev = _ergodicity_deviation_pct_reference(tw_legs_vals)
        actual_dev = wf_path_ergodicity_deviation_pct(tw_legs_vals)
        assert abs(actual_dev - expected_dev) < 1e-6, (
            f"ergodicity 함수 불일치: actual={actual_dev:.4f}, expected={expected_dev:.4f}"
        )

    def test_all_tw_equal_deviation_zero(self) -> None:
        """모든 TW가 동일하면 deviation = 0."""
        tw_legs = [1.05] * 8
        dev_pct = _ergodicity_deviation_pct_reference(tw_legs)
        assert dev_pct < 1e-9, f"동일 TW에서 deviation은 0이어야 함: {dev_pct}"
