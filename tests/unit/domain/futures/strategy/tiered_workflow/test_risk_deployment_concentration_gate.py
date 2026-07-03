"""calibrate_deployment_leverage concentration gate TDD tests (Scenarios B1-B5, C1-C2).

Conventions follow test_risk_deployment_oos_leverage.py:
np.random.default_rng(42), BARS_PER_YEAR=2190.0, 순수 함수(No mock).
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
    calibrate_deployment_leverage,
)

BARS_PER_YEAR = 2190.0


def _make_dr_series(
    n_bars: int, dr_early: float, dr_tail: float, tail_frac: float
) -> NDArray[np.float64]:
    n_tail = int(n_bars * tail_frac)
    return np.concatenate(
        [
            np.full(n_bars - n_tail, dr_early, dtype=np.float64),
            np.full(n_tail, dr_tail, dtype=np.float64),
        ]
    )


class TestConcentrationGate:
    """B scenarios: concentration gate haircut logic."""

    def test_haircuts_linearly_on_dr_collapse(self) -> None:
        """B1: 상관 클러스터링 감지 시 L* 선형 haircut."""
        rng = np.random.default_rng(42)
        fit_rets = rng.normal(0.0005, 0.01, 600).astype(np.float64)
        dr_fit = _make_dr_series(600, dr_early=2.0, dr_tail=0.5, tail_frac=0.1)

        lev_base, _binding_base, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            diversification_gate_enabled=False,
        )

        lev_gated, binding_gated, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            diversification_ratio_fit=dr_fit,
            diversification_gate_enabled=True,
            concentration_recent_window_bars=60,
            concentration_floor=0.15,
        )

        assert binding_gated == "concentration_gate"
        assert lev_gated < lev_base
        # dr_recent/dr_fit_median = 0.5/2.0 = 0.25, floor=0.15보다 큼 → 0.25
        assert lev_gated == pytest.approx(lev_base * 0.25, rel=1e-4)

    def test_noop_when_dr_stable_or_rising(self) -> None:
        """B2: DR 안정/상승 시 게이트 미개입."""
        rng = np.random.default_rng(42)
        fit_rets = rng.normal(0.0005, 0.01, 600).astype(np.float64)
        dr_fit = _make_dr_series(600, dr_early=1.5, dr_tail=1.8, tail_frac=0.1)

        lev_base, _binding_base, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            diversification_gate_enabled=False,
        )

        lev_gated, binding_gated, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            diversification_ratio_fit=dr_fit,
            diversification_gate_enabled=True,
            concentration_recent_window_bars=60,
            concentration_floor=0.15,
        )

        assert lev_gated == pytest.approx(lev_base, rel=1e-6)
        assert binding_gated != "concentration_gate"

    def test_clips_at_floor(self) -> None:
        """B3: concentration_floor 하한 클립."""
        rng = np.random.default_rng(42)
        fit_rets = rng.normal(0.0005, 0.01, 600).astype(np.float64)
        # raw ratio = 0.02/2.0 = 0.01 → floor=0.25에서 클립
        dr_fit = _make_dr_series(600, dr_early=2.0, dr_tail=0.02, tail_frac=0.1)

        lev_base, _, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            diversification_gate_enabled=False,
        )

        lev_gated, binding_gated, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            diversification_ratio_fit=dr_fit,
            diversification_gate_enabled=True,
            concentration_recent_window_bars=60,
            concentration_floor=0.25,
        )

        assert binding_gated == "concentration_gate"
        assert lev_gated == pytest.approx(lev_base * 0.25, rel=1e-4)

    def test_disabled_by_default_matches_existing_behavior(self) -> None:
        """B4: 기본값 시 기존 동작 완전 보존 (S1 fixture)."""
        rng = np.random.default_rng(42)
        fit_rets = rng.normal(-0.002, 0.03, 600).astype(np.float64)
        oos_rets = rng.normal(+0.0006, 0.006, 560).astype(np.float64)

        lev, binding, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            oos_rets=oos_rets,
            mdd_cap=0.30,
            mdd_margin=0.30,
            l_hard_cap=20.0,
            oos_floor_cap=4.0,
        )

        assert lev > 2.0
        assert binding == "oos_blend"

    def test_crisis_gate_and_concentration_gate_compose(self) -> None:
        """B5: crisis_gate와 concentration_gate 동시 활성 시 순차 합성."""
        rng = np.random.default_rng(42)
        # 고-MDD fit 시계열 → crisis_gate 발동
        fit_rets = rng.normal(-0.003, 0.04, 600).astype(np.float64)
        oos_rets = rng.normal(+0.0006, 0.006, 560).astype(np.float64)
        dr_fit = _make_dr_series(600, dr_early=2.0, dr_tail=0.5, tail_frac=0.1)

        _lev_gated, binding_gated, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            oos_rets=oos_rets,
            mdd_cap=0.30,
            mdd_margin=0.30,
            l_hard_cap=20.0,
            oos_floor_cap=4.0,
            fit_mdd_crisis_gate=0.90,
            diversification_ratio_fit=dr_fit,
            diversification_gate_enabled=True,
            concentration_recent_window_bars=60,
            concentration_floor=0.15,
        )

        assert binding_gated == "concentration_gate"

    def test_skips_when_history_insufficient(self) -> None:
        """C1: 표본 부족 시 게이트 skip, 조용히 no-op."""
        rng = np.random.default_rng(42)
        fit_rets = rng.normal(0.0005, 0.01, 600).astype(np.float64)
        dr_fit = np.array([1.0, 1.5], dtype=np.float64)  # size=2 < window=60

        lev_base, _, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            diversification_gate_enabled=False,
        )

        lev_gated, binding_gated, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            diversification_ratio_fit=dr_fit,
            diversification_gate_enabled=True,
            concentration_recent_window_bars=60,
            concentration_floor=0.15,
        )

        assert lev_gated == pytest.approx(lev_base, rel=1e-6)
        assert binding_gated != "concentration_gate"

    def test_raises_when_gate_enabled_without_explicit_floor(self) -> None:
        """C2: floor 미명시 시 ValueError."""
        rng = np.random.default_rng(42)
        fit_rets = rng.normal(0.0005, 0.01, 600).astype(np.float64)

        with pytest.raises(ValueError, match="concentration_floor must be explicitly set"):
            calibrate_deployment_leverage(
                fit_rets=fit_rets,
                diversification_ratio_fit=_make_dr_series(600, 2.0, 0.5, 0.1),
                diversification_gate_enabled=True,
                concentration_recent_window_bars=60,
                concentration_floor=None,
            )
