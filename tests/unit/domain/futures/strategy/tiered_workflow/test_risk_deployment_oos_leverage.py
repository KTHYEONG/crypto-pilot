# tests/unit/domain/futures/strategy/tiered_workflow/test_risk_deployment_oos_leverage.py
"""RC-2: OOS-aware leverage calibrate_deployment_leverage 테스트 (S1~S4)."""
from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
    _mdd_at_leverage,
    calibrate_deployment_leverage,
)

BARS_PER_YEAR = 2190.0


# ---------------------------------------------------------------------------
# S1 (Happy — fit/OOS 역전 시 OOS 예산 사용)
# ---------------------------------------------------------------------------
class TestOosBlendWhenInverted:
    def test_calibrate_leverage_uses_oos_budget_when_fit_inverted(self) -> None:
        """fit Sharpe<0, OOS Sharpe>0, OOS MDD << fit MDD → L* > 2.0, binding=oos_blend."""
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

        assert lev > 2.0, f"L*={lev:.4f} should exceed magic cap 2.0"
        oos_mdd_at_l = _mdd_at_leverage(oos_rets, lev)
        invariant = 0.30 * (1.0 - 0.30 * 0.5)
        assert oos_mdd_at_l <= invariant + 1e-4, (
            f"OOS MDD at L*={oos_mdd_at_l:.6f} > invariant={invariant:.6f}"
        )
        assert binding == "oos_blend", f"binding={binding} != oos_blend"


# ---------------------------------------------------------------------------
# S2 (Bounds — exchange_cap 우선, blend가 cap 초과 시 clip)
# ---------------------------------------------------------------------------
class TestExchangeCapOverOosBlend:
    def test_calibrate_leverage_respects_exchange_cap_over_oos_floor(self) -> None:
        """OOS blend가 exchange_cap=3.0 초과 시 cap으로 clip, binding=exchange_cap."""
        rng = np.random.default_rng(42)
        # fit: 고변동·음드리프트 → fit L* 낮음
        fit_rets = rng.normal(-0.002, 0.03, 600).astype(np.float64)
        # OOS: 극저변동 → l_oos 매우 높음 → blend가 exchange_cap 초과
        oos_rets = rng.normal(+0.0006, 0.003, 560).astype(np.float64)

        lev, binding, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            oos_rets=oos_rets,
            mdd_cap=0.30,
            mdd_margin=0.30,
            l_hard_cap=20.0,
            exchange_leverage_cap=3.0,
            oos_floor_cap=4.0,
        )

        assert binding == "exchange_cap", f"binding={binding}, lev={lev:.4f}"
        assert lev == pytest.approx(3.0, rel=1e-3), f"lev={lev:.4f}"


# ---------------------------------------------------------------------------
# S3 (Edge — fit이 OOS보다 안전, 기존 동작 보존)
# ---------------------------------------------------------------------------
class TestFitCalibrationPreserved:
    def test_calibrate_leverage_preserves_fit_calibration_when_not_inverted(self) -> None:
        """fit MDD < OOS MDD → 기존 fit-기준 동작, binding∈{mdd,cvar,hard_cap}."""
        rng = np.random.default_rng(42)
        fit_rets = rng.normal(0.0006, 0.006, 600).astype(np.float64)
        oos_rets = rng.normal(-0.002, 0.03, 560).astype(np.float64)

        _, binding, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            oos_rets=oos_rets,
            mdd_cap=0.30,
            mdd_margin=0.30,
            l_hard_cap=20.0,
        )

        assert binding in ("mdd", "cvar", "hard_cap", "none")
        _fit_mdd_1 = _mdd_at_leverage(fit_rets, 1.0)
        _oos_mdd_1 = _mdd_at_leverage(oos_rets, 1.0)
        assert _fit_mdd_1 < _oos_mdd_1


# ---------------------------------------------------------------------------
# S4 (Error — oos_rets None → fit-only fallback)
# ---------------------------------------------------------------------------
class TestFallbackWithoutOos:
    def test_calibrate_leverage_without_oos_falls_back_to_fit(self) -> None:
        """oos_rets=None → fit-only 경로, binding≠oos_blend, cross_valid_mdd=0."""
        rng = np.random.default_rng(42)
        fit_rets = rng.normal(-0.002, 0.03, 600).astype(np.float64)

        lev, binding, cv_mdd = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            oos_rets=None,
            mdd_cap=0.30,
            mdd_margin=0.30,
            l_hard_cap=20.0,
        )

        assert cv_mdd == pytest.approx(0.0)
        assert binding != "oos_blend"
        assert lev >= 1.0
