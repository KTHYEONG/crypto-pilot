# tests/unit/domain/futures/strategy/tiered_workflow/test_risk_deployment.py
"""Fix-A risk_deployment 모듈 단위 테스트 (S1~S5 + DSR 불변성)."""
from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
    DeploymentResult,
    _cvar_95_at_leverage,
    _mdd_at_leverage,
    apply_deployment,
    calibrate_deployment_leverage,
)

BARS_PER_YEAR = 2190.0  # 4h 기준


def _make_rets(n: int, mu_bps: float = 6.4, sigma_bps: float = 16.0, seed: int = 0) -> np.ndarray:
    """재현 가능한 per-bar 수익률 생성. sigma_bps=16은 4h 현실적 변동성(≈연율 7.5%)."""
    rng = np.random.default_rng(seed)
    return rng.normal(mu_bps * 1e-4, sigma_bps * 1e-4, n).astype(np.float64)


# ---------------------------------------------------------------------------
# S1: 저변동 → L*=hard_cap (MDD가 낮아 레버리지 예산을 꽉 채움)
# ---------------------------------------------------------------------------
class TestHappyPathHardCap:
    def test_binding_hard_cap(self) -> None:
        """MDD≈6% 경로 → mdd_target=21%, hard_cap=4 → binding=hard_cap."""
        # Arrange: sigma=16 bps, MDD(L=1)≈4-6% ≪ mdd_target=21%
        rets = _make_rets(2190, sigma_bps=16.0)

        # Act
        lev, binding, _ = calibrate_deployment_leverage(
            fit_rets=rets,
            mdd_cap=0.30,
            cvar_cap=0.06,
            mdd_margin=0.30,
            l_hard_cap=4.0,
        )

        # Assert: MDD(L=1)≈6% < 21% → 탐색 결과가 hard_cap에 clip
        assert lev == pytest.approx(4.0, rel=1e-3)
        assert binding == "hard_cap"

    def test_apply_cagr_increases_with_leverage(self) -> None:
        """L=4 → L=1 대비 CAGR 상승 (양의 mu 가정)."""
        # Arrange
        rets = _make_rets(2190, sigma_bps=16.0, seed=1)
        base = apply_deployment(rets=rets, leverage=1.0, bars_per_year=BARS_PER_YEAR)
        lev4 = apply_deployment(rets=rets, leverage=4.0, bars_per_year=BARS_PER_YEAR)

        # Assert
        assert lev4.cagr > base.cagr
        assert lev4.mdd > base.mdd
        assert lev4.leverage == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# S2: MDD-binding (l_hard_cap이 충분히 커서 MDD가 먼저 바인딩됨)
# ---------------------------------------------------------------------------
class TestMddBinding:
    def test_binding_mdd(self) -> None:
        """MDD 제약이 CVaR보다 먼저 바인딩 (mdd_target << cvar_target)."""
        # Arrange: seed=42에서 MDD(1)≈1.21%, CVaR(1)≈0.28%
        #   mdd_target = 0.10*0.40 = 0.06 → l_mdd ≈ 0.06/0.0121 ≈ 5.0
        #   cvar_target = 0.10 (느슨) → l_cvar ≈ 0.10/0.0028 ≈ 35
        #   → l* = min(5, 35) = 5 → binding="mdd"
        rets = _make_rets(2190, sigma_bps=16.0, seed=42)
        mdd_cap = 0.10
        mdd_margin = 0.40
        mdd_target = mdd_cap * (1 - mdd_margin)  # 0.06

        # Act
        lev, binding, _ = calibrate_deployment_leverage(
            fit_rets=rets,
            mdd_cap=mdd_cap,
            cvar_cap=0.10,   # 느슨한 CVaR 제약
            mdd_margin=mdd_margin,
            cvar_margin=0.0,
            l_hard_cap=20.0,
        )

        # Assert: MDD 바인딩, 실제 MDD ≈ 목표
        actual_mdd = _mdd_at_leverage(rets, lev)
        assert binding == "mdd"
        assert actual_mdd == pytest.approx(mdd_target, rel=0.08)  # ±8%


# ---------------------------------------------------------------------------
# S3: 스케일 불변성 — DSR은 L에 무관 (핵심 수치 증명)
# ---------------------------------------------------------------------------
class TestScaleInvariance:
    def test_sharpe_hac_invariant(self) -> None:
        """L=1,2,4 에서 Sharpe_HAC 동일."""
        from src.domain.futures.strategy.tiered_workflow.metrics import _sharpe_hac

        rets = _make_rets(2190, seed=7)

        sharpes = [
            _sharpe_hac(rets * lev, bars_per_year=BARS_PER_YEAR)
            for lev in [1.0, 2.0, 4.0]
        ]

        assert sharpes[0] == pytest.approx(sharpes[1], rel=1e-5)
        assert sharpes[0] == pytest.approx(sharpes[2], rel=1e-5)

    def test_dsr_invariant_to_leverage(self) -> None:
        """동일 pool에서 L=1,2,4 에 대해 DSR 수치 동일."""
        from src.domain.futures.strategy.tiered_workflow.metrics import _deflated_sharpe_probability

        rets = _make_rets(2190, seed=11)
        pool = np.array([1.0, 1.2, 1.5, 1.8, 2.0], dtype=np.float64)

        dsrs = [
            _deflated_sharpe_probability(
                selected_rets=(rets * lev).tolist(),
                completed_trial_sharpes=pool,
                effective_trial_count=5.0,
                bars_per_year=BARS_PER_YEAR,
            )
            for lev in [1.0, 2.0, 4.0]
        ]

        assert dsrs[0] == pytest.approx(dsrs[1], rel=1e-3)
        assert dsrs[0] == pytest.approx(dsrs[2], rel=1e-3)


# ---------------------------------------------------------------------------
# S4: 엣지 케이스
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_rets(self) -> None:
        lev, binding, _ = calibrate_deployment_leverage(fit_rets=np.array([], dtype=np.float64))
        assert lev == pytest.approx(1.0)
        assert binding == "none"

    def test_single_element(self) -> None:
        lev, binding, _ = calibrate_deployment_leverage(fit_rets=np.array([0.01], dtype=np.float64))
        assert lev == pytest.approx(1.0)
        assert binding == "none"

    def test_zero_rets_safe(self) -> None:
        """전부 0 → MDD=0, CVaR=0 → hard_cap 반환."""
        rets = np.zeros(500, dtype=np.float64)
        lev, _, _ = calibrate_deployment_leverage(fit_rets=rets, l_hard_cap=4.0)
        assert 1.0 <= lev <= 4.0 + 1e-6

    def test_apply_deployment_result_type(self) -> None:
        rets = _make_rets(500)
        result = apply_deployment(rets=rets, leverage=2.0, bars_per_year=BARS_PER_YEAR)
        assert isinstance(result, DeploymentResult)
        assert result.leverage == pytest.approx(2.0)
        assert result.mdd >= 0.0
        assert result.cvar_95 >= 0.0
        assert result.scaled_rets.shape == (500,)


# ---------------------------------------------------------------------------
# S5: CVaR-binding (tight cvar_cap + loose mdd_cap)
# ---------------------------------------------------------------------------
class TestCvarBinding:
    def test_cvar_binding_tight_cap(self) -> None:
        """cvar_cap 극도로 낮게, mdd_cap 높게 → CVaR가 먼저 바인딩."""
        # Arrange: sigma=16 bps, CVaR(1)≈26 bps(=0.26%), tight cvar_target=0.4%
        rng = np.random.default_rng(99)
        rets = rng.normal(1e-4, 0.0016, 2190).astype(np.float64)

        mdd_cap = 0.80   # 매우 느슨한 MDD 제약
        cvar_cap = 0.004  # 매우 타이트 CVaR 제약 (0.4%)
        mdd_margin = 0.0
        cvar_margin = 0.0

        # Act
        lev, binding, _ = calibrate_deployment_leverage(
            fit_rets=rets,
            mdd_cap=mdd_cap,
            cvar_cap=cvar_cap,
            mdd_margin=mdd_margin,
            cvar_margin=cvar_margin,
            l_hard_cap=20.0,
        )

        # CVaR(lev) ≈ cvar_target=0.4%
        actual_cvar = _cvar_95_at_leverage(rets, lev)
        cvar_target = cvar_cap  # margin=0

        # Assert
        assert binding == "cvar"
        assert actual_cvar == pytest.approx(cvar_target, rel=0.08)


# ---------------------------------------------------------------------------
# S6: oos_rets 크로스 검증 — L* inflation 감지
# ---------------------------------------------------------------------------
class TestCalibrateWithOosCrossValidation:
    def test_oos_rets_not_provided_returns_zero_cross_valid_mdd(self) -> None:
        """oos_rets 미제공 시 cross_valid_mdd=0.0 반환."""
        rets = _make_rets(2190, sigma_bps=16.0, seed=42)
        l_star, binding, cv_mdd = calibrate_deployment_leverage(
            fit_rets=rets,
            mdd_cap=0.30,
            mdd_margin=0.30,
            l_hard_cap=20.0,
        )
        assert cv_mdd == pytest.approx(0.0)
        assert l_star >= 1.0
        assert binding in ("mdd", "hard_cap", "cvar")

    def test_oos_mdd_greater_than_fit_mdd_detected(self) -> None:
        """fit 보다 OOS MDD가 클 때 inflation 감지 시나리오.

        fit: 저변동(6bps sigma) → fit_MDD_at_L1 작음 → L* 큼.
        OOS: 고변동(40bps sigma) → oos_deployed_MDD가 cap 초과 예상.
        """
        rng_fit = np.random.default_rng(100)
        fit_rets = rng_fit.normal(6.4e-4, 6e-4, 2190).astype(np.float64)
        rng_oos = np.random.default_rng(101)
        oos_rets = rng_oos.normal(6.4e-4, 40e-4, 2190).astype(np.float64)

        _l_star, _binding, cv_mdd = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            oos_rets=oos_rets,
            mdd_cap=0.30,
            mdd_margin=0.30,
            l_hard_cap=20.0,
        )
        assert cv_mdd > 0.0
        # fit-MDD가 매우 낮아 L*는 hard_cap에 걸림
        assert _binding in ("hard_cap", "mdd")

    def test_oos_and_fit_similar_produces_reasonable_cv_mdd(self) -> None:
        """fit과 OOS 분포 유사 → cv_mdd ≈ mdd_target."""
        rets = _make_rets(2190, sigma_bps=16.0, seed=42)
        _, _, cv_mdd = calibrate_deployment_leverage(
            fit_rets=rets,
            oos_rets=rets,
            mdd_cap=0.30,
            mdd_margin=0.30,
            l_hard_cap=20.0,
        )
        mdd_target = 0.30 * (1.0 - 0.30)
        assert cv_mdd == pytest.approx(mdd_target, rel=0.15)

    def test_empty_oos_rets_does_not_crash(self) -> None:
        """빈 oos_rets도 안전하게 처리."""
        rets = _make_rets(2190, seed=42)
        _, _, cv_mdd = calibrate_deployment_leverage(
            fit_rets=rets,
            oos_rets=np.array([], dtype=np.float64),
            mdd_cap=0.30,
            mdd_margin=0.30,
            l_hard_cap=20.0,
        )
        assert cv_mdd == pytest.approx(0.0)
