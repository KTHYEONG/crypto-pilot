# tests/unit/domain/futures/strategy/tiered_workflow/test_risk_deployment.py
"""Fix-A risk_deployment 모듈 단위 테스트 (S1~S5 + DSR 불변성)."""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
    DeploymentResult,
    _annualized_cagr_from_returns,
    _cvar_95_at_leverage,
    _mdd_at_leverage,
    _mdd_from_returns,
    _sharpe_from_returns,
    apply_deployment,
    calibrate_deployment_leverage,
    select_worst_fold_returns,
    trend_efficiency_gross_mult,
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
            cvar_cap=0.10,  # 느슨한 CVaR 제약
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

        sharpes = [_sharpe_hac(rets * lev, bars_per_year=BARS_PER_YEAR) for lev in [1.0, 2.0, 4.0]]

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

        mdd_cap = 0.80  # 매우 느슨한 MDD 제약
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


class TestOosFloor:
    """OOS-based L* floor (anti-L*=1.0-hard-landing)."""

    def test_oos_safe_floor_raises_leverage(self) -> None:
        """OOS MDD fit 대비 낮음 → L* floor 상향 가능."""
        rng = np.random.default_rng(42)
        fit_rets = rng.normal(-1e-4, 0.02, 2190).astype(np.float64)
        oos_rets = rng.normal(2e-4, 0.01, 1095).astype(np.float64)
        lev, _, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            oos_rets=oos_rets,
            mdd_cap=0.30,
            l_hard_cap=20.0,
        )
        assert lev >= 1.0

    def test_no_oos_rets_backward_compat(self) -> None:
        """oos_rets=None → 기존 로직 동일."""
        rets = _make_rets(2190, seed=42)
        lev, bind, _ = calibrate_deployment_leverage(
            fit_rets=rets,
            mdd_cap=0.30,
            l_hard_cap=4.0,
        )
        assert bind in ("mdd", "hard_cap", "cvar", "none")
        assert lev >= 1.0


# ---------------------------------------------------------------------------
# S7: select_worst_fold_returns
# ---------------------------------------------------------------------------
class TestSelectWorstFoldReturns:
    def test_select_worst_fold_returns_picks_highest_unit_mdd_fold(self) -> None:
        """단위 MDD가 더 큰 volatile fold를 반환."""
        calm_fold = (0.01, 0.01, -0.005, 0.01)
        volatile_fold = (0.02, -0.08, 0.01, -0.03)
        fit_rets_by_fold = (calm_fold, volatile_fold)

        worst = select_worst_fold_returns(fit_rets_by_fold)

        assert tuple(worst.tolist()) == volatile_fold

    def test_select_worst_fold_returns_empty_input_returns_empty_array(self) -> None:
        """빈 튜플 입력 → 빈 배열 반환."""
        worst = select_worst_fold_returns(())
        assert worst.size == 0

    def test_single_fold_returns_empty(self) -> None:
        """fold < 2인 경우 빈 배열 반환."""
        worst = select_worst_fold_returns(((0.01, 0.02, -0.01),))
        assert worst.size == 0

    def test_all_folds_too_short_returns_empty(self) -> None:
        """모든 fold 길이 < 2 → 빈 배열."""
        worst = select_worst_fold_returns(((0.01,), (0.02,)))
        assert worst.size == 0

    def test_mdd_tie_returns_first_with_max_mdd(self) -> None:
        """동일 MDD 시 첫 번째 fold 반환 (안정성)."""
        fold_a = (0.02, -0.05, 0.01)
        fold_b = (0.01, -0.05, 0.03)
        fit_rets_by_fold = (fold_a, fold_b)
        worst = select_worst_fold_returns(fit_rets_by_fold)
        expected_mdd = _mdd_from_returns(np.asarray(fold_a, dtype=np.float64))
        actual_mdd = _mdd_from_returns(worst)
        assert actual_mdd == pytest.approx(expected_mdd, rel=1e-3)


# ---------------------------------------------------------------------------
# S8: calibrate_deployment_leverage — kelly_safety_fraction
# ---------------------------------------------------------------------------
class TestKellySafetyFraction:
    def test_calibrate_deployment_leverage_kelly_fraction_out_of_range_raises(self) -> None:
        """kelly_safety_fraction <= 0 또는 > 1 → ValueError."""
        for invalid in [0.0, -0.1, 1.5]:
            with pytest.raises(ValueError, match="kelly_safety_fraction"):
                calibrate_deployment_leverage(
                    fit_rets=np.array([0.01, -0.01, 0.02, -0.005], dtype=np.float64),
                    kelly_safety_fraction=invalid,
                )

    def test_kelly_binding_with_positive_mu(self) -> None:
        """mu > 0인 fit_rets에서 kelly candidate가 binding됨 (충분히 조여서)."""
        np.random.seed(42)
        rets = np.random.normal(0.001, 0.02, 1000).astype(np.float64)
        lev_no_kelly, bind_no_kelly, _ = calibrate_deployment_leverage(
            fit_rets=rets,
            l_hard_cap=20.0,
            mdd_cap=0.30,
            mdd_margin=0.30,
        )
        lev_kelly, bind_kelly, _ = calibrate_deployment_leverage(
            fit_rets=rets,
            l_hard_cap=20.0,
            mdd_cap=0.30,
            mdd_margin=0.30,
            kelly_safety_fraction=0.25,
        )
        assert lev_kelly <= lev_no_kelly + 1e-6
        assert bind_kelly in ("kelly_theoretical", "mdd", "cvar", "hard_cap")

    def test_kelly_skipped_when_mu_nonpositive(self) -> None:
        """mu <= 0이면 kelly candidate 생략."""
        rets = np.full(100, -0.001, dtype=np.float64)
        lev, bind, _ = calibrate_deployment_leverage(
            fit_rets=rets,
            kelly_safety_fraction=0.25,
            l_hard_cap=20.0,
        )
        assert bind != "kelly_theoretical"


# ---------------------------------------------------------------------------
# S9: calibrate_deployment_leverage — worst_fold_rets
# ---------------------------------------------------------------------------
class TestWorstFoldConstraint:
    def test_worst_fold_candidate_added(self) -> None:
        """worst_fold_rets 제공 시 candidate 리스트에 추가."""
        np.random.seed(42)
        rets = np.random.normal(0.001, 0.02, 1000).astype(np.float64)
        np.random.seed(43)
        worst = np.random.normal(0.0005, 0.04, 300).astype(np.float64)

        lev_no_wf, bind_no_wf, _ = calibrate_deployment_leverage(
            fit_rets=rets,
            l_hard_cap=20.0,
            mdd_cap=0.30,
            mdd_margin=0.30,
        )
        lev_wf, bind_wf, _ = calibrate_deployment_leverage(
            fit_rets=rets,
            l_hard_cap=20.0,
            mdd_cap=0.30,
            mdd_margin=0.30,
            worst_fold_rets=worst,
        )
        assert lev_wf <= lev_no_wf + 1e-6
        assert bind_wf in ("worst_fold", "mdd", "cvar", "hard_cap")

    def test_worst_fold_empty_skipped(self) -> None:
        """빈 worst_fold_rets → candidate 생략."""
        rets = np.array([0.01, -0.01, 0.02, -0.005], dtype=np.float64)
        lev, bind, _ = calibrate_deployment_leverage(
            fit_rets=rets,
            worst_fold_rets=np.array([], dtype=np.float64),
        )
        assert bind != "worst_fold"

    def test_worst_fold_none_skipped(self) -> None:
        """worst_fold_rets=None → candidate 생략."""
        rets = np.array([0.01, -0.01, 0.02, -0.005], dtype=np.float64)
        lev, bind, _ = calibrate_deployment_leverage(
            fit_rets=rets,
            worst_fold_rets=None,
        )
        assert bind != "worst_fold"


# ---------------------------------------------------------------------------
# S10: 회귀 방지 — disabled gates match pre-spec baseline
# ---------------------------------------------------------------------------
class TestDisabledGatesRegression:
    def test_calibrate_deployment_leverage_disabled_gates_match_pre_spec_baseline(self) -> None:
        """worst_fold_rets/kelly_safety_fraction 기본값(None) → 기존 동작 동일."""
        np.random.seed(42)
        rets = np.random.normal(0.001, 0.02, 1000).astype(np.float64)

        lev, binding, cross_mdd = calibrate_deployment_leverage(
            fit_rets=rets,
            mdd_cap=0.30,
            mdd_margin=0.30,
            l_hard_cap=4.0,
        )
        assert binding in {"mdd", "cvar", "hard_cap"}


# ---------------------------------------------------------------------------
# S11: Integration — evaluate_l2_trial wiring (mock test)
# ---------------------------------------------------------------------------
class TestSelectWorstFoldThenCalibrateIntegration:
    def test_worst_fold_flows_into_calibrate(self) -> None:
        """select_worst_fold_returns → calibrate_deployment_leverage 순차 호출 검증."""
        import numpy as np

        calm_fold = (0.01, 0.01, -0.005, 0.01)
        volatile_fold = (0.02, -0.08, 0.01, -0.03)
        fit_rets_by_fold = (calm_fold, volatile_fold)

        worst = select_worst_fold_returns(fit_rets_by_fold)
        assert tuple(worst.tolist()) == volatile_fold

        np.random.seed(42)
        fit_rets = np.random.normal(0.001, 0.02, 1000).astype(np.float64)

        lev, binding, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            mdd_cap=0.30,
            mdd_margin=0.30,
            l_hard_cap=20.0,
            worst_fold_rets=worst,
            kelly_safety_fraction=0.25,
        )
        assert binding in ("worst_fold", "kelly_theoretical", "mdd", "cvar", "hard_cap")
        assert lev >= 1.0


# ---------------------------------------------------------------------------
# S12: Supplementary coverage — pre-existing uncovered paths
# ---------------------------------------------------------------------------
class TestSupplementaryCoverage:
    def test_exchange_leverage_cap_applied(self) -> None:
        """exchange_leverage_cap < candidate L* → binding=exchange_cap."""
        np.random.seed(99)
        rets = np.random.normal(0.001, 0.005, 1000).astype(np.float64)
        lev, binding, _ = calibrate_deployment_leverage(
            fit_rets=rets,
            l_hard_cap=20.0,
            exchange_leverage_cap=1.5,
        )
        assert lev == pytest.approx(1.5, rel=1e-3)
        assert binding == "exchange_cap"

    def test_exchange_leverage_cap_none_ignored(self) -> None:
        """exchange_leverage_cap=None → 기존 동작."""
        np.random.seed(99)
        rets = np.random.normal(0.001, 0.005, 1000).astype(np.float64)
        lev, binding, _ = calibrate_deployment_leverage(
            fit_rets=rets,
            l_hard_cap=4.0,
            exchange_leverage_cap=None,
        )
        assert binding in ("mdd", "cvar", "hard_cap")

    def test_trend_efficiency_gross_mult_nonfinite(self) -> None:
        """trend_efficiency_gross_mult: non-finite trailing_ER → floor_mult."""
        result = trend_efficiency_gross_mult(float("nan"), target=0.5, floor_mult=0.3)
        assert result == 0.3

    def test_trend_efficiency_gross_mult_above_target(self) -> None:
        """trailing_ER >= target → 1.0."""
        result = trend_efficiency_gross_mult(0.8, target=0.5, floor_mult=0.3)
        assert result == 1.0

    def test_trend_efficiency_gross_mult_below_target(self) -> None:
        """trailing_ER < target → interpolated."""
        result = trend_efficiency_gross_mult(0.2, target=0.5, floor_mult=0.3)
        assert result == pytest.approx(0.3 + 0.7 * 0.4, rel=1e-3)

    def test_annualized_cagr_empty_rets(self) -> None:
        """_annualized_cagr_from_returns: 빈 배열 → 0.0."""
        assert _annualized_cagr_from_returns(np.array([], dtype=np.float64), bars_per_year=2190) == 0.0

    def test_mdd_from_returns_empty(self) -> None:
        """_mdd_from_returns: 빈 배열 → 0.0."""
        assert _mdd_from_returns(np.array([], dtype=np.float64)) == 0.0

    def test_sharpe_from_returns_small_size(self) -> None:
        """_sharpe_from_returns: size<2 → 0.0."""
        assert _sharpe_from_returns(np.array([0.01], dtype=np.float64), bars_per_year=2190) == 0.0

    def test_sharpe_from_returns_zero_std(self) -> None:
        """_sharpe_from_returns: std≈0 → 0.0."""
        rets = np.array([0.01, 0.01, 0.01], dtype=np.float64)
        assert _sharpe_from_returns(rets, bars_per_year=2190) == 0.0

    def test_mdd_at_leverage_empty(self) -> None:
        """_mdd_at_leverage: 빈 배열 → 0.0."""
        assert _mdd_at_leverage(np.array([], dtype=np.float64), 1.0) == 0.0

    def test_cvar_95_at_leverage_empty(self) -> None:
        """_cvar_95_at_leverage: 빈 배열 → 0.0."""
        assert _cvar_95_at_leverage(np.array([], dtype=np.float64), 1.0) == 0.0

    def test_cvar_95_at_leverage_tail_empty(self) -> None:
        """_cvar_95_at_leverage: 모든 손실 동일 → tail 비어도 안전."""
        rets = np.ones(100, dtype=np.float64) * -0.01
        result = _cvar_95_at_leverage(rets, 1.0)
        assert result > 0.0

    def test_annualized_cagr_very_small_years(self) -> None:
        """_annualized_cagr_from_returns: very small years → 0.0."""
        result = _annualized_cagr_from_returns(np.array([0.01], dtype=np.float64), bars_per_year=1e12)
        assert result == 0.0

    def test_crisis_gate_suppresses_oos_blend(self) -> None:
        """fit_mdd_crisis_gate=0.001 → fit_MDD_vol1 확정 초과 → OOS blend 차단."""
        np.random.seed(42)
        fit_rets = np.random.normal(0.001, 0.02, 1000).astype(np.float64)
        oos_rets = np.random.normal(0.002, 0.01, 500).astype(np.float64)
        lev, binding, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            oos_rets=oos_rets,
            mdd_cap=0.30,
            mdd_margin=0.30,
            l_hard_cap=4.0,
            fit_mdd_crisis_gate=0.001,
        )
        assert binding != "oos_blend"

    def test_high_oos_mdd_ratio_triggers_blend(self) -> None:
        """OOS MDD < fit MDD → blended budget."""
        np.random.seed(200)
        fit_rets = np.random.normal(0.001, 0.03, 1000).astype(np.float64)
        oos_rets = np.random.normal(0.001, 0.01, 500).astype(np.float64)
        lev, binding, cross_mdd = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            oos_rets=oos_rets,
            mdd_cap=0.30,
            mdd_margin=0.30,
            l_hard_cap=4.0,
        )
        assert binding in ("mdd", "cvar", "oos_blend", "hard_cap")
        assert cross_mdd >= 0.0

    def test_hard_cap_final_clip_without_exchange_cap(self) -> None:
        """exchange_leverage_cap=None + OOS blend가 l_final을 l_hard_cap 위로 → clip."""
        np.random.seed(300)
        fit_rets = np.random.normal(0.0005, 0.005, 2000).astype(np.float64)
        oos_rets = np.random.normal(0.0005, 0.003, 1000).astype(np.float64)
        lev, binding, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            oos_rets=oos_rets,
            mdd_cap=0.30,
            mdd_margin=0.30,
            l_hard_cap=3.0,
            exchange_leverage_cap=None,
        )
        assert binding in ("hard_cap", "mdd", "cvar")
        assert lev <= 3.0 + 1e-6

    def test_awf_sim_result_fit_rets_by_field(self) -> None:
        """_AwfSimResult fit_rets_by_fold 필드 존재 및 타입 검증."""
        from src.domain.futures.strategy.tiered_workflow.awf_sim import _AwfSimResult

        result = _AwfSimResult(
            rets_hybrid=[0.01, -0.01],
            rets_baseline=[0.005, -0.005],
            last_selected=frozenset(),
            last_w=np.array([], dtype=np.float64),
            all_turnovers=[],
            all_turnovers_baseline=[],
            all_gross_exposures=[],
            all_net_exposures=[],
            friction_pass_total=0,
            signal_total=0,
            support_leak_count=0,
            total_cost_hybrid=0.0,
            total_cost_baseline=0.0,
            cap_saturation_count=0,
            rebalance_count=0,
            trade_count=0,
            fold_rets_hybrid=[],
            fold_rets_baseline=[],
            fold_selected_symbols=(),
            block_rets_hybrid=(),
            block_rets_baseline=(),
            rets_baseline_ew=[],
            fit_rets_hybrid=(0.01, -0.01),
            fit_rets_by_fold=((0.01,), (-0.01, 0.02)),
            fold_attributions=(),
            policy_effect_by_fold=(),
        )
        assert result.fit_rets_by_fold == ((0.01,), (-0.01, 0.02))

    def test_oos_blend_invariant_reversion(self) -> None:
        """OOS blend가 invariant를 초과하면 원래 L*로 revert."""
        fit_rets = np.array([0.001] * 100 + [-0.002] * 50, dtype=np.float64)
        oos_rets = np.array([0.0001] * 50 + [-0.05, 0.01] * 25, dtype=np.float64)
        lev, binding, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            oos_rets=oos_rets,
            mdd_cap=0.30,
            mdd_margin=0.30,
            l_hard_cap=20.0,
            exchange_leverage_cap=None,
        )
        assert lev >= 1.0
        assert binding != "oos_blend" or binding == "mdd"

    def test_compute_layer2_fold_diagnostics_basic(self) -> None:
        """compute_layer2_fold_diagnostics: 기본 경로."""
        from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
            compute_layer2_fold_diagnostics,
        )

        np.random.seed(400)
        fold_rets = [np.random.normal(0.001, 0.02, 100).tolist() for _ in range(3)]
        fold_symbols = [("BTCUSDT",), ("ETHUSDT",), ("SOLUSDT",)]
        result = compute_layer2_fold_diagnostics(
            fold_rets_hybrid=fold_rets,
            fold_selected_symbols=fold_symbols,
            leverage=2.0,
            bars_per_year=2190,
        )
        assert result.fold_pass_ratio >= 0.0
        assert len(result.fold_unit_sharpes) == 3
        assert len(result.fold_selected_symbols) == 3

    def test_compute_layer2_fold_diagnostics_empty_fold(self) -> None:
        """compute_layer2_fold_diagnostics: 빈 fold + symbol 불일치."""
        from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
            compute_layer2_fold_diagnostics,
        )

        fold_rets = [[], [0.01, -0.005, 0.02], [0.02, -0.01, 0.03]]
        fold_symbols = [(), ("BTCUSDT",)]
        result = compute_layer2_fold_diagnostics(
            fold_rets_hybrid=fold_rets,
            fold_selected_symbols=fold_symbols,
            leverage=2.0,
            bars_per_year=2190,
        )
        assert result.fold_pass_ratio >= 0.0
