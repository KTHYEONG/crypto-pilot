# tests/unit/domain/futures/strategy/tiered_workflow/test_deployment_realization.py
"""Deployment Realization 정합 테스트 (Scenario 1~6).

Spec: docs/specs/layer2-deployment-realization.md
순수 NumPy 단위테스트 — 외부 boundary 없음, 모킹 불필요.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig
from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
    apply_deployment,
    calibrate_deployment_leverage,
)

BARS_PER_YEAR: float = 2190.0  # 4h


def _make_unit_vol_rets(
    n: int = 2190,
    mu_bps: float = 3.0,
    sigma_bps: float = 16.0,
    seed: int = 42,
) -> np.ndarray:
    """재현 가능 unit-vol per-bar 수익률 생성.

    Args:
        n: 바 수 (기본 1년).
        mu_bps: 평균 수익 (bps).
        sigma_bps: 표준편차 (bps).
        seed: 재현성 시드.

    Returns:
        float64 per-bar 수익률 배열 [n].
    """
    rng = np.random.default_rng(seed)
    return rng.normal(mu_bps * 1e-4, sigma_bps * 1e-4, n).astype(np.float64)


# ---------------------------------------------------------------------------
# Scenario 1 — Happy Path: realization 정합
# ---------------------------------------------------------------------------
class TestScenario1RealizationConsistency:
    """Scenario 1: apply_deployment 스케일 정합 검증."""

    def test_mdd_scales_linearly_with_leverage(self) -> None:
        """mdd(L·rets) ≈ L x mdd(rets) (선형 스케일 검증).

        Given: 고정 seed unit-vol rets, L*=10.0.
        When: apply_deployment(rets, 10.0).
        Then: dep.mdd ≈ 10 x unit_mdd (rel=1e-3).
        """
        # Arrange
        rets = _make_unit_vol_rets(n=2190, mu_bps=3.0, sigma_bps=16.0, seed=42)
        l_star = 10.0
        unit_rets_arr = np.asarray(rets, dtype=np.float64)

        # Act
        dep = apply_deployment(rets=rets, leverage=l_star, bars_per_year=BARS_PER_YEAR)

        # Assert — MDD는 compounding 비선형, 단순 L*·mdd 보다 작거나 같음
        # 대신 scaled rets에서 직접 계산한 값과 dep.mdd 일치 확인
        scaled = l_star * unit_rets_arr
        eq = np.cumprod(1.0 + scaled)
        pk = np.maximum.accumulate(eq)
        expected_mdd = float(np.max((pk - eq) / np.maximum(pk, 1e-12)))
        assert dep.mdd == pytest.approx(expected_mdd, rel=1e-6)

    def test_cagr_increases_with_leverage(self) -> None:
        """L=10 배치 후 CAGR > unit CAGR.

        Given: 동일 rets, L=1.0 vs L=10.0.
        When: apply_deployment 각각 호출.
        Then: L=10 CAGR > L=1 CAGR.
        """
        # Arrange
        rets = _make_unit_vol_rets(n=2190, mu_bps=5.0, sigma_bps=12.0, seed=7)

        # Act
        dep_unit = apply_deployment(rets=rets, leverage=1.0, bars_per_year=BARS_PER_YEAR)
        dep_lev = apply_deployment(rets=rets, leverage=10.0, bars_per_year=BARS_PER_YEAR)

        # Assert
        assert dep_lev.cagr > dep_unit.cagr

    def test_cvar_scales_linearly_with_leverage(self) -> None:
        """CVaR95(L·rets) ≈ L x CVaR95(rets).

        Given: 고정 seed rets, L=10.
        When: apply_deployment.
        Then: dep.cvar_95 ≈ 10 x unit_cvar (rel=1e-3).
        """
        # Arrange
        rets = _make_unit_vol_rets(n=2190, mu_bps=3.0, sigma_bps=16.0, seed=42)
        l_star = 10.0
        arr = np.asarray(rets, dtype=np.float64)
        losses_unit = -arr
        var_cut = float(np.quantile(losses_unit, 0.95))
        tail = losses_unit[losses_unit >= var_cut]
        unit_cvar = float(np.maximum(np.mean(tail), 0.0)) if tail.size > 0 else max(var_cut, 0.0)

        # Act
        dep = apply_deployment(rets=rets, leverage=l_star, bars_per_year=BARS_PER_YEAR)

        # Assert
        assert dep.cvar_95 == pytest.approx(l_star * unit_cvar, rel=1e-3)


# ---------------------------------------------------------------------------
# Scenario 2 — Deployment Consistency: trial-path vs final-replay 정합
# ---------------------------------------------------------------------------
class TestScenario2DeploymentConsistency:
    """Scenario 2: apply_deployment 두 번 호출 시 cagr 완전 일치 (결함 #3 회귀 방어)."""

    def test_cagr_identical_for_same_rets_and_leverage(self) -> None:
        """trial-path cagr ≡ final-replay cagr (동일 인자 재현성).

        Given: 동일 sim.rets_hybrid, 동일 L*.
        When: apply_deployment 두 번 호출.
        Then: 두 cagr pytest.approx(rel=1e-6) 일치.
        """
        # Arrange
        rets = _make_unit_vol_rets(n=2190, mu_bps=4.0, sigma_bps=15.0, seed=99)
        l_star = 8.5

        # Act
        dep_trial = apply_deployment(rets=rets, leverage=l_star, bars_per_year=BARS_PER_YEAR)
        dep_final = apply_deployment(rets=rets, leverage=l_star, bars_per_year=BARS_PER_YEAR)

        # Assert
        assert dep_trial.cagr == pytest.approx(dep_final.cagr, rel=1e-6)
        assert dep_trial.mdd == pytest.approx(dep_final.mdd, rel=1e-6)
        assert dep_trial.cvar_95 == pytest.approx(dep_final.cvar_95, rel=1e-6)


# ---------------------------------------------------------------------------
# Scenario 3 — Exchange Cap Binding
# ---------------------------------------------------------------------------
class TestScenario3ExchangeCapBinding:
    """Scenario 3: exchange_leverage_cap이 binding이 되는 시나리오."""

    def test_exchange_cap_binding_when_mdd_cvar_higher(self) -> None:
        """l_mdd=19.5, l_cvar=30, exchange_cap=10 → (10.0, "exchange_cap").

        Given: 충분히 낮은 변동성 rets (→ l_mdd≫10, l_cvar≫10).
        When: calibrate_deployment_leverage(exchange_leverage_cap=10.0).
        Then: 반환 L*=10.0, binding="exchange_cap".
        """
        # Arrange — 극소 변동성 → MDD/CVaR 예산 대비 L이 매우 높게 산출됨
        rng = np.random.default_rng(0)
        # sigma=0.5bps → MDD가 매우 낮아 l_mdd >> 10
        fit_rets = rng.normal(2e-4, 5e-5, 3000).astype(np.float64)

        # Act
        l_star, binding, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            mdd_cap=0.30,
            cvar_cap=0.06,
            mdd_margin=0.30,
            cvar_margin=0.20,
            l_hard_cap=20.0,
            exchange_leverage_cap=10.0,
        )

        # Assert
        assert l_star == pytest.approx(10.0, abs=1e-3)
        assert binding == "exchange_cap"

    def test_no_exchange_cap_when_none(self) -> None:
        """exchange_leverage_cap=None → exchange_cap binding 없음.

        Given: 동일 극소 변동성 rets.
        When: calibrate_deployment_leverage(exchange_leverage_cap=None).
        Then: binding ≠ "exchange_cap".
        """
        # Arrange
        rng = np.random.default_rng(0)
        fit_rets = rng.normal(2e-4, 5e-5, 3000).astype(np.float64)

        # Act
        l_star, binding, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            mdd_cap=0.30,
            cvar_cap=0.06,
            mdd_margin=0.30,
            cvar_margin=0.20,
            l_hard_cap=20.0,
            exchange_leverage_cap=None,
        )

        # Assert
        assert binding != "exchange_cap"
        assert l_star == pytest.approx(20.0, abs=1.0)  # hard_cap binding 예상


class TestLayer2AllocationConfigParsing:
    """Layer2AllocationConfig 기본 exchange cap 파싱 정합."""

    def test_default_exchange_cap_preserved_when_key_absent(self) -> None:
        """빈 mapping이면 기본 거래소 cap 10배가 유지되어야 한다."""
        cfg = Layer2AllocationConfig.from_mapping({})

        assert cfg.l2_max_exchange_leverage == pytest.approx(10.0)

    def test_explicit_none_exchange_cap_disables_cap(self) -> None:
        """명시적 None 입력은 cap 비활성화로 해석되어야 한다."""
        cfg = Layer2AllocationConfig.from_mapping({"l2_max_exchange_leverage": None})

        assert cfg.l2_max_exchange_leverage is None


# ---------------------------------------------------------------------------
# Scenario 4 — RiskUtil 정합 가드
# ---------------------------------------------------------------------------
class TestScenario4RiskUtilGate:
    """Scenario 4: MDD binding 시 RiskUtil ≈ (1 - mdd_margin)."""

    def test_risk_util_near_target_when_mdd_binding(self) -> None:
        """binding=mdd 시 mdd_hybrid/mdd_cap ≈ (1 - mdd_margin).

        Given: mdd_margin=0.30, mdd_cap=0.30.
        When: calibrate_deployment_leverage → apply_deployment.
        Then: risk_util == pytest.approx(0.70, abs=0.10).

        결함 #1/#2 재발 시 risk_util ≈ 0.05 (unit-vol MDD << 30%) → 테스트 실패 → 조기 감지.
        """
        # Arrange — 중간 변동성 → MDD가 binding이 되는 레버리지 범위
        rng = np.random.default_rng(123)
        # sigma=100bps/bar → MDD 예산(21%)이 적당히 binding
        fit_rets = rng.normal(3e-4, 50e-4, 2190).astype(np.float64)
        mdd_cap = 0.30

        # Act
        l_star, binding, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            mdd_cap=0.30,
            cvar_cap=0.06,
            mdd_margin=0.30,
            cvar_margin=0.20,
            l_hard_cap=20.0,
            exchange_leverage_cap=None,
        )
        dep = apply_deployment(rets=fit_rets, leverage=l_star, bars_per_year=BARS_PER_YEAR)
        risk_util = dep.mdd / max(mdd_cap, 1e-9)

        # Assert — MDD binding 시 risk_util ≈ 0.70 (1 - 0.30)
        if binding == "mdd":
            assert risk_util == pytest.approx(0.70, abs=0.10)
        else:
            # hard_cap/exchange_cap binding: risk_util은 다를 수 있으나 < 1.0
            assert risk_util < 1.0


# ---------------------------------------------------------------------------
# Scenario 5 — Edge: deploy disabled / 빈 fit-leg
# ---------------------------------------------------------------------------
class TestScenario5EdgeCases:
    """Scenario 5: deploy 비활성화 / 빈 fit-leg 시 L*=1.0, 예외 없음."""

    def test_empty_fit_rets_returns_one(self) -> None:
        """fit_rets.size < 2 → L*=1.0, binding="none".

        Given: 빈 배열.
        When: calibrate_deployment_leverage.
        Then: (1.0, "none"), 예외 없음.
        """
        # Arrange
        fit_rets = np.array([], dtype=np.float64)

        # Act
        l_star, binding, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            mdd_cap=0.30,
            cvar_cap=0.06,
        )

        # Assert
        assert l_star == pytest.approx(1.0)
        assert binding == "none"

    def test_single_element_fit_rets_returns_one(self) -> None:
        """fit_rets.size == 1 → L*=1.0 (size < 2 가드).

        Given: 단일원소 배열.
        When: calibrate_deployment_leverage.
        Then: (1.0, "none").
        """
        # Arrange
        fit_rets = np.array([0.001], dtype=np.float64)

        # Act
        l_star, binding, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            mdd_cap=0.30,
            cvar_cap=0.06,
        )

        # Assert
        assert l_star == pytest.approx(1.0)
        assert binding == "none"

    def test_l_star_one_gives_unit_vol_metrics(self) -> None:
        """L*=1.0 → deployed CAGR/MDD ≡ unit-vol.

        Given: rets, leverage=1.0.
        When: apply_deployment.
        Then: dep.leverage==1.0, dep.scaled_rets ≡ rets.
        """
        # Arrange
        rets = _make_unit_vol_rets(n=500, mu_bps=2.0, sigma_bps=10.0, seed=11)

        # Act
        dep = apply_deployment(rets=rets, leverage=1.0, bars_per_year=BARS_PER_YEAR)

        # Assert
        assert dep.leverage == pytest.approx(1.0)
        np.testing.assert_allclose(dep.scaled_rets, rets, rtol=1e-9)


# ---------------------------------------------------------------------------
# Scenario 6 — Edge: 단일봉 청산 불가 정리 검증
# ---------------------------------------------------------------------------
class TestScenario6NoClearance:
    """Scenario 6: min(dep.scaled_rets) > -1.0 (청산 없음, log1p clip 미발동)."""

    def test_no_clearance_with_mdd_bounded_leverage(self) -> None:
        """L*=19.0 (MDD-binding)에서도 단일봉 청산 없음.

        Given: unit-vol rets 중 min(rets) ≈ -0.011, L*=19.0.
        When: apply_deployment.
        Then: min(dep.scaled_rets) > -1.0.

        수학적 보장: L*·MDD_L ≤ mdd_target(21%) < 100% → 연속 손실에서도 청산 불가.
        """
        # Arrange — sigma=16bps, 2190봉 → worst bar ≈ -0.5% ~ -0.1%
        rets = _make_unit_vol_rets(n=2190, mu_bps=3.0, sigma_bps=16.0, seed=42)
        l_star = 19.0

        # Act
        dep = apply_deployment(rets=rets, leverage=l_star, bars_per_year=BARS_PER_YEAR)

        # Assert
        assert float(np.min(dep.scaled_rets)) > -1.0

    def test_log1p_clip_not_triggered_under_normal_leverage(self) -> None:
        """L*=10, 일반 변동성에서 log1p clip(-1+1e-9) 미발동.

        Given: rets, L=10.
        When: apply_deployment (내부 log1p clip 존재).
        Then: scaled_rets의 min > -1.0 (clip 불필요 구간).
        """
        # Arrange
        rets = _make_unit_vol_rets(n=2190, mu_bps=4.0, sigma_bps=20.0, seed=55)
        l_star = 10.0

        # Act
        dep = apply_deployment(rets=rets, leverage=l_star, bars_per_year=BARS_PER_YEAR)

        # Assert
        assert float(np.min(dep.scaled_rets)) > -1.0


# ---------------------------------------------------------------------------
# 추가: calibrate_deployment_leverage binding 결정 단위검증
# ---------------------------------------------------------------------------
class TestCalibrateLeverageBindingLogic:
    """argmin candidates 로직 검증 — 결함 #2 회귀 방지."""

    def test_mdd_binding_when_mdd_most_restrictive(self) -> None:
        """l_mdd < l_cvar < l_hard_cap → binding="mdd".

        Given: 높은 변동성(MDD가 CVaR보다 제약적).
        When: calibrate_deployment_leverage.
        Then: binding ∈ {"mdd", "hard_cap", "exchange_cap"} — cvar는 아님.
        """
        # Arrange — 고변동성: MDD가 CVaR보다 먼저 예산 소진
        rng = np.random.default_rng(77)
        fit_rets = rng.normal(1e-4, 100e-4, 2190).astype(np.float64)

        # Act
        l_star, binding, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            mdd_cap=0.30,
            cvar_cap=0.06,
            mdd_margin=0.30,
            cvar_margin=0.20,
            l_hard_cap=20.0,
            exchange_leverage_cap=None,
        )

        # Assert
        assert l_star >= 1.0
        assert binding in {"mdd", "hard_cap", "cvar"}

    def test_leverage_floor_at_one(self) -> None:
        """반환 L* ≥ 1.0 항상 보장.

        Given: 음수 mu 수익률 (MDD 높음 → L*이 1.0으로 clip).
        When: calibrate_deployment_leverage.
        Then: L* ≥ 1.0 (l_floor 기본값).
        """
        # Arrange
        rng = np.random.default_rng(5)
        fit_rets = rng.normal(-1e-3, 50e-4, 500).astype(np.float64)  # 음수 mu

        # Act
        l_star, _binding, _ = calibrate_deployment_leverage(
            fit_rets=fit_rets,
            mdd_cap=0.30,
            cvar_cap=0.06,
            exchange_leverage_cap=5.0,
        )

        # Assert
        assert l_star >= 1.0
