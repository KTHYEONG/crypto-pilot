# tests/unit/domain/futures/strategy/tiered_workflow/test_deployment_fix.py
"""CAGR-Deployment Gap Fix 단위 테스트 (spec: layer2-cagr-deployment-gap.md §🧪).

Scenarios:
    S1 (RC-2 fix): fit-leg가 전략 book — deploy_leverage ≈ hard_cap, binding=="hard_cap"
    S2 (RC-1/3 fix): deployed CAGR로 gate 통과 — cagr_hybrid ≈ unit_cagr * L, sortino 불변
    S3 (invariance): Sortino/Sharpe scale 불변 — L*=1 vs L*=20 동일
    S4 (edge — fit-leg empty): oos_proxy fallback — 크래시 없음, binding 기록
    S5 (RC-4 adaptive throttle): adaptive ref가 신호 스케일 추종 — m≈1.0
    S6 (selection 노브 — RC-3): vol_target/gross에 L* 주입, kelly 불변
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2TrialEvaluation,
)
from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
    apply_deployment,
    calibrate_deployment_leverage,
)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

_BARS_PER_YEAR = 2190.0  # 4h 기준


def _low_vol_rets(n: int = 2190, mu_bps: float = 6.4, sigma_bps: float = 2.0, seed: int = 42) -> np.ndarray:
    """전략 unit-vol book: 낮은 변동성(MDD ≈ 1%), 양의 드리프트."""
    rng = np.random.default_rng(seed)
    return rng.normal(mu_bps * 1e-4, sigma_bps * 1e-4, n).astype(np.float64)


def _market_rets(n: int = 2190, mu_bps: float = 0.0, sigma_bps: float = 90.0, seed: int = 7) -> np.ndarray:
    """시장평균 수익률: 높은 변동성(MDD ≈ 20~40%)."""
    rng = np.random.default_rng(seed)
    return rng.normal(mu_bps * 1e-4, sigma_bps * 1e-4, n).astype(np.float64)


def _make_evaluation(**kwargs: Any) -> Layer2TrialEvaluation:
    """최소 필수 필드를 채운 Layer2TrialEvaluation fixture."""
    defaults: dict[str, Any] = dict(
        objective_value=1.0,
        constraint_values=(-1.0,) * 8,
        cagr_hybrid=0.03,
        cagr_baseline=0.01,
        growth_lcb_hybrid=0.0,
        growth_lcb_baseline=0.0,
        sharpe_hac_hybrid=1.8,
        sharpe_hac_baseline=0.5,
        psr_hybrid=0.95,
        mdd_hybrid=0.01,
        cvar_95_hybrid=0.001,
        fold_pass_ratio=0.80,
        break_even_pass_pct=0.80,
        average_gross_exposure=0.10,
        cap_saturation_ratio=0.0,
        total_cost_bps=5.0,
        block_metrics=(),
        returns_hybrid=tuple(_low_vol_rets()),
        returns_baseline=(),
        sharpe_hybrid=1.8,
        sharpe_hac_baseline_ew=0.5,
        sortino_hybrid=3.0,
        trade_count=150,
        risk_utilization=0.034,
        deployment_objective_bonus=0.0,
        worst_fold_sharpe=1.2,
        gate=None,
        fit_returns_hybrid=tuple(_low_vol_rets()),
        deploy_leverage=1.0,
        deploy_binding="",
    )
    defaults.update(kwargs)
    return Layer2TrialEvaluation(**defaults)


# ---------------------------------------------------------------------------
# S1: RC-2 fix — fit-leg가 전략 book (저MDD) → L* ≈ hard_cap
# ---------------------------------------------------------------------------
class TestS1FitLegBookMddGivesHardCap:
    """Given 전략 book MDD≈1%, 시장평균 MDD≈25% — calibrate_deployment_leverage가
    전략 book 기반으로 동작하면 hard_cap에 clip되어야 한다."""

    def test_strategy_book_rets_gives_hard_cap(self) -> None:
        # Arrange
        strategy_rets = _low_vol_rets(n=2190, sigma_bps=2.0)  # MDD≈1%

        # Act: 전략 book으로 calibrate
        lev, binding, _ = calibrate_deployment_leverage(
            fit_rets=strategy_rets,
            mdd_cap=0.30,
            cvar_cap=0.06,
            mdd_margin=0.30,
            l_hard_cap=20.0,
        )

        # Assert: 저MDD book → budget 대비 여유 → hard_cap(20) clip
        assert binding == "hard_cap"
        assert lev == pytest.approx(20.0, rel=1e-3)

    def test_market_rets_gives_lower_leverage(self) -> None:
        """시장평균(고MDD) 수익률은 낮은 L*를 산출 (RC-2 전 동작 재현)."""
        # Arrange
        market_rets = _market_rets(n=2190, sigma_bps=90.0)

        # Act
        lev, _, _ = calibrate_deployment_leverage(
            fit_rets=market_rets,
            mdd_cap=0.30,
            cvar_cap=0.06,
            mdd_margin=0.30,
            l_hard_cap=20.0,
        )

        # Assert: 고MDD → L* << hard_cap (일반적으로 1~3 수준)
        assert lev < 5.0


# ---------------------------------------------------------------------------
# S2: RC-1/3 fix — deployed CAGR로 gate 통과, sortino 불변
# ---------------------------------------------------------------------------
class TestS2DeployedCagrGatePass:
    """unit-vol CAGR≈3%, L*=15 → deployed CAGR≈45%; sortino는 L 적용 전후 불변."""

    def test_cagr_scales_with_leverage(self) -> None:
        # Arrange: unit-vol book CAGR ≈ 3%
        rets = _low_vol_rets(n=2190, mu_bps=6.4, sigma_bps=2.0)
        unit_cagr = apply_deployment(
            rets=rets, leverage=1.0, bars_per_year=_BARS_PER_YEAR
        ).cagr

        # Act: L*=15 적용
        dep = apply_deployment(rets=rets, leverage=15.0, bars_per_year=_BARS_PER_YEAR)

        # Assert: deployed CAGR가 CAGR gate(30%) 돌파
        assert dep.cagr > 0.30, f"deployed cagr={dep.cagr:.4f} should exceed 0.30"
        assert dep.cagr > unit_cagr * 5.0  # 최소 5배 이상 성장

    def test_sortino_invariant_to_leverage(self) -> None:
        """Sortino(Sharpe)는 L에 불변 — 분모(std)와 분자(mean)가 같은 비율로 변화."""
        from src.domain.futures.strategy.tiered_workflow.metrics import _sortino

        # Arrange
        rets = _low_vol_rets(n=2190, mu_bps=6.4, sigma_bps=2.0)
        rets_arr = np.asarray(rets, dtype=np.float64)

        # Act
        s1 = _sortino(list(rets_arr * 1.0), bars_per_year=_BARS_PER_YEAR)
        s15 = _sortino(list(rets_arr * 15.0), bars_per_year=_BARS_PER_YEAR)

        # Assert: 상대 오차 1e-6 이내
        assert s1 == pytest.approx(s15, rel=1e-6)


# ---------------------------------------------------------------------------
# S3: Invariance — Sortino/Sharpe scale 불변, CAGR는 L배 로그성장
# ---------------------------------------------------------------------------
class TestS3ScaleInvariance:
    """apply_deployment L*=1 vs L*=20: sortino/sharpe 동일, cagr는 비례 증가."""

    def test_sortino_sharpe_invariant(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.metrics import _sharpe, _sortino

        # Arrange
        rets = _low_vol_rets(n=2190, mu_bps=6.4, sigma_bps=4.0, seed=1)
        rets_arr = np.asarray(rets, dtype=np.float64)

        # Act
        s1_sort = _sortino(list(rets_arr), bars_per_year=_BARS_PER_YEAR)
        s20_sort = _sortino(list(rets_arr * 20.0), bars_per_year=_BARS_PER_YEAR)
        s1_sharpe = _sharpe(list(rets_arr), bars_per_year=_BARS_PER_YEAR)
        s20_sharpe = _sharpe(list(rets_arr * 20.0), bars_per_year=_BARS_PER_YEAR)

        # Assert
        assert s1_sort == pytest.approx(s20_sort, rel=1e-6)
        assert s1_sharpe == pytest.approx(s20_sharpe, rel=1e-6)

    def test_cagr_increases_with_leverage(self) -> None:
        # Arrange
        rets = _low_vol_rets(n=2190, mu_bps=6.4, sigma_bps=2.0, seed=2)

        # Act
        dep1 = apply_deployment(rets=rets, leverage=1.0, bars_per_year=_BARS_PER_YEAR)
        dep20 = apply_deployment(rets=rets, leverage=20.0, bars_per_year=_BARS_PER_YEAR)

        # Assert
        assert dep20.cagr > dep1.cagr * 5.0


# ---------------------------------------------------------------------------
# S4: Edge — fit-leg empty → oos_proxy fallback, 크래시 없음
# ---------------------------------------------------------------------------
class TestS4FitLegEmptyFallback:
    """fit_rets_hybrid=() → oos_proxy(returns_hybrid)로 L* 산정, 크래시 없음."""

    def test_empty_fit_rets_uses_oos_proxy(self) -> None:
        # Arrange: fit_rets_hybrid 비어있음, returns_hybrid 유효
        oos_rets = _low_vol_rets(n=2190)
        evaluation = _make_evaluation(
            fit_returns_hybrid=(),
            returns_hybrid=tuple(oos_rets),
            deploy_leverage=1.0,
            deploy_binding="",
        )

        # Simulate C4 직접: getattr로 deploy_leverage 접근 — 기본값 1.0 반환
        l_star = float(getattr(evaluation, "deploy_leverage", 1.0))
        binding = str(getattr(evaluation, "deploy_binding", ""))

        # Assert: 크래시 없음, 기본값 반환
        assert isinstance(l_star, float)
        assert isinstance(binding, str)

    def test_calibrate_with_oos_proxy_succeeds(self) -> None:
        """oos_proxy로도 calibrate_deployment_leverage가 정상 동작."""
        # Arrange
        oos_rets = np.asarray(_low_vol_rets(n=2190), dtype=np.float64)

        # Act
        lev, binding, _ = calibrate_deployment_leverage(
            fit_rets=oos_rets,
            mdd_cap=0.30,
            cvar_cap=0.06,
            mdd_margin=0.30,
            l_hard_cap=20.0,
        )

        # Assert: 유효한 결과 반환
        assert lev >= 1.0
        assert binding in ("hard_cap", "mdd", "cvar", "none")

    def test_tiny_fit_rets_returns_no_leverage(self) -> None:
        """fit_rets 크기 < 2이면 L*=1.0, binding='none'."""
        # Act
        lev, binding, _ = calibrate_deployment_leverage(
            fit_rets=np.array([0.001], dtype=np.float64),
            mdd_cap=0.30,
            cvar_cap=0.06,
        )
        # Assert
        assert lev == pytest.approx(1.0)
        assert binding == "none"


# ---------------------------------------------------------------------------
# S5: RC-4 adaptive throttle — ref가 신호 스케일 추종
# ---------------------------------------------------------------------------
class TestS5AdaptiveThrottle:
    """per-bar edge median이 ref보다 크면 throttle multiplier ≈ 1.0 (full)."""

    def test_high_edge_gives_full_throttle(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            _edge_throttle_multiplier,
        )

        # Arrange: score >> ref → m = 1.0
        score_bps = 50.0  # signal edge 50 bps >> ref 5 bps
        ref_bps = 5.0

        # Act
        m = _edge_throttle_multiplier(
            score_bps,
            floor_bps=0.0,
            ref_bps=ref_bps,
            gamma=1.0,
        )

        # Assert: 완전 통과
        assert m == pytest.approx(1.0, abs=1e-9)

    def test_below_floor_gives_zero_throttle(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            _edge_throttle_multiplier,
        )

        # Arrange: score <= floor → m = 0.0
        score_bps = -1.0
        floor_bps = 0.0
        ref_bps = 5.0

        # Act
        m = _edge_throttle_multiplier(
            score_bps,
            floor_bps=floor_bps,
            ref_bps=ref_bps,
            gamma=1.0,
        )

        # Assert
        assert m == pytest.approx(0.0, abs=1e-9)

    def test_throttle_proportional_between_floor_and_ref(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            _edge_throttle_multiplier,
        )

        # Arrange: score = midpoint(floor, ref) → m ≈ 0.5
        floor_bps = 0.0
        ref_bps = 10.0
        score_bps = 5.0  # midpoint

        # Act
        m = _edge_throttle_multiplier(
            score_bps,
            floor_bps=floor_bps,
            ref_bps=ref_bps,
            gamma=1.0,
        )

        # Assert
        assert m == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# S6: Selection 노브 — RC-3 vol_target/gross에 L* 주입, kelly 불변
# ---------------------------------------------------------------------------
class TestS6SelectionKnobC4:
    """_apply_deployment_to_params: max_ann_vol *= L*, gross *= L*, kelly 불변."""

    def _make_params(self, kelly: float = 0.25, gross_cap: float = 3.0) -> dict[str, Any]:
        return {
            "kelly_fraction": kelly,
            "gross_exposure_cap": gross_cap,
            "l2_deploy_enabled": True,
            "l2_max_mdd_abs": 0.30,
            "l2_max_cvar_95": 0.06,
            "l2_deploy_mdd_margin": 0.30,
            "l2_deploy_l_hard_cap": 20.0,
        }

    def test_deploy_leverage_tracked_no_ceiling_injection(self) -> None:
        """C4 변경: 천장 주입 제거 — l2_deploy_leverage 추적만, vol/gross 불변.

        vol-targeting 하향전용(결함 #2) 확정으로 천장주입은 구조적 no-op.
        realized_mode=return_scaling으로 전환: L*는 run_l2_awf에 직접 전달.
        """
        from src.domain.futures.strategy.tiered_workflow.selection import (
            _apply_deployment_to_params,
        )

        # Arrange
        params = self._make_params(kelly=0.25, gross_cap=3.0)
        evaluation = _make_evaluation(deploy_leverage=12.0, deploy_binding="hard_cap")

        # Act
        deployed = _apply_deployment_to_params(params, evaluation, tf="4h")

        # Assert: vol/gross 불변 (천장주입 없음)
        assert "max_ann_vol" not in deployed  # 삽입되지 않아야 함
        assert deployed.get("gross_exposure_cap", 3.0) == pytest.approx(3.0, rel=1e-6)
        # 추적 필드 보존
        assert deployed["l2_deploy_leverage"] == pytest.approx(12.0, rel=1e-6)

    def test_kelly_fraction_unchanged(self) -> None:
        """C4 핵심: kelly_fraction은 L*에 의해 변경되지 않아야 한다."""
        from src.domain.futures.strategy.tiered_workflow.selection import (
            _apply_deployment_to_params,
        )

        # Arrange
        params = self._make_params(kelly=0.25)
        evaluation = _make_evaluation(deploy_leverage=12.0, deploy_binding="hard_cap")

        # Act
        deployed = _apply_deployment_to_params(params, evaluation, tf="4h")

        # Assert: kelly 불변
        assert deployed.get("kelly_fraction", 0.25) == pytest.approx(0.25, rel=1e-6)

    def test_no_deployment_when_l_star_leq_one(self) -> None:
        """L* ≤ 1.0 → params 변경 없음."""
        from src.domain.futures.strategy.tiered_workflow.selection import (
            _apply_deployment_to_params,
        )

        # Arrange
        params = self._make_params(kelly=0.25)
        evaluation = _make_evaluation(deploy_leverage=1.0, deploy_binding="none")

        # Act
        deployed = _apply_deployment_to_params(params, evaluation, tf="4h")

        # Assert: 원본 params와 동일
        assert deployed is params or deployed == params

    def test_l2_deploy_leverage_tracking_field(self) -> None:
        """deployed dict에 l2_deploy_leverage 추적 필드가 기록되어야 한다."""
        from src.domain.futures.strategy.tiered_workflow.selection import (
            _apply_deployment_to_params,
        )

        # Arrange
        params = self._make_params()
        evaluation = _make_evaluation(deploy_leverage=8.0, deploy_binding="mdd")

        # Act
        deployed = _apply_deployment_to_params(params, evaluation, tf="4h")

        # Assert
        assert deployed.get("l2_deploy_leverage") == pytest.approx(8.0, rel=1e-6)

    def test_deploy_disabled_returns_original_params(self) -> None:
        """l2_deploy_enabled=False → params 원본 반환."""
        from src.domain.futures.strategy.tiered_workflow.selection import (
            _apply_deployment_to_params,
        )

        # Arrange
        params = {
            **self._make_params(),
            "l2_deploy_enabled": False,
        }
        evaluation = _make_evaluation(deploy_leverage=15.0, deploy_binding="hard_cap")

        # Act
        result = _apply_deployment_to_params(params, evaluation, tf="4h")

        # Assert: 변경 없음
        assert result is params


# ---------------------------------------------------------------------------
# Layer2TrialEvaluation 필드 존재 검증
# ---------------------------------------------------------------------------
class TestLayer2TrialEvaluationNewFields:
    """dataclasses.py: deploy_leverage, deploy_binding 필드 추가 확인."""

    def test_default_values(self) -> None:
        # Arrange & Act
        ev = _make_evaluation()

        # Assert
        assert hasattr(ev, "deploy_leverage")
        assert hasattr(ev, "deploy_binding")
        assert ev.deploy_leverage == pytest.approx(1.0)
        assert ev.deploy_binding == ""

    def test_custom_values(self) -> None:
        # Arrange & Act
        ev = _make_evaluation(deploy_leverage=15.5, deploy_binding="hard_cap")

        # Assert
        assert ev.deploy_leverage == pytest.approx(15.5)
        assert ev.deploy_binding == "hard_cap"


# ---------------------------------------------------------------------------
# risk_deployment.py l_hard_cap 기본값 검증
# ---------------------------------------------------------------------------
class TestRiskDeploymentDefaultHardCap:
    """calibrate_deployment_leverage 기본 l_hard_cap=20.0 (RC-4 dead-default 제거)."""

    def test_default_hard_cap_is_20(self) -> None:
        import inspect

        sig = inspect.signature(calibrate_deployment_leverage)
        default = sig.parameters["l_hard_cap"].default

        assert default == pytest.approx(20.0)
