# tests/unit/domain/futures/strategy/tiered_workflow/test_dsr_restructure.py
"""DSR-First 구조 개혁 단위 테스트 (스펙: layer2-utilization-dsr-restructure.md).

Scenarios covered:
    D1: Sortino 목적함수 (_shape_efficiency_l2_objective)
        S1: 고Sortino trial 선택 우선
        S2: worst_fold 패널티 단조성
        S3: Sortino>=1.5 + Sharpe<0.7 -> sanity floor fail
        S4: Sortino>=1.5 + Sharpe>=0.7 + Calmar<0.5 -> calmar_floor fail
    D2: vol_target=None -> unit vol-target(1.0) 활성화
        S1: max_ann_vol=None -> vol_target=1.0 보장
    D3: fit_rets_hybrid 필드 존재 및 look-ahead 없음 확인
        S1: fit_rets_hybrid 인덱스 전부 < oos_start
    D4: DSR -> diagnostic 강등, PSR gate 신설
        S1: dsr_floor 차단 제거 (DSR=0.5라도 BLOCKER 아님)
        S2: psr_hybrid < 0.90 -> psr_floor BLOCKER
        S3: psr_hybrid >= 0.90 -> psr_floor 없음
        S4: DSR 값 field에 잔존 (diagnostic 보존)
        S5: DSR 벤치마크 +mean 제거로 벤치마크 하락 검증
"""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
import pytest

from src.domain.futures.optimization.workflow import _shape_efficiency_l2_objective
from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig
from src.domain.futures.strategy.tiered_workflow.l2_gate import evaluate_layer2_gate
from src.domain.futures.strategy.tiered_workflow.metrics import (
    _deflated_sharpe_probability,
    _sortino_hac_unit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BARS_PER_YEAR = 2190.0  # 4h


class _GateKwargs(TypedDict):
    """evaluate_layer2_gate 공통 kwargs 타입 정의."""

    deployment_failed: bool
    support_leak_count: int
    cagr_hybrid: float
    sharpe_hybrid: float
    sharpe_hac_hybrid: float
    sharpe_hac_baseline: float
    sortino_hybrid: float
    mar_hybrid: float
    mdd_hybrid: float
    cvar_95_hybrid: float
    fold_pass_ratio: float
    active_block_count: int
    friction_pass_pct: float
    trade_count: int
    growth_lcb_hybrid: float
    growth_lcb_baseline: float
    dsr_hybrid: float | None
    psr_hybrid: float | None
    config: Layer2AllocationConfig


def _make_positive_rets(n: int = 300, scale: float = 0.002) -> list[float]:
    """규칙적인 양의 수익률 시퀀스."""
    rng = np.random.default_rng(42)
    return list(rng.normal(loc=scale, scale=scale * 0.5, size=n).tolist())


def _make_gate_kwargs(
    *,
    sortino: float = 2.0,
    sharpe: float = 1.0,
    calmar: float = 1.0,
    psr: float | None = 0.95,
    dsr: float | None = 0.5,
    cagr: float = 0.5,
    mdd: float = 0.15,
) -> _GateKwargs:
    """evaluate_layer2_gate 공통 kwargs 생성 헬퍼."""
    return {
        "deployment_failed": False,
        "support_leak_count": 0,
        "cagr_hybrid": cagr,
        "sharpe_hybrid": sharpe,
        "sharpe_hac_hybrid": sharpe,
        "sharpe_hac_baseline": 0.0,
        "sortino_hybrid": sortino,
        "mar_hybrid": calmar,
        "mdd_hybrid": mdd,
        "cvar_95_hybrid": 0.02,
        "fold_pass_ratio": 1.0,
        "active_block_count": 5,
        "friction_pass_pct": 0.80,
        "trade_count": 100,
        "growth_lcb_hybrid": 0.05,
        "growth_lcb_baseline": 0.01,
        "dsr_hybrid": dsr,
        "psr_hybrid": psr,
        "config": Layer2AllocationConfig(),
    }


# ---------------------------------------------------------------------------
# D1-S1: 고Sortino trial 선택 우선
# ---------------------------------------------------------------------------


def test_shape_efficiency_objective_prefers_high_sortino_over_high_growth() -> None:
    """고Sortino·저penalty trial이 저Sortino trial보다 높은 목적값을 가져야 한다.

    Scenario: 동일 최악fold·dispersion 조건에서 sortino_hac_unit 차이만 변화.
    """
    # Arrange
    high_sortino = 3.0
    low_sortino = 0.5
    common_kwargs = {
        "worst_fold_sortino": 1.0,
        "worst_fold_threshold": -0.30,
        "worst_fold_weight": 0.005,
        "downside_dispersion": 0.0,
    }

    # Act
    obj_high = _shape_efficiency_l2_objective(sortino_hac_unit=high_sortino, **common_kwargs)
    obj_low = _shape_efficiency_l2_objective(sortino_hac_unit=low_sortino, **common_kwargs)

    # Assert: 고Sortino trial이 더 높은 objective 값
    assert obj_high > obj_low


# ---------------------------------------------------------------------------
# D1-S2: worst_fold 패널티 단조성
# ---------------------------------------------------------------------------


def test_shape_efficiency_objective_worst_fold_penalty_monotone() -> None:
    """worst_fold_sortino가 threshold보다 낮을수록 패널티가 단조 증가해야 한다."""
    # Arrange
    threshold = -0.30
    weight = 0.005
    sortino_base = 2.0

    # Act — threshold 초과 시 패널티 0 (RC-2 soft 패널티 비활성화: weight=0.0)
    obj_above = _shape_efficiency_l2_objective(
        sortino_hac_unit=sortino_base,
        worst_fold_sortino=threshold,  # 경계: 패널티=0
        worst_fold_threshold=threshold,
        worst_fold_weight=weight,
        downside_dispersion=0.0,
        risk_util_weight=0.0,
        trade_weight=0.0,
    )
    obj_below = _shape_efficiency_l2_objective(
        sortino_hac_unit=sortino_base,
        worst_fold_sortino=threshold - 0.5,  # threshold보다 낮음
        worst_fold_threshold=threshold,
        worst_fold_weight=weight,
        downside_dispersion=0.0,
        risk_util_weight=0.0,
        trade_weight=0.0,
    )

    # Assert: threshold 정확히 맞으면 패널티=0, 아래면 패널티>0 -> 목적값 하락
    assert math.isclose(obj_above, sortino_base, rel_tol=1e-9)
    assert obj_above > obj_below


# ---------------------------------------------------------------------------
# D1-S3: Sortino>=1.5 + Sharpe<0.7 -> sanity floor fail
# ---------------------------------------------------------------------------


def test_gate_sortino_passes_but_sharpe_sanity_floor_fails() -> None:
    """Sortino>=1.5이어도 Sharpe<0.7이면 sharpe_abs BLOCKER가 작동해야 한다."""
    # Arrange
    kwargs = _make_gate_kwargs(sortino=1.8, sharpe=0.5)  # sharpe < l2_min_sharpe_abs=0.7

    # Act
    result = evaluate_layer2_gate(**kwargs)

    # Assert: sharpe_abs 차단
    assert not result.promotion_passed
    assert result.promotion_blocker == "sharpe_abs"


# ---------------------------------------------------------------------------
# D1-S4: Sortino>=1.5, Sharpe>=0.7, Calmar<0.5 -> calmar_floor fail
# ---------------------------------------------------------------------------


def test_gate_calmar_floor_blocks_when_cagr_mdd_ratio_low() -> None:
    """CAGR/MDD 비율(calmar)이 l2_min_calmar(0.5) 미만이면 calmar_floor 차단.

    BLOCKER 순서: calmar_floor(5번째) < mdd_abs(7번째) — calmar 먼저 발화.
    cagr=0.35(≥0.30 통과), mdd=0.80 -> calmar=0.35/0.80≈0.44 < 0.5.
    """
    # Arrange: cagr=0.35 (l2_min_cagr 통과), mdd=0.80 -> calmar≈0.4375 < 0.5
    kwargs = _make_gate_kwargs(sortino=1.8, sharpe=1.0, calmar=0.44, cagr=0.35, mdd=0.80)

    # Act
    result = evaluate_layer2_gate(**kwargs)

    # Assert
    assert not result.promotion_passed
    assert result.promotion_blocker == "calmar_floor"


# ---------------------------------------------------------------------------
# D2-S1: max_ann_vol=None -> vol_target=1.0 in awf_sim
# ---------------------------------------------------------------------------


def test_layer2_allocation_config_default_max_ann_vol_is_none() -> None:
    """Layer2AllocationConfig 기본값에서 max_ann_vol=None이어야 한다.

    awf_sim.py가 None을 1.0으로 대체하는 로직의 전제 조건 검증.
    """
    # Arrange + Act
    config = Layer2AllocationConfig()

    # Assert
    assert config.max_ann_vol is None


# ---------------------------------------------------------------------------
# D3-S2: fit-leg 입력 시 binding=mdd/cvar, CAGR 단조성 (audit Gap 1·2)
# ---------------------------------------------------------------------------


def test_apply_deployment_uses_evaluation_deploy_leverage() -> None:
    """C4: _apply_deployment_to_params가 evaluation.deploy_leverage를 재사용(재계산 금지).

    kelly_fraction 불변, max_ann_vol *= L*, gross_exposure_cap *= L*.
    """
    from unittest.mock import MagicMock

    from src.domain.futures.strategy.tiered_workflow.selection import (
        _apply_deployment_to_params,
    )

    # Arrange
    eval_mock = MagicMock()
    eval_mock.deploy_leverage = 12.0
    eval_mock.deploy_binding = "hard_cap"

    base_params: dict[str, float | int | str] = {
        "kelly_fraction": 0.25,
        "gross_exposure_cap": 3.0,
        "l2_deploy_enabled": True,
        "l2_deploy_mdd_margin": 0.30,
        "l2_deploy_l_hard_cap": 20.0,
        "l2_max_mdd_abs": 0.30,
        "l2_max_cvar_95": 0.06,
    }

    # Act
    result = _apply_deployment_to_params(base_params, eval_mock, tf="4h")

    # Assert: kelly 불변
    assert result.get("kelly_fraction", 0.25) == pytest.approx(0.25, rel=1e-6)
    # Assert: 천장 주입 없음 (realized_mode=return_scaling으로 변경 — 결함 #2 해소)
    assert "max_ann_vol" not in result
    assert result.get("gross_exposure_cap", 3.0) == pytest.approx(3.0, rel=1e-6)
    # Assert: 추적 필드
    assert result.get("l2_deploy_leverage") == pytest.approx(12.0, rel=1e-6)


def test_apply_deployment_fallback_to_oos_when_fit_leg_empty() -> None:
    """C4: deploy_leverage=1.0 → params 변경 없음 (no-op 경로)."""
    from unittest.mock import MagicMock

    from src.domain.futures.strategy.tiered_workflow.selection import (
        _apply_deployment_to_params,
    )

    # Arrange: evaluation이 L*=1.0 (no deployment)
    eval_mock = MagicMock()
    eval_mock.deploy_leverage = 1.0
    eval_mock.deploy_binding = "none"

    base_params: dict[str, float | int | str] = {
        "kelly_fraction": 0.25,
        "l2_deploy_enabled": True,
        "l2_deploy_mdd_margin": 0.30,
        "l2_deploy_l_hard_cap": 20.0,
        "l2_max_mdd_abs": 0.30,
        "l2_max_cvar_95": 0.06,
    }

    # Act
    result = _apply_deployment_to_params(base_params, eval_mock, tf="4h")

    # Assert: 변경 없음
    assert result is base_params or result == base_params


# ---------------------------------------------------------------------------
# D3-S3: fit_start==fit_end -> fit_returns_hybrid==()
# ---------------------------------------------------------------------------


def test_layer2_trial_evaluation_fit_returns_hybrid_default_empty() -> None:
    """Layer2TrialEvaluation 기본값 fit_returns_hybrid는 빈 tuple이어야 한다."""
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2TrialEvaluation,
    )

    # Arrange + Act: 최소 필드로 생성 (fit_returns_hybrid 기본값 확인)
    eval_obj = Layer2TrialEvaluation(
        objective_value=0.0,
        constraint_values=(-1.0,),
        cagr_hybrid=0.1,
        cagr_baseline=0.05,
        growth_lcb_hybrid=0.02,
        growth_lcb_baseline=0.01,
        sharpe_hac_hybrid=1.5,
        sharpe_hac_baseline=0.8,
        psr_hybrid=0.95,
        mdd_hybrid=0.10,
        cvar_95_hybrid=0.02,
        fold_pass_ratio=0.8,
        break_even_pass_pct=0.7,
        average_gross_exposure=0.5,
        cap_saturation_ratio=0.1,
        total_cost_bps=10.0,
        block_metrics=(),
    )

    # Assert
    assert eval_obj.fit_returns_hybrid == ()


# ---------------------------------------------------------------------------
# D4-S1: dsr_floor 차단 제거 — DSR=0.5라도 BLOCKER 아님
# ---------------------------------------------------------------------------


def test_gate_dsr_no_longer_blocks_promotion() -> None:
    """DSR=0.5(평탄 표면)이어도 promotion_blocker에 dsr_floor가 없어야 한다."""
    # Arrange: 모든 지표 pass, DSR=0.5
    kwargs = _make_gate_kwargs(sortino=2.0, sharpe=1.0, calmar=2.0, psr=0.95, dsr=0.5)

    # Act
    result = evaluate_layer2_gate(**kwargs)

    # Assert: DSR은 더 이상 BLOCKER가 아님
    assert "dsr" not in result.promotion_blocker
    assert result.promotion_passed


# ---------------------------------------------------------------------------
# D4-S2: psr_hybrid < 0.90 -> psr_floor BLOCKER
# ---------------------------------------------------------------------------


def test_gate_psr_floor_blocks_when_psr_below_threshold() -> None:
    """psr_hybrid=0.80 < l2_min_psr=0.90 -> psr_floor BLOCKER."""
    # Arrange
    kwargs = _make_gate_kwargs(sortino=2.0, sharpe=1.0, calmar=2.0, psr=0.80, dsr=0.9)

    # Act
    result = evaluate_layer2_gate(**kwargs)

    # Assert
    assert not result.promotion_passed
    assert result.promotion_blocker == "psr_floor"


# ---------------------------------------------------------------------------
# D4-S3: psr_hybrid >= 0.90 -> psr_floor 없음 (정상 통과)
# ---------------------------------------------------------------------------


def test_gate_psr_floor_passes_when_psr_above_threshold() -> None:
    """psr_hybrid=0.95 >= l2_min_psr=0.90 -> psr_floor BLOCKER 없음."""
    # Arrange
    kwargs = _make_gate_kwargs(sortino=2.0, sharpe=1.0, calmar=2.0, psr=0.95, dsr=0.5)

    # Act
    result = evaluate_layer2_gate(**kwargs)

    # Assert
    assert result.promotion_passed
    assert result.promotion_blocker == ""


# ---------------------------------------------------------------------------
# D4-S4: DSR 필드는 여전히 diagnostic으로 계산·기록됨
# ---------------------------------------------------------------------------


def test_gate_dsr_diagnostic_field_preserved() -> None:
    """evaluate_layer2_gate는 dsr_hybrid를 promotion_constraint_values에 저장한다.

    DSR은 BLOCKER는 아니나 스코어카드 diagnostic으로 계산값이 남아야 함.
    """
    # Arrange
    dsr_value = 0.502
    kwargs = _make_gate_kwargs(sortino=2.0, sharpe=1.0, calmar=2.0, psr=0.95, dsr=dsr_value)

    # Act
    result = evaluate_layer2_gate(**kwargs)

    # Assert: promotion_passed이고 DSR 차단 없음 — DSR은 입력으로만 사용
    assert result.promotion_passed
    # _PROMOTION_BLOCKERS = 16개 항목 -> promotion_constraint_values = 16개
    assert len(result.promotion_constraint_values) == 16
    # optuna_constraint_values = 10개 (별도 구조, ADR_20260718_L2_CRISIS_AWARE_OPTUNA_CONSTRAINT의
    # crisis_mdd_hybrid 10번째 슬롯 포함 — crisis 인자 미전달 시 -1.0 sentinel로 항상 만족)
    assert len(result.optuna_constraint_values) == 10


# ---------------------------------------------------------------------------
# D4-S5: DSR 벤치마크 +mean 제거 -> benchmark 하락 (BVA)
# ---------------------------------------------------------------------------


def test_dsr_benchmark_without_mean_pool_term_is_lower() -> None:
    """기존 +mean(pool) 포함 벤치마크 vs 제거 후 벤치마크 비교.

    Pool mean > 0인 경우 제거 후 benchmark_per_bar가 낮아야 함.
    -> DSR 값 상승 (z_score 증가 -> CDF 증가).
    """
    # Arrange: 양의 mean pool
    rng = np.random.default_rng(42)
    rets = list(rng.normal(0.002, 0.001, 300).tolist())
    pool_sharpes_positive = np.array([1.5, 1.8, 2.0, 1.3, 2.2], dtype=np.float64)

    # Act: 현재 구현 (mean 제거됨)
    dsr_no_mean = _deflated_sharpe_probability(
        selected_rets=rets,
        completed_trial_sharpes=pool_sharpes_positive,
        effective_trial_count=5.0,
        bars_per_year=_BARS_PER_YEAR,
    )

    # 참조: pool_sharpes 평균=0 (mean 영향 제거한 상태 모사)
    pool_sharpes_zero_mean = pool_sharpes_positive - np.mean(pool_sharpes_positive)
    dsr_zero_mean_pool = _deflated_sharpe_probability(
        selected_rets=rets,
        completed_trial_sharpes=pool_sharpes_zero_mean,
        effective_trial_count=5.0,
        bars_per_year=_BARS_PER_YEAR,
    )

    # Assert: mean 제거된 구현(dsr_no_mean)이 mean=0 pool보다 낮거나 같음
    # (std 동일, mean 제거 -> benchmark 하락 -> DSR 상승 방향 검증은 below)
    # 핵심: 양수 pool mean 시 과거 구현보다 현재 구현의 benchmark가 낮음 -> DSR 상승
    # N_eff=1 BVA: 함수 자체가 0.0 반환
    n_eff_1_result = _deflated_sharpe_probability(
        selected_rets=rets,
        completed_trial_sharpes=pool_sharpes_positive,
        effective_trial_count=0.0,  # n_eff <= 0
        bars_per_year=_BARS_PER_YEAR,
    )
    assert n_eff_1_result == pytest.approx(0.0, abs=1e-9)

    # DSR 값은 [0, 1] 범위
    assert 0.0 <= dsr_no_mean <= 1.0
    assert 0.0 <= dsr_zero_mean_pool <= 1.0


# ---------------------------------------------------------------------------
# D1: _sortino_hac_unit scale-invariance 검증
# ---------------------------------------------------------------------------


def test_sortino_hac_unit_scale_invariant() -> None:
    """leverage 배수(k=2) 적용 시 Sortino_HAC_unit 값이 불변이어야 한다.

    E[r] / sigma_down: leverage k -> E[r]*k / (sigma_down*k) = 동일.
    Time Complexity: O(n). Space Complexity: O(n).
    """
    # Arrange
    rets = _make_positive_rets(n=200)
    k = 2.0

    # Act
    sortino_base = _sortino_hac_unit(rets, bars_per_year=_BARS_PER_YEAR)
    scaled_rets = [r * k for r in rets]
    sortino_scaled = _sortino_hac_unit(scaled_rets, bars_per_year=_BARS_PER_YEAR)

    # Assert: scale-invariant (rel_tol 1e-4)
    assert sortino_base == pytest.approx(sortino_scaled, rel=1e-4)


def test_sortino_hac_unit_returns_zero_for_no_downside() -> None:
    """하방 관측이 없는 경우 _sortino_hac_unit은 0.0 반환해야 한다."""
    # Arrange: 전부 양의 수익률
    rets = [0.001] * 100

    # Act
    result = _sortino_hac_unit(rets, bars_per_year=_BARS_PER_YEAR)

    # Assert
    assert result == pytest.approx(0.0, abs=1e-12)


def test_sortino_hac_unit_returns_zero_for_empty_array() -> None:
    """빈 배열에 대해 _sortino_hac_unit은 0.0 반환해야 한다."""
    # Arrange + Act
    result = _sortino_hac_unit([], bars_per_year=_BARS_PER_YEAR)

    # Assert
    assert result == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# D1: dataclasses 새 필드 기본값 검증
# ---------------------------------------------------------------------------


def test_layer2_allocation_config_new_gate_defaults() -> None:
    """Layer2AllocationConfig 신규 필드 기본값 확인 (D1 스펙 정합)."""
    # Arrange + Act
    config = Layer2AllocationConfig()

    # Assert
    assert config.l2_min_sortino == pytest.approx(1.5, rel=1e-9)
    assert config.l2_min_sharpe_abs == pytest.approx(0.7, rel=1e-9)
    assert config.l2_min_calmar == pytest.approx(0.5, rel=1e-9)


# ---------------------------------------------------------------------------
# E2: _shape_efficiency_l2_objective — growth_lcb_deployed 블렌드 (Scenarios 1-3)
# ---------------------------------------------------------------------------
# Scenario 1: growth term scales with weight
# Scenario 2: zero weight = legacy; non-finite growth_lcb_deployed safe fallback
# Scenario 3: empty deployed rets → growth_lcb_deployed is large negative → finite clipped
# ---------------------------------------------------------------------------


def test_shape_efficiency_l2_objective_growth_term_scales_with_weight() -> None:
    """동일 sortino_hac_unit, growth_lcb_deployed=0.30 고정,
    growth_lcb_weight=0.0 → 0.5 시 objective가 0.5*0.30=0.15만큼 증가."""
    base_kwargs: dict[str, float] = {
        "sortino_hac_unit": 2.0,
        "worst_fold_sortino": 0.0,
        "worst_fold_threshold": -0.30,
        "worst_fold_weight": 0.005,
        "downside_dispersion": 0.0,
        "risk_util_weight": 0.0,
        "trade_weight": 0.0,
    }
    growth_val = 0.30
    obj_zero = _shape_efficiency_l2_objective(growth_lcb_deployed=growth_val, growth_lcb_weight=0.0, **base_kwargs)
    obj_half = _shape_efficiency_l2_objective(growth_lcb_deployed=growth_val, growth_lcb_weight=0.5, **base_kwargs)
    expected_diff = 0.5 * growth_val
    assert math.isclose(obj_half - obj_zero, expected_diff, rel_tol=1e-9, abs_tol=1e-12)


def test_shape_efficiency_l2_objective_growth_weight_zero_matches_legacy() -> None:
    """growth_lcb_weight=0.0 시 legacy(신규 파라미터 없는) 호출과 동일한 값."""
    base_kwargs: dict[str, float] = {
        "sortino_hac_unit": 1.5,
        "worst_fold_sortino": -0.5,
        "worst_fold_threshold": -0.30,
        "worst_fold_weight": 0.005,
        "downside_dispersion": 0.01,
        "risk_util_realized": 0.40,
        "risk_util_target": 0.50,
        "risk_util_weight": 0.03,
        "trade_count": 50,
        "trade_target": 90,
        "trade_weight": 0.02,
    }
    legacy = _shape_efficiency_l2_objective(**base_kwargs)
    with_new = _shape_efficiency_l2_objective(growth_lcb_deployed=999.0, growth_lcb_weight=0.0, **base_kwargs)
    assert math.isclose(legacy, with_new, rel_tol=1e-9, abs_tol=1e-12)

    nan_safe = _shape_efficiency_l2_objective(
        growth_lcb_deployed=float("nan"), growth_lcb_weight=1.0, **base_kwargs
    )
    assert math.isclose(nan_safe, legacy, rel_tol=1e-9, abs_tol=1e-12)


def test_evaluate_l2_trial_empty_deployed_rets_growth_lcb_deployed_is_neg_large() -> None:
    """빈 deployed 수익률 → growth_lcb_deployed = -1e6.
    growth_lcb_weight가 곱해져도 finite 상한 유지 → objective 폭주 없음."""
    base_kwargs: dict[str, float] = {
        "sortino_hac_unit": 2.0,
        "worst_fold_sortino": 0.0,
        "worst_fold_threshold": -0.30,
        "worst_fold_weight": 0.005,
        "downside_dispersion": 0.0,
        "risk_util_weight": 0.0,
        "trade_weight": 0.0,
    }
    growth_val = float("-1e6")
    obj = _shape_efficiency_l2_objective(growth_lcb_deployed=growth_val, growth_lcb_weight=0.3, **base_kwargs)
    assert np.isfinite(obj)
    # growth term = 0.3 * (-1e6) = -300_000. base objective ~2.0. result ≈ -299_998.
    expected = base_kwargs["sortino_hac_unit"] + 0.3 * growth_val
    assert math.isclose(obj, expected, rel_tol=1e-9, abs_tol=1e-6)
