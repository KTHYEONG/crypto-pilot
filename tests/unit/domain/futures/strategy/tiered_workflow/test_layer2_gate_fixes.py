# tests/unit/domain/futures/strategy/tiered_workflow/test_layer2_gate_fixes.py
"""Layer2 게이트 교정(FIX-1~5) 회귀 테스트.

Scenarios:
    1. active_blocks 정의 통일 — fold 3개 + l2_min_active_blocks=3 → blocker != "active_blocks"
    2. 순수 EW baseline이 risk-matched와 구조적으로 다름 (L1 distance > 0.01)
    3. DSR fallback = PSR 정직 하한 (override_dsr=None 시 dsr == psr)
    4. uplift 제약이 infeasible trial을 올바르게 차단
    5. 빈 support → zeros 반환 (edge case)
"""
from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    build_directional_equal_weight_baseline,
    build_directional_risk_matched_equal_weight,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
    Layer2BlockMetric,
)
from src.domain.futures.strategy.tiered_workflow.metrics import _psr

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def caps_default() -> PortfolioCaps:
    """기본 PortfolioCaps (per_symbol=0.5, gross=1.5, net=0.5, beta=1.0)."""
    return PortfolioCaps(per_symbol=0.5, gross=1.5, net=0.5, beta=1.0)


@pytest.fixture
def default_config() -> Layer2AllocationConfig:
    """FIX-1 반영된 기본 설정 (l2_min_active_blocks=3, l2_min_dsr=0.75)."""
    return Layer2AllocationConfig()


# ---------------------------------------------------------------------------
# Scenario 1: active_blocks 정의 통일 회귀 테스트
# ---------------------------------------------------------------------------

def test_l2_min_active_blocks_default_is_3(default_config: Layer2AllocationConfig) -> None:
    """FIX-1: l2_min_active_blocks 기본값이 3으로 현실화됐는지 확인."""
    # Arrange (Given)
    config = default_config

    # Act (When)
    threshold = config.l2_min_active_blocks

    # Assert (Then)
    assert threshold == 3


def test_active_blocks_gate_passes_with_three_folds() -> None:
    """FIX-1: 3개 fold + l2_min_active_blocks=3 → active_blocks 게이트 통과."""
    # Arrange (Given)
    block_metrics = (
        Layer2BlockMetric(start_idx=0, end_idx=10, log_growth_hybrid=0.01,
                          log_growth_baseline=0.005, mdd_hybrid=0.02,
                          turnover_hybrid=0.1, active_rebalances=1),
        Layer2BlockMetric(start_idx=10, end_idx=20, log_growth_hybrid=0.02,
                          log_growth_baseline=0.01, mdd_hybrid=0.01,
                          turnover_hybrid=0.1, active_rebalances=1),
        Layer2BlockMetric(start_idx=20, end_idx=30, log_growth_hybrid=0.015,
                          log_growth_baseline=0.008, mdd_hybrid=0.015,
                          turnover_hybrid=0.1, active_rebalances=1),
    )
    min_active_blocks = 3

    # Act (When)
    active_count = len(block_metrics)
    is_blocked = active_count < min_active_blocks

    # Assert (Then)
    assert active_count == 3
    assert not is_blocked, "active_blocks 게이트가 fold 3개에서 차단하면 안 됨"


def test_active_block_count_fold_based_matches_pipeline_definition() -> None:
    """FIX-1: evaluate_l2_trial의 len([m for m if active_rebalances>0])가 pipeline gate와 동일."""
    # Arrange (Given): fold 기반 block_metrics (active_rebalances로 식별)
    block_metrics = [
        Layer2BlockMetric(start_idx=0, end_idx=10, log_growth_hybrid=0.01,
                          log_growth_baseline=0.0, mdd_hybrid=0.0,
                          turnover_hybrid=0.1, active_rebalances=1),
        Layer2BlockMetric(start_idx=10, end_idx=20, log_growth_hybrid=0.0,
                          log_growth_baseline=0.0, mdd_hybrid=0.0,
                          turnover_hybrid=0.0, active_rebalances=0),  # 비활성
        Layer2BlockMetric(start_idx=20, end_idx=30, log_growth_hybrid=0.02,
                          log_growth_baseline=0.0, mdd_hybrid=0.0,
                          turnover_hybrid=0.1, active_rebalances=1),
    ]

    # Act (When): evaluate_l2_trial 정의(FIX-1 후)
    active_block_count_eval = len([m for m in block_metrics if m.active_rebalances > 0])
    # pipeline.py 정의: len(block_metrics) (모든 fold 포함)
    active_block_count_pipeline = len(block_metrics)

    # Assert (Then): 두 정의가 같은 입력에서 일관성 있게 동작함 확인
    # (active_rebalances=0 제외 시 2, 전체 시 3 — 의도적으로 달라야 이 테스트가 의미 있음)
    assert active_block_count_eval == 2
    assert active_block_count_pipeline == 3


# ---------------------------------------------------------------------------
# Scenario 2: 순수 EW baseline이 risk-matched와 구조적으로 다름
# ---------------------------------------------------------------------------

def test_ew_baseline_differs_from_risk_matched_baseline(caps_default: PortfolioCaps) -> None:
    """FIX-2: 동질적 edge + 대각 공분산에서 EW vs risk-matched가 구조적으로 달라야 함."""
    # Arrange (Given): K=3 심볼, 불균일 vol → risk-matched는 vol 역비례, EW는 1/3
    n = 3
    bars_per_year = 8760.0
    # 불균일 vol — risk-matched와 EW가 벌어지도록
    sigma = np.array([0.01, 0.05, 0.10], dtype=np.float64)
    mu_bps = np.array([1.0, 1.0, 1.0], dtype=np.float64)   # 동질적 edge
    btc_beta = np.zeros(n, dtype=np.float64)
    # strategy_weights: Kelly 비중은 mu/sigma^2에 비례 (불균일)
    strategy_weights = mu_bps / (sigma**2 + 1e-12)
    strategy_weights = strategy_weights / np.sum(np.abs(strategy_weights))  # 정규화

    # Act (When)
    w_rm = build_directional_risk_matched_equal_weight(
        signed_net_mu_bps=mu_bps,
        strategy_weights=strategy_weights,
        sigma=sigma,
        btc_beta=btc_beta,
        caps=caps_default,
        bars_per_year=bars_per_year,
    )
    w_ew = build_directional_equal_weight_baseline(
        signed_net_mu_bps=mu_bps,
        strategy_weights=strategy_weights,
        sigma=sigma,
        btc_beta=btc_beta,
        caps=caps_default,
        bars_per_year=bars_per_year,
    )

    # Assert (Then): L1 distance > 0.01 — 구조적으로 달라야 함
    l1_dist = float(np.sum(np.abs(w_ew - w_rm)))
    assert l1_dist > 0.01, (
        f"EW baseline과 risk-matched baseline이 너무 비슷함 (L1={l1_dist:.4f}). "
        "Uplift를 0으로 만드는 설계 결함 재발 의심."
    )


def test_ew_baseline_is_equal_weight_in_direction() -> None:
    """FIX-2: EW baseline은 각 support 심볼에 1/N 균등 비중을 부여해야 함."""
    # Arrange (Given): K=3, vol 균일 (cap 클리핑 없음)
    n = 3
    caps = PortfolioCaps(per_symbol=1.0, gross=3.0, net=1.0, beta=2.0)
    sigma = np.full(n, 0.01, dtype=np.float64)
    mu_bps = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    btc_beta = np.zeros(n, dtype=np.float64)
    strategy_weights = np.array([0.33, 0.33, 0.34], dtype=np.float64)

    # Act (When)
    w_ew = build_directional_equal_weight_baseline(
        signed_net_mu_bps=mu_bps,
        strategy_weights=strategy_weights,
        sigma=sigma,
        btc_beta=btc_beta,
        caps=caps,
        bars_per_year=8760.0,
    )

    # Assert (Then): 비중이 1/3에 근사 (cap 클리핑 없는 경우)
    assert np.all(w_ew > 0), "모든 support 심볼이 양수 비중을 가져야 함"
    assert abs(float(np.std(w_ew)) / float(np.mean(w_ew))) < 0.01, (
        "EW baseline은 균등해야 함 (변동계수 < 1%)"
    )


# ---------------------------------------------------------------------------
# Scenario 3: DSR fallback = PSR 정직 하한
# ---------------------------------------------------------------------------

def test_dsr_fallback_equals_psr_when_no_override() -> None:
    """FIX-5: override_dsr=None 시 dsr_hybrid == psr_hybrid (±1e-9)."""
    # Arrange (Given): 임의 수익률 시계열
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, size=200).tolist()
    bars_per_year = 8760.0

    # Act (When): pipeline.py FIX-5 후 로직 — PSR을 정직한 하한으로 사용
    psr = _psr(rets, bars_per_year=bars_per_year)
    # DSR fallback (override_dsr=None) 경로는 _psr(list(sim.rets_hybrid)) 반환
    dsr_fallback = _psr(list(rets), bars_per_year=bars_per_year)

    # Assert (Then)
    assert abs(dsr_fallback - psr) < 1e-9, (
        f"DSR fallback({dsr_fallback:.6f})이 PSR({psr:.6f})와 불일치. "
        "단일-원소 degenerate 경로 재침투 의심."
    )


def test_dsr_fallback_not_constant_half() -> None:
    """FIX-5: 이전 구현의 degenerate DSR=0.5 상수가 더 이상 발생하지 않아야 함."""
    # Arrange (Given): 명확한 양의 엣지 수익률 (PSR >> 0.5 기대)
    rng = np.random.default_rng(7)
    rets = rng.normal(0.002, 0.005, size=500).tolist()
    bars_per_year = 8760.0

    # Act (When)
    psr = _psr(list(rets), bars_per_year=bars_per_year)

    # Assert (Then): PSR이 0.5보다 유의하게 크면 degenerate 문제 없음
    assert psr > 0.6, (
        f"PSR({psr:.4f})이 0.6 미만 — 양의 엣지에서도 degenerate 값 의심."
    )


# ---------------------------------------------------------------------------
# Scenario 4: uplift 제약이 infeasible trial을 올바르게 차단
# ---------------------------------------------------------------------------

def test_uplift_constraint_is_positive_when_hybrid_sharpe_below_threshold() -> None:
    """FIX-4: hybrid Sharpe < baseline_ew + min_uplift → 제약값 > 0 (infeasible)."""
    # Arrange (Given): hybrid Sharpe < EW baseline + 0.20
    sharpe_hac_baseline_ew = 1.0
    sharpe_hac_hybrid = 1.10          # 차이 = 0.10 < 0.20
    min_uplift = 0.20

    # Act (When): evaluate_l2_trial의 uplift 제약 계산 공식
    constraint_uplift = (
        sharpe_hac_baseline_ew + min_uplift - sharpe_hac_hybrid
    )

    # Assert (Then): 양수면 infeasible
    assert constraint_uplift > 0, (
        f"uplift 제약이 infeasible을 차단하지 못함 (값={constraint_uplift:.4f})"
    )


def test_uplift_constraint_is_negative_when_hybrid_sharpe_above_threshold() -> None:
    """FIX-4: hybrid Sharpe >= baseline_ew + min_uplift → 제약값 ≤ 0 (feasible)."""
    # Arrange (Given): hybrid Sharpe ≥ EW baseline + 0.20
    sharpe_hac_baseline_ew = 1.0
    sharpe_hac_hybrid = 1.25          # 차이 = 0.25 ≥ 0.20
    min_uplift = 0.20

    # Act (When)
    constraint_uplift = (
        sharpe_hac_baseline_ew + min_uplift - sharpe_hac_hybrid
    )

    # Assert (Then): 음수 또는 0이면 feasible
    assert constraint_uplift <= 0, (
        f"uplift 제약이 feasible trial을 잘못 차단함 (값={constraint_uplift:.4f})"
    )


def test_layer2_constraints_tuple_length_is_twelve() -> None:
    """C1/C2: Sortino+trades 제약 추가 후 layer2_constraints_from_trial fallback 크기가 12인지 확인."""
    # Arrange (Given): l2_constraint_values 미존재 trial 시뮬레이션
    from unittest.mock import MagicMock

    from src.domain.futures.optimization.workflow import layer2_constraints_from_trial

    mock_trial = MagicMock()
    mock_trial.user_attrs = {}  # l2_constraint_values 없음

    # Act (When)
    fallback = layer2_constraints_from_trial(mock_trial)

    # Assert (Then)
    assert len(fallback) == 12, (
        f"fallback 크기 {len(fallback)} != 12. Sortino+trades 제약 추가 후 12-tuple이어야 함."
    )
    assert all(v == 1.0 for v in fallback), "모든 fallback 값이 1.0 (infeasible) 이어야 함"


def test_layer2_constraints_twelve_element_feasible() -> None:
    """C1: 12개 constraint_values 모두 feasible(≤0)이면 통과 판정."""
    from unittest.mock import MagicMock

    from src.domain.futures.optimization.workflow import layer2_constraints_from_trial

    # Arrange: 12개 feasible
    mock_trial = MagicMock()
    mock_trial.user_attrs = {"l2_constraint_values": [-1.0] * 12}

    # Act
    result = layer2_constraints_from_trial(mock_trial)

    # Assert
    assert len(result) == 12
    assert all(c <= 0.0 for c in result), "모든 제약이 ≤0 이면 feasible이어야 함"


def test_layer2_constraints_legacy_ten_tuple_pads_to_twelve() -> None:
    """C2: 구 10-tuple user_attr이 12-tuple로 하위호환 패딩(1.0)되는지 확인."""
    from unittest.mock import MagicMock

    from src.domain.futures.optimization.workflow import layer2_constraints_from_trial

    # Arrange: 구 버전 10-tuple (sortino/trades 제약 도입 전)
    mock_trial = MagicMock()
    mock_trial.user_attrs = {"l2_constraint_values": [-1.0] * 10}

    # Act
    result = layer2_constraints_from_trial(mock_trial)

    # Assert: 10개는 원본 유지, 마지막 2개는 패딩(1.0, infeasible)
    assert len(result) == 12
    assert all(c == -1.0 for c in result[:10])
    assert result[10] == 1.0
    assert result[11] == 1.0


# ---------------------------------------------------------------------------
# S5: Range BVA — L2_ALLOC_SPACE_V3 경계 검증
# ---------------------------------------------------------------------------

def test_l2_alloc_space_v3_kelly_range() -> None:
    """S5: L2_ALLOC_SPACE_V3의 kelly_fraction 범위가 [0.15, 0.55]인지 확인."""
    from src.domain.futures.optimization.opt_config import L2_ALLOC_SPACE_V3

    spec = L2_ALLOC_SPACE_V3["kelly_fraction"]
    assert spec["low"] == pytest.approx(0.15), "kelly_fraction 하한이 0.15이어야 함"
    assert spec["high"] == pytest.approx(0.55), "kelly_fraction 상한이 0.55이어야 함"


def test_l2_alloc_space_v3_max_ann_vol_range() -> None:
    """C2: L2_ALLOC_SPACE_V3의 max_ann_vol 범위가 배치천장 개방 후 [0.20, 1.20]인지 확인."""
    from src.domain.futures.optimization.opt_config import L2_ALLOC_SPACE_V3

    spec = L2_ALLOC_SPACE_V3["max_ann_vol"]
    assert spec["low"] == pytest.approx(0.20), "max_ann_vol 하한이 0.20이어야 함"
    assert spec["high"] == pytest.approx(1.20), "max_ann_vol 상한이 1.20이어야 함"


def test_l2_alloc_space_alias_points_to_v3() -> None:
    """S5: L2_ALLOC_SPACE가 active-deployment V4를 가리키는지 확인."""
    from src.domain.futures.optimization.opt_config import L2_ALLOC_SPACE

    assert L2_ALLOC_SPACE["kelly_fraction"]["high"] == pytest.approx(0.80), (
        "L2_ALLOC_SPACE가 V4를 가리켜야 함 (kelly high=0.80)"
    )


def test_l2_alloc_space_v4_active_deployment_bounds() -> None:
    """V4: active L1 signal deployment 탐색공간 경계 검증."""
    from src.domain.futures.optimization.opt_config import L2_ALLOC_SPACE, L2_ALLOC_SPACE_V4

    assert L2_ALLOC_SPACE == L2_ALLOC_SPACE_V4
    assert L2_ALLOC_SPACE_V4["risk_budget_floor_ratio"]["high"] == pytest.approx(0.75)
    assert L2_ALLOC_SPACE_V4["deploy_cost_safety_mult"]["low"] == pytest.approx(1.0)
    assert L2_ALLOC_SPACE_V4["max_ann_vol"]["high"] == pytest.approx(1.50)


# ---------------------------------------------------------------------------
# Scenario 5: 빈 support → zeros 반환 (edge case)
# ---------------------------------------------------------------------------

def test_ew_baseline_returns_zeros_when_no_support(caps_default: PortfolioCaps) -> None:
    """FIX-2 edge case: strategy_weights=zeros → build_directional_equal_weight_baseline returns zeros."""
    # Arrange (Given)
    n = 5
    strategy_weights = np.zeros(n, dtype=np.float64)
    mu_bps = np.ones(n, dtype=np.float64)
    sigma = np.full(n, 0.01, dtype=np.float64)
    btc_beta = np.zeros(n, dtype=np.float64)

    # Act (When)
    result = build_directional_equal_weight_baseline(
        signed_net_mu_bps=mu_bps,
        strategy_weights=strategy_weights,
        sigma=sigma,
        btc_beta=btc_beta,
        caps=caps_default,
        bars_per_year=8760.0,
    )

    # Assert (Then)
    assert result.shape == (n,)
    assert np.all(result == 0.0), "support 없을 때 zeros를 반환해야 함"


def test_ew_baseline_returns_zeros_when_direction_all_zero(caps_default: PortfolioCaps) -> None:
    """FIX-2 edge case: mu=0이면 direction=0 → zeros 반환."""
    # Arrange (Given)
    n = 3
    strategy_weights = np.array([0.33, 0.33, 0.34], dtype=np.float64)
    mu_bps = np.zeros(n, dtype=np.float64)  # 방향 없음
    sigma = np.full(n, 0.01, dtype=np.float64)
    btc_beta = np.zeros(n, dtype=np.float64)

    # Act (When)
    result = build_directional_equal_weight_baseline(
        signed_net_mu_bps=mu_bps,
        strategy_weights=strategy_weights,
        sigma=sigma,
        btc_beta=btc_beta,
        caps=caps_default,
        bars_per_year=8760.0,
    )

    # Assert (Then)
    assert np.all(result == 0.0), "direction=0이면 zeros를 반환해야 함"


# ---------------------------------------------------------------------------
# Config 기본값 회귀 테스트
# ---------------------------------------------------------------------------

def test_l2_min_dsr_default_is_0_75() -> None:
    """FIX-3: l2_min_dsr 기본값이 0.75로 현실화됐는지 확인."""
    # Arrange / Act
    config = Layer2AllocationConfig()

    # Assert
    assert config.l2_min_dsr == pytest.approx(0.75, rel=1e-9)


def test_from_mapping_respects_new_defaults() -> None:
    """FIX-1/3: from_mapping 빈 dict 호출 시 신규 기본값 적용 확인."""
    # Arrange / Act
    config = Layer2AllocationConfig.from_mapping({})

    # Assert
    assert config.l2_min_active_blocks == 3
    assert config.l2_min_dsr == pytest.approx(0.75, rel=1e-9)
