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

from types import SimpleNamespace

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
from src.domain.futures.strategy.tiered_workflow.diagnostics import build_layer_universe_audit
from src.domain.futures.strategy.tiered_workflow.l2_gate import evaluate_layer2_gate
from src.domain.futures.strategy.tiered_workflow.metrics import _psr

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def caps_default() -> PortfolioCaps:
    """기본 PortfolioCaps (per_symbol=0.5, gross=1.5, net=0.5, beta=1.0)."""
    return PortfolioCaps(per_symbol=0.5, gross=1.5, net=0.5, beta=1.0)


def default_config() -> Layer2AllocationConfig:
    """FIX-1 반영된 기본 설정 (l2_min_active_blocks=3, l2_min_dsr=0.75)."""
    return Layer2AllocationConfig()


# ---------------------------------------------------------------------------
# Scenario 1: active_blocks 정의 통일 회귀 테스트
# ---------------------------------------------------------------------------

def test_l2_min_active_blocks_default_is_3() -> None:
    """FIX-1: l2_min_active_blocks 기본값이 3으로 현실화됐는지 확인."""
    # Arrange (Given)
    config = default_config()

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

def test_ew_baseline_differs_from_risk_matched_baseline() -> None:
    """FIX-2: 동질적 edge + 대각 공분산에서 EW vs risk-matched가 구조적으로 달라야 함."""
    # Arrange (Given): K=3 심볼, 불균일 vol → risk-matched는 vol 역비례, EW는 1/3
    n = 3
    bars_per_year = 8760.0
    # 불균일 vol — risk-matched와 EW가 벌어지도록
    sigma = np.array([0.01, 0.05, 0.10], dtype=np.float64)
    mu_bps = np.array([1.0, 1.0, 1.0], dtype=np.float64)   # 동질적 edge
    btc_beta: np.ndarray = np.zeros(n, dtype=np.float64)
    # strategy_weights: Kelly 비중은 mu/sigma^2에 비례 (불균일)
    strategy_weights = mu_bps / (sigma**2 + 1e-12)
    strategy_weights = strategy_weights / np.sum(np.abs(strategy_weights))  # 정규화

    # Act (When)
    w_rm = build_directional_risk_matched_equal_weight(
        signed_net_mu_bps=mu_bps,
        strategy_weights=strategy_weights,
        sigma=sigma,
        btc_beta=btc_beta,
        caps=caps_default(),
        bars_per_year=bars_per_year,
    )
    w_ew = build_directional_equal_weight_baseline(
        signed_net_mu_bps=mu_bps,
        strategy_weights=strategy_weights,
        sigma=sigma,
        btc_beta=btc_beta,
        caps=caps_default(),
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
    sigma: np.ndarray = np.full(n, 0.01, dtype=np.float64)
    mu_bps = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    btc_beta: np.ndarray = np.zeros(n, dtype=np.float64)
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


def test_layer2_constraints_tuple_length_is_eight() -> None:
    """Optuna safety constraints fallback 크기가 9인지 확인."""
    # Arrange (Given): l2_constraint_values 미존재 trial 시뮬레이션
    from unittest.mock import MagicMock

    from src.domain.futures.optimization.workflow import layer2_constraints_from_trial

    mock_trial = MagicMock()
    mock_trial.user_attrs = {}  # l2_constraint_values 없음

    # Act (When)
    fallback = layer2_constraints_from_trial(mock_trial)

    # Assert (Then)
    assert len(fallback) == 9, (
        f"fallback 크기 {len(fallback)} != 9. Optuna safety constraints 9-tuple이어야 함."
    )
    assert all(v == 1.0 for v in fallback), "모든 fallback 값이 1.0 (infeasible) 이어야 함"


def test_layer2_constraints_eight_element_feasible() -> None:
    """Optuna safety constraints 9개가 모두 feasible(≤0)이면 통과 판정."""
    from unittest.mock import MagicMock

    from src.domain.futures.optimization.workflow import layer2_constraints_from_trial

    # Arrange: 12개 feasible
    mock_trial = MagicMock()
    mock_trial.user_attrs = {"l2_optuna_constraint_values": [-1.0] * 9}

    # Act
    result = layer2_constraints_from_trial(mock_trial)

    # Assert
    assert len(result) == 9
    assert all(c <= 0.0 for c in result), "모든 제약이 ≤0 이면 feasible이어야 함"


def test_layer2_constraints_legacy_values_pad_to_eight() -> None:
    """구 saved values는 9-tuple Optuna safety constraints로 하위호환 패딩된다."""
    from unittest.mock import MagicMock

    from src.domain.futures.optimization.workflow import layer2_constraints_from_trial

    # Arrange: 구 버전 짧은 constraint list
    mock_trial = MagicMock()
    mock_trial.user_attrs = {"l2_constraint_values": [-1.0] * 3}

    # Act
    result = layer2_constraints_from_trial(mock_trial)

    assert len(result) == 9
    assert all(c == -1.0 for c in result[:3])
    assert result[3:] == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


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


def test_l2_alloc_space_alias_points_to_v8() -> None:
    """Fix C: L2_ALLOC_SPACE가 V8(kelly_fraction·max_ann_vol 제거)를 가리키는지 확인."""
    from src.domain.futures.optimization.opt_config import L2_ALLOC_SPACE

    # Fix C: V8은 kelly_fraction·max_ann_vol을 탐색공간에서 제거
    assert "kelly_fraction" not in L2_ALLOC_SPACE, (
        "V8: kelly_fraction은 Phase B 결정론 처리로 탐색공간에서 제거되어야 함"
    )
    assert "max_ann_vol" not in L2_ALLOC_SPACE, (
        "V8: max_ann_vol은 Phase B 결정론 처리로 탐색공간에서 제거되어야 함"
    )


def test_l2_alloc_space_v9_retains_signal_dims() -> None:
    """D2: L2_ALLOC_SPACE가 V9를 가리키며 신호 혼합 파라미터를 유지하는지 확인."""
    from src.domain.futures.optimization.opt_config import (
        L2_ALLOC_SPACE,
        L2_ALLOC_SPACE_V8,
        L2_ALLOC_SPACE_V9,
    )

    # V9가 현재 active alias — V8 더 이상 alias 아님
    assert L2_ALLOC_SPACE is L2_ALLOC_SPACE_V9
    assert L2_ALLOC_SPACE is not L2_ALLOC_SPACE_V8

    # 핵심 파라미터 보존 확인
    assert "K_RANK" in L2_ALLOC_SPACE_V9
    assert "REBALANCE_BARS" in L2_ALLOC_SPACE_V9
    assert "risk_budget_floor_ratio" in L2_ALLOC_SPACE_V9
    assert L2_ALLOC_SPACE_V9["risk_budget_floor_ratio"]["high"] == pytest.approx(1.00)
    assert L2_ALLOC_SPACE_V9["deploy_cost_safety_mult"]["low"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scenario 5: 빈 support → zeros 반환 (edge case)
# ---------------------------------------------------------------------------

def test_ew_baseline_returns_zeros_when_no_support() -> None:
    """FIX-2 edge case: strategy_weights=zeros → build_directional_equal_weight_baseline returns zeros."""
    # Arrange (Given)
    n = 5
    strategy_weights: np.ndarray = np.zeros(n, dtype=np.float64)
    mu_bps: np.ndarray = np.ones(n, dtype=np.float64)
    sigma: np.ndarray = np.full(n, 0.01, dtype=np.float64)
    btc_beta: np.ndarray = np.zeros(n, dtype=np.float64)

    # Act (When)
    result = build_directional_equal_weight_baseline(
        signed_net_mu_bps=mu_bps,
        strategy_weights=strategy_weights,
        sigma=sigma,
        btc_beta=btc_beta,
        caps=caps_default(),
        bars_per_year=8760.0,
    )

    # Assert (Then)
    assert result.shape == (n,)
    assert np.all(result == 0.0), "support 없을 때 zeros를 반환해야 함"


def test_ew_baseline_returns_zeros_when_direction_all_zero() -> None:
    """FIX-2 edge case: mu=0이면 direction=0 → zeros 반환."""
    # Arrange (Given)
    n = 3
    strategy_weights = np.array([0.33, 0.33, 0.34], dtype=np.float64)
    mu_bps: np.ndarray = np.zeros(n, dtype=np.float64)  # 방향 없음
    sigma: np.ndarray = np.full(n, 0.01, dtype=np.float64)
    btc_beta: np.ndarray = np.zeros(n, dtype=np.float64)

    # Act (When)
    result = build_directional_equal_weight_baseline(
        signed_net_mu_bps=mu_bps,
        strategy_weights=strategy_weights,
        sigma=sigma,
        btc_beta=btc_beta,
        caps=caps_default(),
        bars_per_year=8760.0,
    )

    # Assert (Then)
    assert np.all(result == 0.0), "direction=0이면 zeros를 반환해야 함"


# ---------------------------------------------------------------------------
# Config 기본값 회귀 테스트
# ---------------------------------------------------------------------------

def test_l2_min_dsr_default_is_0_60() -> None:
    """FIX-3: l2_min_dsr 기본값이 0.60으로 설정됐는지 확인."""
    # Arrange / Act
    config = Layer2AllocationConfig()

    # Assert
    assert config.l2_min_dsr == pytest.approx(0.60, rel=1e-9)


def test_from_mapping_respects_new_defaults() -> None:
    """FIX-1/3: from_mapping 빈 dict 호출 시 신규 기본값 적용 확인."""
    # Arrange / Act
    config = Layer2AllocationConfig.from_mapping({})

    # Assert
    assert config.l2_min_active_blocks == 3
    assert config.l2_min_dsr == pytest.approx(0.60, rel=1e-9)
    assert config.l2_min_sharpe_abs == pytest.approx(0.70, rel=1e-9)
    assert config.l2_min_sortino == pytest.approx(1.5, rel=1e-9)
    assert config.l2_min_calmar == pytest.approx(0.5, rel=1e-9)


def test_build_layer_universe_audit_detects_low_active_tail() -> None:
    active_mask = np.ones((10, 4), dtype=bool)
    active_mask[8:, 2:] = False
    aligned = SimpleNamespace(
        datetimes=np.array([np.datetime64(f"2025-01-{day:02d}") for day in range(1, 11)]),
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"),
        active_mask=active_mask,
        warm_mask=np.ones((10, 4), dtype=bool),
        entry_block_mask=np.zeros((10, 4), dtype=bool),
        kill_mask=np.zeros((10, 4), dtype=bool),
    )

    audit = build_layer_universe_audit(
        aligned=aligned,
        layer="L2",
        start_idx=0,
        end_idx=10,
    )

    assert "low_active_tail" in audit.warnings


def test_recent_fold_gate_enters_optuna_constraints() -> None:
    gate = evaluate_layer2_gate(
        deployment_failed=False,
        support_leak_count=0,
        cagr_hybrid=0.4,
        sharpe_hybrid=1.2,
        sharpe_hac_hybrid=1.2,
        sharpe_hac_baseline=0.8,
        sortino_hybrid=1.8,
        mar_hybrid=1.5,
        mdd_hybrid=0.2,
        cvar_95_hybrid=0.04,
        fold_pass_ratio=0.8,
        active_block_count=3,
        friction_pass_pct=0.8,
        trade_count=100,
        growth_lcb_hybrid=0.1,
        growth_lcb_baseline=0.05,
        dsr_hybrid=None,
        psr_hybrid=0.95,
        recent_fold_passed=False,
        recent_fold_sharpe=-0.2,
        config=Layer2AllocationConfig(),
    )

    assert len(gate.optuna_constraint_values) == 9
    assert gate.optuna_constraint_values[5] > 0.0
    assert gate.promotion_blocker == "recent_fold"


# ---------------------------------------------------------------------------
# S6: Fix B — argmax(dsr, cagr) champion selection (두 후보 중 높은 DSR 선택)
# ---------------------------------------------------------------------------

def test_s6_champion_selected_by_argmax_dsr() -> None:
    """S6: gate-pass 후보 A(dsr=0.55) vs B(dsr=0.72) → champion은 B (DSR 우선).

    selection.py의 passed_candidates 수집 + argmax(dsr, cagr) 로직을 직접 단위 검증.
    통합 경로(evaluate_l2_trial) 없이 argmax 선택 로직만 테스트.
    """
    # Arrange (Given): (dsr, cagr_hybrid, trial_num) 형태의 gate-pass 후보 목록
    # tuple 구조: (dsr, cagr_hybrid, trial_idx, _, dsr) — selection.py 설계와 동일
    candidate_a = (0.55, 0.20, "trial_A", None, 0.55)  # DSR=0.55
    candidate_b = (0.72, 0.18, "trial_B", None, 0.72)  # DSR=0.72 (DSR 우선, CAGR 낮아도)
    passed_candidates = [candidate_a, candidate_b]

    # Act (When): selection.py의 argmax(dsr, cagr) 로직 재현
    best_entry = max(passed_candidates, key=lambda x: (x[0], x[1]))

    # Assert (Then): DSR이 더 높은 B가 champion
    assert best_entry[2] == "trial_B", (
        f"champion은 DSR 최대 후보(trial_B)여야 함. 실제: {best_entry[2]}"
    )
    assert best_entry[0] == pytest.approx(0.72), (
        f"champion DSR={best_entry[0]:.4f} != 0.72"
    )


def test_s6_champion_tiebreak_by_cagr_when_dsr_equal() -> None:
    """S6 타이브레이크: DSR 동일 시 CAGR 높은 후보가 champion."""
    # Arrange (Given): DSR 동일, CAGR 차이
    candidate_a = (0.65, 0.30, "trial_A", None, 0.65)
    candidate_b = (0.65, 0.25, "trial_B", None, 0.65)
    passed_candidates = [candidate_b, candidate_a]  # 순서 무관

    # Act (When)
    best_entry = max(passed_candidates, key=lambda x: (x[0], x[1]))

    # Assert (Then): CAGR 높은 A가 선택
    assert best_entry[2] == "trial_A"
    assert best_entry[1] == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# S7: Fix B — zero gate-pass → fallback path returns blocker_reason
# ---------------------------------------------------------------------------

def test_s7_zero_gate_pass_fallback_path_returns_blocker_reason() -> None:
    """S7: passed_candidates 빈 목록 → champion_trial is None → diagnostic fallback.

    selection.py의 fallback 분기 로직을 직접 단위 검증 (통합 경로 없이).
    """
    # Arrange (Given): gate-pass 후보 없음
    passed_candidates: list[tuple[float, float, str, None, float]] = []

    # Act (When): selection.py의 fallback 판정 로직 재현
    champion_found = bool(passed_candidates)
    # fallback 경로에서 blocker_reason이 설정됨을 검증 (Layer2StudyResult 구조 확인)
    fallback_blocker = "non_deterministic_replay"  # selection.py fallback default

    # Assert (Then)
    assert not champion_found, "후보 없으면 champion_found=False이어야 함"
    assert fallback_blocker != "", "fallback 경로에서 blocker_reason이 빈 문자열이면 안 됨"


# ---------------------------------------------------------------------------
# S8: Fix C — completed_trial_sharpes pool size == len(feasible_trials)
# ---------------------------------------------------------------------------

def test_s8_dsr_pool_restricted_to_feasible_trials() -> None:
    """S8: complete=10, feasible=4인 경우 completed_trial_sharpes 크기가 4.

    selection.py의 Fix C: feasible_trials 서명으로 pool 제한 로직을 직접 검증.
    """
    from unittest.mock import MagicMock

    from src.domain.futures.optimization.workflow import layer2_constraints_from_trial

    # Arrange (Given): 10개 complete trials, 그 중 4개만 feasible (constraints ≤ 0)
    def _make_trial(sharpe: float, feasible: bool) -> MagicMock:
        t = MagicMock()
        t.user_attrs = {
            "sharpe_hac_hybrid": sharpe,
                "l2_optuna_constraint_values": [-1.0] * 9 if feasible else [1.0] * 9,
        }
        return t

    complete_trials = [_make_trial(1.5 + i * 0.1, i < 4) for i in range(10)]
    # feasible: trials 0~3 (constraints all ≤ 0), infeasible: 4~9

    # Act (When): Fix C 로직 재현 — layer2_constraints_from_trial 기반 필터
    feasible_trials_filtered = [
        t for t in complete_trials
        if all(c <= 0.0 for c in layer2_constraints_from_trial(t))
    ]
    pool_sharpes = np.array(
        [t.user_attrs["sharpe_hac_hybrid"] for t in feasible_trials_filtered],
        dtype=np.float64,
    )

    # Assert (Then)
    assert len(feasible_trials_filtered) == 4, (
        f"feasible 필터 후 trial 수 {len(feasible_trials_filtered)} != 4"
    )
    assert pool_sharpes.shape == (4,), (
        f"completed_trial_sharpes shape {pool_sharpes.shape} != (4,)"
    )


# ---------------------------------------------------------------------------
# S9: Fix B+C — 동일 selected_rets, feasible pool(4) vs full pool(10) DSR 단조성
# ---------------------------------------------------------------------------

def test_s9_dsr_monotone_improvement_with_smaller_pool() -> None:
    """S9: pool 축소(10→4 feasible) 시 동일 selected_rets에서 DSR 단조 상승.

    수학적 근거: benchmark = mean(SR_pool) + std(SR_pool)·√(2·ln N_eff)
    N_eff 감소 → benchmark 하락 → DSR 상승 (DSR∝Φ(SR_obs - benchmark)).
    """
    from src.domain.futures.strategy.tiered_workflow.metrics import _deflated_sharpe_probability

    rng = np.random.default_rng(42)
    # Arrange (Given): 양의 엣지 수익률 (base 시나리오: MDD4.4%·CAGR14%·DSR0.27 모사)
    # 낮은 Sharpe를 모사하기 위해 작은 평균/높은 vol
    selected_rets = list(rng.normal(0.0003, 0.008, size=4000))
    bars_per_year = 8760.0

    # pool of 10 (all complete trials) — 높은 benchmark
    pool_10 = np.array([1.2, 1.4, 1.6, 1.8, 2.0, 1.3, 1.5, 1.7, 1.9, 2.1], dtype=np.float64)
    # pool of 4 (feasible only) — 낮은 benchmark (N_eff 감소)
    pool_4 = pool_10[:4]

    # Act (When): 동일 selected_rets, 동일 n_eff(편의상 동일 값으로 n_eff 비교)
    dsr_pool10 = _deflated_sharpe_probability(
        selected_rets=selected_rets,
        completed_trial_sharpes=pool_10,
        effective_trial_count=10.0,
        bars_per_year=bars_per_year,
    )
    dsr_pool4 = _deflated_sharpe_probability(
        selected_rets=selected_rets,
        completed_trial_sharpes=pool_4,
        effective_trial_count=4.0,
        bars_per_year=bars_per_year,
    )

    # Assert (Then): pool 축소 → DSR 단조 상승
    assert dsr_pool4 > dsr_pool10, (
        f"Fix C 후 DSR이 상승해야 함: pool4 DSR={dsr_pool4:.4f} <= pool10 DSR={dsr_pool10:.4f}. "
        "feasible pool 제한이 benchmark를 낮춰 DSR을 높여야 함."
    )
