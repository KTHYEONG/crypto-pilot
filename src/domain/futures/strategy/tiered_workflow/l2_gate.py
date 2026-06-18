from __future__ import annotations

from typing import cast

import numpy as np

from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
    Layer2GateEvaluation,
)

_DEFAULT_L2_CONFIG: Layer2AllocationConfig = Layer2AllocationConfig()

_PROMOTION_BLOCKERS: tuple[str, ...] = (
    "no_deployment",
    "low_trades",
    "cagr",
    "sharpe_abs",
    "sortino_floor",
    "calmar_floor",
    "mar",
    "mdd_abs",
    "cvar_95",
    "fold",
    "active_blocks",
    "friction",
    "growth_lcb",
    "uplift",
    "psr_floor",
)


def _finite_or_fail(value: float, *, default_fail: float = 1.0) -> float:
    return float(value) if np.isfinite(value) else float(default_fail)


def evaluate_layer2_gate(
    *,
    deployment_failed: bool,
    support_leak_count: int,
    cagr_hybrid: float,
    sharpe_hybrid: float,
    sharpe_hac_hybrid: float,
    sharpe_hac_baseline: float,
    sortino_hybrid: float,
    mar_hybrid: float,
    mdd_hybrid: float,
    cvar_95_hybrid: float,
    fold_pass_ratio: float,
    active_block_count: int,
    friction_pass_pct: float,
    trade_count: int,
    growth_lcb_hybrid: float,
    growth_lcb_baseline: float,
    dsr_hybrid: float | None,
    psr_hybrid: float | None = None,
    config: Layer2AllocationConfig = _DEFAULT_L2_CONFIG,
) -> Layer2GateEvaluation:
    """Build Optuna safety constraints and final L2 promotion gate diagnostics.

    Args:
        deployment_failed: 시뮬레이션 배포 실패 여부.
        support_leak_count: look-ahead 누수 카운트.
        cagr_hybrid: 전략 연율화 복리 수익률.
        sharpe_hybrid: 전략 Sharpe 비율.
        sharpe_hac_hybrid: HAC 조정 Sharpe 비율.
        sharpe_hac_baseline: 기준선 HAC Sharpe.
        sortino_hybrid: 전략 Sortino 비율.
        mar_hybrid: CAGR / MDD 비율.
        mdd_hybrid: 최대 낙폭 (양수).
        cvar_95_hybrid: 95% CVaR 손실.
        fold_pass_ratio: 복리 기준 수익 fold 비율.
        active_block_count: 활성 블록 수.
        friction_pass_pct: 마찰 허들 통과 비율.
        trade_count: 총 거래 수.
        growth_lcb_hybrid: 성장 LCB (diagnostic).
        growth_lcb_baseline: 기준선 성장 LCB.
        dsr_hybrid: DSR (diagnostic only — BLOCKER 아님).
        psr_hybrid: PSR — L2 하드게이트 (None 시 통과).
        config: Layer2AllocationConfig.

    Returns:
        Layer2GateEvaluation.
    """
    optuna_constraint_values = (
        1.0 if deployment_failed else -1.0,
        float(max(support_leak_count, 0)),
        _finite_or_fail(mdd_hybrid - float(config.l2_max_mdd_abs)),
        _finite_or_fail(cvar_95_hybrid - float(config.l2_max_cvar_95)),
        _finite_or_fail(float(config.l2_min_fold_pass_ratio) - fold_pass_ratio),
        _finite_or_fail(float(int(config.l2_min_active_blocks) - active_block_count)),
        _finite_or_fail(float(config.l2_min_friction_pass) - friction_pass_pct),
        _finite_or_fail(float(int(config.l2_min_trades) - trade_count)),
    )

    # calmar = CAGR / MDD; mar_hybrid 이미 동일 계산이나 명시적 calmar 분리
    calmar_hybrid = float(cagr_hybrid) / (float(mdd_hybrid) + 1e-9)

    # psr_floor: PSR < threshold → BLOCKER (None 입력 시 -1.0 통과)
    psr_constraint = (
        -1.0
        if psr_hybrid is None
        else _finite_or_fail(float(config.l2_min_psr) - float(psr_hybrid))
    )

    promotion_constraint_values = (
        1.0 if deployment_failed else -1.0,
        _finite_or_fail(float(int(config.l2_min_trades) - trade_count)),
        _finite_or_fail(float(config.l2_min_cagr) - cagr_hybrid),
        _finite_or_fail(float(config.l2_min_sharpe_abs) - sharpe_hybrid),
        _finite_or_fail(float(config.l2_min_sortino) - sortino_hybrid),
        _finite_or_fail(float(config.l2_min_calmar) - calmar_hybrid),
        _finite_or_fail(float(config.l2_min_mar) - mar_hybrid),
        _finite_or_fail(mdd_hybrid - float(config.l2_max_mdd_abs)),
        _finite_or_fail(cvar_95_hybrid - float(config.l2_max_cvar_95)),
        _finite_or_fail(float(config.l2_min_fold_pass_ratio) - fold_pass_ratio),
        _finite_or_fail(float(int(config.l2_min_active_blocks) - active_block_count)),
        _finite_or_fail(float(config.l2_min_friction_pass) - friction_pass_pct),
        _finite_or_fail(
            growth_lcb_baseline + float(config.l2_min_growth_uplift) - growth_lcb_hybrid
        ),
        _finite_or_fail(
            sharpe_hac_baseline + float(config.l2_min_sharpe_uplift) - sharpe_hac_hybrid
        ),
        psr_constraint,
    )

    promotion_passed = True
    promotion_blocker = ""
    for blocker, value in zip(
        _PROMOTION_BLOCKERS,
        promotion_constraint_values,
        strict=True,
    ):
        if value > 0.0:
            promotion_passed = False
            promotion_blocker = blocker
            break

    return Layer2GateEvaluation(
        optuna_constraint_values=cast(tuple[float, ...], optuna_constraint_values),
        promotion_passed=promotion_passed,
        promotion_blocker=promotion_blocker,
        promotion_constraint_values=cast(tuple[float, ...], promotion_constraint_values),
    )
