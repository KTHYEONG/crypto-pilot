from __future__ import annotations

from typing import cast

import numpy as np

from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
    Layer2GateEvaluation,
)

_PROMOTION_BLOCKERS: tuple[str, ...] = (
    "no_deployment",
    "low_trades",
    "cagr",
    "sharpe_abs",
    "sortino",
    "mar",
    "mdd_abs",
    "cvar_95",
    "fold",
    "active_blocks",
    "friction",
    "growth_lcb",
    "uplift",
    "dsr_floor",
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
    config: Layer2AllocationConfig,
) -> Layer2GateEvaluation:
    """Build Optuna safety constraints and final L2 promotion gate diagnostics."""
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

    promotion_constraint_values = (
        1.0 if deployment_failed else -1.0,
        _finite_or_fail(float(int(config.l2_min_trades) - trade_count)),
        _finite_or_fail(float(config.l2_min_cagr) - cagr_hybrid),
        _finite_or_fail(float(config.l2_min_sharpe_abs) - sharpe_hybrid),
        _finite_or_fail(float(config.l2_min_sortino_abs) - sortino_hybrid),
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
        (
            -1.0
            if dsr_hybrid is None
            else _finite_or_fail(float(config.l2_min_dsr) - float(dsr_hybrid))
        ),
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
