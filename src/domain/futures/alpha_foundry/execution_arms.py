"""L0 execution-arm cost/fill-probability modeling.
[ADR_20260708_L0_EDGE_FAILURE_ATTRIBUTION][ADR_20260709_L0_CONDITIONAL_DIAGNOSTIC_WIRING]"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.alpha_foundry.conditional_cells import build_parent_event_mask, evaluate_event_mask_gate
from src.domain.futures.alpha_foundry.contracts import (
    AlphaGateConfig,
    AlphaGateEvidence,
    AlphaRecipe,
    ExecutionArmConfig,
    ExecutionStyle,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel

_KNOWN_STYLES: frozenset[str] = frozenset(
    {
        "taker_now",
        "maker_retest",
        "maker_or_cancel",
        "hybrid",
    }
)


@dataclass(slots=True, frozen=True)
class ExecutionCostArm:
    style: ExecutionStyle
    fill_probability: float
    base_round_trip_bps: float
    adverse_selection_bps: float
    unfilled_opportunity_cost_bps: float


def _build_taker_now_arm(cost_model: ExecutionCostModel) -> ExecutionCostArm:
    return ExecutionCostArm(
        style="taker_now",
        fill_probability=1.0,
        base_round_trip_bps=cost_model.stress_round_trip_bps(),
        adverse_selection_bps=0.0,
        unfilled_opportunity_cost_bps=0.0,
    )


def _build_maker_retest_arm(
    panel: CandidateSignalPanel,
    cost_model: ExecutionCostModel,
    config: ExecutionArmConfig,
) -> ExecutionCostArm:
    base_rt = cost_model.one_way_bps() * 2.0
    return ExecutionCostArm(
        style="maker_retest",
        fill_probability=0.60,
        base_round_trip_bps=base_rt,
        adverse_selection_bps=config.min_adverse_selection_bps * 1.5,
        unfilled_opportunity_cost_bps=base_rt * 0.5,
    )


def _build_maker_or_cancel_arm(
    cost_model: ExecutionCostModel,
    config: ExecutionArmConfig,
) -> ExecutionCostArm:
    base_rt = cost_model.one_way_bps() * 2.0
    return ExecutionCostArm(
        style="maker_or_cancel",
        fill_probability=0.40,
        base_round_trip_bps=base_rt,
        adverse_selection_bps=config.min_adverse_selection_bps,
        unfilled_opportunity_cost_bps=base_rt * 0.25,
    )


def _build_hybrid_arm(
    cost_model: ExecutionCostModel,
    config: ExecutionArmConfig,
) -> ExecutionCostArm:
    taker_rt = cost_model.taker_round_trip_bps()
    maker_rt = cost_model.one_way_bps() * 2.0
    hybrid_rt = 0.4 * maker_rt + 0.6 * taker_rt
    return ExecutionCostArm(
        style="hybrid",
        fill_probability=0.85,
        base_round_trip_bps=hybrid_rt,
        adverse_selection_bps=config.min_adverse_selection_bps * 0.5,
        unfilled_opportunity_cost_bps=taker_rt * 0.3,
    )


_ARM_BUILDERS = {
    "taker_now": _build_taker_now_arm,
    "maker_retest": _build_maker_retest_arm,
    "maker_or_cancel": _build_maker_or_cancel_arm,
    "hybrid": _build_hybrid_arm,
}


def resolve_execution_cost_arms(
    *,
    panel: CandidateSignalPanel,
    aligned: AlignedMarketData,
    recipe: AlphaRecipe,
    cost_model: ExecutionCostModel,
    config: ExecutionArmConfig,
) -> tuple[ExecutionCostArm, ...]:
    if not config.enabled:
        return (_build_taker_now_arm(cost_model),)

    arms: list[ExecutionCostArm] = []
    for style in config.styles:
        if style not in _KNOWN_STYLES:
            raise ValueError(f"unsupported execution style: {style!r}")
        if style == "maker_retest":
            arm = _build_maker_retest_arm(panel, cost_model, config)
        elif style == "maker_or_cancel":
            arm = _build_maker_or_cancel_arm(cost_model, config)
        elif style == "hybrid":
            arm = _build_hybrid_arm(cost_model, config)
        else:
            arm = _build_taker_now_arm(cost_model)
        arms.append(arm)

    if len(arms) > config.max_arm_count_per_cell:
        arms = arms[: config.max_arm_count_per_cell]
    return tuple(arms)


def estimate_execution_arm_cost_bps(
    *,
    event_mask_2d: NDArray[np.bool_],
    arm: ExecutionCostArm,
    aligned: AlignedMarketData,
    holding_bars: int,
) -> NDArray[np.float64]:
    t, n = event_mask_2d.shape
    out = np.full((t, n), np.nan, dtype=np.float64)
    if not np.any(event_mask_2d):
        return out

    if arm.fill_probability <= 0.0:
        out[event_mask_2d] = np.nan
        return out

    base = arm.base_round_trip_bps * (1.0 + arm.adverse_selection_bps / max(arm.base_round_trip_bps, 1e-10))
    unfilled = arm.unfilled_opportunity_cost_bps * (1.0 - arm.fill_probability)
    cost = base + unfilled
    out[event_mask_2d] = cost
    return out


def evaluate_recipe_under_arm(
    *,
    panel: CandidateSignalPanel,
    aligned: AlignedMarketData,
    recipe: AlphaRecipe,
    arm: ExecutionCostArm,
    gate_config: AlphaGateConfig,
    bars_per_year: float,
    run_id: str,
) -> AlphaGateEvidence:
    """Re-evaluates the recipe's full (unsliced) event set under an alternate execution
    cost assumption. Reuses `estimate_execution_arm_cost_bps()` for the cost figure
    (extracted as a scalar — the function is constant across all True mask cells in the
    current implementation) and delegates statistical evaluation to
    `conditional_cells.evaluate_event_mask_gate()`. [LIMIT-03]
    """
    event_mask = build_parent_event_mask(panel=panel, aligned=aligned)

    if not np.any(event_mask):
        raise ValueError("recipe has no valid events for arm evaluation")

    cost_arr = estimate_execution_arm_cost_bps(
        event_mask_2d=event_mask,
        arm=arm,
        aligned=aligned,
        holding_bars=panel.expected_holding_bars,
    )

    cost_scalar = float(cost_arr[event_mask][0])

    return evaluate_event_mask_gate(
        event_mask=event_mask,
        panel=panel,
        aligned=aligned,
        recipe=recipe,
        round_trip_cost_bps=cost_scalar,
        gate_config=gate_config,
        bars_per_year=bars_per_year,
        run_id=run_id,
    )
