"""L0 execution-arm cost/fill-probability modeling. [ADR_20260708_L0_EDGE_FAILURE_ATTRIBUTION]"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.alpha_foundry.contracts import AlphaRecipe, ExecutionArmConfig, ExecutionStyle
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel

_KNOWN_STYLES: frozenset[str] = frozenset({
    "taker_now",
    "maker_retest",
    "maker_or_cancel",
    "hybrid",
})


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
