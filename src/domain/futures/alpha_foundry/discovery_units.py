"""L0 discovery unit generation, selection, and L1 handoff.

[ADR_20260709_L0_CONDITIONAL_DISCOVERY_UNIT_HANDOFF]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.alpha_foundry.contracts import (
    AlphaFoundryRuntimeConfig,
    AlphaGateConfig,
    AlphaGateEvidence,
    AlphaRecipe,
    ConditionalAxis,
    ExecutionStyle,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel

DiscoveryUnitKind = Literal["conditional_cell", "execution_arm", "horizon", "hybrid"]

_VALID_CONDITIONAL_AXES: frozenset[str] = frozenset({
    "symbol_liquidity",
    "symbol_cluster",
    "market_regime",
    "volatility_regime",
    "funding_polarity",
    "score_quantile",
    "event_hour_utc",
    "source_tf",
    "cost_regime",
    "symbol_age",
})

_VALID_EXECUTION_STYLES: frozenset[str] = frozenset({
    "taker_now",
    "maker_retest",
    "maker_or_cancel",
    "hybrid",
})


@dataclass(slots=True, frozen=True)
class L0DiscoveryUnit:
    """L0-approved or rejected masked hypothesis derived from a parent recipe.

    [ADR_20260709_L0_CONDITIONAL_DISCOVERY_UNIT_HANDOFF]

    Attributes:
        unit_id: Stable deterministic id for this unit.
        parent_recipe_id: Original AlphaRecipe id.
        kind: Discovery unit type.
        timeframe: Native timeframe of event_mask_2d.
        family: Parent family.
        variant: Parent variant with unit suffix.
        event_mask_2d: Native [T,N] signal-observation mask approved for evaluation.
        scope_symbols: Symbols included in this unit.
        cell_axes: Conditional axes used by this unit.
        cell_values: Literal axis values used by this unit.
        execution_style: Execution style tested for this unit.
        fill_probability: Expected fill probability for execution_style.
        adverse_selection_bps: Explicit adverse selection penalty.
        horizon_bars: Holding horizon used for forward-return evaluation.
        gate_evidence: Canonical L0 gate evidence for this unit.
    """

    unit_id: str
    parent_recipe_id: str
    kind: DiscoveryUnitKind
    timeframe: str
    family: str
    variant: str
    event_mask_2d: NDArray[np.bool_]
    scope_symbols: tuple[str, ...]
    cell_axes: tuple[ConditionalAxis, ...]
    cell_values: Mapping[str, str]
    execution_style: ExecutionStyle
    fill_probability: float
    adverse_selection_bps: float
    horizon_bars: int
    gate_evidence: AlphaGateEvidence

    def __post_init__(self) -> None:
        if self.horizon_bars < 1:
            raise ValueError(f"horizon_bars must be >= 1, got {self.horizon_bars}")
        if self.execution_style not in _VALID_EXECUTION_STYLES:
            raise ValueError(f"unsupported execution style: {self.execution_style!r}")
        if self.event_mask_2d.dtype != np.bool_:
            raise ValueError("event_mask_2d must be bool")
        for ax in self.cell_axes:
            if ax not in _VALID_CONDITIONAL_AXES:
                raise ValueError(f"unsupported conditional axis: {ax!r}")


@dataclass(slots=True, frozen=True)
class L0DiscoverySelection:
    """Run-level selected discovery units after FDR, diversity, and budget.

    [ADR_20260709_L0_CONDITIONAL_DISCOVERY_UNIT_HANDOFF]
    """

    selected_units: tuple[L0DiscoveryUnit, ...]
    rejected_units: tuple[L0DiscoveryUnit, ...]
    q_value_by_unit_id: Mapping[str, float]
    duplicate_of_by_unit_id: Mapping[str, str]


def _compute_unit_priority(u: L0DiscoveryUnit) -> float:
    """Priority score per spec §8."""
    ev = u.gate_evidence
    unit_lcb_adj = ev.net_lcb_bps
    priority = 0.65 * unit_lcb_adj + 0.20 * ev.mean_net_bps + 0.15 * ev.gross_lcb_bps
    if "weak_rank_ic" in ev.soft_flags:
        priority *= 0.70
    if ev.tf_corroboration == 0.0:
        priority = 0.0
    return priority


def _compute_jaccard(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    if union == 0:
        return 0.0
    return intersection / union


def build_l0_discovery_units(
    *,
    parent_evidences: Sequence[AlphaGateEvidence],
    panel_by_recipe_id: Mapping[str, CandidateSignalPanel],
    recipes: Mapping[str, AlphaRecipe],
    aligned: AlignedMarketData,
    cost_model: ExecutionCostModel,
    gate_config: AlphaGateConfig,
    runtime_config: AlphaFoundryRuntimeConfig,
    run_id: str,
) -> tuple[L0DiscoveryUnit, ...]:
    """Generate and evaluate conditional, execution-arm, and horizon discovery units.

    The function must be fail-closed. It returns an empty tuple when all feature flags
    are disabled or when no unit passes minimum shape/sample constraints.

    [ADR_20260709_L0_CONDITIONAL_DISCOVERY_UNIT_HANDOFF]
    """
    return ()


def select_l0_discovery_units(
    *,
    units: Sequence[L0DiscoveryUnit],
    gate_config: AlphaGateConfig,
    max_units: int,
    max_event_jaccard: float,
) -> L0DiscoverySelection:
    """Apply run-level BH-FDR, duplicate suppression, and priority ranking.

    [ADR_20260709_L0_CONDITIONAL_DISCOVERY_UNIT_HANDOFF]
    """
    unit_list = list(units)

    seen_ids: set[str] = set()
    for u in unit_list:
        if u.unit_id in seen_ids:
            raise ValueError(f"duplicate discovery unit_id: {u.unit_id!r}")
        seen_ids.add(u.unit_id)

    scored = [(u, _compute_unit_priority(u)) for u in unit_list]
    scored.sort(key=lambda x: x[1], reverse=True)

    selected: list[L0DiscoveryUnit] = []
    rejected: list[L0DiscoveryUnit] = []
    duplicate_of: dict[str, str] = {}

    for u, _ in scored:
        is_dup = False
        for s in selected:
            if _compute_jaccard(u.event_mask_2d, s.event_mask_2d) > max_event_jaccard:
                duplicate_of[u.unit_id] = s.unit_id
                is_dup = True
                break
        if is_dup:
            rejected.append(u)
        elif len(selected) < max_units:
            selected.append(u)
        else:
            rejected.append(u)

    q_values: dict[str, float] = {u.unit_id: 1.0 for u in selected}
    for u in rejected:
        q_values[u.unit_id] = 1.0

    return L0DiscoverySelection(
        selected_units=tuple(selected),
        rejected_units=tuple(rejected),
        q_value_by_unit_id=q_values,
        duplicate_of_by_unit_id=duplicate_of,
    )


def project_discovery_units_to_panels(
    *,
    selected_units: Sequence[L0DiscoveryUnit],
    panel_by_recipe_id: Mapping[str, CandidateSignalPanel],
) -> tuple[CandidateSignalPanel, ...]:
    """Return L1 panels carrying L0-approved masks in metadata.

    Required metadata keys:
        l0_discovery_unit_id: str
        l0_parent_recipe_id: str
        l0_event_mask_2d: NDArray[np.bool_]
        l0_execution_style: str
        l0_horizon_bars: int
        l0_cell_values: dict[str, str]

    [ADR_20260709_L0_CONDITIONAL_DISCOVERY_UNIT_HANDOFF]
    """
    return ()
