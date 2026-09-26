"""De-risk-only execution under account-state uncertainty: only intents that strictly shrink an existing position toward its target are allowed, so no action taken while the account state is unexplained can increase exposure or open new risk."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from src.live.planner import OrderIntent

DeriskReason = Literal["reconciliation_breach", "foreign_open_orders", "free_margin_floor"]


@dataclass(frozen=True, slots=True)
class DeriskPlan:
    kept: tuple[OrderIntent, ...]
    blocked: tuple[tuple[OrderIntent, str], ...]  # (intent, block reason)


def derisk_filter(
    intents: Sequence[OrderIntent],
    current_positions: Mapping[str, Decimal],
    *,
    excluded_symbols: Collection[str] = (),
) -> DeriskPlan:
    """Keep an intent only if it reduces |position|: it is flagged `reduce_only`, its side is opposite to the sign of the current venue position, and its quantity does not exceed |current position|. Intents on `excluded_symbols` (symbols with foreign open orders) are blocked.

    Returns: DeriskPlan preserving the input order of kept intents; each blocked intent carries one of `increase`, `flip_open_leg`, `exceeds_position`, `foreign_order_symbol`, `no_position`.
    """
    excluded = set(excluded_symbols)
    kept: list[OrderIntent] = []
    blocked: list[tuple[OrderIntent, str]] = []
    for intent in intents:
        if intent.symbol in excluded:
            blocked.append((intent, "foreign_order_symbol"))
            continue
        current = current_positions.get(intent.symbol, Decimal(0))
        if current == 0:
            blocked.append((intent, "no_position"))
            continue
        if intent.leg_index == 1 and not intent.reduce_only:
            blocked.append((intent, "flip_open_leg"))
            continue
        if not intent.reduce_only:
            blocked.append((intent, "increase"))
            continue
        expected_side = "SELL" if current > 0 else "BUY"
        if intent.side != expected_side:
            blocked.append((intent, "increase"))
            continue
        if intent.quantity > abs(current):
            blocked.append((intent, "exceeds_position"))
            continue
        kept.append(intent)
    return DeriskPlan(kept=tuple(kept), blocked=tuple(blocked))
