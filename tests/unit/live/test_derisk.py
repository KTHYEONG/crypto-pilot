# ruff: noqa
"""Live derisk tests - reduce-only filter."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from src.live.derisk import derisk_filter
from src.live.planner import OrderIntent


def _intent(
    symbol: str,
    side: str,
    quantity: str,
    *,
    reduce_only: bool,
    leg_index: int = 0,
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        reduce_only=reduce_only,
        target_qty=Decimal("0"),
        current_qty=Decimal("0"),
        client_order_prefix="20260914",
        leg_index=leg_index,
        decision_price=Decimal("100"),
    )


def test_derisk_filter_keeps_only_exposure_reducing_intents() -> None:
    """Only exposure-reducing intents survive."""
    current = {"AAAUSDT": Decimal("2"), "BUSDT": Decimal("-1"), "CUSDT": Decimal("0")}
    intents = [
        _intent("AAAUSDT", "SELL", "1", reduce_only=True),
        _intent("BUSDT", "BUY", "1", reduce_only=True, leg_index=0),
        _intent("BUSDT", "BUY", "1", reduce_only=False, leg_index=1),
        _intent("CUSDT", "BUY", "1", reduce_only=False),
    ]
    plan = derisk_filter(intents, current)
    assert [i.symbol for i in plan.kept] == ["AAAUSDT", "BUSDT"]
    assert plan.kept[1].leg_index == 0
    blocked = {i.symbol: reason for i, reason in plan.blocked}
    assert blocked["BUSDT"] == "flip_open_leg"
    assert blocked["CUSDT"] in ("no_position", "increase")


def test_derisk_filter_blocks_oversized_reduce_intent() -> None:
    """Reduce intent larger than the position is blocked."""
    plan = derisk_filter(
        [_intent("AAAUSDT", "SELL", "1.5", reduce_only=True)],
        {"AAAUSDT": Decimal("1")},
    )
    assert plan.kept == ()
    assert plan.blocked[0][1] == "exceeds_position"


def test_derisk_filter_blocks_foreign_order_symbols() -> None:
    """Foreign-order symbols are never traded."""
    plan = derisk_filter(
        [_intent("AAAUSDT", "SELL", "1", reduce_only=True)],
        {"AAAUSDT": Decimal("2")},
        excluded_symbols={"AAAUSDT"},
    )
    assert plan.kept == ()
    assert plan.blocked[0][1] == "foreign_order_symbol"


def test_derisk_filter_kept_intents_are_reduce_only() -> None:
    """Kept intents are always reduce-only."""
    current = {"AAAUSDT": Decimal("3"), "BUSDT": Decimal("-2")}
    intents = [
        _intent("AAAUSDT", "SELL", "3", reduce_only=True),
        _intent("AAAUSDT", "BUY", "1", reduce_only=False),
        _intent("BUSDT", "BUY", "2", reduce_only=True),
        _intent("BUSDT", "SELL", "1", reduce_only=True),
    ]
    plan = derisk_filter(intents, current)
    assert all(i.reduce_only for i in plan.kept)
    for intent in plan.kept:
        before = abs(current[intent.symbol])
        after = abs(current[intent.symbol] + (intent.quantity if intent.side == "BUY" else -intent.quantity))
        assert after <= before
