"""SCENARIO_LIVE_05 / SCENARIO_LIVE_06: intent 분해와 결정론적 client order id."""

from __future__ import annotations

import itertools
import re
from decimal import Decimal

import pytest

from src.live.filters import SymbolFilters
from src.live.planner import build_client_order_id, plan_orders


def _filters(symbol: str) -> SymbolFilters:
    return SymbolFilters(
        symbol=symbol,
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
        max_qty=Decimal("1000000"),
        quantity_precision=3,
        price_precision=2,
    )


def test_SCENARIO_LIVE_05_position_flip_splits_into_two_intents() -> None:
    filters = {"AAAUSDT": _filters("AAAUSDT")}
    marks = {"AAAUSDT": Decimal("100")}

    flip = plan_orders(
        {"AAAUSDT": Decimal("-3")},
        {"AAAUSDT": Decimal("+2")},
        filters,
        marks,
        "run1",
    )
    assert len(flip) == 2
    first, second = flip
    assert (first.side, first.quantity, first.reduce_only) == ("SELL", Decimal("2"), True)
    assert (second.side, second.quantity, second.reduce_only) == ("SELL", Decimal("3"), False)

    reduce_only_scale_down = plan_orders(
        {"AAAUSDT": Decimal("1")}, {"AAAUSDT": Decimal("2")}, filters, marks, "run1"
    )
    assert len(reduce_only_scale_down) == 1
    intent = reduce_only_scale_down[0]
    assert (intent.side, intent.quantity, intent.reduce_only) == ("SELL", Decimal("1"), True)

    scale_up = plan_orders(
        {"AAAUSDT": Decimal("2")}, {"AAAUSDT": Decimal("1")}, filters, marks, "run1"
    )
    assert len(scale_up) == 1
    assert scale_up[0].reduce_only is False
    assert scale_up[0].side == "BUY"


def test_SCENARIO_LIVE_06_client_order_id_deterministic_and_valid() -> None:
    pattern = re.compile(r"^[.A-Za-z0-9_-]{1,36}$")

    once = build_client_order_id("run1", "BTCUSDT", 0, 0, 0)
    twice = build_client_order_id("run1", "BTCUSDT", 0, 0, 0)
    assert once == twice
    assert pattern.fullmatch(once)

    combos = itertools.product(("r0", "r1"), ("AAAUSDT", "BBBUSDT"), range(250), range(2))
    ids = {build_client_order_id(run_id, symbol, 0, s, a) for run_id, symbol, s, a in combos}
    assert len(ids) >= 1000

    with pytest.raises(ValueError, match="client order id"):
        build_client_order_id("x" * 40, "BTCUSDT", 0, 0, 0)
    with pytest.raises(ValueError, match="client order id"):
        build_client_order_id("bad id:콜론", "BTCUSDT", 0, 0, 0)


def test_SCENARIO_LIVE_20_flip_legs_are_distinct_and_bounded() -> None:
    """D2: 반전 분해의 leg 서수가 보존되고 clientOrderId 충돌이 사라진다."""
    filters = {"AAAUSDT": _filters("AAAUSDT")}
    marks = {"AAAUSDT": Decimal("100")}

    flip = plan_orders(
        {"AAAUSDT": Decimal("+1")},
        {"AAAUSDT": Decimal("-1")},
        filters,
        marks,
        "20260824",
    )
    assert len(flip) == 2
    assert [intent.leg_index for intent in flip] == [0, 1]
    first_ids = {
        intent.leg_index: build_client_order_id("20260824", "AAAUSDT", intent.leg_index, 0, 0)
        for intent in flip
    }
    assert len(set(first_ids.values())) == 2
    pattern = re.compile(r"^[.A-Za-z0-9_-]{1,36}$")
    assert all(pattern.fullmatch(coid) for coid in first_ids.values())

    # 최장 심볼 + attempt 99 에서도 36자 상한을 만족한다.
    longest = build_client_order_id("20260824", "1000000BABYDOGEUSDT", 1, 3, 99)
    assert len(longest) <= 36
    assert pattern.fullmatch(longest)

    # decision_price 는 결정 시점 mark 가 앵커로 싣는다.
    for intent in flip:
        assert intent.decision_price == Decimal("100")


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_05_POSITION_FLIP_SPLITS_INTO_TWO_INTENTS",
    "SCENARIO_LIVE_06_CLIENT_ORDER_ID_DETERMINISTIC_AND_VALID",
    "SCENARIO_LIVE_20",  # FLIP_LEGS_ARE_DISTINCT_AND_BOUNDED
)
