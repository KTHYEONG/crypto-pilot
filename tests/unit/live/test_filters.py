"""SCENARIO_LIVE_03: 필터 양자화는 Decimal로 정확하며 미달은 드롭한다."""

from __future__ import annotations

from decimal import Decimal

from src.live.filters import SymbolFilters, parse_exchange_filters, quantize_order


def _filters() -> SymbolFilters:
    return SymbolFilters(
        symbol="TESTUSDT",
        tick_size=Decimal("0.10"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
        max_qty=Decimal("1000000"),
        quantity_precision=3,
        price_precision=2,
    )


def test_SCENARIO_LIVE_03_filter_quantization_exact() -> None:
    filters = _filters()

    buy = quantize_order(
        filters, Decimal("1.23456"), Decimal("100.07"), reduce_only=False
    )
    assert buy is not None
    assert buy.quantity == Decimal("1.234")
    assert buy.price == Decimal("100.00")

    sell = quantize_order(
        filters, Decimal("-1.23456"), Decimal("100.07"), reduce_only=False
    )
    assert sell is not None
    assert sell.price == Decimal("100.10")

    dropped_notional = quantize_order(
        filters, Decimal("0.04"), Decimal("100"), reduce_only=False
    )
    assert dropped_notional is None

    dropped_qty = quantize_order(
        filters, Decimal("0.0004"), Decimal("100"), reduce_only=False
    )
    assert dropped_qty is None

    assert isinstance(buy.quantity, Decimal)
    assert not isinstance(buy.quantity, float)


def test_parse_exchange_filters_keeps_only_tradable_perpetuals() -> None:
    exchange_info = {
        "symbols": [
            {
                "symbol": "AAAUSDT",
                "contractType": "PERPETUAL",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "quantityPrecision": 3,
                "pricePrecision": 2,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                ],
            },
            {
                "symbol": "DELUSDT",
                "contractType": "PERPETUAL",
                "quoteAsset": "USDT",
                "status": "SETTLING",
            },
            {
                "symbol": "BTCUSDC",
                "contractType": "PERPETUAL",
                "quoteAsset": "USDC",
                "status": "TRADING",
            },
        ]
    }
    parsed = parse_exchange_filters(exchange_info)
    assert set(parsed) == {"AAAUSDT"}
    assert parsed["AAAUSDT"].min_notional == Decimal("5")
    assert parsed["AAAUSDT"].position_control_side == "NONE"
    assert parsed["AAAUSDT"].blocks_risk_increase is False

#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_03_FILTER_QUANTIZATION_EXACT",
)


def test_parse_delivery_schedule_and_is_delisted() -> None:
    import pandas as pd
    from src.live.filters import DELISTED_STATUSES, DeliveryInfo, is_delisted, parse_delivery_schedule

    delivery_ms = int(pd.Timestamp("2026-09-10 08:00", tz="UTC").value // 1_000_000)
    info = {
        "symbols": [
            {"symbol": "AUSDT", "status": "SETTLING", "deliveryDate": delivery_ms},
            {"symbol": "BUSDT", "status": "TRADING", "deliveryDate": 4133404800000},
            {"symbol": "CUSDT", "status": "CLOSE"},
            {"status": "TRADING"},
        ]
    }

    schedule = parse_delivery_schedule(info)

    assert frozenset({"SETTLING", "CLOSE"}) == DELISTED_STATUSES
    assert set(schedule) == {"AUSDT", "BUSDT", "CUSDT"}
    assert schedule["AUSDT"] == DeliveryInfo(status="SETTLING", delivery_time=pd.Timestamp("2026-09-10 08:00", tz="UTC"))
    assert schedule["CUSDT"].delivery_time is None
    now = pd.Timestamp("2026-09-14 01:26", tz="UTC")
    assert is_delisted(schedule["AUSDT"], now) is True
    assert is_delisted(schedule["AUSDT"], pd.Timestamp("2026-09-10 07:59", tz="UTC")) is False
    assert is_delisted(schedule["BUSDT"], now) is False
    assert is_delisted(schedule["CUSDT"], now) is False


def test_held_symbols_absent_from_exchange_lists_nonzero_unlisted_positions() -> None:
    from decimal import Decimal

    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.filters import held_symbols_absent_from_exchange

    info = {"symbols": [{"symbol": "AAAUSDT", "status": "TRADING"}, {"symbol": "ZZZUSDT", "status": "SETTLING"}]}
    positions = {"GONEUSDT": Decimal("-2"), "AAAUSDT": Decimal("1"), "ZZZUSDT": Decimal("3"), "FLATUSDT": Decimal("0"), "BGONEUSDT": Decimal("0.5")}

    assert held_symbols_absent_from_exchange(positions, info) == ["BGONEUSDT", "GONEUSDT"]
    assert held_symbols_absent_from_exchange({}, info) == []
    with pytest.raises(DataIntegrityError):
        held_symbols_absent_from_exchange(positions, {})


def _filter_entry(symbol, extra_filters=()):
    return {
        "symbol": symbol,
        "contractType": "PERPETUAL",
        "quoteAsset": "USDT",
        "status": "TRADING",
        "quantityPrecision": 3,
        "pricePrecision": 2,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
            *extra_filters,
        ],
    }


def test_position_risk_control_is_parsed() -> None:
    """Spec 02: POSITION_RISK_CONTROL NONE parses with no restriction."""
    from src.live.filters import parse_exchange_filters

    info = {"symbols": [_filter_entry("AAAUSDT", [{"filterType": "POSITION_RISK_CONTROL", "positionControlSide": "NONE"}])]}
    parsed = parse_exchange_filters(info)
    assert parsed["AAAUSDT"].position_control_side == "NONE"
    assert parsed["AAAUSDT"].blocks_risk_increase is False


def test_unknown_control_side_blocks_risk_increases() -> None:
    """Spec 02: any non-NONE control side (including unknown values) blocks increases."""
    from decimal import Decimal

    from src.live.filters import POSITION_CONTROL_NONE, SymbolFilters, parse_exchange_filters

    assert POSITION_CONTROL_NONE == "NONE"
    for side in ("LONG", "SHORT", "BOTH", "FUTURE_VALUE"):
        info = {"symbols": [_filter_entry("AAAUSDT", [{"filterType": "POSITION_RISK_CONTROL", "positionControlSide": side}])]}
        parsed = parse_exchange_filters(info)
        assert parsed["AAAUSDT"].position_control_side == side
        assert parsed["AAAUSDT"].blocks_risk_increase is True
    assert SymbolFilters(
        symbol="X", tick_size=Decimal("0.1"), step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"), min_notional=Decimal("5"),
        max_qty=Decimal("1000"), quantity_precision=3, price_precision=2,
    ).blocks_risk_increase is False


def test_missing_control_filter_defaults_to_none() -> None:
    """Spec 02: entries without the filter parse and impose no restriction."""
    from src.live.filters import parse_exchange_filters

    parsed = parse_exchange_filters({"symbols": [_filter_entry("AAAUSDT")]})
    assert parsed["AAAUSDT"].position_control_side == "NONE"
    assert parsed["AAAUSDT"].blocks_risk_increase is False
