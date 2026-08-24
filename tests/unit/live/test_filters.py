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

#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_03_FILTER_QUANTIZATION_EXACT",
)
