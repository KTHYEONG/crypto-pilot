"""Exchange symbol filters and exact Decimal order quantization.

float 라운딩은 -1013 필터 위반의 직접 원인이므로 이 모듈의 모든 산술은 Decimal이다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any, cast

from src.common.errors import DataIntegrityError

_ZERO = Decimal(0)
_ONE = Decimal(1)


@dataclass(frozen=True, slots=True)
class QuantizedOrder:
    """양자화를 통과한 주문 파라미터. quantity는 항상 양수다."""

    quantity: Decimal
    price: Decimal
    reduce_only: bool


@dataclass(frozen=True, slots=True)
class SymbolFilters:
    """단일 심볼의 거래소 필터 스냅샷."""

    symbol: str
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal
    max_qty: Decimal
    quantity_precision: int
    price_precision: int


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise DataIntegrityError(f"exchangeInfo missing {context}.{key}")
    return mapping[key]


def _filter_of(entry: Mapping[str, Any], filter_type: str) -> Mapping[str, Any]:
    for item in entry.get("filters", ()):
        if item.get("filterType") == filter_type:
            return cast(Mapping[str, Any], item)
    raise DataIntegrityError(f"symbol {entry.get('symbol')} missing filter {filter_type}")


def _notional_filter(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    for filter_type in ("MIN_NOTIONAL", "NOTIONAL"):
        for item in entry.get("filters", ()):
            if item.get("filterType") == filter_type:
                return cast(Mapping[str, Any], item)
    raise DataIntegrityError(
        f"symbol {entry.get('symbol')} missing filter MIN_NOTIONAL/NOTIONAL"
    )


def parse_exchange_filters(exchange_info: Mapping[str, Any]) -> dict[str, SymbolFilters]:
    """PERPETUAL + USDT quote + TRADING 심볼만 파싱한다. 나머지는 거래 대상이 아니다."""
    parsed: dict[str, SymbolFilters] = {}
    symbols = _required(exchange_info, "symbols", "exchangeInfo")
    for entry in symbols:
        if (
            entry.get("contractType") != "PERPETUAL"
            or entry.get("quoteAsset") != "USDT"
            or entry.get("status") != "TRADING"
        ):
            continue
        symbol = str(_required(entry, "symbol", "symbol entry"))
        price_filter = _filter_of(entry, "PRICE_FILTER")
        lot_size = _filter_of(entry, "LOT_SIZE")
        notional = _notional_filter(entry)
        min_notional_key = "minNotional" if "minNotional" in notional else "notional"
        parsed[symbol] = SymbolFilters(
            symbol=symbol,
            tick_size=Decimal(str(_required(price_filter, "tickSize", f"{symbol} PRICE_FILTER"))),
            step_size=Decimal(str(_required(lot_size, "stepSize", f"{symbol} LOT_SIZE"))),
            min_qty=Decimal(str(_required(lot_size, "minQty", f"{symbol} LOT_SIZE"))),
            min_notional=Decimal(str(_required(notional, min_notional_key, f"{symbol} NOTIONAL"))),
            max_qty=Decimal(str(_required(lot_size, "maxQty", f"{symbol} LOT_SIZE"))),
            quantity_precision=int(_required(entry, "quantityPrecision", symbol)),
            price_precision=int(_required(entry, "pricePrecision", symbol)),
        )
    return parsed


def _quantize(value: Decimal, multiple: Decimal, rounding: str) -> Decimal:
    if multiple <= _ZERO:
        raise ValueError("quantization multiple must be positive")
    return (value / multiple).to_integral_value(rounding=rounding) * multiple


def quantize_order(
    filters: SymbolFilters,
    target_qty: Decimal,
    reference_price: Decimal,
    *,
    reduce_only: bool,
) -> QuantizedOrder | None:
    """수량/가격을 거래소 필터에 정확히 양자화한다. 미달이면 None으로 드롭한다.

    수량은 항상 ROUND_DOWN(오버슛 금지), 가격은 매수 ROUND_DOWN / 매도 ROUND_UP으로
    패시브 측을 유지한다. reduce_only 전량 청산은 최소노셔널 면제 케이스로 허용한다.
    """
    side_buy = target_qty >= _ZERO
    qty = abs(target_qty)

    qty = min(qty, filters.max_qty)
    qty = _quantize(qty, filters.step_size, ROUND_DOWN)

    if qty < filters.min_qty:
        return None

    if side_buy:
        price = _quantize(reference_price, filters.tick_size, ROUND_DOWN)
    else:
        price = _quantize(reference_price, filters.tick_size, ROUND_UP)
    if price <= _ZERO:
        price = filters.tick_size

    notional = qty * price
    if notional < filters.min_notional and not (reduce_only and target_qty != _ZERO):
        return None

    signed_qty = qty if side_buy else -qty
    return QuantizedOrder(quantity=signed_qty, price=price, reduce_only=reduce_only)
