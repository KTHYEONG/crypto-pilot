"""목표비중 -> 필터 반영 목표수량. 드롭된 gross는 절대 재분배하지 않는다(I-FILTER-EXACT)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from src.live.filters import SymbolFilters, quantize_order


@dataclass(frozen=True, slots=True)
class DroppedSymbol:
    symbol: str
    reason: str
    target_notional: Decimal


def target_quantities(
    weights: pd.Series,
    marks: Mapping[str, Decimal],
    filters: Mapping[str, SymbolFilters],
    equity_usdt: Decimal,
) -> tuple[dict[str, Decimal], list[DroppedSymbol]]:
    """종목별 목표 노셔널 = equity * weight, 목표 수량 = 노셔널 / mark."""
    targets: dict[str, Decimal] = {}
    dropped: list[DroppedSymbol] = []
    for symbol, weight in weights.items():
        sym = str(symbol)
        target_notional = equity_usdt * Decimal(str(float(weight)))
        symbol_filters = filters.get(sym)
        mark = marks.get(sym)
        if symbol_filters is None or mark is None or mark <= 0:
            dropped.append(DroppedSymbol(sym, "NOT_TRADABLE", target_notional))
            continue
        raw_qty = target_notional / mark
        quantized = quantize_order(symbol_filters, raw_qty, mark, reduce_only=False)
        if quantized is None:
            reason = "MIN_QTY" if abs(raw_qty) < symbol_filters.min_qty else "MIN_NOTIONAL"
            dropped.append(DroppedSymbol(sym, reason, target_notional))
            continue
        targets[sym] = quantized.quantity
    return targets, dropped
