"""(목표 - 현재) 델타를 OrderIntent로 분해한다. 포지션 반전은 2-intent로 쪼갠다."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from src.live.filters import _ZERO, SymbolFilters, quantize_order

_CLIENT_ORDER_ID_PATTERN = re.compile(r"^[.A-Za-z0-9_-]{1,36}$")


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """단일 자식 주문 의도. quantity는 항상 양수이고 방향은 side가 담는다."""

    symbol: str
    side: str
    quantity: Decimal
    reduce_only: bool
    target_qty: Decimal
    current_qty: Decimal
    client_order_prefix: str


def build_client_order_id(run_id: str, symbol: str, slice_idx: int, attempt: int) -> str:
    """결정론적 client order id. Binance 제약(^[.A-Za-z0-9_-]{1,36}$) 위반 시 ValueError."""
    candidate = f"{run_id}-{symbol}-{slice_idx}-{attempt}"
    if not _CLIENT_ORDER_ID_PATTERN.fullmatch(candidate):
        raise ValueError(
            "client order id violates Binance constraint ^[.A-Za-z0-9_-]{1,36}$: "
            f"{candidate!r}"
        )
    return candidate


def plan_orders(
    targets: Mapping[str, Decimal],
    current: Mapping[str, Decimal],
    filters: Mapping[str, SymbolFilters],
    marks: Mapping[str, Decimal],
    run_id: str,
) -> list[OrderIntent]:
    """부호 반대 델타(포지션 반전)는 reduceOnly 청산 후 신규 진입 두 intent로 분해한다."""
    intents: list[OrderIntent] = []
    for symbol, target in targets.items():
        symbol_filters = filters.get(symbol)
        mark = marks.get(symbol)
        if symbol_filters is None or mark is None or mark <= 0:
            continue
        current_qty = current.get(symbol, Decimal(0))
        delta = target - current_qty

        components: list[tuple[Decimal, bool]] = []
        if (
            current_qty != _ZERO
            and target != _ZERO
            and (target > _ZERO) != (current_qty > _ZERO)
        ):
            # 포지션 반전: reduceOnly 의미 보존을 위해 청산 -> 신규로 분해한다.
            components.append((-current_qty, True))
            components.append((target, False))
        else:
            components.append((delta, abs(target) < abs(current_qty)))

        for _slice_idx, (component_qty, reduce_only) in enumerate(components):
            quantized = quantize_order(
                symbol_filters, component_qty, mark, reduce_only=reduce_only
            )
            if quantized is None:
                continue
            intents.append(
                OrderIntent(
                    symbol=symbol,
                    side="BUY" if quantized.quantity > 0 else "SELL",
                    quantity=abs(quantized.quantity),
                    reduce_only=reduce_only,
                    target_qty=target,
                    current_qty=current_qty,
                    client_order_prefix=run_id,
                )
            )
    return intents
