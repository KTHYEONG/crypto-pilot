"""(목표 - 현재) 델타를 OrderIntent로 분해한다. 포지션 반전은 2-intent로 쪼갠다."""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from src.live.filters import _ZERO, SymbolFilters, quantize_order

_CLIENT_ORDER_ID_PATTERN = re.compile(r"^[.A-Za-z0-9_-]{1,36}$")

CLIENT_ORDER_NAMESPACE = "mh"


def symbol_tag(symbol: str) -> str:
    """심볼 식별 태그: sha256 -> base32 상위 10자(ASCII, Binance 문자셋)."""
    return base64.b32encode(hashlib.sha256(symbol.encode("utf-8")).digest()).decode("ascii")[:10]


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """단일 자식 주문 의도. quantity는 항상 양수이고 방향은 side가 담는다.

    leg_index는 반전 분해의 leg 서수(청산=0, 신규=1)로 clientOrderId 고유성을 보장하고,
    decision_price는 결정 시점 mark로 재호가 밴드(I-CHASE-BAND)의 앵커다.
    """

    symbol: str
    side: str
    quantity: Decimal
    reduce_only: bool
    target_qty: Decimal
    current_qty: Decimal
    client_order_prefix: str
    leg_index: int
    decision_price: Decimal


def build_client_order_id(run_id: str, symbol: str, leg_index: int, generation: int, submit_seq: int) -> str:
    """결정론적 client order id. Binance 제약(^[.A-Za-z0-9_-]{1,36}$) 위반 시 ValueError.

    'mh' 네임스페이스 + 심볼 태그(base32 10자) + leg/generation/submit_seq 로
    계정 수명 전체에서 고유한 id 를 만든다. 비-ASCII 심볼도 태그로 인코딩된다.
    """
    candidate = f"{CLIENT_ORDER_NAMESPACE}{run_id}-{symbol_tag(symbol)}-{leg_index}-{generation}-{submit_seq}"
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
    for symbol in sorted(set(targets) | set(current)):
        target = targets.get(symbol, _ZERO)
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

        for leg_index, (component_qty, reduce_only) in enumerate(components):
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
                    leg_index=leg_index,
                    decision_price=mark,
                )
            )
    return intents
