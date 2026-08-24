"""Peg-and-chase 스마트 지정가 집행기.

I-NO-NAKED-MARKET: MARKET 주문은 절대 생성하지 않는다. 공격적 집행조차
LIMIT + IOC + 가격 상한으로 표현하며 최악 슬리피지를 계약으로 묶는다.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any

from src.live.audit import AuditLog
from src.live.errors import VenueError
from src.live.filters import _ZERO, SymbolFilters
from src.live.planner import OrderIntent, build_client_order_id

_BPS_DENOMINATOR = Decimal(10_000)

#: 부모 intent의 최대 슬라이스 노셔널(등록 상수).
MAX_SLICE_NOTIONAL = Decimal("500")

#: 멈춘 시계 등 비정상 상태에서의 무한 루프 방지 가드.
_MAX_LOOP_ITERATIONS = 10_000


@dataclass(frozen=True, slots=True)
class PassiveExecutionPolicy:
    """집행 파라미터. 전부 등록 상수이며 실행 중 변경되지 않는다."""

    reprice_interval_s: float = 20.0
    chase_ticks: int = 2
    max_chases: int = 8
    passive_deadline_s: float = 2700.0
    window_deadline_s: float = 7200.0
    taker_cap_bps: float = 15.0
    participation_cap: float = 0.005
    max_slices: int = 4


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """단일 intent의 집행 결과."""

    symbol: str
    filled_qty: Decimal
    unfilled_qty: Decimal
    avg_fill_price: Decimal | None
    chases: int
    status: str


def _touches(client: Any, symbol: str) -> tuple[Decimal, Decimal]:
    payload = client.book_ticker(symbol)
    return Decimal(str(payload["bidPrice"])), Decimal(str(payload["askPrice"]))


def _capped_ioc_price(
    opposite_touch: Decimal, *, is_buy: bool, taker_cap_bps: float, tick_size: Decimal
) -> Decimal:
    factor = (
        Decimal(1) + Decimal(str(taker_cap_bps)) / _BPS_DENOMINATOR
        if is_buy
        else Decimal(1) - Decimal(str(taker_cap_bps)) / _BPS_DENOMINATOR
    )
    raw = opposite_touch * factor
    return raw.quantize(tick_size, rounding=ROUND_DOWN if is_buy else ROUND_UP)


def _order_params(
    intent: OrderIntent,
    quantity: Decimal,
    price: Decimal,
    *,
    time_in_force: str,
    client_order_id: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "symbol": intent.symbol,
        "side": intent.side,
        "type": "LIMIT",
        "timeInForce": time_in_force,
        "quantity": format(quantity, "f"),
        "price": format(price, "f"),
        "newClientOrderId": client_order_id,
    }
    if intent.reduce_only:
        params["reduceOnly"] = "true"
    return params


def _cancel_tolerating_benign(client: Any, symbol: str, client_order_id: str) -> None:
    """취소 거절(-2011: 이미 체결/취소)은 benign이므로 무시한다."""
    try:
        client.cancel_order(symbol, client_order_id)
    except VenueError as exc:
        if exc.code != -2011:
            raise


def execute_intent(
    client: Any,
    intent: OrderIntent,
    filters: SymbolFilters,
    policy: PassiveExecutionPolicy,
    audit: AuditLog,
    clock: Callable[[], float],
) -> ExecutionOutcome:
    """PASSIVE(GTX 체이스) -> DEADLINE(IOC 백스톱) -> RESIDUAL 순으로 집행한다."""
    bid, ask = _touches(client, intent.symbol)
    reference_price = bid if intent.side == "BUY" else ask
    raw_slices = float((intent.quantity * reference_price) / MAX_SLICE_NOTIONAL)
    slice_count = min(max(1, math.ceil(raw_slices)), policy.max_slices)
    slice_qty = (intent.quantity / slice_count).quantize(filters.step_size, rounding=ROUND_DOWN)

    is_buy = intent.side == "BUY"
    filled_total = _ZERO
    fill_notional = _ZERO
    total_chases = 0
    window_start = clock()

    for slice_idx in range(slice_count):
        remaining = slice_qty
        ioc_phase = False
        chases_this_slice = 0
        attempt = 0
        active_id: str | None = None
        active_price = _ZERO
        reported_executed = _ZERO
        posted_at = window_start
        iterations = 0

        while remaining > _ZERO:
            iterations += 1
            now = clock()
            if iterations > _MAX_LOOP_ITERATIONS or now - window_start >= policy.window_deadline_s:
                break

            best_bid, best_ask = _touches(client, intent.symbol)
            own_touch = best_bid if is_buy else best_ask
            opposite_touch = best_ask if is_buy else best_bid

            if active_id is not None:
                executed = Decimal(str(client.query_order(intent.symbol, active_id).get("executedQty", "0")))
                delta_fill = executed - reported_executed
                if delta_fill > _ZERO:
                    filled_total += delta_fill
                    fill_notional += delta_fill * active_price
                    remaining -= delta_fill
                    reported_executed = executed
                if remaining <= _ZERO:
                    break
                if not ioc_phase:
                    if now - posted_at >= policy.passive_deadline_s or chases_this_slice >= policy.max_chases:
                        _cancel_tolerating_benign(client, intent.symbol, active_id)
                        active_id = None
                        ioc_phase = True
                    elif (
                        now - posted_at >= policy.reprice_interval_s
                        or abs(own_touch - active_price) >= filters.tick_size * policy.chase_ticks
                    ):
                        _cancel_tolerating_benign(client, intent.symbol, active_id)
                        active_id = None
                        chases_this_slice += 1
                        total_chases += 1
                    else:
                        continue

            if active_id is None:
                if ioc_phase:
                    price = _capped_ioc_price(
                        opposite_touch,
                        is_buy=is_buy,
                        taker_cap_bps=policy.taker_cap_bps,
                        tick_size=filters.tick_size,
                    )
                    time_in_force = "IOC"
                else:
                    price = own_touch.quantize(
                        filters.tick_size, rounding=ROUND_DOWN if is_buy else ROUND_UP
                    )
                    time_in_force = "GTX"
                order_id = build_client_order_id(
                    intent.client_order_prefix, intent.symbol, slice_idx, attempt
                )
                attempt += 1
                params = _order_params(
                    intent, remaining, price, time_in_force=time_in_force, client_order_id=order_id
                )
                try:
                    client.new_order(params)
                except VenueError as exc:
                    # -5022(GTX 거절)는 오류가 아니라 호가 이동 신호다 -> 재호가.
                    if exc.code == -5022 and not ioc_phase:
                        chases_this_slice += 1
                        total_chases += 1
                        continue
                    raise
                active_id = order_id
                active_price = price
                reported_executed = _ZERO
                posted_at = clock()
                audit.record(
                    "order_posted",
                    symbol=intent.symbol,
                    client_order_id=order_id,
                    time_in_force=time_in_force,
                    price=str(price),
                    quantity=str(remaining),
                )

        if remaining > _ZERO and active_id is not None:
            _cancel_tolerating_benign(client, intent.symbol, active_id)
            executed = Decimal(str(client.query_order(intent.symbol, active_id).get("executedQty", "0")))
            delta_fill = executed - reported_executed
            if delta_fill > _ZERO:
                filled_total += delta_fill
                fill_notional += delta_fill * active_price
                remaining -= delta_fill
        if remaining > _ZERO:
            audit.record("order_residual", symbol=intent.symbol, quantity=str(remaining))

    unfilled = max(intent.quantity - filled_total, _ZERO)
    outcome = ExecutionOutcome(
        symbol=intent.symbol,
        filled_qty=filled_total,
        unfilled_qty=unfilled,
        avg_fill_price=(fill_notional / filled_total) if filled_total > _ZERO else None,
        chases=total_chases,
        status="FILLED" if unfilled <= _ZERO else "RESIDUAL",
    )
    audit.record(
        "intent_outcome",
        symbol=outcome.symbol,
        status=outcome.status,
        filled_qty=str(outcome.filled_qty),
        unfilled_qty=str(outcome.unfilled_qty),
        chases=outcome.chases,
    )
    return outcome
