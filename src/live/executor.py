"""Peg-and-chase 스마트 지정가 집행기(단일 협조 루프).

I-NO-NAKED-MARKET: MARKET 주문은 절대 생성하지 않는다. 공격적 집행조차
LIMIT + IOC + 가격 상한으로 표현하며 최악 슬리피지를 계약으로 묶는다.
I-CHASE-BAND: 모든 가격은 decision_price 밴드 안에 있어야 하며, 밴드 이탈 시
재호가하지 않고 대기한다(백테스트의 decision_price 고정 모델과 비용 구조 일치).
I-POLL-BOUNDED: 매 tick 은 반드시 sleep 으로 끝나며 루프 상한은
ceil(window_deadline_s / poll_interval_s) + 1 로 유도된다.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any

from src.live.audit import AuditLog
from src.live.errors import LiveTradingError, OrderObsolete, VenueError
from src.live.filters import _ZERO, SymbolFilters, quantize_to_multiple
from src.live.planner import OrderIntent, build_client_order_id
from src.live.rest import RateLimits, ShadowResponse

_BPS_DENOMINATOR = Decimal(10_000)

#: 부모 intent의 최대 슬라이스 노셔널(등록 상수).
MAX_SLICE_NOTIONAL = Decimal("500")


@dataclass(frozen=True, slots=True)
class PassiveExecutionPolicy:
    """집행 파라미터. 시간/비용 기본값은 ExecutionSpec 계약에서 유도된다.

    passive_deadline_s/window_deadline_s = passive_timeout_minutes(30)*60,
    taker_cap_bps = taker_fee_bps(5) + taker_slippage_bps(3).
    """

    poll_interval_s: float = 3.0
    chase_ticks: int = 2
    max_chases: int = 8
    passive_deadline_s: float = 30 * 60.0
    window_deadline_s: float = 30 * 60.0
    taker_cap_bps: float = 5.0 + 3.0
    chase_band_bps: float = 10.0
    participation_cap: float = 0.005
    max_slices: int = 4
    rate_weight_budget_fraction: float = 0.5


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """단일 intent의 집행 결과.

    status 는 'FILLED' | 'RESIDUAL' | 'SHADOW' | 'OBSOLETE' 폐쇄집합이다.
    SHADOW 는 억제된 주문, OBSOLETE 는 -2022 로 무의미해진 reduceOnly intent 다.
    """

    symbol: str
    filled_qty: Decimal
    unfilled_qty: Decimal
    avg_fill_price: Decimal | None
    chases: int
    status: str


def _slice_quantities(total: Decimal, slice_count: int, step_size: Decimal) -> list[Decimal]:
    """I-SLICE-EXACT: 합이 total 과 정확히 일치하는 step 배수 슬라이스 리스트.

    앞 원소들이 스텝 잔량을 흡수하고, total 이 step 의 배수가 아니면 fail-closed 한다
    (상위 planner/sizing 이 이미 정렬된 수량만 만든다).
    """
    if slice_count < 1:
        raise ValueError("slice_count must be >= 1")
    if step_size <= _ZERO:
        raise ValueError("step_size must be positive")
    total_units = (total / step_size).to_integral_value(rounding=ROUND_DOWN)
    if total_units * step_size != total:
        raise ValueError(f"total {total} is not a multiple of step_size {step_size}")
    units = int(total_units)
    if units == 0:
        return []
    base = units // slice_count
    remainder = units % slice_count
    slices: list[Decimal] = []
    for index in range(min(slice_count, units)):
        count = base + (1 if index < remainder else 0)
        if count > 0:
            slices.append(step_size * count)
    return slices


def _post_quantity(
    remaining: Decimal,
    price: Decimal,
    *,
    filters: SymbolFilters,
    max_slices: int,
    max_slice_notional: Decimal = MAX_SLICE_NOTIONAL,
) -> Decimal:
    """단일 활성 주문이 실을 헤드 슬라이스. 슬라이싱 비활성 시 remaining 전체와 동일.

    I-SLICE-EXACT 분할의 순차 소비형: 각 재게시 시점에 남은 수량을 다시 정확히
    분할하므로 합 보존이 유지된다.
    """
    if price <= _ZERO:
        return remaining
    slice_count = min(
        max_slices,
        max(1, math.ceil(float((remaining * price) / max_slice_notional))),
    )
    if slice_count <= 1:
        return remaining
    slices = _slice_quantities(remaining, slice_count, filters.step_size)
    return slices[0] if slices else remaining


def _touches_from_payload(payload: Any) -> tuple[Decimal, Decimal]:
    return Decimal(str(payload["bidPrice"])), Decimal(str(payload["askPrice"]))


def _fetch_books(client: Any, symbols: Iterable[str]) -> dict[str, tuple[Decimal, Decimal]]:
    """전 심볼 호가를 배치 1회(weight 5)로 수집하고, 구형 클라이언트는 N+1 로 폴백한다."""
    batch_getter = getattr(client, "book_tickers", None)
    books: dict[str, tuple[Decimal, Decimal]] = {}
    if callable(batch_getter):
        payload_map: Mapping[str, Any] = batch_getter()
        for symbol in symbols:
            payload = payload_map.get(symbol)
            if payload is not None:
                books[symbol] = _touches_from_payload(payload)
        return books
    for symbol in symbols:
        books[symbol] = _touches_from_payload(client.book_ticker(symbol))
    return books


def _capped_ioc_price(
    opposite_touch: Decimal,
    *,
    is_buy: bool,
    taker_cap_bps: float,
    tick_size: Decimal,
    band_low: Decimal | None = None,
    band_high: Decimal | None = None,
) -> Decimal | None:
    """IOC 백스톱 가격: opposite*(1±taker_cap) 틱 양자화 후 I-CHASE-BAND 로 클램프한다."""
    factor = (
        Decimal(1) + Decimal(str(taker_cap_bps)) / _BPS_DENOMINATOR
        if is_buy
        else Decimal(1) - Decimal(str(taker_cap_bps)) / _BPS_DENOMINATOR
    )
    raw = opposite_touch * factor
    price = quantize_to_multiple(raw, tick_size, ROUND_DOWN if is_buy else ROUND_UP)
    if band_low is not None and band_high is not None:
        if is_buy:
            high = quantize_to_multiple(band_high, tick_size, ROUND_DOWN)
            price = min(price, high)
            floor = quantize_to_multiple(band_low, tick_size, ROUND_DOWN)
            if price < floor:
                return None
        else:
            low = quantize_to_multiple(band_low, tick_size, ROUND_UP)
            price = max(price, low)
            ceiling = quantize_to_multiple(band_high, tick_size, ROUND_UP)
            if price > ceiling:
                return None
    if price <= _ZERO:
        return None
    return price


def _band(intent: OrderIntent, policy: PassiveExecutionPolicy) -> tuple[Decimal, Decimal]:
    """I-CHASE-BAND 앵커: decision_price ± chase_band_bps."""
    half_band = intent.decision_price * Decimal(str(policy.chase_band_bps)) / _BPS_DENOMINATOR
    return intent.decision_price - half_band, intent.decision_price + half_band


def _gtx_candidate(
    raw: Decimal,
    *,
    is_buy: bool,
    filters: SymbolFilters,
    band_low: Decimal,
    band_high: Decimal,
) -> Decimal | None:
    """GTX 게시 가격: 패시브측 틱 양자화 후 밴드 밖이면 None(HOLD)."""
    price = quantize_to_multiple(raw, filters.tick_size, ROUND_DOWN if is_buy else ROUND_UP)
    if price <= _ZERO or not (band_low <= price <= band_high):
        return None
    return price


def _cancel_tolerating_benign(client: Any, symbol: str, client_order_id: str) -> None:
    """취소 거절(-2011: 이미 체결/취소)은 benign이므로 무시한다."""
    try:
        client.cancel_order(symbol, client_order_id)
    except VenueError as exc:
        if exc.code != -2011:
            raise


def _record_fill(rt: _IntentRuntime, executed: Decimal) -> None:
    delta_fill = executed - rt.reported_executed
    if delta_fill > _ZERO:
        rt.filled_total += delta_fill
        rt.fill_notional += delta_fill * rt.active_price
        rt.reported_executed = executed


@dataclass(slots=True)
class _IntentRuntime:
    """협조 루프가 유지하는 단일 intent 의 진행 상태."""

    intent: OrderIntent
    filters: SymbolFilters | None
    phase: str = "passive"  # 'passive' | 'ioc'
    active_id: str | None = None
    active_price: Decimal = _ZERO
    reported_executed: Decimal = _ZERO
    filled_total: Decimal = _ZERO
    fill_notional: Decimal = _ZERO
    chases: int = 0
    attempts: int = 0
    posted_at: float = 0.0
    terminal_status: str | None = None

    @property
    def done(self) -> bool:
        return self.terminal_status is not None

    def snapshot(self) -> ExecutionOutcome:
        if self.terminal_status is not None:
            status = self.terminal_status
        elif self.intent.quantity > _ZERO and self.filled_total >= self.intent.quantity:
            status = "FILLED"
        else:
            status = "RESIDUAL"
        unfilled = max(self.intent.quantity - self.filled_total, _ZERO)
        return ExecutionOutcome(
            symbol=self.intent.symbol,
            filled_qty=self.filled_total,
            unfilled_qty=unfilled,
            avg_fill_price=(
                self.fill_notional / self.filled_total if self.filled_total > _ZERO else None
            ),
            chases=self.chases,
            status=status,
        )


def cancel_orphan_orders(client: Any, client_order_prefix: str, audit: AuditLog) -> int:
    """prefix 일치 고아 주문을 재조정 이전에 전량 취소하고 건수를 반환한다(-2011 benign)."""
    open_orders = client.open_orders()
    cancelled = 0
    for entry in open_orders:
        order_id = str(entry.get("clientOrderId", ""))
        if not order_id.startswith(client_order_prefix):
            continue
        _cancel_tolerating_benign(client, str(entry["symbol"]), order_id)
        audit.record("orphan_cancelled", symbol=entry.get("symbol"), client_order_id=order_id)
        cancelled += 1
    return cancelled


def execute_intents(
    client: Any,
    intents: Sequence[OrderIntent],
    filters: Mapping[str, SymbolFilters],
    policy: PassiveExecutionPolicy,
    audit: AuditLog,
    clock: Callable[[], float],
    sleep_fn: Callable[[float], None],
    *,
    rate_limits: RateLimits | None = None,
) -> tuple[ExecutionOutcome, ...]:
    """단일 협조 루프(post-all/poll-all/예산 스로틀). 반환 순서는 intents 와 1:1.

    어떤 예외 경로에서도 이미 확인된 체결을 잃지 않도록 진행 중 부분 결과를
    발생 예외에 partial_outcomes 속성으로 붙여 재전파한다(I-LEDGER-DURABLE 채널).
    """
    runtimes = [
        _IntentRuntime(intent=intent, filters=filters.get(intent.symbol))
        for intent in intents
    ]
    for rt in runtimes:
        if rt.filters is None:
            # 필터 부재 심볼은 게시 자체가 불가능하므로 즉시 잔량 확정한다.
            rt.terminal_status = "RESIDUAL"
    try:
        _run_loop(client, runtimes, policy, audit, clock, sleep_fn, rate_limits)
    except LiveTradingError as exc:
        exc.partial_outcomes = tuple(rt.snapshot() for rt in runtimes)
        raise
    for rt in runtimes:
        audit.record(
            "intent_outcome",
            symbol=rt.intent.symbol,
            status=rt.snapshot().status,
            filled_qty=str(rt.filled_total),
            unfilled_qty=str(max(rt.intent.quantity - rt.filled_total, _ZERO)),
            chases=rt.chases,
        )
    return tuple(rt.snapshot() for rt in runtimes)


def _run_loop(
    client: Any,
    runtimes: Sequence[_IntentRuntime],
    policy: PassiveExecutionPolicy,
    audit: AuditLog,
    clock: Callable[[], float],
    sleep_fn: Callable[[float], None],
    rate_limits: RateLimits | None,
) -> None:
    live = [rt for rt in runtimes if not rt.done]
    if not live:
        return
    symbols = sorted({rt.intent.symbol for rt in live})
    start = clock()
    interval = policy.poll_interval_s
    max_ticks = math.ceil(policy.window_deadline_s / policy.poll_interval_s) + 1
    ticks = 0

    while ticks < max_ticks and any(not rt.done for rt in live):
        now = clock()
        if now - start >= policy.window_deadline_s:
            break
        books = _fetch_books(client, symbols)
        for rt in live:
            if not rt.done and rt.intent.symbol in books:
                _poll_or_post(client, rt, books[rt.intent.symbol], now, policy, audit)
        ticks += 1
        if all(rt.done for rt in live):
            break
        interval = _throttled_interval(client, interval, policy, rate_limits)
        # R1: 모든 tick 은 sleep 으로 끝난다(busy-wait 금지).
        sleep_fn(interval)

    _finalize(client, live, audit)


def _throttled_interval(
    client: Any, interval: float, policy: PassiveExecutionPolicy, rate_limits: RateLimits | None
) -> float:
    """R2: 사용 weight 가 예산을 초과하면 poll 간격을 배증한다(상한 window_deadline_s/4)."""
    if rate_limits is None or rate_limits.request_weight_1m <= 0:
        return interval
    rate_state = getattr(client, "rate_state", None)
    used = getattr(rate_state, "used_weight_1m", None)
    budget = policy.rate_weight_budget_fraction * rate_limits.request_weight_1m
    if used is not None and used > budget:
        return min(interval * 2, policy.window_deadline_s / 4)
    return interval


def _poll_or_post(
    client: Any,
    rt: _IntentRuntime,
    touch: tuple[Decimal, Decimal],
    now: float,
    policy: PassiveExecutionPolicy,
    audit: AuditLog,
) -> None:
    assert rt.filters is not None  # filters 부재 intent 는 생성 시 즉시 RESIDUAL 처리된다
    is_buy = rt.intent.side == "BUY"
    bid, ask = touch
    own_touch = bid if is_buy else ask
    opposite_touch = ask if is_buy else bid
    band_low, band_high = _band(rt.intent, policy)

    # 1) 활성 주문 조회: 체결 누적 및 상태 전이(FILL/CHASE/HOLD/IOC).
    if rt.active_id is not None:
        executed = Decimal(
            str(client.query_order(rt.intent.symbol, rt.active_id).get("executedQty", "0"))
        )
        _record_fill(rt, executed)
        if rt.intent.quantity - rt.filled_total <= _ZERO:
            rt.terminal_status = "FILLED"
            return
        if rt.phase == "passive":
            timed_out = now - rt.posted_at >= policy.passive_deadline_s
            exhausted = rt.chases >= policy.max_chases
            moved = abs(own_touch - rt.active_price) >= rt.filters.tick_size * policy.chase_ticks
            if timed_out or exhausted:
                _cancel_tolerating_benign(client, rt.intent.symbol, rt.active_id)
                rt.active_id = None
                rt.phase = "ioc"
            elif moved:
                _cancel_tolerating_benign(client, rt.intent.symbol, rt.active_id)
                rt.active_id = None
                rt.chases += 1
            else:
                return
        else:
            # IOC 미체결분은 즉시 소멸하므로 잔량을 새 IOC 로 재게시한다.
            _cancel_tolerating_benign(client, rt.intent.symbol, rt.active_id)
            rt.active_id = None

    # 2) 게시: 패시브(GTX, 밴드 내) 또는 백스톱(IOC, 캡+밴드 클램프).
    remaining = rt.intent.quantity - rt.filled_total
    if remaining <= _ZERO:
        rt.terminal_status = "FILLED"
        return
    if rt.phase == "passive":
        price = _gtx_candidate(
            own_touch, is_buy=is_buy, filters=rt.filters, band_low=band_low, band_high=band_high
        )
        time_in_force = "GTX"
    else:
        price = _capped_ioc_price(
            opposite_touch,
            is_buy=is_buy,
            taker_cap_bps=policy.taker_cap_bps,
            tick_size=rt.filters.tick_size,
            band_low=band_low,
            band_high=band_high,
        )
        time_in_force = "IOC"
    if price is None:
        return  # I-CHASE-BAND: 밴드 이탈 시 재호가하지 않고 대기한다.

    post_qty = _post_quantity(
        remaining, price, filters=rt.filters, max_slices=policy.max_slices
    )
    order_id = build_client_order_id(
        rt.intent.client_order_prefix, rt.intent.symbol, rt.intent.leg_index, 0, rt.attempts
    )
    rt.attempts += 1
    params: dict[str, Any] = {
        "symbol": rt.intent.symbol,
        "side": rt.intent.side,
        "type": "LIMIT",
        "timeInForce": time_in_force,
        "quantity": format(post_qty, "f"),
        "price": format(price, "f"),
        "newClientOrderId": order_id,
    }
    if rt.intent.reduce_only:
        params["reduceOnly"] = "true"
    try:
        response = client.new_order(params)
    except VenueError as exc:
        # -5022/-4131 은 오류가 아니라 호가 이동 신호다 -> 다음 tick 에 재호가.
        if exc.code in (-5022, -4131):
            rt.chases += 1
            return
        raise
    except OrderObsolete:
        rt.terminal_status = "OBSOLETE"
        audit.record("intent_obsolete", symbol=rt.intent.symbol, client_order_id=order_id)
        return
    if isinstance(response, ShadowResponse):
        # R8: SHADOW 억제 -> query_order 없이 즉시 종료한다.
        rt.terminal_status = "SHADOW"
        return
    rt.active_id = order_id
    rt.active_price = price
    rt.reported_executed = _ZERO
    rt.posted_at = now
    audit.record(
        "order_posted",
        symbol=rt.intent.symbol,
        client_order_id=order_id,
        time_in_force=time_in_force,
        price=str(price),
        quantity=str(post_qty),
    )


def _finalize(client: Any, runtimes: Sequence[_IntentRuntime], audit: AuditLog) -> None:
    """윈도우 종료 시 활성 주문을 정리하고 최종 체결을 확정한다."""
    for rt in runtimes:
        if rt.done:
            continue
        if rt.active_id is not None:
            _cancel_tolerating_benign(client, rt.intent.symbol, rt.active_id)
            executed = Decimal(
                str(client.query_order(rt.intent.symbol, rt.active_id).get("executedQty", "0"))
            )
            _record_fill(rt, executed)
            rt.active_id = None
        if rt.intent.quantity - rt.filled_total > _ZERO:
            rt.terminal_status = "RESIDUAL"
            audit.record(
                "order_residual",
                symbol=rt.intent.symbol,
                quantity=str(rt.intent.quantity - rt.filled_total),
            )
        else:
            rt.terminal_status = "FILLED"


def execute_intent(
    client: Any,
    intent: OrderIntent,
    filters: SymbolFilters,
    policy: PassiveExecutionPolicy,
    audit: AuditLog,
    clock: Callable[[], float],
) -> ExecutionOutcome:
    """execute_intents 의 단일 원소 위임 래퍼(기존 테스트 호환 유지)."""
    outcomes = execute_intents(
        client,
        [intent],
        {intent.symbol: filters},
        policy,
        audit,
        clock,
        lambda _seconds: None,
    )
    return outcomes[0]
