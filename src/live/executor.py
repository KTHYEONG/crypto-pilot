"""Peg-and-chase 스마트 지정가 집행기(단일 협조 루프).

I-NO-NAKED-MARKET: MARKET 주문은 절대 생성하지 않는다. 공격적 집행조차
LIMIT + IOC + 가격 상한으로 표현하며 최악 슬리피지를 계약으로 묶는다.
I-CHASE-BAND: GTX 게시 가격은 decision_price chase 밴드 안에 있어야 하며,
밴드 이탈 시 재호가하지 않고 대기한다. IOC 백스톱은 별도의 max_cross_bps
리스크 레일을 따른다: 레일 안이면 반드시 마케터블 가격으로 크로싱하고,
레일 밖 이상 징후일 때만 대기한다.
I-POLL-BOUNDED: 매 tick 은 반드시 sleep 으로 끝나며 루프 상한은
ceil(window_deadline_s / poll_interval_s) + 1 로 유도된다.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from src.common.errors import DataIntegrityError
from src.core.types import ExecutionSpec
from src.live.audit import AuditLog
from src.live.errors import (
    ErrorAction,
    LiveTradingError,
    OrderObsolete,
    TransientReadError,
    VenueError,
    resolve_error_action,
)
from src.live.filters import _ZERO, SymbolFilters, quantize_to_multiple
from src.live.microstructure import parse_book_quote
from src.live.order_cancel import _LEGACY_CLIENT_ORDER_ID as _LEGACY_CLIENT_ORDER_ID
from src.live.order_cancel import _OPEN_ORDER_STATUSES as _OPEN_ORDER_STATUSES
from src.live.order_cancel import _ORDER_GONE_CODES as _ORDER_GONE_CODES
from src.live.order_cancel import ORDER_DOES_NOT_EXIST_CODE as ORDER_DOES_NOT_EXIST_CODE
from src.live.order_cancel import ORDER_NOT_FOUND_STATUS as ORDER_NOT_FOUND_STATUS
from src.live.order_cancel import CancelConfirmation as CancelConfirmation
from src.live.order_cancel import CancelNotConfirmed as CancelNotConfirmed
from src.live.order_cancel import OrphanSweep as OrphanSweep
from src.live.order_cancel import UnresolvedOrder as UnresolvedOrder
from src.live.order_cancel import cancel_and_confirm as cancel_and_confirm
from src.live.order_cancel import cancel_orphan_orders as cancel_orphan_orders
from src.live.order_journal import OrderJournal
from src.live.planner import OrderIntent, build_client_order_id
from src.live.rest import (
    _HTTP_TIMEOUT_SECONDS,
    OrderStatusUnknown,
    PaperResponse,
    RateLimits,
    ShadowResponse,
)

if TYPE_CHECKING:
    from src.live.order_journal import JournalAttempt, JournalFill

_BPS_DENOMINATOR = Decimal(10_000)

#: 부모 intent의 최대 슬라이스 노셔널(등록 상수).
MAX_SLICE_NOTIONAL = Decimal("500")

#: 백테스트 3m 리플레이 바 하나에 대응하는 패시브 집행 상한(초).
EXECUTION_BAR_SECONDS: float = 180.0

#: 미확인 제출이 실제로 미체결로 확정되기까지 필요한 연속 -2013 조회 횟수.
UNKNOWN_SUBMISSION_MISS_LIMIT: int = 2


#: Closed outcome set for one intent.
OUTCOME_STATUSES: frozenset[str] = frozenset(
    {"FILLED", "RESIDUAL", "RESIDUAL_SUB_MINIMUM", "REJECTED", "SHADOW", "OBSOLETE"}
)


def _default_unknown_outcome_horizon_s() -> float:
    # 모듈 import 시 환경 변수를 읽지 않도록 인스턴스가 아닌 필드 기본값을 쓴다.
    from src.live.settings import LiveSettings

    recv_window_ms = LiveSettings.model_fields["recv_window_ms"].default
    return float(recv_window_ms) / 1000 + _HTTP_TIMEOUT_SECONDS


DEFAULT_UNKNOWN_OUTCOME_HORIZON_S: float = _default_unknown_outcome_horizon_s()


class ForeignOpenOrderError(LiveTradingError):  # noqa: N818 - contract pins the name
    """변이 모드에서 외부(수동) 미체결 주문이 존재해 정리를 거부한다."""


class ExecutionInterrupted(LiveTradingError):  # noqa: N818 - contract pins the name
    """A shutdown request stopped the execution window before its natural end.

    Raised after resting orders were cancelled and settled (within the cleanup budget) so the
    caller can persist the partial outcome without marking the decision executed.
    """


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    maker_fee_bps: float
    taker_fee_bps: float

    def bps_for(self, liquidity: str) -> float:
        if liquidity == "maker":
            return self.maker_fee_bps
        if liquidity == "taker":
            return self.taker_fee_bps
        raise ValueError(f"unknown liquidity {liquidity!r}")


@dataclass(frozen=True, slots=True)
class PassiveExecutionPolicy:
    """집행 파라미터. 시간/비용 기본값은 ExecutionSpec 계약에서 유도된다.

    window_deadline_s = passive_timeout_minutes(30)*60,
    passive_deadline_s 는 창의 60% 로 패시브 단계 상한을 나누며
    window_deadline_s 보다 엄격히 작아야 한다(fail-closed 검증),
    taker_cap_bps = taker_fee_bps(5) + taker_slippage_bps(3).

    chase_band_bps(GTX 알파 레일)와 max_cross_bps(IOC 리스크 레일)는 분리된
    한계다: 전자는 GTX peg 가 얼마나 쫓아가는가(수익 기회 한계), 후자는
    백스톱 크로싱이 포기하는 이상 징후 경계(손실 한계)다.

    ``passive_pricing="anchored"`` mirrors the canonical strict passive fill model: the GTX
    limit rests at the intent's decision price (clamped one tick inside the opposite touch
    so it stays post-only) for the whole passive phase and never chases the book; only an
    unfilled remainder at the passive deadline escalates to the capped IOC backstop.
    """

    poll_interval_s: float = 3.0
    chase_ticks: int = 2
    max_chases: int = 8
    passive_deadline_s: float = 0.6 * 30 * 60.0
    window_deadline_s: float = 30 * 60.0
    taker_cap_bps: float = 5.0 + 3.0
    chase_band_bps: float = 10.0
    max_cross_bps: float = 50.0
    max_slices: int = 4
    rate_weight_budget_fraction: float = 0.5
    max_ioc_attempts: int = 10
    max_consecutive_read_failures: int = 5
    rate_weight_recover_fraction: float = 0.35
    margin_retry_s: float = 60.0
    max_margin_rejects: int = 3
    fee_schedule: FeeSchedule = FeeSchedule(maker_fee_bps=ExecutionSpec().maker_fee_bps, taker_fee_bps=ExecutionSpec().taker_fee_bps)
    taker_slippage_bps: float = ExecutionSpec().taker_slippage_bps
    passive_pricing: Literal["touch_chase", "anchored", "anchored_repeg"] = "touch_chase"
    repeg_interval_s: float = EXECUTION_BAR_SECONDS

    def __post_init__(self) -> None:
        if self.passive_pricing not in ("touch_chase", "anchored", "anchored_repeg"):
            raise ValueError(f"passive_pricing must be 'touch_chase', 'anchored' or 'anchored_repeg', got {self.passive_pricing!r}")
        if self.passive_deadline_s >= self.window_deadline_s:
            raise ValueError(
                f"passive_deadline_s ({self.passive_deadline_s}) must be strictly less than "
                f"window_deadline_s ({self.window_deadline_s})"
            )
        if self.passive_pricing == "anchored_repeg" and not (
            0 < self.repeg_interval_s < self.passive_deadline_s
        ):
            raise ValueError(
                f"repeg_interval_s ({self.repeg_interval_s}) must be strictly less than "
                f"passive_deadline_s ({self.passive_deadline_s})"
            )
        if self.poll_interval_s <= 0:
            raise ValueError(f"poll_interval_s must be > 0, got {self.poll_interval_s}")
        if self.max_ioc_attempts < 1:
            raise ValueError(f"max_ioc_attempts must be >= 1, got {self.max_ioc_attempts}")
        if self.max_cross_bps <= self.chase_band_bps:
            raise ValueError(
                f"max_cross_bps ({self.max_cross_bps}) must strictly exceed "
                f"chase_band_bps ({self.chase_band_bps})"
            )
        if self.max_consecutive_read_failures < 1:
            raise ValueError(
                f"max_consecutive_read_failures must be >= 1, got {self.max_consecutive_read_failures}"
            )
        if not 0 < self.rate_weight_recover_fraction < self.rate_weight_budget_fraction:
            raise ValueError(
                "rate_weight_recover_fraction must satisfy "
                f"0 < recover < budget, got {self.rate_weight_recover_fraction} >= {self.rate_weight_budget_fraction}"
            )
        if self.margin_retry_s <= 0:
            raise ValueError(f"margin_retry_s must be > 0, got {self.margin_retry_s}")
        if self.max_margin_rejects < 1:
            raise ValueError(f"max_margin_rejects must be >= 1, got {self.max_margin_rejects}")


def backtest_parity_execution_policy(
    fee_schedule: FeeSchedule, taker_slippage_bps: float
) -> PassiveExecutionPolicy:
    """백테스트 즉시-테이커 타이밍에 수렴하는 라이브 집행 정책.

    패시브 단계는 3m 리플레이 바 하나(EXECUTION_BAR_SECONDS)로, 전체 윈도우는
    두 바로 묶으며, IOC 백스톱 캡은 taker 수수료 + 테이커 슬리피지로 둔다.
    나머지 기본값은 그대로 둔다.
    """
    return PassiveExecutionPolicy(
        passive_deadline_s=EXECUTION_BAR_SECONDS,
        window_deadline_s=2 * EXECUTION_BAR_SECONDS,
        taker_cap_bps=fee_schedule.taker_fee_bps + taker_slippage_bps,
        fee_schedule=fee_schedule,
        taker_slippage_bps=taker_slippage_bps,
    )


def strict_passive_execution_policy(
    fee_schedule: FeeSchedule, taker_slippage_bps: float, passive_timeout_minutes: int,
) -> PassiveExecutionPolicy:
    """Live policy matching the canonical strict passive (maker) backtest.

    The anchored GTX limit rests for ``passive_timeout_minutes`` (the same timeout the
    OHLCV strict proxy uses), then any remainder crosses through the capped IOC backstop
    within two replay bars; the IOC cap is taker fee + taker slippage, as in the taker
    parity policy.

    Raises:
        ValueError: passive_timeout_minutes < 1.
    """
    if passive_timeout_minutes < 1:
        raise ValueError(f"passive_timeout_minutes must be >= 1, got {passive_timeout_minutes}")
    passive_deadline_s = float(passive_timeout_minutes) * 60.0
    return PassiveExecutionPolicy(
        passive_deadline_s=passive_deadline_s,
        window_deadline_s=passive_deadline_s + 2 * EXECUTION_BAR_SECONDS,
        taker_cap_bps=fee_schedule.taker_fee_bps + taker_slippage_bps,
        fee_schedule=fee_schedule,
        taker_slippage_bps=taker_slippage_bps,
        passive_pricing="anchored",
    )


def strict_passive_repeg_execution_policy(
    fee_schedule: FeeSchedule,
    taker_slippage_bps: float,
    passive_timeout_minutes: int,
    repeg_interval_s: float = EXECUTION_BAR_SECONDS,
) -> PassiveExecutionPolicy:
    """Live policy matching the canonical strict passive (maker) backtest plus re-pegging.

    The anchored GTX limit rests for ``passive_timeout_minutes`` (the same timeout the
    OHLCV strict proxy uses), re-pegging to the live touch every ``repeg_interval_s``
    during the passive phase (band-capped on the adverse side only); then any remainder
    crosses through the capped IOC backstop within two replay bars; the IOC cap is taker
    fee + taker slippage, as in the taker parity policy.

    Raises:
        ValueError: passive_timeout_minutes < 1.
    """
    if passive_timeout_minutes < 1:
        raise ValueError(f"passive_timeout_minutes must be >= 1, got {passive_timeout_minutes}")
    passive_deadline_s = float(passive_timeout_minutes) * 60.0
    return PassiveExecutionPolicy(
        passive_deadline_s=passive_deadline_s,
        window_deadline_s=passive_deadline_s + 2 * EXECUTION_BAR_SECONDS,
        taker_cap_bps=fee_schedule.taker_fee_bps + taker_slippage_bps,
        fee_schedule=fee_schedule,
        taker_slippage_bps=taker_slippage_bps,
        passive_pricing="anchored_repeg",
        repeg_interval_s=repeg_interval_s,
    )


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """Outcome of one intent.

    ``status`` is one of ``OUTCOME_STATUSES``: FILLED, RESIDUAL (window ended with an unfilled
    remainder), RESIDUAL_SUB_MINIMUM (remainder below the venue's minQty/minNotional and not
    reduce-only, so it can never be posted), REJECTED (a symbol-scoped venue rejection ended
    only this intent; ``reject_code`` holds the Binance code), SHADOW, OBSOLETE.
    """

    symbol: str
    filled_qty: Decimal
    unfilled_qty: Decimal
    avg_fill_price: Decimal | None
    chases: int
    status: str
    latency_seconds: float | None = None
    fills: tuple[tuple[Decimal, Decimal, float, str, str, pd.Timestamp], ...] = ()
    maker_qty: Decimal = _ZERO
    taker_qty: Decimal = _ZERO
    reject_code: int | None = None


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


def _note_invalid_quote(
    audit: AuditLog | None, invalid_reported: set[str] | None, symbol: str
) -> None:
    """Record one ``quote_invalid`` row per symbol (per cycle when a set is given)."""
    if audit is None:
        return
    if invalid_reported is not None:
        if symbol in invalid_reported:
            return
        invalid_reported.add(symbol)
    audit.record("quote_invalid", symbol=symbol)


def _fetch_books(
    client: Any,
    symbols: Iterable[str],
    *,
    audit: AuditLog | None = None,
    invalid_reported: set[str] | None = None,
) -> dict[str, tuple[Decimal, Decimal]]:
    """Fetch validated two-sided touches for ``symbols`` in one weight-5 batch (legacy clients: N+1).

    Symbols whose quote fails ``parse_book_quote`` are omitted; the tick loop then skips them,
    exactly as for a symbol the venue did not return.
    """
    wanted = list(symbols)
    batch_getter = getattr(client, "book_tickers", None)
    books: dict[str, tuple[Decimal, Decimal]] = {}
    if callable(batch_getter):
        payload_map: Mapping[str, Any] = batch_getter()
        for symbol in wanted:
            payload = payload_map.get(symbol)
            if payload is None:
                continue
            quote = parse_book_quote(symbol, payload)
            if quote is None:
                _note_invalid_quote(audit, invalid_reported, symbol)
                continue
            books[symbol] = (quote.bid, quote.ask)
        return books
    for symbol in wanted:
        quote = parse_book_quote(symbol, client.book_ticker(symbol))
        if quote is None:
            _note_invalid_quote(audit, invalid_reported, symbol)
            continue
        books[symbol] = (quote.bid, quote.ask)
    return books


@dataclass(slots=True)
class _CycleFlags:
    """Mutable per-cycle loop state (risk-increase freeze)."""

    risk_increase_frozen: bool = False
    freeze_code: int | None = None


def _capped_ioc_price(
    opposite_touch: Decimal,
    *,
    is_buy: bool,
    taker_cap_bps: float,
    tick_size: Decimal,
    band_low: Decimal | None = None,
    band_high: Decimal | None = None,
) -> Decimal | None:
    """IOC 백스톱 가격: opposite*(1±taker_cap) 틱 양자화 후 리스크 레일 클램프.

    ``band_low``/``band_high`` 는 chase 밴드가 아니라 decision_price ±
    max_cross_bps 리스크 레일이다. 레일 안에서는 결과가 반드시 마케터블이다:
    클램프 후 비마케터블(매수는 opposite 미달, 매도는 초과)이면 opposite 터치
    가격으로 올려(내려) 반환한다. opposite 자체가 레일 밖 이상 징후일 때만
    None 을 반환해 체결 거부는 진짜 이상 징후 보호로 한정한다.
    """
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
            if price < opposite_touch:
                price = quantize_to_multiple(opposite_touch, tick_size, ROUND_UP)
        else:
            low = quantize_to_multiple(band_low, tick_size, ROUND_UP)
            price = max(price, low)
            if price > opposite_touch:
                price = quantize_to_multiple(opposite_touch, tick_size, ROUND_DOWN)
    if price <= _ZERO:
        return None
    if band_low is not None and band_high is not None:
        outside_rail = opposite_touch > band_high if is_buy else opposite_touch < band_low
        if outside_rail:
            return None
    return price


def _band(intent: OrderIntent, policy: PassiveExecutionPolicy) -> tuple[Decimal, Decimal]:
    """I-CHASE-BAND 앵커: decision_price ± chase_band_bps (GTX 알파 레일)."""
    half_band = intent.decision_price * Decimal(str(policy.chase_band_bps)) / _BPS_DENOMINATOR
    return intent.decision_price - half_band, intent.decision_price + half_band


def _risk_rail(bid: Decimal, ask: Decimal, policy: PassiveExecutionPolicy) -> tuple[Decimal, Decimal]:
    """IOC anomaly rail: the current book mid ± ``max_cross_bps``.

    Centred on the book observed at IOC time, not on the submit-time decision price, so a
    genuine trend during the passive window is still crossed (the replay ledger crosses at the
    window's last close); only a far touch implausibly far from its own mid -- a spread
    blow-out or broken quote -- is refused.
    """
    mid = (bid + ask) / Decimal(2)
    half_rail = mid * Decimal(str(policy.max_cross_bps)) / _BPS_DENOMINATOR
    return mid - half_rail, mid + half_rail


def _gtx_candidate(
    raw: Decimal,
    *,
    is_buy: bool,
    filters: SymbolFilters,
    band_low: Decimal,
    band_high: Decimal,
) -> Decimal | None:
    """GTX 게시 가격: 패시브측 틱 양자화 후 불리 방향 밴드 이탈만 None(HOLD).

    유리 방향 이탈(매수 중 하락, 매도 중 상승)은 그대로 게시해 체결 기회를
    유지한다. 대칭 밴드 차단은 유리한 체결까지 폐기하는 결함이었다.
    """
    price = quantize_to_multiple(raw, filters.tick_size, ROUND_DOWN if is_buy else ROUND_UP)
    if price <= _ZERO:
        return None
    if is_buy:
        if price > band_high:
            return None
    else:
        if price < band_low:
            return None
    return price


def _post_only_quote_price(
    anchor: Decimal,
    touch: tuple[Decimal, Decimal],
    *,
    is_buy: bool,
    filters: SymbolFilters,
) -> Decimal | None:
    """GTX 게시 가격: quote 수준 ``anchor`` 고정, opposite 터치 안쪽 한 틱으로만 양보한다.

    매수는 ``min(anchor, ask - tick)`` 을 tick 아래로, 매도는
    ``max(anchor, bid + tick)`` 을 tick 위로 양자화한다. 결과는 항상
    post-only 이며 anchor 보다 유리한 방향으로만 벗어난다. 0 이하면 None(HOLD).
    """
    bid, ask = touch
    tick = filters.tick_size
    if is_buy:
        raw = min(anchor, ask - tick)
        price = quantize_to_multiple(raw, tick, ROUND_DOWN)
    else:
        raw = max(anchor, bid + tick)
        price = quantize_to_multiple(raw, tick, ROUND_UP)
    if price <= _ZERO:
        return None
    return price


def _band_capped_peg_price(
    own_touch: Decimal,
    *,
    is_buy: bool,
    filters: SymbolFilters,
    band_low: Decimal,
    band_high: Decimal,
) -> Decimal:
    """Bar-cadence repeg price for ``passive_pricing="anchored_repeg"``.

    Quantize ``own_touch`` to one post-only tick (ROUND_DOWN for a buy, ROUND_UP for a
    sell), then clamp into ``[band_low, band_high]`` on the adverse side only: a buy's
    result is capped at ``band_high``, a sell's result is floored at ``band_low``, while
    a favourable excursion posts unclamped. Unlike ``_gtx_candidate``, never returns
    ``None``: an adverse excursion beyond the band saturates at the band edge.
    """
    price = quantize_to_multiple(own_touch, filters.tick_size, ROUND_DOWN if is_buy else ROUND_UP)
    if is_buy:
        price = min(price, quantize_to_multiple(band_high, filters.tick_size, ROUND_DOWN))
    else:
        price = max(price, quantize_to_multiple(band_low, filters.tick_size, ROUND_UP))
    return price


def _journal_submit(
    journal: OrderJournal,
    *,
    client_order_id: str,
    symbol: str,
    submit_seq: int,
    attempt_seq: int,
    side: str,
    quantity: Decimal,
    reduce_only: bool,
    leg_index: int,
) -> None:
    """Record one submission (fsync) before the order is sent."""
    journal.record_submit(
        client_order_id,
        symbol,
        submit_seq,
        attempt_seq=attempt_seq,
        side=side,
        quantity=quantity,
        reduce_only=reduce_only,
        leg_index=leg_index,
    )


def _journal_execution_fill(
    journal: OrderJournal,
    *,
    attempt_seq: int | None,
    symbol: str,
    side: str,
    quantity: Decimal,
    price: Decimal,
    fee_bps: float,
    liquidity: str,
    reason: str,
    filled_at: pd.Timestamp,
    client_order_id: str | None,
    leg_index: int,
    cumulative_executed_qty: Decimal | None,
    simulated: bool,
) -> JournalFill:
    """Journal one execution delta first (WAL) before it reaches any in-memory outcome."""
    return journal.record_fill(
        kind="execution",
        attempt_seq=attempt_seq,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        fee_bps=fee_bps,
        liquidity=liquidity,
        reason=reason,
        filled_at=filled_at,
        client_order_id=client_order_id,
        leg_index=leg_index,
        cumulative_executed_qty=cumulative_executed_qty,
        simulated=simulated,
    )


def _journal_terminal(rt: _IntentRuntime, order_id: str | None, status: str) -> None:
    """Mark one submitted order id terminal, exactly once per runtime.

    Only ids that were actually submitted through this runtime are journaled, so reposts,
    rejections and never-placed attempts each resolve exactly once.
    """
    # journal_submitted 는 저널이 켜진 경우에만 채워지므로 이 검사가 저널 유무까지 가린다.
    if order_id is None or order_id in rt.journal_terminals or order_id not in rt.journal_submitted:
        return
    assert rt.journal is not None
    rt.journal_terminals.add(order_id)
    rt.journal.record_terminal(order_id, status)


def _cancel_and_settle(
    client: Any,
    rt: _IntentRuntime,
    audit: AuditLog,
    reason: str,
    touch: tuple[Decimal, Decimal] | None = None,
    *,
    now: float,
) -> None:
    """취소 직후 동일 주문을 재조회해 취소 시점 부분체결을 정산한다(-2011 benign).

    cancel 응답의 체결량을 폐기하면 그 사이 체결이 filled_total 에 누락되고
    다음 사이클 정합성 검증에서 HALT 로 이어진다. PAPER 의 가상 주문은 조회할
    실체가 없고 시뮬레이터가 체결을 이미 반영했으므로 로컬로만 해소한다.
    취소 결정은 감사 추적에 ``order_cancelled`` 로 남는다.
    """
    assert rt.active_id is not None
    fields: dict[str, Any] = {
        "symbol": rt.intent.symbol,
        "client_order_id": rt.active_id,
        "reason": reason,
    }
    if touch is not None:
        fields["bid"] = str(touch[0])
        fields["ask"] = str(touch[1])
    audit.record("order_cancelled", **fields)
    if rt.paper_active:
        _release_active(rt, "CANCELED")
        return
    confirmation = cancel_and_confirm(client, rt.intent.symbol, rt.active_id)
    if confirmation.payload is not None:
        _sync_venue_order(client, rt, now=now, audit=audit, touch=touch, payload=confirmation.payload)
    if not confirmation.closed or confirmation.payload is None:
        audit.record(
            "order_cancel_unconfirmed",
            symbol=rt.intent.symbol,
            client_order_id=rt.active_id,
            status=confirmation.status,
        )
        raise CancelNotConfirmed(rt.intent.symbol, str(rt.active_id), confirmation.status)
    _release_active(rt, confirmation.status)


def _emit_fill_event(
    audit: AuditLog,
    rt: _IntentRuntime,
    qty: Decimal,
    price: Decimal,
    liquidity: str,
    simulated: bool,
    touch: tuple[Decimal, Decimal] | None,
    client_order_id: str | None = None,
) -> None:
    """Per-fill audit evidence; sums reconcile exactly with the outcome ledger."""
    fields: dict[str, Any] = {
        "symbol": rt.intent.symbol,
        "client_order_id": client_order_id if client_order_id is not None else rt.active_id,
        "qty": str(qty),
        "price": str(price),
        "liquidity": liquidity,
        "simulated": simulated,
    }
    if touch is not None:
        fields["bid"] = str(touch[0])
        fields["ask"] = str(touch[1])
    audit.record("fill", **fields)


def _fill_time(now: float) -> pd.Timestamp:
    """UTC wall-clock confirmation stamp for one fill tuple entry."""
    return pd.Timestamp(now, unit="s", tz="UTC")


def _parse_avg_price(payload: Mapping[str, Any]) -> Decimal | None:
    """Return a positive venue average fill price when the payload contains one."""
    raw = payload.get("avgPrice") if "avgPrice" in payload else payload.get("avg_price")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        price = Decimal(str(raw))
    except Exception:
        return None
    return price if price.is_finite() and price > _ZERO else None


def _apply_fill(
    rt: _IntentRuntime,
    *,
    quantity: Decimal,
    price: Decimal,
    liquidity: Literal["maker", "taker"],
    client_order_id: str,
    order_cumulative_qty: Decimal,
    simulated: bool,
    now: float,
    audit: AuditLog,
    touch: tuple[Decimal, Decimal] | None = None,
) -> None:
    if quantity <= _ZERO or price <= _ZERO:
        raise ValueError("fill quantity and price must be positive")
    fee_bps = rt.fee_schedule.bps_for(liquidity)
    reason = "maker_fill" if liquidity == "maker" else "timeout_taker"
    filled_at = _fill_time(now)
    if rt.journal is not None and rt.journal_enabled:
        _journal_execution_fill(
            rt.journal,
            attempt_seq=rt.attempt_seq,
            symbol=rt.intent.symbol,
            side=rt.intent.side,
            quantity=quantity,
            price=price,
            fee_bps=float(fee_bps),
            liquidity=liquidity,
            reason=reason,
            filled_at=filled_at,
            client_order_id=client_order_id,
            leg_index=rt.intent.leg_index,
            cumulative_executed_qty=None if simulated else order_cumulative_qty,
            simulated=simulated,
        )
    rt.filled_total += quantity
    rt.fill_notional += quantity * price
    rt.reported_executed = order_cumulative_qty
    rt.fills.append((quantity, price, fee_bps, reason, liquidity, filled_at))
    _emit_fill_event(audit, rt, quantity, price, liquidity, simulated, touch, client_order_id)


def _sync_venue_order(
    client: Any,
    rt: _IntentRuntime,
    *,
    now: float,
    audit: AuditLog,
    touch: tuple[Decimal, Decimal] | None,
    payload: Mapping[str, Any] | None = None,
) -> str:
    """Apply new execution quantity for the current LIVE order from ``payload`` (queried when None)."""
    assert rt.active_id is not None
    assert not rt.paper_active
    if payload is None:
        payload = client.query_order(rt.intent.symbol, rt.active_id)
    status = str(payload.get("status", "") or "")
    executed = Decimal(str(payload.get("executedQty", "0")))
    if executed > rt.active_post_qty:
        raise DataIntegrityError(
            f"order {rt.active_id} executed {executed} exceeds submitted quantity {rt.active_post_qty}"
        )
    if executed > rt.reported_executed:
        delta = executed - rt.reported_executed
        _apply_fill(
            rt,
            quantity=delta,
            price=_parse_avg_price(payload) or rt.active_price,
            liquidity="maker" if rt.phase == "passive" else "taker",
            client_order_id=rt.active_id,
            order_cumulative_qty=executed,
            simulated=False,
            now=now,
            audit=audit,
            touch=touch,
        )
    return status


def _release_active(rt: _IntentRuntime, status: str) -> None:
    """Journal the current order terminal and clear its active-order state."""
    order_id = rt.active_id
    _journal_terminal(rt, order_id, status)
    rt.active_id = None
    rt.active_price = _ZERO
    rt.active_post_qty = _ZERO
    rt.reported_executed = _ZERO
    rt.paper_active = False


def _simulate_paper_fill(
    rt: _IntentRuntime,
    touch: tuple[Decimal, Decimal],
    time_in_force: str,
    price: Decimal,
    post_qty: Decimal,
) -> Decimal:
    """PAPER 로컬 체결 시뮬레이터: 이번 tick 의 관측 touch 만 사용한다(I4).

    GTX는 관측된 opposite touch 가 게시 가격을 관통할 때만(엄밀 trade-through)
    메이커로 전량 체결되고, IOC는 캡 가격이 이미 마케터블이므로 즉시 전량
    체결된다. 미래 호가 참조는 금지며, 체결 실패 시 0 을 반환해 chase 루프가
    다음 tick 을 이어간다.
    """
    bid, ask = touch
    is_buy = rt.intent.side == "BUY"
    if time_in_force == "GTX":
        traded_through = (ask < price) if is_buy else (bid > price)
        if not traded_through:
            return _ZERO
    return post_qty


@dataclass(slots=True)
class _IntentRuntime:
    """협조 루프가 유지하는 단일 intent 의 진행 상태."""

    intent: OrderIntent
    filters: SymbolFilters | None
    fee_schedule: FeeSchedule
    phase: str = "passive"  # 'passive' | 'ioc'
    active_id: str | None = None
    active_price: Decimal = _ZERO
    reported_executed: Decimal = _ZERO  # current active order cumulative quantity; venue-reported or simulated
    filled_total: Decimal = _ZERO
    fill_notional: Decimal = _ZERO
    chases: int = 0
    attempts: int = 0
    ioc_attempts: int = 0
    posted_at: float = 0.0
    finalized_at: float = 0.0
    terminal_status: str | None = None
    # PAPER 모드의 가상 활성 주문: 실제 주문이 없으므로 조회/취소가 아니라
    # 시뮬레이터가 이미 반영한 체결로 정산한다.
    paper_active: bool = False
    active_post_qty: Decimal = _ZERO
    fills: list[tuple[Decimal, Decimal, float, str, str, pd.Timestamp]] = field(default_factory=list)
    # 패시브 단계 진입 시각(phase-level 타임아웃 기준). 첫 _poll_or_post 호출에 기록되며
    # 재게시 때마다 갱신되는 posted_at 과 달리 리포스트로 리셋되지 않는다.
    passive_started_at: float = 0.0
    last_repeg_at: float = 0.0
    # anchored_repeg quote level: decision price until the first bar-cadence re-peg, then the last
    # re-peg price. Slicing, fills and post-only rejections never move it, so order splitting cannot
    # create an off-cadence re-quote.
    passive_quote: Decimal = _ZERO
    # 실행 상태 미확인 제출(lookup-before-resend): 재게시 금지, 조회로 해소한다.
    journal: OrderJournal | None = None
    unresolved_id: str | None = None
    unresolved_price: Decimal = _ZERO
    unresolved_post_qty: Decimal = _ZERO
    unresolved_tif: str = ""
    unknown_misses: int = 0
    read_failures: int = 0
    unresolved_at: float = 0.0
    reject_code: int | None = None
    margin_rejects: int = 0
    margin_wait_until: float = 0.0
    # Journal durability context: attempt this runtime belongs to, ids submitted through it,
    # ids already journaled terminal, and whether the shutdown path finalized it.
    attempt_seq: int | None = None
    shutdown_finalized: bool = False
    # True when a journal is attached; every mode (PAPER simulated fills included) journals
    # fills first because the ledger learns about executions only from the journal.
    journal_enabled: bool = False
    journal_submitted: set[str] = field(default_factory=set)
    journal_terminals: set[str] = field(default_factory=set)

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
        latency = (
            (self.finalized_at - self.posted_at)
            if self.posted_at > 0.0 and self.finalized_at > 0.0
            else None
        )
        # Guard against spurious negative due to clock skew (contract I-LATENCY-NONNEGATIVE).
        if latency is not None and latency < 0.0:
            latency = 0.0
        maker_qty = sum((qty for qty, _, _, _, liq, _ in self.fills if liq == "maker"), _ZERO)
        taker_qty = sum((qty for qty, _, _, _, liq, _ in self.fills if liq == "taker"), _ZERO)
        # If no fills recorded but filled_total>0 (fallback via paper path), assume maker for residual compatibility
        # but maker_qty/taker_qty already zero; downstream cashflow will fallback.
        return ExecutionOutcome(
            symbol=self.intent.symbol,
            filled_qty=self.filled_total,
            unfilled_qty=unfilled,
            avg_fill_price=(
                self.fill_notional / self.filled_total if self.filled_total > _ZERO else None
            ),
            chases=self.chases,
            status=status,
            latency_seconds=latency,
            fills=tuple(self.fills),
            maker_qty=maker_qty,
            taker_qty=taker_qty,
            reject_code=self.reject_code if status == "REJECTED" else None,
        )


def _order_budget_exceeded(client: Any, policy: PassiveExecutionPolicy, rate_limits: RateLimits | None) -> bool:
    if rate_limits is None:
        return False
    rate_state = getattr(client, "rate_state", None)
    if rate_state is None:
        return False
    fraction = policy.rate_weight_budget_fraction
    order_10s = getattr(rate_state, "order_count_10s", None)
    order_1m = getattr(rate_state, "order_count_1m", None)
    return bool(  # noqa: SIM103
        (order_10s is not None and rate_limits.orders_10s > 0 and order_10s > fraction * rate_limits.orders_10s)
        or (order_1m is not None and rate_limits.orders_1m > 0 and order_1m > fraction * rate_limits.orders_1m)
    )


def simulate_immediate_taker_fills(
    intents: Sequence[OrderIntent],
    books: Mapping[str, tuple[Decimal, Decimal]],
    policy: PassiveExecutionPolicy,
    *,
    now: float = 0.0,
) -> tuple[ExecutionOutcome, ...]:
    fee_bps = policy.fee_schedule.taker_fee_bps + policy.taker_slippage_bps
    outcomes: list[ExecutionOutcome] = []
    for intent in intents:
        touch = books.get(intent.symbol)
        if touch is None:
            outcomes.append(
                ExecutionOutcome(
                    symbol=intent.symbol,
                    filled_qty=_ZERO,
                    unfilled_qty=intent.quantity,
                    avg_fill_price=None,
                    chases=0,
                    status="RESIDUAL",
                    latency_seconds=0.0,
                    fills=(),
                    maker_qty=_ZERO,
                    taker_qty=_ZERO,
                )
            )
            continue
        bid, ask = touch
        mid = (bid + ask) / Decimal(2)
        fills = ((intent.quantity, mid, fee_bps, "immediate_taker", "taker", _fill_time(now)),)
        outcomes.append(
            ExecutionOutcome(
                symbol=intent.symbol,
                filled_qty=intent.quantity,
                unfilled_qty=_ZERO,
                avg_fill_price=mid,
                chases=0,
                status="FILLED",
                latency_seconds=0.0,
                fills=fills,
                maker_qty=_ZERO,
                taker_qty=intent.quantity,
            )
        )
    return tuple(outcomes)


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
    outcome_sink: list[ExecutionOutcome] | None = None,
    shutdown: Any | None = None,
    paper_fill_model: str | None = None,
    journal: OrderJournal | None = None,
    attempt: JournalAttempt | None = None,
    shutdown_cleanup_budget_s: float | None = None,
) -> tuple[ExecutionOutcome, ...]:
    """단일 협조 루프(post-all/poll-all/예산 스로틀). 반환 순서는 intents 와 1:1.

    어떤 예외 경로에서도 이미 확인된 체결을 잃지 않도록 진행 중 부분 결과를
    발생 예외에 partial_outcomes 속성으로 붙여 재전파한다(I-LEDGER-DURABLE 채널).

    Every fill delta is journaled (`OrderJournal.record_fill`, kind `execution`) before it is
    added to the in-memory outcome, and every order that can no longer fill is journaled
    terminal. A shutdown request stops posting, cancels and settles every active or unresolved
    order within `shutdown_cleanup_budget_s`, and raises `ExecutionInterrupted` carrying
    `partial_outcomes`; it never returns normally. Returning normally therefore always means
    the window ran to completion.
    """
    if (journal is None) != (attempt is None):
        raise ValueError("journal and attempt must be provided together (both or neither)")
    attempt_seq: int | None = getattr(attempt, "attempt_seq", None) if attempt is not None else None
    if paper_fill_model == "immediate_taker":
        if shutdown is not None and getattr(shutdown, "requested", False):
            interrupted = ExecutionInterrupted("shutdown requested before first tick")
            interrupted.partial_outcomes = ()
            raise interrupted
        if not intents:
            if outcome_sink is not None:
                outcome_sink[:] = []
            return ()
        books = _fetch_books(client, sorted({i.symbol for i in intents}), audit=audit)
        outcomes = simulate_immediate_taker_fills(intents, books, policy, now=clock())
        # PAPER 도 WAL 대상이다: 러너는 저널 fill 만으로 장부를 커밋한다(INV-FILL-WAL).
        if journal is not None:
            try:
                for it, oc in zip(intents, outcomes, strict=False):
                    for qty, price, fee, reason, liq, ts in oc.fills:
                        journal.record_fill(
                            kind="execution",
                            attempt_seq=attempt_seq,
                            symbol=it.symbol,
                            side=it.side,
                            quantity=qty,
                            price=price,
                            fee_bps=float(fee),
                            liquidity=liq,
                            reason=reason,
                            filled_at=ts,
                            client_order_id=None,
                            leg_index=it.leg_index,
                            cumulative_executed_qty=None,
                            simulated=True,
                        )
            except BaseException:
                # 저널에 남지 않은 체결을 러너가 커밋하지 않도록 sink 를 비운다(저널 = 유일한 진실).
                if outcome_sink is not None:
                    outcome_sink[:] = []
                raise
        if outcome_sink is not None:
            outcome_sink[:] = list(outcomes)
        for it, oc in zip(intents, outcomes, strict=False):
            audit.record(
                "intent_outcome",
                symbol=it.symbol,
                status=oc.status,
                filled_qty=str(oc.filled_qty),
                unfilled_qty=str(oc.unfilled_qty),
                chases=0,
            )
        return outcomes
    runtimes = [
        _IntentRuntime(
            intent=intent,
            filters=filters.get(intent.symbol),
            fee_schedule=policy.fee_schedule,
            journal=journal,
        )
        for intent in intents
    ]
    for rt in runtimes:
        rt.attempt_seq = attempt_seq
        rt.journal_enabled = journal is not None
        if rt.filters is None:
            rt.terminal_status = "RESIDUAL"
    try:
        _run_loop(client, runtimes, policy, audit, clock, sleep_fn, rate_limits, shutdown, shutdown_cleanup_budget_s)
    except BaseException as exc:
        _cleanup_on_abort(client, runtimes, audit, clock)
        if isinstance(exc, LiveTradingError):
            exc.partial_outcomes = tuple(rt.snapshot() for rt in runtimes)
        raise
    finally:
        if outcome_sink is not None:
            outcome_sink[:] = [rt.snapshot() for rt in runtimes]
    for rt in runtimes:
        outcome = rt.snapshot()
        fields: dict[str, Any] = {
            "symbol": rt.intent.symbol,
            "status": outcome.status,
            "filled_qty": str(rt.filled_total),
            "unfilled_qty": str(max(rt.intent.quantity - rt.filled_total, _ZERO)),
            "chases": rt.chases,
        }
        if outcome.reject_code is not None:
            fields["reject_code"] = outcome.reject_code
        audit.record("intent_outcome", **fields)
    return tuple(rt.snapshot() for rt in runtimes)


def _run_loop(
    client: Any,
    runtimes: Sequence[_IntentRuntime],
    policy: PassiveExecutionPolicy,
    audit: AuditLog,
    clock: Callable[[], float],
    sleep_fn: Callable[[float], None],
    rate_limits: RateLimits | None,
    shutdown: Any | None = None,
    shutdown_cleanup_budget_s: float | None = None,
) -> None:
    live = [rt for rt in runtimes if not rt.done]
    if not live:
        return
    symbols = sorted({rt.intent.symbol for rt in live})
    start = clock()
    interval = policy.poll_interval_s
    max_ticks = math.ceil(policy.window_deadline_s / policy.poll_interval_s) + 1
    ticks = 0
    book_failures = 0
    quote_invalid_reported: set[str] = set()
    cycle_flags = _CycleFlags()
    tick_order = sorted(live, key=lambda rt: 0 if rt.intent.reduce_only else 1)

    while ticks < max_ticks and any(not rt.done for rt in live):
        if shutdown is not None and getattr(shutdown, "requested", False):
            # Shutdown never returns normally: settle within the budget, audit, and raise
            # so the caller persists a partial outcome without marking the decision executed.
            unsettled = _shutdown_finalize(client, live, audit, clock, shutdown_cleanup_budget_s)
            audit.record("execution_interrupted", unsettled=unsettled)
            raise ExecutionInterrupted(
                f"shutdown requested during execution window with {unsettled} unsettled order(s)"
            )
        now = clock()
        if now - start >= policy.window_deadline_s:
            break
        try:
            books = _fetch_books(
                client, symbols, audit=audit, invalid_reported=quote_invalid_reported
            )
        except TransientReadError:
            book_failures += 1
            audit.record(
                "tick_skipped_read_failure", scope="books", consecutive=book_failures
            )
            if book_failures > policy.max_consecutive_read_failures:
                raise
            sleep_fn(interval)
            ticks += 1
            continue
        book_failures = 0
        for rt in tick_order:
            if not rt.done and rt.intent.symbol in books:
                _poll_or_post(
                    client,
                    rt,
                    books[rt.intent.symbol],
                    now,
                    policy,
                    audit,
                    rate_limits,
                    cycle_flags,
                )
        ticks += 1
        if all(rt.done for rt in live):
            break
        interval = _throttled_interval(client, interval, policy, rate_limits)
        sleep_fn(interval)

    _finalize(client, live, audit, clock, sleep_fn)


def _throttled_interval(
    client: Any, interval: float, policy: PassiveExecutionPolicy, rate_limits: RateLimits | None
) -> float:
    """Adapt the poll interval to the venue-reported budget with hysteresis.

    Doubles (capped at ``window_deadline_s / 4``) while used weight or order counts exceed
    ``rate_weight_budget_fraction`` of the venue limit; halves back toward ``poll_interval_s``
    once every reported counter is at or below ``rate_weight_recover_fraction``; otherwise keeps
    the current interval.
    """
    cap = policy.window_deadline_s / 4
    floor = policy.poll_interval_s
    if rate_limits is None:
        return floor
    rate_state = getattr(client, "rate_state", None)
    if rate_state is None:
        return floor
    used = getattr(rate_state, "used_weight_1m", None)
    order_10s = getattr(rate_state, "order_count_10s", None)
    order_1m = getattr(rate_state, "order_count_1m", None)
    if used is None and order_10s is None and order_1m is None:
        return floor
    budget_fraction = policy.rate_weight_budget_fraction
    recover_fraction = policy.rate_weight_recover_fraction
    over_budget = False
    if used is not None and rate_limits.request_weight_1m > 0 and used > budget_fraction * rate_limits.request_weight_1m:
        over_budget = True
    if order_10s is not None and rate_limits.orders_10s > 0 and order_10s > budget_fraction * rate_limits.orders_10s:
        over_budget = True
    if order_1m is not None and rate_limits.orders_1m > 0 and order_1m > budget_fraction * rate_limits.orders_1m:
        over_budget = True
    if over_budget:
        return min(max(interval * 2, floor), cap)
    recovered = True
    if used is not None and rate_limits.request_weight_1m > 0 and used > recover_fraction * rate_limits.request_weight_1m:
        recovered = False
    if order_10s is not None and rate_limits.orders_10s > 0 and order_10s > recover_fraction * rate_limits.orders_10s:
        recovered = False
    if order_1m is not None and rate_limits.orders_1m > 0 and order_1m > recover_fraction * rate_limits.orders_1m:
        recovered = False
    if recovered:
        return max(interval / 2, floor) if interval > floor else floor
    result = min(max(interval, floor), cap)
    return result


def _shutdown_finalize(
    client: Any,
    runtimes: Sequence[_IntentRuntime],
    audit: AuditLog,
    clock: Callable[[], float],
    budget_s: float | None,
) -> int:
    """Cancel and settle active/unresolved orders after a shutdown request.

    Venue calls stop once ``budget_s`` (default 20 s) elapses; ids left unsettled stay
    unresolved for restart recovery. Returns the number of unsettled order ids.
    Finalized runtimes are flagged so ``_cleanup_on_abort`` does not touch them twice.
    """
    budget = budget_s if isinstance(budget_s, (int, float)) and budget_s > 0 else 20.0
    start = clock()
    unsettled = 0
    for rt in runtimes:
        # Every runtime visited here is owned by the shutdown path (settled, already done,
        # or deliberately left for restart recovery), so _cleanup_on_abort must not touch it.
        rt.shutdown_finalized = True
        if rt.done or (rt.active_id is None and rt.unresolved_id is None):
            continue
        if clock() - start >= budget:
            unsettled += _open_order_ids(rt)
            continue
        try:
            _finalize(client, [rt], audit, clock, lambda _seconds: None)
        except Exception as cleanup_exc:  # noqa: BLE001 - every intent must be attempted
            audit.record(
                "abort_cleanup_failed",
                symbol=rt.intent.symbol,
                client_order_id=rt.active_id or rt.unresolved_id,
                error=type(cleanup_exc).__name__,
            )
        unsettled += _open_order_ids(rt)
    return unsettled


def _open_order_ids(rt: _IntentRuntime) -> int:
    """Order ids of one intent still live or unresolved on the venue (left for restart recovery)."""
    return int(rt.active_id is not None) + int(rt.unresolved_id is not None)


def _cleanup_on_abort(
    client: Any, runtimes: Sequence[_IntentRuntime], audit: AuditLog, clock: Callable[[], float]
) -> None:
    """어떤 예외로 중단되든 활성/미확인 주문을 취소·정산한다.

    정리 실패는 감사 기록만 남기고 원본 예외를 절대 가리지 않는다.
    """
    for rt in runtimes:
        if rt.shutdown_finalized:
            continue
        if rt.done or rt.paper_active or (rt.active_id is None and rt.unresolved_id is None):
            continue
        try:
            _finalize(client, [rt], audit, clock, lambda _seconds: None)
        except Exception as cleanup_exc:  # noqa: BLE001 - every intent must be attempted
            audit.record(
                "abort_cleanup_failed",
                symbol=rt.intent.symbol,
                client_order_id=rt.active_id or rt.unresolved_id,
                error=type(cleanup_exc).__name__,
            )


def _adopt_unresolved(rt: _IntentRuntime, now: float) -> None:
    """조회로 확인된 미확인 제출을 활성 주문으로 입양한다."""
    rt.active_id = rt.unresolved_id
    rt.active_price = rt.unresolved_price
    rt.active_post_qty = rt.unresolved_post_qty
    rt.reported_executed = _ZERO
    rt.posted_at = now
    if rt.unresolved_tif == "IOC":
        rt.ioc_attempts += 1
    rt.unresolved_id = None
    rt.unknown_misses = 0


def _unknown_outcome_horizon_s(client: Any) -> float:
    """Horizon after which an unconfirmed submission can be declared never placed (client property or derived default)."""
    horizon = getattr(client, "unknown_outcome_horizon_s", None)
    if isinstance(horizon, (int, float)):
        value = float(horizon)
        if value > 0:
            return value
    return float(DEFAULT_UNKNOWN_OUTCOME_HORIZON_S)


def _lookup_unknown_submission(client: Any, rt: _IntentRuntime) -> bool:
    """Query the unconfirmed submission by ``origClientOrderId``.

    Returns:
        True when the venue knows the order; False on ``-2013`` (not visible yet, which
        alone never proves the order was not placed).

    Raises:
        VenueError: any other venue code (fail-closed).
        TransientReadError: propagated; each caller decides whether a transient read is
            undecidable (tick path) or fatal for the cleanup attempt (exit path).
    """
    assert rt.unresolved_id is not None
    try:
        client.query_order(rt.intent.symbol, rt.unresolved_id)
    except VenueError as exc:
        if exc.code != ORDER_DOES_NOT_EXIST_CODE:
            raise
        return False
    return True


def _adopt_unknown_submission(rt: _IntentRuntime, now: float, audit: AuditLog) -> None:
    """Adopt a venue-confirmed unknown submission as the intent's active order."""
    audit.record(
        "order_unknown_adopted",
        symbol=rt.intent.symbol,
        client_order_id=rt.unresolved_id,
    )
    _adopt_unresolved(rt, now)


def _declare_not_placed(rt: _IntentRuntime, audit: AuditLog) -> None:
    """Record that an unconfirmed submission can no longer materialize and release it.

    Callers must have established the venue's unknown-outcome horizon; the journal
    terminal is written exactly once so restart recovery never re-resolves the id.
    """
    audit.record(
        "order_unknown_not_placed",
        symbol=rt.intent.symbol,
        client_order_id=rt.unresolved_id,
    )
    _journal_terminal(rt, rt.unresolved_id, "NOT_PLACED")
    rt.unresolved_id = None
    rt.unknown_misses = 0


def _resolve_unknown_submission(
    client: Any, rt: _IntentRuntime, now: float, audit: AuditLog
) -> bool:
    """Resolve an execution-status-unknown submission by ``origClientOrderId`` lookup.

    A found order is adopted. A ``-2013`` answer only proves the order is not visible *yet*; the
    submission is declared never placed only once ``now - rt.unresolved_at`` has reached the
    venue's unknown-outcome horizon (recvWindow + transport timeout) and at least
    ``UNKNOWN_SUBMISSION_MISS_LIMIT`` consecutive lookups returned ``-2013``. Until then the
    intent must not repost.

    Returns:
        True when resolved (adopted or declared not placed), False while still undecidable.
    """
    assert rt.unresolved_id is not None
    try:
        found = _lookup_unknown_submission(client, rt)
    except TransientReadError:
        return False
    if found:
        _adopt_unknown_submission(rt, now, audit)
        return True
    rt.unknown_misses += 1
    if rt.unknown_misses < UNKNOWN_SUBMISSION_MISS_LIMIT:
        return False
    if now - rt.unresolved_at < _unknown_outcome_horizon_s(client):
        return False
    _declare_not_placed(rt, audit)
    return True


def _resolve_unknown_at_exit(
    client: Any,
    rt: _IntentRuntime,
    audit: AuditLog,
    clock: Callable[[], float],
    sleep_fn: Callable[[float], None],
    now: float,
) -> float | None:
    """Resolve an unconfirmed submission when the execution window ends.

    Unlike the tick path there is no later tick, so the rule is horizon-only: a
    ``-2013`` at or beyond the unknown-outcome horizon declares the order never placed
    without a consecutive-miss count. Before the horizon the remaining horizon is
    slept once and the order re-queried once; if the sleeper could not advance the
    clock (shutdown/abort cleanup) the order is left unresolved for restart recovery
    instead of being declared early.

    Returns:
        The clock reading to continue window-end settlement with, or None when the
        order was left unresolved and the intent must be skipped.

    Raises:
        VenueError: a lookup answered with a code other than ``-2013``.
        TransientReadError: a lookup failed transiently (not swallowed at exit).
    """
    horizon = _unknown_outcome_horizon_s(client)
    if _lookup_unknown_submission(client, rt):
        _adopt_unknown_submission(rt, now, audit)
        return now
    if now - rt.unresolved_at >= horizon:
        _declare_not_placed(rt, audit)
        return now
    sleep_fn(horizon - (now - rt.unresolved_at))
    now = clock()
    if _lookup_unknown_submission(client, rt):
        _adopt_unknown_submission(rt, now, audit)
        return now
    if now - rt.unresolved_at < horizon:
        audit.record(
            "order_unknown_left_unresolved",
            symbol=rt.intent.symbol,
            client_order_id=rt.unresolved_id,
        )
        return None
    _declare_not_placed(rt, audit)
    return now


def _poll_active(
    client: Any,
    rt: _IntentRuntime,
    touch: tuple[Decimal, Decimal],
    now: float,
    policy: PassiveExecutionPolicy,
    audit: AuditLog,
) -> None:
    """Poll one active order and advance its phase; transient reads propagate to the caller."""
    assert rt.filters is not None
    assert rt.active_id is not None
    is_buy = rt.intent.side == "BUY"
    bid, ask = touch
    own_touch = bid if is_buy else ask
    if rt.paper_active:
        if (
            policy.passive_pricing in ("anchored", "anchored_repeg")
            and rt.phase == "passive"
            and rt.active_price > _ZERO
        ):
            resting_qty = rt.active_post_qty - rt.reported_executed
            if resting_qty > _ZERO:
                resting_fill = _simulate_paper_fill(rt, touch, "GTX", rt.active_price, resting_qty)
                if resting_fill > _ZERO:
                    order_cumulative = rt.reported_executed + resting_fill
                    _apply_fill(
                        rt,
                        quantity=resting_fill,
                        price=rt.active_price,
                        liquidity="maker",
                        client_order_id=rt.active_id,
                        order_cumulative_qty=order_cumulative,
                        simulated=True,
                        now=now,
                        audit=audit,
                        touch=touch,
                    )
                    if rt.reported_executed >= rt.active_post_qty:
                        _release_active(rt, "FILLED")
                        if rt.intent.quantity - rt.filled_total <= _ZERO:
                            rt.terminal_status = "FILLED"
                            rt.finalized_at = now
                        return
    else:
        _sync_venue_order(client, rt, now=now, audit=audit, touch=touch)
    if rt.intent.quantity - rt.filled_total <= _ZERO:
        if rt.active_id is not None:
            _release_active(rt, "FILLED")
        rt.terminal_status = "FILLED"
        rt.finalized_at = now
        return
    if rt.phase == "passive":
        band_low, band_high = _band(rt.intent, policy)
        timed_out = now - rt.passive_started_at >= policy.passive_deadline_s
        exhausted = policy.passive_pricing == "touch_chase" and rt.chases >= policy.max_chases
        repeg_due = policy.passive_pricing == "anchored_repeg" and now - rt.last_repeg_at >= policy.repeg_interval_s
        moved = abs(own_touch - rt.active_price) >= rt.filters.tick_size * policy.chase_ticks
        slice_done = rt.active_post_qty > _ZERO and rt.reported_executed >= rt.active_post_qty and (rt.intent.quantity - rt.filled_total) > _ZERO
        if timed_out:
            _cancel_and_settle(client, rt, audit, "passive_timeout", touch, now=now)
            rt.phase = "ioc"
        elif slice_done:
            _cancel_and_settle(client, rt, audit, "slice_done", touch, now=now)
        elif exhausted:
            return
        elif repeg_due:
            candidate = _band_capped_peg_price(
                own_touch, is_buy=is_buy, filters=rt.filters, band_low=band_low, band_high=band_high
            )
            rt.last_repeg_at = now
            rt.passive_quote = candidate
            if candidate != rt.active_price:
                _cancel_and_settle(client, rt, audit, "repeg", touch, now=now)
            else:
                return
        elif moved and policy.passive_pricing == "touch_chase":
            _cancel_and_settle(client, rt, audit, "chase", touch, now=now)
            rt.chases += 1
        else:
            return
    else:
        _cancel_and_settle(client, rt, audit, "ioc_expired", touch, now=now)


def _end_rejected(
    rt: _IntentRuntime, audit: AuditLog, now: float, code: int | None, order_id: str | None = None
) -> None:
    """End one intent REJECTED with its venue code; other intents continue."""
    _journal_terminal(rt, order_id, "REJECTED")
    rt.terminal_status = "REJECTED"
    rt.reject_code = code
    rt.finalized_at = now
    fields: dict[str, Any] = {"symbol": rt.intent.symbol, "code": code}
    if order_id is not None:
        fields["client_order_id"] = order_id
    audit.record("intent_rejected", **fields)


def _end_sub_minimum(
    rt: _IntentRuntime,
    audit: AuditLog,
    now: float,
    remaining: Decimal,
    price: Decimal,
) -> None:
    """End one intent RESIDUAL_SUB_MINIMUM without posting; fills are preserved."""
    assert rt.filters is not None
    rt.terminal_status = "RESIDUAL_SUB_MINIMUM"
    rt.finalized_at = now
    audit.record(
        "intent_sub_minimum",
        symbol=rt.intent.symbol,
        remaining=str(remaining),
        price=str(price),
        min_qty=str(rt.filters.min_qty),
        min_notional=str(rt.filters.min_notional),
    )


def _handle_post_error(
    exc: VenueError,
    rt: _IntentRuntime,
    *,
    order_id: str,
    now: float,
    policy: PassiveExecutionPolicy,
    audit: AuditLog,
    cycle_flags: _CycleFlags | None,
) -> bool:
    """Apply the scoped consequence of a rejected ``new_order`` to one intent.

    Policy comes only from ``resolve_error_action``; the one exception is a venue rate
    limit, which the REST layer reports as ``http_status=429`` with no venue code
    (registry would read it as FAIL_CLOSED) and as ``-1003`` (RETRY_BACKOFF_LONG). A
    rate-limited submission was not executed, so it is journaled NOT_PLACED and the
    intent retries on a later tick with a fresh submit_seq. A post-only reprice signal
    (BENIGN_REPRICE) is likewise NOT_PLACED and counts one chase. Scoped rejections
    end only this intent; RISK_INCREASE_FREEZE additionally freezes every later
    risk-increasing intent of the cycle. MARGIN_WAIT never retries a reduce-only
    intent: insufficient margin on a reduction cannot be cured by waiting.

    Returns:
        True when the error was fully handled for this intent; False when the action
        is not scoped to one intent (FAIL_CLOSED, unregistered code, RETRY_BACKOFF,
        BENIGN, RESYNC_*), in which case the caller must re-raise the original
        exception unchanged.
    """
    action = resolve_error_action(exc.code)
    if exc.http_status == HTTPStatus.TOO_MANY_REQUESTS or action is ErrorAction.RETRY_BACKOFF_LONG:
        _journal_terminal(rt, order_id, "NOT_PLACED")
        audit.record(
            "order_rate_limited",
            symbol=rt.intent.symbol,
            client_order_id=order_id,
        )
        return True
    if action is ErrorAction.INTENT_REJECT:
        _end_rejected(rt, audit, now, exc.code, order_id)
        return True
    if action is ErrorAction.MARGIN_WAIT:
        if rt.intent.reduce_only:
            _end_rejected(rt, audit, now, exc.code, order_id)
            return True
        rt.margin_rejects += 1
        if rt.margin_rejects > policy.max_margin_rejects:
            _end_rejected(rt, audit, now, exc.code, order_id)
            return True
        rt.margin_wait_until = now + policy.margin_retry_s
        _journal_terminal(rt, order_id, "NOT_PLACED")
        audit.record(
            "intent_margin_wait",
            symbol=rt.intent.symbol,
            client_order_id=order_id,
            margin_rejects=rt.margin_rejects,
        )
        return True
    if action is ErrorAction.RISK_INCREASE_FREEZE:
        if cycle_flags is not None:
            cycle_flags.risk_increase_frozen = True
            cycle_flags.freeze_code = exc.code
        audit.record(
            "risk_increase_frozen",
            symbol=rt.intent.symbol,
            code=exc.code,
        )
        _end_rejected(rt, audit, now, exc.code, order_id)
        return True
    if action is ErrorAction.BENIGN_REPRICE:
        _journal_terminal(rt, order_id, "NOT_PLACED")
        rt.chases += 1
        return True
    return False


def _mark_posted(
    rt: _IntentRuntime,
    *,
    order_id: str,
    price: Decimal,
    post_qty: Decimal,
    now: float,
    simulated: bool,
) -> None:
    """Make ``order_id`` the intent's active order with zero reported execution.

    Must run before any fill of the new order is applied: fill settlement and
    ``_release_active`` read the active-order fields. ``simulated`` marks a PAPER
    order, which is later settled from the simulator instead of venue queries.
    """
    rt.active_id = order_id
    rt.active_price = price
    rt.active_post_qty = post_qty
    rt.reported_executed = _ZERO
    rt.posted_at = now
    if simulated:
        rt.paper_active = True


def _record_order_posted(
    rt: _IntentRuntime,
    audit: AuditLog,
    *,
    order_id: str,
    time_in_force: str,
    price: Decimal,
    post_qty: Decimal,
    touch: tuple[Decimal, Decimal],
    simulated: bool,
) -> None:
    """Count an IOC attempt and emit the ``order_posted`` evidence for one accepted post.

    An IOC attempt is counted exactly when its post is announced, so a PAPER IOC that
    completes the intent on submission (``paper_filled``) is neither counted nor
    announced. ``simulated`` adds ``simulated=True`` to the event; LIVE events carry no
    ``simulated`` key.
    """
    if time_in_force == "IOC":
        rt.ioc_attempts += 1
    fields: dict[str, Any] = {
        "symbol": rt.intent.symbol,
        "client_order_id": order_id,
        "time_in_force": time_in_force,
        "price": str(price),
        "quantity": str(post_qty),
        "bid": str(touch[0]),
        "ask": str(touch[1]),
        "phase": "passive" if time_in_force == "GTX" else "ioc",
    }
    if simulated:
        fields["simulated"] = True
    audit.record("order_posted", **fields)


def _settle_paper_post(
    rt: _IntentRuntime,
    audit: AuditLog,
    *,
    order_id: str,
    time_in_force: str,
    price: Decimal,
    post_qty: Decimal,
    touch: tuple[Decimal, Decimal],
    now: float,
) -> None:
    """Settle one suppressed PAPER post against this tick's observed touch only.

    The simulator never reads a future quote. A fill that completes the intent ends it
    FILLED with ``paper_filled`` evidence and no ``order_posted``; otherwise the order
    rests (GTX) or is announced as an attempted IOC like a LIVE post.
    """
    executed_qty = _simulate_paper_fill(rt, touch, time_in_force, price, post_qty)
    _mark_posted(rt, order_id=order_id, price=price, post_qty=post_qty, now=now, simulated=True)
    if executed_qty > _ZERO:
        _apply_fill(
            rt,
            quantity=executed_qty,
            price=price,
            liquidity="maker" if time_in_force == "GTX" else "taker",
            client_order_id=order_id,
            order_cumulative_qty=executed_qty,
            simulated=True,
            now=now,
            audit=audit,
            touch=touch,
        )
        if executed_qty >= post_qty:
            _release_active(rt, "FILLED")
        if rt.intent.quantity - rt.filled_total <= _ZERO:
            rt.terminal_status = "FILLED"
            rt.finalized_at = now
            audit.record(
                "paper_filled",
                symbol=rt.intent.symbol,
                client_order_id=order_id,
                quantity=str(executed_qty),
                price=str(price),
                time_in_force=time_in_force,
            )
            return
    _record_order_posted(
        rt, audit, order_id=order_id, time_in_force=time_in_force, price=price,
        post_qty=post_qty, touch=touch, simulated=True,
    )


def _poll_or_post(
    client: Any,
    rt: _IntentRuntime,
    touch: tuple[Decimal, Decimal],
    now: float,
    policy: PassiveExecutionPolicy,
    audit: AuditLog,
    rate_limits: RateLimits | None = None,
    cycle_flags: _CycleFlags | None = None,
) -> None:
    assert rt.filters is not None  # filters 부재 intent 는 생성 시 즉시 RESIDUAL 처리된다
    if rt.passive_started_at == 0.0:
        rt.passive_started_at = now
    if rt.last_repeg_at == 0.0:
        rt.last_repeg_at = now
    is_buy = rt.intent.side == "BUY"
    bid, ask = touch
    own_touch = bid if is_buy else ask
    opposite_touch = ask if is_buy else bid
    band_low, band_high = _band(rt.intent, policy)
    rail_low, rail_high = _risk_rail(bid, ask, policy)

    # 0) 미확인 제출: 조회로 해소될 때까지 재게시 금지.
    if rt.unresolved_id is not None and not _resolve_unknown_submission(client, rt, now, audit):
        return

    # 1) 활성 주문 조회: 체결 누적 및 상태 전이(FILL/CHASE/HOLD/IOC).
    if rt.active_id is not None:
        try:
            _poll_active(client, rt, touch, now, policy, audit)
        except TransientReadError:
            rt.read_failures += 1
            audit.record(
                "tick_skipped_read_failure",
                scope="order",
                symbol=rt.intent.symbol,
                client_order_id=rt.active_id,
                consecutive=rt.read_failures,
            )
            if rt.read_failures > policy.max_consecutive_read_failures:
                raise
            return
        rt.read_failures = 0
        if rt.done:
            return
        if rt.active_id is not None:
            return

    # 2) 게시: 패시브(GTX, 밴드 내) 또는 백스톱(IOC, 캡+밴드 클램프).
    remaining = rt.intent.quantity - rt.filled_total
    if remaining <= _ZERO:
        _journal_terminal(rt, rt.active_id, "FILLED")
        rt.terminal_status = "FILLED"
        rt.finalized_at = now
        return
    if now < rt.margin_wait_until:
        return
    if (
        cycle_flags is not None
        and cycle_flags.risk_increase_frozen
        and not rt.intent.reduce_only
    ):
        _end_rejected(rt, audit, now, cycle_flags.freeze_code)
        return
    if rt.phase == "passive" and now - rt.passive_started_at >= policy.passive_deadline_s:
        # 휴지 주문 없이 밴드 밖에서 대기한 경우에도 phase-level 타임아웃으로
        # 캡 적용 IOC 백스톱으로 상승시킨다(리포스트 리셋 없음).
        rt.phase = "ioc"
    if rt.phase == "passive":
        if policy.passive_pricing in ("anchored", "anchored_repeg"):
            anchor = rt.passive_quote if rt.passive_quote > _ZERO else rt.intent.decision_price
            price = _post_only_quote_price(anchor, touch, is_buy=is_buy, filters=rt.filters)
        else:
            price = _gtx_candidate(
                own_touch, is_buy=is_buy, filters=rt.filters, band_low=band_low, band_high=band_high
            )
        time_in_force = "GTX"
    else:
        if rt.ioc_attempts >= policy.max_ioc_attempts:
            rt.terminal_status = "RESIDUAL"
            rt.finalized_at = now
            return
        price = _capped_ioc_price(
            opposite_touch,
            is_buy=is_buy,
            taker_cap_bps=policy.taker_cap_bps,
            tick_size=rt.filters.tick_size,
            band_low=rail_low,
            band_high=rail_high,
        )
        time_in_force = "IOC"
    if price is None:
        return  # 리스크 레일 밖 이상 징후: 재게시하지 않고 대기한다.

    filters = rt.filters
    if remaining < filters.min_qty:
        _end_sub_minimum(rt, audit, now, remaining, price)
        return
    post_qty = _post_quantity(
        remaining, price, filters=filters, max_slices=policy.max_slices
    )
    if post_qty < filters.min_qty or post_qty * price < filters.min_notional:
        if remaining * price < filters.min_notional and not rt.intent.reduce_only:
            _end_sub_minimum(rt, audit, now, remaining, price)
            return
        post_qty = remaining

    if _order_budget_exceeded(client, policy, rate_limits):
        return

    journaled = rt.journal_enabled and rt.journal is not None
    submit_seq = rt.journal.next_submit_seq() if journaled and rt.journal is not None else rt.attempts
    order_id = build_client_order_id(
        rt.intent.client_order_prefix, rt.intent.symbol, rt.intent.leg_index, 0, submit_seq
    )
    if journaled and rt.journal is not None and rt.attempt_seq is not None:
        _journal_submit(
            rt.journal,
            client_order_id=order_id,
            symbol=rt.intent.symbol,
            submit_seq=submit_seq,
            attempt_seq=rt.attempt_seq,
            side=rt.intent.side,
            quantity=post_qty,
            reduce_only=rt.intent.reduce_only,
            leg_index=rt.intent.leg_index,
        )
        rt.journal_submitted.add(order_id)
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
    except OrderStatusUnknown:
        rt.unresolved_id = order_id
        rt.unresolved_price = price
        rt.unresolved_post_qty = post_qty
        rt.unresolved_tif = time_in_force
        rt.unknown_misses = 0
        rt.unresolved_at = now
        audit.record(
            "order_status_unknown",
            symbol=rt.intent.symbol,
            client_order_id=order_id,
        )
        return
    except VenueError as exc:
        if not _handle_post_error(
            exc, rt, order_id=order_id, now=now, policy=policy, audit=audit, cycle_flags=cycle_flags
        ):
            raise
        return
    except OrderObsolete:
        _journal_terminal(rt, order_id, "OBSOLETE")
        rt.terminal_status = "OBSOLETE"
        rt.finalized_at = now
        audit.record("intent_obsolete", symbol=rt.intent.symbol, client_order_id=order_id)
        return
    if isinstance(response, ShadowResponse):
        rt.terminal_status = "SHADOW"
        rt.finalized_at = now
        return
    if isinstance(response, PaperResponse):
        _settle_paper_post(
            rt, audit, order_id=order_id, time_in_force=time_in_force, price=price,
            post_qty=post_qty, touch=touch, now=now,
        )
        return
    _mark_posted(rt, order_id=order_id, price=price, post_qty=post_qty, now=now, simulated=False)
    _record_order_posted(
        rt, audit, order_id=order_id, time_in_force=time_in_force, price=price,
        post_qty=post_qty, touch=touch, simulated=False,
    )


def _finalize(
    client: Any,
    runtimes: Sequence[_IntentRuntime],
    audit: AuditLog,
    clock: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> None:
    """윈도우 종료 시 활성 주문을 정리하고 최종 체결을 확정한다."""
    for rt in runtimes:
        if rt.done:
            continue
        now = clock()
        if rt.unresolved_id is not None:
            resolved_now = _resolve_unknown_at_exit(client, rt, audit, clock, sleep_fn, now)
            if resolved_now is None:
                continue
            now = resolved_now
        if rt.active_id is not None:
            if not rt.paper_active:
                confirmation = cancel_and_confirm(client, rt.intent.symbol, rt.active_id)
                if confirmation.payload is not None:
                    _sync_venue_order(client, rt, now=now, audit=audit, touch=None, payload=confirmation.payload)
                if not confirmation.closed or confirmation.payload is None:
                    audit.record(
                        "order_cancel_unconfirmed",
                        symbol=rt.intent.symbol,
                        client_order_id=rt.active_id,
                        status=confirmation.status,
                    )
                    raise CancelNotConfirmed(rt.intent.symbol, str(rt.active_id), confirmation.status)
                audit.record(
                    "order_cancelled",
                    symbol=rt.intent.symbol,
                    client_order_id=rt.active_id,
                    reason="window_end",
                )
                _release_active(rt, confirmation.status)
            else:
                _release_active(rt, "CANCELED")
        if rt.intent.quantity - rt.filled_total > _ZERO:
            rt.terminal_status = "RESIDUAL"
            rt.finalized_at = now
            audit.record(
                "order_residual",
                symbol=rt.intent.symbol,
                quantity=str(rt.intent.quantity - rt.filled_total),
            )
        else:
            rt.terminal_status = "FILLED"
            rt.finalized_at = now


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
