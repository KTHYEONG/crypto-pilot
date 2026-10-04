"""Confirmed order cancellation and pre-reconciliation orphan-order sweep.

I-CANCEL-CONFIRMED: an own order is never treated as closed on the strength of a cancel response alone;
only a follow-up order query (or "order does not exist") proves it, so an order still resting at the venue
is reported as unresolved instead of being journaled terminal.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.live.audit import AuditLog
from src.live.errors import LiveTradingError, VenueError
from src.live.filters import _ZERO
from src.live.order_journal import OrderJournal
from src.live.planner import CLIENT_ORDER_NAMESPACE
from src.live.rest import OrderStatusUnknown

if TYPE_CHECKING:
    from src.live.order_journal import JournalFill

#: 취소/조회에서 benign(사라진 주문)으로 취급하는 베뉴 코드.
_ORDER_GONE_CODES: frozenset[int] = frozenset({-2011, -2013})

#: Venue code for "Order does not exist" on an order query: the order is gone (never accepted, or archived).
ORDER_DOES_NOT_EXIST_CODE: int = -2013

#: Terminal status journaled for an order the venue no longer knows.
ORDER_NOT_FOUND_STATUS: str = "NOT_FOUND"

#: 취소 상태 미확인 시 해소 조회에서 '아직 열림'으로 판정하는 상태 집합.
_OPEN_ORDER_STATUSES: frozenset[str] = frozenset({"NEW", "PARTIALLY_FILLED"})

#: 네임스페이스 이전('%Y%m%d-' 접두) 레거시 client order id.
_LEGACY_CLIENT_ORDER_ID = re.compile(r"^\d{8}-")


@dataclass(frozen=True, slots=True)
class UnresolvedOrder:
    """One own order whose venue state is open or not provably closed after a confirmed cancel attempt.

    ``attempt_seq`` / ``leg_index`` come from the journal submit when known so recovered fills keep their
    attempt attribution; orphan-sweep orders (possibly legacy ids) carry ``None`` / ``0``.
    """

    client_order_id: str
    symbol: str
    attempt_seq: int | None = None
    leg_index: int = 0


@dataclass(frozen=True, slots=True)
class CancelConfirmation:
    """Venue-confirmed state of one order after a single cancel request.

    ``closed`` is True only when the confirming query reported a status outside ``_OPEN_ORDER_STATUSES``
    or answered ``ORDER_DOES_NOT_EXIST_CODE``. ``payload`` is the confirming query response, None iff
    ``status == ORDER_NOT_FOUND_STATUS``.
    """

    symbol: str
    client_order_id: str
    closed: bool
    status: str
    payload: Mapping[str, Any] | None


class CancelNotConfirmed(LiveTradingError):  # noqa: N818 - mirrors ForeignOpenOrderError naming
    """An in-execution cancel could not be confirmed closed; the order stays non-terminal for restart recovery."""

    def __init__(self, symbol: str, client_order_id: str, status: str) -> None:
        super().__init__(
            f"cancel of {client_order_id} on {symbol} not confirmed closed (status={status})"
        )
        self.symbol = symbol
        self.client_order_id = client_order_id
        self.status = status


@dataclass(frozen=True, slots=True)
class OrphanSweep:
    fills: tuple[JournalFill, ...]
    foreign_symbols: tuple[str, ...]
    unconfirmed: tuple[UnresolvedOrder, ...] = ()

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.fills)

    def __len__(self) -> int:
        return len(self.fills)

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self.fills[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return list(self.fills) == list(other)
        if isinstance(other, OrphanSweep):
            return (
                self.fills == other.fills
                and self.foreign_symbols == other.foreign_symbols
                and self.unconfirmed == other.unconfirmed
            )
        return NotImplemented


def cancel_and_confirm(client: Any, symbol: str, client_order_id: str) -> CancelConfirmation:
    """Send one cancel and confirm the order's venue state with a follow-up query (I-CANCEL-CONFIRMED).

    Why: ``-2011`` on cancel and a transport-unknown cancel are both ambiguous — Binance answers
    ``-2011`` while an accept is still propagating, so the order may still be ``NEW``. Only the
    order query is authoritative. The cancel is never re-sent; ambiguity is resolved by reading.

    Args:
        client: venue order client exposing ``cancel_order`` and ``query_order``.
        symbol: order symbol.
        client_order_id: our client order id.

    Returns:
        CancelConfirmation; ``closed=False`` means the order is still open at the venue.

    Raises:
        VenueError: the cancel failed with a code outside ``_ORDER_GONE_CODES``, or the query failed
            with a code other than ``ORDER_DOES_NOT_EXIST_CODE``.
        DataIntegrityError: the query payload has no non-empty ``status`` (I-RECOVERY-STATUS).
    """
    try:
        client.cancel_order(symbol, client_order_id)
    except VenueError as exc:
        if exc.code not in _ORDER_GONE_CODES:
            raise
    except OrderStatusUnknown:
        pass
    try:
        payload = client.query_order(symbol, client_order_id)
    except VenueError as exc:
        if exc.code == ORDER_DOES_NOT_EXIST_CODE:
            return CancelConfirmation(
                symbol=symbol,
                client_order_id=client_order_id,
                closed=True,
                status=ORDER_NOT_FOUND_STATUS,
                payload=None,
            )
        raise
    status = str(payload.get("status", "") or "")
    if not status:
        raise DataIntegrityError(f"order {client_order_id} query returned no status")
    return CancelConfirmation(
        symbol=symbol,
        client_order_id=client_order_id,
        closed=status not in _OPEN_ORDER_STATUSES,
        status=status,
        payload=payload,
    )


def cancel_orphan_orders(
    client: Any,
    client_order_prefix: str,
    audit: AuditLog,
    *,
    journal: OrderJournal,
    now: pd.Timestamp,
    taker_fee_bps: float,
) -> OrphanSweep:
    """Cancel namespace orphan orders before reconciliation and report foreign and unconfirmed orders.

    Foreign (non-namespace) open orders are never cancelled; their symbols are returned in
    ``foreign_symbols`` and audited as ``foreign_open_order``. Each own order gets one confirmed cancel
    (``cancel_and_confirm``): ``orphan_cancelled`` is audited only once the venue confirms it closed;
    an order still open is audited ``orphan_cancel_unconfirmed`` and returned in ``unconfirmed`` so the
    caller can freeze its symbol instead of halting the whole cycle.

    Executed quantity beyond the journal's observed quantity is journaled as an ``orphan_settlement``
    fill (open or closed); the ledger learns about it only through ``commit_journal_fills``. Settlements
    are booked at ``taker_fee_bps`` because the order query does not reveal maker/taker.

    Raises:
        DataIntegrityError: a confirming query lacks a status, or a settled order's response lacks a
            parseable executedQty, a BUY/SELL side, or a positive avgPrice.
        VenueError: non-benign cancel or query rejection.
    """
    open_orders = client.open_orders()
    ours: list[Mapping[str, Any]] = []
    foreign_symbols: list[str] = []
    for entry in open_orders:
        order_id = str(entry.get("clientOrderId", ""))
        if order_id.startswith(CLIENT_ORDER_NAMESPACE) or _LEGACY_CLIENT_ORDER_ID.match(order_id):
            ours.append(entry)
        else:
            symbol = str(entry.get("symbol", ""))
            if symbol and symbol not in foreign_symbols:
                foreign_symbols.append(symbol)
            audit.record(
                "foreign_open_order",
                symbol=entry.get("symbol"),
                client_order_id=order_id,
            )
    settlements: list[JournalFill] = []
    unconfirmed: list[UnresolvedOrder] = []
    for entry in ours:
        order_id = str(entry.get("clientOrderId", ""))
        symbol = str(entry["symbol"])
        confirmation = cancel_and_confirm(client, symbol, order_id)
        if confirmation.status == ORDER_NOT_FOUND_STATUS:
            audit.record(
                "orphan_cancelled",
                symbol=entry.get("symbol"),
                client_order_id=order_id,
                current_run=order_id.startswith(f"{CLIENT_ORDER_NAMESPACE}{client_order_prefix}"),
                status=ORDER_NOT_FOUND_STATUS,
            )
            continue
        assert confirmation.payload is not None
        queried = confirmation.payload
        status = confirmation.status
        if not confirmation.closed:
            audit.record(
                "orphan_cancel_unconfirmed",
                symbol=symbol,
                client_order_id=order_id,
                status=status,
            )
        else:
            audit.record(
                "orphan_cancelled",
                symbol=entry.get("symbol"),
                client_order_id=order_id,
                current_run=order_id.startswith(f"{CLIENT_ORDER_NAMESPACE}{client_order_prefix}"),
                status=status,
            )
        executed_raw = queried.get("executedQty")
        try:
            executed_qty = Decimal(str(executed_raw))
        except ArithmeticError as exc:
            raise DataIntegrityError(
                f"orphan order {order_id} has unparseable executedQty {executed_raw!r}"
            ) from exc
        if not executed_qty.is_finite():
            raise DataIntegrityError(f"orphan order {order_id} has unparseable executedQty {executed_raw!r}")
        observed = journal.observed_qty(order_id)
        delta = executed_qty - observed
        if delta <= _ZERO:
            if confirmation.closed:
                journal.record_terminal(order_id, status)
            else:
                unconfirmed.append(UnresolvedOrder(client_order_id=order_id, symbol=symbol))
            continue
        side = queried.get("side") or entry.get("side")
        if side not in ("BUY", "SELL"):
            raise DataIntegrityError(f"orphan order {order_id} settled {delta} without a venue side")
        avg_raw = queried.get("avgPrice", entry.get("avgPrice"))
        try:
            avg_price = Decimal(str(avg_raw)) if avg_raw is not None else None
        except ArithmeticError as exc:
            raise DataIntegrityError(f"orphan order {order_id} has unparseable avgPrice {avg_raw!r}") from exc
        # Never invent a price: a settled quantity without a positive venue price fails closed.
        if avg_price is None or not avg_price.is_finite() or avg_price <= _ZERO:
            raise DataIntegrityError(
                f"orphan order {order_id} settled {delta} with no positive venue avgPrice"
            )
        fill = journal.record_fill(
            kind="orphan_settlement",
            attempt_seq=None,
            symbol=symbol,
            side=str(side),
            quantity=delta,
            price=avg_price,
            fee_bps=float(taker_fee_bps),
            liquidity="taker",
            reason="orphan_settlement",
            filled_at=now,
            client_order_id=order_id,
            leg_index=0,
            cumulative_executed_qty=executed_qty,
            simulated=False,
        )
        settlements.append(fill)
        audit.record(
            "orphan_settled",
            symbol=symbol,
            client_order_id=order_id,
            executed_qty=str(delta),
            previously_observed=str(observed),
        )
        if confirmation.closed:
            journal.record_terminal(order_id, status)
        else:
            unconfirmed.append(UnresolvedOrder(client_order_id=order_id, symbol=symbol))
    return OrphanSweep(
        fills=tuple(settlements),
        foreign_symbols=tuple(sorted(foreign_symbols)),
        unconfirmed=tuple(unconfirmed),
    )
