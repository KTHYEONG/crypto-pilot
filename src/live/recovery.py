"""Restart recovery of fills that the venue executed but the journal never observed (process killed between a fill and the next poll). Runs before reconciliation so the ledger reflects every venue execution of our own orders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.live.audit import AuditLog
from src.live.errors import VenueError
from src.live.order_cancel import (
    _OPEN_ORDER_STATUSES,
    ORDER_DOES_NOT_EXIST_CODE,
    ORDER_NOT_FOUND_STATUS,
    UnresolvedOrder,
    cancel_and_confirm,
)
from src.live.order_journal import OrderJournal

if TYPE_CHECKING:
    from src.live.order_journal import JournalFill

#: Venue statuses that mean the order may still fill; left for the open-order sweep.

@dataclass(frozen=True, slots=True)
class RecoveryReport:
    recovered: tuple[JournalFill, ...]
    resolved_ids: tuple[str, ...]  # ids journaled terminal by this recovery
    unresolved_ids: tuple[str, ...]  # still open on the venue after query
    unresolved: tuple[UnresolvedOrder, ...] = ()  # same orders as unresolved_ids, with symbol and attempt attribution


@dataclass(frozen=True, slots=True)
class UnresolvedSettlement:
    recovered: tuple[JournalFill, ...]
    resolved_ids: tuple[str, ...]
    still_open: tuple[UnresolvedOrder, ...]


def _is_suppressed_client(client: Any) -> bool:
    """True when the client suppresses mutations (NullOrderClient / PAPER / SHADOW)."""
    mode = getattr(client, "mode", None)
    if mode is not None and bool(getattr(mode, "suppresses_mutations", False)):
        return True
    return type(client).__name__ == "NullOrderClient"


def _venue_decimal(payload: Any, key: str, order_id: str) -> Decimal:
    raw = payload.get(key)
    if raw is None:
        raise DataIntegrityError(f"recovery query for {order_id} lacks {key}")
    try:
        value = Decimal(str(raw))
    except ArithmeticError as exc:
        raise DataIntegrityError(f"recovery query for {order_id} has invalid {key}: {raw!r}") from exc
    if not value.is_finite():
        raise DataIntegrityError(f"recovery query for {order_id} has invalid {key}: {raw!r}")
    return value


def _venue_update_time(payload: Any, order_id: str) -> pd.Timestamp:
    """The venue's last-update instant of a closed order, which bounds when its fills happened."""
    raw = payload.get("updateTime")
    try:
        millis = int(raw)
    except (TypeError, ValueError) as exc:
        raise DataIntegrityError(f"recovery query for {order_id} has invalid updateTime: {raw!r}") from exc
    if millis <= 0:
        raise DataIntegrityError(f"recovery query for {order_id} has invalid updateTime: {raw!r}")
    return pd.Timestamp(millis, unit="ms", tz="UTC")


def _book_recovered_delta(
    journal: OrderJournal,
    audit: AuditLog,
    *,
    order_id: str,
    symbol: str,
    attempt_seq: int | None,
    leg_index: int,
    payload: Any,
    taker_fee_bps: float,
) -> JournalFill | None:
    """Journal the venue executed delta over the observed watermark as one ``recovered`` fill.

    Returns None when nothing new executed. Validates side, avgPrice and updateTime only when a
    delta exists, so closed orders without new executions terminal without venue-field demands.
    """
    executed = _venue_decimal(payload, "executedQty", order_id)
    observed = journal.observed_qty(order_id)
    delta = executed - observed
    if delta <= 0:
        return None
    side = payload.get("side", None)
    if side not in ("BUY", "SELL"):
        raise DataIntegrityError(f"recovery query for {order_id} lacks side")
    price = _venue_decimal(payload, "avgPrice", order_id)
    if price <= 0:
        raise DataIntegrityError(f"recovery query for {order_id} reports fills with no positive avgPrice")
    filled_at = _venue_update_time(payload, order_id)
    fill = journal.record_fill(
        kind="recovered",
        attempt_seq=attempt_seq,
        symbol=symbol,
        side=str(side),
        quantity=delta,
        price=price,
        fee_bps=float(taker_fee_bps),
        liquidity="taker",
        reason="recovered_fill",
        filled_at=filled_at,
        client_order_id=order_id,
        leg_index=leg_index,
        cumulative_executed_qty=executed,
        simulated=False,
    )
    audit.record(
        "order_recovered",
        symbol=symbol,
        client_order_id=order_id,
        executed_qty=str(delta),
        price=str(price),
        status=str(payload.get("status", "") or ""),
    )
    return fill


def recover_unresolved_orders(
    client: Any,
    journal: OrderJournal,
    audit: AuditLog,
    *,
    now: pd.Timestamp,
    lookback: pd.Timedelta,
    taker_fee_bps: float,
) -> RecoveryReport:
    """Query every schema-v2 journal submit recorded within `lookback` that has no terminal record, journal the executed delta as a `recovered` fill and mark closed orders terminal.

    Args: client: venue client (mutation-suppressed clients return an empty report without calls). journal: order journal. audit: audit log. now: tz-aware UTC wall clock. lookback: recovery window (`journal_recovery_lookback_hours`). taker_fee_bps: fee booked on recovered fills; the order query does not reveal maker/taker, so the conservative taker fee is used.
    Returns: RecoveryReport.
    Raises: DataIntegrityError: a venue response lacks side/executedQty/updateTime, has no non-empty status (I-RECOVERY-STATUS), or reports executed quantity with a non-positive average price; ids processed earlier in the loop keep their journal records. VenueError: non-benign venue errors propagate (the cycle HALTs; nothing is journaled for the failing id).
    """
    if _is_suppressed_client(client):
        return RecoveryReport(recovered=(), resolved_ids=(), unresolved_ids=())
    since = now - lookback
    submits = journal.unresolved_submits(since=since)
    recovered: list[JournalFill] = []
    resolved: list[str] = []
    unresolved: list[UnresolvedOrder] = []
    for submit in submits:
        order_id = submit.client_order_id
        symbol = submit.symbol
        try:
            payload = client.query_order(symbol, order_id)
        except VenueError as exc:
            if exc.code != ORDER_DOES_NOT_EXIST_CODE:
                raise
            journal.record_terminal(order_id, ORDER_NOT_FOUND_STATUS)
            audit.record(
                "order_recovery_resolved", symbol=symbol, client_order_id=order_id, status=ORDER_NOT_FOUND_STATUS
            )
            resolved.append(order_id)
            continue
        status = str(payload.get("status", "") or "")
        if not status:
            raise DataIntegrityError(f"recovery query for {order_id} lacks status")
        if status in _OPEN_ORDER_STATUSES:
            attempt_seq = getattr(submit, "attempt_seq", None)
            leg_index = getattr(submit, "leg_index", None)
            unresolved.append(
                UnresolvedOrder(
                    client_order_id=order_id,
                    symbol=symbol,
                    attempt_seq=attempt_seq,
                    leg_index=leg_index if isinstance(leg_index, int) else 0,
                )
            )
            continue
        attempt_seq = getattr(submit, "attempt_seq", None)
        leg_index = getattr(submit, "leg_index", None)
        fill = _book_recovered_delta(
            journal,
            audit,
            order_id=order_id,
            symbol=symbol,
            attempt_seq=attempt_seq,
            leg_index=leg_index if isinstance(leg_index, int) else 0,
            payload=payload,
            taker_fee_bps=taker_fee_bps,
        )
        if fill is not None:
            recovered.append(fill)
        journal.record_terminal(order_id, status)
        audit.record("order_recovery_resolved", symbol=symbol, client_order_id=order_id, status=status)
        resolved.append(order_id)
    return RecoveryReport(
        recovered=tuple(recovered),
        resolved_ids=tuple(resolved),
        unresolved_ids=tuple(o.client_order_id for o in unresolved),
        unresolved=tuple(unresolved),
    )


def settle_unresolved_orders(
    client: Any,
    journal: OrderJournal,
    audit: AuditLog,
    unresolved: Sequence[UnresolvedOrder],
    *,
    taker_fee_bps: float,
) -> UnresolvedSettlement:
    """Cancel each order recovery left open, with venue confirmation, and book what it executed.

    Why: a journaled own order still resting at the venue can fill at any time and, because client
    order ids are unique, the venue never dedupes it against new intents. Cancelling first removes
    the exposure; booking the confirmed executed delta keeps the ledger equal to venue fills even when
    the cancel cannot be confirmed. Orders that remain open are returned for the caller's per-symbol
    freeze rather than halting the cycle.

    Args:
        client: venue order client (mutation-suppressed clients return an empty settlement without calls).
        journal: order journal.
        audit: audit log.
        unresolved: orders from ``RecoveryReport.unresolved``.
        taker_fee_bps: fee booked on recovered fills (order query does not reveal maker/taker).

    Returns:
        UnresolvedSettlement; ``still_open`` keeps input order.

    Raises:
        DataIntegrityError: confirming payload lacks status, or reports an executed delta without a
            BUY/SELL side, a positive avgPrice, or a valid updateTime.
        VenueError: non-benign cancel/query rejection.
    """
    if _is_suppressed_client(client):
        return UnresolvedSettlement(recovered=(), resolved_ids=(), still_open=())
    recovered: list[JournalFill] = []
    resolved: list[str] = []
    still_open: list[UnresolvedOrder] = []
    for order in unresolved:
        confirmation = cancel_and_confirm(client, order.symbol, order.client_order_id)
        if confirmation.status == ORDER_NOT_FOUND_STATUS:
            journal.record_terminal(order.client_order_id, ORDER_NOT_FOUND_STATUS)
            audit.record(
                "order_recovery_resolved",
                symbol=order.symbol,
                client_order_id=order.client_order_id,
                status=ORDER_NOT_FOUND_STATUS,
            )
            resolved.append(order.client_order_id)
            continue
        assert confirmation.payload is not None
        fill = _book_recovered_delta(
            journal,
            audit,
            order_id=order.client_order_id,
            symbol=order.symbol,
            attempt_seq=order.attempt_seq,
            leg_index=order.leg_index,
            payload=confirmation.payload,
            taker_fee_bps=taker_fee_bps,
        )
        if fill is not None:
            recovered.append(fill)
        if confirmation.closed:
            journal.record_terminal(order.client_order_id, confirmation.status)
            audit.record(
                "order_recovery_resolved",
                symbol=order.symbol,
                client_order_id=order.client_order_id,
                status=confirmation.status,
            )
            resolved.append(order.client_order_id)
        else:
            still_open.append(order)
    return UnresolvedSettlement(
        recovered=tuple(recovered), resolved_ids=tuple(resolved), still_open=tuple(still_open)
    )
