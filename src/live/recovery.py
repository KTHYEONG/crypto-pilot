"""Restart recovery of fills that the venue executed but the journal never observed (process killed between a fill and the next poll). Runs before reconciliation so the ledger reflects every venue execution of our own orders."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.live.audit import AuditLog
from src.live.errors import VenueError
from src.live.order_journal import OrderJournal

if TYPE_CHECKING:
    from src.live.order_journal import JournalFill

#: Venue statuses that mean the order may still fill; left for the open-order sweep.
_OPEN_ORDER_STATUSES: frozenset[str] = frozenset({"NEW", "PARTIALLY_FILLED"})

@dataclass(frozen=True, slots=True)
class RecoveryReport:
    recovered: tuple[JournalFill, ...]
    resolved_ids: tuple[str, ...]  # ids journaled terminal by this recovery
    unresolved_ids: tuple[str, ...]  # still open on the venue (left to cancel_orphan_orders) or query failed


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
    Raises: DataIntegrityError: a venue response lacks side/executedQty/updateTime or reports executed quantity with a non-positive average price. VenueError: non-benign venue errors propagate (the cycle HALTs; nothing is journaled for the failing id).
    """
    if _is_suppressed_client(client):
        return RecoveryReport(recovered=(), resolved_ids=(), unresolved_ids=())
    since = now - lookback
    submits = journal.unresolved_submits(since=since)
    recovered: list[JournalFill] = []
    resolved: list[str] = []
    unresolved: list[str] = []
    for submit in submits:
        order_id = submit.client_order_id
        symbol = submit.symbol
        try:
            payload = client.query_order(symbol, order_id)
        except VenueError as exc:
            if exc.code != -2013:
                raise
            journal.record_terminal(order_id, "NOT_FOUND")
            audit.record(
                "order_recovery_resolved", symbol=symbol, client_order_id=order_id, status="NOT_FOUND"
            )
            resolved.append(order_id)
            continue
        status = str(payload.get("status", "") or "")
        # 상태 미상(빈 응답)·미체결 주문은 판단 불가 → 오픈 주문 스윕에 맡긴다.
        if not status or status in _OPEN_ORDER_STATUSES:
            unresolved.append(order_id)
            continue
        executed = _venue_decimal(payload, "executedQty", order_id)
        observed = journal.observed_qty(order_id)
        delta = executed - observed
        if delta <= 0:
            journal.record_terminal(order_id, status)
            audit.record("order_recovery_resolved", symbol=symbol, client_order_id=order_id, status=status)
            resolved.append(order_id)
            continue
        side = payload.get("side", None)
        if side not in ("BUY", "SELL"):
            raise DataIntegrityError(f"recovery query for {order_id} lacks side")
        price = _venue_decimal(payload, "avgPrice", order_id)
        if price <= 0:
            raise DataIntegrityError(f"recovery query for {order_id} reports fills with no positive avgPrice")
        filled_at = _venue_update_time(payload, order_id)
        attempt_seq = getattr(submit, "attempt_seq", None)
        leg_index = getattr(submit, "leg_index", None)
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
            leg_index=leg_index if isinstance(leg_index, int) else 0,
            cumulative_executed_qty=executed,
            simulated=False,
        )
        audit.record(
            "order_recovered",
            symbol=symbol,
            client_order_id=order_id,
            executed_qty=str(delta),
            price=str(price),
            status=status,
        )
        journal.record_terminal(order_id, status)
        audit.record("order_recovery_resolved", symbol=symbol, client_order_id=order_id, status=status)
        recovered.append(fill)
        resolved.append(order_id)
    return RecoveryReport(recovered=tuple(recovered), resolved_ids=tuple(resolved), unresolved_ids=tuple(unresolved))
