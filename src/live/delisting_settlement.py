"""Booking of delivered (delisted) perpetual positions into the live ledger.

LIVE trusts the venue's flat position as settlement proof and books no cash (the venue books it);
PAPER books settlement cash only from flat 1h-kline evidence, so an unproven delivery never invents a price.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.core.settlement_evidence import SettlementEvidence
from src.live.account import settled_delisting_symbols
from src.live.filters import is_delisted, parse_delivery_schedule
from src.live.ledger import LedgerState, append_position_snapshot
from src.live.settings import ExecutionMode
from src.live.tax_schema import TaxRecord


@dataclass(frozen=True, slots=True)
class DelistingSettlement:
    """One booked settlement of a delivered perpetual position."""

    symbol: str
    quantity: Decimal  # signed ledger quantity that was settled
    price: Decimal | None  # settlement price (PAPER); None in LIVE where the venue books cash
    fee: Decimal  # PAPER settlement fee in USDT; 0 in LIVE
    delivery_time: pd.Timestamp
    evidence_source: str  # "flat_1h_klines" (PAPER) | "venue_flat_position" (LIVE)


def book_delisting_settlements(
    state: LedgerState,
    *,
    mode: ExecutionMode,
    exchange_info: Mapping[str, Any],
    venue_positions: Mapping[str, Decimal],
    evidence: Mapping[str, SettlementEvidence],
    fee_bps: Decimal,
    now: pd.Timestamp,
) -> tuple[LedgerState, tuple[DelistingSettlement, ...]]:
    """Zero ledger positions the venue has settled at delivery, and book the PAPER settlement cash.

    LIVE: a nonzero ledger quantity is settled when the venue position is flat, the listing
    status is SETTLING/CLOSE, and delivery has passed (``settled_delisting_symbols``). The
    ledger quantity is zeroed. Cash is venue-owned in LIVE, so none is booked here. PAPER: a
    held quantity past delivery is settled only when ``evidence`` holds a venue-evidenced
    settlement price. Cash moves by ``quantity x price`` minus the fee, like a closing fill at
    the settlement price. Without evidence the position stays unresolved and the caller fails
    closed. A settlement appends a position-history snapshot at ``now``, so funding accrual
    stops at delivery. SHADOW books nothing. Pure: audit and alert emission belong to the
    caller, which has the ``AuditLog`` and settings.

    Raises:
        DataIntegrityError: ``now`` is naive, ``fee_bps`` is negative, or PAPER cash_usdt is
            None while a settlement must be booked.
    """
    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is None:
        raise DataIntegrityError("delisting settlement now must be tz-aware")
    now_ts = now_ts.tz_convert("UTC")
    fee_rate = Decimal(str(fee_bps))
    if fee_rate < 0:
        raise DataIntegrityError("delisting settlement fee_bps must be >= 0")
    if mode == ExecutionMode.SHADOW:
        return state, ()
    schedule = parse_delivery_schedule(exchange_info)
    if mode == ExecutionMode.PAPER:
        candidates: list[tuple[str, Decimal, SettlementEvidence]] = []
        for symbol in sorted(state.positions):
            qty = state.positions[symbol]
            if qty == 0:
                continue
            info = schedule.get(symbol)
            if info is None or info.delivery_time is None:
                continue
            if not is_delisted(info, now_ts):
                continue
            proven = evidence.get(symbol)
            if proven is None:
                continue
            candidates.append((symbol, qty, proven))
        if not candidates:
            return state, ()
        if state.cash_usdt is None:
            raise DataIntegrityError("paper delisting settlement requires cash_usdt")
        positions = dict(state.positions)
        cash = state.cash_usdt
        booked: list[DelistingSettlement] = []
        for symbol, qty, proven in candidates:
            price = Decimal(proven.price)
            fee = abs(qty * price) * fee_rate / Decimal(10_000)
            cash = cash + qty * price - fee
            positions[symbol] = Decimal(0)
            delivery = pd.Timestamp(proven.delivery_time).tz_convert("UTC")
            booked.append(
                DelistingSettlement(
                    symbol=symbol,
                    quantity=qty,
                    price=price,
                    fee=fee,
                    delivery_time=delivery,
                    evidence_source="flat_1h_klines",
                )
            )
        history = append_position_snapshot(
            state.position_history, now_ts, positions, watermarks=state.funding_watermarks
        )
        return (
            dataclasses.replace(state, positions=positions, cash_usdt=cash, position_history=history),
            tuple(booked),
        )
    settled = settled_delisting_symbols(exchange_info, venue_positions, state.positions, now=now_ts)
    if not settled:
        return state, ()
    live_positions = dict(state.positions)
    live_booked: list[DelistingSettlement] = []
    for symbol in settled:
        qty = live_positions.get(symbol, Decimal(0))
        info = schedule.get(symbol)
        delivery = (
            pd.Timestamp(info.delivery_time).tz_convert("UTC")
            if info is not None and info.delivery_time is not None
            else now_ts
        )
        live_positions[symbol] = Decimal(0)
        live_booked.append(
            DelistingSettlement(
                symbol=symbol,
                quantity=qty,
                price=None,
                fee=Decimal(0),
                delivery_time=delivery,
                evidence_source="venue_flat_position",
            )
        )
    live_history = append_position_snapshot(
        state.position_history, now_ts, live_positions, watermarks=state.funding_watermarks
    )
    return (
        dataclasses.replace(state, positions=live_positions, position_history=live_history),
        tuple(live_booked),
    )


def delisting_settlement_tax_records(
    settlements: Sequence[DelistingSettlement], *, mode: str
) -> tuple[TaxRecord, ...]:
    """One TRADE TaxRecord per evidenced (PAPER) settlement for the cycle cash reconciliation.

    LIVE settlements carry no price (the venue owns the cash), so they emit no record. The
    record id is deterministic per symbol and delivery instant, so a retried cycle never
    double-books.
    """
    records: list[TaxRecord] = []
    for settlement in settlements:
        if settlement.price is None:
            continue
        qty = abs(settlement.quantity)
        price = settlement.price
        notional = abs(settlement.quantity * price)
        delivery = pd.Timestamp(settlement.delivery_time).tz_convert("UTC")
        delivery_ms = int(delivery.value // 1_000_000)
        records.append(
            TaxRecord(
                record_id=f"simulated:DELISTING_SETTLEMENT:{settlement.symbol}:{delivery_ms}",
                kind="TRADE",
                event_time=delivery,
                symbol=settlement.symbol,
                side="SELL" if settlement.quantity > 0 else "BUY",
                quantity=qty,
                price=price,
                quote_qty=notional,
                fee=settlement.fee,
                fee_asset="USDT",
                realized_pnl=Decimal(0),
                income_asset="USDT",
                is_maker=False,
                venue_id=0,
                source="delisting_settlement",
                mode=str(mode),
            )
        )
    return tuple(records)
