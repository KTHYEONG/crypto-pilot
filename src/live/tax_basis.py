"""Signed-position moving-average cost basis (이동평균법) for one-way perpetual positions.

Moving average is the Korean statutory basis method for virtual assets and the only method
supported: same-direction fills re-average the entry, opposite-direction fills realize only the
closed portion at the current average, a fill larger than the position flips it and opens the
remainder at the fill price, and a flat position has no basis.

Numerics: every operation runs in a Decimal context of TAX_DECIMAL_PRECISION digits that traps
Inexact, so additions and products are exact or raise. The single declared rounding point is
re-averaging: the new average entry is quantized half-even to AVERAGE_ENTRY_QUANTUM, far below
any venue price tick, which keeps every later product exact and bounded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Context, Decimal, Inexact, localcontext
from typing import Final, Literal

import pandas as pd

from src.common.errors import DataIntegrityError
from src.live.tax_schema import (
    DELIVERY_SETTLEMENT_INCOME_TYPE,
    TaxRecord,
    tax_event_sort_key,
    validate_tax_record,
)

COST_BASIS_MOVING_AVERAGE: Final[str] = "moving_average"
TAX_DECIMAL_PRECISION: Final[int] = 60
AVERAGE_ENTRY_QUANTUM: Final[Decimal] = Decimal("1E-18")


def require_moving_average(cost_basis: str) -> None:
    """Reject any cost-basis method other than moving average.

    Raises:
        ValueError: ``cost_basis`` != COST_BASIS_MOVING_AVERAGE (FIFO and every other method are
            removed; a different statutory method needs a new spec, not a flag).
    """
    if cost_basis != COST_BASIS_MOVING_AVERAGE:
        raise ValueError(
            f"unsupported cost basis {cost_basis!r}: only {COST_BASIS_MOVING_AVERAGE!r} is supported"
        )


@dataclass(frozen=True, slots=True)
class AverageCostPosition:
    """Signed position under moving-average basis.

    quantity: signed size (> 0 long, < 0 short, 0 flat).
    avg_entry: average entry price; exactly Decimal(0) when flat, > 0 otherwise.
    opened_at: event time of the fill that opened the current non-flat position (from flat or by a
        flip); None when flat. Re-averaging and partial closes keep it.
    """

    quantity: Decimal
    avg_entry: Decimal
    opened_at: pd.Timestamp | None


FLAT_POSITION: Final[AverageCostPosition] = AverageCostPosition(Decimal(0), Decimal(0), None)


@dataclass(frozen=True, slots=True)
class FillRealization:
    """Effect of one fill (or delivery settlement) on a symbol's moving-average position.

    closed_quantity: absolute size closed against the prior position (0 when the fill only opens
        or adds). closed_direction is the prior position's direction when closed_quantity > 0,
        else None.
    entry_notional_closed: prior avg_entry * closed_quantity.
    exit_notional_closed: fill price * closed_quantity (settlement: derived so realized_pnl holds).
    realized_pnl: gross realized P&L of the closed portion, before fees and funding:
        exit - entry for a long close, entry - exit for a short close.
    opened_quantity: absolute size that opened or increased exposure (fill quantity minus
        closed_quantity).
    """

    record_id: str
    symbol: str
    event_time: pd.Timestamp
    closed_quantity: Decimal
    closed_direction: Literal["long", "short"] | None
    entry_notional_closed: Decimal
    exit_notional_closed: Decimal
    realized_pnl: Decimal
    opened_quantity: Decimal
    position_before: AverageCostPosition
    position_after: AverageCostPosition


def _exact_mul(left: Decimal, right: Decimal) -> Decimal:
    with localcontext(prec=TAX_DECIMAL_PRECISION) as ctx:
        ctx.traps[Inexact] = True
        return left * right


def _exact_add(left: Decimal, right: Decimal) -> Decimal:
    with localcontext(prec=TAX_DECIMAL_PRECISION) as ctx:
        ctx.traps[Inexact] = True
        return left + right


def _reaveraged_entry(total_cost: Decimal, total_quantity: Decimal) -> Decimal:
    # Integer ratios avoid double rounding of a repeating quotient at a half-even tie.
    cost_num, cost_den = total_cost.as_integer_ratio()
    quantity_num, quantity_den = total_quantity.as_integer_ratio()
    quantum_num, quantum_den = AVERAGE_ENTRY_QUANTUM.as_integer_ratio()
    numerator = cost_num * quantity_den * quantum_den
    denominator = cost_den * quantity_num * quantum_num
    units, remainder = divmod(numerator, denominator)
    if remainder * 2 > denominator or (remainder * 2 == denominator and units % 2):
        units += 1
    average = _exact_mul(Decimal(units), AVERAGE_ENTRY_QUANTUM)
    if average <= 0:
        raise DataIntegrityError("moving-average avg_entry rounds to zero at the supported quantum")
    return average


def apply_fill(position: AverageCostPosition, record: TaxRecord) -> FillRealization:
    """Apply one TRADE record to a moving-average position.

    Raises:
        ValueError: ``record.kind`` != "TRADE".
        DataIntegrityError: ``record`` fails ``validate_tax_record`` or re-averaging would
            produce a zero entry price for a non-flat position.
        decimal.Inexact: an exact operation would round (precision exhausted); never swallowed.
    """
    with localcontext(Context(prec=TAX_DECIMAL_PRECISION, rounding=ROUND_HALF_EVEN)) as ctx:
        ctx.traps[Inexact] = True
        return _apply_fill(position, record)


def _apply_fill(position: AverageCostPosition, record: TaxRecord) -> FillRealization:
    if record.kind != "TRADE":
        raise ValueError(f"apply_fill requires kind 'TRADE', got {record.kind!r}")
    validate_tax_record(record)
    signed = record.quantity if record.side == "BUY" else -record.quantity
    before_qty = position.quantity
    after_qty = _exact_add(before_qty, signed)
    same_direction = before_qty == 0 or (before_qty > 0) == (signed > 0)
    if same_direction:
        held = abs(before_qty)
        total_cost = _exact_add(_exact_mul(held, position.avg_entry), _exact_mul(record.quantity, record.price))
        total_held = _exact_add(held, record.quantity)
        if before_qty == 0:
            avg_after = record.price
            opened_at = record.event_time
        else:
            avg_after = _reaveraged_entry(total_cost, total_held)
            opened_at = position.opened_at
        return FillRealization(
            record_id=record.record_id,
            symbol=record.symbol,
            event_time=record.event_time,
            closed_quantity=Decimal(0),
            closed_direction=None,
            entry_notional_closed=Decimal(0),
            exit_notional_closed=Decimal(0),
            realized_pnl=Decimal(0),
            opened_quantity=record.quantity,
            position_before=position,
            position_after=AverageCostPosition(after_qty, avg_after, opened_at),
        )
    held = abs(before_qty)
    closed = held if held < record.quantity else record.quantity
    opened = _exact_add(record.quantity, -closed)
    direction: Literal["long", "short"] = "long" if before_qty > 0 else "short"
    entry_closed = _exact_mul(position.avg_entry, closed)
    exit_closed = _exact_mul(record.price, closed)
    realized = _exact_add(exit_closed, -entry_closed) if direction == "long" else _exact_add(entry_closed, -exit_closed)
    if closed == held:
        after = FLAT_POSITION if opened == 0 else AverageCostPosition(after_qty, record.price, record.event_time)
    else:
        after = AverageCostPosition(after_qty, position.avg_entry, position.opened_at)
    return FillRealization(
        record_id=record.record_id,
        symbol=record.symbol,
        event_time=record.event_time,
        closed_quantity=closed,
        closed_direction=direction,
        entry_notional_closed=entry_closed,
        exit_notional_closed=exit_closed,
        realized_pnl=realized,
        opened_quantity=opened,
        position_before=position,
        position_after=after,
    )


def apply_settlement_close(position: AverageCostPosition, record: TaxRecord) -> FillRealization:
    """Close the whole position at a venue delivery settlement (delisted perpetual).

    The venue books delivery as a DELIVERED_SETTELMENT income row (kind REALIZED_PNL) without a
    userTrades fill, so the fold would otherwise keep a phantom position forever. The venue amount
    is authoritative: realized_pnl == record.realized_pnl and exit_notional_closed is derived as
    entry + realized (long) or entry - realized (short).

    Raises:
        ValueError: record is not kind REALIZED_PNL with income_type DELIVERY_SETTLEMENT_INCOME_TYPE.
        DataIntegrityError: the position is flat (settlement without a known position means the
            ledger history is incomplete) or record fails validate_tax_record.
        decimal.Inexact: an exact operation would round (precision exhausted); never swallowed.
    """
    with localcontext(Context(prec=TAX_DECIMAL_PRECISION, rounding=ROUND_HALF_EVEN)) as ctx:
        ctx.traps[Inexact] = True
        return _apply_settlement_close(position, record)


def _apply_settlement_close(position: AverageCostPosition, record: TaxRecord) -> FillRealization:
    if record.kind != "REALIZED_PNL" or record.income_type != DELIVERY_SETTLEMENT_INCOME_TYPE:
        raise ValueError(
            "apply_settlement_close requires kind 'REALIZED_PNL' with "
            f"income_type {DELIVERY_SETTLEMENT_INCOME_TYPE!r}, got {record.kind!r}/{record.income_type!r}"
        )
    validate_tax_record(record)
    if position.quantity == 0:
        raise DataIntegrityError(
            f"settlement {record.record_id!r} closes no position: ledger history is incomplete"
        )
    direction: Literal["long", "short"] = "long" if position.quantity > 0 else "short"
    closed = abs(position.quantity)
    entry_closed = _exact_mul(position.avg_entry, closed)
    if direction == "long":
        exit_closed = _exact_add(entry_closed, record.realized_pnl)
    else:
        exit_closed = _exact_add(entry_closed, -record.realized_pnl)
    return FillRealization(
        record_id=record.record_id,
        symbol=record.symbol,
        event_time=record.event_time,
        closed_quantity=closed,
        closed_direction=direction,
        entry_notional_closed=entry_closed,
        exit_notional_closed=exit_closed,
        realized_pnl=record.realized_pnl,
        opened_quantity=Decimal(0),
        position_before=position,
        position_after=FLAT_POSITION,
    )


def fold_symbol(
    records: Sequence[TaxRecord],
    *,
    initial: AverageCostPosition = FLAT_POSITION,
) -> tuple[FillRealization, ...]:
    """Fold one symbol's fills and delivery settlements chronologically from ``initial``.

    Input order is irrelevant: records are ordered by ``tax_event_sort_key``. Each step's
    position_before is the previous step's position_after, so the result is a complete, causal
    audit trail: realizations at or before time T depend only on records at or before T.

    Raises:
        ValueError: records span more than one symbol, or contain a kind other than TRADE and
            delivery-settlement REALIZED_PNL.
        DataIntegrityError: a record_id repeats, or a step raises it.
    """
    ordered = sorted(records, key=tax_event_sort_key)
    symbols = {record.symbol for record in ordered}
    if len(symbols) > 1:
        raise ValueError(f"fold_symbol requires one symbol, got {sorted(symbols)}")
    seen: set[str] = set()
    position = initial
    out: list[FillRealization] = []
    for record in ordered:
        if record.record_id in seen:
            raise DataIntegrityError(f"duplicate record_id in fold: {record.record_id!r}")
        seen.add(record.record_id)
        if record.kind == "TRADE":
            step = apply_fill(position, record)
        elif record.kind == "REALIZED_PNL" and record.income_type == DELIVERY_SETTLEMENT_INCOME_TYPE:
            step = apply_settlement_close(position, record)
        else:
            raise ValueError(
                f"fold_symbol rejects kind {record.kind!r} (income_type {record.income_type!r}): "
                "callers fold only TRADE and delivery-settlement rows"
            )
        position = step.position_after
        out.append(step)
    return tuple(out)
