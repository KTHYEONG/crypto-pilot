"""Tax-ledger record schema: Decimal fact type, fail-closed validation, shard rows.

The ledger is a jurisdiction-agnostic FACTUAL record. No tax rate, deduction, or income
classification is computed here; this module only guarantees that every persisted row is a
well-typed fact that any downstream classification can rely on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Final

import pandas as pd

from src.common.errors import DataIntegrityError
from src.live.settings import ExecutionMode

TAX_RECORD_KINDS: Final[frozenset[str]] = frozenset(
    {"TRADE", "REALIZED_PNL", "FUNDING_FEE", "COMMISSION", "TRANSFER", "UNCLASSIFIED"}
)

TRADE_SIDES: Final[frozenset[str]] = frozenset({"BUY", "SELL"})

ONE_WAY_POSITION_SIDE: Final[str] = "BOTH"

DELIVERY_SETTLEMENT_INCOME_TYPE: Final[str] = "DELIVERED_SETTELMENT"

TAX_SOURCE_MODES: Final[Mapping[str, frozenset[str]]] = {
    "venue": frozenset({ExecutionMode.LIVE_TESTNET.value, ExecutionMode.LIVE_MAINNET.value}),
    "simulated": frozenset({ExecutionMode.PAPER.value, ExecutionMode.SHADOW.value}),
    "delisting_settlement": frozenset({ExecutionMode.PAPER.value, ExecutionMode.SHADOW.value}),
}

TAX_DECIMAL_FIELDS: Final[tuple[str, ...]] = ("quantity", "price", "quote_qty", "fee", "realized_pnl")

TAX_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "record_id",
        "kind",
        "event_time",
        "symbol",
        "side",
        "quantity",
        "price",
        "quote_qty",
        "fee",
        "fee_asset",
        "realized_pnl",
        "income_asset",
        "is_maker",
        "venue_id",
        "source",
        "mode",
        "income_type",
        "position_side",
    }
)

LEGACY_OPTIONAL_ROW_KEYS: Final[frozenset[str]] = frozenset({"income_type", "position_side"})

_TAX_ROW_STR_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "record_id",
        "kind",
        "symbol",
        "side",
        "fee_asset",
        "income_asset",
        "source",
        "mode",
        "income_type",
        "position_side",
    }
)

_INCOME_KINDS: Final[frozenset[str]] = frozenset(
    {"REALIZED_PNL", "FUNDING_FEE", "COMMISSION", "TRANSFER", "UNCLASSIFIED"}
)

_SYMBOL_REQUIRED_VENUE_KINDS: Final[frozenset[str]] = frozenset(
    {"REALIZED_PNL", "FUNDING_FEE", "COMMISSION"}
)


@dataclass(frozen=True, slots=True)
class TaxRecord:
    """One immutable tax-ledger fact: a fill (kind TRADE) or a cash-flow event (every other kind).

    Amounts are Decimal end-to-end. Venue strings and the Decimal paper ledgers are carried without
    a binary-float round trip so that yearly sums reconcile exactly and a re-read shard reproduces
    the written values bit for bit.

    Field semantics:
        quantity: TRADE -> absolute fill size (> 0; ``side`` carries direction). Simulated
            FUNDING_FEE -> signed position size held at the funding epoch. Venue income -> 0.
        price: TRADE -> fill price (> 0). Simulated FUNDING_FEE -> mark used for the accrual.
        quote_qty: TRADE -> venue quoteQty / simulated quantity * price. Never cash for
            FUNDING_FEE rows (position notional).
        fee: TRADE commission in ``fee_asset`` (venue sign preserved). 0 for income kinds.
        realized_pnl: venue TRADE -> venue per-fill realized P&L (average-entry, one-way mode);
            simulated TRADE -> always 0 (the fold is the P&L source); every income kind -> the
            signed cash amount in ``income_asset``.
        income_asset: asset that ``realized_pnl`` is denominated in.
        event_time: tz-aware UTC instant. Tax-year bucketing happens downstream in the configured
            tax timezone, never on this field's UTC calendar.
        income_type: raw venue incomeType for venue income rows; "" otherwise.
        position_side: venue positionSide for venue TRADE rows (must be "BOTH": one-way mode is
            the only supported accounting model); "" for every other row.

    Raises:
        TypeError: a field in TAX_DECIMAL_FIELDS is not a ``Decimal`` (a ``float`` would inject
            binary rounding), ``is_maker`` is not a ``bool``, ``venue_id`` is not an ``int`` (or is
            a ``bool``), or ``event_time`` is not a tz-aware ``pd.Timestamp``.
    """

    record_id: str
    kind: str
    event_time: pd.Timestamp
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    quote_qty: Decimal
    fee: Decimal
    fee_asset: str
    realized_pnl: Decimal
    income_asset: str
    is_maker: bool
    venue_id: int
    source: str
    mode: str
    income_type: str = ""
    position_side: str = ""

    def __post_init__(self) -> None:
        for field in TAX_DECIMAL_FIELDS:
            if not isinstance(getattr(self, field), Decimal):
                raise TypeError(
                    f"TaxRecord field {field!r} must be Decimal, got {type(getattr(self, field)).__name__}"
                )
        if type(self.is_maker) is not bool:
            raise TypeError(f"TaxRecord field 'is_maker' must be bool, got {type(self.is_maker).__name__}")
        if isinstance(self.venue_id, bool) or not isinstance(self.venue_id, int):
            raise TypeError(
                f"TaxRecord field 'venue_id' must be int, got {type(self.venue_id).__name__}"
            )
        if not isinstance(self.event_time, pd.Timestamp) or self.event_time.tzinfo is None:
            raise TypeError("TaxRecord field 'event_time' must be a tz-aware pd.Timestamp")


def parse_tax_decimal(value: object, *, field: str) -> Decimal:
    """Convert one persisted or venue numeric to a finite Decimal without a float round trip.

    Accepted: ``Decimal`` (as produced by ``json.loads(parse_float=Decimal)``), ``int`` (not
    ``bool``), or a ``str`` holding a finite decimal literal. Legacy shards store JSON numbers;
    parsing them with ``parse_float=Decimal`` preserves their exact text, so legacy and new rows
    load identically.

    Raises:
        DataIntegrityError: ``value`` is None, a bool, a float, any other type, an empty or
            non-numeric string, or a non-finite value (NaN, sNaN, Infinity). The message names
            ``field``.
    """
    if isinstance(value, bool):
        raise DataIntegrityError(f"tax field {field!r} is a bool, not a decimal")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise DataIntegrityError(f"tax field {field!r} is non-finite: {value!r}")
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        if not value or value.strip() != value:
            raise DataIntegrityError(f"tax field {field!r} is not a decimal literal: {value!r}")
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:
            raise DataIntegrityError(
                f"tax field {field!r} is not a decimal literal: {value!r}"
            ) from exc
        if not parsed.is_finite():
            raise DataIntegrityError(f"tax field {field!r} is non-finite: {value!r}")
        return parsed
    raise DataIntegrityError(
        f"tax field {field!r} has unsupported type {type(value).__name__}"
    )


def _fail(record_id: object, field: str, reason: str) -> DataIntegrityError:
    return DataIntegrityError(f"tax record {record_id!r} has invalid {field!r}: {reason}")


def validate_tax_record(record: TaxRecord) -> None:
    """Enforce the semantic schema every persisted or loaded tax record must satisfy.

    Called before any shard write and on every loaded row, so a row the summary would reject
    can never be written and a corrupt row can never be summarized as zero.

    Raises:
        DataIntegrityError: any rule below is violated; the message names record_id and field.

    Rules:
        all rows: non-empty record_id; kind in TAX_RECORD_KINDS; source in TAX_SOURCE_MODES and
            mode in TAX_SOURCE_MODES[source]; every Decimal field finite.
        TRADE: non-empty symbol; side in TRADE_SIDES; quantity > 0; price > 0; quote_qty >= 0;
            non-empty income_asset; non-empty fee_asset whenever fee != 0; income_type == "".
            source "venue": position_side == "BOTH".
            source "simulated" / "delisting_settlement": position_side == "" and
            realized_pnl == 0 (simulated fills carry no venue P&L; the fold is their P&L source).
        income kinds (REALIZED_PNL, FUNDING_FEE, COMMISSION, TRANSFER, UNCLASSIFIED):
            side == ""; position_side == ""; non-empty income_asset.
            source "venue": non-empty income_type; non-empty symbol for REALIZED_PNL,
            FUNDING_FEE and COMMISSION (account-level rows such as transfers may be "").
            FUNDING_FEE (any source): non-empty symbol.
    """
    rid = getattr(record, "record_id", None)
    if not isinstance(rid, str) or not rid:
        raise _fail(rid, "record_id", "must be a non-empty string")
    for field in sorted(_TAX_ROW_STR_FIELDS):
        if not isinstance(getattr(record, field), str):
            raise _fail(rid, field, "must be a string")
    if record.kind not in TAX_RECORD_KINDS:
        raise _fail(rid, "kind", f"must be one of {sorted(TAX_RECORD_KINDS)}")
    if record.source not in TAX_SOURCE_MODES:
        raise _fail(rid, "source", f"must be one of {sorted(TAX_SOURCE_MODES)}")
    if record.mode not in TAX_SOURCE_MODES[record.source]:
        raise _fail(rid, "mode", f"{record.mode!r} cannot produce source {record.source!r}")
    for field in TAX_DECIMAL_FIELDS:
        amount = getattr(record, field)
        if not isinstance(amount, Decimal) or not amount.is_finite():
            raise _fail(rid, field, "must be a finite Decimal")
    if record.kind == "TRADE":
        _validate_trade(record, rid)
    else:
        _validate_income(record, rid)


def _validate_trade(record: TaxRecord, rid: str) -> None:
    if not isinstance(record.symbol, str) or not record.symbol:
        raise _fail(rid, "symbol", "TRADE rows must name a symbol")
    if record.side not in TRADE_SIDES:
        raise _fail(rid, "side", f"must be one of {sorted(TRADE_SIDES)}")
    if record.quantity <= 0:
        raise _fail(rid, "quantity", "TRADE quantity must be > 0")
    if record.price <= 0:
        raise _fail(rid, "price", "TRADE price must be > 0")
    if record.quote_qty < 0:
        raise _fail(rid, "quote_qty", "TRADE quote_qty must be >= 0")
    if not isinstance(record.income_asset, str) or not record.income_asset:
        raise _fail(rid, "income_asset", "must be a non-empty string")
    if record.fee != 0 and (not isinstance(record.fee_asset, str) or not record.fee_asset):
        raise _fail(rid, "fee_asset", "must be non-empty whenever fee != 0")
    if record.income_type != "":
        raise _fail(rid, "income_type", "TRADE rows carry no venue incomeType")
    if record.source == "venue":
        if record.position_side != ONE_WAY_POSITION_SIDE:
            raise _fail(rid, "position_side", f"venue TRADE rows must be {ONE_WAY_POSITION_SIDE!r}")
    elif record.source in ("simulated", "delisting_settlement"):
        if record.position_side != "":
            raise _fail(rid, "position_side", "simulated TRADE rows carry no positionSide")
        if record.realized_pnl != 0:
            raise _fail(rid, "realized_pnl", "simulated TRADE rows carry no venue P&L")


def _validate_income(record: TaxRecord, rid: str) -> None:
    if record.kind not in _INCOME_KINDS:  # pragma: no cover - guarded by validate_tax_record
        raise _fail(rid, "kind", f"must be one of {sorted(TAX_RECORD_KINDS)}")
    if record.side != "":
        raise _fail(rid, "side", "income rows carry no side")
    if record.position_side != "":
        raise _fail(rid, "position_side", "income rows carry no positionSide")
    if not isinstance(record.income_asset, str) or not record.income_asset:
        raise _fail(rid, "income_asset", "must be a non-empty string")
    if record.source == "venue" and (
        not isinstance(record.income_type, str) or not record.income_type
    ):
        raise _fail(rid, "income_type", "venue income rows must carry the raw incomeType")
    if (
        record.source == "venue"
        and record.kind in _SYMBOL_REQUIRED_VENUE_KINDS
        and (not isinstance(record.symbol, str) or not record.symbol)
    ):
        raise _fail(rid, "symbol", f"venue {record.kind} rows must name a symbol")
    if record.kind == "FUNDING_FEE" and (
        not isinstance(record.symbol, str) or not record.symbol
    ):
        raise _fail(rid, "symbol", "FUNDING_FEE rows must name a symbol")


def tax_record_from_row(row: Mapping[str, Any]) -> TaxRecord:
    """Rebuild and validate one TaxRecord from a decoded shard row.

    Accepts both the current format (Decimal amounts as JSON strings, ``position_side`` present)
    and the legacy format (amounts as JSON numbers, no ``income_type``/``position_side`` keys,
    which default to ""). The row must have been decoded with ``parse_float=Decimal``.

    Raises:
        DataIntegrityError: a key outside TAX_ROW_KEYS is present; a required key is missing;
            a string field is not a ``str``; ``is_maker`` is not a JSON boolean; ``venue_id`` is
            not a JSON integer; ``event_time`` is not an ISO-8601 string with an explicit UTC
            offset; a numeric fails ``parse_tax_decimal``; or ``validate_tax_record`` fails.
    """
    if not isinstance(row, Mapping):
        raise DataIntegrityError(f"tax row must be an object, got {type(row).__name__}")
    unknown = set(row) - set(TAX_ROW_KEYS)
    if unknown:
        raise DataIntegrityError(f"tax row has unknown keys: {sorted(unknown)}")
    required = set(TAX_ROW_KEYS) - set(LEGACY_OPTIONAL_ROW_KEYS)
    missing = required - set(row)
    if missing:
        raise DataIntegrityError(f"tax row is missing keys: {sorted(missing)}")
    for field in sorted(_TAX_ROW_STR_FIELDS):
        if field in LEGACY_OPTIONAL_ROW_KEYS and field not in row:
            continue
        if not isinstance(row[field], str):
            raise DataIntegrityError(
                f"tax row field {field!r} must be a string, got {type(row[field]).__name__}"
            )
    if type(row["is_maker"]) is not bool:
        raise DataIntegrityError("tax row field 'is_maker' must be a JSON boolean")
    if isinstance(row["venue_id"], bool) or not isinstance(row["venue_id"], int):
        raise DataIntegrityError("tax row field 'venue_id' must be a JSON integer")
    raw_time = row["event_time"]
    if not isinstance(raw_time, str):
        raise DataIntegrityError("tax row field 'event_time' must be an ISO-8601 string")
    try:
        event_time = pd.Timestamp(raw_time)
    except (TypeError, ValueError) as exc:
        raise DataIntegrityError(f"tax row field 'event_time' is not ISO-8601: {raw_time!r}") from exc
    if event_time.tzinfo is None:
        raise DataIntegrityError(
            f"tax row field 'event_time' lacks an explicit UTC offset: {raw_time!r}"
        )
    event_time = event_time.tz_convert("UTC")
    record = TaxRecord(
        record_id=row["record_id"],
        kind=row["kind"],
        event_time=event_time,
        symbol=row["symbol"],
        side=row["side"],
        quantity=parse_tax_decimal(row["quantity"], field="quantity"),
        price=parse_tax_decimal(row["price"], field="price"),
        quote_qty=parse_tax_decimal(row["quote_qty"], field="quote_qty"),
        fee=parse_tax_decimal(row["fee"], field="fee"),
        fee_asset=row["fee_asset"],
        realized_pnl=parse_tax_decimal(row["realized_pnl"], field="realized_pnl"),
        income_asset=row["income_asset"],
        is_maker=row["is_maker"],
        venue_id=row["venue_id"],
        source=row["source"],
        mode=row["mode"],
        income_type=row.get("income_type", ""),
        position_side=row.get("position_side", ""),
    )
    validate_tax_record(record)
    return record


def tax_record_to_row(record: TaxRecord) -> dict[str, Any]:
    """Serialize one record to the shard row format.

    Decimal fields are written as fixed-point decimal strings (``format(value, "f")``) so the
    shard text is the exact value; ``event_time`` is ISO-8601 UTC. Inverse of
    ``tax_record_from_row``: ``tax_record_from_row(tax_record_to_row(r)) == r``.
    """
    return {
        "record_id": record.record_id,
        "kind": record.kind,
        "event_time": record.event_time.tz_convert("UTC").isoformat(),
        "symbol": record.symbol,
        "side": record.side,
        "quantity": format(record.quantity, "f"),
        "price": format(record.price, "f"),
        "quote_qty": format(record.quote_qty, "f"),
        "fee": format(record.fee, "f"),
        "fee_asset": record.fee_asset,
        "realized_pnl": format(record.realized_pnl, "f"),
        "income_asset": record.income_asset,
        "is_maker": record.is_maker,
        "venue_id": record.venue_id,
        "source": record.source,
        "mode": record.mode,
        "income_type": record.income_type,
        "position_side": record.position_side,
    }


def tax_event_sort_key(record: TaxRecord) -> tuple[pd.Timestamp, int, str]:
    """Chronological order key: (event_time, venue_id, record_id).

    venue_id (venue trade id / journal fill sequence) breaks same-millisecond ties in execution
    order; lexical record_id order would place "venue:TRADE:10" before "venue:TRADE:9".
    """
    return (record.event_time, record.venue_id, record.record_id)
