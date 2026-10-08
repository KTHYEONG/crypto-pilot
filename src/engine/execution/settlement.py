"""Evidenced delisting-settlement producer owned by the execution core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.core.instrument_settlements import (
    InstrumentSettlementRecord,
    InstrumentSettlementRegistry,
    record_digest,
)
from src.core.types import ExecutionSpec
from src.core.venue_halts import VenueHaltInterval

from .contracts import InstrumentSettlementEvent


def venue_halt_to_json(event: VenueHaltInterval) -> dict[str, object]:
    """Lossless JSON codec payload for one halt interval (epoch-ns integers)."""
    return {
        "halt_id": event.halt_id,
        "start_ns": int(event.start.value),
        "end_ns": int(event.end.value),
        "present_symbols": int(event.present_symbols),
        "zero_symbols": int(event.zero_symbols),
        "evidence": event.evidence,
        "verified_at_ns": int(event.verified_at.value),
    }


def venue_halt_from_json(payload: Mapping[str, object]) -> VenueHaltInterval:
    """Lossless JSON codec for one halt interval (epoch-ns integers)."""
    try:
        halt_id = payload["halt_id"]
        evidence = payload["evidence"]
    except KeyError as exc:
        raise DataIntegrityError(f"venue halt payload missing field {exc}") from exc
    if not isinstance(halt_id, str) or not halt_id:
        raise DataIntegrityError("venue halt halt_id must be a nonempty string")
    if not isinstance(evidence, str) or not evidence.strip():
        raise DataIntegrityError("venue halt evidence must be a non-empty string")
    for name in ("present_symbols", "zero_symbols"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise DataIntegrityError(f"venue halt field {name} must be an integer")
    try:
        start = _require_ts_ns(payload["start_ns"], "start_ns")
        end = _require_ts_ns(payload["end_ns"], "end_ns")
        verified_at = _require_ts_ns(payload["verified_at_ns"], "verified_at_ns")
    except KeyError as exc:
        raise DataIntegrityError(f"venue halt payload missing field {exc}") from exc
    return VenueHaltInterval(
        halt_id=halt_id,
        start=start,
        end=end,
        present_symbols=cast(int, payload["present_symbols"]),
        zero_symbols=cast(int, payload["zero_symbols"]),
        evidence=evidence,
        verified_at=verified_at,
    )

_PROXY_SOURCES: frozenset[str] = frozenset({"flat_1h_klines", "twap30_proxy"})
_INT64_MAX: int = np.iinfo(np.int64).max


def settlement_event_from_record(record: InstrumentSettlementRecord) -> InstrumentSettlementEvent:
    """Map one registry record to the engine event (delivery is venue-available at delivery)."""
    return InstrumentSettlementEvent(
        event_id=record.event_id,
        symbol=record.symbol,
        effective_at=record.delivery_at,
        available_at=record.delivery_at,
        settlement_price=float(record.settlement_price),
        fee_bps=float(record.fee_bps),
        source_digest=record_digest(record),
        announced_at=record.announced_at,
        last_trade_at=record.last_trade_at,
        price_source=record.price_source,
    )


def _is_liquid(value: object) -> bool:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return np.isfinite(number) and number > 0.0


def settlement_events_for_piece(
    registry: InstrumentSettlementRegistry,
    roster: Sequence[str],
    piece_grid: pd.DatetimeIndex,
    quote_volumes: pd.DataFrame,
    *,
    replay_start: pd.Timestamp,
    replay_end: pd.Timestamp,
) -> tuple[InstrumentSettlementEvent, ...]:
    """Events a replay piece must carry: every lifecycle of a roster symbol announced by the piece end.

    Emission is idempotent across overlapping pieces (the accumulator admits each event_id once),
    so any event announced at or before ``piece_grid[-1]`` is re-emitted while its symbol is on the
    roster; the accumulator, not the producer, gates every field by time.

    Raises:
        DataIntegrityError: a roster symbol has more than one lifecycle intersecting
            ``[replay_start, replay_end)`` (relisting inside one replay is unsupported); or a bar
            of ``quote_volumes`` with label in ``[last_trade_at, delivery_at]`` is liquid for the
            event symbol (registry contradicts the lake).
    """
    if len(piece_grid) == 0:
        raise DataIntegrityError("piece_grid must be non-empty")
    piece_end = piece_grid[-1]
    out: list[InstrumentSettlementEvent] = []
    for symbol in roster:
        records = registry.settlements_for(symbol)
        intersecting = [
            r for r in records if not (r.delivery_at <= replay_start or r.announced_at >= replay_end)
        ]
        if len(intersecting) > 1:
            raise DataIntegrityError(
                f"relisting inside one replay is unsupported for {symbol!r}"
            )
        for record in records:
            if record.announced_at > piece_end:
                continue
            event = settlement_event_from_record(record)
            if symbol in quote_volumes.columns:
                col = quote_volumes[symbol]
                mask = (col.index >= record.last_trade_at) & (col.index <= record.delivery_at)
                window = col.loc[mask] if bool(mask.any()) else col.iloc[0:0]
                if any(_is_liquid(v) for v in window.to_numpy()):
                    raise DataIntegrityError(
                        f"settlement registry contradicts lake for {symbol!r}: "
                        f"liquid bar in [{record.last_trade_at}, {record.delivery_at}]"
                    )
            out.append(event)
    out.sort(key=lambda e: (e.effective_at.value, e.event_id))
    return tuple(out)


def settled_before_piece(
    registry: InstrumentSettlementRegistry,
    roster: Sequence[str],
    piece_grid: pd.DatetimeIndex,
    piece_weights: pd.DataFrame,
) -> frozenset[str]:
    """Roster symbols that may be dropped: delivery strictly before ``piece_grid[0]`` and every piece target of the symbol exactly 0.0 (or no decision rows)."""
    if len(piece_grid) == 0:
        return frozenset()
    start = piece_grid[0]
    drop: set[str] = set()
    for symbol in roster:
        delivered = any(r.delivery_at < start for r in registry.settlements_for(symbol))
        if not delivered:
            continue
        if len(piece_weights) == 0 or symbol not in piece_weights.columns:
            drop.add(symbol)
            continue
        col = piece_weights[symbol].to_numpy()
        if col.size and all(float(v) == 0.0 for v in col):
            drop.add(symbol)
    return frozenset(drop)


def settlement_fill_price(event: InstrumentSettlementEvent, units: float, spec: ExecutionSpec) -> float:
    """Adverse-haircut settlement price for one bound (identity for curated/venue sources)."""
    price = float(event.settlement_price)
    haircut = float(spec.settlement_price_haircut_bps)
    if event.price_source in ("curated", "venue") or haircut == 0.0:
        result = price
    else:
        if event.price_source not in _PROXY_SOURCES:
            raise DataIntegrityError(f"unknown settlement price_source {event.price_source!r}")
        sign = 1.0 if units > 0 else (-1.0 if units < 0 else 0.0)
        result = price * (1.0 - sign * haircut / 1e4)
    if not np.isfinite(result) or result <= 0.0:
        raise DataIntegrityError("haircut settlement price must be finite positive")
    return float(result)


def settlement_event_to_json(event: InstrumentSettlementEvent) -> dict[str, object]:
    """Lossless JSON codec payload (epoch-ns integers, float repr) for the window IPC spill."""
    return {
        "event_id": event.event_id,
        "symbol": event.symbol,
        "effective_at_ns": int(event.effective_at.value),
        "available_at_ns": int(event.available_at.value),
        "announced_at_ns": int(event.announced_at.value) if event.announced_at is not None else None,
        "last_trade_at_ns": int(event.last_trade_at.value) if event.last_trade_at is not None else None,
        "settlement_price": float(event.settlement_price),
        "fee_bps": float(event.fee_bps),
        "price_source": event.price_source,
        "source_digest": event.source_digest,
    }


def _require_ts_ns(value: object, field: str) -> pd.Timestamp:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DataIntegrityError(f"settlement event field {field} must be an epoch-ns integer")
    if value == _INT64_MAX or abs(value) > _INT64_MAX:
        raise DataIntegrityError(f"settlement event field {field} out of range")
    try:
        return pd.Timestamp(int(value), unit="ns", tz="UTC")
    except (ValueError, OverflowError) as exc:
        raise DataIntegrityError(f"settlement event field {field} malformed") from exc


def settlement_event_from_json(payload: Mapping[str, object]) -> InstrumentSettlementEvent:
    """Lossless JSON codec (epoch-ns integers, float repr) for the window IPC spill."""
    try:
        event_id = payload["event_id"]
        symbol = payload["symbol"]
        price_source = payload["price_source"]
        source_digest = payload["source_digest"]
        price = payload["settlement_price"]
        fee = payload["fee_bps"]
    except KeyError as exc:
        raise DataIntegrityError(f"settlement event payload missing field {exc}") from exc
    if not isinstance(event_id, str) or not event_id:
        raise DataIntegrityError("settlement event_id must be a nonempty string")
    if not isinstance(symbol, str) or not symbol:
        raise DataIntegrityError("settlement symbol must be a nonempty string")
    if not isinstance(source_digest, str) or not source_digest:
        raise DataIntegrityError("settlement source_digest must be a nonempty string")
    if not isinstance(price_source, str):
        raise DataIntegrityError("settlement price_source must be a string")
    for number, name in ((price, "settlement_price"), (fee, "fee_bps")):
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise DataIntegrityError(f"settlement event field {name} must be a number")
    try:
        effective_at = _require_ts_ns(payload["effective_at_ns"], "effective_at_ns")
        available_at = _require_ts_ns(payload["available_at_ns"], "available_at_ns")
    except KeyError as exc:
        raise DataIntegrityError(f"settlement event payload missing field {exc}") from exc
    try:
        announced_raw = payload["announced_at_ns"]
        last_trade_raw = payload["last_trade_at_ns"]
    except KeyError as exc:
        raise DataIntegrityError(f"settlement event payload missing field {exc}") from exc
    announced_at = None if announced_raw is None else _require_ts_ns(announced_raw, "announced_at_ns")
    last_trade_at = None if last_trade_raw is None else _require_ts_ns(last_trade_raw, "last_trade_at_ns")
    return InstrumentSettlementEvent(
        event_id=event_id,
        symbol=symbol,
        effective_at=effective_at,
        available_at=available_at,
        settlement_price=float(price),  # type: ignore[arg-type]
        fee_bps=float(fee),  # type: ignore[arg-type]
        source_digest=source_digest,
        announced_at=announced_at,
        last_trade_at=last_trade_at,
        price_source=price_source,  # type: ignore[arg-type]
    )


__all__ = [
    "settled_before_piece",
    "settlement_event_from_json",
    "settlement_event_from_record",
    "settlement_event_to_json",
    "settlement_events_for_piece",
    "settlement_fill_price",
    "venue_halt_from_json",
    "venue_halt_to_json",
]
