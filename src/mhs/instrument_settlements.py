"""Point-in-time instrument settlement registry: the only authority replay may settle on.

A delisted perpetual's terminal inventory is settled at a curated, evidenced price;
the pipeline never infers a settlement from inactivity at run time. This module owns
the record contracts and the fail-closed JSONL loader. Lake measurement lives in
``src.mhs.settlement_evidence``; the operator generator lives in
``src.application.ops.settlement_registry``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.params import DELIST_ANNOUNCEMENT_LEAD

SettlementPriceSource = Literal["flat_1h_klines", "twap30_proxy", "curated"]
AnnouncementSource = Literal["proxy_lead", "curated"]

_VALID_PRICE_SOURCES: Final[tuple[str, ...]] = ("flat_1h_klines", "twap30_proxy", "curated")
_VALID_ANNOUNCEMENT_SOURCES: Final[tuple[str, ...]] = ("proxy_lead", "curated")

_SETTLEMENT_FIELDS: Final[tuple[str, ...]] = (
    "kind", "symbol", "event_id", "announced_at", "announcement_source",
    "announcement_evidence", "last_trade_at", "delivery_at", "settlement_price",
    "price_source", "price_evidence", "fee_bps", "evidence_digest", "verified_at",
)
_TRUNCATION_FIELDS: Final[tuple[str, ...]] = ("kind", "symbol", "data_end", "evidence", "verified_at")

_MS_PER_UNIT: Final[int] = 1_000_000


@dataclass(frozen=True, slots=True)
class InstrumentSettlementRecord:
    """One evidenced end of an instrument lifecycle: announcement, last trade, delivery and price.

    The record is the only authority the replay engine may use to settle inventory. It is curated
    and committed ahead of any run; the pipeline never infers a settlement from inactivity at run
    time. Fields are consumed causally: ``announced_at`` only by decisions made at or after it,
    ``last_trade_at`` only by fills at or after it, ``delivery_at`` only at delivery.

    Attributes:
        symbol: Upper-case exchange symbol.
        event_id: ``f"{symbol}:{delivery_at epoch ms}"``; unique and stable.
        announced_at: UTC instant the delisting became public (see announcement_source).
        announcement_source: ``proxy_lead`` (last_trade_at minus DELIST_ANNOUNCEMENT_LEAD) or
            ``curated`` (real announcement with evidence).
        announcement_evidence: Official notice reference; required non-empty when curated.
        last_trade_at: End of the last liquid 3m bar (3m grid).
        delivery_at: Contractual settlement instant (3m grid), >= last_trade_at.
        settlement_price: Finite positive price per unit.
        price_source: Evidence class of ``settlement_price``.
        price_evidence: Human-readable description of the bars or notice used.
        fee_bps: Finite non-negative settlement fee on delivered notional.
        evidence_digest: sha256 of the bars that determined the price (recomputable from the lake).
        verified_at: UTC instant the record was last reconciled against the lake.
    """

    symbol: str
    event_id: str
    announced_at: pd.Timestamp
    announcement_source: AnnouncementSource
    announcement_evidence: str
    last_trade_at: pd.Timestamp
    delivery_at: pd.Timestamp
    settlement_price: float
    price_source: SettlementPriceSource
    price_evidence: str
    fee_bps: float
    evidence_digest: str
    verified_at: pd.Timestamp


@dataclass(frozen=True, slots=True)
class DataTruncationRecord:
    """Operator declaration that a symbol's file ends because collection stopped, not trading.

    Attributes:
        symbol: Upper-case exchange symbol.
        data_end: First absent 3m label (last observed label + 3m), UTC.
        evidence: Why the end is a collection horizon, not a delisting.
        verified_at: UTC instant of the declaration.
    """

    symbol: str
    data_end: pd.Timestamp
    evidence: str
    verified_at: pd.Timestamp


@dataclass(frozen=True, slots=True)
class InstrumentSettlementRegistry:
    """Validated, immutable registry with O(1) symbol lookup and a content digest.

    Attributes:
        settlements: Records sorted by (delivery_at, symbol).
        truncations: Records sorted by (symbol, data_end).
        digest: Canonical registry digest (formatting-independent).
    """

    settlements: tuple[InstrumentSettlementRecord, ...] = ()
    truncations: tuple[DataTruncationRecord, ...] = ()
    digest: str = ""
    _settlement_index: Mapping[str, tuple[InstrumentSettlementRecord, ...]] = field(
        init=False, repr=False, compare=False,
    )
    _truncation_index: Mapping[str, DataTruncationRecord] = field(
        init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        by_symbol: dict[str, list[InstrumentSettlementRecord]] = {}
        for record in self.settlements:
            by_symbol.setdefault(record.symbol, []).append(record)
        object.__setattr__(
            self, "_settlement_index", MappingProxyType({key: tuple(value) for key, value in by_symbol.items()}),
        )
        object.__setattr__(
            self, "_truncation_index", MappingProxyType({record.symbol: record for record in self.truncations}),
        )

    def settlements_for(self, symbol: str) -> tuple[InstrumentSettlementRecord, ...]:
        """Return every settlement record of one symbol (empty when none)."""
        return self._settlement_index.get(symbol, ())

    def truncation_for(self, symbol: str) -> DataTruncationRecord | None:
        """Return the truncation record of one symbol, or None."""
        return self._truncation_index.get(symbol)


def _empty_digest() -> str:
    return "sha256:" + hashlib.sha256(b"").hexdigest()


EMPTY_SETTLEMENT_REGISTRY: Final[InstrumentSettlementRegistry] = InstrumentSettlementRegistry(
    settlements=(), truncations=(), digest=_empty_digest(),
)


def _format_moment(value: pd.Timestamp) -> str:
    return str(value.tz_convert("UTC").isoformat().replace("+00:00", "Z"))


def _canonical_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _settlement_payload(record: InstrumentSettlementRecord) -> dict[str, Any]:
    return {
        "kind": "settlement",
        "symbol": record.symbol,
        "event_id": record.event_id,
        "announced_at": _format_moment(record.announced_at),
        "announcement_source": record.announcement_source,
        "announcement_evidence": record.announcement_evidence,
        "last_trade_at": _format_moment(record.last_trade_at),
        "delivery_at": _format_moment(record.delivery_at),
        "settlement_price": float(record.settlement_price),
        "price_source": record.price_source,
        "price_evidence": record.price_evidence,
        "fee_bps": float(record.fee_bps),
        "evidence_digest": record.evidence_digest,
        "verified_at": _format_moment(record.verified_at),
    }


def _truncation_payload(record: DataTruncationRecord) -> dict[str, Any]:
    return {
        "kind": "data_truncation",
        "symbol": record.symbol,
        "data_end": _format_moment(record.data_end),
        "evidence": record.evidence,
        "verified_at": _format_moment(record.verified_at),
    }


def _sort_key(payload: dict[str, Any]) -> tuple[str, str, str]:
    moment = payload["delivery_at"] if payload["kind"] == "settlement" else payload["data_end"]
    return (payload["kind"], payload["symbol"], moment)


def _registry_digest(payloads: list[dict[str, Any]]) -> str:
    ordered = sorted(payloads, key=_sort_key)
    joined = "\n".join(_canonical_dumps(payload) for payload in ordered)
    return "sha256:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()


def record_digest(record: InstrumentSettlementRecord) -> str:
    """Canonical sha256 of one record; becomes the replay event's ``source_digest``."""
    raw = _canonical_dumps(_settlement_payload(record)).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _parse_moment(value: object, field_name: str, line_no: int, source: str) -> pd.Timestamp:
    if not isinstance(value, str):
        raise DataIntegrityError(f"{source} line {line_no}: {field_name} must be a non-empty string")
    if not value.strip():
        raise DataIntegrityError(f"{source} line {line_no}: {field_name} must not be blank")
    text = value.strip()
    normalized = f"{text[:-1]}+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise DataIntegrityError(f"{source} line {line_no}: {field_name} is not ISO8601") from exc
    if parsed.tzinfo is None:
        raise DataIntegrityError(f"{source} line {line_no}: {field_name} must be tz-aware UTC")
    if parsed.utcoffset() != timedelta(0):
        raise DataIntegrityError(f"{source} line {line_no}: {field_name} must be UTC")
    return pd.Timestamp(normalized).tz_convert("UTC")


def _require_grid(value: pd.Timestamp, field_name: str, line_no: int, source: str) -> None:
    if (
        value.second != 0
        or value.microsecond != 0
        or value.nanosecond != 0
        or (value.hour * 60 + value.minute) % 3 != 0
    ):
        raise DataIntegrityError(f"{source} line {line_no}: {field_name} is off the 3m grid")


def _parse_symbol(value: object, line_no: int, source: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or value != value.upper()
    ):
        raise DataIntegrityError(f"{source} line {line_no}: symbol must be an upper-case string")
    return value


def _parse_price(value: object, line_no: int, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataIntegrityError(f"{source} line {line_no}: settlement_price must be a JSON number")
    price = float(value)
    if price != price or price in (float("inf"), float("-inf")) or price <= 0.0:
        raise DataIntegrityError(f"{source} line {line_no}: settlement_price must be a finite positive price")
    return price


def _parse_fee(value: object, line_no: int, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataIntegrityError(f"{source} line {line_no}: fee_bps must be a JSON number")
    fee = float(value)
    if fee != fee or fee in (float("inf"), float("-inf")) or fee < 0.0:
        raise DataIntegrityError(f"{source} line {line_no}: fee_bps must be a finite non-negative fee")
    return fee


def _parse_settlement(record: dict[str, Any], line_no: int, source: str) -> InstrumentSettlementRecord:
    extra = sorted(set(record) - set(_SETTLEMENT_FIELDS))
    missing = sorted(set(_SETTLEMENT_FIELDS) - set(record))
    if extra or missing:
        raise DataIntegrityError(
            f"{source} line {line_no}: missing or extra field"
            f" (missing={missing}, extra={extra})",
        )
    symbol = _parse_symbol(record["symbol"], line_no, source)
    announcement_source = record["announcement_source"]
    if announcement_source not in _VALID_ANNOUNCEMENT_SOURCES:
        raise DataIntegrityError(f"{source} line {line_no}: unknown announcement_source {announcement_source!r}")
    price_source = record["price_source"]
    if price_source not in _VALID_PRICE_SOURCES:
        raise DataIntegrityError(f"{source} line {line_no}: unknown price_source {price_source!r}")
    announcement_evidence = record["announcement_evidence"]
    if not isinstance(announcement_evidence, str):
        raise DataIntegrityError(f"{source} line {line_no}: announcement_evidence must be a string")
    if announcement_source == "curated" and not announcement_evidence.strip():
        raise DataIntegrityError(f"{source} line {line_no}: curated announcement of {symbol} without evidence")
    price_evidence = record["price_evidence"]
    if not isinstance(price_evidence, str) or not price_evidence.strip():
        raise DataIntegrityError(f"{source} line {line_no}: price_evidence must not be blank")
    evidence_digest = record["evidence_digest"]
    if not isinstance(evidence_digest, str) or not evidence_digest:
        raise DataIntegrityError(f"{source} line {line_no}: evidence_digest must be a non-empty string")
    announced_at = _parse_moment(record["announced_at"], "announced_at", line_no, source)
    last_trade_at = _parse_moment(record["last_trade_at"], "last_trade_at", line_no, source)
    delivery_at = _parse_moment(record["delivery_at"], "delivery_at", line_no, source)
    verified_at = _parse_moment(record["verified_at"], "verified_at", line_no, source)
    for field_name, moment in (("last_trade_at", last_trade_at), ("delivery_at", delivery_at)):
        _require_grid(moment, field_name, line_no, source)
    if not (announced_at <= last_trade_at <= delivery_at):
        raise DataIntegrityError(
            f"{source} line {line_no}: lifecycle ordering violated"
            " (announced_at <= last_trade_at <= delivery_at required)",
        )
    if announcement_source == "proxy_lead" and announced_at != last_trade_at - DELIST_ANNOUNCEMENT_LEAD:
        raise DataIntegrityError(
            f"{source} line {line_no}: proxy_lead announced_at of {symbol} must equal"
            " last_trade_at minus DELIST_ANNOUNCEMENT_LEAD; regenerate the registry",
        )
    event_id = record["event_id"]
    expected = f"{symbol}:{int(delivery_at.value // _MS_PER_UNIT)}"
    if event_id != expected:
        raise DataIntegrityError(
            f"{source} line {line_no}: event_id mismatch for {symbol}"
            f" (got {event_id!r}, want {expected!r})",
        )
    return InstrumentSettlementRecord(
        symbol=symbol,
        event_id=event_id,
        announced_at=announced_at,
        announcement_source=announcement_source,
        announcement_evidence=announcement_evidence,
        last_trade_at=last_trade_at,
        delivery_at=delivery_at,
        settlement_price=_parse_price(record["settlement_price"], line_no, source),
        price_source=price_source,
        price_evidence=price_evidence,
        fee_bps=_parse_fee(record["fee_bps"], line_no, source),
        evidence_digest=evidence_digest,
        verified_at=verified_at,
    )


def _parse_truncation(record: dict[str, Any], line_no: int, source: str) -> DataTruncationRecord:
    extra = sorted(set(record) - set(_TRUNCATION_FIELDS))
    missing = sorted(set(_TRUNCATION_FIELDS) - set(record))
    if extra or missing:
        raise DataIntegrityError(
            f"{source} line {line_no}: missing or extra field"
            f" (missing={missing}, extra={extra})",
        )
    symbol = _parse_symbol(record["symbol"], line_no, source)
    evidence = record["evidence"]
    if not isinstance(evidence, str) or not evidence.strip():
        raise DataIntegrityError(f"{source} line {line_no}: evidence must not be blank")
    data_end = _parse_moment(record["data_end"], "data_end", line_no, source)
    _require_grid(data_end, "data_end", line_no, source)
    verified_at = _parse_moment(record["verified_at"], "verified_at", line_no, source)
    return DataTruncationRecord(
        symbol=symbol, data_end=data_end, evidence=evidence, verified_at=verified_at,
    )


def _assemble(
    settlements: list[InstrumentSettlementRecord],
    truncations: list[DataTruncationRecord],
) -> InstrumentSettlementRegistry:
    ordered_settlements = tuple(sorted(settlements, key=lambda record: (record.delivery_at, record.symbol)))
    ordered_truncations = tuple(sorted(truncations, key=lambda record: (record.symbol, record.data_end)))
    payloads = [_settlement_payload(record) for record in settlements]
    payloads.extend(_truncation_payload(record) for record in truncations)
    return InstrumentSettlementRegistry(
        settlements=ordered_settlements,
        truncations=ordered_truncations,
        digest=_registry_digest(payloads),
    )


def assemble_instrument_settlement_registry(
    settlements: Sequence[InstrumentSettlementRecord],
    truncations: Sequence[DataTruncationRecord],
) -> InstrumentSettlementRegistry:
    """Assemble validated records into a sorted, digested registry."""
    return _assemble(list(settlements), list(truncations))


def settlement_registry_jsonl(registry: InstrumentSettlementRegistry) -> bytes:
    """Serialize a registry to canonical JSONL (sorted keys, no whitespace, one trailing newline).

    Lines follow registry order: settlements by (delivery_at, symbol), then
    truncations by (symbol, data_end).
    """
    lines = [_canonical_dumps(_settlement_payload(record)) for record in registry.settlements]
    lines.extend(_canonical_dumps(_truncation_payload(record)) for record in registry.truncations)
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def _decode_registry_records(
    text: str, source: str,
) -> tuple[list[InstrumentSettlementRecord], list[DataTruncationRecord]]:
    settlements: list[InstrumentSettlementRecord] = []
    truncations: list[DataTruncationRecord] = []
    decoder = json.JSONDecoder()
    position = 0
    end = len(text)
    while True:
        while position < end and text[position] in (" ", "\t", "\r", "\n"):
            position += 1
        if position >= end:
            return settlements, truncations
        line_no = text.count("\n", 0, position) + 1
        try:
            record, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            raise DataIntegrityError(f"{source} line {line_no}: malformed JSON") from exc
        if not isinstance(record, dict):
            raise DataIntegrityError(f"{source} line {line_no}: record must be a JSON object")
        kind = record.get("kind")
        if kind == "settlement":
            settlements.append(_parse_settlement(record, line_no, source))
        elif kind == "data_truncation":
            truncations.append(_parse_truncation(record, line_no, source))
        else:
            raise DataIntegrityError(f"{source} line {line_no}: unknown kind {kind!r}")


def _reject_registry_conflicts(
    settlements: list[InstrumentSettlementRecord],
    truncations: list[DataTruncationRecord],
    source: str,
) -> None:
    seen_ids: set[str] = set()
    for settlement in settlements:
        if settlement.event_id in seen_ids:
            raise DataIntegrityError(f"{source}: duplicate event_id {settlement.event_id!r}")
        seen_ids.add(settlement.event_id)
    by_symbol: dict[str, list[InstrumentSettlementRecord]] = {}
    for settlement in settlements:
        by_symbol.setdefault(settlement.symbol, []).append(settlement)
    for symbol, records in by_symbol.items():
        ordered = sorted(records, key=lambda record: record.announced_at)
        for previous, current in pairwise(ordered):
            if previous.delivery_at >= current.announced_at:
                raise DataIntegrityError(
                    f"{source}: overlapping lifecycles for {symbol}"
                    f" ({previous.event_id!r} and {current.event_id!r})",
                )
    seen_truncations: set[str] = set()
    for truncation in truncations:
        if truncation.symbol in seen_truncations:
            raise DataIntegrityError(f"{source}: more than one truncation record for {truncation.symbol}")
        seen_truncations.add(truncation.symbol)
        for settlement in by_symbol.get(truncation.symbol, ()):
            if settlement.announced_at <= truncation.data_end <= settlement.delivery_at:
                raise DataIntegrityError(
                    f"{source}: truncation data_end of {truncation.symbol} inside"
                    f" settlement lifecycle {settlement.event_id!r}",
                )


def parse_instrument_settlement_registry(raw: bytes, *, source: str) -> InstrumentSettlementRegistry:
    """Parse and validate JSONL registry bytes.

    Raises:
        DataIntegrityError: any line is not a JSON object; unknown ``kind``; missing or extra
            field; malformed/naive/non-UTC timestamp; timestamp off the 3m grid; violated
            ordering ``announced_at <= last_trade_at <= delivery_at``; proxy_lead announcement not
            exactly ``last_trade_at - DELIST_ANNOUNCEMENT_LEAD`` (message names the record and says
            "regenerate the registry"); curated announcement or curated price without evidence;
            non-finite/non-positive price; negative/non-finite fee; ``event_id`` mismatch; duplicate
            ``event_id``; two settlement records of one symbol whose
            ``[announced_at, delivery_at]`` intervals overlap; more than one truncation record per
            symbol; a truncation ``data_end`` inside a settlement record's lifecycle. Messages are
            prefixed with ``source`` and the 1-based line number.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataIntegrityError(f"{source}: file must be UTF-8") from exc
    settlements, truncations = _decode_registry_records(text, source)
    _reject_registry_conflicts(settlements, truncations, source)
    return _assemble(settlements, truncations)


def default_instrument_settlement_registry_path() -> Path:
    """Return the committed registry path (``src/mhs/policy/instrument_settlements.jsonl``)."""
    return Path(__file__).resolve().parent / "policy" / "instrument_settlements.jsonl"


_registry_cache_raw: bytes | None = None
_registry_cache_value: InstrumentSettlementRegistry = EMPTY_SETTLEMENT_REGISTRY


def clear_instrument_settlement_registry_cache() -> None:
    """Drop the cached default-registry parse so tests observe a fresh file."""
    global _registry_cache_raw, _registry_cache_value
    _registry_cache_raw = None
    _registry_cache_value = EMPTY_SETTLEMENT_REGISTRY


def load_instrument_settlement_registry(path: Path | None = None) -> InstrumentSettlementRegistry:
    """Load the committed registry (default ``src/mhs/policy/instrument_settlements.jsonl``).

    The default path is parsed once per distinct file content (byte-compared cache, mirroring
    ``load_source_gap_registry``); an explicit path is always re-read.

    Raises:
        DataIntegrityError: missing/unreadable file or any parse failure.
    """
    global _registry_cache_raw, _registry_cache_value
    target = default_instrument_settlement_registry_path() if path is None else path
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise DataIntegrityError(f"settlement registry unreadable: {target}") from exc
    if path is None and _registry_cache_raw is not None and _registry_cache_raw == raw:
        return _registry_cache_value
    registry = parse_instrument_settlement_registry(raw, source=str(target))
    if path is None:
        _registry_cache_raw = raw
        _registry_cache_value = registry
    return registry


def settlement_registry_for_root(ohlcv_root: str | Path) -> InstrumentSettlementRegistry:
    """Registry whose evidence was measured on ``ohlcv_root``.

    The committed registry is evidence about the canonical lake only. It is returned when
    ``ohlcv_root`` resolves to ``FUTURES_DATA_DIR / "ohlcv"``; any other root (synthetic test
    lakes, operator overrides) gets ``EMPTY_SETTLEMENT_REGISTRY`` unless a caller passes an
    explicit registry. Binding by root prevents real delisting events from being applied to
    unrelated price paths that happen to reuse a real symbol name.
    """
    from src.common.paths import FUTURES_DATA_DIR

    canonical = (FUTURES_DATA_DIR / "ohlcv").resolve()
    if Path(ohlcv_root).resolve() == canonical:
        return load_instrument_settlement_registry()
    return EMPTY_SETTLEMENT_REGISTRY
