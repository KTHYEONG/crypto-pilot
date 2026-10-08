"""Evidenced exchange-wide trading-halt registry on the 3m execution plane."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import pandas as pd

from src.common.errors import DataIntegrityError

_HALT_FIELDS: Final[tuple[str, ...]] = (
    "kind",
    "halt_id",
    "start",
    "end",
    "present_symbols",
    "zero_symbols",
    "evidence",
    "verified_at",
)


@dataclass(frozen=True, slots=True)
class VenueHaltInterval:
    """One evidenced exchange-wide trading halt on the 3m execution plane.

    Attributes:
        halt_id: ISO label of the first halted bar (stable identity).
        start: First halted 3m label, UTC, inclusive.
        end: First label after the halt, UTC, exclusive.
        present_symbols: Live symbols with a bar at ``start`` (zombie tails excluded).
        zero_symbols: Of those, symbols with zero quote volume at ``start``.
        evidence: Human-readable cross-section description.
        verified_at: UTC instant of verification.
    """

    halt_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    present_symbols: int
    zero_symbols: int
    evidence: str
    verified_at: pd.Timestamp


@dataclass(frozen=True, slots=True)
class VenueHaltRegistry:
    """Validated, chronologically sorted, non-overlapping halts with a content digest."""

    halts: tuple[VenueHaltInterval, ...] = ()
    digest: str = ""
    _index: tuple[VenueHaltInterval, ...] = field(init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", tuple(self.halts))

    def overlapping(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[VenueHaltInterval, ...]:
        """Return halts intersecting the half-open range ``[start, end)``."""
        return tuple(h for h in self._index if h.start < end and h.end > start)


def _empty_digest() -> str:
    return "sha256:" + hashlib.sha256(b"").hexdigest()


EMPTY_VENUE_HALT_REGISTRY: Final[VenueHaltRegistry] = VenueHaltRegistry(halts=(), digest=_empty_digest())


def _format_moment(value: pd.Timestamp) -> str:
    return str(value.tz_convert("UTC").isoformat().replace("+00:00", "Z"))


def _canonical_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _halt_payload(halt: VenueHaltInterval) -> dict[str, Any]:
    return {
        "kind": "venue_halt",
        "halt_id": halt.halt_id,
        "start": _format_moment(halt.start),
        "end": _format_moment(halt.end),
        "present_symbols": int(halt.present_symbols),
        "zero_symbols": int(halt.zero_symbols),
        "evidence": halt.evidence,
        "verified_at": _format_moment(halt.verified_at),
    }


def _registry_digest(payloads: list[dict[str, Any]]) -> str:
    ordered = sorted(payloads, key=lambda p: (p["start"], p["halt_id"]))
    joined = "\n".join(_canonical_dumps(p) for p in ordered)
    return "sha256:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _parse_moment(value: object, field_name: str, line_no: int, source: str) -> pd.Timestamp:
    if not isinstance(value, str) or not value.strip():
        raise DataIntegrityError(f"{source} line {line_no}: {field_name} must be a non-empty string")
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


def _parse_halt(record: dict[str, Any], line_no: int, source: str) -> VenueHaltInterval:
    from src.core.params import VENUE_HALT_MIN_PRESENT_SYMBOLS, VENUE_HALT_MIN_ZERO_FRACTION

    extra = sorted(set(record) - set(_HALT_FIELDS))
    missing = sorted(set(_HALT_FIELDS) - set(record))
    if extra or missing:
        raise DataIntegrityError(
            f"{source} line {line_no}: missing or extra field (missing={missing}, extra={extra})",
        )
    if record["kind"] != "venue_halt":
        raise DataIntegrityError(f"{source} line {line_no}: unknown kind {record['kind']!r}")
    halt_id = record["halt_id"]
    if not isinstance(halt_id, str) or not halt_id.strip():
        raise DataIntegrityError(f"{source} line {line_no}: halt_id must be a non-empty string")
    start = _parse_moment(record["start"], "start", line_no, source)
    end = _parse_moment(record["end"], "end", line_no, source)
    _require_grid(start, "start", line_no, source)
    _require_grid(end, "end", line_no, source)
    if not start < end:
        raise DataIntegrityError(f"{source} line {line_no}: start must precede end")
    present = record["present_symbols"]
    zero = record["zero_symbols"]
    if isinstance(present, bool) or not isinstance(present, int):
        raise DataIntegrityError(f"{source} line {line_no}: present_symbols must be an integer")
    if isinstance(zero, bool) or not isinstance(zero, int):
        raise DataIntegrityError(f"{source} line {line_no}: zero_symbols must be an integer")
    if present < VENUE_HALT_MIN_PRESENT_SYMBOLS:
        raise DataIntegrityError(f"{source} line {line_no}: present_symbols below minimum")
    if zero < VENUE_HALT_MIN_ZERO_FRACTION * present:
        raise DataIntegrityError(f"{source} line {line_no}: zero share below minimum")
    if not 0 <= zero <= present:
        raise DataIntegrityError(f"{source} line {line_no}: zero_symbols must satisfy 0 <= zero <= present")
    evidence = record["evidence"]
    if not isinstance(evidence, str) or not evidence.strip():
        raise DataIntegrityError(f"{source} line {line_no}: evidence must not be blank")
    verified_at = _parse_moment(record["verified_at"], "verified_at", line_no, source)
    return VenueHaltInterval(
        halt_id=halt_id,
        start=start,
        end=end,
        present_symbols=present,
        zero_symbols=zero,
        evidence=evidence,
        verified_at=verified_at,
    )


def _assemble(halts: list[VenueHaltInterval]) -> VenueHaltRegistry:
    ordered = tuple(sorted(halts, key=lambda h: (h.start, h.halt_id)))
    for prev, cur in pairwise(ordered):
        if cur.start < prev.end:
            raise DataIntegrityError(
                f"overlapping venue halts {prev.halt_id!r} and {cur.halt_id!r}",
            )
    payloads = [_halt_payload(h) for h in ordered]
    return VenueHaltRegistry(halts=ordered, digest=_registry_digest(payloads))


def assemble_venue_halt_registry(halts: tuple[VenueHaltInterval, ...] | list[VenueHaltInterval]) -> VenueHaltRegistry:
    """Assemble validated intervals into a sorted, digested registry."""
    return _assemble(list(halts))


def venue_halt_registry_jsonl(registry: VenueHaltRegistry) -> bytes:
    """Serialize a registry to canonical JSONL (one trailing newline)."""
    lines = [_canonical_dumps(_halt_payload(h)) for h in registry.halts]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def parse_venue_halt_registry(raw: bytes, *, source: str) -> VenueHaltRegistry:
    """Parse and validate JSONL registry bytes.

    Raises:
        DataIntegrityError: on malformed JSON, missing/extra keys, naive/non-UTC or off-grid
            timestamps, ``start >= end``, overlaps, ``present_symbols < VENUE_HALT_MIN_PRESENT_SYMBOLS``,
            or ``zero_symbols < VENUE_HALT_MIN_ZERO_FRACTION x present_symbols``.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataIntegrityError(f"{source}: file must be UTF-8") from exc
    halts: list[VenueHaltInterval] = []
    decoder = json.JSONDecoder()
    position = 0
    end = len(text)
    while True:
        while position < end and text[position] in (" ", "\t", "\r", "\n"):
            position += 1
        if position >= end:
            break
        line_no = text.count("\n", 0, position) + 1
        try:
            record, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            raise DataIntegrityError(f"{source} line {line_no}: malformed JSON") from exc
        if not isinstance(record, dict):
            raise DataIntegrityError(f"{source} line {line_no}: record must be a JSON object")
        halts.append(_parse_halt(record, line_no, source))
    return _assemble(halts)


def default_venue_halt_registry_path() -> Path:
    """Return the committed registry path (``src/core/policy/venue_halts.jsonl``)."""
    return Path(__file__).resolve().parent / "policy" / "venue_halts.jsonl"


_registry_cache_raw: bytes | None = None
_registry_cache_value: VenueHaltRegistry = EMPTY_VENUE_HALT_REGISTRY


def clear_venue_halt_registry_cache() -> None:
    """Drop the cached default-registry parse so tests observe a fresh file."""
    global _registry_cache_raw, _registry_cache_value
    _registry_cache_raw = None
    _registry_cache_value = EMPTY_VENUE_HALT_REGISTRY


def load_venue_halt_registry(path: Path | None = None) -> VenueHaltRegistry:
    """Load the committed registry (default ``src/core/policy/venue_halts.jsonl``).

    Default ``src/core/policy/venue_halts.jsonl``; byte-compared cache like the settlement registry.

    Raises:
        DataIntegrityError: missing/unreadable file or any parse failure.
    """
    global _registry_cache_raw, _registry_cache_value
    target = default_venue_halt_registry_path() if path is None else path
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise DataIntegrityError(f"venue halt registry unreadable: {target}") from exc
    if path is None and _registry_cache_raw is not None and _registry_cache_raw == raw:
        return _registry_cache_value
    registry = parse_venue_halt_registry(raw, source=str(target))
    if path is None:
        _registry_cache_raw = raw
        _registry_cache_value = registry
    return registry


def venue_halt_registry_for_root(ohlcv_root: str | Path) -> VenueHaltRegistry:
    """Committed registry for the canonical lake root; EMPTY for any other root."""
    from src.common.paths import FUTURES_DATA_DIR

    canonical = (FUTURES_DATA_DIR / "ohlcv").resolve()
    if Path(ohlcv_root).resolve() == canonical:
        return load_venue_halt_registry()
    return EMPTY_VENUE_HALT_REGISTRY


__all__ = [
    "EMPTY_VENUE_HALT_REGISTRY",
    "VenueHaltInterval",
    "VenueHaltRegistry",
    "assemble_venue_halt_registry",
    "clear_venue_halt_registry_cache",
    "default_venue_halt_registry_path",
    "load_venue_halt_registry",
    "parse_venue_halt_registry",
    "venue_halt_registry_for_root",
    "venue_halt_registry_jsonl",
]
