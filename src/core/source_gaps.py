"""Single evidenced registry of source-plane data gaps at interval scope."""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal, cast

import pandas as pd

from src.common.errors import DataIntegrityError

SourceGapPlane = Literal["ohlcv_1h", "ohlcv_3m", "funding"]
SourceGapExtent = Literal["LISTING_EDGE", "OPEN_EDGE", "INTERIOR", "UNSCOPED"]

_VALID_PLANES: Final[tuple[str, ...]] = ("ohlcv_1h", "ohlcv_3m", "funding")
_VALID_REASONS: Final[tuple[str, ...]] = ("SOURCE_ABSENT", "DELISTED", "SETTLING")
_VALID_EXTENTS: Final[tuple[str, ...]] = ("LISTING_EDGE", "OPEN_EDGE", "INTERIOR", "UNSCOPED")
# 과거 레지스트리 행에는 extent 가 없다. 측정되지 않은 범위이므로 가장 보수적인 UNSCOPED 로 읽는다.
_LEGACY_EXTENT: Final[SourceGapExtent] = "UNSCOPED"
_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "symbol",
    "plane",
    "start",
    "end",
    "reason",
    "evidence",
    "verified_at",
    "resolved_at",
)


@dataclass(frozen=True, slots=True)
class SourceGapInterval:
    """One evidenced interval in which a symbol's source plane has no recoverable data.

    The record is the single authority for "this symbol could not be traded or valued
    during this interval". It never asserts why a strategy avoided a symbol, only that
    the exchange source itself is absent, so downstream policy may scope its decision to
    the interval rather than discarding the symbol's entire history.

    Attributes:
        symbol: Upper-case exchange symbol.
        plane: Source plane the absence applies to.
        start: First absent bar, UTC, inclusive.
        end: First recovered bar, UTC, exclusive; None when the absence is open-ended.
        reason: Absence classification used by reporting, never by certification.
        evidence: Human-readable record of which sources were queried and found empty.
        verified_at: UTC time the absence was last empirically confirmed.
        resolved_at: UTC time the source was observed recovered; non-None deactivates it.
        extent: Measured position of the gap relative to the symbol's observed bars.
            ``LISTING_EDGE`` is the span before the first listed bar, ``OPEN_EDGE`` the
            span after the last observed bar of a symbol not known to be DELISTED,
            ``INTERIOR`` a bounded gap between observed bars, and ``UNSCOPED`` a legacy or
            manually curated record whose scope was never measured.
    """

    symbol: str
    plane: SourceGapPlane
    start: datetime
    end: datetime | None
    reason: str
    evidence: str
    verified_at: datetime
    resolved_at: datetime | None
    extent: SourceGapExtent = _LEGACY_EXTENT


def _default_registry_path() -> Path:
    return Path(__file__).resolve().parent / "policy" / "source_gaps.jsonl"


_registry_cache_raw: bytes | None = None
_registry_cache_value: tuple[SourceGapInterval, ...] = ()


def clear_source_gap_registry_cache() -> None:
    """Drop the cached default-registry parse so tests observe a fresh file."""
    global _registry_cache_raw, _registry_cache_value
    _registry_cache_raw = None
    _registry_cache_value = ()


def _parse_utc_moment(value: object, field: str, line_no: int) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise DataIntegrityError(f"source-gap registry line {line_no}: {field} must be a non-empty string")
    text = value.strip()
    normalized = f"{text[:-1]}+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise DataIntegrityError(f"source-gap registry line {line_no}: {field} is not ISO8601") from exc
    if parsed.tzinfo is None:
        raise DataIntegrityError(f"source-gap registry line {line_no}: {field} must be tz-aware UTC")
    if parsed.utcoffset() != timedelta(0):
        raise DataIntegrityError(f"source-gap registry line {line_no}: {field} must be UTC")
    return parsed.astimezone(UTC)


def _parse_record(record: object, line_no: int) -> SourceGapInterval:
    if not isinstance(record, dict):
        raise DataIntegrityError(f"source-gap registry line {line_no}: record must be a JSON object")
    for field in _REQUIRED_FIELDS:
        if field not in record:
            raise DataIntegrityError(f"source-gap registry line {line_no}: missing field {field!r}")
    symbol = record["symbol"]
    if not isinstance(symbol, str) or symbol.strip() != symbol or not symbol or symbol != symbol.upper():
        raise DataIntegrityError(f"source-gap registry line {line_no}: symbol must be an upper-case string")
    plane = record["plane"]
    if not isinstance(plane, str) or plane not in _VALID_PLANES:
        raise DataIntegrityError(f"source-gap registry line {line_no}: unknown plane {plane!r}")
    reason = record["reason"]
    if not isinstance(reason, str) or reason not in _VALID_REASONS:
        raise DataIntegrityError(f"source-gap registry line {line_no}: unknown reason {reason!r}")
    evidence = record["evidence"]
    if not isinstance(evidence, str) or not evidence.strip():
        raise DataIntegrityError(f"source-gap registry line {line_no}: evidence must not be blank")
    extent = record.get("extent", _LEGACY_EXTENT)
    if not isinstance(extent, str) or extent not in _VALID_EXTENTS:
        raise DataIntegrityError(f"source-gap registry line {line_no}: unknown extent {extent!r}")
    start = _parse_utc_moment(record["start"], "start", line_no)
    end_raw = record["end"]
    end: datetime | None = None
    if end_raw is not None:
        if not isinstance(end_raw, str):
            raise DataIntegrityError(f"source-gap registry line {line_no}: end must be ISO8601 UTC or null")
        end = _parse_utc_moment(end_raw, "end", line_no)
    if end is not None and end <= start:
        raise DataIntegrityError(f"source-gap registry line {line_no}: end must exceed start")
    verified_at = _parse_utc_moment(record["verified_at"], "verified_at", line_no)
    resolved_raw = record["resolved_at"]
    resolved_at: datetime | None = None
    if resolved_raw is not None:
        if not isinstance(resolved_raw, str):
            raise DataIntegrityError(f"source-gap registry line {line_no}: resolved_at must be ISO8601 UTC or null")
        resolved_at = _parse_utc_moment(resolved_raw, "resolved_at", line_no)
    if resolved_at is not None and resolved_at < verified_at:
        raise DataIntegrityError(f"source-gap registry line {line_no}: resolved_at must not precede verified_at")
    return SourceGapInterval(
        symbol=symbol,
        plane=cast(SourceGapPlane, plane),
        start=start,
        end=end,
        reason=reason,
        evidence=evidence,
        verified_at=verified_at,
        resolved_at=resolved_at,
        extent=cast(SourceGapExtent, extent),
    )


def _reject_active_overlaps(intervals: tuple[SourceGapInterval, ...], target: Path) -> None:
    active_end: dict[tuple[str, str], datetime | None] = {}
    for iv in intervals:
        if iv.resolved_at is not None:
            continue
        key = (iv.symbol, iv.plane)
        if key in active_end:
            previous = active_end[key]
            if previous is None or iv.start < previous:
                raise DataIntegrityError(f"source-gap registry {target}: overlapping active intervals for {key}")
        active_end[key] = iv.end


def _parse_registry_bytes(raw: bytes, target: Path) -> tuple[SourceGapInterval, ...]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataIntegrityError(f"source-gap registry {target}: file must be UTF-8") from exc
    records: list[SourceGapInterval] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataIntegrityError(f"source-gap registry {target} line {line_no}: malformed JSON") from exc
        records.append(_parse_record(record, line_no))
    ordered = tuple(sorted(records, key=lambda iv: (iv.symbol, iv.plane, iv.start)))
    _reject_active_overlaps(ordered, target)
    return ordered


def load_source_gap_registry(path: Path | None = None) -> tuple[SourceGapInterval, ...]:
    """Load and validate the checked-in source-gap registry.

    Reading is pure: the loader never consults market data, never infers an interval,
    and never repairs a malformed record. A registry that cannot be parsed is a fatal
    policy defect rather than an empty exclusion set, because silently returning no
    exclusions would let unrecoverable gaps enter a certified backtest.

    Args:
        path: Registry override; defaults to the packaged `policy/source_gaps.jsonl`.
    Returns:
        Every record, active and resolved, in normalized `(symbol, plane, start)` order.
    Raises:
        DataIntegrityError: A record is malformed, non-UTC, reversed, or overlaps a
            sibling interval of the same symbol and plane.
    """
    global _registry_cache_raw, _registry_cache_value
    target = _default_registry_path() if path is None else path
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise DataIntegrityError(f"source-gap registry unreadable: {target}") from exc
    if path is None and _registry_cache_raw is not None and _registry_cache_raw == raw:
        return _registry_cache_value
    intervals = _parse_registry_bytes(raw, target)
    if path is None:
        _registry_cache_raw = raw
        _registry_cache_value = intervals
    return intervals


def active_intervals(
    *,
    plane: SourceGapPlane | None = None,
    symbols: Collection[str] | None = None,
    path: Path | None = None,
) -> tuple[SourceGapInterval, ...]:
    """Return unresolved records, optionally narrowed to one plane and symbol set.

    Args:
        plane: Restrict to a single source plane; None returns every plane.
        symbols: Restrict to these symbols; None returns every symbol.
        path: Registry override forwarded to the loader.
    Returns:
        Unresolved records in normalized order.
    """
    wanted: frozenset[str] | None = frozenset(symbols) if symbols is not None else None
    return tuple(
        iv
        for iv in load_source_gap_registry(path)
        if iv.resolved_at is None and (plane is None or iv.plane == plane) and (wanted is None or iv.symbol in wanted)
    )


def _require_utc_index(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise DataIntegrityError("blocked_mask index must be a DatetimeIndex")
    if index.tz is None:
        raise DataIntegrityError("blocked_mask index must be timezone-aware UTC")
    if any(ts.utcoffset() != timedelta(0) for ts in index):
        raise DataIntegrityError("blocked_mask index must be UTC")


def blocked_mask(
    symbols: Sequence[str],
    index: pd.DatetimeIndex,
    *,
    plane: SourceGapPlane,
    path: Path | None = None,
) -> pd.DataFrame:
    """Project active intervals onto an exact time grid as a boolean block mask.

    True marks a bar whose source evidence is absent, so a caller may zero a target,
    drop a roster seat, or skip a finiteness assertion for exactly that cell instead of
    the symbol's whole column.

    Args:
        symbols: Column order of the returned frame; unknown symbols yield all-False.
        index: Timezone-aware UTC grid used verbatim as the frame index.
        plane: Source plane whose intervals are projected.
        path: Registry override forwarded to the loader.
    Returns:
        Boolean frame indexed by `index` with `symbols` as columns.
    Raises:
        DataIntegrityError: `index` is not tz-aware UTC or `symbols` repeats a name.
    """
    cols = list(symbols)
    if len(set(cols)) != len(cols):
        raise DataIntegrityError("blocked_mask symbols must not repeat a name")
    _require_utc_index(index)
    frame = pd.DataFrame(False, index=index, columns=cols, dtype=bool)
    known = set(cols)
    for iv in active_intervals(plane=plane, path=path):
        if iv.symbol not in known:
            continue
        hit = index >= pd.Timestamp(iv.start)
        if iv.end is not None:
            hit = hit & (index < pd.Timestamp(iv.end))
        frame.loc[hit, iv.symbol] = True
    return frame


def _require_utc_window(start: pd.Timestamp, end: pd.Timestamp) -> None:
    for name, moment in (("start", start), ("end", end)):
        if not isinstance(moment, pd.Timestamp):
            raise DataIntegrityError(f"blocked_symbols_between {name} must be a tz-aware UTC Timestamp")
        if moment.tzinfo is None:
            raise DataIntegrityError(f"blocked_symbols_between {name} must be tz-aware UTC")
        if moment.utcoffset() != timedelta(0):
            raise DataIntegrityError(f"blocked_symbols_between {name} must be UTC")
    if end <= start:
        raise DataIntegrityError("blocked_symbols_between end must exceed start")


def blocked_symbols_between(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    plane: SourceGapPlane,
    path: Path | None = None,
) -> frozenset[str]:
    """Return symbols whose active interval intersects the half-open window.

    Args:
        start: Window start, UTC, inclusive.
        end: Window end, UTC, exclusive; must exceed `start`.
        plane: Source plane whose intervals are tested.
        path: Registry override forwarded to the loader.
    Returns:
        Symbols with at least one overlapping unresolved interval.
    Raises:
        DataIntegrityError: The window is empty or not tz-aware UTC.
    """
    _require_utc_window(start, end)
    start_dt = start.to_pydatetime()
    end_dt = end.to_pydatetime()
    return frozenset(
        iv.symbol
        for iv in active_intervals(plane=plane, path=path)
        if iv.start < end_dt and (iv.end is None or iv.end > start_dt)
    )
