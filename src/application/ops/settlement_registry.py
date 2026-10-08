"""Operator generator for the point-in-time instrument settlement registry.

Builds candidate settlement records from the lake, preserves curated fields of the
committed registry, and writes atomically only on explicit request. The replay
pipeline never calls this module; it consumes the committed registry only.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.instrument_settlements import (
    DataTruncationRecord,
    InstrumentSettlementRecord,
    InstrumentSettlementRegistry,
    assemble_instrument_settlement_registry,
    parse_instrument_settlement_registry,
    settlement_registry_jsonl,
)
from src.mhs.params import (
    DELIST_ANNOUNCEMENT_LEAD,
    DELIST_SETTLEMENT_FEE_BPS,
    SETTLEMENT_AUDIT_MIN_TRAILING_FLAT_BARS,
)
from src.mhs.settlement_evidence import (
    SettlementPriceEvidence,
    SymbolTailProfile,
    derive_settlement_price,
    measure_symbol_tail,
)
from src.mhs.venue_halts import VenueHaltRegistry

_logger = logging.getLogger(__name__)

_TRUNCATION_STEP: pd.Timedelta = pd.Timedelta(minutes=3)
_EVENT_ID_MS_DIVISOR: int = 1_000_000


class _UnresolvedLifecycleError(Exception):
    """Internal signal: a required lifecycle has no derivable price (curation required)."""


@dataclass(frozen=True, slots=True)
class SettlementRegistryBuild:
    """Candidate registry plus findings an operator must resolve before committing.

    Attributes:
        registry: Candidate registry (curated fields of the existing registry preserved).
        unresolved: ``(symbol, reason)`` lifecycles with no derivable price (curation required).
        changed: event_ids whose generated content differs from the committed record.
    """

    registry: InstrumentSettlementRegistry
    unresolved: tuple[tuple[str, str], ...] = ()
    changed: tuple[str, ...] = ()


def _require_utc_moment(value: pd.Timestamp, label: str) -> pd.Timestamp:
    if not isinstance(value, pd.Timestamp) or pd.isna(value):
        raise DataIntegrityError(f"{label} must be a valid timestamp")
    if value.tzinfo is None or value.utcoffset() != pd.Timedelta(0):
        raise DataIntegrityError(f"{label} must be timezone-aware UTC")
    return value.tz_convert("UTC")


def _content_equal(generated: InstrumentSettlementRecord, committed: InstrumentSettlementRecord) -> bool:
    return (
        generated.symbol == committed.symbol
        and generated.event_id == committed.event_id
        and generated.announced_at == committed.announced_at
        and generated.announcement_source == committed.announcement_source
        and generated.announcement_evidence == committed.announcement_evidence
        and generated.last_trade_at == committed.last_trade_at
        and generated.delivery_at == committed.delivery_at
        and generated.settlement_price == committed.settlement_price
        and generated.price_source == committed.price_source
        and generated.price_evidence == committed.price_evidence
        and generated.fee_bps == committed.fee_bps
        and generated.evidence_digest == committed.evidence_digest
    )


def _carry_curated_announcement(
    symbol: str, last_trade_at: pd.Timestamp,
    prior: InstrumentSettlementRecord | None,
) -> tuple[pd.Timestamp, str, str]:
    if prior is None or prior.announcement_source != "curated":
        return last_trade_at - DELIST_ANNOUNCEMENT_LEAD, "proxy_lead", ""
    if not prior.announcement_evidence.strip():
        raise DataIntegrityError(f"committed curated announcement of {symbol} without evidence")
    if not prior.announced_at <= last_trade_at:
        raise DataIntegrityError(
            f"committed curated announcement of {symbol} no longer precedes last_trade_at",
        )
    return prior.announced_at, "curated", prior.announcement_evidence


def _carry_curated_price(
    symbol: str, evidence: SettlementPriceEvidence | None,
    prior: InstrumentSettlementRecord | None,
) -> tuple[float, str, str, str]:
    if prior is None or prior.price_source != "curated":
        if evidence is None:
            raise _UnresolvedLifecycleError("no derivable settlement price; curation required")
        return (
            evidence.settlement_price, evidence.price_source,
            evidence.price_evidence, evidence.evidence_digest,
        )
    if not prior.price_evidence.strip():
        raise DataIntegrityError(f"committed curated price of {symbol} without evidence")
    return (
        prior.settlement_price, "curated", prior.price_evidence, prior.evidence_digest,
    )


def _derive_candidate_record(
    root: Path, symbol: str, tail: SymbolTailProfile, end: pd.Timestamp,
    stamped: pd.Timestamp,
    committed_by_event: dict[str, InstrumentSettlementRecord],
    changed: list[str],
) -> InstrumentSettlementRecord | None:
    """Derive one candidate record, None when the lifecycle needs no record.

    Raises:
        _UnresolvedLifecycleError: the lifecycle ends before ``end`` but no price is derivable.
    """
    if tail.first_bar >= end:
        return None
    need_a = tail.last_bar + _TRUNCATION_STEP < end
    need_b = (
        tail.trailing_flat_bars >= SETTLEMENT_AUDIT_MIN_TRAILING_FLAT_BARS
        and tail.last_liquid_bar is not None
        and tail.last_liquid_bar + _TRUNCATION_STEP < end
    )
    if not (need_a or need_b):
        return None
    if tail.last_liquid_bar is None:
        raise _UnresolvedLifecycleError("lifecycle ends with no liquid bar; curation required")
    last_trade_at = tail.last_liquid_bar + _TRUNCATION_STEP
    event_id = f"{symbol}:{int(last_trade_at.value // _EVENT_ID_MS_DIVISOR)}"
    prior = committed_by_event.get(event_id)
    evidence = derive_settlement_price(root, symbol, last_trade_at)
    announced_at, announcement_source, announcement_evidence = _carry_curated_announcement(
        symbol, last_trade_at, prior,
    )
    settlement_price, price_source, price_evidence, evidence_digest = _carry_curated_price(
        symbol, evidence, prior,
    )
    candidate = InstrumentSettlementRecord(
        symbol=symbol,
        event_id=event_id,
        announced_at=announced_at,
        announcement_source=announcement_source,  # type: ignore[arg-type]
        announcement_evidence=announcement_evidence,
        last_trade_at=last_trade_at,
        delivery_at=last_trade_at,
        settlement_price=float(settlement_price),
        price_source=price_source,  # type: ignore[arg-type]
        price_evidence=price_evidence,
        fee_bps=float(DELIST_SETTLEMENT_FEE_BPS),
        evidence_digest=evidence_digest,
        verified_at=stamped,
    )
    if prior is not None:
        if _content_equal(candidate, prior):
            candidate = replace(candidate, verified_at=prior.verified_at)
        else:
            changed.append(event_id)
    return candidate


def _carry_consistent_truncations(
    existing: InstrumentSettlementRegistry, three_dir: Path, scanned: set[str],
) -> list[DataTruncationRecord]:
    return [
        record for record in existing.truncations
        if record.symbol not in scanned
        or (
            (three_dir / f"{record.symbol}.parquet").exists()
            and measure_symbol_tail(three_dir / f"{record.symbol}.parquet").last_bar + _TRUNCATION_STEP == record.data_end
        )
    ]


def build_settlement_registry(
    ohlcv_root: Path, *, horizon: pd.Timestamp, existing: InstrumentSettlementRegistry,
    verified_at: pd.Timestamp, symbols: Sequence[str] | None = None,
) -> SettlementRegistryBuild:
    """Derive one settlement record per lake symbol whose lifecycle ends before ``horizon``.

    Uses the audit rules (a)/(b) of ``audit_settlement_registry`` with ``audit_end = horizon`` over
    every 3m archive (or ``symbols``). For each required symbol: ``last_trade_at`` from the tail
    profile, ``delivery_at = last_trade_at``, price via ``derive_settlement_price``, fee
    ``DELIST_SETTLEMENT_FEE_BPS``, proxy_lead announcement. When ``existing`` holds the same
    ``event_id``, curated announcement fields and curated price fields are carried over unchanged
    and re-validated. Existing truncation records are carried over when still consistent.

    Raises:
        DataIntegrityError: horizon naive/non-UTC; an archive unreadable.
    """
    end = _require_utc_moment(horizon, "horizon")
    stamped = _require_utc_moment(verified_at, "verified_at")
    root = Path(ohlcv_root)
    three_dir = root / "3m"
    if not three_dir.is_dir():
        raise DataIntegrityError(f"settlement registry build unreadable: {three_dir}")
    if symbols is None:
        try:
            names = sorted(path.stem for path in three_dir.glob("*.parquet"))
        except OSError as exc:
            raise DataIntegrityError(f"settlement registry build unreadable: {three_dir}") from exc
    else:
        names = list(symbols)
    committed_by_event = {record.event_id: record for record in existing.settlements}
    records: list[InstrumentSettlementRecord] = []
    unresolved: list[tuple[str, str]] = []
    changed: list[str] = []
    flat_count = 0
    twap_count = 0
    curated_count = 0
    for symbol in names:
        minute_path = three_dir / f"{symbol}.parquet"
        if not minute_path.exists():
            raise DataIntegrityError(f"settlement bars unreadable: {minute_path.name}")
        tail = measure_symbol_tail(minute_path)
        truncation = existing.truncation_for(symbol)
        if (
            truncation is not None
            and truncation.data_end == tail.last_bar + _TRUNCATION_STEP
            and tail.trailing_flat_bars < SETTLEMENT_AUDIT_MIN_TRAILING_FLAT_BARS
        ):
            continue
        try:
            candidate = _derive_candidate_record(
                root, symbol, tail, end, stamped, committed_by_event, changed,
            )
        except _UnresolvedLifecycleError as exc:
            unresolved.append((symbol, str(exc)))
            continue
        if candidate is None:
            continue
        records.append(candidate)
        if candidate.price_source == "flat_1h_klines":
            flat_count += 1
        elif candidate.price_source == "twap30_proxy":
            twap_count += 1
        else:
            curated_count += 1
        _logger.debug(
            "[DATA] stage=build_settlement_registry symbol=%s event_id=%s price_source=%s",
            symbol, candidate.event_id, candidate.price_source,
        )
    carried = _carry_consistent_truncations(existing, three_dir, set(names))
    if symbols is not None:
        records.extend(record for record in existing.settlements if record.symbol not in set(names))
    registry = assemble_instrument_settlement_registry(records, carried)
    registry = parse_instrument_settlement_registry(
        settlement_registry_jsonl(registry), source="generated settlement registry",
    )
    _logger.info(
        "[DATA] stage=build_settlement_registry required=%d flat_1h_klines=%d"
        " twap30_proxy=%d curated=%d unresolved=%d changed=%d",
        len(records) + len(unresolved), flat_count, twap_count, curated_count,
        len(unresolved), len(changed),
    )
    return SettlementRegistryBuild(
        registry=registry, unresolved=tuple(unresolved), changed=tuple(changed),
    )


def write_settlement_registry(build: SettlementRegistryBuild, path: Path) -> int:
    """Atomically replace the registry file (tmp file + os.replace) with canonical JSONL.

    Returns:
        Number of records written.
    Raises:
        DataIntegrityError: ``build.unresolved`` is non-empty (never write a partial registry), or
            the serialized bytes fail ``parse_instrument_settlement_registry``.
    """
    if build.unresolved:
        names = ", ".join(symbol for symbol, _reason in build.unresolved)
        raise DataIntegrityError(
            f"settlement registry has {len(build.unresolved)} unresolved lifecycle(s)"
            f" ({names}); curate before writing",
        )
    payload = settlement_registry_jsonl(build.registry)
    parse_instrument_settlement_registry(payload, source=str(path))
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".settlements-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(tmp_name, target)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise
    return len(build.registry.settlements) + len(build.registry.truncations)


_HALT_STEP: pd.Timedelta = pd.Timedelta(minutes=3)


def build_venue_halt_registry(
    ohlcv_root: Path, *, horizon: pd.Timestamp, settlements: InstrumentSettlementRegistry,
    verified_at: pd.Timestamp,
) -> VenueHaltRegistry:
    """Detect exchange-wide halts from the full 3m cross-section.

    For every 3m label before ``horizon`` count present symbols and zero-quote-volume symbols,
    excluding each symbol's bars at or after its settlement ``last_trade_at`` (a delisted
    forward-filled tail must neither dilute nor fake a venue halt). A label is halted when
    ``present >= VENUE_HALT_MIN_PRESENT_SYMBOLS`` and ``zero >= VENUE_HALT_MIN_ZERO_FRACTION x present``;
    contiguous halted labels form one interval.

    Raises:
        DataIntegrityError: unreadable archive; horizon naive/non-UTC.
    """
    from src.mhs.params import VENUE_HALT_MIN_PRESENT_SYMBOLS, VENUE_HALT_MIN_ZERO_FRACTION
    from src.mhs.venue_halts import VenueHaltInterval, assemble_venue_halt_registry

    end = _require_utc_moment(horizon, "horizon")
    stamped = _require_utc_moment(verified_at, "verified_at")
    root = Path(ohlcv_root)
    three_dir = root / "3m" if (root / "3m").is_dir() else root
    if not three_dir.is_dir():
        raise DataIntegrityError(f"venue halt build unreadable: {three_dir}")
    try:
        archives = sorted(three_dir.glob("*.parquet"))
    except OSError as exc:
        raise DataIntegrityError(f"venue halt build unreadable: {three_dir}") from exc
    if not archives:
        raise DataIntegrityError(f"venue halt build unreadable: {three_dir}")

    zombie_floor: dict[str, int] = {}
    for record in settlements.settlements:
        prev = zombie_floor.get(record.symbol)
        cur = int(record.last_trade_at.value)
        if prev is None or cur < prev:
            zombie_floor[record.symbol] = cur

    horizon_ns = int(end.value)
    global_min: int | None = None
    step_ns = int(_HALT_STEP.value)
    present = np.zeros(0, dtype="int64")
    zero = np.zeros(0, dtype="int64")
    for count, path in enumerate(archives):
        if count % 100 == 0 and count:
            _logger.info("[DATA] stage=build_venue_halts scanned=%d", count)
        try:
            frame = pd.read_parquet(path, columns=["timestamp", "quote_vol"])
        except Exception as exc:
            raise DataIntegrityError(f"venue halt bars unreadable: {path.name}") from exc
        stamps_ms = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(dtype="float64")
        finite = np.isfinite(stamps_ms)
        if not finite.all() or bool((stamps_ms % (step_ns // 1_000_000) != 0).any()):
            raise DataIntegrityError(f"venue halt timestamps invalid or off-grid: {path.name}")
        labels_ns = stamps_ms.astype("int64") * 1_000_000
        qv = pd.to_numeric(frame["quote_vol"], errors="coerce").to_numpy(dtype="float64")
        keep = labels_ns < horizon_ns
        labels_ns = labels_ns[keep]
        qv = qv[keep]
        if labels_ns.size == 0:
            continue
        order = np.argsort(labels_ns, kind="stable")
        labels_ns = labels_ns[order]
        qv = qv[order]
        if bool((np.diff(labels_ns) == 0).any()):
            raise DataIntegrityError(f"venue halt duplicate symbol bars: {path.name}")
        if not np.isfinite(qv).all() or bool((qv < 0.0).any()):
            raise DataIntegrityError(f"venue halt quote volume invalid: {path.name}")
        start_ns = int(labels_ns[0]) // step_ns * step_ns
        if present.size == 0:
            global_min = start_ns
            size = (horizon_ns - start_ns + step_ns - 1) // step_ns
            present = np.zeros(size, dtype="int64")
            zero = np.zeros(size, dtype="int64")
        elif global_min is not None and start_ns < global_min:
            padding = (global_min - start_ns) // step_ns
            present = np.pad(present, (padding, 0))
            zero = np.pad(zero, (padding, 0))
            global_min = start_ns
        origin = start_ns if global_min is None else global_min
        qv_hit = qv
        floor = zombie_floor.get(path.stem)
        if floor is not None:
            live = labels_ns < floor
            labels_ns = labels_ns[live]
            qv_hit = qv_hit[live]
        pos = (labels_ns - origin) // step_ns
        np.add.at(present, pos, 1)
        np.add.at(zero, pos, (qv_hit == 0.0).astype("int64"))
    if global_min is None:
        registry = assemble_venue_halt_registry([])
        _logger.info("[DATA] stage=build_venue_halts halts=0 halted_bars=0")
        return registry
    grid = pd.date_range(
        pd.Timestamp(int(global_min), unit="ns", tz="UTC"), end, freq="3min", tz="UTC",
    )
    grid = grid[grid < end]
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(present > 0, zero / np.maximum(present, 1), 0.0)
    halted = (present >= VENUE_HALT_MIN_PRESENT_SYMBOLS) & (
        zero >= VENUE_HALT_MIN_ZERO_FRACTION * present
    )
    intervals: list[VenueHaltInterval] = []
    idx = np.flatnonzero(halted)
    if idx.size:
        run_start = int(idx[0])
        prev = int(idx[0])
        for cursor in idx[1:].tolist():
            cursor = int(cursor)
            if cursor == prev + 1:
                prev = cursor
                continue
            start = grid[run_start]
            halt_id = start.strftime("%Y-%m-%dT%H:%MZ")
            run_share = float(share[run_start : prev + 1].min())
            intervals.append(
                VenueHaltInterval(
                    halt_id=halt_id, start=start, end=grid[prev] + _HALT_STEP,
                    present_symbols=int(present[run_start]),
                    zero_symbols=int(zero[run_start]),
                    evidence=(
                        f"exchange-wide halt: {prev - run_start + 1} bars,"
                        f" present={int(present[run_start])},"
                        f" zero={int(zero[run_start])},"
                        f" min_zero_share={run_share:.4f}"
                    ),
                    verified_at=stamped,
                )
            )
            run_start = cursor
            prev = cursor
        start = grid[run_start]
        halt_id = start.strftime("%Y-%m-%dT%H:%MZ")
        run_share = float(share[run_start : prev + 1].min())
        intervals.append(
            VenueHaltInterval(
                halt_id=halt_id, start=start, end=grid[prev] + _HALT_STEP,
                present_symbols=int(present[run_start]),
                zero_symbols=int(zero[run_start]),
                evidence=(
                    f"exchange-wide halt: {prev - run_start + 1} bars,"
                    f" present={int(present[run_start])},"
                    f" zero={int(zero[run_start])},"
                    f" min_zero_share={run_share:.4f}"
                ),
                verified_at=stamped,
            )
        )
    registry = assemble_venue_halt_registry(intervals)
    halted_bars = int(halted.sum())
    _logger.info(
        "[DATA] stage=build_venue_halts halts=%d halted_bars=%d",
        len(intervals), halted_bars,
    )
    return registry


def write_venue_halt_registry(registry: VenueHaltRegistry, path: Path) -> int:
    """Atomic canonical JSONL write; re-parses before replace."""
    from src.mhs.venue_halts import parse_venue_halt_registry, venue_halt_registry_jsonl

    payload = venue_halt_registry_jsonl(registry)
    parse_venue_halt_registry(payload, source=str(path))
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".venue-halts-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(tmp_name, target)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise
    return len(registry.halts)
