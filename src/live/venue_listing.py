"""Point-in-time venue listing snapshots for the delisting lifecycle.

Pure over exchangeInfo payloads and disk; no network calls.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError

_REQUIRED_BAR_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
_MS_PER_HOUR = 3_600_000


@dataclass(frozen=True, slots=True)
class VenueListingEntry:
    """One exchangeInfo symbol's lifecycle facts as observed at a capture instant."""

    symbol: str
    status: str  # TRADING / PENDING_TRADING / SETTLING / CLOSE / ...
    contract_type: str  # PERPETUAL / CURRENT_QUARTER / TRADIFI_PERPETUAL / ...
    underlying_type: str  # COIN / EQUITY / ...
    quote_asset: str
    delivery_time: pd.Timestamp | None
    announced_delisting: bool
    delisting_first_seen_at: pd.Timestamp | None


@dataclass(frozen=True, slots=True)
class VenueListingSnapshot:
    """Point-in-time exchangeInfo lifecycle view persisted once per decision day."""

    captured_at: pd.Timestamp
    entries: Mapping[str, VenueListingEntry]
    #: Decision-day slot the snapshot was persisted under. A retry after 00:00 UTC re-captures
    #: slot D with ``captured_at`` on D+1, so applicability must follow the slot, not the capture
    #: instant; None only for in-memory snapshots that were never persisted.
    slot_day: pd.Timestamp | None = None


@dataclass(frozen=True, slots=True)
class SettlementEvidence:
    """Venue-published settlement price of a delivered (delisted) perpetual."""

    symbol: str
    delivery_time: pd.Timestamp
    price: Decimal
    flat_bars: int
    source: str  # "flat_1h_klines"


def _require_utc(stamp: pd.Timestamp, label: str) -> pd.Timestamp:
    ts = pd.Timestamp(stamp)
    if ts.tzinfo is None:
        raise DataIntegrityError(f"{label} must be timezone-aware UTC")
    return ts.tz_convert("UTC")


def _parse_delivery_ms(raw: Any) -> pd.Timestamp | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError("deliveryDate bool")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        raw = int(text)
    if isinstance(raw, float):
        if raw != raw or raw in (float("inf"), float("-inf")) or not raw.is_integer():
            raise ValueError("deliveryDate non-integer")
        raw = int(raw)
    if not isinstance(raw, int):
        raise ValueError("deliveryDate not epoch ms")
    return pd.Timestamp(int(raw), unit="ms", tz="UTC")


def parse_venue_listing(
    payload: Mapping[str, Any],
    *,
    captured_at: pd.Timestamp,
    previous: VenueListingSnapshot | None,
    announcement_horizon: pd.Timedelta,
) -> VenueListingSnapshot:
    """Normalize an exchangeInfo payload into lifecycle entries, carrying announcement first-seen times.

    A PERPETUAL whose ``deliveryDate`` falls within ``announcement_horizon`` of ``captured_at``
    is an announced delisting. ``delisting_first_seen_at`` is the earliest capture that saw the
    announcement, carried forward from ``previous``. This makes pre-delist blocking causal: a
    decision day earlier than the first sighting is never blocked retroactively. A malformed
    individual entry is skipped and counted, not raised, so one odd row cannot poison the listing.

    Args:
        payload: Raw ``/fapi/v1/exchangeInfo`` JSON object.
        captured_at: tz-aware UTC instant the payload was received.
        previous: Latest earlier snapshot, or None on first capture.
        announcement_horizon: Horizon separating an announced delivery from the far-future sentinel.

    Returns:
        Snapshot keyed by symbol.

    Raises:
        DataIntegrityError: payload has no ``symbols`` list, ``captured_at`` is naive, or no entry parses.
    """
    captured = _require_utc(captured_at, "captured_at")
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        raise DataIntegrityError("exchangeInfo payload has no symbols list")
    prev_entries: Mapping[str, VenueListingEntry] = previous.entries if previous is not None else {}
    entries: dict[str, VenueListingEntry] = {}
    for raw_entry in symbols:
        if not isinstance(raw_entry, Mapping):
            continue
        try:
            symbol = raw_entry.get("symbol")
            status = raw_entry.get("status")
            contract_type = raw_entry.get("contractType")
            if not isinstance(symbol, str) or not symbol:
                continue
            if not isinstance(status, str) or not status:
                continue
            if not isinstance(contract_type, str) or not contract_type:
                continue
            delivery_time = _parse_delivery_ms(raw_entry.get("deliveryDate"))
        except (ValueError, TypeError, OverflowError):
            continue
        announced = (
            contract_type == "PERPETUAL" and delivery_time is not None and delivery_time <= captured + announcement_horizon
        )
        first_seen: pd.Timestamp | None = None
        if announced:
            prev = prev_entries.get(symbol)
            if prev is not None and prev.delisting_first_seen_at is not None:
                first_seen = prev.delisting_first_seen_at
            else:
                first_seen = captured
        entries[symbol] = VenueListingEntry(
            symbol=symbol,
            status=str(status),
            contract_type=str(contract_type),
            underlying_type=str(raw_entry.get("underlyingType", "COIN")),
            quote_asset=str(raw_entry.get("quoteAsset", "USDT")),
            delivery_time=delivery_time,
            announced_delisting=announced,
            delisting_first_seen_at=first_seen,
        )
    if not entries:
        raise DataIntegrityError("exchangeInfo payload yields no listing entry")
    return VenueListingSnapshot(captured_at=captured, entries=entries)


def _entry_to_json(entry: VenueListingEntry) -> dict[str, Any]:
    return {
        "symbol": entry.symbol,
        "status": entry.status,
        "contract_type": entry.contract_type,
        "underlying_type": entry.underlying_type,
        "quote_asset": entry.quote_asset,
        "delivery_time": entry.delivery_time.isoformat() if entry.delivery_time is not None else None,
        "announced_delisting": entry.announced_delisting,
        "delisting_first_seen_at": entry.delisting_first_seen_at.isoformat()
        if entry.delisting_first_seen_at is not None
        else None,
    }


def _entry_from_json(raw: Any) -> VenueListingEntry:
    if not isinstance(raw, Mapping):
        raise DataIntegrityError("listing snapshot entry malformed")
    try:
        symbol = raw["symbol"]
        status = raw["status"]
        contract_type = raw["contract_type"]
        if not isinstance(symbol, str) or not symbol:
            raise DataIntegrityError("listing snapshot entry malformed")
        if not isinstance(status, str) or not status:
            raise DataIntegrityError("listing snapshot entry malformed")
        if not isinstance(contract_type, str) or not contract_type:
            raise DataIntegrityError("listing snapshot entry malformed")
        delivery_raw = raw.get("delivery_time")
        first_seen_raw = raw.get("delisting_first_seen_at")
        try:
            delivery_time = pd.Timestamp(delivery_raw).tz_convert("UTC") if delivery_raw is not None else None
            first_seen = pd.Timestamp(first_seen_raw).tz_convert("UTC") if first_seen_raw is not None else None
        except (ValueError, TypeError) as exc:
            raise DataIntegrityError("listing snapshot entry malformed") from exc
        announced = raw["announced_delisting"]
        if not isinstance(announced, bool):
            raise DataIntegrityError("listing snapshot entry malformed")
        return VenueListingEntry(
            symbol=symbol,
            status=status,
            contract_type=contract_type,
            underlying_type=str(raw.get("underlying_type", "COIN")),
            quote_asset=str(raw.get("quote_asset", "USDT")),
            delivery_time=delivery_time,
            announced_delisting=announced,
            delisting_first_seen_at=first_seen,
        )
    except KeyError as exc:
        raise DataIntegrityError("listing snapshot entry malformed") from exc


def _snapshot_to_json(snapshot: VenueListingSnapshot) -> dict[str, Any]:
    return {
        "captured_at": snapshot.captured_at.isoformat(),
        "entries": [_entry_to_json(snapshot.entries[s]) for s in sorted(snapshot.entries)],
    }


def _snapshot_from_json(raw: Any) -> VenueListingSnapshot:
    if not isinstance(raw, Mapping):
        raise DataIntegrityError("listing snapshot file malformed")
    try:
        captured_raw = raw["captured_at"]
        entries_raw = raw["entries"]
    except KeyError as exc:
        raise DataIntegrityError("listing snapshot file malformed") from exc
    try:
        captured_at = pd.Timestamp(str(captured_raw)).tz_convert("UTC")
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError("listing snapshot file malformed") from exc
    if not isinstance(entries_raw, list) or not entries_raw:
        raise DataIntegrityError("listing snapshot file malformed")
    entries = {entry.symbol: entry for entry in (_entry_from_json(item) for item in entries_raw)}
    return VenueListingSnapshot(captured_at=captured_at, entries=entries)


def _read_snapshot_file(path: Path) -> VenueListingSnapshot:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            raw: Any = json.load(handle)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise DataIntegrityError(f"listing snapshot unreadable: {path.name}") from exc
    return _snapshot_from_json(raw)


def write_venue_listing_snapshot(snapshot: VenueListingSnapshot, root: Path, *, slot_day: pd.Timestamp) -> Path:
    """Persist ``<root>/<YYYYMMDD>.json.gz`` atomically for the decision day that captured it.

    Re-capturing an existing slot on a retry is allowed only if the new snapshot does not move
    any ``delisting_first_seen_at`` later. The slot keeps the earliest first-seen evidence.

    Raises:
        DataIntegrityError: ``slot_day`` naive, or an attempted overwrite would move a first-seen time later.
    """
    slot = _require_utc(slot_day, "slot_day")
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    path = root_path / f"{slot.strftime('%Y%m%d')}.json.gz"
    if path.exists():
        existing = _read_snapshot_file(path)
        for symbol, entry in snapshot.entries.items():
            if entry.delisting_first_seen_at is None:
                continue
            old = existing.entries.get(symbol)
            if (
                old is not None
                and old.delisting_first_seen_at is not None
                and entry.delisting_first_seen_at > old.delisting_first_seen_at
            ):
                raise DataIntegrityError(f"listing slot {path.name} would move first-seen later for {symbol}")
    payload = json.dumps(_snapshot_to_json(snapshot), separators=(",", ":"))
    tmp_path = root_path / f".{path.name}.{os.getpid()}.tmp"
    with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
        handle.write(payload)
    os.replace(tmp_path, path)
    return path


def _slot_files(root: Path) -> list[tuple[str, Path]]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    found: list[tuple[str, Path]] = []
    for child in sorted(root_path.glob("*.json.gz")):
        stem = child.name.removesuffix(".json.gz")
        if len(stem) == 8 and stem.isdigit():
            found.append((stem, child))
    return found


def load_venue_listing_history(root: Path, *, through_day: pd.Timestamp) -> tuple[VenueListingSnapshot, ...]:
    """Load every listing snapshot whose slot day is on or before ``through_day``, ordered by slot.

    Raises:
        DataIntegrityError: an existing snapshot file is unreadable or malformed.
    """
    through = _require_utc(through_day, "through_day")
    through_slot = through.strftime("%Y%m%d")
    snapshots: list[VenueListingSnapshot] = []
    for stem, path in _slot_files(root):
        if stem <= through_slot:
            snapshot = _read_snapshot_file(path)
            snapshots.append(replace(snapshot, slot_day=pd.Timestamp(stem, tz="UTC")))
    return tuple(snapshots)


def latest_venue_listing_or_none(root: Path) -> VenueListingSnapshot | None:
    """Newest listing snapshot without an age check (first-seen carry-forward source), or None when none exists.

    Raises:
        DataIntegrityError: the newest existing snapshot is unreadable or malformed.
    """
    files = _slot_files(root)
    if not files:
        return None
    return _read_snapshot_file(files[-1][1])


def latest_venue_listing(
    root: Path, *, now: pd.Timestamp, max_age: pd.Timedelta,
) -> VenueListingSnapshot:
    """Newest listing snapshot, required to be no older than ``max_age`` at ``now``.

    Raises:
        DataIntegrityError: no snapshot exists or the newest is older than ``max_age``.
    """
    now_utc = _require_utc(now, "now")
    newest = latest_venue_listing_or_none(root)
    if newest is None:
        raise DataIntegrityError("no venue listing snapshot exists")
    if now_utc - newest.captured_at > max_age:
        raise DataIntegrityError("venue listing snapshot is stale")
    return newest


def _as_utc_instant(stamp: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(stamp)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _as_utc_day(stamp: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(stamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").normalize()


def delisting_blocked_decisions(
    history: Sequence[VenueListingSnapshot],
    decision_index: pd.DatetimeIndex,
    census: Sequence[str],
    *,
    holding_end_offset: pd.Timedelta,
    lead: pd.Timedelta,
) -> pd.DataFrame:
    """Boolean decision-day frame withdrawing announced-delisting symbols from roster and targets.

    Decision day ``d`` is blocked for symbol ``s`` when the snapshot in force at ``d`` (latest
    slot on or before ``d``) marks ``s`` as announced, ``d`` is on or after the UTC day of
    ``delisting_first_seen_at``, and ``d + holding_end_offset + lead >= delivery_time``. The
    same frame shape and meaning as the research ``blocked_decisions`` contract (daily UTC
    index identical to ``decision_index``, columns in ``census`` order) lets it pass straight
    into ``build_frozen_mhs_candidate``.

    Returns:
        Frame of bool, all False where no snapshot is in force.
    """
    names = [str(symbol) for symbol in census]
    def _applies_from(snapshot: VenueListingSnapshot) -> pd.Timestamp:
        return _as_utc_day(snapshot.slot_day if snapshot.slot_day is not None else snapshot.captured_at)

    ordered = sorted(history, key=lambda snapshot: (_applies_from(snapshot), snapshot.captured_at))
    slot_days = [_applies_from(snapshot) for snapshot in ordered]

    values = np.zeros((len(decision_index), len(names)), dtype=bool)
    if ordered and names:
        for row, day in enumerate(decision_index):
            day_ts = _as_utc_instant(pd.Timestamp(day))
            day_floor = day_ts.normalize()
            in_force: VenueListingSnapshot | None = None
            for slot_day, snapshot in zip(slot_days, ordered, strict=True):
                if slot_day <= day_floor:
                    in_force = snapshot
                else:
                    break
            if in_force is None:
                continue
            horizon_end = day_ts + holding_end_offset + lead
            for col, name in enumerate(names):
                entry = in_force.entries.get(name)
                if entry is None or not entry.announced_delisting:
                    continue
                if entry.delivery_time is None or entry.delisting_first_seen_at is None:
                    continue
                if day_floor < _as_utc_day(entry.delisting_first_seen_at):
                    continue
                if horizon_end >= entry.delivery_time:
                    values[row, col] = True
    frame = pd.DataFrame(values, index=decision_index, columns=names)
    return frame.astype(bool)


def settlement_evidence_from_bars(
    hourly_path: Path,
    *,
    symbol: str,
    delivery_time: pd.Timestamp,
    min_flat_bars: int,
    price_rtol: float,
) -> SettlementEvidence | None:
    """Settlement price read from the venue's post-delivery flat 1h klines, or None when not yet evidenced.

    After delivery the venue publishes zero-volume bars with open == high == low == close at the
    settlement price. Only a contiguous run of at least ``min_flat_bars`` such bars, starting at or
    after ``delivery_time`` and agreeing on close within ``price_rtol``, counts as evidence. Any
    traded bar after delivery, a non-flat bar, or disagreement returns None. The price is
    therefore never synthesized from the last traded close, a mark candle, or zero.

    Raises:
        DataIntegrityError: file unreadable or lacks timestamp/open/high/low/close/volume columns.
    """
    delivery = _require_utc(delivery_time, "delivery_time")
    if min_flat_bars < 1:
        raise DataIntegrityError("min_flat_bars must be >= 1")
    path = Path(hourly_path)
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise DataIntegrityError(f"settlement bars unreadable: {path.name}") from exc
    if not all(col in frame.columns for col in _REQUIRED_BAR_COLUMNS):
        raise DataIntegrityError(f"settlement bars lack required columns: {path.name}")
    if frame.empty:
        return None
    stamps = pd.to_numeric(frame["timestamp"], errors="coerce")
    work = pd.DataFrame(
        {
            "timestamp": stamps,
            "open": pd.to_numeric(frame["open"], errors="coerce"),
            "high": pd.to_numeric(frame["high"], errors="coerce"),
            "low": pd.to_numeric(frame["low"], errors="coerce"),
            "close": pd.to_numeric(frame["close"], errors="coerce"),
            "volume": pd.to_numeric(frame["volume"], errors="coerce"),
        }
    ).dropna()
    if work.empty:
        return None
    work = work.sort_values("timestamp", kind="mergesort")
    delivery_ms = int(delivery.value // 1_000_000)
    post = work.loc[work["timestamp"] >= delivery_ms]
    if len(post) < min_flat_bars:
        return None
    post_times = [int(v) for v in post["timestamp"].tolist()]
    for prev_ms, cur_ms in pairwise(post_times):
        if cur_ms - prev_ms != _MS_PER_HOUR:
            return None
    for _, row in post.iterrows():
        if float(row["volume"]) != 0.0:
            return None
        o, h, low, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        if not (o == h == low == c):
            return None
    closes = [float(v) for v in post["close"].tolist()]
    ref = closes[0]
    if ref == 0.0:
        if any(c != 0.0 for c in closes):
            return None
    elif max(abs(c - ref) for c in closes) / abs(ref) > price_rtol:
        return None
    return SettlementEvidence(
        symbol=str(symbol),
        delivery_time=delivery,
        price=Decimal(str(ref)),
        flat_bars=len(post),
        source="flat_1h_klines",
    )
