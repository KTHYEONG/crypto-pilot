"""Point-in-time snapshots of Binance USD-M public REST state that no archive preserves:
all-symbol top-of-book, premium index / predicted funding, and raw daily reference payloads."""

from __future__ import annotations

import gzip
import json
import logging
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.parquet_io import read_parquet_or_quarantine, write_parquet_atomic

_logger = logging.getLogger(__name__)

BOOK_TICKER_URL: str = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"
PREMIUM_INDEX_URL: str = "https://fapi.binance.com/fapi/v1/premiumIndex"
REFERENCE_URLS: Mapping[str, str] = {
    "exchange_info": "https://fapi.binance.com/fapi/v1/exchangeInfo",
    "funding_info": "https://fapi.binance.com/fapi/v1/fundingInfo",
    "asset_index": "https://fapi.binance.com/fapi/v1/assetIndex",
}

BOOK_TICKER_DATASET: str = "book_ticker"
PREMIUM_INDEX_DATASET: str = "premium_index"
REFERENCE_DIRNAME: str = "reference"

BOOK_TICKER_COLUMNS: tuple[str, ...] = (
    "captured_at",
    "symbol",
    "exchange_time_ms",
    "fetched_at_ms",
    "bid_px",
    "bid_qty",
    "ask_px",
    "ask_qty",
)
PREMIUM_INDEX_COLUMNS: tuple[str, ...] = (
    "captured_at",
    "symbol",
    "exchange_time_ms",
    "fetched_at_ms",
    "mark_price",
    "index_price",
    "estimated_settle_price",
    "last_funding_rate",
    "interest_rate",
    "next_funding_time_ms",
)
NULLABLE_MS_COLUMNS: frozenset[str] = frozenset({"fetched_at_ms", "next_funding_time_ms"})

REJECTED_SYMBOL_SAMPLE: int = 10

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


@dataclass(frozen=True, slots=True)
class SnapshotParse:
    """Result of normalizing one all-symbol snapshot payload with per-row isolation.

    All-symbol endpoints mix instruments in every lifecycle state (trading, settling, settled
    delivery contracts, pre-listing), so one row that violates the row contract is excluded and
    counted instead of discarding every other symbol's observation — the snapshot cannot be
    re-fetched later. Rejections stay observable through the counters so a venue schema drift is
    still detected.

    Attributes:
        frame: Accepted rows in the dataset's canonical column order and dtypes.
        total_rows: Number of list items in the payload.
        rejected_rows: Number of items excluded by the row contract.
        rejected_reasons: Reason token -> count, e.g. ``{"missing_time": 1, "negative_price": 2}``.
        rejected_symbols: Up to ``REJECTED_SYMBOL_SAMPLE`` distinct symbols of rejected rows, in
            payload order (``"<unknown>"`` when the row had no usable symbol).
    """

    frame: pd.DataFrame
    total_rows: int
    rejected_rows: int
    rejected_reasons: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    rejected_symbols: tuple[str, ...] = ()


def next_grid_time(now: pd.Timestamp, interval_s: int) -> pd.Timestamp:
    """Earliest UTC instant strictly after ``now`` lying on the ``interval_s`` wall-clock grid.

    Snapshots are sampled on a fixed epoch-aligned grid so samples from different days, restarts and
    datasets share timestamps; a missed grid point is an observable gap, never a shifted sample.

    Args:
        now: tz-aware timestamp.
        interval_s: grid spacing in seconds; must divide 86400.

    Raises:
        ValueError: naive ``now`` or ``interval_s`` not a positive divisor of 86400.
    """
    ts = pd.Timestamp(now)
    if ts.tzinfo is None:
        raise ValueError("next_grid_time requires tz-aware now")
    if interval_s <= 0 or 86400 % interval_s != 0:
        raise ValueError("interval_s must be a positive divisor of 86400")
    utc = ts.tz_convert("UTC")
    interval_ns = int(interval_s) * 1_000_000_000
    ns = int(utc.value)
    nxt = (ns // interval_ns + 1) * interval_ns
    return pd.Timestamp(nxt, unit="ns", tz="UTC")


def _utc_captured(captured_at: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(captured_at)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _utc_fetched(fetched_at: pd.Timestamp, captured_at: pd.Timestamp) -> pd.Timestamp:
    """Validate the response-receipt instant against the grid label it is filed under."""
    cap = _utc_captured(captured_at)
    ts = pd.Timestamp(fetched_at)
    if ts.tzinfo is None:
        raise DataIntegrityError("snapshot fetched_at is naive or missing")
    out = ts.tz_convert("UTC")
    if out < cap:
        raise DataIntegrityError("snapshot fetched_at precedes captured_at")
    return out


def _strict_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _strict_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return int(out)


def _row_symbol(row: Mapping[str, Any]) -> str | None:
    raw = row.get("symbol")
    if raw is None:
        return None
    text = str(raw).strip()
    return text if text else None


def _rejection_error(
    dataset: str, total: int, rejected: int, reasons: dict[str, int], symbols: list[str]
) -> DataIntegrityError:
    hist = ",".join(f"{key}:{reasons[key]}" for key in sorted(reasons))
    sample = ",".join(symbols[:REJECTED_SYMBOL_SAMPLE])
    return DataIntegrityError(
        f"{dataset} snapshot rejected: {rejected}/{total} rows rejected "
        f"exceeds threshold; reasons={hist}; symbols={sample}"
    )


def _finalize_snapshot(
    dataset: str,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    total: int,
    reasons: dict[str, int],
    rejected_order: list[str],
    max_rejected_fraction: float,
) -> SnapshotParse:
    rejected = sum(reasons.values())
    seen: list[str] = []
    seen_set: set[str] = set()
    for sym in rejected_order:
        if sym not in seen_set:
            seen_set.add(sym)
            seen.append(sym)
        if len(seen) >= REJECTED_SYMBOL_SAMPLE:
            break
    if not rows:
        raise _rejection_error(dataset, total, rejected, reasons, seen)
    if rejected / total > max_rejected_fraction:
        raise _rejection_error(dataset, total, rejected, reasons, seen)
    frame = pd.DataFrame(rows, columns=list(columns))
    return SnapshotParse(
        frame=frame,
        total_rows=total,
        rejected_rows=rejected,
        rejected_reasons=MappingProxyType(dict(reasons)),
        rejected_symbols=tuple(seen),
    )


def _book_row_reject(row: Any, accepted: set[str]) -> tuple[str | None, dict[str, Any] | None, str]:
    if not isinstance(row, Mapping):
        return "malformed_row", None, "<unknown>"
    symbol = _row_symbol(row)
    if symbol is None:
        return "missing_symbol", None, "<unknown>"
    key_reasons = (
        ("bidPrice", "missing_bid_price"),
        ("bidQty", "missing_bid_qty"),
        ("askPrice", "missing_ask_price"),
        ("askQty", "missing_ask_qty"),
        ("time", "missing_time"),
    )
    for key, token in key_reasons:
        if key not in row:
            return token, None, symbol
    values: dict[str, float] = {}
    for key, token in (
        ("bidPrice", "non_numeric_bid_price"),
        ("bidQty", "non_numeric_bid_qty"),
        ("askPrice", "non_numeric_ask_price"),
        ("askQty", "non_numeric_ask_qty"),
    ):
        parsed = _strict_float(row[key])
        if parsed is None:
            return token, None, symbol
        values[key] = parsed
    bid_px = values["bidPrice"]
    ask_px = values["askPrice"]
    bid_qty = values["bidQty"]
    ask_qty = values["askQty"]
    if bid_px < 0 or ask_px < 0:
        return "negative_price", None, symbol
    if bid_qty < 0 or ask_qty < 0:
        return "negative_qty", None, symbol
    exchange_ms = _strict_ms(row["time"])
    if exchange_ms is None or exchange_ms <= 0:
        return "invalid_time", None, symbol
    if symbol in accepted:
        return "duplicate_symbol", None, symbol
    stored_bid = float("nan") if bid_px == 0 else bid_px
    stored_ask = float("nan") if ask_px == 0 else ask_px
    return None, {"exchange_time_ms": exchange_ms, "bid_px": stored_bid, "ask_px": stored_ask,
        "bid_qty": bid_qty, "ask_qty": ask_qty}, symbol


def parse_book_ticker_payload(
    payload: Any,
    *,
    captured_at: pd.Timestamp,
    fetched_at: pd.Timestamp,
    max_rejected_fraction: float,
) -> SnapshotParse:
    """Normalize an all-symbol ``/fapi/v1/ticker/bookTicker`` response into ``BOOK_TICKER_COLUMNS``.

    ``captured_at`` is the wall-clock grid instant the sample is filed under (a join key shared by all
    datasets); ``fetched_at`` is when the response was fully received and is the earliest instant the
    data was observable, so point-in-time consumers must gate on ``fetched_at_ms``, never on
    ``captured_at``. ``exchange_time_ms`` is the venue's own quote timestamp, kept so clock skew and
    quote staleness remain measurable.

    A zero price is the venue's encoding of an empty book side (e.g. a settled delivery contract),
    so it is stored as NaN. Rows violating the row contract are excluded and counted rather than
    rejecting the whole all-symbol snapshot, because one delisting or settled instrument must not
    erase every other symbol's unrecoverable observation.

    Args:
        payload: Decoded JSON response.
        captured_at: Grid label (tz-aware or naive UTC).
        fetched_at: Response receipt instant (tz-aware, not before ``captured_at``).
        max_rejected_fraction: Largest tolerated ``rejected_rows / total_rows``; above it the payload
            is treated as a structural failure (venue schema drift), not as isolated bad rows.

    Returns:
        ``SnapshotParse`` whose frame has one row per accepted symbol; prices float64 (NaN for an
        empty side), quantities float32, ``exchange_time_ms`` int64, ``fetched_at_ms`` Int64,
        ``captured_at`` datetime64[ns, UTC], ``symbol`` string.

    Raises:
        DataIntegrityError: payload is not a non-empty list; ``fetched_at`` is naive or precedes
            ``captured_at``; no row is accepted; or the rejected fraction exceeds
            ``max_rejected_fraction``. The message carries the reason histogram and symbol sample.
    """
    if not isinstance(payload, list) or not payload:
        raise DataIntegrityError("book ticker payload must be a non-empty list")
    cap = _utc_captured(captured_at)
    fetched = _utc_fetched(fetched_at, captured_at)
    fetched_ms = int(fetched.value // 1_000_000)
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    rejected_order: list[str] = []
    accepted: set[str] = set()
    for item in payload:
        reason, parsed, symbol = _book_row_reject(item, accepted)
        if reason is not None or parsed is None:
            assert reason is not None
            reasons[reason] = reasons.get(reason, 0) + 1
            rejected_order.append(symbol)
            continue
        accepted.add(symbol)
        rows.append(
            {
                "captured_at": cap,
                "symbol": symbol,
                "exchange_time_ms": parsed["exchange_time_ms"],
                "fetched_at_ms": fetched_ms,
                "bid_px": parsed["bid_px"],
                "bid_qty": parsed["bid_qty"],
                "ask_px": parsed["ask_px"],
                "ask_qty": parsed["ask_qty"],
            }
        )
    parsed_all = _finalize_snapshot(
        "book ticker", rows, BOOK_TICKER_COLUMNS, len(payload), reasons, rejected_order,
        max_rejected_fraction,
    )
    frame = parsed_all.frame
    frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True)
    frame["symbol"] = frame["symbol"].astype("string")
    frame["exchange_time_ms"] = frame["exchange_time_ms"].astype("int64")
    frame["fetched_at_ms"] = frame["fetched_at_ms"].astype("Int64")
    frame["bid_px"] = frame["bid_px"].astype("float64")
    frame["ask_px"] = frame["ask_px"].astype("float64")
    frame["bid_qty"] = frame["bid_qty"].astype("float32")
    frame["ask_qty"] = frame["ask_qty"].astype("float32")
    return SnapshotParse(
        frame=frame,
        total_rows=parsed_all.total_rows,
        rejected_rows=parsed_all.rejected_rows,
        rejected_reasons=parsed_all.rejected_reasons,
        rejected_symbols=parsed_all.rejected_symbols,
    )


def _premium_row_reject(row: Any, accepted: set[str]) -> tuple[str | None, dict[str, Any] | None, str]:
    if not isinstance(row, Mapping):
        return "malformed_row", None, "<unknown>"
    symbol = _row_symbol(row)
    if symbol is None:
        return "missing_symbol", None, "<unknown>"
    if "markPrice" not in row or row["markPrice"] is None or (
        isinstance(row["markPrice"], str) and row["markPrice"].strip() == ""
    ):
        return "missing_mark_price", None, symbol
    mark_price = _strict_float(row["markPrice"])
    if mark_price is None:
        return "non_numeric_mark_price", None, symbol
    if mark_price <= 0:
        return "non_positive_mark_price", None, symbol
    if "time" not in row or row["time"] is None or (
        isinstance(row["time"], str) and row["time"].strip() == ""
    ):
        return "missing_time", None, symbol
    exchange_ms = _strict_ms(row["time"])
    if exchange_ms is None or exchange_ms <= 0:
        return "invalid_time", None, symbol
    if "nextFundingTime" not in row or row["nextFundingTime"] is None or (
        isinstance(row["nextFundingTime"], str) and row["nextFundingTime"].strip() == ""
    ):
        return "missing_next_funding_time", None, symbol
    next_funding_ms = _strict_ms(row["nextFundingTime"])
    if next_funding_ms is None or next_funding_ms < 0:
        return "invalid_next_funding_time", None, symbol
    optionals: dict[str, float] = {}
    for key, token in (
        ("indexPrice", "non_numeric_index_price"),
        ("estimatedSettlePrice", "non_numeric_estimated_settle_price"),
        ("lastFundingRate", "non_numeric_last_funding_rate"),
        ("interestRate", "non_numeric_interest_rate"),
    ):
        if key not in row or row[key] is None or (
            isinstance(row[key], str) and str(row[key]).strip() == ""
        ):
            optionals[key] = float("nan")
            continue
        parsed = _strict_float(row[key])
        if parsed is None:
            return token, None, symbol
        optionals[key] = parsed
    if symbol in accepted:
        return "duplicate_symbol", None, symbol
    index_price = optionals["indexPrice"]
    estimated = optionals["estimatedSettlePrice"]
    funding = optionals["lastFundingRate"]
    interest = optionals["interestRate"]
    if next_funding_ms == 0:
        next_stored: int | None = None
        funding = float("nan")
        interest = float("nan")
    else:
        next_stored = next_funding_ms
    if not math.isnan(index_price) and index_price <= 0:
        index_price = float("nan")
    if not math.isnan(estimated) and estimated <= 0:
        estimated = float("nan")
    return None, {
        "exchange_time_ms": exchange_ms,
        "next_funding_time_ms": next_stored,
        "mark_price": mark_price,
        "index_price": index_price,
        "estimated_settle_price": estimated,
        "last_funding_rate": funding,
        "interest_rate": interest,
    }, symbol


def parse_premium_index_payload(
    payload: Any,
    *,
    captured_at: pd.Timestamp,
    fetched_at: pd.Timestamp,
    max_rejected_fraction: float,
) -> SnapshotParse:
    """Normalize an all-symbol ``/fapi/v1/premiumIndex`` response into ``PREMIUM_INDEX_COLUMNS``.

    ``last_funding_rate`` is the venue's running estimate for the next settlement (the realized rate is
    archived separately by the funding history endpoint); keeping the estimate path lets research test
    predicted-funding signals without reconstructing the premium-index clamp. ``captured_at`` is the
    grid label and ``fetched_at_ms`` the availability time (see ``parse_book_ticker_payload``).

    The venue lists settled, settling and delivery instruments with placeholder zeros: a
    ``nextFundingTime`` of 0 means no funding is scheduled, so the funding estimate and interest rate
    of that row are placeholders and are stored as null, and a non-positive settlement or index price
    is never a real price. Rows violating the row contract are excluded and counted rather than
    rejecting the whole snapshot.

    Args:
        payload: Decoded JSON response.
        captured_at: Grid label.
        fetched_at: Response receipt instant.
        max_rejected_fraction: See ``parse_book_ticker_payload``.

    Returns:
        ``SnapshotParse``. In its frame, rates and prices are float64 (NaN for blanks and placeholders),
        ``exchange_time_ms`` is int64, and ``next_funding_time_ms`` and ``fetched_at_ms`` are Int64
        (null when no funding is scheduled). ``captured_at`` is datetime64[ns, UTC].

    Raises:
        DataIntegrityError: Same structural conditions as ``parse_book_ticker_payload``.
    """
    if not isinstance(payload, list) or not payload:
        raise DataIntegrityError("premium index payload must be a non-empty list")
    cap = _utc_captured(captured_at)
    fetched = _utc_fetched(fetched_at, captured_at)
    fetched_ms = int(fetched.value // 1_000_000)
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    rejected_order: list[str] = []
    accepted: set[str] = set()
    for item in payload:
        reason, parsed, symbol = _premium_row_reject(item, accepted)
        if reason is not None or parsed is None:
            assert reason is not None
            reasons[reason] = reasons.get(reason, 0) + 1
            rejected_order.append(symbol)
            continue
        accepted.add(symbol)
        rows.append(
            {
                "captured_at": cap,
                "symbol": symbol,
                "exchange_time_ms": parsed["exchange_time_ms"],
                "fetched_at_ms": fetched_ms,
                "mark_price": parsed["mark_price"],
                "index_price": parsed["index_price"],
                "estimated_settle_price": parsed["estimated_settle_price"],
                "last_funding_rate": parsed["last_funding_rate"],
                "interest_rate": parsed["interest_rate"],
                "next_funding_time_ms": parsed["next_funding_time_ms"],
            }
        )
    parsed_all = _finalize_snapshot(
        "premium index", rows, PREMIUM_INDEX_COLUMNS, len(payload), reasons, rejected_order,
        max_rejected_fraction,
    )
    frame = parsed_all.frame
    frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True)
    frame["symbol"] = frame["symbol"].astype("string")
    frame["exchange_time_ms"] = frame["exchange_time_ms"].astype("int64")
    frame["fetched_at_ms"] = pd.to_numeric(frame["fetched_at_ms"], errors="coerce").astype("Int64")
    frame["next_funding_time_ms"] = pd.to_numeric(
        frame["next_funding_time_ms"], errors="coerce"
    ).astype("Int64")
    for col in ("mark_price", "index_price", "estimated_settle_price", "last_funding_rate", "interest_rate"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return SnapshotParse(
        frame=frame,
        total_rows=parsed_all.total_rows,
        rejected_rows=parsed_all.rejected_rows,
        rejected_reasons=parsed_all.rejected_reasons,
        rejected_symbols=parsed_all.rejected_symbols,
    )


def _enforce_snapshot_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "captured_at" in out.columns:
        out["captured_at"] = pd.to_datetime(out["captured_at"], utc=True)
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype("string")
    for col in out.columns:
        if col in ("captured_at", "symbol"):
            continue
        if col in NULLABLE_MS_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        elif col.endswith("_ms"):
            out[col] = pd.to_numeric(out[col], errors="raise").astype("int64")
        elif col.endswith("_qty"):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
        else:
            with pd.option_context("mode.string_storage", "python"):
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def write_hourly_partition(frame: pd.DataFrame, root: Path, dataset: str) -> list[Path]:
    """Merge snapshot rows into ``<root>/<dataset>/<YYYYMMDD>/<HH>.parquet`` keyed by ``captured_at`` hour.

    Hour partitions bound every rewrite to at most one hour of rows and make every closed hour an
    immutable file, so the nightly ``rclone copy`` transfers each finished file exactly once. Each hour
    is atomically replaced; an undecodable existing hour file is quarantined instead of blocking every
    later flush of that hour. Assumes a single writer process per dataset.

    Returns:
        Sorted list of partition files written.

    Raises:
        DataIntegrityError: ``frame`` lacks ``captured_at``/``symbol`` or ``captured_at`` is naive.
        Exception: Merge failures and OS-level read/write failures propagate with the existing hour
            file unchanged.
    """
    if "captured_at" not in frame.columns or "symbol" not in frame.columns:
        raise DataIntegrityError("snapshot frame lacks captured_at/symbol")
    if len(frame) == 0:
        return []
    for value in frame["captured_at"]:
        if pd.isna(value):
            raise DataIntegrityError("snapshot captured_at is naive or missing")
        if pd.Timestamp(value).tzinfo is None:
            raise DataIntegrityError("snapshot captured_at is naive or missing")
    stamps = pd.to_datetime(frame["captured_at"], utc=True, errors="coerce")
    work = frame.copy()
    work["captured_at"] = stamps
    hours = work["captured_at"].dt.floor("h")
    written: list[Path] = []
    base = Path(root) / dataset
    for hour_ts, group in work.groupby(hours):
        hour = pd.Timestamp(hour_ts).tz_convert("UTC")
        day = hour.strftime("%Y%m%d")
        target = base / day / f"{hour.strftime('%H')}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        chunk = group.drop(columns=[c for c in group.columns if c not in frame.columns])
        chunk = _enforce_snapshot_dtypes(chunk)
        existing = read_parquet_or_quarantine(target, stage=f"snapshot_{dataset}")
        if existing is not None:
            existing["captured_at"] = pd.to_datetime(existing["captured_at"], utc=True)
            combined = pd.concat([existing, chunk], ignore_index=True)
            combined = _enforce_snapshot_dtypes(combined)
        else:
            combined = chunk
        combined = combined.drop_duplicates(subset=["symbol", "captured_at"], keep="last")
        combined = combined.sort_values(["symbol", "captured_at"]).reset_index(drop=True)
        combined = _enforce_snapshot_dtypes(combined)
        write_parquet_atomic(combined, target, compression="zstd")
        written.append(target)
    return sorted(written)


def load_snapshot_dataset(
    root: Path, dataset: str, *, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Read ``[start, end)`` rows of one snapshot dataset across hour partitions (research reader).

    Returns:
        Rows sorted by (captured_at, symbol); an empty DataFrame when no partition overlaps.
    """
    start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
    if end_ts <= start_ts:
        return pd.DataFrame()
    base = Path(root) / dataset
    if not base.exists():
        return pd.DataFrame()
    cursor = start_ts.floor("h")
    last = (end_ts - pd.Timedelta(nanoseconds=1)).floor("h")
    frames: list[pd.DataFrame] = []
    cur = cursor
    while cur <= last:
        path = base / cur.strftime("%Y%m%d") / f"{cur.strftime('%H')}.parquet"
        if path.exists():
            try:
                part = pd.read_parquet(path)
            except Exception as exc:
                _logger.debug("snapshot skip unreadable %s: %s", path, exc)
                cur += pd.Timedelta(hours=1)
                continue
            if not part.empty and "captured_at" in part.columns:
                part["captured_at"] = pd.to_datetime(part["captured_at"], utc=True)
                mask = (part["captured_at"] >= start_ts) & (part["captured_at"] < end_ts)
                part = part.loc[mask]
                if not part.empty:
                    frames.append(part)
        cur += pd.Timedelta(hours=1)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["captured_at"] = pd.to_datetime(out["captured_at"], utc=True)
    out = out.sort_values(["captured_at", "symbol"]).reset_index(drop=True)
    return out


def reference_snapshot_path(root: Path, name: str, day: str) -> Path:
    """Path of the byte-exact daily reference file ``<root>/reference/<name>/<day>.json.gz``.

    Raises:
        DataIntegrityError: ``name`` not in ``REFERENCE_URLS`` or ``day`` not ``YYYYMMDD``.
    """
    if name not in REFERENCE_URLS:
        raise DataIntegrityError(f"unknown reference {name}")
    if not re.fullmatch(r"\d{8}", day):
        raise DataIntegrityError(f"reference day must be YYYYMMDD, got {day!r}")
    return Path(root) / REFERENCE_DIRNAME / name / f"{day}.json.gz"


def write_reference_snapshot(
    raw: bytes, root: Path, name: str, *, captured_at: pd.Timestamp
) -> Path | None:
    """Persist the exact response bytes as ``<root>/reference/<name>/<YYYYMMDD>.json.gz`` once per UTC day.

    The payload is stored byte-for-byte (gzip only, no re-serialization) so fields unknown today
    (status, onboardDate, deliveryDate, filters, underlyingType, ...) remain recoverable for
    point-in-time universe reconstruction.

    Returns:
        The written path, or None when that day's file already exists.

    Raises:
        DataIntegrityError: ``raw`` is empty or not valid JSON; ``name`` not in ``REFERENCE_URLS``.
    """
    if name not in REFERENCE_URLS:
        raise DataIntegrityError(f"unknown reference {name}")
    if not raw:
        raise DataIntegrityError("reference payload empty")
    try:
        json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
    except Exception as exc:
        raise DataIntegrityError("reference payload is not valid JSON") from exc
    cap = _utc_captured(captured_at)
    day = cap.strftime("%Y%m%d")
    target = reference_snapshot_path(root, name, day)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return None
    blob = gzip.compress(bytes(raw), compresslevel=9, mtime=0)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, target)
    return target
