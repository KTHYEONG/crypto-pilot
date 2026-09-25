"""Point-in-time snapshots of Binance USD-M public REST state that no archive preserves:
all-symbol top-of-book, premium index / predicted funding, and raw daily reference payloads."""

from __future__ import annotations

import gzip
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
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
NULLABLE_MS_COLUMNS: frozenset[str] = frozenset({"fetched_at_ms"})

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


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


def _parse_float(value: Any, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise DataIntegrityError(f"book ticker row has non-numeric {label}") from exc
    if pd.isna(out):
        raise DataIntegrityError(f"book ticker row has non-numeric {label}")
    return out


def _parse_ms(value: Any, label: str) -> int:
    try:
        out = int(float(str(value)))
    except (TypeError, ValueError) as exc:
        raise DataIntegrityError(f"snapshot row has non-numeric {label}") from exc
    return out


def _parse_opt_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str) and value.strip() == "":
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise DataIntegrityError("premium index row has non-numeric field") from exc
    return out


def parse_book_ticker_payload(
    payload: Any, *, captured_at: pd.Timestamp, fetched_at: pd.Timestamp
) -> pd.DataFrame:
    """Normalize an all-symbol ``/fapi/v1/ticker/bookTicker`` response into ``BOOK_TICKER_COLUMNS``.

    ``captured_at`` is the wall-clock grid instant the sample is filed under (a join key shared by all
    datasets); ``fetched_at`` is when the response was fully received and is the earliest instant the
    data was observable, so point-in-time consumers must gate on ``fetched_at_ms``, never on
    ``captured_at``. ``exchange_time_ms`` is the venue's own quote timestamp, kept so clock skew and
    quote staleness remain measurable.

    Returns:
        One row per symbol; prices float64, quantities float32, ``exchange_time_ms`` int64,
        ``fetched_at_ms`` Int64, ``captured_at`` datetime64[ns, UTC], ``symbol`` string.

    Raises:
        DataIntegrityError: payload is not a non-empty list, a row lacks a required key or has a
            non-numeric / non-positive price, ``fetched_at`` is naive, or ``fetched_at`` precedes
            ``captured_at``.
    """
    if not isinstance(payload, list) or not payload:
        raise DataIntegrityError("book ticker payload must be a non-empty list")
    cap = _utc_captured(captured_at)
    fetched = _utc_fetched(fetched_at, captured_at)
    fetched_ms = int(fetched.value // 1_000_000)
    rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, Mapping):
            raise DataIntegrityError("book ticker row malformed")
        for key in ("symbol", "bidPrice", "bidQty", "askPrice", "askQty", "time"):
            if key not in row:
                raise DataIntegrityError(f"book ticker row lacks {key}")
        symbol = str(row["symbol"])
        bid_px = _parse_float(row["bidPrice"], "bidPrice")
        ask_px = _parse_float(row["askPrice"], "askPrice")
        if not bid_px > 0 or not ask_px > 0:
            raise DataIntegrityError("book ticker row has non-positive price")
        bid_qty = _parse_float(row["bidQty"], "bidQty")
        ask_qty = _parse_float(row["askQty"], "askQty")
        exchange_ms = _parse_ms(row["time"], "time")
        rows.append(
            {
                "captured_at": cap,
                "symbol": symbol,
                "exchange_time_ms": exchange_ms,
                "fetched_at_ms": fetched_ms,
                "bid_px": bid_px,
                "bid_qty": bid_qty,
                "ask_px": ask_px,
                "ask_qty": ask_qty,
            }
        )
    frame = pd.DataFrame(rows, columns=list(BOOK_TICKER_COLUMNS))
    frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True)
    frame["symbol"] = frame["symbol"].astype("string")
    frame["exchange_time_ms"] = frame["exchange_time_ms"].astype("int64")
    frame["fetched_at_ms"] = frame["fetched_at_ms"].astype("Int64")
    frame["bid_px"] = frame["bid_px"].astype("float64")
    frame["ask_px"] = frame["ask_px"].astype("float64")
    frame["bid_qty"] = frame["bid_qty"].astype("float32")
    frame["ask_qty"] = frame["ask_qty"].astype("float32")
    return frame


def parse_premium_index_payload(
    payload: Any, *, captured_at: pd.Timestamp, fetched_at: pd.Timestamp
) -> pd.DataFrame:
    """Normalize an all-symbol ``/fapi/v1/premiumIndex`` response into ``PREMIUM_INDEX_COLUMNS``.

    ``last_funding_rate`` is the venue's running estimate for the next settlement (the realized rate is
    archived separately by the funding history endpoint); keeping the estimate path lets research test
    predicted-funding signals without reconstructing the premium-index clamp. ``captured_at`` is the
    grid label and ``fetched_at_ms`` the availability time (see ``parse_book_ticker_payload``).

    Returns:
        One row per symbol; rates/prices float64, venue ``*_ms`` int64, ``fetched_at_ms`` Int64,
        ``captured_at`` datetime64[ns, UTC]. Empty-string numeric fields (delisting symbols) become
        NaN, never 0.

    Raises:
        DataIntegrityError: payload is not a non-empty list, a row lacks ``symbol``/``markPrice``,
            ``fetched_at`` is naive, or ``fetched_at`` precedes ``captured_at``.
    """
    if not isinstance(payload, list) or not payload:
        raise DataIntegrityError("premium index payload must be a non-empty list")
    cap = _utc_captured(captured_at)
    fetched = _utc_fetched(fetched_at, captured_at)
    fetched_ms = int(fetched.value // 1_000_000)
    rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, Mapping):
            raise DataIntegrityError("premium index row malformed")
        if "symbol" not in row or "markPrice" not in row:
            raise DataIntegrityError("premium index row lacks symbol/markPrice")
        raw_mark = row["markPrice"]
        if raw_mark is None:
            raise DataIntegrityError("premium index row lacks symbol/markPrice")
        symbol = str(row["symbol"])
        if "time" not in row or "nextFundingTime" not in row:
            raise DataIntegrityError("premium index row lacks time/nextFundingTime")
        mark_price = _parse_opt_float(raw_mark)
        if pd.isna(mark_price):
            raise DataIntegrityError("premium index row lacks symbol/markPrice")
        rows.append(
            {
                "captured_at": cap,
                "symbol": symbol,
                "exchange_time_ms": _parse_ms(row["time"], "time"),
                "fetched_at_ms": fetched_ms,
                "mark_price": mark_price,
                "index_price": _parse_opt_float(row.get("indexPrice")),
                "estimated_settle_price": _parse_opt_float(row.get("estimatedSettlePrice")),
                "last_funding_rate": _parse_opt_float(row.get("lastFundingRate")),
                "interest_rate": _parse_opt_float(row.get("interestRate")),
                "next_funding_time_ms": _parse_ms(row["nextFundingTime"], "nextFundingTime"),
            }
        )
    frame = pd.DataFrame(rows, columns=list(PREMIUM_INDEX_COLUMNS))
    frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True)
    frame["symbol"] = frame["symbol"].astype("string")
    frame["exchange_time_ms"] = frame["exchange_time_ms"].astype("int64")
    frame["fetched_at_ms"] = frame["fetched_at_ms"].astype("Int64")
    frame["next_funding_time_ms"] = frame["next_funding_time_ms"].astype("int64")
    for col in ("mark_price", "index_price", "estimated_settle_price", "last_funding_rate", "interest_rate"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return frame


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
    directory = Path(root) / REFERENCE_DIRNAME / name
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{day}.json.gz"
    if target.exists():
        return None
    blob = gzip.compress(bytes(raw), compresslevel=9, mtime=0)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, target)
    return target
