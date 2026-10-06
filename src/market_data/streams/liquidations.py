from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.parquet_io import read_parquet_or_quarantine, write_parquet_atomic
from src.common.paths import DATA_DIR

_logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class LiquidationEvent:
    symbol: str
    event_time: pd.Timestamp
    ingested_at: pd.Timestamp
    side: str
    order_type: str
    time_in_force: str
    orig_qty: float
    price: float
    avg_price: float
    status: str
    last_filled_qty: float
    filled_accum_qty: float
    raw_order_json: str | None = None
    """`raw_order_json`: the venue order object (`o`) serialized as compact, key-sorted JSON (`separators=(",", ":")`, `ensure_ascii=False`). The forceOrder stream has no archive, so fields unknown today (e.g. `ps`, `st`) are preserved verbatim for later research. `None` on rows persisted before the column existed, or when the order object holds a non-JSON value."""


def _serialize_raw_order(o: Mapping[str, Any]) -> str | None:
    try:
        return json.dumps(dict(o), separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def _reject(reason: str, field: str) -> LiquidationEvent | None:
    _logger.debug("[DATA] stage=parse_liquidation status=REJECTED reason=%s field=%s", reason, field)
    return None


def _parse_num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (str, int, float)):
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
    else:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_liquidation(
    msg: Mapping[str, Any],
    *,
    ingested_at: pd.Timestamp,
) -> LiquidationEvent | None:
    """Parse one decoded raw Binance USD-M ``forceOrder`` frame into a liquidation event.

    Accepts only the raw stream shape ``{"e": "forceOrder", "E": <ms>, "o": {...}}``
    journaled by the capture service; ccxt-normalized shapes are not production inputs
    and are rejected. Every rejection returns ``None`` so the normalizer counts it in
    ``parse_failures`` (heartbeat ``force_order.parse_failures_total``); no field is ever
    defaulted, because a fabricated quantity or price would be persisted
    indistinguishably from venue data.

    Field contract on ``msg["o"]`` (all keys required):
        s, S, o, f, X: non-empty ``str`` without surrounding whitespace; ``S`` is
            ``"BUY"`` or ``"SELL"``.
        q, p: decimal ``str`` or real number (not ``bool``), finite and > 0.
        ap, l, z: decimal ``str`` or real number (not ``bool``), finite and >= 0.
            Zero is legitimate: a snapshot of an order without fills carries
            ``ap = l = z = 0``.
        T: order trade time in integer epoch milliseconds, as ``int`` (not ``bool``)
            or an ASCII-digit ``str``; > 0 and representable as a pandas Timestamp
            at nanosecond resolution for persisted storage.

    Args:
        msg: JSON-decoded frame. Envelope keys ``e``/``E`` and unknown order keys
            (e.g. ``ps``, ``st``) are not validated; the order object is preserved
            verbatim in ``raw_order_json``.
        ingested_at: Local receipt time (capture ``recv_ns``); converted to UTC.

    Returns:
        ``LiquidationEvent`` with ``event_time`` = ``T`` (UTC, ms resolution),
        ``symbol`` = ``s`` verbatim, quantities/prices as ``float``; or ``None`` when
        the shape or any field violates the contract or ``ingested_at`` is NaT.

    Raises:
        Never raises for JSON-decoded input: a poison frame must not crash or stall the
        normalizer checkpoint loop.
    """
    if not isinstance(msg, Mapping):
        return _reject("shape", "-")
    o = msg.get("o")
    if not isinstance(o, Mapping):
        return _reject("shape", "-")
    ingested = pd.to_datetime(ingested_at, utc=True)
    if pd.isna(ingested):
        return _reject("ingested_at", "-")
    for key in ("s", "S", "o", "f", "q", "p", "ap", "X", "l", "z", "T"):
        if key not in o or o[key] is None:
            return _reject("missing", key)
    for key in ("s", "S", "o", "f", "X"):
        value = o[key]
        if not isinstance(value, str) or not value or value != value.strip():
            return _reject("type", key)
    if o["S"] not in ("BUY", "SELL"):
        return _reject("side", "S")
    parsed: dict[str, float] = {}
    for key in ("q", "p", "ap", "l", "z"):
        value = _parse_num(o[key])
        if value is None:
            raw = o[key]
            if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
                return _reject("type", key)
            try:
                float(raw)
            except (TypeError, ValueError, OverflowError):
                return _reject("type", key)
            return _reject("non_finite", key)
        parsed[key] = value
    if parsed["q"] <= 0 or parsed["p"] <= 0:
        bad = "q" if parsed["q"] <= 0 else "p"
        return _reject("non_positive", bad)
    for key in ("ap", "l", "z"):
        if parsed[key] < 0:
            return _reject("negative", key)
    t_raw = o["T"]
    t_int: int | None = None
    if isinstance(t_raw, bool):
        return _reject("timestamp", "T")
    if isinstance(t_raw, int):
        t_int = t_raw
    elif isinstance(t_raw, str):
        if not t_raw or not t_raw.isascii() or not t_raw.isdigit():
            return _reject("timestamp", "T")
        try:
            t_int = int(t_raw)
        except (TypeError, ValueError, OverflowError):
            return _reject("timestamp", "T")
    else:
        return _reject("timestamp", "T")
    if t_int is None or t_int <= 0:
        return _reject("timestamp", "T")
    try:
        event_time = pd.Timestamp(t_int, unit="ms", tz="UTC")
        event_time.as_unit("ns")
    except (TypeError, ValueError, OverflowError):
        return _reject("timestamp", "T")
    return LiquidationEvent(
        symbol=o["s"],
        event_time=event_time,
        ingested_at=ingested,
        side=o["S"],
        order_type=o["o"],
        time_in_force=o["f"],
        orig_qty=float(parsed["q"]),
        price=float(parsed["p"]),
        avg_price=float(parsed["ap"]),
        status=o["X"],
        last_filled_qty=float(parsed["l"]),
        filled_accum_qty=float(parsed["z"]),
        raw_order_json=_serialize_raw_order(o),
    )


def _events_to_frame(events: Sequence[LiquidationEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for ev in events:
        # event_time_ms for dedup key
        try:
            etm = int(ev.event_time.value // 1_000_000)
        except Exception:
            etm = int(pd.to_datetime(ev.event_time, utc=True).value // 1_000_000)
        rows.append(
            {
                "symbol": ev.symbol,
                "event_time": pd.to_datetime(ev.event_time, utc=True),
                "ingested_at": pd.to_datetime(ev.ingested_at, utc=True),
                "side": ev.side,
                "order_type": ev.order_type,
                "time_in_force": ev.time_in_force,
                "orig_qty": float(ev.orig_qty),
                "price": float(ev.price),
                "avg_price": float(ev.avg_price),
                "status": ev.status,
                "last_filled_qty": float(ev.last_filled_qty),
                "filled_accum_qty": float(ev.filled_accum_qty),
                "event_time_ms": etm,
                "raw_order_json": ev.raw_order_json,
            }
        )
    df = pd.DataFrame(rows)
    # ensure tz-aware
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True)
    df["raw_order_json"] = df["raw_order_json"].astype("string")
    return df


def _apply_compact_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    # price/avg_price float64
    for col in ("price", "avg_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in ("orig_qty", "last_filled_qty", "filled_accum_qty"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    for col in ("side", "status", "order_type", "time_in_force"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    if "raw_order_json" in df.columns:
        df["raw_order_json"] = df["raw_order_json"].astype("string")
    return df


def append_liquidation_events(
    events: Sequence[LiquidationEvent],
    directory: Path,
) -> list[Path]:
    """Merge liquidation events into hourly ``liquidations_<YYYYMMDD>_<HH>.parquet`` partitions.

    Partitions are keyed by the UTC hour of ``event_time``. Each touched hour is read, merged,
    deduplicated on (symbol, event_time_ms, price, orig_qty, filled_accum_qty) and atomically
    replaced. The copy with the earliest ``ingested_at`` survives: two capture slots receive the same
    frame, and a checkpoint replay re-parses it, and neither may move the recorded receipt time later.
    An undecodable hour file is quarantined and the hour restarts from the new events. Legacy daily
    files are never modified. Assumes a single writer process per ``directory``.

    Args:
        events: Parsed events; an empty batch writes nothing.
        directory: Partition directory (created if missing).

    Returns:
        Sorted hourly partition paths written.

    Raises:
        Exception: Merge and OS-level failures propagate with every existing partition unchanged.
    """
    if not events:
        return []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame = _events_to_frame(events)
    if frame.empty:
        return []
    # group by UTC hour of event_time
    frame["event_hour"] = frame["event_time"].dt.tz_convert("UTC").dt.strftime("%Y%m%d_%H")
    written: list[Path] = []
    dedup_subset = ["symbol", "event_time_ms", "price", "orig_qty", "filled_accum_qty"]
    for hour_str, group in frame.groupby("event_hour"):
        path = directory / f"liquidations_{hour_str}.parquet"
        # prepare group without helper key
        grp = group.drop(columns=["event_hour"])
        grp = _apply_compact_dtypes(grp)
        grp["event_time"] = pd.to_datetime(grp["event_time"], utc=True)
        grp["ingested_at"] = pd.to_datetime(grp["ingested_at"], utc=True)
        existing = read_parquet_or_quarantine(path, stage="liquidations")
        if existing is not None:
            if "event_time_ms" not in existing.columns and "event_time" in existing.columns:
                existing["event_time_ms"] = pd.to_datetime(existing["event_time"], utc=True).astype("int64") // 1_000_000
            combined = _earliest_ingest_merge(existing, grp, dedup_subset)
            write_parquet_atomic(combined, path, compression="zstd")
        else:
            write_parquet_atomic(_earliest_ingest_merge(grp.iloc[0:0], grp, dedup_subset), path, compression="zstd")
        written.append(path)
    # sort and dedup written
    written = sorted(set(written))
    return written


def _earliest_ingest_merge(
    existing: pd.DataFrame, chunk: pd.DataFrame, dedup_subset: list[str]
) -> pd.DataFrame:
    """Combine on-disk events with new events so the earliest receipt per key sorts first.

    The surviving row keeps the minimum ``ingested_at`` (ties go to the row already on disk),
    but it never loses a non-null ``raw_order_json`` to a null duplicate: a non-null payload
    from any duplicate is patched onto the survivor.
    """
    prior = pd.DataFrame({"_on_disk": [0] * len(existing) + [1] * len(chunk)})
    combined = pd.concat([existing, chunk], ignore_index=True)
    combined = _apply_compact_dtypes(combined)
    combined["event_time"] = pd.to_datetime(combined["event_time"], utc=True)
    combined["ingested_at"] = pd.to_datetime(combined["ingested_at"], utc=True)
    combined["_on_disk"] = prior["_on_disk"].to_numpy()
    combined = combined.sort_values(["ingested_at", "_on_disk"], na_position="last")
    best_raw = (
        combined.dropna(subset=["raw_order_json"])
        .drop_duplicates(subset=dedup_subset, keep="first")
        .set_index(dedup_subset)["raw_order_json"]
    )
    survivors = combined.drop_duplicates(subset=dedup_subset, keep="first")
    needs_raw = survivors["raw_order_json"].isna()
    patched = survivors[needs_raw].set_index(dedup_subset).index.map(best_raw).astype("string")
    survivors.loc[needs_raw, "raw_order_json"] = patched.to_numpy()
    combined = survivors.drop(columns=["_on_disk"])
    if "event_time" in combined.columns:
        combined = combined.sort_values("event_time").reset_index(drop=True)
    return combined


def default_liquidations_dir() -> Path:
    return DATA_DIR / "futures" / "liquidations"
