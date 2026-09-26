from __future__ import annotations

import json
import logging
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
    """`raw_order_json`: the venue order object (`o`) serialized as compact, key-sorted JSON (`separators=(",", ":")`, `ensure_ascii=False`). The forceOrder stream has no archive, so fields unknown today (e.g. `ps`, `st`) are preserved verbatim for later research. `None` when the event came through the unified fallback without a raw order object."""


def _normalize_symbol(symbol: Any) -> str:
    raw = str(symbol).strip()
    if "/" in raw or ":" in raw:
        raw = raw.replace("/", "")
        raw = raw.split(":")[0]
    return raw


def _serialize_raw_order(o: Mapping[str, Any]) -> str | None:
    try:
        return json.dumps(dict(o), separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def parse_liquidation(
    msg: Mapping[str, Any],
    *,
    ingested_at: pd.Timestamp,
) -> LiquidationEvent | None:
    try:
        if not isinstance(msg, Mapping):
            return None
        ingested = pd.to_datetime(ingested_at, utc=True)
        if pd.isna(ingested):
            return None

        # 원시 forceOrder 주문 오브젝트 우선. 현행 ccxt(binanceusdm)는 이를 info 로
        # 평탄화해 전달하고(info.s/q/z/T ...), 과거 스키마는 info.o 로 중첩했다.
        info = msg.get("info") if isinstance(msg.get("info"), Mapping) else None
        o: Mapping[str, Any] | None = None
        if isinstance(info, Mapping) and isinstance(info.get("o"), Mapping):
            o = info["o"]
        elif isinstance(info, Mapping) and "s" in info and "T" in info:
            o = info
        elif isinstance(msg.get("o"), Mapping):
            o = msg["o"]

        if o is not None:
            s = o.get("s")
            side_raw = o.get("S")
            otype = o.get("o")
            f = o.get("f")
            q = o.get("q")
            p = o.get("p")
            ap = o.get("ap")
            status_raw = o.get("X")
            l_val = o.get("l")
            z_val = o.get("z")
            t_raw = o.get("T")
            if s is None or t_raw is None:
                return None
            symbol = _normalize_symbol(s)
            side = str(side_raw) if side_raw is not None else ""
            order_type = str(otype) if otype is not None else ""
            tif = str(f) if f is not None else ""
            try:
                orig_qty = float(q) if q is not None else 0.0
                price = float(p) if p is not None else 0.0
                avg_price = float(ap) if ap is not None else price
                last_filled = float(l_val) if l_val is not None else 0.0
                filled_accum = float(z_val) if z_val is not None else 0.0
            except (TypeError, ValueError):
                return None
            try:
                event_time = pd.to_datetime(int(float(str(t_raw))), unit="ms", utc=True)
            except Exception:
                return None
            status = str(status_raw) if status_raw is not None else ""
            return LiquidationEvent(
                symbol=symbol,
                event_time=event_time,
                ingested_at=ingested,
                side=side,
                order_type=order_type,
                time_in_force=tif,
                orig_qty=float(orig_qty),
                price=float(price),
                avg_price=float(avg_price),
                status=status,
                last_filled_qty=float(last_filled),
                filled_accum_qty=float(filled_accum),
                raw_order_json=_serialize_raw_order(o),
            )

        # Unified fallback
        symbol_raw = msg.get("symbol")
        ts = msg.get("timestamp")
        if ts is None:
            ts = msg.get("T")
        if symbol_raw is None or ts is None:
            return None
        symbol = _normalize_symbol(symbol_raw)
        price_raw = msg.get("price")
        if price_raw is None:
            price_raw = msg.get("markPrice")
        if price_raw is None:
            # try p
            price_raw = msg.get("p")
        if price_raw is None:
            return None
        qty_raw = msg.get("baseValue")
        if qty_raw is None:
            qty_raw = msg.get("amount")
        if qty_raw is None:
            qty_raw = msg.get("orig_qty")
        if qty_raw is None:
            qty_raw = msg.get("q")
        if qty_raw is None:
            # try to derive from quoteValue / price ?
            qv = msg.get("quoteValue")
            if qv is not None:
                try:
                    qty_raw = float(qv) / float(price_raw) if float(price_raw) != 0 else None
                except Exception:
                    qty_raw = None
            if qty_raw is None:
                return None
        try:
            price = float(price_raw)
            orig_qty = float(qty_raw)
        except (TypeError, ValueError):
            return None
        # avg price fallback
        ap_raw = msg.get("avg_price")
        if ap_raw is None:
            ap_raw = msg.get("ap")
        if ap_raw is None:
            ap_raw = msg.get("average")
        avg_price = float(ap_raw) if ap_raw is not None else price
        try:
            avg_price = float(avg_price)
        except (TypeError, ValueError):
            avg_price = price
        side = str(msg.get("side", msg.get("S", "")) or "")
        order_type = str(msg.get("order_type", msg.get("o", "")) or "")
        tif = str(msg.get("time_in_force", msg.get("f", "")) or "")
        status = str(msg.get("status", msg.get("X", "")) or "")
        # last filled / accum
        l_raw = msg.get("last_filled_qty")
        if l_raw is None:
            l_raw = msg.get("l")
        z_raw = msg.get("filled_accum_qty")
        if z_raw is None:
            z_raw = msg.get("z")
        try:
            last_filled = float(l_raw) if l_raw is not None else orig_qty
        except (TypeError, ValueError):
            last_filled = orig_qty
        try:
            filled_accum = float(z_raw) if z_raw is not None else orig_qty
        except (TypeError, ValueError):
            filled_accum = orig_qty
        try:
            event_time = pd.to_datetime(int(float(str(ts))), unit="ms", utc=True)
        except Exception:
            return None
        return LiquidationEvent(
            symbol=symbol,
            event_time=event_time,
            ingested_at=ingested,
            side=side,
            order_type=order_type,
            time_in_force=tif,
            orig_qty=float(orig_qty),
            price=float(price),
            avg_price=float(avg_price),
            status=status,
            last_filled_qty=float(last_filled),
            filled_accum_qty=float(filled_accum),
        )
    except Exception:
        return None


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


def load_liquidation_events(
    directory: Path | str | None = None,
    *,
    since: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if directory is None:
        directory = default_liquidations_dir()
    directory = Path(directory)
    if not directory.exists():
        return pd.DataFrame()
    files = sorted(directory.glob("liquidations_*.parquet"))
    if not files:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for p in files:
        try:
            df = pd.read_parquet(p)
            if df.empty:
                continue
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            _logger.debug("load_liquidation_events skip %s: %s", p, exc)
            continue
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    # ensure tz-aware event_time
    if "event_time" in combined.columns:
        combined["event_time"] = pd.to_datetime(combined["event_time"], utc=True)
    if "ingested_at" in combined.columns:
        combined["ingested_at"] = pd.to_datetime(combined["ingested_at"], utc=True)
    if "raw_order_json" in combined.columns:
        combined["raw_order_json"] = combined["raw_order_json"].astype("string")
    # dedup globally (cross-file same key unlikely but keep)
    dedup_subset = [c for c in ["symbol", "event_time_ms", "price", "orig_qty", "filled_accum_qty"] if c in combined.columns]
    if dedup_subset:
        # ensure event_time_ms exists
        if "event_time_ms" not in combined.columns and "event_time" in combined.columns:
            combined["event_time_ms"] = pd.to_datetime(combined["event_time"], utc=True).astype("int64") // 1_000_000
        combined = combined.drop_duplicates(subset=dedup_subset, keep="last")
    if since is not None:
        since_ts = pd.to_datetime(since, utc=True)
        if "event_time" in combined.columns:
            combined = combined[combined["event_time"] >= since_ts]
    # sort by event_time
    if "event_time" in combined.columns:
        combined = combined.sort_values("event_time").reset_index(drop=True)
    return combined


def default_liquidations_dir() -> Path:
    return DATA_DIR / "futures" / "liquidations"


