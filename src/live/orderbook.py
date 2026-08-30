"""Order book snapshot capture and persistence."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import DATA_DIR
from src.common.errors import DataIntegrityError

_ORDERBOOK_TOP_K: int = 20


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    symbol: str
    captured_at: pd.Timestamp
    decision_time: pd.Timestamp
    mode: str
    capture_seq: int
    bids: tuple[tuple[Decimal, Decimal], ...]
    asks: tuple[tuple[Decimal, Decimal], ...]
    last_update_id: int


def fetch_order_book(
    client: Any,
    symbol: str,
    decision_time: pd.Timestamp,
    *,
    mode: str,
    capture_seq: int,
    limit: int,
    now: pd.Timestamp,
) -> OrderBookSnapshot:
    payload = client.depth(symbol, limit=limit)
    if not isinstance(payload, dict) or "bids" not in payload or "asks" not in payload:
        raise DataIntegrityError("depth endpoint returned an unexpected schema")
    # lastUpdateId variants
    last_id = payload.get("lastUpdateId")
    if last_id is None:
        last_id = payload.get("last_update_id", 0)
    try:
        last_update_id = int(str(last_id)) if last_id is not None else 0
    except Exception:
        raise DataIntegrityError("depth missing lastUpdateId")  # noqa: B904
    raw_bids = payload.get("bids") or []
    raw_asks = payload.get("asks") or []
    if not isinstance(raw_bids, list) or not isinstance(raw_asks, list):
        raise DataIntegrityError("depth bids/asks not list")
    bids: list[tuple[Decimal, Decimal]] = []
    for entry in raw_bids[:limit]:
        try:
            px = Decimal(str(entry[0]))
            qty = Decimal(str(entry[1]))
        except Exception as exc:
            raise DataIntegrityError(f"invalid bid entry {entry}") from exc
        bids.append((px, qty))
    asks: list[tuple[Decimal, Decimal]] = []
    for entry in raw_asks[:limit]:
        try:
            px = Decimal(str(entry[0]))
            qty = Decimal(str(entry[1]))
        except Exception as exc:
            raise DataIntegrityError(f"invalid ask entry {entry}") from exc
        asks.append((px, qty))
    captured_at = pd.Timestamp(now)
    captured_at = captured_at.tz_localize("UTC") if captured_at.tzinfo is None else captured_at.tz_convert("UTC")  # noqa: SIM108
    dt = pd.Timestamp(decision_time)
    dt = dt.tz_localize("UTC") if dt.tzinfo is None else dt.tz_convert("UTC")  # noqa: SIM108
    return OrderBookSnapshot(
        symbol=str(symbol),
        captured_at=captured_at,
        decision_time=dt,
        mode=str(mode),
        capture_seq=int(capture_seq),
        bids=tuple(bids),
        asks=tuple(asks),
        last_update_id=last_update_id,
    )


def capture_order_books(
    client: Any,
    symbols: Sequence[str],
    decision_time: pd.Timestamp,
    *,
    mode: str,
    duration_s: float,
    interval_s: float,
    depth_limit: int,
    max_symbols: int,
    clock: Callable[[], float],
    sleep_fn: Callable[[float], None],
    now_fn: Callable[[], pd.Timestamp],
    shutdown: Any | None = None,
) -> list[OrderBookSnapshot]:
    syms = list(symbols)[:max_symbols]
    if not syms:
        return []
    # handle duration <=0 as single tick
    if duration_s <= 0:
        # still one tick if not shutdown
        if shutdown is not None and getattr(shutdown, "requested", False):
            return []
        snapshots_single: list[OrderBookSnapshot] = []
        now = now_fn()
        for s in syms:
            try:
                snap = fetch_order_book(client, s, decision_time, mode=mode, capture_seq=0, limit=depth_limit, now=now)
                snapshots_single.append(snap)
            except Exception:  # noqa: S112
                continue
        return snapshots_single
    max_ticks = math.ceil(duration_s / interval_s) + 1
    start = clock()
    snapshots: list[OrderBookSnapshot] = []
    seq = 0
    ticks = 0
    while ticks < max_ticks:
        if shutdown is not None and getattr(shutdown, "requested", False):
            break
        elapsed = clock() - start
        if elapsed >= duration_s:
            break
        now = now_fn()
        for s in syms:
            try:
                snap = fetch_order_book(client, s, decision_time, mode=mode, capture_seq=seq, limit=depth_limit, now=now)
                snapshots.append(snap)
            except Exception:  # noqa: S112
                continue
        seq += 1
        ticks += 1
        # check elapsed before sleeping
        elapsed_after = clock() - start
        if elapsed_after >= duration_s:
            break
        if ticks >= max_ticks:
            break
        # check shutdown before sleep? spec says every non-final tick ends in sleep; also shutdown checked every tick start, so next loop will break
        # Sleep only if not final
        sleep_fn(interval_s)
    return snapshots


def _flatten_snapshot(snap: OrderBookSnapshot) -> dict[str, Any]:
    # best bid/ask
    best_bid = float(snap.bids[0][0]) if snap.bids else float("nan")
    best_ask = float(snap.asks[0][0]) if snap.asks else float("nan")
    if snap.bids and snap.asks:
        mid = (float(snap.bids[0][0]) + float(snap.asks[0][0])) / 2.0
        spread_bps = (float(snap.asks[0][0]) - float(snap.bids[0][0])) / mid * 10000 if mid != 0 else float("nan")
    else:
        mid = float("nan")
        spread_bps = float("nan")
    row: dict[str, Any] = {
        "symbol": snap.symbol,
        "captured_at": snap.captured_at,
        "decision_time": snap.decision_time,
        "mode": snap.mode,
        "capture_seq": int(snap.capture_seq),
        "last_update_id": int(snap.last_update_id),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread_bps": spread_bps,
        "depth_levels": len(snap.bids) if len(snap.bids) == len(snap.asks) else max(len(snap.bids), len(snap.asks)),
    }
    # flat top-K columns
    for i in range(_ORDERBOOK_TOP_K):
        if i < len(snap.bids):
            row[f"bid_px_{i:02d}"] = float(snap.bids[i][0])
            row[f"bid_qty_{i:02d}"] = float(snap.bids[i][1])
        else:
            row[f"bid_px_{i:02d}"] = float("nan")
            row[f"bid_qty_{i:02d}"] = float("nan")
    for i in range(_ORDERBOOK_TOP_K):
        if i < len(snap.asks):
            row[f"ask_px_{i:02d}"] = float(snap.asks[i][0])
            row[f"ask_qty_{i:02d}"] = float(snap.asks[i][1])
        else:
            row[f"ask_px_{i:02d}"] = float("nan")
            row[f"ask_qty_{i:02d}"] = float("nan")
    return row


def append_order_book_snapshots(
    snapshots: Sequence[OrderBookSnapshot], directory: Path
) -> list[Path]:
    if not snapshots:
        return []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # group by day YYYYMMDD of captured_at UTC
    buckets: dict[str, list[dict[str, Any]]] = {}
    for snap in snapshots:
        key = pd.Timestamp(snap.captured_at).tz_convert("UTC").strftime("%Y%m%d")
        buckets.setdefault(key, []).append(_flatten_snapshot(snap))
    written: list[Path] = []
    for yyyymmdd, rows in sorted(buckets.items()):
        df_new = pd.DataFrame(rows)
        # enforce dtypes
        # qty cols float32, px cols float64
        for col in df_new.columns:
            if col.startswith("bid_qty_") or col.startswith("ask_qty_"):
                df_new[col] = df_new[col].astype("float32")
            elif col.startswith("bid_px_") or col.startswith("ask_px_") or col in ("best_bid", "best_ask", "mid", "spread_bps"):
                df_new[col] = pd.to_numeric(df_new[col], errors="coerce").astype("float64")
        # datetime cols
        for col in ("captured_at", "decision_time"):
            if col in df_new.columns:
                df_new[col] = pd.to_datetime(df_new[col], utc=True).astype("datetime64[ns, UTC]")
        path = directory / f"live_orderbook_{yyyymmdd}.parquet"
        if path.exists():
            try:
                df_existing = pd.read_parquet(path)
                combined = pd.concat([df_existing, df_new], ignore_index=True)
                # dedup
                combined = combined.drop_duplicates(subset=["symbol", "captured_at", "last_update_id"])
                # re-enforce dtypes for qty/px
                for col in combined.columns:
                    if col.startswith("bid_qty_") or col.startswith("ask_qty_"):
                        combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("float32")
                    elif col.startswith("bid_px_") or col.startswith("ask_px_") or col in ("best_bid", "best_ask", "mid", "spread_bps"):
                        combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("float64")
                for col in ("captured_at", "decision_time"):
                    if col in combined.columns:
                        combined[col] = pd.to_datetime(combined[col], utc=True).astype("datetime64[ns, UTC]")
                combined.to_parquet(path, index=False, compression="zstd")
            except Exception:
                df_new.to_parquet(path, index=False, compression="zstd")
        else:
            # dedup within new (in case duplicates in input)
            df_new = df_new.drop_duplicates(subset=["symbol", "captured_at", "last_update_id"])
            df_new.to_parquet(path, index=False, compression="zstd")
        written.append(path)
    return sorted(written)


def load_order_book_snapshots(
    directory: Path | str | None = None, *, since: pd.Timestamp | None = None
) -> pd.DataFrame:
    dir_path = Path(directory) if directory is not None else default_orderbook_dir()
    if not dir_path.exists():
        return pd.DataFrame()
    shards = sorted(dir_path.glob("live_orderbook_*.parquet"))
    if not shards:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for shard in shards:
        try:
            df = pd.read_parquet(shard)
            if not df.empty:
                frames.append(df)
        except Exception:  # noqa: S112
            continue
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # dedup
    if {"symbol", "captured_at", "last_update_id"}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=["symbol", "captured_at", "last_update_id"])
    # ensure tz-aware
    if "captured_at" in combined.columns:
        combined["captured_at"] = pd.to_datetime(combined["captured_at"], utc=True).astype("datetime64[ns, UTC]")
    if "decision_time" in combined.columns:
        combined["decision_time"] = pd.to_datetime(combined["decision_time"], utc=True).astype("datetime64[ns, UTC]")
    if since is not None:
        try:
            since_ts = pd.Timestamp(since)
            since_ts = since_ts.tz_localize("UTC") if since_ts.tzinfo is None else since_ts.tz_convert("UTC")
            mask = pd.to_datetime(combined["captured_at"], utc=True) >= since_ts
            combined = combined[mask]
        except Exception:  # noqa: S110, BLE001
            pass
    return combined


def default_orderbook_dir() -> Path:
    return DATA_DIR / "state" / "live_orderbook"
