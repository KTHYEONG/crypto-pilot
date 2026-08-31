from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import DATA_DIR

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


def _normalize_symbol(symbol: Any) -> str:
    raw = str(symbol).strip()
    if "/" in raw or ":" in raw:
        raw = raw.replace("/", "")
        raw = raw.split(":")[0]
    return raw


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
            }
        )
    df = pd.DataFrame(rows)
    # ensure tz-aware
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True)
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
    return df


def append_liquidation_events(
    events: Sequence[LiquidationEvent],
    directory: Path,
) -> list[Path]:
    if not events:
        return []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame = _events_to_frame(events)
    if frame.empty:
        return []
    # group by date UTC of event_time
    frame["event_date"] = frame["event_time"].dt.tz_convert("UTC").dt.date
    written: list[Path] = []
    for date_val, group in frame.groupby("event_date"):
        date_str = pd.Timestamp(date_val).strftime("%Y%m%d")
        path = directory / f"liquidations_{date_str}.parquet"
        # prepare group without helper date
        grp = group.drop(columns=["event_date"])
        dedup_subset = ["symbol", "event_time_ms", "price", "orig_qty", "filled_accum_qty"]
        grp = _apply_compact_dtypes(grp)
        grp["event_time"] = pd.to_datetime(grp["event_time"], utc=True)
        grp["ingested_at"] = pd.to_datetime(grp["ingested_at"], utc=True)
        grp = grp.drop_duplicates(subset=dedup_subset, keep="last")
        if path.exists():
            try:
                existing = pd.read_parquet(path)
                if "event_time_ms" not in existing.columns and "event_time" in existing.columns:
                    existing["event_time_ms"] = pd.to_datetime(existing["event_time"], utc=True).astype("int64") // 1_000_000
                combined = pd.concat([existing, grp], ignore_index=True)
                combined = _apply_compact_dtypes(combined)
                combined["event_time"] = pd.to_datetime(combined["event_time"], utc=True)
                combined["ingested_at"] = pd.to_datetime(combined["ingested_at"], utc=True)
                combined = combined.drop_duplicates(subset=dedup_subset, keep="last")
                combined.to_parquet(path, index=False, compression="zstd")
            except Exception:
                grp.to_parquet(path, index=False, compression="zstd")
        else:
            grp.to_parquet(path, index=False, compression="zstd")
        written.append(path)
    # sort and dedup written
    written = sorted(set(written))
    return written


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


async def run_liquidation_stream(
    *,
    symbols: list[str] | None,
    directory: Path,
    flush_interval_s: float = 60.0,
    max_buffer: int = 5000,
    shutdown: Any | None = None,
    exchange_factory: Callable[[], Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if exchange_factory is not None:
        ex = exchange_factory()
    else:
        try:
            import ccxt.pro as ccxtpro

            ex = ccxtpro.binanceusdm({"newUpdates": True})
        except Exception:  # noqa: BLE001
            # fallback: try ccxt.pro via ccxt
            try:
                import ccxt.pro as ccxtpro

                ex = ccxtpro.binanceusdm({"newUpdates": True})
            except Exception as exc2:
                raise RuntimeError("ccxt.pro not available for liquidation stream") from exc2

    buffer: list[LiquidationEvent] = []
    last_flush = clock()
    backoff = 1.0
    max_backoff = 60.0

    # helpers to check shutdown
    def _is_shutdown() -> bool:
        try:
            return bool(getattr(shutdown, "requested", False))
        except Exception:
            return False

    while True:
        if _is_shutdown():
            break
        try:
            # ccxt watch_liquidations(symbol) 는 symbol 필수. 전체 마켓은 빈 리스트로 조회한다.
            if hasattr(ex, "watch_liquidations_for_symbols"):
                raw = await ex.watch_liquidations_for_symbols(list(symbols) if symbols else [])
            elif symbols and len(symbols) == 1 and hasattr(ex, "watch_liquidations"):
                raw = await ex.watch_liquidations(symbols[0])
            else:
                raise AttributeError("exchange has no usable liquidation watch method")
            # raw may be list or single dict
            if raw is None:
                items: list[Any] = []
            elif isinstance(raw, list):
                items = raw
            elif isinstance(raw, Mapping):
                items = [raw]
            else:
                try:
                    items = list(raw)
                except Exception:
                    items = [raw]
            now_ingested = pd.Timestamp.now(tz="UTC")
            for msg in items:
                if not isinstance(msg, Mapping):
                    continue
                ev = parse_liquidation(msg, ingested_at=now_ingested)
                if ev is not None:
                    buffer.append(ev)
            # reset backoff on success
            backoff = 1.0
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # log and backoff
            _logger.warning("liquidation stream error: %s backoff=%.1fs", exc, backoff)
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            backoff = min(max_backoff, backoff * 2.0)
            # check shutdown before continue
            if _is_shutdown():
                break
            continue

        # flush condition
        now_clock = clock()
        should_flush = False
        if len(buffer) >= max_buffer:
            should_flush = True
        elif (now_clock - last_flush) >= flush_interval_s:
            if buffer:
                should_flush = True
            else:
                # still update last_flush to avoid tight loop spinning on time?
                # only update if we would have flushed; otherwise keep interval?
                # For flush_interval_s==0, we want frequent checks; update.
                if flush_interval_s == 0:
                    last_flush = now_clock
        if should_flush and buffer:
            to_write = list(buffer)
            try:
                append_liquidation_events(to_write, directory)
                buffer.clear()
                last_flush = clock()
            except Exception as exc:
                _logger.warning("liquidation flush failed: %s", exc)
                # keep buffer for retry; update last_flush to avoid tight retry loop?
                last_flush = clock()
        # cooperative yield to avoid busy loop if watch returns immediately with empty
        # and shutdown not requested
        if not should_flush and not buffer:
            # small yield
            await asyncio.sleep(0)

        if _is_shutdown():
            break

    # final flush and close
    if buffer:
        try:
            append_liquidation_events(buffer, directory)
        except Exception as exc:
            _logger.warning("liquidation final flush failed: %s", exc)
    try:
        close = getattr(ex, "close", None)
        if close is not None:
            res = close()
            if asyncio.iscoroutine(res):
                await res
    except Exception as exc:
        _logger.debug("exchange close failed: %s", exc)
