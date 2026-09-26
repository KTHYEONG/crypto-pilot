from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import aiohttp
import pandas as pd

from src.common.parquet_io import read_parquet_or_quarantine, write_parquet_atomic
from src.common.paths import DATA_DIR

if TYPE_CHECKING:
    from src.market_data.streams.coverage import CoverageTracker

_logger = logging.getLogger(__name__)

# Binance USD-M 웹소켓은 카테고리 경로로 분리되었다. 청산 스트림은 `market` 경로에서만 이벤트가 온다.
# 레거시 `/ws/` 경로는 연결·핑퐁은 정상이면서 이벤트가 0건이라 겉보기로는 "조용한 시장"과 구분되지 않는다.
FORCE_ORDER_STREAM_URL: str = "wss://fstream.binance.com/market/ws/!forceOrder@arr"


@dataclass(frozen=True, slots=True)
class FeedFrame:
    """One observation from a liquidation feed.

    ``kind`` semantics:
      * ``"events"``  — one or more raw forceOrder payloads arrived at ``received_at``.
      * ``"alive"``   — a control frame (pong, or server ping that was answered) arrived at
        ``received_at``; proves the connection was live at that instant without carrying events.
      * ``"timeout"`` — nothing arrived within the requested timeout; carries no liveness evidence.
      * ``"closed"``  — the connection ended (server close, transport error, heartbeat loss).
    """

    kind: Literal["events", "alive", "timeout", "closed"]
    received_at: pd.Timestamp | None
    payloads: tuple[Mapping[str, Any], ...] = ()
    detail: str = ""


class LiquidationFeed(Protocol):
    """Minimal surface a liquidation transport must provide."""

    async def receive(self, timeout_s: float) -> FeedFrame: ...
    async def close(self) -> None: ...


class BinanceForceOrderFeed:
    """Binance USD-M all-market liquidation stream over a raw WebSocket.

    Binance pushes at most one forceOrder snapshot per symbol per second on ``!forceOrder@arr``; the
    payload shape is ``{"e": "forceOrder", "E": ..., "o": {...}}`` which ``parse_liquidation``
    already accepts. Automatic ping handling is disabled so control frames surface as liveness
    evidence: the feed sends a client ping every ``ping_interval_s`` and answers server pings itself.
    """

    def __init__(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        *,
        ping_interval_s: float,
        now_fn: Callable[[], pd.Timestamp],
    ) -> None:
        self._ws = ws
        self._ping_interval_s = ping_interval_s
        self._now_fn = now_fn
        self._last_ping = time.monotonic()
        self._closed = False

    @classmethod
    async def connect(
        cls,
        session: aiohttp.ClientSession,
        *,
        url: str = FORCE_ORDER_STREAM_URL,
        ping_interval_s: float,
        now_fn: Callable[[], pd.Timestamp],
    ) -> BinanceForceOrderFeed:
        """Open the all-market forceOrder stream; raises on handshake failure."""
        ws = await session.ws_connect(url, autoping=False, heartbeat=None)
        return cls(ws, ping_interval_s=ping_interval_s, now_fn=now_fn)

    def _handle_text(self, text: str) -> FeedFrame:
        try:
            decoded = json.loads(text)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("[DATA] stage=liquidation_stream status=BAD_FRAME detail=%s", exc)
            return FeedFrame(kind="alive", received_at=self._now_fn())
        if isinstance(decoded, Mapping):
            return FeedFrame(kind="events", received_at=self._now_fn(), payloads=(decoded,))
        if isinstance(decoded, list):
            usable = tuple(item for item in decoded if isinstance(item, Mapping))
            if usable:
                if len(usable) != len(decoded):
                    _logger.warning(
                        "[DATA] stage=liquidation_stream status=BAD_FRAME detail=%d of %d items dropped",
                        len(decoded) - len(usable),
                        len(decoded),
                    )
                return FeedFrame(kind="events", received_at=self._now_fn(), payloads=usable)
            _logger.warning("[DATA] stage=liquidation_stream status=BAD_FRAME detail=no usable payloads")
            return FeedFrame(kind="alive", received_at=self._now_fn())
        _logger.warning("[DATA] stage=liquidation_stream status=BAD_FRAME detail=non-mapping payload")
        return FeedFrame(kind="alive", received_at=self._now_fn())

    async def receive(self, timeout_s: float) -> FeedFrame:
        """Wait at most ``timeout_s`` for the next frame, sending a client ping when due."""
        if self._closed:
            return FeedFrame(kind="closed", received_at=None, detail="already closed")
        if time.monotonic() - self._last_ping >= self._ping_interval_s:
            try:
                await self._ws.ping()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                return FeedFrame(kind="closed", received_at=None, detail=f"ping failed: {exc}")
            self._last_ping = time.monotonic()
        try:
            msg = await self._ws.receive(timeout=timeout_s)
        except TimeoutError:
            return FeedFrame(kind="timeout", received_at=None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            return FeedFrame(kind="closed", received_at=None, detail=f"receive failed: {exc}")
        try:
            if msg.type == aiohttp.WSMsgType.TEXT:
                return self._handle_text(str(msg.data))
            if msg.type == aiohttp.WSMsgType.BINARY:
                try:
                    text = bytes(msg.data).decode("utf-8")
                except Exception as exc:  # noqa: BLE001
                    _logger.warning("[DATA] stage=liquidation_stream status=BAD_FRAME detail=%s", exc)
                    return FeedFrame(kind="alive", received_at=self._now_fn())
                return self._handle_text(text)
            if msg.type == aiohttp.WSMsgType.PONG:
                return FeedFrame(kind="alive", received_at=self._now_fn())
            if msg.type == aiohttp.WSMsgType.PING:
                try:
                    await self._ws.pong(msg.data)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    return FeedFrame(kind="closed", received_at=None, detail=f"pong failed: {exc}")
                return FeedFrame(kind="alive", received_at=self._now_fn())
            if msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.ERROR,
            ):
                return FeedFrame(kind="closed", received_at=None, detail=f"ws {msg.type.name}: {msg.data}")
            return FeedFrame(kind="closed", received_at=None, detail=f"unexpected ws type: {msg.type}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            return FeedFrame(kind="closed", received_at=None, detail=f"dispatch failed: {exc}")

    async def close(self) -> None:
        """Close the socket; idempotent and never raises."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self._ws.close()


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

    Partitions are keyed by the UTC hour of ``event_time`` so each flush rewrites at most one hour of
    rows and every closed hour becomes an immutable file that the nightly backup copies once. Each
    touched hour is read, merged, deduplicated on (symbol, event_time_ms, price, orig_qty,
    filled_accum_qty) keeping the last copy, and atomically replaced; an undecodable hour file is
    quarantined and the hour restarts from the new events. Legacy daily files
    (``liquidations_<YYYYMMDD>.parquet``) are never modified; ``load_liquidation_events`` reads both
    layouts. Assumes a single writer process per ``directory``.

    Args:
        events: Parsed events; an empty batch writes nothing.
        directory: Partition directory (created if missing).

    Returns:
        Sorted hourly partition paths written.

    Raises:
        Exception: Merge failures and OS-level read/write failures propagate with every existing
            partition unchanged; the caller keeps its buffer and retries on the next flush.
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
        grp = grp.drop_duplicates(subset=dedup_subset, keep="last")
        existing = read_parquet_or_quarantine(path, stage="liquidations")
        if existing is not None:
            if "event_time_ms" not in existing.columns and "event_time" in existing.columns:
                existing["event_time_ms"] = pd.to_datetime(existing["event_time"], utc=True).astype("int64") // 1_000_000
            combined = pd.concat([existing, grp], ignore_index=True)
            combined = _apply_compact_dtypes(combined)
            combined["event_time"] = pd.to_datetime(combined["event_time"], utc=True)
            combined["ingested_at"] = pd.to_datetime(combined["ingested_at"], utc=True)
            combined = combined.drop_duplicates(subset=dedup_subset, keep="last")
            write_parquet_atomic(combined, path, compression="zstd")
        else:
            write_parquet_atomic(grp, path, compression="zstd")
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


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


@dataclass(slots=True)
class LiquidationHealth:
    """Mutable liveness facts of the liquidation stream, published through the recorder heartbeat.

    Updated in place by ``run_liquidation_stream`` on the event loop thread and snapshotted by the
    recorder's heartbeat task; a pong-answering socket that delivers no events is invisible to
    coverage, so an external watchdog needs the time of the last *parseable* event and the count of
    connections that ended without delivering one.

    Attributes:
        last_event_at: Wall-clock receipt time of the latest parseable forceOrder event (UTC).
        last_connected_at: Wall-clock time the latest connection was established (UTC).
        consecutive_failed_connections: Connection attempts in a row that failed to open or closed
            without delivering a parseable event; reset to 0 by the next parseable event.
        last_disconnect_reason: Status token and detail of the latest connection end or connect failure.
        last_persisted_at: Wall-clock instant of the latest successful event flush (UTC).
        consecutive_flush_failures: Flushes in a row that raised.
        pending_events: Events buffered in memory and not yet persisted.
        dropped_events_total: Events discarded by the pending bound since process start.
    """

    last_event_at: pd.Timestamp | None = None
    last_connected_at: pd.Timestamp | None = None
    consecutive_failed_connections: int = 0
    last_disconnect_reason: str | None = None
    last_persisted_at: pd.Timestamp | None = None
    consecutive_flush_failures: int = 0
    pending_events: int = 0
    dropped_events_total: int = 0

    def as_heartbeat_entry(self) -> dict[str, Any]:
        """JSON-ready snapshot: timestamps as ISO-8601 UTC strings (or ``None``), counters as ``int``."""

        def _iso(ts: pd.Timestamp | None) -> str | None:
            if ts is None:
                return None
            out = pd.Timestamp(ts)
            out = out.tz_localize("UTC") if out.tzinfo is None else out.tz_convert("UTC")  # noqa: SIM108
            return str(out.isoformat())

        return {
            "last_event_at": _iso(self.last_event_at),
            "last_connected_at": _iso(self.last_connected_at),
            "consecutive_failed_connections": int(self.consecutive_failed_connections),
            "last_disconnect_reason": self.last_disconnect_reason,
            "last_persisted_at": _iso(self.last_persisted_at),
            "consecutive_flush_failures": int(self.consecutive_flush_failures),
            "pending_events": int(self.pending_events),
            "dropped_events_total": int(self.dropped_events_total),
        }


async def run_liquidation_stream(
    *,
    symbols: list[str] | None,
    directory: Path,
    flush_interval_s: float = 60.0,
    max_buffer: int = 5000,
    shutdown: Any | None = None,
    feed_factory: Callable[[], Awaitable[LiquidationFeed]] | None = None,
    clock: Callable[[], float] = time.monotonic,
    coverage: CoverageTracker | None = None,
    now_fn: Callable[[], pd.Timestamp] | None = None,
    receive_timeout_s: float = 1.0,
    liveness_timeout_s: float = 15.0,
    ping_interval_s: float = 5.0,
    max_backoff_s: float = 60.0,
    event_stall_timeout_s: float = 600.0,
    health: LiquidationHealth | None = None,
    max_pending_events: int = 100_000,
) -> None:
    """Stream Binance forceOrder liquidations into hourly parquet partitions until shutdown.

    Binance pushes at most one liquidation snapshot per symbol per second, so stored events are a lower
    bound of liquidation flow. Coverage is attested only by parseable forceOrder events: within one
    connection the attested segment spans its first to its latest event, so quiet stretches between
    events stay attested while a connection that only answers pings (wrong endpoint, dropped
    subscription) attests nothing. Control frames are liveness evidence for reconnect decisions only.
    Shutdown is observed at least every ``receive_timeout_s`` so SIGTERM completes the final flush
    well inside the container stop grace period.

    Args:
        symbols: optional client-side filter on the all-market stream (normalized symbol names);
            ``None`` keeps every symbol. Filtered-out events still count as evidence that the
            subscription delivers.
        feed_factory: opens a connected feed; defaults to ``BinanceForceOrderFeed.connect`` on a
            session owned (and closed) by this function.
        receive_timeout_s: upper bound on shutdown observation latency.
        liveness_timeout_s: a connection with no event/control frame for this long is treated as dead.
        ping_interval_s: client ping cadence passed to the default feed.
        max_backoff_s: cap of the exponential reconnect backoff.
        event_stall_timeout_s: a connection that answers pings but delivers no parseable event for
            this long is treated as a silently broken subscription and reconnected. Market-wide
            liquidations arrive ~0.3/s and the largest observed quiet gap is ~107 s, so 600 s never
            fires on a healthy feed.
        health: optional liveness record updated in place for the recorder heartbeat.
        max_pending_events: memory bound on unpersisted events; oldest events are dropped
            beyond it after a failed flush.

    Raises:
        ValueError: when ``receive_timeout_s``/``ping_interval_s`` are not positive or
            ``liveness_timeout_s`` is not greater than ``ping_interval_s`` or ``event_stall_timeout_s``
            is not greater than ``liveness_timeout_s``.
        Exception: any unexpected error, re-raised after buffered events and coverage are flushed and
            the feed/session are closed, so the caller's supervisor restarts the stream.
    """
    if receive_timeout_s <= 0:
        raise ValueError("receive_timeout_s must be positive")
    if ping_interval_s <= 0:
        raise ValueError("ping_interval_s must be positive")
    if liveness_timeout_s <= ping_interval_s:
        raise ValueError("liveness_timeout_s must be greater than ping_interval_s")
    if event_stall_timeout_s <= liveness_timeout_s:
        raise ValueError("event_stall_timeout_s must be greater than liveness_timeout_s")
    if max_pending_events < 1:
        raise ValueError("max_pending_events must be >= 1")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    wanted = {_normalize_symbol(s) for s in symbols} if symbols else None
    _now = now_fn if now_fn is not None else _utc_now

    owned_session: aiohttp.ClientSession | None = None
    factory: Callable[[], Awaitable[LiquidationFeed]]
    if feed_factory is not None:
        factory = feed_factory
    else:
        owned_session = aiohttp.ClientSession()

        async def _default_factory() -> LiquidationFeed:
            assert owned_session is not None
            return await BinanceForceOrderFeed.connect(
                owned_session, ping_interval_s=ping_interval_s, now_fn=_now
            )

        factory = _default_factory

    buffer: list[LiquidationEvent] = []
    last_flush = clock()
    last_coverage_flush = last_flush
    backoff = 1.0

    def _is_shutdown() -> bool:
        try:
            return bool(getattr(shutdown, "requested", False))
        except Exception:  # noqa: BLE001
            return False

    async def _sleep_shutdown_aware(delay: float) -> bool:
        """Sleep in ≤1 s chunks; return True when shutdown was observed."""
        remaining = delay
        while remaining > 0:
            if _is_shutdown():
                return True
            try:
                await asyncio.sleep(min(1.0, remaining))
            except asyncio.CancelledError:
                raise
            remaining -= 1.0
        return _is_shutdown()

    def _mark_ok(ts: pd.Timestamp) -> None:
        if coverage is None:
            return
        try:
            coverage.mark_ok(ts)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("[DATA] stage=liquidation_stream status=COVERAGE_MARK_FAILED detail=%s", exc)

    def _mark_error() -> None:
        if coverage is None:
            return
        try:
            coverage.mark_error(_now())
        except Exception as exc:  # noqa: BLE001
            _logger.warning("[DATA] stage=liquidation_stream status=COVERAGE_MARK_FAILED detail=%s", exc)

    def _flush_coverage() -> None:
        if coverage is None:
            return
        try:
            coverage.flush()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("[DATA] stage=liquidation_stream status=COVERAGE_FLUSH_FAILED detail=%s", exc)

    def _flush_events() -> bool:
        """Persist buffered events; return True when the buffer is empty afterwards."""
        nonlocal last_flush
        if not buffer:
            return True
        try:
            append_liquidation_events(buffer, directory)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("[DATA] stage=liquidation_stream status=FLUSH_FAILED detail=%s", exc)
            if health is not None:
                health.consecutive_flush_failures += 1
                health.pending_events = len(buffer)
            if len(buffer) > max_pending_events:
                dropped = len(buffer) - max_pending_events
                del buffer[:dropped]
                if health is not None:
                    health.dropped_events_total += dropped
                    health.pending_events = len(buffer)
                _logger.error(
                    "[DATA] stage=liquidation_stream status=BUFFER_OVERFLOW dropped=%d pending=%d",
                    dropped, len(buffer),
                )
                if coverage is not None:
                    try:
                        coverage.discard_unflushed(_now())
                    except Exception as exc:  # noqa: BLE001
                        _logger.warning(
                            "[DATA] stage=liquidation_stream status=COVERAGE_MARK_FAILED detail=%s", exc
                        )
            return False
        buffer.clear()
        last_flush = clock()
        if health is not None:
            health.last_persisted_at = _now()
            health.consecutive_flush_failures = 0
            health.pending_events = 0
        return True

    async def _close_feed(feed: LiquidationFeed | None) -> None:
        if feed is None:
            return
        try:
            await feed.close()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            _logger.warning("[DATA] stage=liquidation_stream status=CLOSE_FAILED detail=%s", exc)

    feed: LiquidationFeed | None = None
    try:
        while not _is_shutdown():
            try:
                feed = await factory()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "[DATA] stage=liquidation_stream status=CONNECT_FAILED detail=%s backoff=%.1f",
                    exc,
                    backoff,
                )
                if health is not None:
                    health.consecutive_failed_connections += 1
                    health.last_disconnect_reason = f"CONNECT_FAILED: {exc}"
                _mark_error()
                if not buffer:
                    _flush_coverage()
                if await _sleep_shutdown_aware(backoff):
                    break
                backoff = min(max_backoff_s, backoff * 2.0)
                continue
            if health is not None:
                health.last_connected_at = _now()
            has_evidence = False
            connection_event_evidence = False
            disconnect_reason: str | None = None
            last_evidence = clock()
            last_event_at = last_evidence

            def _note_stall(elapsed: float) -> None:
                nonlocal disconnect_reason
                _logger.warning(
                    "[DATA] stage=liquidation_stream status=EVENT_STALL detail=no event for %.0fs "
                    "while connection answers",
                    elapsed,
                )
                disconnect_reason = f"EVENT_STALL: no parseable event for {elapsed:.0f}s"
                _mark_error()
                if not buffer:
                    _flush_coverage()

            while not _is_shutdown():
                try:
                    frame = await feed.receive(receive_timeout_s)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    frame = FeedFrame(kind="closed", received_at=None, detail=f"receive raised: {exc}")
                if frame.kind == "events":
                    # Event evidence is evaluated before the symbol filter: a parseable
                    # forceOrder payload proves the subscription delivers, even when the
                    # event itself is filtered out of storage.
                    evidence: list[LiquidationEvent] = []
                    for msg in frame.payloads:
                        if not isinstance(msg, Mapping):
                            continue
                        received_at = frame.received_at
                        assert received_at is not None
                        ev = parse_liquidation(msg, ingested_at=received_at)
                        if ev is not None:
                            evidence.append(ev)
                    has_evidence = True
                    last_evidence = clock()
                    if evidence:
                        last_event_at = clock()
                        connection_event_evidence = True
                        received_at = frame.received_at
                        assert received_at is not None
                        _mark_ok(received_at)
                        if health is not None:
                            health.last_event_at = received_at
                            health.consecutive_failed_connections = 0
                        for ev in evidence:
                            if wanted is not None and ev.symbol not in wanted:
                                continue
                            buffer.append(ev)
                    elif clock() - last_event_at >= event_stall_timeout_s:
                        _note_stall(clock() - last_event_at)
                        break
                elif frame.kind == "alive":
                    has_evidence = True
                    last_evidence = clock()
                    if clock() - last_event_at >= event_stall_timeout_s:
                        _note_stall(clock() - last_event_at)
                        break
                elif frame.kind == "timeout":
                    if clock() - last_event_at >= event_stall_timeout_s:
                        _note_stall(clock() - last_event_at)
                        break
                    if clock() - last_evidence >= liveness_timeout_s:
                        elapsed = clock() - last_evidence
                        _logger.warning(
                            "[DATA] stage=liquidation_stream status=LIVENESS_TIMEOUT detail=no evidence for "
                            "%.1fs",
                            elapsed,
                        )
                        disconnect_reason = f"LIVENESS_TIMEOUT: no evidence for {elapsed:.1f}s"
                        _mark_error()
                        if not buffer:
                            _flush_coverage()
                        break
                else:  # "closed"
                    _logger.warning(
                        "[DATA] stage=liquidation_stream status=DISCONNECTED detail=%s", frame.detail
                    )
                    disconnect_reason = f"DISCONNECTED: {frame.detail}"
                    _mark_error()
                    if not buffer:
                        _flush_coverage()
                    break
                now_c = clock()
                if buffer and (len(buffer) >= max_buffer or now_c - last_flush >= flush_interval_s):
                    if _flush_events():
                        _flush_coverage()
                        last_coverage_flush = now_c
                elif not buffer and now_c - last_coverage_flush >= flush_interval_s:
                    # 조용한 구간은 프레임(핑 5초)마다가 아니라 flush 주기마다만 기록한다(레코드 폭증 방지).
                    _flush_coverage()
                    last_coverage_flush = now_c
            await _close_feed(feed)
            feed = None
            if disconnect_reason is not None and health is not None:
                health.last_disconnect_reason = disconnect_reason
                if not _is_shutdown() and not connection_event_evidence:
                    health.consecutive_failed_connections += 1
            if _is_shutdown():
                break
            if has_evidence:
                backoff = 1.0
                continue
            if await _sleep_shutdown_aware(backoff):
                break
            backoff = min(max_backoff_s, backoff * 2.0)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        _logger.error("[DATA] stage=liquidation_stream status=FATAL detail=%s", exc, exc_info=True)
        raise
    finally:
        _flush_events()
        if not buffer:
            _flush_coverage()
        await _close_feed(feed)
        if owned_session is not None:
            try:
                await owned_session.close()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                _logger.warning("[DATA] stage=liquidation_stream status=SESSION_CLOSE_FAILED detail=%s", exc)
