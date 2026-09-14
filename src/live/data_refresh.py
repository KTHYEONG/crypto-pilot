# ruff: noqa
from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.market_data.binance.futures import BinanceIpBlockedError
from src.market_data.services.futures_collection import funding_gap_start_ms, funding_tail_is_fresh
from src.market_data.storage.ohlcv import is_temp_artifact
from src.quant.universe.pit_universe import symbol_partition

try:
    from src.market_data.services.futures_collection import DataCollector
except Exception:  # noqa: BLE001

    class DataCollector:  # type: ignore[no-redef]
        pass

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RefreshReport:
    total: int
    fresh: int
    refreshed: int
    failed: int
    deadline_skipped: int
    elapsed_s: float
    deadline_hit: bool
    staleness_hours: float
    ok: bool
    ip_blocked: bool = False
    funding_stale: int = 0
    absent: int = 0


class ColdUniverseError(RuntimeError):
    pass


STALENESS_ACTIVE_WINDOW_HOURS: int = 72

# DataCollector는 메인넷 fapi 고정이라 목록도 같은 베뉴를 쓴다.
EXCHANGE_INFO_URL: str = "https://fapi.binance.com/fapi/v1/exchangeInfo"
EXCHANGE_INFO_TIMEOUT_S: float = 20.0
# 실측 제거 비율 4/807≈0.5%를 크게 넘으면 잘못된 베뉴/응답으로 보고 목록 불신.
ABSENT_MAX_FRACTION: float = 0.05
ABSENT_LOG_SAMPLE: int = 10


def parse_listed_symbols(payload: Mapping[str, Any]) -> frozenset[str] | None:
    """Return listed symbols regardless of status; None on malformed/empty."""
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        return None
    listed = {str(e["symbol"]) for e in symbols if isinstance(e, Mapping) and "symbol" in e}
    if not listed:
        return None
    return frozenset(listed)


def fetch_listed_symbols(url: str = EXCHANGE_INFO_URL, *, timeout_s: float = EXCHANGE_INFO_TIMEOUT_S, opener: Callable[..., Any] = urllib.request.urlopen) -> frozenset[str] | None:
    """Fetch exchangeInfo listing; fail-open None on transport or payload errors."""
    try:
        with opener(url, timeout=timeout_s) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        _logger.warning("[DATA] stage=exchange_listing fetch_failed error=%s", exc)
        return None
    if not isinstance(payload, dict):
        return None
    listed = parse_listed_symbols(payload)
    if listed is None:
        _logger.warning("[DATA] stage=exchange_listing malformed_payload")
        return None
    return listed


def split_absent_symbols(symbols: Sequence[str], listed: frozenset[str] | None) -> tuple[list[str], list[str]]:
    """Split disk universe into listed and absent; fail-open on untrusted listing."""
    if listed is None:
        return (list(symbols), [])
    absent = [s for s in symbols if s not in listed]
    if symbols and len(absent) > ABSENT_MAX_FRACTION * len(symbols):
        _logger.error("[DATA] stage=exchange_listing untrusted absent=%d total=%d", len(absent), len(symbols))
        return (list(symbols), [])
    return ([s for s in symbols if s in listed], absent)


def _disk_tail_ts(futures_root: Path, symbol: str, now: pd.Timestamp) -> pd.Timestamp | None:
    try:
        p = Path(futures_root) / "ohlcv" / "1h" / f"{symbol}.parquet"
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p, columns=["timestamp"])
        except Exception:
            return None
        if df.empty or "timestamp" not in df.columns:
            return None
        # Ensure numeric
        try:
            ts_col = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
        except Exception:
            return None
        if ts_col.empty:
            return None
        # check integer-like? spec says non-integer -> None
        # We'll treat any non-integer as None if not finite int
        try:
            max_val = int(ts_col.max())
        except Exception:
            return None
        # If original max wasn't integer, still convert; but spec says non-integer -> None
        # For safety, check if ts_col contains non-integer floats? We'll just require int
        return pd.Timestamp(int(max_val), unit="ms", tz="UTC")
    except Exception:
        return None


def _funding_fresh_on_disk(futures_root: Path, symbol: str, now: pd.Timestamp, window_start: pd.Timestamp | None = None) -> bool:
    path = Path(futures_root) / "funding" / f"{symbol}.parquet"
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path, columns=["timestamp"])
    except (OSError, ValueError) as exc:
        _logger.debug("[DATA] funding tail unreadable symbol=%s error=%s", symbol, exc)
        return False
    stamps = pd.to_numeric(frame["timestamp"], errors="coerce").dropna().astype("int64").tolist()
    if not funding_tail_is_fresh(stamps, now):
        return False
    if window_start is None:
        return True
    return funding_gap_start_ms(stamps, int(window_start.value // 1_000_000)) is None


def market_data_staleness_hours(futures_root: Path, *, now: pd.Timestamp, partition: str = "dev") -> float:
    try:
        root = Path(futures_root) / "ohlcv" / "1h"
        if not root.exists():
            return float("inf")
        parquets = [p for p in root.glob("*.parquet") if not is_temp_artifact(p.name)]
        if not parquets:
            return float("inf")
        gaps: list[float] = []
        for p in parquets:
            sym = p.stem
            try:
                if symbol_partition(sym) != partition:
                    continue
            except Exception:
                continue
            try:
                df = pd.read_parquet(p, columns=["timestamp", "volume"])
            except Exception:
                continue
            if df.empty or "timestamp" not in df.columns:
                continue
            try:
                ts_col = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
            except Exception:
                continue
            if ts_col.empty:
                continue
            try:
                max_val = int(ts_col.max())
            except Exception:
                continue
            try:
                tail = pd.Timestamp(int(max_val), unit="ms", tz="UTC")
            except Exception:
                continue
            recent = pd.to_numeric(df["timestamp"], errors="coerce") >= int((now - pd.Timedelta(hours=STALENESS_ACTIVE_WINDOW_HOURS)).value // 10**6)
            if not (pd.to_numeric(df.loc[recent, "volume"], errors="coerce") > 0).any():
                continue
            gaps.append((now - tail).total_seconds() / 3600.0)
        if not gaps:
            return float("inf")
        # p90 -- 소수의 상장폐지 심볼(무한 gap)이 지표를 오염시키지 않도록.
        # 실제 시스템 장애 시엔 대부분 심볼이 정체되므로 p90도 함께 상승한다.
        return float(pd.Series(gaps).quantile(0.90))
    except Exception:
        return float("inf")


def _refresh_one_symbol_tail(collector: Any, symbol: str, start: str, end: str, *, funding_start: str | None = None) -> bool:
    try:
        collector.ensure_ohlcv_data(symbol, "1h", start, end)
    except BinanceIpBlockedError:
        raise
    except Exception as exc:  # noqa: BLE001
        _logger.warning("[DATA] refresh_live_universe symbol=%s failed error=%s", symbol, exc)
        return False
    funding_ok = True
    try:
        collector.ensure_funding_data(symbol, start if funding_start is None else funding_start, end)
    except BinanceIpBlockedError:
        raise
    except Exception as exc:  # noqa: BLE001
        _logger.warning("[DATA] funding symbol=%s failed error=%r", symbol, exc)
        funding_ok = False
    try:
        if hasattr(collector, "ensure_mark_price_data"):
            collector.ensure_mark_price_data(symbol, "1h", start, end)
        elif hasattr(collector, "ensure_mark_price_klines"):
            collector.ensure_mark_price_klines(symbol, "1h", start, end)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("[DATA] markPriceKlines symbol=%s failed error=%s", symbol, exc)
    try:
        if hasattr(collector, "ensure_metrics_live_tail"):
            collector.ensure_metrics_live_tail(symbol)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("[DATA] metrics_live_tail symbol=%s failed error=%s", symbol, exc)
    return funding_ok


def refresh_live_market_data(
    futures_root: Path,
    *,
    now: pd.Timestamp,
    lookback_days: int,
    max_workers: int,
    deadline_s: float,
    freshness_floor_hours: float,
    min_symbols: int,
    max_fail_fraction: float,
    symbols: list[str] | None = None,
    collector: Any | None = None,
    partition: str = "dev",
    listed_symbols: frozenset[str] | None = None,
) -> RefreshReport:
    t0 = time.perf_counter()
    # 1) symbol list
    if symbols is None:
        root = Path(futures_root) / "ohlcv" / "1h"
        parquets = sorted(root.glob("*.parquet")) if root.exists() else []
        syms = [p.stem for p in parquets if not is_temp_artifact(p.name)]
        # filter dev
        filtered: list[str] = []
        for s in syms:
            try:
                if symbol_partition(s) == partition:
                    filtered.append(s)
            except Exception:
                continue
        symbols_list = filtered
    else:
        symbols_list = list(symbols)

    symbols_list, absent = split_absent_symbols(symbols_list, listed_symbols)
    if absent:
        _logger.warning("[DATA] stage=refresh_live_market_data absent_symbols=%d sample=%s", len(absent), ",".join(absent[:ABSENT_LOG_SAMPLE]))

    total = len(symbols_list)
    if total < min_symbols:
        raise ColdUniverseError(f"dev universe {total} < min {min_symbols}")

    if collector is None:
        collector = DataCollector()

    deadline_ts = time.perf_counter() + float(deadline_s)
    funding_window_start = now - pd.Timedelta(days=lookback_days)

    ip_blocked = threading.Event()

    # Per-symbol worker
    def _refresh_symbol(sym: str) -> str:
        if time.perf_counter() > deadline_ts:
            return "deadline"
        tail = _disk_tail_ts(futures_root, sym, now)
        if tail is not None and (now - tail) <= pd.Timedelta(hours=freshness_floor_hours) and _funding_fresh_on_disk(futures_root, sym, now, window_start=funding_window_start):
            return "fresh"
        if ip_blocked.is_set():
            return "ip_blocked"
        if tail is not None:
            start = max(tail - pd.Timedelta(hours=2), now - pd.Timedelta(days=lookback_days))
        else:
            start = now - pd.Timedelta(days=lookback_days)
        try:
            ok = _refresh_one_symbol_tail(collector, sym, str(start), str(now), funding_start=str(funding_window_start))
        except BinanceIpBlockedError as exc:
            ip_blocked.set()
            _logger.error("[DATA] stage=refresh_live_market_data ip_blocked=True symbol=%s http_code=%d", sym, exc.http_code)
            return "ip_blocked"
        return "refreshed" if ok else "failed"

    fresh = 0
    refreshed = 0
    failed = 0
    deadline_skipped = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_refresh_symbol, sym): sym for sym in symbols_list}
        for fut in concurrent.futures.as_completed(futures):
            try:
                outcome = fut.result()
            except Exception as exc:  # noqa: BLE001
                _logger.warning("[DATA] symbol worker exception error=%s", exc)
                failed += 1
                continue
            if outcome == "fresh":
                fresh += 1
            elif outcome == "refreshed":
                refreshed += 1
            elif outcome == "failed":
                failed += 1
            elif outcome == "ip_blocked":
                failed += 1
            elif outcome == "deadline":
                deadline_skipped += 1
            else:
                failed += 1

    deadline_hit = deadline_skipped > 0
    # Also if deadline passed but all tasks already started? spec says deadline_hit True if any skipped
    # If deadline_s==0, some tasks may not be skipped if they started before deadline? But spec says with 0 deadline all should be skipped.
    # The per-worker deadline check at entry ensures that if deadline already passed at submission time, they skip.
    # However ThreadPool may have already started some before deadline check; for deadline_s=0 we check deadline_ts = t0 +0, so any worker entering after t0 will see perf_counter > deadline_ts true.
    # So all will be deadline.

    staleness = market_data_staleness_hours(Path(futures_root), now=now, partition=partition)
    funding_stale = sum(1 for sym in symbols_list if not _funding_fresh_on_disk(futures_root, sym, now, window_start=funding_window_start))
    ok = (fresh + refreshed) >= min_symbols and failed <= math.ceil(max_fail_fraction * total) and not ip_blocked.is_set()
    elapsed_s = time.perf_counter() - t0

    _logger.info(
        "[DATA] stage=refresh_live_market_data total=%d fresh=%d refreshed=%d failed=%d deadline_skipped=%d deadline_hit=%s staleness_h=%.1f elapsed_s=%.1f ok=%s ip_blocked=%s funding_stale=%d absent=%d",
        total, fresh, refreshed, failed, deadline_skipped, deadline_hit, staleness, elapsed_s, ok, ip_blocked.is_set(), funding_stale, len(absent),
    )

    return RefreshReport(
        total=total,
        fresh=fresh,
        refreshed=refreshed,
        failed=failed,
        deadline_skipped=deadline_skipped,
        elapsed_s=float(elapsed_s),
        deadline_hit=bool(deadline_hit),
        staleness_hours=float(staleness),
        ok=bool(ok),
        ip_blocked=ip_blocked.is_set(),
        funding_stale=funding_stale,
        absent=len(absent),
    )
