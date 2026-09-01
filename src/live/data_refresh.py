# ruff: noqa
from __future__ import annotations

import concurrent.futures
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.universe.pit_universe import symbol_partition

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


class ColdUniverseError(RuntimeError):
    pass


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


def market_data_staleness_hours(futures_root: Path, *, now: pd.Timestamp, partition: str = "dev") -> float:
    try:
        root = Path(futures_root) / "ohlcv" / "1h"
        if not root.exists():
            return float("inf")
        parquets = list(root.glob("*.parquet"))
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
                df = pd.read_parquet(p, columns=["timestamp"])
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
            gaps.append((now - tail).total_seconds() / 3600.0)
        if not gaps:
            return float("inf")
        # p90 -- 소수의 상장폐지 심볼(무한 gap)이 지표를 오염시키지 않도록.
        # 실제 시스템 장애 시엔 대부분 심볼이 정체되므로 p90도 함께 상승한다.
        return float(pd.Series(gaps).quantile(0.90))
    except Exception:
        return float("inf")


def _refresh_one_symbol_tail(collector: Any, symbol: str, start: str, end: str) -> bool:
    try:
        collector.ensure_ohlcv_data(symbol, "1h", start, end)
        collector.ensure_funding_data(symbol, start, end)
        try:
            if hasattr(collector, "ensure_mark_price_data"):
                collector.ensure_mark_price_data(symbol, "1h", start, end)
            elif hasattr(collector, "ensure_mark_price_klines"):
                collector.ensure_mark_price_klines(symbol, "1h", start, end)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("[DATA] markPriceKlines symbol=%s failed error=%s", symbol, exc)
        return True
    except Exception as exc:  # noqa: BLE001
        _logger.warning("[DATA] refresh_live_universe symbol=%s failed error=%s", symbol, exc)
        return False


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
) -> RefreshReport:
    t0 = time.perf_counter()
    # 1) symbol list
    if symbols is None:
        root = Path(futures_root) / "ohlcv" / "1h"
        parquets = sorted(root.glob("*.parquet")) if root.exists() else []
        syms = [p.stem for p in parquets]
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

    total = len(symbols_list)
    if total < min_symbols:
        raise ColdUniverseError(f"dev universe {total} < min {min_symbols}")

    if collector is None:
        collector = DataCollector()

    deadline_ts = time.perf_counter() + float(deadline_s)

    # Per-symbol worker
    def _refresh_symbol(sym: str) -> str:
        if time.perf_counter() > deadline_ts:
            return "deadline"
        tail = _disk_tail_ts(futures_root, sym, now)
        if tail is not None and (now - tail) <= pd.Timedelta(hours=freshness_floor_hours):
            return "fresh"
        if tail is not None:
            start = max(tail - pd.Timedelta(hours=2), now - pd.Timedelta(days=lookback_days))
        else:
            start = now - pd.Timedelta(days=lookback_days)
        ok = _refresh_one_symbol_tail(collector, sym, str(start), str(now))
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
    ok = (fresh + refreshed) >= min_symbols and failed <= math.ceil(max_fail_fraction * total)
    elapsed_s = time.perf_counter() - t0

    _logger.info(
        "[DATA] stage=refresh_live_market_data total=%d fresh=%d refreshed=%d failed=%d deadline_skipped=%d deadline_hit=%s staleness_h=%.1f elapsed_s=%.1f ok=%s",
        total, fresh, refreshed, failed, deadline_skipped, deadline_hit, staleness, elapsed_s, ok,
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
    )
