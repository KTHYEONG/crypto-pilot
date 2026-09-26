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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.common.errors import DataIntegrityError
from src.live.venue_listing import VenueListingSnapshot
from src.market_data.binance.futures import BinanceIpBlockedError
from src.market_data.binance.venue_rules import EXCHANGE_INFO_URL
from src.market_data.services.futures_collection import funding_gap_start_ms, funding_tail_is_fresh
from src.market_data.storage.ohlcv import is_temp_artifact

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
    funding_blocked: bool = False
    incomplete: int = 0
    required_incomplete: tuple[str, ...] = ()
    not_current_sample: tuple[str, ...] = ()
    klines_elapsed_s: float = 0.0
    funding_elapsed_s: float = 0.0


@dataclass(frozen=True, slots=True)
class FundingPrefetchReport:
    total: int
    fetched: int
    fresh: int
    failed: int
    deadline_skipped: int
    funding_blocked: bool
    elapsed_s: float


class ColdUniverseError(RuntimeError):
    pass


STALENESS_ACTIVE_WINDOW_HOURS: int = 72

# DataCollector는 메인넷 fapi 고정이라 목록도 같은 베뉴를 쓴다.
EXCHANGE_INFO_TIMEOUT_S: float = 20.0
# 실측 제거 비율 4/807≈0.5%를 크게 넘으면 잘못된 베뉴/응답으로 보고 목록 불신.
ABSENT_MAX_FRACTION: float = 0.05
ABSENT_LOG_SAMPLE: int = 10


def listed_crypto_perpetuals(payload: Mapping[str, Any]) -> tuple[frozenset[str], frozenset[str]]:
    """Split an exchangeInfo payload into (tradable USDT crypto perpetuals, non-crypto symbols).

    Crypto = status TRADING, contractType PERPETUAL, quoteAsset USDT and underlyingType
    COIN; every symbol whose underlyingType is not COIN is non-crypto (tokenized equities,
    commodities, indices), matching the research lake exclusion.

    Raises:
        DataIntegrityError: payload has no symbols list or yields no crypto perpetual.
    """
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        raise DataIntegrityError("exchangeInfo payload has no symbols list")
    crypto: set[str] = set()
    non_crypto: set[str] = set()
    for entry in symbols:
        if not isinstance(entry, Mapping) or "symbol" not in entry:
            continue
        name = str(entry["symbol"])
        if str(entry.get("underlyingType", "COIN")) != "COIN":
            non_crypto.add(name)
            continue
        if (
            entry.get("status") == "TRADING"
            and entry.get("contractType") == "PERPETUAL"
            and entry.get("quoteAsset") == "USDT"
        ):
            crypto.add(name)
    if not crypto:
        raise DataIntegrityError("exchangeInfo payload yields no crypto perpetual")
    return frozenset(crypto), frozenset(non_crypto)


@dataclass(frozen=True, slots=True)
class RefreshUniverse:
    """Symbols the nightly refresh must bring current, split by required data planes."""

    trading: tuple[str, ...]  # TRADING COIN USDT perpetuals: klines + funding
    tracked_settled: tuple[str, ...]  # held/booked names past delivery: klines only (funding has ceased)
    tracked_pending: tuple[str, ...]  # held/booked names announced but not yet delivered: klines + funding
    unlisted_required: tuple[str, ...]  # required names absent from exchangeInfo (cannot be fetched)


_SETTLED_LISTING_STATUSES: frozenset[str] = frozenset({"SETTLING", "CLOSE"})


def build_refresh_universe(
    listing: VenueListingSnapshot,
    *,
    required_symbols: Iterable[str],
    non_crypto: frozenset[str],
    now: pd.Timestamp,
) -> RefreshUniverse:
    """Union the tradable crypto perpetual census with every symbol the book or ledger still depends on.

    ``required_symbols`` is the union of nonzero ledger positions and the nonzero columns of the
    latest deployed weight row. A required symbol whose listing status is no longer TRADING stays
    in the refresh: after delivery the venue keeps serving flat klines at the settlement price, and
    they are the only evidence that values the position and lets the unit proxy close. A required
    symbol absent from the listing is reported as ``unlisted_required``, never silently dropped.

    Raises:
        DataIntegrityError: the TRADING crypto census is empty or ``now`` is tz-naive.
    """
    stamp = pd.Timestamp(now)
    if stamp.tzinfo is None:
        raise DataIntegrityError("now must be timezone-aware UTC")
    now_utc = stamp.tz_convert("UTC")
    trading: set[str] = set()
    for name, entry in listing.entries.items():
        if entry.underlying_type != "COIN":
            continue
        if entry.status == "TRADING" and entry.contract_type == "PERPETUAL" and entry.quote_asset == "USDT":
            trading.add(str(name))
    if not trading:
        raise DataIntegrityError("listing yields no TRADING crypto perpetual")
    settled: set[str] = set()
    pending: set[str] = set()
    unlisted: set[str] = set()
    for raw in required_symbols:
        sym = str(raw)
        if sym in trading:
            continue
        if sym in non_crypto:
            unlisted.add(sym)
            continue
        req_entry = listing.entries.get(sym)
        if req_entry is None:
            unlisted.add(sym)
            continue
        if req_entry.underlying_type != "COIN":
            unlisted.add(sym)
            continue
        if (
            req_entry.status in _SETTLED_LISTING_STATUSES
            and req_entry.delivery_time is not None
            and now_utc >= req_entry.delivery_time
        ):
            settled.add(sym)
        else:
            pending.add(sym)
    return RefreshUniverse(
        trading=tuple(sorted(trading)),
        tracked_settled=tuple(sorted(settled)),
        tracked_pending=tuple(sorted(pending)),
        unlisted_required=tuple(sorted(unlisted)),
    )


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


def _kline_tail_state(futures_root: Path, symbol: str, now: pd.Timestamp) -> tuple[pd.Timestamp | None, bool]:
    """Return the stored 1h tail and whether the symbol traded in the recent staleness window.

    One read of ``timestamp`` and ``volume`` serves both the refresh-currency check and the
    staleness gauge, so a refresh never re-reads the same file for each question.
    """
    p = Path(futures_root) / "ohlcv" / "1h" / f"{symbol}.parquet"
    if not p.exists():
        return None, False
    try:
        present = set(pq.read_schema(p).names)
        df = pd.read_parquet(p, columns=[c for c in ("timestamp", "volume") if c in present])
    except Exception:  # noqa: BLE001 - 읽을 수 없는 파일은 tail 부재로 취급(미갱신으로 집계)
        return None, False
    stamps = pd.to_numeric(df["timestamp"], errors="coerce") if "timestamp" in df.columns else pd.Series(dtype="float64")
    valid = stamps.dropna()
    if valid.empty:
        return None, False
    tail = pd.Timestamp(int(valid.max()), unit="ms", tz="UTC")
    recent = stamps >= int((now - pd.Timedelta(hours=STALENESS_ACTIVE_WINDOW_HOURS)).value // 10**6)
    volume = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else pd.Series(dtype="float64")
    active = bool((volume[recent] > 0).any()) if len(volume) else False
    return tail, active


def _disk_tail_ts(futures_root: Path, symbol: str, now: pd.Timestamp) -> pd.Timestamp | None:
    return _kline_tail_state(futures_root, symbol, now)[0]


def _staleness_p90(states: Iterable[tuple[pd.Timestamp | None, bool]], now: pd.Timestamp) -> float:
    gaps = [(now - tail).total_seconds() / 3600.0 for tail, active in states if tail is not None and active]
    if not gaps:
        return float("inf")
    # p90 -- 소수의 상장폐지 심볼(무한 gap)이 지표를 오염시키지 않도록.
    # 실제 시스템 장애 시엔 대부분 심볼이 정체되므로 p90도 함께 상승한다.
    return float(pd.Series(gaps).quantile(0.90))


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


def market_data_staleness_hours(futures_root: Path, *, now: pd.Timestamp, symbols: Iterable[str] | None = None) -> float:
    root = Path(futures_root) / "ohlcv" / "1h"
    if not root.exists():
        return float("inf")
    wanted = {str(s) for s in symbols} if symbols is not None else None
    names = [p.stem for p in root.glob("*.parquet") if not is_temp_artifact(p.name)]
    if wanted is not None:
        names = [name for name in names if name in wanted]
    return _staleness_p90((_kline_tail_state(futures_root, name, now) for name in names), now)


def _expected_kline_tail(now: pd.Timestamp) -> pd.Timestamp:
    now_utc = pd.Timestamp(now).tz_convert("UTC")
    return now_utc.floor("h") - pd.Timedelta(hours=1)


def refresh_live_market_data(
    futures_root: Path,
    *,
    now: pd.Timestamp,
    lookback_days: int,
    max_workers: int,
    deadline_s: float,
    min_symbols: int,
    max_fail_fraction: float,
    symbols: list[str],
    collector: Any | None = None,
    listed_symbols: frozenset[str] | None = None,
    klines_only_symbols: frozenset[str] = frozenset(),
    seed_lookback_days: int = 150,
    required_symbols: frozenset[str] = frozenset(),
) -> RefreshReport:
    """Bring every universe symbol's 1h klines and settled funding current, then verify it on disk.

    Two phases share one deadline. The klines phase runs first for all symbols, then the funding
    phase, so a deadline can only cut funding, never the bars the signal ranks on. Within each phase
    ``required_symbols`` (ledger and book names) are submitted first. Success is measured on disk
    after the phases, not by the absence of an exception. A symbol is current only when its stored
    1h tail reaches the last closed hour and, unless klines-only, its funding tail is fresh. Symbols
    skipped by the deadline, left empty by the venue, or whose 4xx was swallowed are counted as not
    current.

    ``ok`` requires every required symbol to be current on its planes, at least ``min_symbols``
    current symbols, not-current symbols (failed + deadline_skipped + incomplete) within
    ``ceil(max_fail_fraction x total)``, and no IP or funding block.

    Raises:
        ColdUniverseError: fewer listed symbols than ``min_symbols``.
    """
    t0 = time.perf_counter()
    symbols_list = list(symbols)

    symbols_list, absent = split_absent_symbols(symbols_list, listed_symbols)
    if absent:
        _logger.warning("[DATA] stage=refresh_live_market_data absent_symbols=%d sample=%s", len(absent), ",".join(absent[:ABSENT_LOG_SAMPLE]))

    total = len(symbols_list)
    if total < min_symbols:
        raise ColdUniverseError(f"universe {total} < min {min_symbols}")

    if collector is None:
        collector = DataCollector()

    deadline_ts = time.perf_counter() + float(deadline_s)
    funding_window_start = now - pd.Timedelta(days=lookback_days)
    expected_tail = _expected_kline_tail(now)

    ip_blocked = threading.Event()
    funding_blocked = threading.Event()

    klines_only = frozenset(klines_only_symbols) if klines_only_symbols is not None else frozenset()
    required = frozenset(required_symbols) & set(symbols_list) if required_symbols is not None else frozenset()

    # 디스크 상태는 갱신 전·후 각 한 번만 읽는다. 갱신하지 않은(이미 current인) 심볼은 사전 상태를 재사용.
    kline_state: dict[str, tuple[pd.Timestamp | None, bool]] = {}
    funding_fresh: dict[str, bool] = {}

    def _observe(sym: str) -> None:
        kline_state[sym] = _kline_tail_state(futures_root, sym, now)
        funding_fresh[sym] = sym in klines_only or _funding_fresh_on_disk(futures_root, sym, now, window_start=funding_window_start)

    def _symbol_current(sym: str) -> bool:
        tail = kline_state[sym][0]
        return tail is not None and tail >= expected_tail and funding_fresh[sym]

    for sym in symbols_list:
        _observe(sym)

    def _priority_order(names: Iterable[str]) -> list[str]:
        names = list(names)
        req = sorted(s for s in names if s in required)
        rest = sorted(s for s in names if s not in required)
        return req + rest

    fresh_set = {sym for sym in symbols_list if _symbol_current(sym)}
    pending = [sym for sym in symbols_list if sym not in fresh_set]

    def _klines_start(sym: str) -> str:
        tail = kline_state[sym][0]
        if tail is not None:
            start = max(tail - pd.Timedelta(hours=2), now - pd.Timedelta(days=lookback_days))
        else:
            start = now - pd.Timedelta(days=seed_lookback_days)
        return str(start)

    klines_outcome: dict[str, str] = {}

    def _fetch_klines(sym: str) -> str:
        if time.perf_counter() > deadline_ts:
            return "deadline"
        if ip_blocked.is_set():
            return "ip_skip"
        try:
            collector.ensure_ohlcv_data(sym, "1h", _klines_start(sym), str(now))
        except BinanceIpBlockedError as exc:
            ip_blocked.set()
            _logger.error("[DATA] stage=refresh_live_market_data ip_blocked=True symbol=%s http_code=%d", sym, exc.http_code)
            return "ip"
        except Exception as exc:  # noqa: BLE001
            _logger.warning("[DATA] refresh_live_universe symbol=%s failed error=%s", sym, exc)
            return "klines_failed"
        return "klines_ok"

    t_klines = time.perf_counter()
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_klines, sym): sym for sym in _priority_order(pending)}
            for fut in concurrent.futures.as_completed(futures):
                sym = futures[fut]
                try:
                    klines_outcome[sym] = fut.result()
                except Exception as exc:  # noqa: BLE001  # pragma: no cover - _fetch_klines handles all known failures internally
                    _logger.warning("[DATA] symbol worker exception error=%s", exc)
                    klines_outcome[sym] = "klines_failed"
    klines_elapsed_s = time.perf_counter() - t_klines

    funding_outcome: dict[str, str] = {}
    funding_deadline_hit = False

    def _fetch_funding(sym: str) -> str:
        if time.perf_counter() > deadline_ts:
            return "deadline"
        if funding_blocked.is_set():
            return "funding_skip"
        try:
            collector.ensure_funding_data(sym, str(funding_window_start), str(now))
        except BinanceIpBlockedError as exc:
            funding_blocked.set()
            _logger.error("[DATA] stage=refresh_live_market_data funding_blocked=True symbol=%s http_code=%d", sym, exc.http_code)
            return "funding_blocked"
        except Exception as exc:  # noqa: BLE001
            _logger.warning("[DATA] funding symbol=%s failed error=%r", sym, exc)
            return "funding_failed"
        return "funding_ok"

    t_funding = time.perf_counter()
    funding_pending = [sym for sym in pending if sym not in klines_only and klines_outcome.get(sym) not in ("deadline", "ip", "ip_skip", "klines_failed")]
    if funding_pending and not ip_blocked.is_set():
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_funding, sym): sym for sym in _priority_order(funding_pending)}
            for fut in concurrent.futures.as_completed(futures):
                sym = futures[fut]
                try:
                    funding_outcome[sym] = fut.result()
                except Exception as exc:  # noqa: BLE001  # pragma: no cover - _fetch_funding handles all known failures internally
                    _logger.warning("[DATA] symbol worker exception error=%s", exc)
                    funding_outcome[sym] = "funding_failed"
    funding_elapsed_s = time.perf_counter() - t_funding

    fresh = len(fresh_set)
    refreshed = 0
    failed = 0
    deadline_skipped = 0
    incomplete = 0
    not_current: list[str] = []
    for sym in pending:
        _observe(sym)
    for sym in symbols_list:
        if sym in fresh_set:
            continue
        if _symbol_current(sym):
            refreshed += 1
            continue
        not_current.append(sym)
        if klines_outcome.get(sym) == "deadline":
            deadline_skipped += 1
            continue
        if funding_outcome.get(sym) == "deadline":
            funding_deadline_hit = True
            incomplete += 1
            continue
        if klines_outcome.get(sym) in ("ip", "ip_skip") or funding_outcome.get(sym) in ("ip_skip", "funding_blocked", "funding_skip", "funding_failed") or klines_outcome.get(sym) == "klines_failed":
            failed += 1
            continue
        incomplete += 1

    deadline_hit = deadline_skipped > 0 or funding_deadline_hit

    staleness = _staleness_p90((kline_state[sym] for sym in symbols_list), now)
    funding_stale = sum(1 for sym in symbols_list if not funding_fresh[sym])
    required_incomplete = tuple(sorted(sym for sym in required if not _symbol_current(sym)))
    not_current_sample = tuple(sorted(not_current)[:ABSENT_LOG_SAMPLE])
    not_current_count = failed + deadline_skipped + incomplete
    ok = (
        (fresh + refreshed) >= min_symbols
        and not_current_count <= math.ceil(max_fail_fraction * total)
        and not required_incomplete
        and not ip_blocked.is_set()
        and not funding_blocked.is_set()
    )
    elapsed_s = time.perf_counter() - t0

    _logger.info(
        "[DATA] stage=refresh_live_market_data total=%d fresh=%d refreshed=%d failed=%d deadline_skipped=%d incomplete=%d required_incomplete=%d deadline_hit=%s staleness_h=%.1f klines_elapsed_s=%.1f funding_elapsed_s=%.1f elapsed_s=%.1f ok=%s ip_blocked=%s funding_stale=%d absent=%d funding_blocked=%s",
        total, fresh, refreshed, failed, deadline_skipped, incomplete, len(required_incomplete), deadline_hit, staleness, klines_elapsed_s, funding_elapsed_s, elapsed_s, ok, ip_blocked.is_set(), funding_stale, len(absent), funding_blocked.is_set(),
    )
    if not_current_sample:
        _logger.warning("[DATA] stage=refresh_live_market_data not_current_sample=%s", ",".join(not_current_sample))

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
        funding_blocked=funding_blocked.is_set(),
        incomplete=incomplete,
        required_incomplete=required_incomplete,
        not_current_sample=not_current_sample,
        klines_elapsed_s=float(klines_elapsed_s),
        funding_elapsed_s=float(funding_elapsed_s),
    )


def refresh_funding_tails(
    futures_root: Path,
    *,
    now: pd.Timestamp,
    lookback_days: int,
    symbols: Sequence[str],
    deadline_s: float,
    max_workers: int,
    collector: Any | None = None,
) -> FundingPrefetchReport:
    """Best-effort off-window funding fetch so the nightly critical path finds most tails already fresh.

    Uses the same collector, freshness rule and rate limiter as the nightly refresh. It only
    fetches settlements already published at ``now``, so it is causal by construction and never
    marks anything current by itself.

    Returns:
        Counts of fetched / already fresh / failed symbols and elapsed seconds.
    """
    t0 = time.perf_counter()
    names = sorted({str(s) for s in symbols})
    if collector is None:
        collector = DataCollector()
    deadline_ts = time.perf_counter() + float(deadline_s)
    funding_window_start = now - pd.Timedelta(days=lookback_days)
    funding_blocked = threading.Event()

    def _fetch(sym: str) -> str:
        if _funding_fresh_on_disk(futures_root, sym, now, window_start=funding_window_start):
            return "fresh"
        if time.perf_counter() > deadline_ts:
            return "deadline"
        if funding_blocked.is_set():
            return "funding_skip"
        try:
            collector.ensure_funding_data(sym, str(funding_window_start), str(now))
        except BinanceIpBlockedError as exc:
            funding_blocked.set()
            _logger.error("[DATA] stage=funding_prefetch funding_blocked=True symbol=%s http_code=%d", sym, exc.http_code)
            return "funding_blocked"
        except Exception as exc:  # noqa: BLE001
            _logger.warning("[DATA] stage=funding_prefetch symbol=%s failed error=%r", sym, exc)
            return "failed"
        return "fetched"

    fetched = 0
    fresh = 0
    failed = 0
    deadline_skipped = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, sym): sym for sym in names}
        for fut in concurrent.futures.as_completed(futures):
            try:
                outcome = fut.result()
            except Exception as exc:  # noqa: BLE001  # pragma: no cover - _fetch handles all known failures internally
                _logger.warning("[DATA] stage=funding_prefetch worker exception error=%s", exc)
                failed += 1
                continue
            if outcome == "fresh":
                fresh += 1
            elif outcome == "fetched":
                fetched += 1
            elif outcome == "deadline":
                deadline_skipped += 1
            else:
                failed += 1
    elapsed_s = time.perf_counter() - t0
    _logger.info(
        "[DATA] stage=funding_prefetch total=%d fetched=%d fresh=%d failed=%d deadline_skipped=%d funding_blocked=%s elapsed_s=%.1f",
        len(names), fetched, fresh, failed, deadline_skipped, funding_blocked.is_set(), elapsed_s,
    )
    return FundingPrefetchReport(
        total=len(names),
        fetched=fetched,
        fresh=fresh,
        failed=failed,
        deadline_skipped=deadline_skipped,
        funding_blocked=funding_blocked.is_set(),
        elapsed_s=float(elapsed_s),
    )
