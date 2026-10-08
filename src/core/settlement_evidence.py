"""Lake evidence measurement and registry completeness audit for instrument lifecycles.

Read-only measurement over the 3m/1m execution archives: tail profiles, settlement price
derivation (flat-1h-klines rule, then the TWAP proxy), envelope checks, and the I6
completeness audit that fails closed before any replay whose census ends
unexplained. Never writes; never mutates the registry.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.common.errors import DataIntegrityError
from src.core.instrument_settlements import (
    InstrumentSettlementRecord,
    InstrumentSettlementRegistry,
)
from src.core.params import (
    SETTLEMENT_AUDIT_MIN_TRAILING_FLAT_BARS,
    SETTLEMENT_EVIDENCE_MIN_FLAT_BARS,
    SETTLEMENT_EVIDENCE_PRICE_RTOL,
    SETTLEMENT_PRICE_ENVELOPE_LOOKBACK,
    SETTLEMENT_PROXY_MIN_BARS,
    SETTLEMENT_PROXY_TWAP_WINDOW,
)

_logger = logging.getLogger(__name__)

_REQUIRED_BAR_COLUMNS: Final[tuple[str, ...]] = ("timestamp", "open", "high", "low", "close", "volume")
_MS_PER_HOUR: Final[int] = 3_600_000


@dataclass(frozen=True, slots=True)
class SettlementEvidence:
    """Venue-published settlement price of a delivered (delisted) perpetual."""

    symbol: str
    delivery_time: pd.Timestamp
    price: Decimal
    flat_bars: int
    source: str  # "flat_1h_klines"


def _require_utc(stamp: pd.Timestamp, label: str) -> pd.Timestamp:
    ts = pd.Timestamp(stamp)
    if ts.tzinfo is None:
        raise DataIntegrityError(f"{label} must be timezone-aware UTC")
    return ts.tz_convert("UTC")


def settlement_evidence_from_bars(
    hourly_path: Path,
    *,
    symbol: str,
    delivery_time: pd.Timestamp,
    min_flat_bars: int,
    price_rtol: float,
) -> SettlementEvidence | None:
    """Settlement price read from the venue's post-delivery flat 1h klines, or None when not yet evidenced.

    After delivery the venue publishes zero-volume bars with open == high == low == close at the
    settlement price. Only a contiguous run of at least ``min_flat_bars`` such bars, starting at or
    after ``delivery_time`` and agreeing on close within ``price_rtol``, counts as evidence. Any
    traded bar after delivery, a non-flat bar, or disagreement returns None. The price is
    therefore never synthesized from the last traded close, a mark candle, or zero.

    Raises:
        DataIntegrityError: file unreadable or lacks timestamp/open/high/low/close/volume columns.
    """
    delivery = _require_utc(delivery_time, "delivery_time")
    if min_flat_bars < 1:
        raise DataIntegrityError("min_flat_bars must be >= 1")
    path = Path(hourly_path)
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise DataIntegrityError(f"settlement bars unreadable: {path.name}") from exc
    if not all(col in frame.columns for col in _REQUIRED_BAR_COLUMNS):
        raise DataIntegrityError(f"settlement bars lack required columns: {path.name}")
    if frame.empty:
        return None
    stamps = pd.to_numeric(frame["timestamp"], errors="coerce")
    work = pd.DataFrame(
        {
            "timestamp": stamps,
            "open": pd.to_numeric(frame["open"], errors="coerce"),
            "high": pd.to_numeric(frame["high"], errors="coerce"),
            "low": pd.to_numeric(frame["low"], errors="coerce"),
            "close": pd.to_numeric(frame["close"], errors="coerce"),
            "volume": pd.to_numeric(frame["volume"], errors="coerce"),
        }
    ).dropna()
    if work.empty:
        return None
    work = work.sort_values("timestamp", kind="mergesort")
    delivery_ms = int(delivery.value // 1_000_000)
    post = work.loc[work["timestamp"] >= delivery_ms]
    if len(post) < min_flat_bars:
        return None
    post_times = [int(v) for v in post["timestamp"].tolist()]
    for prev_ms, cur_ms in pairwise(post_times):
        if cur_ms - prev_ms != _MS_PER_HOUR:
            return None
    for _, row in post.iterrows():
        if float(row["volume"]) != 0.0:
            return None
        o, h, low, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        if not (o == h == low == c):
            return None
    closes = [float(v) for v in post["close"].tolist()]
    ref = closes[0]
    if ref == 0.0:
        if any(c != 0.0 for c in closes):
            return None
    elif max(abs(c - ref) for c in closes) / abs(ref) > price_rtol:
        return None
    return SettlementEvidence(
        symbol=str(symbol),
        delivery_time=delivery,
        price=Decimal(str(ref)),
        flat_bars=len(post),
        source="flat_1h_klines",
    )

_REQUIRED_3M_COLUMNS: Final[tuple[str, ...]] = ("timestamp", "open", "high", "low", "close", "quote_vol")
_REQUIRED_1H_COLUMNS: Final[tuple[str, ...]] = ("timestamp", "open", "high", "low", "close", "quote_vol")
_BAR_STEP: Final[pd.Timedelta] = pd.Timedelta(minutes=3)
_MS_PER_MINUTE: Final[int] = 60_000


@dataclass(frozen=True, slots=True)
class SymbolTailProfile:
    """End-of-file shape of one 3m archive, measured from the file suffix only.

    Attributes:
        symbol: Upper-case symbol.
        first_bar: First 3m label in the file.
        last_bar: Last 3m label in the file.
        last_liquid_bar: Last label with quote_vol > 0, None if never liquid.
        trailing_flat_bars: Length of the contiguous flat run ending at ``last_bar``.
    """

    symbol: str
    first_bar: pd.Timestamp
    last_bar: pd.Timestamp
    last_liquid_bar: pd.Timestamp | None
    trailing_flat_bars: int


_tail_cache: dict[tuple[str, int, int], SymbolTailProfile] = {}


def _coerce_bars(table: object, name: str) -> pd.DataFrame:
    frame = table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table)
    out = pd.DataFrame()
    out["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    for column in ("open", "high", "low", "close", "quote_vol"):
        out[column] = pd.to_numeric(frame[column], errors="coerce")
    if out.empty:
        raise DataIntegrityError(f"3m archive has no placeable bars: {name}")
    if not np.isfinite(out.to_numpy(dtype="float64")).all():
        raise DataIntegrityError(f"settlement archive contains non-finite bars: {name}")
    return out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def _masks(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    quote_vol = frame["quote_vol"].to_numpy(dtype="float64", na_value=float("nan"))
    opens = frame["open"].to_numpy(dtype="float64", na_value=float("nan"))
    highs = frame["high"].to_numpy(dtype="float64", na_value=float("nan"))
    lows = frame["low"].to_numpy(dtype="float64", na_value=float("nan"))
    closes = frame["close"].to_numpy(dtype="float64", na_value=float("nan"))
    liquid = quote_vol > 0.0
    flat = (
        (quote_vol == 0.0)
        & (opens == highs)
        & (highs == lows)
        & (lows == closes)
    )
    return pd.Series(liquid), pd.Series(flat)


def _ms_to_utc(value: int) -> pd.Timestamp:
    return pd.Timestamp(int(value), unit="ms", tz="UTC")


def _open_tail_parquet(target: Path) -> pq.ParquetFile:
    try:
        handle = pq.ParquetFile(target)
        names = set(handle.schema.names)
    except Exception as exc:
        raise DataIntegrityError(f"3m archive unreadable: {target.name}") from exc
    if any(column not in names for column in _REQUIRED_3M_COLUMNS):
        raise DataIntegrityError(f"3m archive lacks required columns: {target.name}")
    if handle.metadata.num_rows == 0:
        raise DataIntegrityError(f"3m archive is empty: {target.name}")
    return handle


def _suffix_first_ms(handle: pq.ParquetFile) -> int | None:
    try:
        stamp_index = handle.schema.names.index("timestamp")
        statistics = handle.metadata.row_group(0).column(stamp_index).statistics
        if statistics is not None and statistics.has_min_max:
            return int(statistics.min)
    except Exception:
        return None
    return None


def _scan_suffix_chunks(handle: pq.ParquetFile, target: Path) -> tuple[pd.DataFrame, int | None]:
    chunks: list[pd.DataFrame] = []
    last_liquid_ms: int | None = None
    try:
        for group in range(handle.metadata.num_row_groups - 1, -1, -1):
            chunk = _coerce_bars(
                handle.read_row_group(group, columns=list(_REQUIRED_3M_COLUMNS)), target.name,
            )
            chunks.append(chunk)
            liquid, _flat = _masks(chunk)
            hit = chunk.loc[liquid.fillna(False).to_numpy(dtype=bool), "timestamp"]
            if len(hit):
                last_liquid_ms = int(hit.iloc[-1])
                break
    except DataIntegrityError:
        raise
    except Exception as exc:
        raise DataIntegrityError(f"3m archive unreadable: {target.name}") from exc
    combined = pd.concat(list(reversed(chunks)), ignore_index=True)
    return combined.sort_values("timestamp", kind="mergesort").reset_index(drop=True), last_liquid_ms


def measure_symbol_tail(path: Path) -> SymbolTailProfile:
    """Measure one 3m archive's tail with a reverse row-group scan and column pruning.

    Reads row groups from the end until a liquid bar is found (or the file is exhausted), so the
    cost scales with the flat tail, never the full history, for normal symbols. Memoized per
    ``(resolved path, size, mtime_ns)`` within the process.

    Raises:
        DataIntegrityError: file unreadable, missing required columns, or empty.
    """
    target = Path(path)
    try:
        stat = target.stat()
    except OSError as exc:
        raise DataIntegrityError(f"3m archive unreadable: {target.name}") from exc
    try:
        resolved = str(target.resolve())
    except OSError as exc:
        raise DataIntegrityError(f"3m archive unreadable: {target.name}") from exc
    key = (resolved, stat.st_size, stat.st_mtime_ns)
    cached = _tail_cache.get(key)
    if cached is not None:
        return cached
    handle = _open_tail_parquet(target)
    first_ms = _suffix_first_ms(handle)
    combined, last_liquid_ms = _scan_suffix_chunks(handle, target)
    if first_ms is None:
        try:
            first_table = handle.read_row_group(0, columns=["timestamp"])
            first_ms = int(first_table.column("timestamp").to_pylist()[0])
        except Exception as exc:
            raise DataIntegrityError(f"3m archive unreadable: {target.name}") from exc
    _liquid_all, flat_all = _masks(combined)
    flat_values = flat_all.fillna(False).to_numpy(dtype=bool)
    trailing = 0
    for value in reversed(flat_values.tolist()):
        if not value:
            break
        trailing += 1
    profile = SymbolTailProfile(
        symbol=target.stem,
        first_bar=_ms_to_utc(int(first_ms)),
        last_bar=_ms_to_utc(int(combined["timestamp"].iloc[-1])),
        last_liquid_bar=_ms_to_utc(last_liquid_ms) if last_liquid_ms is not None else None,
        trailing_flat_bars=int(trailing),
    )
    _tail_cache[key] = profile
    return profile


@dataclass(frozen=True, slots=True)
class SettlementPriceEvidence:
    """Price, class, evidence description and digest derived from the lake for one lifecycle."""

    settlement_price: float
    price_source: str
    price_evidence: str
    evidence_digest: str


def _digest_bars(symbol: str, price_source: str, bars: list[list[float]]) -> str:
    payload = {"symbol": symbol, "price_source": price_source, "bars": bars}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _bar_row(frame: pd.DataFrame, position: int) -> list[float]:
    row = frame.iloc[position]
    return [
        int(row["timestamp"]),
        float(row["open"]),
        float(row["high"]),
        float(row["low"]),
        float(row["close"]),
        float(row["quote_vol"]),
    ]


def _read_required_frame(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path, columns=list(columns))
    except (OSError, ValueError) as exc:
        raise DataIntegrityError(f"settlement bars unreadable: {path.name}") from exc
    if frame.empty:
        raise DataIntegrityError(f"settlement bars are empty: {path.name}")
    return frame


def _require_utc_moment(value: pd.Timestamp, label: str) -> pd.Timestamp:
    if not isinstance(value, pd.Timestamp) or pd.isna(value):
        raise DataIntegrityError(f"{label} must be a valid timestamp")
    if value.tzinfo is None or value.utcoffset() != pd.Timedelta(0):
        raise DataIntegrityError(f"{label} must be timezone-aware UTC")
    return value.tz_convert("UTC")


def _ceil_hour(value: pd.Timestamp) -> pd.Timestamp:
    floored = value.floor("h")
    return floored if value == floored else floored + pd.Timedelta(hours=1)


def derive_settlement_price(
    ohlcv_root: Path, symbol: str, last_trade_at: pd.Timestamp,
) -> SettlementPriceEvidence | None:
    """Apply the price-source priority (flat_1h_klines, then twap30_proxy) to the lake.

    flat_1h_klines reuses the live rule with ``delivery_time = ceil_hour(last_trade_at)``,
    ``SETTLEMENT_EVIDENCE_MIN_FLAT_BARS`` and ``SETTLEMENT_EVIDENCE_PRICE_RTOL``.
    twap30_proxy averages liquid 3m closes in float64, requiring at least
    ``SETTLEMENT_PROXY_MIN_BARS`` liquid bars. Returns None when neither class
    applies (operator must curate); never synthesizes a price from zero, a mark, or the last close
    alone.

    Raises:
        DataIntegrityError: a required archive is unreadable or malformed.
    """
    last_trade = _require_utc_moment(last_trade_at, "last_trade_at")
    root = Path(ohlcv_root)
    minute_path = root / "3m" / f"{symbol}.parquet"
    if not minute_path.exists():
        raise DataIntegrityError(f"settlement bars unreadable: {minute_path.name}")
    minute = _read_required_frame(minute_path, _REQUIRED_3M_COLUMNS)
    minute = _coerce_bars(minute, minute_path.name)
    last_trade_ms = int(last_trade.value // 1_000_000)
    hourly_path = root / "1h" / f"{symbol}.parquet"
    if hourly_path.exists():
        delivery = _ceil_hour(last_trade)
        evidence = settlement_evidence_from_bars(
            hourly_path,
            symbol=symbol,
            delivery_time=delivery,
            min_flat_bars=SETTLEMENT_EVIDENCE_MIN_FLAT_BARS,
            price_rtol=SETTLEMENT_EVIDENCE_PRICE_RTOL,
        )
        if evidence is not None:
            price = float(evidence.price)
            hourly = _read_required_frame(hourly_path, _REQUIRED_1H_COLUMNS)
            hourly = _coerce_bars(hourly, hourly_path.name)
            delivery_ms = int(delivery.value // 1_000_000)
            post = hourly.loc[hourly["timestamp"] >= delivery_ms].head(SETTLEMENT_EVIDENCE_MIN_FLAT_BARS)
            bars = [_bar_row(hourly, int(index)) for index in post.index.tolist()]
            liquid_mask = minute["quote_vol"].to_numpy(dtype="float64", na_value=float("nan")) > 0.0
            liquid_positions = [index for index, value in enumerate(liquid_mask.tolist()) if value]
            if not liquid_positions:
                raise DataIntegrityError(f"settlement bars have no liquid bar: {minute_path.name}")
            bars.append(_bar_row(minute, int(liquid_positions[-1])))
            description = (
                f"flat_1h_klines delivery={delivery.isoformat()} flat_bars={evidence.flat_bars}"
                f" price={price!r}"
            )
            return SettlementPriceEvidence(
                settlement_price=price,
                price_source="flat_1h_klines",
                price_evidence=description,
                evidence_digest=_digest_bars(symbol, "flat_1h_klines", bars),
            )
    window_ms = int(SETTLEMENT_PROXY_TWAP_WINDOW.value // 1_000_000)
    stamps = minute["timestamp"].to_numpy(dtype="int64")
    quote_volumes = minute["quote_vol"].to_numpy(dtype="float64", na_value=float("nan"))
    in_window = (stamps >= last_trade_ms - window_ms) & (stamps < last_trade_ms) & (quote_volumes > 0.0)
    positions = [index for index, value in enumerate(in_window.tolist()) if value]
    if len(positions) < SETTLEMENT_PROXY_MIN_BARS:
        return None
    closes = minute["close"].iloc[positions].astype("float64")
    price = float(closes.mean())
    bars = [_bar_row(minute, int(position)) for position in positions]
    window_start = _ms_to_utc(last_trade_ms - window_ms)
    description = (
        f"twap30_proxy window=[{window_start.isoformat()}, {last_trade.isoformat()})"
        f" bars={len(positions)} price={price!r}"
    )
    return SettlementPriceEvidence(
        settlement_price=price,
        price_source="twap30_proxy",
        price_evidence=description,
        evidence_digest=_digest_bars(symbol, "twap30_proxy", bars),
    )


def settlement_price_within_envelope(ohlcv_root: Path, record: InstrumentSettlementRecord) -> bool:
    """True iff the price lies in the liquid 3m [min low, max high] over
    ``[last_trade_at - SETTLEMENT_PRICE_ENVELOPE_LOOKBACK, last_trade_at)``."""
    root = Path(ohlcv_root)
    path = root / "3m" / f"{record.symbol}.parquet"
    frame = _read_required_frame(path, ("timestamp", "low", "high", "quote_vol"))
    stamps = pd.to_numeric(frame["timestamp"], errors="coerce")
    lookback_ms = int(SETTLEMENT_PRICE_ENVELOPE_LOOKBACK.value // 1_000_000)
    last_ms = int(record.last_trade_at.value // 1_000_000)
    liquid = pd.to_numeric(frame["quote_vol"], errors="coerce") > 0.0
    in_window = (stamps >= last_ms - lookback_ms) & (stamps < last_ms) & liquid.fillna(False)
    window = frame.loc[in_window]
    if window.empty:
        return False
    low = float(pd.to_numeric(window["low"], errors="coerce").min())
    high = float(pd.to_numeric(window["high"], errors="coerce").max())
    return bool(low <= float(record.settlement_price) <= high)


@dataclass(frozen=True, slots=True)
class SettlementRegistryAuditReport:
    """Census symbols whose lifecycle end is unexplained, and records contradicting the lake.

    Attributes:
        missing: ``(symbol, reason)`` for census symbols that ended or carry a delisted flat tail
            before ``audit_end`` without a settlement or truncation record.
        stale: ``(event_id_or_symbol, reason)`` for records that no longer reconcile with the lake.
    """

    missing: tuple[tuple[str, str], ...]
    stale: tuple[tuple[str, str], ...]

    @property
    def complete(self) -> bool:
        """True when every required lifecycle is explained and every record reconciles."""
        return not self.missing and not self.stale


def audit_settlement_registry(
    ohlcv_root: Path, symbols: Sequence[str], *, audit_end: pd.Timestamp,
    registry: InstrumentSettlementRegistry,
) -> SettlementRegistryAuditReport:
    """Reconcile the registry with the execution archives of one replay census (I6).

    A census symbol whose first bar precedes ``audit_end`` requires a lifecycle explanation when
    (a) ``last_bar + 3m < audit_end`` (data ends inside the replay horizon), or (b)
    ``trailing_flat_bars >= SETTLEMENT_AUDIT_MIN_TRAILING_FLAT_BARS`` and
    ``last_liquid_bar + 3m < audit_end`` (forward-filled delisted tail). (a) is explained by a
    settlement record with ``last_trade_at == last_liquid_bar + 3m`` or a truncation record with
    ``data_end == last_bar + 3m``; (b) only by such a settlement record. Every settlement record of
    a census symbol with ``last_trade_at < audit_end`` must reconcile: the bar labelled
    ``last_trade_at - 3m`` exists and is liquid; no liquid bar has a label in
    ``[last_trade_at, delivery_at]``; the price is inside the envelope; a non-curated record's
    ``evidence_digest`` and ``settlement_price`` equal a fresh ``derive_settlement_price``.
    Symbols without a 3m archive are skipped (existing missing-source checks own them).

    Raises:
        DataIntegrityError: ``audit_end`` naive/non-UTC, or an archive unreadable.
    """
    end = _require_utc_moment(audit_end, "audit_end")
    root = Path(ohlcv_root)
    names = list(symbols)
    by_symbol: dict[str, list[InstrumentSettlementRecord]] = {}
    for record in registry.settlements:
        by_symbol.setdefault(record.symbol, []).append(record)
    missing: list[tuple[str, str]] = []
    stale: list[tuple[str, str]] = []
    required = 0

    for symbol in names:
        minute_path = root / "3m" / f"{symbol}.parquet"
        if not minute_path.exists():
            continue
        tail = measure_symbol_tail(minute_path)
        if tail.first_bar >= end:
            continue
        records = by_symbol.get(symbol, [])
        if _audit_symbol_coverage(symbol, tail, end, records, registry, missing):
            required += 1
        _audit_symbol_records(root, symbol, end, records, stale)
    _logger.info(
        "[DATA] stage=settlement_registry_audit symbols=%d required=%d missing=%d stale=%d audit_end=%s",
        len(names), required, len(missing), len(stale), end.isoformat(),
    )
    return SettlementRegistryAuditReport(missing=tuple(missing), stale=tuple(stale))


def _audit_symbol_coverage(
    symbol: str, tail: SymbolTailProfile, end: pd.Timestamp,
    records: list[InstrumentSettlementRecord],
    registry: InstrumentSettlementRegistry,
    missing: list[tuple[str, str]],
) -> bool:
    need_a = tail.last_bar + _BAR_STEP < end
    need_b = (
        tail.trailing_flat_bars >= SETTLEMENT_AUDIT_MIN_TRAILING_FLAT_BARS
        and tail.last_liquid_bar is not None
        and tail.last_liquid_bar + _BAR_STEP < end
    )
    if not (need_a or need_b):
        return False
    expected_trade = (
        tail.last_liquid_bar + _BAR_STEP if tail.last_liquid_bar is not None else None
    )
    settled = expected_trade is not None and any(
        record.last_trade_at == expected_trade for record in records
    )
    truncation = registry.truncation_for(symbol)
    truncated = truncation is not None and truncation.data_end == tail.last_bar + _BAR_STEP
    if need_b and not settled:
        liquid_since = (
            tail.last_liquid_bar.isoformat() if tail.last_liquid_bar is not None else "never"
        )
        missing.append((
            symbol,
            f"forward-filled flat tail of {tail.trailing_flat_bars} bars"
            f" since {liquid_since} with no settlement record",
        ))
    elif need_a and not settled and not truncated:
        missing.append((
            symbol,
            f"data ends at {tail.last_bar.isoformat()} with no settlement or truncation record",
        ))
    return True


def _audit_symbol_records(
    root: Path, symbol: str, end: pd.Timestamp,
    records: list[InstrumentSettlementRecord],
    stale: list[tuple[str, str]],
) -> None:
    frame: pd.DataFrame | None = None
    for record in records:
        if record.last_trade_at >= end:
            continue
        if frame is None:
            frame = _coerce_bars(
                _read_required_frame(root / "3m" / f"{symbol}.parquet", _REQUIRED_3M_COLUMNS),
                f"{symbol}.parquet",
            )
        reason = _record_is_stale(root, symbol, frame, record)
        if reason is not None:
            stale.append((record.event_id, reason))
    del frame


def _record_is_stale(
    root: Path, symbol: str, frame: pd.DataFrame, record: InstrumentSettlementRecord,
) -> str | None:
    stamps = frame["timestamp"].to_numpy(dtype="int64")
    quote_vol = frame["quote_vol"].to_numpy(dtype="float64", na_value=float("nan"))
    last_ms = int(record.last_trade_at.value // 1_000_000)
    delivery_ms = int(record.delivery_at.value // 1_000_000)
    opening_ms = last_ms - 3 * _MS_PER_MINUTE
    trade_positions = [
        index for index, stamp in enumerate(stamps.tolist()) if stamp == opening_ms
    ]
    if not trade_positions or not bool(quote_vol[trade_positions[-1]] > 0.0):
        return f"last trade bar {_ms_to_utc(opening_ms).isoformat()} absent or illiquid"
    if any(
        last_ms <= stamp <= delivery_ms and quote_vol[index] > 0.0
        for index, stamp in enumerate(stamps.tolist())
    ):
        return "liquid bar inside [last_trade_at, delivery_at]"
    if not settlement_price_within_envelope(root, record):
        return "price outside 24h liquid envelope"
    if record.price_source != "curated":
        fresh = derive_settlement_price(root, symbol, record.last_trade_at)
        if (
            fresh is None
            or fresh.settlement_price != float(record.settlement_price)
            or fresh.evidence_digest != record.evidence_digest
        ):
            return "evidence mismatch: re-derivation differs"
    return None


def assert_settlement_registry_complete(
    ohlcv_root: Path, symbols: Sequence[str], *, audit_end: pd.Timestamp,
    registry: InstrumentSettlementRegistry,
) -> None:
    """Fail closed before any replay when the audit is not complete.

    Raises:
        DataIntegrityError: "settlement registry incomplete: missing=[SYM (reason), ...]
            stale=[ID (reason), ...]; run `data build-settlement-registry` and review" listing
            every finding (never only the first).
    """
    report = audit_settlement_registry(ohlcv_root, symbols, audit_end=audit_end, registry=registry)
    if report.complete:
        return
    missing = ", ".join(f"{symbol} ({reason})" for symbol, reason in report.missing)
    stale = ", ".join(f"{identity} ({reason})" for identity, reason in report.stale)
    raise DataIntegrityError(
        f"settlement registry incomplete: missing=[{missing}] stale=[{stale}];"
        " run `data build-settlement-registry` and review",
    )
