"""Reconcile the committed source-gap registry against the local parquet lake."""

from __future__ import annotations

import contextlib
import itertools
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.paths import FUTURES_DATA_DIR
from src.mhs.source_gaps import (
    SourceGapInterval,
    SourceGapPlane,
    load_source_gap_registry,
)

_PLANE_STEP: dict[str, timedelta] = {
    "ohlcv_3m": timedelta(minutes=3),
    "ohlcv_1h": timedelta(hours=1),
    "funding": timedelta(hours=8),
}

_PLANE_TIMEFRAME: dict[str, str | None] = {
    "ohlcv_3m": "3m",
    "ohlcv_1h": "1h",
    "funding": None,
}


@dataclass(frozen=True, slots=True)
class SourceGapAuditReport:
    """Difference between the committed registry and what the local lake now holds.

    The report is advisory evidence, never an automatic policy change: narrowing or
    resolving an exclusion alters which history a certified backtest may trade, so an
    operator commits the result deliberately.

    Attributes:
        resolved: Committed records whose interval is now fully present locally.
        narrowed: Committed records replaced by the exact measured intervals.
        unchanged: Committed records confirmed still absent with identical bounds.
        discovered: Measured intervals absent from the registry entirely.
    """

    resolved: tuple[SourceGapInterval, ...] = ()
    narrowed: tuple[SourceGapInterval, ...] = ()
    unchanged: tuple[SourceGapInterval, ...] = ()
    discovered: tuple[SourceGapInterval, ...] = ()


def _plane_step(plane: SourceGapPlane) -> timedelta:
    try:
        return _PLANE_STEP[plane]
    except KeyError as exc:
        raise DataIntegrityError(f"unknown source-gap plane {plane!r}") from exc


def _require_window(start: pd.Timestamp, end: pd.Timestamp) -> None:
    for name, moment in (("start", start), ("end", end)):
        if not isinstance(moment, pd.Timestamp):
            raise DataIntegrityError(f"{name} must be a tz-aware UTC Timestamp")
        if moment.tzinfo is None:
            raise DataIntegrityError(f"{name} must be tz-aware UTC")
        if moment.utcoffset() != timedelta(0):
            raise DataIntegrityError(f"{name} must be UTC")
    if end <= start:
        raise DataIntegrityError("window end must exceed start")


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace("_", "")


def _lake_path(symbol: str, plane: SourceGapPlane, data_root: Path | None) -> Path:
    root = Path(data_root) if data_root is not None else FUTURES_DATA_DIR
    safe = _safe_symbol(symbol)
    if plane in ("ohlcv_3m", "ohlcv_1h"):
        timeframe = _PLANE_TIMEFRAME[plane]
        assert timeframe is not None
        return root / "ohlcv" / timeframe / f"{safe}.parquet"
    return root / "funding" / f"{safe}.parquet"


def _lake_symbols(plane: SourceGapPlane, data_root: Path | None) -> frozenset[str]:
    from src.market_data.storage.ohlcv import is_temp_artifact

    root = Path(data_root) if data_root is not None else FUTURES_DATA_DIR
    directory = root / "ohlcv" / str(_PLANE_TIMEFRAME[plane]) if plane in ("ohlcv_3m", "ohlcv_1h") else root / "funding"
    try:
        names = [p.name for p in directory.glob("*.parquet")]
    except OSError:
        return frozenset()
    return frozenset(Path(n).stem for n in names if not is_temp_artifact(n))


def _read_bar_times(path: Path, plane: SourceGapPlane) -> list[pd.Timestamp]:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise DataIntegrityError(f"source-gap audit unreadable parquet: {path}") from exc
    if frame.empty:
        return []
    ts: pd.Series | None = None
    if "timestamp" in frame.columns:
        numeric = pd.to_numeric(frame["timestamp"], errors="coerce").dropna()
        if numeric.empty:
            return []
        ts = pd.to_datetime(numeric.astype("int64"), unit="ms", utc=True)
    elif "datetime" in frame.columns:
        ts = pd.to_datetime(frame["datetime"], utc=True, errors="coerce").dropna()
    else:
        raise DataIntegrityError(f"source-gap audit parquet missing time column: {path}")
    uniq = sorted(set(ts.tolist()))
    out: list[pd.Timestamp] = []
    for t in uniq:
        stamp = pd.Timestamp(t)
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        out.append(stamp)
    return out


def _iso_z(moment: datetime) -> str:
    utc = moment.astimezone(UTC)
    return utc.isoformat().replace("+00:00", "Z")


def _measure_one_symbol(
    symbol: str,
    plane: SourceGapPlane,
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_root: Path | None,
) -> tuple[SourceGapInterval, ...]:
    step = _plane_step(plane)
    path = _lake_path(symbol, plane, data_root)
    verified = end.to_pydatetime()
    if not path.exists():
        return (
            SourceGapInterval(
                symbol=symbol,
                plane=plane,
                start=start.to_pydatetime(),
                end=None,
                reason="SOURCE_ABSENT",
                evidence=(
                    f"Local lake {plane} has no file for {symbol} over "
                    f"[{start.isoformat()}, {end.isoformat()}); open-ended edge, "
                    f"never DELISTED by measurement alone."
                ),
                verified_at=verified,
                resolved_at=None,
            ),
        )
    observed = _read_bar_times(path, plane)
    if not observed:
        return (
            SourceGapInterval(
                symbol=symbol,
                plane=plane,
                start=start.to_pydatetime(),
                end=None,
                reason="SOURCE_ABSENT",
                evidence=(
                    f"Local lake {plane} file for {symbol} holds no bars over "
                    f"[{start.isoformat()}, {end.isoformat()}); open-ended edge."
                ),
                verified_at=verified,
                resolved_at=None,
            ),
        )
    intervals: list[SourceGapInterval] = []
    for left, right in itertools.pairwise(observed):
        if right <= start or left >= end:
            continue
        if right - left > step:
            gap_start = left + step
            gap_end = right
            clipped_start = max(gap_start, start)
            clipped_end = min(gap_end, end)
            if clipped_start < clipped_end:
                n_bars = round((clipped_end - clipped_start) / step)
                intervals.append(
                    SourceGapInterval(
                        symbol=symbol,
                        plane=plane,
                        start=clipped_start.to_pydatetime(),
                        end=clipped_end.to_pydatetime(),
                        reason="SOURCE_ABSENT",
                        evidence=(
                            f"Local lake {plane} interior gap for {symbol}: "
                            f"{n_bars} missing bars over "
                            f"[{clipped_start.isoformat()}, {clipped_end.isoformat()}); "
                            f"bounded by observed bars."
                        ),
                        verified_at=verified,
                        resolved_at=None,
                    )
                )
    first = observed[0]
    last = observed[-1]
    if first > start and first < end:
        n_bars = round((min(first, end) - start) / step)
        intervals.append(
            SourceGapInterval(
                symbol=symbol,
                plane=plane,
                start=start.to_pydatetime(),
                end=min(first, end).to_pydatetime(),
                reason="SOURCE_ABSENT",
                evidence=(
                    f"Local lake {plane} leading edge for {symbol}: "
                    f"{max(n_bars, 0)} bars before first observed bar "
                    f"{first.isoformat()}; listing edge, never DELISTED "
                    f"by measurement alone."
                ),
                verified_at=verified,
                resolved_at=None,
            )
        )
    elif first >= end:
        intervals.append(
            SourceGapInterval(
                symbol=symbol,
                plane=plane,
                start=start.to_pydatetime(),
                end=end.to_pydatetime(),
                reason="SOURCE_ABSENT",
                evidence=(
                    f"Local lake {plane} leading edge for {symbol}: whole window "
                    f"precedes first observed bar {first.isoformat()}."
                ),
                verified_at=verified,
                resolved_at=None,
            )
        )
    if last + step < end:
        tail_start = max(last + step, start)
        intervals.append(
            SourceGapInterval(
                symbol=symbol,
                plane=plane,
                start=tail_start.to_pydatetime(),
                end=None,
                reason="SOURCE_ABSENT",
                evidence=(
                    f"Local lake {plane} trailing edge for {symbol}: no bars at or "
                    f"after {tail_start.isoformat()} through {end.isoformat()} "
                    f"(last observed {last.isoformat()}); open-ended edge, "
                    f"never DELISTED by measurement alone."
                ),
                verified_at=verified,
                resolved_at=None,
            )
        )
    intervals.sort(key=lambda iv: (iv.start, iv.end or datetime.max.replace(tzinfo=UTC)))
    return tuple(intervals)


def measure_source_gaps(
    symbols: Sequence[str],
    *,
    plane: SourceGapPlane,
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_root: Path | None = None,
) -> tuple[SourceGapInterval, ...]:
    """Measure interior absences directly from the local parquet lake.

    Absence is judged by the plane's own native step: a missing bar between two observed
    bars is an interior gap, while the span before a symbol's first bar and after its last
    is reported as an open-ended edge so listing and delisting are never mistaken for a
    recoverable hole.

    Args:
        symbols: Symbols to measure; a symbol without a file yields one open-ended record.
        plane: Source plane to measure.
        start: Inclusive UTC measurement start.
        end: Exclusive UTC measurement end.
        data_root: OHLCV root override; defaults to the canonical futures lake.
    Returns:
        Measured intervals with `reason="SOURCE_ABSENT"` and generated evidence text.
    Raises:
        DataIntegrityError: The window is empty or a parquet file is unreadable.
    """
    _plane_step(plane)
    _require_window(start, end)
    cols = list(symbols)
    if len(set(cols)) != len(cols):
        raise DataIntegrityError("measure_source_gaps symbols must not repeat a name")
    if not cols:
        raise DataIntegrityError("measure_source_gaps symbols must not be empty")
    out: list[SourceGapInterval] = []
    for symbol in cols:
        if not isinstance(symbol, str) or not symbol or symbol != symbol.upper():
            raise DataIntegrityError(f"measure_source_gaps symbol must be upper-case: {symbol!r}")
        out.extend(_measure_one_symbol(symbol, plane, start, end, data_root))
    out.sort(key=lambda iv: (iv.symbol, iv.plane, iv.start))
    return tuple(out)


def _as_inf(moment: datetime | None) -> datetime:
    if moment is None:
        return datetime.max.replace(tzinfo=UTC)
    return moment


def _overlaps(
    a_start: datetime, a_end: datetime | None, b_start: datetime, b_end: datetime | None
) -> bool:
    return a_start < _as_inf(b_end) and b_start < _as_inf(a_end)


def _clip_to_commit(
    measured: SourceGapInterval,
    commit: SourceGapInterval,
    window_start: datetime,
    window_end: datetime,
) -> tuple[datetime, datetime | None] | None:
    clip_start = max(measured.start, commit.start, window_start)
    if measured.end is None and commit.end is None:
        clip_end: datetime | None = None
    else:
        ends: list[datetime] = [window_end]
        if measured.end is not None:
            ends.append(measured.end)
        if commit.end is not None:
            ends.append(commit.end)
        clip_end = min(ends)
    if clip_end is not None and not (clip_start < clip_end):
        return None
    return (clip_start, clip_end)


def _merge_clipped(
    pieces: list[tuple[datetime, datetime | None]],
) -> list[tuple[datetime, datetime | None]]:
    """Merge overlapping or touching clipped spans into a minimal ordered cover."""
    if not pieces:
        return []
    ordered = sorted(pieces, key=lambda p: (p[0], _as_inf(p[1])))
    merged: list[tuple[datetime, datetime | None]] = [ordered[0]]
    for start, end in ordered[1:]:
        cur_start, cur_end = merged[-1]
        # cur_end is None이면 개방 구간이라 뒤따르는 조각을 모두 흡수한다.
        if cur_end is None:
            continue
        if start > cur_end:
            merged.append((start, end))
            continue
        merged[-1] = (cur_start, None if end is None else max(cur_end, end))
    return merged


def audit_source_gap_registry(
    *,
    plane: SourceGapPlane,
    start: pd.Timestamp,
    end: pd.Timestamp,
    symbols: Sequence[str] | None = None,
    registry_path: Path | None = None,
    data_root: Path | None = None,
) -> SourceGapAuditReport:
    """Reconcile the committed registry against freshly measured local evidence.

    Args:
        plane: Source plane to reconcile.
        start: Inclusive UTC reconciliation start.
        end: Exclusive UTC reconciliation end.
        symbols: Symbols to reconcile; None reconciles every symbol the registry names
            plus every symbol present in the lake for that plane.
        registry_path: Registry override forwarded to the loader.
        data_root: OHLCV root override forwarded to measurement.
    Returns:
        Classified differences; an empty report means the registry matches the lake.
    Raises:
        DataIntegrityError: Registry load or measurement fails.
    """
    _plane_step(plane)
    _require_window(start, end)
    records = load_source_gap_registry(registry_path)
    active = [iv for iv in records if iv.resolved_at is None and iv.plane == plane]
    if symbols is None:
        wanted: frozenset[str] | None = None
        scope = sorted(
            {iv.symbol for iv in records if iv.plane == plane}
            | set(_lake_symbols(plane, data_root))
        )
    else:
        wanted = frozenset(symbols)
        scope = sorted(wanted)
    if not scope:
        return SourceGapAuditReport()
    measured = measure_source_gaps(scope, plane=plane, start=start, end=end, data_root=data_root)
    by_symbol: dict[str, list[SourceGapInterval]] = {}
    for gap in measured:
        by_symbol.setdefault(gap.symbol, []).append(gap)
    window_start_dt = start.to_pydatetime()
    window_end_dt = end.to_pydatetime()
    resolved: list[SourceGapInterval] = []
    narrowed: list[SourceGapInterval] = []
    unchanged: list[SourceGapInterval] = []
    discovered: list[SourceGapInterval] = []
    committed_by_symbol: dict[str, list[SourceGapInterval]] = {}
    for iv in active:
        if wanted is not None and iv.symbol not in wanted:
            continue
        if not _overlaps(iv.start, iv.end, window_start_dt, window_end_dt):
            continue
        committed_by_symbol.setdefault(iv.symbol, []).append(iv)
    for symbol, commits in committed_by_symbol.items():
        gaps = by_symbol.get(symbol, [])
        for commit in commits:
            overlapping = [g for g in gaps if _overlaps(g.start, g.end, commit.start, commit.end)]
            if not overlapping:
                resolved.append(commit)
                continue
            pieces: list[tuple[datetime, datetime | None]] = []
            for gap in overlapping:
                clipped = _clip_to_commit(gap, commit, window_start_dt, window_end_dt)
                if clipped is not None:
                    pieces.append(clipped)
            merged = _merge_clipped(pieces)
            if len(merged) == 1 and merged[0][0] == commit.start and merged[0][1] == commit.end:
                unchanged.append(commit)
                continue
            c_start = max(commit.start, window_start_dt)
            if commit.end is None:
                has_open_tail = any(g.end is None for g in overlapping)
                c_end: datetime | None = None if has_open_tail else window_end_dt
            else:
                c_end = min(commit.end, window_end_dt)
            if len(merged) == 1 and merged[0][0] == c_start and merged[0][1] == c_end:
                unchanged.append(commit)
                continue
            for piece_start, piece_end in merged:
                narrowed.append(
                    SourceGapInterval(
                        symbol=commit.symbol,
                        plane=commit.plane,
                        start=piece_start,
                        end=piece_end,
                        reason="SOURCE_ABSENT",
                        evidence=(
                            f"Audited local lake {plane}: residual absence within "
                            f"committed [{_iso_z(commit.start)}, "
                            f"{_iso_z(commit.end) if commit.end is not None else 'open'}) "
                            f"measured as [{_iso_z(piece_start)}, "
                            f"{_iso_z(piece_end) if piece_end is not None else 'open'})."
                        ),
                        verified_at=end.to_pydatetime(),
                        resolved_at=None,
                    )
                )
    # Measured gaps come from the reconciled scope itself, so every gap here
    # necessarily belongs to a scoped symbol.
    for gap in measured:
        commits = committed_by_symbol.get(gap.symbol, [])
        if any(_overlaps(gap.start, gap.end, c.start, c.end) for c in commits):
            continue
        discovered.append(gap)
    def _sort_key(iv: SourceGapInterval) -> tuple[str, str, datetime, datetime]:
        return (iv.symbol, iv.plane, iv.start, _as_inf(iv.end))

    return SourceGapAuditReport(
        resolved=tuple(sorted(resolved, key=_sort_key)),
        narrowed=tuple(sorted(narrowed, key=_sort_key)),
        unchanged=tuple(sorted(unchanged, key=_sort_key)),
        discovered=tuple(sorted(discovered, key=_sort_key)),
    )


def _record_to_row(iv: SourceGapInterval) -> dict[str, object]:
    return {
        "symbol": iv.symbol,
        "plane": iv.plane,
        "start": _iso_z(iv.start),
        "end": _iso_z(iv.end) if iv.end is not None else None,
        "reason": iv.reason,
        "evidence": iv.evidence,
        "verified_at": _iso_z(iv.verified_at),
        "resolved_at": _iso_z(iv.resolved_at) if iv.resolved_at is not None else None,
    }


def write_audited_registry(
    report: SourceGapAuditReport,
    *,
    registry_path: Path | None = None,
    verified_at: pd.Timestamp,
) -> int:
    """Persist an audited registry, preserving history rather than deleting records.

    Args:
        report: Result of `audit_source_gap_registry`.
        registry_path: Registry override; defaults to the packaged file.
        verified_at: UTC stamp written onto every record this audit re-confirmed.
    Returns:
        Number of records written.
    Raises:
        DataIntegrityError: The resulting registry would violate a loader invariant.
    """
    if not isinstance(verified_at, pd.Timestamp) or verified_at.tzinfo is None:
        raise DataIntegrityError("verified_at must be a tz-aware UTC Timestamp")
    if verified_at.utcoffset() != timedelta(0):
        raise DataIntegrityError("verified_at must be UTC")
    verified_dt = verified_at.to_pydatetime()
    from src.mhs.source_gaps import _default_registry_path

    target = _default_registry_path() if registry_path is None else Path(registry_path)
    try:
        target.read_bytes()
    except OSError as exc:
        raise DataIntegrityError(f"source-gap registry unreadable: {target}") from exc
    existing = load_source_gap_registry(registry_path)

    def _key(iv: SourceGapInterval) -> tuple[str, str, datetime, datetime | None]:
        return (iv.symbol, iv.plane, iv.start, iv.end)

    resolved_keys = {_key(iv) for iv in report.resolved}
    unchanged_keys = {_key(iv) for iv in report.unchanged}
    narrowed_new = list(report.narrowed)
    discovered_new = list(report.discovered)
    for iv in (*narrowed_new, *discovered_new):
        if not iv.evidence.strip():
            raise DataIntegrityError("audited record evidence must not be blank")
        if iv.resolved_at is not None:
            raise DataIntegrityError("audited new record must not carry resolved_at")

    rebuilt: list[SourceGapInterval] = []
    for iv in existing:
        if iv.resolved_at is not None:
            rebuilt.append(iv)
            continue
        key = _key(iv)
        if key in resolved_keys:
            rebuilt.append(
                SourceGapInterval(
                    symbol=iv.symbol,
                    plane=iv.plane,
                    start=iv.start,
                    end=iv.end,
                    reason=iv.reason,
                    evidence=iv.evidence,
                    verified_at=verified_dt,
                    resolved_at=verified_dt,
                )
            )
        elif key in unchanged_keys:
            rebuilt.append(
                SourceGapInterval(
                    symbol=iv.symbol,
                    plane=iv.plane,
                    start=iv.start,
                    end=iv.end,
                    reason=iv.reason,
                    evidence=iv.evidence,
                    verified_at=verified_dt,
                    resolved_at=None,
                )
            )
        elif any(
            n.symbol == iv.symbol and n.plane == iv.plane and _overlaps(n.start, n.end, iv.start, iv.end)
            for n in narrowed_new
        ):
            rebuilt.append(
                SourceGapInterval(
                    symbol=iv.symbol,
                    plane=iv.plane,
                    start=iv.start,
                    end=iv.end,
                    reason=iv.reason,
                    evidence=iv.evidence,
                    verified_at=verified_dt,
                    resolved_at=verified_dt,
                )
            )
        else:
            rebuilt.append(iv)
    def _stamped(iv: SourceGapInterval) -> SourceGapInterval:
        return SourceGapInterval(
            symbol=iv.symbol,
            plane=iv.plane,
            start=iv.start,
            end=iv.end,
            reason=iv.reason,
            evidence=iv.evidence,
            verified_at=verified_dt,
            resolved_at=None,
        )

    rebuilt.extend(_stamped(iv) for iv in (*narrowed_new, *discovered_new))
    rebuilt.sort(key=lambda iv: (iv.symbol, iv.plane, iv.start))
    payload = "".join(json.dumps(_record_to_row(iv), sort_keys=True) + "\n" for iv in rebuilt)
    from src.mhs.source_gaps import _parse_registry_bytes

    _parse_registry_bytes(payload.encode("utf-8"), target)
    tmp_name: str | None = None
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".source_gaps_", suffix=".tmp")
        with open(tmp_fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        load_source_gap_registry(Path(tmp_name))
        Path(tmp_name).replace(target)
    except DataIntegrityError:
        if tmp_name is not None:
            with contextlib.suppress(OSError):
                Path(tmp_name).unlink(missing_ok=True)
        raise
    except OSError as exc:
        if tmp_name is not None:
            with contextlib.suppress(OSError):
                Path(tmp_name).unlink(missing_ok=True)
        raise DataIntegrityError(f"source-gap registry unwritable: {target}") from exc
    with contextlib.suppress(Exception):
        from src.mhs.source_gaps import clear_source_gap_registry_cache

        clear_source_gap_registry_cache()
    return len(rebuilt)
