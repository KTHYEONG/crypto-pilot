"""Safe planning and batch collection for MHS execution OHLCV."""

from __future__ import annotations

import concurrent.futures
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import src.market_data.services.futures_collection as _futures_collection
from src.common.errors import DataIntegrityError
from src.common.paths import FUTURES_DATA_DIR, funding_path
from src.market_data.services.futures_collection import DataCollector
from src.mhs.books import phase_tranche_book, rank_weight_book
from src.mhs.data_provenance import resolve_required_mhs_input_paths, seal_mhs_input_manifest
from src.mhs.horizons import horizon_log_return
from src.mhs.panel import liquid_half_eligibility, load_base_panel
from src.mhs.types import BOOK_SPECS

# 유동성 순위(720봉)와 모멘텀 신호가 창 시작 시점에 이미 완성돼 있어야 PIT 계획이 성립하므로
# 창 앞쪽 2000봉을 워밍업으로 읽고, 선택 자체는 창 안 결정 그리드에서만 한다.
MHS_EXECUTION_PLAN_WARMUP_HOURS: int = 2000


@dataclass(frozen=True, slots=True)
class MhsExecutionCollectionPlan:
    timeframe: Literal["3m"]
    start: str
    end: str
    execution_universe_size: int
    symbols: tuple[str, ...]
    manifest_path: str

    def to_payload(self) -> dict[str, object]:
        return {
            "timeframe": self.timeframe,
            "start": self.start,
            "end": self.end,
            "execution_universe_size": self.execution_universe_size,
            "symbols": list(self.symbols),
            "symbol_count": len(self.symbols),
            "manifest_path": self.manifest_path,
        }


def _funded_symbols(symbols: list[str]) -> list[str]:
    return [s for s in symbols if funding_path(s).exists()]


def _manifest_path(timeframe: Literal["3m"], start: pd.Timestamp, end: pd.Timestamp) -> Path:
    safe_start = start.strftime("%Y%m%d")
    safe_end = end.strftime("%Y%m%d")
    return FUTURES_DATA_DIR / "mhs_execution" / f"{timeframe}_{safe_start}_{safe_end}.json"


def build_mhs_execution_plan(
    start: str, end: str, timeframe: Literal["3m"] = "3m", execution_universe_size: int = 30,
) -> MhsExecutionCollectionPlan:
    """Derive the exact PIT replay symbol union without network access."""
    if timeframe != "3m":
        raise ValueError(f"unknown execution_timeframe '{timeframe}': timeframe must be '3m' ('1m', '3m' or '5m' are recognized but only 3m is supported)")
    if execution_universe_size < 8:
        raise ValueError("execution_universe_size must be >= 8")
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    if start_ts >= end_ts:
        raise ValueError("start must precede end")
    root = str(FUTURES_DATA_DIR / "ohlcv")
    panel = load_base_panel(root, "1h", ("close", "quote_vol"), start_ts - pd.Timedelta(hours=MHS_EXECUTION_PLAN_WARMUP_HOURS), end_ts, partition="dev", min_bars=2000)
    symbols = _funded_symbols(list(panel["close"].columns))
    if not symbols:
        raise RuntimeError("no funded dev symbols available for MHS execution plan")
    quote_volume = panel["quote_vol"][symbols]
    eligible = liquid_half_eligibility(quote_volume, lookback_bars=720, min_history_bars=720)
    trailing = quote_volume.rolling(720, min_periods=720).mean()
    ranked = trailing.where(eligible).rank(axis=1, ascending=False, method="first")
    top = ranked.le(execution_universe_size).fillna(False)
    log_close = np.log(panel["close"][symbols].where(panel["close"][symbols] > 0))
    decision_grids = (
        pd.date_range(start_ts, end_ts, freq="6h", tz="UTC"),
        pd.date_range(start_ts, end_ts, freq="24h", tz="UTC"),
    )
    selected: set[str] = set()
    for name in ("fast_reversal", "slow_momentum"):
        spec = BOOK_SPECS[name]
        grid = decision_grids[0 if name == "fast_reversal" else 1]
        signal = horizon_log_return(log_close, spec.horizon_hours).reindex(grid)
        eligibility = eligible.reindex(grid)
        weights = phase_tranche_book(
            rank_weight_book(signal, eligibility, spec.band.sign, spec.min_symbols),
            spec.tranche_count(),
        )
        weights = weights.where(top.reindex(grid, method="ffill").fillna(False), other=0.0)
        selected.update(weights.ne(0.0).any(axis=0).loc[lambda s: s].index)
    if not selected:
        raise RuntimeError("PIT execution plan selected no symbols")
    manifest = _manifest_path(timeframe, start_ts, end_ts)
    return MhsExecutionCollectionPlan(
        timeframe=timeframe,
        start=start_ts.isoformat(),
        end=end_ts.isoformat(),
        execution_universe_size=execution_universe_size,
        symbols=tuple(sorted(selected)),
        manifest_path=str(manifest),
    )


def _coverage(
    symbol: str, timeframe: Literal["3m"], start: str, end: str, root: str | None = None,
) -> dict[str, object]:
    if timeframe != "3m":
        raise ValueError(f"unknown execution_timeframe '{timeframe}'")
    base = Path(root) if root else FUTURES_DATA_DIR / "ohlcv"
    path = base / timeframe / f"{symbol}.parquet"
    if not path.exists():
        return {"status": "MISSING", "rows": 0}
    table = pq.read_table(path, columns=["timestamp"])
    idx = pd.to_datetime(table.column("timestamp").to_numpy(), unit="ms", utc=True)
    idx = pd.DatetimeIndex(idx).drop_duplicates().sort_values()
    req_start = pd.Timestamp(start)
    req_end = pd.Timestamp(end)
    observed = idx[(idx >= req_start) & (idx <= req_end)]
    missing_internal = 0
    if len(observed) > 1:
        step = pd.Timedelta(minutes=3)
        expected = int((observed[-1] - observed[0]) / step) + 1
        missing_internal = max(0, expected - len(observed))
    return {
        "status": "GAPPED" if missing_internal else "PRESENT",
        "rows": len(observed),
        "missing_internal_bars": missing_internal,
        "first": observed[0].isoformat() if len(observed) else None,
        "last": observed[-1].isoformat() if len(observed) else None,
        "bytes": path.stat().st_size,
    }


def assert_execution_data_coverage(
    symbols: Sequence[str], timeframe: Literal["3m"], start: str, end: str, root: str | None = None,
) -> None:
    """Fail closed unless every symbol has full ``[start, end]`` execution cache coverage.

    Reuses ``_coverage`` (local Parquet metadata reads only -- no network, no
    ``DataCollector``) and raises ``DataIntegrityError`` naming every symbol
    whose status is not ``PRESENT`` (``MISSING`` file or ``GAPPED`` internal
    bars), so a pre-flight diagnostic gate fails with an actionable symbol list
    instead of a late opaque ``MISSING_DATA`` termination count. ``root`` is the
    synthetic-cache root for tests; when ``None`` the canonical
    ``FUTURES_DATA_DIR / 'ohlcv'`` path is used (backward compatible with the
    existing ``_coverage`` call sites).
    """
    deficient = {
        symbol: status
        for symbol in symbols
        if (status := str(_coverage(symbol, timeframe, start, end, root)["status"])) != "PRESENT"
    }
    if deficient:
        listed = ", ".join(f"{s} ({status})" for s, status in sorted(deficient.items()))
        raise DataIntegrityError(
            f"execution data coverage incomplete for {len(deficient)} symbols: {listed}"
        )


def roster_membership_intervals(
    execution_mask: pd.DataFrame,
) -> dict[str, tuple[tuple[pd.Timestamp, pd.Timestamp], ...]]:
    """Extract per-symbol contiguous roster-membership intervals from the PIT mask.

    A symbol that leaves and re-enters the roster yields one ``(start, end)``
    interval PER contiguous True-run -- never a single collapsed ``(first,
    last)`` span, which would re-introduce the over-broad scope this design
    removes. Symbols never in the roster (all-False column) are absent from the
    returned mapping entirely (not mapped to an empty tuple).

    Vectorized ``np.diff`` edge detection over the int8 view of the mask: no
    ``pd.apply`` and no per-row Python loops over the wide (rows x symbols)
    mask.
    """
    if execution_mask.shape[0] == 0 or execution_mask.shape[1] == 0:
        return {}
    arr = execution_mask.to_numpy(dtype=bool)
    padded = np.zeros((arr.shape[0] + 2, arr.shape[1]), dtype=np.int8)
    padded[1:-1] = arr
    diff = np.diff(padded, axis=0)
    enter_rows, enter_cols = np.nonzero(diff == 1)
    exit_rows, exit_cols = np.nonzero(diff == -1)
    intervals: dict[str, tuple[tuple[pd.Timestamp, pd.Timestamp], ...]] = {}
    index = execution_mask.index
    for j, col in enumerate(execution_mask.columns):
        enters = enter_rows[enter_cols == j]
        if enters.size == 0:
            continue
        exits = exit_rows[exit_cols == j]
        ivs = tuple(
            (index[int(enters[k])], index[int(exits[k]) - 1])
            for k in range(enters.size)
        )
        intervals[col] = ivs
    return intervals


def assert_relevant_execution_data_coverage(
    execution_mask: pd.DataFrame,
    timeframe: Literal["3m"],
    root: str | None = None,
) -> None:
    """Fail closed unless every roster-membership interval has full cache coverage.

    Same local-Parquet-metadata semantics as ``assert_execution_data_coverage``
    (``_coverage`` only -- no network, no ``DataCollector``) but scoped to the
    roster's per-symbol contiguous membership intervals instead of the full
    universe x full period Cartesian product. A gap that lies entirely OUTSIDE a
    symbol's membership interval is correctly ignored; a gap INSIDE an interval
    fails closed naming the offending ``(symbol, interval, status)``. An empty
    mapping (no symbol ever in the roster) is a no-op.
    """
    intervals = roster_membership_intervals(execution_mask)
    deficient: list[str] = []
    for symbol, ivs in sorted(intervals.items()):
        for iv_start, iv_end in ivs:
            status = str(_coverage(symbol, timeframe, str(iv_start), str(iv_end), root)["status"])
            if status != "PRESENT":
                deficient.append(f"{symbol} ({iv_start}..{iv_end}) {status}")
    if deficient:
        listed = "; ".join(deficient)
        raise DataIntegrityError(
            f"relevant execution data coverage incomplete for {len(deficient)} "
            f"interval(s): {listed}"
        )


def _mark_availability_index(path: Path) -> pd.DatetimeIndex:
    """Sorted unique mark-availability times for one symbol's mark parquet.

    Causal availability rules: only rows whose ``close`` is finite AND
    strictly positive count, and a row at time ``t`` becomes available only
    at ``t + 1h``.
    """
    if not path.exists():
        return pd.DatetimeIndex([], tz="UTC")
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return pd.DatetimeIndex([], tz="UTC")
    if df.empty or "close" not in df.columns:
        return pd.DatetimeIndex([], tz="UTC")
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    else:
        return pd.DatetimeIndex([], tz="UTC")
    close = pd.to_numeric(df["close"], errors="coerce")
    valid = dt.notna() & close.notna() & (close > 0)
    if not bool(valid.any()):
        return pd.DatetimeIndex([], tz="UTC")
    avail = dt[valid] + pd.Timedelta(hours=1)
    return pd.DatetimeIndex(avail.drop_duplicates()).sort_values()


def _mark_covers_grid(
    avail: pd.DatetimeIndex,
    grid: pd.DatetimeIndex,
    ffill_limit: int,
) -> bool:
    """Whether every hour of ``grid`` is causally covered by available marks.

    Availability points are reindexed onto the grid with ``method='ffill'``
    and the same ``limit`` the gate derives from ``stale_hours``, so a gate
    that passes cannot die mid-replay from a missing mark.
    """
    if avail.empty or grid.empty:
        return False
    source = pd.Series(True, index=avail)
    aligned = (
        source.reindex(grid, method="ffill", limit=ffill_limit)
        if ffill_limit > 0
        else source.reindex(grid)
    )
    return bool(aligned.fillna(False).all())


def assert_relevant_mark_price_coverage(
    execution_mask: pd.DataFrame,
    timeframe: str = "1h",
    stale_hours: int = 0,
) -> None:
    """Fail closed unless every roster hour has a causally available mark.

    Reuses ``roster_membership_intervals`` and checks each hour of each
    membership interval against the symbol's ``markPriceKlines`` parquet
    (resolved via ``futures_collection._mark_price_path``) using EXACTLY the
    causal availability rules of the mark panel: rows with finite,
    strictly-positive ``close`` become available at ``datetime + 1h``, and a stale-carry allowance of ``stale_hours`` is
    honored with the same ``ffill`` limit. A gate that is more permissive than
    the replay would let a run pass and then die mid-replay -- this gate is
    deliberately strict.
    """
    if timeframe != "1h":
        raise ValueError(f"unsupported timeframe '{timeframe}' (mark coverage is hourly)")
    if stale_hours < 0:
        raise ValueError("stale_hours must be non-negative")
    intervals = roster_membership_intervals(execution_mask)
    if not intervals:
        return
    ffill_limit = max(0, stale_hours - 1) if stale_hours > 0 else 0
    deficient: list[str] = []
    for symbol, ivs in sorted(intervals.items()):
        path = _futures_collection._mark_price_path(symbol, timeframe)
        avail = _mark_availability_index(path)
        for iv_start, iv_end in ivs:
            grid = pd.date_range(iv_start, iv_end, freq="1h", tz="UTC")
            if not _mark_covers_grid(avail, grid, ffill_limit):
                deficient.append(f"{symbol} ({iv_start}..{iv_end}) mark coverage deficient")
    if deficient:
        listed = "; ".join(deficient)
        raise DataIntegrityError(
            f"relevant mark-price coverage incomplete for {len(deficient)} "
            f"interval(s): {listed}"
        )


# Dynamic gap-exclusion threshold: reuses the SAME 720h invariant
# ``liquid_half_eligibility(min_history_bars=720)`` already requires for a
# symbol to become liquidity-eligible at all (src/mhs/panel.py). A contiguous
# gap at or above this bound already structurally breaks that trailing-history
# requirement through the gap, so exclusion at this threshold is not a policy
# choice layered on top of a separate magic number -- it is the same bound the
# eligibility computation already enforces. Not a hardcoded symbol list: this
# is recomputed from the live cache and the live roster mask on every call, so
# a symbol excluded today is automatically re-admitted once its gap is
# backfilled, and a symbol excluded tomorrow if its cache degrades.
DYNAMIC_GAP_EXCLUSION_HOURS = 720.0

_OHLCV_BAR_STEP_MINUTES = {"3m": 3, "1h": 60}
_MARK_AVAILABILITY_LAG_HOURS = 1


def _require_gap_threshold(min_gap_hours: float) -> float:
    """Validate the structural history-gap threshold shared by gap exclusion."""
    try:
        value = float(min_gap_hours)
    except (TypeError, ValueError):
        raise ValueError(f"min_gap_hours must be finite, got {min_gap_hours}") from None
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"min_gap_hours must be finite and > 0, got {min_gap_hours}")
    return value


def _row_group_ms_min_ns(column: object) -> int | None:
    """Minimum label of an ms timestamp chunk in nanoseconds, or None if unknown."""
    try:
        if column is None:
            return None
        stats = getattr(column, "statistics", None)
        if stats is None or not getattr(column, "is_stats_set", False) or not getattr(stats, "has_min_max", False):
            return None
        minimum = stats.min
        if minimum is None or (isinstance(minimum, float) and not np.isfinite(minimum)):
            return None
        if isinstance(minimum, bool) or not isinstance(minimum, (int, np.integer, float)):
            return None
        return int(minimum) * 1_000_000
    except Exception:  # noqa: BLE001
        return None


def _row_group_datetime_min_ns(column: object) -> int | None:
    """Minimum label of a datetime chunk in nanoseconds, or None if unknown."""
    try:
        if column is None:
            return None
        stats = getattr(column, "statistics", None)
        if stats is None or not getattr(column, "is_stats_set", False) or not getattr(stats, "has_min_max", False):
            return None
        minimum = stats.min
        if minimum is None:
            return None
        stamp = pd.to_datetime(minimum, utc=True, errors="coerce")
        if not isinstance(stamp, pd.Timestamp) or pd.isna(stamp):
            return None
        return int(stamp.as_unit("ns").value)
    except Exception:  # noqa: BLE001
        return None


def _read_ohlcv_labels(
    symbol: str, timeframe: Literal["3m", "1h"], root: str | None,
    *, observed_through_ns: int | None = None,
) -> np.ndarray | None:
    """Read compact OHLCV labels without decoding irrelevant source planes.

    Args:
        symbol: Existing registered source symbol.
        timeframe: Existing hourly or three-minute source interval.
        root: Existing OHLCV root override.
        observed_through_ns: Inclusive latest useful label, or full audit range.

    Returns:
        Sorted unique int64 UTC labels, or None for an absent file.

    Raises:
        DataIntegrityError: Existing source schema or decoding is inconsistent.
    """
    base = Path(root) if root else FUTURES_DATA_DIR / "ohlcv"
    path = base / timeframe / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        reader = pq.ParquetFile(path)
        names = list(reader.schema_arrow.names)
    except Exception as exc:
        raise DataIntegrityError(f"execution source unreadable symbol={symbol!r} path={path}") from exc
    if "timestamp" not in names:
        raise DataIntegrityError(f"execution source schema inconsistent symbol={symbol!r} path={path}")
    try:
        meta = reader.metadata
        positions = {meta.row_group(0).column(j).path_in_schema: j for j in range(meta.row_group(0).num_columns)} if meta.num_row_groups else {"timestamp": 0}
        stamp_pos = positions.get("timestamp", names.index("timestamp"))
        chunks: list[np.ndarray] = []
        for group in range(meta.num_row_groups):
            if observed_through_ns is not None:
                earliest = _row_group_ms_min_ns(meta.row_group(group).column(stamp_pos))
                if earliest is not None and earliest > int(observed_through_ns):
                    continue
            try:
                table = reader.read_row_group(group, columns=["timestamp"], use_threads=False)
                raw = table.column("timestamp").to_numpy()
                del table
            except Exception as exc:
                raise DataIntegrityError(f"execution source unreadable symbol={symbol!r} path={path}") from exc
            idx = pd.to_datetime(raw, unit="ms", utc=True, errors="coerce")
            del raw
            valid = pd.DatetimeIndex(idx).dropna()
            del idx
            if len(valid) == 0:
                continue
            values = np.asarray(valid.as_unit("ns").asi8, dtype="int64")
            del valid
            if observed_through_ns is not None:
                values = values[values <= int(observed_through_ns)]
                if len(values) == 0:
                    continue
            chunks.append(values)
    except DataIntegrityError:
        raise
    except Exception as exc:
        raise DataIntegrityError(f"execution source unreadable symbol={symbol!r} path={path}") from exc
    if not chunks:
        return np.zeros(0, dtype="int64")
    combined = np.concatenate(chunks)
    del chunks
    return np.unique(combined).astype("int64", copy=False)


def _read_mark_labels(
    symbol: str, timeframe: str, *, observed_through_ns: int | None = None,
) -> np.ndarray | None:
    """Read compact valid mark labels from their established source root.

    Args:
        symbol: Existing mark source symbol.
        timeframe: Existing hourly mark interval.
        observed_through_ns: Inclusive latest useful label, or full audit range.

    Returns:
        Sorted unique int64 UTC labels with finite positive closes, or None.

    Raises:
        DataIntegrityError: Existing mark provenance cannot be established.
    """
    path = _futures_collection._mark_price_path(symbol, timeframe)
    if not path.exists():
        return None
    try:
        reader = pq.ParquetFile(path)
        names = list(reader.schema_arrow.names)
    except Exception as exc:
        raise DataIntegrityError(f"mark source unreadable symbol={symbol!r} path={path}") from exc
    if "close" not in names or ("datetime" not in names and "timestamp" not in names):
        raise DataIntegrityError(f"mark source schema inconsistent symbol={symbol!r} path={path}")
    use_datetime = "datetime" in names
    time_column = "datetime" if use_datetime else "timestamp"
    columns = [time_column, "close"]
    try:
        meta = reader.metadata
        if meta.num_rows == 0:
            raise DataIntegrityError(f"mark source schema inconsistent symbol={symbol!r} path={path}")
        positions = {meta.row_group(0).column(j).path_in_schema: j for j in range(meta.row_group(0).num_columns)} if meta.num_row_groups else {time_column: 0}
        time_pos = positions.get(time_column, names.index(time_column))
        chunks: list[np.ndarray] = []
        for group in range(meta.num_row_groups):
            if observed_through_ns is not None:
                chunk_meta = meta.row_group(group).column(time_pos)
                earliest = _row_group_datetime_min_ns(chunk_meta) if use_datetime else _row_group_ms_min_ns(chunk_meta)
                if earliest is not None and earliest > int(observed_through_ns):
                    continue
            try:
                table = reader.read_row_group(group, columns=columns, use_threads=False)
                if use_datetime:
                    time_raw = table.column("datetime").to_pandas()
                else:
                    time_raw = table.column("timestamp").to_numpy()
                close_raw = table.column("close").to_numpy()
                del table
            except Exception as exc:
                raise DataIntegrityError(f"mark source unreadable symbol={symbol!r} path={path}") from exc
            if use_datetime:
                moments = pd.to_datetime(time_raw, utc=True, errors="coerce")
            else:
                moments = pd.to_datetime(time_raw, unit="ms", utc=True, errors="coerce")
            del time_raw
            closes = np.asarray(pd.to_numeric(close_raw, errors="coerce"), dtype="float64")
            del close_raw
            keep = np.asarray(moments.notna()) & np.isfinite(closes) & (closes > 0.0)
            del closes
            if not bool(np.any(keep)):
                del moments, keep
                continue
            valid = pd.DatetimeIndex(moments[keep]).sort_values()
            del moments, keep
            values = np.asarray(valid.as_unit("ns").asi8, dtype="int64")
            del valid
            if observed_through_ns is not None:
                values = values[values <= int(observed_through_ns)]
                if len(values) == 0:
                    continue
            chunks.append(values)
    except DataIntegrityError:
        raise
    except Exception as exc:
        raise DataIntegrityError(f"mark source unreadable symbol={symbol!r} path={path}") from exc
    if not chunks:
        return np.zeros(0, dtype="int64")
    combined = np.concatenate(chunks)
    del chunks
    return np.unique(combined).astype("int64", copy=False)


def _causal_gap_excluded(
    member: np.ndarray,
    decision_ns: np.ndarray,
    labels_ns: np.ndarray | None,
    lag_ns: int,
    gap_ns: int,
) -> np.ndarray:
    """Exclude new exposure using only observations published at each decision.

    Args:
        member: Aligned boolean point-in-time membership.
        decision_ns: Chronological UTC decision nanoseconds.
        labels_ns: Sorted unique source labels, or absent provenance.
        lag_ns: Positive source publication lag in nanoseconds.
        gap_ns: Positive existing structural-gap threshold in nanoseconds.

    Returns:
        Aligned boolean exclusions; no observed history fails closed.

    Raises:
        ValueError: Array alignment or temporal parameters are invalid.
    """
    if (
        not isinstance(member, np.ndarray)
        or not isinstance(decision_ns, np.ndarray)
        or member.ndim != 1
        or decision_ns.ndim != 1
        or member.shape != decision_ns.shape
    ):
        raise ValueError("member and decision_ns must be aligned one-dimensional arrays")
    if labels_ns is not None and (not isinstance(labels_ns, np.ndarray) or labels_ns.ndim != 1):
        raise ValueError("labels_ns must be a one-dimensional array or absent provenance")
    try:
        lag = int(lag_ns)
        gap = int(gap_ns)
    except (TypeError, ValueError):
        raise ValueError("lag_ns and gap_ns must be positive integers") from None
    if lag <= 0 or gap <= 0 or lag != lag_ns or gap != gap_ns:
        raise ValueError("lag_ns and gap_ns must be positive integers")
    n = len(decision_ns)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if labels_ns is None or len(labels_ns) == 0:
        return np.ones(n, dtype=bool)
    pos = np.searchsorted(labels_ns, decision_ns - lag, side="right") - 1
    excluded = np.ones(n, dtype=bool)
    has = pos >= 0
    latest = labels_ns[np.maximum(pos, 0)]
    excluded[has] = (decision_ns[has] - latest[has]) >= gap
    return excluded


def _apply_causal_gap_exclusion(
    execution_mask: pd.DataFrame,
    intervals: dict[str, tuple[tuple[pd.Timestamp, pd.Timestamp], ...]],
    labels_by_symbol: dict[str, np.ndarray | None],
    lag_ns: int,
    gap_ns: int,
) -> tuple[pd.DataFrame, dict[str, tuple[tuple[pd.Timestamp, pd.Timestamp], ...]]]:
    """Zero membership cells under structural gaps; report newly excluded runs."""
    decision_ns = np.asarray(execution_mask.index.as_unit("ns").asi8, dtype="int64")
    adjusted = execution_mask.copy()
    for symbol in intervals:
        member = execution_mask[symbol].to_numpy(dtype=bool)
        dropped = _causal_gap_excluded(member, decision_ns, labels_by_symbol[symbol], lag_ns, gap_ns)
        adjusted[symbol] = member & ~dropped
    dropped_frame = execution_mask & ~adjusted
    return adjusted, roster_membership_intervals(dropped_frame)


def apply_dynamic_gap_exclusion(
    execution_mask: pd.DataFrame,
    timeframe: Literal["3m", "1h"],
    root: str | None = None,
    min_gap_hours: float = DYNAMIC_GAP_EXCLUSION_HOURS,
) -> tuple[pd.DataFrame, dict[str, tuple[tuple[pd.Timestamp, pd.Timestamp], ...]]]:
    """Restrict membership using only execution gaps observable at each decision.

    Args:
        execution_mask: UTC decision-time membership.
        timeframe: Three-minute execution interval (1h panel-stage masks keep
            their hourly source interval).
        root: Execution source root.
        min_gap_hours: Existing structural history-gap threshold.

    Returns:
        Causal membership and observed exclusion intervals for diagnostics.

    Raises:
        ValueError: Interval or threshold is unsupported.
        DataIntegrityError: Source provenance cannot be established.
    """
    if timeframe not in _OHLCV_BAR_STEP_MINUTES:
        raise ValueError(f"unknown execution_timeframe '{timeframe}'")
    gap_hours = _require_gap_threshold(min_gap_hours)
    intervals = roster_membership_intervals(execution_mask)
    if not intervals:
        return execution_mask, {}
    lag_ns = _OHLCV_BAR_STEP_MINUTES[timeframe] * 60_000_000_000
    gap_ns = int(gap_hours * 3_600_000_000_000)
    decision_ns = np.asarray(execution_mask.index.as_unit("ns").asi8, dtype="int64")
    observed_through_ns = int(decision_ns[-1]) - lag_ns
    adjusted = execution_mask.copy()
    for symbol in intervals:
        member = execution_mask[symbol].to_numpy(dtype=bool)
        labels = _read_ohlcv_labels(
            symbol, timeframe, root, observed_through_ns=observed_through_ns,
        )
        dropped = _causal_gap_excluded(member, decision_ns, labels, lag_ns, gap_ns)
        del labels
        adjusted[symbol] = member & ~dropped
    return adjusted, roster_membership_intervals(execution_mask & ~adjusted)


def apply_dynamic_mark_gap_exclusion(
    execution_mask: pd.DataFrame,
    timeframe: str = "1h",
    root: str | None = None,
    min_gap_hours: float = DYNAMIC_GAP_EXCLUSION_HOURS,
) -> tuple[pd.DataFrame, dict[str, tuple[tuple[pd.Timestamp, pd.Timestamp], ...]]]:
    """Restrict membership using published hourly mark observations.

    Args:
        execution_mask: UTC decision-time membership.
        timeframe: Hourly mark source interval.
        root: Existing compatibility argument; it must not imply root isolation.
        min_gap_hours: Existing structural history-gap threshold.

    Returns:
        Causal membership and observed mark exclusion intervals.

    Raises:
        ValueError: Mark interval or threshold is unsupported.
        DataIntegrityError: Mark provenance is inconsistent.
    """
    if timeframe != "1h":
        raise ValueError(f"unsupported timeframe '{timeframe}' (mark coverage is hourly)")
    gap_hours = _require_gap_threshold(min_gap_hours)
    intervals = roster_membership_intervals(execution_mask)
    if not intervals:
        return execution_mask, {}
    lag_ns = _MARK_AVAILABILITY_LAG_HOURS * 3_600_000_000_000
    gap_ns = int(gap_hours * 3_600_000_000_000)
    decision_ns = np.asarray(execution_mask.index.as_unit("ns").asi8, dtype="int64")
    observed_through_ns = int(decision_ns[-1]) - lag_ns
    adjusted = execution_mask.copy()
    for symbol in intervals:
        member = execution_mask[symbol].to_numpy(dtype=bool)
        labels = _read_mark_labels(
            symbol, timeframe, observed_through_ns=observed_through_ns,
        )
        dropped = _causal_gap_excluded(member, decision_ns, labels, lag_ns, gap_ns)
        del labels
        adjusted[symbol] = member & ~dropped
    return adjusted, roster_membership_intervals(execution_mask & ~adjusted)


def collect_mhs_execution_data(
    plan: MhsExecutionCollectionPlan, *, execute: bool = False, workers: int = 4,
) -> dict[str, object]:
    """Persist a plan and optionally execute its resumable per-symbol collection."""
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    path = Path(plan.manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = plan.to_payload()
    payload["mode"] = "execute" if execute else "dry_run"
    payload["statuses"] = {s: {"status": "PLANNED"} for s in plan.symbols}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if execute:
        def collect_one(symbol: str) -> tuple[str, dict[str, object]]:
            DataCollector().ensure_ohlcv_data(symbol, plan.timeframe, plan.start, plan.end)
            return symbol, _coverage(symbol, plan.timeframe, plan.start, plan.end)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = dict(pool.map(collect_one, plan.symbols))
        payload["statuses"] = results
        payload["mode"] = "completed"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def refresh_mhs_execution_manifest(manifest_path: str | Path) -> dict[str, object]:
    """Refresh per-symbol coverage from local files without network access."""
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    timeframe = str(payload["timeframe"])
    if timeframe != "3m":
        raise ValueError(f"unknown execution_timeframe '{timeframe}'")
    execution_timeframe: Literal["3m"] = "3m"
    statuses = {
        symbol: _coverage(symbol, execution_timeframe, str(payload["start"]), str(payload["end"]))
        for symbol in payload["symbols"]
    }
    payload["statuses"] = statuses
    payload["mode"] = "validated_local"
    symbols = [str(symbol) for symbol in payload["symbols"]]
    attestation_path = path.with_name(path.stem + ".inputs.json")
    payload["input_manifest_digest"] = seal_mhs_input_manifest(
        [
            candidate
            for candidate in resolve_required_mhs_input_paths(
                data_root=FUTURES_DATA_DIR,
                panel_symbols=symbols,
                execution_symbols=symbols,
                execution_timeframe=execution_timeframe,
            )
            if candidate.exists()
        ],
        data_root=FUTURES_DATA_DIR,
        output_path=attestation_path,
    )
    payload["input_manifest_path"] = str(attestation_path)
    payload["input_seal"] = "unverified"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cast(dict[str, object], payload)
