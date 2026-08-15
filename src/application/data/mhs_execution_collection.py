"""Safe planning and batch collection for MHS execution OHLCV."""

from __future__ import annotations

import concurrent.futures
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import src.market_data.services.futures_collection as _futures_collection
from src.common.config import FUTURES_DATA_DIR, funding_path
from src.common.errors import DataIntegrityError
from src.market_data.services.futures_collection import DataCollector
from src.mhs.books import phase_tranche_book, rank_weight_book
from src.mhs.contracts import PHASE_1_BOOK_SPECS
from src.mhs.horizons import horizon_log_return
from src.mhs.panel import liquid_half_eligibility, load_base_panel


@dataclass(frozen=True, slots=True)
class MhsExecutionCollectionPlan:
    timeframe: str
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


def _manifest_path(timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    safe_start = start.strftime("%Y%m%d")
    safe_end = end.strftime("%Y%m%d")
    return FUTURES_DATA_DIR / "mhs_execution" / f"{timeframe}_{safe_start}_{safe_end}.json"


def build_mhs_execution_plan(
    start: str, end: str, timeframe: str = "3m", execution_universe_size: int = 30,
) -> MhsExecutionCollectionPlan:
    """Derive the exact PIT replay symbol union without network access."""
    if timeframe not in ("1m", "3m", "5m"):
        raise ValueError("timeframe must be '1m', '3m' or '5m'")
    if execution_universe_size < 8:
        raise ValueError("execution_universe_size must be >= 8")
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    if start_ts >= end_ts:
        raise ValueError("start must precede end")
    root = str(FUTURES_DATA_DIR / "ohlcv")
    panel = load_base_panel(root, "1h", ("close", "quote_vol"), start_ts, end_ts, partition="dev", min_bars=2000)
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
        spec = PHASE_1_BOOK_SPECS[name]
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
    symbol: str, timeframe: str, start: str, end: str, root: str | None = None,
) -> dict[str, object]:
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
        step = pd.Timedelta(minutes={"1m": 1, "3m": 3, "5m": 5}[timeframe])
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
    symbols: Sequence[str], timeframe: str, start: str, end: str, root: str | None = None,
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
    timeframe: str,
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

    Mirrors ``_cached_mark_panel``'s causal rules exactly: only rows whose
    ``close`` is finite AND strictly positive count, and a row at time ``t``
    becomes available only at ``t + 1h``.
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

    Exact replica of ``_cached_mark_panel``'s reindex semantics: availability
    points are reindexed onto the grid with ``method='ffill'`` and the same
    ``limit`` the replay derives from ``stale_hours``, so a gate that passes
    cannot die mid-replay from a missing mark.
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
    causal availability rules ``_cached_mark_panel`` applies during replay:
    rows with finite, strictly-positive ``close`` become available at
    ``datetime + 1h``, and a stale-carry allowance of ``stale_hours`` is
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
    statuses = {
        symbol: _coverage(symbol, str(payload["timeframe"]), str(payload["start"]), str(payload["end"]))
        for symbol in payload["symbols"]
    }
    payload["statuses"] = statuses
    payload["mode"] = "validated_local"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cast(dict[str, object], payload)
