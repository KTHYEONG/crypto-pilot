from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.domain.futures.compound.contracts import (
    CompoundUniverseResult,
    UniverseLedgerCoverage,
)
from src.domain.futures.data_lake.contracts import DataSnapshot
from src.domain.futures.universe.config import PITUniverseConfig, UniverseConfig
from src.domain.futures.universe.contracts import UniverseStateCube
from src.domain.futures.universe.models import SymbolMeta, UniverseSnapshot, load_ledger_slice
from src.domain.futures.universe.pipeline import load_or_build_universe_snapshot
from src.domain.futures.universe.storage import run_historical_sync

_logger = logging.getLogger(__name__)


class UniverseCoverageError(RuntimeError):
    ...


class EmptyPITUniverseError(RuntimeError):
    ...


class UniverseAxisLimitError(RuntimeError):
    ...


def _to_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def _date_range_dates(start: date, end: date) -> list[date]:
    return [start + pd.Timedelta(days=i) for i in range((end - start).days + 1)]


def _normalize_symbol_id(raw: str) -> str:
    normalized = raw.strip().upper()
    if not normalized:
        raise ValueError(f"empty symbol after normalization: {raw!r}")
    return normalized


def ensure_universe_ledger_coverage(
    *, config: CompoundRunConfig, start_date: date, end_date: date
) -> UniverseLedgerCoverage:
    if start_date > end_date:
        raise UniverseCoverageError(f"start_date {start_date} > end_date {end_date}")

    covered_start = start_date
    covered_end = end_date
    synced = False

    if config.sync == "auto":
        _logger.info("Running auto sync for 4h ledger: start=%s end=%s", start_date, end_date)
        try:
            run_historical_sync(
                start_date=start_date,
                end_date=end_date,
                sync_4h=True,
                sync_1d=False,
                sync_1m=False,
                max_workers=4,
            )
            synced = True
        except Exception as exc:
            _logger.warning("Auto sync failed: %s; checking cache coverage", exc)

    try:
        sample = load_ledger_slice(
            as_of=end_date,
            tf="4h",
            columns=("symbol", "date", "knowledge_date"),
            enforce_eligibility=False,
        )
    except Exception as exc:
        raise UniverseCoverageError(f"cannot read ledger: {exc}") from exc

    if sample.empty:
        raise UniverseCoverageError(
            f"4h ledger empty for start={start_date} end={end_date}; need sync"
        )

    ledger_dates = pd.to_datetime(sample["date"], utc=True)
    covered_start = ledger_dates.min().date()
    covered_end = ledger_dates.max().date()

    complete = covered_start <= start_date and covered_end >= end_date
    if not complete:
        raise UniverseCoverageError(
            f"4h ledger covers {covered_start}..{covered_end}, "
            f"requested {start_date}..{end_date}"
        )

    return UniverseLedgerCoverage(
        requested_start=start_date,
        requested_end=end_date,
        covered_start=covered_start,
        covered_end=covered_end,
        timeframe="4h",
        complete=True,
        synced=synced,
    )


def load_or_build_daily_pit_snapshot(
    *, as_of: date, cfg: CompoundRunConfig
) -> UniverseSnapshot:
    pit_config = PITUniverseConfig(k_max=cfg.max_daily_symbols)
    universe_cfg = UniverseConfig(pit_config=pit_config)
    snap, _, _ = load_or_build_universe_snapshot(
        as_of=as_of.isoformat(), tf="4h", cfg=universe_cfg
    )
    return snap


def project_pit_snapshots_to_execution_calendar(
    *,
    snapshots: list[UniverseSnapshot],
    execution_calendar: pd.DatetimeIndex,
    max_axis_symbols: int,
) -> UniverseStateCube:
    if not snapshots:
        raise EmptyPITUniverseError("received empty snapshots list")

    snapshots_sorted = sorted(snapshots, key=lambda s: _to_date(s.as_of))
    for i in range(1, len(snapshots_sorted)):
        prev_d = _to_date(snapshots_sorted[i - 1].as_of)
        curr_d = _to_date(snapshots_sorted[i].as_of)
        if curr_d <= prev_d:
            raise ValueError(
                f"non-monotonic snapshot times: {prev_d} >= {curr_d}"
            )

    union_symbols: list[str] = []
    seen: set[str] = set()
    for snap in snapshots_sorted:
        for sel in snap.selected:
            sid = _normalize_symbol_id(sel.symbol)
            if sid in seen:
                continue
            seen.add(sid)
            union_symbols.append(sid)

    if len(union_symbols) > max_axis_symbols:
        raise UniverseAxisLimitError(
            f"union symbols {len(union_symbols)} exceeds max_axis_symbols {max_axis_symbols}"
        )

    n_bars = len(execution_calendar)
    n_syms = len(union_symbols)
    eligible = np.zeros((n_bars, n_syms), dtype=np.bool_)
    entry_block = np.ones((n_bars, n_syms), dtype=np.bool_)
    exit_required = np.zeros((n_bars, n_syms), dtype=np.bool_)
    capacity = np.zeros((n_bars, n_syms), dtype=np.float64)
    risk_scale = np.ones((n_bars, n_syms), dtype=np.float64)
    cost_bps = np.full((n_bars, n_syms), 12.0, dtype=np.float64)

    # Pandas may store timezone-aware ranges in microseconds depending on the
    # input construction.  Convert both axes to the canonical internal ns unit
    # before PIT lookup.
    calendar_values = execution_calendar
    if execution_calendar.tz is not None:
        calendar_values = execution_calendar.tz_convert("UTC").tz_localize(None)
    cal_ns = calendar_values.to_numpy(dtype="datetime64[ns]").astype(np.int64)

    snap_data: dict[str, list[tuple[int, bool, bool, float, float, float]]] = {
        s: [] for s in union_symbols
    }

    for snap in snapshots_sorted:
        snap_date = _to_date(snap.as_of)
        snap_ns = np.datetime64(snap_date, "ns").astype(np.int64)
        eligible_set: set[str] = set()
        meta_map: dict[str, SymbolMeta] = {}
        for meta_item in snap.selected:
            sid = _normalize_symbol_id(meta_item.symbol)
            eligible_set.add(sid)
            meta_map[sid] = meta_item
        for sym in union_symbols:
            is_eligible = sym in eligible_set
            meta: SymbolMeta | None = meta_map.get(sym)
            cap = 0.0
            if meta is not None and meta.capacity_clip_usdt_list:
                cap = float(meta.capacity_clip_usdt_list[0])
            snap_data[sym].append(
                (int(snap_ns), is_eligible, not is_eligible, cap, 1.0, 12.0)
            )

    for col, sym in enumerate(union_symbols):
        states = snap_data[sym]
        if not states:
            continue  # pragma: no cover - union symbols are populated from snapshots
        snap_ns_arr = np.array([s[0] for s in states], dtype=np.int64)
        eligible_arr = np.array([s[1] for s in states], dtype=np.bool_)
        entry_block_arr = np.array([s[2] for s in states], dtype=np.bool_)
        capacity_arr = np.array([s[3] for s in states], dtype=np.float64)
        risk_scale_arr = np.array([s[4] for s in states], dtype=np.float64)
        cost_bps_arr = np.array([s[5] for s in states], dtype=np.float64)

        idx = np.searchsorted(snap_ns_arr, cal_ns, side="right") - 1
        valid = idx >= 0
        clipped = np.clip(idx, 0, len(snap_ns_arr) - 1)
        eligible[valid, col] = eligible_arr[clipped[valid]]
        entry_block[valid, col] = entry_block_arr[clipped[valid]]
        capacity[valid, col] = capacity_arr[clipped[valid]]
        risk_scale[valid, col] = risk_scale_arr[clipped[valid]]
        cost_bps[valid, col] = cost_bps_arr[clipped[valid]]

        for t in range(1, n_bars):
            if not eligible[t, col] and eligible[t - 1, col]:
                exit_required[t, col] = True

    return UniverseStateCube(
        calendar=execution_calendar,
        instrument_ids=tuple(union_symbols),
        eligible=eligible,
        entry_block=entry_block,
        exit_required=exit_required,
        capacity_usdt=capacity,
        risk_scale=risk_scale,
        cost_bps=cost_bps,
    )


def resolve_compound_universe(
    *, config: CompoundRunConfig, execution_calendar: pd.DatetimeIndex
) -> CompoundUniverseResult:
    ref_date = _to_date(config.reference_date) if config.reference_date else date.today()
    start_date = ref_date - pd.Timedelta(days=config.history_days)
    end_date = ref_date

    coverage = ensure_universe_ledger_coverage(
        config=config, start_date=start_date, end_date=end_date
    )

    dates = _date_range_dates(start_date, end_date)
    snapshots: list[UniverseSnapshot] = []
    for d in dates:
        snap = load_or_build_daily_pit_snapshot(as_of=d, cfg=config)
        if snap.selected:
            snapshots.append(snap)

    if not snapshots:
        raise EmptyPITUniverseError(
            f"no eligible symbols for {start_date}..{end_date}; "
            "BTC/ETH fallback is forbidden"
        )

    state_cube = project_pit_snapshots_to_execution_calendar(
        snapshots=snapshots,
        execution_calendar=execution_calendar,
        max_axis_symbols=config.max_axis_symbols,
    )

    symbols = tuple(state_cube.instrument_ids)

    return CompoundUniverseResult(
        symbols=symbols,
        state_cube=state_cube,
        snapshots=tuple(snapshots),
        coverage=coverage,
    )


class DailyPITUniverse:
    def __init__(self, symbols: tuple[str, ...], decision_dates: tuple[date, ...]) -> None:
        self.symbols = symbols
        self.decision_dates = decision_dates
        self.union_symbols = symbols
        self.historical_union: tuple[str, ...] = symbols


def build_daily_pit_universe(
    *, snapshot: DataSnapshot, config: object
) -> DailyPITUniverse:
    symbols: set[str] = set()
    for p in snapshot.partitions:
        symbols.add(p.symbol)

    if len(symbols) < 20:
        raise EmptyPITUniverseError(
            f"only {len(symbols)} eligible symbols, minimum 20 required"
        )

    _logger.info(
        "built daily PIT universe: %d symbols from snapshot %s",
        len(symbols), snapshot.snapshot_id,
    )

    return DailyPITUniverse(
        symbols=tuple(sorted(symbols)),
        decision_dates=(date.today(),),
    )


__all__ = [
    "CompoundUniverseResult",
    "DailyPITUniverse",
    "EmptyPITUniverseError",
    "UniverseAxisLimitError",
    "UniverseCoverageError",
    "build_daily_pit_universe",
    "ensure_universe_ledger_coverage",
    "project_pit_snapshots_to_execution_calendar",
    "resolve_compound_universe",
]
