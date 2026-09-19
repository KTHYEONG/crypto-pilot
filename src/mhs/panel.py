"""PIT uniform-grid panel loading for the MHS pipeline.

The uniform grid built by ``build_uniform_grid`` is the single decision clock:
every panel is reindexed onto it so phase offsets are integer row offsets.
"""

from __future__ import annotations

import glob
import logging
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.errors import DataIntegrityError
from src.market_data.storage.ohlcv import is_temp_artifact
from src.mhs.data_policy import MHS_DATA_POLICY_DEFAULT, MhsDataPolicy
from src.mhs.params import PANEL_MIN_HISTORY_BARS
from src.mhs.types import FILL_MARK_MAX_LOG_DIVERGENCE
from src.quant.universe.pit_universe import symbol_partition

DATA_POLICY_LEGACY: str = "legacy"
DATA_POLICY_ZOMBIE_MASK_V1: str = "zombie_mask_v1"
DATA_POLICIES: frozenset[str] = frozenset({DATA_POLICY_LEGACY, DATA_POLICY_ZOMBIE_MASK_V1})
ZOMBIE_FLAT_RUN_BARS: int = 24

logger = logging.getLogger("MhsPanel")


@dataclass(frozen=True, slots=True)
class QuarantineRecord:
    symbol: str
    reason: str


QUARANTINE_MAX_SYMBOLS: int = 5
QUARANTINE_MAX_FRACTION: float = 0.01
DECISION_BAR_LOOKBACK: pd.Timedelta = pd.Timedelta(hours=72)


@dataclass(slots=True)
class PanelQuarantine:
    protected: frozenset[str]
    records: list[QuarantineRecord] = field(default_factory=list)

    @property
    def symbols(self) -> frozenset[str]:
        return frozenset(r.symbol for r in self.records)

    def add(self, symbol: str, reason: str) -> None:
        if symbol in self.protected:
            raise DataIntegrityError(f"protected symbol {symbol} failed signal input check: {reason}")
        if symbol in self.symbols:
            return
        self.records.append(QuarantineRecord(symbol=symbol, reason=reason))
        logger.warning("[DATA] stage=signal_quarantine symbol=%s reason=%s", symbol, reason)

    def enforce_limit(self, universe_size: int) -> None:
        limit = min(QUARANTINE_MAX_SYMBOLS, math.ceil(QUARANTINE_MAX_FRACTION * universe_size))
        if len(self.records) > limit:
            raise DataIntegrityError(f"signal quarantine {len(self.records)} symbols exceeds limit {limit} of universe {universe_size}")


def build_uniform_grid(start: pd.Timestamp, end: pd.Timestamp, interval: str) -> pd.DatetimeIndex:
    """Return a tz-aware UTC grid inclusive of both endpoints.

    This grid is the single decision clock: every panel is reindexed onto it so
    phase offsets are integer row offsets.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be tz-aware")
    if start >= end:
        raise ValueError(f"start must be < end, got start={start} end={end}")
    return pd.date_range(start, end, freq=interval, tz="UTC")


def partition_symbols(
    symbols: Sequence[str], partition: Literal["dev", "holdout", "all"],
) -> list[str]:
    """Order-preserving delegate to ``pit_universe.symbol_partition``.

    The holdout partition must stay unread for all of Phase 1; routing every
    symbol list through this helper enforces that in one place.
    """
    if partition == "all":
        return list(symbols)
    if partition not in ("dev", "holdout"):
        raise ValueError(f"unknown partition '{partition}'")
    return [s for s in symbols if symbol_partition(s) == partition]


def zombie_masked_timestamps(
    path: str, start_ms: int, end_ms: int, interval_ms: int, run_bars: int = ZOMBIE_FLAT_RUN_BARS
) -> np.ndarray:
    """좀비 구간 마스크: run_bars 이상 연속된 flat(volume==0 and high==low) 바의 타임스탬프.

    인과적이다: 각 바의 마스크 여부는 그 바 이전(포함) 바들로만 판단되며,
    판정은 윈도우 시작 전 run_bars-1 구간을 함께 읽어 워밍업한다. 중복
    타임스탬프는 keep-last 로 해소한다.
    """
    missing = {"volume", "high", "low"} - set(pq.read_schema(path).names)
    if missing:
        raise DataIntegrityError(f"zombie mask requires {sorted(missing)} in {path}")
    table = pq.read_table(
        path,
        columns=["timestamp", "volume", "high", "low"],
        filters=[[("timestamp", ">=", start_ms - (run_bars - 1) * interval_ms), ("timestamp", "<=", end_ms)]],
    )
    ts = table.column("timestamp").to_numpy().astype("int64", copy=False)
    if ts.size == 0:
        return np.empty(0, dtype="int64")
    order = np.argsort(ts, kind="stable")
    ordered = ts[order]
    last = np.empty(ordered.size, dtype=bool)
    last[:-1] = ordered[:-1] != ordered[1:]
    last[-1] = True
    rows = order[last]
    stamps = ts[rows]
    volume = table.column("volume").to_numpy(zero_copy_only=False).astype("float64")[rows]
    high = table.column("high").to_numpy(zero_copy_only=False).astype("float64")[rows]
    low = table.column("low").to_numpy(zero_copy_only=False).astype("float64")[rows]
    flat = (volume == 0.0) & (high == low)
    idx = np.arange(stamps.size)
    last_break = np.maximum.accumulate(np.where(flat, -1, idx))
    run = idx - last_break
    return np.asarray(stamps[flat & (run >= run_bars) & (stamps >= start_ms)], dtype=np.int64)


def load_base_panel(
    root: str,
    interval: str,
    columns: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    partition: Literal["dev", "holdout", "all"] = "dev",
    min_bars: int = PANEL_MIN_HISTORY_BARS,
    data_policy: str = DATA_POLICY_LEGACY,
    *,
    quarantine: PanelQuarantine | None = None,
    allocation_admission: Callable[[int], None] | None = None,
    selection_mode: Literal["legacy_window", "causal_history"] = "legacy_window",
) -> dict[str, pd.DataFrame]:
    """Read ``<root>/<interval>/<SYMBOL>.parquet`` into wide per-column panels.

    Returns one wide DataFrame per requested column, all sharing
    ``build_uniform_grid(start, end, interval)`` as index and identical sorted
    column order. No survivorship filter: symbols that delisted inside the
    window are kept with NaN outside their life. ``data_policy`` selects the
    input-data contract: ``'legacy'`` keeps every bar, ``'zombie_mask_v1'``
    masks causally-detected zombie (long flat) bars from both the survivor
    count and every panel column. Causal-history mode retains source
    instruments independently of whole-window row counts and endpoint
    coverage. History qualification occurs at each decision. A physical source
    roster is not a point-in-time investable universe.

    Args:
        allocation_admission: Optional pre-allocation admission of additional dense panel and conversion working memory after source survivor discovery. Rejection preserves source coverage and prevents wide-plane allocation.
        selection_mode: ``'legacy_window'`` keeps the whole-window survivor
            filter; ``'causal_history'`` retains instruments with any in-window
            bar so history qualification stays local to each decision.

    Raises:
        DataIntegrityError: Supplied resource admission rejects wide panel construction.
    """
    if selection_mode not in ("legacy_window", "causal_history"):
        raise ValueError(f"unknown selection_mode '{selection_mode}'")
    causal_history = selection_mode == "causal_history"
    survivor_min_bars = 1 if causal_history else min_bars
    if data_policy not in DATA_POLICIES:
        raise ValueError(f"unknown data_policy '{data_policy}' (shared default {MHS_DATA_POLICY_DEFAULT})")
    data_policy = str(MhsDataPolicy(data_policy))
    grid = build_uniform_grid(start, end, interval)
    paths = sorted(p for p in glob.glob(os.path.join(root, interval, "*.parquet")) if not is_temp_artifact(os.path.basename(p)))
    names = [os.path.basename(p).removesuffix(".parquet") for p in paths]
    keep = set(partition_symbols(names, partition))

    start_ms = int(start.value // 1_000_000)
    end_ms = int(end.value // 1_000_000)
    masking = data_policy == DATA_POLICY_ZOMBIE_MASK_V1
    interval_ms = int(pd.Timedelta(interval).total_seconds() * 1000)
    masks: dict[str, np.ndarray] = {}

    # Discover survivors before allocating the wide panel.  The prior
    # ``dict[Series] -> DataFrame -> reindex`` construction held as many as
    # three full copies of every requested field while assembling long MHS
    # folds.  A full 2021--2025 dev panel can contain hundreds of symbols, so
    # that transient amplification terminates the process before the
    # fail-closed replay/report path can run.
    survivors: list[tuple[str, str]] = []
    scan_quarantined = 0
    for path, sym in zip(paths, names, strict=True):
        if sym not in keep:
            continue
        try:
            table = pq.read_table(
                path,
                columns=["timestamp"],
                filters=[[("timestamp", ">=", start_ms), ("timestamp", "<=", end_ms)]],
            )
        except (OSError, pa.ArrowException) as exc:
            if quarantine is None:
                raise
            quarantine.add(sym, f"unreadable:{type(exc).__name__}")
            scan_quarantined += 1
            continue
        idx = pd.to_datetime(table.column("timestamp").to_numpy(), unit="ms", utc=True)
        ts_ms = table.column("timestamp").to_numpy()
        keep_rows = (idx >= start) & (idx <= end)
        if masking:
            masks[path] = zombie_masked_timestamps(path, start_ms, end_ms, interval_ms)
            window_ts = ts_ms[keep_rows]
            # 좀비 꼬리(상장폐지 후 거래소가 계속 내주는 flat 봉)는 수집 누락이 아니라 생애 종료이므로 격리 대상이 아니다.
            zombie_tail = bool(window_ts.size) and bool(np.isin(window_ts.max(), masks[path]))
            keep_rows &= ~np.isin(ts_ms, masks[path])
        else:
            zombie_tail = False
        idx = idx[keep_rows]
        if len(idx.drop_duplicates(keep="last")) < survivor_min_bars:
            continue
        if quarantine is not None and not causal_history and not zombie_tail and idx.max() < end and idx.max() >= end - DECISION_BAR_LOOKBACK:
            quarantine.add(sym, "decision_bar_missing")
            scan_quarantined += 1
            continue
        survivors.append((path, sym))

    if not survivors:
        raise ValueError("no symbol survived the panel filters")

    if allocation_admission is not None:
        allocation_admission(len(grid) * len(survivors) * len(columns) * 8 * 2)

    values = {
        column: np.full((len(grid), len(survivors)), np.nan, dtype="float64")
        for column in columns
    }
    failed_columns: list[int] = []
    for column_index, (path, sym) in enumerate(survivors):
        try:
            table = pq.read_table(
                path,
                columns=["timestamp", *columns],
                filters=[[("timestamp", ">=", start_ms), ("timestamp", "<=", end_ms)]],
            )
        except (OSError, pa.ArrowException) as exc:
            if quarantine is None:
                raise
            quarantine.add(sym, f"unreadable:{type(exc).__name__}")
            failed_columns.append(column_index)
            continue
        idx = pd.to_datetime(table.column("timestamp").to_numpy(), unit="ms", utc=True)
        in_window = (idx >= start) & (idx <= end)
        if masking:
            in_window &= ~np.isin(table.column("timestamp").to_numpy(), masks[path])
        window_sources = np.flatnonzero(in_window)
        positions = grid.get_indexer(idx[in_window])
        valid_positions = positions >= 0
        source_positions = window_sources[valid_positions]
        target_positions = positions[valid_positions]
        if not len(target_positions):
            continue

        # Stable sorting makes the final source row win for duplicate
        # timestamps, exactly matching ``duplicated(keep='last')``.
        order = np.argsort(target_positions, kind="stable")
        ordered_targets = target_positions[order]
        keep_last = np.empty(len(order), dtype=bool)
        keep_last[:-1] = ordered_targets[:-1] != ordered_targets[1:]
        keep_last[-1] = True
        selected_sources = source_positions[order[keep_last]]
        selected_targets = ordered_targets[keep_last]
        for column in columns:
            field = table.column(column).to_numpy().astype("float64", copy=False)
            values[column][selected_targets, column_index] = field[selected_sources]

    if quarantine is not None:
        quarantine.enforce_limit(len(survivors) + scan_quarantined)

    if not failed_columns:
        symbols = [sym for _, sym in survivors]
        return {
            column: pd.DataFrame(values[column], index=grid, columns=symbols, copy=False)
            for column in columns
        }
    keep_mask = np.ones(len(survivors), dtype=bool)
    keep_mask[failed_columns] = False
    symbols = [sym for (_, sym), keep in zip(survivors, keep_mask, strict=True) if keep]
    if not symbols:
        raise ValueError("no symbol survived the panel filters")
    return {
        column: pd.DataFrame(values[column][:, keep_mask], index=grid, columns=symbols, copy=False)
        for column in columns
    }


def slice_base_panel(
    base_panel: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_bars: int = PANEL_MIN_HISTORY_BARS,
) -> dict[str, pd.DataFrame]:
    """Slice a pre-loaded 1h base panel to ``[start, end]`` with survivor filtration.

    In-memory counterpart of ``load_base_panel`` for fork-shared panels:
    each field is sliced with ``.loc[start:end]`` (float64 preserved) and only
    symbols with at least ``min_bars`` valid (non-NaN close, or first-field)
    rows are retained, identically to the disk loader's survivor filter.
    """
    if not base_panel:
        raise ValueError("base_panel must be non-empty")
    ref_key = "close" if "close" in base_panel else next(iter(base_panel))
    ref = base_panel[ref_key]
    sliced_ref = ref.loc[start:end]
    valid_counts = sliced_ref.notna().sum(axis=0)
    survivors = [c for c in ref.columns if valid_counts[c] >= min_bars]
    if not survivors:
        raise ValueError("no symbol survived the panel filters")
    return {
        column: frame.loc[start:end, survivors].astype("float64")
        for column, frame in base_panel.items()
    }


def liquid_half_eligibility(
    quote_volume: pd.DataFrame,
    lookback_bars: int,
    min_history_bars: int,
) -> pd.DataFrame:
    """Boolean PIT liquidity eligibility using a trailing cross-sectional median.

    At timestamp ``t`` each symbol's trailing mean quote volume uses only bars
    at or before ``t``; a symbol is eligible exactly when that mean is at least
    the valid-symbol cross-sectional median at ``t`` and it has observed
    ``min_history_bars`` bars. Missing history is False, never zero-filled.
    """
    if lookback_bars < 1 or min_history_bars < 1 or min_history_bars > lookback_bars:
        raise ValueError(
            "lookback_bars and min_history_bars must satisfy 1 <= min_history_bars <= lookback_bars"
        )
    trailing_mean = quote_volume.rolling(
        lookback_bars, min_periods=min_history_bars
    ).mean()
    median = trailing_mean.median(axis=1)
    eligible = trailing_mean.ge(median, axis=0)
    return eligible.fillna(False)


def fill_mark_parity_mask(
    fill_close: pd.DataFrame,
    mark_close: pd.DataFrame,
    max_log_divergence: float = FILL_MARK_MAX_LOG_DIVERGENCE,
) -> pd.DataFrame:
    """Boolean mask: True where the bar is tradeable at the modelled fill price.

    False only where both prices are finite and strictly positive AND
    ``abs(log(fill/mark)) > max_log_divergence``.  NaN, zero, or negative
    prices on either axis yield True (fail-open per I2/I6).
    """
    if max_log_divergence <= 0:
        raise ValueError(f"max_log_divergence must be > 0, got {max_log_divergence}")

    if not fill_close.index.equals(mark_close.index):
        raise ValueError("fill_close and mark_close must have identical index")
    if not fill_close.columns.equals(mark_close.columns):
        raise ValueError("fill_close and mark_close must have identical columns")

    fill_vals = fill_close.to_numpy(dtype="float64", copy=False)
    mark_vals = mark_close.to_numpy(dtype="float64", copy=False)

    both_positive = (fill_vals > 0) & (mark_vals > 0)
    both_finite = np.isfinite(fill_vals) & np.isfinite(mark_vals)
    comparable = both_positive & both_finite

    log_div = np.full_like(fill_vals, np.nan)
    log_div[comparable] = np.abs(
        np.log(fill_vals[comparable]) - np.log(mark_vals[comparable])
    )

    over_band = comparable & (log_div > max_log_divergence)

    mask = np.ones(fill_vals.shape, dtype=bool)
    mask[over_band] = False

    return pd.DataFrame(mask, index=fill_close.index, columns=fill_close.columns)
