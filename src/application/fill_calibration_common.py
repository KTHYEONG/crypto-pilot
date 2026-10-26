"""Shared tape I/O and notional-weighted statistics for the fill calibration cohorts."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.market_data.public_archive.catalog import AGG_TRADES_SLICE_COLUMNS

if TYPE_CHECKING:
    from src.engine.execution.tape_fill import TapeTrades

PROGRESS_INTERVAL_S = 60.0


def bump(exclusions: dict[str, int], reason: str) -> None:
    exclusions[reason] = exclusions.get(reason, 0) + 1


def side_sign(side: str) -> float:
    return 1.0 if side == "BUY" else -1.0


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, q))


def bootstrap_diff_ci(
    days: np.ndarray,
    day_of_row: np.ndarray,
    metric_rule: np.ndarray,
    metric_base: np.ndarray,
    weights: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Day-block bootstrap of the weighted rule-minus-base difference; returns (point, lo95, hi95)."""
    uniq = np.unique(day_of_row)
    n_days = len(uniq)
    if n_days == 0:
        return float("nan"), float("nan"), float("nan")
    _, inverse = np.unique(day_of_row, return_inverse=True)
    numerators = np.bincount(inverse, weights=(metric_rule - metric_base) * weights)
    denominators = np.bincount(inverse, weights=weights)
    rng = np.random.default_rng(int(seed))
    point_num_r = float(np.sum(metric_rule * weights))
    point_num_b = float(np.sum(metric_base * weights))
    denom = float(np.sum(weights))
    point = (point_num_r - point_num_b) / denom if denom > 0 else float("nan")
    diffs = np.empty(samples, dtype="float64")
    batch_size = max(1, min(samples, 65536 // n_days))
    for start in range(0, samples, batch_size):
        picks = rng.integers(0, n_days, size=(min(batch_size, samples - start), n_days))
        num = numerators[picks].sum(axis=1)
        den = denominators[picks].sum(axis=1)
        diffs[start : start + len(picks)] = np.divide(num, den, out=np.full(len(picks), np.nan), where=den > 0)
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        return point, float("nan"), float("nan")
    return point, min(point, float(np.percentile(finite, 2.5))), max(point, float(np.percentile(finite, 97.5)))


def read_tape_partition(path: Path) -> pd.DataFrame:
    """Read only the five slice columns, rejecting incomplete partition schemas."""
    import pyarrow.parquet as pq

    try:
        missing = set(AGG_TRADES_SLICE_COLUMNS) - set(pq.ParquetFile(path).schema_arrow.names)
        if missing:
            raise DataIntegrityError(f"tape partition {path} missing columns: {sorted(missing)}")
        return pd.read_parquet(path, columns=list(AGG_TRADES_SLICE_COLUMNS))
    except DataIntegrityError:
        raise
    except Exception as exc:
        raise DataIntegrityError(f"unreadable tape partition {path}: {exc}") from exc


def tape_partition_path(archive_root: Path, symbol: str, month: str) -> Path:
    from src.market_data.public_archive import store

    try:
        return store.partition_path(archive_root, "aggTrades", symbol, month, kind="slice")
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f"malformed tape partition key: {exc}") from exc


def slice_bounds_ms(day: date) -> tuple[int, int]:
    from src.market_data.public_archive.catalog import TAPE_SLICE_HOURS

    base = pd.Timestamp(day, tz="UTC")
    start = base + pd.Timedelta(hours=TAPE_SLICE_HOURS[0])
    end = base + pd.Timedelta(hours=TAPE_SLICE_HOURS[1])
    return int(start.value // 10**6), int(end.value // 10**6)


def day_tape(partition: pd.DataFrame, symbol: str, day: date) -> TapeTrades:
    """Build the day's TapeTrades from a monthly slice partition; raises DataIntegrityError on malformed rows."""
    from src.engine.execution.tape_fill import TapeTrades

    s_ms, e_ms = slice_bounds_ms(day)
    times = partition["transact_time_ms"]
    day_frame = partition.loc[(times >= s_ms) & (times < e_ms), list(AGG_TRADES_SLICE_COLUMNS)]
    try:
        return TapeTrades.from_slice_frame(day_frame, coverage_ms=(s_ms, e_ms))
    except DataIntegrityError as exc:
        raise DataIntegrityError(f"malformed tape slice {(symbol, day)}: {exc}") from exc


def design_cutoff() -> pd.Timestamp:
    from src.strategy.targets import FLOW_MOM_TOP20

    return FLOW_MOM_TOP20.design_data_cutoff
