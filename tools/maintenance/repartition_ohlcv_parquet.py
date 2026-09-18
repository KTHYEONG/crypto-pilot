"""One-time OHLCV parquet row-group re-layout (E1).

Every 3m OHLCV file is historically a SINGLE row group, so the timestamp
filter predicate in ``_load_window_minute_frames`` prunes nothing and the
filtered read decodes the whole group before masking. Rewriting each file with
``row_group_size = window_days * bars_per_day`` makes the predicate actually
prune (measured 7.70 -> 3.65 ms per window read, rows bit-identical,
+57% disk). Fail-closed: every column must round-trip byte-equal before the
original is atomically replaced.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.errors import DataIntegrityError

#: Bars per UTC day for each supported OHLCV timeframe.
BARS_PER_DAY: dict[str, int] = {
    "1m": 1440,
    "3m": 480,
    "5m": 288,
    "15m": 96,
    "1h": 24,
}


def _column_roundtrip_equal(
    original: pa.ChunkedArray | pa.Array,
    rewritten: pa.ChunkedArray | pa.Array,
) -> bool:
    """Byte-level round-trip equality of one column (NaN-aware for floats)."""
    if len(original) != len(rewritten):
        return False
    a = original.to_numpy(zero_copy_only=False)
    b = rewritten.to_numpy(zero_copy_only=False)
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def repartition_ohlcv_parquet(
    root: Path,
    timeframe: str,
    *,
    window_days: int = 31,
    dry_run: bool = True,
) -> dict[str, int]:
    """Rewrite every ``root/<timeframe>/*.parquet`` with day-bounded row groups.

    ``row_group_size = window_days * bars_per_day`` so the per-window timestamp
    filter can prune whole row groups. Each candidate is written to a temp path
    in the same directory, verified column-for-column against the original
    (fail-closed ``DataIntegrityError`` on any mismatch; the original stays
    untouched), then atomically renamed over it. ``dry_run=True`` only counts.

    Returns ``{"files", "rewritten", "rows", "row_groups_after"}``.
    """
    if timeframe not in BARS_PER_DAY:
        raise ValueError(
            f"unsupported timeframe '{timeframe}'; expected one of {sorted(BARS_PER_DAY)}"
        )
    if window_days < 1:
        raise ValueError(f"window_days must be >= 1, got {window_days}")
    directory = Path(root) / timeframe
    stats = {
        "files": 0,
        "rewritten": 0,
        "rows": 0,
        "row_groups_after": 0,
    }
    if not directory.is_dir():
        return stats

    row_group_size = window_days * BARS_PER_DAY[timeframe]
    for path in sorted(directory.glob("*.parquet")):
        original_table = pq.read_table(path)
        stats["files"] += 1
        stats["rows"] += original_table.num_rows
        if dry_run:
            continue
        tmp_path = path.with_name(path.name + ".repartition-tmp")
        try:
            pq.write_table(original_table, tmp_path, row_group_size=row_group_size)
            rewritten_table = pq.read_table(tmp_path)
            if original_table.schema != rewritten_table.schema:
                raise DataIntegrityError(
                    f"repartition schema drift for {path.name}: "
                    f"{original_table.schema} != {rewritten_table.schema}"
                )
            for field_index, field in enumerate(original_table.schema):
                if not _column_roundtrip_equal(
                    original_table.column(field_index),
                    rewritten_table.column(field_index),
                ):
                    raise DataIntegrityError(
                        f"repartition verification failed for {path.name}: "
                        f"column '{field.name}' does not round-trip byte-equal"
                    )
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        metadata = pq.read_metadata(path)
        stats["rewritten"] += 1
        stats["row_groups_after"] += metadata.num_row_groups
    return stats


def overlapping_row_groups(
    path: Path,
    start_ms: int,
    end_ms: int,
    *,
    timestamp_column: str = "timestamp",
) -> int:
    """Row groups whose timestamp statistics intersect ``[start_ms, end_ms]``."""
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    schema_arrow = metadata.schema.to_arrow_schema()
    column_index = schema_arrow.get_field_index(timestamp_column)
    touched = 0
    for group in range(metadata.num_row_groups):
        group_metadata = metadata.row_group(group)
        column_metadata = group_metadata.column(column_index)
        statistics = column_metadata.statistics
        if statistics is None or not statistics.has_min_max:
            touched += 1  # no statistics: assume it intersects (conservative)
            continue
        if statistics.min <= end_ms and statistics.max >= start_ms:
            touched += 1
    return touched
