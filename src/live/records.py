"""Typed monthly partition storage — evidence-immutable append layer."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from src.common.parquet_io import read_parquet_or_quarantine, write_parquet_atomic


def _enforce_dtypes(df: pd.DataFrame, dtypes: Mapping[str, str]) -> pd.DataFrame:
    for col, dtype in dtypes.items():
        if col not in df.columns:
            continue
        if "datetime64" in dtype:
            df[col] = pd.to_datetime(df[col], utc=True).astype(dtype)
        elif dtype in ("float64", "float32", "float"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        elif dtype.startswith("int"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
        elif dtype == "bool":
            df[col] = df[col].astype(bool)
        else:
            df[col] = df[col].astype(dtype)
    return df


def append_typed_frame(
    frame: pd.DataFrame,
    directory: Path,
    prefix: str,
    *,
    time_column: str,
    dtypes: Mapping[str, str],
    compression: str = "zstd",
) -> list[Path]:
    """Merge ``frame`` into monthly ``<prefix>_<YYYYMM>.parquet`` partitions keyed by ``time_column`` (UTC).

    Evidence partitions are append-only: existing rows are never dropped. Each touched month is
    read, concatenated with the new rows, dtype-normalized and atomically replaced, so a crash or a
    backup snapshot can only observe the previous or the new complete partition. An undecodable
    existing partition is quarantined (see ``read_parquet_or_quarantine``) and the month restarts
    from the new rows. Assumes a single writer process per ``directory``.

    Args:
        frame: New rows; ``time_column`` must be parseable as UTC timestamps.
        directory: Partition directory (created if missing).
        prefix: Partition file prefix.
        time_column: Column whose UTC month selects the partition.
        dtypes: Column dtype contract enforced on new and merged rows.
        compression: Parquet codec.

    Returns:
        Sorted partition paths written; empty when ``frame`` is empty.

    Raises:
        Exception: Merge or dtype-enforcement failures and OS-level read/write failures propagate;
            the existing partition is left unchanged.
    """
    if frame.empty:
        return []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # ensure time_column is datetime
    frame = frame.copy()
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
    # enforce dtypes
    frame = _enforce_dtypes(frame, dtypes)
    # group by YYYYMM of time_column UTC
    frame["_yyyymm"] = frame[time_column].dt.tz_convert("UTC").dt.strftime("%Y%m")
    written: list[Path] = []
    for yyyymm, group in frame.groupby("_yyyymm"):
        grp = group.drop(columns=["_yyyymm"])
        path = directory / f"{prefix}_{yyyymm}.parquet"
        existing = read_parquet_or_quarantine(path, stage=f"records_{prefix}")
        if existing is not None:
            combined = pd.concat([existing, grp], ignore_index=True)
            combined = _enforce_dtypes(combined, dtypes)
            # ensure time column remains datetime
            if time_column in combined.columns:
                combined[time_column] = pd.to_datetime(combined[time_column], utc=True).astype("datetime64[ns, UTC]")
            write_parquet_atomic(combined, path, compression=compression)
        else:
            write_parquet_atomic(grp, path, compression=compression)
        written.append(path)
    return sorted(written)
