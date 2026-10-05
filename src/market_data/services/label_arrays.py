"""Exact array codecs for compact source-label reads.

Label readers decode epoch-millisecond parquet columns and merge per-row-group
chunks into sorted unique UTC nanoseconds. Both operations sit on the full
multi-year three-minute history of every symbol, so they avoid per-element
datetime boxing and redundant sorting while staying bit-identical to the
pandas/numpy reference expressions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

MS_TO_NS: Final[int] = 1_000_000

_INT64_MAX_MS: Final[int] = int(np.iinfo(np.int64).max // MS_TO_NS)


def decode_ms_labels_ns(raw: np.ndarray) -> np.ndarray:
    """Decode one row group of epoch-millisecond labels to valid UTC epoch nanoseconds.

    Integer millisecond columns whose every value is representable in int64
    nanoseconds are scaled exactly; any other input (float columns carrying NaN
    for nulls, non-int64 dtypes, the int64 NaT sentinel, or values outside the
    nanosecond range) takes the pandas coercion path so invalid rows are dropped
    or rejected exactly as before.

    Args:
        raw: Raw ``timestamp`` values of one row group as returned by
            ``ChunkedArray.to_numpy()``.
    Returns:
        int64 nanoseconds of the valid labels in input order (no sort, no
        deduplication); a zero-length int64 array when no label is valid.
    Raises:
        pandas.errors.OutOfBoundsDatetime: A value lies outside the int64
            nanosecond range (callers wrap it as ``DataIntegrityError``).
    """
    if raw.size == 0:
        return np.zeros(0, dtype="int64")
    if raw.dtype == np.dtype("int64") and int(raw.min()) >= -_INT64_MAX_MS and int(raw.max()) <= _INT64_MAX_MS:
        return np.asarray(raw * MS_TO_NS, dtype="int64")
    idx = pd.to_datetime(raw, unit="ms", utc=True, errors="coerce")
    valid = pd.DatetimeIndex(idx).dropna()
    if len(valid) == 0:
        return np.zeros(0, dtype="int64")
    return np.asarray(valid.as_unit("ns").asi8, dtype="int64")


def sorted_unique_labels(chunks: Sequence[np.ndarray]) -> np.ndarray:
    """Merge per-row-group int64 label chunks into sorted unique labels.

    Source files are written chronologically, so the concatenation is almost
    always already strictly increasing; an O(n) monotonicity check then replaces
    the O(n log n) sort that ``np.unique`` performs on the full history.

    Args:
        chunks: int64 label arrays in file row-group order.
    Returns:
        int64 array equal in values and dtype to
        ``np.unique(np.concatenate(chunks))``; a zero-length int64 array when
        ``chunks`` is empty.
    """
    if len(chunks) == 0:
        return np.zeros(0, dtype="int64")
    combined = np.concatenate(chunks).astype("int64", copy=False)
    if combined.size < 2 or bool((combined[1:] > combined[:-1]).all()):
        return combined
    return np.unique(combined).astype("int64", copy=False)
