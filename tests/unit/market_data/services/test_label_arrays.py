"""Invariant guards for exact ms-to-ns label codecs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.market_data.services.label_arrays import MS_TO_NS, decode_ms_labels_ns, sorted_unique_labels

B = int(np.iinfo(np.int64).max // MS_TO_NS)


def _ref_decode(raw: np.ndarray) -> np.ndarray:
    idx = pd.to_datetime(raw, unit="ms", utc=True, errors="coerce")
    valid = pd.DatetimeIndex(idx).dropna()
    if len(valid) == 0:
        return np.zeros(0, dtype="int64")
    return np.asarray(valid.as_unit("ns").asi8, dtype="int64")


def _ref_merge(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype="int64")
    return np.unique(np.concatenate(chunks)).astype("int64", copy=False)


def test_decode_matches_pandas_within_exact_bounds() -> None:
    for raw in (
        np.array([0, 1_700_000_000_000, -86_400_000], dtype="int64"),
        np.array([B], dtype="int64"),
        np.array([-B], dtype="int64"),
    ):
        out = decode_ms_labels_ns(raw)
        ref = _ref_decode(raw)
        assert out.dtype == ref.dtype == np.dtype("int64")
        assert np.array_equal(out, ref)


def test_decode_out_of_range_raises_like_pandas() -> None:
    for raw in (
        np.array([B + 1], dtype="int64"),
        np.array([-(B + 1)], dtype="int64"),
        np.array([32503680000000], dtype="int64"),
    ):
        with pytest.raises(pd.errors.OutOfBoundsDatetime):
            decode_ms_labels_ns(raw)
        with pytest.raises(pd.errors.OutOfBoundsDatetime):
            _ref_decode(raw)


def test_decode_drops_nat_sentinel_like_pandas() -> None:
    raw = np.array([np.iinfo(np.int64).min, 5], dtype="int64")
    out = decode_ms_labels_ns(raw)
    ref = _ref_decode(raw)
    assert np.array_equal(out, np.array([5_000_000], dtype="int64"))
    assert np.array_equal(out, ref)


def test_decode_non_int64_inputs_use_reference_path() -> None:
    cases = [
        np.array([1.0, float("nan")], dtype="float64"),
        np.array([5], dtype="int32"),
        np.array([5], dtype="uint64"),
        np.zeros(0, dtype="int64"),
    ]
    for raw in cases:
        out = decode_ms_labels_ns(raw)
        ref = _ref_decode(raw)
        assert out.dtype == ref.dtype == np.dtype("int64")
        assert np.array_equal(out, ref)


def test_decode_does_not_alias_input() -> None:
    raw = np.array([1_700_000_000_000, 5], dtype="int64")
    out = decode_ms_labels_ns(raw)
    raw[:] = 0
    assert np.array_equal(out, np.array([1_700_000_000_000_000_000, 5_000_000], dtype="int64"))


def test_merge_equals_unique_on_boundary_chunks() -> None:
    a = np.array([1, 3, 5], dtype="int64")
    b = np.array([7, 9], dtype="int64")
    cases: list[list[np.ndarray]] = [
        [],
        [np.zeros(0, dtype="int64")],
        [np.array([42], dtype="int64")],
        [a, b],
        [np.array([1, 3, 3], dtype="int64"), np.array([3, 5], dtype="int64")],
        [np.array([1, 2, 2, 3], dtype="int64")],
        [np.array([9, 7, 5], dtype="int64")],
        [np.array([3, 1, 2], dtype="int64")],
    ]
    for chunks in cases:
        out = sorted_unique_labels(chunks)
        ref = _ref_merge(chunks)
        assert out.dtype == ref.dtype == np.dtype("int64")
        assert np.array_equal(out, ref)


def test_merge_leaves_inputs_untouched() -> None:
    chunks = [np.array([3, 1, 3], dtype="int64"), np.array([2, 5], dtype="int64")]
    copies = [c.copy() for c in chunks]
    sorted_unique_labels(chunks)
    for got, want in zip(chunks, copies, strict=True):
        assert np.array_equal(got, want)
