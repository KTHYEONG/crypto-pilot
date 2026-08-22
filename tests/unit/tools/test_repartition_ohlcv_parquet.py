"""SCENARIO_MHS_PERF_P3_01_PARQUET_ROUNDTRIP: OHLCV row-group re-layout tool.

- After repartitioning a synthetic 876480-row 3m file with window_days=31,
  metadata.num_row_groups >= 50.
- Every column reads back np.array_equal to the original.
- A 31-day filtered read touches strictly fewer row groups than the total.
- dry_run=True writes nothing; a verification failure leaves the original
  file untouched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.common.errors import DataIntegrityError
from tools.devops.repartition_ohlcv_parquet import (
    overlapping_row_groups,
    repartition_ohlcv_parquet,
)

N_ROWS = 876_480  # the production 31-day 3m bar count
WINDOW_DAYS = 31


@pytest.fixture(scope="module")
def synthetic_market_dir(tmp_path_factory) -> Path:
    """One synthetic single-row-group 3m file with the production row count."""
    root = tmp_path_factory.mktemp("ohlcv_repartition")
    directory = root / "3m"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(20260807)
    start_ms = int(pd.Timestamp("2021-01-01", tz="UTC").value // 1_000_000)
    timestamps = start_ms + np.arange(N_ROWS, dtype=np.int64) * 180_000
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0 + rng.normal(0, 0.1, N_ROWS),
            "high": 100.5 + rng.normal(0, 0.1, N_ROWS),
            "low": 99.5 + rng.normal(0, 0.1, N_ROWS),
            "close": 100.0 + rng.normal(0, 0.1, N_ROWS),
        }
    )
    frame.to_parquet(directory / "AAAUSDT.parquet", index=False)
    return root


def _column_arrays(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    return {
        field.name: table.column(field.name).to_numpy(zero_copy_only=False)
        for field in table.schema
    }


def test_dry_run_writes_nothing(synthetic_market_dir) -> None:
    path = synthetic_market_dir / "3m" / "AAAUSDT.parquet"
    before_bytes = path.read_bytes()
    stats = repartition_ohlcv_parquet(synthetic_market_dir, "3m", dry_run=True)

    assert stats["files"] == 1
    assert stats["rewritten"] == 0
    assert stats["rows"] == N_ROWS
    assert path.read_bytes() == before_bytes
    assert pq.read_metadata(path).num_row_groups == 1


def test_repartition_produces_day_bounded_row_groups_and_identical_rows(
    synthetic_market_dir,
) -> None:
    original = _column_arrays(synthetic_market_dir / "3m" / "AAAUSDT.parquet")

    stats = repartition_ohlcv_parquet(
        synthetic_market_dir, "3m", window_days=WINDOW_DAYS, dry_run=False,
    )

    assert stats["rewritten"] == 1
    metadata = pq.read_metadata(synthetic_market_dir / "3m" / "AAAUSDT.parquet")
    expected_groups = -(-N_ROWS // (WINDOW_DAYS * 480))
    assert metadata.num_row_groups >= 50
    assert metadata.num_row_groups == expected_groups
    rewritten = _column_arrays(synthetic_market_dir / "3m" / "AAAUSDT.parquet")
    assert set(rewritten) == set(original)
    for name, column in original.items():
        assert np.array_equal(column, rewritten[name]), name


def test_filtered_read_touches_strictly_fewer_row_groups(synthetic_market_dir) -> None:
    path = synthetic_market_dir / "3m" / "AAAUSDT.parquet"
    total = pq.read_metadata(path).num_row_groups

    # A 31-day window in the middle of the file's span.
    start_ms = int(pd.Timestamp("2021-01-10", tz="UTC").value // 1_000_000)
    end_ms = int(pd.Timestamp("2021-02-10", tz="UTC").value // 1_000_000)
    touched = overlapping_row_groups(path, start_ms, end_ms)

    assert touched < total
    assert touched >= 1
    assert touched <= total


def test_verification_failure_leaves_original_untouched(
    synthetic_market_dir, monkeypatch
) -> None:
    from tools.devops import repartition_ohlcv_parquet as tool

    path = synthetic_market_dir / "3m" / "AAAUSDT.parquet"
    before_bytes = path.read_bytes()
    before_meta = pq.read_metadata(path).num_row_groups

    monkeypatch.setattr(tool, "_column_roundtrip_equal", lambda *_a, **_k: False)

    with pytest.raises(DataIntegrityError, match="round-trip"):
        repartition_ohlcv_parquet(synthetic_market_dir, "3m", dry_run=False)

    assert path.read_bytes() == before_bytes
    assert pq.read_metadata(path).num_row_groups == before_meta
    leftovers = list((synthetic_market_dir / "3m").glob("*repartition-tmp*"))
    assert not leftovers


def test_unsupported_timeframe_and_bad_window_days_raise(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported timeframe"):
        repartition_ohlcv_parquet(tmp_path, "2h")
    with pytest.raises(ValueError, match="window_days"):
        repartition_ohlcv_parquet(tmp_path, "3m", window_days=0)


def test_missing_directory_is_a_clean_noop(tmp_path) -> None:
    stats = repartition_ohlcv_parquet(tmp_path, "3m")
    assert stats["files"] == 0
    assert stats["rewritten"] == 0
