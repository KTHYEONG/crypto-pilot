from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.public_archive.store import (
    CoverageRow,
    append_coverage,
    load_coverage,
    merge_partition,
)

_DAY1 = date(2024, 1, 15)
_DAY2 = date(2024, 1, 16)


def _frame(day: date, values: list[float]) -> pd.DataFrame:
    stamps = [pd.Timestamp(day, tz="UTC").as_unit("ms") + pd.Timedelta(minutes=5 * i) for i in range(len(values))]
    return pd.DataFrame({"timestamp": stamps, "value": values})


def test_merge_is_idempotent(tmp_path: Path) -> None:
    frame = _frame(_DAY1, [1.0, 2.0])
    first = merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY1, frame)
    second = merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY1, frame)
    assert first == second == 2
    back = pd.read_parquet(tmp_path / "metrics" / "BTCUSDT" / "202401.parquet")
    assert len(back) == 2


def test_merge_replaces_only_that_day(tmp_path: Path) -> None:
    merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY1, _frame(_DAY1, [1.0]))
    merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY2, _frame(_DAY2, [9.0]))
    merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY1, _frame(_DAY1, [2.0]))
    back = pd.read_parquet(tmp_path / "metrics" / "BTCUSDT" / "202401.parquet")
    assert sorted(back["value"].tolist()) == [2.0, 9.0]


def _coverage_row(status: str, recorded_at: str) -> CoverageRow:
    return CoverageRow(
        dataset="metrics",
        symbol="BTCUSDT",
        day=_DAY1,
        status=status,  # type: ignore[arg-type]
        raw_sha256=None,
        rows=1,
        recorded_at=pd.Timestamp(recorded_at, tz="UTC"),
        detail=None,
    )


def test_coverage_last_row_wins(tmp_path: Path) -> None:
    append_coverage(tmp_path, _coverage_row("failed", "2024-01-17 00:00"))
    append_coverage(tmp_path, _coverage_row("ok", "2024-01-17 01:00"))
    loaded = load_coverage(tmp_path, "metrics")
    assert loaded[("BTCUSDT", _DAY1)].status == "ok"


def test_coverage_malformed_line_fails_closed(tmp_path: Path) -> None:
    ledger = tmp_path / "_coverage"
    ledger.mkdir(parents=True, exist_ok=True)
    (ledger / "metrics.jsonl").write_text("{not-json\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_coverage(tmp_path, "metrics")


def test_month_mismatch_rejected(tmp_path: Path) -> None:
    merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY1, _frame(_DAY1, [1.0]))
    before = (tmp_path / "metrics" / "BTCUSDT" / "202401.parquet").read_bytes()
    with pytest.raises(DataIntegrityError):
        merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY1, _frame(date(2024, 2, 1), [5.0]))
    assert (tmp_path / "metrics" / "BTCUSDT" / "202401.parquet").read_bytes() == before
