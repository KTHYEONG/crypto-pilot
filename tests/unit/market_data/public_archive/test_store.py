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


def _ok_row(day: date, rows: int = 10) -> CoverageRow:
    return CoverageRow(
        dataset="metrics",
        symbol="BTCUSDT",
        day=day,
        status="ok",
        raw_sha256=None,
        rows=rows,
        recorded_at=pd.Timestamp("2024-01-17", tz="UTC"),
        detail=None,
    )


def test_archive_lock_invalid_wait(tmp_path: Path) -> None:
    from src.market_data.public_archive.store import archive_lock

    with pytest.raises(ValueError, match="wait_s"), archive_lock(tmp_path, wait_s=-1):
        pass
    with pytest.raises(ValueError, match="wait_s"), archive_lock(tmp_path, wait_s=float("inf")):
        pass


def _spawn_holder(root: Path, hold_s: float, holder: str = "child"):  # type: ignore[no-untyped-def]
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        from pathlib import Path
        import time
        from src.market_data.public_archive.store import archive_lock
        root = Path({str(root)!r})
        with archive_lock(root, wait_s=0, holder={holder!r}):
            print("ready", flush=True)
            time.sleep({hold_s})
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)  # noqa: S603
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "ready"
    return proc


def test_archive_lock_excludes_second_process(tmp_path: Path) -> None:
    import time

    from src.market_data.public_archive.store import ArchiveBusyError, archive_lock

    proc = _spawn_holder(tmp_path, 3.0, holder="child-holder")
    try:
        deadline = time.monotonic() + 10
        seen = ""
        while time.monotonic() < deadline:
            try:
                with archive_lock(tmp_path, wait_s=0):
                    pass
            except ArchiveBusyError as exc:
                seen = str(exc)
                break
            time.sleep(0.05)
        assert seen != ""
        assert "child-holder" in seen
        with pytest.raises(ArchiveBusyError), archive_lock(tmp_path, wait_s=0):
            pass
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    with archive_lock(tmp_path, wait_s=0):
        pass


def test_archive_lock_waits_then_succeeds(tmp_path: Path) -> None:
    import time

    from src.market_data.public_archive.store import archive_lock

    proc = _spawn_holder(tmp_path, 0.5, holder="waiter")
    try:
        start = time.monotonic()
        with archive_lock(tmp_path, wait_s=3):
            elapsed = time.monotonic() - start
        assert elapsed >= 0.45
    finally:
        proc.wait(timeout=10)


def test_nested_acquire_fails_fast(tmp_path: Path) -> None:
    from src.market_data.public_archive.store import ArchiveBusyError, archive_lock

    with archive_lock(tmp_path, wait_s=0), pytest.raises(ArchiveBusyError), archive_lock(tmp_path, wait_s=10):
        pass


def test_lock_released_on_exception(tmp_path: Path) -> None:
    from src.market_data.public_archive.store import archive_lock

    with pytest.raises(RuntimeError), archive_lock(tmp_path, wait_s=0):
        raise RuntimeError("boom")
    with archive_lock(tmp_path, wait_s=0):
        pass


def test_lock_io_failure_is_not_reported_as_contention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import store

    def fail(*args: object) -> None:
        raise OSError("lock filesystem failure")

    with monkeypatch.context() as patch:
        patch.setattr(store.fcntl, "flock", fail)
        with pytest.raises(OSError, match="filesystem failure"), store.archive_lock(tmp_path, wait_s=30):
            pass
    with store.archive_lock(tmp_path, wait_s=0):
        pass


def test_lock_metadata_failure_releases_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import store

    def fail(*args: object) -> None:
        raise OSError("metadata failure")

    with monkeypatch.context() as patch:
        patch.setattr(store.os, "fsync", fail)
        with pytest.raises(OSError, match="metadata failure"), store.archive_lock(tmp_path, wait_s=0):
            pass
    with store.archive_lock(tmp_path, wait_s=0):
        pass


def test_sealed_month_detection(tmp_path: Path) -> None:
    from src.market_data.public_archive.store import sealed_months

    cov = {("BTCUSDT", date(2023, 3, 5)): _ok_row(date(2023, 3, 5))}
    assert ("derived", "BTCUSDT", "202303") in sealed_months(tmp_path, "metrics", cov)
    cov_zero = {("BTCUSDT", date(2023, 3, 5)): _ok_row(date(2023, 3, 5), rows=0)}
    assert sealed_months(tmp_path, "metrics", cov_zero) == frozenset()
    part = tmp_path / "metrics" / "BTCUSDT" / "202303.parquet"
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"stub")
    assert sealed_months(tmp_path, "metrics", cov) == frozenset()


def test_sealed_slice_only_when_derived_present(tmp_path: Path) -> None:
    from src.market_data.public_archive.store import sealed_months

    cov = {("BTCUSDT", date(2023, 3, 5)): _ok_row(date(2023, 3, 5))}
    derived = tmp_path / "aggTrades" / "BTCUSDT" / "202303.parquet"
    derived.parent.mkdir(parents=True, exist_ok=True)
    derived.write_bytes(b"stub")
    sealed = sealed_months(tmp_path, "aggTrades", cov)
    assert ("derived", "BTCUSDT", "202303") not in sealed
    assert ("slice", "BTCUSDT", "202303") in sealed


def test_archive_lock_busy_without_readable_holder(tmp_path: Path, monkeypatch) -> None:
    from pathlib import Path as _Path

    from src.market_data.public_archive.store import ArchiveBusyError, archive_lock

    proc = _spawn_holder(tmp_path, 2.0, holder="x")
    try:
        import time as _time

        deadline = _time.monotonic() + 10
        while _time.monotonic() < deadline:
            try:
                with archive_lock(tmp_path, wait_s=0):
                    pass
            except ArchiveBusyError:
                break
            _time.sleep(0.05)
        monkeypatch.setattr(_Path, "read_text", lambda *a, **k: (_ for _ in ()).throw(OSError("gone")))
        with pytest.raises(ArchiveBusyError, match="Another process holds"), archive_lock(tmp_path, wait_s=0):
            pass
    finally:
        proc.terminate()
        proc.wait(timeout=10)
