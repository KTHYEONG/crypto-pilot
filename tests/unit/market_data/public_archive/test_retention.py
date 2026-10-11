from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest

from src.market_data.public_archive.retention import ArchivePruneReport, prune_backed_up_archive
from src.market_data.streams.retention import BackupStatus

_NOW = pd.Timestamp("2024-06-15T12:00:00Z")
_STARTED = pd.Timestamp("2024-06-01T00:00:00Z")
_FINISHED = pd.Timestamp("2024-06-14T12:00:00Z")


def _status(started: pd.Timestamp = _STARTED, finished: pd.Timestamp = _FINISHED) -> BackupStatus:
    return BackupStatus(started_at=started, finished_at=finished)


def _partition(root: Path, dataset: str, symbol: str, month: str, *, mtime: float | None = None) -> Path:
    path = root / dataset / symbol / f"{month}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"parquet-stub")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _old_mtime() -> float:
    return time.mktime(pd.Timestamp("2024-01-10T00:00:00Z").to_pydatetime().timetuple())


def test_blocked_without_backup_status(tmp_path: Path) -> None:
    target = _partition(tmp_path, "metrics", "BTCUSDT", "202403", mtime=_old_mtime())
    report = prune_backed_up_archive(tmp_path, status=None, now=_NOW)
    assert report == ArchivePruneReport(prune_blocked=True, blocked_reason="status_missing", pruned_files=0, kept_files=0)
    assert target.exists()


def test_stale_backup_blocks(tmp_path: Path) -> None:
    target = _partition(tmp_path, "metrics", "BTCUSDT", "202403", mtime=_old_mtime())
    stale = _status(finished=_NOW - pd.Timedelta(hours=80), started=_NOW - pd.Timedelta(hours=81))
    report = prune_backed_up_archive(tmp_path, status=stale, now=_NOW)
    assert report.prune_blocked
    assert report.blocked_reason == "status_stale"
    assert target.exists()


def test_only_backed_up_old_months_pruned(tmp_path: Path) -> None:
    for month in ("202403", "202404", "202405", "202406"):
        _partition(tmp_path, "metrics", "BTCUSDT", month, mtime=_old_mtime())
    report = prune_backed_up_archive(tmp_path, status=_status(), now=_NOW, keep_days=45)
    assert not report.prune_blocked
    assert report.pruned_files == 2
    assert report.kept_files == 2
    assert not (tmp_path / "metrics" / "BTCUSDT" / "202403.parquet").exists()
    assert not (tmp_path / "metrics" / "BTCUSDT" / "202404.parquet").exists()
    assert (tmp_path / "metrics" / "BTCUSDT" / "202405.parquet").exists()
    assert (tmp_path / "metrics" / "BTCUSDT" / "202406.parquet").exists()


def test_modified_after_backup_is_kept(tmp_path: Path) -> None:
    fresh_mtime = time.mktime(pd.Timestamp("2024-06-10T00:00:00Z").to_pydatetime().timetuple())
    target = _partition(tmp_path, "bookDepth", "ETHUSDT", "202403", mtime=fresh_mtime)
    report = prune_backed_up_archive(tmp_path, status=_status(), now=_NOW)
    assert report.pruned_files == 0
    assert target.exists()


def test_ledger_and_status_never_pruned(tmp_path: Path) -> None:
    coverage = tmp_path / "_coverage" / "metrics.jsonl"
    coverage.parent.mkdir(parents=True, exist_ok=True)
    coverage.write_text("{}\n", encoding="utf-8")
    os.utime(coverage, (_old_mtime(), _old_mtime()))
    status_dir = tmp_path / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    beat = status_dir / "collect_heartbeat.json"
    beat.write_text("{}\n", encoding="utf-8")
    victim = _partition(tmp_path, "aggTrades", "BTCUSDT", "202403", mtime=_old_mtime())
    report = prune_backed_up_archive(tmp_path, status=_status(), now=_NOW)
    assert not victim.exists()
    assert report.pruned_files == 1
    assert coverage.exists()
    assert beat.exists()


def test_empty_symbol_dir_removed_but_root_kept(tmp_path: Path) -> None:
    _partition(tmp_path, "metrics", "BTCUSDT", "202403", mtime=_old_mtime())
    prune_backed_up_archive(tmp_path, status=_status(), now=_NOW)
    assert not (tmp_path / "metrics" / "BTCUSDT").exists()
    assert (tmp_path / "metrics").is_dir()


def test_keep_days_floor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_days"):
        prune_backed_up_archive(tmp_path, status=_status(), now=_NOW, keep_days=10)


@pytest.mark.parametrize("stamp", ["2024-03-31T23:59:59Z", "2024-01-31T23:59:59Z"])
def test_previous_calendar_month_always_kept(tmp_path: Path, stamp: str) -> None:
    now = pd.Timestamp(stamp)
    previous = now.replace(day=1) - pd.Timedelta(days=1)
    target = _partition(tmp_path, "metrics", "BTCUSDT", previous.strftime("%Y%m"), mtime=1.0)
    report = prune_backed_up_archive(
        tmp_path, status=_status(started=now - pd.Timedelta(hours=2), finished=now - pd.Timedelta(hours=1)),
        now=now, keep_days=31,
    )
    assert report.pruned_files == 0
    assert target.exists()


def test_exact_retention_boundary_kept(tmp_path: Path) -> None:
    target = _partition(tmp_path, "metrics", "BTCUSDT", "202403", mtime=_old_mtime())
    now = pd.Timestamp("2024-05-16T00:00:00Z")
    status = _status(started=now - pd.Timedelta(hours=2), finished=now - pd.Timedelta(hours=1))
    report = prune_backed_up_archive(tmp_path, status=status, now=now, keep_days=45)
    assert report.pruned_files == 0
    assert target.exists()


def test_slice_dataset_pruned(tmp_path: Path) -> None:
    target = _partition(tmp_path, "aggTrades_slice", "BTCUSDT", "202403", mtime=_old_mtime())
    report = prune_backed_up_archive(tmp_path, status=_status(), now=_NOW, keep_days=45)
    assert report.pruned_files == 1
    assert not target.exists()


def test_malformed_filename_kept(tmp_path: Path) -> None:
    target = _partition(tmp_path, "metrics", "BTCUSDT", "notes", mtime=_old_mtime())
    report = prune_backed_up_archive(tmp_path, status=_status(), now=_NOW)
    assert report.pruned_files == 0
    assert target.exists()


def test_december_rollover_pruned(tmp_path: Path) -> None:
    target = _partition(tmp_path, "metrics", "BTCUSDT", "202312", mtime=_old_mtime())
    report = prune_backed_up_archive(tmp_path, status=_status(), now=_NOW, keep_days=45)
    assert report.pruned_files == 1
    assert not target.exists()


def test_nested_file_kept(tmp_path: Path) -> None:
    target = tmp_path / "metrics" / "BTCUSDT" / "extra" / "202403.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"parquet-stub")
    os.utime(target, (_old_mtime(), _old_mtime()))
    report = prune_backed_up_archive(tmp_path, status=_status(), now=_NOW)
    assert report.pruned_files == 0
    assert target.exists()


def test_stat_failure_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import Any

    target = _partition(tmp_path, "metrics", "BTCUSDT", "202403", mtime=_old_mtime())
    real_stat = Path.stat

    def _fail(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.suffix == ".parquet":
            raise OSError("injected")
        return real_stat(self, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "stat", _fail)
        with pytest.raises(OSError, match="injected"):
            prune_backed_up_archive(tmp_path, status=_status(), now=_NOW)
    assert target.exists()


def test_unlink_failure_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import Any

    target = _partition(tmp_path, "metrics", "BTCUSDT", "202403", mtime=_old_mtime())

    def _fail(self: Path, *args: Any, **kwargs: Any) -> Any:
        raise OSError("injected")

    with monkeypatch.context() as patched:
        patched.setattr(Path, "unlink", _fail)
        with pytest.raises(OSError, match="injected"):
            prune_backed_up_archive(tmp_path, status=_status(), now=_NOW)
    assert target.exists()


def test_top_level_file_kept(tmp_path: Path) -> None:
    stray = tmp_path / "metrics" / "202403.parquet"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"parquet-stub")
    os.utime(stray, (_old_mtime(), _old_mtime()))
    report = prune_backed_up_archive(tmp_path, status=_status(), now=_NOW)
    assert report.pruned_files == 0
    assert stray.exists()


def test_rmdir_failure_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import Any

    _partition(tmp_path, "metrics", "BTCUSDT", "202403", mtime=_old_mtime())
    real_rmdir = Path.rmdir

    def _fail(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.name == "BTCUSDT":
            raise OSError("injected")
        return real_rmdir(self, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "rmdir", _fail)
        with pytest.raises(OSError, match="injected"):
            prune_backed_up_archive(tmp_path, status=_status(), now=_NOW)
    assert (tmp_path / "metrics" / "BTCUSDT").is_dir()
