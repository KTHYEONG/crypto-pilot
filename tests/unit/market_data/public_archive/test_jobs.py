from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.public_archive import jobs


def _job(dataset: str = "metrics", second: int = 0) -> jobs.BackfillJob:
    return jobs.BackfillJob(
        f"{dataset}-2024-01-01-2024-01-02-20240101T0000{second:02d}Z",
        dataset,
        date(2024, 1, 1),
        date(2024, 1, 2),
        "/census",
        None,
        None,
        pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(seconds=second),
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"dataset": "unknown"},
        {"end": date(2023, 1, 1)},
        {"start": date(2020, 1, 1)},
        {"created_at": pd.Timestamp("2024-01-01")},
        {"created_at": pd.NaT},
        {"job_id": "../../escape"},
        {"job_id": "metrics-2024-02-01-2024-02-02-20240101T000000Z"},
        {"census_root": ""},
        {"breadth": 0},
        {"preroll_days": -1},
        {"breadth": True},
    ],
)
def test_invalid_requests_fail_before_writing(fields: dict) -> None:
    with pytest.raises(ValueError, match=r"."):
        replace(_job(), **fields)


def test_order_duplicate_overlap_and_lifecycle(tmp_path: Path) -> None:
    first, second = _job(), _job("aggTrades", 1)
    assert jobs.pending_jobs(tmp_path) == []
    assert jobs.done_jobs(tmp_path) == []
    jobs.enqueue_backfill(tmp_path, second)
    jobs.enqueue_backfill(tmp_path, first)
    assert jobs.pending_jobs(tmp_path) == [first, second]
    with pytest.raises(FileExistsError):
        jobs.enqueue_backfill(tmp_path, first)
    with pytest.raises(ValueError, match=first.job_id):
        jobs.enqueue_backfill(tmp_path, _job(second=2))
    done = jobs.finish_job(tmp_path, first.job_id, summary={"ok": 2})
    assert json.loads(done.read_text())["summary"] == {"ok": 2}
    assert jobs.done_jobs(tmp_path) == [first]
    jobs.cancel_job(tmp_path, second.job_id)
    assert jobs.cancelled_jobs(tmp_path) == [second]
    assert jobs.pending_jobs(tmp_path) == []
    with pytest.raises(FileNotFoundError):
        jobs.finish_job(tmp_path, first.job_id, summary={})
    with pytest.raises(ValueError, match="job_id"):
        jobs.cancel_job(tmp_path, "../escape")


@pytest.mark.parametrize(
    "payload",
    [
        b"{bad",
        b"\xff",
        b"[]",
        b"{}",
        {"breadth": True},
        {"preroll_days": "1"},
        {"census_root": 3},
        {"created_at": "NaT"},
    ],
)
def test_malformed_pending_fails_closed(tmp_path: Path, payload: bytes | dict) -> None:
    path = jobs.enqueue_backfill(tmp_path, _job())
    if isinstance(payload, dict):
        original = json.loads(path.read_text())
        original.update(payload)
        path.write_text(json.dumps(original))
    else:
        path.write_bytes(payload)
    with pytest.raises(DataIntegrityError):
        jobs.pending_jobs(tmp_path)


def test_filename_mismatch_and_finish_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = jobs.enqueue_backfill(tmp_path, _job())
    other = path.with_name("other.json")
    path.rename(other)
    with pytest.raises(DataIntegrityError):
        jobs.pending_jobs(tmp_path)
    with pytest.raises(DataIntegrityError):
        jobs.cancel_job(tmp_path, "other")
    other.write_text("{")
    with pytest.raises(DataIntegrityError):
        jobs.cancel_job(tmp_path, "other")
    other.unlink()
    path = jobs.enqueue_backfill(tmp_path, _job())
    done = tmp_path / "_jobs/done" / path.name
    done.parent.mkdir()
    done.write_text("{}")
    with pytest.raises(FileExistsError):
        jobs.finish_job(tmp_path, _job().job_id, summary={})
    assert path.exists()

    def fail(*args: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(jobs.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        jobs.enqueue_backfill(tmp_path, _job("bookDepth"))
    assert list(path.parent.glob(".job-*")) == []
