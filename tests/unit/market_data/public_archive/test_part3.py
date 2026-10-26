from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.public_archive.catalog import daily_key
from src.market_data.public_archive.service import CollectionPlan, execute_plan

_HDR = (
    "create_time,symbol,sum_open_interest,sum_open_interest_value,"
    "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
    "count_long_short_ratio,sum_taker_long_short_vol_ratio"
)


def _zip(body: str) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as zf:
        zf.writestr("data.csv", body)
    return b.getvalue()


class _Fake:
    def __init__(self) -> None:
        self.keys: list[str] = []
        self.blobs: dict[str, bytes] = {}

    def add(self, key: str, payload: bytes) -> None:
        d = hashlib.sha256(payload).hexdigest()
        s = key.removeprefix("data/futures/um/")
        self.keys.extend([key, key + ".CHECKSUM"])
        self.blobs[s] = payload
        self.blobs[s + ".CHECKSUM"] = f"{d}  {key}\n".encode()

    def list_object_keys(self, *, prefix: str, timeout: int | None = None) -> list[str] | None:
        del timeout
        return [k for k in self.keys if k.startswith(prefix)]

    def fetch_archive_bytes(self, *parts: str) -> bytes | None:
        return self.blobs.get("/".join(parts))


def _mk(tmp: Path, days: list[date]) -> _Fake:
    f = _Fake()
    for d in days:
        stamp = pd.Timestamp(d, tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
        f.add(daily_key("metrics", "BTCUSDT", d), _zip(f"{_HDR}\n{stamp},BTCUSDT,1,2,3,4,5,6\n"))
    return f


def test_deadline_stops_and_naive_rejected(tmp_path: Path) -> None:
    days = [date(2024, 1, 15), date(2024, 2, 15), date(2024, 3, 15)]
    fake = _mk(tmp_path, days)
    plans = [CollectionPlan("metrics", "BTCUSDT", (d,), None) for d in days]
    calls = [0]

    def now_fn() -> pd.Timestamp:
        calls[0] += 1
        if calls[0] <= 4:
            return pd.Timestamp("2024-04-01T00:00", tz="UTC")
        return pd.Timestamp("2024-04-02", tz="UTC")

    rep = execute_plan(
        fake,
        plans,
        root=tmp_path,
        workers=1,
        now_fn=now_fn,
        deadline=pd.Timestamp("2024-04-01T12:00", tz="UTC"),
    )
    from src.market_data.public_archive.store import load_coverage

    cov = load_coverage(tmp_path, "metrics")
    assert ("BTCUSDT", days[0]) in cov
    assert ("BTCUSDT", days[1]) not in cov
    assert ("BTCUSDT", days[2]) not in cov
    assert rep.stopped_at_deadline is True
    assert rep.remaining_days == 2
    with pytest.raises(ValueError, match="UTC"):
        execute_plan(fake, plans, root=tmp_path, workers=1, deadline=pd.Timestamp("2024-04-01"))


def test_jobs_roundtrip_overlap_finish_cancel(tmp_path: Path) -> None:
    import pandas as pd

    from src.market_data.public_archive import jobs as jm

    now = pd.Timestamp("2024-01-01", tz="UTC")
    j1 = jm.BackfillJob(
        "aggTrades-2023-01-01-2023-06-30-20240101T000000Z",
        "aggTrades",
        date(2023, 1, 1),
        date(2023, 6, 30),
        "/census",
        None,
        None,
        now,
    )
    j2 = jm.BackfillJob(
        "aggTrades-2023-07-01-2023-12-31-20240101T000001Z",
        "aggTrades",
        date(2023, 7, 1),
        date(2023, 12, 31),
        "/census",
        10,
        5,
        now,
    )
    jm.enqueue_backfill(tmp_path, j1)
    jm.enqueue_backfill(tmp_path, j2)
    got = jm.pending_jobs(tmp_path)
    assert [j.job_id for j in got] == [j1.job_id, j2.job_id]
    with pytest.raises(ValueError, match=j1.job_id):
        jm.enqueue_backfill(
            tmp_path,
            jm.BackfillJob(
                "aggTrades-2023-06-01-2023-12-31-20240101T000002Z",
                "aggTrades",
                date(2023, 6, 1),
                date(2023, 12, 31),
                "/census",
                None,
                None,
                now,
            ),
        )
    with pytest.raises(FileExistsError):
        jm.enqueue_backfill(tmp_path, j1)
    p = jm.finish_job(tmp_path, j1.job_id, summary={"ok": 1})
    assert p.parent.name == "done"
    assert all(j.job_id != j1.job_id for j in jm.pending_jobs(tmp_path))
    jm.cancel_job(tmp_path, j2.job_id)
    assert jm.pending_jobs(tmp_path) == []
    (tmp_path / "_jobs" / "pending" / "bad.json").write_text("{bad")
    with pytest.raises(DataIntegrityError):
        jm.pending_jobs(tmp_path)


def _metrics_cov_row(sym: str, day: date, sha: str = "a" * 64, rows: int = 1, status: str = "ok"):
    from src.market_data.public_archive import store as sm

    import pandas as pd

    return sm.CoverageRow("metrics", sym, day, status, sha, rows, pd.Timestamp("2024-01-01", tz="UTC"), None)


def _write_metrics_day(root: Path, sym: str, day: date, val: int = 1) -> str:
    import hashlib

    frame = pd.DataFrame({"timestamp": [pd.Timestamp(day, tz="UTC")], "v": [val]})
    from src.market_data.public_archive import store as sm

    sm.merge_partition(root, "metrics", sym, day, frame)
    import pandas as _pd

    raw = _pd.read_parquet(sm.partition_path(root, "metrics", sym, day.strftime("%Y%m"))).to_csv(index=False).encode()
    return hashlib.sha256(raw).hexdigest()


def test_importer_absent_merge_conflict_sealed(tmp_path: Path) -> None:
    from src.market_data.public_archive import store as sm
    from src.market_data.public_archive.importer import import_archive

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    day1, day2 = date(2024, 1, 5), date(2024, 1, 15)
    _write_metrics_day(src, "BTCUSDT", day1)
    sha = _write_metrics_day(src, "BTCUSDT", day2)
    for d in (day1, day2):
        sm.append_coverage(src, sm.CoverageRow("metrics", "BTCUSDT", d, "ok", sha, 1, pd.Timestamp.now(tz="UTC"), None))
    rep = import_archive(src, dst, datasets=["metrics"])
    assert rep.months_copied == 1
    assert rep.days_recorded == 2
    assert (
        sm.partition_path(dst, "metrics", "BTCUSDT", "202401").read_bytes()
        == sm.partition_path(src, "metrics", "BTCUSDT", "202401").read_bytes()
    )
    # present merge: dest has day1..10, source day5..20 -> add 11..20
    src2 = tmp_path / "src2"
    dst2 = tmp_path / "dst2"
    for d in [date(2024, 1, d) for d in range(1, 11)]:
        _write_metrics_day(dst2, "BTCUSDT", d)
        sm.append_coverage(dst2, _metrics_cov_row("BTCUSDT", d))
    for d in [date(2024, 1, d) for d in range(5, 21)]:
        _write_metrics_day(src2, "BTCUSDT", d)
        sm.append_coverage(src2, _metrics_cov_row("BTCUSDT", d))
    rep2 = import_archive(src2, dst2, datasets=["metrics"])
    assert rep2.months_merged == 1
    assert rep2.days_recorded == 10
    # conflict
    src3 = tmp_path / "src3"
    dst3 = tmp_path / "dst3"
    _write_metrics_day(src3, "BTCUSDT", day1)
    _write_metrics_day(dst3, "BTCUSDT", day1)
    sm.append_coverage(src3, _metrics_cov_row("BTCUSDT", day1, sha="a" * 64))
    sm.append_coverage(dst3, _metrics_cov_row("BTCUSDT", day1, sha="b" * 64))
    rep3 = import_archive(src3, dst3, datasets=["metrics"])
    assert rep3.months_skipped_conflict == 1
    # sealed
    dst4 = tmp_path / "dst4"
    src4 = tmp_path / "src4"
    _write_metrics_day(src4, "BTCUSDT", day1)
    sm.append_coverage(src4, _metrics_cov_row("BTCUSDT", day1))
    sm.append_coverage(dst4, _metrics_cov_row("BTCUSDT", day1))
    rep4 = import_archive(src4, dst4, datasets=["metrics"])
    assert rep4.months_skipped_sealed == 1
    assert not sm.partition_path(dst4, "metrics", "BTCUSDT", "202401").exists()
