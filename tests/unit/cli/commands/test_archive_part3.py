from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

from src.cli.commands.archive import add_archive_commands
from src.market_data.public_archive.catalog import daily_key

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


class Fake:
    def __init__(self, days):
        self.keys = []
        self.blobs = {}
        self.calls = []
        for d in days:
            stamp = pd.Timestamp(d, tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
            p = _zip(f"{_HDR}\n{stamp},BTCUSDT,1,2,3,4,5,6\n")
            k = daily_key("metrics", "BTCUSDT", d)
            s = k.removeprefix("data/futures/um/")
            h = hashlib.sha256(p).hexdigest()
            self.keys.extend([k, k + ".CHECKSUM"])
            self.blobs[s] = p
            self.blobs[s + ".CHECKSUM"] = f"{h}  {k}\n".encode()

    def list_object_keys(self, *, prefix, timeout=None):
        del timeout
        return [k for k in self.keys if k.startswith(prefix)]

    def fetch_archive_bytes(self, *parts):
        k = "/".join(parts)
        self.calls.append(k)
        return self.blobs.get(k)

    def list_all_symbols(self, **kw):
        del kw
        return ["BTCUSDT"]

    def list_symbols_from_s3_xml_listing(self, **kw):
        del kw
        return ["BTCUSDT"]


def _ns(tmp: Path, argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_archive_commands(p)
    return p.parse_args([*argv, "--root", str(tmp)])


def _census(monkeypatch, start="2020-01-01", periods=2000):
    idx = pd.DatetimeIndex(pd.date_range(start, periods=periods, freq="D", tz="UTC"), tz="UTC")
    close = pd.DataFrame({"BTCUSDT": 100.0}, index=idx, dtype="float64")
    qv = pd.DataFrame({"BTCUSDT": 1e7}, index=idx, dtype="float64")
    import src.market_data.public_archive.mask as mm

    monkeypatch.setattr(mm, "load_daily_census", lambda path: (close, qv))


def test_backfill_add_list_cancel(tmp_path, monkeypatch, capsys):
    _census(monkeypatch)
    ns = _ns(tmp_path, ["backfill", "add", "--dataset", "metrics", "--start", "2024-01-15", "--census-root", "/c"])
    ns.downloader = Fake([date(2024, 1, 15)])
    ns.handler(ns)
    jid = capsys.readouterr().out.strip().splitlines()[-1]
    assert "metrics-2024-01-15" in jid
    ns2 = _ns(tmp_path, ["backfill", "list"])
    ns2.handler(ns2)
    out = capsys.readouterr().out
    assert jid in out
    ns3 = _ns(tmp_path, ["backfill", "cancel", jid])
    ns3.handler(ns3)
    assert "cancelled" in capsys.readouterr().out


def test_backfill_add_short(tmp_path, monkeypatch):
    idx = pd.DatetimeIndex(pd.date_range("2025-08-22", "2026-10-25", freq="D", tz="UTC"), tz="UTC")
    close = pd.DataFrame({"BTCUSDT": 100.0}, index=idx, dtype="float64")
    qv = pd.DataFrame({"BTCUSDT": 1e7}, index=idx, dtype="float64")
    import src.market_data.public_archive.mask as mm

    monkeypatch.setattr(mm, "load_daily_census", lambda path: (close, qv))
    ns = _ns(tmp_path, ["backfill", "add", "--dataset", "aggTrades", "--start", "2023-01-01", "--census-root", "/c"])
    ns.downloader = Fake([])
    import pytest

    with pytest.raises(SystemExit) as e:
        ns.handler(ns)
    assert e.value.code == 2


def test_collect_drain_done_and_partial(tmp_path, monkeypatch, capsys):

    _census(monkeypatch)
    # empty plan -> DONE
    from src.market_data.public_archive import jobs as jm

    now = pd.Timestamp.now(tz="UTC")
    job = jm.BackfillJob(
        "metrics-2024-01-15-2024-01-15-20240101T000000Z",
        "metrics",
        date(2024, 1, 15),
        date(2024, 1, 15),
        "/c",
        None,
        None,
        now,
    )
    # pre-cover the day so plan is empty
    from src.market_data.public_archive import store as sm

    sm.append_coverage(tmp_path, sm.CoverageRow("metrics", "BTCUSDT", date(2024, 1, 15), "ok", "a" * 64, 1, now, None))
    # need partition file to avoid sealed? sealed requires file absent -> sealed! So create file to make it not sealed but ok -> plan empty (no candidates? candidates from mask include day but already ok -> no candidates -> plan empty)
    # Actually with coverage ok but file present, sealed=False, candidate filtered (already ok) -> no plan. Good.
    p = tmp_path / "metrics" / "BTCUSDT" / "202401.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": [pd.Timestamp(date(2024, 1, 15), tz="UTC")]}).to_parquet(p)
    jm.enqueue_backfill(tmp_path, job)
    ns = _ns(
        tmp_path,
        [
            "collect",
            "--dataset",
            "metrics",
            "--symbols",
            "BTCUSDT",
            "--start",
            "2024-01-15",
            "--end",
            "2024-01-15",
            "--execute",
            "--drain-jobs-until",
            "23:59",
        ],
    )
    ns.downloader = Fake([date(2024, 1, 15)])
    ns.handler(ns)
    out = capsys.readouterr().out
    assert "status=DONE" in out
    assert (tmp_path / "_jobs" / "done" / f"{job.job_id}.json").exists()


def test_import_cmd(tmp_path, capsys):
    import json

    from src.market_data.public_archive import store as sm

    src = tmp_path / "src"
    day = date(2024, 1, 5)
    frame = pd.DataFrame({"timestamp": [pd.Timestamp(day, tz="UTC")], "v": [1.0]})
    sm.merge_partition(src, "metrics", "BTCUSDT", day, frame)
    sm.append_coverage(
        src, sm.CoverageRow("metrics", "BTCUSDT", day, "ok", "a" * 64, 1, pd.Timestamp.now(tz="UTC"), None)
    )
    ns = _ns(tmp_path / "dest", ["import", "--from", str(src), "--dataset", "metrics"])
    ns.handler(ns)
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["months_copied"] == 1


def test_edge_branches(tmp_path, monkeypatch, capsys):
    import argparse

    import pytest

    from src.cli.commands import archive as am
    from src.common.errors import DataIntegrityError

    with pytest.raises(ValueError, match="HH:MM"):
        am._parse_hhmm("bad")
    with pytest.raises(ValueError, match="HH:MM"):
        am._parse_hhmm("25:00")
    d = am._drain_deadline("00:00")
    assert d is not None
    # _job_remaining exception path
    import src.market_data.public_archive.store as sm

    monkeypatch.setattr(sm, "load_coverage", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    from src.market_data.public_archive import jobs as jm

    now = pd.Timestamp.now(tz="UTC")
    job = jm.BackfillJob(
        "metrics-2024-01-15-2024-01-16-20240101T000000Z",
        "metrics",
        date(2024, 1, 15),
        date(2024, 1, 16),
        "/c",
        None,
        None,
        now,
    )
    with pytest.raises(OSError, match="boom"):
        am._job_remaining(tmp_path, job)
    # unknown dataset
    ns = argparse.Namespace(
        dataset="bad",
        start="2024-01-15",
        end=None,
        census_root="/c",
        breadth=None,
        preroll_days=None,
        root=str(tmp_path),
    )
    with pytest.raises(SystemExit) as e:
        am._run_backfill_add(ns)
    assert e.value.code == 2
    # lake missing
    import src.market_data.public_archive.mask as mm

    def boom(path):
        raise DataIntegrityError("no lake")

    monkeypatch.setattr(mm, "load_daily_census", boom)
    ns2 = _ns(tmp_path, ["backfill", "add", "--dataset", "metrics", "--start", "2024-01-15", "--census-root", "/c"])
    with pytest.raises(SystemExit) as e:
        ns2.handler(ns2)
    assert e.value.code == 2
    # busy paths
    with sm.archive_lock(tmp_path, wait_s=0, holder="other"):
        for argv in (
            ["backfill", "add", "--dataset", "metrics", "--start", "2024-01-15", "--census-root", "/c"],
            ["backfill", "list"],
            ["import", "--from", str(tmp_path)],
        ):
            nsb = _ns(tmp_path, argv)
            if "backfill" in argv and "add" in argv:
                nsb.downloader = Fake([])
            with pytest.raises(SystemExit) as e:
                nsb.handler(nsb)
            assert e.value.code == 75
        nsb = _ns(tmp_path, ["backfill", "cancel", "nope"])
        with pytest.raises(SystemExit) as e:
            nsb.handler(nsb)
        assert e.value.code == 75
    # cancel not found
    ns3 = _ns(tmp_path, ["backfill", "cancel", "missing-id"])
    with pytest.raises(SystemExit) as e:
        ns3.handler(ns3)
    assert e.value.code == 1
    # import unknown dataset
    ns4 = argparse.Namespace(from_dir=str(tmp_path), dataset=["bad"], root=str(tmp_path), lock_wait_s=0)
    with pytest.raises(SystemExit) as e:
        am._run_import(ns4)
    assert e.value.code == 2
    # drain past deadline prints PARTIAL
    _census(monkeypatch)
    jm.enqueue_backfill(tmp_path, job)
    past = pd.Timestamp("2000-01-01", tz="UTC")
    am._drain_jobs(argparse.Namespace(), tmp_path, Fake([date(2024, 1, 15)]), past, 1)
    assert jm.pending_jobs(tmp_path) == [job]


def test_more_branches(tmp_path, monkeypatch, capsys):
    import argparse

    import pytest

    from src.cli.commands import archive as am
    from src.market_data.public_archive import jobs as jm

    _census(monkeypatch)
    now = pd.Timestamp.now(tz="UTC")
    base = jm.BackfillJob(
        "metrics-2024-01-01-2024-01-31-20240101T000000Z",
        "metrics",
        date(2024, 1, 1),
        date(2024, 1, 31),
        "/c",
        None,
        None,
        now,
    )
    jm.enqueue_backfill(tmp_path, base)
    # overlap via CLI
    ns = _ns(
        tmp_path,
        [
            "backfill",
            "add",
            "--dataset",
            "metrics",
            "--start",
            "2024-01-15",
            "--end",
            "2024-02-01",
            "--census-root",
            "/c",
        ],
    )
    with pytest.raises(SystemExit) as e:
        ns.handler(ns)
    assert e.value.code == 1
    # census detail generic exception
    import src.market_data.public_archive.mask as mm

    idx = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=2000, freq="D", tz="UTC"), tz="UTC")
    close = pd.DataFrame({"BTCUSDT": 100.0}, index=idx, dtype="float64")
    qv = pd.DataFrame({"BTCUSDT": 1e7}, index=idx, dtype="float64")
    monkeypatch.setattr(mm, "load_daily_census", lambda p: (close, qv))
    monkeypatch.setattr(mm, "require_census_coverage", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("weird")))
    ns2 = _ns(tmp_path, ["backfill", "add", "--dataset", "metrics", "--start", "2024-01-15", "--census-root", "/c"])
    with pytest.raises(SystemExit) as e:
        ns2.handler(ns2)
    assert e.value.code == 2
    # drain census fail + listing fail + plan None
    import src.market_data.public_archive.service as smod

    fut = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
    # census fail
    monkeypatch.setattr(mm, "require_census_coverage", lambda *a, **k: (_ for _ in ()).throw(ValueError("short")))
    am._drain_jobs(argparse.Namespace(), tmp_path, Fake([]), fut, 1)
    assert "PARTIAL" in capsys.readouterr().out
    # listing fail
    monkeypatch.setattr(mm, "require_census_coverage", lambda *a, **k: None)

    class NoList(Fake):
        def list_all_symbols(self, **kw):
            del kw
            return []

    am._drain_jobs(argparse.Namespace(), tmp_path, NoList([]), fut, 1)
    # plan None
    monkeypatch.setattr(smod, "plan_collection", lambda *a, **k: None)
    am._drain_jobs(argparse.Namespace(), tmp_path, Fake([]), fut, 1)


def test_drain_exec_and_finish_fail(tmp_path, monkeypatch, capsys):
    import argparse

    from src.cli.commands import archive as am
    from src.market_data.public_archive import jobs as jm

    _census(monkeypatch)
    fut = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
    # execution path with real days
    now = pd.Timestamp.now(tz="UTC")
    job = jm.BackfillJob(
        "metrics-2024-01-15-2024-01-16-20240101T000009Z",
        "metrics",
        date(2024, 1, 15),
        date(2024, 1, 16),
        "/c",
        None,
        None,
        now,
    )
    jm.enqueue_backfill(tmp_path, job)
    am._drain_jobs(argparse.Namespace(), tmp_path, Fake([date(2024, 1, 15), date(2024, 1, 16)]), fut, 1)
    out = capsys.readouterr().out
    assert "status=DONE" in out
    assert f"job={job.job_id}" in out
    assert not jm.pending_jobs(tmp_path)
    # finish fail: pre-create done file so finish raises FileExistsError
    job2 = jm.BackfillJob(
        "metrics-2023-05-01-2023-05-02-20240101T000010Z",
        "metrics",
        date(2023, 5, 1),
        date(2023, 5, 2),
        "/c",
        None,
        None,
        now,
    )
    jm.enqueue_backfill(tmp_path, job2)
    done_dir = tmp_path / "_jobs" / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    (done_dir / f"{job2.job_id}.json").write_text("{}")
    import src.market_data.public_archive.service as smod

    monkeypatch.setattr(smod, "plan_collection", lambda *a, **k: [])
    am._drain_jobs(argparse.Namespace(), tmp_path, Fake([]), fut, 1)
    assert "PARTIAL" in capsys.readouterr().out


def test_corrupt_queue_does_not_fail_incremental(tmp_path, caplog):

    pending = tmp_path / "_jobs/pending"
    pending.mkdir(parents=True)
    (pending / "bad.json").write_text("{")
    ns = _ns(
        tmp_path,
        [
            "collect",
            "--dataset",
            "metrics",
            "--symbols",
            "BTCUSDT",
            "--start",
            "2024-01-15",
            "--end",
            "2024-01-15",
            "--execute",
            "--drain-jobs-until",
            "21:00",
        ],
    )
    ns.downloader = Fake([date(2024, 1, 15)])
    ns.handler(ns)
    assert "pending_jobs_failed" in caplog.text
    assert (pending / "bad.json").exists()


def test_job_workers_zero_rejected(tmp_path):
    import pytest

    ns = _ns(tmp_path, ["collect", "--job-workers", "0"])
    with pytest.raises(ValueError, match="workers"):
        ns.handler(ns)


def test_job_failures_retry_and_lock_covers_incremental_and_job(tmp_path, monkeypatch, capsys):
    import subprocess
    import sys

    from src.cli.commands import archive as am
    from src.market_data.public_archive import jobs, store

    _census(monkeypatch)
    first, last = date(2024, 1, 15), date(2024, 1, 16)
    now = pd.Timestamp.now(tz="UTC")
    job = jobs.BackfillJob(
        "metrics-2024-01-15-2024-01-15-20240101T000000Z", "metrics", first, first, "/c", None, None, now
    )
    jobs.enqueue_backfill(tmp_path, job)
    fake = Fake([first, last])
    original_fetch = fake.fetch_archive_bytes
    order = []

    def fetch(*parts):
        key = "/".join(parts)
        if key.endswith(".zip"):
            busy = subprocess.run(  # noqa: S603
                [
                    sys.executable,
                    "-m",
                    "src.cli.main",
                    "archive",
                    "collect",
                    "--execute",  # noqa: S603
                    "--root",
                    str(tmp_path),
                    "--lock-wait-s",
                    "0",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert busy.returncode == 75
            order.append(key)
        return original_fetch(*parts)

    fake.fetch_archive_bytes = fetch
    bad = daily_key("metrics", "BTCUSDT", first).removeprefix("data/futures/um/") + ".CHECKSUM"
    fake.blobs[bad] = b"invalid checksum"
    ns = _ns(
        tmp_path,
        [
            "collect",
            "--dataset",
            "metrics",
            "--symbols",
            "BTCUSDT",
            "--start",
            str(last),
            "--end",
            str(last),
            "--execute",
            "--drain-jobs-until",
            "21:00",
            "--job-workers",
            "1",
        ],
    )
    ns.downloader = fake
    ns.handler(ns)
    assert str(last) in order[0]
    assert str(first) in order[1]
    assert jobs.pending_jobs(tmp_path) == [job]
    assert store.load_coverage(tmp_path, "metrics")["BTCUSDT", first].status == "failed"
    assert "failed=1" in capsys.readouterr().out
    # A later drain retries the failure and finishes immediately after committing.
    am._drain_jobs(ns, tmp_path, Fake([first]), now + pd.Timedelta(days=1), 1)
    assert jobs.pending_jobs(tmp_path) == []
    assert jobs.done_jobs(tmp_path) == [job]


def test_list_counts_symbol_days_not_calendar_days(tmp_path, monkeypatch, capsys):
    from src.cli.commands import archive as am
    from src.market_data.public_archive import jobs, mask, store

    index = pd.date_range("2023-08-01", "2024-01-02", freq="D", tz="UTC")
    close = pd.DataFrame({"BTCUSDT": 100.0, "ETHUSDT": 100.0}, index=index)
    monkeypatch.setattr(mask, "load_daily_census", lambda path: (close, close * 1e6))
    job = jobs.BackfillJob(
        "metrics-2024-01-01-2024-01-02-20240101T000000Z",
        "metrics",
        date(2024, 1, 1),
        date(2024, 1, 2),
        "/c",
        2,
        0,
        pd.Timestamp.now(tz="UTC"),
    )
    jobs.enqueue_backfill(tmp_path, job)
    store.append_coverage(
        tmp_path, store.CoverageRow("metrics", "BTCUSDT", job.start, "ok", "sha", 1, pd.Timestamp.now(tz="UTC"), None)
    )
    assert am._job_remaining(tmp_path, job) == 3
    ns = _ns(tmp_path, ["backfill", "list"])
    ns.handler(ns)
    assert "remaining=3" in capsys.readouterr().out


def test_drain_deadline_keeps_committed_month_and_resumes(tmp_path, monkeypatch, capsys):
    from src.cli.commands import archive as am
    from src.market_data.public_archive import jobs, service, store

    _census(monkeypatch)
    january, february, march = date(2024, 1, 15), date(2024, 2, 15), date(2024, 3, 15)
    deadline = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1)
    job = jobs.BackfillJob(
        "metrics-2024-01-15-2024-03-15-20240101T000000Z",
        "metrics",
        january,
        march,
        "/c",
        None,
        None,
        deadline - pd.Timedelta(hours=2),
    )
    jobs.enqueue_backfill(tmp_path, job)
    real_execute = service.execute_plan

    def bounded(downloader, plans, **kwargs):
        def clock():
            committed = ("BTCUSDT", january) in store.load_coverage(tmp_path, "metrics")
            return deadline if committed else deadline - pd.Timedelta(seconds=1)

        return real_execute(downloader, plans, now_fn=clock, **kwargs)

    monkeypatch.setattr(service, "execute_plan", bounded)
    am._drain_jobs(argparse.Namespace(), tmp_path, Fake([january, february, march]), deadline, 1)
    assert "remaining=2 status=PARTIAL" in capsys.readouterr().out
    assert jobs.pending_jobs(tmp_path) == [job]
    assert set(store.load_coverage(tmp_path, "metrics")) == {("BTCUSDT", january)}
    before = store.partition_path(tmp_path, "metrics", "BTCUSDT", "202401").read_bytes()
    monkeypatch.setattr(service, "execute_plan", real_execute)
    am._drain_jobs(argparse.Namespace(), tmp_path, Fake([january, february, march]), deadline, 1)
    assert "status=DONE" in capsys.readouterr().out
    assert not jobs.pending_jobs(tmp_path)
    assert store.partition_path(tmp_path, "metrics", "BTCUSDT", "202401").read_bytes() == before


def test_repeated_import_dataset_flags_preserve_all(tmp_path):
    ns = _ns(tmp_path, ["import", "--from", "/source", "--dataset", "metrics", "--dataset", "bookDepth", "aggTrades"])
    assert ns.dataset == ["metrics", "bookDepth", "aggTrades"]
