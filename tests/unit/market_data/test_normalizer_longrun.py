"""Thirty-day-style bound verification: bounded state after a week of simulated operation."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.capture.journal import SegmentWriter
from src.live.lifecycle import ShutdownFlag
from src.market_data.streams.normalizer import NormalizerConfig, run_normalizer

BOOK_BODY = (
    '[{"symbol":"BTCUSDT","bidPrice":"70000.5","bidQty":"1.2","askPrice":"70001.0","askQty":"0.8",'
    '"time":1758576000000},{"symbol":"ETHUSDT","bidPrice":"3000.0","bidQty":"2.0","askPrice":"3000.5",'
    '"askQty":"1.5","time":1758576000000}]'
)
PREMIUM_BODY = (
    '[{"symbol":"BTCUSDT","markPrice":"70000.0","indexPrice":"69999.0","estimatedSettlePrice":"70001.0",'
    '"lastFundingRate":"0.0001","nextFundingTime":1758681600000,"interestRate":"0.0003",'
    '"time":1758576000000}]'
)


def _frame_text(day: int) -> str:
    return (
        '{"e":"forceOrder","E":1758576001000,"o":{"s":"BTCUSDT","S":"SELL","o":"LIMIT","f":"GTC",'
        f'"q":"1.0","p":"7000{day}.0","ap":"70000.0","X":"FILLED","l":"1.0","z":"1.0","T":1758576001000}}'
    )


class _Clock:
    def __init__(self, start: pd.Timestamp) -> None:
        """Start the fake clock at ``start``."""
        self.now = start
        self.sleeps = 0

    def __call__(self) -> pd.Timestamp:
        """Return the current fake time."""
        return self.now

    def sleep(self, delay: float) -> None:
        """Advance the fake clock and count the scheduling round."""
        self.now += pd.Timedelta(seconds=delay)
        self.sleeps += 1


def _write_day(root: Path, day: int) -> None:
    slot = "blue" if day % 2 == 0 else "green"
    grid = f"2026-09-{day:02d}T10:00:00Z"
    moment = pd.Timestamp(grid)
    recv_ns = int(moment.value) + 5_000_000_000
    for stream, body in (("book_ticker", BOOK_BODY), ("premium_index", PREMIUM_BODY)):
        writer = SegmentWriter(root, stream, slot)
        writer.add({"v": 1, "stream": stream, "slot": slot, "kind": "rest", "recv_ns": recv_ns,
                    "grid": grid, "status": 200, "body": body})
        writer.flush()
    writer = SegmentWriter(root, "force_order", slot)
    text = _frame_text(day)
    writer.add({"v": 1, "stream": "force_order", "slot": slot, "kind": "ws_open",
                "recv_ns": recv_ns, "url": "wss://example.invalid"})
    writer.add({"v": 1, "stream": "force_order", "slot": slot, "kind": "frame",
                "recv_ns": recv_ns + 1_000_000_000, "frame": text})
    writer.add({"v": 1, "stream": "force_order", "slot": slot, "kind": "frame",
                "recv_ns": recv_ns + 2_000_000_000, "frame": text})
    writer.add({"v": 1, "stream": "force_order", "slot": slot, "kind": "ws_close",
                "recv_ns": recv_ns + 3_000_000_000, "reason": "server_close"})
    writer.flush()
    noon = datetime(day=day, month=9, year=2026, hour=12, tzinfo=UTC).timestamp()
    for path in (root / "raw" / "hot").rglob("*.jsonl.gz"):
        os.utime(path, (noon, noon))


def _write_status(path: Path, now: pd.Timestamp) -> None:
    started = (now - pd.Timedelta(hours=1)).isoformat()
    finished = (now - pd.Timedelta(minutes=30)).isoformat()
    path.write_text(json.dumps({"started_at": started, "finished_at": finished, "rc": 0}))


def test_week_leaves_only_bounded_state(tmp_path: Path) -> None:
    """Eight simulated days keep hot to current, archives to retention, and zero residue."""
    config = NormalizerConfig(
        normalize_interval_s=30.0,
        retention_interval_s=60.0,
        heartbeat_interval_s=30.0,
        segment_final_grace_s=60.0,
        compaction_grace_s=60.0,
        raw_archive_local_retention_days=2,
        parquet_local_retention_days=2,
    )
    status_path = tmp_path / "last_success.json"
    liq = tmp_path / "liq"
    clock = _Clock(pd.Timestamp(datetime(day=20, month=9, year=2026, hour=1, tzinfo=UTC)))
    for day in range(20, 28):
        _write_day(tmp_path, day)
        clock.now = pd.Timestamp(datetime(day=day + 1, month=9, year=2026, hour=1, tzinfo=UTC))
        _write_status(status_path, clock.now)
        flag = ShutdownFlag()
        target_sleeps = clock.sleeps + 8

        def _sleep(delay: float, _flag: ShutdownFlag = flag, _target: int = target_sleeps) -> None:
            clock.sleep(delay)
            if clock.sleeps >= _target:
                _flag.requested = True

        run_normalizer(
            tmp_path, liq, config, backup_status_path=status_path, shutdown=flag,
            now_fn=clock, sleep_fn=_sleep,
        )
    hot_book = tmp_path / "raw" / "hot" / "book_ticker"
    hot_days = sorted(p.name for p in hot_book.iterdir()) if hot_book.is_dir() else []
    assert set(hot_days) <= {"20260927"}
    for stream in ("book_ticker", "premium_index", "force_order"):
        archived = sorted(p.stem.split(".")[0] for p in (tmp_path / "raw" / "archive" / stream).glob("*.jsonl.xz"))
        assert len(archived) <= 2, (stream, archived)
    assert list(tmp_path.rglob("*.partial")) == []
    assert [p for p in tmp_path.rglob("*.tmp") if ".parquet." in p.name] == []
    for root in ("raw/hot", "raw/archive", "book_ticker", "premium_index"):
        base = tmp_path / root
        if base.is_dir():
            for path in base.rglob("*"):
                if path.is_dir() and path != base:
                    assert list(path.iterdir()), path
    checkpoint = json.loads((tmp_path / "raw" / "normalizer_checkpoint.json").read_text())
    for rel in checkpoint["files"]:
        assert (tmp_path / "raw" / "hot" / rel).exists(), rel
    for dataset in ("book_ticker", "premium_index"):
        for day_dir in (tmp_path / dataset).iterdir():
            assert list(day_dir.iterdir()), day_dir
    heartbeat = json.loads((tmp_path / "recorder_heartbeat.json").read_text())
    assert heartbeat["schema_version"] == 3
    assert heartbeat["retention"]["prune_blocked"] is False
    assert heartbeat["normalizer"]["consecutive_failures"] == 0
