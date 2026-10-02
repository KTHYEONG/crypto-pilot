"""Retention and compaction invariants of the raw-to-derived normalizer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.market_data.streams.normalizer import NormalizerCheckpoint, NormalizerConfig

def _now(day: int = 26, hour: int = 12) -> pd.Timestamp:
    return pd.Timestamp(datetime(day=day, month=9, year=2026, hour=hour, tzinfo=UTC))


def _empty() -> NormalizerCheckpoint:
    return NormalizerCheckpoint(files={}, coverage={})


def _write_backup_status(path: Path, *, finished: pd.Timestamp, rc: int = 0) -> None:
    path.write_text(
        json.dumps(
            {
                "started_at": (finished - pd.Timedelta(minutes=1)).isoformat(),
                "finished_at": finished.isoformat(),
                "rc": rc,
            }
        ),
        encoding="utf-8",
    )



def test_retention_stage_failures_are_counted(tmp_path: Path, monkeypatch) -> None:
    """Sweep, compaction-list, compaction-item and prune failures never escape the pass."""
    from src.market_data.streams import normalizer as normalizer_mod
    from src.market_data.streams.normalizer import _run_retention_pass

    checkpoint = _empty()
    compaction_state: dict = {}
    retention_state: dict = {}
    now = _now()
    checkpoint_path = tmp_path / "raw" / "normalizer_checkpoint.json"

    def _boom(*args, **kwargs):
        raise RuntimeError("stage down")

    monkeypatch.setattr(normalizer_mod, "sweep_partials", _boom)
    out = _run_retention_pass(tmp_path, tmp_path / "liq", NormalizerConfig(), checkpoint,
                              tmp_path / "missing.json", compaction_state, retention_state,
                              now, checkpoint_path)
    assert out[0].files == checkpoint.files
    assert out[0].coverage == checkpoint.coverage
    monkeypatch.undo()
    monkeypatch.setattr(normalizer_mod, "due_compactions", _boom)
    _run_retention_pass(tmp_path, tmp_path / "liq", NormalizerConfig(), checkpoint,
                        tmp_path / "missing.json", dict(compaction_state), dict(retention_state),
                        now, checkpoint_path)
    monkeypatch.undo()
    monkeypatch.setattr(normalizer_mod, "prune_backed_up", _boom)
    _run_retention_pass(tmp_path, tmp_path / "liq", NormalizerConfig(), checkpoint,
                        tmp_path / "missing.json", dict(compaction_state), dict(retention_state),
                        now, checkpoint_path)


def test_compaction_item_failures_recorded(tmp_path: Path, monkeypatch) -> None:
    """Item-level compaction errors record last_result error and continue."""
    from src.common.errors import DataIntegrityError
    from src.market_data.streams import normalizer as normalizer_mod
    from src.market_data.streams.normalizer import _run_retention_pass

    def _due(*args, **kwargs):
        return [("book_ticker", "20260920")]

    def _fail(*args, **kwargs):
        raise DataIntegrityError("bad day")

    def _fail_other(*args, **kwargs):
        raise RuntimeError("weird")

    monkeypatch.setattr(normalizer_mod, "due_compactions", _due)
    monkeypatch.setattr(normalizer_mod, "compact_day", _fail)
    _, compaction_state, _ = _run_retention_pass(
        tmp_path, tmp_path / "liq", NormalizerConfig(), _empty(),
        tmp_path / "missing.json", {}, {}, _now(), tmp_path / "raw" / "normalizer_checkpoint.json")
    assert compaction_state["last_result"] == "error"
    assert compaction_state["last_day"] == "20260920"
    monkeypatch.setattr(normalizer_mod, "compact_day", _fail_other)
    _, compaction_state, _ = _run_retention_pass(
        tmp_path, tmp_path / "liq", NormalizerConfig(), _empty(),
        tmp_path / "missing.json", {}, {}, _now(), tmp_path / "raw" / "normalizer_checkpoint.json")
    assert compaction_state["last_result"] == "error"


def test_retention_pass_sweeps_before_pruning(tmp_path: Path) -> None:
    """The retention pass sweeps temps, then prunes, logging the sweep count."""
    from src.market_data.streams.normalizer import _run_retention_pass

    raw = tmp_path / "raw"
    raw.mkdir()
    doomed = raw / "z.partial"
    doomed.write_bytes(b"x")
    old = _now().value // 1_000_000_000 - 7200
    import os as _os

    _os.utime(doomed, (old, old))
    checkpoint, compaction_state, retention_state = _run_retention_pass(
        tmp_path, tmp_path / "liq", NormalizerConfig(), _empty(),
        tmp_path / "missing.json", {}, {}, _now(), tmp_path / "raw" / "normalizer_checkpoint.json")
    assert not doomed.exists()
    assert retention_state["prune_blocked"] is True
    assert retention_state["blocked_reason"] == "status_missing"


def test_retention_due_only_when_block_can_clear(tmp_path: Path) -> None:
    """Between cadence ticks the pass re-runs only if it is blocked and a fresh success status exists."""
    from src.market_data.streams.normalizer import _retention_due

    config = NormalizerConfig()
    status_path = tmp_path / "last_success.json"
    now = _now()
    recent = now - pd.Timedelta(seconds=60)
    blocked: dict[str, Any] = {"prune_blocked": True}
    assert _retention_due(None, blocked, config, status_path, now) is True
    assert _retention_due(recent, blocked, config, status_path, now) is False
    _write_backup_status(status_path, finished=now)
    assert _retention_due(recent, blocked, config, status_path, now) is True
    assert _retention_due(recent, {"prune_blocked": False}, config, status_path, now) is False
    _write_backup_status(status_path, finished=now - pd.Timedelta(hours=100))
    assert _retention_due(recent, blocked, config, status_path, now) is False
    assert _retention_due(now - pd.Timedelta(hours=2), {"prune_blocked": False}, config, status_path, now) is True


def test_run_clears_prune_block_within_one_cycle_of_status_appearing(tmp_path: Path) -> None:
    """The loop re-evaluates a blocked prune every cycle, not once per retention interval."""
    import src.market_data.streams.normalizer as normalizer_mod
    from src.live.lifecycle import ShutdownFlag

    status_path = tmp_path / "last_success.json"
    clock = [_now()]
    flag = ShutdownFlag()
    sleeps = 0
    seen: list[bool] = []

    def _sleep(delay: float) -> None:
        nonlocal sleeps
        sleeps += 1
        clock[0] += pd.Timedelta(seconds=delay)
        heartbeat = tmp_path / "recorder_heartbeat.json"
        if heartbeat.exists():
            seen.append(json.loads(heartbeat.read_text())["retention"]["prune_blocked"])
        if sleeps == 40:
            _write_backup_status(status_path, finished=clock[0])
        if sleeps >= 120:
            flag.requested = True

    normalizer_mod.run_normalizer(
        tmp_path,
        tmp_path / "liq",
        NormalizerConfig(heartbeat_interval_s=30.0),
        backup_status_path=status_path,
        shutdown=flag,
        now_fn=lambda: clock[0],
        sleep_fn=_sleep,
    )
    assert True in seen
    assert seen[-1] is False
    heartbeat = json.loads((tmp_path / "recorder_heartbeat.json").read_text())
    assert heartbeat["retention"]["blocked_since"] is None
    assert (
        normalizer_mod.load_checkpoint(tmp_path / "raw" / "normalizer_checkpoint.json").retention_blocked_since is None
    )


def test_successful_compaction_drops_only_that_days_cursors(tmp_path: Path, monkeypatch) -> None:
    """A compacted day's hot cursors are removed while other days and the block start survive."""
    from src.market_data.streams import normalizer as normalizer_mod
    from src.market_data.streams.normalizer import FileCursor, _run_retention_pass

    monkeypatch.setattr(normalizer_mod, "due_compactions", lambda *a, **k: [("book_ticker", "20260920")])
    monkeypatch.setattr(normalizer_mod, "compact_day", lambda *a, **k: True)
    since = (_now() - pd.Timedelta(hours=1)).isoformat()
    checkpoint = NormalizerCheckpoint(
        files={
            "book_ticker/20260920/00.blue.jsonl.gz": FileCursor(offset=5, final=True),
            "book_ticker/20260921/00.blue.jsonl.gz": FileCursor(offset=7, final=False),
        },
        coverage={},
        retention_blocked_since=since,
    )
    out, compaction_state, retention_state = _run_retention_pass(
        tmp_path, tmp_path / "liq", NormalizerConfig(), checkpoint, tmp_path / "missing.json",
        {}, {"pruned_files_total": 0}, _now(), tmp_path / "raw" / "normalizer_checkpoint.json",
    )
    assert set(out.files) == {"book_ticker/20260921/00.blue.jsonl.gz"}
    assert compaction_state["last_result"] == "ok"
    assert out.retention_blocked_since == since
    assert retention_state["blocked_since"] == since


def test_retention_pass_pulse_precedes_each_unit_in_order(tmp_path: Path, monkeypatch) -> None:
    """Pulse fires exactly once before each due unit, in due order."""
    import src.market_data.streams.normalizer as normalizer_mod
    from src.market_data.streams.normalizer import _run_retention_pass

    units = [("book_ticker", "20260920"), ("premium_index", "20260920"), ("book_ticker", "20260921")]
    monkeypatch.setattr(normalizer_mod, "due_compactions", lambda *a, **k: list(units))
    events: list[tuple] = []

    def _fake_compact(capture_root: Path, stream: str, day: str, **kwargs: object) -> bool:
        events.append(("compact", stream, day))
        return True

    monkeypatch.setattr(normalizer_mod, "compact_day", _fake_compact)

    def _pulse(checkpoint: object, compaction_state: object) -> None:
        events.append(("pulse",))

    _run_retention_pass(
        tmp_path, tmp_path / "liq", NormalizerConfig(), _empty(), tmp_path / "missing.json",
        {}, {"pruned_files_total": 0}, _now(), tmp_path / "raw" / "normalizer_checkpoint.json",
        pulse=_pulse,
    )
    assert events == [
        ("pulse",), ("compact", "book_ticker", "20260920"),
        ("pulse",), ("compact", "premium_index", "20260920"),
        ("pulse",), ("compact", "book_ticker", "20260921"),
    ]


def test_retention_pass_pulse_sees_completed_unit_state(tmp_path: Path, monkeypatch) -> None:
    """A pulse observes compaction state from units already finished in the pass."""
    import src.market_data.streams.normalizer as normalizer_mod
    from src.market_data.streams.normalizer import _run_retention_pass

    units = [("book_ticker", "20260920"), ("book_ticker", "20260921")]
    monkeypatch.setattr(normalizer_mod, "due_compactions", lambda *a, **k: list(units))
    monkeypatch.setattr(normalizer_mod, "compact_day", lambda *a, **k: True)
    seen: list[dict] = []

    def _pulse(checkpoint: object, compaction_state: object) -> None:
        assert isinstance(compaction_state, dict)
        seen.append(dict(compaction_state))

    initial = {"last_day": "input", "last_result": None}
    _run_retention_pass(
        tmp_path, tmp_path / "liq", NormalizerConfig(), _empty(), tmp_path / "missing.json",
        dict(initial), {"pruned_files_total": 0}, _now(),
        tmp_path / "raw" / "normalizer_checkpoint.json", pulse=_pulse,
    )
    assert len(seen) == 2
    assert seen[0] == initial
    assert seen[1]["last_day"] == "20260920"
    assert seen[1]["last_result"] == "ok"


def test_retention_pass_failing_pulse_never_aborts(tmp_path: Path, monkeypatch, caplog) -> None:
    """A raising pulse logs PULSE_FAILED while every unit still compacts."""
    import logging

    import src.market_data.streams.normalizer as normalizer_mod
    from src.market_data.streams.normalizer import _run_retention_pass

    units = [("book_ticker", "20260920"), ("book_ticker", "20260921")]
    monkeypatch.setattr(normalizer_mod, "due_compactions", lambda *a, **k: list(units))
    compacted: list[tuple] = []
    monkeypatch.setattr(
        normalizer_mod, "compact_day",
        lambda *a, **k: compacted.append((a[1] if len(a) > 1 else k.get("stream"), a[2] if len(a) > 2 else k.get("day"))) or True,
    )

    def _boom(checkpoint: object, compaction_state: object) -> None:
        raise RuntimeError("pulse down")

    now = _now()
    ckpt_path = tmp_path / "raw" / "normalizer_checkpoint.json"
    with caplog.at_level(logging.ERROR, logger="src.market_data.streams.normalizer"):
        with_pulse = _run_retention_pass(
            tmp_path, tmp_path / "liq", NormalizerConfig(), _empty(), tmp_path / "missing.json",
            {}, {"pruned_files_total": 0}, now, ckpt_path, pulse=_boom,
        )
    assert len(compacted) == 2
    assert "PULSE_FAILED" in caplog.text
    without_pulse = _run_retention_pass(
        tmp_path, tmp_path / "liq", NormalizerConfig(), _empty(), tmp_path / "missing.json",
        {}, {"pruned_files_total": 0}, now, tmp_path / "raw" / "ckpt2.json",
    )
    assert with_pulse[1] == without_pulse[1]
    assert with_pulse[2]["prune_blocked"] == without_pulse[2]["prune_blocked"]
