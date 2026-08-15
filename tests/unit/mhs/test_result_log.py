"""Run-history ledger contract: append / rotate / prune / latest snapshot."""

from __future__ import annotations

import json
from pathlib import Path

from src.mhs.result_log import (
    MHS_RUN_HISTORY_MAX_SHARDS,
    MHS_RUN_HISTORY_SHARD_MAX_BYTES,
    append_run_history_record,
    mhs_run_history_dir,
)


def test_append_creates_history_dir_active_line_and_latest(tmp_path) -> None:
    record = {"run_id": "abc", "status": "COMPLETE", "perf": {"run_elapsed_seconds": 1.5}}
    history_dir = tmp_path / "history"
    active = append_run_history_record(record, history_dir)

    assert active.name == "active.jsonl"
    assert active == history_dir / "active.jsonl"
    assert history_dir.is_dir()
    lines = active.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == record
    latest = history_dir / "latest.json"
    assert latest.exists()
    assert json.loads(latest.read_text(encoding="utf-8")) == record


def test_append_rotates_active_shard_when_over_budget(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.mhs.result_log.MHS_RUN_HISTORY_SHARD_MAX_BYTES", 1)
    history_dir = tmp_path / "history"

    append_run_history_record({"run_id": "first"}, history_dir)
    append_run_history_record({"run_id": "second"}, history_dir)
    append_run_history_record({"run_id": "third"}, history_dir)

    archives = sorted(history_dir.glob("mhs_run_history_*.jsonl"))
    assert len(archives) == 2
    assert json.loads(archives[0].read_text(encoding="utf-8").splitlines()[0])["run_id"] == "first"
    assert json.loads(archives[1].read_text(encoding="utf-8").splitlines()[0])["run_id"] == "second"

    active_lines = (history_dir / "active.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(active_lines) == 1
    assert json.loads(active_lines[0])["run_id"] == "third"


def test_append_prunes_oldest_archive_at_retention_cap(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.mhs.result_log.MHS_RUN_HISTORY_SHARD_MAX_BYTES", 1)
    monkeypatch.setattr("src.mhs.result_log.MHS_RUN_HISTORY_MAX_SHARDS", 3)
    history_dir = tmp_path / "history"
    history_dir.mkdir(parents=True)
    (history_dir / "mhs_run_history_100.jsonl").write_text('{"run_id": "oldest"}\n', encoding="utf-8")
    (history_dir / "mhs_run_history_200.jsonl").write_text('{"run_id": "mid"}\n', encoding="utf-8")
    (history_dir / "mhs_run_history_300.jsonl").write_text('{"run_id": "recent"}\n', encoding="utf-8")
    (history_dir / "active.jsonl").write_text('{"run_id": "pre-rotation"}\n', encoding="utf-8")

    append_run_history_record({"run_id": "trigger"}, history_dir)

    archives = sorted(history_dir.glob("mhs_run_history_*.jsonl"))
    assert len(archives) == 3
    names = [p.name for p in archives]
    assert "mhs_run_history_100.jsonl" not in names
    assert "mhs_run_history_200.jsonl" in names
    assert "mhs_run_history_300.jsonl" in names
    active_lines = (history_dir / "active.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(active_lines) == 1
    assert json.loads(active_lines[0])["run_id"] == "trigger"


def test_append_no_rotation_when_under_budget(tmp_path) -> None:
    history_dir = tmp_path / "history"
    append_run_history_record({"run_id": "a"}, history_dir)
    append_run_history_record({"run_id": "b"}, history_dir)
    assert not list(history_dir.glob("mhs_run_history_*.jsonl"))
    lines = (history_dir / "active.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["run_id"] == "a"
    assert json.loads(lines[1])["run_id"] == "b"


def test_latest_snapshot_tracks_most_recent_record(tmp_path) -> None:
    history_dir = tmp_path / "history"
    append_run_history_record({"run_id": "first"}, history_dir)
    append_run_history_record({"run_id": "second"}, history_dir)
    latest = json.loads((history_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest["run_id"] == "second"


def test_mhs_run_history_dir_derives_from_target_parent() -> None:
    assert mhs_run_history_dir(Path("results/report.json")) == Path("results/mhs_run_history")


def test_shard_constants_are_fixed_bounds() -> None:
    assert MHS_RUN_HISTORY_SHARD_MAX_BYTES == 262144
    assert MHS_RUN_HISTORY_MAX_SHARDS == 12
