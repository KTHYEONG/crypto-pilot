"""Invariant guards for daily compaction into verified LZMA archives."""

from __future__ import annotations

import gzip
import json
import lzma
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.capture.journal import SegmentWriter
from src.common.errors import DataIntegrityError
from src.market_data.streams.compaction import compact_day, due_compactions, verify_archive
from src.market_data.streams.normalizer import (
    DedupeWindow,
    NormalizerCheckpoint,
    NormalizerConfig,
    normalize_once,
    save_checkpoint,
)

BOOK_ROW = ('{"symbol":"S%04dUSDT","bidPrice":"%d","bidQty":"1","askPrice":"%d","askQty":"1",'
            '"time":1758576000000}')


def _ns(day: int, hour: int = 10, minute: int = 0) -> int:
    return int(datetime(day=day, month=9, year=2026, hour=hour, minute=minute,
                        tzinfo=UTC).timestamp() * 1_000_000_000)


def _now() -> pd.Timestamp:
    return pd.Timestamp(datetime(day=26, month=9, year=2026, hour=12, tzinfo=UTC))


def _rest(stream: str, slot: str, grid: str, recv_ns: int, body: str) -> dict[str, Any]:
    return {"v": 1, "stream": stream, "slot": slot, "kind": "rest", "recv_ns": recv_ns,
            "grid": grid, "status": 200, "body": body}


def _config(**overrides: Any) -> NormalizerConfig:
    values: dict[str, Any] = {"segment_final_grace_s": 60.0, "compaction_grace_s": 60.0}
    values.update(overrides)
    return NormalizerConfig(**values)


def _derive(root: Path, liq: Path, day_end_hour: int = 12) -> NormalizerCheckpoint:
    checkpoint = NormalizerCheckpoint(files={}, coverage={})
    dedupe = DedupeWindow(rest_window_s=7200.0, ws_window_s=7200.0)
    checkpoint, _ = normalize_once(
        root, liq, _config(), checkpoint=checkpoint,
        now=pd.Timestamp(datetime(day=26, month=9, year=2026, hour=day_end_hour, tzinfo=UTC)),
        dedupe=dedupe,
    )
    save_checkpoint(root / "raw" / "normalizer_checkpoint.json", checkpoint)
    return checkpoint


def _age_hot(root: Path, days: int = 2) -> None:
    old = _now().value // 1_000_000_000 - days * 86400
    for path in (root / "raw" / "hot").rglob("*.jsonl.gz"):
        os.utime(path, (old, old))


def _write_day(root: Path, stream: str, day: str, grids: list[str], slots: tuple[str, ...] = ("blue", "green")) -> int:
    total = 0
    for slot in slots:
        writer = SegmentWriter(root, stream, slot)
        for grid in grids:
            moment = pd.Timestamp(grid)
            body = "[" + ",".join(BOOK_ROW % (i, 70000 + i, 70001 + i) for i in range(5)) + "]"
            writer.add(_rest(stream, slot, grid, int(moment.value) + 5_000_000_000, body))
            total += 1
        writer.flush()
    return total


def test_closed_day_compacts_verifies_deletes(tmp_path: Path) -> None:
    """A fully derived FINAL day archives receipt-ordered records, then the hot dir is gone."""
    grids = [f"2026-09-20T{h:02d}:00:00Z" for h in (10, 11)]
    total = _write_day(tmp_path, "book_ticker", "20260920", grids)
    checkpoint = _derive(tmp_path, tmp_path / "liq")
    _age_hot(tmp_path)
    manifest = compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint, config=_config(), now=_now())
    assert manifest is not None
    archive = tmp_path / "raw" / "archive" / "book_ticker" / "20260920.jsonl.xz"
    assert archive.is_file()
    assert (tmp_path / "raw" / "archive" / "book_ticker" / "20260920.manifest.json").is_file()
    lines = lzma.decompress(archive.read_bytes()).decode().splitlines()
    assert len(lines) == manifest.records
    assert manifest.records + manifest.duplicates_dropped == total
    recvs = [json.loads(line)["recv_ns"] for line in lines]
    assert recvs == sorted(recvs)
    assert not (tmp_path / "raw" / "hot" / "book_ticker" / "20260920").exists()


def test_not_eligible_before_grace_undrained_or_open(tmp_path: Path) -> None:
    """Grace, drainage and finality each gate compaction independently."""
    grids = [f"2026-09-20T10:{m:02d}:00Z" for m in (0, 5)]
    _write_day(tmp_path, "book_ticker", "20260920", grids, slots=("blue",))
    early = pd.Timestamp(datetime(day=21, month=9, year=2026, hour=0, minute=0, second=30, tzinfo=UTC))
    assert compact_day(tmp_path, "book_ticker", "20260920", checkpoint=NormalizerCheckpoint(files={}, coverage={}),
                       config=_config(), now=early) is None
    assert (tmp_path / "raw" / "hot" / "book_ticker" / "20260920").exists()
    checkpoint = _derive(tmp_path, tmp_path / "liq")
    assert compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint,
                       config=_config(), now=_now()) is None
    fresh = tmp_path / "raw" / "hot" / "book_ticker" / "20260920" / "10.blue.jsonl.gz"
    assert fresh.stat().st_mtime_ns > 0
    _age_hot(tmp_path)
    stale_checkpoint = NormalizerCheckpoint(files={}, coverage={})
    assert compact_day(tmp_path, "book_ticker", "20260920", checkpoint=stale_checkpoint,
                       config=_config(), now=_now()) is None


def test_interrupted_write_recovers_without_residue(tmp_path: Path) -> None:
    """A leftover archive .partial is swept, then compaction completes cleanly."""
    from src.market_data.streams.retention import sweep_partials

    grids = [f"2026-09-20T10:{m:02d}:00Z" for m in (0, 5)]
    _write_day(tmp_path, "book_ticker", "20260920", grids, slots=("blue",))
    checkpoint = _derive(tmp_path, tmp_path / "liq")
    _age_hot(tmp_path)
    archive = tmp_path / "raw" / "archive" / "book_ticker" / "20260920.jsonl.xz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name(archive.name + ".partial")
    partial.write_bytes(b"interrupted")
    old = _now().value // 1_000_000_000 - 7200
    os.utime(partial, (old, old))
    assert sweep_partials(tmp_path, tmp_path / "liq", now=_now(), max_age_s=3600.0) == 1
    manifest = compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint, config=_config(), now=_now())
    assert manifest is not None
    assert archive.is_file()
    assert not (tmp_path / "raw" / "hot" / "book_ticker" / "20260920").exists()
    assert list((tmp_path / "raw" / "archive").rglob("*.partial")) == []


def test_archive_without_manifest_redone(tmp_path: Path) -> None:
    """An unverifiable archive with hot present is rewritten from the hot day."""
    grids = [f"2026-09-20T10:{m:02d}:00Z" for m in (0, 5)]
    _write_day(tmp_path, "book_ticker", "20260920", grids, slots=("blue",))
    checkpoint = _derive(tmp_path, tmp_path / "liq")
    _age_hot(tmp_path)
    archive = tmp_path / "raw" / "archive" / "book_ticker" / "20260920.jsonl.xz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"garbage-not-lzma")
    manifest = compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint, config=_config(), now=_now())
    assert manifest is not None
    assert (tmp_path / "raw" / "archive" / "book_ticker" / "20260920.manifest.json").is_file()
    assert not (tmp_path / "raw" / "hot" / "book_ticker" / "20260920").exists()
    verify_archive(archive, manifest)


def test_archive_without_manifest_hot_gone_fails_closed(tmp_path: Path) -> None:
    """An orphan archive that cannot be re-derived is never deleted."""
    archive = tmp_path / "raw" / "archive" / "book_ticker" / "20260920.jsonl.xz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"orphan-bytes")
    with pytest.raises(DataIntegrityError, match="without manifest"):
        compact_day(tmp_path, "book_ticker", "20260920",
                    checkpoint=NormalizerCheckpoint(files={}, coverage={}), config=_config(), now=_now())
    assert archive.read_bytes() == b"orphan-bytes"


def test_verified_leftover_hot_completes_idempotently(tmp_path: Path) -> None:
    """A verified archive plus leftover hot dir deletes the dir without touching the archive."""
    grids = [f"2026-09-20T10:{m:02d}:00Z" for m in (0, 5)]
    _write_day(tmp_path, "book_ticker", "20260920", grids, slots=("blue",))
    checkpoint = _derive(tmp_path, tmp_path / "liq")
    _age_hot(tmp_path)
    first = compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint, config=_config(), now=_now())
    assert first is not None
    archive = tmp_path / "raw" / "archive" / "book_ticker" / "20260920.jsonl.xz"
    before = archive.read_bytes()
    _write_day(tmp_path, "book_ticker", "20260920", grids, slots=("blue",))
    second = compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint, config=_config(), now=_now())
    assert second is not None
    assert archive.read_bytes() == before
    assert not (tmp_path / "raw" / "hot" / "book_ticker" / "20260920").exists()


def test_tampered_archive_never_trusted(tmp_path: Path) -> None:
    """Altered bytes fail verification, and the hot dir survives compact_day."""
    grids = [f"2026-09-20T10:{m:02d}:00Z" for m in (0, 5)]
    _write_day(tmp_path, "book_ticker", "20260920", grids, slots=("blue",))
    checkpoint = _derive(tmp_path, tmp_path / "liq")
    _age_hot(tmp_path)
    manifest = compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint, config=_config(), now=_now())
    assert manifest is not None
    archive = tmp_path / "raw" / "archive" / "book_ticker" / "20260920.jsonl.xz"
    raw = bytearray(archive.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    archive.write_bytes(bytes(raw))
    with pytest.raises(DataIntegrityError, match="archive"):
        verify_archive(archive, manifest)
    _write_day(tmp_path, "book_ticker", "20260920", grids, slots=("blue",))
    with pytest.raises(DataIntegrityError, match="archive"):
        compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint, config=_config(), now=_now())
    assert (tmp_path / "raw" / "hot" / "book_ticker" / "20260920").exists()


def test_zero_record_day_still_closes(tmp_path: Path) -> None:
    """A day with only a torn member archives zero records with torn bytes counted."""
    hot_file = tmp_path / "raw" / "hot" / "book_ticker" / "20260920" / "10.blue.jsonl.gz"
    hot_file.parent.mkdir(parents=True, exist_ok=True)
    member = gzip.compress(b'{"v":1}\n', compresslevel=6)
    hot_file.write_bytes(member[: len(member) - 5])
    old = _now().value // 1_000_000_000 - 172800
    os.utime(hot_file, (old, old))
    manifest = compact_day(tmp_path, "book_ticker", "20260920",
                           checkpoint=NormalizerCheckpoint(files={}, coverage={}),
                           config=_config(), now=_now())
    assert manifest is not None
    assert manifest.records == 0
    assert manifest.torn_tail_bytes > 0
    assert not (tmp_path / "raw" / "hot" / "book_ticker" / "20260920").exists()


def test_current_day_never_listed(tmp_path: Path) -> None:
    """due_compactions excludes today even with hot files present."""
    grids = ["2026-09-26T10:00:00Z"]
    _write_day(tmp_path, "book_ticker", "20260926", grids, slots=("blue",))
    due = due_compactions(tmp_path, config=_config(), now=_now())
    assert ("book_ticker", "20260926") not in due
    _write_day(tmp_path, "book_ticker", "20260920", ["2026-09-20T10:00:00Z"], slots=("blue",))
    _age_hot(tmp_path)
    due = due_compactions(tmp_path, config=_config(), now=_now())
    assert ("book_ticker", "20260920") in due


def test_compression_beats_per_member_gzip(tmp_path: Path) -> None:
    """Thirty similar snapshots compress smaller as one LZMA stream than per-member gzip."""
    grids = [f"2026-09-20T{h:02d}:{m:02d}:00Z" for h in (10, 11) for m in range(0, 60, 4)]
    assert len(grids) == 30
    total = _write_day(tmp_path, "book_ticker", "20260920", grids, slots=("blue",))
    assert total == 30
    checkpoint = _derive(tmp_path, tmp_path / "liq")
    _age_hot(tmp_path)
    hot_bytes = sum(p.stat().st_size for p in (tmp_path / "raw" / "hot").rglob("*.jsonl.gz"))
    manifest = compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint, config=_config(), now=_now())
    assert manifest is not None
    archive = tmp_path / "raw" / "archive" / "book_ticker" / "20260920.jsonl.xz"
    assert manifest.compressed_bytes < hot_bytes
    assert archive.stat().st_size == manifest.compressed_bytes


def test_manifest_schema_fails_closed(tmp_path: Path) -> None:
    """Undecodable, non-object and shape-violating manifests never load."""
    from src.market_data.streams.compaction import _load_manifest, _manifest_from_json

    assert _load_manifest(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe")
    assert _load_manifest(bad) is None
    with pytest.raises(DataIntegrityError, match="must be an object"):
        _manifest_from_json(b"[1,2]")
    with pytest.raises(DataIntegrityError, match="violates schema"):
        _manifest_from_json(b'{"day": "20260920"}')


def test_due_and_final_corners(tmp_path: Path) -> None:
    """Non-day dirs are skipped; bad dates and fresh mtimes are never FINAL."""
    from src.market_data.streams.compaction import _is_final, due_compactions

    (tmp_path / "raw" / "hot" / "book_ticker" / "notes.txt").parent.mkdir(parents=True)
    (tmp_path / "raw" / "hot" / "book_ticker" / "notes.txt").write_text("x")
    (tmp_path / "raw" / "hot" / "book_ticker" / "2026-09-20").mkdir()
    assert due_compactions(tmp_path, config=_config(), now=_now()) == []
    assert _is_final(tmp_path / "x", hour=10, day="not-a-day", now_ns=0, grace_s=60.0) is False


def test_iter_day_records_empty_day(tmp_path: Path) -> None:
    """A day directory without segments yields nothing and no torn bytes."""
    from src.market_data.streams.compaction import _iter_day_records

    hot_dir = tmp_path / "raw" / "hot" / "book_ticker" / "20260920"
    hot_dir.mkdir(parents=True)
    tally = [0, 0]
    assert list(_iter_day_records(hot_dir, "book_ticker", tally)) == []
    assert tally == [0, 0]


def test_verify_count_mismatch_fails(tmp_path: Path) -> None:
    """A manifest disagreeing on count or digest is rejected."""
    grids = [f"2026-09-20T10:{m:02d}:00Z" for m in (0, 5)]
    _write_day(tmp_path, "book_ticker", "20260920", grids, slots=("blue",))
    checkpoint = _derive(tmp_path, tmp_path / "liq")
    _age_hot(tmp_path)
    manifest = compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint, config=_config(), now=_now())
    assert manifest is not None
    archive = tmp_path / "raw" / "archive" / "book_ticker" / "20260920.jsonl.xz"
    import dataclasses as _dataclasses

    wrong = _dataclasses.replace(manifest, records=manifest.records + 1)
    with pytest.raises(DataIntegrityError, match="verification failed"):
        verify_archive(archive, wrong)
    wrong_digest = _dataclasses.replace(manifest, sha256="0" * 64)
    with pytest.raises(DataIntegrityError, match="verification failed"):
        verify_archive(archive, wrong_digest)


def test_write_failures_clean_partials(tmp_path: Path, monkeypatch) -> None:
    """Failed archive and manifest writes raise with no partial residue."""
    import os as _os

    grids = [f"2026-09-20T10:{m:02d}:00Z" for m in (0, 5)]
    _write_day(tmp_path, "book_ticker", "20260920", grids, slots=("blue",))
    checkpoint = _derive(tmp_path, tmp_path / "liq")
    _age_hot(tmp_path)
    monkeypatch.setattr(_os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    with pytest.raises(OSError, match="no"):
        compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint, config=_config(), now=_now())
    assert list((tmp_path / "raw" / "archive").rglob("*.partial")) == []


def test_manifest_dir_fsync_tolerated(tmp_path: Path, monkeypatch) -> None:
    """Unopenable or unsyncable archive directories still leave verified output."""
    import os as _os

    grids = [f"2026-09-20T10:{m:02d}:00Z" for m in (0, 5)]
    _write_day(tmp_path, "book_ticker", "20260920", grids, slots=("blue",))
    checkpoint = _derive(tmp_path, tmp_path / "liq")
    _age_hot(tmp_path)
    calls: list[int] = []
    real_open = _os.open

    def selective_open(*args: object, **kwargs: object) -> int:
        calls.append(1)
        if len(calls) == 1:
            raise OSError("no")
        return real_open(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_os, "open", selective_open)
    manifest = compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint, config=_config(), now=_now())
    assert manifest is not None


def test_idempotent_no_hot_returns_manifest_and_empty_none(tmp_path: Path) -> None:
    """Verified archive without hot completes; unknown days return None."""
    grids = [f"2026-09-20T10:{m:02d}:00Z" for m in (0, 5)]
    _write_day(tmp_path, "book_ticker", "20260920", grids, slots=("blue",))
    checkpoint = _derive(tmp_path, tmp_path / "liq")
    _age_hot(tmp_path)
    first = compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint, config=_config(), now=_now())
    assert first is not None
    import shutil as _shutil

    _shutil.rmtree(tmp_path / "raw" / "hot" / "book_ticker" / "20260920", ignore_errors=True)
    second = compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint, config=_config(), now=_now())
    assert second is not None
    assert second.sha256 == first.sha256
    assert compact_day(tmp_path, "book_ticker", "20250101", checkpoint=checkpoint,
                       config=_config(), now=_now()) is None


def test_bad_hour_filename_blocks_compaction(tmp_path: Path) -> None:
    """An unparseable segment name fails the day closed, never the process."""
    hot_file = tmp_path / "raw" / "hot" / "book_ticker" / "20260920" / "xx.blue.jsonl.gz"
    hot_file.parent.mkdir(parents=True, exist_ok=True)
    hot_file.write_bytes(b"junk")
    old = _now().value // 1_000_000_000 - 172800
    os.utime(hot_file, (old, old))
    assert compact_day(tmp_path, "book_ticker", "20260920",
                       checkpoint=NormalizerCheckpoint(files={}, coverage={}),
                       config=_config(), now=_now()) is None
    assert hot_file.exists()


def test_final_corners_and_missing_files(tmp_path: Path) -> None:
    """Hour-open segments, absent files and bad dates are never FINAL."""
    from src.market_data.streams.compaction import _is_final

    assert _is_final(tmp_path / "missing.gz", hour=10, day="20260920",
                     now_ns=_now().value, grace_s=60.0) is False
    recent = tmp_path / "recent.gz"
    recent.write_bytes(b"x")
    assert _is_final(recent, hour=_now().hour, day=_now().strftime("%Y%m%d"),
                     now_ns=_now().value, grace_s=3600.0) is False


def test_iter_day_records_fails_closed_on_vanished_or_foreign(tmp_path: Path, monkeypatch) -> None:
    """A segment vanishing mid-compaction or a non-object member aborts instead of archiving a partial day."""
    import gzip as _gzip
    from pathlib import Path as _Path

    from src.market_data.streams.compaction import _iter_day_records

    hot_dir = tmp_path / "raw" / "hot" / "book_ticker" / "20260920"
    hot_dir.mkdir(parents=True)
    target = hot_dir / "10.blue.jsonl.gz"
    target.write_bytes(_gzip.compress(b"[1,2]\n", compresslevel=6))
    real_stat = _Path.stat

    def _vanishing(self: Path, *args: object, **kwargs: object) -> object:
        if self == target:
            raise OSError("gone")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(_Path, "stat", _vanishing)
    with pytest.raises(OSError, match="gone"):
        list(_iter_day_records(hot_dir, "book_ticker", [0, 0]))
    monkeypatch.undo()
    with pytest.raises(ValueError, match="non-JSON-object"):
        list(_iter_day_records(hot_dir, "book_ticker", [0, 0]))


def test_iter_day_records_holds_one_hour_at_a_time(tmp_path: Path, monkeypatch) -> None:
    """Each source hour is read only after the previous hour was fully emitted (bounded memory)."""
    import src.capture.journal as journal_mod
    from src.market_data.streams.compaction import _iter_day_records

    _write_day(tmp_path, "book_ticker", "20260920", [f"2026-09-20T{h:02d}:00:00Z" for h in (0, 1, 2)])
    hot_dir = tmp_path / "raw" / "hot" / "book_ticker" / "20260920"
    opened: list[str] = []
    real_iter = journal_mod.iter_complete_records

    def _spy(path: Path, start_offset: int = 0):
        opened.append(path.name)
        return real_iter(path, start_offset)

    monkeypatch.setattr(journal_mod, "iter_complete_records", _spy)
    stream = _iter_day_records(hot_dir, "book_ticker", [0, 0])
    first = next(stream)
    assert sorted(opened) == ["00.blue.jsonl.gz", "00.green.jsonl.gz"]
    rest = list(stream)
    recvs = [first["recv_ns"], *(record["recv_ns"] for record in rest)]
    assert recvs == sorted(recvs)
    assert len(opened) == 6
    assert len(recvs) == 3


def test_manifest_dir_fsync_failure_tolerated(tmp_path: Path, monkeypatch) -> None:
    """A manifest directory that cannot fsync still leaves a verified archive."""
    import os as _os

    grids = [f"2026-09-20T10:{m:02d}:00Z" for m in (0, 5)]
    _write_day(tmp_path, "book_ticker", "20260920", grids, slots=("blue",))
    checkpoint = _derive(tmp_path, tmp_path / "liq")
    _age_hot(tmp_path)
    calls: list[int] = []
    real_fsync = _os.fsync

    def _selective(fd: int) -> None:
        calls.append(fd)
        if len(calls) == 3:
            raise OSError("no")
        real_fsync(fd)

    monkeypatch.setattr(_os, "fsync", _selective)
    manifest = compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint, config=_config(), now=_now())
    assert manifest is not None
    assert len(calls) == 3


def test_archive_write_failure_cleans_up(tmp_path: Path, monkeypatch) -> None:
    """A failed archive replace raises with no partial residue."""
    import os as _os

    grids = [f"2026-09-20T10:{m:02d}:00Z" for m in (0, 5)]
    _write_day(tmp_path, "book_ticker", "20260920", grids, slots=("blue",))
    checkpoint = _derive(tmp_path, tmp_path / "liq")
    _age_hot(tmp_path)
    monkeypatch.setattr(_os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    with pytest.raises(OSError, match="no"):
        compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint, config=_config(), now=_now())
    assert list((tmp_path / "raw" / "archive").rglob("*.partial")) == []


def test_manifest_write_failure_cleans_up(tmp_path: Path, monkeypatch) -> None:
    """A failed manifest replace raises with no partial residue."""
    import os as _os

    grids = [f"2026-09-20T10:{m:02d}:00Z" for m in (0, 5)]
    _write_day(tmp_path, "book_ticker", "20260920", grids, slots=("blue",))
    checkpoint = _derive(tmp_path, tmp_path / "liq")
    _age_hot(tmp_path)
    calls: list[int] = []
    real_replace = _os.replace

    def _selective(first: object, second: object) -> None:
        calls.append(1)
        if len(calls) == 2:
            raise OSError("no")
        real_replace(first, second)  # type: ignore[arg-type]

    monkeypatch.setattr(_os, "replace", _selective)
    with pytest.raises(OSError, match="no"):
        compact_day(tmp_path, "book_ticker", "20260920", checkpoint=checkpoint, config=_config(), now=_now())
    assert list((tmp_path / "raw" / "archive").rglob("*.partial")) == []
