"""Invariant guards for hot-segment paths, multi-member gzip journaling and write-once files."""

from __future__ import annotations

import gzip
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from src.capture.journal import (
    SegmentWriter,
    hot_segment_path,
    iter_complete_records,
    last_complete_offset,
    repair_torn_tail,
    write_bytes_once,
)


def _ns(year: int, month: int, day: int, hour: int, minute: int = 0, second: int = 0) -> int:
    moment = datetime(year, month, day, hour, minute, second, tzinfo=UTC)
    return int(moment.timestamp() * 1_000_000_000)


def _record(stream: str, slot: str, recv_ns: int, body: str = '{"a":1}') -> dict[str, object]:
    return {
        "v": 1,
        "stream": stream,
        "slot": slot,
        "kind": "rest",
        "recv_ns": recv_ns,
        "grid": "2026-09-26T10:00:00Z",
        "status": 200,
        "body": body,
    }


def test_hot_segment_path_per_hour_and_slot(tmp_path: Path) -> None:
    """Spec 01: one file per (stream, UTC hour, slot)."""
    before = _ns(2026, 9, 26, 10, 59, 59) + 900_000_000
    after = _ns(2026, 9, 26, 11, 0, 0)
    first = hot_segment_path(tmp_path, "force_order", "blue", before)
    second = hot_segment_path(tmp_path, "force_order", "blue", after)
    assert first == tmp_path / "raw" / "hot" / "force_order" / "20260926" / "10.blue.jsonl.gz"
    assert second == tmp_path / "raw" / "hot" / "force_order" / "20260926" / "11.blue.jsonl.gz"
    green = hot_segment_path(tmp_path, "force_order", "green", before)
    assert green != first
    with pytest.raises(ValueError, match="unknown stream"):
        hot_segment_path(tmp_path, "nope", "blue", before)
    with pytest.raises(ValueError, match="unknown slot"):
        hot_segment_path(tmp_path, "force_order", "purple", before)
    with pytest.raises(ValueError, match="non-negative"):
        hot_segment_path(tmp_path, "force_order", "blue", -1)


def test_flush_appends_one_complete_member(tmp_path: Path) -> None:
    """Spec 01: each flush appends exactly one gzip member; records replay in order."""
    writer = SegmentWriter(tmp_path, "book_ticker", "blue")
    base = _ns(2026, 9, 26, 10)
    for index in range(3):
        writer.add(_record("book_ticker", "blue", base + index))
    assert writer.flush() == 3
    for index in range(3, 5):
        writer.add(_record("book_ticker", "blue", base + index))
    assert writer.flush() == 2
    dest = hot_segment_path(tmp_path, "book_ticker", "blue", base)
    with open(dest, "rb") as handle:
        raw = handle.read()
    assert len(raw) > 0
    replayed = list(iter_complete_records(dest, 0))
    assert len(replayed) == 5
    assert [item[0]["recv_ns"] for item in replayed] == [base + index for index in range(5)]
    assert replayed[-1][1] == dest.stat().st_size


def test_resume_from_checkpoint_offset(tmp_path: Path) -> None:
    """Spec 01: iterating from a member end offset yields only later members."""
    writer = SegmentWriter(tmp_path, "book_ticker", "blue")
    base = _ns(2026, 9, 26, 10)
    writer.add(_record("book_ticker", "blue", base))
    writer.flush()
    dest = hot_segment_path(tmp_path, "book_ticker", "blue", base)
    checkpoint = last_complete_offset(dest)
    writer.add(_record("book_ticker", "blue", base + 1))
    writer.flush()
    replayed = list(iter_complete_records(dest, checkpoint))
    assert len(replayed) == 1
    assert replayed[0][0]["recv_ns"] == base + 1


def test_torn_tail_tolerated_by_reader(tmp_path: Path) -> None:
    """Spec 01: a truncated final member is skipped silently; the file is unchanged."""
    writer = SegmentWriter(tmp_path, "force_order", "blue")
    base = _ns(2026, 9, 26, 10)
    writer.add(_record("force_order", "blue", base))
    writer.flush()
    writer.add(_record("force_order", "blue", base + 1))
    writer.flush()
    dest = hot_segment_path(tmp_path, "force_order", "blue", base)
    size_before = dest.stat().st_size
    with open(dest, "r+b") as handle:
        handle.truncate(size_before - 7)
    replayed = list(iter_complete_records(dest, 0))
    assert len(replayed) == 1
    assert replayed[0][0]["recv_ns"] == base
    assert dest.stat().st_size == size_before - 7


def test_last_complete_offset_matches_repair_point(tmp_path: Path) -> None:
    """Spec 01: last_complete_offset equals the point repair_torn_tail truncates to."""
    writer = SegmentWriter(tmp_path, "force_order", "blue")
    base = _ns(2026, 9, 26, 10)
    writer.add(_record("force_order", "blue", base))
    writer.flush()
    writer.add(_record("force_order", "blue", base + 1))
    writer.flush()
    dest = hot_segment_path(tmp_path, "force_order", "blue", base)
    complete = last_complete_offset(dest)
    torn = gzip.compress(b'{"v":9}\n', compresslevel=6)[:10]
    with open(dest, "ab") as handle:
        handle.write(torn)
    assert dest.stat().st_size == complete + len(torn)
    assert last_complete_offset(dest) == complete
    assert repair_torn_tail(dest) == len(torn)
    assert dest.stat().st_size == complete
    assert len(list(iter_complete_records(dest, 0))) == 2


def test_writer_repairs_torn_tail_before_append(tmp_path: Path) -> None:
    """Spec 01: a new writer truncates a torn tail before appending its own member."""
    first = SegmentWriter(tmp_path, "force_order", "blue")
    base = _ns(2026, 9, 26, 10)
    first.add(_record("force_order", "blue", base))
    first.flush()
    dest = hot_segment_path(tmp_path, "force_order", "blue", base)
    size_before = dest.stat().st_size
    torn = gzip.compress(b'{"v":9}\n', compresslevel=6)[:12]
    with open(dest, "ab") as handle:
        handle.write(torn)
    second = SegmentWriter(tmp_path, "force_order", "blue")
    second.add(_record("force_order", "blue", base + 1))
    assert second.flush() == 1
    replayed = list(iter_complete_records(dest, 0))
    assert [item[0]["recv_ns"] for item in replayed] == [base, base + 1]
    assert dest.stat().st_size > size_before


def test_mid_file_corruption_fails_closed(tmp_path: Path) -> None:
    """Spec 01: garbage in the middle of the file raises ValueError."""
    writer = SegmentWriter(tmp_path, "force_order", "blue")
    base = _ns(2026, 9, 26, 10)
    for index in range(3):
        writer.add(_record("force_order", "blue", base + index))
        writer.flush()
    dest = hot_segment_path(tmp_path, "force_order", "blue", base)
    raw = dest.read_bytes()
    first_end = last_complete_offset(dest)
    assert first_end > 0
    corrupted = raw[:first_end] + b"\x00\x01NOT-GZIP" + raw[first_end + 12 :]
    dest.write_bytes(corrupted)
    with pytest.raises(ValueError, match="boundary"):
        list(iter_complete_records(dest, 0))
    with pytest.raises(ValueError, match="boundary"):
        last_complete_offset(dest)


def test_non_boundary_offset_rejected(tmp_path: Path) -> None:
    """Spec 01: a start offset inside a member raises ValueError."""
    writer = SegmentWriter(tmp_path, "book_ticker", "blue")
    base = _ns(2026, 9, 26, 10)
    writer.add(_record("book_ticker", "blue", base))
    writer.flush()
    dest = hot_segment_path(tmp_path, "book_ticker", "blue", base)
    with pytest.raises(ValueError, match="boundary"):
        list(iter_complete_records(dest, 7))


def test_hour_spanning_flush(tmp_path: Path) -> None:
    """Spec 01: records route by the UTC hour of their own recv_ns."""
    writer = SegmentWriter(tmp_path, "force_order", "blue")
    before = _ns(2026, 9, 26, 10, 59, 59)
    after = _ns(2026, 9, 26, 11, 0, 1)
    writer.add(_record("force_order", "blue", before))
    writer.add(_record("force_order", "blue", after))
    assert writer.flush() == 2
    first = hot_segment_path(tmp_path, "force_order", "blue", before)
    second = hot_segment_path(tmp_path, "force_order", "blue", after)
    assert len(list(iter_complete_records(first, 0))) == 1
    assert len(list(iter_complete_records(second, 0))) == 1


def test_flush_failure_retains_buffer(tmp_path: Path) -> None:
    """Spec 01: a failed flush keeps the buffer so the next flush writes every record once."""
    writer = SegmentWriter(tmp_path, "book_ticker", "blue")
    blocker = tmp_path / "raw"
    blocker.mkdir(parents=True)
    (blocker / "hot").write_text("not-a-directory")
    base = _ns(2026, 9, 26, 10)
    writer.add(_record("book_ticker", "blue", base))
    with pytest.raises(OSError, match="Errno"):
        writer.flush()
    assert writer.pending() == 1
    (blocker / "hot").unlink()
    assert writer.flush() == 1
    assert writer.pending() == 0
    dest = hot_segment_path(tmp_path, "book_ticker", "blue", base)
    assert len(list(iter_complete_records(dest, 0))) == 1


def test_body_kept_byte_exact(tmp_path: Path) -> None:
    """Spec 01: unicode and escaped quotes survive the journal round-trip exactly."""
    writer = SegmentWriter(tmp_path, "book_ticker", "blue")
    base = _ns(2026, 9, 26, 10)
    body = '{"s":"BTCUSDT","b":"70000.5","note":"한글 \\"quoted\\" \\u2603"}'
    writer.add(_record("book_ticker", "blue", base, body=body))
    writer.flush()
    dest = hot_segment_path(tmp_path, "book_ticker", "blue", base)
    replayed = list(iter_complete_records(dest, 0))
    assert replayed[0][0]["body"] == body


def test_write_once_reference(tmp_path: Path) -> None:
    """Spec 01: the second create loses; content is the first payload; no partial remains."""
    dest = tmp_path / "reference" / "exchange_info" / "20260926.json.gz"
    assert write_bytes_once(dest, b"first") is True
    assert write_bytes_once(dest, b"second") is False
    assert dest.read_bytes() == b"first"
    assert list(tmp_path.rglob("*.partial")) == []


def test_stale_partial_does_not_block(tmp_path: Path) -> None:
    """Spec 01: a leftover partial older than 60 s is swept so the write succeeds."""
    dest = tmp_path / "reference" / "exchange_info" / "20260926.json.gz"
    dest.parent.mkdir(parents=True)
    partial = dest.with_name(dest.name + ".partial")
    partial.write_bytes(b"stale")
    old = datetime.now(tz=UTC).timestamp() - 120.0
    os.utime(partial, (old, old))
    assert write_bytes_once(dest, b"fresh") is True
    assert dest.read_bytes() == b"fresh"
    assert list(tmp_path.rglob("*.partial")) == []


def test_record_without_recv_ns_rejected(tmp_path: Path) -> None:
    """Writer refuses records that cannot be routed to an hour file."""
    writer = SegmentWriter(tmp_path, "book_ticker", "blue")
    with pytest.raises(ValueError, match="recv_ns"):
        writer.add({"v": 1})
    assert writer.flush() == 0


def _append_member(dest: Path, payload: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "ab") as handle:
        handle.write(gzip.compress(payload, compresslevel=6))


def test_single_byte_file_yields_nothing(tmp_path: Path) -> None:
    """A one-byte file cannot hold a member; the reader stops silently."""
    dest = tmp_path / "seg.jsonl.gz"
    dest.write_bytes(b"\x1f")
    assert list(iter_complete_records(dest, 0)) == []


def test_corrupt_deflate_mid_file_fails_closed(tmp_path: Path) -> None:
    """A magic-valid but corrupt mid-file member raises ValueError."""
    dest = tmp_path / "seg.jsonl.gz"
    good = gzip.compress(b'{"v":1}\n', compresslevel=6)
    broken = bytearray(good)
    broken[15] ^= 0xFF
    with open(dest, "wb") as handle:
        handle.write(bytes(broken))
        handle.write(good)
    with pytest.raises(ValueError, match="corrupt"):
        list(iter_complete_records(dest, 0))


def test_non_object_and_incomplete_records_rejected(tmp_path: Path) -> None:
    """Members with a JSON array or a key-missing object fail closed."""
    dest = tmp_path / "seg.jsonl.gz"
    _append_member(dest, b'[1,2]\n')
    with pytest.raises(ValueError, match="non-JSON-object"):
        list(iter_complete_records(dest, 0))
    dest.unlink()
    _append_member(dest, b'{"v":1}\n')
    with pytest.raises(ValueError, match="lacks"):
        list(iter_complete_records(dest, 0))


def test_start_offset_edge_cases(tmp_path: Path) -> None:
    """Negative offsets fail; missing files read empty; end offsets yield nothing."""
    assert list(iter_complete_records(tmp_path / "missing.jsonl.gz", 0)) == []
    writer = SegmentWriter(tmp_path, "book_ticker", "blue")
    base = _ns(2026, 9, 26, 10)
    writer.add(_record("book_ticker", "blue", base))
    writer.flush()
    dest = hot_segment_path(tmp_path, "book_ticker", "blue", base)
    with pytest.raises(ValueError, match="non-negative"):
        list(iter_complete_records(dest, -1))
    with pytest.raises(ValueError, match="beyond file size"):
        list(iter_complete_records(dest, dest.stat().st_size + 1))
    assert list(iter_complete_records(dest, dest.stat().st_size)) == []


def test_non_utf8_and_non_json_members_rejected(tmp_path: Path) -> None:
    """Complete members with undecodable or non-JSON payloads fail closed."""
    dest = tmp_path / "seg.jsonl.gz"
    _append_member(dest, b"\xff\xfe\x00bad")
    with pytest.raises(ValueError, match="UTF-8"):
        list(iter_complete_records(dest, 0))
    dest.unlink()
    _append_member(dest, b"just text\n")
    with pytest.raises(ValueError, match="JSON lines"):
        list(iter_complete_records(dest, 0))


def test_missing_and_clean_repair_points(tmp_path: Path) -> None:
    """Absent files repair to zero; clean files need no repair."""
    assert last_complete_offset(tmp_path / "missing.jsonl.gz") == 0
    assert repair_torn_tail(tmp_path / "missing.jsonl.gz") == 0
    writer = SegmentWriter(tmp_path, "book_ticker", "blue")
    base = _ns(2026, 9, 26, 10)
    writer.add(_record("book_ticker", "blue", base))
    writer.flush()
    dest = hot_segment_path(tmp_path, "book_ticker", "blue", base)
    assert repair_torn_tail(dest) == 0


def test_write_once_partial_races(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stat races, fresh partials and relink collisions all fail safe without residue."""
    from pathlib import Path as _Path

    dest = tmp_path / "reference" / "x" / "20260926.json.gz"
    dest.parent.mkdir(parents=True)
    partial = dest.with_name(dest.name + ".partial")
    partial.write_bytes(b"stale")
    old = datetime.now(tz=UTC).timestamp() - 120.0
    os.utime(partial, (old, old))
    real_stat = _Path.stat

    def vanishing_stat(self: Path, *args: object, **kwargs: object) -> Any:
        if self == partial:
            raise OSError("gone")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(_Path, "stat", vanishing_stat)
    assert write_bytes_once(dest, b"fresh") is False
    monkeypatch.undo()
    partial.write_bytes(b"stale-fresh")
    assert write_bytes_once(dest, b"fresh") is False
    assert not dest.exists()
    os.utime(partial, (old, old))

    def failing_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self == partial:
            raise OSError("kept")
        _Path.unlink(self, *args, **kwargs)  # type: ignore[call-arg]

    monkeypatch.setattr(_Path, "unlink", failing_unlink)
    assert write_bytes_once(dest, b"fresh") is False
    assert list(tmp_path.rglob("*.partial")) != []
    monkeypatch.undo()
    partial.unlink()


def test_write_once_recreated_partial_loses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial recreated between unlink and open still loses safely."""
    dest = tmp_path / "reference" / "x" / "20260926.json.gz"
    dest.parent.mkdir(parents=True)
    partial = dest.with_name(dest.name + ".partial")
    partial.write_bytes(b"stale")
    old = datetime.now(tz=UTC).timestamp() - 120.0
    os.utime(partial, (old, old))
    real_open = os.open

    def racing_open(path: object, *args: object, **kwargs: object) -> int:
        raise FileExistsError("raced")

    monkeypatch.setattr(os, "open", racing_open)
    assert write_bytes_once(dest, b"fresh") is False
    assert not dest.exists()
    _ = real_open


def test_write_once_write_failure_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed payload write raises and leaves no partial behind."""
    dest = tmp_path / "reference" / "x" / "20260926.json.gz"

    def failing_write(fd: int, data: object) -> int:
        raise OSError("disk gone")

    monkeypatch.setattr(os, "write", failing_write)
    with pytest.raises(OSError, match="disk gone"):
        write_bytes_once(dest, b"fresh")
    assert list(tmp_path.rglob("*.partial")) == []


def test_write_once_concurrent_dest_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A destination appearing before link keeps its bytes; the loser cleans up."""
    dest = tmp_path / "reference" / "x" / "20260926.json.gz"
    real_link = os.link

    def racing_link(src: object, dst: object, *args: object, **kwargs: object) -> None:
        raise FileExistsError("taken")

    monkeypatch.setattr(os, "link", racing_link)
    assert write_bytes_once(dest, b"fresh") is False
    assert list(tmp_path.rglob("*.partial")) == []
    _ = real_link


def test_fsync_dir_failures_are_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unopenable or unsyncable directories never break the journal."""
    from src.capture.journal import _fsync_dir

    monkeypatch.setattr(os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    _fsync_dir(tmp_path)
    monkeypatch.undo()

    def failing_fsync(fd: int) -> None:
        raise OSError("no")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    _fsync_dir(tmp_path)


def test_writer_rejects_unknown_stream_slot_and_bad_recv_ns(tmp_path: Path) -> None:
    """Writer binding and flush routing validate their inputs."""
    with pytest.raises(ValueError, match="unknown stream"):
        SegmentWriter(tmp_path, "nope", "blue")
    with pytest.raises(ValueError, match="unknown slot"):
        SegmentWriter(tmp_path, "book_ticker", "purple")
    writer = SegmentWriter(tmp_path, "book_ticker", "blue")
    writer.add({"v": 1, "recv_ns": "not-an-int"})
    with pytest.raises(ValueError, match="non-negative int"):
        writer.flush()
    assert writer.discard_oldest(0) == 0
    assert writer.discard_oldest(-3) == 0


def test_close_flushes_buffer(tmp_path: Path) -> None:
    """Close persists every buffered record."""
    writer = SegmentWriter(tmp_path, "book_ticker", "blue")
    base = _ns(2026, 9, 26, 10)
    writer.add(_record("book_ticker", "blue", base))
    writer.close()
    assert writer.pending() == 0
    dest = hot_segment_path(tmp_path, "book_ticker", "blue", base)
    assert len(list(iter_complete_records(dest, 0))) == 1


def test_final_byte_corruption_tolerated_as_torn(tmp_path: Path) -> None:
    """A corrupt byte at EOF fails open as a torn tail; mid-file stays fatal."""
    writer = SegmentWriter(tmp_path, "book_ticker", "blue")
    base = _ns(2026, 9, 26, 10)
    writer.add(_record("book_ticker", "blue", base))
    writer.flush()
    dest = hot_segment_path(tmp_path, "book_ticker", "blue", base)
    raw = bytearray(dest.read_bytes())
    raw[-1] ^= 0xFF
    dest.write_bytes(bytes(raw))
    assert list(iter_complete_records(dest, 0)) == []
    assert dest.stat().st_size == len(raw)
    assert repair_torn_tail(dest) == len(raw)
    assert dest.stat().st_size == 0


class _HalfWriteFile:
    """File proxy whose first ``write`` persists half the bytes then raises ENOSPC."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def __enter__(self) -> _HalfWriteFile:
        return self

    def __exit__(self, *exc: object) -> None:
        self._handle.close()

    def write(self, data: bytes) -> int:
        self._handle.write(data[: len(data) // 2])
        self._handle.flush()
        raise OSError(28, "No space left on device")

    def flush(self) -> None:
        self._handle.flush()

    def fileno(self) -> int:
        return int(self._handle.fileno())


def test_enospc_mid_member_retry_repairs_before_append(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A half-written member from a failed flush is truncated before the retry appends, never glued."""
    import builtins

    writer = SegmentWriter(tmp_path, "book_ticker", "blue")
    base = _ns(2026, 9, 26, 10)
    writer.add(_record("book_ticker", "blue", base, body='{"n":1}'))
    assert writer.flush() == 1
    dest = hot_segment_path(tmp_path, "book_ticker", "blue", base)
    writer.add(_record("book_ticker", "blue", base + 1, body='{"n":2}'))
    real_open = builtins.open
    calls = {"n": 0}

    def half_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if Path(file) == dest and "a" in mode and calls["n"] == 0:
            calls["n"] += 1
            return _HalfWriteFile(real_open(file, mode, *args, **kwargs))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", half_open)
    with pytest.raises(OSError, match="No space"):
        writer.flush()
    assert last_complete_offset(dest) < dest.stat().st_size
    assert writer.pending() == 1
    assert writer.flush() == 1
    bodies = [record["body"] for record, _ in iter_complete_records(dest, 0)]
    assert bodies == ['{"n":1}', '{"n":2}']
    assert last_complete_offset(dest) == dest.stat().st_size


def test_partial_multi_hour_flush_never_rewrites_written_hour(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the second hour fails, the retry writes only that hour; the first hour holds each record once."""
    writer = SegmentWriter(tmp_path, "force_order", "blue")
    first_ns = _ns(2026, 9, 26, 10, 59, 59)
    second_ns = _ns(2026, 9, 26, 11, 0, 1)
    writer.add(_record("force_order", "blue", first_ns, body="a"))
    writer.add(_record("force_order", "blue", second_ns, body="b"))
    second = hot_segment_path(tmp_path, "force_order", "blue", second_ns)
    import builtins

    real_open = builtins.open
    failed = {"done": False}

    def failing_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if Path(file) == second and "a" in mode and not failed["done"]:
            failed["done"] = True
            raise OSError(28, "No space left on device")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing_open)
    with pytest.raises(OSError, match="No space"):
        writer.flush()
    assert writer.pending() == 1
    assert writer.flush() == 1
    first = hot_segment_path(tmp_path, "force_order", "blue", first_ns)
    assert [r["body"] for r, _ in iter_complete_records(first, 0)] == ["a"]
    assert [r["body"] for r, _ in iter_complete_records(second, 0)] == ["b"]
