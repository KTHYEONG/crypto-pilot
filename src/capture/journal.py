"""Append-only multi-member gzip journal for raw-first capture."""

from __future__ import annotations

import contextlib
import gzip
import json
import os
import time
import zlib
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

Stream = Literal["book_ticker", "premium_index", "force_order"]
Slot = Literal["blue", "green"]

STREAMS: Final[tuple[str, ...]] = ("book_ticker", "premium_index", "force_order")
SLOTS: Final[tuple[str, ...]] = ("blue", "green")
RECORD_VERSION: Final[int] = 1
PARTIAL_SUFFIX: Final[str] = ".partial"
_PARTIAL_STALE_AGE_S: Final[float] = 60.0
_GZIP_MAGIC: Final[bytes] = b"\x1f\x8b"
_READ_CHUNK: Final[int] = 65536
_REQUIRED_KEYS: Final[tuple[str, ...]] = ("v", "stream", "slot", "kind", "recv_ns")


def hot_segment_path(capture_root: Path, stream: str, slot: str, recv_ns: int) -> Path:
    """Return ``raw/hot/<stream>/<YYYYMMDD>/<HH>.<slot>.jsonl.gz`` for the UTC hour of ``recv_ns``.

    One file per (stream, UTC hour, slot): the two handover slots never share a file, so an
    overlapping pair of writers cannot race on the same inode.

    Raises:
        ValueError: unknown stream or slot, or ``recv_ns`` negative.
    """
    if stream not in STREAMS:
        raise ValueError(f"unknown stream: {stream!r}")
    if slot not in SLOTS:
        raise ValueError(f"unknown slot: {slot!r}")
    if recv_ns < 0:
        raise ValueError("recv_ns must be non-negative")
    moment = datetime.fromtimestamp(recv_ns / 1_000_000_000, tz=UTC)
    return capture_root / "raw" / "hot" / stream / moment.strftime("%Y%m%d") / f"{moment.strftime('%H')}.{slot}.jsonl.gz"


def _read_member_at(path: Path, start: int, file_size: int) -> tuple[bytes, int] | None:
    """Decompress the single gzip member at ``start``; return ``(payload, end_offset)``.

    Returns ``None`` when the member is torn (truncated by a crash or still being written).
    Raises ``ValueError`` when bytes at ``start`` are not a gzip member (mid-file corruption
    fails closed) or a complete member is corrupt.
    """
    with open(path, "rb") as handle:
        handle.seek(start)
        head = handle.read(2)
        if len(head) < 2:
            return None
        if bytes(head) != _GZIP_MAGIC:
            raise ValueError(f"offset {start} is not a gzip member boundary in {path}")
        handle.seek(start)
        decoder = zlib.decompressobj(31)
        chunks: list[bytes] = []
        fed = 0
        while True:
            data = handle.read(_READ_CHUNK)
            if not data:
                return None
            fed += len(data)
            try:
                chunks.append(decoder.decompress(data))
            except zlib.error as exc:
                error_at = start + fed - len(decoder.unconsumed_tail)
                if error_at < file_size:
                    raise ValueError(f"corrupt gzip member at offset {start} in {path}: {exc}") from exc
                return None
            if decoder.eof or decoder.unused_data:
                end = start + fed - len(decoder.unused_data)
                return b"".join(chunks), end
        raise AssertionError("unreachable")


def _check_record(record: Any, path: Path, offset: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"non-JSON-object record at offset {offset} in {path}")
    for key in _REQUIRED_KEYS:
        if key not in record:
            raise ValueError(f"record at offset {offset} in {path} lacks {key!r}")
    return record


def iter_complete_records(path: Path, start_offset: int = 0) -> Iterator[tuple[dict[str, Any], int]]:
    """Yield ``(record, end_offset)`` for every record in complete gzip members at or after ``start_offset``.

    ``end_offset`` is the byte offset just past the member containing the record. A consumer
    that checkpoints it after processing a member resumes exactly there. A torn or still-being-
    written final member is silently not yielded (the file is not modified). This function is
    the normalizer's only reader of hot segments.

    Raises:
        ValueError: ``start_offset`` is not a member boundary (the stream at that offset is not a
            gzip header), or a complete member decompresses to a non-JSON line or a record
            lacking ``v``/``stream``/``slot``/``kind``/``recv_ns``. Mid-file corruption fails
            closed; only the unterminated tail is tolerated.
    """
    if start_offset < 0:
        raise ValueError("start_offset must be non-negative")
    try:
        file_size = path.stat().st_size
    except OSError:
        return
    if start_offset > file_size:
        raise ValueError(f"start_offset {start_offset} beyond file size {file_size} in {path}")
    if start_offset == file_size:
        return
    offset = start_offset
    while offset < file_size:
        member = _read_member_at(path, offset, file_size)
        if member is None:
            return
        payload, end = member
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"member at offset {offset} in {path} is not UTF-8: {exc}") from exc
        records: list[dict[str, Any]] = []
        for line in text.split("\n"):
            if not line:
                continue
            try:
                records.append(_check_record(json.loads(line), path, offset))
            except json.JSONDecodeError as exc:
                raise ValueError(f"member at offset {offset} in {path} is not JSON lines: {exc}") from exc
        for record in records:
            yield record, end
        offset = end


def last_complete_offset(path: Path) -> int:
    """Return the byte offset just past the last complete gzip member of ``path`` (0 if none).

    Read-only: never modifies the file. The normalizer compares it with the file size to tell a
    fully derived FINAL segment from one that still ends in a torn member, and ``repair_torn_tail``
    truncates to exactly this offset, so both agree on one definition of "complete".

    Raises:
        ValueError: a complete member is followed by bytes that are not a gzip header (mid-file
            corruption fails closed, identical to ``iter_complete_records``).
    """
    try:
        file_size = path.stat().st_size
    except OSError:
        return 0
    offset = 0
    while offset < file_size:
        member = _read_member_at(path, offset, file_size)
        if member is None:
            return offset
        _, offset = member
    return offset


def repair_torn_tail(path: Path) -> int:
    """Truncate ``path`` to the end of its last complete gzip member; return bytes removed.

    Uses ``os.truncate`` plus fsync in place: no copy and no temp file.
    """
    try:
        file_size = path.stat().st_size
    except OSError:
        return 0
    complete = last_complete_offset(path)
    removed = file_size - complete
    if removed <= 0:
        return 0
    with open(path, "r+b") as handle:
        os.ftruncate(handle.fileno(), complete)
        handle.flush()
        os.fsync(handle.fileno())
    return removed


def write_bytes_once(dest: Path, payload: bytes) -> bool:
    """Create ``dest`` exactly once with ``payload``; return False if it already exists.

    Writes ``dest`` + ``PARTIAL_SUFFIX`` with ``O_CREAT|O_EXCL`` in the destination directory,
    fsyncs it, then ``os.link`` to ``dest`` (fails if present) and unlinks the partial. A
    partial from a concurrent slot or a crash never blocks a later attempt: an ``O_EXCL``
    collision on the partial is retried once after unlinking a partial older than
    ``partial_age_s`` (60 s).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return False
    partial = dest.with_name(dest.name + PARTIAL_SUFFIX)
    try:
        fd = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            age_s = time.time() - partial.stat().st_mtime
        except OSError:
            age_s = 0.0
        if age_s <= _PARTIAL_STALE_AGE_S:
            return False
        try:
            partial.unlink()
        except OSError:
            return False
        try:
            fd = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    except OSError:
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            partial.unlink()
        raise
    os.close(fd)
    try:
        os.link(partial, dest)
    except FileExistsError:
        with contextlib.suppress(OSError):
            partial.unlink()
        return False
    with contextlib.suppress(OSError):
        partial.unlink()
    _fsync_dir(dest.parent)
    return True


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class SegmentWriter:
    """Append-only multi-member gzip journal for one (stream, slot), rolling at each UTC hour.

    Each ``flush`` appends exactly one complete gzip member and fsyncs the file (and fsyncs the
    parent directory when the file is created), so a crash can only leave a torn *final*
    member. Before the first append to an existing file, the writer truncates any torn tail
    back to the last complete member (``repair_torn_tail``) so later members are never glued to
    garbage.
    """

    def __init__(self, capture_root: Path, stream: str, slot: str) -> None:
        """Bind the writer to one (stream, slot) journal tree under ``capture_root``."""
        if stream not in STREAMS:
            raise ValueError(f"unknown stream: {stream!r}")
        if slot not in SLOTS:
            raise ValueError(f"unknown slot: {slot!r}")
        self._capture_root = capture_root
        self._stream = stream
        self._slot = slot
        self._buffer: list[dict[str, Any]] = []
        self._repaired: set[Path] = set()

    @property
    def stream(self) -> str:
        """Return the bound stream name."""
        return self._stream

    @property
    def slot(self) -> str:
        """Return the bound slot name."""
        return self._slot

    def add(self, record: Mapping[str, Any]) -> None:
        """Buffer one record (must carry ``recv_ns``); no I/O."""
        if "recv_ns" not in record:
            raise ValueError("record must carry recv_ns")
        self._buffer.append(dict(record))

    def pending(self) -> int:
        """Return the number of buffered, not yet flushed records."""
        return len(self._buffer)

    def discard_oldest(self, count: int) -> int:
        """Drop up to ``count`` oldest buffered records; return the number dropped."""
        if count <= 0:
            return 0
        removed = min(count, len(self._buffer))
        del self._buffer[:removed]
        return removed

    def flush(self) -> int:
        """Write buffered records as one gzip member per destination hour file; return records written.

        Groups are appended one destination at a time and removed from the buffer as soon as their
        member is durable, so a retry after a partial failure never rewrites an hour that already
        succeeded. A destination is marked repaired only after a successful append: when a write or
        fsync fails mid-member, the next flush truncates the torn tail before appending again, so a
        new member is never glued to garbage bytes.

        Raises:
            OSError: write, fsync or truncate failed. Records of the failed and later destinations
                stay buffered for the next flush. A failure after the member reached disk (fsync)
                can make the retry write the same records twice; downstream dedupe by grid label
                and exact frame text absorbs that, whereas losing them would be irrecoverable.
        """
        if not self._buffer:
            return 0
        groups: dict[Path, list[dict[str, Any]]] = {}
        for record in self._buffer:
            recv_ns = record["recv_ns"]
            if not isinstance(recv_ns, int) or recv_ns < 0:
                raise ValueError("record recv_ns must be a non-negative int")
            dest = hot_segment_path(self._capture_root, self._stream, self._slot, recv_ns)
            groups.setdefault(dest, []).append(record)
        written = 0
        for dest, records in groups.items():
            try:
                if dest not in self._repaired and dest.exists():
                    repair_torn_tail(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                created = not dest.exists()
                text = "".join(
                    json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in records
                )
                member = gzip.compress(text.encode("utf-8"), compresslevel=6)
                with open(dest, "ab") as handle:
                    handle.write(member)
                    handle.flush()
                    os.fsync(handle.fileno())
                if created:
                    _fsync_dir(dest.parent)
            except BaseException:
                # 반쯤 쓴 멤버가 남았을 수 있으므로 다음 flush에서 꼬리를 다시 복구하게 한다.
                self._repaired.discard(dest)
                raise
            self._repaired.add(dest)
            done = {id(item) for item in records}
            self._buffer = [item for item in self._buffer if id(item) not in done]
            written += len(records)
        return written

    def close(self) -> None:
        """Flush any buffered records and release the writer."""
        self.flush()
