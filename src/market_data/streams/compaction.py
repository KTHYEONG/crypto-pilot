"""Daily compaction of closed hot days into verified LZMA archives.

Cross-snapshot redundancy is invisible to per-member gzip (32 KB window) but captured by one
continuous LZMA stream, so archiving each closed day once bounds long-term raw storage without
losing a byte of what was received. The hot day directory is deleted only after the written
archive has been decompressed end to end and matched against its manifest.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import lzma
import os
import shutil
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.common.errors import DataIntegrityError

if TYPE_CHECKING:
    from src.market_data.streams.normalizer import NormalizerCheckpoint, NormalizerConfig

_logger = logging.getLogger(__name__)

ARCHIVE_SCHEMA_VERSION: int = 1
ARCHIVE_STREAMS: tuple[str, str, str] = ("book_ticker", "premium_index", "force_order")


@dataclass(frozen=True, slots=True)
class ArchiveManifest:
    """Verified description of one compacted stream-day.

    Attributes:
        day: ``YYYYMMDD`` (UTC).
        stream: ``book_ticker`` | ``premium_index`` | ``force_order``.
        records: Records in the archive after dedupe.
        duplicates_dropped: Records dropped by cross-slot dedupe.
        torn_tail_bytes: Undecodable trailing bytes discarded across source segments.
        sha256: SHA-256 of the uncompressed JSONL byte stream.
        compressed_bytes: Size of the ``.jsonl.xz`` file.
        source_files: Hot segment paths (relative to ``raw/hot``) with their byte sizes.
        lzma_preset: Preset used.
        created_at: ISO UTC.
    """

    day: str
    stream: str
    records: int
    duplicates_dropped: int
    torn_tail_bytes: int
    sha256: str
    compressed_bytes: int
    source_files: tuple[Mapping[str, Any], ...]
    lzma_preset: int
    created_at: str


def _manifest_to_json(manifest: ArchiveManifest) -> bytes:
    return json.dumps(
        {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "day": manifest.day,
            "stream": manifest.stream,
            "records": manifest.records,
            "duplicates_dropped": manifest.duplicates_dropped,
            "torn_tail_bytes": manifest.torn_tail_bytes,
            "sha256": manifest.sha256,
            "compressed_bytes": manifest.compressed_bytes,
            "source_files": [
                {"path": entry["path"], "bytes": entry["bytes"]} for entry in manifest.source_files
            ],
            "lzma_preset": manifest.lzma_preset,
            "created_at": manifest.created_at,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _manifest_from_json(raw: bytes) -> ArchiveManifest:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise DataIntegrityError(f"archive manifest undecodable: {exc}") from exc
    if not isinstance(data, dict):
        raise DataIntegrityError("archive manifest must be an object")
    try:
        sources = tuple(
            {"path": str(entry["path"]), "bytes": int(entry["bytes"])} for entry in data["source_files"]
        )
        return ArchiveManifest(
            day=str(data["day"]),
            stream=str(data["stream"]),
            records=int(data["records"]),
            duplicates_dropped=int(data["duplicates_dropped"]),
            torn_tail_bytes=int(data["torn_tail_bytes"]),
            sha256=str(data["sha256"]),
            compressed_bytes=int(data["compressed_bytes"]),
            source_files=sources,
            lzma_preset=int(data["lzma_preset"]),
            created_at=str(data["created_at"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DataIntegrityError(f"archive manifest violates schema: {exc}") from exc


def _day_end(day: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:8]}T00:00:00Z") + pd.Timedelta(days=1)


def due_compactions(capture_root: Path, *, config: NormalizerConfig, now: pd.Timestamp) -> list[tuple[str, str]]:
    """``(stream, day)`` pairs whose hot day directory exists and ``now ≥ day_end + compaction_grace_s``."""
    now_utc = pd.Timestamp(now, tz="UTC") if pd.Timestamp(now).tzinfo is None else pd.Timestamp(now).tz_convert("UTC")
    due: list[tuple[str, str]] = []
    for stream in ARCHIVE_STREAMS:
        stream_dir = Path(capture_root) / "raw" / "hot" / stream
        if not stream_dir.is_dir():
            continue
        for day_dir in sorted(stream_dir.iterdir()):
            if not day_dir.is_dir() or len(day_dir.name) != 8 or not day_dir.name.isdigit():
                continue
            if now_utc >= _day_end(day_dir.name) + pd.Timedelta(seconds=config.compaction_grace_s):
                due.append((stream, day_dir.name))
    return due


def _is_final(path: Path, *, hour: int, day: str, now_ns: int, grace_s: float) -> bool:
    """Mirror of the spec 01 FINAL definition: hour ended past grace and file untouched past grace."""
    try:
        moment = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:8]}T{hour:02d}:00:00Z")
    except (TypeError, ValueError):
        return False
    hour_end_ns = int(moment.value) + 3_600_000_000_000
    if now_ns < hour_end_ns + int(grace_s * 1_000_000_000):
        return False
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return False
    return mtime_ns + int(grace_s * 1_000_000_000) <= now_ns


def _hour_of(path: Path) -> int:
    return int(path.name.split(".")[0])


def _iter_day_records(hot_dir: Path, stream: str, torn: list[int]) -> Iterator[dict[str, Any]]:
    """Yield the deduped records of one stream-day in receipt order, holding one source hour at a time.

    Hot segments are partitioned by UTC receipt hour, so ordering each hour's slot files by receipt
    and emitting hours in order yields the day in global receipt order while memory stays bounded by
    one hour of both slots instead of the whole day. Cross-slot duplicates are dropped against
    day-scoped digests (REST by successful grid, WS by exact frame text), which are small next to the
    records themselves. ``torn`` accumulates torn-tail bytes of the source files.
    """
    from src.capture.journal import iter_complete_records, last_complete_offset

    by_hour: dict[int, list[Path]] = {}
    for path in sorted(hot_dir.glob("*.jsonl.gz")):
        by_hour.setdefault(_hour_of(path), []).append(path)
    seen_grids: set[str] = set()
    seen_frames: set[bytes] = set()
    rest_stream = stream in ("book_ticker", "premium_index")
    for hour in sorted(by_hour):
        ordered: list[tuple[int, str, str, dict[str, Any]]] = []
        for path in by_hour[hour]:
            size = path.stat().st_size
            torn[0] += max(0, size - last_complete_offset(path))
            for record, _end in iter_complete_records(path, 0):
                recv = record.get("recv_ns")
                recv_ns = recv if isinstance(recv, int) and not isinstance(recv, bool) else 0
                ordered.append((recv_ns, str(record.get("slot", "")), path.name, record))
        ordered.sort(key=lambda item: (item[0], item[1], item[2]))
        for _recv, _slot, _name, record in ordered:
            if rest_stream:
                grid = record.get("grid")
                if record.get("kind") == "rest" and record.get("status") == 200 and isinstance(grid, str):
                    if grid in seen_grids:
                        torn[1] += 1
                        continue
                    seen_grids.add(grid)
            else:
                frame = record.get("frame")
                if record.get("kind") == "frame" and isinstance(frame, str):
                    digest = hashlib.sha256(frame.encode("utf-8")).digest()
                    if digest in seen_frames:
                        torn[1] += 1
                        continue
                    seen_frames.add(digest)
            yield record
        del ordered


def _serialize_record(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(record, separators=(",", ":"), sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def verify_archive(archive: Path, manifest: ArchiveManifest) -> None:
    """Stream-decompress ``archive`` and require record count and SHA-256 to equal the manifest.

    Raises:
        DataIntegrityError: Mismatch or undecodable archive.
    """
    digest = hashlib.sha256()
    count = 0
    try:
        with lzma.open(archive, "rb") as handle:
            while True:
                chunk = handle.read(1_048_576)
                if not chunk:
                    break
                digest.update(chunk)
                count += chunk.count(b"\n")
    except (OSError, lzma.LZMAError) as exc:
        raise DataIntegrityError(f"archive undecodable: {exc}") from exc
    if count != manifest.records or digest.hexdigest() != manifest.sha256:
        raise DataIntegrityError(
            f"archive verification failed: records={count} want={manifest.records} "
            f"sha256={digest.hexdigest()} want={manifest.sha256}"
        )


def _write_lzma_archive(archive: Path, lines: Iterable[bytes], preset: int) -> tuple[str, int, int]:
    """Stream lines through LZMA into ``archive`` via ``.partial``; return ``(sha256, size, lines)``."""
    partial = archive.with_name(archive.name + ".partial")
    archive.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    compressor = lzma.LZMACompressor(preset=preset)
    count = 0
    try:
        with open(partial, "wb") as handle:
            for line in lines:
                digest.update(line)
                handle.write(compressor.compress(line))
                count += 1
            handle.write(compressor.flush())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, archive)
    except OSError:
        with contextlib.suppress(OSError):
            partial.unlink()
        raise
    return digest.hexdigest(), archive.stat().st_size, count


def _write_manifest(manifest_path: Path, manifest: ArchiveManifest) -> None:
    partial = manifest_path.with_name(manifest_path.name + ".partial")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(partial, "wb") as handle:
            handle.write(_manifest_to_json(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, manifest_path)
        try:
            fd = os.open(manifest_path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)
    except OSError:
        with contextlib.suppress(OSError):
            partial.unlink()
        raise


def _load_manifest(path: Path) -> ArchiveManifest | None:
    try:
        return _manifest_from_json(path.read_bytes())
    except (OSError, DataIntegrityError):
        return None


def compact_day(
    capture_root: Path,
    stream: str,
    day: str,
    *,
    checkpoint: NormalizerCheckpoint,
    config: NormalizerConfig,
    now: pd.Timestamp,
) -> ArchiveManifest | None:
    """Compact one closed stream-day of hot segments into ``raw/archive/<stream>/<day>.jsonl.xz`` and delete the hot day.

    Cross-snapshot redundancy is invisible to per-member gzip (32 KB window) but captured by one
    continuous LZMA stream (about 2.3x smaller for bookTicker). Archiving each closed day once
    therefore bounds long-term raw storage without losing a byte of what was received. The hot day
    directory is deleted only after the written archive has been decompressed end to end and matched
    against its manifest, so a crash or a bad write can never leave the day in neither place.

    Returns:
        The manifest when a compaction or an idempotent completion happened; ``None`` when the day
        is not yet eligible.

    Raises:
        DataIntegrityError: Verification failed, or an archive exists without a manifest while the
            hot day is already gone (it cannot be re-derived, so it is never deleted and is reported
            as ``compaction_failed``).
    """
    from src.capture.journal import last_complete_offset

    capture_root = Path(capture_root)
    now_utc = pd.Timestamp(now, tz="UTC") if pd.Timestamp(now).tzinfo is None else pd.Timestamp(now).tz_convert("UTC")
    now_ns = int(now_utc.value)
    if now_utc < _day_end(day) + pd.Timedelta(seconds=config.compaction_grace_s):
        return None
    hot_dir = capture_root / "raw" / "hot" / stream / day
    archive = capture_root / "raw" / "archive" / stream / f"{day}.jsonl.xz"
    manifest_path = capture_root / "raw" / "archive" / stream / f"{day}.manifest.json"
    manifest = _load_manifest(manifest_path) if manifest_path.is_file() else None
    if not hot_dir.is_dir():
        if manifest is not None and archive.is_file():
            verify_archive(archive, manifest)
            return manifest
        if archive.is_file() and manifest is None:
            raise DataIntegrityError(f"archive without manifest and no hot day: {archive}")
        return None
    if manifest is not None and archive.is_file():
        verify_archive(archive, manifest)
        shutil.rmtree(hot_dir)
        _logger.info("[DATA] stage=compaction stream=%s day=%s status=IDEMPOTENT", stream, day)
        return manifest
    if archive.is_file() and manifest is None:
        with contextlib.suppress(OSError):
            archive.unlink()
    files = sorted(hot_dir.glob("*.jsonl.gz"))
    grace_s = config.segment_final_grace_s
    for path in files:
        try:
            hour = int(path.name.split(".")[0])
        except (ValueError, IndexError):
            return None
        if not _is_final(path, hour=hour, day=day, now_ns=now_ns, grace_s=grace_s):
            return None
        rel = f"{stream}/{day}/{path.name}"
        cursor = checkpoint.files.get(rel)
        complete = last_complete_offset(path)
        if (cursor.offset if cursor is not None else 0) != complete:
            return None
    source_files = tuple(
        {"path": f"{stream}/{day}/{path.name}", "bytes": path.stat().st_size}
        for path in files
        if path.is_file()
    )
    # [torn_tail_bytes, duplicates_dropped]: 제너레이터가 스트리밍 중 누적한다.
    tally = [0, 0]
    lines = (_serialize_record(record) for record in _iter_day_records(hot_dir, stream, tally))
    sha256, compressed, kept_count = _write_lzma_archive(archive, lines, config.archive_lzma_preset)
    torn, dropped = tally
    fresh = ArchiveManifest(
        day=day,
        stream=stream,
        records=kept_count,
        duplicates_dropped=dropped,
        torn_tail_bytes=torn,
        sha256=sha256,
        compressed_bytes=compressed,
        source_files=source_files,
        lzma_preset=config.archive_lzma_preset,
        created_at=now_utc.isoformat(),
    )
    _write_manifest(manifest_path, fresh)
    verify_archive(archive, fresh)
    shutil.rmtree(hot_dir)
    _logger.info(
        "[DATA] stage=compaction stream=%s day=%s status=OK records=%d dropped=%d torn=%d",
        stream, day, kept_count, dropped, torn,
    )
    return fresh
