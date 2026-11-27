from __future__ import annotations

import contextlib
import functools
import hashlib
import http.client
import itertools
import logging
import re
import shutil
import time
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, cast

from src.common.errors import DataIntegrityError
from src.market_data.binance.vision import BinanceVisionDownloader
from src.market_data.mirror.codec import (
    MIRROR_FORMAT_VERSION,
    MirrorObjectWriter,
    SourceLayout,
    UnencodableSourceError,
    encode_source,
    inspect_csv,
    verify_object,
)
from src.market_data.mirror.config import MirrorConfig
from src.market_data.mirror.datasets import MirrorDataset
from src.market_data.mirror.index import Sidecar, SourceRecord, sidecar_path, write_sidecar
from src.market_data.mirror.planning import PartitionPlan, SourceRef

logger = logging.getLogger("MirrorBuild")
GIB_BYTES = 1 << 30
_DIGEST_READ_BYTES = 1 << 20
_HEARTBEAT_SECONDS = 60.0
_MD5_HEX = re.compile(r"[0-9a-f]{32}")
_CHECKSUM_HEX = re.compile(r"[0-9a-f]{64}")
_REJECTED_SOURCE_ERRORS = (UnencodableSourceError, zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError)


def utc_now() -> datetime:
    return datetime.now(UTC)


def local_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def file_digests(path: Path) -> tuple[str, str]:
    """(md5, sha256) hex digests of a file, streamed in fixed chunks."""
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_DIGEST_READ_BYTES):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


class Capacity:
    """Byte accounting for the mirror's Drive usage and local scratch space; every check fails closed."""

    def __init__(
        self,
        config: MirrorConfig,
        *,
        remote_bytes: int,
        drive_free_bytes: int,
        committed_bytes: int,
        outbox_bytes: int,
    ) -> None:
        self._root = config.root
        self._budget = config.drive_budget_gib * GIB_BYTES
        self._drive_floor = config.drive_free_floor_gib * GIB_BYTES
        self._local_floor = config.local_free_floor_gib * GIB_BYTES
        self._base = max(remote_bytes, committed_bytes)
        self._drive_free = drive_free_bytes
        self.outbox_bytes = outbox_bytes

    def reason(self, extra: int) -> str | None:
        if self._base + self.outbox_bytes + extra > self._budget:
            return "BUDGET"
        if self._drive_free - self.outbox_bytes - extra < self._drive_floor:
            return "DRIVE_FREE_FLOOR"
        return self.local_reason(extra)

    def local_reason(self, extra: int) -> str | None:
        return "LOCAL_FREE_FLOOR" if local_free_bytes(self._root) - extra < self._local_floor else None

    def published(self, nbytes: int) -> None:
        self.outbox_bytes += nbytes


@dataclass(slots=True)
class PartitionOutcome:
    built_objects: int = 0
    built_sources: int = 0
    verbatim_sources: int = 0
    failed_sources: int = 0
    published_sources: int = 0
    processed_sources: int = 0
    verbatim_seen: int = 0
    capacity_reason: str | None = None
    deadline_hit: bool = False
    format_drift: bool = False


@dataclass(frozen=True, slots=True)
class _Source:
    ref: SourceRef
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class _Columnar:
    source: _Source
    name: str
    layout: SourceLayout


def _download_path(config: MirrorConfig, dataset: MirrorDataset, ref: SourceRef) -> Path:
    token = hashlib.sha256(f"{ref.key}\n{ref.etag}".encode()).hexdigest()[:16]
    return config.work_dir / "downloads" / dataset.name / f"{token}.zip"


def _verified_sha256(downloader: BinanceVisionDownloader, ref: SourceRef, dest: Path) -> str | None:
    md5_hex, sha_hex = file_digests(dest)
    if _MD5_HEX.fullmatch(ref.etag):
        return sha_hex if md5_hex == ref.etag else None
    checksum = dest.with_name(dest.name + ".CHECKSUM")
    try:
        if not downloader.download_object(key=f"{ref.key}.CHECKSUM", dest=checksum):
            return None
        fields = checksum.read_text(encoding="ascii").split()
    finally:
        checksum.unlink(missing_ok=True)
    if not fields or not _CHECKSUM_HEX.fullmatch(fields[0]):
        return None
    return sha_hex if sha_hex == fields[0] else None


def _fetch_one(downloader: BinanceVisionDownloader, ref: SourceRef, dest: Path) -> str | None:
    """Verified zip sha256, or None when the upstream object is missing or fails its integrity check."""
    try:
        if dest.exists():
            sha = _verified_sha256(downloader, ref, dest)
            if sha is not None:
                return sha
            dest.unlink(missing_ok=True)
        if not downloader.download_object(key=ref.key, dest=dest):
            logger.warning("[DATA] upstream object not found: %s", ref.key)
            return None
        sha = _verified_sha256(downloader, ref, dest)
        if sha is None:
            dest.unlink(missing_ok=True)
            logger.warning("[DATA] integrity check failed: %s", ref.key)
        return sha
    except (OSError, ValueError, http.client.HTTPException) as exc:
        dest.unlink(missing_ok=True)
        logger.warning("[DATA] download failed for %s: %s", ref.key, exc)
        return None


def _fetch_partition(
    downloader: BinanceVisionDownloader,
    config: MirrorConfig,
    dataset: MirrorDataset,
    refs: tuple[SourceRef, ...],
    deadline: datetime | None,
) -> dict[str, str | None] | None:
    """Download every ref through a bounded pool; None when the deadline stops submission.

    At most one ref per worker is in flight, so nothing queued can start after the deadline.
    """
    results: dict[str, str | None] = {}
    remaining = iter(refs)
    in_flight: dict[Future[str | None], str] = {}
    expired = False
    last_beat = time.monotonic()
    with ThreadPoolExecutor(max_workers=config.download_workers) as pool:
        while True:
            for ref in itertools.islice(remaining, config.download_workers - len(in_flight)):
                if deadline is not None and utc_now() >= deadline:
                    expired = True
                    break
                dest = _download_path(config, dataset, ref)
                in_flight[pool.submit(_fetch_one, downloader, ref, dest)] = ref.key
            if expired:
                remaining = iter(())
            if not in_flight:
                break
            done, _ = wait(in_flight, timeout=_HEARTBEAT_SECONDS, return_when=FIRST_COMPLETED)
            for future in done:
                results[in_flight.pop(future)] = future.result()
            if time.monotonic() - last_beat >= _HEARTBEAT_SECONDS:
                last_beat = time.monotonic()
                logger.info("[DATA] download heartbeat %s: %s/%s done", dataset.name, len(results), len(refs))
    return None if expired else results


def _csv_member(path: Path) -> str:
    """The single CSV member of an official zip; anything else is stored verbatim."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".csv")]
    except zipfile.BadZipFile as exc:
        raise UnencodableSourceError("bad zip") from exc
    if len(names) != 1:
        raise UnencodableSourceError("zip does not contain exactly one csv member")
    return names[0]


@contextlib.contextmanager
def _member_stream(path: Path, name: str) -> Iterator[IO[bytes]]:
    with zipfile.ZipFile(path) as archive, archive.open(name) as stream:
        yield stream


def _opener(path: Path, name: str) -> Callable[[], IO[bytes]]:
    return cast(Callable[[], IO[bytes]], functools.partial(_member_stream, path, name))


def _classify(dataset: MirrorDataset, source: _Source) -> _Columnar | None:
    """Columnar candidate, or None when the source must be kept as its official zip."""
    try:
        name = _csv_member(source.path)
        layout = inspect_csv(_opener(source.path, name), dataset)
    except _REJECTED_SOURCE_ERRORS:
        return None
    return _Columnar(source=source, name=name, layout=layout)


def _append_source(
    writer: MirrorObjectWriter,
    dataset: MirrorDataset,
    candidate: _Columnar,
    *,
    source_index: int,
    first_block: int,
    streaming: bool,
) -> int:
    """Append one source's blocks and return their count.

    A multi-source pack materialises the source first, so a rejection never leaves partial rows in the object.
    """
    tables = encode_source(
        _opener(candidate.source.path, candidate.name),
        dataset,
        candidate.layout,
        symbol=candidate.source.ref.symbol,
        source_index=source_index,
        first_block=first_block,
    )
    if streaming:
        count = 0
        for table in tables:
            writer.append(table)
            count += 1
        return count
    materialised = list(tables)
    for table in materialised:
        writer.append(table)
    return len(materialised)


def _object_token(object_sha256: str, refs: Iterable[SourceRef], now: datetime) -> str:
    """Name token covering content, source revisions and build instant.

    A re-uploaded zip with an identical CSV yields identical object bytes; a content-only name would then
    reuse a committed sidecar path, which the immutable uploader rejects forever.
    """
    identity = "\n".join([object_sha256, now.isoformat(), *(f"{ref.key}\t{ref.etag}" for ref in refs)])
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]


def _publish_file(src: Path, dst: Path, sidecar_file: Path) -> None:
    if dst.exists() or sidecar_file.exists():
        raise DataIntegrityError(f"mirror object name collision: {dst.name}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.replace(dst)


def _verbatim_publish(
    config: MirrorConfig,
    dataset: MirrorDataset,
    partition: PartitionPlan,
    source: _Source,
    capacity: Capacity,
    *,
    now: datetime,
    outcome: PartitionOutcome,
) -> str | None:
    """Publish one official zip as its own verbatim object; returns a capacity reason when blocked."""
    md5_hex, _ = file_digests(source.path)
    size = source.path.stat().st_size
    reason = capacity.reason(size)
    if reason is not None:
        return reason
    name = f"{Path(source.ref.key).stem}-{_object_token(source.sha256, (source.ref,), now)}.zip"
    relative = f"verbatim/{dataset.name}/{partition.month}/{name}"
    object_path = f"objects/{relative}"
    sidecar_file = sidecar_path(config.outbox_index_dir, object_path)
    try:
        _publish_file(source.path, config.outbox_objects_dir / relative, sidecar_file)
    except DataIntegrityError:
        logger.error("[DATA] verbatim name collision for %s", source.ref.key, exc_info=True)
        outcome.failed_sources += 1
        return None
    write_sidecar(
        sidecar_file,
        Sidecar(
            format_version=MIRROR_FORMAT_VERSION,
            dataset=dataset.name,
            month=partition.month,
            object_path=object_path,
            object_bytes=size,
            object_md5=md5_hex,
            object_sha256=source.sha256,
            created_at=now.isoformat(),
            sources=(
                SourceRecord(
                    key=source.ref.key,
                    etag=source.ref.etag,
                    zip_sha256=source.sha256,
                    symbol=source.ref.symbol,
                    period=source.ref.period,
                    source_index=0,
                    storage="verbatim",
                    layout=None,
                ),
            ),
        ),
    )
    capacity.published(size)
    outcome.built_objects += 1
    outcome.verbatim_sources += 1
    outcome.published_sources += 1
    logger.info("[DATA] published verbatim %s bytes=%s", name, size)
    return None


def _columnar_publish(
    config: MirrorConfig,
    dataset: MirrorDataset,
    partition: PartitionPlan,
    candidates: list[_Columnar],
    capacity: Capacity,
    *,
    now: datetime,
    outcome: PartitionOutcome,
) -> tuple[list[_Columnar], str | None, bool]:
    """Pack accepted candidates into one object and publish it.

    Returns ``(rejected during encoding, capacity reason, verified)``; ``verified`` False means nothing was published.
    """
    work_path = config.work_dir / "objects" / dataset.name / partition.month / f"{partition.partition}-pending.parquet"
    work_path.parent.mkdir(parents=True, exist_ok=True)
    writer = MirrorObjectWriter(work_path, dataset)
    streaming = len(candidates) == 1
    accepted: list[_Columnar] = []
    rejected: list[_Columnar] = []
    next_block = 0
    for candidate in candidates:
        try:
            blocks = _append_source(
                writer,
                dataset,
                candidate,
                source_index=len(accepted),
                first_block=next_block,
                streaming=streaming,
            )
        except _REJECTED_SOURCE_ERRORS:
            rejected.append(candidate)
            continue
        next_block += blocks
        accepted.append(candidate)
    if not accepted:
        writer.abort()
        return rejected, None, True
    try:
        writer.close()
        verify_object(work_path, dataset, {index: item.layout for index, item in enumerate(accepted)})
    except (DataIntegrityError, ValueError, OSError):
        work_path.unlink(missing_ok=True)
        logger.error("[DATA] object verification failed for %s/%s", dataset.name, partition.partition, exc_info=True)
        outcome.failed_sources += len(accepted)
        return rejected, None, False
    md5_hex, sha_hex = file_digests(work_path)
    size = work_path.stat().st_size
    reason = capacity.reason(size)
    if reason is not None:
        work_path.unlink(missing_ok=True)
        return rejected, reason, True
    name = f"{partition.partition}-{_object_token(sha_hex, (item.source.ref for item in accepted), now)}.parquet"
    relative = f"{dataset.name}/{partition.month}/{name}"
    object_path = f"objects/{relative}"
    sidecar_file = sidecar_path(config.outbox_index_dir, object_path)
    records = tuple(
        SourceRecord(
            key=item.source.ref.key,
            etag=item.source.ref.etag,
            zip_sha256=item.source.sha256,
            symbol=item.source.ref.symbol,
            period=item.source.ref.period,
            source_index=index,
            storage="columnar",
            layout=item.layout,
        )
        for index, item in enumerate(accepted)
    )
    try:
        _publish_file(work_path, config.outbox_objects_dir / relative, sidecar_file)
    except DataIntegrityError:
        work_path.unlink(missing_ok=True)
        logger.error("[DATA] object name collision for %s", name, exc_info=True)
        outcome.failed_sources += len(accepted)
        return rejected, None, False
    write_sidecar(
        sidecar_file,
        Sidecar(
            format_version=MIRROR_FORMAT_VERSION,
            dataset=dataset.name,
            month=partition.month,
            object_path=object_path,
            object_bytes=size,
            object_md5=md5_hex,
            object_sha256=sha_hex,
            created_at=now.isoformat(),
            sources=records,
        ),
    )
    capacity.published(size)
    outcome.built_objects += 1
    outcome.built_sources += len(accepted)
    outcome.published_sources += len(accepted)
    logger.info("[DATA] published %s sources=%s bytes=%s", name, len(accepted), size)
    return rejected, None, True


def _verified_sources(
    config: MirrorConfig,
    dataset: MirrorDataset,
    partition: PartitionPlan,
    fetched: dict[str, str | None],
    outcome: PartitionOutcome,
) -> list[_Source]:
    sources: list[_Source] = []
    for ref in partition.sources:
        sha = fetched[ref.key]
        if sha is None:
            outcome.failed_sources += 1
        else:
            sources.append(_Source(ref=ref, path=_download_path(config, dataset, ref), sha256=sha))
    outcome.processed_sources = len(sources)
    return sources


def _publish_verbatim_all(
    config: MirrorConfig,
    dataset: MirrorDataset,
    partition: PartitionPlan,
    sources: list[_Source],
    capacity: Capacity,
    *,
    now: datetime,
    outcome: PartitionOutcome,
) -> None:
    for source in sources:
        outcome.capacity_reason = _verbatim_publish(
            config, dataset, partition, source, capacity, now=now, outcome=outcome
        )
        if outcome.capacity_reason is not None:
            return


def run_partition(
    downloader: BinanceVisionDownloader,
    config: MirrorConfig,
    dataset: MirrorDataset,
    partition: PartitionPlan,
    capacity: Capacity,
    *,
    now: datetime,
    deadline: datetime | None,
    drifted: Callable[[int, int], bool],
) -> PartitionOutcome:
    """Download, classify, pack and publish one partition.

    Every source ends published, counted failed, or held back: when ``drifted(processed, verbatim)`` is true the
    verbatim fallbacks are not published, so an upstream format change cannot flood the budget with raw zips.
    """
    outcome = PartitionOutcome()
    missing = sum(ref.size for ref in partition.sources if not _download_path(config, dataset, ref).exists())
    outcome.capacity_reason = capacity.local_reason(missing)
    if outcome.capacity_reason is not None:
        return outcome
    fetched = _fetch_partition(downloader, config, dataset, partition.sources, deadline)
    if fetched is None:
        outcome.deadline_hit = True
        return outcome
    columnar: list[_Columnar] = []
    verbatim: list[_Source] = []
    for source in _verified_sources(config, dataset, partition, fetched, outcome):
        classified = _classify(dataset, source)
        if classified is None:
            verbatim.append(source)
        else:
            columnar.append(classified)
    rejected: list[_Columnar] = []
    if columnar:
        rejected, outcome.capacity_reason, verified = _columnar_publish(
            config, dataset, partition, columnar, capacity, now=now, outcome=outcome
        )
        if not verified:
            outcome.failed_sources += len(verbatim) + len(rejected)
            _remove_downloads(config, dataset, partition.sources)
            return outcome
    verbatim.extend(item.source for item in rejected)
    outcome.verbatim_seen = len(verbatim)
    if outcome.capacity_reason is not None:
        return outcome
    outcome.format_drift = drifted(outcome.processed_sources, outcome.verbatim_seen)
    held: set[str] = set()
    if outcome.format_drift:
        held = {source.ref.key for source in verbatim}
        logger.error("[DATA] format drift in %s: %s verbatim sources held back", dataset.name, len(held))
    else:
        _publish_verbatim_all(config, dataset, partition, verbatim, capacity, now=now, outcome=outcome)
    if outcome.capacity_reason is None:
        _remove_downloads(config, dataset, tuple(ref for ref in partition.sources if ref.key not in held))
    return outcome


def _remove_downloads(config: MirrorConfig, dataset: MirrorDataset, refs: tuple[SourceRef, ...]) -> None:
    for ref in refs:
        _download_path(config, dataset, ref).unlink(missing_ok=True)
