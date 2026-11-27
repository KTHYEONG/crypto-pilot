from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.errors import DataIntegrityError
from src.market_data.mirror.codec import SourceLayout, iter_source_csv
from src.market_data.mirror.datasets import MIRROR_DATASET_SPECS, MirrorDataset
from src.market_data.mirror.index import Sidecar, read_sidecar, sidecar_path
from src.market_data.mirror.packing import file_digests
from src.market_data.mirror.status import update_status

_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class RemoteEntry:
    path: str
    size: int
    md5: str | None


@dataclass(frozen=True, slots=True)
class ConfirmReport:
    confirmed: int
    unconfirmed: int
    committed_bytes: int


@dataclass(frozen=True, slots=True)
class RestoredSource:
    path: Path
    sha256: str


def parse_remote_listing(text: str) -> tuple[RemoteEntry, ...]:
    """Entries of ``rclone lsjson -R --hash --files-only``; any malformed item fails the whole listing."""
    try:
        payload: Any = json.loads(text)
    except ValueError as exc:
        raise DataIntegrityError(f"malformed remote listing: {exc}") from exc
    if not isinstance(payload, list):
        raise DataIntegrityError("malformed remote listing: top level is not a list")
    entries: list[RemoteEntry] = []
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("Path"), str) or not isinstance(item.get("Size"), int):
            raise DataIntegrityError("malformed remote listing: entry without Path or Size")
        hashes = item.get("Hashes")
        md5 = hashes.get("md5") if isinstance(hashes, dict) else None
        entries.append(RemoteEntry(path=item["Path"], size=item["Size"], md5=md5 if isinstance(md5, str) else None))
    return tuple(entries)


def _md5_of(path: Path) -> str:
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            md5.update(chunk)
    return md5.hexdigest()


def _object_matches(entry: RemoteEntry | None, sidecar: Sidecar) -> bool:
    return entry is not None and entry.size == sidecar.object_bytes and entry.md5 == sidecar.object_md5


def _commit_sidecar(root: Path, sidecar_file: Path, relative: Path) -> None:
    target = root / "index" / relative
    if target.exists():
        if _md5_of(target) != _md5_of(sidecar_file):
            raise DataIntegrityError(f"committed sidecar differs from outbox copy: {relative}")
        sidecar_file.unlink()
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(sidecar_file, target)


def confirm_uploads(root: Path, remote: Sequence[RemoteEntry], *, now: datetime) -> ConfirmReport:
    """Commit outbox pairs whose object and sidecar both appear remotely with matching size and md5."""
    by_path = {entry.path: entry for entry in remote}
    outbox_index = root / "outbox" / "index"
    confirmed = unconfirmed = committed_bytes = 0
    for sidecar_file in sorted(outbox_index.rglob("*.json.zst")):
        sidecar = read_sidecar(sidecar_file)
        relative = sidecar_file.relative_to(outbox_index)
        sidecar_remote = by_path.get(f"index/{relative.as_posix()}")
        local_md5 = _md5_of(sidecar_file)
        object_remote = by_path.get(sidecar.object_path)
        if not _object_matches(object_remote, sidecar) or sidecar_remote is None or sidecar_remote.md5 != local_md5:
            unconfirmed += 1
            continue
        if sidecar_remote.size != sidecar_file.stat().st_size:
            unconfirmed += 1
            continue
        _commit_sidecar(root, sidecar_file, relative)
        (root / "outbox" / sidecar.object_path).unlink(missing_ok=True)
        confirmed += 1
        committed_bytes += sidecar.object_bytes
    _drop_objects_already_committed(root, by_path)
    if confirmed:
        update_status(root / "_status", lambda data: _stamp_confirm(data, now))
    return ConfirmReport(confirmed=confirmed, unconfirmed=unconfirmed, committed_bytes=committed_bytes)


def _stamp_confirm(data: dict[str, Any], now: datetime) -> None:
    data["last_confirmed_at"] = now.isoformat()


def _drop_objects_already_committed(root: Path, by_path: dict[str, RemoteEntry]) -> None:
    """Repair a crash between sidecar move and object delete: remove the leftover object once remote matches."""
    objects = root / "outbox" / "objects"
    if not objects.is_dir():
        return
    for obj in objects.rglob("*"):
        if not obj.is_file():
            continue
        object_path = "objects/" + obj.relative_to(objects).as_posix()
        if sidecar_path(root / "outbox" / "index", object_path).exists():
            continue
        committed = sidecar_path(root / "index", object_path)
        if not committed.exists():
            continue
        sidecar = read_sidecar(committed)
        if _object_matches(by_path.get(object_path), sidecar):
            obj.unlink()


def restore_source(
    sidecar_file: Path, object_file: Path, key: str, out: Path, *, etag: str | None = None
) -> RestoredSource:
    """Rebuild one source (columnar CSV or verbatim zip) from its object, verifying every checksum before replacing ``out``."""
    sidecar = read_sidecar(sidecar_file)
    _, object_sha = file_digests(object_file)
    if object_sha != sidecar.object_sha256:
        raise DataIntegrityError(f"object sha256 differs from sidecar: {object_file}")
    matches = [record for record in sidecar.sources if record.key == key and (etag is None or record.etag == etag)]
    if not matches:
        raise ValueError(f"key {key} not found in sidecar")
    if len({record.etag for record in matches}) > 1:
        raise ValueError(f"key {key} has several ETags; pass etag")
    record = matches[0]
    tmp = out.with_name(out.name + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if record.storage == "verbatim":
            digest = _copy_hashed(object_file, tmp)
            if digest != record.zip_sha256:
                raise DataIntegrityError(f"verbatim zip sha256 differs from sidecar: {key}")
            sha = digest
        else:
            assert record.layout is not None
            layout = record.layout
            sha, size = _write_csv(object_file, MIRROR_DATASET_SPECS[sidecar.dataset], record.source_index, layout, tmp)
            if sha != layout.csv_sha256 or size != layout.csv_bytes:
                raise DataIntegrityError(f"rebuilt csv differs from sidecar: {key}")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return RestoredSource(path=out, sha256=sha)


def _copy_hashed(src: Path, dst: Path) -> str:
    sha = hashlib.sha256()
    with src.open("rb") as reader, dst.open("wb") as writer:
        while chunk := reader.read(_CHUNK_BYTES):
            sha.update(chunk)
            writer.write(chunk)
    return sha.hexdigest()


def _write_csv(
    object_file: Path, dataset: MirrorDataset, source_index: int, layout: SourceLayout, dst: Path
) -> tuple[str, int]:
    sha = hashlib.sha256()
    size = 0
    with dst.open("wb") as writer:
        for chunk in iter_source_csv(object_file, dataset, source_index, layout):
            sha.update(chunk)
            size += len(chunk)
            writer.write(chunk)
    return sha.hexdigest(), size
