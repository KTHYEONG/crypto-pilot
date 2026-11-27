from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa

from src.common.errors import DataIntegrityError
from src.market_data.mirror.codec import SourceLayout

SIDECAR_SUFFIX = ".json.zst"
_STORAGE_KINDS = ("columnar", "verbatim")


@dataclass(frozen=True, slots=True)
class SourceRecord:
    key: str
    etag: str
    zip_sha256: str
    symbol: str
    period: str
    source_index: int
    storage: Literal["columnar", "verbatim"]
    layout: SourceLayout | None


@dataclass(frozen=True, slots=True)
class Sidecar:
    format_version: int
    dataset: str
    month: str
    object_path: str
    object_bytes: int
    object_md5: str
    object_sha256: str
    created_at: str
    sources: tuple[SourceRecord, ...]


def sidecar_path(index_root: Path, object_path: str) -> Path:
    """Sidecar file under ``index_root`` for an object path such as ``objects/<dataset>/<month>/<name>.parquet``."""
    if not object_path.startswith("objects/"):
        raise DataIntegrityError(f"object path must start with objects/: {object_path}")
    relative = Path(object_path[len("objects/") :])
    return index_root / relative.parent / (relative.stem + SIDECAR_SUFFIX)


def _record_to_json(record: SourceRecord) -> dict[str, Any]:
    layout = record.layout
    return {
        "key": record.key,
        "etag": record.etag,
        "zip_sha256": record.zip_sha256,
        "symbol": record.symbol,
        "period": record.period,
        "source_index": record.source_index,
        "storage": record.storage,
        "header_line": layout.header_line if layout else None,
        "trailing_newline": layout.trailing_newline if layout else None,
        "csv_sha256": layout.csv_sha256 if layout else None,
        "csv_bytes": layout.csv_bytes if layout else None,
    }


def _malformed(path: Path, reason: str) -> DataIntegrityError:
    return DataIntegrityError(f"malformed sidecar {path}: {reason}")


def _expect(payload: Mapping[str, Any], path: Path, field: str, kind: type | tuple[type, ...]) -> Any:
    value = payload.get(field)
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is int):
        raise _malformed(path, f"{field} missing or wrong type")
    return value


def _record_from_json(raw: Any, path: Path) -> SourceRecord:
    if not isinstance(raw, dict):
        raise _malformed(path, "source record is not an object")
    storage = _expect(raw, path, "storage", str)
    if storage not in _STORAGE_KINDS:
        raise _malformed(path, f"unknown storage {storage}")
    layout: SourceLayout | None = None
    if storage == "columnar":
        header_line = raw.get("header_line")
        if header_line is not None and not isinstance(header_line, str):
            raise _malformed(path, "header_line wrong type")
        layout = SourceLayout(
            header_line=header_line,
            trailing_newline=_expect(raw, path, "trailing_newline", bool),
            csv_sha256=_expect(raw, path, "csv_sha256", str),
            csv_bytes=_expect(raw, path, "csv_bytes", int),
        )
    elif any(raw.get(name) is not None for name in ("header_line", "trailing_newline", "csv_sha256", "csv_bytes")):
        raise _malformed(path, "verbatim record carries layout fields")
    return SourceRecord(
        key=_expect(raw, path, "key", str),
        etag=_expect(raw, path, "etag", str),
        zip_sha256=_expect(raw, path, "zip_sha256", str),
        symbol=_expect(raw, path, "symbol", str),
        period=_expect(raw, path, "period", str),
        source_index=_expect(raw, path, "source_index", int),
        storage=storage,
        layout=layout,
    )


def write_sidecar(path: Path, sidecar: Sidecar) -> None:
    payload = {
        "format_version": sidecar.format_version,
        "dataset": sidecar.dataset,
        "month": sidecar.month,
        "object_path": sidecar.object_path,
        "object_bytes": sidecar.object_bytes,
        "object_md5": sidecar.object_md5,
        "object_sha256": sidecar.object_sha256,
        "created_at": sidecar.created_at,
        "sources": [_record_to_json(record) for record in sidecar.sources],
    }
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with pa.CompressedOutputStream(str(tmp), "zstd") as stream:
        stream.write(encoded)
    os.replace(tmp, path)


def read_sidecar(path: Path) -> Sidecar:
    try:
        with pa.input_stream(str(path), compression="zstd") as stream:
            payload = json.loads(stream.read())
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise _malformed(path, str(exc)) from exc
    if not isinstance(payload, dict):
        raise _malformed(path, "top level is not an object")
    raw_sources = _expect(payload, path, "sources", list)
    object_path = _expect(payload, path, "object_path", str)
    parts = object_path.split("/")
    if parts[0] != "objects" or len(parts) < 2 or any(part in ("", ".", "..") for part in parts):
        raise _malformed(path, "object_path must stay under objects/")
    return Sidecar(
        format_version=_expect(payload, path, "format_version", int),
        dataset=_expect(payload, path, "dataset", str),
        month=_expect(payload, path, "month", str),
        object_path=object_path,
        object_bytes=_expect(payload, path, "object_bytes", int),
        object_md5=_expect(payload, path, "object_md5", str),
        object_sha256=_expect(payload, path, "object_sha256", str),
        created_at=_expect(payload, path, "created_at", str),
        sources=tuple(_record_from_json(raw, path) for raw in raw_sources),
    )


def iter_index(directory: Path, dataset: str) -> Iterator[Sidecar]:
    """Sidecars of one dataset under an index root (columnar and verbatim), one at a time."""
    roots = (directory / dataset, directory / "verbatim" / dataset)
    for path in sorted(path for root in roots if root.is_dir() for path in root.rglob(f"*{SIDECAR_SUFFIX}")):
        yield read_sidecar(path)


def load_index(directory: Path, dataset: str) -> tuple[Sidecar, ...]:
    """All sidecars of one dataset under an index root; use ``iter_index`` when the records need not be kept."""
    return tuple(iter_index(directory, dataset))


@dataclass(slots=True)
class IndexSummary:
    """What planning needs from an index: upstream ETags held per key, their symbols and the object bytes."""

    mirrored: dict[str, frozenset[str]]
    symbols: set[str]
    object_bytes: int = 0

    def add(self, sidecars: Iterable[Sidecar]) -> int:
        """Fold sidecars in without retaining them and return their object bytes.

        A revised key keeps every ETag it was mirrored under.
        """
        added = 0
        for sidecar in sidecars:
            added += sidecar.object_bytes
            for record in sidecar.sources:
                held = self.mirrored.get(record.key)
                if held is None:
                    self.mirrored[record.key] = frozenset((record.etag,))
                elif record.etag not in held:
                    self.mirrored[record.key] = held | {record.etag}
                self.symbols.add(record.symbol)
        self.object_bytes += added
        return added
