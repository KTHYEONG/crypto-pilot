"""Sealed MHS input manifests and forward-observation evidence (P3 provenance).

Collection/refresh seals every consumed parquet once (SHA-256 over bytes plus
row bounds); hot replay validates only path/size/mtime metadata plus the
required-path set derived from the configured panel/execution roster, so the
7.8GB corpus is never rehashed per run (INV-INPUT-SEAL). Manifest entries are
canonical (data-root-relative POSIX, deduplicated, sorted) and written
atomically; a partial write never clobbers the last good manifest
(INV-MANIFEST-CANONICAL-ATOMIC). Forward live observations are immutable
evidence: digest-less or foreign-digest rows are preserved but excluded, and
forward data can never relabel the frozen strategy (INV-FORWARD-EVIDENCE-IMMUTABILITY).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow.parquet as pq

from src.live.execution_quality import EXECUTION_QUALITY_MIN_EVIDENCE_DAYS


class DataEvidenceTier(StrEnum):
    """Evidence strength of the data behind a reported result."""

    UNSEALED_ARCHIVE = "unsealed_archive"
    ARCHIVE_PROXY = "archive_proxy"
    REPRODUCIBLE_ARCHIVE = "reproducible_archive"
    FORWARD_OBSERVED = "forward_observed"


@dataclass(frozen=True, slots=True)
class MhsInputFileAttestation:
    """One sealed input file: identity hash plus row-bound provenance."""

    relative_path: str
    sha256: str
    size_bytes: int
    mtime_ns: int
    row_count: int
    first_timestamp: str
    last_timestamp: str
    data_kind: str
    symbol: str


@dataclass(frozen=True, slots=True)
class DataProvenanceResult:
    """Provenance verdict for one evidence tier."""

    tier: DataEvidenceTier
    valid: bool
    reason_codes: tuple[str, ...]
    manifest_digest: str | None
    files_checked: int


def _canonical_entries_json(entries: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        {"version": 1, "files": entries}, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _attest_one(path: Path, data_root: Path) -> MhsInputFileAttestation:
    resolved = Path(path).resolve()
    root = Path(data_root).resolve()
    rel = resolved.relative_to(root).as_posix()
    blob = resolved.read_bytes()
    stat = resolved.stat()
    parquet_file = pq.ParquetFile(resolved)
    schema_names = set(parquet_file.schema_arrow.names)
    ts_column = "timestamp" if "timestamp" in schema_names else "datetime"
    bounds = pd.read_parquet(resolved, columns=[ts_column]).iloc[:, 0]
    # Binance futures Parquet ``timestamp`` columns are epoch milliseconds;
    # omitting ``unit`` silently interpreted them as nanoseconds and sealed
    # every bound in 1970, making provenance metadata non-auditable.
    if pd.api.types.is_numeric_dtype(bounds):
        stamps = pd.to_datetime(bounds, unit="ms", utc=True, errors="coerce")
    else:
        stamps = pd.to_datetime(bounds, utc=True, errors="coerce")
    parts = Path(rel).parts
    kind = "funding" if "funding" in parts else ("mark" if any("mark" in part for part in parts) else "ohlcv")
    return MhsInputFileAttestation(
        relative_path=rel,
        sha256=hashlib.sha256(blob).hexdigest(),
        size_bytes=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        row_count=int(parquet_file.metadata.num_rows),
        first_timestamp=stamps.min().isoformat() if len(stamps) else "",
        last_timestamp=stamps.max().isoformat() if len(stamps) else "",
        data_kind=kind,
        symbol=Path(rel).stem,
    )


def seal_mhs_input_manifest(
    paths: Sequence[Path], *, data_root: Path, output_path: Path
) -> str:
    """Seal the canonical input manifest; returns its digest (atomic write)."""
    root = Path(data_root).resolve()
    unique = list(dict.fromkeys(Path(p).resolve() for p in paths))
    attestations = [_attest_one(p, root) for p in unique]
    entries = [
        {
            "relative_path": a.relative_path,
            "sha256": a.sha256,
            "size_bytes": a.size_bytes,
            "mtime_ns": a.mtime_ns,
            "row_count": a.row_count,
            "first_timestamp": a.first_timestamp,
            "last_timestamp": a.last_timestamp,
            "data_kind": a.data_kind,
            "symbol": a.symbol,
        }
        for a in sorted(attestations, key=lambda a: a.relative_path)
    ]
    digest = hashlib.sha256(_canonical_entries_json(entries)).hexdigest()
    payload = {"version": 1, "digest": digest, "files": entries}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, out)
    return digest


def validate_mhs_input_manifest(
    manifest_path: Path | None, *, data_root: Path, required_paths: Sequence[Path]
) -> DataProvenanceResult:
    """Validate sealed provenance without rehashing (metadata-only hot path)."""
    root = Path(data_root).resolve()
    if manifest_path is None or not Path(manifest_path).exists():
        return DataProvenanceResult(DataEvidenceTier.UNSEALED_ARCHIVE, False, ("MISSING_INPUT_MANIFEST",), None, 0)
    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest_digest = raw.get("digest")
    attested = {str(entry["relative_path"]): entry for entry in raw.get("files", [])}
    required = [Path(p).resolve().relative_to(root).as_posix() for p in required_paths]
    for rel in required:
        entry = attested.get(rel)
        if entry is None:
            return DataProvenanceResult(
                DataEvidenceTier.UNSEALED_ARCHIVE, False, ("UNATTESTED_REQUIRED_FILE",), manifest_digest, len(required),
            )
        candidate = root / rel
        stat = candidate.stat() if candidate.exists() else None
        if stat is None or int(stat.st_size) != int(entry["size_bytes"]) or int(stat.st_mtime_ns) != int(entry["mtime_ns"]):
            return DataProvenanceResult(
                DataEvidenceTier.UNSEALED_ARCHIVE, False, ("STALE_INPUT_MANIFEST",), manifest_digest, len(required),
            )
    return DataProvenanceResult(
        DataEvidenceTier.REPRODUCIBLE_ARCHIVE, True, (), manifest_digest, len(required),
    )


def resolve_required_mhs_input_paths(
    *, data_root: Path, panel_symbols: Sequence[str], execution_symbols: Sequence[str], execution_timeframe: Literal["3m"]
) -> tuple[Path, ...]:
    """Resolve exactly the files consumed by the registered MHS panel and replay: 1h OHLCV for panel symbols, 3m OHLCV for executed symbols, and funding for symbols whose rates affect signals or held cash flow. Missing required files remain missing evidence; mark and daily metrics never enter the input identity."""
    if execution_timeframe != "3m":
        raise ValueError(f"unknown execution_timeframe {execution_timeframe!r}")
    root = Path(data_root)
    paths = [
        *(root / "ohlcv" / "1h" / f"{symbol}.parquet" for symbol in dict.fromkeys(panel_symbols)),
        *(root / "ohlcv" / execution_timeframe / f"{symbol}.parquet" for symbol in dict.fromkeys(execution_symbols)),
        *(root / "funding" / f"{symbol}.parquet" for symbol in dict.fromkeys([*panel_symbols, *execution_symbols])),
    ]
    ordered = sorted({p.as_posix() for p in paths})
    return tuple(Path(p) for p in ordered)


def validate_forward_execution_observations(
    records: pd.DataFrame, *, frozen_strategy_digest: str
) -> DataProvenanceResult:
    """Gate forward evidence: frozen-digest, causally-timed, sufficiently long."""
    required = ("decision_time", "observed_at", "strategy_digest")
    absent = [column for column in required if column not in records.columns]
    if absent or records.empty:
        return DataProvenanceResult(DataEvidenceTier.UNSEALED_ARCHIVE, False, ("FORWARD_OBSERVATION_SCHEMA_MISMATCH",), None, 0)
    frame = records.copy()
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
    digest_ok = frame["strategy_digest"].notna() & (frame["strategy_digest"].astype(str) == str(frozen_strategy_digest))
    time_ok = frame["observed_at"].notna() & frame["decision_time"].notna() & (frame["observed_at"] >= frame["decision_time"])
    usable = frame[digest_ok & time_ok]
    reasons: list[str] = []
    if bool((~digest_ok).any()):
        reasons.append("STRATEGY_DIGEST_MISMATCH")
    if bool(((frame["observed_at"].isna()) | (frame["decision_time"].isna()) | (frame["observed_at"] < frame["decision_time"])).any()):
        reasons.append("FORWARD_OBSERVATION_TIME_INVALID")
    if usable.empty:
        return DataProvenanceResult(DataEvidenceTier.UNSEALED_ARCHIVE, False, tuple(reasons), None, 0)
    span_days = int((usable["observed_at"].max().normalize() - usable["observed_at"].min().normalize()).days)
    if span_days < EXECUTION_QUALITY_MIN_EVIDENCE_DAYS:
        return DataProvenanceResult(
            DataEvidenceTier.UNSEALED_ARCHIVE, False, (*reasons, "FORWARD_EVIDENCE_INCOMPLETE"), None, len(usable),
        )
    return DataProvenanceResult(DataEvidenceTier.FORWARD_OBSERVED, True, (), None, len(usable))


def mhs_sealable_input_paths(*, data_root: Path, execution_timeframe: Literal["3m"]) -> tuple[Path, ...]:
    """Enumerate sealable inputs: existing required files for discovered symbols.

    Symbols are discovered from ``<data_root>/ohlcv/1h/*.parquet`` sorted by
    stem; the required-path layout comes solely from
    :func:`resolve_required_mhs_input_paths`. Collection-time sealing includes
    only existing files, while evaluation-time validation reports each missing
    required file as incomplete evidence.
    """
    if execution_timeframe != "3m":
        raise ValueError(f"unknown execution_timeframe {execution_timeframe!r}")
    root = Path(data_root)
    symbols = sorted({path.stem for path in (root / "ohlcv" / "1h").glob("*.parquet")})
    required = resolve_required_mhs_input_paths(
        data_root=root,
        panel_symbols=symbols,
        execution_symbols=symbols,
        execution_timeframe=execution_timeframe,
    )
    sealed = [path for path in required if path.exists()]
    return tuple(dict.fromkeys(sealed))
