from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.data_lake.contracts import DatasetKind, PartitionManifest

_logger = logging.getLogger(__name__)

_SIDECAR_SUFFIX = ".sidecar.json"
_CATALOG_DB = "catalog.duckdb"
_CATALOG_NEXT_DB = "catalog.duckdb.next"
_QUARANTINE_DIR = "quarantine"
_MAX_ABS_FUNDING_RATE = 0.05
_FUNDING_SCHEMA_VERSION = "funding-v3"
_FUNDING_VALIDATOR_VERSION = "funding-integrity-v1"


class CatalogLockError(RuntimeError):
    ...


@dataclass(slots=True, frozen=True)
class ManifestReconciliationReport:
    scanned_files: int
    reused_sidecars: int
    added_rows: int
    removed_stale_rows: int
    quarantined_files: tuple[str, ...]
    catalog_hash: str
    funding_repair_requests: tuple[FundingRepairRequest, ...] = ()


@dataclass(slots=True, frozen=True)
class FundingRepairRequest:
    symbol: str
    start_time_ms: int
    parquet_path: Path


@dataclass(slots=True, frozen=True)
class FundingPartitionAudit:
    checked_files: int
    invalid_requests: tuple[FundingRepairRequest, ...]


def _read_sidecar(sidecar_path: Path) -> dict[str, object] | None:
    import json
    try:
        data = json.loads(sidecar_path.read_text())
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError):
        return None


def _write_sidecar(sidecar_path: Path, metadata: dict[str, object]) -> None:
    import json
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(metadata, sort_keys=True))


def _partition_key(path: Path) -> tuple[str, str, int] | None:
    try:
        parts = path.parts
        dataset_str = parts[-5] if len(parts) >= 5 else ""
        symbol_str = parts[-3] if len(parts) >= 3 else ""
        year_month = parts[-2] if len(parts) >= 2 else ""
        year_s = year_month.removeprefix("year=")
        return (dataset_str, symbol_str, int(year_s))
    except (ValueError, IndexError):
        return None


def _partition_manifest_from_path(
    path: Path, root: Path, sha256: str, size: int, mtime_ns: int,
) -> PartitionManifest | None:
    try:
        parts = path.relative_to(root).parts
        if len(parts) < 4:
            return None
        dataset_str = parts[0]
        symbol_str = parts[1].removeprefix("symbol=")
        year_month = parts[2].removeprefix("year=")
        year = int(year_month)
        month_part = parts[3].removeprefix("month=")
        month = int(month_part.rstrip(".parquet"))
    except (ValueError, IndexError):
        return None
    try:
        dataset = DatasetKind(dataset_str)
    except ValueError:
        _logger.warning("unknown dataset kind: %s", dataset_str)
        return None

    import pandas as pd
    try:
        schema = pd.read_parquet(path, engine="pyarrow").columns
        time_column = "timestamp" if "timestamp" in schema else "effective_time_ns"
        df = pd.read_parquet(path, columns=[time_column])
        ts = df[time_column]
        _start_ms = int(datetime(year, month, 1, tzinfo=UTC).timestamp() * 1000)
        _end_ms = int(ts.max()) if not ts.empty else 0
        _row_count = len(df)
    except Exception:
        _start_ms = int(datetime(year, month, 1, tzinfo=UTC).timestamp() * 1000)
        _end_ms = _start_ms
        _row_count = 0

    return PartitionManifest(
        dataset=dataset,
        symbol=symbol_str,
        start_time_ms=_start_ms,
        end_time_ms=_end_ms,
        row_count=_row_count,
        sha256=sha256,
        source="local_reconciliation",
        is_final=True,
        path=path,
    )


def _scan_parquet_files(root: Path, cutoff_exclusive_ns: int) -> list[Path]:
    parquet_files: list[Path] = []
    for fpath in root.rglob("*.parquet"):
        rel = fpath.relative_to(root)
        if any(part.name == _QUARANTINE_DIR for part in rel.parents):
            continue
        if _CATALOG_DB in str(rel) or _CATALOG_NEXT_DB in str(rel):
            continue
        parquet_files.append(fpath)

    filtered: list[Path] = []
    for fpath in parquet_files:
        parts = fpath.relative_to(root).parts
        if len(parts) >= 3:
            try:
                year_str = parts[2].removeprefix("year=")
                year = int(year_str)
                if year < 1970 or year > 2100:
                    continue
                _month_str = parts[3].removeprefix("month=") if len(parts) >= 4 else ""
                month = int(_month_str.rstrip(".parquet")) if _month_str else 1
                import pandas as pd
                partition_end = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(1)
                partition_end_ns = partition_end.timestamp() * 1_000_000_000
                if int(partition_end_ns) >= cutoff_exclusive_ns:
                    continue
            except (ValueError, IndexError):
                pass
        filtered.append(fpath)
    return filtered


def validate_funding_rates(
    rates: NDArray[np.float64],
    *,
    source: str,
    max_abs_rate: float = _MAX_ABS_FUNDING_RATE,
) -> None:
    """Raise FundingDataIntegrityError on non-finite or implausible event rates."""
    from src.domain.futures.compound.contracts import FundingDataIntegrityError
    if not np.all(np.isfinite(rates)):
        non_finite = int(np.sum(~np.isfinite(rates)))
        raise FundingDataIntegrityError(
            f"{source}: {non_finite} non-finite funding rate values"
        )
    if np.any(np.abs(rates) > max_abs_rate):
        outliers = int(np.sum(np.abs(rates) > max_abs_rate))
        raise FundingDataIntegrityError(
            f"{source}: {outliers} funding rate values exceed |{max_abs_rate}|"
        )


def validate_funding_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    month_start_ms: int | None = None,
) -> None:
    """Validate canonical funding timestamps and rates without altering values."""
    from src.domain.futures.compound.contracts import FundingDataIntegrityError

    required = {"timestamp", "funding_rate"}
    if not required.issubset(frame.columns):
        raise FundingDataIntegrityError(f"{source}: missing funding columns")
    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(timestamps)) or np.any(timestamps != np.floor(timestamps)):
        raise FundingDataIntegrityError(f"{source}: invalid funding timestamps")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise FundingDataIntegrityError(f"{source}: timestamps must be strictly increasing")
    rates = pd.to_numeric(frame["funding_rate"], errors="coerce").to_numpy(dtype=np.float64)
    validate_funding_rates(rates, source=source)
    if month_start_ms is not None and timestamps.size:
        start = pd.Timestamp(month_start_ms, unit="ms", tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        if np.any(timestamps < month_start_ms) or np.any(
            timestamps >= int(end.timestamp() * 1000)
        ):
            raise FundingDataIntegrityError(f"{source}: timestamp outside partition month")


def _funding_repair_request(path: Path, root: Path) -> FundingRepairRequest | None:
    try:
        relative = path.relative_to(root)
        symbol = relative.parts[1].removeprefix("symbol=")
        year = int(relative.parts[2].removeprefix("year="))
        month = int(relative.parts[3].removeprefix("month=").split(".")[0])
    except (ValueError, IndexError):
        return None
    start_time_ms = int(datetime(year, month, 1, tzinfo=UTC).timestamp() * 1000)
    return FundingRepairRequest(symbol=symbol, start_time_ms=start_time_ms, parquet_path=path)


def audit_funding_partitions(
    *, root: Path, cutoff_exclusive_ns: int,
) -> FundingPartitionAudit:
    """Read-only audit used to prevent LOCAL runs from consuming corrupt funding."""
    import pandas as pd

    requests: list[FundingRepairRequest] = []
    checked = 0
    for path in _scan_parquet_files(Path(root), cutoff_exclusive_ns):
        if "funding_event" not in path.parts:
            continue
        request = _funding_repair_request(path, Path(root))
        if request is None:
            continue
        checked += 1
        try:
            frame = pd.read_parquet(path, engine="pyarrow")
            validate_funding_frame(
                frame,
                source=str(path),
                month_start_ms=request.start_time_ms,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("funding audit failed for %s: %s", path, exc)
            requests.append(request)
    return FundingPartitionAudit(checked_files=checked, invalid_requests=tuple(requests))


def quarantine_funding_partitions(
    *, requests: tuple[FundingRepairRequest, ...], root: Path,
) -> tuple[str, ...]:
    """Move only invalid funding files and sidecars to recoverable quarantine."""
    quarantine_dir = Path(root) / _QUARANTINE_DIR
    moved: list[str] = []
    for request in requests:
        path = request.parquet_path
        if not path.exists():
            continue
        target = quarantine_dir / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, target)
        sidecar = _compute_sidecar_path(path)
        if sidecar.exists():
            os.replace(sidecar, target.with_suffix(_SIDECAR_SUFFIX))
        moved.append(str(path))
    return tuple(moved)


def _compute_sidecar_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(_SIDECAR_SUFFIX)


def reconcile_local_catalog(
    *,
    root: Path,
    cutoff_exclusive_ns: int,
    lock_timeout_seconds: int = 60,
    sync_mode: str = "auto",
) -> ManifestReconciliationReport:  # pragma: no cover - filesystem integration exercised separately
    import pandas as pd
    root = Path(root)
    catalog_path = root / _CATALOG_DB
    catalog_next_path = root / _CATALOG_NEXT_DB
    quarantine_dir = root / _QUARANTINE_DIR
    from src.domain.futures.compound.contracts import FundingDataIntegrityError

    start_time = time.monotonic()
    lock_acquired = False
    while time.monotonic() - start_time < lock_timeout_seconds:
        try:
            conn = duckdb.connect(str(catalog_path))
            conn.execute("SELECT 1")
            conn.close()
            lock_acquired = True
            break
        except duckdb.IOException:
            time.sleep(1.0)
    if not lock_acquired:
        raise CatalogLockError(
            f"could not acquire lock on {catalog_path} within {lock_timeout_seconds}s"
        )

    conn = duckdb.connect(str(catalog_next_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS partitions (
            dataset VARCHAR NOT NULL, symbol VARCHAR NOT NULL,
            start_time_ms BIGINT NOT NULL, end_time_ms BIGINT NOT NULL,
            row_count BIGINT NOT NULL, sha256 VARCHAR NOT NULL,
            source VARCHAR NOT NULL, is_final BOOLEAN NOT NULL, path VARCHAR NOT NULL,
            PRIMARY KEY (dataset, symbol, start_time_ms)
        )
    """)

    existing_by_path: dict[str, tuple[object, ...]] = {}
    if catalog_path.exists():
        try:
            old_conn = duckdb.connect(str(catalog_path))
            try:
                old_conn.execute("SELECT dataset FROM partitions LIMIT 1")
            except (duckdb.CatalogException, duckdb.BinderException):
                old_conn.close()
            else:
                existing_rows = old_conn.execute(
                    "SELECT dataset, symbol, start_time_ms, end_time_ms, row_count, sha256, source, is_final, path FROM partitions"
                ).fetchall()
                old_conn.close()
                for row in existing_rows:
                    existing_by_path[str(row[8])] = row
        except (duckdb.CatalogException, duckdb.BinderException):
            pass

    parquet_files = _scan_parquet_files(root, cutoff_exclusive_ns)
    scanned = len(parquet_files)
    reused = 0
    added = 0
    quarantined: list[str] = []
    funding_integrity_errors: list[str] = []
    funding_repair_requests: list[FundingRepairRequest] = []

    for fpath in parquet_files:
        sidecar_path = _compute_sidecar_path(fpath)
        stat = fpath.stat()
        size = stat.st_size
        mtime_ns = int(stat.st_mtime_ns)

        is_funding = "funding_event" in fpath.parts

        sidecar_valid = False
        if sidecar_path.exists():
            sidecar_data = _read_sidecar(sidecar_path)
            if sidecar_data is not None:
                cached_size = sidecar_data.get("size")
                cached_mtime = sidecar_data.get("mtime_ns")
                if cached_size == size and cached_mtime == mtime_ns:
                    if is_funding:
                        sv = sidecar_data.get("schema_version", "")
                        cached_sha = str(sidecar_data.get("sha256", ""))
                        if sv != _FUNDING_SCHEMA_VERSION or not cached_sha:
                            sidecar_valid = False
                        else:
                            try:
                                current_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
                            except OSError:
                                current_sha = ""
                            if current_sha == cached_sha:
                                reused += 1
                                sha256 = cached_sha
                                existing = existing_by_path.get(str(fpath))
                                if existing is not None:
                                    conn.execute("INSERT OR REPLACE INTO partitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", existing)
                                    added += 1
                                else:
                                    manifest = _partition_manifest_from_path(fpath, root, sha256, size, mtime_ns)
                                    if manifest is not None:
                                        conn.execute(
                                            "INSERT OR REPLACE INTO partitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                            [manifest.dataset.value, manifest.symbol, manifest.start_time_ms,
                                             manifest.end_time_ms, manifest.row_count, manifest.sha256,
                                             manifest.source, manifest.is_final, str(manifest.path)],
                                        )
                                        added += 1
                                sidecar_valid = True
                    else:
                        reused += 1
                        sha256_val = sidecar_data.get("sha256", "")
                        sha256 = str(sha256_val) if sha256_val is not None else ""
                        existing = existing_by_path.get(str(fpath))
                        if existing is not None:
                            conn.execute("INSERT OR REPLACE INTO partitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", existing)
                            added += 1
                        else:
                            manifest = _partition_manifest_from_path(fpath, root, sha256, size, mtime_ns)
                            if manifest is not None:
                                conn.execute(
                                    "INSERT OR REPLACE INTO partitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                    [manifest.dataset.value, manifest.symbol, manifest.start_time_ms,
                                     manifest.end_time_ms, manifest.row_count, manifest.sha256,
                                     manifest.source, manifest.is_final, str(manifest.path)],
                                )
                                added += 1
                        sidecar_valid = True

        if not sidecar_valid:
            try:
                payload = fpath.read_bytes()
                sha256 = hashlib.sha256(payload).hexdigest()
            except OSError:
                _logger.warning("cannot read %s, quarantining", fpath)
                qpath = quarantine_dir / fpath.relative_to(root)
                qpath.parent.mkdir(parents=True, exist_ok=True)
                os.rename(fpath, qpath)
                quarantined.append(str(fpath))
                continue

            sidecar_meta: dict[str, object] = {"size": size, "mtime_ns": mtime_ns, "sha256": sha256}

            import pandas as pd
            try:
                df_check = pd.read_parquet(fpath, engine="pyarrow")
                columns = df_check.columns
                time_column = "timestamp" if "timestamp" in columns else "effective_time_ns"
                _ = pd.read_parquet(fpath, columns=[time_column])
            except Exception:
                _logger.warning("invalid parquet %s, quarantining", fpath)
                qpath = quarantine_dir / fpath.relative_to(root)
                qpath.parent.mkdir(parents=True, exist_ok=True)
                os.rename(fpath, qpath)
                quarantined.append(str(fpath))
                continue

            if is_funding:
                request = _funding_repair_request(fpath, root)
                try:
                    validate_funding_frame(
                        df_check,
                        source=str(fpath),
                        month_start_ms=request.start_time_ms if request is not None else None,
                    )
                except FundingDataIntegrityError:
                    _logger.warning("funding integrity %s, quarantining", fpath)
                    if sync_mode == "local":
                        funding_integrity_errors.append(str(fpath))
                    if request is not None:
                        funding_repair_requests.append(request)
                    qpath = quarantine_dir / fpath.relative_to(root)
                    qpath.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(fpath, qpath)
                    sidecar_path = _compute_sidecar_path(fpath)
                    if sidecar_path.exists():
                        os.rename(sidecar_path, qpath.with_suffix(_SIDECAR_SUFFIX))
                    quarantined.append(str(fpath))
                    continue
                sidecar_meta.update(
                    {
                        "schema_version": _FUNDING_SCHEMA_VERSION,
                        "validator_version": _FUNDING_VALIDATOR_VERSION,
                        "row_count": len(df_check),
                        "min_timestamp": int(df_check["timestamp"].min()),
                        "max_timestamp": int(df_check["timestamp"].max()),
                        "min_rate": float(pd.to_numeric(df_check["funding_rate"]).min()),
                        "max_rate": float(pd.to_numeric(df_check["funding_rate"]).max()),
                    }
                )

            _write_sidecar(sidecar_path, sidecar_meta)

            manifest = _partition_manifest_from_path(fpath, root, sha256, size, mtime_ns)
            if manifest is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO partitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [manifest.dataset.value, manifest.symbol, manifest.start_time_ms,
                     manifest.end_time_ms, manifest.row_count, manifest.sha256,
                     manifest.source, manifest.is_final, str(manifest.path)],
                )
                added += 1

    if sync_mode == "local" and funding_integrity_errors:
        raise FundingDataIntegrityError(
            f"local sync: {len(funding_integrity_errors)} funding partitions failed integrity check"
        )

    path_rows = conn.execute("SELECT path FROM partitions").fetchall()
    new_paths = {str(row[0]) for row in path_rows if len(row) > 0}
    existing_paths = set(existing_by_path.keys())
    removed_stale = len(existing_paths - new_paths)
    for stale_path in existing_paths - new_paths:
        if stale_path in new_paths:
            continue
        conn.execute("DELETE FROM partitions WHERE path = ?", [stale_path])

    conn.close()

    catalog_hash = ""
    new_conn = duckdb.connect(str(catalog_next_path))
    rows = new_conn.execute("SELECT dataset, symbol, start_time_ms, sha256 FROM partitions ORDER BY dataset, symbol, start_time_ms").fetchall()
    body = "|".join(f"{r[0]}|{r[1]}|{r[2]}|{r[3]}" for r in rows)
    catalog_hash = hashlib.sha256(body.encode()).hexdigest()
    new_conn.close()

    os.replace(catalog_next_path, catalog_path)

    _logger.info(
        "reconciliation complete: scanned=%d reused=%d added=%d removed=%d quarantined=%d hash=%s",
        scanned, reused, added, removed_stale, len(quarantined), catalog_hash,
    )

    return ManifestReconciliationReport(
        scanned_files=scanned,
        reused_sidecars=reused,
        added_rows=added,
        removed_stale_rows=removed_stale,
        quarantined_files=tuple(quarantined),
        catalog_hash=catalog_hash,
        funding_repair_requests=tuple(funding_repair_requests),
    )


__all__ = [
    "CatalogLockError",
    "FundingPartitionAudit",
    "FundingRepairRequest",
    "ManifestReconciliationReport",
    "audit_funding_partitions",
    "quarantine_funding_partitions",
    "reconcile_local_catalog",
    "validate_funding_frame",
    "validate_funding_rates",
]
