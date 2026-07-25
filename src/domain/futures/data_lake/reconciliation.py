from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

from src.domain.futures.data_lake.contracts import DatasetKind, PartitionManifest

_logger = logging.getLogger(__name__)

_SIDECAR_SUFFIX = ".sidecar.json"
_CATALOG_DB = "catalog.duckdb"
_CATALOG_NEXT_DB = "catalog.duckdb.next"
_QUARANTINE_DIR = "quarantine"


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
        _start_ms = int(ts.min()) if not ts.empty else 0
        _end_ms = int(ts.max()) if not ts.empty else 0
        _row_count = len(df)
    except Exception:
        _start_ms = int(pd.Timestamp(year, month, 1).timestamp() * 1000)
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


def _compute_sidecar_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(_SIDECAR_SUFFIX)


def reconcile_local_catalog(
    *,
    root: Path,
    cutoff_exclusive_ns: int,
    lock_timeout_seconds: int = 60,
) -> ManifestReconciliationReport:  # pragma: no cover - filesystem integration exercised separately
    import pandas as pd
    root = Path(root)
    catalog_path = root / _CATALOG_DB
    catalog_next_path = root / _CATALOG_NEXT_DB
    quarantine_dir = root / _QUARANTINE_DIR

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

    for fpath in parquet_files:
        sidecar_path = _compute_sidecar_path(fpath)
        stat = fpath.stat()
        size = stat.st_size
        mtime_ns = int(stat.st_mtime_ns)

        sidecar_valid = False
        if sidecar_path.exists():
            sidecar_data = _read_sidecar(sidecar_path)
            if sidecar_data is not None:
                cached_size = sidecar_data.get("size")
                cached_mtime = sidecar_data.get("mtime_ns")
                if cached_size == size and cached_mtime == mtime_ns:
                    reused += 1
                    sha256_val = sidecar_data.get("sha256", "")
                    sha256 = str(sha256_val) if sha256_val is not None else ""
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

            _write_sidecar(sidecar_path, {"size": size, "mtime_ns": mtime_ns, "sha256": sha256})

            import pandas as pd
            try:
                columns = pd.read_parquet(fpath, engine="pyarrow").columns
                time_column = "timestamp" if "timestamp" in columns else "effective_time_ns"
                _ = pd.read_parquet(fpath, columns=[time_column])
            except Exception:
                _logger.warning("invalid parquet %s, quarantining", fpath)
                qpath = quarantine_dir / fpath.relative_to(root)
                qpath.parent.mkdir(parents=True, exist_ok=True)
                os.rename(fpath, qpath)
                quarantined.append(str(fpath))
                continue

            manifest = _partition_manifest_from_path(fpath, root, sha256, size, mtime_ns)
            if manifest is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO partitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [manifest.dataset.value, manifest.symbol, manifest.start_time_ms,
                     manifest.end_time_ms, manifest.row_count, manifest.sha256,
                     manifest.source, manifest.is_final, str(manifest.path)],
                )
                added += 1

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
    )


__all__ = [
    "CatalogLockError",
    "ManifestReconciliationReport",
    "reconcile_local_catalog",
]
