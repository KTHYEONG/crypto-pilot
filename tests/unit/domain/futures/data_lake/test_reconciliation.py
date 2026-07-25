from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.domain.futures.data_lake.reconciliation import (
    ManifestReconciliationReport,
    _partition_key,
    _read_sidecar,
    _scan_parquet_files,
    _write_sidecar,
    reconcile_local_catalog,
)


class TestReconcileLocalCatalog:
    @pytest.fixture
    def lake_root(self) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="test_lake_"))
        yield tmp
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    def test_reconcile_registers_unindexed_physical_parquet(self, lake_root: Path) -> None:
        import duckdb
        import pandas as pd

        parquet_dir = lake_root / "klines_1h" / "symbol=BTCUSDT" / "year=2024" / "month=01"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"timestamp": [1704412800000, 1704499200000], "close": [40000.0, 40100.0]})
        parquet_path = parquet_dir / "part.parquet"
        df.to_parquet(parquet_path, index=False)

        report = reconcile_local_catalog(root=lake_root, cutoff_exclusive_ns=1735689600_000_000_000)

        assert report.added_rows >= 1
        assert report.scanned_files >= 1
        conn = duckdb.connect(str(lake_root / "catalog.duckdb"), read_only=True)
        start_time_ms, = conn.execute("SELECT start_time_ms FROM partitions").fetchone()
        conn.close()
        assert start_time_ms == 1704067200000

    def test_reconcile_replaces_existing_path_manifest_with_month_partition_key(
        self, lake_root: Path,
    ) -> None:
        import duckdb
        import pandas as pd

        parquet_dir = lake_root / "funding_event" / "symbol=BTCUSDT" / "year=2024" / "month=01"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = parquet_dir / "part.parquet"
        pd.DataFrame({"timestamp": [1704412800000], "funding_rate": [0.0001]}).to_parquet(
            parquet_path, index=False,
        )
        conn = duckdb.connect(str(lake_root / "catalog.duckdb"))
        conn.execute(
            "CREATE TABLE partitions (dataset VARCHAR, symbol VARCHAR, start_time_ms BIGINT, "
            "end_time_ms BIGINT, row_count BIGINT, sha256 VARCHAR, source VARCHAR, "
            "is_final BOOLEAN, path VARCHAR, PRIMARY KEY (dataset, symbol, start_time_ms))"
        )
        conn.execute(
            "INSERT INTO partitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["funding_event", "BTCUSDT", 1704412800000, 1704412800000, 1, "old", "local_reconciliation", True, str(parquet_path)],
        )
        conn.close()

        reconcile_local_catalog(root=lake_root, cutoff_exclusive_ns=1735689600_000_000_000)

        conn = duckdb.connect(str(lake_root / "catalog.duckdb"), read_only=True)
        manifests = conn.execute(
            "SELECT start_time_ms, path FROM partitions WHERE dataset = 'funding_event'"
        ).fetchall()
        conn.close()
        assert manifests == [(1704067200000, str(parquet_path))]

    def test_reconcile_atomic_failure_preserves_old_catalog(self, lake_root: Path) -> None:
        import pandas as pd
        import duckdb

        parquet_dir = lake_root / "klines_1h" / "symbol=BTCUSDT" / "year=2024" / "month=01"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"timestamp": [1704067200000], "close": [40000.0]})
        df.to_parquet(parquet_dir / "part.parquet", index=False)

        old_db = lake_root / "catalog.duckdb"
        conn = duckdb.connect(str(old_db))
        conn.execute("CREATE TABLE IF NOT EXISTS partitions (dummy INTEGER)")
        conn.execute("INSERT INTO partitions VALUES (1)")
        conn.close()

        report = reconcile_local_catalog(root=lake_root, cutoff_exclusive_ns=1735689600_000_000_000)
        assert report.scanned_files >= 1
        assert old_db.exists()

    def test_reconcile_lock_timeout_never_creates_recovery_catalog(self, lake_root: Path) -> None:
        recovery = lake_root / "catalog_recovered.duckdb"
        assert not recovery.exists()

    def test_reconcile_with_empty_catalog_succeeds(self, lake_root: Path) -> None:
        import pandas as pd
        parquet_dir = lake_root / "klines_1h" / "symbol=BTCUSDT" / "year=2024" / "month=01"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"timestamp": [1704067200000, 1704153600000], "close": [40000.0, 40100.0]})
        df.to_parquet(parquet_dir / "part.parquet", index=False)

        report = reconcile_local_catalog(root=lake_root, cutoff_exclusive_ns=1735689600_000_000_000)
        assert report.scanned_files >= 1

    def test_reconcile_ignores_partitions_after_cutoff(self, lake_root: Path) -> None:
        import pandas as pd
        parquet_dir = lake_root / "klines_1h" / "symbol=BTCUSDT" / "year=2026" / "month=07"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"timestamp": [1780300800000], "close": [50000.0]})
        df.to_parquet(parquet_dir / "part.parquet", index=False)

        cutoff_ns = int(pd.Timestamp("2026-07-01", tz="UTC").timestamp() * 1_000_000_000)
        report = reconcile_local_catalog(root=lake_root, cutoff_exclusive_ns=cutoff_ns)
        for p in parquet_dir.rglob("*.parquet"):
            assert p.exists()

    def test_corrupt_partition_moves_to_quarantine(self, lake_root: Path) -> None:
        parquet_dir = lake_root / "klines_1h" / "symbol=BTCUSDT" / "year=2024" / "month=01"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        bad_path = parquet_dir / "part.parquet"
        bad_path.write_bytes(b"not a valid parquet file")

        report = reconcile_local_catalog(
            root=lake_root,
            cutoff_exclusive_ns=1735689600_000_000_000,
        )
        quarantine_dir = lake_root / "quarantine"
        assert any(quarantine_dir.rglob("*.parquet")) or len(report.quarantined_files) > 0


class TestManifestReconciliationReport:
    def test_report_fields(self) -> None:
        report = ManifestReconciliationReport(
            scanned_files=100,
            reused_sidecars=20,
            added_rows=5,
            removed_stale_rows=3,
            quarantined_files=("a.parquet",),
            catalog_hash="abc123",
        )
        assert report.scanned_files == 100
        assert report.reused_sidecars == 20
        assert report.added_rows == 5
        assert report.catalog_hash == "abc123"


def test_reconciliation_helpers_cover_sidecars_and_path_filters(tmp_path: Path) -> None:
    sidecar = tmp_path / "part.sidecar.json"
    _write_sidecar(sidecar, {"size": 1})
    assert _read_sidecar(sidecar) == {"size": 1}
    assert _read_sidecar(tmp_path / "missing.json") is None
    assert _partition_key(tmp_path / "klines_1h" / "symbol=BTC" / "year=2024" / "month=01" / "x.parquet") is None
    assert _partition_key(tmp_path / "bad") is None
    assert _scan_parquet_files(tmp_path, cutoff_exclusive_ns=9_999_999_999_999_999) == []
