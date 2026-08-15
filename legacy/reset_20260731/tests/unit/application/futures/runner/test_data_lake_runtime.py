from __future__ import annotations

import hashlib
import io
from datetime import date
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.application.futures.runner.data_lake_runtime import (
    DataLakeRuntime,
    build_data_lake_runtime,
    finalize_quarterly_signal_data,
    prepare_data_snapshot,
    prepare_quarterly_bootstrap,
)
from src.domain.futures.data_lake.contracts import (
    DataLakeConfig,
    DataSnapshot,
    IngestionPlan,
    PartitionManifest,
    SyncMode,
    PreparedBootstrap,
)
from src.domain.futures.data_lake.ingestion import (
    ChecksumMismatchError,
    DataCoverageError,
    StorageBudgetError,
    sync_futures_data_lake,
)
from src.domain.futures.universe.config import PITUniverseConfig
from src.domain.futures.data_lake.run_windows import QuarterlyWindowConfig, resolve_completed_quarter_window


def _snap(
    snapshot_id: str = "s1",
    reference_time_ms: int = 1_000_000,
    partitions: tuple = (),
    manifest_hash: str = "h1",
    total_bytes: int = 0,
) -> DataSnapshot:
    return DataSnapshot(
        snapshot_id=snapshot_id,
        reference_time_ms=reference_time_ms,
        partitions=partitions,
        manifest_hash=manifest_hash,
        universe_state_hash="",
        total_bytes=total_bytes,
    )


class FakeClient:
    def __init__(self) -> None:
        self.download_calls = 0

    def download_partition(self, *args: object, **kwargs: object) -> bytes:
        self.download_calls += 1
        buffer = io.BytesIO()
        pd.DataFrame({"timestamp": [1_783_440_000_000], "close": [100.0]}).to_parquet(buffer, index=False)
        return buffer.getvalue()

    def download_checksum(self, *args: object, **kwargs: object) -> str:
        return hashlib.sha256(self.download_partition()).hexdigest()


class FakeCatalog:
    def __init__(self, snapshot: DataSnapshot, *, complete: bool) -> None:
        self.snapshot = snapshot
        self.complete = complete
        self.committed: list[PartitionManifest] = []
        self._bytes = 0

    def load_snapshot(self, reference_time_ms: int) -> DataSnapshot:
        return self.snapshot

    def has_complete_coverage(self, snapshot: DataSnapshot, plan: IngestionPlan) -> bool:
        return self.complete

    def commit_partition(self, manifest: PartitionManifest) -> None:
        self.committed.append(manifest)

    def partition_exists(self, dataset: object, symbol: str, start_time_ms: int) -> bool:
        return False

    def total_bytes(self) -> int:
        return self._bytes


class TestConfigDefaults:
    def test_config_disables_network_sync_by_default(self) -> None:
        from src.application.futures.runner.compound_config import (
            build_compound_run_config,
        )

        config = build_compound_run_config({"sync": "local", "seed": 42})
        assert config.sync == SyncMode.LOCAL
        assert isinstance(config.data_lake, DataLakeConfig)
        assert isinstance(config.universe, PITUniverseConfig)


class TestRuntimeFactory:
    def test_runtime_factory_does_not_download(self, tmp_path: Path) -> None:
        from src.domain.futures.data_lake.query import LocalDataCatalog

        lake_root = tmp_path / "lake"
        writable_catalog = LocalDataCatalog(lake_root)
        writable_catalog._connection.close()
        config = CompoundRunConfig(
            reference_date="2026-07-08",
            sync=SyncMode.LOCAL,
            refresh_universe=False,
            data_lake=DataLakeConfig(root=lake_root),
        )
        runtime = build_data_lake_runtime(config)
        assert isinstance(runtime, DataLakeRuntime)
        assert runtime.client.download_calls == 0


class TestLocalSnapshot:
    def test_complete_local_snapshot_avoids_network(self) -> None:
        snap = _snap(snapshot_id="test-complete")
        runtime = DataLakeRuntime(
            client=FakeClient(),
            catalog=FakeCatalog(snap, complete=True),
        )
        config = CompoundRunConfig(
            reference_date="2026-07-08",
            sync=SyncMode.LOCAL,
            refresh_universe=False,
        )
        result = prepare_data_snapshot(config=config, runtime=runtime)
        assert result.snapshot_id == "test-complete"


class TestCoverageFailure:
    def test_incomplete_cache_without_approval_fails_closed(self) -> None:
        snap = _snap(snapshot_id="test-incomplete")
        runtime = DataLakeRuntime(
            client=FakeClient(),
            catalog=FakeCatalog(snap, complete=False),
        )
        config = CompoundRunConfig(
            reference_date="2026-07-08",
            sync=SyncMode.LOCAL,
            refresh_universe=False,
        )
        with pytest.raises(DataCoverageError):
            prepare_data_snapshot(config=config, runtime=runtime)


def test_prepare_quarterly_bootstrap_returns_window_bound_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.futures.runner.data_lake_runtime as runtime_module

    snapshot = _snap(snapshot_id="bootstrap")
    monkeypatch.setattr(runtime_module, "prepare_data_snapshot", lambda **_: snapshot)
    monkeypatch.setattr(
        runtime_module,
        "audit_funding_partitions",
        lambda **_: type("Audit", (), {"invalid_requests": ()})(),
    )
    config = CompoundRunConfig(reference_date="2026-07-25", sync=SyncMode.LOCAL, refresh_universe=False)
    window = resolve_completed_quarter_window(date(2026, 7, 25), QuarterlyWindowConfig())
    prepared = prepare_quarterly_bootstrap(
        config=config,
        runtime=DataLakeRuntime(client=FakeClient(), catalog=FakeCatalog(snapshot, complete=True)),
        window=window,
    )
    assert prepared.snapshot is snapshot
    assert prepared.window is window


def test_prepare_quarterly_bootstrap_auto_reconciles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.application.futures.runner.data_lake_runtime as runtime_module

    snapshot = _snap(snapshot_id="auto-bootstrap")
    monkeypatch.setattr(runtime_module, "prepare_data_snapshot", lambda **_: snapshot)
    report = type(
        "Report",
        (),
        {"checked_files": 1, "invalid_requests": ()},
    )()
    monkeypatch.setattr(runtime_module, "audit_funding_partitions", lambda **_: report)
    config = CompoundRunConfig(
        reference_date="2026-07-25", sync=SyncMode.AUTO, refresh_universe=False,
        data_lake=DataLakeConfig(root=tmp_path / "lake"),
    )
    window = resolve_completed_quarter_window(date(2026, 7, 25), QuarterlyWindowConfig())
    prepared = prepare_quarterly_bootstrap(
        config=config,
        runtime=DataLakeRuntime(client=FakeClient(), catalog=FakeCatalog(snapshot, complete=True)),
        window=window,
    )
    assert prepared.reconciliation_report is report


def test_auto_bootstrap_repairs_exact_corrupt_funding_partition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import src.application.futures.runner.data_lake_runtime as runtime_module
    import src.domain.futures.data_lake.ingestion as ingestion_module
    from src.domain.futures.data_lake.reconciliation import FundingRepairRequest

    snapshot = _snap(snapshot_id="auto-repaired")
    request = FundingRepairRequest("BTCUSDT", 1714521600000, tmp_path / "quarantine" / "part.parquet")
    audit = type(
        "Report", (), {
            "checked_files": 1,
            "invalid_requests": (request,),
        },
    )()
    calls: list[tuple[FundingRepairRequest, ...]] = []
    monkeypatch.setattr(runtime_module, "prepare_data_snapshot", lambda **_: snapshot)
    monkeypatch.setattr(
        runtime_module,
        "audit_funding_partitions",
        lambda **_: audit if not calls else type("Audit", (), {"invalid_requests": ()})(),
    )
    monkeypatch.setattr(
        ingestion_module,
        "repair_funding_partitions",
        lambda **kwargs: calls.append(kwargs["requests"]) or (),
    )
    config = CompoundRunConfig(
        reference_date="2026-07-25", sync=SyncMode.AUTO, refresh_universe=False,
        data_lake=DataLakeConfig(root=tmp_path / "lake"),
    )
    window = resolve_completed_quarter_window(date(2026, 7, 25), QuarterlyWindowConfig())

    prepare_quarterly_bootstrap(
        config=config,
        runtime=DataLakeRuntime(client=FakeClient(), catalog=FakeCatalog(snapshot, complete=True)),
        window=window,
    )

    assert calls == [(request,)]


def test_auto_bootstrap_fails_closed_when_funding_repair_is_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import src.application.futures.runner.data_lake_runtime as runtime_module
    from src.domain.futures.compound.contracts import FundingDataIntegrityError
    from src.domain.futures.data_lake.reconciliation import FundingRepairRequest

    request = FundingRepairRequest("BTCUSDT", 1714521600000, tmp_path / "part.parquet")
    monkeypatch.setattr(
        runtime_module,
        "audit_funding_partitions",
        lambda **_: type("Audit", (), {"checked_files": 1, "invalid_requests": (request,)})(),
    )
    config = CompoundRunConfig(reference_date="2026-07-25", sync=SyncMode.LOCAL, refresh_universe=False)
    window = resolve_completed_quarter_window(date(2026, 7, 25), QuarterlyWindowConfig())

    with pytest.raises(FundingDataIntegrityError, match="local funding audit failed"):
        prepare_quarterly_bootstrap(
            config=config,
            runtime=DataLakeRuntime(client=FakeClient(), catalog=FakeCatalog(_snap(), complete=True)),
            window=window,
        )


def test_auto_bootstrap_rejects_unresolved_post_repair_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import src.application.futures.runner.data_lake_runtime as runtime_module
    import src.domain.futures.data_lake.ingestion as ingestion_module
    from src.domain.futures.data_lake.reconciliation import FundingRepairRequest

    request = FundingRepairRequest("BTCUSDT", 1714521600000, tmp_path / "part.parquet")
    monkeypatch.setattr(ingestion_module, "repair_funding_partitions", lambda **_: ())
    monkeypatch.setattr(
        runtime_module,
        "audit_funding_partitions",
        lambda **_: type("Audit", (), {"checked_files": 1, "invalid_requests": (request,)})(),
    )
    config = CompoundRunConfig(
        reference_date="2026-07-25", sync=SyncMode.AUTO, refresh_universe=False,
        data_lake=DataLakeConfig(root=tmp_path / "lake"),
    )
    window = resolve_completed_quarter_window(date(2026, 7, 25), QuarterlyWindowConfig())

    with pytest.raises(DataCoverageError, match="funding repair left invalid"):
        prepare_quarterly_bootstrap(
            config=config,
            runtime=DataLakeRuntime(client=FakeClient(), catalog=FakeCatalog(_snap(), complete=True)),
            window=window,
        )


class TestApprovedSync:
    def test_approved_sync_revalidates_snapshot(self, tmp_path: Path) -> None:
        snap = _snap(snapshot_id="test-incomplete")
        client = FakeClient()
        catalog = FakeCatalog(snap, complete=False)
        runtime = DataLakeRuntime(client=client, catalog=catalog)
        config = CompoundRunConfig(
            reference_date="2026-07-08", sync=SyncMode.AUTO, refresh_universe=False,
            data_lake=DataLakeConfig(root=tmp_path / "lake"),
        )
        with pytest.raises(DataCoverageError, match="still incomplete after sync"):
            prepare_data_snapshot(config=config, runtime=runtime)


def test_sync_report_survives_quant_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.futures.runner.data_lake_runtime as runtime_module

    monkeypatch.chdir(tmp_path)
    window = resolve_completed_quarter_window(date(2026, 7, 25), QuarterlyWindowConfig())
    bootstrap = PreparedBootstrap(window=window, snapshot=_snap(), reconciliation_report=None)
    monkeypatch.setattr(runtime_module, "evaluate_layered_coverage", lambda **_: ())
    from src.domain.futures.compound.alpha_catalog import build_multiscale_alpha_catalog
    recipe = build_multiscale_alpha_catalog()[0]
    monkeypatch.setattr(
        runtime_module,
        "resolve_recipe_availability",
        lambda **_: (type("Availability", (), {"status": type("Status", (), {"value": "enabled"})(), "recipe_id": recipe.recipe_id, "reasons": ()})(),),
    )

    from src.domain.futures.universe.contracts import UniverseStateCube

    universe = UniverseStateCube(
        calendar=pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC"),
        instrument_ids=("BTCUSDT",),
        eligible=np.ones((1, 1), dtype=np.bool_),
        entry_block=np.zeros((1, 1), dtype=np.bool_),
        exit_required=np.zeros((1, 1), dtype=np.bool_),
        capacity_usdt=np.ones((1, 1), dtype=np.float64),
        risk_scale=np.ones((1, 1), dtype=np.float64),
        cost_bps=np.ones((1, 1), dtype=np.float64),
    )

    prepared = finalize_quarterly_signal_data(
        config=CompoundRunConfig(reference_date="2026-07-25", sync=SyncMode.LOCAL, refresh_universe=False),
        runtime=DataLakeRuntime(client=FakeClient(), catalog=FakeCatalog(_snap(), complete=True)),
        bootstrap=bootstrap,
        universe=universe,
        catalog=(recipe,),
    )

    report_path = tmp_path / "logs/futures/compound/data_sync_report.json"
    assert report_path.exists()
    assert prepared.downloaded_partitions == 0
    wrapped = type("WrappedUniverse", (), {"state_cube": universe})()
    finalize_quarterly_signal_data(
        config=CompoundRunConfig(reference_date="2026-07-25", sync=SyncMode.LOCAL, refresh_universe=False),
        runtime=DataLakeRuntime(client=FakeClient(), catalog=FakeCatalog(_snap(), complete=True)),
        bootstrap=bootstrap,
        universe=wrapped,
        catalog=(recipe,),
    )
    with pytest.raises(TypeError, match="universe must be UniverseStateCube"):
        finalize_quarterly_signal_data(
            config=CompoundRunConfig(reference_date="2026-07-25", sync=SyncMode.LOCAL, refresh_universe=False),
            runtime=DataLakeRuntime(client=FakeClient(), catalog=FakeCatalog(_snap(), complete=True)),
            bootstrap=bootstrap,
            universe=object(),
            catalog=(recipe,),
        )


def test_finalize_signal_data_excludes_unsupported_cost_calibration_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.application.futures.runner.data_lake_runtime as runtime_module
    from src.domain.futures.compound.alpha_catalog import build_multiscale_alpha_catalog
    from src.domain.futures.universe.contracts import UniverseStateCube

    monkeypatch.chdir(tmp_path)
    window = resolve_completed_quarter_window(date(2026, 7, 25), QuarterlyWindowConfig())
    bootstrap = PreparedBootstrap(window=window, snapshot=_snap(), reconciliation_report=None)
    captured: dict[str, tuple] = {}

    def capture_coverage(**kwargs: object) -> tuple:
        captured["requirements"] = kwargs["requirements"]  # type: ignore[assignment,index]
        return ()

    monkeypatch.setattr(runtime_module, "evaluate_layered_coverage", capture_coverage)
    monkeypatch.setattr(runtime_module, "resolve_recipe_availability", lambda **_: ())
    universe = UniverseStateCube(
        calendar=pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC"),
        instrument_ids=("BTCUSDT",),
        eligible=np.ones((1, 1), dtype=np.bool_),
        entry_block=np.zeros((1, 1), dtype=np.bool_),
        exit_required=np.zeros((1, 1), dtype=np.bool_),
        capacity_usdt=np.ones((1, 1), dtype=np.float64),
        risk_scale=np.ones((1, 1), dtype=np.float64),
        cost_bps=np.full((1, 1), 12.0, dtype=np.float64),
    )

    finalize_quarterly_signal_data(
        config=CompoundRunConfig(reference_date="2026-07-25", sync=SyncMode.LOCAL, refresh_universe=False),
        runtime=DataLakeRuntime(client=FakeClient(), catalog=FakeCatalog(_snap(), complete=True)),
        bootstrap=bootstrap,
        universe=universe,
        catalog=tuple(build_multiscale_alpha_catalog()),
    )

    assert {requirement.dataset.value for requirement in captured["requirements"]} == {
        "klines_1h", "funding_event", "premium_5m", "mark_1m", "index_1m", "metrics_5m", "klines_1m",
    }


class TestChecksumFailure:
    def test_checksum_failure_is_not_committed(self) -> None:
        from src.domain.futures.data_lake.contracts import DatasetKind

        plan = IngestionPlan(
            reference_date=date(2026, 7, 8),
            broad_symbols=("BTCUSDT",),
            selected_symbols=("BTCUSDT",),
            datasets=(DatasetKind.KLINES_1H,),
            config=DataLakeConfig(root=Path("/tmp/lake")),  # noqa: S108
        )
        cat = FakeCatalog(
            _snap(snapshot_id="s1", reference_time_ms=1),
            complete=False,
        )

        class BadClient(FakeClient):
            def download_partition(self, *args: object, **kwargs: object) -> bytes:
                self.download_calls += 1
                return b"tampered"

            def download_checksum(self, *args: object, **kwargs: object) -> str:
                return hashlib.sha256(b"valid").hexdigest()

        with pytest.raises(ChecksumMismatchError):
            sync_futures_data_lake(plan=plan, client=BadClient(), catalog=cat)
        assert len(cat.committed) == 0


class TestHardCap:
    def test_hard_cap_preserves_canonical_partitions(self) -> None:
        from src.domain.futures.data_lake.contracts import DatasetKind

        plan = IngestionPlan(
            reference_date=date(2026, 7, 8),
            broad_symbols=("BTCUSDT",),
            selected_symbols=("BTCUSDT",),
            datasets=(DatasetKind.KLINES_1H,),
            config=DataLakeConfig(root=Path("/tmp/lake"), hard_cap_gib=64),  # noqa: S108
        )

        class FullCatalog(FakeCatalog):
            def total_bytes(self) -> int:
                return 100 * 1024**3

        cat = FullCatalog(
            _snap(snapshot_id="s1", reference_time_ms=1),
            complete=False,
        )
        with pytest.raises(StorageBudgetError):
            sync_futures_data_lake(plan=plan, client=FakeClient(), catalog=cat)
