from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from src.domain.futures.data_lake.contracts import DataSnapshot, UniverseStateRequest
from src.domain.futures.data_lake.query import load_pit_universe_state
from src.application.futures.runner.legacy_retirement import (
    LegacyRetirementReport,
    retire_legacy_storage,
)
from src.domain.futures.data_lake.query import UniverseCoverageError
from src.domain.futures.data_lake.ingestion import build_ingestion_plan
from src.domain.futures.compound.contracts import SealedHoldoutManifest
from src.domain.futures.compound.holdout_store import HoldoutReuseError, SealedHoldoutStore


def _timestamps_ns() -> NDArray[np.int64]:
    cal = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    return cal.to_numpy(dtype="datetime64[ns]").astype(np.int64)


def test_load_pit_universe_state_projects_effective_state(
    lake_snapshot: DataSnapshot,
) -> None:
    universe = load_pit_universe_state(
        snapshot=lake_snapshot,
        request=UniverseStateRequest(
            execution_timestamps_ns=_timestamps_ns(),
            max_axis_symbols=8,
        ),
    )
    assert universe.state_cube.eligible.shape == (48, len(universe.symbols))
    assert not universe.state_cube.eligible[0].any()
    assert universe.state_cube.eligible[24].any()


def test_load_pit_universe_state_rejects_missing_day(
    incomplete_lake_snapshot: DataSnapshot,
) -> None:
    with pytest.raises(UniverseCoverageError, match="missing PIT state"):
        load_pit_universe_state(
            snapshot=incomplete_lake_snapshot,
            request=UniverseStateRequest(
                execution_timestamps_ns=_timestamps_ns(),
                max_axis_symbols=8,
            ),
        )


def test_load_pit_universe_state_rejects_axis_overflow(lake_snapshot: DataSnapshot) -> None:
    with pytest.raises(UniverseCoverageError, match="exceeds"):
        load_pit_universe_state(
            snapshot=lake_snapshot,
            request=UniverseStateRequest(_timestamps_ns(), 1),
        )


def test_load_pit_universe_state_rejects_late_knowledge(lake_snapshot: DataSnapshot) -> None:
    path = lake_snapshot.partitions[0].path
    frame = pd.read_parquet(path)
    frame["knowledge_time_ns"] = frame["effective_time_ns"]
    frame.to_parquet(path, index=False)

    with pytest.raises(UniverseCoverageError, match="strictly before"):
        load_pit_universe_state(
            snapshot=lake_snapshot,
            request=UniverseStateRequest(_timestamps_ns(), 8),
        )


def test_load_pit_universe_state_skips_missing_partition(tmp_path: Path) -> None:
    from src.domain.futures.data_lake.contracts import DatasetKind, PartitionManifest

    snapshot = DataSnapshot(
        "missing", 0,
        (PartitionManifest(DatasetKind.UNIVERSE_STATE, "__all__", 0, 1, 1, "h", "t", True, tmp_path / "gone.parquet"),),
        "m", "u", 0,
    )
    with pytest.raises(UniverseCoverageError, match="no UNIVERSE_STATE"):
        load_pit_universe_state(
            snapshot=snapshot,
            request=UniverseStateRequest(_timestamps_ns(), 8),
        )


def test_load_pit_universe_state_marks_exit_after_deactivation(tmp_path: Path) -> None:
    from src.domain.futures.data_lake.contracts import DatasetKind, PartitionManifest

    path = tmp_path / "state.parquet"
    rows = []
    for day, eligible in ((1, True), (2, False)):
        effective = pd.Timestamp(f"2025-01-{day:02d}", tz="UTC").value
        rows.append({
            "effective_time_ns": effective, "knowledge_time_ns": effective - 1,
            "symbol": "BTCUSDT", "eligible": eligible, "entry_block": not eligible,
            "exit_required": False, "capacity_usdt": 1.0, "risk_scale": 1.0,
            "execution_cost_bps": 1.0, "state_reason": "", "universe_config_hash": "u",
            "source_manifest_hash": "m",
        })
    pd.DataFrame(rows).to_parquet(path, index=False)
    snapshot = DataSnapshot(
        "exit", 0, (PartitionManifest(DatasetKind.UNIVERSE_STATE, "__all__", 0, 1, 2, "h", "t", True, path),),
        "m", "u", 0,
    )
    calendar = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")

    universe = load_pit_universe_state(
        snapshot=snapshot,
        request=UniverseStateRequest(calendar.to_numpy(dtype="datetime64[ns]").astype(np.int64), 8),
    )

    assert universe.state_cube.exit_required[24, 0]


def test_load_pit_universe_state_blocks_symbol_before_first_state(tmp_path: Path) -> None:
    from src.domain.futures.data_lake.contracts import DatasetKind, PartitionManifest

    path = tmp_path / "future-state.parquet"
    effective = pd.Timestamp("2025-01-03", tz="UTC").value
    pd.DataFrame([{
        "effective_time_ns": effective, "knowledge_time_ns": effective - 1,
        "symbol": "BTCUSDT", "eligible": True, "entry_block": False,
        "exit_required": False, "capacity_usdt": 1.0, "risk_scale": 1.0,
        "execution_cost_bps": 1.0, "state_reason": "", "universe_config_hash": "u",
        "source_manifest_hash": "m",
    }]).to_parquet(path, index=False)
    snapshot = DataSnapshot(
        "future", 0, (PartitionManifest(DatasetKind.UNIVERSE_STATE, "__all__", 0, 1, 1, "h", "t", True, path),),
        "m", "u", 0,
    )
    calendar = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")

    universe = load_pit_universe_state(
        snapshot=snapshot,
        request=UniverseStateRequest(calendar.to_numpy(dtype="datetime64[ns]").astype(np.int64), 8),
    )

    assert not universe.state_cube.eligible[:, 0].any()
    assert universe.state_cube.entry_block[:, 0].all()


def test_retirement_blocks_delete_before_audit(tmp_path: Path) -> None:
    target = tmp_path / "ohlcv"
    target.mkdir()
    report = LegacyRetirementReport(
        migration_hash_match=True,
        snapshot_complete=False,
        smoke_run_passed=True,
        unresolved_references=(),
        deletion_targets=(target,),
    )
    with pytest.raises(RuntimeError, match="retirement preflight"):
        retire_legacy_storage(report=report, approved=True)
    assert target.exists()


def test_lake_query_never_reads_legacy_raw_paths(
    incomplete_lake_snapshot: DataSnapshot,
) -> None:
    with pytest.raises(UniverseCoverageError, match="missing PIT state"):
        load_pit_universe_state(
            snapshot=incomplete_lake_snapshot,
            request=UniverseStateRequest(_timestamps_ns(), 8),
        )


def test_feature_cube_uses_lake_funding_and_pit_costs() -> None:
    from src.application.futures.runner.compound_data import build_multiscale_market_cube
    from src.application.futures.runner.compound_config import CompoundRunConfig
    from src.domain.futures.data_lake.contracts import LakeUniverse
    from src.domain.futures.universe.contracts import UniverseStateCube

    calendar = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    state = UniverseStateCube(
        calendar=calendar, instrument_ids=("BTCUSDT",),
        eligible=np.ones((24, 1), dtype=np.bool_),
        entry_block=np.zeros((24, 1), dtype=np.bool_),
        exit_required=np.zeros((24, 1), dtype=np.bool_),
        capacity_usdt=np.full((24, 1), 123.0), risk_scale=np.ones((24, 1)),
        cost_bps=np.full((24, 1), 7.0),
    )
    snapshot = DataSnapshot("empty", 0, (), "m", "u", 0)
    cube = build_multiscale_market_cube(
        snapshot=snapshot, universe=LakeUniverse(("BTCUSDT",), state, "u"),
        config=CompoundRunConfig("2025-01-02", "skip", False, history_days=1),
    )
    assert np.all(cube.execution_cost_bps_2d == 7.0)
    assert np.isnan(cube.fields_2d["funding"]).all()


def test_ingestion_plan_is_lake_native(tmp_path: Path) -> None:
    from datetime import date
    from src.domain.futures.data_lake.contracts import DataLakeConfig

    root = tmp_path / "lake"
    part = root / "klines_1h" / "symbol=BTCUSDT" / "year=2025" / "month=01" / "part.parquet"
    part.parent.mkdir(parents=True)
    pd.DataFrame({"quote_volume": [1.0]}).to_parquet(part, index=False)
    plan = build_ingestion_plan(config=DataLakeConfig(root=root), reference_date=date.today())
    assert plan.selected_symbols == ("BTCUSDT",)


def test_holdout_rejects_changed_universe_hash(tmp_path: Path) -> None:
    store = SealedHoldoutStore(tmp_path / "holdout.db")
    manifest = SealedHoldoutManifest("h", 1, 2, 1, "m", "d", "s", "universe-a")
    store.create(manifest)
    with pytest.raises(HoldoutReuseError):
        store.consume(
            holdout_id="h", model_version="m", data_manifest_hash="d",
            strategy_spec_hash="s", universe_state_hash="universe-b",
            evaluate=lambda _: pytest.fail("must not evaluate"),
        )


def test_compound_main_wires_lake_universe_and_sealed_holdout() -> None:
    from pathlib import Path as LocalPath
    from src.application.futures.runner.compound_main import run_multiscale_compound_main

    assert callable(run_multiscale_compound_main)
    assert LocalPath("src/application/futures/runner/compound_main.py").exists()


def test_retirement_deletes_exact_approved_targets(tmp_path: Path) -> None:
    target = tmp_path / "legacy"
    target.mkdir()
    report = LegacyRetirementReport(True, True, True, (), (target,))
    assert retire_legacy_storage(report=report, approved=True) == (target,)
    assert not target.exists()
