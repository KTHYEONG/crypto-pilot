from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.domain.futures.data_lake.contracts import (
    DataSnapshot,
    DatasetKind,
    PartitionManifest,
)


def _write_universe_state_partition(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False, compression="zstd")


@pytest.fixture
def lake_snapshot(tmp_path: Path) -> DataSnapshot:
    """Create a lake snapshot with two daily UNIVERSE_STATE partitions."""
    state_path = tmp_path / DatasetKind.UNIVERSE_STATE.value / "symbol=__all__" / "year=2025" / "month=01" / "part.parquet"
    rows = []
    symbols = [f"S{i:02d}USDT" for i in range(8)]
    for day in range(1, 3):
        eff_ns = pd.Timestamp(f"2025-01-{day:02d}", tz="UTC").as_unit("ns").value
        knl_ns = eff_ns - 1
        rows.extend({
                "effective_time_ns": eff_ns,
                "knowledge_time_ns": knl_ns,
                "symbol": sym,
                "eligible": day >= 2,
                "entry_block": day < 2,
                "exit_required": False,
                "capacity_usdt": 1_000_000.0,
                "risk_scale": 1.0,
                "execution_cost_bps": 10.0,
                "state_reason": "",
                "universe_config_hash": "test",
                "source_manifest_hash": "test",
            } for sym in symbols)
    _write_universe_state_partition(state_path, rows)

    partitions = (
        PartitionManifest(
            dataset=DatasetKind.UNIVERSE_STATE,
            symbol="__all__",
            start_time_ms=1_735_699_200_000,
            end_time_ms=1_735_699_200_000,
            row_count=len(rows),
            sha256="test-hash-1",
            source="test",
            is_final=True,
            path=state_path,
        ),
    )
    return DataSnapshot(
        snapshot_id="lake-test",
        reference_time_ms=1_735_699_200_000,
        partitions=partitions,
        manifest_hash="test-manifest-hash",
        universe_state_hash="test-state-hash",
        total_bytes=0,
    )


@pytest.fixture
def incomplete_lake_snapshot(tmp_path: Path) -> DataSnapshot:
    return DataSnapshot(
        snapshot_id="incomplete-lake",
        reference_time_ms=1_735_699_200_000,
        partitions=(),
        manifest_hash="empty",
        universe_state_hash="",
        total_bytes=0,
    )
