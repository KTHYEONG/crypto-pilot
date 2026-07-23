from __future__ import annotations

from datetime import date
from pathlib import Path

from src.domain.futures.compound.alpha_catalog import build_multiscale_alpha_catalog
from src.domain.futures.data_lake.contracts import (
    DataLakeConfig,
    DataSnapshot,
    DatasetKind,
    IngestionPlan,
    PartitionManifest,
)
from src.domain.futures.data_lake.coverage import validate_strategy_data_coverage


def test_validate_strategy_data_coverage_marks_missing_recipe_fields() -> None:
    snapshot = DataSnapshot(
        snapshot_id="snapshot",
        reference_time_ms=1,
        universe_state_hash="",
        partitions=(
            PartitionManifest(
                dataset=DatasetKind.KLINES_1H,
                symbol="BTCUSDT",
                start_time_ms=0,
                end_time_ms=1,
                row_count=1,
                sha256="hash",
                source="test",
                is_final=True,
                path=Path("missing.parquet"),
            ),
        ),
        manifest_hash="manifest",
        total_bytes=0,
    )
    plan = IngestionPlan(
        reference_date=date(2026, 1, 1),
        broad_symbols=("BTCUSDT",),
        selected_symbols=("BTCUSDT",),
        datasets=(DatasetKind.KLINES_1H,),
        config=DataLakeConfig(root=Path("/tmp/lake")),
    )
    result = validate_strategy_data_coverage(
        snapshot=snapshot,
        plan=plan,
        catalog=build_multiscale_alpha_catalog(),
    )
    assert not result.all_ready
    assert any(entry.readiness == "shadow" for entry in result.entries)
