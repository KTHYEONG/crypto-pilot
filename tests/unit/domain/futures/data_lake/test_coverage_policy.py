from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.domain.futures.compound.contracts import (
    AlphaCandidateState,
    MultiscaleAlphaDefinition,
)
from src.domain.futures.data_lake.coverage_policy import (
    DataCoverageError,
    DataCriticality,
    DatasetCoverage,
    DatasetRequirement,
    RecipeDataStatus,
    evaluate_layered_coverage,
    resolve_recipe_availability,
)
from src.domain.futures.data_lake.contracts import (
    DatasetKind,
    DataSnapshot,
    PartitionManifest,
)


@dataclass(slots=True, frozen=True)
class FakeUniverseStateCube:
    instrument_ids: tuple[str, ...]
    eligible: NDArray[np.bool_]


class TestDatasetRequirement:
    def test_valid_optional_requirement(self) -> None:
        req = DatasetRequirement(
            dataset=DatasetKind.MARK_1M,
            fields=("close",),
            criticality=DataCriticality.OPTIONAL,
            start_time_ns=1000,
            end_time_ns=2000,
            min_coverage_ratio=0.5,
            max_gap_ns=86_400_000_000_000,
            recipe_ids=("basis_reversion_1h_h8",),
        )
        assert req.criticality == DataCriticality.OPTIONAL
        assert req.recipe_ids == ("basis_reversion_1h_h8",)

    def test_empty_fields_raises(self) -> None:
        with pytest.raises(ValueError, match="fields must not be empty"):
            DatasetRequirement(
                dataset=DatasetKind.KLINES_1H,
                fields=(),
                criticality=DataCriticality.CORE,
                start_time_ns=0,
                end_time_ns=1000,
                min_coverage_ratio=0.5,
                max_gap_ns=0,
                recipe_ids=(),
            )

    def test_invalid_coverage_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="min_coverage_ratio"):
            DatasetRequirement(
                dataset=DatasetKind.KLINES_1H,
                fields=("close",),
                criticality=DataCriticality.CORE,
                start_time_ns=0,
                end_time_ns=1000,
                min_coverage_ratio=0.0,
                max_gap_ns=0,
                recipe_ids=(),
            )

    def test_negative_max_gap_raises(self) -> None:
        with pytest.raises(ValueError, match="max_gap_ns"):
            DatasetRequirement(
                dataset=DatasetKind.KLINES_1H,
                fields=("close",),
                criticality=DataCriticality.CORE,
                start_time_ns=0,
                end_time_ns=1000,
                min_coverage_ratio=0.5,
                max_gap_ns=-1,
                recipe_ids=(),
            )


class TestEvaluateLayeredCoverage:
    def test_prelisting_absence_is_not_coverage_gap(self) -> None:
        path1 = Path("/nonexistent/part.parquet")
        snapshot = DataSnapshot(
            snapshot_id="test",
            reference_time_ms=0,
            partitions=(PartitionManifest(
                dataset=DatasetKind.KLINES_1H,
                symbol="BTCUSDT",
                start_time_ms=1767225600000,
                end_time_ms=1767312000000,
                row_count=24,
                sha256="h",
                source="test",
                is_final=True,
                path=path1,
            ),),
            manifest_hash="h",
            universe_state_hash="",
            total_bytes=0,
        )
        n_hours = (1767312000000000000 - 1767225600000000000) // 3600000000000
        eligible_2d = np.ones((int(n_hours), 1), dtype=np.bool_)
        universe = FakeUniverseStateCube(
            instrument_ids=("BTCUSDT",),
            eligible=eligible_2d,
        )
        req = DatasetRequirement(
            dataset=DatasetKind.KLINES_1H,
            fields=("close",),
            criticality=DataCriticality.CORE,
            start_time_ns=1767225600000000000,
            end_time_ns=1767312000000000000,
            min_coverage_ratio=0.3,
            max_gap_ns=864_000_000_000_000,
            recipe_ids=(),
        )
        result = evaluate_layered_coverage(
            snapshot=snapshot,
            universe=universe,
            requirements=(req,),
        )
        assert len(result) == 1
        assert result[0].passed

    def test_core_hourly_gap_above_six_hours_fails(self) -> None:
        ns_per_hour = 3_600_000_000_000
        gap_ns = 7 * ns_per_hour
        req = DatasetRequirement(
            dataset=DatasetKind.KLINES_1H,
            fields=("close",),
            criticality=DataCriticality.CORE,
            start_time_ns=0,
            end_time_ns=10 * ns_per_hour,
            min_coverage_ratio=0.98,
            max_gap_ns=6 * ns_per_hour,
            recipe_ids=(),
        )
        snapshot = DataSnapshot(
            snapshot_id="test",
            reference_time_ms=0,
            partitions=(),
            manifest_hash="h",
            universe_state_hash="",
            total_bytes=0,
        )
        eligible_2d = np.ones((100, 1), dtype=np.bool_)
        universe = FakeUniverseStateCube(
            instrument_ids=("BTCUSDT",),
            eligible=eligible_2d,
        )
        with pytest.raises(DataCoverageError, match="CORE"):
            evaluate_layered_coverage(
                snapshot=snapshot,
                universe=universe,
                requirements=(req,),
            )

    def test_optional_failure_does_not_raise(self) -> None:
        req = DatasetRequirement(
            dataset=DatasetKind.MARK_1M,
            fields=("close",),
            criticality=DataCriticality.OPTIONAL,
            start_time_ns=0,
            end_time_ns=86_400_000_000_000,
            min_coverage_ratio=0.9,
            max_gap_ns=0,
            recipe_ids=("basis_reversion_1h_h8",),
        )
        snapshot = DataSnapshot(
            snapshot_id="test",
            reference_time_ms=0,
            partitions=(),
            manifest_hash="h",
            universe_state_hash="",
            total_bytes=0,
        )
        eligible_2d = np.ones((100, 1), dtype=np.bool_)
        universe = FakeUniverseStateCube(
            instrument_ids=("BTCUSDT",),
            eligible=eligible_2d,
        )
        result = evaluate_layered_coverage(
            snapshot=snapshot,
            universe=universe,
            requirements=(req,),
        )
        assert len(result) == 1
        assert not result[0].passed


class TestResolveRecipeAvailability:
    def test_missing_mark_disables_basis_not_trend(self) -> None:
        catalog = (
            MultiscaleAlphaDefinition(
                recipe_id="basis_reversion_1h_h8",
                family="basis_reversion",
                native_timeframe="1h",
                lookback_hours=(24, 72),
                horizon_hours=8,
                required_fields=("mark", "index"),
                initial_state=AlphaCandidateState.SHADOW_RESEARCH,
                max_half_life_hours=4.0,
            ),
            MultiscaleAlphaDefinition(
                recipe_id="ts_trend_4h_h24",
                family="trend",
                native_timeframe="4h",
                lookback_hours=(72, 168),
                horizon_hours=24,
                required_fields=("close", "quote_volume"),
                initial_state=AlphaCandidateState.CORE_CANDIDATE,
                max_half_life_hours=12.0,
            ),
        )

        mark_requirement = DatasetRequirement(
            dataset=DatasetKind.MARK_1M,
            fields=("close",),
            criticality=DataCriticality.OPTIONAL,
            start_time_ns=0,
            end_time_ns=86_400_000_000_000,
            min_coverage_ratio=0.9,
            max_gap_ns=0,
            recipe_ids=("basis_reversion_1h_h8",),
        )

        coverage = (
            DatasetCoverage(
                requirement=mark_requirement,
                expected_observations=100,
                observed_observations=0,
                coverage_ratio=0.0,
                max_gap_ns=86_400_000_000_000,
                passed=False,
                reasons=("coverage_ratio 0.0000 < min 0.9",),
            ),
        )

        result = resolve_recipe_availability(catalog=catalog, coverage=coverage)
        result_map = {r.recipe_id: r.status for r in result}
        assert result_map["basis_reversion_1h_h8"] == RecipeDataStatus.DISABLED_DATA
        assert result_map["ts_trend_4h_h24"] == RecipeDataStatus.ENABLED
