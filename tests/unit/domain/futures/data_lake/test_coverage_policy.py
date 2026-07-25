from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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
    exclude_symbols_with_funding_gaps,
    resolve_recipe_availability,
)
from src.domain.futures.data_lake.contracts import (
    DatasetKind,
    DataSnapshot,
    PartitionManifest,
)
from src.domain.futures.universe.contracts import UniverseStateCube


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
    def test_exclude_symbols_with_funding_gaps_masks_only_affected_symbol(self) -> None:
        hour_ns = 3_600_000_000_000
        hour_ms = 3_600_000
        calendar = pd.date_range("2025-01-01", periods=6, freq="h", tz="UTC")
        start_ms = int(calendar[0].value // 1_000_000)
        universe = UniverseStateCube(
            calendar=calendar,
            instrument_ids=("GOODUSDT", "GAPUSDT"),
            eligible=np.ones((6, 2), dtype=np.bool_),
            entry_block=np.zeros((6, 2), dtype=np.bool_),
            exit_required=np.zeros((6, 2), dtype=np.bool_),
            capacity_usdt=np.full((6, 2), 100.0, dtype=np.float64),
            risk_scale=np.ones((6, 2), dtype=np.float64),
            cost_bps=np.full((6, 2), 12.0, dtype=np.float64),
        )
        snapshot = DataSnapshot(
            snapshot_id="test",
            reference_time_ms=0,
            partitions=(
                PartitionManifest(
                    dataset=DatasetKind.FUNDING_EVENT,
                    symbol="GOODUSDT",
                    start_time_ms=start_ms,
                    end_time_ms=start_ms + 6 * hour_ms,
                    row_count=2,
                    sha256="good",
                    source="test",
                    is_final=True,
                    path=Path("/nonexistent/good.parquet"),
                ),
                PartitionManifest(
                    dataset=DatasetKind.FUNDING_EVENT,
                    symbol="GAPUSDT",
                    start_time_ms=start_ms,
                    end_time_ms=start_ms + hour_ms,
                    row_count=1,
                    sha256="gap-a",
                    source="test",
                    is_final=True,
                    path=Path("/nonexistent/gap-a.parquet"),
                ),
                PartitionManifest(
                    dataset=DatasetKind.FUNDING_EVENT,
                    symbol="GAPUSDT",
                    start_time_ms=start_ms + 5 * hour_ms,
                    end_time_ms=start_ms + 6 * hour_ms,
                    row_count=1,
                    sha256="gap-b",
                    source="test",
                    is_final=True,
                    path=Path("/nonexistent/gap-b.parquet"),
                ),
            ),
            manifest_hash="h",
            universe_state_hash="",
            total_bytes=0,
        )

        filtered, excluded = exclude_symbols_with_funding_gaps(
            snapshot=snapshot,
            universe=universe,
            start_time_ns=int(calendar[0].value),
            end_time_ns=int(calendar[-1].value + hour_ns),
            max_gap_ns=hour_ns,
        )

        assert excluded == ("GAPUSDT",)
        assert filtered.eligible[:, 0].tolist() == [True] * 6
        assert filtered.eligible[:, 1].tolist() == [False] * 6
        assert filtered.entry_block[:, 1].tolist() == [True] * 6
        assert filtered.capacity_usdt[:, 1].tolist() == [0.0] * 6

    def test_exclude_symbols_with_funding_gaps_excludes_active_symbol_without_partitions(self) -> None:
        calendar = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        universe = UniverseStateCube(
            calendar=calendar,
            instrument_ids=("EMPTYUSDT",),
            eligible=np.ones((2, 1), dtype=np.bool_),
            entry_block=np.zeros((2, 1), dtype=np.bool_),
            exit_required=np.zeros((2, 1), dtype=np.bool_),
            capacity_usdt=np.full((2, 1), 100.0, dtype=np.float64),
            risk_scale=np.ones((2, 1), dtype=np.float64),
            cost_bps=np.full((2, 1), 12.0, dtype=np.float64),
        )
        snapshot = DataSnapshot("test", 0, (), "h", "", 0)

        filtered, excluded = exclude_symbols_with_funding_gaps(
            snapshot=snapshot,
            universe=universe,
            start_time_ns=int(calendar[0].value),
            end_time_ns=int(calendar[-1].value + 3_600_000_000_000),
            max_gap_ns=3_600_000_000_000,
        )

        assert excluded == ("EMPTYUSDT",)
        assert filtered.eligible[:, 0].tolist() == [False, False]

    def test_exclude_symbols_with_funding_gaps_rejects_invalid_window(self) -> None:
        calendar = pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC")
        universe = UniverseStateCube(
            calendar=calendar,
            instrument_ids=("BTCUSDT",),
            eligible=np.ones((1, 1), dtype=np.bool_),
            entry_block=np.zeros((1, 1), dtype=np.bool_),
            exit_required=np.zeros((1, 1), dtype=np.bool_),
            capacity_usdt=np.full((1, 1), 100.0, dtype=np.float64),
            risk_scale=np.ones((1, 1), dtype=np.float64),
            cost_bps=np.full((1, 1), 12.0, dtype=np.float64),
        )
        snapshot = DataSnapshot("test", 0, (), "h", "", 0)

        with pytest.raises(ValueError, match="end_time_ns"):
            exclude_symbols_with_funding_gaps(
                snapshot=snapshot,
                universe=universe,
                start_time_ns=1,
                end_time_ns=1,
                max_gap_ns=1,
            )

    def test_exclude_symbols_with_funding_gaps_keeps_inactive_symbol_unchanged(self) -> None:
        calendar = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        universe = UniverseStateCube(
            calendar=calendar,
            instrument_ids=("INACTIVEUSDT",),
            eligible=np.zeros((2, 1), dtype=np.bool_),
            entry_block=np.ones((2, 1), dtype=np.bool_),
            exit_required=np.zeros((2, 1), dtype=np.bool_),
            capacity_usdt=np.zeros((2, 1), dtype=np.float64),
            risk_scale=np.ones((2, 1), dtype=np.float64),
            cost_bps=np.full((2, 1), 12.0, dtype=np.float64),
        )
        snapshot = DataSnapshot("test", 0, (), "h", "", 0)

        filtered, excluded = exclude_symbols_with_funding_gaps(
            snapshot=snapshot,
            universe=universe,
            start_time_ns=int(calendar[0].value),
            end_time_ns=int(calendar[-1].value + 3_600_000_000_000),
            max_gap_ns=3_600_000_000_000,
        )

        assert excluded == ()
        assert filtered.eligible[:, 0].tolist() == [False, False]

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

    def test_monthly_partition_duration_is_not_counted_as_gap(self) -> None:
        req = DatasetRequirement(
            dataset=DatasetKind.KLINES_1H,
            fields=("close",),
            criticality=DataCriticality.CORE,
            start_time_ns=1704067200000000000,
            end_time_ns=1709251200000000000,
            min_coverage_ratio=0.1,
            max_gap_ns=6 * 3_600_000_000_000,
            recipe_ids=(),
        )
        snapshot = DataSnapshot(
            snapshot_id="test",
            reference_time_ms=0,
            partitions=(
                PartitionManifest(
                    dataset=DatasetKind.KLINES_1H,
                    symbol="BTCUSDT",
                    start_time_ms=1704067200000,
                    end_time_ms=1706742000000,
                    row_count=744,
                    sha256="a",
                    source="test",
                    is_final=True,
                    path=Path("/nonexistent/january.parquet"),
                ),
                PartitionManifest(
                    dataset=DatasetKind.KLINES_1H,
                    symbol="BTCUSDT",
                    start_time_ms=1706745600000,
                    end_time_ms=1709161200000,
                    row_count=672,
                    sha256="b",
                    source="test",
                    is_final=True,
                    path=Path("/nonexistent/february.parquet"),
                ),
            ),
            manifest_hash="h",
            universe_state_hash="",
            total_bytes=0,
        )
        universe = FakeUniverseStateCube(
            instrument_ids=("BTCUSDT",),
            eligible=np.ones((100, 1), dtype=np.bool_),
        )

        result = evaluate_layered_coverage(snapshot=snapshot, universe=universe, requirements=(req,))

        assert result[0].max_gap_ns == 3_600_000_000_000
        assert result[0].passed

    def test_gap_outside_eligible_window_is_ignored(self) -> None:
        hour = 3_600_000_000_000
        req = DatasetRequirement(
            dataset=DatasetKind.FUNDING_EVENT,
            fields=("funding_rate",),
            criticality=DataCriticality.CORE,
            start_time_ns=0,
            end_time_ns=10 * hour,
            min_coverage_ratio=0.1,
            max_gap_ns=hour,
            recipe_ids=(),
        )
        snapshot = DataSnapshot(
            snapshot_id="test",
            reference_time_ms=0,
            partitions=(
                PartitionManifest(
                    dataset=DatasetKind.FUNDING_EVENT,
                    symbol="BTCUSDT",
                    start_time_ms=0,
                    end_time_ms=2 * 3_600_000,
                    row_count=1,
                    sha256="a",
                    source="test",
                    is_final=True,
                    path=Path("/nonexistent/a.parquet"),
                ),
                PartitionManifest(
                    dataset=DatasetKind.FUNDING_EVENT,
                    symbol="BTCUSDT",
                    start_time_ms=8 * 3_600_000,
                    end_time_ms=10 * 3_600_000,
                    row_count=1,
                    sha256="b",
                    source="test",
                    is_final=True,
                    path=Path("/nonexistent/b.parquet"),
                ),
            ),
            manifest_hash="h",
            universe_state_hash="",
            total_bytes=0,
        )
        eligible = np.zeros((10, 1), dtype=np.bool_)
        eligible[:2, 0] = True
        universe = FakeUniverseStateCube(instrument_ids=("BTCUSDT",), eligible=eligible)

        result = evaluate_layered_coverage(snapshot=snapshot, universe=universe, requirements=(req,))

        assert result[0].max_gap_ns == 0
        assert result[0].passed

    def test_coverage_uses_universe_calendar_for_gap_eligibility(self) -> None:
        hour_ns = 3_600_000_000_000
        hour_ms = 3_600_000
        calendar = pd.date_range("1970-01-01 03:00", periods=3, freq="h", tz="UTC")
        universe = UniverseStateCube(
            calendar=calendar,
            instrument_ids=("BTCUSDT",),
            eligible=np.ones((3, 1), dtype=np.bool_),
            entry_block=np.zeros((3, 1), dtype=np.bool_),
            exit_required=np.zeros((3, 1), dtype=np.bool_),
            capacity_usdt=np.full((3, 1), 100.0, dtype=np.float64),
            risk_scale=np.ones((3, 1), dtype=np.float64),
            cost_bps=np.full((3, 1), 12.0, dtype=np.float64),
        )
        snapshot = DataSnapshot(
            snapshot_id="test",
            reference_time_ms=0,
            partitions=(
                PartitionManifest(
                    dataset=DatasetKind.FUNDING_EVENT,
                    symbol="BTCUSDT",
                    start_time_ms=0,
                    end_time_ms=hour_ms,
                    row_count=1,
                    sha256="a",
                    source="test",
                    is_final=True,
                    path=Path("/nonexistent/a.parquet"),
                ),
                PartitionManifest(
                    dataset=DatasetKind.FUNDING_EVENT,
                    symbol="BTCUSDT",
                    start_time_ms=2 * hour_ms,
                    end_time_ms=6 * hour_ms,
                    row_count=1,
                    sha256="b",
                    source="test",
                    is_final=True,
                    path=Path("/nonexistent/b.parquet"),
                ),
            ),
            manifest_hash="h",
            universe_state_hash="",
            total_bytes=0,
        )
        req = DatasetRequirement(
            dataset=DatasetKind.FUNDING_EVENT,
            fields=("funding_rate",),
            criticality=DataCriticality.CORE,
            start_time_ns=0,
            end_time_ns=6 * hour_ns,
            min_coverage_ratio=0.1,
            max_gap_ns=0,
            recipe_ids=(),
        )

        result = evaluate_layered_coverage(snapshot=snapshot, universe=universe, requirements=(req,))

        assert result[0].max_gap_ns == 0
        assert result[0].passed

    def test_funding_coverage_uses_eight_hour_event_frequency(self) -> None:
        hour_ns = 3_600_000_000_000
        hour_ms = 3_600_000
        calendar = pd.date_range("1970-01-01", periods=24, freq="h", tz="UTC")
        universe = UniverseStateCube(
            calendar=calendar,
            instrument_ids=("BTCUSDT",),
            eligible=np.ones((24, 1), dtype=np.bool_),
            entry_block=np.zeros((24, 1), dtype=np.bool_),
            exit_required=np.zeros((24, 1), dtype=np.bool_),
            capacity_usdt=np.full((24, 1), 100.0, dtype=np.float64),
            risk_scale=np.ones((24, 1), dtype=np.float64),
            cost_bps=np.full((24, 1), 12.0, dtype=np.float64),
        )
        snapshot = DataSnapshot(
            snapshot_id="test",
            reference_time_ms=0,
            partitions=(PartitionManifest(
                dataset=DatasetKind.FUNDING_EVENT,
                symbol="BTCUSDT",
                start_time_ms=0,
                end_time_ms=24 * hour_ms,
                row_count=3,
                sha256="funding",
                source="test",
                is_final=True,
                path=Path("/nonexistent/funding.parquet"),
            ),),
            manifest_hash="h",
            universe_state_hash="",
            total_bytes=0,
        )
        req = DatasetRequirement(
            dataset=DatasetKind.FUNDING_EVENT,
            fields=("funding_rate",),
            criticality=DataCriticality.CORE,
            start_time_ns=0,
            end_time_ns=24 * hour_ns,
            min_coverage_ratio=0.98,
            max_gap_ns=hour_ns,
            recipe_ids=(),
        )

        result = evaluate_layered_coverage(snapshot=snapshot, universe=universe, requirements=(req,))

        assert result[0].expected_observations == 3
        assert result[0].observed_observations == 3
        assert result[0].coverage_ratio == 1.0

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
