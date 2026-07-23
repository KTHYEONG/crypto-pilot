from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.application.futures.runner.compound_universe import DailyPITUniverse
from src.application.futures.runner.compound_universe import (
    EmptyPITUniverseError,
    UniverseCoverageError,
    UniverseAxisLimitError,
    build_daily_pit_universe,
    load_or_build_daily_pit_snapshot,
    project_pit_snapshots_to_execution_calendar,
    resolve_compound_universe,
    ensure_universe_ledger_coverage,
    _normalize_symbol_id,
    _date_range_dates,
)
from src.domain.futures.data_lake.contracts import (
    DataSnapshot,
    DatasetKind,
    PartitionManifest,
)
from src.domain.futures.universe.models import SymbolMeta, UniverseSnapshot


def test_daily_pit_universe_defaults() -> None:
    u = DailyPITUniverse(symbols=("BTCUSDT", "ETHUSDT"), decision_dates=())
    assert u.symbols == ("BTCUSDT", "ETHUSDT")
    assert u.decision_dates == ()


def _snapshot(as_of: str, symbols: tuple[str, ...]) -> UniverseSnapshot:
    selected = tuple(
        SymbolMeta(
            symbol=symbol,
            role="core",
            adv_usdt=30_000_000.0,
            execution_cost_bps=10.0,
            funding_carry_8h=0.0,
            beta_vs_market=1.0,
            cluster_id=index,
            tradeable_rank=index + 1,
            basis_annualized_mean=None,
            basis_vol=None,
            capacity_clip_usdt_list=(100_000.0,),
        )
        for index, symbol in enumerate(symbols)
    )
    return UniverseSnapshot(
        as_of=as_of,
        tf="4h",
        schema_version=1,
        config_hash="c1",
        data_manifest_hash="d1",
        basket_ref=symbols,
        basket_weights=tuple(1.0 / len(symbols) for _ in symbols),
        selected=selected,
        rejected={},
        generated_at_utc=f"{as_of}T00:00:00Z",
        ledger_confidence="observed",
        n_stage0=len(symbols),
        n_stage1_pass=len(symbols),
        n_stage2_pass=len(symbols),
        n_stage3_pass=len(symbols),
        n_stage4_pass=len(symbols),
        n_stage5_pass=len(symbols),
        n_stage6_selected=len(symbols),
    )


def test_projection_uses_prior_snapshot_and_marks_membership_exit() -> None:
    symbols = tuple(f"S{index:02d}USDT" for index in range(20))
    later_symbols = symbols[:-1]
    calendar = pd.date_range("2026-01-01", periods=28, freq="h", tz="UTC")

    cube = project_pit_snapshots_to_execution_calendar(
        snapshots=[_snapshot("2026-01-01", symbols), _snapshot("2026-01-02", later_symbols)],
        execution_calendar=calendar,
        max_axis_symbols=240,
    )

    assert cube.instrument_ids == symbols
    assert cube.eligible[0, 0]
    assert cube.capacity_usdt[0, 0] == pytest.approx(100_000.0)
    assert not cube.eligible[-1, -1]
    assert cube.exit_required[0, -1] is np.False_ or not cube.exit_required[0, -1]


def test_projection_rejects_empty_and_oversized_universe() -> None:
    calendar = pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC")

    with pytest.raises(EmptyPITUniverseError):
        project_pit_snapshots_to_execution_calendar(
            snapshots=[], execution_calendar=calendar, max_axis_symbols=240
        )

    symbols = tuple(f"S{index:03d}USDT" for index in range(3))
    with pytest.raises(UniverseAxisLimitError):
        project_pit_snapshots_to_execution_calendar(
            snapshots=[_snapshot("2026-01-01", symbols)],
            execution_calendar=calendar,
            max_axis_symbols=2,
        )


def test_projection_rejects_non_monotonic_snapshot_dates() -> None:
    symbols = tuple(f"S{index:02d}USDT" for index in range(20))
    calendar = pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC")

    with pytest.raises(ValueError, match="non-monotonic"):
        project_pit_snapshots_to_execution_calendar(
            snapshots=[_snapshot("2026-01-01", symbols), _snapshot("2026-01-01", symbols)],
            execution_calendar=calendar,
            max_axis_symbols=240,
        )


def test_daily_pit_universe_requires_twenty_symbols_and_sorts_membership() -> None:
    partitions = tuple(
        PartitionManifest(
            dataset=DatasetKind.KLINES_1H,
            symbol=f"S{index:02d}USDT",
            start_time_ms=1,
            end_time_ms=2,
            row_count=1,
            sha256=f"h{index}",
            source="vision",
            is_final=True,
            path=f"/tmp/S{index:02d}.parquet",
        )
        for index in range(20)
    )
    snapshot = DataSnapshot(
        snapshot_id="s1",
        reference_time_ms=2,
        partitions=partitions,
        manifest_hash="m1",
        total_bytes=20,
    )

    universe = build_daily_pit_universe(snapshot=snapshot, config=object())

    assert len(universe.symbols) == 20
    assert universe.symbols == tuple(sorted(universe.symbols))

    with pytest.raises(EmptyPITUniverseError, match="minimum 20"):
        build_daily_pit_universe(
            snapshot=DataSnapshot(
                snapshot_id="small",
                reference_time_ms=2,
                partitions=partitions[:19],
                manifest_hash="m1",
                total_bytes=19,
            ),
            config=object(),
        )


def test_symbol_normalization_rejects_empty_identifier() -> None:
    with pytest.raises(ValueError, match="empty symbol"):
        _normalize_symbol_id("   ")


def test_date_range_dates_includes_both_boundaries() -> None:
    assert _date_range_dates(date(2026, 1, 1), date(2026, 1, 3)) == [
        date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)
    ]


def test_resolve_compound_universe_builds_historical_union(mocker) -> None:
    symbols = tuple(f"S{index:02d}USDT" for index in range(20))
    config = mocker.Mock(
        reference_date="2026-01-02",
        history_days=1,
        max_axis_symbols=240,
        sync="skip",
    )
    coverage = mocker.Mock(complete=True)
    mocker.patch(
        "src.application.futures.runner.compound_universe.ensure_universe_ledger_coverage",
        return_value=coverage,
    )
    mocker.patch(
        "src.application.futures.runner.compound_universe._date_range_dates",
        return_value=[date(2026, 1, 1)],
    )
    mocker.patch(
        "src.application.futures.runner.compound_universe.load_or_build_daily_pit_snapshot",
        return_value=_snapshot("2026-01-01", symbols),
    )
    calendar = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")

    result = resolve_compound_universe(config=config, execution_calendar=calendar)

    assert result.symbols == symbols
    assert result.coverage is coverage
    assert result.state_cube.instrument_ids == symbols


def test_resolve_compound_universe_fails_without_selected_snapshot(mocker) -> None:
    config = mocker.Mock(
        reference_date="2026-01-02",
        history_days=1,
        max_axis_symbols=240,
        sync="skip",
    )
    mocker.patch(
        "src.application.futures.runner.compound_universe.ensure_universe_ledger_coverage",
        return_value=mocker.Mock(complete=True),
    )
    mocker.patch(
        "src.application.futures.runner.compound_universe._date_range_dates",
        return_value=[date(2026, 1, 1)],
    )
    empty = _snapshot("2026-01-01", ())
    mocker.patch(
        "src.application.futures.runner.compound_universe.load_or_build_daily_pit_snapshot",
        return_value=empty,
    )

    with pytest.raises(EmptyPITUniverseError, match="fallback is forbidden"):
        resolve_compound_universe(
            config=config,
            execution_calendar=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
        )


def test_load_or_build_daily_pit_snapshot_passes_point_in_time_config(mocker) -> None:
    expected = _snapshot("2026-01-01", tuple(f"S{i:02d}USDT" for i in range(20)))
    loader = mocker.patch(
        "src.application.futures.runner.compound_universe.load_or_build_universe_snapshot",
        return_value=(expected, object(), object()),
    )
    config = mocker.Mock(max_daily_symbols=45)

    result = load_or_build_daily_pit_snapshot(
        as_of=date(2026, 1, 1), cfg=config
    )

    assert result is expected
    loader.assert_called_once()
    assert loader.call_args.kwargs["as_of"] == "2026-01-01"
    assert loader.call_args.kwargs["tf"] == "4h"


def test_ledger_coverage_auto_sync_and_pit_range_are_verified(mocker) -> None:
    sync = mocker.patch("src.application.futures.runner.compound_universe.run_historical_sync")
    mocker.patch(
        "src.application.futures.runner.compound_universe.load_ledger_slice",
        return_value=pd.DataFrame({"date": ["2025-01-01", "2026-01-02"]}),
    )
    config = mocker.Mock(sync="auto")

    result = ensure_universe_ledger_coverage(
        config=config, start_date=date(2025, 1, 1), end_date=date(2026, 1, 2)
    )

    assert result.complete
    assert result.synced
    sync.assert_called_once()

    sync.side_effect = RuntimeError("temporary sync failure")
    result = ensure_universe_ledger_coverage(
        config=config, start_date=date(2025, 1, 1), end_date=date(2026, 1, 2)
    )
    assert result.complete
    assert not result.synced


def test_ledger_coverage_rejects_invalid_or_unreadable_cache(mocker) -> None:
    config = mocker.Mock(sync="skip")
    with pytest.raises(UniverseCoverageError, match="start_date"):
        ensure_universe_ledger_coverage(
            config=config, start_date=date(2026, 1, 2), end_date=date(2026, 1, 1)
        )

    mocker.patch(
        "src.application.futures.runner.compound_universe.load_ledger_slice",
        side_effect=RuntimeError("db down"),
    )
    with pytest.raises(UniverseCoverageError, match="cannot read ledger"):
        ensure_universe_ledger_coverage(
            config=config, start_date=date(2026, 1, 1), end_date=date(2026, 1, 2)
        )


def test_ledger_coverage_rejects_empty_or_incomplete_ranges(mocker) -> None:
    config = mocker.Mock(sync="skip")
    loader = mocker.patch(
        "src.application.futures.runner.compound_universe.load_ledger_slice",
        return_value=pd.DataFrame({"date": []}),
    )
    with pytest.raises(UniverseCoverageError, match="ledger empty"):
        ensure_universe_ledger_coverage(
            config=config, start_date=date(2026, 1, 1), end_date=date(2026, 1, 2)
        )
    loader.return_value = pd.DataFrame({"date": ["2026-01-02"]})
    with pytest.raises(UniverseCoverageError, match="covers"):
        ensure_universe_ledger_coverage(
            config=config, start_date=date(2026, 1, 1), end_date=date(2026, 1, 2)
        )
