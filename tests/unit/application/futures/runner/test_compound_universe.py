from __future__ import annotations

import pandas as pd
import pytest

from src.application.futures.runner.compound_universe import (
    EmptyPITUniverseError,
    build_daily_pit_universe,
)
from src.domain.futures.data_lake.contracts import (
    DataSnapshot,
    LakeUniverse,
)
from src.domain.futures.universe.contracts import UniverseStateCube


def _make_snap() -> DataSnapshot:
    return DataSnapshot(
        snapshot_id="s1",
        reference_time_ms=2,
        partitions=(),
        manifest_hash="m1",
        universe_state_hash="test-state-hash",
        total_bytes=0,
    )


def test_build_daily_pit_universe_delegates_to_lake(mocker) -> None:
    n_bars = 24
    n_syms = 3
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    calendar = pd.date_range("2026-07-08", periods=n_bars, freq="h", tz="UTC")
    cube = UniverseStateCube(
        calendar=calendar,
        instrument_ids=symbols,
        eligible=__import__("numpy").ones((n_bars, n_syms), dtype=bool),
        entry_block=__import__("numpy").zeros((n_bars, n_syms), dtype=bool),
        exit_required=__import__("numpy").zeros((n_bars, n_syms), dtype=bool),
        capacity_usdt=__import__("numpy").full((n_bars, n_syms), 1_000_000.0),
        risk_scale=__import__("numpy").ones((n_bars, n_syms)),
        cost_bps=__import__("numpy").full((n_bars, n_syms), 12.0),
    )
    mock_lake = LakeUniverse(symbols=symbols, state_cube=cube, state_hash="h1")
    mocker.patch(
        "src.domain.futures.data_lake.query.load_pit_universe_state",
        return_value=mock_lake,
    )
    from src.application.futures.runner.compound_config import CompoundRunConfig
    config = CompoundRunConfig(
        reference_date="2026-07-08", sync="skip", refresh_universe=False,
        max_axis_symbols=240,
    )
    result = build_daily_pit_universe(
        snapshot=_make_snap(),
        execution_calendar=calendar,
        config=config,
    )
    assert result.symbols == symbols
    assert result.state_hash == "h1"


def test_build_daily_pit_universe_raises_on_empty(mocker) -> None:
    calendar = pd.date_range("2026-07-08", periods=1, freq="h", tz="UTC")
    mock_lake = LakeUniverse(symbols=(), state_cube=mocker.Mock(), state_hash="")
    mocker.patch(
        "src.domain.futures.data_lake.query.load_pit_universe_state",
        return_value=mock_lake,
    )
    from src.application.futures.runner.compound_config import CompoundRunConfig
    config = CompoundRunConfig(
        reference_date="2026-07-08", sync="skip", refresh_universe=False,
    )
    with pytest.raises(EmptyPITUniverseError, match="no eligible symbols"):
        build_daily_pit_universe(
            snapshot=_make_snap(),
            execution_calendar=calendar,
            config=config,
        )
