from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.application.futures.runner.compound_data import (
    build_multiscale_market_cube,
    check_data_readiness,
)
from src.application.futures.runner.compound_universe import DailyPITUniverse
from src.domain.futures.data_lake.contracts import DataSnapshot, DatasetKind, PartitionManifest


def test_empty_data_is_not_ready() -> None:
    assert not check_data_readiness({})


def test_readiness_requires_eighty_percent_symbols() -> None:
    ready = {f"S{i}": {"1h": pd.DataFrame({"close": [1.0]})} for i in range(4)}
    ready["S4"] = {}
    assert check_data_readiness(ready)


def test_build_multiscale_market_cube_with_empty_snapshot() -> None:
    snap = DataSnapshot(
        snapshot_id="empty",
        reference_time_ms=1_000_000,
        partitions=(),
        manifest_hash="h1",
        total_bytes=0,
    )
    universe = DailyPITUniverse(
        symbols=("BTCUSDT", "ETHUSDT"),
        decision_dates=(),
    )
    config = CompoundRunConfig(
        reference_date="2026-07-08",
        sync="skip",
        refresh_universe=False,
        history_days=1,
    )
    cube = build_multiscale_market_cube(
        snapshot=snap, universe=universe, config=config,
    )
    assert cube.timestamps_ns.shape[0] == 24
    assert cube.symbols == ("BTCUSDT", "ETHUSDT")
    assert "close" in cube.fields_2d
    assert "funding" in cube.fields_2d
    assert np.all(np.isnan(cube.fields_2d["close"][:, 0]))


def test_build_multiscale_market_cube_reads_snapshot_partitions(tmp_path: Path) -> None:
    part = tmp_path / "part.parquet"
    ref_date = pd.Timestamp("2026-07-08", tz="UTC")
    dates = pd.date_range(ref_date - pd.Timedelta(days=7), periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({
        "datetime": dates,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "quote_volume": 1_000_000.0, "funding_rate_sum": 0.0001,
    })
    df.to_parquet(part, index=False)

    snap = DataSnapshot(
        snapshot_id="funding-sum",
        reference_time_ms=int(ref_date.timestamp() * 1000),
        partitions=(PartitionManifest(DatasetKind.KLINES_1H, "BTCUSDT", int(dates[0].timestamp() * 1000), int(dates[-1].timestamp() * 1000), len(df), "h", "test", True, part),),
        manifest_hash="h1",
        total_bytes=0,
    )
    universe = DailyPITUniverse(symbols=("BTCUSDT",), decision_dates=())
    config = CompoundRunConfig(
        reference_date="2026-07-08", sync="skip", refresh_universe=False, history_days=7,
    )
    cube = build_multiscale_market_cube(snapshot=snap, universe=universe, config=config)
    assert np.isfinite(cube.fields_2d["close"]).any()


def test_build_multiscale_market_cube_data_manifest_hash() -> None:
    snap = DataSnapshot(
        snapshot_id="s1",
        reference_time_ms=1_000_000,
        partitions=(),
        manifest_hash="expected-hash",
        total_bytes=0,
    )
    universe = DailyPITUniverse(
        symbols=("BTCUSDT",),
        decision_dates=(),
    )
    config = CompoundRunConfig(
        reference_date="2026-07-08",
        sync="skip",
        refresh_universe=False,
        history_days=1,
    )
    cube = build_multiscale_market_cube(
        snapshot=snap, universe=universe, config=config,
    )
    assert cube.data_manifest_hash == "expected-hash"
