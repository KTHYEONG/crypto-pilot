from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.application.futures.runner.compound_data import (
    build_multiscale_market_cube,
)
from src.domain.futures.data_lake.contracts import (
    DataSnapshot,
    DatasetKind,
    LakeUniverse,
    PartitionManifest,
    SyncMode,
)
from src.domain.futures.universe.contracts import UniverseStateCube


def _lake_universe(symbols: tuple[str, ...], n_bars: int) -> LakeUniverse:
    n_syms = len(symbols)
    calendar = pd.date_range("2026-07-01", periods=n_bars, freq="h", tz="UTC")
    cube = UniverseStateCube(
        calendar=calendar,
        instrument_ids=symbols,
        eligible=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block=np.zeros((n_bars, n_syms), dtype=np.bool_),
        exit_required=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt=np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64),
        risk_scale=np.ones((n_bars, n_syms), dtype=np.float64),
        cost_bps=np.full((n_bars, n_syms), 12.0, dtype=np.float64),
    )
    return LakeUniverse(symbols=symbols, state_cube=cube, state_hash="test-hash")


def _make_snap(partitions: tuple = (), *, hash_val: str = "h1") -> DataSnapshot:
    return DataSnapshot(
        snapshot_id="s1",
        reference_time_ms=1_000_000,
        partitions=partitions,
        manifest_hash=hash_val,
        universe_state_hash="",
        total_bytes=0,
    )


def test_build_multiscale_market_cube_with_empty_snapshot() -> None:
    snap = _make_snap()
    universe = _lake_universe(symbols=("BTCUSDT", "ETHUSDT"), n_bars=24)
    config = CompoundRunConfig(
        reference_date="2026-07-08",
        sync=SyncMode.LOCAL,
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
    assert {"premium", "mark", "index", "taker_buy_quote", "open_interest"}.issubset(
        cube.fields_2d
    )
    assert np.all(np.isnan(cube.fields_2d["close"][:, 0]))


def test_build_multiscale_market_cube_reads_snapshot_partitions(tmp_path: Path) -> None:
    part = tmp_path / "part.parquet"
    ref_date = pd.Timestamp("2026-07-08", tz="UTC")
    dates = pd.date_range(ref_date - pd.Timedelta(days=7), periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({
        "datetime": dates,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "quote_volume": 1_000_000.0,
    })
    df["timestamp"] = dates.as_unit("ns").view(np.int64) // 1_000_000
    df.to_parquet(part, index=False)

    snap = _make_snap(
        partitions=(PartitionManifest(
            DatasetKind.KLINES_1H, "BTCUSDT",
            int(dates[0].timestamp() * 1000), int(dates[-1].timestamp() * 1000),
            len(df), "h", "test", True, part,
        ),),
    )
    universe = _lake_universe(symbols=("BTCUSDT",), n_bars=168)
    config = CompoundRunConfig(
        reference_date="2026-07-08", sync=SyncMode.LOCAL, refresh_universe=False, history_days=7,
    )
    cube = build_multiscale_market_cube(snapshot=snap, universe=universe, config=config)
    assert np.isfinite(cube.fields_2d["close"]).any()


def test_build_multiscale_market_cube_data_manifest_hash() -> None:
    snap = _make_snap(hash_val="expected-hash")
    universe = _lake_universe(symbols=("BTCUSDT",), n_bars=24)
    config = CompoundRunConfig(
        reference_date="2026-07-08",
        sync=SyncMode.LOCAL,
        refresh_universe=False,
        history_days=1,
    )
    cube = build_multiscale_market_cube(
        snapshot=snap, universe=universe, config=config, field_plan=("close",),
    )
    assert cube.data_manifest_hash == "expected-hash"
    assert set(cube.fields_2d) == {"close"}
