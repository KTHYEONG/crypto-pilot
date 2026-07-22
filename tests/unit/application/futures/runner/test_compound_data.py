from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.application.futures.runner.compound_data import (
    check_data_readiness,
    load_hourly_data,
    resolve_cached_symbols,
)


def test_empty_data_is_not_ready() -> None:
    assert not check_data_readiness({})


def test_readiness_requires_eighty_percent_symbols() -> None:
    ready = {f"S{i}": {"1h": pd.DataFrame({"close": [1.0]})} for i in range(4)}
    ready["S4"] = {}
    assert check_data_readiness(ready)


def test_resolve_cached_symbols_reads_hourly_cache(tmp_path: Path) -> None:
    hourly = tmp_path / "ohlcv" / "1h"
    hourly.mkdir(parents=True)
    (hourly / "BTCUSDT.parquet").touch()
    assert resolve_cached_symbols(base_dir=tmp_path) == ("BTCUSDT",)


def test_load_hourly_data_returns_empty_map_for_missing_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.application.futures.runner.compound_data.FUTURES_DATA_DIR", tmp_path)
    config = CompoundRunConfig(reference_date="2026-01-01", sync="skip", refresh_universe=False)
    result = load_hourly_data(config, ("BTCUSDT",))
    assert result == {"BTCUSDT": {}}
