from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.domain.futures.universe.storage import (
    SymbolSyncProfile,
    _requested_sync_caches_missing,
    _resolve_effective_sync_window,
    run_historical_sync,
    sync_single_symbol_data,
)


class _DummyCollector:
    def ensure_ohlcv_data(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not be called for non-overlapping lifecycle window")


def test_resolve_effective_sync_window_clips_to_onboard_and_delivery_dates() -> None:
    profile = SymbolSyncProfile(
        symbol="X",
        onboard_date=date(2024, 2, 1),
        delivery_date=date(2024, 11, 30),
        status="TRADING",
    )
    window = _resolve_effective_sync_window(
        profile=profile,
        requested_start=date(2024, 1, 1),
        requested_end=date(2024, 12, 31),
    )
    assert window == (date(2024, 2, 1), date(2024, 11, 30))


def test_sync_single_symbol_data_when_no_lifecycle_overlap_skips_symbol() -> None:
    profile = SymbolSyncProfile(
        symbol="X",
        onboard_date=date(2026, 1, 1),
        delivery_date=None,
        status="TRADING",
    )
    rows, count = sync_single_symbol_data(
        symbol="X",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        downloader=None,  # type: ignore[arg-type]
        collector=_DummyCollector(),
        sync_profile=profile,
    )
    assert rows == []
    assert count == 0


def test_requested_sync_caches_missing_detects_missing_requested_timeframes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.domain.futures.universe.storage.FUTURES_DATA_DIR", tmp_path)
    symbol = "XUSDT"
    dt = pd.date_range("2024-01-01", "2024-01-31", freq="4h", tz="UTC")
    pd.DataFrame({"datetime": dt}).to_parquet(tmp_path / f"{symbol}_1h.parquet")

    assert _requested_sync_caches_missing(
        symbol,
        sync_1d=True,
        sync_4h=True,
        sync_1m=False,
        requested_start=date(2024, 1, 1),
        requested_end=date(2024, 1, 31),
    )

    for timeframe in ("1d", "4h"):
        pd.DataFrame({"datetime": dt}).to_parquet(tmp_path / f"{symbol}_{timeframe}.parquet")

    assert not _requested_sync_caches_missing(
        symbol,
        sync_1d=True,
        sync_4h=True,
        sync_1m=False,
        requested_start=date(2024, 1, 1),
        requested_end=date(2024, 1, 31),
    )


def test_requested_sync_caches_missing_when_coverage_is_partial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.domain.futures.universe.storage.FUTURES_DATA_DIR", tmp_path)
    symbol = "YUSDT"
    # only one month coverage: should be treated as missing for wider requested window
    dt = pd.date_range("2026-03-01", "2026-03-31", freq="4h", tz="UTC")
    for timeframe in ("1h", "1d", "4h"):
        pd.DataFrame({"datetime": dt}).to_parquet(tmp_path / f"{symbol}_{timeframe}.parquet")
    assert _requested_sync_caches_missing(
        symbol,
        sync_1d=True,
        sync_4h=True,
        sync_1m=False,
        requested_start=date(2022, 10, 1),
        requested_end=date(2026, 3, 31),
    )


def test_run_historical_sync_when_caches_missing_uses_requested_start_date(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.domain.futures.universe.storage.FUTURES_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        "src.domain.futures.universe.storage._load_symbol_sync_profiles",
        lambda: {},
    )
    monkeypatch.setattr(
        "src.domain.futures.universe.storage._requested_sync_caches_missing",
        lambda *_args, **_kwargs: True,
    )

    # ledger 최신일(last>=end)인데 caches_missing=True 상황 구성
    (tmp_path / "universe_ledger.parquet").touch()
    df_ledger = pd.DataFrame({"symbol": ["AAAUSDT"], "date": ["2026-03-31"]})
    monkeypatch.setattr(
        "src.domain.futures.universe.storage.pd.read_parquet",
        lambda *_args, **_kwargs: df_ledger,
    )

    captured: list[tuple] = []

    def _fake_pool(*_args: object, **_kwargs: object):
        class _Pool:
            def __enter__(self):
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

            def map(self, _worker_fn, tasks):
                captured.extend(tasks)
                return [([], 0) for _ in tasks]

        return _Pool()

    monkeypatch.setattr("src.domain.futures.universe.storage.multiprocessing.Pool", _fake_pool)
    run_historical_sync(
        start_date=date(2022, 10, 1),
        end_date=date(2026, 3, 31),
        symbols=["AAAUSDT"],
        sync_1d=True,
        sync_4h=True,
        sync_1m=False,
    )
    assert captured
    assert captured[0][1] == date(2022, 10, 1)
