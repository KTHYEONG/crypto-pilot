from __future__ import annotations

from datetime import date

import pytest

from src.common.errors import DataIntegrityError
from src.market_data.public_archive.catalog import (
    DATASETS,
    DEFAULT_DATASETS,
    MASK_POLICIES,
    TAPE_SLICE_HOURS,
    daily_key,
    is_perpetual_usdt,
    key_day,
)


def test_daily_key_roundtrip() -> None:
    day = date(2024, 1, 15)
    key = daily_key("metrics", "BTCUSDT", day)
    assert key == "data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2024-01-15.zip"
    assert key_day(key) == day


def test_key_day_rejects_malformed() -> None:
    with pytest.raises(DataIntegrityError):
        key_day("garbage")


def test_first_days() -> None:
    assert DATASETS["metrics"].first_day == date(2020, 9, 1)
    assert DATASETS["bookDepth"].first_day == date(2023, 1, 1)
    assert DATASETS["aggTrades"].first_day == date(2019, 9, 1)


def test_is_perpetual_usdt() -> None:
    assert is_perpetual_usdt("BTCUSDT") is True
    assert is_perpetual_usdt("BTCUSDT_250627") is False
    assert is_perpetual_usdt("ETHBUSD") is False


def test_default_datasets_include_aggtrades() -> None:
    assert DEFAULT_DATASETS == ("metrics", "bookDepth", "aggTrades")
    assert all(name in DATASETS and name in MASK_POLICIES for name in DEFAULT_DATASETS)


def test_aggtrades_mask_breadth_matches_exec_depth_candidates() -> None:
    from src.live.settings import LiveSettings

    declared = LiveSettings.model_fields["exec_depth_candidate_breadth"].default
    policy = MASK_POLICIES["aggTrades"]
    assert policy.breadth == declared == 30
    assert policy.start == date(2023, 1, 1)


def test_tape_slice_window_covers_execution_cycle() -> None:
    start, end = TAPE_SLICE_HOURS
    assert 0 <= start < end <= 24
    assert start <= 23 < end
