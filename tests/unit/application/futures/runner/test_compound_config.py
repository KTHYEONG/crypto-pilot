from __future__ import annotations

import argparse

import pytest

from src.application.futures.runner.compound_config import (
    CompoundRunConfig,
    build_compound_run_config,
)


def test_build_compound_config_is_fixed_to_hourly() -> None:
    config = build_compound_run_config(argparse.Namespace(sync="skip", seed=42))
    assert config.base_timeframe == "1h"


def test_allow_network_sync_can_be_enabled() -> None:
    config = build_compound_run_config(argparse.Namespace(
        sync="skip", seed=42, allow_network_sync=True,
    ))
    assert config.allow_network_sync is True


def test_allow_network_sync_disabled_by_default() -> None:
    config = CompoundRunConfig(
        reference_date=None, sync="skip", refresh_universe=False,
    )
    assert config.allow_network_sync is False


def test_new_fields_defaults() -> None:
    config = build_compound_run_config(argparse.Namespace(sync="skip", seed=42))
    assert config.history_days == 730
    assert config.portfolio_nav_usdt == 100_000.0
    assert config.max_daily_symbols == 120
    assert config.max_axis_symbols == 240
    assert config.allow_network_sync is False


def test_positive_new_fields_override() -> None:
    config = build_compound_run_config(argparse.Namespace(
        sync="skip", seed=42, history_days=365, portfolio_nav_usdt=50_000.0,
        max_daily_symbols=30, max_axis_symbols=60, allow_network_sync=True,
    ))
    assert config.history_days == 365
    assert config.portfolio_nav_usdt == 50_000.0
    assert config.max_daily_symbols == 30
    assert config.max_axis_symbols == 60
    assert config.allow_network_sync is True


def test_invalid_history_days_raises() -> None:
    with pytest.raises(ValueError, match="history_days must be >= 1"):
        build_compound_run_config(argparse.Namespace(
            sync="skip", seed=42, history_days=0,
        ))


def test_invalid_nav_raises() -> None:
    with pytest.raises(ValueError, match="portfolio_nav_usdt must be positive"):
        build_compound_run_config(argparse.Namespace(
            sync="skip", seed=42, portfolio_nav_usdt=-1,
        ))


def test_invalid_max_daily_symbols_raises() -> None:
    with pytest.raises(ValueError, match="max_daily_symbols must be >= 1"):
        build_compound_run_config(argparse.Namespace(
            sync="skip", seed=42, max_daily_symbols=0,
        ))


def test_invalid_max_axis_symbols_raises() -> None:
    with pytest.raises(ValueError, match="max_axis_symbols"):
        build_compound_run_config(argparse.Namespace(
            sync="skip", seed=42, max_daily_symbols=10, max_axis_symbols=5,
        ))


def test_negative_max_rss_raises() -> None:
    with pytest.raises(ValueError, match="max_rss_mb must be positive"):
        build_compound_run_config(argparse.Namespace(
            sync="skip", seed=42, max_rss_mb=-1,
        ))


def test_negative_seed_raises() -> None:
    with pytest.raises(ValueError, match="seed out of valid range"):
        build_compound_run_config(argparse.Namespace(
            sync="skip", seed=-1,
        ))
