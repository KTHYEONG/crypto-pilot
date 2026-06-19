"""Tests for Phase 4-A: k_in cap on PITUniverseConfig and PIT as default engine."""

import pytest

from src.domain.futures.universe.config import PITUniverseConfig, UniverseConfig


def test_pit_universe_config_k_in_defaults_to_50() -> None:
    """PITUniverseConfig.k_in must default to 50."""
    # Arrange / Act
    cfg = PITUniverseConfig()

    # Assert
    assert cfg.k_in == 50


def test_universe_config_engine_defaults_to_pit() -> None:
    """UniverseConfig.universe_engine must default to 'pit'."""
    # Arrange / Act
    cfg = UniverseConfig()

    # Assert
    assert cfg.universe_engine == "pit"


def test_universe_config_k_in_zero_means_no_cap() -> None:
    """UniverseConfig accepts k_in=0 and propagates it through pit_config."""
    # Arrange / Act
    cfg = UniverseConfig(pit_config=PITUniverseConfig(k_in=0))

    # Assert
    assert cfg.pit_config.k_in == 0
