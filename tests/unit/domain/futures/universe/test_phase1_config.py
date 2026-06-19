"""Unit tests for Phase 1 PIT Universe — config additions.

Tests:
  - universe_engine field default and pit variant
  - pit_config field default values
  - hash isolation between engines
"""

from __future__ import annotations

import pytest

from src.domain.futures.universe.config import PITUniverseConfig, UniverseConfig


class TestUniverseEngineField:
    """Verify universe_engine and pit_config additions to UniverseConfig."""

    def test_universe_config_default_engine_is_pit(self) -> None:
        # Arrange
        cfg = UniverseConfig()

        # Act / Assert
        assert cfg.universe_engine == "pit"

    def test_universe_config_pit_engine(self) -> None:
        # Arrange / Act
        cfg = UniverseConfig(universe_engine="pit")

        # Assert
        assert cfg.universe_engine == "pit"

    def test_universe_config_hash_differs_between_engines(self) -> None:
        # Arrange
        cfg_stage6 = UniverseConfig(universe_engine="stage6")
        cfg_pit = UniverseConfig(universe_engine="pit")

        # Act
        h_stage6 = cfg_stage6.config_hash()
        h_pit = cfg_pit.config_hash()

        # Assert
        assert h_stage6 != h_pit, "Different engine must produce different config hash"

    def test_universe_config_pit_config_default_factory(self) -> None:
        # Arrange / Act
        cfg = UniverseConfig()

        # Assert — pit_config populated with defaults
        assert isinstance(cfg.pit_config, PITUniverseConfig)
        assert cfg.pit_config.schema_version == 2


class TestPITUniverseConfigDefaults:
    """Verify PITUniverseConfig default values."""

    def test_pit_config_default_values(self) -> None:
        # Arrange / Act
        pit = PITUniverseConfig()

        # Assert
        assert pit.schema_version == 2
        assert pit.max_round_trip_cost_bps == pytest.approx(50.0)
        assert pit.max_market_data_staleness_bars == 1
        assert pit.min_metric_observations == 20
        assert pit.decision_timeframe == "4h"
        assert pit.contract_market == "binance_usdt_perpetual"
        assert pit.default_intended_notional_usdt == pytest.approx(10_000.0)
        assert pit.min_data_confidence == "reconstructed"

    def test_pit_config_is_frozen(self) -> None:
        # Arrange
        pit = PITUniverseConfig()

        # Act / Assert
        with pytest.raises((AttributeError, TypeError)):
            pit.schema_version = 99  # type: ignore[misc]

    def test_universe_config_with_custom_pit_config(self) -> None:
        # Arrange
        custom_pit = PITUniverseConfig(max_round_trip_cost_bps=30.0)
        cfg = UniverseConfig(universe_engine="pit", pit_config=custom_pit)

        # Assert
        assert cfg.pit_config.max_round_trip_cost_bps == pytest.approx(30.0)
