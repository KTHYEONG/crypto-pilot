from __future__ import annotations

import pytest

from src.domain.futures.allocation.replay import (
    ReversalRiskConfig,
    default_reversal_risk_config,
)


class TestReversalRiskConfig:
    """Scenario 11: Latest reversal default only."""

    def test_default_reversal_risk_config(self) -> None:
        config = default_reversal_risk_config()
        assert isinstance(config, ReversalRiskConfig)
        assert config.enabled is True
        assert config.dd_threshold == pytest.approx(0.12)
        assert config.persistence_bars == 3
        assert config.recovery_cooldown_bars == 8
