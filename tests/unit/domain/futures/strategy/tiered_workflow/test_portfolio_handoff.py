from __future__ import annotations

from src.domain.futures.strategy.tiered_workflow.portfolio_handoff import (
    PortfolioHandoffConfig,
)


def test_portfolio_handoff_defaults_are_conservative() -> None:
    config = PortfolioHandoffConfig()
    assert config.max_candidate_sleeves == 32
    assert config.min_calibration_windows == 3
