from __future__ import annotations

import pytest

from src.domain.futures.strategy.config import CandidateStrategyConfig, StrategyConfig


def test_strategy_config_accepts_only_active_names() -> None:
    StrategyConfig(name="candidate_ml")
    StrategyConfig(name="rule_baseline")

    with pytest.raises(ValueError, match="unsupported strategy name"):
        StrategyConfig(name="momentum")


def test_candidate_strategy_config_name_contract() -> None:
    CandidateStrategyConfig(name="candidate_ml")
    CandidateStrategyConfig(name="rule_baseline")

    with pytest.raises(ValueError, match="candidate strategy name"):
        CandidateStrategyConfig(name="legacy")  # type: ignore[arg-type]
