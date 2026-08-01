from __future__ import annotations

import pytest

from src.research.contracts import CostModel, StrategySpec

pytest_plugins = [
    "tests.fixtures.bars",
    "tests.fixtures.cash_carry",
    "tests.fixtures.catalog",
    "tests.fixtures.market_data",
]


@pytest.fixture
def spec() -> StrategySpec:
    return StrategySpec()


@pytest.fixture
def costs() -> CostModel:
    return CostModel()
