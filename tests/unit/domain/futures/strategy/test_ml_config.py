from __future__ import annotations

import pytest

from src.domain.futures.strategy.config import StrategyConfig, StrategyMLConfig


def test_ml_strategy_name_is_supported() -> None:
    cfg = StrategyConfig(name="ml_lambdamart_v1")
    assert cfg.name == "ml_lambdamart_v1"


def test_ml_config_validates_leaf_bound() -> None:
    with pytest.raises(ValueError, match="num_leaves"):
        StrategyMLConfig(num_leaves=64)
