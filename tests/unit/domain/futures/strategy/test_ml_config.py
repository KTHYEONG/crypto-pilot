from __future__ import annotations

import pytest

from src.domain.futures.strategy.config import StrategyConfig, StrategyMLConfig


def test_ml_strategy_name_is_supported() -> None:
    cfg = StrategyConfig(name="ml_lambdamart_v1")
    assert cfg.name == "ml_lambdamart_v1"


def test_ml_config_validates_leaf_bound() -> None:
    with pytest.raises(ValueError, match="num_leaves"):
        StrategyMLConfig(num_leaves=64)


def test_ml_config_requires_candidates_when_horizon_experiment_enabled() -> None:
    with pytest.raises(ValueError, match="horizon_candidates"):
        StrategyMLConfig(horizon_experiment_enabled=True, horizon_candidates=())


def test_ml_config_rejects_unknown_feature_group() -> None:
    with pytest.raises(ValueError, match="unsupported feature group"):
        StrategyMLConfig(feature_groups_enabled=("trend", "unknown"))  # type: ignore[arg-type]


def test_ml_config_rejects_invalid_ev_mode() -> None:
    with pytest.raises(ValueError, match="ev_mode"):
        StrategyMLConfig(ev_mode="invalid")  # type: ignore[arg-type]


def test_ml_config_accepts_model_family_default() -> None:
    cfg = StrategyMLConfig()
    assert cfg.model_family == "lgbm_regression"


def test_ml_config_accepts_model_family_huber() -> None:
    cfg = StrategyMLConfig(model_family="lgbm_huber")
    assert cfg.model_family == "lgbm_huber"


def test_ml_config_rejects_negative_alpha_gate_tolerance() -> None:
    with pytest.raises(ValueError, match="alpha_gate_cost_wall_tolerance_bps"):
        StrategyMLConfig(alpha_gate_cost_wall_tolerance_bps=-0.01)


def test_ml_config_cost_wall_tolerance_default_is_zero() -> None:
    cfg = StrategyMLConfig()
    assert cfg.alpha_gate_cost_wall_tolerance_bps == 0.0
