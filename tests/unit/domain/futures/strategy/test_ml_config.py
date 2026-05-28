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
    assert cfg.ranking_mode == "group_ndcg"


def test_ml_config_accepts_model_family_huber() -> None:
    cfg = StrategyMLConfig(model_family="lgbm_huber", ranking_mode="pointwise")
    assert cfg.model_family == "lgbm_huber"


def test_ml_config_rejects_invalid_ranking_mode() -> None:
    with pytest.raises(ValueError, match="ranking_mode"):
        StrategyMLConfig(ranking_mode="invalid")  # type: ignore[arg-type]


def test_ml_config_rejects_negative_alpha_gate_tolerance() -> None:
    with pytest.raises(ValueError, match="alpha_gate_cost_wall_tolerance_bps"):
        StrategyMLConfig(alpha_gate_cost_wall_tolerance_bps=-0.01)


def test_ml_config_cost_wall_tolerance_default_is_zero() -> None:
    cfg = StrategyMLConfig()
    assert cfg.alpha_gate_cost_wall_tolerance_bps == 0.0


def test_ml_config_rejects_invalid_tradable_thresholds() -> None:
    with pytest.raises(ValueError, match="alpha_gate_min_tradable_long_nz"):
        StrategyMLConfig(alpha_gate_min_tradable_long_nz=1.1)
    with pytest.raises(ValueError, match="alpha_gate_min_tradable_short_nz"):
        StrategyMLConfig(alpha_gate_min_tradable_short_nz=-0.1)


def test_ml_config_rejects_invalid_ev_tail_blend_weight() -> None:
    with pytest.raises(ValueError, match="ev_tail_blend_weight"):
        StrategyMLConfig(ev_tail_blend_weight=1.1)


# ---------------------------------------------------------------------------
# ranker_enabled (T-B)
# ---------------------------------------------------------------------------


def test_strategy_ml_config_ranker_enabled_default() -> None:
    """Default StrategyMLConfig must have ranker_enabled=False (empirically better OOS IC)."""
    # Arrange / Act
    cfg = StrategyMLConfig()

    # Assert
    assert cfg.ranker_enabled is False


def test_strategy_ml_config_ranker_enabled_false() -> None:
    """Setting ranker_enabled=False must not raise any error."""
    # Arrange / Act
    cfg = StrategyMLConfig(ranker_enabled=False)

    # Assert
    assert cfg.ranker_enabled is False


# ---------------------------------------------------------------------------
# Regime gate defaults & validation
# ---------------------------------------------------------------------------


def test_strategy_ml_config_regime_gate_defaults() -> None:
    """Default regime gate fields must be enabled with specified exposure scalars."""
    # Arrange / Act
    cfg = StrategyMLConfig()

    # Assert
    assert cfg.regime_gate_enabled is True
    assert cfg.regime_exposure_bull == pytest.approx(1.0)
    assert cfg.regime_exposure_bear == pytest.approx(0.0)
    assert cfg.regime_exposure_chop == pytest.approx(1.0)


def test_strategy_ml_config_regime_gate_exposure_validation_bull() -> None:
    """regime_exposure_bull > 1.0 must raise ValueError."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="regime_exposure_bull"):
        StrategyMLConfig(regime_exposure_bull=1.5)


def test_strategy_ml_config_regime_gate_exposure_validation_chop() -> None:
    """regime_exposure_chop < 0.0 must raise ValueError."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="regime_exposure_chop"):
        StrategyMLConfig(regime_exposure_chop=-0.1)


def test_strategy_ml_config_regime_gate_exposure_validation_bear() -> None:
    """regime_exposure_bear > 1.0 must raise ValueError."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="regime_exposure_bear"):
        StrategyMLConfig(regime_exposure_bear=2.0)
