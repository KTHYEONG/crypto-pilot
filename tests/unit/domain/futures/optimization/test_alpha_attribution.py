from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.optimization.final_evaluator import (
    _build_oos_alpha_attribution_report,
    _rebuild_member_strategy_config,
)
from src.domain.futures.strategy import StrategyConfig, StrategyMLConfig


def test_alpha_attribution_fallback_is_finite_with_missing_data() -> None:
    report = _build_oos_alpha_attribution_report(
        oos_port={"equity_curve": np.array([100.0, 100.5, 100.2], dtype=np.float64)},
        oos_data_maps={},
        symbols=["BTCUSDT"],
        tf="4h",
    )

    assert report["status"] == "ok"
    assert int(report["n_obs"]) == 2
    for key in ("total", "market", "factor_proxy", "residual"):
        assert np.isfinite(float(report["pnl_pct"][key]))
    for key in ("market", "factor_proxy", "residual"):
        assert np.isfinite(float(report["share_pct"][key]))


def test_alpha_attribution_allows_negative_residual_without_failure() -> None:
    eq = np.array([100.0, 99.7, 99.2, 99.0, 98.8], dtype=np.float64)
    report = _build_oos_alpha_attribution_report(
        oos_port={"equity_curve": eq},
        oos_data_maps={},
        symbols=["BTCUSDT"],
        tf="4h",
    )

    assert report["status"] == "ok"
    assert int(report["n_obs"]) == len(eq) - 1
    assert np.isfinite(float(report["pnl_pct"]["residual"]))
    assert float(report["pnl_pct"]["residual"]) < 0.0


def test_rebuild_member_strategy_config_does_not_mutate_frozen_base() -> None:
    base_ml = StrategyMLConfig(label_horizon_bars=6, embargo_bars=6, learning_rate=0.03)
    base_cfg = StrategyConfig(name="ml_lambdamart_v1", ml=base_ml)

    member_cfg = _rebuild_member_strategy_config(base_cfg, {"learning_rate": 0.07})

    assert member_cfg is not base_cfg
    assert member_cfg.ml is not base_cfg.ml
    assert float(member_cfg.ml.learning_rate) == 0.07
    assert float(base_cfg.ml.learning_rate) == 0.03


def test_rebuild_member_strategy_config_ignores_non_ml_params_with_mixed_inputs() -> None:
    base_cfg = StrategyConfig(name="ml_lambdamart_v1", ml=StrategyMLConfig())
    member_cfg = _rebuild_member_strategy_config(
        base_cfg,
        {
            "learning_rate": 0.05,
            "ranker_n_estimators": 900,
            "NOT_A_CONFIG_KEY": 123,
            "STRATEGY_MODE": True,
        },
    )

    assert float(member_cfg.ml.learning_rate) == 0.05
    assert int(member_cfg.ml.ranker_n_estimators) == 900
    assert float(base_cfg.ml.learning_rate) != 0.05


def test_rebuild_member_strategy_config_invalid_scalar_type_fails_fast() -> None:
    base_cfg = StrategyConfig(name="ml_lambdamart_v1", ml=StrategyMLConfig())

    with pytest.raises(ValueError, match="learning_rate"):
        _rebuild_member_strategy_config(base_cfg, {"learning_rate": "bad"})


@pytest.mark.parametrize("ranking_mode", [123, "invalid_mode"])
def test_rebuild_member_strategy_config_invalid_literal_fails_fast(ranking_mode: object) -> None:
    base_cfg = StrategyConfig(name="ml_lambdamart_v1", ml=StrategyMLConfig())

    with pytest.raises(ValueError, match="ranking_mode"):
        _rebuild_member_strategy_config(base_cfg, {"ranking_mode": ranking_mode})


def test_rebuild_member_strategy_config_ignores_unknown_key_and_applies_valid_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    base_cfg = StrategyConfig(name="ml_lambdamart_v1", ml=StrategyMLConfig(learning_rate=0.03))

    with caplog.at_level("WARNING", logger="final_evaluator"):
        member_cfg = _rebuild_member_strategy_config(
            base_cfg,
            {"learning_rate": 0.05, "UNKNOWN_OVERRIDE": 999},
        )

    assert float(member_cfg.ml.learning_rate) == 0.05
    assert "UNKNOWN_OVERRIDE" in caplog.text


def test_rebuild_member_strategy_config_mixed_valid_invalid_known_fails_no_partial_apply() -> None:
    base_cfg = StrategyConfig(name="ml_lambdamart_v1", ml=StrategyMLConfig(learning_rate=0.03))

    with pytest.raises(ValueError, match="ranking_mode"):
        _rebuild_member_strategy_config(
            base_cfg,
            {"learning_rate": 0.05, "ranking_mode": 123},
        )
