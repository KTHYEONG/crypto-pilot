from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application.futures.optimization.config import FuturesRunConfig
from src.application.futures.optimization.strategy_service import (
    assert_strategy_alpha_ready,
    pick_strategy_data_maps,
    run_active_strategy_output_bridge,
)
from src.domain.futures.strategy.config import StrategyConfig
from src.domain.futures.strategy_runtime.bridge import MLPipelineOutput


def test_pick_strategy_data_maps_always_returns_is_maps_without_is_start_key() -> None:
    """IS maps are returned with is_start_idx stripped so alignment uses fetch_start."""
    frame = pd.DataFrame({"datetime": pd.to_datetime(["2026-01-01"]), "close": [1.0]})
    oos = {"BTCUSDT": {"4h": frame}}
    is_maps = {
        "BTCUSDT": {
            "4h": frame,
            "is_start_idx_4h": 100,
            "other_key": "value",
        }
    }
    picked = pick_strategy_data_maps(oos, is_maps, ["BTCUSDT"], "4h")
    assert "is_start_idx_4h" not in picked["BTCUSDT"]
    assert picked["BTCUSDT"]["other_key"] == "value"
    assert picked["BTCUSDT"]["4h"] is frame


def test_assert_strategy_alpha_ready_rejects_missing_alpha_columns() -> None:
    ml_out = MLPipelineOutput(alpha_panel=pd.DataFrame({"alpha_long": [0.1]}))
    oos = {"BTCUSDT": {"4h": pd.DataFrame({"datetime": pd.to_datetime(["2026-01-01"])})}}
    with pytest.raises(RuntimeError, match="missing required column: alpha_short"):
        assert_strategy_alpha_ready(
            ml_out=ml_out,
            oos_data_maps=oos,
            valid_symbols=["BTCUSDT"],
            tf="4h",
        )


def test_assert_strategy_alpha_ready_accepts_merged_nonzero_long_short() -> None:
    alpha_panel = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "alpha_long": [0.001, 0.0],
            "alpha_short": [0.0, 0.002],
        }
    ).set_index(["datetime", "symbol"])
    ml_out = MLPipelineOutput(alpha_panel=alpha_panel)
    oos = {
        "BTCUSDT": {
            "4h": pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-01-01"]),
                    "alpha_long": [0.001],
                    "alpha_short": [0.0],
                }
            )
        },
        "ETHUSDT": {
            "4h": pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-01-01"]),
                    "alpha_long": [0.0],
                    "alpha_short": [0.002],
                }
            )
        },
    }

    report = assert_strategy_alpha_ready(
        ml_out=ml_out,
        oos_data_maps=oos,
        valid_symbols=["BTCUSDT", "ETHUSDT"],
        tf="4h",
    )
    assert report.merged_symbols == 2


def test_assert_strategy_alpha_ready_preflight_warns_when_target_oos_absent() -> None:
    alpha_panel = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 04:00:00"]),
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "alpha_long": [0.001, 0.001],
            "alpha_short": [0.001, 0.001],
        }
    ).set_index(["datetime", "symbol"])
    ml_out = MLPipelineOutput(alpha_panel=alpha_panel)
    oos = {
        "BTCUSDT": {
            "4h": pd.DataFrame(
                {
                    "datetime": pd.to_datetime(
                        [
                            "2026-01-01 00:00:00",
                            "2026-01-01 04:00:00",
                            "2026-01-01 08:00:00",
                            "2026-01-01 12:00:00",
                        ]
                    ),
                    "alpha_long": [0.001, 0.001, 0.0, 0.0],
                    "alpha_short": [0.001, 0.001, 0.0, 0.0],
                }
            ),
            "oos_start_idx_4h": 3,
        }
    }
    report = assert_strategy_alpha_ready(
        ml_out=ml_out,
        oos_data_maps=oos,
        valid_symbols=["BTCUSDT"],
        tf="4h",
    )
    assert "target_oos_alpha_absent_preflight" in report.warnings


def test_assert_strategy_alpha_ready_requires_target_oos_alpha_when_requested() -> None:
    alpha_panel = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 04:00:00"]),
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "alpha_long": [0.001, 0.001],
            "alpha_short": [0.001, 0.001],
        }
    ).set_index(["datetime", "symbol"])
    ml_out = MLPipelineOutput(alpha_panel=alpha_panel)
    oos = {
        "BTCUSDT": {
            "4h": pd.DataFrame(
                {
                    "datetime": pd.to_datetime(
                        [
                            "2026-01-01 00:00:00",
                            "2026-01-01 04:00:00",
                            "2026-01-01 08:00:00",
                            "2026-01-01 12:00:00",
                        ]
                    ),
                    "alpha_long": [0.001, 0.001, 0.0, 0.0],
                    "alpha_short": [0.001, 0.001, 0.0, 0.0],
                }
            ),
            "oos_start_idx_4h": 3,
        }
    }
    with pytest.raises(RuntimeError, match="strategy target OOS alpha is zero-only"):
        assert_strategy_alpha_ready(
            ml_out=ml_out,
            oos_data_maps=oos,
            valid_symbols=["BTCUSDT"],
            tf="4h",
            require_target_oos_alpha=True,
        )


def test_assert_strategy_alpha_ready_rejects_zero_only_panel() -> None:
    alpha_panel = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 04:00:00"]),
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "alpha_long": [0.0, 0.0],
            "alpha_short": [0.0, 0.0],
        }
    ).set_index(["datetime", "symbol"])
    ml_out = MLPipelineOutput(alpha_panel=alpha_panel)
    oos = {
        "BTCUSDT": {
            "4h": pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 04:00:00"]),
                    "alpha_long": [1.0, 1.0],
                    "alpha_short": [1.0, 1.0],
                }
            )
        }
    }
    with pytest.raises(RuntimeError, match="strategy alpha_panel is zero-only"):
        assert_strategy_alpha_ready(
            ml_out=ml_out,
            oos_data_maps=oos,
            valid_symbols=["BTCUSDT"],
            tf="4h",
        )


def test_assert_strategy_alpha_ready_rejects_zero_only_merged_panel_window() -> None:
    alpha_panel = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 04:00:00"]),
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "alpha_long": [0.001, 0.002],
            "alpha_short": [0.001, 0.002],
        }
    ).set_index(["datetime", "symbol"])
    ml_out = MLPipelineOutput(alpha_panel=alpha_panel)
    oos = {
        "BTCUSDT": {
            "4h": pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 04:00:00"]),
                    "alpha_long": [0.0, 0.0],
                    "alpha_short": [0.0, 0.0],
                }
            )
        }
    }
    with pytest.raises(
        RuntimeError, match="strategy merge produced zero-only alpha columns in panel window"
    ):
        assert_strategy_alpha_ready(
            ml_out=ml_out,
            oos_data_maps=oos,
            valid_symbols=["BTCUSDT"],
            tf="4h",
        )


def test_assert_strategy_alpha_ready_rejects_non_finite_metadata() -> None:
    alpha_panel = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-01"]),
            "symbol": ["BTCUSDT"],
            "alpha_long": [0.001],
            "alpha_short": [0.0],
        }
    ).set_index(["datetime", "symbol"])
    alpha_panel.attrs["alpha_forecast_metadata"] = {
        "q10_long": np.array([0.0], dtype=np.float32),
        "q50_long": np.array([np.inf], dtype=np.float32),
    }
    ml_out = MLPipelineOutput(alpha_panel=alpha_panel)
    oos = {
        "BTCUSDT": {
            "4h": pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-01-01"]),
                    "alpha_long": [0.001],
                    "alpha_short": [0.0],
                }
            )
        }
    }

    with pytest.raises(RuntimeError, match="metadata contains non-finite values"):
        assert_strategy_alpha_ready(
            ml_out=ml_out,
            oos_data_maps=oos,
            valid_symbols=["BTCUSDT"],
            tf="4h",
        )


def test_run_active_strategy_output_bridge_allows_quick_backtest_neutral() -> None:
    cfg = FuturesRunConfig(
        timeframe="4h",
        reference_date=None,
        trials=1,
        mode="quick-backtest",
        sync_mode="full_history_master",
        force_universe_rebuild=False,
    )
    out = run_active_strategy_output_bridge(
        run_config=cfg,
        symbols=["BTCUSDT"],
        tf="4h",
        fetch_start=None,
        end_date=None,
        opt_config={},
        preloaded_data_maps=None,
    )
    assert isinstance(out, MLPipelineOutput)
    assert out.alpha_panel.empty


def test_run_active_strategy_output_bridge_historical_stage5_union_uses_inference_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ml_scope=historical_stage5_union 시 inference_panel을 학습 심볼로 사용해야 한다."""
    # Arrange
    cfg = FuturesRunConfig(
        timeframe="4h",
        reference_date=None,
        trials=1,
        mode="strategy",
        sync_mode="full_history_master",
        force_universe_rebuild=False,
    )
    inference_panel = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> MLPipelineOutput:
        captured["symbols"] = kwargs["symbols"]
        return MLPipelineOutput()

    monkeypatch.setattr(
        "src.application.futures.optimization.strategy_service.run_ml_pipeline_for_universe",
        fake_run,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.legacy.ml_builder.StrategyMLConfig.training_universe_scope",
        "historical_stage5_union",
        raising=False,
    )

    # Act — strategy_cfg 내 scope를 패치하기 위해 StrategyConfig 자체를 mock
    mock_ml = pytest.importorskip("unittest.mock").MagicMock()
    mock_ml.training_universe_scope = "historical_stage5_union"
    mock_ml.trading_symbols = None
    mock_strategy_cfg = pytest.importorskip("unittest.mock").MagicMock()
    mock_strategy_cfg.ml = mock_ml

    monkeypatch.setattr(
        "src.application.futures.optimization.strategy_service.StrategyConfig",
        lambda **kwargs: mock_strategy_cfg,
    )

    out = run_active_strategy_output_bridge(
        run_config=cfg,
        symbols=["BTCUSDT"],
        tf="4h",
        fetch_start=None,
        end_date=None,
        opt_config={},
        preloaded_data_maps={},
        inference_panel=inference_panel,
    )

    # Assert
    assert captured["symbols"] == list(inference_panel)
    assert isinstance(out, MLPipelineOutput)


def test_run_active_strategy_output_bridge_accepts_strategy_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = FuturesRunConfig(
        timeframe="4h",
        reference_date=None,
        trials=1,
        mode="strategy",
        sync_mode="full_history_master",
        force_universe_rebuild=False,
    )
    expected = MLPipelineOutput(
        alpha_panel=pd.DataFrame(
            {
                "alpha_long": [0.001],
                "alpha_short": [0.0],
            },
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2026-01-01"), "BTCUSDT")],
                names=["datetime", "symbol"],
            ),
        )
    )

    def fake_run_ml_pipeline_for_universe(**kwargs: object) -> MLPipelineOutput:
        strategy_cfg = kwargs["strategy_cfg"]
        assert isinstance(strategy_cfg, StrategyConfig)
        assert strategy_cfg.name == "lambdamart"
        assert kwargs["preloaded_data_maps"] == {}
        return expected

    monkeypatch.setattr(
        "src.application.futures.optimization.strategy_service.run_ml_pipeline_for_universe",
        fake_run_ml_pipeline_for_universe,
    )

    out = run_active_strategy_output_bridge(
        run_config=cfg,
        symbols=["BTCUSDT"],
        tf="4h",
        fetch_start=None,
        end_date=None,
        opt_config={"FUTURES_STRATEGY_NAME": "lambdamart"},
        preloaded_data_maps={},
    )

    assert out is expected
