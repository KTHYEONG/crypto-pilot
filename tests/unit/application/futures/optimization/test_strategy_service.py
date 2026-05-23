from __future__ import annotations

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

    assert_strategy_alpha_ready(
        ml_out=ml_out,
        oos_data_maps=oos,
        valid_symbols=["BTCUSDT", "ETHUSDT"],
        tf="4h",
    )


def test_run_active_strategy_output_bridge_allows_quick_backtest_neutral() -> None:
    cfg = FuturesRunConfig(
        tf="4h",
        reference_date=None,
        symbols=("BTCUSDT",),
        trials=1,
        mode="quick-backtest",
        strategy=None,
        sync_mode="full_history_master",
        skip_universe=False,
        skip_data_sync=False,
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


def test_run_active_strategy_output_bridge_accepts_strategy_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = FuturesRunConfig(
        tf="4h",
        reference_date=None,
        symbols=("BTCUSDT",),
        trials=1,
        mode="strategy-smoke",
        strategy="ml_lambdamart_v1",
        sync_mode="full_history_master",
        skip_universe=False,
        skip_data_sync=False,
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
        assert strategy_cfg.name == "ml_lambdamart_v1"
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
        opt_config={},
        preloaded_data_maps={},
    )

    assert out is expected
