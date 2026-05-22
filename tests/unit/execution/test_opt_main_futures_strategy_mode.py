from __future__ import annotations

import pandas as pd
import pytest

from src.domain.futures.strategy_runtime.bridge import MLPipelineOutput
from src.execution.opt_main_futures import (
    _apply_strategy_p0_overrides,
    _assert_strategy_alpha_ready,
    _pick_strategy_data_maps,
    _strategy_smoke_engine_params,
)


def _build_symbol_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=n, freq="4h"),
            "close": [100.0 + i for i in range(n)],
        }
    )


def test_pick_strategy_data_maps_prefers_oos_when_not_empty() -> None:
    oos_df = _build_symbol_df()
    is_df = _build_symbol_df()
    oos_maps = {"BTCUSDT": {"4h": oos_df}}
    is_maps = {"BTCUSDT": {"4h": is_df}}
    chosen = _pick_strategy_data_maps(oos_maps, is_maps, ["BTCUSDT"], "4h")
    assert chosen is oos_maps


def test_pick_strategy_data_maps_falls_back_to_is_when_oos_empty() -> None:
    oos_maps = {"BTCUSDT": {"4h": pd.DataFrame()}}
    is_maps = {"BTCUSDT": {"4h": _build_symbol_df()}}
    chosen = _pick_strategy_data_maps(oos_maps, is_maps, ["BTCUSDT"], "4h")
    assert chosen is is_maps


def test_assert_strategy_alpha_ready_raises_when_panel_empty() -> None:
    with pytest.raises(RuntimeError, match="non-empty alpha_panel"):
        _assert_strategy_alpha_ready(
            ml_out=MLPipelineOutput(alpha_panel=pd.DataFrame()),
            oos_data_maps={"BTCUSDT": {"4h": _build_symbol_df()}},
            valid_symbols=["BTCUSDT"],
            tf="4h",
        )


def test_assert_strategy_alpha_ready_raises_when_merged_values_all_zero() -> None:
    panel = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=5, freq="4h"),
            "symbol": ["BTCUSDT"] * 5,
            "alpha_long": [0.0] * 5,
            "alpha_short": [0.0] * 5,
        }
    ).set_index(["datetime", "symbol"])
    oos_df = _build_symbol_df()
    oos_df["alpha_long"] = 0.0
    oos_df["alpha_short"] = 0.0
    with pytest.raises(RuntimeError, match="zero-only alpha columns"):
        _assert_strategy_alpha_ready(
            ml_out=MLPipelineOutput(alpha_panel=panel),
            oos_data_maps={"BTCUSDT": {"4h": oos_df}},
            valid_symbols=["BTCUSDT"],
            tf="4h",
        )


def test_assert_strategy_alpha_ready_passes_with_non_zero_alpha() -> None:
    panel = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=5, freq="4h"),
            "symbol": ["BTCUSDT"] * 5,
            "alpha_long": [1.0, 0.0, 0.0, 0.0, 0.0],
            "alpha_short": [0.0, 1.0, 0.0, 0.0, 0.0],
        }
    ).set_index(["datetime", "symbol"])
    oos_df = _build_symbol_df()
    oos_df["alpha_long"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    oos_df["alpha_short"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    _assert_strategy_alpha_ready(
        ml_out=MLPipelineOutput(alpha_panel=panel),
        oos_data_maps={"BTCUSDT": {"4h": oos_df}},
        valid_symbols=["BTCUSDT"],
        tf="4h",
    )


def test_apply_strategy_p0_overrides_sets_defaults() -> None:
    cfg: dict[str, float | bool] = {
        "FUTURES_WF_HMM_LEG_REFIT": True,
        "FUTURES_DEFAULT_BETA_ALPHA": 1.0,
        "FUTURES_DEFAULT_EV_HURDLE_BPS": 40.0,
    }
    _apply_strategy_p0_overrides(cfg)
    assert cfg["FUTURES_WF_HMM_LEG_REFIT"] is False
    assert float(cfg["FUTURES_DEFAULT_BETA_ALPHA"]) == 1.0
    assert float(cfg["FUTURES_DEFAULT_EV_HURDLE_BPS"]) == 1.5


def test_strategy_smoke_engine_params_uses_strategy_defaults() -> None:
    params = _strategy_smoke_engine_params("4h")
    assert float(params["BETA_ALPHA"]) == 1.0
    assert float(params["EV_HURDLE_BPS"]) == 1.5
    assert int(params["REBALANCE_BARS"]) == 6
