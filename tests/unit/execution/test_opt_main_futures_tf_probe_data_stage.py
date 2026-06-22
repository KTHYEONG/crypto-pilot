from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.application.futures.optimization.config import FuturesRunConfig
from src.application.futures.optimization.data_readiness import (
    DataReadinessResult,
    DataWindowContract,
)
from src.execution import opt_main_futures


def _window() -> opt_main_futures.QuarterlyWindow:
    return opt_main_futures.QuarterlyWindow(
        fetch_start="2023-01-01",
        is_start="2024-01-01",
        oos_start="2025-10-01",
        end_date="2026-03-31",
        fetch_start_date=datetime.strptime("2023-01-01", "%Y-%m-%d").date(),
        is_start_date=datetime.strptime("2024-01-01", "%Y-%m-%d").date(),
        oos_start_date=datetime.strptime("2025-10-01", "%Y-%m-%d").date(),
        end_date_value=datetime.strptime("2026-03-31", "%Y-%m-%d").date(),
    )


def test_run_data_stage_does_not_require_virtual_probe_tf_parquet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    window = _window()
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2023-01-01", periods=8, freq="4h", tz="UTC"),
            "open": [1.0] * 8,
            "high": [2.0] * 8,
            "low": [0.5] * 8,
            "close": [1.5] * 8,
            "volume": [100.0] * 8,
        }
    )
    data_maps = {"BTCUSDT": {"4h": frame.copy(), "1h": frame.copy()}}
    oos_maps = {"BTCUSDT": {"4h": frame.copy(), "1h": frame.copy()}}

    def _fake_loader(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[
        dict[str, dict[str, pd.DataFrame]],
        dict[str, dict[str, pd.DataFrame]],
        list[str],
    ]:
        captured["target_tfs"] = kwargs.get("target_tfs")
        return data_maps, oos_maps, ["BTCUSDT"]

    def _fake_readiness(**kwargs: Any) -> DataReadinessResult:
        return DataReadinessResult(
            kept_symbols=("BTCUSDT",),
            filtered_is_maps=kwargs["data_maps"],
            filtered_oos_maps=kwargs["oos_data_maps"],
            report=pd.DataFrame({"pass": [True], "reason": ["ok"]}),
            contract=DataWindowContract(
                fetch_start=kwargs["fetch_start"],
                is_start=kwargs["is_start"],
                oos_start=kwargs["oos_start"],
                end=kwargs["end"],
                tf=kwargs["tf"],
                warmup_bars=60,
                require_exec_1m=kwargs["require_exec_1m"],
            ),
        )

    monkeypatch.setattr(opt_main_futures, "FUTURES_ANCHOR_SYMBOLS", ())
    monkeypatch.setattr(opt_main_futures, "FUTURES_MACRO_INDEX_SYMBOLS", ())
    monkeypatch.setattr(opt_main_futures, "load_futures_data_maps_for_symbols", _fake_loader)
    monkeypatch.setattr(opt_main_futures, "evaluate_data_readiness", _fake_readiness)
    opt_cfg = cast(dict[str, Any], opt_main_futures.__dict__["OPT_FUTURES_CONFIG"])
    monkeypatch.setitem(opt_cfg, "ENABLE_TF_PROBE", True)
    monkeypatch.setitem(
        opt_cfg,
        "TF_PROBE_GRID",
        ["1h", "2h", "4h", "6h", "8h", "12h"],
    )

    result = opt_main_futures._run_data_stage(
        cast(FuturesRunConfig, SimpleNamespace(timeframe="4h")),
        window,
        ["BTCUSDT"],
        {},
    )

    assert captured["target_tfs"] is None
    assert result.valid_symbols == ["BTCUSDT"]
