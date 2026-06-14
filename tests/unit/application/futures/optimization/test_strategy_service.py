from __future__ import annotations

import pandas as pd
import pytest

from src.application.futures.optimization.config import build_run_config_from_args
from src.application.futures.optimization.strategy_service import (
    assert_candidate_output_ready,
    pick_strategy_data_maps,
    run_active_strategy_output_bridge,
)
from src.domain.futures.strategy_runtime.bridge import CandidatePipelineOutput


def test_pick_strategy_data_maps_strips_is_start_idx() -> None:
    frame = pd.DataFrame({"datetime": pd.to_datetime(["2026-01-01"]), "close": [1.0]})
    picked = pick_strategy_data_maps(
        {"BTCUSDT": {"4h": frame}},
        {"BTCUSDT": {"4h": frame, "is_start_idx_4h": 10, "x": 1}},
        ["BTCUSDT"],
        "4h",
    )
    assert "is_start_idx_4h" not in picked["BTCUSDT"]


def test_assert_candidate_output_ready_requires_target_weight() -> None:
    out = CandidatePipelineOutput(
        alpha_panel=pd.DataFrame({"alpha_long": [0.1]}),
    )
    with pytest.raises(RuntimeError, match="target_weight"):
        assert_candidate_output_ready(
            candidate_out=out,
            oos_data_maps={"BTCUSDT": {"4h": pd.DataFrame({"datetime": pd.to_datetime(["2026-01-01"])})}},
            valid_symbols=["BTCUSDT"],
            tf="4h",
        )


def test_assert_candidate_output_ready_accepts_nonzero_target_weight() -> None:
    panel = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-01"]),
            "symbol": ["BTCUSDT"],
            "target_weight": [0.1],
        }
    ).set_index(["datetime", "symbol"])
    out = CandidatePipelineOutput(alpha_panel=panel)
    oos = {
        "BTCUSDT": {
            "4h": pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-01-01"]),
                    "target_weight": [0.1],
                }
            )
        }
    }
    report = assert_candidate_output_ready(
        candidate_out=out,
        oos_data_maps=oos,
        valid_symbols=["BTCUSDT"],
        tf="4h",
    )
    assert report.merged_symbols == 1


def test_run_active_strategy_output_bridge_uses_stage6_trading_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_build_strategy_config(**_kwargs: object) -> object:
        return object()

    def _fake_run_candidate_strategy_for_universe(**kwargs: object) -> CandidatePipelineOutput:
        captured.update(kwargs)
        return CandidatePipelineOutput()

    monkeypatch.setattr(
        "src.application.futures.optimization.strategy_service.build_candidate_strategy_config",
        _fake_build_strategy_config,
    )
    monkeypatch.setattr(
        "src.application.futures.optimization.strategy_service.run_candidate_strategy_for_universe",
        _fake_run_candidate_strategy_for_universe,
    )

    run_config = build_run_config_from_args(
        {"phase": "l3", "timeframe": "4h", "trials": 1, "sync": "full"}
    )
    out = run_active_strategy_output_bridge(
        run_config=run_config,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        tf="4h",
        fetch_start="2025-01-01",
        end_date="2025-04-01",
        opt_config={"FUTURES_STRATEGY_NAME": "candidate_ml"},
        preloaded_data_maps={"BTCUSDT": {}, "ETHUSDT": {}, "SOLUSDT": {}},
        training_panel=("BTCUSDT", "SOLUSDT"),
        trading_symbols=("BTCUSDT",),
    )

    assert isinstance(out, CandidatePipelineOutput)
    assert captured["symbols"] == ["BTCUSDT"]


def test_run_active_strategy_output_bridge_when_scope_is_empty_raises_value_error() -> None:
    run_config = build_run_config_from_args(
        {"phase": "l3", "timeframe": "4h", "trials": 1, "sync": "full"}
    )

    with pytest.raises(ValueError, match="candidate ML scope is empty"):
        run_active_strategy_output_bridge(
            run_config=run_config,
            symbols=["BTCUSDT"],
            tf="4h",
            fetch_start="2025-01-01",
            end_date="2025-04-01",
            opt_config={"FUTURES_STRATEGY_NAME": "candidate_ml"},
            preloaded_data_maps={"ETHUSDT": {}},
            training_panel=("BTCUSDT",),
            trading_symbols=("BTCUSDT",),
        )
