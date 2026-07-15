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


def test_pick_strategy_data_maps_merges_is_and_oos_frames_s11() -> None:
    """S11: IS+OOS 병합 — 결과 프레임은 IS 시작부터 OOS 끝까지 정렬·중복없이 이어진다."""
    # Arrange
    is_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2022-04-01", "2022-04-02", "2022-04-03"]),
            "close": [1.0, 2.0, 3.0],
        }
    )
    oos_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2022-04-03", "2022-04-04", "2022-04-05"]),
            "close": [3.0, 4.0, 5.0],
        }
    )
    is_data_maps = {"A": {"4h": is_df, "is_start_idx_4h": 0}}
    oos_data_maps = {"A": {"4h": oos_df}}

    # Act
    result = pick_strategy_data_maps(oos_data_maps, is_data_maps, ["A"], "4h")

    # Assert
    merged = result["A"]["4h"]
    assert isinstance(merged, pd.DataFrame)
    assert merged["datetime"].iloc[0] == pd.Timestamp("2022-04-01")
    assert merged["datetime"].iloc[-1] == pd.Timestamp("2022-04-05")
    assert merged["datetime"].is_monotonic_increasing
    assert merged["datetime"].duplicated().sum() == 0
    assert len(merged) == 5  # 3 + 3 - 1 duplicate (2022-04-03)


def test_pick_strategy_data_maps_missing_oos_symbol_keeps_is_frame_s12() -> None:
    """S12: OOS 누락 심볼 — IS 프레임이 에러 없이 그대로 반환된다."""
    # Arrange
    is_df = pd.DataFrame({"datetime": pd.to_datetime(["2022-04-01", "2022-04-02"]), "close": [1.0, 2.0]})
    is_data_maps = {"A": {"4h": is_df}}
    oos_data_maps: dict[str, dict[str, pd.DataFrame]] = {}

    # Act
    result = pick_strategy_data_maps(oos_data_maps, is_data_maps, ["A"], "4h")

    # Assert
    merged = result["A"]["4h"]
    pd.testing.assert_frame_equal(merged.reset_index(drop=True), is_df.reset_index(drop=True))


def test_pick_strategy_data_maps_filters_by_valid_symbols_s13() -> None:
    """S13: valid_symbols 필터링 — is_data_maps에 3개 심볼이 있어도 2개만 통과한다."""
    # Arrange
    frame = pd.DataFrame({"datetime": pd.to_datetime(["2022-04-01"]), "close": [1.0]})
    is_data_maps = {
        "A": {"4h": frame.copy()},
        "B": {"4h": frame.copy()},
        "C": {"4h": frame.copy()},
    }
    oos_data_maps: dict[str, dict[str, pd.DataFrame]] = {}

    # Act
    result = pick_strategy_data_maps(oos_data_maps, is_data_maps, ["A", "B"], "4h")

    # Assert
    assert sorted(result.keys()) == ["A", "B"]
    assert "C" not in result


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

    run_config = build_run_config_from_args({"phase": "l3", "timeframe": "4h", "trials": 1, "sync": "auto"})
    out = run_active_strategy_output_bridge(
        run_config=run_config,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        tf="4h",
        fetch_start="2025-01-01",
        end_date="2025-04-01",
        opt_config={"FUTURES_STRATEGY_NAME": "candidate_ml"},
        preloaded_data_maps={"BTCUSDT": {}, "ETHUSDT": {}, "SOLUSDT": {}},
        trading_symbols=("BTCUSDT",),
    )

    assert isinstance(out, CandidatePipelineOutput)
    assert captured["symbols"] == ["BTCUSDT"]


def test_run_active_strategy_output_bridge_when_scope_is_empty_raises_value_error() -> None:
    run_config = build_run_config_from_args({"phase": "l3", "timeframe": "4h", "trials": 1, "sync": "auto"})

    with pytest.raises(ValueError, match="candidate ML scope is empty"):
        run_active_strategy_output_bridge(
            run_config=run_config,
            symbols=["BTCUSDT"],
            tf="4h",
            fetch_start="2025-01-01",
            end_date="2025-04-01",
            opt_config={"FUTURES_STRATEGY_NAME": "candidate_ml"},
            preloaded_data_maps={"ETHUSDT": {}},
            trading_symbols=("BTCUSDT",),
        )


def test_run_active_strategy_output_bridge_accepts_l0_phase(
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

    from src.application.futures.run_contracts import FuturesRunConfig
    from src.domain.futures.alpha_foundry.contracts import AlphaFoundryRuntimeConfig

    run_config = FuturesRunConfig(
        timeframe="4h",
        date=None,
        trials=1,
        phase="l0",
        sync="skip",
        refresh_universe=False,
        sync_metrics=False,
        seed=42,
        l0_runtime=AlphaFoundryRuntimeConfig(mode="gate"),
    )
    out = run_active_strategy_output_bridge(
        run_config=run_config,
        symbols=["BTCUSDT"],
        tf="4h",
        fetch_start="2025-01-01",
        end_date="2025-04-01",
        opt_config={"FUTURES_STRATEGY_NAME": "candidate_ml"},
        preloaded_data_maps={"BTCUSDT": {}},
        trading_symbols=("BTCUSDT",),
    )

    assert isinstance(out, CandidatePipelineOutput)
    assert captured["symbols"] == ["BTCUSDT"]
