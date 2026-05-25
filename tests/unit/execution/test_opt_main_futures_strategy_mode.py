from __future__ import annotations

from datetime import datetime

import pytest

from src.application.futures.optimization.config import build_run_config_from_args
from src.execution import opt_main_futures


def test_strategy_mode_pipeline_orchestration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_config = build_run_config_from_args(
        {
            "mode": "strategy",
            "strategy": "momentum_v0",
            "symbols": ("BTCUSDT",),
            "tf": "4h",
            "trials": 1,
            "sync_mode": "full_history_master",
        }
    )
    called: list[str] = []
    window = opt_main_futures.QuarterlyWindow(
        fetch_start="2025-01-01",
        is_start="2025-04-01",
        oos_start="2026-01-01",
        end_date="2026-04-01",
        fetch_start_date=datetime.strptime("2025-01-01", "%Y-%m-%d").date(),
        is_start_date=datetime.strptime("2025-04-01", "%Y-%m-%d").date(),
        oos_start_date=datetime.strptime("2026-01-01", "%Y-%m-%d").date(),
        end_date_value=datetime.strptime("2026-04-01", "%Y-%m-%d").date(),
    )
    data_stage = opt_main_futures.DataStageResult(
        data_maps={"BTCUSDT": {}},
        oos_data_maps={"BTCUSDT": {}},
        valid_symbols=["BTCUSDT"],
    )

    def fake_window(reference_date: str | None) -> opt_main_futures.QuarterlyWindow:
        _ = reference_date
        called.append("window")
        return window

    def fake_universe(
        rc: object,
        win: object,
    ) -> tuple[list[str], dict[object, frozenset[str]]]:
        _ = rc
        _ = win
        called.append("universe")
        return ["BTCUSDT"], {}

    def fake_data(
        rc: object,
        win: object,
        discovered: list[str],
        timeline: dict[object, frozenset[str]],
    ) -> opt_main_futures.DataStageResult:
        _ = rc
        _ = win
        _ = discovered
        _ = timeline
        called.append("data")
        return data_stage

    def fake_strategy(
        rc: object,
        win: object,
        ds: object,
    ) -> None:
        _ = rc
        _ = win
        _ = ds
        called.append("strategy")

    def fake_optimization(
        rc: object,
        win: object,
        ds: object,
        *,
        seed: int,
        resume: bool,
    ) -> opt_main_futures.RunnerResult:
        _ = rc
        _ = win
        _ = ds
        _ = seed
        _ = resume
        called.append("optimization")
        return opt_main_futures.RunnerResult(exit_code=0, reason="ok")

    monkeypatch.setattr(opt_main_futures, "_resolve_quarterly_window", fake_window)
    monkeypatch.setattr(opt_main_futures, "_run_universe_stage", fake_universe)
    monkeypatch.setattr(opt_main_futures, "_run_data_stage", fake_data)
    monkeypatch.setattr(opt_main_futures, "_run_strategy_stage", fake_strategy)
    monkeypatch.setattr(opt_main_futures, "_run_optimization_stage", fake_optimization)

    result = opt_main_futures.run_pipeline(run_config, seed=13, resume=True)
    assert result.exit_code == 0
    assert called == ["window", "universe", "data", "strategy", "optimization"]


def test_strategy_smoke_skips_optimization_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_config = build_run_config_from_args(
        {
            "mode": "strategy-smoke",
            "strategy": "eh_st_v1",
            "symbols": ("BTCUSDT",),
            "tf": "4h",
            "trials": 1,
            "sync_mode": "full_history_master",
        }
    )
    window = opt_main_futures.QuarterlyWindow(
        fetch_start="2025-01-01",
        is_start="2025-04-01",
        oos_start="2026-01-01",
        end_date="2026-04-01",
        fetch_start_date=datetime.strptime("2025-01-01", "%Y-%m-%d").date(),
        is_start_date=datetime.strptime("2025-04-01", "%Y-%m-%d").date(),
        oos_start_date=datetime.strptime("2026-01-01", "%Y-%m-%d").date(),
        end_date_value=datetime.strptime("2026-04-01", "%Y-%m-%d").date(),
    )
    data_stage = opt_main_futures.DataStageResult(
        data_maps={"BTCUSDT": {}},
        oos_data_maps={"BTCUSDT": {}},
        valid_symbols=["BTCUSDT"],
    )

    monkeypatch.setattr(opt_main_futures, "_resolve_quarterly_window", lambda _: window)
    monkeypatch.setattr(opt_main_futures, "_run_universe_stage", lambda *_: (["BTCUSDT"], {}))
    monkeypatch.setattr(opt_main_futures, "_run_data_stage", lambda *_: data_stage)
    monkeypatch.setattr(opt_main_futures, "_run_strategy_stage", lambda *_: None)

    def fail_if_called(*args: object, **kwargs: object) -> opt_main_futures.RunnerResult:
        raise AssertionError("optimization stage should not be called in strategy-smoke")

    monkeypatch.setattr(opt_main_futures, "_run_optimization_stage", fail_if_called)
    result = opt_main_futures.run_pipeline(run_config)
    assert result.exit_code == 0
    assert result.reason == "strategy_smoke_done"


def test_run_from_cli_when_pipeline_returns_nonzero_propagates_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    argv = ["--mode", "strategy-smoke", "--strategy", "momentum_v0", "--symbols", "BTCUSDT"]
    monkeypatch.setattr(
        opt_main_futures,
        "run_pipeline",
        lambda *_args, **_kwargs: opt_main_futures.RunnerResult(exit_code=7, reason="failed"),
    )

    # Act
    exit_code = opt_main_futures.run_from_cli(argv)

    # Assert
    assert exit_code == 7


def test_run_from_cli_when_pipeline_raises_runtime_error_returns_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    argv = ["--mode", "strategy-smoke", "--strategy", "momentum_v0", "--symbols", "BTCUSDT"]

    def _raise_runtime_error(*_args: object, **_kwargs: object) -> opt_main_futures.RunnerResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(opt_main_futures, "run_pipeline", _raise_runtime_error)

    # Act
    exit_code = opt_main_futures.run_from_cli(argv)

    # Assert
    assert exit_code == 1
