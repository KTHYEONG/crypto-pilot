from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.application.futures.optimization.config import build_run_config_from_args
from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG
from src.domain.futures.strategy.alpha_evaluation import AlphaEvaluationReport
from src.execution import opt_main_futures


def test_strategy_mode_pipeline_orchestration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_config = build_run_config_from_args(
        {
            "mode": "strategy",
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
    ) -> tuple[
        list[str],
        dict[object, frozenset[str]],
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        dict[object, frozenset[str]],
    ]:
        _ = rc
        _ = win
        called.append("universe")
        return ["BTCUSDT"], {}, (), (), ("BTCUSDT",), {}

    def fake_data(
        rc: object,
        win: object,
        discovered: list[str],
        timeline: dict[object, frozenset[str]],
        inference_panel: tuple[str, ...] = (),
        live_inference_panel: tuple[str, ...] = (),
        inference_timeline: dict[object, frozenset[str]] | None = None,
    ) -> opt_main_futures.DataStageResult:
        _ = rc
        _ = win
        _ = discovered
        _ = timeline
        _ = inference_panel
        _ = live_inference_panel
        _ = inference_timeline
        called.append("data")
        return data_stage

    def fake_strategy(
        rc: object,
        win: object,
        ds: object,
        inference_panel: tuple[str, ...] = (),
        live_inference_panel: tuple[str, ...] = (),
        trading_symbols: tuple[str, ...] = (),
    ) -> None:
        _ = rc
        _ = win
        _ = ds
        _ = inference_panel
        _ = live_inference_panel
        _ = trading_symbols
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
    monkeypatch.setattr(opt_main_futures, "_ensure_universe_ledger_sync", lambda *_: None)
    monkeypatch.setattr(
        opt_main_futures,
        "_ensure_cached_symbol_data_for_targets",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(opt_main_futures, "_run_universe_stage", fake_universe)
    monkeypatch.setattr(opt_main_futures, "_run_data_stage", fake_data)
    monkeypatch.setattr(opt_main_futures, "_run_strategy_stage", fake_strategy)
    monkeypatch.setattr(opt_main_futures, "_run_optimization_stage", fake_optimization)

    result = opt_main_futures.run_pipeline(run_config, seed=13, resume=True)
    assert result.exit_code == 0
    assert called == ["window", "universe", "data", "strategy", "optimization"]


def test_alpha_mode_skips_optimization_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alpha 모드는 strategy bridge 후 optimization 없이 종료해야 한다."""
    run_config = build_run_config_from_args(
        {
            "mode": "alpha",
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
    monkeypatch.setattr(opt_main_futures, "_ensure_universe_ledger_sync", lambda *_: None)
    monkeypatch.setattr(
        opt_main_futures,
        "_ensure_cached_symbol_data_for_targets",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        opt_main_futures,
        "_run_universe_stage",
        lambda *_: (["BTCUSDT"], {}, (), (), ("BTCUSDT",), {}),
    )
    monkeypatch.setattr(opt_main_futures, "_run_data_stage", lambda *_: data_stage)
    monkeypatch.setattr(opt_main_futures, "_run_strategy_stage", lambda *_: None)

    def fail_if_called(*args: object, **kwargs: object) -> opt_main_futures.RunnerResult:
        raise AssertionError("optimization stage should not be called in alpha mode")

    monkeypatch.setattr(opt_main_futures, "_run_optimization_stage", fail_if_called)
    result = opt_main_futures.run_pipeline(run_config)
    assert result.exit_code == 0
    assert result.reason == "alpha_evaluation_done"


def test_run_from_cli_when_pipeline_returns_nonzero_propagates_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    argv = ["--mode", "strategy"]
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
    argv = ["--mode", "strategy"]

    def _raise_runtime_error(*_args: object, **_kwargs: object) -> opt_main_futures.RunnerResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(opt_main_futures, "run_pipeline", _raise_runtime_error)

    # Act
    exit_code = opt_main_futures.run_from_cli(argv)

    # Assert
    assert exit_code == 1


def test_requires_exec_1m_returns_false_for_alpha_mode() -> None:
    run_config = build_run_config_from_args(
        {
            "mode": "alpha",
            "symbols": ("BTCUSDT",),
            "tf": "4h",
            "trials": 1,
            "sync_mode": "full_history_master",
        }
    )
    assert opt_main_futures._requires_exec_1m(run_config) is False


def test_summarize_alpha_phase1_verdict_uses_report_passes_not_port_ic() -> None:
    report = AlphaEvaluationReport(
        net_ic=-0.0030,
        net_icir=0.0,
        ic_t_stat_nw=-0.95,
        breakeven_ic=0.0386,
        effective_breadth=1.74,
        net_sharpe=float("nan"),
        quantile_coverage=float("nan"),
        q50_sign_hit=float("nan"),
        per_regime_ic={"bull": 0.011, "bear": 0.024, "chop": -0.006},
        per_regime_breakeven={"bull": 0.01, "bear": 0.01, "chop": 0.01},
        deflated_sharpe=1.0,
        cost_drag={},
        passes=True,
        fail_reasons=[],
        resid_ic=0.0141,
        resid_t_stat_nw=3.62,
        n_eff=15.0,
        breakeven_ic_eff=0.0131,
    )

    verdict = opt_main_futures._summarize_alpha_phase1_verdict(
        report,
        basket_net_bps=-36.94,   # 실행 로그 기준 음수 → G2b FAIL
        basket_ir_t=-2.61,
        sweep_pass_count=0,      # sweep 0/3 → G2d FAIL
    )

    # G1은 passes=True이지만 G2(basket/sweep/gap_raw) FAIL → alpha_pass=False
    assert verdict["alpha_pass"] is False
    assert "portfolio_ic_below_raw_breakeven" in verdict["fail_reasons"]
    assert verdict["blocker_categories"]["rank_skill"] == ["portfolio_ic_below_raw_breakeven"]
    assert verdict["blocker_categories"]["breadth"] == [
        "signal_lost_after_selection",
        "no_profitable_horizon_found",
    ]
    assert verdict["blocker_categories"]["cost_turnover"] == ["basket_net_not_profitable"]
    assert verdict["blocker_categories"]["regime_stability"] == []
    assert verdict["gap_eff"] == pytest.approx(0.0010, abs=1e-6)
    assert verdict["port_ic"] == pytest.approx(-0.0030, abs=1e-6)
    assert verdict["bear_pass"] is True


def test_summarize_exec_diag_verdict_fails_on_negative_portfolio_edge() -> None:
    report = AlphaEvaluationReport(
        net_ic=-0.0030,
        net_icir=0.0,
        ic_t_stat_nw=-0.95,
        breakeven_ic=0.0386,
        effective_breadth=1.74,
        net_sharpe=float("nan"),
        quantile_coverage=float("nan"),
        q50_sign_hit=float("nan"),
        per_regime_ic={"bull": 0.011, "bear": 0.024, "chop": -0.006},
        per_regime_breakeven={"bull": 0.01, "bear": 0.01, "chop": 0.01},
        deflated_sharpe=1.0,
        cost_drag={},
        passes=True,
        fail_reasons=[],
        resid_ic=0.0141,
        resid_t_stat_nw=3.62,
        n_eff=15.0,
        breakeven_ic_eff=0.0131,
    )

    verdict = opt_main_futures._summarize_exec_diag_verdict(
        report=report,
        basket_net_bps=-36.94,
    )

    assert verdict["status"] == "FAIL"
    assert "portfolio_ic_not_positive" in verdict["fail_reasons"]
    assert "portfolio_ic_below_raw_breakeven" in verdict["fail_reasons"]
    assert "basket_net_returns_negative" in verdict["fail_reasons"]


def test_summarize_exec_diag_verdict_passes_on_positive_execution_edge() -> None:
    report = AlphaEvaluationReport(
        net_ic=0.0100,
        net_icir=0.0,
        ic_t_stat_nw=2.5,
        breakeven_ic=0.0060,
        effective_breadth=2.0,
        net_sharpe=float("nan"),
        quantile_coverage=float("nan"),
        q50_sign_hit=float("nan"),
        per_regime_ic={"bull": 0.01, "bear": 0.01, "chop": 0.00},
        per_regime_breakeven={"bull": 0.01, "bear": 0.01, "chop": 0.01},
        deflated_sharpe=0.99,
        cost_drag={},
        passes=True,
        fail_reasons=[],
        resid_ic=0.02,
        resid_t_stat_nw=2.6,
        n_eff=10.0,
        breakeven_ic_eff=0.01,
    )

    verdict = opt_main_futures._summarize_exec_diag_verdict(
        report=report,
        basket_net_bps=5.5,
    )

    assert verdict["status"] == "PASS"
    assert verdict["fail_reasons"] == []


def test_requires_exec_1m_respects_intrabar_config_for_quick_backtest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_config = build_run_config_from_args(
        {
            "mode": "quick-backtest",
            "symbols": ("BTCUSDT",),
            "tf": "4h",
            "trials": 1,
            "sync_mode": "full_history_master",
        }
    )
    monkeypatch.setitem(OPT_FUTURES_CONFIG, "FUTURES_EXECUTION_MODE", "intrabar_1m")
    assert opt_main_futures._requires_exec_1m(run_config) is True


def test_resolve_data_collection_symbols_uses_inference_panel() -> None:
    run_config = build_run_config_from_args(
        {
            "mode": "strategy",
            "tf": "4h",
            "trials": 1,
            "sync_mode": "full_history_master",
        }
    )
    out = opt_main_futures._resolve_data_collection_symbols(
        run_config=run_config,
        discovered_symbols=["AAAUSDT"],
        inference_panel=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        live_inference_panel=("BTCUSDT",),
    )
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        assert sym in out


def test_ensure_universe_ledger_sync_always_passes_none_symbols(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_config = build_run_config_from_args(
        {
            "mode": "strategy",
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
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        opt_main_futures,
        "run_historical_sync",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(opt_main_futures, "FUTURES_DATA_DIR", tmp_path)
    opt_main_futures._ensure_universe_ledger_sync(run_config, window)
    assert calls
    assert calls[0].get("symbols") is None


def test_ensure_cached_symbol_data_uses_fetch_start_for_backfill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_config = build_run_config_from_args(
        {
            "mode": "strategy",
            "tf": "4h",
            "trials": 1,
            "sync_mode": "full_history_master",
        }
    )
    window = opt_main_futures.QuarterlyWindow(
        fetch_start="2022-10-01",
        is_start="2023-10-01",
        oos_start="2025-10-01",
        end_date="2026-03-31",
        fetch_start_date=datetime.strptime("2022-10-01", "%Y-%m-%d").date(),
        is_start_date=datetime.strptime("2023-10-01", "%Y-%m-%d").date(),
        oos_start_date=datetime.strptime("2025-10-01", "%Y-%m-%d").date(),
        end_date_value=datetime.strptime("2026-03-31", "%Y-%m-%d").date(),
    )
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        opt_main_futures,
        "run_historical_sync",
        lambda **kwargs: calls.append(kwargs),
    )
    opt_main_futures._ensure_cached_symbol_data_for_targets(
        run_config,
        window,
        ("BTCUSDT", "ETHUSDT"),
        require_exec_1m=True,
    )

    assert len(calls) == 2
    assert calls[0].get("start_date") == window.fetch_start_date
    assert calls[1].get("start_date") == window.fetch_start_date
