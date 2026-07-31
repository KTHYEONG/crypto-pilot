from __future__ import annotations

import pytest

from src.cli import collect_data
from src.cli import run_backtest
from src.cli import run_cash_carry_backtest
from src.cli import run_portfolio_backtest


def test_run_backtest_flags_and_defaults(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        "src.cli.run_backtest.run_baseline_evaluation",
        lambda request: calls.append(request),
    )
    monkeypatch.setattr("sys.argv", ["run_backtest", "--symbol", "ETHUSDT", "--no-log-run"])
    run_backtest.main()
    assert len(calls) == 1
    request = calls[0]
    assert request.symbol == "ETHUSDT"
    assert request.log_run is False
    assert request.unseal_holdout is False


def test_run_portfolio_flags_and_defaults(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        "src.cli.run_portfolio_backtest.run_portfolio_evaluation",
        lambda request: calls.append(request),
    )
    monkeypatch.setattr("sys.argv", [
        "run_portfolio_backtest", "--symbols", "BTCUSDT", "ETHUSDT", "--no-log-run",
    ])
    run_portfolio_backtest.main()
    assert len(calls) == 1
    request = calls[0]
    assert request.symbols == ("BTCUSDT", "ETHUSDT")
    assert request.log_run is False


def test_run_cash_carry_flags_and_defaults(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        "src.cli.run_cash_carry_backtest.run_cash_carry_evaluation",
        lambda request: calls.append(request),
    )
    monkeypatch.setattr("sys.argv", [
        "run_cash_carry_backtest", "run", "--symbol", "BTCUSDT", "--no-log-run",
    ])
    run_cash_carry_backtest.main()
    assert len(calls) == 1
    request = calls[0]
    assert request.symbol == "BTCUSDT"
    assert request.log_run is False


def test_collect_data_surface_commands(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("sys.argv", [
        "collect_data", "spot-ohlcv", "BTCUSDT", "1h", "--start", "2024-01-01",
    ])
    monkeypatch.setattr(
        "src.cli.collect_data.collection.collect_spot_ohlcv",
        lambda *args: calls.append("spot_ohlcv"),
    )
    collect_data.main()
    assert calls == ["spot_ohlcv"]


@pytest.mark.slow
def test_sealed_holdout_policy_shared_across_clis() -> None:
    from src.research.evaluation.policy import HOLDOUT_CUTOFF, resolve_evaluation_end

    assert resolve_evaluation_end(None, unseal_holdout=False) == HOLDOUT_CUTOFF
    with pytest.raises(RuntimeError, match="Holdout sealed"):
        resolve_evaluation_end("2026-01-01", unseal_holdout=False)
    assert resolve_evaluation_end("2026-01-01", unseal_holdout=True) == "2026-01-01"


def test_compare_runs_renders_empty_and_populated(tmp_path, capsys, monkeypatch) -> None:
    import pandas as pd

    from src.cli import compare_runs

    monkeypatch.setattr(compare_runs, "RUNS_LOG_PATH", tmp_path / "runs.jsonl")
    monkeypatch.setattr(compare_runs, "load_runs", lambda: pd.DataFrame())
    monkeypatch.setattr("sys.argv", ["compare_runs", "--last", "5"])
    compare_runs.main()
    assert "No runs recorded yet" in capsys.readouterr().out

    populated = pd.DataFrame([{
        "ts": "2026-07-31T00:00:00+00:00", "git_sha": "abc", "git_dirty": False,
        "symbol": "BTCUSDT", "end": "2025-12-31",
        "metrics.trade_count": 30, "metrics.cagr": 0.2, "metrics.mdd": -0.1,
        "metrics.sharpe": 1.5, "metrics.profit_factor": 1.5, "metrics.win_rate": 0.5,
        "reliability.observation.verdict": "PASS",
        "reliability.observation.lcb90_cagr": 0.16,
        "reliability.fold_distribution.max_period_contribution": 0.2,
        "reliability.stress_test.verdict": "PASS",
    }])
    monkeypatch.setattr(compare_runs, "load_runs", lambda: populated)
    monkeypatch.setattr("sys.argv", ["compare_runs", "--full", "--sort-by", "ts"])
    compare_runs.main()
    assert "BTCUSDT" in capsys.readouterr().out
