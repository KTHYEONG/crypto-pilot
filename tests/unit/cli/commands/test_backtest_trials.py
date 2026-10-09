"""Hermetic neighbor and account trial recording scenarios."""

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.cli.commands.backtest as backtest_mod
from tests.unit.cli.commands._backtest_helpers import (
    _account_namespace,
    _patch_releases_root,
)
from tests.unit.cli.commands.test_backtest import (
    _isolated_release_ledgers as _isolated_release_ledgers,
)


def test_run_neighbor_set_persists_siblings(tmp_path, monkeypatch):
    import src.engine.backtest_persist as persist_mod
    import src.engine.strategy_backtest as run_mod
    from src.common.errors import DataIntegrityError
    from src.strategy.targets import FLOW_MOM_TOP20

    calls = []
    returns = pd.Series(np.linspace(0.001, 0.002, 31), index=pd.date_range("2025-01-01", periods=31, tz="UTC"))
    monkeypatch.setattr(run_mod, "StrategyBacktestRequest", lambda **kwargs: types.SimpleNamespace(**kwargs))
    monkeypatch.setattr(run_mod, "run_strategy_backtest", lambda request: types.SimpleNamespace(
        request=request, evidence=types.SimpleNamespace(base_daily=types.SimpleNamespace(returns=returns))))
    monkeypatch.setattr(backtest_mod, "_strategy_run_statistics", lambda run: {})

    def persist(run, output, **kwargs):
        calls.append(output)
        Path(output).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(persist_mod, "persist_strategy_backtest", persist)
    output = tmp_path / "runs" / "unit" / "result.json"
    output.parent.mkdir(parents=True)
    kwargs = {"source_start": pd.Timestamp("2024-01-01", tz="UTC"),
              "start": pd.Timestamp("2025-01-01", tz="UTC"), "end": pd.Timestamp("2025-02-01", tz="UTC"),
              "base_spec": None, "stress_spec": None, "report_periods": (), "data_root": None,
              "budget": None, "execution_bound": "OHLCV_IMMEDIATE_TAKER", "output": output}
    backtest_mod._run_neighbor_set(FLOW_MOM_TOP20, **kwargs)
    assert len(calls) == 7
    assert sorted(path.parent.name for path in calls) == sorted(f"unit_neighbor{i}" for i in range(7))

    def fail(request):
        raise DataIntegrityError("boom")

    monkeypatch.setattr(run_mod, "run_strategy_backtest", fail)
    with pytest.raises(SystemExit, match="neighbor backtest failed"):
        backtest_mod._run_neighbor_set(FLOW_MOM_TOP20, **kwargs)


def test_account_replay_records_trial(tmp_path, monkeypatch):
    import src.application.strategy_account as account_mod
    from src.application.strategy_account import AccountReplayError

    root = _patch_releases_root(monkeypatch, tmp_path)
    equity = pd.Series(np.linspace(2100, 2200, 31), index=pd.date_range("2025-01-01", periods=31, tz="UTC"))
    report = types.SimpleNamespace(
        result=types.SimpleNamespace(daily_equity=equity, liquidated_at=None, maker_fill_fraction=0.98),
        payload={"moment_source": "stub", "cagr": 0.5, "mdd": 0.05})
    monkeypatch.setattr(account_mod, "run_account_replay", lambda request: report)
    backtest_mod.run_account_replay_command(_account_namespace(tmp_path))
    rows = (root / "flow_mom.trials.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["source"] == "cli"

    def fail(request):
        raise AccountReplayError("replay failed")

    monkeypatch.setattr(account_mod, "run_account_replay", fail)
    with pytest.raises(SystemExit, match="replay failed"):
        backtest_mod.run_account_replay_command(_account_namespace(tmp_path, "2025-03-01", "2025-04-01"))
    rows = (root / "flow_mom.trials.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert json.loads(rows[1])["daily_sharpe"] is None
