"""Persisted daily funding is a return contribution, not a currency total."""

import dataclasses

import pandas as pd
import pytest

from src.engine.backtest_persist import persist_strategy_backtest, strategy_daily_frame
from tests.unit.engine.test_strategy_backtest_report import _run


def test_daily_funding_uses_opening_equity_and_income_sign(tmp_path):
    run = _run()
    days = run.evidence.stress_daily.returns.index
    funding = pd.DataFrame({"AAA": -100.0, "BBB": 10.0}, index=days)
    ledger = dataclasses.replace(run.evidence.stress.ledger, funding_by_symbol_daily=funding)
    replay = dataclasses.replace(run.evidence.stress, ledger=ledger)
    run = dataclasses.replace(run, evidence=dataclasses.replace(run.evidence, stress=replay))
    daily = strategy_daily_frame(run)
    assert daily["stress_funding_income_AAA"].iloc[0] == pytest.approx(100 / run.request.initial_equity)
    assert daily["stress_funding_income_BBB"].iloc[0] == pytest.approx(-10 / run.request.initial_equity)
    persist_strategy_backtest(run, tmp_path / "result.json")
    pd.testing.assert_frame_equal(daily, pd.read_parquet(tmp_path / "daily.parquet"))
