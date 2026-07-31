from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.types import CostModel, StrategySpec
from src.data.loader import DataIntegrityError, load_ohlcv_4h
from src.engine.backtest import run_backtest
from src.validation.metrics import compute_metrics
from src.validation.reliability_gate import (
    compute_reliability_gate,
    compute_stress_test_gate,
)

BTC_PATH = Path("data/futures/ohlcv/1h/BTCUSDT.parquet")
FUNDING_PATH = Path("data/futures/funding/BTCUSDT.parquet")


def _funding_rates(path: Path, bar_index: pd.DatetimeIndex) -> pd.Series:
    """Load funding scoped to the backtest bar window (mirrors the CLI)."""
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["datetime"], utc=True)
    rates = pd.to_numeric(df["funding_rate"], errors="coerce")
    series = pd.Series(rates.to_numpy(dtype="float64"), index=pd.DatetimeIndex(ts)).sort_index()
    window_end = bar_index[-1] + (bar_index[1] - bar_index[0])
    return series[(series.index >= bar_index[0]) & (series.index < window_end)]


@pytest.mark.slow
class TestTakerFlowCandidate:
    def test_candidate_retains_min_trades_and_stress_measured(self) -> None:
        # SC-FLOW-05: on the sealed BTC observation window the .52 candidate keeps
        # the reliability-gate minimum trade count and its stress result is
        # measured net of funding.
        df = load_ohlcv_4h(BTC_PATH, end="2025-12-31 23:59:59")
        funding_rates = _funding_rates(FUNDING_PATH, df.index)
        spec = StrategySpec(min_taker_buy_ratio=0.52)
        costs = CostModel()

        result = run_backtest(df, spec, costs, funding_rates=funding_rates)
        m = compute_metrics(result.equity, result.trades)
        assert m.trade_count >= 30, f"candidate must retain >=30 trades, got {m.trade_count}"

        stress = compute_stress_test_gate(df, spec, costs, funding_rates=funding_rates)
        assert stress.trade_count >= 30, f"stressed candidate trades={stress.trade_count}"
        assert stress.verdict in {"PASS", "FAIL", "PENDING"}

        years = (result.equity.index[-1] - result.equity.index[0]).total_seconds() / (365.25 * 86400)
        gate = compute_reliability_gate(
            result.trades, years=years, sharpe=m.sharpe, mdd=m.mdd,
        )
        assert gate.verdict in {"PASS", "FAIL", "PENDING"}
        assert gate.trade_count == m.trade_count

    def test_candidate_without_funding_rejected_by_engine(self) -> None:
        df = load_ohlcv_4h(BTC_PATH, end="2025-12-31 23:59:59")
        spec = StrategySpec(min_taker_buy_ratio=0.52)

        with pytest.raises(DataIntegrityError, match="funding"):
            run_backtest(df, spec, CostModel())
