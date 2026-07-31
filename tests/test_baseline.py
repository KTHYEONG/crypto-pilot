from __future__ import annotations

from pathlib import Path

import pytest

from src.config import CostModel, StrategySpec
from src.data.loader import load_ohlcv_4h
from src.engine import run_backtest
from src.metrics import compute_metrics

BTC_PATH = Path("data/futures/ohlcv/1h/BTCUSDT.parquet")


@pytest.mark.slow
class TestBaselineReplication:
    def test_reproduces_prototype_figures(self) -> None:
        df = load_ohlcv_4h(BTC_PATH, end="2025-12-31 23:59:59")
        spec = StrategySpec()
        costs = CostModel()
        result = run_backtest(df, spec, costs, initial_equity=10_000.0)
        m = compute_metrics(result.equity, result.trades)

        # Structural figures (trade timing/placement) must match the frozen
        # prototype exactly -- these are what the ATR-causality regression
        # (see ADR: signal-bar ATR must be atr[t-1], not atr[t]) broke.
        assert m.trade_count == 91, f"trade_count={m.trade_count}"
        assert abs(m.mdd - (-0.0747)) < 1e-3, f"mdd={m.mdd}"
        reason_counts = result.trades["reason"].value_counts().to_dict()
        assert reason_counts.get("channel", 0) == 46, f"channel exits: {reason_counts.get('channel', 0)}"
        assert reason_counts.get("stop", 0) == 39, f"stop exits: {reason_counts.get('stop', 0)}"
        assert reason_counts.get("stop_entrybar", 0) == 6, f"stop_entrybar exits: {reason_counts.get('stop_entrybar', 0)}"

        # Aggregate PnL statistics: allow implementation-level bookkeeping
        # variance (fee/notional deduction order) relative to the scratch
        # prototype, but stay tight enough to catch a structural regression.
        assert abs(m.cagr - 0.0988) < 5e-3, f"cagr={m.cagr}"
        assert abs(m.sharpe - 1.419) < 5e-2, f"sharpe={m.sharpe}"
        assert abs(m.sortino - 1.109) < 5e-2, f"sortino={m.sortino}"
        assert abs(m.calmar - 1.323) < 5e-2, f"calmar={m.calmar}"
        assert abs(m.profit_factor - 2.337) < 1e-1, f"pf={m.profit_factor}"
        assert abs(m.win_rate - 0.341) < 1e-2, f"win_rate={m.win_rate}"
        assert abs(m.exposure - 0.268) < 1e-2, f"exposure={m.exposure}"
