import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import FUTURES_INITIAL_BALANCE, DATA_DIR
from src.common.walk_forward_base import BaseWalkForwardAnalyzer
from src.futures_strategy.strategies_futures import UltimateStrategy
from src.futures_strategy.funding_utils import merge_funding_into_ohlcv


try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    sys.path.append(os.getcwd())


class FuturesWalkForwardAnalyzer(BaseWalkForwardAnalyzer):
    def __init__(self, hourly_df, daily_df, params, eval_start_time=None, symbol=None):
        super().__init__(hourly_df, daily_df, params, eval_start_time=eval_start_time)
        self.symbol = symbol

    @staticmethod
    def _calculate_mdd_from_pnl(pnl_series):
        if pnl_series is None or len(pnl_series) == 0:
            return 0.0
        equity = FUTURES_INITIAL_BALANCE + pd.Series(pnl_series).cumsum().values
        running_max = np.maximum.accumulate(equity)
        running_max[running_max == 0] = 1e-9
        drawdown = (equity - running_max) / running_max * 100.0
        return float(np.min(drawdown)) if len(drawdown) else 0.0

    def run_backtest_segment(self, segment_hourly, segment_daily, actual_start_time, actual_end_time, warmup_bars):
        from .engine_fast_futures import BacktestEngineFast

        if getattr(self, "symbol", None):
            segment_hourly = merge_funding_into_ohlcv(self.symbol, segment_hourly, DATA_DIR)
        strategy = UltimateStrategy("WFA_Segment", self.params)
        engine = BacktestEngineFast(
            segment_hourly,
            segment_daily,
            strategy,
            initial_balance=FUTURES_INITIAL_BALANCE,
        )
        engine.leverage = self.params.get("LEVERAGE", 1)
        engine.risk_per_trade = self.params.get("RISK_PER_TRADE", 0.02)
        engine.funding_events_per_bar = 3 if self.params.get("TIMEFRAME") == "1d" else 1
        res = engine.run()

        trades_df = res["trades_df"]
        if trades_df.empty:
            return 0.0, 0.0

        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
        segment_end = pd.Timestamp(actual_end_time)
        eval_start = pd.Timestamp(actual_start_time)
        if self.eval_start_time is not None:
            eval_start = max(eval_start, self.eval_start_time)

        valid_trades = trades_df[
            (trades_df["entry_time"] >= eval_start) & (trades_df["entry_time"] <= segment_end)
        ].copy()
        if valid_trades.empty:
            return 0.0, 0.0

        total_pnl = valid_trades["pnl"].sum()
        ret_pct = (total_pnl / FUTURES_INITIAL_BALANCE) * 100.0
        mdd = self._calculate_mdd_from_pnl(valid_trades["pnl"])
        return float(ret_pct), float(mdd)


if __name__ == "__main__":
    print("This module is designed to be imported or run with data loaded.")
