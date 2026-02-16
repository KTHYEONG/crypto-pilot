import os
import sys
from pathlib import Path

from src.common.walk_forward_base import BaseWalkForwardAnalyzer
from src.spot_strategy.engine_fast_spot import BacktestEngineFastSpot, backtest_loop_spot_numba
from src.strategy.strategies import UltimateStrategy


try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    sys.path.append(os.getcwd())


class SpotWalkForwardAnalyzer(BaseWalkForwardAnalyzer):
    def run_backtest_segment(self, segment_hourly, segment_daily, actual_start_time, actual_end_time, warmup_bars):
        strategy = UltimateStrategy("WFA_Segment", self.params)
        engine = BacktestEngineFastSpot(
            segment_hourly,
            segment_daily,
            strategy,
            backtest_loop_spot_numba,
            initial_balance=1_000_000,
            fee_rate=0.0005,
            slippage_rate=0.0003,
        )
        engine.risk_per_trade = self.params.get("RISK_PER_TRADE_SPOT", 0.99)
        engine._warmup_bars = warmup_bars
        if hasattr(segment_hourly, "attrs"):
            segment_hourly.attrs["warmup_bars"] = warmup_bars
        res = engine.run()
        return float(res["total_return_pct"]), float(res["mdd_pct"])


if __name__ == "__main__":
    print("This module is designed to be imported or run with data loaded.")
