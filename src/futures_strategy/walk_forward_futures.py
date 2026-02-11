
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Project Root Setup
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    sys.path.append(os.getcwd())

from .strategies_futures import UltimateStrategy
from config.settings import FUTURES_INITIAL_BALANCE

class FuturesWalkForwardAnalyzer:
    def __init__(self, hourly_df, daily_df, params, eval_start_time=None):
        self.hourly_df = hourly_df
        self.daily_df = daily_df
        self.params = params
        self.eval_start_time = (
            pd.Timestamp(eval_start_time) if eval_start_time is not None else None
        )

    @staticmethod
    def _calculate_mdd_from_pnl(pnl_series):
        if pnl_series is None or len(pnl_series) == 0:
            return 0.0
        equity = FUTURES_INITIAL_BALANCE + pd.Series(pnl_series).cumsum().values
        running_max = np.maximum.accumulate(equity)
        running_max[running_max == 0] = 1e-9
        drawdown = (equity - running_max) / running_max * 100.0
        return float(np.min(drawdown)) if len(drawdown) else 0.0
        
    def run_backtest_segment(self, segment_hourly, segment_daily, actual_start_time, actual_end_time):
        """
        Run backtest on a single WFA segment using BacktestEngineFast.
        Uses overlap buffer for warmup, then filters trades for actual period.
        """
        from .engine_fast_futures import BacktestEngineFast
        
        # Create Strategy
        strategy = UltimateStrategy("WFA_Segment", self.params)
        
        # Use BacktestEngineFast with buffered data
        engine = BacktestEngineFast(
            segment_hourly,
            segment_daily,
            strategy,
            initial_balance=FUTURES_INITIAL_BALANCE,
        )
        engine.leverage = self.params.get('LEVERAGE', 1)
        engine.risk_per_trade = self.params.get('RISK_PER_TRADE', 0.02)
        
        # Run backtest
        res = engine.run()
        
        # [CRITICAL] Filter trades to start from ACTUAL start time (skip overlap buffer)
        trades_df = res['trades_df']
        
        if not trades_df.empty:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            segment_end = pd.Timestamp(actual_end_time)
            eval_start = actual_start_time
            if self.eval_start_time is not None:
                eval_start = max(pd.Timestamp(actual_start_time), self.eval_start_time)
            
            # Filter trades that started within the actual test segment
            valid_trades = trades_df[
                (trades_df['entry_time'] >= eval_start)
                & (trades_df['entry_time'] <= segment_end)
            ].copy()
            
            if not valid_trades.empty:
                # Re-calculate Return from valid trades PnL
                # Return = (Sum of PnL) / Initial Balance
                total_pnl = valid_trades['pnl'].sum()
                ret_pct = (total_pnl / FUTURES_INITIAL_BALANCE) * 100
                mdd = self._calculate_mdd_from_pnl(valid_trades['pnl'])
                return ret_pct, mdd
        
        return 0.0, 0.0

    def run(self, n_splits=5):
        """
        Split dataset into N segments and test parameters on each.
        Includes overlapping buffer for hourly data to ensure indicators are ready at segment start.
        """
        n = len(self.hourly_df)
        segment_size = n // n_splits
        
        results = []
        
        # Warmup Buffer for Hourly Data (overlap)
        # 4h chart: 300 bars = 1200h = 50 days (Sufficient for most indicators)
        HOURLY_BUFFER = 300
        
        for i in range(n_splits):
            start_idx = i * segment_size
            end_idx = n if i == (n_splits - 1) else (start_idx + segment_size)
            
            # Slice Hourly Data with Buffer (Overlapping Window)
            buf_start_idx = max(0, start_idx - HOURLY_BUFFER)
            segment_hourly = self.hourly_df.iloc[buf_start_idx:end_idx].copy()
            
            if len(segment_hourly) < 100: continue
            
            # Determine ACTUAL start time of this segment (without buffer)
            actual_start_time = self.hourly_df.iloc[start_idx]['datetime']
            actual_end_time = self.hourly_df.iloc[end_idx-1]['datetime'] if end_idx < n else self.hourly_df.iloc[-1]['datetime']
            if self.eval_start_time is not None and pd.Timestamp(actual_end_time) < self.eval_start_time:
                continue
            
            # Slice Daily Data (Include 200 days of buffer for indicators)
            start_time_buffered = segment_hourly['datetime'].iloc[0] # Time including hourly buffer
            end_time = segment_hourly['datetime'].iloc[-1]
            
            daily_buffer_start = start_time_buffered - pd.Timedelta(days=200)
            segment_daily = self.daily_df[(self.daily_df['datetime'] >= daily_buffer_start) & (self.daily_df['datetime'] <= end_time)].copy()
            
            period_str = f"{actual_start_time} ~ {actual_end_time}"
            
            # Pass segment boundaries for strict in-segment filtering.
            ret, mdd = self.run_backtest_segment(
                segment_hourly, segment_daily, actual_start_time, actual_end_time
            )
            
            results.append({
                'Split': i+1,
                'Period': period_str,
                'Return': ret,
                'MDD': mdd
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    print("This module is designed to be imported or run with data loaded.")
