
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

from src.strategy.strategies import UltimateStrategy
from src.spot_strategy.engine_fast_spot import BacktestEngineFastSpot, backtest_loop_spot_numba

class SpotWalkForwardAnalyzer:
    def __init__(self, hourly_df, daily_df, params):
        self.hourly_df = hourly_df
        self.daily_df = daily_df
        self.params = params
        
    def run_backtest_segment(self, segment_hourly, segment_daily, warmup_bars):
        """
        Run backtest on a single WFA segment using BacktestEngineFastSpot.
        Uses _warmup_bars to skip overlaps (buffer).
        """
        # Create Strategy
        strategy = UltimateStrategy("WFA_Segment", self.params)
        
        # Use BacktestEngineFastSpot with buffered data
        engine = BacktestEngineFastSpot(
            segment_hourly, segment_daily, strategy, backtest_loop_spot_numba,
            initial_balance=1_000_000,
            fee_rate=0.0005,
            slippage_rate=0.0003
        )
        engine.risk_per_trade = self.params.get('RISK_PER_TRADE_SPOT', 0.99)
        
        # [CRITICAL] Set warmup bars to skip trading during the buffer period
        # Engine will skip valid logic/trading for these first N bars.
        engine._warmup_bars = warmup_bars
        if hasattr(segment_hourly, 'attrs'):
            segment_hourly.attrs['warmup_bars'] = warmup_bars
        
        # Run backtest
        res = engine.run()
        
        # Since engine skipped warmup bars, result metrics are for the valid period only.
        ret_pct = res['total_return_pct']
        mdd = res['mdd_pct']
        
        return ret_pct, mdd

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
            end_idx = start_idx + segment_size
            
            # Slice Hourly Data with Buffer (Overlapping Window)
            buf_start_idx = max(0, start_idx - HOURLY_BUFFER)
            segment_hourly = self.hourly_df.iloc[buf_start_idx:end_idx].copy()
            
            if len(segment_hourly) < 100: continue
            
            # Identify how many bars are buffer (warmup)
            # If start_idx == 0, buffer is 0.
            # If start_idx > 0, buffer is (start_idx - buf_start_idx).
            warmup_bars = start_idx - buf_start_idx
            
            # Determine ACTUAL start time of this segment (without buffer) for display
            actual_start_time = self.hourly_df.iloc[start_idx]['datetime']
            actual_end_time = self.hourly_df.iloc[end_idx-1]['datetime'] if end_idx < n else self.hourly_df.iloc[-1]['datetime']
            
            # Slice Daily Data (Include 200 days of buffer for indicators)
            start_time_buffered = segment_hourly['datetime'].iloc[0] # Time including hourly buffer
            end_time = segment_hourly['datetime'].iloc[-1]
            
            daily_buffer_start = start_time_buffered - pd.Timedelta(days=200)
            segment_daily = self.daily_df[(self.daily_df['datetime'] >= daily_buffer_start) & (self.daily_df['datetime'] <= end_time)].copy()
            
            period_str = f"{actual_start_time} ~ {actual_end_time}"
            
            # Run segment verification
            ret, mdd = self.run_backtest_segment(segment_hourly, segment_daily, warmup_bars)
            
            results.append({
                'Split': i+1,
                'Period': period_str,
                'Return': ret,
                'MDD': mdd
            })
            
        if not results:
            return pd.DataFrame(columns=['Split', 'Period', 'Return', 'MDD'])

        return pd.DataFrame(results)

if __name__ == "__main__":
    print("This module is designed to be imported or run with data loaded.")
