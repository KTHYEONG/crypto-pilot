
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
from .backtest_utils_futures import run_backtest_segment_futures, prepare_futures_data

class FuturesWalkForwardAnalyzer:
    def __init__(self, hourly_df, daily_df, params):
        self.hourly_df = hourly_df
        self.daily_df = daily_df
        self.params = params
        
    def run_backtest_segment(self, segment_hourly, segment_daily):
        # Create Strategy
        strategy = UltimateStrategy("WFA_Segment", self.params)
        
        # Prepare Data using unified logic
        # This will merge and shift daily indicators correctly
        prepared_df = prepare_futures_data(segment_hourly, segment_daily, strategy)
        
        # Use initial balance 750 for consistency
        ret_pct, mdd = run_backtest_segment_futures(prepared_df, self.params, initial_balance=750, return_series=False)
        return ret_pct, mdd

    def run(self, n_splits=5):
        """
        Split dataset into N segments and test parameters on each.
        Includes buffer for daily indicators to prevent NaN issues.
        """
        n = len(self.hourly_df)
        segment_size = n // n_splits
        
        results = []
        
        for i in range(n_splits):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            
            # Slice Hourly Data
            segment_hourly = self.hourly_df.iloc[start_idx:end_idx].copy()
            if len(segment_hourly) < 100: continue
            
            # Slice Daily Data (Include 200 days of buffer for indicators)
            start_time = segment_hourly['datetime'].iloc[0]
            end_time = segment_hourly['datetime'].iloc[-1]
            
            # Include historical buffer for indicators (MA, ATR, etc.)
            buffer_start = start_time - pd.Timedelta(days=200)
            segment_daily = self.daily_df[(self.daily_df['datetime'] >= buffer_start) & (self.daily_df['datetime'] <= end_time)].copy()
            
            period_str = f"{start_time} ~ {end_time}"
            
            ret, mdd = self.run_backtest_segment(segment_hourly, segment_daily)
            
            results.append({
                'Split': i+1,
                'Period': period_str,
                'Return': ret,
                'MDD': mdd
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    print("This module is designed to be imported or run with data loaded.")
