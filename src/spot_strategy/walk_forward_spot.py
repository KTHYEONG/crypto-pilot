import pandas as pd
import numpy as np
from src.strategy.strategies import UltimateStrategy
from src.spot_strategy.backtest_utils import run_backtest_segment as run_backtest_utils
# Re-using verify logic but simpler for loop
# To avoid circular imports or redefining backtest logic, we implement a simple backtest runner here

class SpotWalkForwardAnalyzer:
    def __init__(self, df, params):
        self.df = df
        self.params = params
        
    def run_backtest_segment(self, segment_df):
        # Generate Signals
        strategy = UltimateStrategy("WFA", self.params)
        df = strategy.generate_signals(segment_df.copy())
        
        # Call shared utility
        ret_pct, mdd = run_backtest_utils(df, self.params, return_series=False)
        return ret_pct, mdd

    def run(self, n_splits=5):
        """
        Split dataset into N segments and test parameters on each.
        """
        n = len(self.df)
        segment_size = n // n_splits
        
        results = []
        
        for i in range(n_splits):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            
            segment_df = self.df.iloc[start_idx:end_idx].copy()
            if len(segment_df) < 100: continue
            
            period_str = f"{segment_df['datetime'].iloc[0]} ~ {segment_df['datetime'].iloc[-1]}"
            ret, mdd = self.run_backtest_segment(segment_df)
            
            results.append({
                'Split': i+1,
                'Period': period_str,
                'Return': ret,
                'MDD': mdd
            })
            
        if not results:
            return pd.DataFrame(columns=['Split', 'Period', 'Return', 'MDD'])
            
        return pd.DataFrame(results)

