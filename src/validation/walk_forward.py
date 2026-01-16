
import pandas as pd
import numpy as np
import logging
from src.backtest.engine_fast import BacktestEngineFast

class WalkForwardAnalyzer:
    def __init__(self, hourly_df, daily_df, strategy_cls, strategy_name, base_params, common_params):
        self.hourly_df = hourly_df
        self.daily_df = daily_df
        self.strategy_cls = strategy_cls
        self.strategy_name = strategy_name
        self.base_params = base_params
        self.common_params = common_params
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, n_splits=5, train_ratio=0.7):
        """
        Run Walk-Forward Analysis with Rolling Window
        """
        # 1. Total timeline check
        full_dates = self.hourly_df['datetime'].sort_values().unique()
        n_periods = len(full_dates)
        
        if n_periods < 1000:
            self.logger.warning("Not enough data for 5-split walk-forward. Reducing splits to 3.")
            n_splits = 3
            
        # Segment size
        segment_size = n_periods // n_splits
        
        results = []
        
        for i in range(n_splits):
            # Define window indices
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            
            # Use raw indices to slice dataframes (assuming sorted by date)
            # Find timestamp range for safe slicing
            start_time = full_dates[start_idx]
            end_time = full_dates[min(end_idx, n_periods - 1)]
            
            # Split Train/Test within this segment
            # Actually, standard Walk-Forward is: Train on Period N, Test on Period N+1
            # But here we implement Rolling Window: Train on 70% of Segment, Test on 30% of Segment
            
            segment_duration = end_time - start_time
            split_point = start_time + (segment_duration * train_ratio)
            
            # Slicing
            train_hourly = self.hourly_df[(self.hourly_df['datetime'] >= start_time) & (self.hourly_df['datetime'] < split_point)].copy()
            train_daily = self.daily_df[(self.daily_df['datetime'] >= start_time) & (self.daily_df['datetime'] < split_point)].copy()
            
            test_hourly = self.hourly_df[(self.hourly_df['datetime'] >= split_point) & (self.hourly_df['datetime'] <= end_time)].copy()
            test_daily = self.daily_df[(self.daily_df['datetime'] >= split_point) & (self.daily_df['datetime'] <= end_time)].copy()
            
            if len(test_hourly) == 0:
                continue
                
            # Run Backtest (Verification Only - No Optimization loop here for speed)
            # In a full WFA, we would re-optimize params on Train set.
            # Here, we validate if the 'Global Best Params' work across all rolling periods.
            # This is "Robustness Check" mode.
            
            full_params = {**self.base_params, **self.common_params}
            
            # Test on OOS (Out-of-Sample)
            strategy = self.strategy_cls(f"{self.strategy_name}_Split_{i}", full_params)
            
            engine = BacktestEngineFast(test_hourly, test_daily, strategy, initial_balance=1_000_000)
            engine.leverage = full_params.get('LEVERAGE', 1)
            engine.risk_per_trade = full_params.get('RISK_PER_TRADE', 0.02)
            
            result = engine.run()
            
            results.append({
                'split_id': i,
                'start_date': split_point,
                'end_date': end_time,
                'return': result['total_return_pct'],
                'mdd': result['mdd_pct'],
                'win_rate': result['win_rate'],
                'trades': result['total_trades']
            })
            
        return pd.DataFrame(results)

    def calculate_robustness_score(self, wf_results_df):
        """
        Calculate stability score based on WFA results
        """
        if wf_results_df.empty:
            return 0.0
            
        avg_return = wf_results_df['return'].mean()
        std_return = wf_results_df['return'].std()
        
        if std_return == 0:
            return 1.0 # Perfectly consistent (unlikely)
            
        # Coefficient of Variation (lower is better, so we invert)
        # However, returns can be negative. 
        # Metric: Sharpe-like stability across periods
        
        stability_ratio = avg_return / (std_return + 1.0) 
        
        # Consistency: Ratio of positive periods
        positive_periods = len(wf_results_df[wf_results_df['return'] > 0])
        consistency = positive_periods / len(wf_results_df)
        
        return (stability_ratio * 0.5) + (consistency * 0.5)
