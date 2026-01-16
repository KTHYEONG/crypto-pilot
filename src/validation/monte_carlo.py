
import numpy as np
import pandas as pd
from src.analysis.metrics import calculate_sharpe_ratio, calculate_var, calculate_cvar

class MonteCarloSimulator:
    def __init__(self, trades_df):
        self.trades_df = trades_df
        
    def run(self, n_simulations=10000, initial_balance=1_000_000):
        """
        Run Monte Carlo Simulation by shuffling trade sequence
        """
        if self.trades_df.empty:
            return {
                'prob_profit': 0.0,
                'mean_return': 0.0,
                'worst_case_return': 0.0
            }
            
        pnl_list = self.trades_df['pnl'].values
        
        simulation_returns = []
        simulation_mdds = []
        
        for _ in range(n_simulations):
            # Shuffle trades
            shuffled_pnl = np.random.permutation(pnl_list)
            
            # Calculate Balance Curve
            balance_curve = np.concatenate(([initial_balance], initial_balance + np.cumsum(shuffled_pnl)))
            
            # Final Return
            final_bal = balance_curve[-1]
            ret_pct = (final_bal - initial_balance) / initial_balance * 100
            simulation_returns.append(ret_pct)
            
            # MDD
            running_max = np.maximum.accumulate(balance_curve)
            drawdown = (balance_curve - running_max) / running_max * 100
            mdd = np.min(drawdown)
            simulation_mdds.append(mdd)
            
        simulation_returns = np.array(simulation_returns)
        simulation_mdds = np.array(simulation_mdds)
        
        # Statistics
        prob_profit = np.mean(simulation_returns > 0) * 100
        mean_return = np.mean(simulation_returns)
        median_return = np.median(simulation_returns)
        
        # Confidence Intervals (95%)
        lower_bound = np.percentile(simulation_returns, 2.5)
        upper_bound = np.percentile(simulation_returns, 97.5)
        
        # Risk Metrics
        worst_case_mdd = np.percentile(simulation_mdds, 5) # 5th percentile (most negative)
        
        return {
            'prob_profit': prob_profit,
            'mean_return': mean_return,
            'median_return': median_return,
            'lower_bound_95': lower_bound,
            'upper_bound_95': upper_bound,
            'worst_case_mdd': worst_case_mdd,
            'simulations': n_simulations
        }
