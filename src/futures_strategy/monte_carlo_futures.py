
import numpy as np
import pandas as pd

class FuturesMonteCarloSimulator:
    def __init__(self, trades):
        """
        :param trades: List of percentage returns from trades
        """
        self.trades = trades
    
    def run(self, n_simulations=10000, initial_balance=1_000_000.0):
        """
        Run Monte Carlo Simulation.
        Uses Bootstrap Sampling (Sampling with Replacement) to create realistic variance.
        """
        if len(self.trades) < 5:
            return {
                'prob_profit': 0.0,
                'mean_return_pct': 0.0,
                'median_return_pct': 0.0,
                'worst_case_mdd': 0.0,
                'lower_bound_95': 0.0,
                'upper_bound_95': 0.0
            }
            
        simulation_final_balances = []
        simulation_mdds = []
        
        n_trades = len(self.trades)
        trades_arr = np.array(self.trades)
        
        for _ in range(n_simulations):
            # Bootstrap sampling: Randomly pick N trades with replacement
            # This creates variety in final returns, unlike simple shuffling.
            shuffled_rets = np.random.choice(trades_arr, size=n_trades, replace=True)
            
            # Simple cumulative sum simulation (Conservative)
            cumulative_ret_pct = np.cumsum(shuffled_rets)
            equity_curve = initial_balance * (1 + cumulative_ret_pct / 100.0)
            equity_curve = np.insert(equity_curve, 0, initial_balance)
            
            # Final Balance
            final_bal = equity_curve[-1]
            simulation_final_balances.append(final_bal)
            
            # MDD Calc
            running_max = np.maximum.accumulate(equity_curve)
            with np.errstate(divide='ignore', invalid='ignore'):
                drawdown = (equity_curve - running_max) / running_max * 100
                mdd = np.min(drawdown)
                if np.isnan(mdd): mdd = 0.0
            simulation_mdds.append(mdd)
            
        simulation_final_balances = np.array(simulation_final_balances)
        simulation_mdds = np.array(simulation_mdds)
        
        # Calculate Returns % relative to initial balance
        sim_returns_pct = (simulation_final_balances - initial_balance) / initial_balance * 100
        
        # Stats
        prob_profit = np.mean(sim_returns_pct > 0) * 100
        mean_return = np.mean(sim_returns_pct)
        median_return = np.median(sim_returns_pct)
        
        # 95% Confidence Interval for Returns
        lower_bound = np.percentile(sim_returns_pct, 2.5)
        upper_bound = np.percentile(sim_returns_pct, 97.5)
        
        # Risk (Worst 5% MDD)
        worst_case_mdd = np.percentile(simulation_mdds, 5)
        
        return {
            'prob_profit': prob_profit,
            'mean_return_pct': mean_return,
            'median_return_pct': median_return,
            'worst_case_mdd': worst_case_mdd,
            'lower_bound_95': lower_bound,
            'upper_bound_95': upper_bound
        }
