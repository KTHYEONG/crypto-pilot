
import numpy as np
import pandas as pd

class SpotMonteCarloSimulator:
    def __init__(self, trades_list):
        """
        trades_list: list of trade returns in percent (e.g., [5.0, -2.0, 10.0])
        """
        self.trades = np.array(trades_list)
        
    def run(self, n_simulations=10000, initial_balance=10000000.0):
        """
        Run Monte Carlo Simulation by shuffling trade sequence
        """
        if len(self.trades) == 0:
            return {
                'prob_profit': 0.0,
                'mean_return_pct': 0.0,
                'worst_case_mdd': 0.0
            }
            
        simulation_final_balances = []
        simulation_mdds = []
        
        for _ in range(n_simulations):
            # Shuffle trades percentage returns
            shuffled_rets = np.random.permutation(self.trades)
            
            # Simple compounding balance simulation
            # Balance[i] = Balance[i-1] * (1 + ret/100)
            
            # Vectorized cumulative product for speed
            # (1 + r1) * (1 + r2) ...
            equity_curve = np.cumprod(1 + shuffled_rets / 100.0) * initial_balance
            equity_curve = np.insert(equity_curve, 0, initial_balance)
            
            # Final Balance
            final_bal = equity_curve[-1]
            simulation_final_balances.append(final_bal)
            
            # MDD Calc
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max * 100
            mdd = np.min(drawdown)
            simulation_mdds.append(mdd)
            
        simulation_final_balances = np.array(simulation_final_balances)
        simulation_mdds = np.array(simulation_mdds)
        
        # Calculate Returns %
        sim_returns_pct = (simulation_final_balances - initial_balance) / initial_balance * 100
        
        # Stats
        prob_profit = np.mean(sim_returns_pct > 0) * 100
        mean_return = np.mean(sim_returns_pct)
        median_return = np.median(sim_returns_pct)
        
        # 95% Confidence Interval
        lower_bound = np.percentile(sim_returns_pct, 2.5)
        upper_bound = np.percentile(sim_returns_pct, 97.5)
        
        # Risk (Worst 5% MDD)
        worst_case_mdd = np.percentile(simulation_mdds, 5)
        
        return {
            'prob_profit': prob_profit,
            'mean_return_pct': mean_return,
            'median_return_pct': median_return,
            'lower_bound_95': lower_bound,
            'upper_bound_95': upper_bound,
            'worst_case_mdd': worst_case_mdd,
            'simulations': n_simulations
        }
