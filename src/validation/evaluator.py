
import logging
import pandas as pd
from src.analysis import metrics, statistics
from src.validation.walk_forward import WalkForwardAnalyzer
from src.validation.monte_carlo import MonteCarloSimulator
from config.validation_config import *

class StrategyEvaluator:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def evaluate(self, hourly_df, daily_df, strategy_cls, strategy_name, base_params, common_params, initial_result):
        """
        Comprehensive Strategy Evaluation
        """
        report = {
            'status': 'PASSED',
            'reason': [],
            'metrics': {},
            'wfa': {},
            'monte_carlo': {}
        }
        
        trades_df = initial_result['trades_df']
        
        # [ADD] Basic statistics from base backtest
        report['metrics']['total_return_pct'] = initial_result['total_return_pct']
        report['metrics']['mdd_pct'] = initial_result['mdd_pct']
        
        # 1. Basic Data Sufficiency Check
        if len(trades_df) < MIN_TRADES_COUNT:
            report['status'] = 'FAILED'
            report['reason'].append(f"Not enough trades: {len(trades_df)} < {MIN_TRADES_COUNT}")
            return report
            
        # 2. Statistical Significance Test
        returns = trades_df['pnl'].values # Need percentage returns for t-test? Usually PnL is fine for 0 mean test if normalized
        # Using returns pct is better
        # Approximation: simple returns per trade
        # trades_df['pnl_pct'] = trades_df['pnl'] / (trades_df['entry_price'] * trades_df['amount']) # Hard to reconstruct
        # Let's use PnL directly. H0: Mean PnL = 0
        
        t_stat, p_value = statistics.perform_t_test(trades_df['pnl'])
        is_significant = p_value < STAT_SIGNIFICANCE_ALPHA
        report['metrics']['p_value'] = p_value
        
        if not is_significant:
            report['status'] = 'FAILED'
            report['reason'].append(f"Not statistically significant (p={p_value:.4f})")

        # 3. Financial Metrics Check
        # returns_pct is needed for Sharpe.
        # We can approximate daily returns from daily balance
        # But here we have list of trades.
        # Let's use simple Sharpe on trade returns
        trade_returns = trades_df['pnl'] / 1_000_000 # Normalized by initial capital approx
        sharpe = metrics.calculate_sharpe_ratio(trade_returns, periods_per_year=len(trade_returns)) # Trade-based Sharpe
        sortino = metrics.calculate_sortino_ratio(trade_returns, periods_per_year=len(trade_returns))
        
        report['metrics']['sharpe'] = sharpe
        report['metrics']['sortino'] = sortino
        
        if sharpe < MIN_SHARPE_RATIO:
            report['status'] = 'WARNING' # Downgrade to Warning instead of Fail for Sharpe
            report['reason'].append(f"Low Sharpe Ratio ({sharpe:.2f})")

        # 4. Walk-Forward Analysis (Robustness)
        self.logger.info("Running Walk-Forward Analysis...")
        wfa = WalkForwardAnalyzer(hourly_df, daily_df, strategy_cls, strategy_name, base_params, common_params)
        wfa_df = wfa.run(n_splits=5)
        
        wfa_score = wfa.calculate_robustness_score(wfa_df)
        report['wfa']['score'] = wfa_score
        report['wfa']['details'] = wfa_df.to_dict('records')
        
        if wfa_score < WFA_MIN_ROBUSTNESS_SCORE:
            report['status'] = 'FAILED'
            report['reason'].append(f"Low Robustness Score ({wfa_score:.2f})")
            
        # 5. Monte Carlo Simulation (Probability)
        self.logger.info("Running Monte Carlo Simulation...")
        mc = MonteCarloSimulator(trades_df)
        mc_result = mc.run(n_simulations=5000)
        
        report['monte_carlo'] = mc_result
        
        if mc_result['prob_profit'] < MC_MIN_PROB_PROFIT:
             report['status'] = 'FAILED'
             report['reason'].append(f"Low Probability of Profit ({mc_result['prob_profit']:.1f}%)")
             
        if mc_result['worst_case_mdd'] < MC_MAX_WORST_CASE_MDD:
             report['status'] = 'FAILED'
             report['reason'].append(f"High Risk in Worst Case ({mc_result['worst_case_mdd']:.1f}%)")
             
        return report
