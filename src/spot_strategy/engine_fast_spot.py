
import pandas as pd
import numpy as np
import logging

class BacktestEngineFastSpot:
    """
    Numba-accelerated Backtest Engine for Spot (Long-Only)
    Reuses architecture from BacktestEngineFast (Futures) for consistency.
    """
    def __init__(self, df, strategy, backtest_func, initial_balance=10_000_000, fee_rate=0.0005, slippage_rate=0.0003):
        self.df = df
        self.strategy = strategy
        self.backtest_func = backtest_func  # Injected from outside to avoid circular import
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate  # Upbit: 0.05%
        self.slippage_rate = slippage_rate  # Upbit: 0.03%
        
        # Injected by optimization script
        self.risk_per_trade = 0.99  # Spot default
        
        self.logger = logging.getLogger(__name__)
        self._prepare_data()
    
    def _prepare_data(self):
        """
        Generate signals once and extract as numpy arrays.
        This is the key optimization: signals are computed once per engine instance,
        not per trial.
        """
        # Generate all indicators and signals (EXPENSIVE OPERATION - done once!)
        self.df = self.strategy.generate_signals(self.df)
        
        # Extract arrays (cheap)
        self.close = self.df['close'].values
        self.high = self.df['high'].values
        self.low = self.df['low'].values
        
        # Entry signals
        self.entry_upper = self.df['entry_upper'].values
        
        # Trend filter
        self.trend_dir = self.df['trend_direction'].values
        
        # Strength filter
        self.strength_filter = self.df['strength_filter'].values
        
        # Volume Filter (Ratio)
        self.volume_ratio = self.df['volume_ratio'].values
        
        # ATR for risk
        self.atr = self.df['atr'].values
        
        # Parabolic SAR (optional)
        self.parabolic_sar = self.df['parabolic_sar'].values
    
    def run(self):
        """
        Execute backtest using Numba-accelerated loop.
        """
        # Extract strategy params
        exit_type = 1 if self.strategy.params.get('EXIT_TYPE') == 'PARABOLIC_SAR' else 0
        stop_loss_type = 1 if self.strategy.params.get('STOP_LOSS_TYPE') == 'ATR' else 0
        stop_loss_pct = self.strategy.params.get('STOP_LOSS_PCT', 0.03)
        atr_sl_mult = self.strategy.params.get('ATR_STOP_LOSS_MULT', 1.5)
        atr_mult = self.strategy.params.get('ATR_MULTIPLIER', 3.0)
        
        use_volume_filter = self.strategy.params.get('USE_VOLUME_FILTER', False)
        vol_threshold = self.strategy.params.get('VOLUME_THRESHOLD_MULT', 1.0)
        
        use_take_profit = self.strategy.params.get('USE_TAKE_PROFIT', False)
        tp_atr_mult = self.strategy.params.get('TAKE_PROFIT_ATR_MULT', 3.0)
        
        # [NEW] Time-Based Exit & Trailing Activation
        max_holding_bars = self.strategy.params.get('MAX_HOLDING_BARS', 999999)
        trailing_activation_atr = self.strategy.params.get('TRAILING_ACTIVATION_ATR', 0.0)
        
        # Run Numba loop (using injected function)
        trades, equity, final_bal = self.backtest_func(
            self.close, self.high, self.low, self.entry_upper,
            self.trend_dir, self.strength_filter, self.volume_ratio,
            self.atr, self.parabolic_sar,
            self.initial_balance, self.fee_rate, self.slippage_rate,
            exit_type,
            stop_loss_type, stop_loss_pct, atr_sl_mult,
            atr_mult, self.risk_per_trade,
            use_volume_filter, vol_threshold,
            use_take_profit, tp_atr_mult,
            max_holding_bars, trailing_activation_atr  # [NEW]
        )
        
        # Calculate metrics
        total_return_pct = (final_bal - self.initial_balance) / self.initial_balance * 100
        
        # MDD calculation
        peak = np.maximum.accumulate(equity)
        with np.errstate(divide='ignore', invalid='ignore'):
            mdd_series = np.where(peak > 0, (equity - peak) / peak * 100, 0.0)
            mdd_pct = np.min(mdd_series)
            if np.isnan(mdd_pct):
                mdd_pct = 0.0
        
        # Trade statistics
        num_trades = len(trades)
        if num_trades > 0:
            pnl_pcts = trades[:, 0]
            win_rate = (len(pnl_pcts[pnl_pcts > 0]) / num_trades * 100)
        else:
            win_rate = 0.0
        
        # Convert trades to DataFrame
        if num_trades > 0:
            trades_df = pd.DataFrame(trades, columns=['pnl_pct', 'duration', 'dummy'])
        else:
            trades_df = pd.DataFrame()
        
        return {
            'total_return_pct': total_return_pct,
            'mdd_pct': mdd_pct,
            'total_trades': num_trades,
            'win_rate': win_rate,
            'final_balance': final_bal,
            'trades_df': trades_df,
            'equity_curve': equity
        }
