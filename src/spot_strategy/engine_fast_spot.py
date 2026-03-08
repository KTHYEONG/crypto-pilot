from __future__ import annotations
from typing import Optional
import logging
import numpy as np
import pandas as pd
from numba import njit
from config.settings import TRADING_FEE_RATE, SLIPPAGE_RATE

class BacktestEngineFastSpot:
    def __init__(
        self,
        hourly_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        strategy,
        initial_balance: float = 1_000_000,
        merge_index_map=None,
        precomputed_daily_df: Optional[pd.DataFrame] = None,
        warmup_bars: Optional[int] = None,
        execution_start_idx: int = 0,
    ):
        self.hourly_df = hourly_df.copy(deep=False)
        self.daily_df = daily_df.copy(deep=False)
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self._precomputed_daily_df = precomputed_daily_df
        self._warmup_bars_override = warmup_bars
        self._execution_start_idx = max(0, int(execution_start_idx))

        self.risk_per_trade = self.strategy.params.get("RISK_PER_TRADE", 0.015)

        self.fee_rate = TRADING_FEE_RATE
        self.slippage_rate = SLIPPAGE_RATE

        if merge_index_map is not None:
            self._merge_index_map = merge_index_map

        self.logger = logging.getLogger(__name__)
        self._prepare_data()

    _REQUIRED_INDICATOR_COLS: frozenset = frozenset({
        "entry_upper", "trend_direction", "strength_filter", "atr"
    })

    def _prepare_data(self) -> None:
        exclude_cols = {"date_key", "datetime", "date", "open", "high", "low", "close", "volume", "timestamp"}
        if all(c in self.hourly_df.columns for c in self._REQUIRED_INDICATOR_COLS):
            signal_df = self.hourly_df
        else:
            signal_df = self.strategy.generate_signals(self.hourly_df.copy(deep=True))

        indicator_cols = [
            c for c in signal_df.columns
            if c not in exclude_cols and c in self._REQUIRED_INDICATOR_COLS
        ]

        self.merged_df = self.hourly_df.copy(deep=False)
        for col in indicator_cols:
            self.merged_df[f"daily_{col}"] = signal_df[col].values
    
    def run(self):
        self.logger.debug(f"Running FAST backtest for Spot {self.strategy.name}...")
        df = self.merged_df
        n = len(df)
        
        open_prices = df['open'].values 
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        entry_upper = df['daily_entry_upper'].values
        trend_dir = df['daily_trend_direction'].values
        strength_filter = df['daily_strength_filter'].values
        atr = df['daily_atr'].values
        
        long_atr_mult = float(self.strategy.params.get('LONG_ATR_MULT', 3.0))
        long_trail_mult = float(self.strategy.params.get('LONG_TRAIL_MULT', 3.0))
        long_tp_mult = float(self.strategy.params.get('LONG_TP_MULT', 5.0))
        
        timestamps = df["timestamp"].values 
        
        if getattr(self, "_warmup_bars_override", None) is not None:
            warmup_bars = self._warmup_bars_override
        else:
            warmup_bars = getattr(df, "attrs", {}).get(
                "warmup_bars",
                self.strategy.get_required_warmup(freq="hourly") if hasattr(self.strategy, "get_required_warmup") else 100,
            )
        self._warmup_bars = warmup_bars  
        self._effective_start_idx = max(warmup_bars, self._execution_start_idx)
        datetime_values = df['datetime'].values
        
        use_compounding = self.strategy.params.get('USE_COMPOUNDING', True)
        max_capital_usage = self.strategy.params.get('MAX_CAPITAL_USAGE', 1e12) 

        trades, final_balance, equity_curve = backtest_loop_numba_spot(
            close, high, low, open_prices,
            entry_upper,
            trend_dir, strength_filter, atr,
            self.initial_balance, self.fee_rate, self.slippage_rate,
            self.risk_per_trade, timestamps,
            long_atr_mult, long_trail_mult, long_tp_mult,
            warmup_bars, self._execution_start_idx,
            use_compounding, max_capital_usage
        )
        
        self.balance = final_balance
        self._equity_curve = equity_curve

        self.merged_df = None
        self.hourly_df = None
        self.daily_df = None
        
        self.trades = []
        for i in range(len(trades)):
            entry_idx = int(trades[i][0])
            exit_idx = int(trades[i][1])
            self.trades.append({
                "entry_time": datetime_values[entry_idx],
                "exit_time": datetime_values[exit_idx],
                "side": "LONG",
                "entry_price": trades[i][2],
                "exit_price": trades[i][3],
                "pnl": trades[i][4],
                "amount": trades[i][5],
                "entry_fee": trades[i][6],
            })
        
        return self.get_results()
    
    def get_results(self):
        if not np.isfinite(self.balance):
            return self._empty_result()

        total_return = self.balance - self.initial_balance
        total_return_pct = (total_return / self.initial_balance) * 100

        if not self.trades:
            return self._empty_result()

        n_trades = len(self.trades)
        pnl_arr = np.fromiter((t["pnl"] for t in self.trades), dtype=np.float64, count=n_trades)
        entry_fee_arr = np.fromiter((t["entry_fee"] for t in self.trades), dtype=np.float64, count=n_trades)
        entry_p = np.fromiter((t["entry_price"] for t in self.trades), dtype=np.float64, count=n_trades)
        amount_arr = np.fromiter((t["amount"] for t in self.trades), dtype=np.float64, count=n_trades)

        equity = getattr(self, "_equity_curve", None)
        warmup_bars = getattr(self, "_warmup_bars", 0)
        equity_for_mdd: np.ndarray = np.array([])
        if equity is not None and len(equity) > 0 and np.isfinite(equity).all():
            effective_start = int(getattr(self, "_effective_start_idx", warmup_bars))
            if len(equity) > effective_start:
                equity_for_mdd = equity[effective_start:]
            else:
                equity_for_mdd = equity
            running_max = np.maximum.accumulate(equity_for_mdd)
            running_max[running_max == 0] = 1e-9
            drawdown = (equity_for_mdd - running_max) / running_max * 100
            drawdown = np.nan_to_num(drawdown, nan=0.0)
            mdd = float(drawdown.min())
        else:
            cumulative = np.empty(n_trades + 1, dtype=np.float64)
            cumulative[0] = self.initial_balance
            np.cumsum(pnl_arr, out=cumulative[1:])
            cumulative[1:] += self.initial_balance
            if not np.isfinite(cumulative).all():
                mdd = 0.0
            else:
                running_max = np.maximum.accumulate(cumulative)
                running_max[running_max == 0] = 1e-9
                drawdown = (cumulative - running_max) / running_max * 100
                drawdown = np.nan_to_num(drawdown, nan=0.0)
                mdd = float(drawdown.min())

        true_pnl = pnl_arr - entry_fee_arr
        capital_used = amount_arr * entry_p
        capital_used = np.where(np.isfinite(capital_used) & (capital_used > 0), capital_used, np.nan)

        entry_fee_cumsum = np.concatenate(([0.0], np.cumsum(entry_fee_arr)[:-1]))
        pnl_cumsum = np.concatenate(([0.0], np.cumsum(pnl_arr)[:-1]))
        balance_before = self.initial_balance - entry_fee_cumsum + pnl_cumsum
        balance_before = np.where(balance_before == 0, 1e-9, balance_before)

        denom = np.where(
            np.isfinite(capital_used) & (capital_used > 0),
            capital_used,
            np.asarray(balance_before, dtype=np.float64),
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            pnl_pct = (true_pnl / denom) * 100
        pnl_pct = np.nan_to_num(pnl_pct, nan=0.0, posinf=0.0, neginf=0.0)

        win_trades = int(np.sum(true_pnl > 0))
        loss_trades = int(np.sum(true_pnl <= 0))
        win_rate = (win_trades / n_trades) * 100 if n_trades > 0 else 0.0

        trades_df = pd.DataFrame(self.trades)
        trades_df["balance_before"] = balance_before
        trades_df["pnl_pct"] = pnl_pct

        return {
            'total_trades': n_trades,
            'win_trades': win_trades,
            'loss_trades': loss_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'final_balance': self.balance,
            'mdd_pct': mdd,
            'trades_df': trades_df,
            'equity_curve': equity_for_mdd,
        }
        
    def _empty_result(self):
        return {
            'total_trades': 0, 'win_trades': 0, 'loss_trades': 0, 'win_rate': 0,
            'total_return_pct': 0, 'final_balance': self.initial_balance, 'mdd_pct': 0,
            'trades_df': pd.DataFrame(), 'equity_curve': np.array([]),
        }

@njit(nogil=True, cache=True)
def backtest_loop_numba_spot(
    close, high, low, open_prices,
    entry_upper,
    trend_dir, strength_filter, atr,
    initial_balance, fee_rate, slippage_rate,
    risk_per_trade, timestamps,
    long_atr_mult, long_trail_mult, long_tp_mult,
    warmup_bars, execution_start_idx,
    use_compounding, max_capital_usage
):
    n = len(close)
    balance = initial_balance
    peak_equity = initial_balance
    equity_curve = np.zeros(n)

    in_position = False
    entry_price = 0.0
    entry_idx = 0
    amount = 0.0
    entry_fee_stored = 0.0
    
    stop_price = 0.0 
    tp_price = 0.0
    highest = 0.0
    
    max_trades = 30000
    trades = np.zeros((max_trades, 7))  
    trade_count = 0
    
    for i in range(n):
        if i < warmup_bars or i < execution_start_idx:
            equity_curve[i] = initial_balance
            continue

        c_open = open_prices[i]
        c_price = close[i]
        c_high = high[i]
        c_low = low[i]
        bar_processed = False

        current_eq_check = balance
        if in_position:
            current_eq_check = balance + amount * c_open
        if current_eq_check <= 0:
            equity_curve[i] = current_eq_check
            if in_position:
                exit_price = c_open
                pnl = (exit_price - entry_price) * amount
                exit_fee = amount * exit_price * fee_rate
                if trade_count < max_trades:
                    trades[trade_count] = [entry_idx, i, entry_price, exit_price, pnl - exit_fee, amount, entry_fee_stored]
                    trade_count += 1
                in_position = False
                balance = 0.0
            break

        if in_position:
            exit_triggered = False
            exit_price = 0.0
            
            if c_high > highest:
                highest = c_high
            
            pos_atr = atr[entry_idx]
            
            # Check Stop Loss first (Gap down)
            if c_open <= stop_price:
                exit_price = c_open * (1 - slippage_rate)
                exit_triggered = True
            elif c_low <= stop_price:
                exit_price = stop_price * (1 - slippage_rate)
                exit_triggered = True
            # Check Take Profit (Gap up or high breach)
            elif c_open >= tp_price:
                exit_price = c_open * (1 - slippage_rate)
                exit_triggered = True
            elif c_high >= tp_price:
                exit_price = tp_price * (1 - slippage_rate)
                exit_triggered = True
            
            if not exit_triggered:
                new_stop = highest - (pos_atr * long_trail_mult)
                if new_stop > stop_price:
                    stop_price = new_stop

            if exit_triggered:
                pnl = (exit_price - entry_price) * amount
                exit_fee = amount * exit_price * fee_rate
                pnl -= exit_fee
                balance += (amount * entry_price) + pnl
                
                if trade_count < max_trades:
                    trades[trade_count] = [entry_idx, i, entry_price, exit_price, pnl, amount, entry_fee_stored]
                    trade_count += 1
                in_position = False
                bar_processed = True

        if not in_position and not bar_processed:
            prev_i = i - 1 if i > 0 else 0
            if strength_filter[prev_i] == 0 or np.isnan(strength_filter[prev_i]):
                equity_curve[i] = balance
                continue

            do_entry = False
            fill_price = 0.0
            
            if trend_dir[prev_i] == 1:
                if c_high > entry_upper[prev_i]:
                    fill_price = max(c_open, entry_upper[prev_i]) * (1 + slippage_rate)
                    do_entry = True

            if do_entry:
                prev_atr = atr[prev_i]
                if np.isnan(prev_atr) or prev_atr <= 0.0:
                    equity_curve[i] = balance
                    continue
                
                stop_price = fill_price - (prev_atr * long_atr_mult)
                stop_distance = abs(fill_price - stop_price)
                if stop_distance > 0:
                    current_equity = max_capital_usage if use_compounding and balance > max_capital_usage else balance
                    
                    # Spot Allocation: Allocate direct percentage of available equity
                    target_capital = current_equity * risk_per_trade
                    
                    # Prevent allocating more than we have
                    available_capital = balance * 0.99  # 1% buffer for fees
                    allocate_capital = min(target_capital, available_capital)
                    
                    if allocate_capital > 0:
                        amount = allocate_capital / fill_price
                        
                        required_capital = amount * fill_price
                        entry_fee = required_capital * fee_rate
                        
                        if balance >= required_capital + entry_fee:
                            balance -= (required_capital + entry_fee)
                            entry_fee_stored = entry_fee
                            in_position = True
                            entry_price = fill_price
                            entry_idx = i
                            highest = fill_price
                            tp_price = fill_price + (prev_atr * long_tp_mult)
                            
                            if c_low <= stop_price:
                                intra_exit_price = stop_price * (1 - slippage_rate)
                                pnl = (intra_exit_price - entry_price) * amount
                                exit_fee_intra = amount * intra_exit_price * fee_rate
                                pnl -= exit_fee_intra
                                balance += (amount * entry_price) + pnl
                                if trade_count < max_trades:
                                    trades[trade_count] = [entry_idx, i, entry_price, intra_exit_price, pnl, amount, entry_fee_stored]
                                    trade_count += 1
                                in_position = False

        if in_position:
            unrealized = (c_price - entry_price) * amount
            equity_curve[i] = balance + (amount * entry_price) + unrealized
        else:
            equity_curve[i] = balance
            
        if equity_curve[i] > peak_equity:
            peak_equity = equity_curve[i]

    if in_position and n > 0:
        last_idx = n - 1
        last_close = close[last_idx]
        exit_price = last_close * (1 - slippage_rate)
        pnl = (exit_price - entry_price) * amount

        exit_fee = amount * exit_price * fee_rate
        pnl -= exit_fee
        balance += (amount * entry_price) + pnl
        if trade_count < max_trades:
            trades[trade_count] = [entry_idx, last_idx, entry_price, exit_price, pnl, amount, entry_fee_stored]
            trade_count += 1

    return trades[:trade_count], balance, equity_curve
