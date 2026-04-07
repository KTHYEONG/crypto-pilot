from __future__ import annotations

from typing import Optional

import logging
import numpy as np
import pandas as pd
from numba import njit

from config.settings import TRADING_FEE_RATE, SLIPPAGE_RATE, FUNDING_FEE_RATE

class BacktestEngineFast:
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

        self.leverage = self.strategy.params.get("LEVERAGE", 1)
        self.risk_per_trade = self.strategy.params.get("RISK_PER_TRADE", 0.015)
        self.funding_events_per_bar = 1  

        self.fee_rate = TRADING_FEE_RATE
        self.slippage_rate = SLIPPAGE_RATE

        if merge_index_map is not None:
            self._merge_index_map = merge_index_map

        self.logger = logging.getLogger(__name__)
        self._prepare_data()

    _REQUIRED_INDICATOR_COLS: frozenset[str] = frozenset({
        "entry_upper", "entry_lower", "trend_direction", "strength_filter", "atr", "macro_ema",
    })
    _OPTIONAL_MERGE_COLS: frozenset[str] = frozenset({"garch_kelly_f"})

    def _prepare_data(self) -> None:
        exclude_cols = {"date_key", "datetime", "date", "open", "high", "low", "close", "volume", "timestamp"}
        if all(c in self.hourly_df.columns for c in self._REQUIRED_INDICATOR_COLS):
            signal_df = self.hourly_df
        else:
            signal_df = self.strategy.generate_signals(self.hourly_df.copy(deep=True))

        merge_keys = self._REQUIRED_INDICATOR_COLS | self._OPTIONAL_MERGE_COLS
        indicator_cols = [
            c for c in signal_df.columns
            if c not in exclude_cols and c in merge_keys
        ]

        self.merged_df = self.hourly_df.copy(deep=False)
        for col in indicator_cols:
            self.merged_df[f"daily_{col}"] = signal_df[col].values
    
    def run(self):
        self.logger.debug(f"Running RSM-VT FAST backtest for {self.strategy.name}...")
        df = self.merged_df
        n = len(df)
        
        open_prices = df['open'].values 
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        entry_upper = df['daily_entry_upper'].values
        entry_lower = df['daily_entry_lower'].values
        trend_dir = df['daily_trend_direction'].values
        strength_filter = df['daily_strength_filter'].values
        atr = df['daily_atr'].values
        macro_ema = df['daily_macro_ema'].values
        if "daily_garch_kelly_f" in df.columns:
            garch_kelly_f = np.asarray(df["daily_garch_kelly_f"].values, dtype=np.float64)
            if len(garch_kelly_f) < n:
                garch_kelly_f = np.pad(garch_kelly_f, (0, n - len(garch_kelly_f)), constant_values=1.0)
            garch_kelly_f = np.nan_to_num(garch_kelly_f[:n], nan=1.0, posinf=1.0, neginf=1.0)
        else:
            garch_kelly_f = np.ones(n, dtype=np.float64)

        # RSM-VT Params
        long_atr_mult = float(self.strategy.params.get('LONG_ATR_MULT', 3.0))
        long_trail_mult = float(self.strategy.params.get('LONG_TRAIL_MULT', 3.0))
        short_atr_mult = float(self.strategy.params.get('SHORT_ATR_MULT', 2.0))
        short_tp_mult = float(self.strategy.params.get('SHORT_TP_MULT', 3.0))
        long_scale_atr_mult = float(self.strategy.params.get('LONG_SCALE_ATR_MULT', 3.0))
        short_trail_mult = float(self.strategy.params.get('SHORT_TRAIL_MULT', 3.0))
        leverage = float(self.leverage)
        
        timestamps = df["timestamp"].values 

        if "funding_rate_sum" in self.hourly_df.columns:
            _fr_sum = np.asarray(self.hourly_df["funding_rate_sum"].values, dtype=np.float64)
            if len(_fr_sum) < n:
                _fr_sum = np.pad(_fr_sum, (0, n - len(_fr_sum)), constant_values=0.0)
            funding_rate_sums = np.nan_to_num(_fr_sum[:n], nan=0.0, posinf=0.0, neginf=0.0)
        else:
            if "funding_rate" in self.hourly_df.columns:
                _fr = np.asarray(self.hourly_df["funding_rate"].values, dtype=np.float64)
                if len(_fr) < n:
                    _fr = np.pad(_fr, (0, n - len(_fr)), constant_values=_fr[-1] if len(_fr) > 0 else FUNDING_FEE_RATE)
                funding_rates = np.nan_to_num(_fr[:n], nan=0.0, posinf=0.0, neginf=0.0)
            else:
                funding_rates = np.full(n, FUNDING_FEE_RATE, dtype=np.float64)

            funding_hours = ((timestamps // 1000) % 86400 // 3600).astype(np.int64)
            is_funding_bar = np.isin(funding_hours, np.array([0, 8, 16], dtype=np.int64))
            funding_rate_sums = funding_rates * is_funding_bar.astype(np.float64) * float(self.funding_events_per_bar)
        
        if getattr(self, "_warmup_bars_override", None) is not None:
            warmup_bars = self._warmup_bars_override
        else:
            warmup_bars = getattr(df, "attrs", {}).get(
                "warmup_bars",
                self.strategy.get_required_warmup(freq="hourly"),
            )
        self._warmup_bars = warmup_bars  
        self._effective_start_idx = max(warmup_bars, self._execution_start_idx)
        datetime_values = df['datetime'].values
        
        use_compounding = self.strategy.params.get('USE_COMPOUNDING', True)
        max_capital_usage = self.strategy.params.get('MAX_CAPITAL_USAGE', 1e12) 

        trades, final_balance, equity_curve, funding_paid_total = backtest_loop_numba(
            close, high, low, open_prices,
            entry_upper, entry_lower,
            trend_dir, strength_filter, atr, macro_ema,
            garch_kelly_f,
            self.initial_balance, leverage, self.fee_rate, self.slippage_rate,
            self.risk_per_trade, timestamps, funding_rate_sums,
            long_atr_mult, long_trail_mult, short_atr_mult, short_tp_mult,
            long_scale_atr_mult, short_trail_mult,
            warmup_bars, self._execution_start_idx,
            use_compounding, max_capital_usage
        )
        
        self.balance = final_balance
        self._equity_curve = equity_curve
        self._total_funding_paid = float(funding_paid_total)

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
                "side": "LONG" if trades[i][2] == 1 else "SHORT",
                "entry_price": trades[i][3],
                "exit_price": trades[i][4],
                "pnl": trades[i][5],
                "amount": trades[i][6],
                "entry_fee": trades[i][7],
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
        leverage_f = float(self.leverage)
        margin_used = amount_arr * entry_p / leverage_f
        margin_used = np.where(np.isfinite(margin_used) & (margin_used > 0), margin_used, np.nan)

        entry_fee_cumsum = np.concatenate(([0.0], np.cumsum(entry_fee_arr)[:-1]))
        pnl_cumsum = np.concatenate(([0.0], np.cumsum(pnl_arr)[:-1]))
        balance_before = self.initial_balance - entry_fee_cumsum + pnl_cumsum
        balance_before = np.where(balance_before == 0, 1e-9, balance_before)

        denom = np.where(
            np.isfinite(margin_used) & (margin_used > 0),
            margin_used,
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

        gross_pnl_abs = float(np.sum(np.abs(true_pnl)))

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
            'total_funding_paid': float(getattr(self, '_total_funding_paid', 0.0)),
            'gross_pnl_abs': gross_pnl_abs,
        }
        
    def _empty_result(self):
        return {
            'total_trades': 0, 'win_trades': 0, 'loss_trades': 0, 'win_rate': 0,
            'total_return_pct': 0, 'final_balance': self.initial_balance, 'mdd_pct': 0,
            'trades_df': pd.DataFrame(), 'equity_curve': np.array([]),
            'total_funding_paid': 0.0, 'gross_pnl_abs': 0.0,
        }

@njit(nogil=True, cache=True)
def backtest_loop_numba(
    close, high, low, open_prices,
    entry_upper, entry_lower,
    trend_dir, strength_filter, atr, macro_ema_arr,
    garch_kelly_f,
    initial_balance, leverage, fee_rate, slippage_rate,
    risk_per_trade, timestamps, funding_rate_sums,
    long_atr_mult, long_trail_mult, short_atr_mult, short_tp_mult,
    long_scale_atr_mult, short_trail_mult,
    warmup_bars, execution_start_idx,
    use_compounding, max_capital_usage
):
    funding_paid_total = 0.0
    n = len(close)
    balance = initial_balance
    peak_equity = initial_balance
    equity_curve = np.zeros(n)

    in_position = False
    pos_side = 0
    entry_price = 0.0
    entry_idx = 0
    amount = 0.0
    entry_fee_stored = 0.0

    stop_price = 0.0
    highest = 0.0
    has_scaled_out = False
    lowest = 0.0

    max_trades = 30000
    trades = np.zeros((max_trades, 8))
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

        # --- Bankruptcy Check ---
        current_eq_check = balance
        if in_position:
            current_eq_check = balance + (amount * entry_price) / leverage + (c_open - entry_price) * amount * pos_side
        if current_eq_check <= 0:
            equity_curve[i] = current_eq_check
            if in_position:
                exit_price = c_open
                if pos_side == 1: pnl = (exit_price - entry_price) * amount
                else:             pnl = (entry_price - exit_price) * amount
                exit_fee = amount * exit_price * fee_rate
                if trade_count < max_trades:
                    trades[trade_count] = [entry_idx, i, pos_side, entry_price, exit_price, pnl - exit_fee, amount, entry_fee_stored]
                    trade_count += 1
                in_position = False
                balance = 0.0
            break

        # --- 1. EXIT LOGIC ---
        if in_position:
            # Funding
            funding_rate_sum = funding_rate_sums[i]
            if not np.isnan(funding_rate_sum) and funding_rate_sum != 0.0:
                funding_cost = (amount * c_open) * funding_rate_sum * pos_side
                balance -= funding_cost
                funding_paid_total += funding_cost
                funding_eq_check = balance + (amount * entry_price) / leverage + (c_open - entry_price) * amount * pos_side
                if funding_eq_check <= 0:
                    exit_price = c_open
                    if pos_side == 1: pnl = (exit_price - entry_price) * amount
                    else:             pnl = (entry_price - exit_price) * amount
                    exit_fee = amount * exit_price * fee_rate
                    if trade_count < max_trades:
                        trades[trade_count] = [entry_idx, i, pos_side, entry_price, exit_price, pnl - exit_fee, amount, entry_fee_stored]
                        trade_count += 1
                    in_position = False
                    balance = 0.0
                    break
            
            exit_triggered = False
            exit_price = 0.0

            if pos_side == 1:
                if c_high > highest:
                    highest = c_high
                pos_atr = atr[entry_idx]
                
                # Long: scale-out 50% at ATR-based target from entry
                if not has_scaled_out:
                    scale_price = entry_price + (pos_atr * long_scale_atr_mult)
                    if c_high >= scale_price:
                        scale_exit_price = c_open if c_open >= scale_price else scale_price
                        scale_amount = amount / 2.0
                        pnl_scale = (scale_exit_price - entry_price) * scale_amount
                        exit_fee_scale = scale_amount * scale_exit_price * fee_rate
                        pnl_scale -= exit_fee_scale
                        balance += (scale_amount * entry_price) / leverage + pnl_scale
                        if trade_count < max_trades:
                            trades[trade_count] = [entry_idx, i, pos_side, entry_price, scale_exit_price, pnl_scale, scale_amount, entry_fee_stored / 2.0]
                            trade_count += 1
                        amount -= scale_amount
                        entry_fee_stored -= (entry_fee_stored / 2.0)
                        has_scaled_out = True

                # Long trailing stop for remainder
                if c_open <= stop_price:
                    exit_price = c_open * (1 - slippage_rate)
                    exit_triggered = True
                elif c_low <= stop_price:
                    exit_price = stop_price * (1 - slippage_rate)
                    exit_triggered = True
                if not exit_triggered:
                    new_stop = highest - (pos_atr * long_trail_mult)
                    if new_stop > stop_price:
                        stop_price = new_stop

            elif pos_side == -1:
                if c_low < lowest:
                    lowest = c_low
                tp_price = entry_price - (atr[entry_idx] * short_tp_mult)

                if not has_scaled_out:
                    if c_open <= tp_price or c_low <= tp_price:
                        scale_exit_price = c_open if c_open <= tp_price else tp_price
                        scale_amount = amount / 2.0
                        pnl_scale = (entry_price - scale_exit_price) * scale_amount
                        exit_fee_scale = scale_amount * scale_exit_price * fee_rate
                        pnl_scale -= exit_fee_scale
                        balance += (scale_amount * entry_price) / leverage + pnl_scale
                        if trade_count < max_trades:
                            trades[trade_count] = [entry_idx, i, pos_side, entry_price, scale_exit_price, pnl_scale, scale_amount, entry_fee_stored / 2.0]
                            trade_count += 1
                        amount -= scale_amount
                        entry_fee_stored -= (entry_fee_stored / 2.0)
                        has_scaled_out = True
                        breakeven_price = entry_price - (entry_price * fee_rate * 2.0)
                        stop_price = breakeven_price
                else:
                    new_stop = lowest + (atr[entry_idx] * short_trail_mult)
                    if new_stop < stop_price:
                        stop_price = new_stop

                if c_open >= stop_price:
                    exit_price = c_open * (1 + slippage_rate)
                    exit_triggered = True
                elif c_high >= stop_price:
                    exit_price = stop_price * (1 + slippage_rate)
                    exit_triggered = True

            if exit_triggered:
                if pos_side == 1: pnl = (exit_price - entry_price) * amount
                else:             pnl = (entry_price - exit_price) * amount
                
                exit_fee = amount * exit_price * fee_rate
                pnl -= exit_fee
                margin = (amount * entry_price) / leverage
                balance += margin + pnl
                
                if trade_count < max_trades:
                    trades[trade_count] = [entry_idx, i, pos_side, entry_price, exit_price, pnl, amount, entry_fee_stored]
                    trade_count += 1
                in_position = False
                bar_processed = True

        # --- 2. ENTRY LOGIC ---
        if not in_position and not bar_processed:
            prev_i = i - 1 if i > 0 else 0
            sf_raw = strength_filter[prev_i]
            if sf_raw <= 0.0 or np.isnan(sf_raw):
                equity_curve[i] = balance
                continue

            do_entry = False
            fill_price = 0.0
            pending_side = 0
            
            # Engine A: Bull Regime -> Long Breakout
            if trend_dir[prev_i] == 1:
                if c_high > entry_upper[prev_i]:
                    fill_price = max(c_open, entry_upper[prev_i]) * (1 + slippage_rate)
                    pending_side = 1
                    do_entry = True
                    
            # Engine B: Bear Regime -> Short Breakdown
            elif trend_dir[prev_i] == -1:
                if c_low < entry_lower[prev_i]:
                    fill_price = min(c_open, entry_lower[prev_i]) * (1 - slippage_rate)
                    pending_side = -1
                    do_entry = True

            if do_entry:
                prev_atr = atr[prev_i]
                if np.isnan(prev_atr) or prev_atr <= 0.0:
                    equity_curve[i] = balance
                    continue
                
                # Fixed Stop Loss Distance for Sizing
                if pending_side == 1:
                    stop_price = fill_price - (prev_atr * long_atr_mult)
                else:
                    stop_price = fill_price + (prev_atr * short_atr_mult)
                
                stop_distance = abs(fill_price - stop_price)
                if stop_distance > 0:
                    current_equity = max_capital_usage if use_compounding and balance > max_capital_usage else balance
                    kf = garch_kelly_f[prev_i]
                    if np.isnan(kf) or kf <= 0.0:
                        kf = 1.0
                    eff_risk = risk_per_trade * float(kf)
                    amount = (current_equity * eff_risk) / stop_distance
                    max_amount = (current_equity * leverage) / fill_price
                    amount = min(amount, max_amount)
                    sf_clamped = sf_raw if sf_raw <= 1.0 else 1.0
                    if sf_clamped < 0.0:
                        sf_clamped = 0.0
                    amount *= sf_clamped
                    
                    required_margin = (amount * fill_price) / leverage
                    entry_fee = amount * fill_price * fee_rate
                    if balance >= required_margin + entry_fee:
                        balance -= (required_margin + entry_fee)
                        entry_fee_stored = entry_fee
                        in_position = True
                        pos_side = pending_side
                        entry_price = fill_price
                        entry_idx = i
                        highest = fill_price
                        lowest = fill_price
                        has_scaled_out = False

                        # Intra-bar instant stop check
                        if pos_side == 1 and c_low <= stop_price:
                            intra_exit_price = stop_price * (1 - slippage_rate)
                            pnl = (intra_exit_price - entry_price) * amount
                            exit_fee_intra = amount * intra_exit_price * fee_rate
                            pnl -= exit_fee_intra
                            balance += (amount * entry_price) / leverage + pnl
                            if trade_count < max_trades:
                                trades[trade_count] = [entry_idx, i, pos_side, entry_price, intra_exit_price, pnl, amount, entry_fee_stored]
                                trade_count += 1
                            in_position = False
                        elif pos_side == -1 and c_high >= stop_price:
                            intra_exit_price = stop_price * (1 + slippage_rate)
                            pnl = (entry_price - intra_exit_price) * amount
                            exit_fee_intra = amount * intra_exit_price * fee_rate
                            pnl -= exit_fee_intra
                            balance += (amount * entry_price) / leverage + pnl
                            if trade_count < max_trades:
                                trades[trade_count] = [entry_idx, i, pos_side, entry_price, intra_exit_price, pnl, amount, entry_fee_stored]
                                trade_count += 1
                            in_position = False

        if in_position:
            margin = (amount * entry_price) / leverage
            unrealized = (c_price - entry_price) * amount * pos_side
            equity_curve[i] = balance + margin + unrealized
        else:
            equity_curve[i] = balance
            
        if equity_curve[i] > peak_equity:
            peak_equity = equity_curve[i]

    if in_position and n > 0:
        last_idx = n - 1
        last_close = close[last_idx]
        if pos_side == 1:
            exit_price = last_close * (1 - slippage_rate)
            pnl = (exit_price - entry_price) * amount
        else:
            exit_price = last_close * (1 + slippage_rate)
            pnl = (entry_price - exit_price) * amount

        exit_fee = amount * exit_price * fee_rate
        pnl -= exit_fee
        margin = (amount * entry_price) / leverage
        balance += margin + pnl
        if trade_count < max_trades:
            trades[trade_count] = [entry_idx, last_idx, pos_side, entry_price, exit_price, pnl, amount, entry_fee_stored]
            trade_count += 1

    return trades[:trade_count], balance, equity_curve, funding_paid_total
