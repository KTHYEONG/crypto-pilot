
from __future__ import annotations

from typing import Optional

import logging
import numpy as np
import pandas as pd
from numba import njit

from config.settings import TRADING_FEE_RATE, SLIPPAGE_RATE, FUNDING_FEE_RATE

class BacktestEngineFast:
    """
    Numba-accelerated Backtest Engine (5-10x faster)
    """
    def __init__(
        self,
        hourly_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        strategy,
        initial_balance: float = 1_000_000,
        merge_index_map=None,
        precomputed_daily_df: Optional[pd.DataFrame] = None,
        warmup_bars: Optional[int] = None,
    ):
        # Shallow copy (deep=False): new DataFrame object so _prepare_data can add columns without
        # mutating the caller's DataFrame (e.g. same df reused across optimization trials).
        # Underlying array data is shared—do not mutate existing columns in place.
        self.hourly_df = hourly_df.copy(deep=False)
        self.daily_df = daily_df.copy(deep=False)
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self._precomputed_daily_df = precomputed_daily_df
        self._warmup_bars_override = warmup_bars

        # injected by optimization script
        self.leverage = self.strategy.params.get("LEVERAGE", 1)
        self.risk_per_trade = self.strategy.params.get("RISK_PER_TRADE", 0.02)
        self.funding_events_per_bar = 1  # 1 for 4h/hourly; 3 for 1d (UTC 00, 08, 16 per day)

        self.fee_rate = TRADING_FEE_RATE
        self.slippage_rate = SLIPPAGE_RATE

        # Optional fast merge index injection (must be set before _prepare_data call)
        if merge_index_map is not None:
            self._merge_index_map = merge_index_map

        self.logger = logging.getLogger(__name__)
        self._prepare_data()

    # Columns from strategy.generate_signals() actually consumed by run() / backtest_loop_numba (V2).
    _REQUIRED_INDICATOR_COLS: frozenset = frozenset({
        "entry_upper", "entry_lower", "trend_direction", "strength_filter", "atr"
    })

    def _prepare_data(self) -> None:
        # [REFACTOR: INTRADAY NATIVE] Signals computed on target timeframe (hourly_df).
        # If hourly_df already has all required indicator cols (e.g. from signal cache), skip generate_signals.
        exclude_cols = {"date_key", "datetime", "date", "open", "high", "low", "close", "volume", "timestamp"}
        if all(c in self.hourly_df.columns for c in self._REQUIRED_INDICATOR_COLS):
            signal_df = self.hourly_df
        else:
            signal_df = self.strategy.generate_signals(self.hourly_df.copy(deep=True))

        indicator_cols = [
            c for c in signal_df.columns
            if c not in exclude_cols and c in self._REQUIRED_INDICATOR_COLS
        ]

        # Signals from generate_signals() are already computed on the correct bar index.
        # A second shift(1) here was causing: (1) NaN on the first bar of every sliced fold,
        # (2) 1-bar latency on all entries, (3) catastrophic -100% on short OOS windows
        # where entry_lower (float_max) became NaN after shift and bypassed NaN guard.
        self.merged_df = self.hourly_df.copy(deep=False)

        for col in indicator_cols:
            self.merged_df[f"daily_{col}"] = signal_df[col].values
    
    def run(self):
        self.logger.debug(f"Running FAST backtest for {self.strategy.name}...")
        
        df = self.merged_df
        
        # Daily signals are already aligned to prior day in _prepare_data_* via shift(1).
        # Do not shift again here to avoid extra 1-bar latency.
                
        # Extract all columns as numpy arrays for speed
        n = len(df)
        
        # Price columns
        open_prices = df['open'].values # [NEW] For Entry
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        # volume = df['volume'].values # Not needed for daily ratio check
        
        # Entry signals (Donchian or other)
        # Columns guaranteed by strategy.generate_signals
        entry_upper = df['daily_entry_upper'].values
        entry_lower = df['daily_entry_lower'].values
        
        # Trend filter
        trend_dir = df['daily_trend_direction'].values
        
        # Strength filter
        strength_filter = df['daily_strength_filter'].values
        
        # ATR for risk
        atr = df['daily_atr'].values
        
        # Strategy params
        atr_mult = self.strategy.params.get('ATR_MULTIPLIER', 3.0) # For Trailing Stop
        leverage = self.leverage
        
        # Extract timestamps for funding fee calculation
        timestamps = df["timestamp"].values  # milliseconds

        # Funding rate: trust only when present on hourly_df (true hourly series). Else constant + log.
        # merged_df may carry daily-mapped funding_rate (shift(1) daily → 24 identical values/day), which is unreliable.
        # Align length to n to avoid IndexError when merged_df is longer than hourly_df (e.g. after merge).
        if "funding_rate" in self.hourly_df.columns:
            _fr = np.asarray(self.hourly_df["funding_rate"].values, dtype=np.float64)
            if len(_fr) < n:
                _fr = np.pad(_fr, (0, n - len(_fr)), constant_values=_fr[-1] if len(_fr) > 0 else FUNDING_FEE_RATE)
            funding_rates = np.nan_to_num(_fr[:n], nan=0.0, posinf=0.0, neginf=0.0)
        else:
            funding_rates = np.full(n, FUNDING_FEE_RATE, dtype=np.float64)
            self.logger.debug("funding_rate column not found in hourly_df; using constant fallback.")
        
        # [NEW] Time-Based Exit & Trailing Activation
        max_holding_bars = self.strategy.params.get('MAX_HOLDING_BARS', 999999)  # Default: No limit
        trailing_activation_atr = self.strategy.params.get('TRAILING_ACTIVATION_ATR', 0.0)  # Default: Immediate activation
        time_exit_profit_threshold = self.strategy.params.get('TIME_EXIT_PROFIT_THRESHOLD', 0.5)  # Default: 0.5 ATR profit required to hold
        enable_trend_exit = self.strategy.params.get('ENABLE_TREND_EXIT', True)
        
        # [WARMUP OPTIMIZATION] Warmup in execution (hourly) bar count; fallback converts daily bars to hourly
        if getattr(self, "_warmup_bars_override", None) is not None:
            warmup_bars = self._warmup_bars_override
        else:
            warmup_bars = getattr(df, "attrs", {}).get(
                "warmup_bars",
                self.strategy.get_required_warmup(freq="hourly"),
            )
        self._warmup_bars = warmup_bars  # for get_results() MDD exclusion


        # [OPTIMIZATION] Extract timestamps for fast lookup (avoid iloc)
        datetime_values = df['datetime'].values
        
        # [NEW] Compounding Control (FIXED BUG: Removed artificial cap restricting geometric growth)
        use_compounding = self.strategy.params.get('USE_COMPOUNDING', True)
        max_capital_usage = self.strategy.params.get('MAX_CAPITAL_USAGE', 1e12) # Default to infinity for pure math
        
        # [NEW] Kelly Parameters (from opt_params or default)
        kelly_p = float(self.strategy.params.get('KELLY_P', 0.5))
        kelly_b = float(self.strategy.params.get('KELLY_B', 1.0))
        kelly_kappa = float(self.strategy.params.get('KELLY_KAPPA', 0.3))

        trades, final_balance, equity_curve = backtest_loop_numba(
            close, high, low, open_prices,
            entry_upper, entry_lower,
            trend_dir, strength_filter, atr,
            self.initial_balance,
            leverage,
            self.fee_rate,
            self.slippage_rate,
            atr_mult,
            self.risk_per_trade,
            timestamps, funding_rates,
            max_holding_bars, trailing_activation_atr, time_exit_profit_threshold,
            enable_trend_exit,
            warmup_bars,
            use_compounding,
            max_capital_usage,
            self.funding_events_per_bar,
            kelly_p,
            kelly_b,
            kelly_kappa,
        )
        
        self.balance = final_balance
        self._equity_curve = equity_curve

        # [MEMORY] Release large DataFrames before processing results
        self.merged_df = None
        self.hourly_df = None
        self.daily_df = None
        
        # Convert trades to list of dicts; Numba returns trades[:trade_count] so no padding rows
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
        
        result = self.get_results()
        return result
    
    def get_results(self):
        # [ROBUSTNESS] Exploded strategy (inf/nan balance): return invalid result so optimizer penalizes it
        if not np.isfinite(self.balance):
            self.logger.warning("Balance is non-finite (exploded strategy). Returning invalid result.")
            return {
                'total_trades': 0,
                'win_trades': 0,
                'loss_trades': 0,
                'win_rate': 0,
                'total_return_pct': 0,
                'final_balance': self.initial_balance,
                'mdd_pct': 0,
                'trades_df': pd.DataFrame(),
                'equity_curve': np.array([]),
            }

        total_return = self.balance - self.initial_balance
        total_return_pct = (total_return / self.initial_balance) * 100

        if not self.trades:
            return {
                'total_trades': 0,
                'win_trades': 0,
                'loss_trades': 0,
                'win_rate': 0,
                'total_return_pct': 0,
                'final_balance': self.initial_balance,
                'mdd_pct': 0,
                'trades_df': pd.DataFrame(),
                'equity_curve': np.array([]),
            }

        # Extract columns once as numpy arrays (avoid repeated DataFrame access)
        n_trades = len(self.trades)
        pnl_arr = np.fromiter((t["pnl"] for t in self.trades), dtype=np.float64, count=n_trades)
        entry_fee_arr = np.fromiter((t["entry_fee"] for t in self.trades), dtype=np.float64, count=n_trades)
        entry_p = np.fromiter((t["entry_price"] for t in self.trades), dtype=np.float64, count=n_trades)
        amount_arr = np.fromiter((t["amount"] for t in self.trades), dtype=np.float64, count=n_trades)

        # MDD from bar-level equity (includes unrealized P&L during open positions)
        equity = getattr(self, "_equity_curve", None)
        warmup_bars = getattr(self, "_warmup_bars", 0)
        equity_for_mdd: np.ndarray = np.array([])
        if equity is not None and len(equity) > 0 and np.isfinite(equity).all():
            if len(equity) > warmup_bars:
                equity_for_mdd = equity[warmup_bars:]
            else:
                equity_for_mdd = equity
            running_max = np.maximum.accumulate(equity_for_mdd)
            running_max[running_max == 0] = 1e-9
            drawdown = (equity_for_mdd - running_max) / running_max * 100
            drawdown = np.nan_to_num(drawdown, nan=0.0)
            mdd = float(drawdown.min())
        else:
            # Fallback: trade-level cumulative (numpy only)
            equity_for_mdd = np.array([])
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

        # ROE per trade: balance_before, pnl_pct (numpy-only)
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

        # Build DataFrame once for API contract (evaluator, metrics expect trades_df)
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



@njit(nogil=True, cache=True)
def backtest_loop_numba(
    close, high, low, open_prices,
    entry_upper, entry_lower,
    trend_dir, strength_filter, atr,
    initial_balance, leverage, fee_rate, slippage_rate,
    atr_mult,
    risk_per_trade,
    timestamps, funding_rates,
    max_holding_bars, trailing_activation_atr, time_exit_profit_threshold,
    enable_trend_exit,
    warmup_bars,  # [WARMUP] Number of bars to skip at start for indicator warmup
    use_compounding,   # [NEW] Bool
    max_capital_usage,  # [NEW] Float
    funding_events_per_bar,  # 1 for 4h/hourly; 3 for 1d (UTC 00, 08, 16 per bar)
    kelly_p,  # [NEW] Float (Win rate)
    kelly_b,  # [NEW] Float (Payoff ratio)
    kelly_kappa,  # [NEW] Float (Min Kelly fraction during DD)
):
    """
    Numba JIT-compiled backtest loop for Futures (Long/Short).
    Simulation premise:
    - Entry: Intra-bar breakout allowed. At bar i we use high[i]/low[i] to detect
      breakout and fill at max(open, level) (long) or min(open, level) (short).
      Real-time: you would not know bar high/low until bar close; this assumes
      tick-level touch within the bar is observable at bar processing time.
    - Exit: Slippage on all exits. Sequential stop logic. Funding at UTC 0/8/16.
    - [v16] Compounding control & capital caps.
    """
    n = len(close)
    balance = initial_balance
    peak_equity = initial_balance
    equity_curve = np.zeros(n)

    # Position state
    in_position = False
    pending_side = 0
    
    pos_side = 0  # 1: LONG, -1: SHORT
    entry_price = 0.0
    entry_idx = 0
    amount = 0.0
    entry_fee_stored = 0.0
    highest = 0.0
    lowest = 0.0
    pos_atr = 0.0
    stop_price = 0.0 # Initial Stop Loss 
    
    # Funding Fee Tracking (UTC-based: 00:00, 08:00, 16:00)
    last_funding_hour = -1
    
    # Trades storage (max 30000 trades)
    max_trades = 30000
    trades = np.zeros((max_trades, 8))  # [entry_idx, exit_idx, side, entry_p, exit_p, pnl, amount, entry_fee]
    trade_count = 0
    
    exec_risk = risk_per_trade # Local execution risk
    
    for i in range(n):
        # [WARMUP] Skip trading during warmup period
        if i < warmup_bars:
            equity_curve[i] = initial_balance
            continue

        # [SAFETY] Bankruptcy Check
        if balance <= 0:
            equity_curve[i] = balance
            break

        c_open = open_prices[i]
        c_price = close[i]
        c_high = high[i]
        c_low = low[i]
        current_timestamp = timestamps[i]
        execute_intra_bar = False
        bar_processed = False  # No re-entry on same bar after exit (Zombie fix)

        # --- 1. POSITION MANAGEMENT (Exits & Funding) ---
        if in_position:
            # A. Funding Fee (UTC 00, 08, 16)
            current_hour_utc = int((current_timestamp // 1000) % 86400 // 3600)
            is_funding_hour = (current_hour_utc in (0, 8, 16))
            
            if is_funding_hour and last_funding_hour != current_hour_utc:
                notional_value = amount * c_price
                rate = funding_rates[i]
                if np.isnan(rate):
                    rate = 0.0
                # Exchange convention: rate = what longs pay; long pays when rate>0, short receives.
                # funding_events_per_bar: 1 for 4h/hourly; 3 for 1d (one bar = one day = 3 UTC events).
                funding_cost = notional_value * rate * pos_side * funding_events_per_bar
                balance -= funding_cost
                last_funding_hour = current_hour_utc
                
                # Bankruptcy check from Funding (forced liquidation)
                # On liquidation do NOT add margin+pnl back: balance must go to 0 (or stay <= 0).
                if balance <= 0:
                    exit_price = c_price
                    if pos_side == 1: pnl = (exit_price - entry_price) * amount
                    else:             pnl = (entry_price - exit_price) * amount
                    
                    exit_fee = amount * exit_price * fee_rate
                    pnl -= exit_fee
                    balance = 0.0
                    
                    if trade_count < max_trades:
                        trades[trade_count] = [entry_idx, i, pos_side, entry_price, exit_price, pnl, amount, entry_fee_stored]
                        trade_count += 1
                    in_position = False
                    last_funding_hour = -1  # Reset so next position can be charged in same funding hour
                    break
            
            # Start-of-bar stop for same-bar exit (guard: no intra-bar time-order assumption)
            start_of_bar_stop = stop_price
            
            # [SEQ-1] Update High/Low & Trailing Stop in real time (exit_triggered-agnostic)
            # [INSTITUTIONAL] Dynamic Trailing Activation: Lower hurdle in chop, higher in trends.
            dynamic_trailing_act = trailing_activation_atr
            
            if pos_side == 1:
                if c_high > highest:
                    highest = c_high
                unreal_profit = (highest - entry_price) / pos_atr if pos_atr > 0 else 0
                if unreal_profit >= dynamic_trailing_act:
                    new_stop = max(highest - (pos_atr * atr_mult), entry_price * 0.01)
                    if new_stop > stop_price:
                        stop_price = new_stop
            else:
                if c_low < lowest:
                    lowest = c_low
                unreal_profit = (entry_price - lowest) / pos_atr if pos_atr > 0 else 0
                if unreal_profit >= dynamic_trailing_act:
                    new_stop = min(lowest + (pos_atr * atr_mult), entry_price * 1.99)
                    if new_stop < stop_price:
                        stop_price = new_stop
            
            # B. Exit Checks (use start_of_bar_stop so same-bar exit does not use same-bar-updated stop)
            exit_triggered = False
            exit_price = 0.0
            
            # [SEQ-2] Stop Loss (Gap-Adjusted Realism)
            current_stop = start_of_bar_stop

            if pos_side == 1:  # Long exit
                if c_open < current_stop:
                    # Gap down: liquidate at open (worst fill)
                    exit_price = c_open * (1 - slippage_rate)
                    exit_triggered = True
                elif c_low <= current_stop:
                    # Intraday touch: fill at stop level
                    exit_price = current_stop * (1 - slippage_rate)
                    exit_triggered = True
            else:  # Short exit
                if c_open > current_stop:
                    # Gap up: cover at open (worst fill)
                    exit_price = c_open * (1 + slippage_rate)
                    exit_triggered = True
                elif c_high >= current_stop:
                    exit_price = current_stop * (1 + slippage_rate)
                    exit_triggered = True


            # [SEQ-4] Conditional Market Exits
            if not exit_triggered:
                # Time Exit: after MAX_HOLDING_BARS, exit if profit below threshold; hard cap at 2x bars
                # Time Exit (Vol_Drag Early Exit): after MAX_HOLDING_BARS, exit if profit below threshold
                bars_held = i - entry_idx
                if bars_held > max_holding_bars:
                    # Evaluate unrealized PnL at open (no look-ahead)
                    if pos_side == 1:
                        unreal_p = (c_open - entry_price) / pos_atr if pos_atr > 0 else 0.0
                    else:
                        unreal_p = (entry_price - c_open) / pos_atr if pos_atr > 0 else 0.0

                    if unreal_p < time_exit_profit_threshold:
                        exit_price = c_open  # Market at bar open
                        if pos_side == 1:
                            exit_price *= (1 - slippage_rate)
                        else:
                            exit_price *= (1 + slippage_rate)
                        exit_triggered = True
                # Trend Reversal & Neutralization Exit (Edge loss exit)
                if not exit_triggered and enable_trend_exit:
                    prev_i = i - 1 if i > 0 else 0
                    if pos_side == 1 and trend_dir[prev_i] <= 0:  # Trend inverted or neutral
                        exit_price = c_open * (1 - slippage_rate)
                        exit_triggered = True
                    elif pos_side == -1 and trend_dir[prev_i] >= 0: # Trend inverted or neutral
                        exit_price = c_open * (1 + slippage_rate)
                        exit_triggered = True

            if exit_triggered:
                # PnL Calc with Slippage
                if pos_side == 1:
                    pnl = (exit_price - entry_price) * amount
                else:
                    pnl = (entry_price - exit_price) * amount
                
                exit_fee = amount * exit_price * fee_rate
                pnl -= exit_fee
                
                margin = (amount * entry_price) / leverage
                balance += margin + pnl
                
                if trade_count < max_trades:
                    trades[trade_count] = [entry_idx, i, pos_side, entry_price, exit_price, pnl, amount, entry_fee_stored]
                    trade_count += 1
                in_position = False
                last_funding_hour = -1  # Reset so next position can be charged in same funding hour
                bar_processed = True   # No re-entry on this bar (Zombie/Phantom fix)

        # --- 2. ENTRY: Intra-bar breakout (simulation premise: bar i high/low known at bar i) ---
        # Only check entry when no position AND this bar did not just exit (Zombie fix)
        elif not in_position and not bar_processed:
            prev_i = i - 1 if i > 0 else 0
            if np.isnan(entry_upper[prev_i]) or np.isnan(entry_lower[prev_i]):
                equity_curve[i] = balance
                continue
            if strength_filter[prev_i] == 0 or np.isnan(strength_filter[prev_i]):
                equity_curve[i] = balance
                continue

            # [NEW] Fixed Fractional Risk (1% Rule) - Removed Lookahead Kelly Compounding
            exec_risk = risk_per_trade

            do_entry = False
            fill_price = 0.0
            # Breakout: use bar i high/low (intra-bar premise); fill at first touch (open or level)
            if c_high > entry_upper[prev_i] and trend_dir[prev_i] == 1:
                fill_price = max(c_open, entry_upper[prev_i]) * (1 + slippage_rate)
                pending_side = 1
                do_entry = True
            elif c_low < entry_lower[prev_i] and trend_dir[prev_i] == -1:
                fill_price = min(c_open, entry_lower[prev_i]) * (1 - slippage_rate)
                pending_side = -1
                do_entry = True

            if do_entry and exec_risk > 0.0:
                # Use atr[prev_i] safely
                current_atr = atr[prev_i]
                if np.isnan(current_atr) or current_atr <= 0.0:
                    current_atr = atr[prev_i - 1] if prev_i > 0 else 0.0
                if np.isnan(current_atr) or current_atr <= 0.0:
                    current_atr = 0.0
                if current_atr <= 0.0:
                    equity_curve[i] = balance
                    continue
                
                if fill_price <= 0.0:
                    fill_price = 1e-9
                
                # ATR-based Initial Stop Loss
                if pending_side == 1:
                    stop_price = max(fill_price - (current_atr * atr_mult), fill_price * 0.01)
                else:
                    stop_price = min(fill_price + (current_atr * atr_mult), fill_price * 1.99)
                
                stop_distance = abs(fill_price - stop_price)
                if use_compounding:
                    current_equity = max_capital_usage if balance > max_capital_usage else balance
                else:
                    current_equity = initial_balance
                if stop_distance > 0:
                    amount = (current_equity * exec_risk) / stop_distance
                else:
                    amount = (current_equity * 0.01) / fill_price
                max_amount = (current_equity * leverage) / fill_price
                if amount > max_amount:
                    amount = max_amount
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
                    pos_atr = current_atr
                    execute_intra_bar = True

        # --- 3. INTRA-BAR SL/TP CHECK (same-bar exit after intra-bar entry; SL priority) ---
        if in_position and execute_intra_bar:
            intra_exit_triggered = False
            intra_exit_price = 0.0
            if pos_side == 1 and c_low <= stop_price:
                intra_exit_price = stop_price * (1 - slippage_rate)
                intra_exit_triggered = True
            elif pos_side == -1 and c_high >= stop_price:
                intra_exit_price = stop_price * (1 + slippage_rate)
                intra_exit_triggered = True

            if intra_exit_triggered:
                if pos_side == 1:
                    pnl = (intra_exit_price - entry_price) * amount
                else:
                    pnl = (entry_price - intra_exit_price) * amount
                exit_fee = amount * intra_exit_price * fee_rate
                pnl -= exit_fee
                margin = (amount * entry_price) / leverage
                balance += margin + pnl
                if trade_count < max_trades:
                    trades[trade_count] = [entry_idx, i, pos_side, entry_price, intra_exit_price, pnl, amount, entry_fee_stored]
                    trade_count += 1
                in_position = False
                last_funding_hour = -1  # Reset so next position can be charged in same funding hour
                pos_side = 0
                execute_intra_bar = False

        # [EQUITY] Bar-end equity for MDD (includes unrealized when in position)
        if in_position:
            margin = (amount * entry_price) / leverage
            unrealized = (c_price - entry_price) * amount * pos_side
            equity_curve[i] = balance + margin + unrealized
        else:
            equity_curve[i] = balance
            
        if equity_curve[i] > peak_equity:
            peak_equity = equity_curve[i]

    # --- 4. END-OF-DATA FORCED LIQUIDATION ---
    # Realize remaining position at the final bar close to avoid hidden unrealized PnL.
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

    return trades[:trade_count], balance, equity_curve
