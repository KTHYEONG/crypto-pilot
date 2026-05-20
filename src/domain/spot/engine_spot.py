from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numba import njit

from config.settings import SLIPPAGE_RATE, TRADING_FEE_RATE


class BacktestEngineFastSpot:
    def __init__(
        self,
        hourly_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        strategy,
        initial_balance: float = 1_000_000,
        merge_index_map=None,
        precomputed_daily_df: pd.DataFrame | None = None,
        warmup_bars: int | None = None,
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

    _REQUIRED_INDICATOR_COLS: frozenset = frozenset(
        {
            "entry_upper",
            "trend_direction",
            "strength_filter",
            "atr",
            "regime_risk_mult",
            "regime_state",
            "kill_signal",
            "garch_kelly_f",
            "fractal_high_flag",
            "bb_upper",
            "trail_tighten_flag",
        }
    )

    def _prepare_data(self) -> None:
        exclude_cols = {
            "date_key",
            "datetime",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp",
        }
        if all(c in self.hourly_df.columns for c in self._REQUIRED_INDICATOR_COLS):
            signal_df = self.hourly_df
        else:
            signal_df = self.strategy.generate_signals(self.hourly_df.copy(deep=True))

        indicator_cols = [
            c
            for c in signal_df.columns
            if c not in exclude_cols and c in self._REQUIRED_INDICATOR_COLS
        ]

        self.merged_df = self.hourly_df.copy(deep=False)
        for col in indicator_cols:
            self.merged_df[f"daily_{col}"] = signal_df[col].values

    def run(self):
        self.logger.debug(f"Running FAST backtest for Spot {self.strategy.name}...")
        df = self.merged_df
        n = len(df)

        open_prices = df["open"].values
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        entry_upper = df["daily_entry_upper"].values
        trend_dir = df["daily_trend_direction"].values
        strength_filter = df["daily_strength_filter"].values
        atr = df["daily_atr"].values

        long_atr_mult = float(self.strategy.params.get("LONG_ATR_MULT", 3.0))
        long_trail_mult = float(
            self.strategy.params.get(
                "TRAIL_ATR_MULT", self.strategy.params.get("LONG_TRAIL_MULT", 3.0)
            )
        )
        use_trailing_stop = 1 if bool(self.strategy.params.get("USE_TRAILING_STOP", True)) else 0
        long_tp_mult = float(self.strategy.params.get("LONG_TP_MULT", 5.0))
        long_trail_lock_mult = float(self.strategy.params.get("LONG_TRAIL_LOCK_MULT", 1.5))
        tp_lock_mult = float(self.strategy.params.get("TP_LOCK_ATR_MULT", 3.0))

        timestamps = df["timestamp"].values

        if getattr(self, "_warmup_bars_override", None) is not None:
            warmup_bars = self._warmup_bars_override
        else:
            tf = self.strategy.params.get("TIMEFRAME", "1h")
            warmup_bars = getattr(df, "attrs", {}).get(
                "warmup_bars",
                self.strategy.get_required_warmup(freq=tf)
                if hasattr(self.strategy, "get_required_warmup")
                else 100,
            )
        self._warmup_bars = warmup_bars
        self._effective_start_idx = max(warmup_bars, self._execution_start_idx)
        datetime_values = df["datetime"].values

        use_compounding = self.strategy.params.get("USE_COMPOUNDING", True)
        max_capital_usage = self.strategy.params.get("MAX_CAPITAL_USAGE", 1e12)

        n_bars = len(close)
        regime_rm = (
            df["daily_regime_risk_mult"].values
            if "daily_regime_risk_mult" in df.columns
            else np.ones(n_bars, dtype=np.float64)
        )
        regime_entry_gate = (
            df["daily_regime_entry_gate"].values
            if "daily_regime_entry_gate" in df.columns
            else np.ones(n_bars, dtype=np.float64)
        )
        regime_state_1d = (
            df["daily_regime_state"].values
            if "daily_regime_state" in df.columns
            else np.full(n_bars, 2.0, dtype=np.float64)
        )
        garch_k = (
            df["daily_garch_kelly_f"].values
            if "daily_garch_kelly_f" in df.columns
            else np.ones(n_bars, dtype=np.float64)
        )
        kill_sig = (
            df["daily_kill_signal"].values
            if "daily_kill_signal" in df.columns
            else np.zeros(n_bars, dtype=np.float64)
        )
        fractal_hf = (
            df["daily_fractal_high_flag"].values
            if "daily_fractal_high_flag" in df.columns
            else np.zeros(n_bars, dtype=np.float64)
        )
        bb_upper = (
            df["daily_bb_upper"].values
            if "daily_bb_upper" in df.columns
            else np.full(n_bars, np.inf, dtype=np.float64)
        )
        trail_tighten = (
            df["daily_trail_tighten_flag"].values
            if "daily_trail_tighten_flag" in df.columns
            else np.zeros(n_bars, dtype=np.float64)
        )
        scale_out_ratio = float(self.strategy.params.get("SCALE_OUT_RATIO", 0.5))
        time_stop_bars = int(self.strategy.params.get("TIME_STOP_BARS", 0))

        kill_cd = int(self.strategy.params.get("KILL_COOLDOWN_BARS", 6))
        delta_gate = float(self.strategy.params.get("DELTA_GATE", 0.08))

        trades, final_balance, equity_curve = backtest_loop_numba_spot(
            close,
            high,
            low,
            open_prices,
            entry_upper,
            trend_dir,
            strength_filter,
            atr,
            regime_rm,
            garch_k,
            kill_sig,
            regime_entry_gate,
            regime_state_1d,
            self.initial_balance,
            self.fee_rate,
            self.slippage_rate,
            self.risk_per_trade,
            timestamps,
            long_atr_mult,
            long_trail_mult,
            long_tp_mult,
            long_trail_lock_mult,
            tp_lock_mult,
            use_trailing_stop,
            warmup_bars,
            self._execution_start_idx,
            use_compounding,
            max_capital_usage,
            kill_cd,
            delta_gate,
            fractal_hf,
            scale_out_ratio,
            time_stop_bars,
            bb_upper,
            trail_tighten,
            float(self.strategy.params.get("MAX_CAP_PER_COIN", 1.0)),
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
            self.trades.append(
                {
                    "entry_time": datetime_values[entry_idx],
                    "exit_time": datetime_values[exit_idx],
                    "side": "LONG",
                    "entry_price": trades[i][2],
                    "exit_price": trades[i][3],
                    "pnl": trades[i][4],
                    "amount": trades[i][5],
                    "entry_fee": trades[i][6],
                }
            )

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
        entry_fee_arr = np.fromiter(
            (t["entry_fee"] for t in self.trades), dtype=np.float64, count=n_trades
        )
        entry_p = np.fromiter(
            (t["entry_price"] for t in self.trades), dtype=np.float64, count=n_trades
        )
        amount_arr = np.fromiter(
            (t["amount"] for t in self.trades), dtype=np.float64, count=n_trades
        )

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

        true_pnl = pnl_arr
        capital_used = amount_arr * entry_p
        capital_used = np.where(
            np.isfinite(capital_used) & (capital_used > 0), capital_used, np.nan
        )

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
            "total_trades": n_trades,
            "win_trades": win_trades,
            "loss_trades": loss_trades,
            "win_rate": win_rate,
            "total_return_pct": total_return_pct,
            "final_balance": self.balance,
            "mdd_pct": mdd,
            "trades_df": trades_df,
            "equity_curve": equity_for_mdd,
        }

    def _empty_result(self):
        return {
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "win_rate": 0,
            "total_return_pct": 0,
            "final_balance": self.initial_balance,
            "mdd_pct": 0,
            "trades_df": pd.DataFrame(),
            "equity_curve": np.array([]),
        }


@njit(nogil=True, cache=True)
def backtest_loop_numba_spot(
    close,
    high,
    low,
    open_prices,
    entry_upper,
    trend_dir,
    strength_filter,
    atr,
    regime_risk_mult,
    garch_kelly_f,
    kill_signal,
    regime_entry_gate,
    regime_state,
    initial_balance,
    fee_rate,
    slippage_rate,
    risk_per_trade,
    timestamps,
    long_atr_mult,
    long_trail_mult,
    long_tp_mult,
    long_trail_lock_mult,
    tp_lock_mult,
    use_trailing_stop,
    warmup_bars,
    execution_start_idx,
    use_compounding,
    max_capital_usage,
    kill_cooldown_bars,
    delta_gate,
    fractal_high_flag,
    scale_out_ratio,
    time_stop_bars,
    bb_upper,
    trail_tighten_flag,
    max_position_pct,
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

    cooldown_remaining = 0
    last_entry_risk_pct = 0.0
    skip_cooldown_tick = False
    fractal_scale_done = False

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
                    trades[trade_count] = [
                        entry_idx,
                        i,
                        entry_price,
                        exit_price,
                        pnl - exit_fee,
                        amount,
                        entry_fee_stored,
                    ]
                    trade_count += 1
                in_position = False
                fractal_scale_done = False
                balance = 0.0
            break

        if in_position:
            exit_triggered = False
            exit_price = 0.0

            # Causal: kill_signal[i] uses bar-i close; exits at bar-i open — use prior bar.
            kill_now = (kill_signal[i - 1] > 0.5) if i > 0 and i - 1 < len(kill_signal) else False
            if kill_now:
                exit_price = c_open * (1.0 - slippage_rate)
                exit_triggered = True
                cooldown_remaining = kill_cooldown_bars
                skip_cooldown_tick = True

            if not exit_triggered:
                if c_high > highest:
                    highest = c_high

            if (
                (not exit_triggered)
                and (not fractal_scale_done)
                and fractal_high_flag[i] > 0.5
                and amount > 0.0
            ):
                cap_amt = amount * 0.99
                eff_scale = scale_out_ratio
                rst_fr = regime_state[i] if i < len(regime_state) else 2.0
                if rst_fr > 0.5 and rst_fr < 1.5:
                    eff_scale = min(0.95, scale_out_ratio * 1.12)
                scale_amount = amount * eff_scale
                if scale_amount > cap_amt:
                    scale_amount = cap_amt
                if scale_amount > 0.0:
                    scale_exit_price = c_open * (1.0 - slippage_rate)
                    scale_pnl = (scale_exit_price - entry_price) * scale_amount
                    scale_fee = scale_amount * scale_exit_price * fee_rate
                    scale_pnl -= scale_fee
                    balance += (scale_amount * entry_price) + scale_pnl
                    amount -= scale_amount
                    fractal_scale_done = True
                if amount < 1e-12:
                    exit_price = c_open * (1.0 - slippage_rate)
                    exit_triggered = True

            # Check Stop Loss first (Gap down)
            if not exit_triggered and c_open <= stop_price:
                exit_price = c_open * (1 - slippage_rate)
                exit_triggered = True
            elif not exit_triggered and c_low <= stop_price:
                exit_price = stop_price * (1 - slippage_rate)
                exit_triggered = True
            # Check Take Profit (Gap up or high breach)
            elif not exit_triggered and c_open >= tp_price:
                exit_price = c_open * (1 - slippage_rate)
                exit_triggered = True
            elif not exit_triggered and c_high >= tp_price:
                exit_price = tp_price * (1 - slippage_rate)
                exit_triggered = True
            elif (
                (not exit_triggered)
                and i < len(bb_upper)
                and np.isfinite(bb_upper[i])
                and bb_upper[i] < 1e18
                and c_high >= bb_upper[i]
            ):
                exit_price = bb_upper[i] * (1.0 - slippage_rate)
                exit_triggered = True

            if use_trailing_stop != 0 and (not exit_triggered):
                # Parabolic Tightening: If price moved enough from entry relative to ATR, use tighter trail
                pos_atr = atr[entry_idx]
                dist = highest - entry_price
                rst_i = regime_state[i] if i < len(regime_state) else 2.0
                trail_regime_factor = max(0.5, 1.0 - (2.0 - rst_i) * 0.14)
                current_trail_mult = long_trail_mult * trail_regime_factor
                if i < len(trail_tighten_flag) and trail_tighten_flag[i] > 0.5:
                    current_trail_mult = long_trail_lock_mult
                elif dist > (pos_atr * tp_lock_mult):
                    current_trail_mult = long_trail_lock_mult

                new_stop = highest - (pos_atr * current_trail_mult)
                if new_stop > stop_price:
                    stop_price = new_stop

            rst_ts = regime_state[i] if i < len(regime_state) else 2.0
            ts_regime_factor = max(0.50, 1.0 - (2.0 - rst_ts) * 0.35)
            eff_ts = max(1, int(time_stop_bars * ts_regime_factor)) if time_stop_bars > 0 else 0
            if (not exit_triggered) and time_stop_bars > 0 and (i - entry_idx) >= eff_ts:
                exit_price = c_open * (1.0 - slippage_rate)
                exit_triggered = True

            if exit_triggered:
                pnl = (exit_price - entry_price) * amount
                exit_fee = amount * exit_price * fee_rate
                pnl -= exit_fee
                balance += (amount * entry_price) + pnl

                if trade_count < max_trades:
                    trades[trade_count] = [
                        entry_idx,
                        i,
                        entry_price,
                        exit_price,
                        pnl,
                        amount,
                        entry_fee_stored,
                    ]
                    trade_count += 1
                in_position = False
                last_entry_risk_pct = 0.0
                fractal_scale_done = False
                bar_processed = True

        if not in_position and not bar_processed:
            prev_i = i - 1 if i > 0 else 0
            if strength_filter[prev_i] == 0 or np.isnan(strength_filter[prev_i]):
                equity_curve[i] = balance
                continue

            if cooldown_remaining > 0:
                equity_curve[i] = balance
                continue

            do_entry = False
            fill_price = 0.0

            if trend_dir[prev_i] == 1:
                eu_prev = entry_upper[prev_i]
                if eu_prev < 1.0:
                    fill_price = c_open * (1.0 + slippage_rate)
                    do_entry = True
                elif c_high > eu_prev:
                    fill_price = max(c_open, eu_prev) * (1.0 + slippage_rate)
                    do_entry = True

            if do_entry:
                ge = regime_entry_gate[prev_i] if prev_i < len(regime_entry_gate) else 1.0
                if ge < 0.5:
                    equity_curve[i] = balance
                    continue
                prev_atr = atr[prev_i]
                if np.isnan(prev_atr) or prev_atr <= 0.0:
                    equity_curve[i] = balance
                    continue

                stop_price = fill_price - (prev_atr * long_atr_mult)
                stop_distance = abs(fill_price - stop_price)
                if stop_distance > 0:
                    current_equity = (
                        max_capital_usage
                        if use_compounding and balance > max_capital_usage
                        else balance
                    )

                    rr = regime_risk_mult[prev_i] if prev_i < len(regime_risk_mult) else 1.0
                    gk = garch_kelly_f[prev_i] if prev_i < len(garch_kelly_f) else 1.0
                    if np.isnan(rr):
                        equity_curve[i] = balance
                        continue
                    if rr < 1e-9:
                        equity_curve[i] = balance
                        continue
                    # Numba nopython: np.clip(scalar) unsupported; use min/max.
                    rr = max(0.05, min(1.0, rr))
                    if np.isnan(gk) or gk <= 0.0:
                        gk = 1.0
                    eff = rr * gk
                    rst_ent = regime_state[prev_i] if prev_i < len(regime_state) else 2.0
                    if rst_ent > 0.5 and rst_ent < 1.5:
                        eff = eff * 0.90
                    if eff < 0.05:
                        eff = 0.05
                    if eff > 1.0:
                        eff = 1.0

                    new_risk_pct = risk_per_trade * eff
                    if (
                        last_entry_risk_pct > 1e-12
                        and abs(new_risk_pct - last_entry_risk_pct) < delta_gate
                    ):
                        equity_curve[i] = balance
                        continue

                    # Spot Allocation: Align with shared-cash Risk-based sizing
                    risk_budget = current_equity * new_risk_pct
                    raw_amount = risk_budget / stop_distance

                    # Cap by max_position_pct (Portfolio-aligned)
                    max_notional = current_equity * max_position_pct
                    amount_pos_cap = max_notional / fill_price

                    # Prevent allocating more than we have (Buffer for fees)
                    max_affordable = (balance * 0.99) / (fill_price * (1.0 + fee_rate))

                    amount = min(raw_amount, amount_pos_cap, max_affordable)

                    if amount > 1e-12:
                        required_capital = amount * fill_price
                        entry_fee = required_capital * fee_rate

                        if balance >= required_capital + entry_fee:
                            balance -= required_capital + entry_fee
                            entry_fee_stored = entry_fee
                            in_position = True
                            entry_price = fill_price
                            entry_idx = i
                            highest = fill_price
                            last_entry_risk_pct = new_risk_pct
                            fractal_scale_done = False
                            tp_price = fill_price + (prev_atr * long_tp_mult)

                            if c_low <= stop_price:
                                intra_exit_price = stop_price * (1 - slippage_rate)
                                pnl = (intra_exit_price - entry_price) * amount
                                exit_fee_intra = amount * intra_exit_price * fee_rate
                                pnl -= exit_fee_intra
                                balance += (amount * entry_price) + pnl
                                if trade_count < max_trades:
                                    trades[trade_count] = [
                                        entry_idx,
                                        i,
                                        entry_price,
                                        intra_exit_price,
                                        pnl,
                                        amount,
                                        entry_fee_stored,
                                    ]
                                    trade_count += 1
                                in_position = False
                                last_entry_risk_pct = 0.0
                                fractal_scale_done = False

        if in_position:
            unrealized = (c_price - entry_price) * amount
            equity_curve[i] = balance + (amount * entry_price) + unrealized
        else:
            equity_curve[i] = balance

        if equity_curve[i] > peak_equity:
            peak_equity = equity_curve[i]

        if not skip_cooldown_tick and cooldown_remaining > 0:
            cooldown_remaining -= 1
        skip_cooldown_tick = False

    if in_position and n > 0:
        last_idx = n - 1
        last_close = close[last_idx]
        exit_price = last_close * (1 - slippage_rate)
        pnl = (exit_price - entry_price) * amount

        exit_fee = amount * exit_price * fee_rate
        pnl -= exit_fee
        balance += (amount * entry_price) + pnl
        if trade_count < max_trades:
            trades[trade_count] = [
                entry_idx,
                last_idx,
                entry_price,
                exit_price,
                pnl,
                amount,
                entry_fee_stored,
            ]
            trade_count += 1

    return trades[:trade_count], balance, equity_curve
