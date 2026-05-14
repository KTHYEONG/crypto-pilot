"""Futures backtest orchestration: delegates Numba loops to ``portfolio.execution_sim``."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import (
    FUNDING_FEE_RATE,
    MAKER_FEE_RATE,
    SLIPPAGE_RATE,
    SMART_ORDER_OFFSET,
    TAKER_FEE_RATE,
)
from src.domain.futures.portfolio.execution_sim import (
    backtest_loop_single_numba,
    backtest_target_weights_numba,
    calculate_position_size,
    check_intra_bar_stop,
    check_long_exit,
    check_short_exit,
    process_long_scale_out,
    process_short_scale_out,
)
from src.domain.futures.portfolio.portfolio_constructor import (
    portfolio_weight_params_from_optuna,
    precompute_rebalance_weights,
)


def hours_per_bar_from_timeframe(tf: str) -> float:
    """Bars as hour-equivalents for vertical time barriers (supports Nh / Nd)."""
    t = (tf or "4h").strip().lower()
    if t.endswith("h"):
        try:
            return max(float(t[:-1]), 1e-6)
        except ValueError:
            return 4.0
    if t.endswith("d"):
        try:
            return max(float(t[:-1]) * 24.0, 1e-6)
        except ValueError:
            return 24.0
    return 4.0


def max_hold_bars_from_time_barrier(strategy_params: dict[str, Any]) -> int:
    """Max held bars before flat (0 = disabled). Uses TIME_BARRIER_H in hours vs bar width."""
    tb = float(strategy_params.get("TIME_BARRIER_H", 0.0))
    if tb <= 0.0:
        return 0
    hpb = hours_per_bar_from_timeframe(str(strategy_params.get("TIMEFRAME", "4h")))
    return max(1, int(math.ceil(tb / hpb)))


__all__ = [
    "FuturesBacktestEngine",
    "MultiSymbolEngine",
    "SingleSymbolEngine",
    "hours_per_bar_from_timeframe",
    "max_hold_bars_from_time_barrier",
    "backtest_loop_single_numba",
    "backtest_target_weights_numba",
    "calculate_position_size",
    "check_intra_bar_stop",
    "check_long_exit",
    "check_short_exit",
    "process_long_scale_out",
    "process_short_scale_out",
]


# =============================================================================
# ENGINE CLASSES
# =============================================================================


class SingleSymbolEngine:
    """Consolidated Single-symbol Futures Backtest Engine."""

    _REQUIRED_INDICATOR_COLS = frozenset({"entry_upper", "entry_lower", "trend_direction", "strength_filter", "atr", "macro_ema"})
    _OPTIONAL_MERGE_COLS = frozenset({"garch_kelly_f"})

    def __init__(
        self,
        hourly_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        strategy: Any,
        initial_balance: float = 1_000_000,
        precomputed_daily_df: pd.DataFrame | None = None,
        warmup_bars: int | None = None,
        execution_start_idx: int = 0,
    ) -> None:
        self.hourly_df = hourly_df.copy(deep=False)
        self.daily_df = daily_df.copy(deep=False)
        self.strategy = strategy
        self.initial_balance = initial_balance
        self._precomputed_daily_df = precomputed_daily_df
        self._warmup_bars_override = warmup_bars
        self._execution_start_idx = max(0, int(execution_start_idx))

        self.leverage: float = self.strategy.params.get("LEVERAGE", 1.0)
        self.risk_per_trade: float = self.strategy.params.get("RISK_PER_TRADE", 0.015)
        self.maker_fee = MAKER_FEE_RATE
        self.taker_fee = TAKER_FEE_RATE
        self.slippage_rate = SLIPPAGE_RATE
        self.smart_offset = SMART_ORDER_OFFSET

        self._prepare_data()

    def _prepare_data(self) -> None:
        exclude_cols = {"date_key", "datetime", "date", "open", "high", "low", "close", "volume", "timestamp"}
        if all(c in self.hourly_df.columns for c in self._REQUIRED_INDICATOR_COLS):
            signal_df = self.hourly_df
        else:
            signal_df = self.strategy.generate_signals(self.hourly_df.copy(deep=True))

        merge_keys = self._REQUIRED_INDICATOR_COLS | self._OPTIONAL_MERGE_COLS
        indicator_cols = [c for c in signal_df.columns if c not in exclude_cols and c in merge_keys]
        self.merged_df = self.hourly_df.copy(deep=False)
        for col in indicator_cols:
            self.merged_df[f"daily_{col}"] = signal_df[col].values

    def run(self) -> dict[str, Any]:
        df = self.merged_df
        n = len(df)
        open_prices, close, high, low = df["open"].values, df["close"].values, df["high"].values, df["low"].values
        entry_upper, entry_lower = df["daily_entry_upper"].values, df["daily_entry_lower"].values
        trend_dir, strength_filter = df["daily_trend_direction"].values, df["daily_strength_filter"].values
        atr, macro_ema = df["daily_atr"].values, df["daily_macro_ema"].values

        garch_kelly_f = df["daily_garch_kelly_f"].values if "daily_garch_kelly_f" in df.columns else np.ones(n)
        garch_kelly_f = np.nan_to_num(garch_kelly_f, nan=1.0)

        atr_mult = float(self.strategy.params.get("ATR_MULT", 3.0))
        trail_mult = float(self.strategy.params.get("TRAIL_MULT", 3.0))
        short_tp_mult = float(self.strategy.params.get("SHORT_TP_MULT", 3.0))
        long_scale_atr_mult = float(self.strategy.params.get("LONG_SCALE_ATR_MULT", 3.0))

        timestamps = df["timestamp"].values
        funding_rate_sums = df["funding_rate_sum"].values if "funding_rate_sum" in df.columns else np.full(n, FUNDING_FEE_RATE / 3.0)

        warmup_bars = self._warmup_bars_override if self._warmup_bars_override is not None else int(getattr(df, "attrs", {}).get("warmup_bars", self.strategy.get_required_warmup(freq=self.strategy.params.get("TIMEFRAME", "1h"))))
        self._warmup_bars = warmup_bars
        self._effective_start_idx = max(warmup_bars, self._execution_start_idx)

        hmm_crisis = df["hmm_prob_crisis"].values if "hmm_prob_crisis" in df.columns else np.zeros(n)
        hmm_mod_long = df["hmm_modulator_long"].values if "hmm_modulator_long" in df.columns else np.ones(n)
        long_mod_floor = float(self.strategy.params.get("LONG_MOD_FLOOR", 0.70))
        hmm_mod_long = np.maximum(hmm_mod_long, long_mod_floor)

        trades_raw, final_balance, equity_curve, funding_total = backtest_loop_single_numba(
            close, high, low, open_prices, entry_upper, entry_lower, trend_dir, strength_filter, atr, macro_ema, garch_kelly_f,
            self.initial_balance, self.leverage, self.maker_fee, self.taker_fee, self.slippage_rate, self.smart_offset, self.risk_per_trade, timestamps, funding_rate_sums,
            atr_mult, trail_mult, atr_mult, short_tp_mult, long_scale_atr_mult, trail_mult, warmup_bars, self._execution_start_idx,
            bool(self.strategy.params.get("USE_COMPOUNDING", True)), float(self.strategy.params.get("MAX_CAPITAL_USAGE", 1e12)),
            float(self.strategy.params.get("MAX_EXPOSURE_PER_COIN", 1.5)), float(self.strategy.params.get("DD_SCALING_THRESHOLD", 0.15)),
            hmm_crisis, hmm_mod_long
        )

        self.balance = final_balance
        self._equity_curve = equity_curve
        self._total_funding_paid = funding_total

        datetime_vals = df["datetime"].values
        self.trades = []
        for i in range(len(trades_raw)):
            e_idx, x_idx = int(trades_raw[i][0]), int(trades_raw[i][1])
            self.trades.append({
                "entry_time": datetime_vals[e_idx], "exit_time": datetime_vals[x_idx],
                "side": "LONG" if trades_raw[i][2] == 1 else "SHORT",
                "entry_price": trades_raw[i][3], "exit_price": trades_raw[i][4],
                "pnl": trades_raw[i][5], "amount": trades_raw[i][6], "entry_fee": trades_raw[i][7]
            })
        return self.get_results()

    def get_results(self) -> dict[str, Any]:
        if not np.isfinite(self.balance) or not self.trades:
            return self._empty_result()

        total_return_pct = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        pnl_arr = np.array([t["pnl"] for t in self.trades])
        win_trades = int(np.sum(pnl_arr > 0))

        equity = self._equity_curve[self._effective_start_idx:] if len(self._equity_curve) > self._effective_start_idx else self._equity_curve
        if len(equity) > 0:
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / np.where(running_max == 0, 1e-9, running_max) * 100
            mdd = float(drawdown.min())
        else: mdd = 0.0

        return {
            "total_trades": len(self.trades), "win_trades": win_trades, "loss_trades": len(self.trades) - win_trades,
            "win_rate": (win_trades / len(self.trades)) * 100, "total_return_pct": total_return_pct,
            "final_balance": self.balance, "mdd_pct": mdd, "trades_df": pd.DataFrame(self.trades),
            "equity_curve": equity, "total_funding_paid": float(self._total_funding_paid),
            "gross_pnl_abs": float(np.sum(np.abs(pnl_arr)))
        }

    def _empty_result(self) -> dict[str, Any]:
        return {"total_trades": 0, "win_trades": 0, "loss_trades": 0, "win_rate": 0, "total_return_pct": 0, "final_balance": self.initial_balance, "mdd_pct": 0, "trades_df": pd.DataFrame(), "equity_curve": np.array([]), "total_funding_paid": 0.0, "gross_pnl_abs": 0.0}


class MultiSymbolEngine:
    """Multi-symbol portfolio: QP-style target weights + ``backtest_target_weights_numba``."""

    def __init__(
        self,
        aligned_data: dict[str, np.ndarray],
        symbol_names: list[str],
        strategy_params: dict[str, Any],
        initial_balance: float = 1_000_000,
        fee_rate: float | None = None,
        slippage_rate: float | None = None,
        maker_fee: float | None = None,
        taker_fee: float | None = None,
        smart_offset: float | None = None,
    ) -> None:
        self.data = aligned_data
        self.symbols = symbol_names
        self.params = strategy_params
        self.initial_balance = initial_balance
        self.maker_fee = maker_fee if maker_fee is not None else MAKER_FEE_RATE
        self.taker_fee = taker_fee if taker_fee is not None else TAKER_FEE_RATE
        self.slippage_rate = slippage_rate if slippage_rate is not None else SLIPPAGE_RATE
        self.smart_offset = smart_offset if smart_offset is not None else SMART_ORDER_OFFSET

        self.leverage = float(self.params.get("LEVERAGE", 1.0))
        self.risk_per_trade = float(self.params.get("RISK_PER_TRADE", 0.02))
        self.max_exposure = float(self.params.get("MAX_EXPOSURE", 0.8))
        self.max_concurrent_positions = int(OPT_FUTURES_CONFIG.get("FUTURES_MAX_CONCURRENT_POSITIONS", 2))

    def run(self) -> tuple[pd.DataFrame, np.ndarray, float, np.ndarray]:
        d = self.data
        c2d = d["close"]
        lev_2d = d.get("dyn_leverage", np.full(c2d.shape, self.leverage))

        buf = float(
            self.params.get(
                "SLIPPAGE_BPS_BUFFER_MULT",
                OPT_FUTURES_CONFIG.get("SLIPPAGE_BPS_BUFFER_MULT", 1.0),
            )
        )
        slip_eff = float(self.slippage_rate) * max(buf, 1e-9)
        mx_hold = max_hold_bars_from_time_barrier(self.params)
        sborr = float(OPT_FUTURES_CONFIG.get("FUTURES_SHORT_BORROW_DAILY", 0.0))

        tw_raw = d.get("target_weights")
        tw_arr: np.ndarray
        if tw_raw is not None:
            tw_arr = np.asarray(tw_raw, dtype=np.float64)
        else:
            pw = portfolio_weight_params_from_optuna(self.params, OPT_FUTURES_CONFIG)
            hpb = hours_per_bar_from_timeframe(str(self.params.get("TIMEFRAME", "4h")))
            bars_py = (365.0 * 24.0) / max(hpb, 1e-9)
            hmm_probs_2d = None
            hmm_cols = [
                "hmm_prob_bull_calm",
                "hmm_prob_bull_vol_up",
                "hmm_prob_bear_trend",
                "hmm_prob_chop",
                "hmm_prob_crisis",
            ]
            hmm_blocks = [d.get(c) for c in hmm_cols]
            if all(b is not None for b in hmm_blocks):
                hmm_arr_cols = []
                for b in hmm_blocks:
                    arr = np.asarray(b, dtype=np.float64)
                    hmm_arr_cols.append(arr[:, 0] if arr.ndim == 2 else arr)
                hmm_probs_2d = np.stack(hmm_arr_cols, axis=1)
            tw_arr = precompute_rebalance_weights(
                c2d,
                np.asarray(d.get("xs_score_long", np.zeros_like(c2d)), dtype=np.float64),
                np.asarray(d.get("xs_score_short", np.zeros_like(c2d)), dtype=np.float64),
                rebalance_bars=max(1, int(self.params.get("REBALANCE_BARS", 6))),
                lookback=int(pw["lookback"]),
                bars_per_year=bars_py,
                kappa=float(pw["kappa"]),
                f_kelly_max=float(pw["f_kelly_max"]),
                sigma_target_ann=float(pw["sigma_target_ann"]),
                gross_cap=float(pw["gross_cap"]),
                per_symbol_cap=float(pw["per_symbol_cap"]),
                current_dd=0.0,
                composer_sigma_2d=(
                    np.asarray(d["composer_sigma_bar"], dtype=np.float64)
                    if d.get("composer_sigma_bar") is not None
                    else None
                ),
                hmm_probs_2d=hmm_probs_2d,
                regime_policy_enabled=bool(pw.get("regime_policy_enabled", False)),
                chop_gross_damp=float(pw.get("chop_gross_damp", 0.50)),
                crisis_gross_damp=float(pw.get("crisis_gross_damp", 0.80)),
                entropy_gross_damp=float(pw.get("entropy_gross_damp", 0.35)),
                bear_gross_damp=float(pw.get("bear_gross_damp", 0.10)),
                gross_floor_mult=float(pw.get("gross_floor_mult", 0.15)),
                crisis_long_suppress_thr=float(pw.get("crisis_long_suppress_thr", 0.60)),
                crisis_long_suppress_mult=float(pw.get("crisis_long_suppress_mult", 0.10)),
            )

        if tw_arr.shape != c2d.shape:
            raise ValueError(
                f"target_weights shape {tw_arr.shape} != close shape {c2d.shape}; "
                "aligned OOS/portfolio data is inconsistent."
            )

        use_simple_atr_i = (
            1 if bool(OPT_FUTURES_CONFIG.get("FUTURES_SIMPLE_ATR_STOP", True)) else 0
        )

        trades_arr, final_bal, equity, diag = backtest_target_weights_numba(
            c2d,
            d["high"],
            d["low"],
            d["open"],
            d.get("funding_rate_sum", np.zeros_like(c2d)),
            d.get("kill_signal", np.zeros_like(c2d)),
            tw_arr,
            self.initial_balance,
            lev_2d,
            self.maker_fee,
            self.taker_fee,
            slip_eff,
            max(1, int(self.params.get("REBALANCE_BARS", 6))),
            mx_hold,
            sborr,
            d["atr"],
            float(self.params.get("ATR_MULT", 3.0)),
            float(self.params.get("TRAIL_MULT", 3.0)),
            use_simple_atr_i,
            self.max_concurrent_positions,
            self.max_exposure,
            float(self.params.get("MAX_EXPOSURE_PER_COIN", 1.5)),
            float(self.params.get("DD_SCALING_THRESHOLD", 0.0)),
        )
        if trades_arr.size == 0:
            return pd.DataFrame(), equity, final_bal, diag
        df = pd.DataFrame(trades_arr, columns=["sym_idx", "entry_idx", "exit_idx", "side_val", "entry_price", "exit_price", "pnl", "amount", "entry_fee", "funding_fee"])
        df["symbol"] = [self.symbols[int(i)] for i in df["sym_idx"]]
        df["side"] = np.where(df["side_val"] == 1.0, "LONG", "SHORT")
        return df[["symbol", "entry_idx", "exit_idx", "side", "entry_price", "exit_price", "pnl", "amount", "entry_fee", "funding_fee"]], equity, final_bal, diag


class FuturesBacktestEngine:
    """Unified entry point for Futures Backtesting."""

    @staticmethod
    def run_single(hourly_df: pd.DataFrame, daily_df: pd.DataFrame, strategy: Any, **kwargs) -> dict[str, Any]:
        engine = SingleSymbolEngine(hourly_df, daily_df, strategy, **kwargs)
        return engine.run()

    @staticmethod
    def run_multi(aligned_data: dict[str, np.ndarray], symbol_names: list[str], strategy_params: dict[str, Any], **kwargs) -> tuple[pd.DataFrame, np.ndarray, float, np.ndarray]:
        engine = MultiSymbolEngine(aligned_data, symbol_names, strategy_params, **kwargs)
        return engine.run()
