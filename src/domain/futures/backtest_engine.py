"""Futures portfolio backtest engine: multi-symbol, target-weight driven, Numba-accelerated."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import (
    MAKER_FEE_RATE,
    SLIPPAGE_RATE,
    SMART_ORDER_OFFSET,
    TAKER_FEE_RATE,
)
from src.domain.futures.backtest_preparation import prepare_backtest_inputs
from src.domain.futures.portfolio.execution_sim import (
    backtest_target_weights_intrabar_numba,
    backtest_target_weights_numba,
    check_long_exit,
    check_short_exit,
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
    return max(1, math.ceil(tb / hpb))


__all__ = [
    "FuturesBacktestEngine",
    "PortfolioBacktestEngine",
    "backtest_target_weights_intrabar_numba",
    "backtest_target_weights_numba",
    "check_long_exit",
    "check_short_exit",
    "hours_per_bar_from_timeframe",
    "max_hold_bars_from_time_barrier",
]


# =============================================================================
# ENGINE CLASSES
# =============================================================================


class PortfolioBacktestEngine:
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
        """Initialize engine state and execution parameters."""
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
        """Execute multi-symbol futures backtest."""
        prepared = prepare_backtest_inputs(self.data, self.params)
        d = prepared.aligned_data
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

        use_intrabar = (
            prepared.execution_mode == "intrabar_1m"
            and prepared.exec_bar_start_1m_idx is not None
            and prepared.exec_bar_end_1m_idx is not None
            and d.get("exec_open_1m") is not None
            and d.get("exec_high_1m") is not None
            and d.get("exec_low_1m") is not None
            and d.get("exec_close_1m") is not None
        )
        if use_intrabar:
            trades_arr, final_bal, equity, diag = backtest_target_weights_intrabar_numba(
                c2d,
                d["high"],
                d["low"],
                d["open"],
                tw_arr,
                lev_2d,
                d["atr"],
                d.get("kill_signal", np.zeros_like(c2d)),
                np.asarray(d["exec_open_1m"], dtype=np.float64),
                np.asarray(d["exec_high_1m"], dtype=np.float64),
                np.asarray(d["exec_low_1m"], dtype=np.float64),
                np.asarray(d["exec_close_1m"], dtype=np.float64),
                prepared.exec_bar_start_1m_idx,
                prepared.exec_bar_end_1m_idx,
                self.initial_balance,
                self.maker_fee,
                self.taker_fee,
                slip_eff,
                max(1, int(self.params.get("REBALANCE_BARS", 6))),
                mx_hold,
                sborr,
                float(self.params.get("ATR_MULT", 3.0)),
                float(self.params.get("TRAIL_MULT", 3.0)),
                use_simple_atr_i,
                self.max_concurrent_positions,
                self.max_exposure,
                float(self.params.get("MAX_EXPOSURE_PER_COIN", 1.5)),
                float(self.params.get("DD_SCALING_THRESHOLD", 0.0)),
                funding_event_mask_1m=d.get("funding_event_mask_1m"),
                funding_rate_1m=d.get("funding_rate_event_1m"),
                volume_1m_2d=d.get("exec_volume_1m"),
            )
        else:
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
                d.get("volume"),
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
    def run_multi(
        aligned_data: dict[str, np.ndarray],
        symbol_names: list[str],
        strategy_params: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, np.ndarray, float, np.ndarray]:
        """Run the portfolio backtest from aligned arrays."""
        engine = PortfolioBacktestEngine(aligned_data, symbol_names, strategy_params, **kwargs)
        return engine.run()
