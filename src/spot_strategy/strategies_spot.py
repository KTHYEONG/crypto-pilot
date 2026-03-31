from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.strategy.base.ultimate import UltimateStrategyBase
from src.futures_strategy.strategies_futures import _FUTURES_INDICATORS as _SPOT_INDICATORS
from src.spot_strategy.regimes import REGIME_REGISTRY
from src.spot_strategy.signals import SIGNAL_REGISTRY
from src.spot_strategy.signals.numpy_ops import compute_atr_numpy, compute_rsi_numpy
from src.spot_strategy.sizing import SIZING_REGISTRY


class UltimateSpotStrategy(UltimateStrategyBase):
    """
    Thin orchestrator: Signal + Regime + Sizing plugins → engine columns.
    Set `_portfolio_eval_ctx` to ``{"data_maps": ..., "symbols": ..., "tf": ...}`` for
    multi-symbol regime (e.g. market breadth) during portfolio optimization.
    """

    INDICATORS = _SPOT_INDICATORS
    ENTRY_SHIFT = False

    _portfolio_eval_ctx: Optional[Dict[str, Any]] = None

    def _compute_warmup_bars(self) -> int:
        return 300

    def _regime_data_maps(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        ctx = self._portfolio_eval_ctx
        if ctx is not None and isinstance(ctx.get("data_maps"), dict):
            return ctx["data_maps"]
        tf = str(self.params.get("TIMEFRAME", "4h"))
        return {"_spot_single": {tf: df}}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = df[col].astype(np.float64)

        if "btc_close" not in df.columns:
            df["btc_close"] = df["close"].astype(np.float64)

        st = str(self.params.get("SIGNAL_TYPE", "ADX_BREAKOUT")).upper()
        sig_engine = SIGNAL_REGISTRY.get(st) or SIGNAL_REGISTRY["ADX_BREAKOUT"]
        sig_out = sig_engine.compute(df, self.params)

        rt = str(self.params.get("REGIME_TYPE", "MARKET_BREADTH")).upper()
        reg_engine = REGIME_REGISTRY.get(rt) or REGIME_REGISTRY["MARKET_BREADTH"]
        reg_arr = reg_engine.compute(self._regime_data_maps(df), self.params)
        n = len(df)
        if int(reg_arr.size) != n:
            raise ValueError(f"regime length {reg_arr.size} != df {n}")

        df["long_entry_signal"] = sig_out.entry_signal.astype(np.float64)
        df["entry_upper"] = np.where(sig_out.entry_signal, 0.0, 999999.0)
        df["trend_direction"] = np.where(sig_out.entry_signal, 1, 0).astype(np.int32)
        df["strength_filter"] = np.where(sig_out.entry_signal, 1, 0).astype(np.int32)
        df["atr"] = compute_atr_numpy(
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["close"].to_numpy(),
            int(self.params.get("ATR_PERIOD", 14)),
        )
        df["slot_rank_score"] = sig_out.rank_score.astype(np.float64)
        df["kill_signal"] = sig_out.kill_signal.astype(np.float64)
        df["fractal_high_flag"] = np.zeros(n, dtype=np.float64)
        df["regime_risk_mult"] = reg_arr.astype(np.float64)
        df["regime_label"] = np.where(reg_arr >= 0.5, 3, 0).astype(np.int32)
        out = self.apply_sizing(df, self.params)
        return self._attach_exit_overlay(out)

    def _attach_exit_overlay(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].to_numpy(dtype=np.float64)
        bb_per = int(self.params.get("BB_EXIT_PERIOD", 20))
        bb_std_n = float(self.params.get("BB_EXIT_STD", 2.0))
        ma = pd.Series(close).rolling(max(2, bb_per)).mean().to_numpy(dtype=np.float64)
        std = pd.Series(close).rolling(max(2, bb_per)).std(ddof=0).to_numpy(dtype=np.float64)
        bb_up = ma + bb_std_n * std
        bb_up = np.where(np.isfinite(bb_up), bb_up, np.inf)
        df["bb_upper"] = bb_up.astype(np.float64)
        rsi_p = int(self.params.get("RSI_EXIT_PERIOD", 14))
        rsi_series = compute_rsi_numpy(close, max(2, rsi_p))
        th = float(self.params.get("RSI_EXIT_THRESHOLD", 85.0))
        df["trail_tighten_flag"] = (rsi_series > th).astype(np.float64)
        return df

    def apply_sizing(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        sm = str(params.get("SIZING_METHOD", "vol_target")).lower()
        sizer = SIZING_REGISTRY.get(sm) or SIZING_REGISTRY["vol_target"]
        df["garch_kelly_f"] = sizer.compute(df, params)
        return df
