from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.core.indicators.indicators import get_indicator_engine
from src.core.indicators.numpy_ops_spot import compute_atr_numpy, compute_rsi_numpy
from src.domain.spot.regimes import REGIME_REGISTRY
from src.domain.spot.signals import SIGNAL_REGISTRY
from src.domain.spot.sizing import SIZING_REGISTRY
from src.strategy_base.pipeline_base import PipelineStrategyBase

_SPOT_INDICATORS = get_indicator_engine(domain="spot")

_REGIME_OFF_EPS: float = 1e-9
_SOFT_FULL_SPLIT: float = 0.85


def merge_exit_family_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Biases sampled engine params toward exit-family intent (tmp.md EXIT_FAMILY).
    Does not replace Optuna samples; tightens ranges per family.
    """
    p = dict(params)
    fam = str(p.get("EXIT_FAMILY", "BALANCED")).upper()
    if fam == "BALANCED":
        return p
    if fam == "TREND_HOLD":
        ts = int(p.get("TIME_STOP_BARS", 24))
        p["TIME_STOP_BARS"] = int(max(ts, 36))
        p["TRAIL_ATR_MULT"] = float(max(float(p.get("TRAIL_ATR_MULT", 4.0)), 4.5))
        p["SCALE_OUT_PCT"] = float(min(float(p.get("SCALE_OUT_PCT", 0.4)), 0.28))
        p["USE_TRAILING_STOP"] = True
        p["LONG_TP_MULT"] = 0.0
        return p
    if fam == "FAST_REALIZE":
        ts = int(p.get("TIME_STOP_BARS", 24))
        p["TIME_STOP_BARS"] = int(min(ts, 18))
        p["TRAIL_ATR_MULT"] = float(min(float(p.get("TRAIL_ATR_MULT", 4.0)), 4.0))
        p["SCALE_OUT_PCT"] = float(max(float(p.get("SCALE_OUT_PCT", 0.4)), 0.48))
        p["RSI_EXIT_THRESHOLD"] = float(min(float(p.get("RSI_EXIT_THRESHOLD", 85.0)), 78.0))
        p["LONG_TP_MULT"] = float(min(float(p.get("LONG_TP_MULT", 5.0)), 3.5))
        return p
    return p


class SpotPipelineStrategy(PipelineStrategyBase):
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

        n = len(df)

        if "btc_close" not in df.columns:
            df["btc_close"] = df["close"].astype(np.float64)
        ctx_maps = self._portfolio_eval_ctx
        if (
            "eth_close" not in df.columns
            and ctx_maps is not None
            and isinstance(ctx_maps.get("data_maps"), dict)
        ):
            dm = ctx_maps["data_maps"]
            tf_ctx = str(self.params.get("TIMEFRAME", "4h"))
            eth_pack = dm.get("KRW-ETH")
            eth_df = eth_pack.get(tf_ctx) if isinstance(eth_pack, dict) else None
            if eth_df is not None and hasattr(eth_df, "__len__") and len(eth_df) == n:
                df["eth_close"] = eth_df["close"].to_numpy(dtype=np.float64)

        eff_params = merge_exit_family_params(dict(self.params))

        st = str(eff_params.get("SIGNAL_TYPE", "ADX_BREAKOUT")).upper()
        sig_engine = SIGNAL_REGISTRY.get(st) or SIGNAL_REGISTRY["ADX_BREAKOUT"]
        sig_out = sig_engine.compute(df, eff_params)

        rt = str(eff_params.get("REGIME_TYPE", "MARKET_BREADTH")).upper()
        reg_engine = REGIME_REGISTRY.get(rt) or REGIME_REGISTRY["MARKET_BREADTH"]
        reg_arr = reg_engine.compute(self._regime_data_maps(df), eff_params)
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
            int(eff_params.get("ATR_PERIOD", 14)),
        )
        df["slot_rank_score"] = sig_out.rank_score.astype(np.float64)
        df["kill_signal"] = sig_out.kill_signal.astype(np.float64)
        df["fractal_high_flag"] = np.zeros(n, dtype=np.float64)
        df["regime_risk_mult"] = reg_arr.astype(np.float64)
        df["regime_entry_gate"] = np.where(reg_arr > _REGIME_OFF_EPS, 1.0, 0.0).astype(np.float64)
        rs_i = np.where(
            reg_arr <= _REGIME_OFF_EPS,
            0,
            np.where(reg_arr < _SOFT_FULL_SPLIT, 1, 2),
        ).astype(np.int32)
        df["regime_state"] = rs_i.astype(np.float64)
        df["regime_label"] = rs_i.astype(np.int32)
        out = self.apply_sizing(df, eff_params)
        return self._attach_exit_overlay(out, eff_params)

    def _attach_exit_overlay(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        close = df["close"].to_numpy(dtype=np.float64)
        bb_per = int(params.get("BB_EXIT_PERIOD", 20))
        bb_std_n = float(params.get("BB_EXIT_STD", 2.0))
        ma = pd.Series(close).rolling(max(2, bb_per)).mean().to_numpy(dtype=np.float64)
        std = pd.Series(close).rolling(max(2, bb_per)).std(ddof=0).to_numpy(dtype=np.float64)
        bb_up = ma + bb_std_n * std
        bb_up = np.where(np.isfinite(bb_up), bb_up, np.inf)
        df["bb_upper"] = bb_up.astype(np.float64)
        rsi_p = int(params.get("RSI_EXIT_PERIOD", 14))
        rsi_series = compute_rsi_numpy(close, max(2, rsi_p))
        th = float(params.get("RSI_EXIT_THRESHOLD", 85.0))
        df["trail_tighten_flag"] = (rsi_series > th).astype(np.float64)
        return df

    def apply_sizing(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        sm = str(params.get("SIZING_METHOD", "vol_target")).lower()
        sizer = SIZING_REGISTRY.get(sm) or SIZING_REGISTRY["vol_target"]
        df["garch_kelly_f"] = sizer.compute(df, params)
        return df
