"""
RSM-VT squeeze breakout (PipelineStrategyBase) adapted to FuturesSignalOutput.
Causal; no positive shifts on price paths.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.core.indicators.indicators import get_indicator_engine
from src.domain.futures.signals.base import FuturesSignalOutput
from src.domain.futures.signals.registry import register_futures_signal


# Futures 전용 엔진 확보
_ind = get_indicator_engine(domain="futures")


@register_futures_signal
class RsmVtFuturesSignal:
    name: ClassVar[str] = "RSM_VT"
    param_space: ClassVar[Dict[str, Any]] = {
        "MACRO_EMA_PERIOD": {"type": "int", "low": 50, "high": 200, "step": 10},
        "KC_MULT": {"type": "float", "low": 1.0, "high": 2.5, "step": 0.25},
        "SQUEEZE_WINDOW": {"type": "int", "low": 3, "high": 15, "step": 2},
        "MOMENTUM_PERIOD": {"type": "int", "low": 10, "high": 30, "step": 5},
        "VOL_Z_THRESHOLD": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.5},
        "EXHAUSTION_MULT": {"type": "float", "low": 2.0, "high": 5.0, "step": 0.5},
        "CVD_WINDOW": {"type": "int", "low": 3, "high": 20, "step": 2},
        "TAKER_RATIO_THRESHOLD": {"type": "float", "low": 0.4, "high": 0.7, "step": 0.05},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FuturesSignalOutput:
        work = df
        macro_ema_period = int(params.get("MACRO_EMA_PERIOD", 200))
        momentum_period = int(params.get("MOMENTUM_PERIOD", 20))
        kc_mult = float(params.get("KC_MULT", 1.5))
        atr_period = int(params.get("ATR_PERIOD", 20))
        vol_z_threshold = float(params.get("VOL_Z_THRESHOLD", 1.5))
        exhaustion_mult = float(params.get("EXHAUSTION_MULT", 4.0))
        squeeze_window = int(params.get("SQUEEZE_WINDOW", 5))

        work["atr"] = _ind.calculate_atr(work, window=atr_period)
        work["macro_ema"] = _ind.calculate_ema(work["close"], window=macro_ema_period)
        work["bb_mid"] = _ind.calculate_sma(work["close"], window=20)
        std_dev = work["close"].rolling(window=20).std()
        work["bb_upper"] = work["bb_mid"] + (std_dev * 2.0)
        work["bb_lower"] = work["bb_mid"] - (std_dev * 2.0)
        work["kc_mid"] = _ind.calculate_ema(work["close"], window=20)
        work["kc_upper"] = work["kc_mid"] + (work["atr"] * kc_mult)
        work["kc_lower"] = work["kc_mid"] - (work["atr"] * kc_mult)
        work["is_squeezing"] = (work["bb_upper"] < work["kc_upper"]) & (work["bb_lower"] > work["kc_lower"])
        work["recent_squeeze"] = work["is_squeezing"].rolling(window=squeeze_window).sum() > 0

        if "datetime" in work.columns:
            hours = work["datetime"].dt.hour
            grouped_vol = work.groupby(hours)["volume"]
            vol_mean = grouped_vol.transform(lambda x: x.rolling(window=10, min_periods=3).mean())
            vol_std = grouped_vol.transform(lambda x: x.rolling(window=10, min_periods=3).std())
        else:
            vol_mean = work["volume"].rolling(window=20).mean()
            vol_std = work["volume"].rolling(window=20).std()

        vol_std = vol_std.replace(0, 1e-8)
        work["vol_zscore"] = (work["volume"] - vol_mean) / vol_std
        vol_spike = work["vol_zscore"] > vol_z_threshold
        candle_range = work["high"] - work["low"]
        work["is_exhausted"] = candle_range > (work["atr"] * exhaustion_mult)

        cvd_window = int(params.get("CVD_WINDOW", 5))
        taker_ratio_threshold = float(params.get("TAKER_RATIO_THRESHOLD", 1.1))

        if "taker_buy_base_volume" in work.columns:
            work["taker_buy_base_volume"] = work["taker_buy_base_volume"].astype(np.float64).clip(lower=0.0)
            work["volume"] = work["volume"].astype(np.float64).clip(lower=0.0)
            work["taker_sell_volume"] = np.maximum(work["volume"] - work["taker_buy_base_volume"], 0.0)
            work["vol_delta"] = work["taker_buy_base_volume"] - work["taker_sell_volume"]
            work["cvd"] = work["vol_delta"].rolling(window=cvd_window).sum()
            safe_sell = work["taker_sell_volume"].replace(0.0, 1e-8)
            work["taker_ratio"] = work["taker_buy_base_volume"] / safe_sell
            cvd_bull_filter = (work["cvd"] > 0.0) & (work["taker_ratio"] >= taker_ratio_threshold)
            cvd_bear_filter = (work["cvd"] < 0.0) & (work["taker_ratio"] <= (1.0 / taker_ratio_threshold))
        else:
            cvd_bull_filter = pd.Series(True, index=work.index)
            cvd_bear_filter = pd.Series(True, index=work.index)

        work["dc_upper"] = work["high"].rolling(window=momentum_period).max()
        work["dc_lower"] = work["low"].rolling(window=momentum_period).min()

        macro_uptrend = work["close"] > work["macro_ema"]
        macro_downtrend = work["close"] < work["macro_ema"]

        bull_breakout = (
            work["recent_squeeze"]
            & macro_uptrend
            & (work["close"] > work["kc_upper"])
            & vol_spike
            & (~work["is_exhausted"])
            & cvd_bull_filter
        )
        bear_breakout = (
            work["recent_squeeze"]
            & macro_downtrend
            & (work["close"] < work["kc_lower"])
            & vol_spike
            & (~work["is_exhausted"])
            & cvd_bear_filter
        )

        bull = bull_breakout.to_numpy(dtype=np.bool_)
        bear = bear_breakout.to_numpy(dtype=np.bool_)
        n = len(bull)
        if n == 0:
            empty = np.array([], dtype=np.float64)
            return FuturesSignalOutput(
                long_entry=np.array([], dtype=np.bool_),
                short_entry=np.array([], dtype=np.bool_),
                kill_long=empty,
                kill_short=empty,
                rank_score=empty,
            )

        kl = work["is_exhausted"].to_numpy(dtype=np.float64)
        ks = work["is_exhausted"].to_numpy(dtype=np.float64)
        vz = work["vol_zscore"].replace(np.nan, 0.0).to_numpy(dtype=np.float64)
        rank = np.where(bull & ~bear, vz, np.where(bear & ~bull, -vz, 0.0)).astype(np.float64)
        return FuturesSignalOutput(
            long_entry=bull,
            short_entry=bear,
            kill_long=np.clip(kl, 0.0, 1.0),
            kill_short=np.clip(ks, 0.0, 1.0),
            rank_score=rank,
        )
