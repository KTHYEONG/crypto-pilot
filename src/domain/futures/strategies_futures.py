from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.indicators.indicators import get_indicator_engine
from src.domain.futures.regimes import FUTURES_REGIME_REGISTRY
from src.domain.futures.signals import FUTURES_SIGNAL_REGISTRY
from src.domain.futures.sizing import FUTURES_SIZING_REGISTRY
from src.strategy_base import (
    MasterStrategyBase,
    PipelineStrategyBase,
    StrategyBase,
)


class Strategy(StrategyBase):
    pass


class MasterStrategy(MasterStrategyBase):
    pass


_FUTURES_INDICATORS = get_indicator_engine(domain="futures")


class UltimateStrategy(PipelineStrategyBase):
    """
    Futures: plugin signal × regime × sizing dispatch.
    ENTRY_SHIFT: signal at i uses bar i-1 for entry columns (engine reads prev bar).
    """

    INDICATORS = _FUTURES_INDICATORS
    ENTRY_SHIFT = False

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)

        st_key = str(self.params.get("SIGNAL_TYPE", "RSM_VT")).upper()
        sig_engine = FUTURES_SIGNAL_REGISTRY.get(st_key) or FUTURES_SIGNAL_REGISTRY["RSM_VT"]
        sig_out = sig_engine.compute(df, self.params)

        rt_key = str(self.params.get("REGIME_TYPE", "EMA_ATR")).upper()
        reg_engine = FUTURES_REGIME_REGISTRY.get(rt_key) or FUTURES_REGIME_REGISTRY["EMA_ATR"]
        long_mult, short_mult = reg_engine.compute_long_short_mult(df, self.params)

        sm_key = str(self.params.get("SIZING_METHOD", "vol_target")).lower()
        sizer = FUTURES_SIZING_REGISTRY.get(sm_key) or FUTURES_SIZING_REGISTRY["vol_target"]

        long_e = sig_out.long_entry.astype(np.bool_)
        short_e = sig_out.short_entry.astype(np.bool_)
        close_a = df["close"].to_numpy(dtype=np.float64)
        df["trend_direction"] = np.where(long_e, 1.0, np.where(short_e, -1.0, 0.0))
        df["entry_upper"] = np.where(long_e, 0.0, 999999.0)
        df["entry_lower"] = np.where(short_e, close_a, 0.0)

        df["entry_upper"] = self._shift_if_needed(df["entry_upper"])
        df["entry_lower"] = self._shift_if_needed(df["entry_lower"])

        td = df["trend_direction"].to_numpy(dtype=np.float64)
        df["kill_signal"] = np.where(td == 1.0, sig_out.kill_long, sig_out.kill_short)
        df["slot_rank_score"] = sig_out.rank_score

        atr_period = int(self.params.get("ATR_PERIOD", 20))
        macro_ema_period = int(self.params.get("MACRO_EMA_PERIOD", 200))
        ind = self._ind()
        if "atr" not in df.columns or df["atr"].isna().all():
            df["atr"] = ind.calculate_atr(df, window=atr_period)
        df["atr"] = df["atr"].ffill().fillna(df["close"] * 0.01)
        if "macro_ema" not in df.columns or df["macro_ema"].isna().all():
            df["macro_ema"] = ind.calculate_ema(df["close"], window=macro_ema_period)

        kelly_f = sizer.compute(df, self.params)
        df["garch_kelly_f"] = kelly_f
        # Kelly scales risk in engine via garch_kelly_f; regime scales conviction here only.
        df["strength_filter"] = np.where(
            long_e,
            long_mult.astype(np.float64),
            np.where(short_e, short_mult.astype(np.float64), 0.0),
        )
        
        # Expose continuous regime state for TIER 4 OOS diagnostic block
        df["regime_risk_mult"] = long_mult.astype(np.float64)

        return df


# Alias for backward compatibility
FuturesPipelineStrategy = UltimateStrategy
