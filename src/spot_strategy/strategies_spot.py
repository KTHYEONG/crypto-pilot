from __future__ import annotations

import numpy as np
import pandas as pd

from config.opt_config import OPT_SPOT_CONFIG
from src.strategy.base.ultimate import UltimateStrategyBase
from src.futures_strategy.strategies_futures import _FUTURES_INDICATORS as _SPOT_INDICATORS


class UltimateSpotStrategy(UltimateStrategyBase):
    """
    Spot trend-following: macro alignment, ADX strength, BB expansion + volume Z (AND or OR via VOL_CONFIRM_OR_MODE),
    Donchian breakout fill on next bar. BTC distance-based regime_risk_mult (continuous).
    """

    INDICATORS = _SPOT_INDICATORS
    ENTRY_SHIFT = False

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = df[col].astype(np.float64)

        if "btc_close" not in df.columns:
            df["btc_close"] = df["close"].astype(np.float64)

        ind = self._ind()

        macro_ema_period = int(self.params.get("MACRO_EMA_PERIOD", 200))
        adx_period = int(self.params.get("ADX_PERIOD", 14))
        adx_threshold = float(self.params.get("ADX_THRESHOLD", 25.0))
        momentum_period = int(self.params.get("MOMENTUM_PERIOD", 20))
        atr_period = int(self.params.get("ATR_PERIOD", 20))
        vol_z_threshold = float(self.params.get("VOL_Z_THRESHOLD", 1.0))

        bb_window = int(self.params.get("BB_WINDOW", OPT_SPOT_CONFIG.get("BB_WINDOW", 20)))
        vol_z_window = int(self.params.get("VOL_Z_WINDOW", OPT_SPOT_CONFIG.get("VOL_Z_WINDOW", 20)))
        vol_expansion_mult = float(
            self.params.get("VOL_EXPANSION_MULT", OPT_SPOT_CONFIG.get("VOL_EXPANSION_MULT", 1.05))
        )
        btc_regime_period = int(
            self.params.get("BTC_REGIME_SMA_PERIOD", self.params.get("MACRO_EMA_PERIOD", macro_ema_period))
        )

        df["atr"] = ind.calculate_atr(df, window=atr_period)
        df["macro_ema"] = ind.calculate_ema(df["close"], window=macro_ema_period)
        df["adx"] = ind.calculate_adx(df, window=adx_period)

        upper_bb, mid_bb, lower_bb = ind.calculate_bollinger_bands(df, window=bb_window)
        bb_width = (upper_bb - lower_bb) / mid_bb.replace(0.0, np.nan)
        bb_width_ma = bb_width.rolling(window=max(3, bb_window // 4), min_periods=1).mean()
        vol_expansion = (bb_width > (bb_width_ma * vol_expansion_mult)).fillna(False)

        vol = df["volume"].astype(np.float64)
        v_mu = vol.rolling(window=vol_z_window, min_periods=2).mean()
        v_sd = vol.rolling(window=vol_z_window, min_periods=2).std().replace(0.0, np.nan)
        vol_z = (vol - v_mu) / v_sd
        vol_z_ok = (vol_z > vol_z_threshold) & vol_z.notna()

        vol_confirm_or = bool(self.params.get("VOL_CONFIRM_OR_MODE", False))
        vol_confirm = (
            (vol_expansion | vol_z_ok) if vol_confirm_or else (vol_expansion & vol_z_ok)
        )

        df["dc_upper"] = df["high"].rolling(window=momentum_period).max().shift(1)

        bull_alignment = df["close"] > df["macro_ema"]
        strong_trend = df["adx"] > adx_threshold
        bull_breakout = bull_alignment & strong_trend & vol_confirm & (df["close"] > df["dc_upper"])

        df["long_entry_signal"] = np.where(bull_breakout, 1.0, 0.0)
        df["sig_long_entry_signal"] = df["long_entry_signal"]

        df["entry_upper"] = np.where(bull_breakout, df["dc_upper"], 999999.0)
        df["entry_lower"] = 999999.0

        df["strength_filter"] = np.where(bull_breakout, 1, 0)
        df["trend_direction"] = np.where(bull_breakout, 1, 0)

        if "btc_close" in df.columns:
            btc_s = df["btc_close"].astype(np.float64)
            btc_sma = ind.calculate_sma(btc_s, window=btc_regime_period)
        else:
            btc_s = df["close"]
            btc_sma = df["macro_ema"]

        dist = (btc_s - btc_sma) / (btc_sma.abs() + 1e-9)
        regime_risk_mult = 0.05 + 0.95 / (1.0 + np.exp(-dist.to_numpy(dtype=np.float64) * 3.0))
        df["regime_risk_mult"] = regime_risk_mult

        rs_num = df["close"] / df["close"].shift(20).replace(0.0, np.nan)
        if "btc_close" in df.columns:
            rs_den = (
                df["btc_close"] / df["btc_close"].shift(20).replace(0.0, np.nan)
            ).replace(0.0, np.nan)
            rs_ratio = (rs_num / rs_den).replace([np.inf, -np.inf], np.nan)
        else:
            rs_ratio = rs_num
        rs_score = np.log(rs_ratio.clip(lower=1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rs_score = np.clip(rs_score, -3.0, 3.0)

        break_score = ((df["close"] - df["dc_upper"]) / (df["atr"] + 1e-9)).clip(-6.0, 6.0)
        vol_persist = vol_expansion.astype(np.float64).rolling(3, min_periods=1).mean().fillna(0.0)
        regime_component = pd.Series(regime_risk_mult, index=df.index)

        df["slot_rank_score"] = (
            0.35 * rs_score
            + 0.25 * break_score
            + 0.20 * vol_persist
            + 0.20 * regime_component
        ).astype(np.float64)

        df["entry_upper"] = self._shift_if_needed(df["entry_upper"])
        df["entry_lower"] = self._shift_if_needed(df["entry_lower"])

        df["atr"] = df["atr"].ffill().fillna(df["close"] * 0.01)
        return df
