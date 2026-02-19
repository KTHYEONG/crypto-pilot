from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from .core import StrategyBase


class UltimateStrategyBase(StrategyBase):
    """
    Shared signal-generation pipeline for Spot/Futures.
    Subclasses only provide indicator bindings and a few policy toggles.
    """

    INDICATORS: SimpleNamespace | None = None
    ENTRY_SHIFT: bool = False
    ATR_ALWAYS_ON: bool = True
    RSI_OVERBOUGHT_KEYS: tuple[str, ...] = ("RSI_OVERBOUGHT",)

    def _ind(self) -> SimpleNamespace:
        if self.INDICATORS is None:
            raise RuntimeError("INDICATORS bindings are not configured.")
        return self.INDICATORS

    def _should_compute_atr(self) -> bool:
        if self.ATR_ALWAYS_ON:
            return True
        use_tp = self.params.get("USE_TAKE_PROFIT", False)
        use_atr_sl = self.params.get("STOP_LOSS_TYPE") == "ATR"
        use_trailing = self.params.get("EXIT_TYPE") == "ATR" or self.params.get("TRAILING_ACTIVATION_ATR", 0) > 0
        return bool(use_tp or use_atr_sl or use_trailing)

    def _shift_if_needed(self, series: pd.Series) -> pd.Series:
        return series.shift(1) if self.ENTRY_SHIFT else series

    def _get_rsi_overbought(self) -> float:
        for key in self.RSI_OVERBOUGHT_KEYS:
            if key in self.params:
                return float(self.params[key])
        return float(self.params.get("RSI_OVERBOUGHT", 75))

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ind = self._ind()

        atr_period = int(self.params.get("ATR_PERIOD", 14))
        if self._should_compute_atr():
            df["atr"] = ind.calculate_atr(df, window=atr_period)
        else:
            df["atr"] = np.float32(0.0)

        strength_period = int(self.params.get("STRENGTH_FILTER_PERIOD", 14))
        df["rsi"] = ind.calculate_rsi(df["close"], window=strength_period)
        df["hurst"] = ind.calculate_hurst_exponent(df["close"], window=int(self.params.get("HURST_PERIOD", 200)))
        df["natr"] = ind.calculate_natr(df, window=strength_period)

        entry_type = self.params.get("ENTRY_TYPE", "DONCHIAN")
        entry_period = int(self.params.get("ENTRY_PERIOD", 20))
        df["entry_upper"] = np.nan
        df["entry_lower"] = np.nan

        if entry_type == "DONCHIAN":
            upper = df["high"].rolling(window=entry_period).max()
            lower = df["low"].rolling(window=entry_period).min()
            df["entry_upper"] = self._shift_if_needed(upper)
            df["entry_lower"] = self._shift_if_needed(lower)
        elif entry_type == "BOLLINGER":
            std_dev = float(self.params.get("BB_STD", 2.0))
            upper, lower, _ = ind.calculate_bollinger_bands(df, window=entry_period, std_dev=std_dev)
            df["entry_upper"] = self._shift_if_needed(upper)
            df["entry_lower"] = self._shift_if_needed(lower)
        elif entry_type == "KELTNER":
            atr_mult = float(self.params.get("KELTNER_ATR_MULT", 1.5))
            upper, lower = ind.calculate_keltner_channel(df, window=entry_period, atr_mult=atr_mult)
            df["entry_upper"] = self._shift_if_needed(upper)
            df["entry_lower"] = self._shift_if_needed(lower)
        elif entry_type == "CCI":
            cci = ind.calculate_cci(df, window=entry_period)
            cci_ref = cci.shift(1) if self.ENTRY_SHIFT else cci
            high_ref = df["high"].shift(1) if self.ENTRY_SHIFT else df["high"]
            low_ref = df["low"].shift(1) if self.ENTRY_SHIFT else df["low"]
            cci_thresh = float(self.params.get("CCI_THRESHOLD", 100))
            df["entry_upper"] = np.where(cci_ref > cci_thresh, high_ref, np.inf)
            df["entry_lower"] = np.where(cci_ref < -cci_thresh, low_ref, -np.inf)

        filter_type = self.params.get("TREND_FILTER_TYPE", "EMA")
        ma_period = int(self.params.get("MA_PERIOD", 50))
        df["trend_direction"] = 0

        if filter_type == "SMA":
            df["trend_line"] = ind.calculate_sma(df["close"], ma_period)
            df["trend_direction"] = np.where(df["close"] > df["trend_line"], 1, -1)
        elif filter_type == "EMA":
            df["trend_line"] = ind.calculate_ema(df["close"], ma_period)
            df["trend_direction"] = np.where(df["close"] > df["trend_line"], 1, -1)
        elif filter_type == "HMA":
            df["trend_line"] = ind.calculate_hma(df["close"], ma_period)
            df["trend_direction"] = np.where(df["close"] > df["trend_line"], 1, -1)
        elif filter_type == "DEMA":
            df["trend_line"] = ind.calculate_dema(df["close"], ma_period)
            df["trend_direction"] = np.where(df["close"] > df["trend_line"], 1, -1)
        elif filter_type == "TEMA":
            df["trend_line"] = ind.calculate_tema(df["close"], ma_period)
            df["trend_direction"] = np.where(df["close"] > df["trend_line"], 1, -1)
        elif filter_type == "SUPERTREND":
            df["trend_direction"] = ind.calculate_supertrend(
                df,
                period=int(self.params.get("SUPERTREND_PERIOD", 10)),
                multiplier=float(self.params.get("SUPERTREND_MULT", 3.0)),
            )
        elif filter_type == "MACD":
            macd_line, signal_line, _ = ind.calculate_macd(
                df,
                fast=int(self.params.get("MACD_FAST", 12)),
                slow=int(self.params.get("MACD_SLOW", 26)),
                signal=int(self.params.get("MACD_SIGNAL", 9)),
            )
            df["trend_direction"] = np.where(macd_line > signal_line, 1, -1)
        elif filter_type == "ICHIMOKU":
            _, _, senkou_a, senkou_b = ind.calculate_ichimoku(
                df,
                tenkan_window=int(self.params.get("ICHIMOKU_TENKAN", 9)),
                kijun_window=int(self.params.get("ICHIMOKU_KIJUN", 26)),
                senkou_span_b_window=int(self.params.get("ICHIMOKU_SENKOU_B", 52)),
            )
            cloud_top = np.maximum(senkou_a.to_numpy(), senkou_b.to_numpy())
            cloud_bottom = np.minimum(senkou_a.to_numpy(), senkou_b.to_numpy())
            close_np = df["close"].to_numpy()
            trend = np.zeros(len(df), dtype=np.int8)
            trend[close_np > cloud_top] = 1
            trend[close_np < cloud_bottom] = -1
            df["trend_direction"] = trend
        elif filter_type == "VWAP":
            vwap, vwap_upper, vwap_lower = ind.calculate_vwap(
                df, window=ma_period, std_mult=float(self.params.get("VWAP_STD_MULT", 1.5))
            )
            df["vwap"] = vwap
            df["vwap_upper"] = vwap_upper
            df["vwap_lower"] = vwap_lower
            df["trend_direction"] = np.where(df["close"] > df["vwap"], 1, -1)
        elif filter_type == "DMI":
            dmi_period = int(self.params.get("DMI_PERIOD", 14))
            plus_di, minus_di = ind.calculate_dmi(df, window=dmi_period)
            df["trend_direction"] = np.where(plus_di > minus_di, 1, -1)
        elif filter_type == "AROON":
            aroon_period = int(self.params.get("AROON_PERIOD", 14))
            aroon_up, aroon_down = ind.calculate_aroon(df, window=aroon_period)
            df["trend_direction"] = np.where(aroon_up > aroon_down, 1, -1)

        df["strength_filter"] = 1
        strength_type = self.params.get("STRENGTH_FILTER_TYPE", "NONE")

        if strength_type == "ADX":
            df["adx"] = ind.calculate_adx(df, window=strength_period)
            df.loc[df["adx"] < float(self.params.get("ADX_THRESHOLD", 20)), "strength_filter"] = 0
        elif strength_type == "VHF":
            df["vhf"] = ind.calculate_vhf(df["close"], window=strength_period)
            df.loc[df["vhf"] < float(self.params.get("VHF_THRESHOLD", 0.4)), "strength_filter"] = 0
        elif strength_type == "MFI":
            df["mfi"] = ind.calculate_mfi(df, window=strength_period)
            df.loc[df["mfi"] < float(self.params.get("MFI_THRESHOLD", 25)), "strength_filter"] = 0
        elif strength_type == "RSI":
            rsi_overbought = self._get_rsi_overbought()
            rsi_oversold = float(self.params.get("RSI_OVERSOLD", 25))
            df.loc[(df["rsi"] > rsi_overbought) | (df["rsi"] < rsi_oversold), "strength_filter"] = 0
        elif strength_type == "STOCHASTIC":
            stoch_k, _ = ind.calculate_stochastic(df, window=strength_period)
            df["stoch_k"] = stoch_k
            df.loc[
                (df["stoch_k"] > float(self.params.get("STOCH_OVERBOUGHT", 85)))
                | (df["stoch_k"] < float(self.params.get("STOCH_OVERSOLD", 15))),
                "strength_filter",
            ] = 0
        elif strength_type == "STOCH_RSI":
            stoch_rsi_k, _ = ind.calculate_stoch_rsi(df["close"], window=strength_period)
            df["stoch_rsi_k"] = stoch_rsi_k
            df.loc[
                (df["stoch_rsi_k"] > float(self.params.get("STOCH_RSI_OVERBOUGHT", 80)))
                | (df["stoch_rsi_k"] < float(self.params.get("STOCH_RSI_OVERSOLD", 20))),
                "strength_filter",
            ] = 0
        elif strength_type == "CMF":
            cmf_period = int(self.params.get("CMF_PERIOD", 20))
            df["cmf"] = ind.calculate_cmf(df, window=cmf_period)
            df.loc[df["cmf"] < float(self.params.get("CMF_THRESHOLD", 0.05)), "strength_filter"] = 0
        elif strength_type == "HURST":
            random_threshold = float(self.params.get("HURST_RANDOM_THRESHOLD", 0.50))
            df.loc[df["hurst"] < random_threshold, "strength_filter"] = 0
            trend_threshold = self.params.get("HURST_TREND_THRESHOLD")
            if trend_threshold is not None:
                df.loc[df["hurst"] < float(trend_threshold), "strength_filter"] = 0
        elif strength_type == "ER":
            er_period = int(self.params.get("STRENGTH_FILTER_PERIOD", 10))
            df["er"] = ind.calculate_efficiency_ratio(df["close"], window=er_period)
            df.loc[df["er"] < float(self.params.get("ER_THRESHOLD", 0.6)), "strength_filter"] = 0
        elif strength_type == "NATR":
            df.loc[df["natr"] < float(self.params.get("NATR_THRESHOLD", 1.0)), "strength_filter"] = 0
        elif strength_type == "GARMAN_KLASS":
            gk_period = int(self.params.get("GK_PERIOD", 30))
            df["gk_vol"] = ind.calculate_garman_klass_vol(df, window=gk_period)
            df.loc[
                df["gk_vol"] < float(self.params.get("GK_THRESHOLD", 0.0001)),
                "strength_filter",
            ] = 0
        elif strength_type == "FORCE_INDEX":
            fi_period = int(self.params.get("FORCE_INDEX_PERIOD", 2))
            df["force_index"] = ind.calculate_force_index(df, smooth_period=fi_period)
            fi_thresh = float(self.params.get("FORCE_INDEX_THRESHOLD", 0.0))
            df.loc[
                (df["trend_direction"] > 0) & (df["force_index"] < fi_thresh),
                "strength_filter",
            ] = 0
            df.loc[
                (df["trend_direction"] < 0) & (df["force_index"] > -fi_thresh),
                "strength_filter",
            ] = 0
        elif strength_type == "WILLIAMS_R":
            df["willr"] = ind.calculate_williams_r(df, window=strength_period)
            willr_ob = float(self.params.get("WILLR_OVERBOUGHT", -20.0))
            willr_os = float(self.params.get("WILLR_OVERSOLD", -80.0))
            df.loc[
                (df["willr"] > willr_ob) | (df["willr"] < willr_os),
                "strength_filter",
            ] = 0
        elif strength_type == "OBV":
            df["obv"] = ind.calculate_obv(df)
            obv_ma_period = int(self.params.get("OBV_MA_PERIOD", 20))
            obv_ma = df["obv"].rolling(window=obv_ma_period).mean()
            df.loc[
                (df["trend_direction"] > 0) & (df["obv"] < obv_ma) & obv_ma.notna(),
                "strength_filter",
            ] = 0
            df.loc[
                (df["trend_direction"] < 0) & (df["obv"] > obv_ma) & obv_ma.notna(),
                "strength_filter",
            ] = 0

        rsi_entry_max = self.params.get("RSI_ENTRY_MAX")
        if rsi_entry_max is not None:
            rsi_entry_max = float(rsi_entry_max)
            df.loc[(df["trend_direction"] > 0) & (df["rsi"] > rsi_entry_max), "strength_filter"] = 0
            df.loc[(df["trend_direction"] < 0) & (df["rsi"] < (100.0 - rsi_entry_max)), "strength_filter"] = 0

        if self.params.get("EXIT_TYPE") == "PARABOLIC_SAR":
            sar_line, _ = ind.calculate_parabolic_sar(df, step=float(self.params.get("SAR_STEP", 0.02)))
            df["parabolic_sar"] = sar_line
        else:
            df["parabolic_sar"] = 0.0

        if self.params.get("USE_VOLUME_FILTER", False):
            vol_ma_period = int(self.params.get("VOLUME_MA_PERIOD", 20))
            log_vol = np.log1p(df["volume"])
            log_vol_mean = log_vol.rolling(window=vol_ma_period).mean()
            log_vol_std = log_vol.rolling(window=vol_ma_period).std()
            z_score = (log_vol - log_vol_mean) / log_vol_std.replace(0, 1).fillna(1)
            df["volume_ratio"] = z_score.fillna(-10.0)
        else:
            df["volume_ratio"] = 100.0

        df["atr"] = df["atr"].ffill().fillna(df["close"] * 0.01)
        df["natr"] = df["natr"].ffill().fillna(1.0)
        df["rsi"] = df["rsi"].ffill().fillna(50.0)
        df["hurst"] = df["hurst"].ffill().fillna(0.5)
        if "gk_vol" in df.columns:
            df["gk_vol"] = df["gk_vol"].ffill().fillna(1e-6)
        if "force_index" in df.columns:
            df["force_index"] = df["force_index"].ffill().fillna(0.0)
        if "willr" in df.columns:
            df["willr"] = df["willr"].ffill().fillna(-50.0)
        if "obv" in df.columns:
            df["obv"] = df["obv"].ffill().fillna(0.0)
        df["strength_filter"] = df["strength_filter"].fillna(0).astype(int)
        df["trend_direction"] = df["trend_direction"].fillna(0).astype(int)
        df["entry_upper"] = df["entry_upper"].astype(float)
        df["entry_lower"] = df["entry_lower"].astype(float)
        return df
