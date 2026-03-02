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

        # --- 0. Baseline Indicators (ATR, RSI, Hurst) ---
        atr_period = int(self.params.get("ATR_PERIOD", 14))
        
        tf = self.params.get("TIMEFRAME", "1h")
        bars_per_day = {"1h": 24, "4h": 6}.get(tf, 24)

        if self._should_compute_atr():
            df["atr"] = ind.calculate_atr(df, window=atr_period)
        else:
            df["atr"] = np.float32(0.0)

        # Baseline Momentum & Volatility for Regimes
        strength_period = int(self.params.get("STRENGTH_FILTER_PERIOD", 14))
        df["rsi"] = ind.calculate_rsi(df["close"], window=strength_period)
        raw_hurst = ind.calculate_hurst_exponent(df["close"], window=int(self.params.get("HURST_PERIOD", 200)))
        log_ret = np.log(df["close"] / df["close"].shift(1))

        # --- [NEW] Kaufman Efficiency Ratio (ER) - The Institutional Chop Filter ---
        # ER = (Net Change over N days) / (Sum of Absolute Daily Changes over N days)
        # Using a fixed 14-day window (Kaufman's standard) to evaluate crypto market noise.
        er_period = int(14 * bars_per_day)
        net_change = (df["close"] - df["close"].shift(er_period)).abs()
        abs_change = (df["close"] - df["close"].shift(1)).abs()
        sum_abs_change = abs_change.rolling(window=er_period).sum().replace(0, 1e-9).fillna(1e-9)
        df["er"] = (net_change / sum_abs_change).fillna(0.0)
        
        # [REMOVED] ER-Based Risk Scalar (구속복 해체)
        # 횡보장 끝자락(가장 ER이 낮은 시점)에서 터지는 극초기 대추세 돌파 때, 
        # 비중이 10%로 줄어드는 치명적 모순을 방지하기 위해 1.0으로 고정.
        # 비중 조절은 오직 엔진의 ATR 기반 Vol-Targeting에 100% 위임함.
        df["risk_scalar"] = 1.0
        
        # [REMOVED] Dynamic Exit Scalar: 횡보장에서 방패를 이중으로 조이는 치명적 자해(Double-Tightening) 현상 방지를 위해 1.0 고정
        df["exit_scalar"] = 1.0

        # --- 1. Factor Families (Independent Scales) ---
        n_bars = 7 * bars_per_day
        atr_ma = df["atr"].rolling(window=n_bars).mean().replace(0, 1e-9).fillna(1e-9)

        # 1-1. Breakout Family (Donchian, BB, KC, SuperTrend)
        entry_period = int(self.params.get("ENTRY_PERIOD", 20))
        scale_breakout = (atr_ma / np.sqrt(bars_per_day)) * np.sqrt(entry_period)
        
        donchian_upper = df["high"].rolling(window=entry_period).max()
        donchian_lower = df["low"].rolling(window=entry_period).min()
        donchian_mid = (donchian_upper + donchian_lower) / 2
        donchian_z = (df["close"] - donchian_mid) / scale_breakout
        donchian_score = np.tanh(donchian_z)

        bb_std = float(self.params.get("BB_STD", 2.0))
        bb_upper, bb_lower, _ = ind.calculate_bollinger_bands(df, window=entry_period, std_dev=bb_std)
        bb_mid = (bb_upper + bb_lower) / 2.0
        bb_z = (df["close"] - bb_mid) / scale_breakout
        bb_score = np.tanh(bb_z)

        kc_mult = float(self.params.get("KC_MULT", 1.5))
        _, _ = ind.calculate_keltner_channel(df, window=entry_period, atr_mult=kc_mult)
        kc_mid = ind.calculate_sma(df["close"], entry_period)
        kc_z = (df["close"] - kc_mid) / scale_breakout
        kc_score = np.tanh(kc_z)

        st_mult = float(self.params.get("SUPERTREND_MULTIPLIER", 3.0))
        st_dir = ind.calculate_supertrend(df, period=10, multiplier=st_mult)
        st_score = st_dir.fillna(0)

        # 1-2. Trend Family (EMA, HMA, VWMA, MACD, ADX)
        trend_p = int(self.params.get("TREND_PERIOD", 50))
        ema_line = ind.calculate_ema(df["close"], trend_p)
        ema_z = np.log(df["close"] / ema_line) / (log_ret.rolling(window=trend_p).std() * np.sqrt(trend_p))
        ema_score = np.tanh(ema_z.fillna(0))

        hma_line = ind.calculate_hma(df["close"], window=trend_p)
        hma_z = np.log(df["close"] / hma_line) / (log_ret.rolling(window=trend_p).std() * np.sqrt(trend_p))
        hma_score = np.tanh(hma_z.fillna(0))

        vwma_line = ind.calculate_vwma(df, window=trend_p)
        vwma_z = np.log(df["close"] / vwma_line) / (log_ret.rolling(window=trend_p).std() * np.sqrt(trend_p))
        vwma_score = np.tanh(vwma_z.fillna(0))

        adx_thresh = float(self.params.get("ADX_THRESHOLD", 25))
        df["adx"] = ind.calculate_adx(df, window=14)
        df["plus_di"], df["minus_di"] = ind.calculate_dmi(df, window=14)
        adx_filter = np.where(df["adx"] > adx_thresh, 1.0, 0.0)
        adx_dir = np.where(df["plus_di"] > df["minus_di"], 1.0, -1.0)

        macd_f = int(self.params.get("MACD_FAST", 12))
        macd_s = int(self.params.get("MACD_SLOW", 26))
        _, _, macd_hist = ind.calculate_macd(df, fast=macd_f, slow=macd_s)
        macd_score = np.tanh(macd_hist / (df["close"] * 0.001))

        # 1-3. Volume Family (Vol-Z, CMF)
        vol_p = int(self.params.get("VOL_WINDOW", 20))
        log_vol = np.log1p(df["volume"])
        vol_z = (log_vol - log_vol.rolling(window=vol_p).mean()) / log_vol.rolling(window=vol_p).std().replace(0, 1e-9)
        
        cmf_period = int(self.params.get("CMF_PERIOD", 20))
        df["cmf"] = ind.calculate_cmf(df, window=cmf_period)
        cmf_score = np.tanh(df["cmf"] / 0.15)
        
        vol_combined = (vol_z + cmf_score) / 2.0

        # 1-4. Mean Reversion Family (VWAP, StochRSI, RSI, Inv-BB)
        # [IMPROVEMENT] Rolling VWAP (24h) to fix lifetime cumulative divergence bug.
        # 24h represents one complete institutional liquidity cycle in 24/7 crypto markets.
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        rolling_vol_price = (typical_price * df["volume"]).rolling(window=24).sum()
        rolling_vol = df["volume"].rolling(window=24).sum().replace(0, 1e-9)
        df["vwap"] = rolling_vol_price / rolling_vol
        
        # [IMPROVEMENT] Use ATR for volatility normalization instead of raw standard deviation
        # This prevents the Z-score from exploding to infinity during tight consolidation (low vol)
        vwap_atr_ma = df["atr"].rolling(window=24).mean().replace(0, 1e-9).fillna(1e-9)
        vwap_z = (df["close"] - df["vwap"]) / vwap_atr_ma
        vwap_mult = float(self.params.get("VWAP_STD_MULT", 2.5))
        vwap_mr_score = np.where(vwap_z > vwap_mult, -1.0, np.where(vwap_z < -vwap_mult, 1.0, 0.0))

        stoch_rsi_p = int(self.params.get("STOCH_RSI_PERIOD", 14))
        stoch_rsi_ext = float(self.params.get("STOCH_RSI_EXTREME", 20)) / 100.0
        stoch_k, _ = ind.calculate_stoch_rsi(df["close"], window=stoch_rsi_p)
        df["stoch_rsi"] = stoch_k
        stoch_rsi_score = np.where(df["stoch_rsi"] < stoch_rsi_ext, 1.0, 
                                   np.where(df["stoch_rsi"] > (1.0 - stoch_rsi_ext), -1.0, 0.0))
        
        # [IMPROVEMENT] Digital threshold for RSI to prevent weak continuous alpha accumulation
        rsi_mr_score = np.where(df["rsi"] < 30.0, 1.0, np.where(df["rsi"] > 70.0, -1.0, 0.0))
        
        # [IMPROVEMENT] Digital threshold for Bollinger Bands (Price completely outside band)
        bb_mr_score = np.where(df["close"] < bb_lower, 1.0, np.where(df["close"] > bb_upper, -1.0, 0.0))

        # --- 2. Ensemble Alpha Aggregation (Pure Trend Following) ---
        eps = 1e-9
        
        # Style Factor 1: Breakout Team
        # [NEW] Priority 5: Factor Combination (Single Indicator Dependency 타파)
        # 켈트너(변동성)와 돈치안(가격) 돌파를 5:5로 섞어 알파 점수의 신뢰도를 높임.
        breakout_alpha = (kc_score + donchian_score) / 2.0

        # Style Factor 2: Trend Team (Multi-Timeframe TSMOM)
        def calc_tsmom_z(p: int) -> pd.Series:
            ret = np.log(df["close"] / df["close"].shift(p).replace(0, np.nan))
            vol = ret.rolling(window=p).std().replace(0, 1e-9).fillna(1e-9) * np.sqrt(p)
            return ret / vol

        tsmom_z_combined = (calc_tsmom_z(20) + calc_tsmom_z(60) + calc_tsmom_z(120)) / 3.0
        trend_alpha = np.tanh(tsmom_z_combined.fillna(0))

        # [REMOVED] Structural Macro Gatekeeper (Hard Boolean Gate)
        # 낡고 후행성이 심한 200일선 차단기를 완전히 철거하여, 
        # 다중 시계열 모멘텀(tsmom_z_combined)과 돈치안 돌파 로직이 가진 본연의 엣지를 100% 해방함.

        # Style Factor 3: Mean Reversion Team
        mean_reversion_alpha = (rsi_mr_score + bb_mr_score + vwap_mr_score + stoch_rsi_score) / 4.0

        # Layer 3: Strategic Macro Weights
        w_breakout = float(self.params.get("W_BREAKOUT", 1.0))
        w_trend = float(self.params.get("W_TREND", 1.0))
        w_volume = float(self.params.get("W_VOLUME", 1.0))
        w_mean_rev = float(self.params.get("W_MEAN_REVERSION", 0.0))

        # Style Factor 4: Volume Team
        price_consensus = (w_breakout * breakout_alpha) + (w_trend * trend_alpha) + (w_mean_rev * mean_reversion_alpha)
        volume_alpha = vol_combined * np.sign(price_consensus)
        
        tot_w = w_breakout + w_trend + w_volume + w_mean_rev + eps
        total_alpha = (w_breakout*breakout_alpha + w_trend*trend_alpha + w_volume*volume_alpha + w_mean_rev*mean_reversion_alpha) / tot_w
        total_alpha = np.nan_to_num(total_alpha, nan=0.0)

        # --- [NEW] Priority 1: Adaptive Threshold (Rolling Percentile) ---
        # Instead of a static threshold (e.g., > 0.5), we normalize the signal by recent market history.
        # We only enter if the current alpha is in the top/bottom X percentile of the lookback window.
        lookback_bars = int(self.params.get("THRESHOLD_LOOKBACK", 180)) # Default 30 days on 4H
        q_val = float(self.params.get("THRESHOLD_QUANTILE", 0.85)) # e.g., Top 15%
        
        alpha_series = pd.Series(total_alpha, index=df.index)
        
        # Calculate rolling quantiles. Fillna with 1.0/-1.0 to prevent early accidental triggers during warmup.
        roll_q_long = alpha_series.rolling(window=lookback_bars).quantile(q_val).fillna(1.0)
        roll_q_short = alpha_series.rolling(window=lookback_bars).quantile(1.0 - q_val).fillna(-1.0)

        ensemble_dir = np.zeros(len(df))
        # Trigger only if alpha exceeds the dynamic percentile AND is directionally correct (>0 or <0)
        ensemble_dir[(total_alpha > roll_q_long) & (total_alpha > 0)] = 1
        ensemble_dir[(total_alpha < roll_q_short) & (total_alpha < 0)] = -1

        df["trend_direction"] = self._shift_if_needed(pd.Series(ensemble_dir, index=df.index))
        
        # --- [NEW] Priority 2: Price Confirmation Gate (Liquidity Sweep) ---
        # 4H(추세추종)는 실제 가격이 전고/전저점을 물리적으로 뚫을 때만 진입하도록 방어막을 침.
        # 1H(역추세)는 과매도/과매수 신호 발생 즉시(Buy the Dip) 시장가로 진입해야 하므로 방어막을 현재가로 해제함.
        if tf == "4h":
            df["entry_upper"] = donchian_upper
            df["entry_lower"] = donchian_lower
        else:
            df["entry_upper"] = df["close"]
            df["entry_lower"] = df["close"]
        
        df["strength_filter"] = np.where(df["trend_direction"] != 0, 1, 0)
        
        # --- 4. Special Logic ---
        if self.params.get("EXIT_TYPE") == "PARABOLIC_SAR":
            sar_step = float(self.params.get("PSAR_STEP", 0.02))
            sar_max = float(self.params.get("PSAR_MAX", 0.2))
            df["parabolic_sar"], _ = ind.calculate_parabolic_sar(df, step=sar_step, max_step=sar_max)
        else:
            df["parabolic_sar"] = 0.0

        df["volume_ratio"] = 100.0 # Standard override

        # --- 5. Cleanup ---
        df["atr"] = df["atr"].ffill().fillna(df["close"] * 0.01)
        df["rsi"] = df["rsi"].ffill().fillna(50.0)
        df["strength_filter"] = df["strength_filter"].fillna(0).astype(int)
        df["trend_direction"] = df["trend_direction"].fillna(0).astype(int)
        
        return df
