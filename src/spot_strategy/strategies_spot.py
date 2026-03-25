from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.spot_strategy.ec3_filters import apply_ec3_persistence
from src.spot_strategy.garch_sizer import compute_garch_kelly_series
from src.spot_strategy.hmm_regime import compute_walk_forward_hmm
from src.strategy.base.ultimate import UltimateStrategyBase
from src.futures_strategy.strategies_futures import _FUTURES_INDICATORS as _SPOT_INDICATORS


class UltimateSpotStrategy(UltimateStrategyBase):
    """
    Spot: SuperTrend + TSMOM + HMM regime + t-GARCH Kelly scaling + EC-3 persistence + L0 kill-switch.
    """

    INDICATORS = _SPOT_INDICATORS
    ENTRY_SHIFT = False

    def _compute_warmup_bars(self) -> int:
        p = self.params
        hmm_w = int(p.get("HMM_TRAIN_WINDOW", 360))
        g_w = int(p.get("GARCH_WINDOW", 360))
        max_p = max(
            int(p.get("SUPERTREND_PERIOD", 10)),
            int(p.get("ATR_RATIO_PERIOD", 14)),
            int(p.get("ATR_RATIO_LONG_PERIOD", 40)),
            int(p.get("EMA_TREND_PERIOD", 100)),
            int(p.get("MOMENTUM_ROC_PERIOD", 14)),
            int(p.get("RSI_PERIOD", 14)),
            hmm_w,
            g_w,
            50,
        )
        return max(300, int(max_p * 3))

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = df[col].astype(np.float64)

        if "btc_close" not in df.columns:
            df["btc_close"] = df["close"].astype(np.float64)

        ind = self._ind()

        st_period = int(self.params.get("SUPERTREND_PERIOD", 10))
        st_mult = float(self.params.get("SUPERTREND_MULT", 3.0))
        atr_s_period = int(self.params.get("ATR_RATIO_PERIOD", 14))
        atr_l_period = int(self.params.get("ATR_RATIO_LONG_PERIOD", 42))
        atr_exp_thr = float(self.params.get("ATR_EXPANSION_THRESHOLD", 1.2))
        ema_trend_period = int(self.params.get("EMA_TREND_PERIOD", 100))
        roc_period = int(self.params.get("MOMENTUM_ROC_PERIOD", 14))
        rsi_period = int(self.params.get("RSI_PERIOD", 14))

        hmm_train = int(self.params.get("HMM_TRAIN_WINDOW", 360))
        hmm_rf = int(self.params.get("HMM_RETRAIN_FREQ", 24))
        garch_w = int(self.params.get("GARCH_WINDOW", 360))
        garch_rf = int(self.params.get("GARCH_RETRAIN_FREQ", 24))
        kill_k = float(self.params.get("KILL_ATR_K", 4.0))

        if atr_l_period <= atr_s_period:
            atr_l_period = atr_s_period * 3

        df["atr"] = ind.calculate_atr(df, window=atr_s_period)
        atr_short = df["atr"]
        atr_long = ind.calculate_atr(df, window=atr_l_period)
        atr_ratio = (atr_short / (atr_long + 1e-12)).replace([np.inf, -np.inf], np.nan)

        st_trend = ind.calculate_supertrend(df, period=st_period, multiplier=st_mult)
        st_bull = st_trend == 1

        ema_trend = ind.calculate_ema(df["close"], window=ema_trend_period)
        rsi = ind.calculate_rsi(df["close"], window=rsi_period)
        roc_pct = ind.calculate_roc(df["close"], window=roc_period)

        hurst_w = int(self.params.get("HURST_WINDOW", 40))
        df["hurst_40"] = ind.calculate_hurst_exponent(df["close"], window=hurst_w).astype(np.float64)
        df["hurst_40"] = df["hurst_40"].fillna(0.5)

        log_ret_1 = np.log(df["close"] / df["close"].shift(1).replace(0.0, np.nan)).fillna(0.0)
        log_ret_6 = log_ret_1.rolling(6, min_periods=6).sum()
        log_ret_18 = log_ret_1.rolling(18, min_periods=18).sum()
        tsmom_ok = (log_ret_1 > 0.0) & (log_ret_6 > 0.0) & (log_ret_18 > 0.0)

        momentum_soft = (rsi > 50.0) | (roc_pct > 0.0)
        atr_expansion = atr_ratio > atr_exp_thr
        core_long = st_bull & tsmom_ok
        is_bear = df["close"] < ema_trend

        bull_raw = (core_long & momentum_soft & (~is_bear)).astype(np.float64).values
        ec3_out = apply_ec3_persistence(bull_raw, m_entry=2)

        viterbi, p_bull_hmm, p_side_hmm = compute_walk_forward_hmm(
            df,
            train_window=hmm_train,
            retrain_freq=hmm_rf,
        )

        df["hmm_viterbi"] = viterbi.astype(np.int32)
        df["p_bull"] = p_bull_hmm.astype(np.float64)
        df["p_side"] = p_side_hmm.astype(np.float64)

        bear_hmm = viterbi == 0
        bull_hmm = viterbi == 2
        regime_risk_hmm = np.where(
            bear_hmm,
            0.0,
            np.where(bull_hmm, p_bull_hmm, 0.3),
        ).astype(np.float64)

        bear_reg = is_bear
        trending_reg = (~is_bear) & atr_expansion & st_bull
        regime_risk_atr = np.where(
            bear_reg,
            0.0,
            np.where(trending_reg, 1.0, 0.5),
        ).astype(np.float64)

        use_hmm = bool(self.params.get("USE_HMM_REGIME", True))
        regime_risk_mult = np.where(use_hmm, regime_risk_hmm, regime_risk_atr).astype(np.float64)
        df["regime_risk_mult"] = regime_risk_mult

        close_series = cast(pd.Series, df["close"])
        garch_series = compute_garch_kelly_series(
            close_series,
            window=garch_w,
            retrain_freq=garch_rf,
        )
        df["garch_kelly_f"] = garch_series.astype(np.float64).values

        prev_close = df["close"].shift(1)
        price_drop = df["close"] - prev_close
        df["kill_signal"] = (price_drop < (-kill_k * df["atr"])).astype(np.float64)
        df.loc[df["atr"].isna() | (df["atr"] <= 0), "kill_signal"] = 0.0

        bull_breakout = ec3_out > 0.5

        df["long_entry_signal"] = np.where(bull_breakout, 1.0, 0.0)
        df["sig_long_entry_signal"] = df["long_entry_signal"]

        df["entry_upper"] = np.where(bull_breakout, df["close"], 999999.0)
        df["entry_lower"] = np.full(len(df), 999999.0, dtype=np.float64)

        df["strength_filter"] = np.where(bull_breakout, 1, 0)
        df["trend_direction"] = np.where(bull_breakout, 1, 0)

        roc_score = (roc_pct / 100.0).clip(-3.0, 3.0).fillna(0.0)

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

        df["slot_rank_score"] = (0.5 * rs_score + 0.5 * roc_score).astype(np.float64)

        df["entry_upper"] = self._shift_if_needed(cast(pd.Series, df["entry_upper"]))
        df["entry_lower"] = self._shift_if_needed(cast(pd.Series, df["entry_lower"]))

        df["atr"] = df["atr"].ffill().fillna(df["close"] * 0.01)
        df["garch_kelly_f"] = df["garch_kelly_f"].ffill().bfill().clip(0.15, 1.0)
        df["regime_risk_mult"] = df["regime_risk_mult"].ffill().fillna(0.5)
        df["kill_signal"] = df["kill_signal"].fillna(0.0)
        return df
