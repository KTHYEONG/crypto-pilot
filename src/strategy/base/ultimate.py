from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from .core import StrategyBase


class UltimateStrategyBase(StrategyBase):
    """
    Shared signal-generation pipeline for Spot/Futures.
    Pure TSMOM (Time-Series Momentum) Z-Score Strategy.
    """

    INDICATORS: SimpleNamespace | None = None
    ENTRY_SHIFT: bool = False  # Engine already uses prev_i (i-1), so shift here causes 2-bar delay

    def _ind(self) -> SimpleNamespace:
        if self.INDICATORS is None:
            raise RuntimeError("INDICATORS bindings are not configured.")
        return self.INDICATORS

    def _shift_if_needed(self, series: pd.Series) -> pd.Series:
        return series.shift(1) if self.ENTRY_SHIFT else series

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ind = self._ind()
        
        # Ensure parameters
        atr_window = int(self.params.get("ATR_WINDOW", 14))
        weight_decay = float(self.params.get("TSMOM_WEIGHT_DECAY", 0.0))
        entry_threshold = float(self.params.get("TSMOM_ENTRY_THRESHOLD", 1.5))
        
        # Epsilon guard for zero-division avoidance
        EPS = 1e-8

        # --- 1. Baseline Indicators ---
        df["atr"] = ind.calculate_atr(df, window=atr_window)
        # Engine expects 'atr' for its trailing exit logic.
        
        # --- 1.2 Volatility Regime Filter ---
        # [INSTITUTIONAL] Prevent entries during low-volatility regimes where signals are noisy.
        # Use ATR percentile to define regime: ignore signals if ATR < 20th percentile.
        prc_window = int(self.params.get("ATR_PRC_WINDOW", 250))
        # Percentile calculation
        def calc_rank(x):
            return (x[-1] >= x).mean()
        atr_rank = df["atr"].rolling(prc_window).apply(calc_rank, raw=True)
        vol_regime_mask = atr_rank > 0.20 # Ignore if in bottom 20% vol
        
        # Stationary 1-bar log return
        r1 = np.log(df["close"] / df["close"].shift(1))
        
        # Multi-Timeframe TSMOM Calc (1H Crypto Native Cycles)
        lookbacks = [12, 24, 72, 168]
        weights = [1.0 / (lb ** weight_decay) for lb in lookbacks]
        tot_w = sum(weights)
        w_norm = [w / tot_w for w in weights]
        
        total_tsmom_z = pd.Series(0.0, index=df.index)
        
        for w, n in zip(w_norm, lookbacks):
            rn = np.log(df["close"] / df["close"].shift(n))
            sigma_n = r1.rolling(n).std() * np.sqrt(n)
            tsmom_n = rn / (sigma_n + EPS)
            total_tsmom_z += w * tsmom_n
            
        # Total alpha (Re-standardize to restore N(0,1) variance lost by weighted sum)
        total_tsmom_z = total_tsmom_z.fillna(0.0)
        roll_std = total_tsmom_z.rolling(window=lookbacks[-1], min_periods=max(100, lookbacks[0])).std()
        total_tsmom_z = total_tsmom_z / (roll_std + EPS)
        total_tsmom_z = total_tsmom_z.fillna(0.0)

        # --- 2. Signal Generation ---
        # 1H Macro Trend Filter (1W vs 3D EMA)
        ema_fast = df["close"].ewm(span=72, min_periods=72).mean()
        ema_slow = df["close"].ewm(span=168, min_periods=168).mean()
        macro_bull = ema_fast > ema_slow
        macro_bear = ema_fast < ema_slow

        # Crossing Filter
        crossed_up = (total_tsmom_z > entry_threshold) & (total_tsmom_z.shift(1) <= entry_threshold)
        crossed_down = (total_tsmom_z < -entry_threshold) & (total_tsmom_z.shift(1) >= -entry_threshold)
        
        # Velocity Filter
        velocity_k = int(self.params.get("VELOCITY_K", 12))
        # [REFINED] Use short-term velocity for entry gating to ensure momentum persistence at crossing.
        dz_short = total_tsmom_z - total_tsmom_z.shift(max(velocity_k // 3, 2))
        
        # [CTA-Style] Pulse Entry: Allow re-entry during regime if momentum accelerates (dz cross-up/down)
        in_long_regime = total_tsmom_z > entry_threshold
        in_short_regime = total_tsmom_z < -entry_threshold
        
        dz_crossed_up = (dz_short > 0) & (dz_short.shift(1) <= 0)
        dz_crossed_down = (dz_short < 0) & (dz_short.shift(1) >= 0)
        
        # Lower velocity importance on initial breakout to ensure we catch moves early
        long_cond = ((in_long_regime & dz_crossed_up) | crossed_up) & vol_regime_mask & macro_bull
        short_cond = ((in_short_regime & dz_crossed_down) | crossed_down) & vol_regime_mask & macro_bear

        df_len = len(df)
        trend_state = np.zeros(df_len, dtype=int)
        tsmom_z_vals = total_tsmom_z.values
        long_cond_vals = long_cond.values
        short_cond_vals = short_cond.values
        
        current_state = 0
        for i in range(df_len):
            if current_state == 0:
                if long_cond_vals[i]:
                    current_state = 1
                elif short_cond_vals[i]:
                    current_state = -1
            elif current_state == 1:
                if tsmom_z_vals[i] < 0:
                    current_state = 0
                if short_cond_vals[i]:
                    current_state = -1
            elif current_state == -1:
                if tsmom_z_vals[i] > 0:
                    current_state = 0
                if long_cond_vals[i]:
                    current_state = 1
            trend_state[i] = current_state

        df["trend_direction"] = self._shift_if_needed(pd.Series(trend_state, index=df.index))
        
        # Market entry: set upper/lower to 0/inf so engine always triggers at open price.
        df["entry_upper"] = 0.0          # LONG  fills at c_open if trend_dir == 1
        df["entry_lower"] = np.finfo(np.float64).max  # SHORT fills at c_open if trend_dir == -1
        
        # Strength filter acts as master toggle
        df["strength_filter"] = np.where(df["trend_direction"] != 0, 1, 0)
        
        # Cleanup
        df["atr"] = df["atr"].ffill().fillna(df["close"] * 0.01)
        df["strength_filter"] = df["strength_filter"].fillna(0).astype(int)
        df["trend_direction"] = df["trend_direction"].fillna(0).astype(int)

        return df

