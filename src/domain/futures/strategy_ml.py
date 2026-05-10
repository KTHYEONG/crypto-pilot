from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.core.indicators.indicators import get_indicator_engine
from src.strategy_base import PipelineStrategyBase

_logger = logging.getLogger(__name__)
_FUTURES_INDICATORS = get_indicator_engine(domain="futures")


class FuturesMLStrategy(PipelineStrategyBase):
    """Pure ML-focused Futures Strategy.
    
    Uses ML alpha signals (ml_alpha_*) and HMM probabilities/modulators 
    for signal generation, regime filtering, and ranking.
    """

    INDICATORS = _FUTURES_INDICATORS
    ENTRY_SHIFT = False

    def _compute_warmup_bars(self) -> int:
        """ML strategies typically need at least 300 bars for HMM/Regime stabilization."""
        from src.strategy_base.core import calculate_required_warmup_bars
        return calculate_required_warmup_bars(self.params, min_bars=300)

    def generate_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tier 1: Fixed indicators independent of optimization params."""
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)
        
        atr_period = int(self.params.get("ATR_PERIOD", 20))
        macro_period = int(self.params.get("MACRO_EMA_PERIOD", 200))
        ind = self._ind()
        
        if "atr" not in df.columns or df["atr"].isna().all():
            df["atr"] = ind.calculate_atr(df, window=atr_period)
        df["atr"] = df["atr"].ffill().fillna(df["close"] * 0.01)
        
        if "macro_ema" not in df.columns or df["macro_ema"].isna().all():
            df["macro_ema"] = ind.calculate_ema(df["close"], window=macro_period)
            
        if "btc_close" in df.columns:
            df["btc_ema"] = ind.calculate_ema(df["btc_close"], window=macro_period)
            
        return df

    def compute_signal_regime_component(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tier 2: ML Signal and HMM Regime computation."""
        if "atr" not in df.columns:
            df = self.generate_base_indicators(df)

        n = len(df)
        
        # 1. Identify ML Alpha features
        # We prefer a mean of all ml_alpha_XX columns if multiple exist, otherwise ml_alpha_00
        surviving_gp_cols = [
            c for c in df.columns
            if c.startswith("ml_alpha_") and c[-2:].isdigit() and float(df[c].std()) > 1e-6
        ]
        
        if surviving_gp_cols:
            gp = df[surviving_gp_cols].mean(axis=1).to_numpy(dtype=np.float64)
        elif "ml_alpha_00" in df.columns:
            gp = df["ml_alpha_00"].to_numpy(dtype=np.float64, copy=False)
        else:
            _logger.warning("No ml_alpha_00 found, using neutral 0.5")
            gp = np.full(n, 0.5, dtype=np.float64)

        # 2. Extract HMM Modulators & Crisis Prob
        gp_long = (
            df["ml_alpha_long"].to_numpy(dtype=np.float64)
            if "ml_alpha_long" in df.columns else gp
        )
        gp_short = (
            df["ml_alpha_short"].to_numpy(dtype=np.float64)
            if "ml_alpha_short" in df.columns else gp
        )
        
        hml = (
            df["hmm_modulator_long"].to_numpy(dtype=np.float64)
            if "hmm_modulator_long" in df.columns else np.ones(n)
        )
        hms = (
            df["hmm_modulator_short"].to_numpy(dtype=np.float64)
            if "hmm_modulator_short" in df.columns else np.ones(n)
        )
        
        # 3. Cross-sectional Scoring
        # xs_score_long: higher is better for LONG
        # xs_score_short: lower is better for SHORT (inverted in backtest engine)
        df["xs_score_long"] = gp_long * hml
        df["xs_score_short"] = gp_short / np.maximum(hms, 0.1)
        
        if "hmm_prob_crisis" not in df.columns:
            df["hmm_prob_crisis"] = 0.0

        # 4. Signal Outputs for Backtest Engine
        # ml_calib_prob is set by the ML pipeline (Platt/MetaLabeler); preserve it if present.
        # Only compute here as fallback when the merge did not populate it.
        if "ml_calib_prob" not in df.columns:
            pl_arr = (
                df["ml_calib_prob_long"].to_numpy(dtype=np.float64)
                if "ml_calib_prob_long" in df.columns
                else np.full(n, 0.5, dtype=np.float64)
            )
            ps_arr = (
                df["ml_calib_prob_short"].to_numpy(dtype=np.float64)
                if "ml_calib_prob_short" in df.columns
                else np.full(n, 0.5, dtype=np.float64)
            )
            df["ml_calib_prob"] = np.maximum(pl_arr, ps_arr)
        df["strength_filter"] = gp  # Using raw GP as strength indicator
        
        # Rank score used for symbol selection in multi-symbol engine
        df["slot_rank_score"] = df["xs_score_long"]
        
        # Trend Direction: 1.0 for LONG, -1.0 for SHORT, 0.0 for Neutral
        # We use a threshold around 0.5
        gp_centered = gp - 0.5
        df["trend_direction"] = np.where(np.abs(gp_centered) > 0.01, np.sign(gp_centered), 0.0)
        
        # Entry thresholds (ML usually enters at market, so we use 0.0/inf)
        df["entry_upper"] = 0.0
        df["entry_lower"] = 999999.0
        
        # Kill signal (HMM-based emergency exit)
        df["kill_signal"] = 0.0
        if "hmm_prob_crisis" in df.columns:
            # Emergency exit if crisis probability > 0.8
            df["kill_signal"] = np.where(df["hmm_prob_crisis"] > 0.8, 1.0, 0.0)

        # 5. Macro Filter (Optional BTC trend)
        use_macro = bool(self.params.get("USE_MACRO_FILTER", False))
        if use_macro and "btc_close" in df.columns and "btc_ema" in df.columns:
            btc_bull = (df["btc_close"] > df["btc_ema"]).to_numpy()
            # Only allow LONG if BTC is bull
            df["strength_filter"] = np.where(
                (df["trend_direction"] == 1.0) & (~btc_bull), 0.0, df["strength_filter"]
            )
            df["trend_direction"] = np.where(
                (df["trend_direction"] == 1.0) & (~btc_bull), 0.0, df["trend_direction"]
            )

        return df

    def compute_sizing_component(self, df: pd.DataFrame) -> pd.Series:
        """Tier 3: Sizing computation (ML-centric)."""
        # Default to a simple volatility target logic if no registry is used
        # or just return a default Kelly factor if pre-calculated
        if "garch_kelly_f" in df.columns:
            return df["garch_kelly_f"]
        
        # Default fallback
        return pd.Series(1.0, index=df.index)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Entry point for backtest engine."""
        df = self.generate_base_indicators(df)
        df = self.compute_signal_regime_component(df)
        df["garch_kelly_f"] = self.compute_sizing_component(df)
        return df


# --- Standardized Aliases for Phase 2 Migration ---
UltimateStrategy = FuturesMLStrategy
FuturesMLPipelineStrategy = FuturesMLStrategy
