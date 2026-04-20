from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.indicators.indicators import get_indicator_engine
from src.domain.futures.regimes import FUTURES_REGIME_REGISTRY
from src.domain.futures.signals import FUTURES_SIGNAL_REGISTRY
from src.domain.futures.signals.ml_calib_prob_futures import apply_ml_calib_gate_column
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
    Futures: plugin signal x regime x sizing dispatch.
    ENTRY_SHIFT: signal at i uses bar i-1 for entry columns (engine reads prev bar).
    """

    INDICATORS = _FUTURES_INDICATORS
    ENTRY_SHIFT = False

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
        """Tier 2: Signal and Regime computation with Ensemble & Dynamic Weighting."""
        if "atr" not in df.columns:
            df = self.generate_base_indicators(df)

        st_key = str(self.params.get("SIGNAL_TYPE", "RSM_VT")).upper()
        rt_key = str(self.params.get("REGIME_TYPE", "EMA_ATR")).upper()
        reg_engine = FUTURES_REGIME_REGISTRY.get(rt_key) or FUTURES_REGIME_REGISTRY["EMA_ATR"]
        long_mult, short_mult = reg_engine.compute_long_short_mult(df, self.params)

        # [Step 3] Dynamic Regime Weights (Chameleon Factor)
        # If discovery phase provided weights for each regime, apply them here
        reg_weights = self.params.get("REGIME_WEIGHTS")
        if isinstance(reg_weights, dict):
            # Scale multiplier based on current regime's proven IC performance
            # rt_key is the active regime class; if we have weights for it, apply
            w = float(reg_weights.get(rt_key, 1.0))
            long_mult *= w
            short_mult *= w

        # [Step 2] Ensemble Support
        # Check if we are running in ensemble mode (Multiple signals)
        ensemble_sigs = self.params.get("ENSEMBLE_SIGNALS")
        if isinstance(ensemble_sigs, list) and len(ensemble_sigs) > 1:
            # Aggregate signals: For now, we take the mean rank_score and AND/OR for entry
            # In institutional quant, we usually sum the Z-scores of signals
            all_long = np.zeros(len(df), dtype=bool)
            all_short = np.zeros(len(df), dtype=bool)
            all_rank = np.zeros(len(df), dtype=np.float64)
            all_kill_l = np.zeros(len(df), dtype=np.float64)
            all_kill_s = np.zeros(len(df), dtype=np.float64)
            
            for s_info in ensemble_sigs:
                s_name = s_info["name"]
                s_params = {**self.params, **s_info.get("params", {})}
                s_engine = FUTURES_SIGNAL_REGISTRY[s_name]
                s_out = s_engine.compute(df, s_params)
                
                # Simple OR logic for entry, mean for rank
                all_long |= s_out.long_entry
                all_short |= s_out.short_entry
                all_rank += s_out.rank_score
                all_kill_l = np.maximum(all_kill_l, s_out.kill_long)
                all_kill_s = np.maximum(all_kill_s, s_out.kill_short)
            
            long_e = all_long
            short_e = all_short
            rank_score = all_rank / len(ensemble_sigs)
            kill_long = all_kill_l
            kill_short = all_kill_s
        else:
            sig_engine = FUTURES_SIGNAL_REGISTRY.get(st_key) or FUTURES_SIGNAL_REGISTRY["RSM_VT"]
            sig_out = sig_engine.compute(df, self.params)
            long_e = sig_out.long_entry.astype(np.bool_)
            short_e = sig_out.short_entry.astype(np.bool_)
            rank_score = sig_out.rank_score
            kill_long = sig_out.kill_long
            kill_short = sig_out.kill_short

        close_a = df["close"].to_numpy(dtype=np.float64)
        if st_key == "ML_CALIB_PROB" and bool(self.params.get("USE_CS_RANK_ENGINE", True)):
            n = len(df)
            gp = (
                df["gp_alpha_00"].to_numpy(dtype=np.float64, copy=False)
                if "gp_alpha_00" in df.columns
                else np.zeros(n, dtype=np.float64)
            )
            hml = (
                df["hmm_modulator_long"].to_numpy(dtype=np.float64, copy=False)
                if "hmm_modulator_long" in df.columns
                else np.ones(n, dtype=np.float64)
            )
            hms = (
                df["hmm_modulator_short"].to_numpy(dtype=np.float64, copy=False)
                if "hmm_modulator_short" in df.columns
                else np.ones(n, dtype=np.float64)
            )
            df["xs_score_long"] = gp * hml
            df["xs_score_short"] = (-gp) * hms
            if "hmm_prob_crisis" not in df.columns:
                df["hmm_prob_crisis"] = 0.0
            df["ml_calib_prob"] = 1.0
            df["ml_calib_prob_long"] = 1.0
            df["ml_calib_prob_short"] = 1.0
            long_e = np.zeros(n, dtype=np.bool_)
            short_e = np.zeros(n, dtype=np.bool_)
            rank_score = df["xs_score_long"].to_numpy(dtype=np.float64, copy=False)
            df["trend_direction"] = 0.0
            df["entry_upper"] = 0.0
            df["entry_lower"] = 999999.0
        else:
            if st_key == "ML_CALIB_PROB":
                apply_ml_calib_gate_column(df, self.params)
            df["trend_direction"] = np.where(long_e, 1.0, np.where(short_e, -1.0, 0.0))
            df["entry_upper"] = np.where(long_e, 0.0, 999999.0)
            df["entry_lower"] = np.where(short_e, close_a, 0.0)

        df["entry_upper"] = self._shift_if_needed(df["entry_upper"])
        df["entry_lower"] = self._shift_if_needed(df["entry_lower"])

        td = df["trend_direction"].to_numpy(dtype=np.float64)
        df["kill_signal"] = np.where(td == 1.0, kill_long, kill_short)
        df["slot_rank_score"] = rank_score

        # Macro Filter - Skip if using direct ML signals or NONE regime to avoid interference
        if rt_key != "NONE" and "btc_close" in df.columns and "btc_ema" in df.columns:
            btc_bull = (df["btc_close"] > df["btc_ema"]).to_numpy()
            long_mult = np.where(btc_bull, long_mult, 0.0)

        df["strength_filter"] = np.where(
            long_e,
            long_mult.astype(np.float64),
            np.where(short_e, short_mult.astype(np.float64), 0.0),
        )
        df["regime_risk_mult"] = np.maximum(long_mult, short_mult).astype(np.float64)
        
        return df

    def compute_sizing_component(self, df: pd.DataFrame) -> pd.Series:
        """Tier 3: Sizing computation."""
        sm_key = str(self.params.get("SIZING_METHOD", "vol_target")).lower()
        sizer = FUTURES_SIZING_REGISTRY.get(sm_key) or FUTURES_SIZING_REGISTRY["vol_target"]
        return pd.Series(sizer.compute(df, self.params), index=df.index)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Legacy entry point for monolithic computation."""
        df = self.generate_base_indicators(df)
        df = self.compute_signal_regime_component(df)
        df["garch_kelly_f"] = self.compute_sizing_component(df)
        return df


# Alias for backward compatibility
FuturesPipelineStrategy = UltimateStrategy
