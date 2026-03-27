from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from src.spot_strategy.frama_evr_poc import compute_evr_zscore, compute_frama_series
from src.spot_strategy.garch_sizer import compute_garch_kelly_series
from src.spot_strategy.student_t_hmm_regime import compute_walk_forward_hmm_student_t
from src.strategy.base.ultimate import UltimateStrategyBase
from src.futures_strategy.strategies_futures import _FUTURES_INDICATORS as _SPOT_INDICATORS


def _build_frama_evr_signals(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, np.ndarray]:
    """FRAMA position+slope gate + directional-bull EvR rolling z-score."""
    frama_period = int(params.get("FRAMA_PERIOD", 16))
    evr_window = int(params.get("EVR_WINDOW", 20))
    frama_min_slope = float(params.get("FRAMA_MIN_SLOPE", 0.0005))
    close_arr = df["close"].to_numpy(dtype=np.float64)
    frama = compute_frama_series(
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        close_arr,
        frama_period,
    )
    frama_prev = np.roll(frama, 1)
    slope_pct = (frama - frama_prev) / np.maximum(np.abs(frama_prev), 1e-12)
    frama_bull = (close_arr > frama) & (slope_pct > frama_min_slope)
    frama_bull[0] = False
    evr_z = compute_evr_zscore(
        df["open"].to_numpy(dtype=np.float64),
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        df["close"].to_numpy(dtype=np.float64),
        df["volume"].to_numpy(dtype=np.float64),
        evr_window,
        directional_bull=True,
    )
    return {
        "frama_bull": frama_bull.astype(np.bool_, copy=False),
        "evr_z": evr_z.astype(np.float64, copy=False),
    }


class UltimateSpotStrategy(UltimateStrategyBase):
    """
    Spot: FRAMA + EvR entry filter; Student-t HMM (state-anchored walk-forward) for regime only.
    Execution uses ATR from a single period; sizing uses fixed Kelly column (no GARCH); no kill ladder.
    """

    INDICATORS = _SPOT_INDICATORS
    ENTRY_SHIFT = False

    def _compute_warmup_bars(self) -> int:
        p = self.params
        hmm_w = int(p.get("HMM_TRAIN_WINDOW", 360))
        frama_p = int(p.get("FRAMA_PERIOD", 16))
        evr_w = int(p.get("EVR_WINDOW", 20))
        atr_p = int(p.get("ATR_PERIOD", 14))
        max_p = max(hmm_w, frama_p * 2, evr_w, atr_p, 50)
        return max(300, int(max_p * 3))

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = df[col].astype(np.float64)

        if "btc_close" not in df.columns:
            df["btc_close"] = df["close"].astype(np.float64)

        ind = self._ind()
        atr_period = int(self.params.get("ATR_PERIOD", 14))

        hmm_train = int(self.params.get("HMM_TRAIN_WINDOW", 360))
        hmm_rf = int(self.params.get("HMM_RETRAIN_FREQ", 24))

        df["atr"] = ind.calculate_atr(df, window=atr_period)

        frama_signals = _build_frama_evr_signals(df, self.params)
        frama_bull_s = pd.Series(frama_signals["frama_bull"], index=df.index, dtype=bool)
        evr_z_s = pd.Series(frama_signals["evr_z"], index=df.index, dtype=np.float64)
        evr_thr = float(self.params.get("EVR_THRESHOLD", 0.5))

        bull_raw = (frama_bull_s & (evr_z_s > evr_thr)).astype(np.float64).values

        viterbi, p_bull_hmm, p_side_hmm = compute_walk_forward_hmm_student_t(
            df,
            train_window=hmm_train,
            retrain_freq=hmm_rf,
        )

        df["hmm_viterbi"] = viterbi.astype(np.int32)
        df["p_bull"] = p_bull_hmm.astype(np.float64)
        df["p_side"] = p_side_hmm.astype(np.float64)

        p_bear_hmm = np.clip(1.0 - p_bull_hmm - p_side_hmm, 0.0, 1.0)
        bear_hmm = viterbi == 0
        soft_risk_mult = np.clip(1.0 - 2.0 * p_bear_hmm, 0.05, 1.0)
        regime_risk_mult = np.where(bear_hmm, 0.0, soft_risk_mult).astype(np.float64)
        df["regime_risk_mult"] = regime_risk_mult

        garch_window = int(self.params.get("GARCH_WINDOW", 240))
        garch_retrain_freq = int(self.params.get("GARCH_RETRAIN_FREQ", 24))
        garch_nu_fallback = float(self.params.get("GARCH_NU_FALLBACK", 5.0))
        df["garch_kelly_f"] = compute_garch_kelly_series(
            df["close"].astype(np.float64),
            window=garch_window,
            retrain_freq=garch_retrain_freq,
            nu_fallback=garch_nu_fallback,
        ).to_numpy(dtype=np.float64, copy=False)
        df["kill_signal"] = np.zeros(len(df), dtype=np.float64)

        bull_breakout = bull_raw > 0.5

        df["long_entry_signal"] = np.where(bull_breakout, 1.0, 0.0)
        df["sig_long_entry_signal"] = df["long_entry_signal"]

        df["entry_upper"] = np.where(bull_breakout, df["close"], 999999.0)
        df["entry_lower"] = np.full(len(df), 999999.0, dtype=np.float64)

        df["strength_filter"] = np.where(bull_breakout, 1, 0)
        df["trend_direction"] = np.where(bull_breakout, 1, 0)

        evr_clip = np.clip(frama_signals["evr_z"], -3.0, 3.0).astype(np.float64, copy=False)
        df["slot_rank_score"] = evr_clip.astype(np.float64)

        df["entry_upper"] = self._shift_if_needed(cast(pd.Series, df["entry_upper"]))
        df["entry_lower"] = self._shift_if_needed(cast(pd.Series, df["entry_lower"]))

        df["atr"] = df["atr"].ffill().fillna(df["close"] * 0.01)
        rr = df["regime_risk_mult"].ffill().fillna(0.5)
        df["regime_risk_mult"] = np.where(
            rr <= 0.0,
            0.0,
            np.clip(rr.to_numpy(dtype=np.float64), 0.05, 1.0),
        )
        return df
