"""Rolling profit-factor Kelly on signed bar returns."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from src.domain.futures.sizing.registry import register_futures_sizing


@register_futures_sizing
class ProfitFactorKellyFuturesSizing:
    """Rolling profit-factor Kelly sizing with regime-conditional overrides."""

    name: ClassVar[str] = "profit_factor_kelly"
    param_space: ClassVar[dict[str, Any]] = {
        "PFK_WINDOW": {"type": "int", "low": 30, "high": 120, "step": 10},
        "PFK_MIN_F": {"type": "float", "low": 0.05, "high": 0.30, "step": 0.05},
        "KELLY_FRACTION": {"type": "float", "low": 0.2, "high": 0.8, "step": 0.1},
        "PFK_TARGET_VOL": {"type": "float", "low": 0.005, "high": 0.020, "step": 0.005},
        # Stress Regime Parameters
        "STRESS_VOL_Z": {"type": "float", "low": 2.0, "high": 3.5, "step": 0.5},
        "STRESS_FR_Z": {"type": "float", "low": 2.0, "high": 3.5, "step": 0.5},
    }

    def compute(self, df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
        """Compute Kelly-based position sizes with HMM-based regime overrides."""
        close = df["close"].to_numpy(dtype=np.float64)
        n = len(close)
        window = int(max(5, params.get("PFK_WINDOW", 60)))
        min_f = float(params.get("PFK_MIN_F", 0.1))
        kelly_frac = float(params.get("KELLY_FRACTION", 0.5))
        max_exp = float(params.get("MAX_EXPOSURE", 1.0))
        min_obs = int(max(10, window // 3))
        
        target_vol = float(params.get("PFK_TARGET_VOL", 0.01))
        stress_vol_z_thr = float(params.get("STRESS_VOL_Z", 2.5))
        stress_fr_z_thr = float(params.get("STRESS_FR_Z", 2.5))

        r_series = pd.Series(close).pct_change()
        funding_adj = np.zeros(n, dtype=np.float64)
        fr_series = pd.Series(np.zeros(n, dtype=np.float64))
        if "funding_rate" in df.columns:
            fr_series = df["funding_rate"].fillna(0.0).reset_index(drop=True)
            funding_adj = np.nan_to_num(
                fr_series.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
            )
        net_r = r_series - pd.Series(funding_adj)

        # --- Proactive Stress Regime Indicator ---
        # 1. Volatility Spike (Crypto VIX)
        vol = r_series.rolling(window=window).std()
        vol_mean = vol.rolling(window=window * 2, min_periods=window).mean()
        vol_std = vol.rolling(window=window * 2, min_periods=window).std()
        vol_z = (vol - vol_mean) / np.maximum(vol_std, 1e-12)
        
        # --- Volatility Targeting Scaling ---
        # final_f = f * (target_vol / current_realized_vol)
        vol_scaler = (target_vol / np.maximum(vol.to_numpy(dtype=np.float64), 1e-12))
        vol_scaler = np.nan_to_num(vol_scaler, nan=1.0)
        
        # 2. Funding Rate Extreme
        fr_mean = fr_series.rolling(window=window).mean()
        fr_std = fr_series.rolling(window=window).std()
        fr_z = (fr_series - fr_mean) / np.maximum(fr_std, 1e-12)
        fr_abs_extreme = fr_series.abs() > 0.0015  # Absolute extreme (approx > 164% APR)
        
        # Combine Stress Conditions
        stress_regime = (vol_z > stress_vol_z_thr) | (fr_z.abs() > stress_fr_z_thr) | fr_abs_extreme
        regime_penalty = np.where(stress_regime, 0.25, 1.0)  # 75% risk reduction during chaos

        td = (
            df["trend_direction"].to_numpy(dtype=np.float64)
            if "trend_direction" in df.columns
            else np.zeros(n, dtype=np.float64)
        )
        signed_np = np.where(
            td > 0,
            net_r.to_numpy(dtype=np.float64),
            np.where(td < 0, -net_r.to_numpy(dtype=np.float64), 0.0),
        )

        s = pd.Series(signed_np)
        n_pos = s.gt(0.0).rolling(window, min_periods=window).sum()
        n_neg = s.lt(0.0).rolling(window, min_periods=window).sum()
        sum_pos = s.where(s > 0.0, 0.0).rolling(window, min_periods=window).sum()
        sum_neg = (-s.where(s < 0.0, 0.0)).rolling(window, min_periods=window).sum()
        n_fin = s.notna().rolling(window, min_periods=window).sum()

        n_pos_a = n_pos.to_numpy(dtype=np.float64)
        n_neg_a = n_neg.to_numpy(dtype=np.float64)
        sum_pos_a = sum_pos.to_numpy(dtype=np.float64)
        sum_neg_a = sum_neg.to_numpy(dtype=np.float64)
        n_fin_a = n_fin.to_numpy(dtype=np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            avg_win = np.where(n_pos_a > 0.0, sum_pos_a / n_pos_a, np.nan)
            avg_loss = np.where(n_neg_a > 0.0, sum_neg_a / n_neg_a, np.nan)
            r_safe = np.maximum(avg_loss, 1e-12)
            r_payoff = avg_win / r_safe
            w_win = np.where((n_pos_a + n_neg_a) > 0.0, n_pos_a / (n_pos_a + n_neg_a), np.nan)
            f_star = (w_win * r_payoff - (1.0 - w_win)) / np.maximum(r_payoff, 1e-12)

        f_star = np.nan_to_num(f_star, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply Kelly fraction, Volatility Scaling and Regime Penalty
        f = f_star * kelly_frac * vol_scaler * regime_penalty

        low_data = n_fin_a < float(min_obs)
        f = np.where(low_data, min_f, f)
        active = (td != 0.0).astype(np.float64)
        floored = np.where(active > 0, np.maximum(f, min_f), f)
        return np.clip(floored, 0.0, max_exp).astype(np.float64)
