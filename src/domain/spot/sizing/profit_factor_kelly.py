"""
Profit-factor Kelly sizing: f* = (W*R - (1-W)) / R from rolling win rate W and payoff R.

W = share of positive bar returns in window; R = avg_win / avg_loss on signed returns.
Causal; uses shared KELLY_FRACTION and MAX_EXPOSURE like rolling_kelly.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.domain.spot.sizing.registry import register_sizing


@register_sizing
class ProfitFactorKellySizing:
    name: ClassVar[str] = "profit_factor_kelly"
    param_space: ClassVar[Dict[str, Any]] = {
        "PFK_WINDOW": {"type": "int", "low": 30, "high": 120, "step": 10},
        "PFK_MIN_F": {"type": "float", "low": 0.05, "high": 0.30, "step": 0.05},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        close = df["close"].to_numpy(dtype=np.float64)
        n = len(close)
        window = int(max(5, params.get("PFK_WINDOW", 60)))
        min_f = float(params.get("PFK_MIN_F", 0.1))
        kelly_frac = float(params.get("KELLY_FRACTION", 0.5))
        max_exp = float(params.get("MAX_EXPOSURE", 1.0))
        min_obs = int(max(10, window // 3))

        r = pd.Series(close).pct_change()
        n_pos = r.gt(0.0).rolling(window, min_periods=window).sum()
        n_neg = r.lt(0.0).rolling(window, min_periods=window).sum()
        sum_pos = r.where(r > 0.0, 0.0).rolling(window, min_periods=window).sum()
        sum_neg = (-r.where(r < 0.0, 0.0)).rolling(window, min_periods=window).sum()
        n_fin = r.notna().rolling(window, min_periods=window).sum()

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
        f = f_star * kelly_frac
        low_data = n_fin_a < float(min_obs)
        f = np.where(low_data, min_f, f)

        entry_mask = (
            df["long_entry_signal"].to_numpy(dtype=np.float64)
            if "long_entry_signal" in df.columns
            else np.ones(n, dtype=np.float64)
        )
        floored = np.where(entry_mask > 0, np.maximum(f, min_f), f)
        return np.clip(floored, 0.0, max_exp).astype(np.float64)
