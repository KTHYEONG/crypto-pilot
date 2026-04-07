"""
Rolling profit-factor Kelly on signed bar returns; funding subtracted from pct change.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.futures_strategy.sizing.registry import register_futures_sizing


@register_futures_sizing
class ProfitFactorKellyFuturesSizing:
    name: ClassVar[str] = "profit_factor_kelly"
    param_space: ClassVar[Dict[str, Any]] = {
        "PFK_WINDOW": {"type": "int", "low": 30, "high": 120, "step": 10},
        "PFK_MIN_F": {"type": "float", "low": 0.05, "high": 0.30, "step": 0.05},
        "KELLY_FRACTION": {"type": "float", "low": 0.2, "high": 0.8, "step": 0.1},
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
        funding_adj = np.zeros(n, dtype=np.float64)
        if "funding_rate" in df.columns:
            funding_adj = np.nan_to_num(
                df["funding_rate"].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
            )
        net_r = r - pd.Series(funding_adj)

        td = (
            df["trend_direction"].to_numpy(dtype=np.float64)
            if "trend_direction" in df.columns
            else np.zeros(n, dtype=np.float64)
        )
        signed_np = np.where(td > 0, net_r.to_numpy(dtype=np.float64), np.where(td < 0, -net_r.to_numpy(dtype=np.float64), 0.0))

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
        f = f_star * kelly_frac
        low_data = n_fin_a < float(min_obs)
        f = np.where(low_data, min_f, f)
        active = (td != 0.0).astype(np.float64)
        floored = np.where(active > 0, np.maximum(f, min_f), f)
        return np.clip(floored, 0.0, max_exp).astype(np.float64)
