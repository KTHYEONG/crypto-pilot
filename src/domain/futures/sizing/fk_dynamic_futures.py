"""Fractional Kelly Dynamic Sizing: EWMA volatility balanced Kelly sizing."""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.domain.futures.sizing.registry import register_futures_sizing


@register_futures_sizing
class FractionalKellyDynamicSizing:
    name: ClassVar[str] = "fk_dynamic"
    param_space: ClassVar[Dict[str, Any]] = {
        "FK_EWMA_LAMBDA": {"type": "float", "low": 0.90, "high": 0.99, "step": 0.01},
        "FK_TARGET_VOL": {"type": "float", "low": 0.01, "high": 0.05, "step": 0.005},
        "FK_FRACTION": {"type": "float", "low": 0.2, "high": 0.6, "step": 0.1},
        "FK_MAX_SIZE": {"type": "float", "low": 0.5, "high": 1.5, "step": 0.1},
        "FK_WINDOW": {"type": "int", "low": 30, "high": 120, "step": 10},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        ewma_lambda = float(params.get("FK_EWMA_LAMBDA", 0.94))
        target_vol = float(params.get("FK_TARGET_VOL", 0.02)) # Step-wise daily target vol approx
        kelly_fraction = float(params.get("FK_FRACTION", 0.3))
        max_size = float(params.get("FK_MAX_SIZE", 1.0))
        window = int(params.get("FK_WINDOW", 60))

        close = df["close"].to_numpy(dtype=np.float64)
        n = len(close)
        
        if n < window:
            return np.ones(n, dtype=np.float64) * 0.1

        # Calculate log returns
        returns = np.log(close[1:] / close[:-1])
        returns = np.insert(returns, 0, 0.0)
        ret_s = pd.Series(returns)

        # 1. EWMA Volatility (RiskMetrics style)
        # vol^2_t = lambda * vol^2_{t-1} + (1-lambda) * ret^2_t
        ewma_var = ret_s.pow(2).ewm(alpha=1.0 - ewma_lambda, adjust=False).mean().to_numpy()
        ewma_vol = np.sqrt(np.maximum(ewma_var, 1e-12))

        # 2. Performance-based Edge (Kelly)
        td = (
            df["trend_direction"].to_numpy(dtype=np.float64)
            if "trend_direction" in df.columns
            else np.zeros(n, dtype=np.float64)
        )
        signed_ret = np.where(td > 0, returns, np.where(td < 0, -returns, 0.0))
        s = pd.Series(signed_ret)

        n_pos = (s > 0).rolling(window).sum().to_numpy()
        n_neg = (s < 0).rolling(window).sum().to_numpy()
        sum_pos = s.where(s > 0, 0).rolling(window).sum().to_numpy()
        sum_neg = (-s.where(s < 0, 0)).rolling(window).sum().to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            avg_win = np.where(n_pos > 0, sum_pos / n_pos, 0.0)
            avg_loss = np.where(n_neg > 0, sum_neg / n_neg, 1e-12)
            r_payoff = avg_win / avg_loss
            w_win = np.where((n_pos + n_neg) > 0, n_pos / (n_pos + n_neg), 0.5)
            # Kelly f* = (p*b - q) / b  where p=w_win, b=r_payoff, q=1-p
            f_star = (w_win * r_payoff - (1.0 - w_win)) / np.maximum(r_payoff, 1e-12)

        f_star = np.nan_to_num(f_star, nan=0.0)
        
        # 3. Dynamic Volatility Adjustment (Target Vol / Realized Vol)
        # This scales the final bet based on how volatile the market currently is vs our target
        vol_adj = target_vol / np.maximum(ewma_vol, 1e-6)
        
        # Combined size: Fractional Kelly inhibited by Volatility
        final_size = f_star * kelly_fraction * vol_adj
        
        # Only bet if trend_direction is set by signal
        active = (td != 0).astype(np.float64)
        final_size = final_size * active

        return np.clip(final_size, 0.0, max_size).astype(np.float64)
