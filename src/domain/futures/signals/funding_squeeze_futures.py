"""
Funding Rate Squeeze: Extreme funding rate z-score -> crowded position reversal.

**[Math/Quant Basis]**
- Perpetual futures funding rate is a direct measure of long/short leverage imbalance
  (Cong & He 2022). Extreme positive funding (longs paying shorts) indicates
  over-leveraged longs -> short squeeze reversal risk.
- Z-score of rolling funding_rate_sum detects statistically extreme crowding.
- Mean-reversion half-life of funding extremes is empirically short (1-3 bars at 4H),
  making this a high-frequency reversal signal orthogonal to price-based signals.
- OOS robustness is high because funding rate is a microstructure reality, not a
  data-mined price pattern.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.domain.futures.signals.base import FuturesSignalOutput
from src.domain.futures.signals.registry import register_futures_signal


@register_futures_signal
class FundingSqueezeFuturesSignal:
    name: ClassVar[str] = "FUNDING_SQUEEZE"
    param_space: ClassVar[Dict[str, Any]] = {
        # Rolling window to compute funding rate stats (2-12 days at 4H)
        "FUND_Z_WINDOW": {"type": "int", "low": 12, "high": 72, "step": 12},
        # Z-score threshold to detect extreme crowding
        "FUND_Z_THRESHOLD": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.25},
        # Z-score level at which position is exited (return toward neutral)
        "FUND_EXIT_THRESHOLD": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.25},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FuturesSignalOutput:
        window = int(params.get("FUND_Z_WINDOW", 24))
        z_thresh = float(params.get("FUND_Z_THRESHOLD", 2.0))
        exit_thresh = float(params.get("FUND_EXIT_THRESHOLD", 0.25))

        n = len(df)
        if n == 0:
            z = np.array([], dtype=np.float64)
            return FuturesSignalOutput(
                long_entry=np.array([], dtype=np.bool_),
                short_entry=np.array([], dtype=np.bool_),
                kill_long=z,
                kill_short=z,
                rank_score=z,
            )

        # Prefer funding_rate_sum (sum within bar); fallback to funding_rate
        if "funding_rate_sum" in df.columns:
            raw = df["funding_rate_sum"].to_numpy(dtype=np.float64)
        elif "funding_rate" in df.columns:
            raw = df["funding_rate"].to_numpy(dtype=np.float64)
        else:
            # No funding data available: return zero signal
            z = np.zeros(n, dtype=np.float64)
            return FuturesSignalOutput(
                long_entry=np.zeros(n, dtype=np.bool_),
                short_entry=np.zeros(n, dtype=np.bool_),
                kill_long=z,
                kill_short=z,
                rank_score=z,
            )

        raw = np.nan_to_num(raw, nan=0.0)
        fund_s = pd.Series(raw)

        rolling_mean = fund_s.rolling(window=window, min_periods=window // 2).mean().to_numpy()
        rolling_std = fund_s.rolling(window=window, min_periods=window // 2).std().to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            z_fund = (raw - rolling_mean) / np.maximum(rolling_std, 1e-12)

        z_fund = np.nan_to_num(z_fund, nan=0.0)

        # Negative funding extreme: shorts over-crowded -> long squeeze reversal
        long_entry = z_fund < -z_thresh
        # Positive funding extreme: longs over-crowded -> short squeeze reversal
        short_entry = z_fund > z_thresh

        # Kill when funding returns toward neutral
        kill_long = (z_fund > -exit_thresh).astype(np.float64)
        kill_short = (z_fund < exit_thresh).astype(np.float64)

        # rank_score: negative z_fund -> higher score favours long (more negative funding)
        rank_score = -z_fund

        return FuturesSignalOutput(
            long_entry=long_entry.astype(np.bool_),
            short_entry=short_entry.astype(np.bool_),
            kill_long=kill_long,
            kill_short=kill_short,
            rank_score=rank_score,
        )
