"""Binary Crisis Detector — rules-based soft-scoring layer.

Computes a 0~1 crisis probability score from macro features.
Designed to be causal (uses only past data) and used as an overlay
on top of the 4-state HMM posterior.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# Rolling window size (4h bars → 168 = 4 weeks)
_WINDOW: int = 168
_MIN_PERIODS: int = 24

# Rule weights
_W_R1: float = 0.40  # Extreme negative return
_W_R2: float = 0.25  # Liquidity stress
_W_R3: float = 0.20  # Funding collapse + downside vol surge
_W_R4: float = 0.15  # OI liquidation + return below 2-sigma
_W_R5: float = 0.15  # Cross-sectional dispersion spike
_W_R6: float = 0.25  # Persistent downside vol elevation


def _rolling_mean(ser: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Compute causal rolling mean (no look-ahead)."""
    return ser.rolling(window=window, min_periods=min_periods).mean()


def _rolling_std(ser: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Compute causal rolling std; replace 0 with NaN to avoid division issues."""
    return ser.rolling(window=window, min_periods=min_periods).std().replace(0.0, np.nan)


def _rolling_quantile(ser: pd.Series, q: float, window: int, min_periods: int) -> pd.Series:
    """Compute causal rolling quantile (not expanding)."""
    return ser.rolling(window=window, min_periods=min_periods).quantile(q)


class CrisisDetector:
    """Rules-based soft crisis scoring layer.

    Scores each bar from 0 to 1 based on multiple macro/micro signals.
    All statistics use causal rolling windows to prevent look-ahead bias.

    Rules:
        R1 (weight=0.40): Return < mean - 3.5*std  → extreme left-tail event
        R2 (weight=0.25): Liquidity proxy > 99th pct  → funding market stress
        R3 (weight=0.20): Funding momentum < 1st pct AND downside vol > 80th pct
        R4 (weight=0.15): OI delta < 2nd pct AND return < mean - 2*std  → liquidation cascade
        R5 (weight=0.15): Cross-sectional dispersion > 95th pct  → fragmentation
        R6 (weight=0.25): Downside vol > 90th pct  → persistent risk elevation
    """

    def score(
        self,
        features_df: pd.DataFrame,
        returns_ser: pd.Series,
    ) -> np.ndarray:
        """Compute per-bar crisis score in [0, 1].

        Args:
            features_df: DataFrame indexed by datetime with optional macro columns.
            returns_ser: Log-return series aligned to features_df.index.

        Returns:
            np.ndarray of shape (N,) with values in [0.0, 1.0].
            Leading rows with insufficient history are filled with 0.0.

        Note:
            All rolling statistics are computed on past data only (causal).
            Missing optional macro columns are silently skipped (rule weight omitted).

        """
        idx = features_df.index
        n = len(idx)
        if n == 0:
            return np.empty(0, dtype=np.float64)

        ret = returns_ser.reindex(idx).fillna(0.0).astype(np.float64)

        # Pre-compute return rolling statistics (used by R1, R4)
        ret_mean = _rolling_mean(ret, _WINDOW, _MIN_PERIODS)
        ret_std = _rolling_std(ret, _WINDOW, _MIN_PERIODS)

        total_score = np.zeros(n, dtype=np.float64)

        # ── R1: Extreme negative return (always computed) ─────────────────────
        # ret < mean - 3.5*std
        threshold_r1 = ret_mean - 3.5 * ret_std.fillna(np.inf)
        r1_mask = (ret < threshold_r1).fillna(False).to_numpy()
        total_score += _W_R1 * r1_mask.astype(np.float64)
        _logger.debug("R1 triggered pct=%.3f", r1_mask.mean())

        # ── R2: Liquidity proxy > 99th percentile ─────────────────────────────
        col_liq = "macro_liq_proxy_24h"
        if col_liq in features_df.columns:
            liq = features_df[col_liq].astype(np.float64)
            liq_q99 = _rolling_quantile(liq, 0.99, _WINDOW, _MIN_PERIODS)
            r2_mask = (liq > liq_q99).fillna(False).to_numpy()
            total_score += _W_R2 * r2_mask.astype(np.float64)
            _logger.debug("R2 triggered pct=%.3f", r2_mask.mean())
        else:
            _logger.debug("R2 skipped: %s not found", col_liq)

        # ── R3: Funding collapse AND downside vol surge ───────────────────────
        col_fund = "macro_funding_mom_24h"
        col_dvol = "macro_downside_vol_24h"
        if col_fund in features_df.columns and col_dvol in features_df.columns:
            fund = features_df[col_fund].astype(np.float64)
            dvol = features_df[col_dvol].astype(np.float64)
            fund_q01 = _rolling_quantile(fund, 0.01, _WINDOW, _MIN_PERIODS)
            dvol_q80 = _rolling_quantile(dvol, 0.80, _WINDOW, _MIN_PERIODS)
            r3_mask = ((fund < fund_q01) & (dvol > dvol_q80)).fillna(False).to_numpy()
            total_score += _W_R3 * r3_mask.astype(np.float64)
            _logger.debug("R3 triggered pct=%.3f", r3_mask.mean())
        else:
            _logger.debug("R3 skipped: %s or %s not found", col_fund, col_dvol)

        # ── R4: OI liquidation cascade AND return below 2*std ────────────────
        col_oi = "macro_oi_delta_24h"
        if col_oi in features_df.columns:
            oi = features_df[col_oi].astype(np.float64)
            oi_q02 = _rolling_quantile(oi, 0.02, _WINDOW, _MIN_PERIODS)
            threshold_r4 = ret_mean - 2.0 * ret_std.fillna(np.inf)
            r4_mask = ((oi < oi_q02) & (ret < threshold_r4)).fillna(False).to_numpy()
            total_score += _W_R4 * r4_mask.astype(np.float64)
            _logger.debug("R4 triggered pct=%.3f", r4_mask.mean())
        else:
            _logger.debug("R4 skipped: %s not found", col_oi)

        # ── R5: Cross-sectional dispersion spike ──────────────────────────────
        col_cs = "macro_cs_dispersion_24h"
        if col_cs in features_df.columns:
            cs = features_df[col_cs].astype(np.float64)
            cs_q95 = _rolling_quantile(cs, 0.95, _WINDOW, _MIN_PERIODS)
            r5_mask = (cs > cs_q95).fillna(False).to_numpy()
            total_score += _W_R5 * r5_mask.astype(np.float64)
            _logger.debug("R5 triggered pct=%.3f", r5_mask.mean())
        else:
            _logger.debug("R5 skipped: %s not found", col_cs)

        # ── R6: Persistent downside vol elevation ─────────────────────────────
        if col_dvol in features_df.columns:
            # dvol already loaded if R3 ran; re-load safely to avoid scope issues
            dvol6 = features_df[col_dvol].astype(np.float64)
            dvol_q90 = _rolling_quantile(dvol6, 0.90, _WINDOW, _MIN_PERIODS)
            r6_mask = (dvol6 > dvol_q90).fillna(False).to_numpy()
            total_score += _W_R6 * r6_mask.astype(np.float64)
            _logger.debug("R6 triggered pct=%.3f", r6_mask.mean())
        else:
            _logger.debug("R6 skipped: %s not found", col_dvol)

        # Clip to [0, 1]
        total_score = np.clip(total_score, 0.0, 1.0)

        return total_score

    def detect(
        self,
        features_df: pd.DataFrame,
        returns_ser: pd.Series,
        threshold: float = 0.40,
    ) -> np.ndarray:
        """Return binary crisis flag array.

        Args:
            features_df: Macro feature DataFrame.
            returns_ser: Log-return series.
            threshold: Score threshold above which crisis is flagged.

        Returns:
            np.ndarray of shape (N,) with dtype bool.

        """
        scores = self.score(features_df, returns_ser)
        return scores >= threshold
