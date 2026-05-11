"""Layer 3: Rules-Based Crisis Detector.

Architecture:
    Crash events are sparse point processes, not Markov states.
    Soft-scoring via additive rule triggers, clipped to [0, 1].

Rules:
    1. Calm Before Storm (Weight 0.35): funding_z > 1.5 while in vol_low.
    2. Structural Bearish Transition (Weight 0.45): vol_high AND dir_bear.
    3. Non-linear Price Shock (Weight 0.20): sigmoid(-ret_z - 2.5).
    4. Liquidity Washout Decay: Massive liq_z suppresses score.

Threshold for hard activation: >= 0.35 (tightened from 0.25).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.special import expit

_logger = logging.getLogger(__name__)


def detect_crisis(
    features_df: pd.DataFrame,
    returns_ser: pd.Series,
    vol_probs: pd.DataFrame,
    dir_probs: pd.DataFrame | None = None,
) -> pd.Series:
    """Compute soft crisis probability for each bar using continuous scoring (v9.2.0).

    Args:
        features_df: DataFrame with SYSTEMIC_HMM_FEATURE_COLUMNS columns.
        returns_ser: Raw returns aligned to features_df.index.
        vol_probs:   Layer 1 posteriors (columns: vol_low, vol_mid, vol_high).
        dir_probs:   Layer 2 posteriors (columns: dir_bull, dir_range, dir_bear).
                     Required for Rule 2. If None, Rule 2 is skipped.

    Returns:
        pd.Series (same index as features_df) with values in [0, 1].
        Threshold for hard activation: >= 0.35.

    """
    idx = features_df.index
    score = pd.Series(0.0, index=idx, dtype=np.float64)

    # ── Shared Data ──────────────────────────────────────────────────────────
    ret = returns_ser.reindex(idx).fillna(0.0)
    roll_mu = ret.rolling(168, min_periods=24).mean().fillna(0.0)
    roll_std = (
        ret.rolling(168, min_periods=24)
        .std()
        .fillna(float(ret.std()) if float(ret.std()) > 0 else 0.01)
    )

    # ── Rule 1: Calm Before Storm (Weight 0.35) ──────────────────────────────
    # High funding deviation during low volatility periods.
    if "macro_funding_mom_24h" in features_df.columns and "vol_low" in vol_probs.columns:
        fund = features_df["macro_funding_mom_24h"].fillna(0.0)
        fund_mu = fund.rolling(168, min_periods=24).mean().fillna(0.0)
        fund_std = fund.rolling(168, min_periods=24).std().replace(0, 0.001).fillna(0.001)
        funding_z = (fund - fund_mu) / fund_std
        
        v_low = vol_probs["vol_low"].reindex(idx).fillna(0.0)
        score += 0.35 * (v_low * expit(funding_z - 1.5))

    # ── Rule 2: Structural Bearish Transition (Weight 0.45) ───────────────────
    # Combination of high volatility and bearish direction.
    if dir_probs is not None and "vol_high" in vol_probs.columns and "dir_bear" in dir_probs.columns:
        v_high = vol_probs["vol_high"].reindex(idx).fillna(0.0)
        d_bear = dir_probs["dir_bear"].reindex(idx).fillna(0.0)
        score += 0.45 * (v_high * d_bear)

    # ── Rule 3: Non-linear Price Shock (Weight 0.20) ─────────────────────────
    # Rapid price drops (Z-score based) passed through sigmoid.
    ret_z = (ret - roll_mu) / roll_std
    score += 0.20 * expit(-ret_z - 2.5)

    # ── Decay & Suppression ──────────────────────────────────────────────────
    # 1. Liquidity Washout Decay
    if "macro_liq_proxy_24h" in features_df.columns:
        liq = features_df["macro_liq_proxy_24h"].fillna(0.0)
        liq_mu = liq.rolling(168, min_periods=24).mean().fillna(0.0)
        liq_std = liq.rolling(168, min_periods=24).std().replace(0, 1.0).fillna(1.0)
        liq_z = (liq - liq_mu) / liq_std
        
        decay_factor = 1.0 - 0.7 * expit(liq_z - 3.0)
        score = score * decay_factor

    # 2. Positive Return Penalty
    # If the current bar is positive, we aggressively suppress the crisis score.
    score = pd.Series(np.where(ret > 0, score * 0.5, score), index=idx)

    result = score.clip(0.0, 1.0)
    _logger.info(
        "Crisis detector (v9.2.0): %d bars with score>=0.35 (%.2f%% of total).",
        int((result >= 0.35).sum()),
        100.0 * float((result >= 0.35).mean()),
    )
    return result
