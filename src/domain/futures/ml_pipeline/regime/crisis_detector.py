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


def detect_crisis_components(
    features_df: pd.DataFrame,
    returns_ser: pd.Series,
    vol_probs: pd.DataFrame,
    dir_probs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute split crisis scores (pre-crisis / realized-crisis) and blended output.

    Args:
        features_df: DataFrame with SYSTEMIC_HMM_FEATURE_COLUMNS columns.
        returns_ser: Raw returns aligned to features_df.index.
        vol_probs:   Layer 1 posteriors (columns: vol_low, vol_mid, vol_high).
        dir_probs:   Layer 2 posteriors (columns: dir_bull, dir_range, dir_bear).
                     Required for Rule 2. If None, Rule 2 is skipped.

    Returns:
        DataFrame with columns:
          - pre_crisis_score: overheat / transition warning score in [0, 1]
          - realized_crisis_score: realized downside-liquidity stress score in [0, 1]
          - crisis_score: blended score used by backward-compatible hmm_prob_crisis
        Threshold for hard activation guidance: >= 0.35.

    """
    idx = features_df.index
    pre_score = pd.Series(0.0, index=idx, dtype=np.float64)
    realized_score = pd.Series(0.0, index=idx, dtype=np.float64)

    # ── Shared Data ──────────────────────────────────────────────────────────
    ret = returns_ser.reindex(idx).fillna(0.0)
    roll_mu = ret.rolling(168, min_periods=24).mean().fillna(0.0)
    roll_std = (
        ret.rolling(168, min_periods=24)
        .std()
        .fillna(float(ret.std()) if float(ret.std()) > 0 else 0.01)
    )

    # ── Rule 1: Calm Before Storm (pre-crisis, Weight 0.35) ──────────────────
    # High funding deviation during low volatility periods.
    if "macro_funding_mom_24h" in features_df.columns and "vol_low" in vol_probs.columns:
        fund = features_df["macro_funding_mom_24h"].fillna(0.0)
        fund_mu = fund.rolling(168, min_periods=24).mean().fillna(0.0)
        fund_std = fund.rolling(168, min_periods=24).std().replace(0, 0.001).fillna(0.001)
        funding_z = (fund - fund_mu) / fund_std
        
        v_low = vol_probs["vol_low"].reindex(idx).fillna(0.0)
        v_mid = vol_probs["vol_mid"].reindex(idx).fillna(0.0)
        # v_low: Higher threshold for extreme overheating
        pre_score += 0.30 * (v_low * expit(funding_z - 2.0))
        # v_mid: Capture transitional overheating
        pre_score += 0.20 * (v_mid * expit(funding_z - 1.5))

    # ── Rule 2: Structural Bearish Transition (realized, Weight 0.45) ─────────
    # Rollback baseline: multiplicative vol_high * dir_bear interaction.
    if dir_probs is not None and "vol_high" in vol_probs.columns and "dir_bear" in dir_probs.columns:
        v_high = vol_probs["vol_high"].reindex(idx).fillna(0.0)
        d_bear = dir_probs["dir_bear"].reindex(idx).fillna(0.0)
        realized_score += 0.45 * (v_high * d_bear)

    # ── Rule 3: Non-linear Price Shock (realized, Weight 0.20) ───────────────
    # Rapid price drops (Z-score based) passed through sigmoid.
    ret_z = (ret - roll_mu) / roll_std
    realized_score += 0.20 * expit(-ret_z - 2.5)

    # ── Decay & Suppression ──────────────────────────────────────────────────
    # 1. Liquidity Washout Decay
    if "macro_liq_proxy_24h" in features_df.columns:
        liq = features_df["macro_liq_proxy_24h"].fillna(0.0)
        liq_mu = liq.rolling(168, min_periods=24).mean().fillna(0.0)
        liq_std = liq.rolling(168, min_periods=24).std().replace(0, 1.0).fillna(1.0)
        liq_z = (liq - liq_mu) / liq_std
        
        decay_factor = 1.0 - 0.7 * expit(liq_z - 3.0)
        pre_score = pre_score * decay_factor
        realized_score = realized_score * decay_factor

    # 2. Positive Return Penalty
    # If the current bar is positive, we aggressively suppress the crisis score.
    ret_z = (ret - roll_mu) / roll_std
    pre_score = pd.Series(
        np.where(ret_z > 2.0, pre_score * 0.2, np.where(ret > 0, pre_score * 0.7, pre_score)),
        index=idx,
    )
    realized_score = pd.Series(
        np.where(ret_z > 2.0, realized_score * 0.2, np.where(ret > 0, realized_score * 0.7, realized_score)),
        index=idx,
    )

    pre_crisis = pre_score.clip(0.0, 1.0)
    realized_crisis = realized_score.clip(0.0, 1.0)
    # Rollback baseline blend closer to pre-split aggregate crisis scoring.
    crisis_blend = (pre_crisis + realized_crisis).clip(0.0, 1.0)

    result = pd.DataFrame(
        {
            "pre_crisis_score": pre_crisis.to_numpy(dtype=np.float64),
            "realized_crisis_score": realized_crisis.to_numpy(dtype=np.float64),
            "crisis_score": crisis_blend.to_numpy(dtype=np.float64),
        },
        index=idx,
    )
    _logger.debug(
        "🚨 Crisis detector | score>=0.35: %d bars (%.1f%%)",
        int((result["crisis_score"] >= 0.35).sum()),
        100.0 * float((result["crisis_score"] >= 0.35).mean()),
    )
    return result


def detect_crisis(
    features_df: pd.DataFrame,
    returns_ser: pd.Series,
    vol_probs: pd.DataFrame,
    dir_probs: pd.DataFrame | None = None,
) -> pd.Series:
    """Backward-compatible single crisis probability API."""
    comps = detect_crisis_components(features_df, returns_ser, vol_probs, dir_probs)
    return comps["crisis_score"]
