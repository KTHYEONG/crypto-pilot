"""Layer 3: Rules-Based Crisis Detector.

Architecture:
    Crash events are sparse point processes, not Markov states.
    Soft-scoring via additive rule triggers, clipped to [0, 1].

Rules:
    1. return < rolling_mean - 4*rolling_std (4-sigma event)            → +0.5
    2. macro_liq_proxy_24h > p99.5 (lifetime)                           → +0.25
    3. macro_funding_mom_24h < p0.5 AND Layer1 HIGH_VOL > 0.7           → +0.3
    4. vol_high > 0.8 AND return < rolling_mean - 2*rolling_std         → +0.15
    5. vol_high > 0.6 AND dir_bear > 0.5  (NEW: early bear+vol signal)  → +0.25
    6. return < rolling_mean - 3*rolling_std (3-sigma, broader net)     → +0.35

Threshold for hard activation: >= 0.25 (relaxed from 0.6).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


def detect_crisis(
    features_df: pd.DataFrame,
    returns_ser: pd.Series,
    vol_probs: pd.DataFrame,
    dir_probs: pd.DataFrame | None = None,
) -> pd.Series:
    """Compute soft crisis probability for each bar.

    Args:
        features_df: DataFrame with SYSTEMIC_HMM_FEATURE_COLUMNS columns.
        returns_ser: Raw returns aligned to features_df.index.
        vol_probs:   Layer 1 posteriors (columns: vol_low, vol_mid, vol_high).
        dir_probs:   Layer 2 posteriors (columns: dir_bull, dir_range, dir_bear).
                     Required for Rule 5. If None, Rule 5 is skipped.

    Returns:
        pd.Series (same index as features_df) with values in [0, 1].
        Threshold for hard activation: >= 0.25.

    """
    idx = features_df.index
    score = pd.Series(0.0, index=idx, dtype=np.float64)

    # ── Rolling stats (shared by Rule 1, 4, 6) ──────────────────────────────
    ret = returns_ser.reindex(idx).fillna(0.0)
    roll_mu = ret.rolling(168, min_periods=24).mean().fillna(0.0)
    roll_std = (
        ret.rolling(168, min_periods=24)
        .std()
        .fillna(float(ret.std()) if float(ret.std()) > 0 else 0.01)
    )

    # ── Rule 1: 4-sigma return event ─────────────────────────────────────────
    rule1 = (ret < roll_mu - 4.0 * roll_std).astype(np.float64)
    score += 0.5 * rule1
    _logger.debug("Crisis Rule-1 (4-sigma): %d bars triggered.", int(rule1.sum()))

    # ── Rule 2: extreme liquidation proxy ────────────────────────────────────
    if "macro_liq_proxy_24h" in features_df.columns:
        liq = features_df["macro_liq_proxy_24h"].fillna(0.0)
        p995 = float(liq.quantile(0.995))
        rule2 = (liq > p995).astype(np.float64)
        score += 0.25 * rule2
        _logger.debug("Crisis Rule-2 (liq p99.5): %d bars triggered.", int(rule2.sum()))

    # ── Rule 3: funding extreme + HIGH_VOL posterior ──────────────────────────
    if "macro_funding_mom_24h" in features_df.columns and "vol_high" in vol_probs.columns:
        fund = features_df["macro_funding_mom_24h"].fillna(0.0)
        p005 = float(fund.quantile(0.005))
        vol_high_flag = vol_probs["vol_high"].reindex(idx).fillna(0.0) > 0.7
        rule3 = ((fund < p005) & vol_high_flag).astype(np.float64)
        score += 0.3 * rule3
        _logger.debug("Crisis Rule-3 (fund+volhigh): %d bars triggered.", int(rule3.sum()))

    # ── Rule 4: HIGH_VOL + 2-sigma ────────────────────────────────────────────
    if "vol_high" in vol_probs.columns:
        vol_high_ser = vol_probs["vol_high"].reindex(idx).fillna(0.0)
        rule4 = (
            (vol_high_ser > 0.8) & (ret < roll_mu - 2.0 * roll_std)
        ).astype(np.float64)
        score += 0.15 * rule4
        _logger.debug("Crisis Rule-4 (vol>0.8+2sigma): %d bars triggered.", int(rule4.sum()))

    # ── Rule 5 (NEW): vol_high > 0.6 AND dir_bear > 0.5 ─────────────────────
    has_rule5 = (
        dir_probs is not None
        and "vol_high" in vol_probs.columns
        and "dir_bear" in dir_probs.columns
    )
    if has_rule5:
        vol_high_ser = vol_probs["vol_high"].reindex(idx).fillna(0.0)
        p_bear_ser = dir_probs["dir_bear"].reindex(idx).fillna(0.0)
        rule5 = ((vol_high_ser > 0.6) & (p_bear_ser > 0.5)).astype(np.float64)
        score += 0.25 * rule5
        _logger.debug("Crisis Rule-5 (vol>0.6+bear>0.5): %d bars triggered.", int(rule5.sum()))

    # ── Rule 6 (NEW): 3-sigma return event (broader net than Rule 1) ─────────
    rule6 = (ret < roll_mu - 3.0 * roll_std).astype(np.float64)
    score += 0.35 * rule6
    _logger.debug("Crisis Rule-6 (3-sigma): %d bars triggered.", int(rule6.sum()))

    result = score.clip(0.0, 1.0)
    _logger.info(
        "Crisis detector: %d bars with score>=0.25 (%.2f%% of total).",
        int((result >= 0.25).sum()),
        100.0 * float((result >= 0.25).mean()),
    )
    return result
