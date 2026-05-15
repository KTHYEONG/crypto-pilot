"""Policy mapper: posterior + hazards -> execution controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

POLICY_COLUMNS: tuple[str, ...] = (
    "gross_cap_mult",
    "kelly_mult",
    "long_mult",
    "short_mult",
    "flat_gate",
    "rebalance_gate",
)


def _clip01(v: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _clip_mult(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.nan_to_num(v, nan=0.0, posinf=hi, neginf=lo), lo, hi)


def map_policy_controls(
    regime_df: pd.DataFrame,
    hazard_df: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = cfg or {}
    idx = regime_df.index

    p_calm = _clip01(regime_df.get("regime_prob_risk_on_calm", 0.25))
    p_off = _clip01(regime_df.get("regime_prob_risk_off_trend", 0.25))
    p_chop = _clip01(regime_df.get("regime_prob_chop_liquidity_thin", 0.25))
    ent = _clip01(regime_df.get("regime_entropy", 0.5))

    pre = _clip01(hazard_df.get("pre_crisis_hazard", 0.0))
    realized = _clip01(hazard_df.get("realized_crisis_hazard", 0.0))
    tail8 = _clip01(hazard_df.get("tail_hazard_8h", 0.0))

    gross = 1.0 - 0.35 * p_chop - 0.55 * p_off - 0.85 * realized - 0.20 * ent
    kelly = 1.0 - 0.25 * p_off - 0.60 * tail8 - 0.20 * ent
    long_m = 1.0 - 0.60 * p_off - 0.80 * realized
    short_m = 1.0 + 0.20 * p_off - 0.20 * p_calm

    gross = _clip_mult(gross, float(cfg.get("FUTURES_POLICY_GROSS_CAP_MIN", 0.0)), float(cfg.get("FUTURES_POLICY_GROSS_CAP_MAX", 1.5)))
    kelly = _clip_mult(kelly, float(cfg.get("FUTURES_POLICY_KELLY_MIN", 0.0)), float(cfg.get("FUTURES_POLICY_KELLY_MAX", 1.5)))
    long_m = _clip_mult(long_m, float(cfg.get("FUTURES_POLICY_LONG_MIN", 0.0)), float(cfg.get("FUTURES_POLICY_LONG_MAX", 1.5)))
    short_m = _clip_mult(short_m, float(cfg.get("FUTURES_POLICY_SHORT_MIN", 0.0)), float(cfg.get("FUTURES_POLICY_SHORT_MAX", 1.5)))

    flat_thr = float(cfg.get("FUTURES_POLICY_FLAT_REALIZED_CRISIS_THR", 0.80))
    damp_thr = float(cfg.get("FUTURES_POLICY_DAMP_PRE_CRISIS_THR", 0.65))
    pre_damp = float(cfg.get("FUTURES_POLICY_PRE_DAMP_FACTOR", 0.85))

    flat_gate = (realized > flat_thr).astype(np.float64)
    damp_mask = pre > damp_thr
    gross = np.where(damp_mask, gross * pre_damp, gross)
    kelly = np.where(damp_mask, kelly * pre_damp, kelly)

    # Rebalance gate: suppress churn in highly uncertain/choppy zone.
    rebalance_gate = ((ent > 0.80) & (p_chop > 0.45)).astype(np.float64)

    out = pd.DataFrame(
        {
            "gross_cap_mult": gross,
            "kelly_mult": kelly,
            "long_mult": long_m,
            "short_mult": short_m,
            "flat_gate": flat_gate,
            "rebalance_gate": rebalance_gate,
        },
        index=idx,
    )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
