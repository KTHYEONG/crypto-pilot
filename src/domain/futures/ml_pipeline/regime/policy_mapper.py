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
    "soft_damp_gate",
    "hard_damp_gate",
    "near_flat_gate",
    "flat_gate",
    "rebalance_gate",
)


def _clip01(v: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _clip_mult(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.nan_to_num(v, nan=0.0, posinf=hi, neginf=lo), lo, hi)


def _rank_pct01(v: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method="average", pct=True).fillna(0.0).to_numpy(dtype=np.float64)


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
    tail8 = _clip01(hazard_df.get("tail_hazard_8h", hazard_df.get("hmm_tail_risk_8bar", 0.0)))
    sup_soft = _clip01(hazard_df.get("sup_score_soft", hazard_df.get("sup_score_q10_h8", tail8)))
    sup_hard = _clip01(hazard_df.get("sup_score_hard", hazard_df.get("sup_score_q05_h8", tail8)))
    sup_near = _clip01(hazard_df.get("sup_score_near_flat", hazard_df.get("sup_score_q03_h16", sup_hard)))

    gross = 1.0 - 0.30 * p_chop - 0.45 * p_off - 0.55 * realized - 0.15 * ent
    kelly = 1.0 - 0.20 * p_off - 0.35 * tail8 - 0.20 * realized - 0.15 * ent
    long_m = 1.0 - 0.50 * p_off - 0.50 * realized - 0.20 * pre
    short_m = 1.0 + 0.20 * p_off - 0.20 * p_calm

    gross = _clip_mult(gross, float(cfg.get("FUTURES_POLICY_GROSS_CAP_MIN", 0.0)), float(cfg.get("FUTURES_POLICY_GROSS_CAP_MAX", 1.5)))
    kelly = _clip_mult(kelly, float(cfg.get("FUTURES_POLICY_KELLY_MIN", 0.0)), float(cfg.get("FUTURES_POLICY_KELLY_MAX", 1.5)))
    long_m = _clip_mult(long_m, float(cfg.get("FUTURES_POLICY_LONG_MIN", 0.0)), float(cfg.get("FUTURES_POLICY_LONG_MAX", 1.5)))
    short_m = _clip_mult(short_m, float(cfg.get("FUTURES_POLICY_SHORT_MIN", 0.0)), float(cfg.get("FUTURES_POLICY_SHORT_MAX", 1.5)))

    flat_thr = float(
        cfg.get(
            "FUTURES_POLICY_FLAT_REALIZED_CRISIS_THR",
            cfg.get("FUTURES_HMM_REALIZED_CRISIS_FLAT_THRESHOLD", 0.80),
        )
    )
    flat_tail_thr = float(cfg.get("FUTURES_POLICY_FLAT_TAIL8_THR", 0.96))
    flat_mix_tail_w = float(np.clip(cfg.get("FUTURES_POLICY_FLAT_MIX_TAIL8_W", 0.15), 0.0, 1.0))
    flat_extreme_thr = float(
        np.clip(cfg.get("FUTURES_POLICY_FLAT_REALIZED_EXTREME_THR", max(flat_thr + 0.08, 0.90)), 0.0, 1.0)
    )
    damp_thr = float(cfg.get("FUTURES_POLICY_DAMP_PRE_CRISIS_THR", 0.65))
    pre_damp = float(cfg.get("FUTURES_POLICY_PRE_DAMP_FACTOR", 0.85))
    mix_realized_w = float(np.clip(cfg.get("FUTURES_POLICY_DAMP_MIX_REALIZED_W", 0.65), 0.0, 1.0))
    mix_pre_w = float(np.clip(cfg.get("FUTURES_POLICY_DAMP_MIX_PRE_W", 0.20), 0.0, 1.0))
    mix_tail_w = float(np.clip(cfg.get("FUTURES_POLICY_DAMP_MIX_TAIL8_W", 0.15), 0.0, 1.0))
    mix_sum = max(1e-6, mix_realized_w + mix_pre_w + mix_tail_w)
    mix_realized_w /= mix_sum
    mix_pre_w /= mix_sum
    mix_tail_w /= mix_sum

    soft_thr = float(np.clip(cfg.get("FUTURES_POLICY_DEFENSE_SOFT_THR", 0.48), 0.0, 1.0))
    hard_thr = float(np.clip(cfg.get("FUTURES_POLICY_DEFENSE_HARD_THR", 0.66), 0.0, 1.0))
    near_flat_thr = float(np.clip(cfg.get("FUTURES_POLICY_DEFENSE_NEAR_FLAT_THR", 0.84), 0.0, 1.0))
    soft_mult = float(np.clip(cfg.get("FUTURES_POLICY_DEFENSE_SOFT_MULT", 0.82), 0.0, 1.0))
    hard_mult = float(np.clip(cfg.get("FUTURES_POLICY_DEFENSE_HARD_MULT", 0.52), 0.0, 1.0))
    near_flat_mult = float(np.clip(cfg.get("FUTURES_POLICY_DEFENSE_NEAR_FLAT_MULT", 0.24), 0.0, 1.0))
    tail_hard_thr = float(np.clip(cfg.get("FUTURES_POLICY_DEFENSE_TAIL_HARD_THR", 0.92), 0.0, 1.0))
    hard_realized_thr = float(np.clip(cfg.get("FUTURES_POLICY_DEFENSE_HARD_REALIZED_THR", 0.55), 0.0, 1.0))
    near_flat_realized_thr = float(np.clip(cfg.get("FUTURES_POLICY_DEFENSE_NEAR_FLAT_REALIZED_THR", 0.72), 0.0, 1.0))
    hard_tail_rank_thr = float(np.clip(cfg.get("FUTURES_POLICY_DEFENSE_HARD_TAIL_RANK_THR", 0.90), 0.0, 1.0))
    near_flat_tail_rank_thr = float(np.clip(cfg.get("FUTURES_POLICY_DEFENSE_NEAR_FLAT_TAIL_RANK_THR", 0.95), 0.0, 1.0))
    soft_realized_floor = float(np.clip(cfg.get("FUTURES_POLICY_DEFENSE_SOFT_REALIZED_FLOOR", 0.40), 0.0, 1.0))
    soft_tail_thr = float(np.clip(cfg.get("FUTURES_POLICY_DEFENSE_SOFT_TAIL_THR", 0.82), 0.0, 1.0))
    soft_sup_thr = float(np.clip(cfg.get("FUTURES_POLICY_DEFENSE_SOFT_SUP_THR", 0.80), 0.0, 1.0))
    hard_sup_thr = float(np.clip(cfg.get("FUTURES_POLICY_DEFENSE_HARD_SUP_THR", 0.86), 0.0, 1.0))
    near_flat_sup_thr = float(np.clip(cfg.get("FUTURES_POLICY_DEFENSE_NEAR_FLAT_SUP_THR", 0.92), 0.0, 1.0))

    flat_signal = np.maximum(realized, (1.0 - flat_mix_tail_w) * realized + flat_mix_tail_w * tail8)
    hard_realized_mask = realized > flat_extreme_thr
    combo_tail_mask = (tail8 > flat_tail_thr) & (flat_signal > flat_thr) & (realized > (flat_thr * 0.85))
    flat_gate = (hard_realized_mask | combo_tail_mask).astype(np.float64)

    defense_signal = _clip01(mix_realized_w * realized + mix_pre_w * pre + mix_tail_w * tail8)
    tail_rank = _clip01(_rank_pct01(tail8))
    near_flat_gate = (
        (defense_signal >= near_flat_thr)
        & (realized >= near_flat_realized_thr)
        & (tail_rank >= near_flat_tail_rank_thr)
        & (sup_near >= near_flat_sup_thr)
    ).astype(np.float64)
    hard_damp_gate = (
        (
            (defense_signal >= hard_thr)
            & (realized >= hard_realized_thr)
            & ((tail_rank >= hard_tail_rank_thr) | (tail8 >= tail_hard_thr))
            & (sup_hard >= hard_sup_thr)
        )
        | (near_flat_gate > 0.5)
    ).astype(np.float64)
    soft_damp_gate = (
        (defense_signal >= soft_thr)
        | (realized >= soft_realized_floor)
        | (tail8 >= soft_tail_thr)
        | (sup_soft >= soft_sup_thr)
        | (hard_damp_gate > 0.5)
    ).astype(np.float64)
    tier_scale = np.ones_like(defense_signal, dtype=np.float64)
    tier_scale = np.where(soft_damp_gate > 0.5, np.minimum(tier_scale, soft_mult), tier_scale)
    tier_scale = np.where(hard_damp_gate > 0.5, np.minimum(tier_scale, hard_mult), tier_scale)
    tier_scale = np.where(near_flat_gate > 0.5, np.minimum(tier_scale, near_flat_mult), tier_scale)
    gross = gross * tier_scale
    kelly = kelly * tier_scale
    long_m = long_m * tier_scale

    damp_mask = pre > damp_thr
    gross = np.where(damp_mask, gross * pre_damp, gross)
    kelly = np.where(damp_mask, kelly * pre_damp, kelly)
    long_m = np.where(damp_mask, long_m * pre_damp, long_m)

    gross = _clip_mult(gross, float(cfg.get("FUTURES_POLICY_GROSS_CAP_MIN", 0.0)), float(cfg.get("FUTURES_POLICY_GROSS_CAP_MAX", 1.5)))
    kelly = _clip_mult(kelly, float(cfg.get("FUTURES_POLICY_KELLY_MIN", 0.0)), float(cfg.get("FUTURES_POLICY_KELLY_MAX", 1.5)))
    long_m = _clip_mult(long_m, float(cfg.get("FUTURES_POLICY_LONG_MIN", 0.0)), float(cfg.get("FUTURES_POLICY_LONG_MAX", 1.5)))

    # Rebalance gate: suppress churn in highly uncertain/choppy zone.
    rebalance_gate = ((ent > 0.80) & (p_chop > 0.45)).astype(np.float64)

    out = pd.DataFrame(
        {
            "gross_cap_mult": gross,
            "kelly_mult": kelly,
            "long_mult": long_m,
            "short_mult": short_m,
            "soft_damp_gate": soft_damp_gate,
            "hard_damp_gate": hard_damp_gate,
            "near_flat_gate": near_flat_gate,
            "flat_gate": flat_gate,
            "rebalance_gate": rebalance_gate,
        },
        index=idx,
    )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
