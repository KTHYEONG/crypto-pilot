"""Tail/crisis hazard overlay from 1h features + regime posterior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


HAZARD_COLUMNS: tuple[str, ...] = (
    "pre_crisis_hazard",
    "realized_crisis_hazard",
    "tail_hazard_4h",
    "tail_hazard_8h",
    "tail_hazard_24h",
)


@dataclass
class TailHazardOverlayResult:
    hazards: pd.DataFrame
    report: dict[str, float]


@dataclass
class TailOverlayResult:
    risk: pd.Series
    report: dict[str, float]
    method: str


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(np.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


def _safe01(v: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _zscore_robust(s: pd.Series, win: int = 168) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").astype(np.float64)
    med = x.rolling(win, min_periods=max(12, win // 8)).median()
    q75 = x.rolling(win, min_periods=max(12, win // 8)).quantile(0.75)
    q25 = x.rolling(win, min_periods=max(12, win // 8)).quantile(0.25)
    iqr = (q75 - q25).replace(0.0, np.nan)
    z = ((x - med) / iqr).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z.to_numpy(dtype=np.float64)


def compute_tail_hazard_overlay(
    features_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    returns_ser: pd.Series,
    cfg: dict[str, Any] | None = None,
) -> TailHazardOverlayResult:
    cfg = cfg or {}
    idx = regime_df.index
    f = features_df.reindex(idx).copy()
    r = pd.to_numeric(returns_ser.reindex(idx), errors="coerce").fillna(0.0).astype(np.float64)

    p_calm = pd.to_numeric(regime_df.get("regime_prob_risk_on_calm", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
    p_vol = pd.to_numeric(regime_df.get("regime_prob_risk_on_volatile", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
    p_off = pd.to_numeric(regime_df.get("regime_prob_risk_off_trend", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
    p_chop = pd.to_numeric(regime_df.get("regime_prob_chop_liquidity_thin", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
    ent = pd.to_numeric(regime_df.get("regime_entropy", 0.5), errors="coerce").fillna(0.5).to_numpy(dtype=np.float64)

    rv24 = _zscore_robust(f.get("rv_24h", r.abs().rolling(24, min_periods=4).std().fillna(0.0)))
    down24 = _zscore_robust(f.get("downside_vol_24h", (-r).clip(lower=0.0).rolling(24, min_periods=4).std().fillna(0.0)))
    liq24 = _zscore_robust(f.get("liq_proxy_24h", f.get("liq_proxy_8h", pd.Series(0.0, index=idx))))
    oi24 = _zscore_robust(f.get("oi_delta_24h", pd.Series(0.0, index=idx)))
    jump = _zscore_robust((-r).clip(lower=0.0))

    pre_logit = (
        -1.8
        + 1.25 * p_off
        + 0.85 * p_chop
        + 0.35 * p_vol
        + 0.45 * ent
        + 0.30 * rv24
        + 0.25 * oi24
        + 0.30 * liq24
    )
    realized_logit = (
        -2.3
        + 1.60 * p_off
        + 0.70 * p_chop
        + 0.55 * down24
        + 0.50 * jump
        + 0.35 * liq24
        + 0.25 * ent
    )

    pre = _safe01(_sigmoid(pre_logit))
    realized = _safe01(_sigmoid(realized_logit))
    tail4 = _safe01(0.55 * realized + 0.35 * pre + 0.10 * _safe01(_sigmoid(-1.0 + 1.0 * down24 + 0.5 * p_off)))
    tail8 = _safe01(0.45 * realized + 0.40 * pre + 0.15 * _safe01(_sigmoid(-0.9 + 0.8 * down24 + 0.6 * p_off + 0.3 * ent)))
    tail24 = _safe01(0.30 * realized + 0.45 * pre + 0.25 * _safe01(_sigmoid(-0.6 + 0.7 * p_off + 0.4 * p_chop + 0.3 * liq24 + 0.3 * ent)))

    hazards = pd.DataFrame(
        {
            "pre_crisis_hazard": pre,
            "realized_crisis_hazard": realized,
            "tail_hazard_4h": tail4,
            "tail_hazard_8h": tail8,
            "tail_hazard_24h": tail24,
        },
        index=idx,
    )
    hazards = hazards.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)

    report = {
        "pre_crisis_hazard_mean": float(hazards["pre_crisis_hazard"].mean()),
        "realized_crisis_hazard_mean": float(hazards["realized_crisis_hazard"].mean()),
        "tail_hazard_8h_p95": float(hazards["tail_hazard_8h"].quantile(0.95)),
    }
    return TailHazardOverlayResult(hazards=hazards, report=report)


def fit_predict_tail_overlay(
    market_probs: pd.DataFrame,
    market_hmm_feats: pd.DataFrame | None,
    market_returns: pd.Series,
    is_end_idx: int,
    cfg: dict[str, Any],
) -> TailOverlayResult:
    """Backward-compatible API returning 8h tail risk series."""
    del is_end_idx  # New overlay is deterministic and does not refit on OOS.
    _ = market_hmm_feats  # Reserved for future feature expansion.
    idx = pd.to_datetime(market_probs["datetime"], utc=True) if "datetime" in market_probs.columns else market_probs.index
    regime_df = pd.DataFrame(index=idx)
    regime_df["regime_prob_risk_on_calm"] = pd.to_numeric(market_probs.get("hmm_prob_bull_calm", 0.25), errors="coerce").fillna(0.25)
    regime_df["regime_prob_risk_on_volatile"] = pd.to_numeric(market_probs.get("hmm_prob_bull_vol_up", 0.25), errors="coerce").fillna(0.25)
    regime_df["regime_prob_risk_off_trend"] = pd.to_numeric(market_probs.get("hmm_prob_bear_trend", 0.25), errors="coerce").fillna(0.25)
    regime_df["regime_prob_chop_liquidity_thin"] = pd.to_numeric(market_probs.get("hmm_prob_chop", 0.25), errors="coerce").fillna(0.25)
    regime_df["regime_entropy"] = 0.0
    features_df = market_hmm_feats if market_hmm_feats is not None else pd.DataFrame(index=idx)
    res = compute_tail_hazard_overlay(
        features_df=features_df,
        regime_df=regime_df,
        returns_ser=market_returns.reindex(idx).fillna(0.0),
        cfg=cfg,
    )
    risk = res.hazards["tail_hazard_8h"].rename("hmm_tail_risk_8bar")
    return TailOverlayResult(risk=risk, report=res.report, method="hazard_overlay")
