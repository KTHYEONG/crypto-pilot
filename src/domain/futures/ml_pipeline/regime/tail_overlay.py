"""Tail-event supervised overlay for systemic HMM outputs.

Produces a bounded 1~8 bar forward tail-risk probability (`hmm_tail_risk_8bar`)
using a lightweight logistic model with optional isotonic calibration.
Falls back to a deterministic rules score when training data is insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class TailOverlayResult:
    risk: pd.Series
    report: dict[str, float]
    method: str


def _safe_prob01(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def _forward_tail_label(returns: pd.Series, horizon: int, q: float) -> pd.Series:
    idx = returns.index
    fwd_worst = np.full(len(returns), np.nan, dtype=np.float64)
    r = returns.to_numpy(dtype=np.float64)
    for i in range(len(r)):
        j0 = i + 1
        j1 = min(len(r), i + 1 + horizon)
        if j0 >= j1:
            continue
        seg = r[j0:j1]
        if seg.size > 0 and np.isfinite(seg).any():
            fwd_worst[i] = float(np.nanmin(seg))
    ser = pd.Series(fwd_worst, index=idx, dtype=np.float64)
    thr = float(ser.quantile(q)) if ser.notna().any() else np.nan
    if not np.isfinite(thr):
        return pd.Series(np.zeros(len(ser), dtype=np.float64), index=idx)
    return (ser <= thr).astype(np.float64)


def _rule_fallback(
    market_probs: pd.DataFrame,
    market_hmm_feats: pd.DataFrame | None,
) -> np.ndarray:
    p_crisis = market_probs.get("hmm_prob_crisis", 0.0)
    p_bear = market_probs.get("hmm_prob_bear_trend", 0.0)
    p_chop = market_probs.get("hmm_prob_chop", 0.0)
    score = 0.65 * np.asarray(p_crisis, dtype=np.float64) + 0.25 * np.asarray(p_bear, dtype=np.float64) + 0.10 * np.asarray(p_chop, dtype=np.float64)
    if market_hmm_feats is not None and not market_hmm_feats.empty and "macro_vol_24h" in market_hmm_feats.columns:
        vol = market_hmm_feats["macro_vol_24h"].reindex(market_probs["datetime"]).ffill().bfill().to_numpy(dtype=np.float64)
        v_med = float(np.nanmedian(vol)) if np.isfinite(vol).any() else 0.0
        v_iqr = float(np.nanpercentile(vol, 75) - np.nanpercentile(vol, 25)) if np.isfinite(vol).any() else 0.0
        vol_z = (vol - v_med) / max(v_iqr, 1e-6)
        score = score + 0.08 * np.clip(vol_z, -3.0, 3.0)
    return _safe_prob01(score)


def fit_predict_tail_overlay(
    market_probs: pd.DataFrame,
    market_hmm_feats: pd.DataFrame | None,
    market_returns: pd.Series,
    is_end_idx: int,
    cfg: dict[str, Any],
) -> TailOverlayResult:
    horizon = int(cfg.get("FUTURES_HMM_TAIL_OVERLAY_HORIZON", 8))
    q = float(cfg.get("FUTURES_HMM_TAIL_OVERLAY_LABEL_Q", 0.10))
    min_train = int(cfg.get("FUTURES_HMM_TAIL_OVERLAY_MIN_TRAIN", 240))
    min_pos = int(cfg.get("FUTURES_HMM_TAIL_OVERLAY_MIN_POS", 20))
    use_isotonic = bool(cfg.get("FUTURES_HMM_TAIL_OVERLAY_USE_ISOTONIC", True))

    idx = pd.to_datetime(market_probs["datetime"], utc=True)
    feats = pd.DataFrame(index=idx)
    for c in ("hmm_prob_bull_calm", "hmm_prob_bull_vol_up", "hmm_prob_bear_trend", "hmm_prob_chop", "hmm_prob_crisis"):
        if c in market_probs.columns:
            feats[c] = pd.to_numeric(market_probs[c], errors="coerce").astype(np.float64)
    if market_hmm_feats is not None and not market_hmm_feats.empty:
        for c in ("macro_vol_24h", "macro_cost_168h", "macro_trend_24h", "cs_dispersion", "market_breadth"):
            if c in market_hmm_feats.columns:
                feats[c] = pd.to_numeric(market_hmm_feats[c].reindex(idx), errors="coerce").astype(np.float64)
    feats = feats.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    y = _forward_tail_label(market_returns.reindex(idx).fillna(0.0), horizon=max(1, horizon), q=min(max(q, 0.01), 0.40))

    n = len(feats)
    is_end_idx = int(max(0, min(is_end_idx, n)))
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:is_end_idx] = True
    train_mask &= y.notna().to_numpy()
    y_train = y.to_numpy(dtype=np.float64)[train_mask]
    pos_n = int(np.sum(y_train > 0.5))

    method = "fallback_rule"
    if int(np.sum(train_mask)) >= min_train and pos_n >= min_pos and pos_n < int(np.sum(train_mask)) - 5:
        try:
            x_tr = feats.to_numpy(dtype=np.float64)[train_mask]
            x_all = feats.to_numpy(dtype=np.float64)
            lr = LogisticRegression(
                penalty="l2",
                C=float(cfg.get("FUTURES_HMM_TAIL_OVERLAY_LR_C", 0.8)),
                solver="lbfgs",
                max_iter=400,
                class_weight="balanced",
                random_state=int(cfg.get("seed", 42)),
            )
            lr.fit(x_tr, y_train.astype(np.int32))
            p_all = lr.predict_proba(x_all)[:, 1]
            method = "logistic"
            if use_isotonic:
                p_tr = lr.predict_proba(x_tr)[:, 1]
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(p_tr, y_train)
                p_all = iso.predict(p_all)
                method = "logistic+isotonic"
            pred = _safe_prob01(p_all)
        except Exception:
            pred = _rule_fallback(market_probs, market_hmm_feats)
    else:
        pred = _rule_fallback(market_probs, market_hmm_feats)

    report = {
        "hmm_tail_overlay_train_n": float(int(np.sum(train_mask))),
        "hmm_tail_overlay_train_pos_n": float(pos_n),
        "hmm_tail_overlay_mean": float(np.nanmean(pred) * 100.0),
        "hmm_tail_overlay_p95": float(np.nanpercentile(pred, 95) * 100.0),
    }
    if y.notna().any():
        y_all = y.to_numpy(dtype=np.float64)
        top_thr = float(np.nanpercentile(pred, 90))
        top_mask = pred >= top_thr
        if np.any(top_mask):
            report["hmm_tail_overlay_top_decile_hit_rate"] = float(np.mean(y_all[top_mask] > 0.5) * 100.0)

    return TailOverlayResult(
        risk=pd.Series(_safe_prob01(pred), index=idx, name="hmm_tail_risk_8bar"),
        report=report,
        method=method,
    )
