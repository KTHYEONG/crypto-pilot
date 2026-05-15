"""Tail/crisis hazard overlay from 1h features + regime posterior."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

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
    supervised_score: np.ndarray | None = None,
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
    # supervised_score 주입: 0.5 * sigmoid(realized_logit) + 0.5 * supervised_score
    if supervised_score is not None:
        sup = np.clip(np.nan_to_num(supervised_score, nan=0.0), 0.0, 1.0)
        realized = _safe01(0.5 * _sigmoid(realized_logit) + 0.5 * sup)
    else:
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


def _fit_supervised_tail_score(
    X: np.ndarray,
    r_ser: pd.Series,
    idx: pd.Index,
    is_end_idx: int,
) -> np.ndarray | None:
    """IS logistic+isotonic supervised score; causal — no look-ahead.

    Args:
        X: Feature matrix (n, 7), full timeline.
        r_ser: Returns series aligned to idx.
        idx: Full timeline index.
        is_end_idx: Exclusive end of IS window.

    Returns:
        Calibrated probability array of shape (n,) or None on failure.
    """
    try:
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    except ImportError:
        _logger.warning("scikit-learn not available; using hardcoded fallback")
        return None

    try:
        n = len(X)

        # 8-bar forward worst return (causal label on IS)
        fwd_worst = np.full(n, np.inf, dtype=np.float64)
        for k in range(1, 9):
            shifted = r_ser.shift(-k).to_numpy(dtype=np.float64)
            fwd_worst = np.minimum(fwd_worst, shifted)
        fwd_ser = pd.Series(fwd_worst, index=idx).replace([np.inf, -np.inf], np.nan)

        fwd_is = fwd_ser.iloc[:is_end_idx]
        valid_is = fwd_is.dropna()
        if len(valid_is) < 20:
            raise ValueError(f"Too few valid IS returns: {len(valid_is)}")
        q10_is = float(valid_is.quantile(0.10))
        if not np.isfinite(q10_is):
            raise ValueError("q10 not finite")

        y_is = (fwd_ser.iloc[:is_end_idx].fillna(0.0).to_numpy() <= q10_is).astype(int)
        if y_is.sum() < 10:
            raise ValueError(f"Too few positive labels: {y_is.sum()}")

        X_is = X[:is_end_idx]
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=500, random_state=42)
        clf.fit(X_is, y_is)

        prob_is = clf.predict_proba(X_is)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(prob_is, y_is.astype(np.float64))

        prob_all = clf.predict_proba(X)[:, 1]
        score: np.ndarray = iso.transform(prob_all).astype(np.float64)
        _logger.info(
            "fit_predict_tail_overlay Step3 | is_end=%d | n=%d | score_mean=%.3f | pos_rate=%.3f",
            is_end_idx,
            n,
            float(score.mean()),
            float(y_is.mean()),
        )
        return score
    except Exception as exc:
        _logger.warning("Step3 supervised calibration failed: %s; using hardcoded", exc)
        return None


def fit_predict_tail_overlay(
    market_probs: pd.DataFrame,
    market_hmm_feats: pd.DataFrame | None,
    market_returns: pd.Series,
    is_end_idx: int,
    cfg: dict[str, Any],
) -> TailOverlayResult:
    """IS supervised logistic+isotonic calibration on tail hazard.

    When is_end_idx > 50, learns a logistic+isotonic model on IS labels
    (fwd_worst_8bar <= q10_IS) and blends its score into realized_crisis_hazard.
    Falls back to hardcoded logits on any failure.
    """
    idx = pd.to_datetime(market_probs["datetime"], utc=True) if "datetime" in market_probs.columns else market_probs.index

    # ── feature matrix X (7 features) ──────────────────────────────────────
    p_off_s = pd.to_numeric(market_probs.get("hmm_prob_bear_trend", 0.25), errors="coerce").fillna(0.25)
    p_chop_s = pd.to_numeric(market_probs.get("hmm_prob_chop", 0.25), errors="coerce").fillna(0.25)
    ent_s = pd.to_numeric(market_probs.get("regime_entropy", 0.5), errors="coerce").fillna(0.5)

    r_ser = market_returns.reindex(idx).fillna(0.0).astype(np.float64)
    r_np = r_ser.to_numpy(dtype=np.float64)

    feats_df = market_hmm_feats.reindex(idx) if market_hmm_feats is not None else pd.DataFrame(index=idx)

    rv24_z = _zscore_robust(pd.Series(np.abs(r_np), index=idx).rolling(24, min_periods=4).std().fillna(0.0))
    down24_z = _zscore_robust(pd.Series(np.clip(-r_np, 0.0, None), index=idx).rolling(24, min_periods=4).std().fillna(0.0))
    if "macro_liq_proxy_24h" in feats_df.columns:
        liq24_z = _zscore_robust(pd.to_numeric(feats_df["macro_liq_proxy_24h"], errors="coerce").fillna(0.0))
    else:
        liq24_z = np.zeros(len(idx), dtype=np.float64)
    jump_z = _zscore_robust(pd.Series(np.clip(-r_np, 0.0, None), index=idx))

    X = np.column_stack([
        p_off_s.to_numpy(dtype=np.float64),
        p_chop_s.to_numpy(dtype=np.float64),
        ent_s.to_numpy(dtype=np.float64),
        rv24_z,
        down24_z,
        liq24_z,
        jump_z,
    ]).astype(np.float64)

    # ── supervised score (IS only, causal) ─────────────────────────────────
    supervised_score: np.ndarray | None = None
    method = "hardcoded"
    if is_end_idx > 50:
        supervised_score = _fit_supervised_tail_score(X, r_ser, idx, is_end_idx)
        if supervised_score is not None:
            method = "logistic+isotonic"

    # ── regime_df for compute_tail_hazard_overlay ───────────────────────────
    regime_df = pd.DataFrame(index=idx)
    regime_df["regime_prob_risk_on_calm"] = pd.to_numeric(market_probs.get("hmm_prob_bull_calm", 0.25), errors="coerce").fillna(0.25)
    regime_df["regime_prob_risk_on_volatile"] = pd.to_numeric(market_probs.get("hmm_prob_bull_vol_up", 0.25), errors="coerce").fillna(0.25)
    regime_df["regime_prob_risk_off_trend"] = p_off_s
    regime_df["regime_prob_chop_liquidity_thin"] = p_chop_s
    regime_df["regime_entropy"] = ent_s

    res = compute_tail_hazard_overlay(
        features_df=feats_df,
        regime_df=regime_df,
        returns_ser=r_ser,
        cfg=cfg,
        supervised_score=supervised_score,
    )
    risk = res.hazards["tail_hazard_8h"].rename("hmm_tail_risk_8bar")
    # method 문자열은 float 변환 불가이므로 report에서 제외, method 필드로만 전달
    return TailOverlayResult(risk=risk, report=res.report, method=method)
