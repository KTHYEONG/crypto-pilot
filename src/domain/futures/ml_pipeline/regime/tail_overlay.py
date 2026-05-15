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


def _rank_pct01(v: np.ndarray) -> np.ndarray:
    s = pd.Series(np.nan_to_num(v, nan=0.0), copy=False)
    return s.rank(method="average", pct=True).fillna(0.0).to_numpy(dtype=np.float64)


def _as_score_bundle(supervised_score: np.ndarray | dict[str, np.ndarray] | None, n: int) -> dict[str, np.ndarray]:
    if supervised_score is None:
        return {}
    if isinstance(supervised_score, dict):
        out: dict[str, np.ndarray] = {}
        for k, v in supervised_score.items():
            arr = np.asarray(v, dtype=np.float64)
            if arr.shape[0] == n:
                out[k] = _safe01(arr)
        return out
    arr = np.asarray(supervised_score, dtype=np.float64)
    if arr.shape[0] != n:
        return {}
    s = _safe01(arr)
    return {
        "sup_score_q10_h8": s,
        "sup_score_soft": s,
    }


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
    supervised_score: np.ndarray | dict[str, np.ndarray] | None = None,
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
        -2.8
        + 2.20 * (p_off * down24)
        + 0.60 * p_chop
        + 0.40 * jump
        + 0.30 * liq24
    )

    pre = _safe01(_sigmoid(pre_logit))
    sup_bundle = _as_score_bundle(supervised_score, len(idx))
    sup_main = sup_bundle.get("sup_score_q10_h8", np.zeros(len(idx), dtype=np.float64))
    # supervised_score 주입: 0.5 * sigmoid(realized_logit) + 0.5 * supervised main score
    if sup_bundle:
        realized = _safe01(0.5 * _sigmoid(realized_logit) + 0.5 * sup_main)
    else:
        realized = _safe01(_sigmoid(realized_logit))
    tail4 = _safe01(0.55 * realized + 0.35 * pre + 0.10 * _safe01(_sigmoid(-1.0 + 1.0 * down24 + 0.5 * p_off)))
    tail8_struct = _safe01(0.45 * realized + 0.40 * pre + 0.15 * _safe01(_sigmoid(-0.9 + 0.8 * down24 + 0.6 * p_off + 0.3 * ent)))
    tail8 = tail8_struct.copy()
    tail24 = _safe01(0.30 * realized + 0.45 * pre + 0.25 * _safe01(_sigmoid(-0.6 + 0.7 * p_off + 0.4 * p_chop + 0.3 * liq24 + 0.3 * ent)))

    # Step2: strengthen hazard path for top supervised-score zones (posterior purity preserved).
    if sup_bundle:
        sup = sup_main
        top_q = float(cfg.get("FUTURES_HMM_STEP2_SUPERVISED_TOP_Q", 0.90))
        top_q = float(np.clip(top_q, 0.0, 1.0))
        top_thr = float(np.quantile(sup, top_q))
        top_scale_den = max(1e-6, 1.0 - top_thr)
        top_scale = np.where(sup >= top_thr, (sup - top_thr) / top_scale_den, 0.0)
        top_scale = _safe01(top_scale)

        pre_boost = float(np.clip(cfg.get("FUTURES_HMM_STEP2_PRE_HAZARD_BOOST", 0.30), 0.0, 1.0))
        realized_boost = float(np.clip(cfg.get("FUTURES_HMM_STEP2_REALIZED_HAZARD_BOOST", 0.45), 0.0, 1.0))
        tail8_boost = float(np.clip(cfg.get("FUTURES_HMM_STEP2_TAIL8_HAZARD_BOOST", 0.35), 0.0, 1.0))

        pre_lift = pre + pre_boost * top_scale * (1.0 - pre)
        pre = _safe01(np.maximum(pre, pre_lift))

        realized_lift = realized + realized_boost * top_scale * (1.0 - realized)
        realized = _safe01(np.maximum(realized, realized_lift))

        sup_rank = _safe01(_rank_pct01(sup))
        sup_rank_pow = float(np.clip(cfg.get("FUTURES_HMM_STEP2_TAIL8_SUP_RANK_POW", 1.15), 0.5, 3.0))
        sup_rank = _safe01(sup_rank**sup_rank_pow)
        crash_q05 = float(np.clip(cfg.get("FUTURES_HMM_STEP2_SUP_CRASH_Q05", 0.95), 0.50, 0.999))
        crash_q03 = float(np.clip(cfg.get("FUTURES_HMM_STEP2_SUP_CRASH_Q03", 0.97), 0.50, 0.999))
        q05_thr = float(np.quantile(sup_rank, crash_q05))
        q03_thr = float(np.quantile(sup_rank, crash_q03))
        crash05 = _safe01(np.clip((sup_rank - q05_thr) / max(1e-6, 1.0 - q05_thr), 0.0, 1.0))
        crash03 = _safe01(np.clip((sup_rank - q03_thr) / max(1e-6, 1.0 - q03_thr), 0.0, 1.0))
        w_rank = float(np.clip(cfg.get("FUTURES_HMM_STEP2_TAIL8_RANK_BLEND_W", 0.55), 0.0, 1.0))
        w_sup = float(np.clip(cfg.get("FUTURES_HMM_STEP2_TAIL8_SUP_BLEND_W", 0.20), 0.0, 1.0))
        w_real = float(np.clip(cfg.get("FUTURES_HMM_STEP2_TAIL8_REALIZED_BLEND_W", 0.15), 0.0, 1.0))
        w_pre = float(np.clip(cfg.get("FUTURES_HMM_STEP2_TAIL8_PRE_BLEND_W", 0.10), 0.0, 1.0))
        w_struct = float(np.clip(cfg.get("FUTURES_HMM_STEP2_TAIL8_STRUCT_BLEND_W", 0.15), 0.0, 1.0))
        w_crash05 = float(np.clip(cfg.get("FUTURES_HMM_STEP2_TAIL8_CRASH05_BLEND_W", 0.10), 0.0, 1.0))
        w_crash03 = float(np.clip(cfg.get("FUTURES_HMM_STEP2_TAIL8_CRASH03_BLEND_W", 0.10), 0.0, 1.0))
        blend_sum = max(1e-6, w_rank + w_sup + w_real + w_pre + w_struct + w_crash05 + w_crash03)
        w_rank /= blend_sum
        w_sup /= blend_sum
        w_real /= blend_sum
        w_pre /= blend_sum
        w_struct /= blend_sum
        w_crash05 /= blend_sum
        w_crash03 /= blend_sum
        tail8_target = (
            w_rank * sup_rank
            + w_sup * sup
            + w_real * realized
            + w_pre * pre
            + w_struct * tail8_struct
            + w_crash05 * crash05
            + w_crash03 * crash03
        )
        sup_soft = _safe01(sup_bundle.get("sup_score_soft", sup))
        sup_hard = _safe01(sup_bundle.get("sup_score_hard", sup))
        sup_near_flat = _safe01(sup_bundle.get("sup_score_near_flat", sup))
        w_soft = float(np.clip(cfg.get("FUTURES_HMM_STEP2_TAIL8_SOFT_BLEND_W", 0.10), 0.0, 1.0))
        w_hard = float(np.clip(cfg.get("FUTURES_HMM_STEP2_TAIL8_HARD_BLEND_W", 0.10), 0.0, 1.0))
        w_near_flat = float(np.clip(cfg.get("FUTURES_HMM_STEP2_TAIL8_NEAR_FLAT_BLEND_W", 0.10), 0.0, 1.0))
        tier_sum = max(1e-6, 1.0 + w_soft + w_hard + w_near_flat)
        tail8_target = _safe01(
            (tail8_target + w_soft * sup_soft + w_hard * sup_hard + w_near_flat * sup_near_flat) / tier_sum
        )
        tail8_lift = tail8 + tail8_boost * top_scale * (tail8_target - tail8)
        tail8 = _safe01(np.maximum(tail8, tail8_lift))

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
    if sup_bundle:
        for k, arr in sup_bundle.items():
            hazards[k] = _safe01(arr)
    hazards = hazards.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)

    report = {
        "pre_crisis_hazard_mean": float(hazards["pre_crisis_hazard"].mean()),
        "realized_crisis_hazard_mean": float(hazards["realized_crisis_hazard"].mean()),
        "tail_hazard_8h_p95": float(hazards["tail_hazard_8h"].quantile(0.95)),
    }
    try:
        diag_specs = (
            ("sup_score_q10_h8", 8, 0.10),
            ("sup_score_q05_h8", 8, 0.05),
            ("sup_score_q03_h16", 16, 0.03),
        )
        r = pd.to_numeric(returns_ser.reindex(idx), errors="coerce").fillna(0.0)
        for score_col, horizon, q in diag_specs:
            if score_col not in hazards.columns:
                continue
            fwd_worst = pd.Series(np.inf, index=idx, dtype=np.float64)
            for k in range(1, int(max(2, horizon)) + 1):
                fwd_worst = np.minimum(fwd_worst, r.shift(-k).to_numpy(dtype=np.float64))
            fwd_worst = pd.Series(fwd_worst, index=idx).replace([np.inf, -np.inf], np.nan)
            q_thr = float(fwd_worst.quantile(q)) if fwd_worst.notna().any() else np.nan
            if not np.isfinite(q_thr):
                continue
            top_mask = hazards[score_col] >= float(hazards[score_col].quantile(0.90))
            if bool(np.any(top_mask)):
                hit = float((fwd_worst[top_mask] <= q_thr).mean() * 100.0)
                report[f"hmm_{score_col}_top_decile_hit"] = hit
    except Exception:
        pass
    return TailHazardOverlayResult(hazards=hazards, report=report)


def _fit_supervised_tail_score(
    X: np.ndarray,
    r_ser: pd.Series,
    idx: pd.Index,
    is_end_idx: int,
    cfg: dict[str, Any] | None = None,
) -> dict[str, np.ndarray] | None:
    """IS logistic+isotonic supervised score; causal — no look-ahead.

    Args:
        X: Feature matrix (n, 7), full timeline.
        r_ser: Returns series aligned to idx.
        idx: Full timeline index.
        is_end_idx: Exclusive end of IS window.

    Returns:
        Dict of calibrated multi-label scores or None on failure.
    """
    try:
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    except ImportError:
        _logger.warning("scikit-learn not available; using hardcoded fallback")
        return None

    try:
        cfg = cfg or {}
        n = len(X)

        horizons_raw = cfg.get("FUTURES_HMM_SUP_HORIZONS", [4, 8, 16])
        try:
            horizons = tuple(int(h) for h in horizons_raw)
        except Exception:
            horizons = (4, 8, 16)
        horizons = tuple(sorted({h for h in horizons if h > 1})) or (4, 8, 16)
        quantiles = {
            "q10": float(np.clip(cfg.get("FUTURES_HMM_SUP_LABEL_Q10", 0.10), 0.001, 0.40)),
            "q05": float(np.clip(cfg.get("FUTURES_HMM_SUP_LABEL_Q05", 0.05), 0.001, 0.30)),
            "q03": float(np.clip(cfg.get("FUTURES_HMM_SUP_LABEL_Q03", 0.03), 0.001, 0.20)),
        }
        min_pos = int(max(5, cfg.get("FUTURES_HMM_SUP_MIN_POS", 12)))
        rank_blend_w = float(np.clip(cfg.get("FUTURES_HMM_SUP_RANK_BLEND_W", 0.30), 0.0, 0.95))
        rank_blend_pow = float(np.clip(cfg.get("FUTURES_HMM_SUP_RANK_BLEND_POW", 1.20), 0.5, 3.0))
        X_is = X[:is_end_idx]
        scores: dict[str, np.ndarray] = {}
        fwd_map: dict[int, pd.Series] = {}
        for h in horizons:
            fwd_worst = np.full(n, np.inf, dtype=np.float64)
            for k in range(1, int(h) + 1):
                shifted = r_ser.shift(-k).to_numpy(dtype=np.float64)
                fwd_worst = np.minimum(fwd_worst, shifted)
            fwd_map[h] = pd.Series(fwd_worst, index=idx).replace([np.inf, -np.inf], np.nan)

        for h in horizons:
            fwd_ser = fwd_map[h]
            valid_is = fwd_ser.iloc[:is_end_idx].dropna()
            if len(valid_is) < 20:
                continue
            for q_name, q_v in quantiles.items():
                q_thr = float(valid_is.quantile(q_v))
                if not np.isfinite(q_thr):
                    continue
                y_is = (fwd_ser.iloc[:is_end_idx].fillna(0.0).to_numpy() <= q_thr).astype(int)
                if int(y_is.sum()) < min_pos:
                    continue
                clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=500, random_state=42)
                clf.fit(X_is, y_is)
                prob_is = clf.predict_proba(X_is)[:, 1]
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(prob_is, y_is.astype(np.float64))
                prob_all = clf.predict_proba(X)[:, 1]
                score_iso = _safe01(iso.transform(prob_all).astype(np.float64))
                rank = _safe01(_rank_pct01(score_iso) ** rank_blend_pow)
                score = _safe01((1.0 - rank_blend_w) * score_iso + rank_blend_w * rank)
                scores[f"sup_score_{q_name}_h{h}"] = score

        if "sup_score_q10_h4" in scores:
            scores["sup_score_soft"] = scores["sup_score_q10_h4"]
        elif "sup_score_q10_h8" in scores:
            scores["sup_score_soft"] = scores["sup_score_q10_h8"]
        if "sup_score_q05_h8" in scores:
            scores["sup_score_hard"] = scores["sup_score_q05_h8"]
        elif "sup_score_q05_h16" in scores:
            scores["sup_score_hard"] = scores["sup_score_q05_h16"]
        if "sup_score_q03_h16" in scores:
            scores["sup_score_near_flat"] = scores["sup_score_q03_h16"]
        elif "sup_score_q03_h8" in scores:
            scores["sup_score_near_flat"] = scores["sup_score_q03_h8"]
        if "sup_score_q10_h8" not in scores and "sup_score_soft" in scores:
            scores["sup_score_q10_h8"] = scores["sup_score_soft"]
        if not scores:
            raise ValueError("No valid supervised multi-label score trained")
        _logger.info(
            "fit_predict_tail_overlay Step3 | is_end=%d | n=%d | multi_scores=%d | soft_mean=%.3f",
            is_end_idx,
            n,
            len(scores),
            float(np.mean(scores.get("sup_score_soft", scores.get("sup_score_q10_h8", np.zeros(n, dtype=np.float64))))),
        )
        return scores
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
    p_calm_s = pd.to_numeric(market_probs.get("hmm_prob_bull_calm", 0.25), errors="coerce").fillna(0.25)
    p_off_s = pd.to_numeric(market_probs.get("hmm_prob_bear_trend", 0.25), errors="coerce").fillna(0.25)
    p_chop_s = pd.to_numeric(market_probs.get("hmm_prob_chop", 0.25), errors="coerce").fillna(0.25)
    ent_s = pd.to_numeric(market_probs.get("regime_entropy", 0.5), errors="coerce").fillna(0.5)

    r_ser = market_returns.reindex(idx).fillna(0.0).astype(np.float64)
    r_np = r_ser.to_numpy(dtype=np.float64)

    feats_df = market_hmm_feats.reindex(idx) if market_hmm_feats is not None else pd.DataFrame(index=idx)

    rv24_z = _zscore_robust(pd.Series(np.abs(r_np), index=idx).rolling(24, min_periods=4).std().fillna(0.0))
    down24_z = _zscore_robust(pd.Series(np.clip(-r_np, 0.0, None), index=idx).rolling(24, min_periods=4).std().fillna(0.0))
    mom6_z = _zscore_robust(pd.Series(r_np, index=idx).rolling(6, min_periods=2).mean().fillna(0.0))
    mom24_z = _zscore_robust(pd.Series(r_np, index=idx).rolling(24, min_periods=4).mean().fillna(0.0))
    vol24 = pd.Series(r_np, index=idx).rolling(24, min_periods=4).std().fillna(0.0)
    vol96 = pd.Series(r_np, index=idx).rolling(96, min_periods=12).std().replace(0.0, np.nan).fillna(vol24 + 1e-9)
    vol_ratio_z = _zscore_robust((vol24 / vol96).replace([np.inf, -np.inf], np.nan).fillna(0.0))
    px = (1.0 + pd.Series(r_np, index=idx)).cumprod()
    dd96 = (px / px.rolling(96, min_periods=8).max() - 1.0).fillna(0.0)
    dd96_z = _zscore_robust(dd96)
    spread_off_calm = _zscore_robust(pd.Series((p_off_s - p_calm_s).to_numpy(dtype=np.float64), index=idx))
    spread_off_chop = _zscore_robust(pd.Series((p_off_s - p_chop_s).to_numpy(dtype=np.float64), index=idx))
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
        mom6_z,
        mom24_z,
        vol_ratio_z,
        dd96_z,
        spread_off_calm,
        spread_off_chop,
    ]).astype(np.float64)

    # ── supervised score (IS only, causal) ─────────────────────────────────
    supervised_score: dict[str, np.ndarray] | None = None
    method = "hardcoded"
    if is_end_idx > 50:
        supervised_score = _fit_supervised_tail_score(X, r_ser, idx, is_end_idx, cfg=cfg)
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
