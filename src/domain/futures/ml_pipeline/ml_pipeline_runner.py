"""ML pipeline execution orchestration for Cross-Sectional Ranking Portfolio."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import FUTURES_CACHE_DIR, FUTURES_DATA_DIR
from src.domain.futures.data_collector import DataCollector
from src.domain.futures.funding_utils import merge_funding_into_ohlcv
from src.domain.futures.metrics_utils import merge_metrics_into_ohlcv
from src.domain.futures.ml_pipeline.cross_sectional_utils import CrossSectionalPipelineUtils
from src.domain.futures.ml_pipeline.feature_engineering import (
    GP_ENGINEERED_FEATURE_NAMES,
    HMM_SEMANTIC_PROB_COLUMNS,
    build_gp_input_features,
)
from src.domain.futures.ml_pipeline.gp_alpha_miner import GPAlphaMiner
from src.domain.futures.ml_pipeline.gp_multiobjective import is_deap_available
from src.domain.futures.ml_pipeline.hmm_state_inferrer import HMMStateInferrer
from src.domain.futures.ml_pipeline.meta_labeler import MetaLabeler
from src.domain.futures.ml_pipeline.triple_barrier import label_triple_barrier

_logger = logging.getLogger(__name__)


@dataclass
class MLPipelineOutput:
    """Container for ML pipeline results by symbol."""

    calib_prob_by_symbol: dict[str, pd.Series] = field(default_factory=dict)
    calib_prob_long_by_symbol: dict[str, pd.Series] = field(default_factory=dict)
    calib_prob_short_by_symbol: dict[str, pd.Series] = field(default_factory=dict)
    meta_feature_frame_by_symbol: dict[str, pd.DataFrame] = field(default_factory=dict)
    health_metrics_by_symbol: dict[str, dict[str, float]] = field(default_factory=dict)
    alpha_panel: pd.DataFrame = field(default_factory=pd.DataFrame)


def _sorted_hmm_prob_columns(df: pd.DataFrame) -> list[str]:
    sem = [c for c in HMM_SEMANTIC_PROB_COLUMNS if c in df.columns]
    if sem:
        return sem
    # Only include columns where the last part is a digit to avoid ValueError (e.g., hmm_prob_0_x)
    legacy = [
        c for c in df.columns
        if str(c).startswith("hmm_prob_") and str(c).split("_")[-1].isdigit()
    ]
    return sorted(legacy, key=lambda x: int(str(x).split("_")[-1]))


def _meta_feature_column_names(wide_1h: pd.DataFrame) -> tuple[str, ...]:
    hmm_cols = _sorted_hmm_prob_columns(wide_1h)
    if "gp_alpha_00" in wide_1h.columns:
        return ("gp_alpha_00", *hmm_cols)
    return tuple(hmm_cols)


def _attach_tbm_gp_weights(
    sym: str,
    df_1h: pd.DataFrame,
    label_start: str,
    end: str,
    collector: DataCollector,
    df_1m: pd.DataFrame | None,
) -> pd.DataFrame:
    """Per tmp.md 2-B: up-weight rows with clear triple-barrier (+1 / -1) hits for GP fitness."""
    out = df_1h.copy()
    lab = _try_tbm_labels_per_1h_row(sym, out, label_start, end, df_1m=df_1m, collector=collector)
    if lab is None or len(lab) != len(out):
        out["tbm_gp_weight"] = 1.0
        return out
    out["tbm_gp_weight"] = np.where(
        np.isfinite(lab) & (np.abs(lab) > 0.9), 1.5, 1.0
    ).astype(np.float64)
    return out


def _merge_gp_into_1h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Append GP microstructure / momentum columns to 1h OHLCV (sorted by datetime)."""
    out = df_1h.copy()
    if "open" not in out.columns:
        out["open"] = pd.to_numeric(out["close"], errors="coerce").shift(1).fillna(out["close"])
    w = out.sort_values("datetime").reset_index(drop=True)
    idx = pd.DatetimeIndex(pd.to_datetime(w["datetime"], utc=True))
    gp = build_gp_input_features(w.set_index(idx))
    for col in gp.columns:
        w[col] = gp[col].to_numpy()
    return w


def _ensure_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" in df.columns:
        return df
    out = df.reset_index()
    if "datetime" not in out.columns and len(out.columns) > 0:
        c0 = str(out.columns[0])
        if c0 != "datetime":
            out = out.rename(columns={c0: "datetime"})
    return out


def _is_kelly_per_semantic_state(
    p_mat: np.ndarray,
    fwd_ret: np.ndarray,
    is_mask: np.ndarray,
    n_states: int,
    cols: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
    """James-Stein shrunk Kelly on global variance, plus soft Isotonic blend (tmp v4)."""
    kelly_long: np.ndarray = np.zeros(n_states, dtype=np.float64)
    kelly_short: np.ndarray = np.zeros(n_states, dtype=np.float64)
    state_hard = np.argmax(p_mat, axis=1).astype(np.int64)

    fwd_is = fwd_ret[is_mask]
    if fwd_is.size < 2:
        return kelly_long, kelly_short
    global_mu = float(np.mean(fwd_is))
    global_v = float(np.var(fwd_is, ddof=1)) + 1e-12

    for s in range(n_states):
        m = is_mask & (state_hard == s)
        n_s = int(np.sum(m))
        if n_s < 30:
            continue
        r = fwd_ret[m]
        mu = float(np.mean(r))

        # [REFACTORED] Stronger Relative Kelly (Demeaning)
        # Subtract FULL market average to force alpha-only Longs.
        mu_relative = mu - global_mu

        # James-Stein shrink towards zero (neutral)
        alpha_js = 30.0 / (30.0 + float(n_s))
        mu_shrunk = alpha_js * 0.0 + (1.0 - alpha_js) * mu_relative

        kelly_long[s] = float(np.clip(mu_shrunk / global_v, -1.0, 1.0))
        kelly_short[s] = float(np.clip(-mu_shrunk / global_v, -1.0, 1.0))

    alpha_iso = 0.5
    try:
        ir = IsotonicRegression(increasing=False)

        idx_bull = cols.index("hmm_prob_bull_trend") if "hmm_prob_bull_trend" in cols else -1
        idx_chop = cols.index("hmm_prob_chop") if "hmm_prob_chop" in cols else -1
        idx_bear = cols.index("hmm_prob_bear_trend") if "hmm_prob_bear_trend" in cols else -1
        idx_crisis = cols.index("hmm_prob_crisis") if "hmm_prob_crisis" in cols else -1

        if all(i >= 0 for i in [idx_bull, idx_chop, idx_bear, idx_crisis]):
            x = np.array([0, 1, 2, 3], dtype=np.float64)
            y_l = np.array(
                [
                    kelly_long[idx_bull],
                    kelly_long[idx_chop],
                    kelly_long[idx_bear],
                    kelly_long[idx_crisis],
                ],
                dtype=np.float64,
            )
            y_l_adj = ir.fit_transform(x, y_l)
            kelly_long[idx_bull] = (1.0 - alpha_iso) * y_l[0] + alpha_iso * y_l_adj[0]
            kelly_long[idx_chop] = (1.0 - alpha_iso) * y_l[1] + alpha_iso * y_l_adj[1]
            kelly_long[idx_bear] = (1.0 - alpha_iso) * y_l[2] + alpha_iso * y_l_adj[2]
            kelly_long[idx_crisis] = (1.0 - alpha_iso) * y_l[3] + alpha_iso * y_l_adj[3]

            y_s = np.array(
                [
                    kelly_short[idx_bear],
                    kelly_short[idx_chop],
                    kelly_short[idx_bull],
                    kelly_short[idx_crisis],
                ],
                dtype=np.float64,
            )
            y_s_adj = ir.fit_transform(x, y_s)
            kelly_short[idx_bear] = (1.0 - alpha_iso) * y_s[0] + alpha_iso * y_s_adj[0]
            kelly_short[idx_chop] = (1.0 - alpha_iso) * y_s[1] + alpha_iso * y_s_adj[1]
            kelly_short[idx_bull] = (1.0 - alpha_iso) * y_s[2] + alpha_iso * y_s_adj[2]
            kelly_short[idx_crisis] = (1.0 - alpha_iso) * y_s[3] + alpha_iso * y_s_adj[3]
    except Exception as e:
        _logger.warning("Isotonic Kelly adjustment failed: %s", e)

    return kelly_long, kelly_short


def _hmm_modulator_kelly_values(
    market_probs: pd.DataFrame,
    alpha_panel: pd.DataFrame,
    is_end_utc: pd.Timestamp,
    btc_df: pd.DataFrame | None,
    shrink: float,
    crisis_thr: float,
    market_hmm_feats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """[REFACTORED] Posterior-weighted asymmetric modulators (Long/Short).

    Returns DataFrame with columns ['hmm_modulator_long', 'hmm_modulator_short',
    'hmm_modulator_base_long', 'btc_trend_vol_adj_24h'].
    hmm_modulator_base_long = pre-crisis-kill modulator (for probe replay).
    """
    cols = _sorted_hmm_prob_columns(market_probs)
    n = len(market_probs)
    if not cols or btc_df is None or len(btc_df) < 80:
        return pd.DataFrame({
            "hmm_modulator_long": np.full(n, 1.0, dtype=np.float64),
            "hmm_modulator_short": np.full(n, 1.0, dtype=np.float64),
            "hmm_modulator_base_long": np.full(n, 1.0, dtype=np.float64),
            "btc_trend_vol_adj_24h": np.zeros(n, dtype=np.float64),
        })

    k_states = len(cols)
    b = btc_df.sort_values("datetime").copy()
    b["datetime"] = pd.to_datetime(b["datetime"], utc=True)
    c = b["close"].astype(np.float64)
    fwd_1h = np.log(c.shift(-1) / c.clip(lower=1e-12)).fillna(0.0)
    fwd_24h = np.log(c.shift(-24) / c.clip(lower=1e-12)).fillna(0.0) / np.sqrt(24.0)
    b["fwd_ret"] = 0.3 * fwd_1h + 0.7 * fwd_24h

    mp = market_probs[["datetime", *cols]].copy()
    mp["datetime"] = pd.to_datetime(mp["datetime"], utc=True)
    merged = mp.merge(b[["datetime", "fwd_ret"]], on="datetime", how="left")
    merged["fwd_ret"] = merged["fwd_ret"].fillna(0.0)

    fwd_ret = merged["fwd_ret"].to_numpy(dtype=np.float64)
    p_mat = (
        merged[cols].replace([np.inf, -np.inf], np.nan).fillna(1.0 / float(k_states)).to_numpy()
    )
    dt_mp = pd.to_datetime(merged["datetime"], utc=True)
    is_mask = (dt_mp < is_end_utc).to_numpy()

    k_long, k_short = _is_kelly_per_semantic_state(p_mat, fwd_ret, is_mask, k_states, cols)

    # Apply shrink to deviations from 1.0
    mod_long = np.clip(1.0 + float(shrink) * (p_mat @ k_long), 0.3, 2.0).astype(np.float64)
    mod_short = np.clip(1.0 + float(shrink) * (p_mat @ k_short), 0.3, 2.0).astype(np.float64)

    # [NEW] Asymmetric Regime Override (Hard Capping & Boosting)
    p_bull = merged["hmm_prob_bull_trend"].to_numpy(dtype=np.float64)
    p_bear = merged["hmm_prob_bear_trend"].to_numpy(dtype=np.float64)
    p_chop = merged["hmm_prob_chop"].to_numpy(dtype=np.float64)
    p_crisis = merged["hmm_prob_crisis"].to_numpy(dtype=np.float64)

    # BEAR_TREND Override: Aggressively cap long, allow short boost
    mod_long = np.where(p_bear > 0.25, np.minimum(mod_long, 0.6), mod_long)
    mod_short = np.where(p_bear > 0.25, np.maximum(mod_short, 1.5), mod_short)

    # BULL_TREND Override: Allow long boost, cap short
    mod_long = np.where(p_bull > 0.25, np.maximum(mod_long, 1.5), mod_long)
    mod_short = np.where(p_bull > 0.25, np.minimum(mod_short, 0.6), mod_short)

    # CHOP Override: Defensive sizing for both
    mod_long = np.where(p_chop > 0.4, np.minimum(mod_long, 0.8), mod_long)
    mod_short = np.where(p_chop > 0.4, np.minimum(mod_short, 0.8), mod_short)

    # Capture pre-kill state for probe replay (before crisis suppression)
    mod_long_base: np.ndarray = mod_long.copy()

    # Apply Crisis Kill-Switch (Safety First)
    if "hmm_prob_crisis" in market_probs.columns:
        pc = p_crisis
        mod_long = np.where(pc > float(crisis_thr), 0.3, mod_long)
        mod_short = np.where(
            pc > float(crisis_thr), np.minimum(mod_short, 1.1), mod_short
        )

    # RECOVERY Override: crisis label + positive 24h trend → partial lift
    # Activated only when CRISIS_RECOVERY_TREND_THR < 1e8 in OPT_FUTURES_CONFIG.
    trend_24h: np.ndarray = np.zeros(n, dtype=np.float64)
    if market_hmm_feats is not None and "btc_trend_vol_adj_24h" in market_hmm_feats.columns:
        feat_tmp = market_hmm_feats[["btc_trend_vol_adj_24h"]].copy()
        feat_tmp.index = pd.to_datetime(feat_tmp.index, utc=True)
        feat_tmp = feat_tmp.reset_index()
        idx_col = str(feat_tmp.columns[0])
        if idx_col != "datetime":
            feat_tmp = feat_tmp.rename(columns={idx_col: "datetime"})
        feat_tmp["datetime"] = pd.to_datetime(feat_tmp["datetime"], utc=True)
        merged_rec = merged[["datetime"]].merge(feat_tmp, on="datetime", how="left")
        trend_24h = merged_rec["btc_trend_vol_adj_24h"].fillna(0.0).to_numpy(dtype=np.float64)

    rec_thr = float(OPT_FUTURES_CONFIG.get("CRISIS_RECOVERY_TREND_THR", 1e9))
    rec_floor = float(OPT_FUTURES_CONFIG.get("CRISIS_RECOVERY_FLOOR", 0.30))
    if rec_thr < 1e8 and "hmm_prob_crisis" in market_probs.columns:
        is_recovery = (p_crisis > float(crisis_thr)) & (trend_24h > rec_thr)
        mod_long = np.where(is_recovery, np.maximum(mod_long, rec_floor), mod_long)

    return pd.DataFrame({
        "hmm_modulator_long": mod_long,
        "hmm_modulator_short": mod_short,
        "hmm_modulator_base_long": mod_long_base,
        "btc_trend_vol_adj_24h": trend_24h,
    })


def _try_tbm_labels_per_1h_row(
    sym: str,
    df_1h: pd.DataFrame,
    fetch_start: str,
    end: str,
    df_1m: pd.DataFrame | None = None,
    collector: DataCollector | None = None,
) -> np.ndarray | None:
    """Triple-barrier labels aligned to each 1h row."""
    try:
        need = {"open", "high", "low", "close"}
        if not need.issubset(set(df_1h.columns)):
            return None
        if df_1m is None:
            if collector is None:
                return None
            df_1m = collector.collect_1m_ohlcv(sym, fetch_start, end)
        if df_1m is None or len(df_1m) < 200:
            return None

        # [REFACTORED] Use 1h directly for TBM labeling (No 4h resampling)
        d1 = df_1h[[*list(need), "datetime"]].sort_values("datetime").copy()
        d1["datetime"] = pd.to_datetime(d1["datetime"], utc=True)

        if len(d1) < 30:
            return None

        tbm = label_triple_barrier(
            d1,
            df_1m,
            time_stop_bars=int(OPT_FUTURES_CONFIG.get("FUTURES_TBM_TIME_STOP_BARS", 1440)),
            vol_scale_window=int(OPT_FUTURES_CONFIG.get("FUTURES_TBM_VOL_SCALE_WINDOW", 24)),
        )
        if tbm is None or len(tbm) == 0:
            return None

        lab = tbm.rename("tbm_label").reset_index()
        lab["datetime"] = pd.to_datetime(lab["datetime"], utc=True)
        lab = lab.sort_values("datetime")

        tmp = df_1h[["datetime"]].copy()
        tmp["_ord"] = np.arange(len(tmp), dtype=np.int64)
        tmp["datetime"] = pd.to_datetime(tmp["datetime"], utc=True)
        tmp = tmp.sort_values("datetime")

        # Align labels back to 1h rows
        merged = pd.merge_asof(tmp, lab, on="datetime", direction="backward")
        merged = merged.sort_values("_ord")
        return cast(np.ndarray, merged["tbm_label"].to_numpy(dtype=np.float64))
    except Exception:
        return None


def _meta_probs_wf_refit(
    X_w: pd.DataFrame,
    y_ser: pd.Series,
    aligned_tf: pd.DataFrame,
    meta_feats: tuple[str, ...],
    hmm_m_long: np.ndarray,
    hmm_m_short: np.ndarray,
    is_end_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 3: expanding-window MetaLabeler refit on each WF-OOS segment (no softmax)."""
    cfg = OPT_FUTURES_CONFIG
    n_rows = len(X_w)
    feats = list(meta_feats)
    vb = int(cfg.get("FUTURES_META_VERTICAL_BARRIER_BARS", 24))
    mi = int(cfg.get("FUTURES_META_MIN_POS_ISOTONIC", 200))
    wf_on = bool(cfg.get("FUTURES_ML_WF_REFIT_ENABLED", True))
    n_wf = max(1, int(cfg.get("FUTURES_ML_WF_REFIT_LEGS", 3)))

    pl_out: np.ndarray = np.zeros(n_rows, dtype=np.float64)
    ps_out: np.ndarray = np.zeros(n_rows, dtype=np.float64)

    if not wf_on or n_wf <= 1 or is_end_idx >= n_rows - 1:
        meta = MetaLabeler(vertical_barrier_bars=vb, min_pos_isotonic=mi)
        meta.fit(X_w, y_ser, is_end_idx)
        X_a = aligned_tf[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pl, ps = meta.predict_proba_calibrated(X_a)
        rl = np.clip(pl.astype(np.float64) * hmm_m_long, 0.0, 1.0)
        rs = np.clip(ps.astype(np.float64) * hmm_m_short, 0.0, 1.0)
        return rl, rs

    meta = MetaLabeler(vertical_barrier_bars=vb, min_pos_isotonic=mi)
    meta.fit(X_w.iloc[:is_end_idx], y_ser.iloc[:is_end_idx], is_end_idx=is_end_idx)
    X_is = aligned_tf.iloc[:is_end_idx][feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pl_i, ps_i = meta.predict_proba_calibrated(X_is)
    pl_out[:is_end_idx] = np.clip(pl_i.astype(np.float64) * hmm_m_long[:is_end_idx], 0.0, 1.0)
    ps_out[:is_end_idx] = np.clip(ps_i.astype(np.float64) * hmm_m_short[:is_end_idx], 0.0, 1.0)

    n_seg = min(n_wf, max(2, n_rows - is_end_idx))
    edges = np.linspace(is_end_idx, n_rows, n_seg + 1)
    edges_i = np.unique(np.clip(np.round(edges).astype(np.int64), 0, n_rows))

    for k in range(len(edges_i) - 1):
        t0, t1 = int(edges_i[k]), int(edges_i[k + 1])
        if t1 <= t0 or t0 < is_end_idx:
            continue
        train_end = t0
        if train_end < 80:
            continue
        meta.fit(X_w.iloc[:train_end], y_ser.iloc[:train_end], is_end_idx=train_end)
        X_seg = aligned_tf.iloc[t0:t1][feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pl_s, ps_s = meta.predict_proba_calibrated(X_seg)
        pl_out[t0:t1] = np.clip(pl_s.astype(np.float64) * hmm_m_long[t0:t1], 0.0, 1.0)
        ps_out[t0:t1] = np.clip(ps_s.astype(np.float64) * hmm_m_short[t0:t1], 0.0, 1.0)

    return pl_out, ps_out


def _apply_ml_calib_probs(
    aligned_tf: pd.DataFrame,
    wide_1h: pd.DataFrame,
    collector: DataCollector,
    sym: str,
    fetch_start: str,
    end: str,
    is_end_utc: pd.Timestamp,
    use_meta: bool,
    df_1m_prefetch: pd.DataFrame | None = None,
) -> None:
    """Cross-sectional scores (gp x HMM); ml_calib_* = 1.0; optional MetaLabeler refit."""
    hmm_m_long = (
        aligned_tf["hmm_modulator_long"].to_numpy(dtype=np.float64)
        if "hmm_modulator_long" in aligned_tf.columns
        else np.ones(len(aligned_tf), dtype=np.float64)
    )
    hmm_m_short = (
        aligned_tf["hmm_modulator_short"].to_numpy(dtype=np.float64)
        if "hmm_modulator_short" in aligned_tf.columns
        else np.ones(len(aligned_tf), dtype=np.float64)
    )
    gp = (
        aligned_tf["gp_alpha_00"].to_numpy(dtype=np.float64)
        if "gp_alpha_00" in aligned_tf.columns
        else np.zeros(len(aligned_tf), dtype=np.float64)
    )
    aligned_tf["xs_score_long"] = gp * hmm_m_long
    # Invert hms for short ranking: lower xs_short = better short in numba CS ranker.
    # BEAR/CRISIS (hms>1): gp/hms < gp → lower → ranked higher as short. ✓
    aligned_tf["xs_score_short"] = gp / np.maximum(hmm_m_short, 0.1)

    meta_on = bool(use_meta) and bool(OPT_FUTURES_CONFIG.get("FUTURES_USE_META_LABELER", False))

    y_tbm = (
        _try_tbm_labels_per_1h_row(
            sym, wide_1h, fetch_start, end, df_1m=df_1m_prefetch, collector=collector
        )
        if meta_on
        else None
    )
    meta_feats = tuple(c for c in _meta_feature_column_names(wide_1h) if c in wide_1h.columns)
    can_meta = (
        meta_on
        and y_tbm is not None
        and len(y_tbm) == len(wide_1h)
        and len(meta_feats) >= 2
        and len(_sorted_hmm_prob_columns(wide_1h)) > 0
        and all(c in aligned_tf.columns for c in meta_feats)
    )
    n = len(aligned_tf)
    pl = np.ones(n, dtype=np.float64)
    ps = np.ones(n, dtype=np.float64)

    if can_meta:
        X_w = wide_1h[list(meta_feats)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_ser = pd.Series(y_tbm, index=wide_1h.index)
        wdt = pd.to_datetime(wide_1h["datetime"], utc=True)
        is_end_idx = int((wdt < is_end_utc).sum())
        if is_end_idx >= 80:
            try:
                pl, ps = _meta_probs_wf_refit(
                    X_w,
                    y_ser,
                    aligned_tf,
                    meta_feats,
                    hmm_m_long,
                    hmm_m_short,
                    is_end_idx,
                )
            except Exception as exc:
                _logger.debug("MetaLabeler WF refit skipped: %s", exc)

    aligned_tf["ml_calib_prob_long"] = pl
    aligned_tf["ml_calib_prob_short"] = ps
    aligned_tf["ml_calib_prob"] = np.maximum(pl, ps)


class _Step4FusionOutcome(NamedTuple):
    sym: str
    aligned_tf: pd.DataFrame | None
    cp_long: pd.Series | None
    cp_short: pd.Series | None
    error: str | None


def _step4_fusion_one_symbol(
    sym: str,
    tf: str,
    data_maps: dict[str, dict[str, Any]],
    prefetched_1h: dict[str, pd.DataFrame],
    alpha_by_sym: dict[str, pd.DataFrame],
    valid_alpha_set: set[str],
    market_probs: pd.DataFrame,
    hmm_modulator: pd.DataFrame,
    fetch_start: str,
    end: str,
    is_end_utc: pd.Timestamp,
    df_1m_prefetch: pd.DataFrame | None,
    collector: DataCollector,
) -> _Step4FusionOutcome:
    """Per-symbol merge + asof alignment + MetaLabeler (Phase 3: raw x modulator, WF refit)."""
    try:
        df_1h = prefetched_1h[sym].copy()
        df_1h["datetime"] = pd.to_datetime(df_1h["datetime"], utc=True)

        # Clear any existing ML/HMM columns to prevent merge collisions (_x, _y)
        ml_reserved = [
            "gp_alpha_00", "hmm_modulator_long", "hmm_modulator_short",
            "slot_rank_score", "xs_score_long", "xs_score_short",
            "ml_calib_prob", "ml_calib_prob_long", "ml_calib_prob_short",
            "hmm_prob_bull_trend", "hmm_prob_bear_trend", "hmm_prob_chop", "hmm_prob_crisis"
        ]
        # Also drop any legacy or dynamic HMM probability columns
        hmm_to_drop = [c for c in df_1h.columns if str(c).startswith("hmm_prob_")]

        drop_exist = list(set(ml_reserved + hmm_to_drop))
        drop_exist = [c for c in drop_exist if c in df_1h.columns]

        if drop_exist:
            df_1h = df_1h.drop(columns=drop_exist)
        hmm_cols_ref = _sorted_hmm_prob_columns(market_probs)
        k_fb = len(hmm_cols_ref) if hmm_cols_ref else len(HMM_SEMANTIC_PROB_COLUMNS)

        if sym not in valid_alpha_set:
            wide_1h = df_1h.copy()
            wide_1h["gp_alpha_00"] = 0.0
            # [REFACTORED] Merge asymmetric modulators
            wide_1h = pd.merge(wide_1h, hmm_modulator, on="datetime", how="left")
            if "hmm_modulator_long" not in wide_1h.columns:
                 _logger.error(
                     "[%s] Merge failed to add hmm_modulator_long. "
                     "wide_1h.cols=%s, hmm_modulator.cols=%s",
                     sym, list(wide_1h.columns), list(hmm_modulator.columns)
                 )
            wide_1h["hmm_modulator_long"] = wide_1h["hmm_modulator_long"].fillna(0.8)
            wide_1h["hmm_modulator_short"] = wide_1h["hmm_modulator_short"].fillna(0.8)
            wide_1h["slot_rank_score"] = 0.0

            if hmm_cols_ref:
                mp_h = market_probs[["datetime", *hmm_cols_ref]]
                wide_1h = wide_1h.merge(mp_h, on="datetime", how="left")
                for c in hmm_cols_ref:
                    wide_1h[c] = wide_1h[c].fillna(1.0 / float(k_fb))
            else:
                for c in HMM_SEMANTIC_PROB_COLUMNS:
                    wide_1h[c] = 1.0 / float(len(HMM_SEMANTIC_PROB_COLUMNS))
        else:
            sym_alpha = alpha_by_sym[sym].copy()
            sym_alpha["datetime"] = pd.to_datetime(sym_alpha["datetime"], utc=True)
            if "gp_alpha_00" not in sym_alpha.columns:
                _logger.warning(
                    "[%s] gp_alpha_00 missing in sym_alpha. cols=%s",
                    sym, list(sym_alpha.columns)
                )
                sym_alpha["gp_alpha_00"] = 0.0

            wide_1h = pd.merge(df_1h, sym_alpha, on="datetime", how="left").fillna(0.0)

            # [REFACTORED] Merge asymmetric modulators
            wide_1h = pd.merge(wide_1h, hmm_modulator, on="datetime", how="left")
            if "hmm_modulator_long" not in wide_1h.columns:
                 _logger.error(
                     "[%s] Merge failed to add hmm_modulator_long. "
                     "wide_1h.cols=%s, hmm_modulator.cols=%s",
                     sym, list(wide_1h.columns), list(hmm_modulator.columns)
                 )
            wide_1h["hmm_modulator_long"] = wide_1h["hmm_modulator_long"].fillna(0.8)
            wide_1h["hmm_modulator_short"] = wide_1h["hmm_modulator_short"].fillna(0.8)

            # Use mean of long/short modulator for ranking score
            m_avg = (wide_1h["hmm_modulator_long"] + wide_1h["hmm_modulator_short"]) / 2.0
            wide_1h["slot_rank_score"] = wide_1h["gp_alpha_00"] * m_avg

            if hmm_cols_ref:
                mp_h = market_probs[["datetime", *hmm_cols_ref]]
                wide_1h = wide_1h.merge(mp_h, on="datetime", how="left")
                for c in hmm_cols_ref:
                    wide_1h[c] = wide_1h[c].fillna(1.0 / float(k_fb))
            else:
                for c in HMM_SEMANTIC_PROB_COLUMNS:
                    wide_1h[c] = 1.0 / float(len(HMM_SEMANTIC_PROB_COLUMNS))

        df_tf_full = data_maps[sym][tf].copy()
        df_tf_full["datetime"] = pd.to_datetime(df_tf_full["datetime"], utc=True)
        aligned_tf = pd.merge(df_tf_full, wide_1h, on="datetime", how="left").fillna(0.0)

        _apply_ml_calib_probs(
            aligned_tf,
            wide_1h,
            collector,
            sym,
            fetch_start,
            end,
            is_end_utc,
            use_meta=(sym in valid_alpha_set),
            df_1m_prefetch=df_1m_prefetch,
        )

        cp_long = aligned_tf.set_index("datetime")["ml_calib_prob_long"]
        cp_short = aligned_tf.set_index("datetime")["ml_calib_prob_short"]
        return _Step4FusionOutcome(sym, aligned_tf, cp_long, cp_short, None)
    except Exception as e:
        return _Step4FusionOutcome(sym, None, None, None, str(e))


def _print_hmm_summary(
    market_probs: pd.DataFrame,
    market_hmm_feats: pd.DataFrame,
    hmm_modulator: pd.DataFrame,
    btc_1h: pd.DataFrame | None,
    mode_label: str = "",
) -> None:
    """Print a detailed financial and semantic profile of the HMM states."""
    _logger.info("-" * 85)
    _logger.info(f" [HMM AUDIT] Semantic State Profile {mode_label}")
    _logger.info("-" * 85)


    # 1. Input Data Diagnostics
    _logger.info(" [1] INPUT DATA DIAGNOSTICS")
    total_rows = len(market_hmm_feats)
    nan_count = market_hmm_feats.isna().sum().sum()
    zero_var_feats = [c for c in market_hmm_feats.columns if market_hmm_feats[c].std() < 1e-6]
    _logger.info(f"   - Total Samples:      {total_rows}")
    _logger.info(f"   - Total NaNs:         {nan_count}")
    _logger.info(f"   - Zero Var Features:  {zero_var_feats}")
    _logger.info("-" * 85)

    # 2. State Distribution & Financial Impact
    _logger.info(" [2] SEMANTIC STATE PROFILING & FINANCIAL IMPACT")
    cols = [c for c in HMM_SEMANTIC_PROB_COLUMNS if c in market_probs.columns]

    if cols and len(market_probs) == len(market_hmm_feats):
        # Align everything
        df_eval = market_probs[["datetime", *cols]].copy()
        df_eval["dominant_state"] = df_eval[cols].idxmax(axis=1)

        # Merge Modulator only (feat_tmp will supply btc_trend_vol_adj_24h to avoid _x/_y collision)
        mod_cols = ["datetime", "hmm_modulator_long", "hmm_modulator_short"]
        mod_tmp = hmm_modulator[[c for c in mod_cols if c in hmm_modulator.columns]]
        df_eval = pd.merge(df_eval, mod_tmp, on="datetime", how="left")

        # Merge Features
        feat_tmp = market_hmm_feats.copy().reset_index()
        feat_cols_to_merge = ["datetime", "btc_trend_vol_adj_24h", "realized_vol_regime"]
        df_eval = pd.merge(df_eval, feat_tmp[feat_cols_to_merge], on="datetime", how="left")

        # Calculate BTC 24h Forward Return if available
        has_return = False
        if btc_1h is not None and not btc_1h.empty:
            btc_tmp = btc_1h[["datetime", "close"]].copy()
            btc_tmp["datetime"] = pd.to_datetime(btc_tmp["datetime"], utc=True)
            btc_tmp["fwd_ret_24h"] = btc_tmp["close"].pct_change(24).shift(-24)
            df_eval = pd.merge(
                df_eval, btc_tmp[["datetime", "fwd_ret_24h"]], on="datetime", how="left"
            )
            has_return = True

        # Print Global Averages
        _logger.info("  [Global Averages]")
        means = market_probs[cols].mean()
        for c, m in means.items():
            _logger.info(f"   - {c:<25}: {m:.4f}")
        avg_mod_l = df_eval["hmm_modulator_long"].mean()
        avg_mod_s = df_eval["hmm_modulator_short"].mean()
        _logger.info(f"   - Global Modulator (L/S): {avg_mod_l:.2f} / {avg_mod_s:.2f}")
        _logger.info("-" * 40)

        # Print Per-State Profile
        _logger.info("  [Per-State Financial Profile]")
        for state in cols:
            g = df_eval[df_eval["dominant_state"] == state]
            pct = len(g) / len(df_eval) * 100
            if len(g) > 0:
                avg_trend = g["btc_trend_vol_adj_24h"].mean()
                avg_vol = g["realized_vol_regime"].mean()
                avg_mod_l = g["hmm_modulator_long"].mean()
                avg_mod_s = g["hmm_modulator_short"].mean()

                ret_str = "N/A"
                if has_return:
                    avg_ret = g["fwd_ret_24h"].mean() * 100
                    ret_str = f"{avg_ret:>+6.2f}%"

                st_name = state.replace("hmm_prob_", "").upper()
                _logger.info(f"   ► {st_name:<11} ({pct:>5.1f}% time)")
                _logger.info(
                    f"      Kelly Mod (L/S): {avg_mod_l:.2f} / {avg_mod_s:.2f} | "
                    f"24h Fwd Ret: {ret_str}"
                )
                _logger.info(
                    f"      Avg Trend: {avg_trend:>+5.2f} | Avg Vol:     {avg_vol:.2f}"
                )
            else:
                st_name = state.replace("hmm_prob_", "").upper()
                _logger.info(f"   ► {st_name:<11} (  0.0% time)")

    _logger.info("-" * 85)
    _logger.info("-" * 85 + "\n")



def _build_panel_with_targets(
    data_maps: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    utils = CrossSectionalPipelineUtils()
    panel_df = utils.build_panel_df(data_maps, tf="1h")
    panel_df = utils.add_cross_sectional_features(panel_df)
    panel_df = utils.add_systemic_features(panel_df)
    impute_cols = [c for c in GP_ENGINEERED_FEATURE_NAMES if c in panel_df.columns]
    if impute_cols:
        panel_df = utils.cs_median_impute_panel(panel_df, impute_cols)
    raw_h = cfg.get("FUTURES_ML_GP_HORIZONS", (3, 6, 12, 24))
    default_h = (3, 6, 12, 24)
    h_src = raw_h if isinstance(raw_h, (list, tuple)) else default_h
    horizons = tuple(int(x) for x in h_src)
    panel_df["target"] = utils.create_multi_horizon_rank_targets(panel_df, horizons=horizons)
    return panel_df


def merge_ml_output_into_data_maps(
    ml_out: MLPipelineOutput,
    maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    *,
    log_tag: str = "",
) -> None:
    """Merge ML fusion columns from ml_out into each symbol's tf DataFrame in maps (in-place)."""
    for sym in symbols:
        if sym not in ml_out.meta_feature_frame_by_symbol:
            continue
        mff = ml_out.meta_feature_frame_by_symbol[sym].copy()
        mff["datetime"] = pd.to_datetime(mff["datetime"], utc=True)
        hmm_dyn = [c for c in HMM_SEMANTIC_PROB_COLUMNS if c in mff.columns]
        if not hmm_dyn:
            # Only include columns where the last part is a digit to avoid ValueError
            # (e.g., hmm_prob_0_x)
            hmm_candidates = [
                c for c in mff.columns
                if str(c).startswith("hmm_prob_") and str(c).split("_")[-1].isdigit()
            ]
            hmm_dyn = sorted(
                hmm_candidates,
                key=lambda x: int(str(x).split("_")[-1]),
            )
        ml_cols = [
            "datetime",
            "gp_alpha_00",
            "hmm_modulator",
            "hmm_modulator_long",
            "hmm_modulator_short",
            "hmm_modulator_base_long",
            "btc_trend_vol_adj_24h",
            "slot_rank_score",
            "ml_calib_prob",
            "ml_calib_prob_long",
            "ml_calib_prob_short",
            "xs_score_long",
            "xs_score_short",
            *hmm_dyn,
        ]
        # Ensure hmm_prob_crisis is there even if not in hmm_dyn, but only once
        if "hmm_prob_crisis" in mff.columns and "hmm_prob_crisis" not in ml_cols:
            ml_cols.append("hmm_prob_crisis")

        ml_cols = [c for c in ml_cols if c in mff.columns]
        ml_features = mff[ml_cols].copy()
        drop_cols = [c for c in ml_cols if c != "datetime"]
        if sym not in maps or tf not in maps[sym]:
            continue
        original_df = maps[sym][tf].copy()
        original_df["datetime"] = pd.to_datetime(original_df["datetime"], utc=True)

        # Aggressively drop any existing ML/HMM columns to prevent _x, _y suffixes
        reserved_patterns = [
            "gp_alpha_", "hmm_modulator", "ml_calib_prob", "xs_score_", "slot_rank_"
        ]
        exist_ml_cols = [
            c for c in original_df.columns
            if any(p in str(c) for p in reserved_patterns) or str(c).startswith("hmm_prob_")
        ]
        to_drop = list(set(drop_cols + exist_ml_cols))
        is_cols_to_drop = [c for c in to_drop if c in original_df.columns]

        if is_cols_to_drop:
            original_df = original_df.drop(columns=is_cols_to_drop)
        merged = pd.merge(original_df, ml_features, on="datetime", how="left")
        merged = merged.sort_values("datetime")
        if "gp_alpha_00" in merged.columns:
            nan_pct = float(merged["gp_alpha_00"].isna().mean() * 100.0)
            _logger.debug(
                " [MERGE%s] %s %s gp_alpha_00 NaN ratio: %.4f%%",
                log_tag,
                sym,
                tf,
                nan_pct,
            )
        maps[sym][tf] = merged.fillna(0.0)


def merge_ml_output_into_is_and_oos(
    ml_out: MLPipelineOutput,
    data_maps: dict[str, dict[str, Any]],
    oos_data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
) -> None:
    merge_ml_output_into_data_maps(ml_out, data_maps, symbols, tf, log_tag=" IS")
    merge_ml_output_into_data_maps(ml_out, oos_data_maps, symbols, tf, log_tag=" OOS")


def copy_data_maps_tf_clone(
    maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
) -> dict[str, dict[str, Any]]:
    """Shallow clone per symbol dict with a copied tf OHLCV frame (for WF leg isolation)."""
    out: dict[str, dict[str, Any]] = {}
    for sym in symbols:
        if sym not in maps or tf not in maps[sym]:
            continue
        inner = dict(maps[sym])
        inner[tf] = maps[sym][tf].copy(deep=False)
        out[sym] = inner
    return out


def run_hmm_fusion_for_is_end(
    symbols: list[str],
    tf: str,
    fetch_start: str,
    end: str,
    cfg: dict[str, Any],
    data_maps: dict[str, dict[str, Any]],
    prefetched_1h: dict[str, pd.DataFrame],
    panel_df: pd.DataFrame | None,
    alpha_panel: pd.DataFrame,
    is_end_date: str | pd.Timestamp,
    collector: DataCollector,
    workers: int = 4,
    n_jobs: int = 4,
    *,
    include_fusion: bool = True,
    summary_mode_label: str = "",
    prefetch_label_start: str | None = None,
) -> MLPipelineOutput:
    """Walk-forward leg anchor: retrain systemic HMM with is_end_date cutoff.

    GP alpha_panel frozen.
    When include_fusion=False (hmm_only preview), skips per-symbol fusion.
    """
    from src.domain.futures.ml_pipeline.feature_engineering import build_systemic_hmm_features

    if panel_df is None:
        panel_df = _build_panel_with_targets(data_maps, cfg)

    _logger.info("  --> Systemic HMM Inference (is_end=%s)...", is_end_date)

    market_hmm_feats = build_systemic_hmm_features(panel_df, None)
    if market_hmm_feats.index.tz is None:
        market_hmm_feats.index = market_hmm_feats.index.tz_localize("UTC")
    else:
        market_hmm_feats.index = market_hmm_feats.index.tz_convert("UTC")

    hmm_k = int(cfg.get("FUTURES_HMM_K_STATES", 4))
    hmm_inferrer = HMMStateInferrer(n_states=hmm_k)

    is_end_dt = pd.to_datetime(is_end_date)
    is_end_utc = (
        is_end_dt.tz_localize("UTC")
        if is_end_dt.tzinfo is None
        else is_end_dt.tz_convert("UTC")
    )
    is_end_idx_market = int((market_hmm_feats.index < is_end_utc).sum())

    market_probs = hmm_inferrer.fit_predict_systemic(
        market_hmm_feats,
        market_hmm_feats["btc_trend_vol_adj_24h"],
        is_end_idx=is_end_idx_market,
        symbol="Market",
        tf=tf,
    )
    market_probs = _ensure_datetime_column(market_probs)
    market_probs["datetime"] = pd.to_datetime(market_probs["datetime"], utc=True)
    btc_anchor = next((s for s in symbols if "BTC" in s), None)
    btc_1h = prefetched_1h.get(btc_anchor) if btc_anchor else None

    hmm_modulator = _hmm_modulator_kelly_values(
        market_probs,
        alpha_panel,
        is_end_utc,
        btc_1h,
        float(cfg.get("FUTURES_HMM_KELLY_SHRINKAGE", 0.4)),
        float(cfg.get("FUTURES_HMM_CRISIS_THRESHOLD", 0.7)),
        market_hmm_feats,
    )
    hmm_modulator["datetime"] = market_probs["datetime"]

    _print_hmm_summary(
        market_probs,
        market_hmm_feats,
        hmm_modulator,
        btc_1h,
        mode_label=summary_mode_label,
    )

    out = MLPipelineOutput()
    out.alpha_panel = alpha_panel
    if not include_fusion:
        return out

    label_start = prefetch_label_start or fetch_start
    syms_step4 = [s for s in symbols if s in data_maps]
    valid_alpha_symbols = alpha_panel.index.get_level_values("symbol").unique()
    valid_alpha_set = set(valid_alpha_symbols.tolist())
    need_1m = [s for s in syms_step4 if s in valid_alpha_set]

    def _prefetch_1m(s: str) -> tuple[str, pd.DataFrame | None]:
        try:
            d = collector.collect_1m_ohlcv(s, label_start, end)
            return (s, d if d is not None and len(d) >= 200 else None)
        except Exception:
            return (s, None)

    prefetch_workers = max(1, min(len(need_1m) or 1, 4))
    one_m_cache: dict[str, pd.DataFrame | None] = {}
    if need_1m:
        with ThreadPoolExecutor(max_workers=prefetch_workers) as ex:
            one_m_cache = {s: dm for s, dm in ex.map(_prefetch_1m, need_1m)}

    def _fusion_job(s: str) -> _Step4FusionOutcome:
        return _step4_fusion_one_symbol(
            s,
            tf,
            data_maps,
            prefetched_1h,
            alpha_by_sym,
            valid_alpha_set,
            market_probs,
            hmm_modulator,
            fetch_start,
            end,
            is_end_utc,
            one_m_cache.get(s),
            collector,
        )

    alpha_by_sym: dict[str, pd.DataFrame] = {
        s: alpha_panel.xs(s, level="symbol").reset_index()
        for s in syms_step4
        if s in valid_alpha_set
    }

    fusion_workers = max(1, min(len(syms_step4) or 1, max(workers, n_jobs, 2), 12))
    with ThreadPoolExecutor(max_workers=fusion_workers) as ex:
        fusion_results = list(ex.map(_fusion_job, syms_step4))

    for res in fusion_results:
        if res.error is not None:
            _logger.error("[%s] R-6 fusion failed: %s", res.sym, res.error)
            continue
        if res.aligned_tf is None or res.cp_long is None or res.cp_short is None:
            _logger.error("[%s] R-6 fusion returned empty.", res.sym)
            continue
        out.meta_feature_frame_by_symbol[res.sym] = res.aligned_tf
        out.calib_prob_long_by_symbol[res.sym] = res.cp_long
        out.calib_prob_short_by_symbol[res.sym] = res.cp_short
        out.calib_prob_by_symbol[res.sym] = res.cp_long

    return out


def run_ml_pipeline_for_universe(
    symbols: list[str],
    tf: str,
    fetch_start: str,
    end: str,
    cfg: dict[str, Any],
    workers: int = 4,
    n_jobs: int = 4,
    is_end_date: str | None = None,
    is_start_date: str | None = None,
    gp_only: bool = False,
    hmm_only: bool = False,
) -> MLPipelineOutput:
    """[Phase 2] Universal Cross-Sectional ML Pipeline."""
    _logger.info("  --> Initiating Universal Cross-Sectional ML Pipeline")


    collector = DataCollector()
    data_maps: dict[str, dict[str, Any]] = {}
    prefetched_1h: dict[str, pd.DataFrame] = {}

    # --- Step 1: Market-Wide Data Collection & Panel Building ---
    _logger.info("  --> Step 1: Panel Construction & Asset Screening (%d symbols)", len(symbols))


    for sym in symbols:
        try:
            df_tf = collector.collect_and_save(sym, tf, fetch_start, end)
            df_1h = collector.collect_and_save(sym, "1h", fetch_start, end)
            if df_tf is not None and df_1h is not None:
                # IMPORTANT: Merge funding and metrics before ML feature engineering
                df_tf = merge_funding_into_ohlcv(sym, df_tf, Path(FUTURES_DATA_DIR))
                df_tf = merge_metrics_into_ohlcv(sym, df_tf, Path(FUTURES_DATA_DIR))
                if tf != "1h":
                    df_1h = merge_funding_into_ohlcv(sym, df_1h, Path(FUTURES_DATA_DIR))
                    df_1h = merge_metrics_into_ohlcv(sym, df_1h, Path(FUTURES_DATA_DIR))
                else:
                    df_1h = df_tf

                try:
                    df_1h_enriched = _merge_gp_into_1h(df_1h)
                except Exception as e:
                    _logger.warning("[%s] GP feature merge failed: %s", sym, e)
                    df_1h_enriched = df_1h
                data_maps[sym] = {tf: df_tf, "1h": df_1h_enriched}
                prefetched_1h[sym] = df_1h_enriched
        except Exception as e:
            _logger.warning("[%s] Data fetch failed: %s", sym, e)

    if not data_maps:
        _logger.error("No data collected. Pipeline aborted.")
        return MLPipelineOutput()

    label_start = is_start_date or fetch_start
    if bool(cfg.get("FUTURES_ML_GP_USE_TBM_WEIGHT", True)):
        _logger.info("  --> Step 1b: Triple-Barrier Micro-Weighting (1m prefetch)")


        one_m_gp: dict[str, pd.DataFrame | None] = {}
        for sym in list(data_maps.keys()):
            try:
                one_m_gp[sym] = collector.collect_1m_ohlcv(sym, label_start, end)
            except Exception:
                one_m_gp[sym] = None
        for sym in list(data_maps.keys()):
            try:
                data_maps[sym]["1h"] = _attach_tbm_gp_weights(
                    sym,
                    data_maps[sym]["1h"],
                    label_start,
                    end,
                    collector,
                    one_m_gp.get(sym),
                )
            except Exception as e:
                _logger.warning("[%s] TBM weight attach failed: %s", sym, e)
        for sym in prefetched_1h:
            if sym in data_maps and "1h" in data_maps[sym]:
                prefetched_1h[sym] = data_maps[sym]["1h"]

    panel_df = _build_panel_with_targets(data_maps, cfg)
    raw_h = cfg.get("FUTURES_ML_GP_HORIZONS", (3, 6, 12, 24))
    default_h = (3, 6, 12, 24)
    h_src = raw_h if isinstance(raw_h, (list, tuple)) else default_h
    horizons = tuple(int(x) for x in h_src)

    # --- Step 2: Universal GP Model Training (Cross-Sectional IC) ---
    _logger.info("  --> Step 2: Training Cross-Sectional GP Alpha Model")


    if bool(cfg.get("FUTURES_ML_GP_NSGA2_ENABLED", False)) and not is_deap_available():
        _logger.warning(
            "FUTURES_ML_GP_NSGA2_ENABLED=True but `deap` is not installed; "
            "continuing with scalarized gplearn fitness (see gp_multiobjective.py)."
        )
    miner = GPAlphaMiner(
        population_size=int(cfg.get("FUTURES_ML_GP_POPULATION", 1000)),
        generations=int(cfg.get("FUTURES_ML_GP_GENERATIONS", 20)),
        n_jobs=n_jobs,
        target_horizons=horizons,
        parsimony_coefficient=float(cfg.get("FUTURES_ML_GP_PARSIMONY", 0.001)),
    )

    gp_cache = Path(FUTURES_CACHE_DIR) / "universal_cs_gp_v8.parquet"
    filter_opts = {
        "use_newey_west": bool(cfg.get("FUTURES_ML_IC_FILTER_USE_HAC", False)),
        "use_ewma_ic_stat": bool(cfg.get("FUTURES_ML_IC_FILTER_USE_EWMA", False)),
        "ewma_half_life": float(cfg.get("FUTURES_ML_IC_EWMA_HALF_LIFE", 540.0)),
        "symbol_balance_max": float(cfg.get("FUTURES_ML_IC_SYMBOL_BALANCE_MAX", 3.0)),
        "require_regime_gate": bool(cfg.get("FUTURES_ML_IC_REGIME_GATE", True)),
        "fdr_q": float(cfg.get("FUTURES_ML_IC_FDR_Q", 0.10)),
    }
    alpha_panel = miner.mine_alphas_cs(
        panel_df,
        cache_path=gp_cache,
        is_end_date=is_end_date,
        filter_options=filter_opts,
    )

    # --- Print GP IC Validation Results immediately after Step 2 ---
    best_fitness = alpha_panel.attrs.get("best_fitness", 0.0)
    filter_meta = alpha_panel.attrs.get("gp_alpha_filter", {})

    _logger.info("-" * 85)
    _logger.info(" [GP AUDIT] LightGBM IC Validation")
    _logger.info("-" * 85)

    _logger.info(f" IS Best Fitness (Composite ICIR): {best_fitness:.6f}")
    _logger.info(f" Alpha Components Tried:          {filter_meta.get('n_components', 0):.0f}")
    _logger.info(f" Alpha Components Surviving:       {filter_meta.get('n_surviving', 0):.0f}")
    neu_p = bool(filter_meta.get("neutralize_primary", 0))
    _logger.info(f" Primary Alpha Neutralized:       {neu_p}")

    _logger.info("-" * 85)
    _logger.info(" [DIAGNOSTICS] Elimination Breakdown (Why they failed):")
    _logger.info(f"   - Failed FDR (Stat. Luck):    {filter_meta.get('fail_fdr', 0):.0f}")
    _logger.info(f"   - Failed DSR (Risk/Reward):   {filter_meta.get('fail_dsr', 0):.0f}")
    _logger.info(f"   - Failed OOS (Overfit):       {filter_meta.get('fail_oos', 0):.0f}")
    _logger.info(f"   - Failed Half-Life (Noise):   {filter_meta.get('fail_half_life', 0):.0f}")
    _logger.info(f"   - Failed Symbol Balance:      {filter_meta.get('fail_sym_bal', 0):.0f}")
    _logger.info(f"   - Failed Regime Consistency:  {filter_meta.get('fail_regime', 0):.0f}")

    _logger.info("-" * 85)
    _logger.info(" [BEST ALPHA (gp_alpha_00) METRICS]")
    _logger.info(f"   - IS Mean IC:           {filter_meta.get('primary_is_mu', 0.0):.4f}")
    _logger.info(f"   - OOS Mean IC:          {filter_meta.get('primary_oos_mu', 0.0):.4f}")
    _logger.info(f"   - IC Half-Life:         {filter_meta.get('primary_half_life', 0.0):.1f} bars")
    _logger.info(f"   - Symbol IC Dispersion: {filter_meta.get('primary_sym_dispersion', 0.0):.2f}")
    _logger.info(f"   - Raw T-Stat:           {filter_meta.get('primary_t_stat', 0.0):.2f}")
    _logger.info("-" * 85)

    if best_fitness > 0.01:
        _logger.info(" [RESULT] Reasonable IC/Fitness detected. Track A is healthy.")
    else:
        _logger.warning(" [RESULT] Low Fitness detected. Check market volatility or features.")

    if gp_only:
        # Step 2 이후 즉시 종료
        _logger.info("  [SUCCESS] GP-Only analysis complete.")

        out = MLPipelineOutput()
        out.alpha_panel = alpha_panel
        return out

    _logger.info("  --> Step 3: HMM Regime Fusion & Meta-Labeling")


    out = run_hmm_fusion_for_is_end(
        list(data_maps.keys()),
        tf,
        fetch_start,
        end,
        cfg,
        data_maps,
        prefetched_1h,
        panel_df,
        alpha_panel,
        is_end_date or end,
        collector,
        workers,
        n_jobs,
        include_fusion=not hmm_only,
        summary_mode_label="(HMM-ONLY MODE)" if hmm_only else "",
        prefetch_label_start=is_start_date or fetch_start,
    )
    out.alpha_panel = alpha_panel
    if hmm_only:
        _logger.info("  [SUCCESS] Pipeline complete (HMM-only).")
        return out


    _logger.info("  [SUCCESS] Universal ML Pipeline processing complete.")

    out.alpha_panel = alpha_panel
    return out
