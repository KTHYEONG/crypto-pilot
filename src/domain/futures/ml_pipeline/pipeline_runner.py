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
from src.domain.futures.data_loader import (
    DataCollector,
    merge_funding_into_ohlcv,
    merge_metrics_into_ohlcv,
)
from src.domain.futures.ml_pipeline.alpha.miner import MLAlphaMiner
from src.domain.futures.ml_pipeline.features.cross_sectional import CrossSectionalPipelineUtils
from src.domain.futures.ml_pipeline.features.engineering import (
    ALPHA_ENGINEERED_FEATURE_NAMES,
    HMM_SEMANTIC_PROB_COLUMNS,
    build_gp_input_features,
)
from src.domain.futures.ml_pipeline.labels.meta_labeler import MetaLabeler
from src.domain.futures.ml_pipeline.labels.triple_barrier import label_triple_barrier
from src.domain.futures.ml_pipeline.regime.hmm_inferrer import HMMStateInferrer
from src.domain.futures.optimization.optimizer import SignalCalibrator

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
    hmm_report: dict[str, float] = field(default_factory=dict)
    market_probs: pd.DataFrame = field(default_factory=pd.DataFrame)


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


def _drop_ml_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Unify removal of existing ML/HMM columns to prevent merge collisions."""
    ml_reserved = [
        "ml_alpha_00", "ml_alpha_long", "ml_alpha_short",
        "hmm_modulator_long", "hmm_modulator_short",
        "slot_rank_score", "xs_score_long", "xs_score_short",
        "ml_calib_prob", "ml_calib_prob_long", "ml_calib_prob_short",
        "hmm_prob_bull_trend", "hmm_prob_bear_trend", "hmm_prob_chop", "hmm_prob_crisis"
    ]
    # Drop reserved patterns and any column starting with hmm_
    to_drop = [
        c for c in df.columns 
        if any(p in str(c) for p in ml_reserved) or str(c).startswith("hmm_")
    ]
    if to_drop:
        return df.drop(columns=to_drop)
    return df


_META_EXTRA_FEATS: tuple[str, ...] = (
    "funding_z_72",
    "realized_vol_yz_24",
    "vol_surface_24_168",
    "corr_btc_24",
    "vpin_proxy_12",
    "ret_vol_adj_24",
    "downside_jump_24",
)


def _meta_feature_column_names(wide_1h: pd.DataFrame) -> tuple[str, ...]:
    """Resolve which columns to use as features for the meta-labeler."""
    hmm_cols = _sorted_hmm_prob_columns(wide_1h)
    base: list[str] = []
    if "ml_alpha_00" in wide_1h.columns:
        base.append("ml_alpha_00")
    base.extend(hmm_cols)
    # 추가 피처: wide_1h에 실제 존재하는 것만 포함
    for c in _META_EXTRA_FEATS:
        if c in wide_1h.columns and c not in base:
            base.append(c)
    return tuple(base)


def _attach_tbm_gp_weights(
    sym: str,
    df_1h: pd.DataFrame,
    label_start: str,
    end: str,
    collector: DataCollector,
    df_1m: pd.DataFrame | None,
) -> pd.DataFrame:
    """Up-weight rows with clear triple-barrier (+1 / -1) hits for GP fitness."""
    out = df_1h.copy()
    lab = _try_tbm_labels_per_1h_row(sym, out, label_start, end, df_1m=df_1m, collector=collector)
    if lab is None or len(lab) != len(out):
        out["tbm_gp_weight"] = 1.0
        return out
    out["tbm_gp_weight"] = np.where(
        np.isfinite(lab) & (np.abs(lab) > 0.9), 1.5, 1.0
    ).astype(np.float64)
    return out


def _enrich_with_gp_features(df: pd.DataFrame, tf: str = "1h") -> pd.DataFrame:
    """Append GP microstructure / momentum columns to OHLCV (sorted by datetime)."""
    out = df.copy()
    if "open" not in out.columns:
        out["open"] = pd.to_numeric(out["close"], errors="coerce").shift(1).fillna(out["close"])
    w = out.sort_values("datetime").reset_index(drop=True)
    idx = pd.DatetimeIndex(pd.to_datetime(w["datetime"], utc=True))
    gp = build_gp_input_features(w.set_index(idx), tf=tf)
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

        if (
            len(cols) == 4
            and "hmm_prob_bull_trend" in cols
            and all(c in cols for c in ("hmm_prob_chop", "hmm_prob_bear_trend", "hmm_prob_crisis"))
        ):
            idx_bull = cols.index("hmm_prob_bull_trend")
            idx_chop = cols.index("hmm_prob_chop")
            idx_bear = cols.index("hmm_prob_bear_trend")
            idx_crisis = cols.index("hmm_prob_crisis")
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
            kelly_long[idx_bull] = float(max(0.0, kelly_long[idx_bull]))

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
        elif len(cols) == 5 and all(c in cols for c in HMM_SEMANTIC_PROB_COLUMNS):
            order_idx = [cols.index(c) for c in HMM_SEMANTIC_PROB_COLUMNS]
            x = np.arange(5, dtype=np.float64)
            y_l = np.array([kelly_long[i] for i in order_idx], dtype=np.float64)
            y_l_adj = ir.fit_transform(x, y_l)
            for j, i in enumerate(order_idx):
                kelly_long[i] = (1.0 - alpha_iso) * y_l[j] + alpha_iso * y_l_adj[j]
            for j in (0, 1):
                kelly_long[order_idx[j]] = float(max(0.0, kelly_long[order_idx[j]]))

            y_s = np.array(
                [
                    kelly_short[order_idx[2]],
                    kelly_short[order_idx[3]],
                    kelly_short[order_idx[0]],
                    kelly_short[order_idx[1]],
                    kelly_short[order_idx[4]],
                ],
                dtype=np.float64,
            )
            y_s_adj = ir.fit_transform(x, y_s)
            kelly_short[order_idx[2]] = (1.0 - alpha_iso) * y_s[0] + alpha_iso * y_s_adj[0]
            kelly_short[order_idx[3]] = (1.0 - alpha_iso) * y_s[1] + alpha_iso * y_s_adj[1]
            kelly_short[order_idx[0]] = (1.0 - alpha_iso) * y_s[2] + alpha_iso * y_s_adj[2]
            kelly_short[order_idx[1]] = (1.0 - alpha_iso) * y_s[3] + alpha_iso * y_s_adj[3]
            kelly_short[order_idx[4]] = (1.0 - alpha_iso) * y_s[4] + alpha_iso * y_s_adj[4]
    except Exception as e:
        _logger.warning("Isotonic Kelly adjustment failed: %s", e)

    return kelly_long, kelly_short


def _hmm_modulator_kelly_values(
    market_probs: pd.DataFrame,
    alpha_panel: pd.DataFrame,
    is_end_utc: pd.Timestamp,
    price_df_1h: pd.DataFrame | None,
    shrink: float,
    crisis_thr: float,
    market_hmm_feats: pd.DataFrame | None = None,
    precomputed_risk: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Posterior-weighted asymmetric modulators from IS Kelly; fwd returns from ``price_df_1h``."""
    cols = _sorted_hmm_prob_columns(market_probs)
    n = len(market_probs)
    if not cols or price_df_1h is None or len(price_df_1h) < 80:
        return pd.DataFrame({
            "hmm_modulator_long": np.full(n, 1.0, dtype=np.float64),
            "hmm_modulator_short": np.full(n, 1.0, dtype=np.float64),
            "hmm_modulator_base_long": np.full(n, 1.0, dtype=np.float64),
            "btc_trend_vol_adj_24h": np.zeros(n, dtype=np.float64),
        })

    k_states = len(cols)
    b = price_df_1h.sort_values("datetime").copy()
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

    # [REFACTORED] Natural Risk-Adjusted Scaling (Phase 8) - Components for Tunable Optimizer
    mod_long_base = 1.0 + float(shrink) * (p_mat @ k_long)
    mod_short_base = 1.0 + float(shrink) * (p_mat @ k_short)

    if precomputed_risk is not None:
        risk_multiplier, vol_scalar_clipped = precomputed_risk
    else:
        # Fallback to local calculation if not provided
        risk_multiplier, vol_scalar_clipped = _precompute_hmm_risk_components(
            market_probs, market_hmm_feats
        )

    mod_long = np.clip(mod_long_base * risk_multiplier * vol_scalar_clipped, 0.1, 2.5).astype(np.float64)
    mod_short = np.clip(mod_short_base * risk_multiplier * vol_scalar_clipped, 0.1, 2.5).astype(np.float64)

    # BTC trend for diagnostic report only
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

    return pd.DataFrame({
        "hmm_modulator_long": mod_long,
        "hmm_modulator_short": mod_short,
        "hmm_modulator_base_long": mod_long_base,
        "hmm_modulator_base_short": mod_short_base,
        "btc_trend_vol_adj_24h": trend_24h,
    })


def _precompute_hmm_risk_components(
    market_probs: pd.DataFrame,
    market_hmm_feats: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate market-wide risk components (multiplier & vol scalar) once."""
    n = len(market_probs)
    p_crisis = market_probs["hmm_prob_crisis"].to_numpy(dtype=np.float64)
    p_bear = (
        market_probs["hmm_prob_bear_trend"].to_numpy(dtype=np.float64)
        if "hmm_prob_bear_trend" in market_probs.columns
        else np.zeros(n)
    )
    p_bull = (
        market_probs["hmm_prob_bull_calm"].to_numpy(dtype=np.float64)
        if "hmm_prob_bull_calm" in market_probs.columns
        else np.zeros(n)
    )

    if market_hmm_feats is not None and "macro_vol_24h" in market_hmm_feats.columns:
        vol_ser = market_hmm_feats["macro_vol_24h"].reindex(market_probs["datetime"]).ffill().bfill().fillna(0.01)
        vol_1h = vol_ser.to_numpy(dtype=np.float64)
        expected_variance = (vol_1h * np.sqrt(8760))**2
    else:
        expected_variance = np.full(n, 0.25, dtype=np.float64)

    pos_var = expected_variance[expected_variance > 0]
    target_var = float(np.median(pos_var)) if pos_var.size > 0 else 0.25
    # risk_multiplier: 1.0 (neutral), >1.0 (risk-on), <1.0 (risk-off)
    risk_multiplier = 1.0 - (0.75 * p_crisis) - (0.4 * p_bear) + (0.2 * p_bull)
    
    # vol_scalar: Ratio of target variance to current expected variance
    vol_scalar = target_var / (expected_variance + 1e-6)
    vol_scalar_clipped = np.clip(vol_scalar, 0.5, 1.5)

    return risk_multiplier, vol_scalar_clipped

def _hmm_modulator_kelly_per_symbol(
    market_probs: pd.DataFrame,
    alpha_panel: pd.DataFrame,
    is_end_utc: pd.Timestamp,
    prefetched_1h: dict[str, pd.DataFrame],
    universe_syms: list[str],
    shrink: float,
    crisis_thr: float,
    market_hmm_feats: pd.DataFrame | None,
    anchor_symbol: str | None,
) -> dict[str, pd.DataFrame]:
    """One Kelly modulator per symbol using that symbol's 1h OHLC fwd returns; fallback = anchor (e.g. BTC)."""
    # [Precomputation] Calculate shared risk components once
    shared_risk = _precompute_hmm_risk_components(market_probs, market_hmm_feats)
    
    anchor_df = prefetched_1h.get(anchor_symbol) if anchor_symbol else None
    fallback = _hmm_modulator_kelly_values(
        market_probs,
        alpha_panel,
        is_end_utc,
        anchor_df,
        shrink,
        crisis_thr,
        market_hmm_feats,
        precomputed_risk=shared_risk,
    )
    out: dict[str, pd.DataFrame] = {}

    def _calc_one(sym: str) -> tuple[str, pd.DataFrame]:
        df = prefetched_1h.get(sym)
        if (
            df is not None
            and len(df) >= 80
            and "close" in df.columns
            and "high" in df.columns
            and "low" in df.columns
        ):
            val = _hmm_modulator_kelly_values(
                market_probs,
                alpha_panel,
                is_end_utc,
                df,
                shrink,
                crisis_thr,
                market_hmm_feats,
                precomputed_risk=shared_risk,
            )
            return sym, val
        return sym, fallback

    # [Institutional Quant] Parallel execution for symbol-specific modulators
    with ThreadPoolExecutor(max_workers=min(len(universe_syms), 8)) as executor:
        results = list(executor.map(_calc_one, universe_syms))

    for s, v in results:
        out[s] = v

    return out


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


def _platt_calib_probs_wf(
    aligned_tf: pd.DataFrame,
    gp_base: np.ndarray,
    gp_long: np.ndarray,
    gp_short: np.ndarray,
    is_end_utc: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    """Platt on IS (alpha vs forward return), then predict p(win) for long/short scores.

    Matches ``precompute_ml_optimization_context`` / ``SignalCalibrator`` usage so WF OOS
    backtests agree with Phase-D precompute when MetaLabeler is off or fails.
    """
    horizon = int(OPT_FUTURES_CONFIG.get("FUTURES_ML_ALPHA_TARGET_HORIZON", 12))
    min_is = 80
    _close_col = "close" if "close" in aligned_tf.columns else "close_x"
    close = aligned_tf[_close_col].astype(np.float64)
    fwd_ret = close.pct_change(horizon).shift(-horizon).to_numpy(dtype=np.float64)
    dt = pd.to_datetime(aligned_tf["datetime"], utc=True)
    is_mask = (dt < is_end_utc).to_numpy()
    valid = is_mask & np.isfinite(gp_base) & np.isfinite(fwd_ret)

    calib = SignalCalibrator()
    if int(valid.sum()) >= min_is:
        y_bin = (fwd_ret[valid] > 0.0001).astype(int)
        if len(np.unique(y_bin)) > 1:
            calib.fit(gp_base[valid], fwd_ret[valid])

    p_long = np.clip(calib.predict_prob(gp_long.astype(np.float64, copy=False)), 0.0, 1.0)
    p_short = np.clip(calib.predict_prob(gp_short.astype(np.float64, copy=False)), 0.0, 1.0)
    return p_long.astype(np.float64), p_short.astype(np.float64)


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
    """Cross-sectional scores (gp x HMM); ml_calib_* via MetaLabeler or Platt (IS WF)."""
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
    gp_base = (
        aligned_tf["ml_alpha_00"].to_numpy(dtype=np.float64)
        if "ml_alpha_00" in aligned_tf.columns
        else np.zeros(len(aligned_tf), dtype=np.float64)
    )
    gp_long = (
        aligned_tf["ml_alpha_long"].to_numpy(dtype=np.float64)
        if "ml_alpha_long" in aligned_tf.columns
        else gp_base
    )
    gp_short = (
        aligned_tf["ml_alpha_short"].to_numpy(dtype=np.float64)
        if "ml_alpha_short" in aligned_tf.columns
        else gp_base
    )
    # [Improvement 1] Friction-Aware EV Hurdle
    hurdle_ratio = float(OPT_FUTURES_CONFIG.get("FUTURES_ML_EV_HURDLE_RATIO", 0.0))
    if hurdle_ratio > 0:
        from config.settings import SLIPPAGE_RATE, TRADING_FEE_RATE
        # Round-trip cost (Taker fee + Slippage) * 2
        rt_cost = (TRADING_FEE_RATE + SLIPPAGE_RATE) * 2.0
        # Heuristic: 0.1% expected return corresponds to ~0.02 deviation from 0.5 in rank space.
        score_hurdle = hurdle_ratio * rt_cost * 10.0
        
        # Zero out signals that don't clear the hurdle
        gp_long_mask = (gp_long > (0.5 + score_hurdle)).astype(np.float64)
        gp_short_mask = (gp_short < (0.5 - score_hurdle)).astype(np.float64)
        
        aligned_tf["xs_score_long"] = gp_long * hmm_m_long * gp_long_mask
        aligned_tf["xs_score_short"] = (gp_short / np.maximum(hmm_m_short, 0.1)) * gp_short_mask + (1.0 * (1.0 - gp_short_mask))
    else:
        aligned_tf["xs_score_long"] = gp_long * hmm_m_long
        aligned_tf["xs_score_short"] = gp_short / np.maximum(hmm_m_short, 0.1)

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
    used_meta = False

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
                used_meta = True
            except Exception as exc:
                _logger.debug("MetaLabeler WF refit skipped: %s", exc)

    if not used_meta:
        pl, ps = _platt_calib_probs_wf(aligned_tf, gp_base, gp_long, gp_short, is_end_utc)

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
        _logger.debug("[%s] Step 4 Fusion started.", sym)
        df_1h = prefetched_1h[sym].copy()
        df_1h["datetime"] = pd.to_datetime(df_1h["datetime"], utc=True)

        # [Optimization #11] Unify column drop logic
        df_1h = _drop_ml_columns(df_1h)
        
        hmm_cols_ref = _sorted_hmm_prob_columns(market_probs)
        k_fb = len(hmm_cols_ref) if hmm_cols_ref else len(HMM_SEMANTIC_PROB_COLUMNS)

        if sym not in valid_alpha_set:
            wide_1h = df_1h.copy()
            wide_1h["ml_alpha_00"] = 0.5
            # [REFACTORED] Merge asymmetric modulators
            wide_1h = pd.merge(wide_1h, hmm_modulator, on="datetime", how="left")
            # [Institutional Quant] Dual-TF Persistence: ffill 4h modulators into 1h bars
            mod_cols = [c for c in hmm_modulator.columns if c != "datetime"]
            wide_1h[mod_cols] = wide_1h[mod_cols].ffill().fillna(1.0)
            
            wide_1h["slot_rank_score"] = 0.0

            # [NEW] Merge all HMM related columns
            hmm_cols_all = [c for c in market_probs.columns if str(c).startswith("hmm_")]
            if hmm_cols_all:
                mp_h = market_probs[["datetime", *hmm_cols_all]]
                wide_1h = wide_1h.merge(mp_h, on="datetime", how="left")
                # [Institutional Quant] Dual-TF Persistence: ffill 4h HMM states into 1h bars
                wide_1h[hmm_cols_all] = wide_1h[hmm_cols_all].ffill()
                
                # Fill probabilities with uniform, others with 0
                prob_cols = [c for c in hmm_cols_all if "prob_" in c]
                other_hmm_cols = [c for c in hmm_cols_all if "prob_" not in c]
                wide_1h[prob_cols] = wide_1h[prob_cols].fillna(1.0 / float(k_fb))
                wide_1h[other_hmm_cols] = wide_1h[other_hmm_cols].fillna(0.0)
            else:
                for c in HMM_SEMANTIC_PROB_COLUMNS:
                    wide_1h[c] = 1.0 / float(len(HMM_SEMANTIC_PROB_COLUMNS))
        else:
            sym_alpha = alpha_by_sym[sym].copy()
            sym_alpha["datetime"] = pd.to_datetime(sym_alpha["datetime"], utc=True)
            if "ml_alpha_00" not in sym_alpha.columns:
                _logger.warning(
                    "[%s] ml_alpha_00 missing in sym_alpha. cols=%s",
                    sym, list(sym_alpha.columns)
                )
                sym_alpha["ml_alpha_00"] = 0.5

            wide_1h = pd.merge(df_1h, sym_alpha, on="datetime", how="left")
            # [Institutional Quant] Dual-TF Alpha persistence: 
            # If sym_alpha is 4h and df_1h is 1h, ffill to maintain the thesis across bars.
            wide_1h["ml_alpha_00"] = wide_1h["ml_alpha_00"].ffill().fillna(0.5)

            # [REFACTORED] Merge asymmetric modulators
            wide_1h = pd.merge(wide_1h, hmm_modulator, on="datetime", how="left")
            # [Institutional Quant] Dual-TF Persistence: ffill 4h modulators into 1h bars
            mod_cols = [c for c in hmm_modulator.columns if c != "datetime"]
            wide_1h[mod_cols] = wide_1h[mod_cols].ffill().fillna(1.0)

            # Use mean of long/short modulator for ranking score
            # m_avg = (wide_1h["hmm_modulator_long"] + wide_1h["hmm_modulator_short"]) / 2.0
            wide_1h["slot_rank_score"] = wide_1h["ml_alpha_00"]

            # [NEW] Merge all HMM related columns
            hmm_cols_all = [c for c in market_probs.columns if str(c).startswith("hmm_")]
            if hmm_cols_all:
                mp_h = market_probs[["datetime", *hmm_cols_all]]
                wide_1h = wide_1h.merge(mp_h, on="datetime", how="left")
                # [Institutional Quant] Dual-TF Persistence: ffill 4h HMM states into 1h bars
                wide_1h[hmm_cols_all] = wide_1h[hmm_cols_all].ffill()

                # Fill probabilities with uniform, others with 0
                prob_cols = [c for c in hmm_cols_all if "prob_" in c]
                other_hmm_cols = [c for c in hmm_cols_all if "prob_" not in c]
                wide_1h[prob_cols] = wide_1h[prob_cols].fillna(1.0 / float(k_fb))
                wide_1h[other_hmm_cols] = wide_1h[other_hmm_cols].fillna(0.0)
            else:
                for c in HMM_SEMANTIC_PROB_COLUMNS:
                    wide_1h[c] = 1.0 / float(len(HMM_SEMANTIC_PROB_COLUMNS))

        df_tf_full = data_maps[sym][tf].copy()
        df_tf_full["datetime"] = pd.to_datetime(df_tf_full["datetime"], utc=True)

        # [Optimization #11] Unify column drop logic
        df_tf_full = _drop_ml_columns(df_tf_full)

        # Drop OHLCV columns from wide_1h to prevent _x/_y suffix collision when TF != 1h.
        # We keep the trading-TF OHLCV from df_tf_full; 1h OHLCV is not needed post-merge.
        _ohlcv_base = {"open", "high", "low", "close", "volume"}
        wide_1h_for_merge = wide_1h.drop(
            columns=[c for c in _ohlcv_base if c in wide_1h.columns and c in df_tf_full.columns],
        )

        aligned_tf = pd.merge(df_tf_full, wide_1h_for_merge, on="datetime", how="left").fillna(0.0)

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
) -> dict[str, float]:
    """Print institutional-grade audit of HMM states and return report metrics."""
    _logger.info(" ┌───────────────────────────────────────────────────────────────────────────────────┐")
    _logger.info(f" │ [HMM INSTITUTIONAL AUDIT] Log-Wealth & Tail-Risk Analysis {mode_label:<24} │")
    _logger.info(" ├───────────────────────────────────────────────────────────────────────────────────┤")

    report: dict[str, float] = {}
    cols = [c for c in HMM_SEMANTIC_PROB_COLUMNS if c in market_probs.columns]
    if not (cols and len(market_probs) == len(market_hmm_feats)):
        _logger.warning(" │ [WARN] HMM data alignment failed for audit.                                    │")
        _logger.info(" └───────────────────────────────────────────────────────────────────────────────────┘")
        return report

    df_eval = market_probs[["datetime", *cols]].copy()
    df_eval["dominant_state"] = df_eval[cols].idxmax(axis=1)

    mod_cols = ["datetime", "hmm_modulator_long", "hmm_modulator_short"]
    mod_tmp = hmm_modulator[[c for c in mod_cols if c in hmm_modulator.columns]]
    if not mod_tmp.empty and "datetime" in mod_tmp.columns:
        df_eval = pd.merge(df_eval, mod_tmp, on="datetime", how="left")
    else:
        # Add placeholder columns if modulators are missing
        df_eval["hmm_modulator_long"] = 1.0
        df_eval["hmm_modulator_short"] = 1.0

    feat_tmp = market_hmm_feats.copy().reset_index()
    # [v15 Alignment] Use new macro features for diagnostic merge
    target_feats = ["macro_vol_24h", "macro_cost_168h"]
    feat_cols_to_merge = ["datetime"] + [f for f in target_feats if f in feat_tmp.columns]
    df_eval = pd.merge(df_eval, feat_tmp[feat_cols_to_merge], on="datetime", how="left")

    if btc_1h is not None and not btc_1h.empty:
        btc_tmp = btc_1h[["datetime", "close"]].copy()
        btc_tmp["datetime"] = pd.to_datetime(btc_tmp["datetime"], utc=True)
        btc_tmp["ret"] = btc_tmp["close"].pct_change().fillna(0.0)
        df_eval = pd.merge(df_eval, btc_tmp[["datetime", "ret"]], on="datetime", how="left")

    # 1. Log-Wealth Dispersion
    _logger.info(" │ [A] LOG-WEALTH DISPERSION (g = mu - 0.5*sigma^2)                                 │")
    _logger.info(" ├────────────┬────────┬─────────────┬───────────┬───────────┬───────────┬──────────┤")
    _logger.info(" │ REGIME     │ TIME % │ MOD (L / S) │ MU (%)    │ SIG (%)   │ G_LOG (%) │ VERDICT  │")
    _logger.info(" ├────────────┼────────┼─────────────┼───────────┼───────────┼───────────┼──────────┤")

    for state in cols:
        g_df = df_eval[df_eval["dominant_state"] == state]
        pct = len(g_df) / len(df_eval) * 100
        st_name = state.replace("hmm_prob_", "").upper()
        report[state] = pct

        if len(g_df) > 0:
            m_l = float(g_df["hmm_modulator_long"].mean())
            m_s = float(g_df["hmm_modulator_short"].mean())
            
            mu = 0.0
            sig = 0.0
            g_log = 0.0
            if "ret" in g_df.columns:
                mu = float(g_df["ret"].mean() * 100.0)
                sig = float(g_df["ret"].std() * 100.0)
                # Geometric Growth Approximation in % units: mu - 0.5 * (sig^2 / 100)
                g_log = mu - 0.5 * (sig**2 / 100.0)
            
            verdict = "CHOP"
            if g_log > 0.015:
                verdict = "BULL"
            elif g_log < -0.04:
                verdict = "CRISIS"
            elif g_log < -0.01:
                verdict = "BEAR"
            
            if st_name == "CRISIS": report["hmm_crisis_g_log"] = g_log
            if st_name == "BULL_TREND": report["hmm_bull_g_log"] = g_log

            _logger.info(f" │ {st_name:<10} │ {pct:>5.1f}% │ {m_l:.2f} / {m_s:.2f} │ {mu:>9.3f} │ {sig:>9.3f} │ {g_log:>9.3f} │ {verdict:<8} │")
        else:
            _logger.info(f" │ {st_name:<10} │   0.0% │ ---- / ---- │      ---- │      ---- │      ---- │ -------- │")

    # 2. Tail Capture
    if "ret" in df_eval.columns:
        _logger.info(" ├────────────┴────────┴─────────────┴───────────┴───────────┴───────────┴──────────┤")
        q05 = float(df_eval["ret"].quantile(0.05))
        worst_mask = df_eval["ret"] <= q05
        worst_df = df_eval[worst_mask]
        if not worst_df.empty:
            capture = float((worst_df["dominant_state"].isin(["hmm_prob_crisis", "hmm_prob_bear_trend"])).mean() * 100.0)
            report["hmm_tail_capture"] = capture
            _logger.info(f" │ [B] LEFT-TAIL CAPTURE (Worst 5%): {capture:>5.1f}% Caught in CRISIS/BEAR {' '*20} │")

    # 3. Friction & Stability
    switches = int((df_eval["dominant_state"] != df_eval["dominant_state"].shift(1)).sum())
    avg_dur = float(len(df_eval) / max(1, switches))
    report["hmm_switches"] = float(switches)
    report["hmm_avg_duration"] = avg_dur
    _logger.info(f" │ [C] STABILITY: {switches} Switches | Avg Duration: {avg_dur:>6.1f} bars {' '*31} │")
    _logger.info(" └───────────────────────────────────────────────────────────────────────────────────┘\n")

    return report



def _build_panel_with_targets(
    data_maps: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    *,
    skip_targets: bool = False,
) -> pd.DataFrame:
    utils = CrossSectionalPipelineUtils()
    panel_df = utils.build_panel_df(data_maps, tf="1h")
    
    if skip_targets:
        # For HMM inference, we only need systemic features (market breadth, dispersion)
        # We can completely skip the expensive cross-sectional Z-scoring and imputation
        panel_df = utils.add_systemic_features(panel_df)
        return panel_df

    panel_df = utils.add_cross_sectional_features(panel_df)
    
    # [Neutralization] Apply Cross-Sectional Z-Score (Mean/Std) to relevant features
    # to remove market beta and ensure relative strength evaluation.
    cs_neutral_cols = [
        "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", 
        "realized_vol_yz_24", "vol_ratio_24", "vol_ratio_168",
        "mom_proxy_12", "acceleration_24", "beta_neutral_momentum"
    ]
    panel_df = utils.apply_cs_zscore(panel_df, cs_neutral_cols)
    
    panel_df = utils.add_systemic_features(panel_df)
    impute_cols = [c for c in ALPHA_ENGINEERED_FEATURE_NAMES if c in panel_df.columns]
    if impute_cols:
        panel_df = utils.cs_median_impute_panel(panel_df, impute_cols)

    raw_h = cfg.get("FUTURES_ML_ALPHA_HORIZONS", (3, 6, 12, 24))
    default_h = (3, 6, 12, 24)
    h_src = raw_h if isinstance(raw_h, (list, tuple)) else default_h
    horizons = tuple(int(x) for x in h_src)
    _ic_hl = float(OPT_FUTURES_CONFIG.get("FUTURES_ML_IC_HALF_LIFE", 2.3))
    _h_weights = tuple(float(np.exp(-h / _ic_hl)) for h in horizons)
    panel_df["target"] = utils.create_multi_horizon_rank_targets(panel_df, horizons=horizons, weights=_h_weights)
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
    # [Optimization #12] Parallelize per-symbol merge
    def _merge_one(sym: str) -> None:
        if sym not in ml_out.meta_feature_frame_by_symbol:
            return
        mff = ml_out.meta_feature_frame_by_symbol[sym].copy()
        mff["datetime"] = pd.to_datetime(mff["datetime"], utc=True)
        
        hmm_cols_in_mff = [c for c in mff.columns if str(c).startswith("hmm_")]
        ml_cols = [
            "datetime", "ml_alpha_00", "ml_alpha_long", "ml_alpha_short",
            "btc_trend_vol_adj_24h", "hmm_modulator_long", "hmm_modulator_short",
            "slot_rank_score", "ml_calib_prob", "ml_calib_prob_long", "ml_calib_prob_short",
            "xs_score_long", "xs_score_short", *hmm_cols_in_mff,
        ]
        for c in _META_EXTRA_FEATS:
            if c not in ml_cols: ml_cols.append(c)

        unique_ml_cols = []
        seen = set()
        for x in ml_cols:
            if x in mff.columns and x not in seen:
                unique_ml_cols.append(x)
                seen.add(x)
        
        if sym not in maps or tf not in maps[sym]:
            return
            
        ml_features = mff[unique_ml_cols].copy()
        original_df = maps[sym][tf].copy()
        original_df["datetime"] = pd.to_datetime(original_df["datetime"], utc=True)

        # Optimization #11
        original_df = _drop_ml_columns(original_df)
        
        # Optimization #12: Use sort=True inside merge instead of separate sort_values
        merged = pd.merge(original_df, ml_features, on="datetime", how="left", sort=True)

        # Forward fill for Dual-TF
        ml_non_dt_cols = [c for c in unique_ml_cols if c != "datetime" and c in merged.columns]
        if ml_non_dt_cols:
            merged[ml_non_dt_cols] = merged[ml_non_dt_cols].ffill()

        if "ml_alpha_00" not in merged.columns:
            merged["ml_alpha_00"] = 0.5
        else:
            merged["ml_alpha_00"] = merged["ml_alpha_00"].fillna(0.5)
            
        # [Institutional Quant] Prevent total flattening of signal by fillna(0.0)
        # We fill probabilities with uniform, modulators with 1.0, others with 0
        p_cols = [c for c in merged.columns if "prob_" in c]
        mod_cols = [c for c in merged.columns if "modulator" in c]
        k_fb_local = len(p_cols) if p_cols else 5
        merged[p_cols] = merged[p_cols].fillna(1.0 / float(k_fb_local))
        merged[mod_cols] = merged[mod_cols].fillna(1.0)
        
        maps[sym][tf] = merged.fillna(0.0)

    workers = min(len(symbols), 12)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(_merge_one, symbols))


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
    prefetched_1m: dict[str, pd.DataFrame] | None = None,
    prefetched_market_probs: pd.DataFrame | None = None,
    prefetched_market_hmm_feats: pd.DataFrame | None = None,
) -> MLPipelineOutput:
    """Walk-forward leg anchor: retrain systemic HMM with is_end_date cutoff.

    GP alpha_panel frozen.
    When include_fusion=False (hmm_only preview), skips per-symbol fusion.
    """
    from src.domain.futures.ml_pipeline.features.engineering import build_systemic_hmm_features

    is_end_dt = pd.to_datetime(is_end_date)
    is_end_utc = is_end_dt.tz_localize("UTC") if is_end_dt.tzinfo is None else is_end_dt.tz_convert("UTC")

    if prefetched_market_probs is not None and prefetched_market_hmm_feats is not None:
        _logger.info("  --> Reusing precomputed HMM results for fusion.")
        market_probs = prefetched_market_probs
        market_hmm_feats = prefetched_market_hmm_feats
    else:
        if panel_df is None:
            if prefetched_1h:
                tmp_maps = {sym: {"1h": df} for sym, df in prefetched_1h.items()}
                panel_df = _build_panel_with_targets(tmp_maps, cfg, skip_targets=True)
            else:
                panel_df = _build_panel_with_targets(data_maps, cfg, skip_targets=True)
        
        _logger.info("  --> Systemic HMM Inference (is_end=%s)%s...", is_end_date, summary_mode_label)
        market_hmm_feats = build_systemic_hmm_features(panel_df, None, tf="1h")
        if market_hmm_feats.index.tz is None:
            market_hmm_feats.index = market_hmm_feats.index.tz_localize("UTC")
        else:
            market_hmm_feats.index = market_hmm_feats.index.tz_convert("UTC")

        hmm_k = int(cfg.get("FUTURES_HMM_K_STATES", 5))
        # [Optimization #6] Reduce n_iter and add convergence tolerance
        hmm_n_iter = int(cfg.get("FUTURES_HMM_N_ITER", 200))
        hmm_inferrer = HMMStateInferrer(n_states=hmm_k, n_iter=hmm_n_iter, tol=1e-4)
        
        is_end_idx_market = int((market_hmm_feats.index < is_end_utc).sum())
        _btc_anchor = next((s for s in symbols if "BTC" in s), None)
        _btc_df = prefetched_1h.get(_btc_anchor) if _btc_anchor else None
        if _btc_df is not None and "close" in _btc_df.columns:
            _btc_rets = _btc_df.set_index("datetime")["close"].pct_change().reindex(market_hmm_feats.index).fillna(0.0)
        else:
            _btc_rets = market_hmm_feats["macro_trend_168h"]

        market_probs = hmm_inferrer.fit_predict_systemic(
            market_hmm_feats, _btc_rets, is_end_idx=is_end_idx_market, symbol="Market", tf=tf
        )
        market_probs = _ensure_datetime_column(market_probs)
        market_probs["datetime"] = pd.to_datetime(market_probs["datetime"], utc=True)

    btc_anchor = next((s for s in symbols if "BTC" in s), None)
    btc_1h = prefetched_1h.get(btc_anchor) if btc_anchor else None

    hmm_modulator_by_sym = _hmm_modulator_kelly_per_symbol(
        market_probs,
        alpha_panel,
        is_end_utc,
        prefetched_1h,
        symbols,
        float(cfg.get("FUTURES_HMM_KELLY_SHRINKAGE", 0.4)),
        float(cfg.get("FUTURES_HMM_CRISIS_THRESHOLD", 0.7)),
        market_hmm_feats,
        btc_anchor,
    )
    dt_series = market_probs["datetime"]
    for _mod_df in hmm_modulator_by_sym.values():
        _mod_df["datetime"] = dt_series

    hmm_modulator_audit = (
        hmm_modulator_by_sym[btc_anchor]
        if btc_anchor and btc_anchor in hmm_modulator_by_sym
        else (next(iter(hmm_modulator_by_sym.values())) if hmm_modulator_by_sym else pd.DataFrame())
    )

    h_rep = _print_hmm_summary(
        market_probs,
        market_hmm_feats,
        hmm_modulator_audit,
        btc_1h,
        mode_label=summary_mode_label,
    )

    out = MLPipelineOutput()
    out.hmm_report = h_rep
    out.alpha_panel = alpha_panel
    out.market_probs = market_probs
    if not include_fusion:
        return out

    label_start = prefetch_label_start or fetch_start
    syms_step4 = [s for s in symbols if s in data_maps]
    _logger.info("    --> [STEP 4] Fusing %d symbols...", len(syms_step4))
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
    
    # [Fix] Use passed cfg instead of global OPT_FUTURES_CONFIG
    meta_on_any = bool(cfg.get("FUTURES_USE_META_LABELER", False))

    if prefetched_1m is not None:
        # Use existing memory cache if provided (Smart bypass)
        one_m_cache = prefetched_1m
    elif need_1m and meta_on_any:
        with ThreadPoolExecutor(max_workers=prefetch_workers) as ex:
            one_m_cache = {s: dm for s, dm in ex.map(_prefetch_1m, need_1m)}

    def _fusion_job(s: str) -> _Step4FusionOutcome:
        mod_df = hmm_modulator_by_sym.get(s, hmm_modulator_audit)
        return _step4_fusion_one_symbol(
            s,
            tf,
            data_maps,
            prefetched_1h,
            alpha_by_sym,
            valid_alpha_set,
            market_probs,
            mod_df,
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

    fusion_workers = max(1, min(len(syms_step4) or 1, max(workers, n_jobs), 12))
    if fusion_workers > 1:
        with ThreadPoolExecutor(max_workers=fusion_workers) as ex:
            fusion_results = list(ex.map(_fusion_job, syms_step4))
    else:
        fusion_results = [_fusion_job(s) for s in syms_step4]

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
    fetch_start_date: str,
    end: str,
    cfg: dict[str, Any],
    workers: int = 4,
    n_jobs: int = 4,
    is_end_date: str | None = None,
    is_start_date: str | None = None,
    gp_only: bool = False,
    hmm_only: bool = False,
    preloaded_data_maps: dict[str, dict[str, Any]] | None = None,
    preloaded_1h_maps: dict[str, pd.DataFrame] | None = None,
) -> MLPipelineOutput:
    """[Phase 2] Universal Cross-Sectional ML Pipeline.
    
    Logical Flow:
    1. Data Collection & Panel Construction (TF-aware)
    2. Systemic HMM Inference (Regime Discovery)
    3. Alpha Mining (Regime-Aware LightGBM/GP)
    4. Signal Fusion & Meta-Labeling (Unified Meta-Feature Frame)
    """
    _logger.info("  --> Initiating Universal Cross-Sectional ML Pipeline (TF: %s)", tf)

    collector = DataCollector()
    data_maps: dict[str, dict[str, Any]] = preloaded_data_maps or {}
    prefetched_1h: dict[str, pd.DataFrame] = preloaded_1h_maps or {}

    # --- Step 1: Market-Wide Data Collection & Enrichment ---
    _logger.info("  --> Step 1: Panel Construction & Asset Screening (%d symbols)", len(symbols))
    
    missing_any = False
    for sym in symbols:
        # Check if enrichment is needed for target TF
        needs_enrich = False
        if sym in data_maps and tf in data_maps[sym]:
            df_check = data_maps[sym][tf]
            if "ret_1" not in df_check.columns:
                needs_enrich = True
        else:
            needs_enrich = True

        # Check if enrichment is needed for 1h reference
        if tf != "1h" and sym not in prefetched_1h:
            needs_enrich = True
        elif tf != "1h" and sym in prefetched_1h:
            if "ret_1" not in prefetched_1h[sym].columns:
                needs_enrich = True

        if needs_enrich:
            try:
                # Fetch if missing from maps
                if sym not in data_maps or tf not in data_maps[sym]:
                    df_tf = collector.collect_and_save(sym, tf, fetch_start_date, end)
                    if df_tf is not None:
                        df_tf = merge_funding_into_ohlcv(sym, df_tf, Path(FUTURES_DATA_DIR))
                        df_tf = merge_metrics_into_ohlcv(sym, df_tf, Path(FUTURES_DATA_DIR))
                        data_maps.setdefault(sym, {})[tf] = df_tf
                
                if tf != "1h" and sym not in prefetched_1h:
                    df_1h = collector.collect_and_save(sym, "1h", fetch_start_date, end)
                    if df_1h is not None:
                        df_1h = merge_funding_into_ohlcv(sym, df_1h, Path(FUTURES_DATA_DIR))
                        df_1h = merge_metrics_into_ohlcv(sym, df_1h, Path(FUTURES_DATA_DIR))
                        prefetched_1h[sym] = df_1h

                # Enrich if not in hmm_only mode
                if not hmm_only:
                    if sym in data_maps and tf in data_maps[sym]:
                        if "ret_1" not in data_maps[sym][tf].columns:
                            data_maps[sym][tf] = _enrich_with_gp_features(data_maps[sym][tf], tf=tf)
                    
                    if tf != "1h" and sym in prefetched_1h:
                        if "ret_1" not in prefetched_1h[sym].columns:
                            prefetched_1h[sym] = _enrich_with_gp_features(prefetched_1h[sym], tf="1h")
                    elif tf == "1h" and sym in data_maps and "1h" in data_maps[sym]:
                        prefetched_1h[sym] = data_maps[sym]["1h"]

            except Exception as e:
                _logger.warning("[%s] Data fetch/enrich failed: %s", sym, e)
        else:
            # Already enriched, just ensure prefetched_1h is synced for 1h mode
            if tf == "1h" and sym not in prefetched_1h:
                prefetched_1h[sym] = data_maps[sym]["1h"]

    if not data_maps:
        _logger.error("No data collected. Pipeline aborted.")
        return MLPipelineOutput()

    # --- Step 2: Systemic HMM Inference (Regime Discovery) ---
    _logger.info("  --> Step 2: Systemic HMM Inference (Macro Regime Discovery)")
    from src.domain.futures.ml_pipeline.features.engineering import build_systemic_hmm_features

    # [Optimization #5] Build panel once and reuse for systemic features
    h_maps = {s: {"1h": prefetched_1h[s]} for s in prefetched_1h}
    h_utils = CrossSectionalPipelineUtils()
    h_panel = h_utils.build_panel_df(h_maps, tf="1h")
    
    # Check if systemic features already exist (from GP enrichment)
    systemic_cols = ["macro_trend_24h", "macro_vol_24h", "cs_dispersion"]
    if not all(c in h_panel.columns for c in systemic_cols):
        h_panel = h_utils.add_systemic_features(h_panel)
    
    market_hmm_feats = build_systemic_hmm_features(h_panel, None, tf="1h")
    if market_hmm_feats.index.tz is None:
        market_hmm_feats.index = market_hmm_feats.index.tz_localize("UTC")
    else:
        market_hmm_feats.index = market_hmm_feats.index.tz_convert("UTC")

    hmm_k = int(cfg.get("FUTURES_HMM_K_STATES", 5))
    # [Optimization #6] Reduce n_iter and add convergence tolerance
    hmm_n_iter = int(cfg.get("FUTURES_HMM_N_ITER", 200))
    hmm_inferrer = HMMStateInferrer(n_states=hmm_k, n_iter=hmm_n_iter, tol=1e-4)
    
    is_end_dt = pd.to_datetime(is_end_date or end)
    is_end_utc = is_end_dt.tz_localize("UTC") if is_end_dt.tzinfo is None else is_end_dt.tz_convert("UTC")
    is_end_idx_market = int((market_hmm_feats.index < is_end_utc).sum())

    _btc_anchor = next((s for s in symbols if "BTC" in s), None)
    _btc_df = prefetched_1h.get(_btc_anchor) if _btc_anchor else None
    if _btc_df is not None and "close" in _btc_df.columns:
        _btc_rets = _btc_df.set_index("datetime")["close"].pct_change().reindex(market_hmm_feats.index).fillna(0.0)
    else:
        _btc_rets = market_hmm_feats["macro_trend_168h"]

    market_probs = hmm_inferrer.fit_predict_systemic(
        market_hmm_feats, _btc_rets, is_end_idx=is_end_idx_market, symbol="Market", tf=tf
    )
    market_probs = _ensure_datetime_column(market_probs)
    market_probs["datetime"] = pd.to_datetime(market_probs["datetime"], utc=True)

    if hmm_only:
        _logger.info("  [SUCCESS] HMM Inference complete (HMM-only mode).")
        out = MLPipelineOutput(market_probs=market_probs)
        out.hmm_report = _print_hmm_summary(market_probs, market_hmm_feats, pd.DataFrame(), _btc_df, "(HMM-ONLY)")
        return out

    # --- Step 3: Regime-Aware Alpha Mining ---
    _logger.info("  --> Step 3: Regime-Aware Alpha Model Training (LightGBM)")
    
    # Build final panel on target TF
    panel_df = h_utils.build_panel_df(data_maps, tf=tf)
    panel_df = h_utils.add_cross_sectional_features(panel_df)
    
    # [Optimization #5] Avoid redundant systemic feature recalculation
    if not all(c in panel_df.columns for c in systemic_cols):
        panel_df = h_utils.add_systemic_features(panel_df)
    
    # Inject HMM features into training panel
    _logger.info("  --> Injecting HMM regimes into Alpha features...")
    hmm_cols_all = [c for c in market_probs.columns if str(c).startswith("hmm_")]
    
    # [Fix] Drop overlapping columns before join
    drop_overlap = [c for c in hmm_cols_all if c in panel_df.columns]
    if drop_overlap:
        panel_df = panel_df.drop(columns=drop_overlap)

    mp_feats = market_probs.set_index("datetime")[hmm_cols_all]
    panel_df = panel_df.join(mp_feats, on="datetime", how="left")
    panel_df[hmm_cols_all] = panel_df[hmm_cols_all].ffill().fillna(1.0 / float(hmm_k))

    # Add targets
    raw_h = cfg.get("FUTURES_ML_ALPHA_HORIZONS", (3, 6, 12, 24))
    horizons = tuple(int(x) for x in (raw_h if isinstance(raw_h, (list, tuple)) else (3, 6, 12, 24)))
    _ic_hl = float(OPT_FUTURES_CONFIG.get("FUTURES_ML_IC_HALF_LIFE", 2.3))
    _h_weights = tuple(float(np.exp(-h / _ic_hl)) for h in horizons)
    panel_df["target"] = h_utils.create_multi_horizon_rank_targets(panel_df, horizons=horizons, weights=_h_weights)

    miner = MLAlphaMiner(
        n_jobs=n_jobs, 
        target_horizons=horizons, 
        slots_per_theme=max(5, min(12, int(cfg.get("FUTURES_ML_ALPHA_SLOTS_PER_THEME", 6))))
    )
    alpha_panel = miner.mine_alphas_cs(
        panel_df, 
        cache_path=Path(FUTURES_CACHE_DIR) / "universal_cs_gp_v8.parquet",
        is_end_date=is_end_date
    )

    # --- Step 4: Signal Fusion & Meta-Labeling ---
    _logger.info("  --> Step 4: Signal Fusion & Meta-Labeling (Reusing HMM results)")
    
    # Use existing HMM results for fusion to avoid redundant training
    out = run_hmm_fusion_for_is_end(
        list(data_maps.keys()), tf, fetch_start_date, end, cfg, data_maps, 
        prefetched_1h, panel_df, alpha_panel, is_end_date or end, collector,
        workers, n_jobs, include_fusion=not gp_only,
        prefetched_market_probs=market_probs,
        prefetched_market_hmm_feats=market_hmm_feats
    )
    
    _logger.info("  [SUCCESS] Universal ML Pipeline processing complete.")
    return out

