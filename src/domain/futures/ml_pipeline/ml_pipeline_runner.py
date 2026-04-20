"""
ML pipeline execution orchestration for Cross-Sectional Ranking Portfolio.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

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
    legacy = [c for c in df.columns if str(c).startswith("hmm_prob_")]
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
    """
    [REFACTORED] Asymmetric Long/Short Kelly with Isotonic constraints (P0 + P2).
    Based on raw market forward returns instead of GP alpha product.
    """
    kelly_long: np.ndarray = np.zeros(n_states, dtype=np.float64)
    kelly_short: np.ndarray = np.zeros(n_states, dtype=np.float64)
    state_hard = np.argmax(p_mat, axis=1).astype(np.int64)
    
    for s in range(n_states):
        m = is_mask & (state_hard == s)
        if np.sum(m) < 30:
            continue
            
        r = fwd_ret[m]
        mu = float(np.mean(r))
        v = float(np.var(r, ddof=1)) + 1e-12
        # Use 1.5 as clip to allow for stronger conviction in Bull/Bear
        kelly_long[s] = float(np.clip(mu / v, -1.5, 1.5))
        
        mu_s = float(np.mean(-r))
        kelly_short[s] = float(np.clip(mu_s / v, -1.5, 1.5))
        
    # Apply Isotonic Constraints for semantic ordering
    # BULL=0, CHOP=1, BEAR=2, CRISIS=3 in HMM_SEMANTIC_PROB_COLUMNS
    try:
        # monotone decreasing: BULL >= CHOP >= BEAR >= CRISIS
        ir = IsotonicRegression(increasing=False)
        
        idx_bull = cols.index("hmm_prob_bull_trend") if "hmm_prob_bull_trend" in cols else -1
        idx_chop = cols.index("hmm_prob_chop") if "hmm_prob_chop" in cols else -1
        idx_bear = cols.index("hmm_prob_bear_trend") if "hmm_prob_bear_trend" in cols else -1
        idx_crisis = cols.index("hmm_prob_crisis") if "hmm_prob_crisis" in cols else -1
        
        if all(i >= 0 for i in [idx_bull, idx_chop, idx_bear, idx_crisis]):
            # Long Mono: BULL >= CHOP >= BEAR >= CRISIS
            x = np.array([0, 1, 2, 3])
            y_l = np.array([
                kelly_long[idx_bull],
                kelly_long[idx_chop],
                kelly_long[idx_bear],
                kelly_long[idx_crisis]
            ])
            y_l_adj = ir.fit_transform(x, y_l)
            kelly_long[idx_bull], kelly_long[idx_chop], \
                kelly_long[idx_bear], kelly_long[idx_crisis] = y_l_adj
            
            # Short Mono: BEAR >= CHOP >= BULL >= CRISIS
            y_s = np.array([
                kelly_short[idx_bear],
                kelly_short[idx_chop],
                kelly_short[idx_bull],
                kelly_short[idx_crisis]
            ])
            y_s_adj = ir.fit_transform(x, y_s)
            kelly_short[idx_bear], kelly_short[idx_chop], \
                kelly_short[idx_bull], kelly_short[idx_crisis] = y_s_adj
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
) -> pd.DataFrame:
    """
    [REFACTORED] Posterior-weighted asymmetric modulators (Long/Short).
    Returns DataFrame with columns ['hmm_modulator_long', 'hmm_modulator_short'].
    """
    cols = _sorted_hmm_prob_columns(market_probs)
    n = len(market_probs)
    if not cols or btc_df is None or len(btc_df) < 80:
        return pd.DataFrame({
            "hmm_modulator_long": np.full(n, 1.0, dtype=np.float64),
            "hmm_modulator_short": np.full(n, 1.0, dtype=np.float64)
        })
        
    k_states = len(cols)
    b = btc_df.sort_values("datetime").copy()
    b["datetime"] = pd.to_datetime(b["datetime"], utc=True)
    c = b["close"].astype(np.float64)
    # Forward return for Kelly calculation (Shifted back by 1)
    b["fwd_ret"] = np.log(c.shift(-1) / c.clip(lower=1e-12)).fillna(0.0)

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
    
    # Apply Crisis Kill-Switch
    if "hmm_prob_crisis" in market_probs.columns:
        pc = market_probs["hmm_prob_crisis"].fillna(0.0).to_numpy(dtype=np.float64)
        # In crisis, both directions are forced down to 0.3
        mod_long = np.where(pc > float(crisis_thr), 0.3, mod_long)
        mod_short = np.where(pc > float(crisis_thr), 0.3, mod_short)
        
    return pd.DataFrame({
        "hmm_modulator_long": mod_long,
        "hmm_modulator_short": mod_short
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
            
        tbm = label_triple_barrier(d1, df_1m)
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
    """Set ml_calib_prob_{long,short} via MetaLabeler (+ softmax) or GP x HMM softmax only."""
    # [REFACTORED] Use Asymmetric Long/Short Modulators
    hmm_m_long = aligned_tf["hmm_modulator_long"].to_numpy(dtype=np.float64)
    hmm_m_short = aligned_tf["hmm_modulator_short"].to_numpy(dtype=np.float64)
    
    raw_long: np.ndarray | None = None
    raw_short: np.ndarray | None = None

    y_tbm = (
        _try_tbm_labels_per_1h_row(
            sym, wide_1h, fetch_start, end, df_1m=df_1m_prefetch, collector=collector
        )
        if use_meta
        else None
    )
    meta_feats = tuple(c for c in _meta_feature_column_names(wide_1h) if c in wide_1h.columns)
    can_meta = (
        use_meta
        and y_tbm is not None
        and len(y_tbm) == len(wide_1h)
        and len(meta_feats) >= 2
        and len(_sorted_hmm_prob_columns(wide_1h)) > 0
        and all(c in aligned_tf.columns for c in meta_feats)
    )
    if can_meta:
        X_w = wide_1h[list(meta_feats)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_ser = pd.Series(y_tbm, index=wide_1h.index)
        wdt = pd.to_datetime(wide_1h["datetime"], utc=True)
        is_end_idx = int((wdt < is_end_utc).sum())
        if is_end_idx >= 80:
            try:
                meta = MetaLabeler(vertical_barrier_bars=48)
                meta.fit(X_w, y_ser, is_end_idx)
                X_a = aligned_tf[list(meta_feats)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                pl, ps = meta.predict_proba_calibrated(X_a)
                # Apply asymmetric modulators
                raw_long = np.clip(pl.astype(np.float64) * hmm_m_long, 0.0, None)
                raw_short = np.clip(ps.astype(np.float64) * hmm_m_short, 0.0, None)
            except Exception:
                raw_long = None
                raw_short = None

    if raw_long is None or raw_short is None:
        gp = aligned_tf["gp_alpha_00"].to_numpy(dtype=np.float64)
        raw_long = np.clip(gp * hmm_m_long, 0.0, None)
        raw_short = np.clip((1.0 - gp) * hmm_m_short, 0.0, None)

    denom = raw_long + raw_short + 1e-12
    aligned_tf["ml_calib_prob_long"] = raw_long / denom
    aligned_tf["ml_calib_prob_short"] = raw_short / denom
    aligned_tf["ml_calib_prob"] = np.maximum(
        aligned_tf["ml_calib_prob_long"], aligned_tf["ml_calib_prob_short"]
    )


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
    """Per-symbol merge + asof alignment + MetaLabeler/softmax."""
    try:
        df_1h = prefetched_1h[sym].copy()
        df_1h["datetime"] = pd.to_datetime(df_1h["datetime"], utc=True)

        hmm_cols_ref = _sorted_hmm_prob_columns(market_probs)
        k_fb = len(hmm_cols_ref) if hmm_cols_ref else len(HMM_SEMANTIC_PROB_COLUMNS)

        if sym not in valid_alpha_set:
            wide_1h = df_1h.copy()
            wide_1h["gp_alpha_00"] = 0.0
            # [REFACTORED] Merge asymmetric modulators
            wide_1h = pd.merge(wide_1h, hmm_modulator, on="datetime", how="left")
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
            wide_1h = pd.merge(df_1h, sym_alpha, on="datetime", how="left").fillna(0.0)
            
            # [REFACTORED] Merge asymmetric modulators
            wide_1h = pd.merge(wide_1h, hmm_modulator, on="datetime", how="left")
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
    """
    [Phase 2] Universal Cross-Sectional ML Pipeline.
    """
    _logger.info("=" * 85)
    _logger.info(" [PHASE 2] Starting Universal Cross-Sectional ML Pipeline")
    _logger.info("=" * 85)

    collector = DataCollector()
    data_maps: dict[str, dict[str, Any]] = {}
    prefetched_1h: dict[str, pd.DataFrame] = {}
    
    # --- Step 1: Market-Wide Data Collection & Panel Building ---
    _logger.info("Step 1/4: Collecting panel data for %d symbols...", len(symbols))
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
        _logger.info("Step 1b: prefetch 1m for TBM-based GP sample weights...")
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

    # --- Step 2: Universal GP Model Training (Cross-Sectional IC) ---
    _logger.info("Step 2/4: Training Universal Cross-Sectional GP Model...")
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

    if gp_only:
        # Step 2 이후 즉시 종료 (HMM 및 Fusion 스킵)
        best_fitness = alpha_panel.attrs.get("best_fitness", 0.0)
        filter_meta = alpha_panel.attrs.get("gp_alpha_filter", {})
        
        _logger.info("\n" + "-" * 85)
        _logger.info(" [PHASE 2] GP IC VALIDATION RESULTS (GP-ONLY MODE)")
        _logger.info("-" * 85)
        _logger.info(f" IS Best Fitness (Composite ICIR): {best_fitness:.6f}")
        _logger.info(f" GP Alpha Components Tried:      {filter_meta.get('n_components', 0):.0f}")
        _logger.info(f" GP Alpha Components Surviving:   {filter_meta.get('n_surviving', 0):.0f}")
        
        _logger.info("-" * 85)
        _logger.info(" [DIAGNOSTICS] Elimination Breakdown:")
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
        _logger.info(f"   - Raw T-Stat:           {filter_meta.get('primary_t_stat', 0.0):.2f}")
        _logger.info("-" * 85)
        _logger.info(" [RESULT] GP-Only analysis complete. Skipping HMM & Fusion.")
        _logger.info("=" * 85 + "\n")
        
        out = MLPipelineOutput()
        out.alpha_panel = alpha_panel
        return out

    # --- Step 3: Systemic Market HMM & Bayesian Alpha Fusion ---
    _logger.info("Step 3/4: Training Systemic Macro HMM (GP-Independent)...")
    from src.domain.futures.ml_pipeline.feature_engineering import build_systemic_hmm_features
    
    # HMM is now independent of alpha_panel to avoid circular IS overfitting.
    market_hmm_feats = build_systemic_hmm_features(panel_df, None)
    if market_hmm_feats.index.tz is None:
        market_hmm_feats.index = market_hmm_feats.index.tz_localize("UTC")
    else:
        market_hmm_feats.index = market_hmm_feats.index.tz_convert("UTC")

    hmm_k = int(cfg.get("FUTURES_HMM_K_STATES", 4))
    hmm_inferrer = HMMStateInferrer(n_states=hmm_k)

    is_end_dt = pd.to_datetime(is_end_date or end)
    is_end_utc = (
        is_end_dt.tz_localize("UTC")
        if is_end_dt.tzinfo is None
        else is_end_dt.tz_convert("UTC")
    )
    is_end_idx_market = int((market_hmm_feats.index < is_end_utc).sum())

    # Use BTC_Trend_24h as the performance proxy for state ordering (instead of GP LS Spread)
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
    
    # [REFACTORED] Calculate Asymmetric Modulators
    hmm_modulator = _hmm_modulator_kelly_values(
        market_probs,
        alpha_panel,
        is_end_utc,
        btc_1h,
        float(cfg.get("FUTURES_HMM_KELLY_SHRINKAGE", 0.4)),
        float(cfg.get("FUTURES_HMM_CRISIS_THRESHOLD", 0.7)),
    )
    hmm_modulator["datetime"] = market_probs["datetime"]

    if hmm_only:
        _logger.info("\n" + "=" * 85)
        _logger.info(" [PHASE 3] HMM REGIME ANALYSIS RESULTS (HMM-ONLY MODE)")
        _logger.info("=" * 85)
        
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
        from src.domain.futures.ml_pipeline.feature_engineering import HMM_SEMANTIC_PROB_COLUMNS
        cols = [c for c in HMM_SEMANTIC_PROB_COLUMNS if c in market_probs.columns]
        
        if cols and len(market_probs) == len(market_hmm_feats):
            # Align everything
            df_eval = market_probs[['datetime', *cols]].copy()
            df_eval['dominant_state'] = df_eval[cols].idxmax(axis=1)
            
            # Merge Modulator
            df_eval = pd.merge(df_eval, hmm_modulator, on='datetime', how='left')
            
            # Merge Features
            feat_tmp = market_hmm_feats.copy().reset_index()
            feat_cols_to_merge = ['datetime', 'btc_trend_vol_adj_24h', 'realized_vol_regime']
            df_eval = pd.merge(df_eval, feat_tmp[feat_cols_to_merge], on='datetime', how='left')
            
            # Calculate BTC 24h Forward Return if available
            has_return = False
            if btc_1h is not None and not btc_1h.empty:
                btc_tmp = btc_1h[['datetime', 'close']].copy()
                btc_tmp['datetime'] = pd.to_datetime(btc_tmp['datetime'], utc=True)
                btc_tmp['fwd_ret_24h'] = btc_tmp['close'].pct_change(24).shift(-24)
                df_eval = pd.merge(
                    df_eval, btc_tmp[['datetime', 'fwd_ret_24h']], on='datetime', how='left'
                )
                has_return = True
            
            # Print Global Averages
            _logger.info("  [Global Averages]")
            means = market_probs[cols].mean()
            for c, m in means.items():
                _logger.info(f"   - {c:<25}: {m:.4f}")
            avg_mod_l = df_eval['hmm_modulator_long'].mean()
            avg_mod_s = df_eval['hmm_modulator_short'].mean()
            _logger.info(f"   - Global Modulator (L/S): {avg_mod_l:.2f} / {avg_mod_s:.2f}")
            _logger.info("-" * 40)
            
            # Print Per-State Profile
            _logger.info("  [Per-State Financial Profile]")
            for state in cols:
                g = df_eval[df_eval['dominant_state'] == state]
                pct = len(g) / len(df_eval) * 100
                if len(g) > 0:
                    avg_trend = g['btc_trend_vol_adj_24h'].mean()
                    avg_vol = g['realized_vol_regime'].mean()
                    avg_mod_l = g['hmm_modulator_long'].mean()
                    avg_mod_s = g['hmm_modulator_short'].mean()
                    
                    ret_str = "N/A"
                    if has_return:
                        avg_ret = g['fwd_ret_24h'].mean() * 100
                        ret_str = f"{avg_ret:>+6.2f}%"
                        
                    st_name = state.replace('hmm_prob_', '').upper()
                    _logger.info(f"   ► {st_name:<11} ({pct:>5.1f}% time)")
                    _logger.info(
                        f"      Kelly Mod (L/S): {avg_mod_l:.2f} / {avg_mod_s:.2f} | "
                        f"24h Fwd Ret: {ret_str}"
                    )
                    _logger.info(
                        f"      Avg Trend: {avg_trend:>+5.2f} | Avg Vol:     {avg_vol:.2f}"
                    )
                else:
                    st_name = state.replace('hmm_prob_', '').upper()
                    _logger.info(f"   ► {st_name:<11} (  0.0% time)")

        _logger.info("-" * 85)
        _logger.info(" [RESULT] HMM-Only analysis complete. Skipping Fusion.")
        _logger.info("=" * 85 + "\n")
        
        out = MLPipelineOutput()
        out.alpha_panel = alpha_panel
        return out

    
    out = MLPipelineOutput()
    
    # --- Step 4: Individual Symbol Processing (Fusion) ---
    # [FIX] Use focused start date for 1m prefetch
    label_start = is_start_date or fetch_start
    _logger.info("Step 4/4: Applying Fusion to individual symbols...")
    _logger.info(f" [FUSION] 1m range: {label_start} ~ {end}")
    
    valid_alpha_symbols = alpha_panel.index.get_level_values("symbol").unique()
    valid_alpha_set = set(valid_alpha_symbols.tolist())

    syms_step4 = [s for s in symbols if s in data_maps]
    need_1m = [s for s in syms_step4 if s in valid_alpha_set]
    
    def _prefetch_1m(s: str) -> tuple[str, pd.DataFrame | None]:
        try:
            d = collector.collect_1m_ohlcv(s, label_start, end)
            return (s, d if d is not None and len(d) >= 200 else None)
        except Exception:
            return (s, None)

    # 스마트 스로틀링(API Header 기반)이 적용되었으므로 스레드를 최대 4개로 상향
    prefetch_workers = max(1, min(len(need_1m) or 1, 4))
    one_m_cache: dict[str, pd.DataFrame | None] = {}
    if need_1m:
        _logger.info(
            " Step 4a: prefetch 1m OHLCV for %d symbols (threads=%d)...",
            len(need_1m),
            prefetch_workers,
        )
        with ThreadPoolExecutor(max_workers=prefetch_workers) as ex:
            one_m_cache = {s: dm for s, dm in ex.map(_prefetch_1m, need_1m)}

    def _fusion_job(s: str) -> _Step4FusionOutcome:
        return _step4_fusion_one_symbol(
            s, tf, data_maps, prefetched_1h, alpha_by_sym, valid_alpha_set,
            market_probs, hmm_modulator, fetch_start, end, is_end_utc,
            one_m_cache.get(s), collector,
        )

    alpha_by_sym: dict[str, pd.DataFrame] = {
        s: alpha_panel.xs(s, level="symbol").reset_index()
        for s in syms_step4 if s in valid_alpha_set
    }

    fusion_workers = max(1, min(len(syms_step4) or 1, max(workers, n_jobs, 2), 12))
    _logger.info(" Step 4b: fusion + MetaLabeler (threads=%d)...", fusion_workers)
    with ThreadPoolExecutor(max_workers=fusion_workers) as ex:
        fusion_results = list(ex.map(_fusion_job, syms_step4))

    for res in fusion_results:
        if res.error is not None:
            _logger.error("[%s] Processing failed: %s", res.sym, res.error)
            continue
        if res.aligned_tf is None or res.cp_long is None or res.cp_short is None:
            _logger.error("[%s] Fusion returned empty frames.", res.sym)
            continue
        out.meta_feature_frame_by_symbol[res.sym] = res.aligned_tf
        out.calib_prob_long_by_symbol[res.sym] = res.cp_long
        out.calib_prob_short_by_symbol[res.sym] = res.cp_short
        out.calib_prob_by_symbol[res.sym] = res.cp_long

    # --- Final GP IC Validation Results ---
    best_fitness = alpha_panel.attrs.get("best_fitness", 0.0)
    filter_meta = alpha_panel.attrs.get("gp_alpha_filter", {})

    _logger.info("\n" + "-" * 85)
    _logger.info(" [PHASE 2] GP IC VALIDATION RESULTS")
    _logger.info("-" * 85)
    _logger.info(f" IS Best Fitness (Composite ICIR): {best_fitness:.6f}")
    _logger.info(f" GP Alpha Components Tried:      {filter_meta.get('n_components', 0):.0f}")
    _logger.info(f" GP Alpha Components Surviving:   {filter_meta.get('n_surviving', 0):.0f}")
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

    _logger.info("-" * 85)
    _logger.info(" [PHASE 2] Universal Pipeline Processing Complete")
    _logger.info("=" * 85 + "\n")
    out.alpha_panel = alpha_panel
    return out
