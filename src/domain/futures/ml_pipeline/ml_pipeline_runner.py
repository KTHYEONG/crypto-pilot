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

from config.settings import FUTURES_CACHE_DIR
from src.domain.futures.data_collector import DataCollector
from src.domain.futures.ml_pipeline.cross_sectional_utils import CrossSectionalPipelineUtils
from src.domain.futures.ml_pipeline.feature_engineering import (
    GP_ENGINEERED_FEATURE_NAMES,
    build_gp_input_features,
)
from src.domain.futures.ml_pipeline.gp_alpha_miner import GPAlphaMiner
from src.domain.futures.ml_pipeline.gp_multiobjective import is_deap_available
from src.domain.futures.ml_pipeline.hmm_state_inferrer import HMMStateInferrer
from src.domain.futures.ml_pipeline.market_regime_pre_gp import (
    attach_regime_pre_to_panel,
    infer_pre_gp_regime_ids,
)
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


def _align_asof_1h_to_tf(df_tf: pd.DataFrame, df_1h_feats: pd.DataFrame) -> pd.DataFrame:
    left = df_tf[["datetime"]].drop_duplicates().sort_values("datetime").copy()
    right = df_1h_feats.copy()
    if "datetime" not in right.columns and right.index.name == "datetime":
        right = right.reset_index()
    right = right.sort_values("datetime")
    left["datetime"] = pd.to_datetime(left["datetime"], utc=True)
    right["datetime"] = pd.to_datetime(right["datetime"], utc=True)
    return pd.merge_asof(left, right, on="datetime", direction="backward")


def _sorted_hmm_prob_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith("hmm_prob_")]
    return sorted(cols, key=lambda x: int(str(x).split("_")[-1]))


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


def _hmm_modulator_values(market_probs: pd.DataFrame) -> np.ndarray:
    cols = _sorted_hmm_prob_columns(market_probs)
    if not cols:
        return np.full(len(market_probs), 0.8, dtype=np.float64)
    mp = market_probs[cols].replace([np.inf, -np.inf], np.nan).fillna(1.0 / float(len(cols)))
    if len(cols) >= 3:
        mod = (
            mp.iloc[:, 2].to_numpy(dtype=np.float64, copy=False) * 1.4
            + mp.iloc[:, 1].to_numpy(dtype=np.float64, copy=False) * 1.0
            + mp.iloc[:, 0].to_numpy(dtype=np.float64, copy=False) * 0.4
        )
    elif len(cols) == 2:
        mod = (
            mp.iloc[:, 1].to_numpy(dtype=np.float64, copy=False) * 1.15
            + mp.iloc[:, 0].to_numpy(dtype=np.float64, copy=False) * 0.55
        )
    else:
        mod = mp.iloc[:, 0].to_numpy(dtype=np.float64, copy=False)
    clipped = np.clip(mod, 0.3, 1.8).astype(np.float64, copy=False)
    return cast(np.ndarray, clipped)


def _try_tbm_labels_per_1h_row(
    sym: str,
    df_1h: pd.DataFrame,
    fetch_start: str,
    end: str,
    df_1m: pd.DataFrame | None = None,
    collector: DataCollector | None = None,
) -> np.ndarray | None:
    """Triple-barrier labels aligned to each 1h row (merge_asof backward from 4h)."""
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
        d1 = df_1h[[*list(need), "datetime"]].sort_values("datetime").copy()
        d1["datetime"] = pd.to_datetime(d1["datetime"], utc=True)
        d1i = d1.set_index("datetime")
        df_4h = (
            d1i.resample("4h", label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna(how="any")
            .reset_index()
        )
        if len(df_4h) < 30:
            return None
        tbm = label_triple_barrier(df_4h, df_1m)
        if tbm is None or len(tbm) == 0:
            return None
        lab = tbm.rename("tbm_label").reset_index()
        lab["datetime"] = pd.to_datetime(lab["datetime"], utc=True)
        lab = lab.sort_values("datetime")
        tmp = df_1h[["datetime"]].copy()
        tmp["_ord"] = np.arange(len(tmp), dtype=np.int64)
        tmp["datetime"] = pd.to_datetime(tmp["datetime"], utc=True)
        tmp = tmp.sort_values("datetime")
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
    hmm_m = aligned_tf["hmm_modulator"].to_numpy(dtype=np.float64)
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
                raw_long = np.clip(pl.astype(np.float64) * hmm_m, 0.0, None)
                raw_short = np.clip(ps.astype(np.float64) * hmm_m, 0.0, None)
            except Exception:
                raw_long = None
                raw_short = None

    if raw_long is None or raw_short is None:
        gp = aligned_tf["gp_alpha_00"].to_numpy(dtype=np.float64)
        raw_long = np.clip(gp * hmm_m, 0.0, None)
        raw_short = np.clip((1.0 - gp) * hmm_m, 0.0, None)

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
        k_fb = len(hmm_cols_ref) if hmm_cols_ref else 3

        if sym not in valid_alpha_set:
            wide_1h = df_1h.copy()
            wide_1h["gp_alpha_00"] = 0.0
            wide_1h["slot_rank_score"] = 0.0
            wide_1h = pd.merge(wide_1h, hmm_modulator, on="datetime", how="left")
            wide_1h["hmm_modulator"] = wide_1h["hmm_modulator"].fillna(0.8)
            if hmm_cols_ref:
                mp_h = market_probs[["datetime", *hmm_cols_ref]]
                wide_1h = wide_1h.merge(mp_h, on="datetime", how="left")
                for c in hmm_cols_ref:
                    wide_1h[c] = wide_1h[c].fillna(1.0 / float(k_fb))
            else:
                for i in range(k_fb):
                    wide_1h[f"hmm_prob_{i}"] = 1.0 / float(k_fb)
        else:
            sym_alpha = alpha_by_sym[sym].copy()
            sym_alpha["datetime"] = pd.to_datetime(sym_alpha["datetime"], utc=True)
            wide_1h = pd.merge(df_1h, sym_alpha, on="datetime", how="left").fillna(0.0)
            wide_1h = pd.merge(wide_1h, hmm_modulator, on="datetime", how="left")
            wide_1h["hmm_modulator"] = wide_1h["hmm_modulator"].fillna(0.8)
            wide_1h["slot_rank_score"] = wide_1h["gp_alpha_00"] * wide_1h["hmm_modulator"]
            if hmm_cols_ref:
                mp_h = market_probs[["datetime", *hmm_cols_ref]]
                wide_1h = wide_1h.merge(mp_h, on="datetime", how="left")
                for c in hmm_cols_ref:
                    wide_1h[c] = wide_1h[c].fillna(1.0 / float(k_fb))
            else:
                for i in range(max(1, k_fb)):
                    wide_1h[f"hmm_prob_{i}"] = 1.0 / float(max(1, k_fb))

        df_tf_full = data_maps[sym][tf].copy()
        df_tf_full["datetime"] = pd.to_datetime(df_tf_full["datetime"], utc=True)
        aligned_tf = _align_asof_1h_to_tf(df_tf_full, wide_1h)

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
) -> MLPipelineOutput:
    """
    [Phase 4] Universal Cross-Sectional ML Pipeline.
    """
    _logger.info("=" * 85)
    _logger.info(" [PHASE 4] Starting Universal Cross-Sectional ML Pipeline")
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
    if bool(cfg.get("FUTURES_ML_PRE_GP_REGIME", True)):
        try:
            n_reg = int(cfg.get("FUTURES_ML_PRE_GP_REGIME_STATES", 3))
            rser = infer_pre_gp_regime_ids(
                panel_df, is_end_date=is_end_date, n_states=max(2, n_reg)
            )
            panel_df = attach_regime_pre_to_panel(panel_df, rser)
        except Exception as e:
            _logger.warning("pre-GP regime HMM skipped: %s", e)
            panel_df = panel_df.copy()
            panel_df["regime_pre_hmm"] = np.int64(0)
    raw_h = cfg.get("FUTURES_ML_GP_HORIZONS", (3, 6, 12, 24))
    horizons = tuple(int(x) for x in (raw_h if isinstance(raw_h, (list, tuple)) else (3, 6, 12, 24)))
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

    # --- Step 3: Systemic Market HMM & Bayesian Alpha Fusion ---
    _logger.info("Step 3/4: Training Systemic Market HMM & Bayesian Alpha Fusion...")
    from src.domain.futures.ml_pipeline.feature_engineering import build_systemic_hmm_features
    
    market_hmm_feats = build_systemic_hmm_features(panel_df, alpha_panel)
    if market_hmm_feats.index.tz is None:
        market_hmm_feats.index = market_hmm_feats.index.tz_localize("UTC")
    else:
        market_hmm_feats.index = market_hmm_feats.index.tz_convert("UTC")

    hmm_inferrer = HMMStateInferrer(n_states=0)

    is_end_dt = pd.to_datetime(is_end_date or end)
    is_end_utc = (
        is_end_dt.tz_localize("UTC")
        if is_end_dt.tzinfo is None
        else is_end_dt.tz_convert("UTC")
    )
    is_end_idx_market = int((market_hmm_feats.index < is_end_utc).sum())

    market_probs = hmm_inferrer.fit_predict_systemic(
        market_hmm_feats,
        market_hmm_feats["GP_LS_Spread"],
        is_end_idx=is_end_idx_market,
    )
    market_probs = _ensure_datetime_column(market_probs)
    market_probs["datetime"] = pd.to_datetime(market_probs["datetime"], utc=True)
    hmm_modulator = pd.DataFrame(
        {
            "datetime": market_probs["datetime"],
            "hmm_modulator": _hmm_modulator_values(market_probs),
        }
    )
    
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

    # --- Final Health Report ---
    _logger.info("=" * 85)
    _logger.info(" [ML Pipeline Health Report]")
    _logger.info("-" * 85)
    # 1. GP Quality (IC)
    try:
        ic_val = float(alpha_panel.attrs.get("best_fitness", 0.0))
        _logger.info(f" Universal CS-GP IC: {ic_val:.4f}")
    except Exception:  # noqa: S110
        pass
    _logger.info("-" * 85)
    _logger.info(" [PHASE 4] Universal Pipeline Processing Complete")
    _logger.info("=" * 85)
    return out
