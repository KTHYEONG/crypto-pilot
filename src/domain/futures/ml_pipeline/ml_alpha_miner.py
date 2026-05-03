"""GP Alpha Miner - Cross-Sectional Ranking Edition (LightGBM Architecture Swap).

Learns a universal formula across multiple symbols using panel data and CS-IC fitness.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.preprocessing import QuantileTransformer

from config.opt_config import OPT_FUTURES_CONFIG
from src.core.utils.cache_manager import CacheManager
from src.domain.futures.ml_pipeline.alpha_component_filter import filter_alpha_components
from src.domain.futures.ml_pipeline.cross_sectional_utils import CrossSectionalPipelineUtils
from src.domain.futures.ml_pipeline.feature_engineering import (
    GP_ENGINEERED_FEATURE_NAMES,
    GP_FEATURE_SCHEMA_VERSION,
)

_logger = logging.getLogger(__name__)


def _pct_uniform_cs_is_fit(unstacked: pd.DataFrame, is_row_mask: np.ndarray) -> pd.DataFrame:
    """Apply IS-fitted empirical CDF to achieve uniform [0,1] distribution across rows.

    Args:
        unstacked: Wide-form dataframe of component values.
        is_row_mask: Boolean mask for In-Sample rows.

    Returns:
        Transformed uniform dataframe.

    """
    if unstacked.shape[1] < 2 or len(unstacked) != int(is_row_mask.shape[0]):
        return pd.DataFrame(0.5, index=unstacked.index, columns=unstacked.columns)
    mat = unstacked.to_numpy(dtype=np.float64)
    is_rows = np.asarray(is_row_mask, dtype=bool)[: mat.shape[0]]
    is_vals = mat[is_rows, :].ravel()
    is_vals = is_vals[np.isfinite(is_vals)]
    if is_vals.size < 100 or float(np.nanstd(is_vals)) < 1e-14:
        return pd.DataFrame(0.5, index=unstacked.index, columns=unstacked.columns)
    nq = int(min(1000, max(10, is_vals.size)))
    qt = QuantileTransformer(
        n_quantiles=nq,
        output_distribution="uniform",
        random_state=42,
        subsample=min(50_000, int(is_vals.size)),
    )
    qt.fit(is_vals.reshape(-1, 1))
    flat = np.where(np.isfinite(mat), mat, np.nan).reshape(-1, 1)
    out_flat = np.full(flat.shape[0], 0.5, dtype=np.float64)
    m = np.isfinite(flat[:, 0])
    if m.any():
        out_flat[m] = qt.transform(flat[m, :].reshape(-1, 1)).ravel()
    out = out_flat.reshape(mat.shape)
    return pd.DataFrame(out, index=unstacked.index, columns=unstacked.columns)


def _resolve_gp_feature_columns(panel_df: pd.DataFrame) -> list[str]:
    """Identify columns to be used as features for the GP/ML model.

    Args:
        panel_df: Input panel dataframe.

    Returns:
        List of feature column names.

    """
    blocked = {"target", "close", "tbm_gp_weight", "regime_pre_hmm"}
    cs_names = sorted(c for c in panel_df.columns if c.startswith("cs_"))

    base_features = set()
    if cs_names:
        # T2: "market_breadth" 제거 — feature_engineering.py 미구현 zombie feature
        extras = [
            c for c in ("cross_vol_rank", "cross_ret_24h_rank")
            if c in panel_df.columns
        ]
        base_features = {c for c in cs_names + extras if c not in blocked}
    else:
        # T2: market_breadth for-loop 제거 — 미구현 feature 참조 차단
        base_features = {
            c for c in panel_df.columns if c not in blocked and not c.startswith("hmm_prob_")
        }

    return sorted(base_features)


@dataclass
class MLAlphaMiner:
    """Miner for evolving cross-sectional alpha components using LightGBM.

    Replaces gplearn for improved performance and robustness.
    """

    n_features_to_select: int = 15
    population_size: int = 1000
    generations: int = 20
    target_horizon: int = 6
    target_horizons: tuple[int, ...] = (3, 6, 12, 24)
    n_jobs: int = 1
    parsimony_coefficient: float = 0.001

    # Internal state
    _st_global: LGBMRegressor | None = field(default=None, init=False)
    _st_bull: LGBMRegressor | None = field(default=None, init=False)
    _st_bear: LGBMRegressor | None = field(default=None, init=False)
    _cb_global: CatBoostRegressor | None = field(default=None, init=False)
    _cb_bull: CatBoostRegressor | None = field(default=None, init=False)
    _cb_bear: CatBoostRegressor | None = field(default=None, init=False)

    _kept_indices: list[int] | None = field(default=None, init=False)
    _mp_mean: np.ndarray | None = field(default=None, init=False)
    _mp_std: np.ndarray | float | None = field(default=None, init=False)
    _mp_evecs: np.ndarray | None = field(default=None, init=False)

    def mine_alphas_cs(
        self,
        panel_df: pd.DataFrame,
        cache_path: Path | None = None,
        is_end_date: str | None = None,
        filter_options: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Train and select alpha components using a cross-sectional approach.

        Args:
            panel_df: Input panel dataframe.
            cache_path: Optional path for caching raw results.
            is_end_date: Cutoff date for In-Sample data.
            filter_options: Options for alpha component filtering.

        Returns:
            Dataframe of selected and filtered alpha components.

        """
        if panel_df.empty:
            return pd.DataFrame()

        work_panel = panel_df.copy()
        src_cols = [c for c in GP_ENGINEERED_FEATURE_NAMES if c in work_panel.columns]
        if src_cols:
            work_panel = CrossSectionalPipelineUtils.cs_rank_transform(work_panel, src_cols)

        versioned_cache: Path | None = None
        raw_alpha_df: pd.DataFrame | None = None
        force_retrain = bool(OPT_FUTURES_CONFIG.get("FUTURES_ML_FORCE_RETRAIN_ALPHA", False))
        
        if cache_path is not None:
            # SOTA: Dependency-Aware Hashing & LRU Cleanup
            cm = CacheManager(cache_path.parent, max_files=10, max_size_mb=1000.0)
            cm.cleanup_lru(pattern="raw_lgbm_univ_*")
            
            symbols = sorted(list(work_panel.index.get_level_values("symbol").unique()))
            h_str = "-".join(map(str, self.target_horizons))
            prefix = f"lgbm_univ_s{len(symbols)}_h{h_str}"
            
            deps = {
                "pop": self.population_size,
                "gen": self.generations,
                "horizons": self.target_horizons,
                "symbols": symbols,
                "is_end_date": is_end_date,
                "long_bias": float(OPT_FUTURES_CONFIG.get("FUTURES_GP_LONG_BIAS", 1.0)),
                "feature_schema_version": GP_FEATURE_SCHEMA_VERSION,
                "feature_columns": sorted(list(GP_ENGINEERED_FEATURE_NAMES)),
                "ver": "v2_smart_lru"
            }
            # Track relevant source files for automatic invalidation
            src_files = [
                Path(__file__).resolve(),
                Path(__file__).resolve().parent / "alpha_component_filter.py",
                Path(__file__).resolve().parent / "feature_engineering.py"
            ]
            
            tag = cm.generate_hash(deps, source_files=src_files)
            versioned_cache = cm.get_cache_path(f"raw_{prefix}", ".parquet", tag)
            meta_cache = versioned_cache.with_suffix(".json")

            if force_retrain and versioned_cache.exists():
                _logger.info("LightGBM CS Cache Bypass: force retrain enabled.")
            elif versioned_cache.exists():
                try:
                    loaded = pd.read_parquet(versioned_cache)
                    if len(loaded) == len(panel_df):
                        _logger.info("LightGBM CS Cache Hit (SOTA): %s.", versioned_cache.name)
                        if meta_cache.exists():
                            with open(meta_cache) as f:
                                loaded.attrs.update(json.load(f))
                        raw_alpha_df = loaded
                except Exception as e:
                    _logger.debug("LightGBM Smart Cache load failed: %s", e)

        unstacked_y = work_panel["target"].unstack(level="symbol")
        unique_times = unstacked_y.index
        unique_symbols = unstacked_y.columns

        if is_end_date:
            _cut = pd.to_datetime(is_end_date, utc=True)
            if getattr(unique_times, "tz", None) is None:
                _ut = pd.to_datetime(unique_times, utc=True)
            else:
                _ut = unique_times.tz_convert("UTC")
            is_row_mask_utc = np.asarray(_ut < _cut, dtype=bool)
        else:
            is_row_mask_utc = np.ones(len(unique_times), dtype=bool)

        full_grid_index = pd.MultiIndex.from_product(
            [unique_times, unique_symbols], names=["datetime", "symbol"]
        )

        if raw_alpha_df is None:
            # --- Multi-Horizon Ensemble ---
            horizons = tuple(int(h) for h in self.target_horizons) or (3, 6, 12, 24)
            feat_cols = _resolve_gp_feature_columns(work_panel)
            template = pd.DataFrame(index=full_grid_index)

            # OHLC for multi-horizon target calculation
            ohlc_cols = [c for c in ["close", "open", "high", "low"] if c in work_panel.columns]
            join_cols = [*feat_cols, *ohlc_cols, "target"]
            if "tbm_gp_weight" in work_panel.columns:
                join_cols = [*feat_cols, *ohlc_cols, "tbm_gp_weight", "target"]

            aligned_df = template.join(work_panel[join_cols]).fillna(np.nan)
            x_grid = aligned_df[feat_cols].values

            # Base sample weight (non-NaN features)
            base_sw = np.ones(len(aligned_df), dtype=np.float64)
            if "tbm_gp_weight" in aligned_df.columns:
                tw = aligned_df["tbm_gp_weight"].fillna(1.0).to_numpy(dtype=np.float64)
                base_sw = base_sw * np.clip(tw, 0.25, 3.0)

            # --- Regime-balanced weighting (inverse-frequency soft assignment) ---
            p_bull = aligned_df.get(
                "hmm_prob_bull_trend", pd.Series(0.0, index=aligned_df.index)
            ).fillna(0.0).to_numpy(dtype=np.float64)
            p_bear = aligned_df.get(
                "hmm_prob_bear_trend", pd.Series(0.0, index=aligned_df.index)
            ).fillna(0.0).to_numpy(dtype=np.float64)
            p_crisis = aligned_df.get(
                "hmm_prob_crisis", pd.Series(0.0, index=aligned_df.index)
            ).fillna(0.0).to_numpy(dtype=np.float64)
            p_chop = aligned_df.get(
                "hmm_prob_chop", pd.Series(0.0, index=aligned_df.index)
            ).fillna(0.0).to_numpy(dtype=np.float64)
            p_sum = np.clip(p_bull + p_bear + p_crisis + p_chop, 1e-9, None)
            p_bull = p_bull / p_sum
            p_bear = p_bear / p_sum
            p_crisis = p_crisis / p_sum
            p_chop = p_chop / p_sum

            m_bull = float(np.sum(base_sw * p_bull))
            m_bear = float(np.sum(base_sw * p_bear))
            m_crisis = float(np.sum(base_sw * p_crisis))
            m_chop = float(np.sum(base_sw * p_chop))
            masses = np.array([m_bull, m_bear, m_crisis, m_chop], dtype=np.float64)
            freqs = masses / (np.sum(masses) + 1e-12)
            inv = 1.0 / (freqs + 1e-6)
            inv = inv / (np.mean(inv) + 1e-12)
            reg_weight = (
                (p_bull * inv[0]) + (p_bear * inv[1]) + (p_crisis * inv[2]) + (p_chop * inv[3])
            )
            base_sw *= np.clip(reg_weight, 0.6, 1.8)

            if is_end_date:
                times_arr = aligned_df.index.get_level_values("datetime")
                if times_arr.tz is None:
                    times_arr = times_arr.tz_localize("UTC")
                else:
                    times_arr = times_arr.tz_convert("UTC")
                cutoff_ts = pd.to_datetime(is_end_date, utc=True)
                delta_ns = (cutoff_ts - times_arr).total_seconds() / 3600.0
                tau = 13140.0  # 1.5 year decay — better preserves IS=15mo distribution
                time_decay = np.exp(-np.clip(delta_ns, 0.0, None) / tau).astype(np.float64)
                base_sw *= np.asarray(times_arr < cutoff_ts) * time_decay

            nan_row = np.isnan(x_grid).any(axis=1)
            base_sw *= (~nan_row).astype(np.float64)
            x_clean = np.where(np.isfinite(x_grid), x_grid, 0.0)

            # --- MI Audit & MP Denoising (representative target) ---
            y_audit = aligned_df["target"].fillna(0.0).values
            mask_audit = (base_sw > 0) & (~np.isnan(aligned_df["target"]))

            if mask_audit.sum() > 100:
                from sklearn.feature_selection import mutual_info_regression
                x_fit_audit = x_clean[mask_audit]
                y_fit_audit = y_audit[mask_audit]
                n_samples_fit = x_fit_audit.shape[0]
                sample_size = min(5000, n_samples_fit)
                idx = np.random.choice(n_samples_fit, sample_size, replace=False)
                mi_scores = mutual_info_regression(
                    x_fit_audit[idx], y_fit_audit[idx], random_state=42
                )
                kept_indices = [i for i, s in enumerate(mi_scores) if s > 1e-4]
                if len(kept_indices) < max(5, len(feat_cols) // 4):
                    kept_indices = cast(
                        list[int],
                        np.argsort(mi_scores)[-max(5, len(feat_cols) // 4) :].tolist()
                    )
                x_clean = x_clean[:, kept_indices]
                self._kept_indices = kept_indices
                _logger.debug(
                    " [Phase C] Kept %d/%d features via MI Audit.",
                    len(kept_indices),
                    len(feat_cols)
                )
            else:
                self._kept_indices = list(range(len(feat_cols)))

            # --- Multi-Horizon Loop ---
            ensemble_preds = np.zeros(len(aligned_df), dtype=np.float64)
            horizon_ics = []
            slot_components: list[np.ndarray] = []
            long_components: list[np.ndarray] = []
            short_components: list[np.ndarray] = []
            close_ser = aligned_df.get("close", pd.Series(np.nan, index=aligned_df.index))
            ensemble_preds_long = np.zeros(len(aligned_df), dtype=np.float64)
            ensemble_preds_short = np.zeros(len(aligned_df), dtype=np.float64)
            horizon_scores: list[float] = []

            reg_cfg = OPT_FUTURES_CONFIG.get("FUTURES_GP_REGIME_SPECIFIC_LEARNING", False)
            regime_learning = bool(reg_cfg)
            p_bull_ser = aligned_df.get(
                "hmm_prob_bull_trend", pd.Series(0.0, index=aligned_df.index)
            )
            p_bull_arr = p_bull_ser.fillna(0.0).to_numpy()
            p_bear_ser = aligned_df.get(
                "hmm_prob_bear_trend", pd.Series(0.0, index=aligned_df.index)
            )
            p_bear_arr = p_bear_ser.fillna(0.0).to_numpy()

            lgbm_h = cb_h = lgbm_bull = cb_bull = lgbm_bear = cb_bear = None

            def _train_ml_pair(
                idx: np.ndarray, x_mat: np.ndarray, y_vec: np.ndarray, sw_vec: np.ndarray
            ) -> tuple[LGBMRegressor, CatBoostRegressor]:
                v_cut = int(len(idx) * 0.80)
                t_idx, v_idx = idx[:v_cut], idx[v_cut:]

                l_m = LGBMRegressor(
                    boosting_type="gbdt", objective="huber", n_estimators=300,
                    learning_rate=0.03, num_leaves=7, max_depth=3,
                    min_child_samples=200, subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=2.0, reg_lambda=2.0,
                    n_jobs=self.n_jobs, random_state=42, verbosity=-1,
                )
                l_m.fit(
                    x_mat[t_idx], y_vec[t_idx], sample_weight=sw_vec[t_idx],
                    eval_set=[(x_mat[v_idx], y_vec[v_idx])],
                    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
                )
                c_m = CatBoostRegressor(
                    iterations=300, learning_rate=0.03, depth=3,
                    loss_function="Huber:delta=1.35", eval_metric="RMSE",
                    random_seed=42, verbose=0, od_type="Iter", od_wait=20,
                    allow_writing_files=False
                )
                c_m.fit(
                    x_mat[t_idx], y_vec[t_idx], sample_weight=sw_vec[t_idx],
                    eval_set=(x_mat[v_idx], y_vec[v_idx]),
                    use_best_model=True
                )
                return l_m, c_m

            def _fit_predict_horizon(
                y_vec: np.ndarray, sw_vec: np.ndarray
            ) -> tuple[np.ndarray, LGBMRegressor, CatBoostRegressor, LGBMRegressor, CatBoostRegressor, LGBMRegressor, CatBoostRegressor]:
                mask_vec = sw_vec > 0
                if regime_learning:
                    bull_mask = (p_bull_arr > 0.4) & mask_vec
                    bear_mask = (p_bear_arr > 0.4) & mask_vec

                    lgbm_m, cb_m = _train_ml_pair(np.where(mask_vec)[0], x_clean, y_vec, sw_vec)
                    pred_global_m = 0.6 * lgbm_m.predict(x_clean) + 0.4 * cb_m.predict(x_clean)

                    if bull_mask.sum() > 500:
                        lgbm_bull_m, cb_bull_m = _train_ml_pair(
                            np.where(bull_mask)[0], x_clean, y_vec, sw_vec
                        )
                        pred_bull_m = 0.6 * lgbm_bull_m.predict(x_clean) + 0.4 * cb_bull_m.predict(x_clean)
                    else:
                        lgbm_bull_m, cb_bull_m = lgbm_m, cb_m
                        pred_bull_m = pred_global_m

                    if bear_mask.sum() > 500:
                        lgbm_bear_m, cb_bear_m = _train_ml_pair(
                            np.where(bear_mask)[0], x_clean, y_vec, sw_vec
                        )
                        pred_bear_m = 0.6 * lgbm_bear_m.predict(x_clean) + 0.4 * cb_bear_m.predict(x_clean)
                    else:
                        lgbm_bear_m, cb_bear_m = lgbm_m, cb_m
                        pred_bear_m = pred_global_m

                    p_ch = 1.0 - p_bull_arr - p_bear_arr
                    pred_m = (p_bull_arr * pred_bull_m) + (p_bear_arr * pred_bear_m) + (p_ch * pred_global_m)
                    return pred_m, lgbm_m, cb_m, lgbm_bull_m, cb_bull_m, lgbm_bear_m, cb_bear_m

                lgbm_m, cb_m = _train_ml_pair(np.where(mask_vec)[0], x_clean, y_vec, sw_vec)
                pred_m = 0.6 * lgbm_m.predict(x_clean) + 0.4 * cb_m.predict(x_clean)
                return pred_m, lgbm_m, cb_m, lgbm_m, cb_m, lgbm_m, cb_m

            times_arr_utc = pd.to_datetime(
                aligned_df.index.get_level_values("datetime"), utc=True
            )
            bar_delta = pd.Timedelta(hours=4)
            if len(unique_times) >= 2:
                uniq_t = pd.to_datetime(unique_times, utc=True)
                diffs = uniq_t.to_series().diff().dropna()
                if not diffs.empty:
                    med = diffs.median()
                    if pd.notna(med) and med > pd.Timedelta(0):
                        bar_delta = med

            for h in horizons:
                # 1. Target Construction for horizon h
                if "close" in aligned_df.columns:
                    close_wide = close_ser.unstack(level="symbol")
                    fwd_ret = np.log(close_wide.shift(-h) / close_wide)
                    vol = (
                        close_wide.pct_change()
                        .rolling(max(24, h * 2), min_periods=h + 5)
                        .std()
                        .ffill()
                        .fillna(0.01)
                    )
                    target_h_wide = fwd_ret / (vol * np.sqrt(h) + 1e-9)
                    target_h = (
                        target_h_wide.rank(axis=1, pct=True)
                        .stack(future_stack=True)
                        .reindex(aligned_df.index)
                        .fillna(0.5)
                        .values
                    )
                else:
                    # Normalize [-1, 1] target from panel to [0, 1] for Alpha Miner compatibility
                    target_h = ((aligned_df["target"] + 1.0) / 2.0).values

                target_h_short = 1.0 - target_h
                y_h = np.where(np.isfinite(target_h), target_h, 0.5)
                y_h_short = np.where(np.isfinite(target_h_short), target_h_short, 0.5)
                sw_h = base_sw * (~np.isnan(target_h)).astype(np.float64)

                # Tail-focused emphasis: increase weight near top/bottom quantiles.
                tail_dist = np.abs(y_h - 0.5)
                tail_w = np.clip(1.0 + 1.6 * (tail_dist / 0.5), 1.0, 1.8)
                sw_h = sw_h * tail_w

                if is_end_date:
                    cutoff_ts = pd.to_datetime(is_end_date, utc=True)
                    safe_mask = (times_arr_utc + (bar_delta * int(h))) < cutoff_ts
                    sw_h = sw_h * safe_mask.astype(np.float64)

                # Apply Asymmetric Alpha Training with directional separation.
                # Long head keeps long bias, short head applies the reciprocal side to avoid collapse.
                long_bias = float(OPT_FUTURES_CONFIG.get("FUTURES_GP_LONG_BIAS", 1.0))
                inv_bias = 1.0 / max(long_bias, 1e-6)
                sw_h_long = sw_h.copy()
                sw_h_short = sw_h.copy()
                if long_bias != 1.0:
                    sw_h_long = np.where(target_h > 0.5, sw_h_long * long_bias, sw_h_long)
                    sw_h_short = np.where(target_h <= 0.5, sw_h_short * inv_bias, sw_h_short)

                mask_h = (sw_h_long > 0) | (sw_h_short > 0)
                if mask_h.sum() < 200:
                    continue

                pred_h_long, lgbm_h, cb_h, lgbm_bull, cb_bull, lgbm_bear, cb_bear = _fit_predict_horizon(
                    y_h, sw_h_long
                )
                pred_h_short, _, _, _, _, _, _ = _fit_predict_horizon(y_h_short, sw_h_short)

                is_mask = (sw_h_long > 0) & (sw_h_short > 0)
                if is_mask.sum() > 80:
                    idx_is = np.where(is_mask)[0]
                    n_chunks = 5 if len(idx_is) >= 250 else 4
                    edges = np.linspace(0, len(idx_is), num=n_chunks + 1, dtype=int)
                    chunk_scores: list[float] = []
                    for i in range(n_chunks):
                        c_idx = idx_is[edges[i]:edges[i + 1]]
                        if len(c_idx) < 20:
                            continue
                        ic_l, _ = spearmanr(pred_h_long[c_idx], y_h[c_idx])
                        ic_s, _ = spearmanr(pred_h_short[c_idx], y_h_short[c_idx])
                        ic_l = float(ic_l) if np.isfinite(ic_l) else 0.0
                        ic_s = float(ic_s) if np.isfinite(ic_s) else 0.0
                        chunk_scores.append(0.5 * (ic_l + ic_s))

                    if chunk_scores:
                        score_arr = np.asarray(chunk_scores, dtype=np.float64)
                        mean_ic = float(np.mean(score_arr))
                        std_ic = float(np.std(score_arr))
                        worst_ic = float(np.min(score_arr))
                        # Penalize directional imbalance to reduce one-sided legs.
                        long_share = float(np.mean(pred_h_long[idx_is] > 0.5))
                        short_share = float(np.mean(pred_h_short[idx_is] > 0.5))
                        imbalance = abs(long_share - short_share)
                        robust_score = mean_ic - (0.5 * std_ic) + min(0.0, worst_ic) - (0.10 * imbalance)
                        _logger.debug(
                            " [Phase D] Horizon h=%d: robust=%.4f (mean=%.4f std=%.4f worst=%.4f imb=%.3f)",
                            h, robust_score, mean_ic, std_ic, worst_ic, imbalance
                        )
                    else:
                        robust_score = 0.0
                else:
                    robust_score = 0.0

                if robust_score <= 0.0:
                    _logger.debug(" [Phase D] Horizon h=%d skipped (non-positive robust score)", h)
                    continue
                horizon_ics.append(robust_score)
                horizon_scores.append(robust_score)
                ensemble_preds += pred_h_long * robust_score
                ensemble_preds_long += pred_h_long * robust_score
                ensemble_preds_short += pred_h_short * robust_score
                slot_components.append(pred_h_long)
                long_components.append(pred_h_long)
                short_components.append(pred_h_short)

            if sum(horizon_scores) > 0:
                out_pred = ensemble_preds / sum(horizon_scores)
                out_pred_long = ensemble_preds_long / sum(horizon_scores)
                out_pred_short = ensemble_preds_short / sum(horizon_scores)
            else:
                out_pred = np.zeros(len(aligned_df)) + 0.5
                out_pred_long = np.zeros(len(aligned_df)) + 0.5
                out_pred_short = np.zeros(len(aligned_df)) + 0.5

            slot_components.insert(0, out_pred)
            long_components.insert(0, out_pred_long)
            short_components.insert(0, out_pred_short)

            self._st_global = lgbm_h
            self._cb_global = cb_h
            self._st_bull = lgbm_bull
            self._cb_bull = cb_bull
            self._st_bear = lgbm_bear
            self._cb_bear = cb_bear

            cols = [f"gp_alpha_{i:02d}" for i in range(self.n_features_to_select)]
            full_alpha_df = pd.DataFrame(0.5, index=full_grid_index, columns=cols)
            for i, comp in enumerate(slot_components[: self.n_features_to_select]):
                full_alpha_df[f"gp_alpha_{i:02d}"] = comp

            alpha_series_list = []
            for col in cols:
                unstacked = full_alpha_df[col].unstack(level="symbol")
                if unstacked.std(axis=1).mean() < 1e-12:
                    s = pd.Series(0.5, index=full_grid_index, name=col)
                else:
                    ranked = _pct_uniform_cs_is_fit(unstacked, is_row_mask_utc)
                    s = ranked.stack(future_stack=True)
                    s.name = col
                alpha_series_list.append(s)

            raw_alpha_df = pd.concat(alpha_series_list, axis=1)

            # Apply rank-transformation to directional components to avoid signal collapse from raw LightGBM clustering.
            for key, arr in [("gp_alpha_long_raw", out_pred_long), ("gp_alpha_short_raw", out_pred_short)]:
                tmp_df = pd.Series(arr, index=full_grid_index).unstack(level="symbol")
                if tmp_df.std(axis=1).mean() < 1e-12:
                    raw_alpha_df[key] = 0.5
                else:
                    ranked = _pct_uniform_cs_is_fit(tmp_df, is_row_mask_utc)
                    raw_alpha_df[key] = ranked.stack(future_stack=True)

            raw_alpha_df = raw_alpha_df.loc[:, ~raw_alpha_df.columns.duplicated()].copy()

            best_fitness = 1.0
            raw_alpha_df.attrs["best_fitness"] = float(best_fitness)

            if versioned_cache is not None:
                try:
                    versioned_cache.parent.mkdir(parents=True, exist_ok=True)
                    raw_alpha_df.to_parquet(versioned_cache)
                    meta_cache = versioned_cache.with_suffix(".json")
                    with open(meta_cache, "w") as f:
                        json.dump(dict(raw_alpha_df.attrs), f, indent=2)
                except Exception as e:
                    _logger.warning("LightGBM Raw Cache Write Failed: %s", e)

        fo = filter_options or {}
        alpha_df_all, filt_meta = filter_alpha_components(
            raw_alpha_df.copy(),
            panel_df,
            is_end_date=is_end_date,
            n_trials=self.n_features_to_select,
            fdr_q=float(fo.get("fdr_q", 0.10)),
            use_newey_west=bool(fo.get("use_newey_west", False)),
            use_ewma_ic_stat=bool(fo.get("use_ewma_ic_stat", False)),
            ewma_half_life=float(fo.get("ewma_half_life", 540.0)),
            symbol_balance_max=float(fo.get("symbol_balance_max", 3.0)),
            require_regime_gate=bool(fo.get("require_regime_gate", True)),
        )

        surviving_cols = [
            c
            for c in alpha_df_all.columns
            if c.startswith("gp_alpha_") and c[-2:].isdigit() and alpha_df_all[c].std() > 1e-6
        ]

        if "gp_alpha_00" not in surviving_cols and surviving_cols:
            best_surv = surviving_cols[0]
            _logger.info(" [PROMOTION] Promoting %s to slot 00.", best_surv)
            tmp = alpha_df_all["gp_alpha_00"].copy()
            alpha_df_all["gp_alpha_00"] = alpha_df_all[best_surv]
            alpha_df_all[best_surv] = tmp

        if float(filt_meta.get("neutralize_primary", 0.0)) > 0.5:
            _logger.warning(" [FILTER] gp_alpha_00 neutralized.")
            alpha_df_all["gp_alpha_00"] = 0.5

        surviving_cols = [
            c
            for c in alpha_df_all.columns
            if c.startswith("gp_alpha_") and c[-2:].isdigit() and alpha_df_all[c].std() > 1e-6
        ]
        _logger.warning(" [CRITICAL_DEBUG] Reached mine_alphas_cs with unique run marker.")
        if alpha_df_all.columns.duplicated().any():
            _logger.warning(" [FILTER] Post-filter duplicate columns detected; cleaning. Cols: %s", list(alpha_df_all.columns))
            alpha_df_all = alpha_df_all.loc[:, ~alpha_df_all.columns.duplicated()].copy()

        _logger.info(" [FILTER] alpha_df_all columns: %s", list(alpha_df_all.columns))
        if {"gp_alpha_long_raw", "gp_alpha_short_raw"}.issubset(alpha_df_all.columns):
            val_long = alpha_df_all["gp_alpha_long_raw"]
            _logger.info(" [FILTER] gp_alpha_long_raw type: %s, shape: %s", type(val_long), getattr(val_long, "shape", "N/A"))
            if isinstance(val_long, pd.DataFrame):
                _logger.warning(" [FILTER] gp_alpha_long_raw is still a DataFrame! Columns: %s", list(val_long.columns))
                alpha_df_all["gp_alpha_long"] = val_long.iloc[:, 0].clip(0.0, 1.0)
            else:
                alpha_df_all["gp_alpha_long"] = val_long.clip(0.0, 1.0)

            val_short = alpha_df_all["gp_alpha_short_raw"]
            if isinstance(val_short, pd.DataFrame):
                alpha_df_all["gp_alpha_short"] = 1.0 - val_short.iloc[:, 0].clip(0.0, 1.0)
            else:
                alpha_df_all["gp_alpha_short"] = 1.0 - val_short.clip(0.0, 1.0)
        elif surviving_cols:
            alpha_df_all["gp_alpha_long"] = alpha_df_all[surviving_cols].mean(axis=1)
            alpha_df_all["gp_alpha_short"] = 1.0 - alpha_df_all[surviving_cols].mean(axis=1)
        else:
            alpha_df_all["gp_alpha_long"] = 0.5
            alpha_df_all["gp_alpha_short"] = 0.5

        for tmp_col in ("gp_alpha_long_raw", "gp_alpha_short_raw"):
            if tmp_col in alpha_df_all.columns:
                alpha_df_all.drop(columns=[tmp_col], inplace=True)

        alpha_df = alpha_df_all.reindex(panel_df.index).fillna(0.5)
        alpha_df.attrs["best_fitness"] = raw_alpha_df.attrs.get("best_fitness", 0.0)
        alpha_df.attrs["alpha_component_filter"] = filt_meta

        return alpha_df

    def transform_cs(self, panel_df: pd.DataFrame, cache_path: Path | None = None) -> pd.DataFrame:
        """Transform panel features using the trained alpha model.

        Args:
            panel_df: Input panel dataframe.
            cache_path: Optional path to cache (not used in transform currently).

        Returns:
            Dataframe of predicted alpha components.

        """
        _ = cache_path
        if panel_df.empty:
            return pd.DataFrame()

        work = panel_df.copy()
        src_cols = [c for c in GP_ENGINEERED_FEATURE_NAMES if c in work.columns]
        if src_cols:
            work = CrossSectionalPipelineUtils.cs_rank_transform(work, src_cols)
        feat_cols = _resolve_gp_feature_columns(work)
        x = work[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)

        if hasattr(self, "_st_global") and self._st_global is not None:
            kept = getattr(self, "_kept_indices", list(range(x.shape[1])))
            if x.shape[1] >= len(kept):
                x = x[:, kept]

            mp_mean = getattr(self, "_mp_mean", None)
            mp_evecs = getattr(self, "_mp_evecs", None)
            if mp_mean is not None and mp_evecs is not None:
                mp_std = getattr(self, "_mp_std", 1.0)
                x_norm = (x - mp_mean) / mp_std
                x_denoised = np.dot(np.dot(x_norm, mp_evecs), mp_evecs.T)
                x = x_denoised * mp_std + mp_mean

            # Prediction with regime blending
            p_bull_ser = work.get("hmm_prob_bull_trend", pd.Series(0.0, index=work.index))
            p_bull = p_bull_ser.fillna(0.0).values
            p_bear_ser = work.get("hmm_prob_bear_trend", pd.Series(0.0, index=work.index))
            p_bear = p_bear_ser.fillna(0.0).values
            
            # Global prediction
            pred_global = 0.6 * self._st_global.predict(x)
            if self._cb_global is not None:
                pred_global += 0.4 * self._cb_global.predict(x)
            else:
                pred_global = pred_global / 0.6
            
            # Bull prediction
            if self._st_bull is not None:
                pred_bull = 0.6 * self._st_bull.predict(x)
                if self._cb_bull is not None:
                    pred_bull += 0.4 * self._cb_bull.predict(x)
                else:
                    pred_bull = pred_bull / 0.6
            else:
                pred_bull = pred_global
                
            # Bear prediction
            if self._st_bear is not None:
                pred_bear = 0.6 * self._st_bear.predict(x)
                if self._cb_bear is not None:
                    pred_bear += 0.4 * self._cb_bear.predict(x)
                else:
                    pred_bear = pred_bear / 0.6
            else:
                pred_bear = pred_global
                
            p_ch = 1.0 - p_bull - p_bear
            out_arr_raw = (p_bull * pred_bull) + (p_bear * pred_bear) + (p_ch * pred_global)
            
            # Rank-transform raw output to get uniform distribution, matching training logic
            unstacked = pd.Series(out_arr_raw, index=panel_df.index).unstack(level="symbol")
            ranked = unstacked.rank(axis=1, pct=True)
            out_arr = ranked.stack(future_stack=True).reindex(panel_df.index).fillna(0.5)

            cols = [f"gp_alpha_{i:02d}" for i in range(self.n_features_to_select)]
            out_grid = np.full((x.shape[0], self.n_features_to_select), 0.5, dtype=np.float64)
            out_grid[:, 0] = out_arr
            out_df = pd.DataFrame(out_grid, index=panel_df.index, columns=cols)
            out_df["gp_alpha_long"] = out_arr
            out_df["gp_alpha_short"] = 1.0 - out_arr
            return out_df

        _logger.warning("transform_cs: _st_global is missing. Returning neutral features.")
        cols = [f"gp_alpha_{i:02d}" for i in range(self.n_features_to_select)]
        out_df = pd.DataFrame(0.5, index=panel_df.index, columns=cols)
        out_df["gp_alpha_long"] = 0.5
        out_df["gp_alpha_short"] = 0.5
        return out_df
