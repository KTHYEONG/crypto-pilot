"""ML Alpha Miner v5 - LambdaRank + Theme Subspacing Edition.

Replaces regression with 'Learning to Rank' for cross-sectional alpha mining.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import numba
from lightgbm import LGBMRegressor, LGBMRanker

from config.opt_config import OPT_FUTURES_CONFIG
from src.core.utils.cache_manager import CacheManager
from src.domain.futures.ml_pipeline.alpha.component_filter import filter_alpha_components
from src.domain.futures.ml_pipeline.features.engineering import (
    GP_FEATURE_SCHEMA_VERSION,
    HMM_SEMANTIC_PROB_COLUMNS,
    add_macro_interaction_features,
)

# Semantic HMM posteriors — Vol/MR group only (near-zero CS spread but useful with vol features).
HMM_COLS = list(HMM_SEMANTIC_PROB_COLUMNS)

_logger = logging.getLogger(__name__)

@numba.njit(parallel=False, cache=True)
def _fast_rank_2d_numba(array: np.ndarray) -> np.ndarray:
    """Vectorized percentile ranking across rows (axis=1) with tie-breaking 'average'.
    
    Matches pd.DataFrame(array).rank(axis=1, pct=True, method='average') but 10-20x faster.
    """
    n, m = array.shape
    out = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        row = array[i]
        mask = ~np.isnan(row)
        n_valid = np.sum(mask)
        if n_valid <= 1:
            out[i, :] = 0.5
            continue
            
        valid_data = row[mask]
        sort_idx = np.argsort(valid_data)
        sorted_data = valid_data[sort_idx]
        ranks = np.empty(n_valid, dtype=np.float64)
        
        j = 0
        while j < n_valid:
            k = j + 1
            while k < n_valid and sorted_data[k] == sorted_data[j]:
                k += 1
            avg_rank = j + (k - j - 1) / 2.0
            for m_idx in range(j, k):
                ranks[sort_idx[m_idx]] = avg_rank
            j = k
            
        out_row = np.full(m, 0.5)
        out_row[mask] = ranks / (n_valid - 1)
        out[i] = out_row
    return out

# v5 Theme Subspacing Definitions
THEME_GROUPS = {
    0: [  # Group 1: Trend/Momentum (Slots 00-04) + Vol-Adjusted
        "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", 
        "ma_dist_24", "ma_dist_168", "ret_vol_adj_24",
        "realized_vol_yz_24", "orderflow_price_divergence",
        "taker_absorption_score"
    ],
    1: [  # Group 2: Volatility/Mean-Reversion (Slots 05-09)
        "vol_ratio_24", "vol_ratio_168", "macro_vol_regime_shift", 
        "cs_dispersion", "dist_from_weekly_vwap",
        "liq_intensity_proxy", "capitulation_proxy", "tail_risk_24"
    ],
    2: [  # Group 3: Structural/Regime (Slots 10-14)
        "btc_beta_x_bull_trend",
        "realized_vol_x_crisis",
        "funding_x_bear_trend",
        "macro_trend_24h",
        "funding_rate",
    ]
}


def _train_ranker_slot(
    slot_idx: int,
    slots_per_theme: int,
    work_df: pd.DataFrame,
    y_labels: np.ndarray,
    is_mask: np.ndarray,
    sample_weights_all: np.ndarray,
    is_group_sizes: np.ndarray,
    n_jobs: int = 1,
) -> Tuple[int, Optional[LGBMRanker], list[str], np.ndarray]:
    """Helper for parallel Ranker training."""
    theme_idx = min(2, slot_idx // slots_per_theme)
    base_feats = THEME_GROUPS[theme_idx]
    if theme_idx == 1:
        feat_cols = list(dict.fromkeys(base_feats + HMM_COLS))
    else:
        feat_cols = list(dict.fromkeys(base_feats))

    feat_cols = [c for c in feat_cols if c in work_df.columns]
    
    if not feat_cols:
        return slot_idx, None, [], np.full(len(work_df), 0.5)

    # [Optimization] Float32 downcasting for memory bandwidth
    X = work_df[feat_cols].values.astype(np.float32)

    X_is = X[is_mask]
    y_is = y_labels[is_mask].astype(np.float32)
    w_is = sample_weights_all[is_mask].astype(np.float32)

    theme_idx = min(2, slot_idx // slots_per_theme)
    # [Optimization] Differential n_estimators by theme (DART is non-early-stopping)
    # Theme 0 (Trend): 80, Theme 1 (Vol): 70, Theme 2 (Structural): 60
    n_est = {0: 80, 1: 70, 2: 60}.get(theme_idx, 100)

    model = LGBMRanker(
        boosting_type="dart",
        objective="lambdarank",
        metric="ndcg",
        eval_at=[1, 3],
        n_estimators=n_est,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        random_state=42 + slot_idx,
        deterministic=True,
        n_jobs=n_jobs,
        importance_type="gain",
        verbosity=-1,
        drop_rate=0.1,
        skip_drop=0.5,
        max_drop=15,
        uniform_drop=True,
        max_bin=63,
    )

    model.fit(X_is, y_is, group=is_group_sizes, sample_weight=w_is)
    raw_scores = model.predict(X)
    return slot_idx, model, feat_cols, raw_scores


def _train_regressor_slot(
    slot_idx: int,
    slots_per_theme: int,
    feat_cols: list[str],
    work_df: pd.DataFrame,
    y_mag_vals: np.ndarray,
    is_mask: np.ndarray,
    mag_finite: np.ndarray,
    sample_weights_all: np.ndarray,
    n_jobs: int = 1,
) -> Tuple[int, Optional[LGBMRegressor], np.ndarray]:
    """Helper for parallel Regressor training."""
    if not feat_cols:
        return slot_idx, None, np.zeros(len(work_df))

    # [Optimization] Float32 downcasting for memory bandwidth
    X = work_df[feat_cols].values.astype(np.float32)
    train_mag = is_mask & mag_finite
    if int(train_mag.sum()) < 200:
        return slot_idx, None, np.zeros(len(work_df))

    X_mag = X[train_mag]
    y_mag = y_mag_vals[train_mag].astype(np.float32)
    w_mag = sample_weights_all[train_mag].astype(np.float32)

    theme_idx = min(2, slot_idx // slots_per_theme)
    n_est = {0: 80, 1: 70, 2: 60}.get(theme_idx, 100)

    reg = LGBMRegressor(
        objective="regression_l1",
        n_estimators=n_est,
        learning_rate=0.05,
        random_state=42 + slot_idx,
        n_jobs=n_jobs,
        verbosity=-1,
    )
    reg.fit(X_mag, y_mag, sample_weight=w_mag)
    mag_raw = reg.predict(X)
    return slot_idx, reg, mag_raw


def _train_combined_slot(
    slot_idx: int,
    slots_per_theme: int,
    work_df: pd.DataFrame,
    y_labels: np.ndarray,
    y_mag_vals: np.ndarray,
    is_mask: np.ndarray,
    mag_finite: np.ndarray,
    sample_weights_all: np.ndarray,
    is_group_sizes: np.ndarray,
    n_jobs: int = 1,
) -> Tuple[int, Optional[LGBMRanker], Optional[LGBMRegressor], list[str], np.ndarray, np.ndarray]:
    """Execute both Ranker and Regressor training in one task for pipeline efficiency."""
    s_idx, model, feat_cols, raw_scores = _train_ranker_slot(
        slot_idx, slots_per_theme, work_df, y_labels, is_mask, sample_weights_all, is_group_sizes, n_jobs
    )
    
    if model is not None and feat_cols:
        _, reg, mag_raw = _train_regressor_slot(
            slot_idx, slots_per_theme, feat_cols, work_df, y_mag_vals, is_mask, mag_finite, sample_weights_all, n_jobs
        )
    else:
        reg, mag_raw = None, np.zeros(len(work_df))
        
    return slot_idx, model, reg, feat_cols, raw_scores, mag_raw


# Bars ahead for magnitude targets / hybrid scaling (stacked OHLC timeline).
_MAG_HORIZON_BARS = 24


@dataclass
class MLAlphaMiner:
    """Miner for evolving cross-sectional alpha components using LightGBM LambdaRank."""

    # Total rank heads = slots_per_theme × 3 thematic buckets (see THEME_GROUPS).
    slots_per_theme: int = 5
    n_features_to_select: int = 15
    target_horizons: tuple[int, ...] = (3, 6, 12, 24)
    n_jobs: int = 4

    def __post_init__(self):
        # Ensure n_features_to_select is perfectly aligned with thematic themes
        self.n_features_to_select = self.slots_per_theme * 3

    # Internal state
    _models: dict[int, LGBMRanker] = field(default_factory=dict, init=False)
    _mag_models: dict[int, LGBMRegressor] = field(default_factory=dict, init=False)
    _feature_sets: dict[int, list[str]] = field(default_factory=dict, init=False)
    # IS Spearman IC^2 weights per slot (matches mine_alphas_cs ensemble → transform_cs)
    _ic_weights: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    ic_by_slot: dict[str, float] = field(default_factory=dict, init=False)

    def _prepare_labels(
        self, 
        target: pd.Series, 
        raw_returns: pd.Series | np.ndarray | None = None,
        dispersion: pd.Series | None = None,
        atr_24h_pct: pd.Series | np.ndarray | None = None,
        friction_bps: float = 7.0
    ) -> np.ndarray:
        """Convert continuous rank targets [-1, 1] into discrete LambdaRank labels {0, 1, 2, 3}.
        
        v6.5.0 Dynamic Friction Hurdle:
        - Replaces fixed friction with max(1.5*friction, 0.4*ATR_24h_pct).
        - Boosts signal magnitude by ensuring volatility-relative targets.
        """
        # Default to Label 1 (Chop)
        labels = np.ones(len(target), dtype=np.int32)
        
        # Calculate friction threshold in decimal (e.g. 3.5 bps = 0.00035)
        friction = friction_bps / 10000.0
        
        # Convert to numpy for faster vectorized operations
        t = (target.to_numpy() / 2.0) + 0.5
        ret = raw_returns.to_numpy() if hasattr(raw_returns, "to_numpy") else raw_returns
        atr = atr_24h_pct.to_numpy() if hasattr(atr_24h_pct, "to_numpy") else atr_24h_pct
        
        friction = friction_bps / 10000.0
        labels = np.ones(len(t), dtype=np.int32)

        if ret is None:
            labels[t > 0.85] = 3
            labels[t < 0.15] = 0
            labels[(t > 0.60) & (labels == 1)] = 2
        else:
            hurdle = np.maximum(1.5 * friction, 0.4 * atr) if atr is not None else 1.5 * friction
            
            # [Optimization #9] Vectorized Conditional Assignment
            is_strong_long = (t > 0.85) & (ret > hurdle)
            is_strong_short = (t < 0.15) & (ret < -hurdle)
            is_mild_long = (~is_strong_long) & (~is_strong_short) & (t > 0.60) & (ret > friction)
            
            labels[is_strong_long] = 3
            labels[is_strong_short] = 0
            labels[is_mild_long] = 2
        
        return labels

    def mine_alphas_cs(
        self,
        panel_df: pd.DataFrame,
        cache_path: Path | None = None,
        is_end_date: str | None = None,
        filter_options: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Train 3×slots_per_theme LambdaRank heads (Trend / Vol+MR+HMMraw / Interaction)."""
        if panel_df.empty:
            return pd.DataFrame()

        if self.n_features_to_select != self.slots_per_theme * 3:
            raise ValueError(
                "n_features_to_select must equal 3 × slots_per_theme "
                f"({self.n_features_to_select} vs slots_per_theme={self.slots_per_theme})"
            )

        force_retrain = bool(OPT_FUTURES_CONFIG.get("FUTURES_ML_FORCE_RETRAIN_ALPHA", False))
        versioned_cache: Path | None = None
        raw_alpha_df: pd.DataFrame | None = None

        if cache_path is not None:
            cm = CacheManager(cache_path.parent, max_files=5)
            symbols = sorted(list(panel_df.index.get_level_values("symbol").unique()))
            # [Optimization] Use data fingerprint instead of source-file hash.
            # Source-file hash invalidates cache on every code edit, even unrelated ones.
            # miner_version must be bumped manually when model logic changes intentionally.
            _times = panel_df.index.get_level_values("datetime")
            _last_ts = str(_times.max()) if len(_times) > 0 else "none"
            deps = {
                "horizons": self.target_horizons,
                "symbols": symbols,
                "is_end_date": is_end_date,
                "feature_schema_version": GP_FEATURE_SCHEMA_VERSION,
                "miner_version": "v6_lambdarank_dart_mag_g2_fast_v3",
                "slots_per_theme": int(self.slots_per_theme),
                "n_rows": len(panel_df),
                "last_ts": _last_ts,
            }
            tag = cm.generate_hash(deps)
            versioned_cache = cm.get_cache_path("raw_lgbm_v6", ".parquet", tag)

            if not force_retrain and versioned_cache.exists():
                try:
                    raw_alpha_df = pd.read_parquet(versioned_cache)
                    self._mag_models.clear()
                    _logger.info("MLAlphaMiner v6 Cache Hit: %s", versioned_cache.name)
                except Exception as e:
                    _logger.warning("Cache load failed: %s", e)

        if raw_alpha_df is None:
            _logger.info("Training MLAlphaMiner v6 (DART + Dynamic Labeling)...")
            
            # Sort by datetime for group calculation
            work_df = panel_df.sort_index(level="datetime")
            
            # [Localization] Add macro interaction features (Theme Group 3)
            work_df = add_macro_interaction_features(work_df)
            
            # [Optimization] Pre-clean feature columns once
            all_feat_cols = set()
            for group_feats in THEME_GROUPS.values():
                all_feat_cols.update(group_feats)
            all_feat_cols.update(HMM_COLS)
            
            existing_feats = [c for c in all_feat_cols if c in work_df.columns]
            if existing_feats:
                work_df[existing_feats] = work_df[existing_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # [Optimization] Efficient wide-flat mapping
            close_wide = work_df["close"].unstack(level="symbol")
            close_clipped = close_wide.clip(lower=1e-12)
            valid_mask = close_wide.notna().values # Since work_df is sorted by (datetime, symbol), this matches perfectly
            idx_shape = close_wide.shape
            
            # Efficient vectorized calculation of forward returns and ATR
            fwd_ret_6_wide = np.log(close_wide.shift(-6) / close_clipped)
            fwd_log_mag_horizon_wide = np.log(close_wide.shift(-_MAG_HORIZON_BARS) / close_clipped)
            
            high_wide = work_df["high"].unstack(level="symbol")
            low_wide = work_df["low"].unstack(level="symbol")
            close_shifted_1 = close_wide.shift(1)
            tr_wide = np.maximum(high_wide - low_wide, 
                                np.maximum((high_wide - close_shifted_1).abs(), 
                                           (low_wide - close_shifted_1).abs()))
            atr_24_wide = tr_wide.rolling(24).mean()
            atr_24_pct_wide = atr_24_wide / close_clipped

            # Map wide calculations back to flat work_df alignment (row-major)
            fwd_ret_6 = np.nan_to_num(fwd_ret_6_wide.values[valid_mask])
            atr_24_pct = np.nan_to_num(atr_24_pct_wide.values[valid_mask])
            y_mag_h_vals = np.abs(fwd_log_mag_horizon_wide.values[valid_mask])
            
            y_mag_h = pd.Series(y_mag_h_vals, index=work_df.index).replace([np.inf, -np.inf], np.nan)
            
            y_labels = self._prepare_labels(
                work_df["target"], 
                raw_returns=fwd_ret_6,
                dispersion=work_df.get("cs_dispersion"),
                atr_24h_pct=atr_24_pct
            )
            
            # In-Sample Masking
            if is_end_date:
                cutoff = pd.to_datetime(is_end_date, utc=True)
                is_mask = work_df.index.get_level_values("datetime") < cutoff
            else:
                is_mask = np.ones(len(work_df), dtype=bool)

            # [Plan B-1] Sample Weighting (Dispersion-Aware)
            if "cs_dispersion" in work_df.columns:
                raw_weights = work_df["cs_dispersion"].to_numpy()
            else:
                raw_weights = work_df["target"].abs().to_numpy()
            
            sample_weights_all = raw_weights / (raw_weights.mean() + 1e-12)

            slots_df = pd.DataFrame(index=work_df.index)
            is_group_sizes = work_df[is_mask].groupby("datetime").size().to_numpy()
            # [Optimization] Integrated 1-Pass Training (Ranker + Regressor)
            y_mag_vals_raw = y_mag_h.to_numpy(dtype=np.float64)
            mag_finite = np.isfinite(y_mag_vals_raw) & (y_mag_vals_raw >= 0)
            self._mag_models.clear()
            max_workers = 12

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _train_combined_slot,
                        i,
                        self.slots_per_theme,
                        work_df,
                        y_labels,
                        y_mag_vals_raw,
                        is_mask,
                        mag_finite,
                        sample_weights_all,
                        is_group_sizes,
                        n_jobs=1
                    )
                    for i in range(self.n_features_to_select)
                ]
                
                for future in futures:
                    slot_idx, model, reg, feat_cols, raw_scores, mag_raw = future.result()
                    
                    # Store Ranker Results
                    if model is not None:
                        self._models[slot_idx] = model
                        self._feature_sets[slot_idx] = feat_cols
                        scores_matrix = np.full(idx_shape, np.nan)
                        scores_matrix[valid_mask] = raw_scores
                        ranked_matrix = _fast_rank_2d_numba(scores_matrix)
                        slots_df[f"ml_alpha_{slot_idx:02d}"] = ranked_matrix[valid_mask]
                    else:
                        slots_df[f"ml_alpha_{slot_idx:02d}"] = 0.5
                    
                    # Store Regressor Results
                    if reg is not None:
                        self._mag_models[slot_idx] = reg
                        mu_m, sig_m = float(np.mean(mag_raw)), float(np.std(mag_raw) + 1e-9)
                        slots_df[f"ml_mag_{slot_idx:02d}"] = np.clip((mag_raw - mu_m) / sig_m, -3.0, 3.0)
                    else:
                        slots_df[f"ml_mag_{slot_idx:02d}"] = 0.0

            raw_alpha_df = slots_df
            
            if versioned_cache:
                versioned_cache.parent.mkdir(parents=True, exist_ok=True)
                raw_alpha_df.to_parquet(versioned_cache)

        # Apply filtering
        filter_opts = filter_options or {}
        alpha_df_all, filt_meta = filter_alpha_components(
            raw_alpha_df.copy(),
            panel_df,
            is_end_date=is_end_date,
            n_trials=self.n_features_to_select,
            fdr_q=float(filter_opts.get("fdr_q", 0.10)),
            symbol_balance_max=float(filter_opts.get("symbol_balance_max", 3.0)),
            require_regime_gate=bool(filter_opts.get("require_regime_gate", True)),
        )

        surviving = [c for c in alpha_df_all.columns if c.startswith("ml_alpha_") and alpha_df_all[c].std() > 1e-6]
        if surviving:
            ic_map = filt_meta.get("ic_by_slot", {})
            weights = [max(0.0, ic_map.get(c, 0.0)) ** 2 for c in surviving]
            w_arr = np.array(weights)
            if w_arr.sum() > 1e-9:
                w_norm = w_arr / w_arr.sum()
                long_rank = (alpha_df_all[surviving] * w_norm).sum(axis=1)
                self._ic_weights = {int(c.split("_")[2]): float(w) for c, w in zip(surviving, w_norm)}
            else:
                long_rank = alpha_df_all[surviving].mean(axis=1)
                self._ic_weights = {int(c.split("_")[2]): 1.0 / len(surviving) for c in surviving}

            mag_surv_cols = [f"ml_mag_{c.split('_')[2]}" for c in surviving if f"ml_mag_{c.split('_')[2]}" in alpha_df_all.columns]
            if mag_surv_cols:
                mag_blend = alpha_df_all[mag_surv_cols].mean(axis=1)
                alpha_df_all["ml_alpha_long"] = np.clip(long_rank * (1.0 + 0.3 * mag_blend), 0.02, 0.98)
            else:
                alpha_df_all["ml_alpha_long"] = long_rank
            alpha_df_all["ml_alpha_short"] = 1.0 - alpha_df_all["ml_alpha_long"]
        else:
            self._ic_weights = {}
            alpha_df_all["ml_alpha_long"] = 0.5
            alpha_df_all["ml_alpha_short"] = 0.5

        out_df = alpha_df_all.reindex(panel_df.index).fillna(0.5)
        out_df.attrs["alpha_component_filter"] = filt_meta
        out_df.attrs["best_fitness"] = 1.0 if len(surviving) > 5 else 0.1
        return out_df

    def transform_cs(self, panel_df: pd.DataFrame, cache_path: Path | None = None) -> pd.DataFrame:
        """Apply trained v5 models to new panel data."""
        if panel_df.empty or not self._models:
            _logger.warning("transform_cs: No models trained or empty panel.")
            cols = [f"ml_alpha_{i:02d}" for i in range(self.n_features_to_select)]
            return pd.DataFrame(0.5, index=panel_df.index, columns=cols)

        out_df = pd.DataFrame(index=panel_df.index)
        work_df = add_macro_interaction_features(panel_df).sort_index(level="datetime")
        
        all_feat_cols = set()
        for group_feats in THEME_GROUPS.values():
            all_feat_cols.update(group_feats)
        all_feat_cols.update(HMM_COLS)
        existing_feats = [c for c in all_feat_cols if c in work_df.columns]
        if existing_feats:
            work_df[existing_feats] = work_df[existing_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # [Optimization] Efficient wide-flat mapping
        close_wide = work_df["close"].unstack(level="symbol")
        valid_mask = close_wide.notna().values
        idx_shape = close_wide.shape

        for slot_idx, model in self._models.items():
            feat_cols = self._feature_sets.get(slot_idx, [])
            if not feat_cols:
                out_df[f"ml_alpha_{slot_idx:02d}"] = 0.5
                continue

            # [Optimization] Float32 downcasting for memory bandwidth
            X = work_df[feat_cols].values.astype(np.float32)
            raw_scores = model.predict(X)

            # [Optimization] Fast Vectorized Ranking
            scores_matrix = np.full(idx_shape, np.nan)
            scores_matrix[valid_mask] = raw_scores
            ranked_matrix = _fast_rank_2d_numba(scores_matrix)
            out_df[f"ml_alpha_{slot_idx:02d}"] = ranked_matrix[valid_mask]

            mag_model = self._mag_models.get(slot_idx)
            if mag_model is not None:
                mag_raw = mag_model.predict(X)
                mu_m, sig_m = float(np.mean(mag_raw)), float(np.std(mag_raw) + 1e-9)
                out_df[f"ml_mag_{slot_idx:02d}"] = np.clip((mag_raw - mu_m) / sig_m, -3.0, 3.0)
            else:
                out_df[f"ml_mag_{slot_idx:02d}"] = 0.0

        surviving = [c for c in out_df.columns if c.startswith("ml_alpha_") and out_df[c].std() > 1e-6]
        if surviving:
            if self._ic_weights:
                w = np.array([self._ic_weights.get(int(c.split("_")[2]), 0.0) for c in surviving])
                s = float(w.sum())
                if s > 1e-9:
                    long_rank = (out_df[surviving] * (w / s)).sum(axis=1)
                else:
                    long_rank = out_df[surviving].mean(axis=1)
            else:
                long_rank = out_df[surviving].mean(axis=1)

            mag_surv_cols = [f"ml_mag_{c.split('_')[2]}" for c in surviving if f"ml_mag_{c.split('_')[2]}" in out_df.columns]
            if mag_surv_cols:
                mag_blend = out_df[mag_surv_cols].mean(axis=1)
                out_df["ml_alpha_long"] = np.clip(long_rank * (1.0 + 0.3 * mag_blend), 0.02, 0.98)
            else:
                out_df["ml_alpha_long"] = long_rank
            out_df["ml_alpha_short"] = 1.0 - out_df["ml_alpha_long"]
        else:
            out_df["ml_alpha_long"] = 0.5
            out_df["ml_alpha_short"] = 0.5

        return out_df.reindex(panel_df.index).fillna(0.5)
