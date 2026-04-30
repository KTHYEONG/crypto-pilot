"""GP Alpha Miner - Cross-Sectional Ranking Edition (LightGBM Architecture Swap).

Learns a universal formula across multiple symbols using panel data and CS-IC fitness.
"""

from __future__ import annotations

import hashlib
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

from src.domain.futures.ml_pipeline.cross_sectional_utils import CrossSectionalPipelineUtils
from src.domain.futures.ml_pipeline.feature_engineering import (
    GP_ENGINEERED_FEATURE_NAMES,
    GP_FEATURE_SCHEMA_VERSION,
)
from src.domain.futures.ml_pipeline.gp_alpha_filter import filter_gp_alpha_columns

_logger = logging.getLogger(__name__)


def _generate_smart_cache_stem(
    pop: int, gen: int, horizons: tuple[int, ...], symbols: list[str], is_end_date: str | None
) -> str:
    """Generate a unique cache stem based on miner configuration and data scope.

    Args:
        pop: Population size.
        gen: Number of generations.
        horizons: Target horizons.
        symbols: List of symbols used.
        is_end_date: Cutoff date for In-Sample data.

    Returns:
        A hash-tagged string stem for filenames.

    """
    h_str = "-".join(map(str, horizons))
    prefix = f"lgbm_univ_s{len(symbols)}_h{h_str}"

    dna = {
        "pop": pop,
        "gen": gen,
        "horizons": horizons,
        "symbols": sorted(symbols),
        "is_end_date": is_end_date,
        "version": "v1_lgbm",
        "feature_schema_version": GP_FEATURE_SCHEMA_VERSION,
        "feature_columns": list(GP_ENGINEERED_FEATURE_NAMES),
    }
    dna_json = json.dumps(dna, sort_keys=True)
    short_hash = hashlib.sha256(dna_json.encode()).hexdigest()[:8]

    return f"{prefix}_{short_hash}"


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
    # Include HMM probability features if present
    hmm_cols = [c for c in panel_df.columns if c.startswith("hmm_prob_")]
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

    return sorted(base_features | set(hmm_cols))


@dataclass
class GPAlphaMiner:
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
    _st: LGBMRegressor | None = field(default=None, init=False)
    _kept_indices: list[int] | None = field(default=None, init=False)
    _mp_mean: np.ndarray | None = field(default=None, init=False)
    _mp_std: np.ndarray | float | None = field(default=None, init=False)
    _mp_evecs: np.ndarray | None = field(default=None, init=False)

    def _cleanup_old_caches(self, cache_path: Path, max_age_days: int = 7) -> None:
        """Remove stale cache files.

        Args:
            cache_path: Path to cache file (used for directory resolution).
            max_age_days: Maximum age of cache files in days.

        """
        try:
            import time

            now = time.time()
            cache_dir = cache_path.parent
            if not cache_dir.exists():
                return
            patterns = ["lgbm_univ_*", "raw_lgbm_univ_*", "gp_univ_*", "raw_gp_univ_*"]
            for pattern in patterns:
                for f in cache_dir.glob(pattern):
                    if f.is_file() and (now - f.stat().st_mtime) > (max_age_days * 86400):
                        try:
                            f.unlink()
                        except Exception as e:
                            _logger.debug("Failed to unlink old cache: %s", e)
        except Exception as e:
            _logger.debug("Cleanup failed: %s", e)

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
        if cache_path is not None:
            self._cleanup_old_caches(cache_path)
            symbols = list(work_panel.index.get_level_values("symbol").unique())
            tag = "raw_" + _generate_smart_cache_stem(
                self.population_size, self.generations, self.target_horizons, symbols, is_end_date
            )
            versioned_cache = cache_path.with_name(f"{tag}.parquet")
            meta_cache = versioned_cache.with_suffix(".json")
            if versioned_cache.exists():
                try:
                    loaded = pd.read_parquet(versioned_cache)
                    if len(loaded) == len(panel_df):
                        _logger.info("LightGBM CS Cache Hit (RAW): %s.", versioned_cache.name)
                        if meta_cache.exists():
                            with open(meta_cache) as f:
                                loaded.attrs.update(json.load(f))
                        raw_alpha_df = loaded
                except Exception as e:
                    _logger.debug("LightGBM Raw Cache load failed: %s", e)

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
            horizons = (3, 6, 12, 24)
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

            # --- Ergodicity & Time Decay Weighting ---
            if "hmm_prob_crisis" in aligned_df.columns:
                p_crisis = aligned_df["hmm_prob_crisis"].fillna(0.0).to_numpy(dtype=np.float64)
                erg_weight = np.clip(1.0 - p_crisis * 0.7, 0.1, 1.0)
                base_sw *= erg_weight

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
                _logger.info(
                    " [Phase C] Kept %d/%d features via MI Audit.",
                    len(kept_indices),
                    len(feat_cols)
                )
            else:
                self._kept_indices = list(range(len(feat_cols)))

            # --- Multi-Horizon Loop ---
            ensemble_preds = np.zeros(len(aligned_df), dtype=np.float64)
            horizon_ics = []
            close_ser = aligned_df.get("close", pd.Series(np.nan, index=aligned_df.index))
            lgbm_h = None

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
                    target_h = aligned_df["target"].values

                y_h = np.where(np.isfinite(target_h), target_h, 0.5)
                sw_h = base_sw * (~np.isnan(target_h)).astype(np.float64)
                mask_h = sw_h > 0

                if mask_h.sum() < 200:
                    continue

                mask_idx = np.where(mask_h)[0]
                val_cutoff = int(len(mask_idx) * 0.80)
                tr_idx, val_idx = mask_idx[:val_cutoff], mask_idx[val_cutoff:]

                lgbm_h = LGBMRegressor(
                    boosting_type="gbdt", objective="huber", n_estimators=300,
                    learning_rate=0.03, num_leaves=31, max_depth=6,
                    min_child_samples=70, subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=1.2, reg_lambda=1.2,
                    n_jobs=self.n_jobs, random_state=42,
                )
                lgbm_h.fit(
                    x_clean[tr_idx], y_h[tr_idx], sample_weight=sw_h[tr_idx],
                    eval_set=[(x_clean[val_idx], y_h[val_idx])],
                    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
                )

                # --- CatBoost Ensemble (Phase D) ---
                cb_h = CatBoostRegressor(
                    iterations=300, learning_rate=0.03, depth=5,
                    loss_function="Huber:delta=1.35", eval_metric="RMSE",
                    random_seed=42, verbose=0, od_type="Iter", od_wait=20,
                    allow_writing_files=False
                )
                cb_h.fit(
                    x_clean[tr_idx], y_h[tr_idx], sample_weight=sw_h[tr_idx],
                    eval_set=(x_clean[val_idx], y_h[val_idx]),
                    use_best_model=True
                )

                # Blend: 0.6 * LGBM + 0.4 * CatBoost
                pred_h = 0.6 * lgbm_h.predict(x_clean) + 0.4 * cb_h.predict(x_clean)

                is_mask = (sw_h > 0) & (np.arange(len(sw_h)) < len(sw_h) * 0.8)
                if is_mask.sum() > 50:
                    ic_h, _ = spearmanr(pred_h[is_mask], y_h[is_mask])
                else:
                    ic_h = 0.0

                _logger.info(" [Phase D] Horizon h=%d: Blended IS Rank-IC = %.4f", h, ic_h)
                if ic_h <= 0.0:
                    _logger.info(" [Phase D] Horizon h=%d skipped (non-positive IC)", h)
                    continue
                horizon_ics.append(ic_h)
                ensemble_preds += pred_h * ic_h

            if sum(horizon_ics) > 0:
                out_pred = ensemble_preds / sum(horizon_ics)
            else:
                out_pred = np.zeros(len(aligned_df)) + 0.5

            self._st = lgbm_h
            cols = [f"gp_alpha_{i:02d}" for i in range(self.n_features_to_select)]
            full_alpha_df = pd.DataFrame(0.5, index=full_grid_index, columns=cols)
            full_alpha_df["gp_alpha_00"] = out_pred

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
        alpha_df_all, filt_meta = filter_gp_alpha_columns(
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
            if c.startswith("gp_alpha_") and alpha_df_all[c].std() > 1e-6
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

        alpha_df = alpha_df_all.reindex(panel_df.index).fillna(0.5)
        alpha_df.attrs["best_fitness"] = raw_alpha_df.attrs.get("best_fitness", 0.0)
        alpha_df.attrs["gp_alpha_filter"] = filt_meta

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

        if hasattr(self, "_st") and self._st is not None:
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

            out_arr = self._st.predict(x)
            cols = [f"gp_alpha_{i:02d}" for i in range(self.n_features_to_select)]
            out_grid = np.full((x.shape[0], self.n_features_to_select), 0.5, dtype=np.float64)
            out_grid[:, 0] = out_arr
            return pd.DataFrame(out_grid, index=panel_df.index, columns=cols)

        _logger.warning("transform_cs: _st is missing. Returning neutral features.")
        cols = [f"gp_alpha_{i:02d}" for i in range(self.n_features_to_select)]
        return pd.DataFrame(0.5, index=panel_df.index, columns=cols)
