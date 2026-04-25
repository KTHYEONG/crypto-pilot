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

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
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
    cs_names = sorted(c for c in panel_df.columns if c.startswith("cs_"))
    if cs_names:
        extras = [c for c in ("cross_vol_rank", "cross_ret_24h_rank") if c in panel_df.columns]
        return sorted({c for c in cs_names + extras if c not in blocked})
    return [c for c in panel_df.columns if c not in blocked]


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
        n_symbols = len(unique_symbols)

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
            feat_cols = _resolve_gp_feature_columns(work_panel)
            template = pd.DataFrame(index=full_grid_index)
            join_cols = [*feat_cols, "target"]
            if "tbm_gp_weight" in work_panel.columns:
                join_cols = [*feat_cols, "tbm_gp_weight", "target"]
            aligned_df = template.join(work_panel[join_cols]).fillna(np.nan)

            y_raw = aligned_df["target"].values
            x_grid = aligned_df[feat_cols].values

            sample_weight = np.where(pd.isna(y_raw), 0.0, 1.0)
            if "tbm_gp_weight" in aligned_df.columns:
                tw = aligned_df["tbm_gp_weight"].fillna(1.0).to_numpy(dtype=np.float64)
                sample_weight = sample_weight * np.clip(tw, 0.25, 3.0)
            if is_end_date:
                times = aligned_df.index.get_level_values("datetime")
                if times.tz is None:
                    times = times.tz_localize("UTC")
                else:
                    times = times.tz_convert("UTC")
                cutoff_dt = pd.to_datetime(is_end_date, utc=True)
                is_mask = np.asarray(times < cutoff_dt)
                sample_weight = sample_weight * is_mask

            n_time = len(unique_times)
            train_t_limit = max(1, int(n_time * 0.8))
            row_t: np.ndarray = np.arange(len(aligned_df), dtype=np.int64) // n_symbols
            fit_time_ok = row_t < train_t_limit
            nan_row = np.isnan(x_grid).any(axis=1) | np.isnan(y_raw)
            w_fit = fit_time_ok.astype(np.float64)
            w_nan = (~nan_row).astype(np.float64)
            sample_weight = sample_weight * w_fit * w_nan

            x_clean = np.where(np.isfinite(x_grid), x_grid, 0.0)
            y_clean = np.where(np.isfinite(y_raw), y_raw, 0.0)

            mask = sample_weight > 0
            # --- MI Audit & MP Denoising ---
            if mask.sum() > 100:
                from sklearn.feature_selection import mutual_info_regression

                x_fit = x_clean[mask]
                y_fit = y_clean[mask]

                # 1. MI Audit
                n_samples_fit = x_fit.shape[0]
                sample_size = min(10000, n_samples_fit)
                idx = np.random.choice(n_samples_fit, sample_size, replace=False)
                mi_scores = mutual_info_regression(x_fit[idx], y_fit[idx], random_state=42)
                max_mi = float(np.max(mi_scores)) if len(mi_scores) > 0 else 0.0

                if max_mi < 0.001:
                    _logger.warning(
                        " [STAGNATION DETECTED] Max MI is %.5f. Plateau reached.", max_mi
                    )

                kept_indices = [i for i, score in enumerate(mi_scores) if score > 1e-4]
                if len(kept_indices) < max(5, len(feat_cols) // 4):
                    kept_indices = cast(
                        list[int], np.argsort(mi_scores)[-max(5, len(feat_cols) // 4) :].tolist()
                    )

                _logger.info(
                    " [MI AUDIT] Kept %d/%d features based on MI. Max MI=%.5f",
                    len(kept_indices),
                    len(feat_cols),
                    max_mi,
                )

                # Subset features
                x_clean = x_clean[:, kept_indices]
                x_fit = x_fit[:, kept_indices]
                self._kept_indices = kept_indices  # Store for transform

                # 2. Marchenko-Pastur Denoising
                n_s, n_f = x_fit.shape
                if n_s > n_f and n_f > 1:
                    x_mean = np.mean(x_fit, axis=0)
                    x_std = np.std(x_fit, axis=0)
                    x_std[x_std == 0] = 1.0
                    x_norm = (x_fit - x_mean) / x_std

                    corr = np.dot(x_norm.T, x_norm) / n_s
                    evals, evecs = np.linalg.eigh(corr)

                    q = n_s / float(n_f)
                    e_max = (1 + np.sqrt(1 / q)) ** 2

                    n_facts = int(np.sum(evals > e_max))
                    if 0 < n_facts < n_f:
                        _logger.info(
                            " [MP DENOISE] Keeping %d PCs out of %d",
                            n_facts,
                            n_f,
                        )
                        evecs_kept = evecs[:, -n_facts:]
                        x_clean_norm = (x_clean - x_mean) / x_std
                        x_clean_denoised = np.dot(np.dot(x_clean_norm, evecs_kept), evecs_kept.T)
                        x_clean = x_clean_denoised * x_std + x_mean
                        self._mp_mean = x_mean
                        self._mp_std = x_std
                        self._mp_evecs = evecs_kept
            else:
                self._kept_indices = list(range(len(feat_cols)))
                self._mp_mean = None

            # --- LightGBM Model ---
            _logger.info("Training LightGBM Regressor for alpha generation...")
            lgbm = LGBMRegressor(
                objective="regression",
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=6,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=self.n_jobs,
                random_state=42,
            )

            mask = sample_weight > 0
            if mask.sum() > 100:
                lgbm.fit(x_clean[mask], y_clean[mask], sample_weight=sample_weight[mask])
            else:
                _logger.warning("Not enough samples to train LightGBM!")
                lgbm.fit(x_clean[:100], y_clean[:100])

            self._st = lgbm

            out_pred = lgbm.predict(x_clean)

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
