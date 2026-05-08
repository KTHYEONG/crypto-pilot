"""ML Alpha Miner v5 - LambdaRank + Theme Subspacing Edition.

Replaces regression with 'Learning to Rank' for cross-sectional alpha mining.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from config.opt_config import OPT_FUTURES_CONFIG
from src.core.utils.cache_manager import CacheManager
from src.domain.futures.ml_pipeline.alpha.component_filter import filter_alpha_components
from src.domain.futures.ml_pipeline.features.engineering import (
    GP_FEATURE_SCHEMA_VERSION,
    HMM_SEMANTIC_PROB_COLUMNS,
    add_macro_interaction_features,
)

_logger = logging.getLogger(__name__)

# v5 Theme Subspacing Definitions
THEME_GROUPS = {
    0: [  # Group 1: Trend/Momentum (Slots 00-04) + Vol-Adjusted
        "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", 
        "ma_dist_24", "ma_dist_168", "btc_trend_vol_adj_24h",
        "realized_vol_yz_24", "orderflow_price_divergence"
    ],
    1: [  # Group 2: Volatility/Mean-Reversion (Slots 05-09)
        "vol_ratio_24", "vol_ratio_168", "realized_vol_regime", 
        "downside_vol_ratio", "cs_dispersion"
    ],
    2: [  # Group 3: Structural/Regime (Slots 10-14)
        "btc_beta_x_bull_trend",
        "realized_vol_x_crisis",
        "funding_x_bear_trend",
        "volume_momentum_24h",
        "funding_level",
    ]
}

HMM_COLS = list(HMM_SEMANTIC_PROB_COLUMNS)

@dataclass
class MLAlphaMiner:
    """Miner for evolving cross-sectional alpha components using LightGBM LambdaRank."""

    n_features_to_select: int = 15
    target_horizons: tuple[int, ...] = (3, 6, 12, 24)
    n_jobs: int = 4

    # Internal state
    _models: dict[int, LGBMRanker] = field(default_factory=dict, init=False)
    _feature_sets: dict[int, list[str]] = field(default_factory=dict, init=False)

    def _prepare_labels(
        self, 
        target: pd.Series, 
        raw_returns: pd.Series | None = None,
        dispersion: pd.Series | None = None,
        friction_bps: float = 3.5
    ) -> np.ndarray:
        """Convert continuous rank targets [-1, 1] into discrete LambdaRank labels {0, 1, 2, 3}.
        
        Institutional Friction-Aware Logic:
        - Incorporates friction hurdle to prevent learning noise from small movements.
        - Labels are assigned based on both relative rank (target) and absolute magnitude.
        """
        labels = np.zeros(len(target), dtype=np.int32)
        
        # Calculate friction threshold in decimal (e.g. 3.5 bps = 0.00035)
        friction = friction_bps / 10000.0
        
        # If raw_returns not provided, fallback to simple ranking (v5 legacy)
        if raw_returns is None:
            labels[target > 0.85] = 3
            labels[(target > 0.7) & (target <= 0.85)] = 2
            labels[(target > 0.4) & (target <= 0.7)] = 1
        else:
            # Plan B: Nonlinear Magnitude Amplification + Friction Hurdle
            # Assuming target is rank-scaled [-1, 1] -> [0, 1] by ((target/2)+0.5)
            # Actually mine_alphas_cs passes work_df["target"] which is [-1, 1]
            t = (target / 2.0) + 0.5  # Scale to [0, 1]
            
            abs_ret = raw_returns.abs()
            
            # Label 3: Super Winners (Top 25% of ranks AND > 3x friction)
            labels[(t > 0.75) & (abs_ret > 3.0 * friction)] = 3
            # Label 2: Winners (Top 50% of ranks AND > 2x friction)
            labels[(labels == 0) & (t > 0.50) & (abs_ret > 2.0 * friction)] = 2
            # Label 1: Mild (Top 75% of ranks AND > 1.5x friction)
            labels[(labels == 0) & (t > 0.25) & (abs_ret > 1.5 * friction)] = 1
        
        # C: Dynamic Target Thresholding
        # Suppress labels in quiet regimes (low dispersion) to avoid learning noise.
        if dispersion is not None:
            # Use 20th percentile as quiet threshold if not fixed
            threshold = dispersion.quantile(0.20)
            disp_mask = dispersion < threshold
            labels[disp_mask.to_numpy()] = 0
            
        return labels

    def mine_alphas_cs(
        self,
        panel_df: pd.DataFrame,
        cache_path: Path | None = None,
        is_end_date: str | None = None,
        filter_options: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Train 15 deterministic LambdaRank models using thematic subspacing."""
        if panel_df.empty:
            return pd.DataFrame()

        force_retrain = bool(OPT_FUTURES_CONFIG.get("FUTURES_ML_FORCE_RETRAIN_ALPHA", False))
        versioned_cache: Path | None = None
        raw_alpha_df: pd.DataFrame | None = None

        if cache_path is not None:
            cm = CacheManager(cache_path.parent, max_files=10)
            symbols = sorted(list(panel_df.index.get_level_values("symbol").unique()))
            deps = {
                "horizons": self.target_horizons,
                "symbols": symbols,
                "is_end_date": is_end_date,
                "feature_schema_version": GP_FEATURE_SCHEMA_VERSION,
                "miner_version": "v6_lambdarank_dart"
            }
            tag = cm.generate_hash(deps, source_files=[Path(__file__).resolve()])
            versioned_cache = cm.get_cache_path("raw_lgbm_v6", ".parquet", tag)

            if not force_retrain and versioned_cache.exists():
                try:
                    raw_alpha_df = pd.read_parquet(versioned_cache)
                    _logger.info("MLAlphaMiner v6 Cache Hit: %s", versioned_cache.name)
                except Exception as e:
                    _logger.warning("Cache load failed: %s", e)

        if raw_alpha_df is None:
            _logger.info("Training MLAlphaMiner v6 (DART + Dynamic Labeling)...")
            
            # Sort by datetime for group calculation
            work_df = panel_df.sort_index(level="datetime")
            
            # [Localization] Add macro interaction features (Theme Group 3)
            work_df = add_macro_interaction_features(work_df)
            
            # [Friction-Aware] Calculate 6h forward returns for labeling hurdle
            # Use unstacked approach to ensure no cross-symbol leakage
            close_wide = work_df["close"].unstack(level="symbol")
            fwd_ret_6 = np.log(close_wide.shift(-6) / close_wide).stack(future_stack=True).reindex(work_df.index).fillna(0.0)
            
            # C: Get dispersion for dynamic thresholding
            dispersion = work_df["cs_dispersion"] if "cs_dispersion" in work_df.columns else None
            y_labels = self._prepare_labels(
                work_df["target"], 
                raw_returns=fwd_ret_6,
                dispersion=dispersion
            )
            
            # In-Sample Masking
            if is_end_date:
                cutoff = pd.to_datetime(is_end_date, utc=True)
                is_mask = work_df.index.get_level_values("datetime") < cutoff
            else:
                is_mask = np.ones(len(work_df), dtype=bool)

            # [Plan B-1] Sample Weighting (Magnitude-Aware)
            # Higher weight on bars with large cross-sectional dispersion or high absolute returns.
            # Here we use absolute target value as a proxy for profit opportunity.
            raw_weights = work_df["target"].abs().to_numpy()
            # Normalize weights to mean 1.0 to maintain learning rate stability
            sample_weights_all = raw_weights / (raw_weights.mean() + 1e-12)

            # Features are combined from theme + HMM probabilities
            slots_df = pd.DataFrame(index=work_df.index)
            
            for slot_idx in range(self.n_features_to_select):
                theme_idx = slot_idx // 5
                base_feats = THEME_GROUPS[theme_idx]
                
                # Inject HMM probabilities into models as per spec
                # For Group 3, we use interactions only to ensure cross-sectional variance
                if theme_idx == 2:
                    feat_cols = list(dict.fromkeys(base_feats))
                else:
                    feat_cols = list(dict.fromkeys(base_feats + HMM_COLS))
                
                feat_cols = [c for c in feat_cols if c in work_df.columns]
                
                self._feature_sets[slot_idx] = feat_cols
                
                X = work_df[feat_cols].values
                y = y_labels

                # Filter to In-Sample for training
                X_is = X[is_mask]
                y_is = y[is_mask]
                w_is = sample_weights_all[is_mask]
                
                # For LambdaRank, we need to recalculate group sizes for IS data
                is_df = work_df[is_mask]
                is_group_sizes = is_df.groupby("datetime").size().to_numpy()

                # B-1: DART Booster for better generalization
                model = LGBMRanker(
                    boosting_type="dart",
                    objective="lambdarank",
                    metric="ndcg",
                    eval_at=[1, 3],
                    n_estimators=100,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=-1,
                    random_state=42 + slot_idx,
                    deterministic=True,
                    n_jobs=self.n_jobs,
                    importance_type="gain",
                    verbosity=-1,
                    drop_rate=0.1,
                    skip_drop=0.5
                )
                
                model.fit(
                    X_is, y_is,
                    group=is_group_sizes,
                    sample_weight=w_is
                )
                
                self._models[slot_idx] = model
                
                # Generate full-period predictions (In-Sample + Out-of-Sample)
                raw_scores = model.predict(X)
                
                # Rank-transform raw scores cross-sectionally to [0, 1]
                scores_ser = pd.Series(raw_scores, index=work_df.index)
                unstacked = scores_ser.unstack(level="symbol")
                ranked = unstacked.rank(axis=1, pct=True)
                slots_df[f"ml_alpha_{slot_idx:02d}"] = ranked.stack(future_stack=True).reindex(work_df.index).fillna(0.5)

            raw_alpha_df = slots_df
            
            if versioned_cache:
                versioned_cache.parent.mkdir(parents=True, exist_ok=True)
                raw_alpha_df.to_parquet(versioned_cache)

        # Apply filtering (FDR, DSR, Regime Gate, etc.)
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

        # B-2: IC-Weighted Ensemble (instead of simple mean)
        surviving = [c for c in alpha_df_all.columns if c.startswith("ml_alpha_") and alpha_df_all[c].std() > 1e-6]
        if surviving:
            weights = []
            is_sub = panel_df.loc[alpha_df_all.index]
            if is_end_date:
                cut = pd.to_datetime(is_end_date, utc=True)
                is_mask_ens = is_sub.index.get_level_values("datetime") < cut
                is_sub = is_sub[is_mask_ens]
                alpha_is = alpha_df_all[is_mask_ens]
            else:
                alpha_is = alpha_df_all

            for c in surviving:
                ic = is_sub["target"].corr(alpha_is[c], method="spearman")
                weights.append(max(0.0, ic) ** 2)

            w_arr = np.array(weights)
            if w_arr.sum() > 1e-9:
                w_norm = w_arr / w_arr.sum()
                _logger.info("Ensembling %d components with IC weights: %s", len(surviving), w_norm)
                alpha_df_all["ml_alpha_long"] = (alpha_df_all[surviving] * w_norm).sum(axis=1)
            else:
                alpha_df_all["ml_alpha_long"] = alpha_df_all[surviving].mean(axis=1)

            alpha_df_all["ml_alpha_short"] = 1.0 - alpha_df_all["ml_alpha_long"]
        else:
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
        
        # [Localization] Add macro interaction features (Theme Group 3)
        work_df = add_macro_interaction_features(panel_df)
        
        for slot_idx, model in self._models.items():
            feat_cols = self._feature_sets.get(slot_idx, [])
            if not feat_cols:
                out_df[f"ml_alpha_{slot_idx:02d}"] = 0.5
                continue
                
            # Use work_df which contains interaction features
            X = work_df[feat_cols].fillna(0.0).values
            raw_scores = model.predict(X)
            
            # Cross-sectional rank normalization
            scores_ser = pd.Series(raw_scores, index=panel_df.index)
            unstacked = scores_ser.unstack(level="symbol")
            ranked = unstacked.rank(axis=1, pct=True)
            t_vals = ranked.stack(future_stack=True)
            out_df[f"ml_alpha_{slot_idx:02d}"] = t_vals.reindex(panel_df.index).fillna(0.5)

        surviving = [c for c in out_df.columns if c.startswith("ml_alpha_") and out_df[c].std() > 1e-6]
        if surviving:
            out_df["ml_alpha_long"] = out_df[surviving].mean(axis=1)
            out_df["ml_alpha_short"] = 1.0 - out_df["ml_alpha_long"]
        else:
            out_df["ml_alpha_long"] = 0.5
            out_df["ml_alpha_short"] = 0.5
            
        return out_df
