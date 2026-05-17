"""ML Alpha Miner v5 - LambdaRank + Theme Subspacing Edition.

Replaces regression with 'Learning to Rank' for cross-sectional alpha mining.
"""

from __future__ import annotations

import gc
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import numba
import time
from catboost import CatBoostRanker, CatBoostRegressor, Pool

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

# Verify GPU availability once at module load via a short-lived subprocess (max 3.0s)
# to handle WSL2 CUDA hangs or driver mismatch situations safely.
_GPU_AVAILABLE: bool = False
try:
    import subprocess
    import sys
    _check_code = """
import sys
try:
    from catboost.utils import get_gpu_device_count
    count = get_gpu_device_count()
    if count > 0:
        sys.exit(0)
    else:
        sys.exit(1)
except Exception:
    sys.exit(2)
"""
    _res = subprocess.run(
        [sys.executable, "-c", _check_code],
        capture_output=True,
        timeout=3.0
    )
    _GPU_AVAILABLE = (_res.returncode == 0)
except Exception as _e:
    _logger.warning("Failed to check GPU availability: %s. Defaulting to CPU.", _e)
    _GPU_AVAILABLE = False

_logger.info("CatBoost GPU availability verified: %s", _GPU_AVAILABLE)
_DEFAULT_TASK_TYPE = "GPU" if _GPU_AVAILABLE else "CPU"


@numba.njit(parallel=True, cache=True)
def _fast_rank_2d_numba(array: np.ndarray) -> np.ndarray:
    """Vectorized percentile ranking across rows (axis=1) with tie-breaking 'average'.
    
    Matches pd.DataFrame(array).rank(axis=1, pct=True, method='average') but 10-20x faster.
    """
    n, m = array.shape
    out = np.empty((n, m), dtype=np.float64)
    for i in numba.prange(n):
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

SHORT_THEME_GROUPS = {
    0: [  # Group 1: Funding/LSR/Crowding
        "funding_rate", "funding_chg_8", "funding_z_72", "funding_mom_24",
        "funding_intensity_24h", "top_trader_lsr_z_24h", "global_lsr_z_24h",
        "oi_funding_trap_24h", "motif_crowded_long_unwind",
    ],
    1: [  # Group 2: Liquidation/Orderflow
        "liq_proxy_6", "liq_intensity_proxy", "taker_imbalance_z_24",
        "cvd_divergence_24h", "price_impact_asymmetry", "orderflow_price_divergence",
        "taker_absorption_score",
    ],
    2: [  # Group 3: Downside structure
        "downside_jump_24", "tail_rejection_24", "exhaustion_cascade_score",
        "capitulation_proxy", "tail_risk_24", "ret_vol_adj_24",
    ],
}

_LONG_SLOT_COL_RE = re.compile(r"^alpha_long_(\d{2})$")
_SHORT_SLOT_COL_RE = re.compile(r"^alpha_short_(\d{2})$")


def _train_ranker_slot(
    slot_idx: int,
    slots_per_theme: int,
    X_pool: Optional[Pool],
    train_pool: Optional[Pool],
    feat_cols: list[str],
    seed_offset: int = 0,
) -> Tuple[int, Optional[CatBoostRanker], list[str], np.ndarray]:
    """Helper for Ranker training using CatBoost."""
    if not feat_cols or train_pool is None:
        return slot_idx, None, [], np.array([])

    theme_idx = min(2, slot_idx // slots_per_theme)
    n_est = {0: 80, 1: 70, 2: 60}.get(theme_idx, 100)

    ranker_params = {
        "loss_function": "YetiRank",
        "task_type": _DEFAULT_TASK_TYPE,
        "iterations": n_est,
        "depth": 8,
        "border_count": 254,
        "learning_rate": 0.05,
        "metric_period": 10000,
        "verbose": 0,
        "random_seed": 42 + seed_offset + slot_idx,
        "allow_writing_files": False,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.85,
    }
    if _DEFAULT_TASK_TYPE == "GPU":
        ranker_params["devices"] = "0"
        ranker_params["gpu_ram_part"] = 0.5

    model = CatBoostRanker(**ranker_params)
    try:
        model.fit(train_pool)
    except CatBoostError as exc:
        _logger.warning("CatBoost Ranker slot fit failed: %s. Falling back to CPU.", exc)
        ranker_params["task_type"] = "CPU"
        ranker_params.pop("devices", None)
        ranker_params.pop("gpu_ram_part", None)
        model = CatBoostRanker(**ranker_params)
        model.fit(train_pool)
    
    raw_scores = model.predict(X_pool) if X_pool is not None else np.array([])
    return slot_idx, model, feat_cols, raw_scores


def _train_regressor_slot(
    slot_idx: int,
    slots_per_theme: int,
    X_pool: Optional[Pool],
    train_pool: Optional[Pool],
    feat_cols: list[str],
) -> Tuple[int, Optional[CatBoostRegressor], np.ndarray]:
    """Helper for Regressor training using CatBoost."""
    if not feat_cols or train_pool is None:
        return slot_idx, None, np.array([])

    theme_idx = min(2, slot_idx // slots_per_theme)
    n_est = {0: 80, 1: 70, 2: 60}.get(theme_idx, 100)

    reg_params = {
        "loss_function": "MAE",
        "task_type": _DEFAULT_TASK_TYPE,
        "iterations": n_est,
        "depth": 8,
        "border_count": 254,
        "learning_rate": 0.05,
        "verbose": 0,
        "random_seed": 42 + slot_idx,
        "allow_writing_files": False,
    }
    if _DEFAULT_TASK_TYPE == "GPU":
        reg_params["devices"] = "0"
        reg_params["gpu_ram_part"] = 0.5

    reg = CatBoostRegressor(**reg_params)
    try:
        reg.fit(train_pool)
    except CatBoostError as exc:
        _logger.warning("CatBoost Regressor slot fit failed: %s. Falling back to CPU.", exc)
        reg_params["task_type"] = "CPU"
        reg_params.pop("devices", None)
        reg_params.pop("gpu_ram_part", None)
        reg = CatBoostRegressor(**reg_params)
        reg.fit(train_pool)
    
    mag_raw = reg.predict(X_pool) if X_pool is not None else np.array([])
    return slot_idx, reg, mag_raw


# Removed _train_combined_slot as it's no longer used in the optimized theme-batch pipeline.


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
    _models: dict[int, CatBoostRanker] = field(default_factory=dict, init=False)
    _short_models: dict[int, CatBoostRanker] = field(default_factory=dict, init=False)
    _mag_models: dict[int, CatBoostRegressor] = field(default_factory=dict, init=False)
    _feature_sets: dict[int, list[str]] = field(default_factory=dict, init=False)
    _short_feature_sets: dict[int, list[str]] = field(default_factory=dict, init=False)
    # ntree_end per slot for staged-prediction virtual slots (0 = full ensemble)
    _ntree_ends: dict[int, int] = field(default_factory=dict, init=False)
    _short_ntree_ends: dict[int, int] = field(default_factory=dict, init=False)
    # IS Spearman IC^2 weights per slot (matches mine_alphas_cs ensemble → transform_cs)
    _ic_weights: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _short_ic_weights: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    ic_by_slot: dict[str, float] = field(default_factory=dict, init=False)

    def _prepare_labels(
        self, 
        target: pd.Series, 
        raw_returns: pd.Series | np.ndarray | None = None,
        dispersion: pd.Series | None = None,
        atr_24h_pct: pd.Series | np.ndarray | None = None,
        friction_bps: float = 7.0,
        short_oriented: bool = False,
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
            is_mild_short = (~is_strong_long) & (~is_strong_short) & (t < 0.40) & (ret < -friction)
            
            labels[is_strong_long] = 3
            labels[is_strong_short] = 0
            labels[is_mild_long] = 2
            labels[is_mild_short] = 0

            if short_oriented:
                labels = 3 - labels
        
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
                "miner_version": "v8_staged_pred_cpu_opt",
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
            _logger.info("Training MLAlphaMiner v6 (CatBoost GPU + Dynamic Labeling)...")
            
            # Sort by datetime and symbol for alignment and group calculation
            work_df = panel_df.sort_index(level=["datetime", "symbol"])
            
            # [Localization] Add macro interaction features (Theme Group 3)
            work_df = add_macro_interaction_features(work_df)
            
            # [Optimization] Pre-clean feature columns once
            all_feat_cols = set()
            for group_feats in THEME_GROUPS.values():
                all_feat_cols.update(group_feats)
            for group_feats in SHORT_THEME_GROUPS.values():
                all_feat_cols.update(group_feats)
            all_feat_cols.update(HMM_COLS)
            
            existing_feats = [c for c in all_feat_cols if c in work_df.columns]
            if existing_feats:
                # [Optimization ⑨] replace/fillna 체인 → numpy in-place (pandas 중간 DataFrame 생성 제거)
                _feat_arr = work_df[existing_feats].to_numpy(dtype=np.float64, copy=True)
                np.nan_to_num(_feat_arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
                work_df[existing_feats] = _feat_arr

            # [Optimization ⑧] 3회 unstack → 1회 통합 (pandas groupby 오버헤드 2회 절감)
            _ohlc_wide = work_df[["close", "high", "low"]].unstack(level="symbol")
            close_wide = _ohlc_wide["close"]
            high_wide = _ohlc_wide["high"]
            low_wide = _ohlc_wide["low"]
            del _ohlc_wide

            # [Optimization] Efficient wide-flat mapping
            close_clipped = close_wide.clip(lower=1e-12)
            valid_mask = close_wide.notna().values  # Since work_df is sorted by (datetime, symbol), this matches perfectly
            idx_shape = close_wide.shape

            # Efficient vectorized calculation of forward returns and ATR
            fwd_ret_6_wide = np.log(close_wide.shift(-6) / close_clipped)
            fwd_log_mag_horizon_wide = np.log(close_wide.shift(-_MAG_HORIZON_BARS) / close_clipped)
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
            # [Optimization ⑥] short 라벨: _prepare_labels 재호출 대신 flip (3 - labels)
            # short_oriented=True 경로는 마지막에 `labels = 3 - labels`만 수행하므로 동일
            y_labels_short = (3 - y_labels).astype(np.int32)
            
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

            y_mag_vals_raw = y_mag_h.to_numpy(dtype=np.float64)
            # [Optimization ⑤] 재사용 버퍼 사전 할당 + dict-of-arrays 일괄 구성 (컬럼별 할당 제거)
            _m_buf = np.empty(idx_shape, dtype=np.float64)
            slot_arrays: dict[str, np.ndarray] = {}
            self._mag_models.clear()
            self._short_models.clear()
            self._short_feature_sets.clear()
            self._ntree_ends.clear()
            self._short_ntree_ends.clear()

            # GPU-Native Mining Loop v8.0
            # Strategy: 3 themes × 3 types = 9 fit() (vs prior 54+).
            # Virtual slot diversity comes from staged ntree_end checkpoints,
            # not seed clones — early/late tree ensembles capture different
            # bias-variance trade-offs and pass the FDR independence assumption
            # better than seed-only replicas.
            _logger.info("GPU-Native Mining Loop v8.0 (9 fit + staged predictions)...")
            total_loop_start = time.time()
            mag_finite = np.isfinite(y_mag_vals_raw)

            # Compute index arrays once (was recomputed inside the loop)
            train_idx = is_mask.nonzero()[0]
            train_mag_idx = (is_mask & mag_finite).nonzero()[0]
            full_group_ids = work_df.index.get_level_values("datetime").factorize()[0]

            # iterations per theme — larger values let the GPU amortize session overhead
            _LONG_ITERS = {0: 400, 1: 350, 2: 300}
            _SHORT_ITERS = {0: 350, 1: 300, 2: 250}

            for theme_idx in range(3):
                if theme_idx == 1:
                    fc = list(dict.fromkeys(THEME_GROUPS[theme_idx] + HMM_COLS))
                else:
                    fc = list(dict.fromkeys(THEME_GROUPS[theme_idx]))
                fc = [c for c in fc if c in work_df.columns]
                short_fc = list(dict.fromkeys(SHORT_THEME_GROUPS[theme_idx]))
                short_fc = [c for c in short_fc if c in work_df.columns]

                if not fc and not short_fc:
                    continue

                # --- Build pools: quantize ONCE, reuse across all staged predictions ---
                master_pool = train_pool = pred_pool = None
                master_mag_pool = mag_train_pool = None
                master_short_pool = short_train_pool = short_pred_pool = None

                if fc:
                    # [Optimization ⑦] X_long 1회 생성 후 두 Pool에 공유 (이중 복사 제거)
                    X_long = work_df[fc].values.astype(np.float32)
                    # YetiRank: no object weights (pairwise loss ignores them)
                    master_pool = Pool(
                        data=X_long,
                        label=y_labels,
                        group_id=full_group_ids,
                        feature_names=fc,
                    )
                    master_pool.quantize(border_count=254)
                    train_pool = master_pool.slice(train_idx)
                    pred_pool = master_pool

                    master_mag_pool = Pool(  # MAE regressor supports per-sample weights
                        data=X_long,  # 동일 배열 참조 (복사 없음)
                        label=y_mag_vals_raw,
                        weight=sample_weights_all,
                        feature_names=fc,
                    )
                    master_mag_pool.quantize(border_count=254)
                    mag_train_pool = master_mag_pool.slice(train_mag_idx)

                if short_fc:
                    master_short_pool = Pool(
                        data=work_df[short_fc].values.astype(np.float32),
                        label=y_labels_short,
                        group_id=full_group_ids,
                        feature_names=short_fc,
                    )
                    master_short_pool.quantize(border_count=254)
                    short_train_pool = master_short_pool.slice(train_idx)
                    short_pred_pool = master_short_pool

                theme_start = theme_idx * self.slots_per_theme
                theme_end = min(self.n_features_to_select, (theme_idx + 1) * self.slots_per_theme)
                n_virtual = theme_end - theme_start  # number of staged-prediction slots

                # --- Phase 1: ONE GPU fit per direction (amortize session init overhead) ---
                # rsm=0.8: each tree randomly samples 80% of features →
                #          intra-model subspace diversity (replaces seed-clone diversity)
                long_ranker: CatBoostRanker | None = None
                short_ranker: CatBoostRanker | None = None

                long_iters = _LONG_ITERS[theme_idx]
                short_iters = _SHORT_ITERS[theme_idx]

                if train_pool is not None:
                    ranker_params = {
                        "loss_function": "YetiRank",
                        "task_type": _DEFAULT_TASK_TYPE,
                        "iterations": long_iters,
                        "depth": 8,
                        "border_count": 254,
                        "learning_rate": 0.04,
                        "metric_period": 10000,
                        "verbose": 0,
                        "random_seed": 42 + theme_idx * 100,
                        "allow_writing_files": False,
                        "bootstrap_type": "Bernoulli",
                        "subsample": 0.85,
                    }
                    if _DEFAULT_TASK_TYPE == "GPU":
                        ranker_params["devices"] = "0"
                        ranker_params["gpu_ram_part"] = 0.4 # Use less memory for stability in WSL2
                    
                    long_ranker = CatBoostRanker(**ranker_params)
                    try:
                        _logger.info("  [THEME %d] Fitting separate LONG ranker...", theme_idx)
                        long_ranker.fit(train_pool)
                    except CatBoostError as exc:
                        _logger.warning("CatBoost Long Ranker fit failed: %s. Falling back to CPU.", exc)
                        ranker_params["task_type"] = "CPU"
                        ranker_params.pop("devices", None)
                        ranker_params.pop("gpu_ram_part", None)
                        long_ranker = CatBoostRanker(**ranker_params)
                        long_ranker.fit(train_pool)

                if short_train_pool is not None:
                    short_ranker_params = {
                        "loss_function": "YetiRank",
                        "task_type": _DEFAULT_TASK_TYPE,
                        "iterations": short_iters,
                        "depth": 8,
                        "border_count": 254,
                        "learning_rate": 0.04,
                        "metric_period": 10000,
                        "verbose": 0,
                        "random_seed": 42 + theme_idx * 100 + 1000,
                        "allow_writing_files": False,
                        "bootstrap_type": "Bernoulli",
                        "subsample": 0.85,
                    }
                    if _DEFAULT_TASK_TYPE == "GPU":
                        short_ranker_params["devices"] = "0"
                        short_ranker_params["gpu_ram_part"] = 0.4
                    
                    short_ranker = CatBoostRanker(**short_ranker_params)
                    try:
                        _logger.info("  [THEME %d] Fitting separate SHORT ranker...", theme_idx)
                        short_ranker.fit(short_train_pool)
                    except CatBoostError as exc:
                        _logger.warning("CatBoost Short Ranker fit failed: %s. Falling back to CPU.", exc)
                        short_ranker_params["task_type"] = "CPU"
                        short_ranker_params.pop("devices", None)
                        short_ranker_params.pop("gpu_ram_part", None)
                        short_ranker = CatBoostRanker(**short_ranker_params)
                        short_ranker.fit(short_train_pool)

                # Magnitude: one regressor per theme (shared across virtual slots)
                _, mag_reg, _ = _train_regressor_slot(
                    theme_start, self.slots_per_theme, None, mag_train_pool, fc
                )
                mag_z: np.ndarray | None = None
                if mag_reg is not None and pred_pool is not None:
                    mag_raw_all = mag_reg.predict(pred_pool)
                    mu_m = float(np.mean(mag_raw_all))
                    sig_m = float(np.std(mag_raw_all) + 1e-9)
                    mag_z = np.clip((mag_raw_all - mu_m) / sig_m, -3.0, 3.0)

                # --- Phase 2: Staged predictions → virtual slots ---
                # ntree_end checkpoints: e.g. iters=350, slots=5 → [70,140,210,280,350]
                # Each checkpoint = different bias-variance trade-off = genuine diversity
                # Pool is already GPU-resident from fit() → zero re-quantization cost
                step_l = max(1, long_iters // n_virtual)
                step_s = max(1, short_iters // n_virtual)
                long_checkpoints = [min(long_iters, step_l * (k + 1)) for k in range(n_virtual)]
                short_checkpoints = [min(short_iters, step_s * (k + 1)) for k in range(n_virtual)]

                for v_idx in range(n_virtual):
                    s_idx = theme_start + v_idx

                    if long_ranker is not None and pred_pool is not None:
                        ntree_l = long_checkpoints[v_idx]
                        raw_scores = long_ranker.predict(pred_pool, ntree_end=ntree_l)
                        self._models[s_idx] = long_ranker
                        self._feature_sets[s_idx] = fc
                        self._ntree_ends[s_idx] = ntree_l
                        # [Optimization ⑤] 재사용 버퍼: np.full() 매번 할당 제거
                        _m_buf.fill(np.nan)
                        _m_buf[valid_mask] = raw_scores
                        slot_arrays[f"alpha_long_{s_idx:02d}"] = _fast_rank_2d_numba(_m_buf)[valid_mask]
                        if mag_z is not None:
                            self._mag_models[s_idx] = mag_reg  # type: ignore[assignment]
                            slot_arrays[f"mag_long_{s_idx:02d}"] = mag_z
                        else:
                            slot_arrays[f"mag_long_{s_idx:02d}"] = np.zeros(int(valid_mask.sum()), dtype=np.float64)
                    else:
                        slot_arrays[f"alpha_long_{s_idx:02d}"] = np.full(int(valid_mask.sum()), 0.5)
                        slot_arrays[f"mag_long_{s_idx:02d}"] = np.zeros(int(valid_mask.sum()), dtype=np.float64)

                    if short_ranker is not None and short_pred_pool is not None:
                        ntree_s = short_checkpoints[v_idx]
                        short_raw = short_ranker.predict(short_pred_pool, ntree_end=ntree_s)
                        self._short_models[s_idx] = short_ranker
                        self._short_feature_sets[s_idx] = short_fc
                        self._short_ntree_ends[s_idx] = ntree_s
                        # [Optimization ⑤] 동일 버퍼 재사용 (long 처리 후 short)
                        _m_buf.fill(np.nan)
                        _m_buf[valid_mask] = short_raw
                        slot_arrays[f"alpha_short_{s_idx:02d}"] = _fast_rank_2d_numba(_m_buf)[valid_mask]
                    else:
                        slot_arrays[f"alpha_short_{s_idx:02d}"] = np.full(int(valid_mask.sum()), 0.5)

                # Free VRAM/RAM before next theme
                del pred_pool, short_pred_pool, train_pool, mag_train_pool, short_train_pool
                del master_pool, master_mag_pool, master_short_pool
                gc.collect()

            loop_elapsed = time.time() - total_loop_start
            _logger.info("GPU-Native Mining Loop completed in %.4f seconds.", loop_elapsed)

            # [Optimization ⑤] slot_arrays → DataFrame 일괄 구성 (컬럼별 시리즈 할당 제거)
            # valid_mask.sum() == len(work_df) (close가 항상 유효한 경우)이므로 직접 구성 가능
            raw_alpha_df = pd.DataFrame(slot_arrays, index=work_df.index)
            
            if versioned_cache:
                versioned_cache.parent.mkdir(parents=True, exist_ok=True)
                raw_alpha_df.to_parquet(versioned_cache)

        # Apply filtering (long heads only). Short heads are preserved and
        # aggregated separately because they are trained with short-oriented labels.
        long_slot_cols = [c for c in raw_alpha_df.columns if _LONG_SLOT_COL_RE.match(c)]
        short_slot_cols = [c for c in raw_alpha_df.columns if _SHORT_SLOT_COL_RE.match(c)]
        filter_opts = filter_options or {}
        alpha_df_all, filt_meta = filter_alpha_components(
            raw_alpha_df.copy(),
            panel_df,
            is_end_date=is_end_date,
            n_trials=max(1, len(long_slot_cols)),
            fdr_q=float(filter_opts.get("fdr_q", 0.10)),
            alpha_cols=long_slot_cols,
            symbol_balance_max=float(filter_opts.get("symbol_balance_max", 3.0)),
            use_newey_west=bool(filter_opts.get("use_newey_west", False)),
            use_ewma_ic_stat=bool(filter_opts.get("use_ewma_ic_stat", False)),
            ewma_half_life=float(filter_opts.get("ewma_half_life", 540.0)),
            require_regime_gate=bool(filter_opts.get("require_regime_gate", True)),
            step3_regime_alpha_enabled=bool(filter_opts.get("step3_regime_alpha_enabled", False)),
            step3_chop_support_min=float(filter_opts.get("step3_chop_support_min", 0.25)),
            step3_chop_ic_min=float(filter_opts.get("step3_chop_ic_min", -0.01)),
            step3_chop_weight_mult=float(filter_opts.get("step3_chop_weight_mult", 0.50)),
            step3_weight_mult_floor=float(filter_opts.get("step3_weight_mult_floor", 0.20)),
        )
        for c in short_slot_cols:
            alpha_df_all[c] = raw_alpha_df[c]

        surviving = [c for c in alpha_df_all.columns if _LONG_SLOT_COL_RE.match(c) and alpha_df_all[c].std() > 1e-6]
        surviving_short = [c for c in alpha_df_all.columns if _SHORT_SLOT_COL_RE.match(c) and alpha_df_all[c].std() > 1e-6]
        if surviving:
            ic_map = filt_meta.get("ic_weight_by_slot", {}) or filt_meta.get("ic_by_slot", {})
            weights = [max(0.0, float(ic_map.get(c, 0.0))) ** 2 for c in surviving]
            w_arr = np.array(weights)
            if w_arr.sum() > 1e-9:
                w_norm = w_arr / w_arr.sum()
                long_rank = (alpha_df_all[surviving] * w_norm).sum(axis=1)
                self._ic_weights = {int(_LONG_SLOT_COL_RE.match(c).group(1)): float(w) for c, w in zip(surviving, w_norm)}
            else:
                long_rank = alpha_df_all[surviving].mean(axis=1)
                self._ic_weights = {int(_LONG_SLOT_COL_RE.match(c).group(1)): 1.0 / len(surviving) for c in surviving}

            mag_surv_cols = [f"mag_long_{c.split('_')[2]}" for c in surviving if f"mag_long_{c.split('_')[2]}" in alpha_df_all.columns]
            if mag_surv_cols:
                mag_blend = alpha_df_all[mag_surv_cols].mean(axis=1)
                alpha_df_all["alpha_long"] = np.clip(long_rank * (1.0 + 0.3 * mag_blend), 0.02, 0.98)
            else:
                alpha_df_all["alpha_long"] = long_rank
        else:
            self._ic_weights = {}
            alpha_df_all["alpha_long"] = 0.5

        if surviving_short:
            ic_map_short = filt_meta.get("ic_weight_by_slot", {}) or filt_meta.get("ic_by_slot", {})
            short_weights = [max(0.0, float(ic_map_short.get(c, 0.0))) ** 2 for c in surviving_short]
            sw_arr = np.array(short_weights)
            if sw_arr.sum() > 1e-9:
                sw_norm = sw_arr / sw_arr.sum()
                short_rank = (alpha_df_all[surviving_short] * sw_norm).sum(axis=1)
                self._short_ic_weights = {int(_SHORT_SLOT_COL_RE.match(c).group(1)): float(w) for c, w in zip(surviving_short, sw_norm)}
            else:
                short_rank = alpha_df_all[surviving_short].mean(axis=1)
                self._short_ic_weights = {int(_SHORT_SLOT_COL_RE.match(c).group(1)): 1.0 / len(surviving_short) for c in surviving_short}
            alpha_df_all["alpha_short"] = short_rank
        elif "alpha_long" in alpha_df_all.columns:
            self._short_ic_weights = {}
            alpha_df_all["alpha_short"] = 1.0 - alpha_df_all["alpha_long"]
        else:
            self._short_ic_weights = {}
            alpha_df_all["alpha_short"] = 0.5

        out_df = alpha_df_all.reindex(panel_df.index).fillna(0.5)
        out_df.attrs["alpha_component_filter"] = filt_meta
        # Minimal Step3 audit rollups for telemetry/logging without refactor.
        ic_chop = filt_meta.get("ic_chop_by_slot", {}) if isinstance(filt_meta, dict) else {}
        ic_bear = filt_meta.get("ic_bear_by_slot", {}) if isinstance(filt_meta, dict) else {}
        chop_support = filt_meta.get("chop_support_by_slot", {}) if isinstance(filt_meta, dict) else {}
        tail_ic = filt_meta.get("tail_ic_by_slot", {}) if isinstance(filt_meta, dict) else {}
        if isinstance(ic_chop, dict) and ic_chop:
            out_df.attrs["step3_ic_chop_mean"] = float(np.mean(list(ic_chop.values())))
        if isinstance(ic_bear, dict) and ic_bear:
            out_df.attrs["step3_ic_bear_mean"] = float(np.mean(list(ic_bear.values())))
        if isinstance(chop_support, dict) and chop_support:
            out_df.attrs["step3_chop_support_mean"] = float(np.mean(list(chop_support.values())))
        if isinstance(tail_ic, dict) and tail_ic:
            out_df.attrs["step3_tail_ic_mean"] = float(np.mean(list(tail_ic.values())))
        if surviving:
            ic_by_slot = filt_meta.get("ic_by_slot", {})
            ic_vals = [float(ic_by_slot.get(c, 0.0)) for c in surviving if c in ic_by_slot]
            survival_rate = len(surviving) / max(len(alpha_df_all.columns), 1)
            out_df.attrs["best_fitness"] = float(np.mean(ic_vals)) if ic_vals else survival_rate
        else:
            out_df.attrs["best_fitness"] = 0.0

        if "target" in panel_df.columns:
            out_df["target"] = panel_df["target"]

        return out_df

    def _get_lgbm_params(self, seed_offset: int = 0, iterations: int = 250) -> dict:
        """CatBoostRanker 기본 파라미터. GPU 포화 및 CPU fallback 제거 최적화.

        Args:
            seed_offset: random_seed에 더할 오프셋 (슬롯 다양성 확보).
            iterations: 학습 iteration 수 (모델 수 감소분으로 amortize).

        Returns:
            CatBoostRanker 생성자에 전달할 파라미터 dict.
        """
        return dict(
            loss_function="YetiRank",
            task_type="GPU",
            devices="0",
            iterations=iterations,
            depth=8,             # 6→8: GPU는 깊은 트리에서 병렬성 확보
            border_count=254,    # 128→254: GPU histogram 작업 증가
            learning_rate=0.04,
            metric_period=10_000,  # NDCG 평가 비활성화 (NDCG:Base는 GPU 미지원)
            verbose=0,
            random_seed=42 + seed_offset,
            allow_writing_files=False,
            bootstrap_type="Bernoulli",  # Bayesian보다 GPU 동기화 overhead 적음
            subsample=0.85,
        )

    def transform_cs(self, panel_df: pd.DataFrame, cache_path: Path | None = None) -> pd.DataFrame:
        """Apply trained v5 models to new panel data."""
        if panel_df.empty or not self._models:
            _logger.warning("transform_cs: No models trained or empty panel.")
            cols = [f"alpha_long_{i:02d}" for i in range(self.n_features_to_select)]
            return pd.DataFrame(0.5, index=panel_df.index, columns=cols)

        out_df = pd.DataFrame(index=panel_df.index)
        work_df = add_macro_interaction_features(panel_df).sort_index(level=["datetime", "symbol"])
        
        all_feat_cols = set()
        for group_feats in THEME_GROUPS.values():
            all_feat_cols.update(group_feats)
        for group_feats in SHORT_THEME_GROUPS.values():
            all_feat_cols.update(group_feats)
        all_feat_cols.update(HMM_COLS)
        existing_feats = [c for c in all_feat_cols if c in work_df.columns]
        if existing_feats:
            work_df[existing_feats] = work_df[existing_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # [Optimization] Efficient wide-flat mapping
        close_wide = work_df["close"].unstack(level="symbol")
        valid_mask = close_wide.notna().values
        idx_shape = close_wide.shape

        # [Optimization] Efficient Theme-Based Transformation with Pool reuse
        for theme_idx in range(3):
            # Resolve Long Features
            if theme_idx == 1:
                fc = list(dict.fromkeys(THEME_GROUPS[theme_idx] + HMM_COLS))
            else:
                fc = list(dict.fromkeys(THEME_GROUPS[theme_idx]))
            fc = [c for c in fc if c in work_df.columns]
            
            # Resolve Short Features
            short_fc = list(dict.fromkeys(SHORT_THEME_GROUPS[theme_idx]))
            short_fc = [c for c in short_fc if c in work_df.columns]

            # Build Prediction Pools for this theme
            pred_pool = Pool(data=work_df[fc].values.astype(np.float32)) if fc else None
            short_pred_pool = Pool(data=work_df[short_fc].values.astype(np.float32)) if short_fc else None

            theme_start = theme_idx * self.slots_per_theme
            theme_end = min(self.n_features_to_select, (theme_idx + 1) * self.slots_per_theme)

            # mag prediction computed once per theme (all slots share same mag model)
            _theme_mag_z: dict[int, np.ndarray] = {}
            for slot_idx in range(theme_start, theme_end):
                # Long Predictions — use stored ntree_end for staged-prediction consistency
                model = self._models.get(slot_idx)
                ntree_l = self._ntree_ends.get(slot_idx, 0)
                if model is not None and pred_pool is not None:
                    raw_scores = model.predict(pred_pool, ntree_end=ntree_l)
                    scores_matrix = np.full(idx_shape, np.nan)
                    scores_matrix[valid_mask] = raw_scores
                    out_df[f"alpha_long_{slot_idx:02d}"] = _fast_rank_2d_numba(scores_matrix)[valid_mask]

                    mag_model = self._mag_models.get(slot_idx)
                    if mag_model is not None:
                        # Cache mag prediction at theme level (same model for all slots)
                        t_key = theme_idx
                        if t_key not in _theme_mag_z:
                            mag_raw = mag_model.predict(pred_pool)
                            mu_m = float(np.mean(mag_raw))
                            sig_m = float(np.std(mag_raw) + 1e-9)
                            _theme_mag_z[t_key] = np.clip((mag_raw - mu_m) / sig_m, -3.0, 3.0)
                        out_df[f"mag_long_{slot_idx:02d}"] = _theme_mag_z[t_key]
                    else:
                        out_df[f"mag_long_{slot_idx:02d}"] = 0.0
                else:
                    out_df[f"alpha_long_{slot_idx:02d}"] = 0.5
                    out_df[f"mag_long_{slot_idx:02d}"] = 0.0

                # Short Predictions — use stored ntree_end
                short_model = self._short_models.get(slot_idx)
                ntree_s = self._short_ntree_ends.get(slot_idx, 0)
                if short_model is not None and short_pred_pool is not None:
                    short_raw_scores = short_model.predict(short_pred_pool, ntree_end=ntree_s)
                    short_scores_matrix = np.full(idx_shape, np.nan)
                    short_scores_matrix[valid_mask] = short_raw_scores
                    out_df[f"alpha_short_{slot_idx:02d}"] = _fast_rank_2d_numba(short_scores_matrix)[valid_mask]
                else:
                    out_df[f"alpha_short_{slot_idx:02d}"] = 0.5

            del pred_pool, short_pred_pool
            gc.collect()

        surviving = [c for c in out_df.columns if _LONG_SLOT_COL_RE.match(c) and out_df[c].std() > 1e-6]
        surviving_short = [c for c in out_df.columns if _SHORT_SLOT_COL_RE.match(c) and out_df[c].std() > 1e-6]
        if surviving:
            if self._ic_weights:
                w = np.array([self._ic_weights.get(int(_LONG_SLOT_COL_RE.match(c).group(1)), 0.0) for c in surviving])
                s = float(w.sum())
                if s > 1e-9:
                    long_rank = (out_df[surviving] * (w / s)).sum(axis=1)
                else:
                    long_rank = out_df[surviving].mean(axis=1)
            else:
                long_rank = out_df[surviving].mean(axis=1)

            mag_surv_cols = [f"mag_long_{c.split('_')[2]}" for c in surviving if f"mag_long_{c.split('_')[2]}" in out_df.columns]
            if mag_surv_cols:
                mag_blend = out_df[mag_surv_cols].mean(axis=1)
                out_df["alpha_long"] = np.clip(long_rank * (1.0 + 0.3 * mag_blend), 0.02, 0.98)
            else:
                out_df["alpha_long"] = long_rank
        else:
            out_df["alpha_long"] = 0.5

        if surviving_short:
            if self._short_ic_weights:
                w_short = np.array([self._short_ic_weights.get(int(_SHORT_SLOT_COL_RE.match(c).group(1)), 0.0) for c in surviving_short])
                s_short = float(w_short.sum())
                if s_short > 1e-9:
                    short_rank = (out_df[surviving_short] * (w_short / s_short)).sum(axis=1)
                else:
                    short_rank = out_df[surviving_short].mean(axis=1)
            else:
                short_rank = out_df[surviving_short].mean(axis=1)
            out_df["alpha_short"] = short_rank
        else:
            out_df["alpha_short"] = 1.0 - out_df["alpha_long"]

        if "target" in panel_df.columns:
            out_df["target"] = panel_df["target"]

        return out_df.reindex(panel_df.index).fillna(0.5)
