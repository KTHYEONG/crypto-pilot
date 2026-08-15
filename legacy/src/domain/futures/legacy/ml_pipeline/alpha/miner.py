"""ML Alpha Miner v5 - LambdaRank + Theme Subspacing Edition.

Replaces regression with 'Learning to Rank' for cross-sectional alpha mining.
"""

from __future__ import annotations

import gc
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numba
import numpy as np
import pandas as pd
from catboost import CatBoostError, CatBoostRanker, CatBoostRegressor, Pool

from src.domain.futures.ml_pipeline.alpha.component_filter import filter_alpha_components
from src.domain.futures.ml_pipeline.features.engineering import (
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
    _logger.debug("Failed to check GPU availability: %s. Defaulting to CPU.", _e)
    _GPU_AVAILABLE = False

_logger.debug("CatBoost GPU availability verified: %s", _GPU_AVAILABLE)
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


def _compute_triple_barrier_labels_wide(
    close_wide: pd.DataFrame,
    atr_wide: pd.DataFrame,
    horizon: int = 6,
    pt_multiplier: float = 1.5,
    market_neutral: bool = True,
) -> np.ndarray:
    """Triple-barrier per-symbol 레이블 (wide format 완전 벡터화).

    horizon-length 루프만 사용 (per-bar Python 루프 없음, Zero-Loop Policy 준수).
    market_neutral=True: 각 수평선에서 cross-sectional 평균 return 차감 (B2 통합).

    Args:
        close_wide: wide format close DataFrame (rows=datetime, cols=symbols).
        atr_wide: wide format ATR DataFrame, price units (rows=datetime, cols=symbols).
        horizon: 최대 대기 bar 수.
        pt_multiplier: ATR 배수 (TP/SL barrier 크기).
        market_neutral: True면 CS 평균 return 차감 (BTC-beta 노출 중립화).

    Returns:
        numpy array shape (n_bars, n_syms), dtype float32, values in {0.0, 0.5, 1.0}.

    """
    close_arr = close_wide.to_numpy(dtype=np.float64, copy=True)
    atr_arr = (
        atr_wide.reindex(close_wide.index, fill_value=np.nan).to_numpy(dtype=np.float64, copy=True)
    )
    np.nan_to_num(atr_arr, nan=0.0, copy=False)

    n_bars, n_syms = close_arr.shape
    close_safe = np.where(close_arr > 1e-12, close_arr, 1.0)

    # Barrier as return fraction: pt_multiplier * ATR / close
    barrier_frac = pt_multiplier * np.abs(atr_arr) / close_safe
    barrier_frac = np.clip(barrier_frac, 0.002, 0.20)  # 0.2%~20%

    # Track first hit bar for TP and SL (horizon = sentinel for "no hit")
    tp_bar = np.full((n_bars, n_syms), horizon, dtype=np.int32)
    sl_bar = np.full((n_bars, n_syms), horizon, dtype=np.int32)

    for h in range(1, horizon + 1):  # horizon=6 iterations — NOT a per-bar loop
        # Future close at h bars ahead
        future_close = np.empty_like(close_arr)
        future_close[:-h] = close_arr[h:]
        future_close[-h:] = np.nan

        ret = future_close / close_safe - 1.0  # (n_bars, n_syms)

        if market_neutral:
            # [B2] CS 평균 차감 → 시장 요인(BTC-beta) 중립화
            cs_mean = np.nanmean(ret, axis=1, keepdims=True)
            ret = ret - cs_mean

        # First hit only: 이미 hit된 위치는 갱신 안 함
        tp_hit_h = np.isfinite(ret) & (ret >= barrier_frac) & (tp_bar == horizon)
        sl_hit_h = np.isfinite(ret) & (ret <= -barrier_frac) & (sl_bar == horizon)
        tp_bar[tp_hit_h] = h
        sl_bar[sl_hit_h] = h

    # Label: TP 먼저 = 1.0, SL 먼저 = 0.0, 동시/없음 = 0.5
    labels = np.where(tp_bar < sl_bar, 1.0,
             np.where(sl_bar < tp_bar, 0.0, 0.5)).astype(np.float32)
    labels[-horizon:, :] = 0.5  # 미래 데이터 없는 마지막 bars
    return labels


def _compute_ic_linear_slot(
    wide_target: pd.DataFrame,
    feat_wide_list: list[pd.DataFrame],
    feat_names: list[str],
    is_mask_dt: pd.DatetimeIndex,
) -> np.ndarray:
    """IS IC-가중 선형 alpha 슬롯 계산 (G-ALPHA v9.0 A2 baseline).

    IS 구간에서 각 feature의 Spearman IC를 구하고,
    IC-가중 CS rank 합산 → 전체 기간 CS rank → [0,1] 알파 신호.

    Args:
        wide_target: target wide DataFrame (datetime × symbol).
        feat_wide_list: list of feature wide DataFrames (each datetime × symbol).
        feat_names: feature 이름 (로깅 용).
        is_mask_dt: datetime index values that belong to IS set.

    Returns:
        1D numpy array of linear alpha scores (float64), valid_mask 기준 길이.

    """
    if not feat_wide_list:
        n_valid = int(wide_target.notna().any(axis=1).sum())
        return np.full(n_valid, 0.5, dtype=np.float64)

    # IS 구간만 사용해 IC 계산
    is_idx = wide_target.index.isin(is_mask_dt)
    tgt_is = wide_target.loc[is_idx]
    r_tgt_is = tgt_is.rank(axis=1)

    feature_ics: list[float] = []
    for fw in feat_wide_list:
        fw_is = fw.reindex(wide_target.index).loc[is_idx]
        r_fw = fw_is.rank(axis=1)
        # per-bar Spearman IC (min 3 symbols), then IS mean
        valid_counts = (r_fw.notna() & r_tgt_is.notna()).sum(axis=1)
        valid_rows = valid_counts >= 3
        if not valid_rows.any():
            feature_ics.append(0.0)
            continue
        ic_series = r_fw.loc[valid_rows].corrwith(r_tgt_is.loc[valid_rows], axis=1).dropna()
        feature_ics.append(float(ic_series.mean()) if len(ic_series) > 0 else 0.0)

    ic_arr = np.array(feature_ics, dtype=np.float64)

    # IC-가중 CS rank 합산 (음수 IC → 해당 feature 반전하여 활용)
    score_wide = pd.DataFrame(0.0, index=wide_target.index, columns=wide_target.columns)
    for fw, w in zip(feat_wide_list, ic_arr):
        if abs(w) < 1e-6:
            continue
        fw_all = fw.reindex(wide_target.index)
        r_fw_all = fw_all.rank(axis=1)
        score_wide = score_wide.add(r_fw_all.multiply(w), fill_value=0.0)

    # CS rank per bar → percentile [0, 1]
    score_ranked = score_wide.rank(axis=1, pct=True).fillna(0.5)

    # valid_mask (target notna) 기준으로 flatten
    valid_mask_2d = wide_target.notna().values
    scores = score_ranked.values[valid_mask_2d].astype(np.float64)
    _logger.debug(
        "[IC-Linear] features=%d, mean_IC=%.4f, score_shape=%d",
        len(feat_names),
        float(np.mean(np.abs(ic_arr))),
        len(scores),
    )
    return scores


# v5 Theme Subspacing Definitions
THEME_GROUPS = {
    0: [  # Group 1: Trend/Momentum (Slots 00-04) + Vol-Adjusted
        "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", 
        "ma_dist_24", "ma_dist_168", "ret_vol_adj_24",
        "realized_vol_yz_24", "orderflow_price_divergence",
        "taker_absorption_score"
    ],
    1: [  # Group 2: Volatility/Mean-Reversion (Slots 05-09)
        # [G-ALPHA v9.0] CS-constant features 제거:
        # macro_vol_regime_shift — market-wide regime signal, 모든 symbol 동일값 (CS rank 무의미)
        # cs_dispersion — cross-sectional dispersion by definition, 모든 symbol 동일값
        "vol_ratio_24", "vol_ratio_168",
        "dist_from_weekly_vwap",
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
    X_pool: Pool | None,
    train_pool: Pool | None,
    feat_cols: list[str],
    seed_offset: int = 0,
) -> tuple[int, CatBoostRanker | None, list[str], np.ndarray]:
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
    X_pool: Pool | None,
    train_pool: Pool | None,
    feat_cols: list[str],
) -> tuple[int, CatBoostRegressor | None, np.ndarray]:
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
    # [G-ALPHA v9.0] slots_per_theme: 5→3 축소. staged prediction "fake breadth" 완화.
    # 총 slots = 3 themes × 3 = 9 (long), 9 (short). n_features_to_select은 __post_init__에서 자동 연동.
    slots_per_theme: int = 3
    n_features_to_select: int = 9
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
        """Convert continuous rank targets into Risk-Adjusted Soft Labels (G-ALPHA v8.1).
        
        Uses tanh transformation of returns normalized by ATR to scale the 
        cross-sectional rank targets, prioritizing high-conviction moves.
        """
        if raw_returns is None or atr_24h_pct is None:
            # Fallback to rank-based continuous labels if metadata is missing
            t = (target.to_numpy() / 2.0) + 0.5
            return (t if not short_oriented else 1.0 - t).astype(np.float32)

        # Convert to numpy for vectorized operations
        ret = raw_returns.to_numpy() if hasattr(raw_returns, "to_numpy") else raw_returns
        atr = atr_24h_pct.to_numpy() if hasattr(atr_24h_pct, "to_numpy") else atr_24h_pct
        
        if hasattr(target, "to_numpy"):
            target_vals = target.fillna(0.5).to_numpy()
        else:
            target_vals = np.nan_to_num(target, nan=0.5)
        
        # G-ALPHA v8.1: Combine Rank Direction with Return-based Magnitude.
        # 1. Center target [0, 1] to [-1, 1] to preserve cross-sectional ordering.
        target_centered = (target_vals - 0.5) * 2.0
        
        # G-ALPHA v8.2: Enhanced Conviction Scaling (tanh multiplier 2.0x)
        # Uses 2.0x multiplier to increase label variance and prevent CatBoost GPU pruning.
        mag = np.tanh(2.0 * np.abs(ret) / (atr + 1e-8))
        
        # 3. Combine rank direction with return-based magnitude.
        # Multiplicative combination scales the rank signal by absolute return conviction.
        soft_labels = target_centered * mag
        
        # 4. Shift back to [0, 1] for CatBoost PairLogit/RMSE compatibility.
        labels = (soft_labels / 2.0) + 0.5

        if short_oriented:
            # Note: y_labels_short is now calculated as 1.0 - y_labels externally
            # but we keep this for internal consistency/fallback.
            labels = 1.0 - labels
        
        return labels.astype(np.float32)

    def mine_alphas_cs(
        self,
        panel_df: pd.DataFrame,
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

        _logger.debug("Training MLAlphaMiner v6 (CatBoost %s + Dynamic Labeling)...", _DEFAULT_TASK_TYPE)
        
        # [Fix] Standardize indices to UTC aware to prevent alignment mismatches.
        def _standardize_idx(idx: pd.Index) -> pd.Index:
            if not isinstance(idx, pd.MultiIndex):
                return idx
            d_idx = -1
            s_idx = -1
            for i, name in enumerate(idx.names):
                if name and name.lower() in ("datetime", "time", "timestamp"): d_idx = i
                elif name and name.lower() in ("symbol", "ticker", "asset"): s_idx = i
            if d_idx == -1 or s_idx == -1: return idx
            
            d_vals = idx.get_level_values(d_idx)
            if getattr(d_vals, "tz", None) is None:
                d_vals = pd.to_datetime(d_vals, utc=True)
            else:
                d_vals = d_vals.tz_convert("UTC")
            return pd.MultiIndex.from_arrays(
                [d_vals, idx.get_level_values(s_idx)],
                names=["datetime", "symbol"]
            )

        # Work with sorted and standardized copy
        work_df = panel_df.sort_index(level=["datetime", "symbol"]).copy()
        work_df.index = _standardize_idx(work_df.index)
        
        # Standardized panel for filtering
        panel_df_std = panel_df.copy()
        panel_df_std.index = _standardize_idx(panel_df_std.index)

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

        # [Fix] Strict Data Alignment: Use groupby('symbol') for shifts to ensure row integrity.
        # This replaces the unstack() -> shift() -> mask approach which is fragile for mismatched indices.
        work_grouped = work_df.groupby("symbol", group_keys=False)
        
        # 1. Magnitude Horizon Log Returns (for auxiliary regression)
        # [B1] Forward 6-bar return은 triple-barrier 내부에서 처리.
        fwd_close_h = work_grouped["close"].transform(lambda x: x.shift(-_MAG_HORIZON_BARS))
        fwd_log_mag_h = np.log(fwd_close_h / work_df["close"].clip(lower=1e-12))
        y_mag_h_vals = np.abs(fwd_log_mag_h.fillna(0.0).to_numpy())
        y_mag_h = pd.Series(y_mag_h_vals, index=work_df.index)
        
        # 3. ATR (24 bars) for risk-normalization
        close_shift_1 = work_grouped["close"].shift(1)
        tr = np.maximum(
            work_df["high"] - work_df["low"],
            np.maximum(
                (work_df["high"] - close_shift_1).abs(),
                (work_df["low"] - close_shift_1).abs()
            )
        )
        # Use transform with lambda for reliable per-symbol rolling mean alignment
        atr_24 = tr.groupby(level="symbol", group_keys=False).transform(lambda x: x.rolling(24).mean())
        # [B1] atr_24_pct (pct-normalized) 불필요 — triple-barrier에서 ATR price units 직접 사용

        # Metadata + Wide format (B1 triple-barrier에 필요하므로 먼저 생성, 나중에 삭제)
        _close_wide = work_df["close"].unstack(level="symbol")
        valid_mask = _close_wide.notna().values
        idx_shape = _close_wide.shape

        # [G-ALPHA v9.0 B1+B2] Triple-barrier 레이블 (CS 평균 차감 포함)
        # _compute_triple_barrier_labels_wide: horizon 루프만 사용, per-bar 루프 없음
        _atr_wide = atr_24.unstack(level="symbol")  # ATR price units (per-symbol rolling mean of TR)
        _tb_labels_wide = _compute_triple_barrier_labels_wide(
            _close_wide, _atr_wide, horizon=6, pt_multiplier=1.5, market_neutral=True
        )
        # valid_mask 기준으로 flatten (valid_mask와 동일한 순서)
        y_labels = _tb_labels_wide[valid_mask].astype(np.float32)
        del _close_wide, _atr_wide, _tb_labels_wide

        _logger.info(
            "Label Stats - Triple-barrier: mean=%.4f, std=%.4f, nans=%d | "
            "TP=%.1f%%, SL=%.1f%%, neutral=%.1f%%",
            float(np.mean(y_labels)), float(np.std(y_labels)), int(np.isnan(y_labels).sum()),
            float(np.mean(y_labels == 1.0)) * 100,
            float(np.mean(y_labels == 0.0)) * 100,
            float(np.mean(y_labels == 0.5)) * 100,
        )
        
        # In-Sample Masking & Time-Series Split
        if is_end_date:
            cutoff = pd.to_datetime(is_end_date, utc=True)
            is_mask = work_df.index.get_level_values("datetime") < cutoff
        else:
            is_mask = np.ones(len(work_df), dtype=bool)

        # [Fix] Pre-calculate group IDs to ensure split happens at a group boundary
        full_group_ids = work_df.index.get_level_values("datetime").factorize()[0]

        # Time-Series Split for Validation (Last 20% of IS)
        is_indices = is_mask.nonzero()[0]
        if len(is_indices) > 1:
            split_idx = int(len(is_indices) * 0.8)
            
            # Ensure split_idx is at a group boundary to avoid CatBoostError:
            # "Subset's last group size is less than corresponding source group size"
            while split_idx < len(is_indices) and \
                  full_group_ids[is_indices[split_idx]] == full_group_ids[is_indices[split_idx-1]]:
                split_idx += 1
            
            if split_idx >= len(is_indices):
                # Fallback: move backward if we hit the end
                split_idx = int(len(is_indices) * 0.8)
                while split_idx > 0 and \
                      full_group_ids[is_indices[split_idx]] == full_group_ids[is_indices[split_idx-1]]:
                    split_idx -= 1
            
            train_idx = is_indices[:split_idx]
            eval_idx = is_indices[split_idx:]
        else:
            train_idx = eval_idx = is_indices

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
        _logger.info("GPU-Native Mining Loop v9.0 (A1: 독립모델 3개/theme, A3: short=1-long)...")
        total_loop_start = time.time()
        mag_finite = np.isfinite(y_mag_vals_raw)

        # iterations per theme
        _LONG_ITERS = {0: 1000, 1: 800, 2: 600}

        for theme_idx in range(3):
            if theme_idx == 1:
                fc = list(dict.fromkeys(THEME_GROUPS[theme_idx] + HMM_COLS))
            else:
                fc = list(dict.fromkeys(THEME_GROUPS[theme_idx]))
            fc = [c for c in fc if c in work_df.columns]

            if not fc:
                continue

            # --- Build pools: quantize ONCE, reuse across all staged predictions ---
            master_pool = train_pool = eval_pool = pred_pool = None
            master_mag_pool = mag_train_pool = mag_eval_pool = None

            if fc:
                X_long = work_df[fc].values.astype(np.float32)
                master_pool = Pool(
                    data=X_long,
                    label=y_labels,
                    group_id=full_group_ids,
                    feature_names=fc,
                )
                master_pool.quantize(border_count=254)
                train_pool = master_pool.slice(train_idx)
                eval_pool = master_pool.slice(eval_idx)
                pred_pool = master_pool

                master_mag_pool = Pool(
                    data=X_long,
                    label=y_mag_vals_raw,
                    weight=sample_weights_all,
                    feature_names=fc,
                )
                master_mag_pool.quantize(border_count=254)
                mag_train_mask = mag_finite[train_idx]
                mag_eval_mask = mag_finite[eval_idx]
                mag_train_pool = master_mag_pool.slice(train_idx[mag_train_mask])
                mag_eval_pool = master_mag_pool.slice(eval_idx[mag_eval_mask])

            theme_start = theme_idx * self.slots_per_theme
            theme_end = min(self.n_features_to_select, (theme_idx + 1) * self.slots_per_theme)
            n_virtual = theme_end - theme_start

            long_iters = _LONG_ITERS[theme_idx]

            # --- Phase 1: Magnitude Regressor only (theme당 1개, long/short ranker는 Phase 2에서 slot별 독립 학습) ---
            # [G-ALPHA v9.0 A1] long_ranker/short_ranker 단일 fit 제거.
            # staged checkpoint fake breadth 완전 폐기 → 독립 subsample 모델 3개/theme 로 대체.
            mag_params = {
                "loss_function": "MAE",
                "task_type": _DEFAULT_TASK_TYPE,
                "iterations": 500,
                "depth": 4,
                "learning_rate": 0.03,
                "logging_level": "Silent",
                "random_seed": 42 + theme_idx,
                "early_stopping_rounds": 50,
                "use_best_model": True,
            }
            if _DEFAULT_TASK_TYPE == "GPU":
                mag_params["devices"] = "0"
                mag_params["gpu_ram_part"] = 0.4

            mag_reg = CatBoostRegressor(**mag_params)
            if mag_train_pool is not None:
                try:
                    mag_reg.fit(mag_train_pool, eval_set=mag_eval_pool)
                except CatBoostError as exc:
                    _logger.warning("CatBoost Mag fit failed: %s. CPU fallback.", exc)
                    mag_params["task_type"] = "CPU"
                    mag_params.pop("devices", None)
                    mag_reg = CatBoostRegressor(**mag_params)
                    mag_reg.fit(mag_train_pool, eval_set=mag_eval_pool)

            mag_z: np.ndarray | None = None
            if mag_reg is not None and pred_pool is not None:
                mag_raw_all = mag_reg.predict(pred_pool)
                mu_m = float(np.mean(mag_raw_all))
                sig_m = float(np.std(mag_raw_all) + 1e-9)
                mag_z = np.clip((mag_raw_all - mu_m) / sig_m, -3.0, 3.0)

            # --- [G-ALPHA v9.0 A1] Phase 2: Independent models per slot (genuine breadth) ---
            # Staged checkpoint fake breadth 제거. slot당 subsample 다양화 독립 모델 = 진짜 diversity.
            # 각 슬롯은 (subsample, rsm, seed) 조합이 달라 상관관계 0.85~0.95 문제 해소.
            _SLOT_DIVERSITY = [
                {"subsample": 0.85, "rsm": 1.00, "seed_extra": 0},    # slot 0: full features
                {"subsample": 0.75, "rsm": 0.85, "seed_extra": 100},  # slot 1: feature subsample
                {"subsample": 0.65, "rsm": 0.70, "seed_extra": 200},  # slot 2: aggressive subsample
            ]
            n_independent = min(n_virtual, len(_SLOT_DIVERSITY))

            for v_idx in range(n_independent):
                s_idx = theme_start + v_idx
                div_cfg = _SLOT_DIVERSITY[v_idx]

                # --- Long independent model ---
                slot_long_ranker: CatBoostRanker | None = None
                if train_pool is not None:
                    slot_params = self._get_lgbm_params(
                        seed_offset=theme_idx * 100 + div_cfg["seed_extra"],
                        iterations=long_iters,
                    )
                    slot_params["logging_level"] = "Silent"
                    slot_params["subsample"] = div_cfg["subsample"]
                    slot_params["rsm"] = div_cfg["rsm"]
                    slot_long_ranker = CatBoostRanker(**slot_params)
                    try:
                        slot_long_ranker.fit(train_pool, eval_set=eval_pool)
                    except CatBoostError as exc:
                        _logger.debug("CatBoost Long slot%d fit failed: %s. CPU fallback.", v_idx, exc)
                        slot_params["task_type"] = "CPU"
                        slot_params.pop("devices", None)
                        slot_params.pop("gpu_ram_part", None)
                        slot_long_ranker = CatBoostRanker(**slot_params)
                        slot_long_ranker.fit(train_pool, eval_set=eval_pool)

                if slot_long_ranker is not None and pred_pool is not None:
                    raw_scores = slot_long_ranker.predict(pred_pool)
                    self._models[s_idx] = slot_long_ranker
                    self._feature_sets[s_idx] = fc
                    self._ntree_ends[s_idx] = 0  # 0 = full model, staged prediction 없음
                    _m_buf.fill(np.nan)
                    _m_buf[valid_mask] = raw_scores
                    slot_arrays[f"alpha_long_{s_idx:02d}"] = _fast_rank_2d_numba(_m_buf)[valid_mask]
                    if mag_z is not None:
                        self._mag_models[s_idx] = mag_reg
                        slot_arrays[f"mag_long_{s_idx:02d}"] = mag_z
                    else:
                        slot_arrays[f"mag_long_{s_idx:02d}"] = np.zeros(int(valid_mask.sum()), dtype=np.float64)
                else:
                    slot_arrays[f"alpha_long_{s_idx:02d}"] = np.full(int(valid_mask.sum()), 0.5)
                    slot_arrays[f"mag_long_{s_idx:02d}"] = np.zeros(int(valid_mask.sum()), dtype=np.float64)

                # [G-ALPHA v9.0 A3] Short 독립 모델 폐기: alpha_short = 1 - alpha_long
                # 별도 short ranker 반복 실패 검증됨. 단일 CS ranker 역순 활용.
                long_slot_arr = slot_arrays.get(f"alpha_long_{s_idx:02d}")
                if long_slot_arr is not None:
                    slot_arrays[f"alpha_short_{s_idx:02d}"] = 1.0 - long_slot_arr
                else:
                    slot_arrays[f"alpha_short_{s_idx:02d}"] = np.full(int(valid_mask.sum()), 0.5)

            # Free VRAM/RAM before next theme
            del pred_pool, train_pool, mag_train_pool
            del master_pool, master_mag_pool
            gc.collect()

        loop_elapsed = time.time() - total_loop_start
        _logger.info("GPU-Native Mining Loop completed in %.4f seconds.", loop_elapsed)

        # --- [G-ALPHA v9.0 A2] IC-가중 선형 Baseline 슬롯 (CatBoost vs 선형 결합 비교용) ---
        # CatBoost가 단순 IC-가중 선형 결합 대비 OOS에서 나은지 실증 데이터 확보.
        # Long baseline만 생성 (SHORT는 우선순위 낮음). 슬롯 인덱스: n_features_to_select + theme_idx.
        _logger.info("[G-ALPHA v9.0 A2] Computing IC-linear baseline slots...")
        try:
            if "target" in panel_df_std.columns:
                _u_tgt_wide = panel_df_std["target"].unstack(level="symbol")
                _is_dt_idx = pd.DatetimeIndex(
                    work_df.index.get_level_values("datetime")[is_mask].unique()
                )
                n_valid_total = int(valid_mask.sum())

                for _lin_theme_idx in range(3):
                    _lin_fc = list(dict.fromkeys(THEME_GROUPS[_lin_theme_idx]))
                    _lin_fc = [c for c in _lin_fc if c in work_df.columns]
                    if not _lin_fc:
                        _logger.debug("[A2] theme%d: no valid features, skip.", _lin_theme_idx)
                        continue

                    _feat_wide_list = [
                        work_df[f].unstack(level="symbol").reindex(_u_tgt_wide.index)
                        for f in _lin_fc
                    ]
                    _lin_slot_idx = self.n_features_to_select + _lin_theme_idx

                    _lin_scores = _compute_ic_linear_slot(
                        wide_target=_u_tgt_wide,
                        feat_wide_list=_feat_wide_list,
                        feat_names=_lin_fc,
                        is_mask_dt=_is_dt_idx,
                    )
                    if len(_lin_scores) == n_valid_total:
                        slot_arrays[f"alpha_long_{_lin_slot_idx:02d}"] = _lin_scores
                        slot_arrays[f"mag_long_{_lin_slot_idx:02d}"] = np.zeros(n_valid_total, dtype=np.float64)
                        _logger.info(
                            "  [A2] Linear baseline long theme%d → slot alpha_long_%02d (%d features)",
                            _lin_theme_idx, _lin_slot_idx, len(_lin_fc),
                        )
                    else:
                        _logger.warning(
                            "  [A2] Linear baseline theme%d length mismatch: %d vs %d, skip.",
                            _lin_theme_idx, len(_lin_scores), n_valid_total,
                        )
            else:
                _logger.warning("[A2] 'target' column missing from panel_df_std; skipping linear baseline.")
        except Exception as _a2_exc:
            _logger.warning("[A2] Linear baseline computation failed (non-fatal): %s", _a2_exc)

        # [Optimization ⑤] slot_arrays → DataFrame 일괄 구성 (컬럼별 시리즈 할당 제거)
        # valid_mask.sum() == len(work_df) (close가 항상 유효한 경우)이므로 직접 구성 가능
        raw_alpha_df = pd.DataFrame(slot_arrays, index=work_df.index)

        # Apply filtering on both long/short slots. Filter metadata will expose
        # direction-separated survivors while preserving backward-compatible aliases.
        long_slot_cols = [c for c in raw_alpha_df.columns if _LONG_SLOT_COL_RE.match(c)]
        short_slot_cols = [c for c in raw_alpha_df.columns if _SHORT_SLOT_COL_RE.match(c)]
        alpha_slot_cols = list(dict.fromkeys(long_slot_cols + short_slot_cols))
        filter_opts = filter_options or {}
        alpha_df_all, filt_meta = filter_alpha_components(
            raw_alpha_df.copy(),
            panel_df_std, # [Fix] Use standardized panel
            is_end_date=is_end_date,
            n_trials=max(1, 2 * 3 * 3),  # 3 themes × 2 directions × 3 independent models (A1: genuine breadth)
            fdr_q=float(filter_opts.get("fdr_q", 0.10)),
            alpha_cols=alpha_slot_cols,
            symbol_balance_max=float(filter_opts.get("symbol_balance_max", 3.0)),
            use_newey_west=bool(filter_opts.get("use_newey_west", True)),  # [B1] overlapping label t-stat 보정
            use_ewma_ic_stat=bool(filter_opts.get("use_ewma_ic_stat", False)),
            ewma_half_life=float(filter_opts.get("ewma_half_life", 540.0)),
            require_regime_gate=bool(filter_opts.get("require_regime_gate", True)),
            step3_regime_alpha_enabled=bool(filter_opts.get("step3_regime_alpha_enabled", False)),
            step3_chop_support_min=float(filter_opts.get("step3_chop_support_min", 0.25)),
            step3_chop_ic_min=float(filter_opts.get("step3_chop_ic_min", -0.01)),
            step3_chop_weight_mult=float(filter_opts.get("step3_chop_weight_mult", 0.50)),
            step3_weight_mult_floor=float(filter_opts.get("step3_weight_mult_floor", 0.20)),
        )
        # [Audit Fix] Use direction-separated survivors from metadata.
        # Keep backward compatibility by falling back to survived_cols parsing.
        surviving = list(filt_meta.get("survived_long_cols", []))
        surviving_short = list(filt_meta.get("survived_short_cols", []))
        if not surviving and not surviving_short:
            surviving_all = list(filt_meta.get("survived_cols", []))
            surviving_short = [c for c in surviving_all if _SHORT_SLOT_COL_RE.match(c)]
            surviving = [c for c in surviving_all if _LONG_SLOT_COL_RE.match(c)]
        pre_agg_long_count = int(float(filt_meta.get("n_surviving_long", len(surviving))))
        pre_agg_short_count = int(float(filt_meta.get("n_surviving_short", len(surviving_short))))
        filt_meta["pre_agg_surviving_long_count"] = float(pre_agg_long_count)
        filt_meta["pre_agg_surviving_short_count"] = float(pre_agg_short_count)
        
        mag_surv_cols = [c.replace("alpha_", "mag_") for c in surviving if c.replace("alpha_", "mag_") in alpha_df_all.columns]
        final_selected_long_cols: list[str] = []
        if surviving:
            ic_map = filt_meta.get("ic_weight_by_slot", {}) or filt_meta.get("ic_by_slot", {})
            weights = [max(0.0, float(ic_map.get(c, 0.0))) ** 2 for c in surviving]
            w_arr = np.array(weights)
            if w_arr.sum() > 1e-9:
                w_norm = w_arr / w_arr.sum()
                long_rank = (alpha_df_all[surviving] * w_norm).sum(axis=1)
                final_selected_long_cols = [c for c, w in zip(surviving, w_norm) if float(w) > 0.0]
                self._ic_weights = {int(_LONG_SLOT_COL_RE.match(c).group(1)): float(w) for c, w in zip(surviving, w_norm)}
            else:
                long_rank = alpha_df_all[surviving].mean(axis=1)
                final_selected_long_cols = list(surviving)
                self._ic_weights = {int(_LONG_SLOT_COL_RE.match(c).group(1)): 1.0 / len(surviving) for c in surviving}

            if mag_surv_cols:
                mag_blend = alpha_df_all[mag_surv_cols].mean(axis=1)
                long_rank = np.clip(long_rank * (1.0 + 0.3 * mag_blend), 0.02, 0.98)
            
            # G-ALPHA v8.0: Signal Masking (Middle 40% -> 0.5)
            # Neutralize signals with low conviction to increase sparsity and reduce noise.
            mask = (long_rank > 0.3) & (long_rank < 0.7)
            long_rank[mask] = 0.5
            alpha_df_all["alpha_long"] = long_rank
        else:
            self._ic_weights = {}
            alpha_df_all["alpha_long"] = 0.5

        final_selected_short_cols: list[str] = []
        if surviving_short:
            ic_map_short = filt_meta.get("ic_weight_by_slot", {}) or filt_meta.get("ic_by_slot", {})
            short_weights = [max(0.0, float(ic_map_short.get(c, 0.0))) ** 2 for c in surviving_short]
            sw_arr = np.array(short_weights)
            if sw_arr.sum() > 1e-9:
                sw_norm = sw_arr / sw_arr.sum()
                short_rank = (alpha_df_all[surviving_short] * sw_norm).sum(axis=1)
                final_selected_short_cols = [c for c, w in zip(surviving_short, sw_norm) if float(w) > 0.0]
                self._short_ic_weights = {int(_SHORT_SLOT_COL_RE.match(c).group(1)): float(w) for c, w in zip(surviving_short, sw_norm)}
            else:
                short_rank = alpha_df_all[surviving_short].mean(axis=1)
                final_selected_short_cols = list(surviving_short)
                self._short_ic_weights = {int(_SHORT_SLOT_COL_RE.match(c).group(1)): 1.0 / len(surviving_short) for c in surviving_short}
            
            # G-ALPHA v8.0: Signal Masking for Short
            short_mask = (short_rank > 0.3) & (short_rank < 0.7)
            short_rank[short_mask] = 0.5
            alpha_df_all["alpha_short"] = short_rank
            filt_meta["alpha_short_degraded_mode"] = 0.0
            filt_meta["alpha_short_degraded_reason"] = ""
        elif "alpha_long" in alpha_df_all.columns:
            self._short_ic_weights = {}
            alpha_df_all["alpha_short"] = 1.0 - alpha_df_all["alpha_long"]
            filt_meta["alpha_short_degraded_mode"] = 1.0
            filt_meta["alpha_short_degraded_reason"] = "no_survived_short_slots_fallback_to_1_minus_alpha_long"
            _logger.warning(
                "alpha_short fallback engaged: no survived short slots, using 1 - alpha_long."
            )
        else:
            self._short_ic_weights = {}
            alpha_df_all["alpha_short"] = 0.5
            filt_meta["alpha_short_degraded_mode"] = 1.0
            filt_meta["alpha_short_degraded_reason"] = "no_alpha_long_and_no_survived_short_slots_fallback_to_flat"
            _logger.warning(
                "alpha_short fallback engaged: no alpha_long and no survived short slots, using flat 0.5."
            )

        post_agg_long_count = len(final_selected_long_cols)
        post_agg_short_count = len(final_selected_short_cols)
        filt_meta["post_agg_selected_long_count"] = float(post_agg_long_count)
        filt_meta["post_agg_selected_short_count"] = float(post_agg_short_count)
        filt_meta["post_agg_selected_long_cols"] = final_selected_long_cols
        filt_meta["post_agg_selected_short_cols"] = final_selected_short_cols
        filt_meta["final_selection_fail_long"] = float(max(0, pre_agg_long_count - post_agg_long_count))
        filt_meta["final_selection_fail_short"] = float(max(0, pre_agg_short_count - post_agg_short_count))
        filt_meta["elite_zero_after_survival"] = 1.0 if (pre_agg_long_count > 0 and post_agg_long_count == 0) else 0.0
        if pre_agg_long_count != post_agg_long_count or pre_agg_short_count != post_agg_short_count:
            _logger.warning(
                "Alpha final aggregation mismatch | long %d->%d | short %d->%d",
                pre_agg_long_count,
                post_agg_long_count,
                pre_agg_short_count,
                post_agg_short_count,
            )

        # [Fix] Final reindex using standardized panel index, then restore original index
        out_df = alpha_df_all.reindex(panel_df_std.index).fillna(0.5)
        out_df.index = panel_df.index
        
        out_df.attrs["alpha_component_filter"] = filt_meta
        out_df.attrs["alpha_final_aggregation_counts"] = {
            "pre_agg_surviving_long_count": float(pre_agg_long_count),
            "post_agg_selected_long_count": float(post_agg_long_count),
            "pre_agg_surviving_short_count": float(pre_agg_short_count),
            "post_agg_selected_short_count": float(post_agg_short_count),
            "elite_zero_after_survival": float(filt_meta.get("elite_zero_after_survival", 0.0)),
        }
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
        """CatBoostRanker base parameters for G-ALPHA v8.0.

        Args:
            seed_offset: Offset for random_seed.
            iterations: Number of training iterations.

        Returns:
            Dictionary of CatBoost parameters.

        """
        params = {
            "loss_function": "PairLogit",
            "eval_metric": "NDCG",
            "task_type": _DEFAULT_TASK_TYPE,
            "iterations": iterations,
            "depth": 4,             # G-ALPHA v8.0: Shallow trees for generalization
            "border_count": 254,
            "learning_rate": 0.03,
            "l2_leaf_reg": 30.0,    # Strong regularization
            "random_strength": 1.5,
            "metric_period": 100,
            "logging_level": "Silent",
            "random_seed": 42 + seed_offset,
            "allow_writing_files": False,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.85,
            "early_stopping_rounds": 50,
            "use_best_model": True,
        }
        if _DEFAULT_TASK_TYPE == "GPU":
            params["devices"] = "0"
            params["gpu_ram_part"] = 0.4
        return params

    def transform_cs(self, panel_df: pd.DataFrame) -> pd.DataFrame:
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
            
            # Build Prediction Pool for this theme (long only; A3: short=1-long)
            pred_pool = Pool(data=work_df[fc].values.astype(np.float32)) if fc else None

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

                # [G-ALPHA v9.0 A3] Short 독립 모델 폐기: alpha_short = 1 - alpha_long
                long_col = f"alpha_long_{slot_idx:02d}"
                if long_col in out_df.columns:
                    out_df[f"alpha_short_{slot_idx:02d}"] = 1.0 - out_df[long_col]
                else:
                    out_df[f"alpha_short_{slot_idx:02d}"] = 0.5

            del pred_pool
            gc.collect()

        surviving = [c for c in out_df.columns if _LONG_SLOT_COL_RE.match(c) and out_df[c].std() > 1e-6]
        surviving_short = [c for c in out_df.columns if _SHORT_SLOT_COL_RE.match(c) and out_df[c].std() > 1e-6]
        mag_surv_cols = [c.replace("alpha_", "mag_") for c in surviving if c.replace("alpha_", "mag_") in out_df.columns]
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

            if mag_surv_cols:
                mag_blend = out_df[mag_surv_cols].mean(axis=1)
                long_rank = np.clip(long_rank * (1.0 + 0.3 * mag_blend), 0.02, 0.98)
            
            # G-ALPHA v8.0: Signal Masking (Middle 40% -> 0.5)
            mask = (long_rank > 0.3) & (long_rank < 0.7)
            long_rank[mask] = 0.5
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
            
            # G-ALPHA v8.0: Signal Masking for Short
            short_mask = (short_rank > 0.3) & (short_rank < 0.7)
            short_rank[short_mask] = 0.5
            out_df["alpha_short"] = short_rank
        else:
            out_df["alpha_short"] = 1.0 - out_df["alpha_long"]

        if "target" in panel_df.columns:
            out_df["target"] = panel_df["target"]

        return out_df.reindex(panel_df.index).fillna(0.5)
