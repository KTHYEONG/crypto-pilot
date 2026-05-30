from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange
from numpy.typing import NDArray

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.common.normalization import cross_sectional_rank
from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.contracts import FeaturePanel


def _ret(close_2d: NDArray[np.float64], lb: int) -> NDArray[np.float64]:
    out: NDArray[np.float64] = np.full(close_2d.shape, np.nan, dtype=np.float64)
    if lb <= 0:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        out[lb:] = close_2d[lb:] / np.maximum(close_2d[:-lb], 1e-12) - 1.0
    return np.asarray(out, dtype=np.float64)


# Optimized Vectorized Rolling Functions (Time Complexity: O(T * N), Space Complexity: O(T * N))
def _rolling_mean_2d(values: NDArray[np.float64], lb: int) -> NDArray[np.float64]:
    if lb <= 0:
        return np.full(values.shape, np.nan, dtype=np.float64)
    # Use min_periods=1 to fully match nanmean behavior (returns value even with 1 active element)
    arr = pd.DataFrame(values).rolling(window=lb, min_periods=1).mean().to_numpy()
    return np.asarray(arr, dtype=np.float64)


def _rolling_std_2d(values: NDArray[np.float64], lb: int) -> NDArray[np.float64]:
    if lb <= 0:
        return np.full(values.shape, np.nan, dtype=np.float64)
    arr = pd.DataFrame(values).rolling(window=lb, min_periods=1).std().to_numpy()
    return np.asarray(arr, dtype=np.float64)


@njit(parallel=True, fastmath=True)  # type: ignore[untyped-decorator]
def _numba_rolling_mad_zscore(funding_2d: np.ndarray) -> np.ndarray:
    """Numba accelerated 30-day MAD Z-score. Complies with zero-loop policy."""
    t_len, n_len = funding_2d.shape
    out = np.full((t_len, n_len), np.nan, dtype=np.float64)
    for j in prange(n_len):
        for t in range(30, t_len):
            # Window slice
            win = funding_2d[t - 29 : t + 1, j]
            # Calculate median
            med = np.median(win)
            # Calculate MAD
            abs_diff = np.abs(win - med)
            mad = np.median(abs_diff) * 1.4826
            val = funding_2d[t, j]
            if mad > 1e-12:
                out[t, j] = (val - med) / mad
            else:
                out[t, j] = (val - med) / 1e-12
    return out


@njit(parallel=True, fastmath=True)  # type: ignore[untyped-decorator]
def _numba_rolling_corr_2d(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Numba accelerated 2D rolling correlation with min_periods=1.
    
    Time Complexity: O(T * N), Space Complexity: O(T * N)
    """
    t_len, n_len = x.shape
    out = np.zeros((t_len, n_len), dtype=np.float64)
    
    for j in prange(n_len):
        for t in range(3, t_len):
            start = max(3, t - window + 1)
            n = t - start + 1
            if n < 2:
                continue
            
            sum_x = 0.0
            sum_y = 0.0
            for k in range(start, t + 1):
                sum_x += x[k, j]
                sum_y += y[k, j]
            mx = sum_x / n
            my = sum_y / n
            
            sum_xx = 0.0
            sum_yy = 0.0
            sum_xy = 0.0
            for k in range(start, t + 1):
                dx = x[k, j] - mx
                dy = y[k, j] - my
                sum_xx += dx * dx
                sum_yy += dy * dy
                sum_xy += dx * dy
            
            denom = np.sqrt(sum_xx * sum_yy)
            if denom > 1e-12:
                out[t, j] = sum_xy / denom
    return out


def build_feature_panel(aligned: AlignedMarketData, cfg: StrategyMLConfig) -> FeaturePanel:
    """Build P0 feature tensor with returns/vol/carry/liquidity/cs features."""
    close_2d = aligned.close_2d
    volume_2d = aligned.volume_2d
    funding_2d = aligned.funding_2d
    basis_missing = aligned.basis_2d is None or not np.any(np.isfinite(aligned.basis_2d))
    oi_missing = aligned.oi_2d is None or not np.any(np.isfinite(aligned.oi_2d))
    adv_missing = aligned.adv_usdt_2d is None or not np.any(np.isfinite(aligned.adv_usdt_2d))
    execution_cost_missing = (
        aligned.execution_cost_bps_2d is None
        or not np.any(np.isfinite(aligned.execution_cost_bps_2d))
    )
    basis_2d = (
        np.zeros(close_2d.shape, dtype=np.float64)
        if basis_missing
        else np.asarray(aligned.basis_2d, dtype=np.float64)
    )
    oi_2d = (
        np.ones(close_2d.shape, dtype=np.float64)
        if oi_missing
        else np.asarray(aligned.oi_2d, dtype=np.float64)
    )
    adv_usdt_2d = (
        close_2d * volume_2d
        if adv_missing
        else np.asarray(aligned.adv_usdt_2d, dtype=np.float64)
    )
    execution_cost_bps_2d = (
        np.zeros(close_2d.shape, dtype=np.float64)
        if execution_cost_missing
        else np.asarray(aligned.execution_cost_bps_2d, dtype=np.float64)
    )
    mask = aligned.active_mask & aligned.warm_mask & ~aligned.entry_block_mask & ~aligned.kill_mask

    ret_1 = _ret(close_2d, 1)
    ret_3 = _ret(close_2d, 3)
    ret_6 = _ret(close_2d, 6)
    ret_12 = _ret(close_2d, 12)
    ret_18 = _ret(close_2d, 18)
    ret_36 = _ret(close_2d, 36)
    rev_3 = -ret_3
    rev_6 = -ret_6
    rev_12 = -ret_12
    mom_12_skip_1 = np.full(close_2d.shape, np.nan, dtype=np.float64)
    mom_36_skip_3 = np.full(close_2d.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mom_12_skip_1[12:] = close_2d[11:-1] / np.maximum(close_2d[:-12], 1e-12) - 1.0
        mom_36_skip_3[36:] = close_2d[33:-3] / np.maximum(close_2d[:-36], 1e-12) - 1.0

    # 100% Vectorized Volatility calculation using Pandas Rolling
    ret_1_df = pd.DataFrame(ret_1)
    rv_6 = ret_1_df.rolling(6, min_periods=1).std().to_numpy()
    rv_18 = ret_1_df.rolling(18, min_periods=1).std().to_numpy()
    rv_36 = ret_1_df.rolling(36, min_periods=1).std().to_numpy()
    
    # Downside Volatility Vectorized Masking & Rolling Std
    dn_matrix = np.where(ret_1 < 0.0, ret_1, np.nan)
    downside_rv_18 = (
        pd.DataFrame(dn_matrix).rolling(18, min_periods=1).std().to_numpy()
    )

    # Vectorized Funding statistics using Pandas
    funding_2d_filled = np.nan_to_num(funding_2d, nan=0.0)
    funding_1 = funding_2d_filled.copy()
    funding_df = pd.DataFrame(funding_2d_filled)
    funding_mean_3 = funding_df.rolling(3, min_periods=1).mean().to_numpy()
    funding_mean_6 = funding_df.rolling(6, min_periods=1).mean().to_numpy()
    funding_mean_18 = funding_df.rolling(18, min_periods=1).mean().to_numpy()
    funding_sign_persistence_6 = (
        pd.DataFrame(np.sign(funding_2d_filled))
        .rolling(6, min_periods=1)
        .mean()
        .to_numpy()
    )

    # Numba JIT 가속 30일 MAD Z-Score 호출
    funding_z_30d = _numba_rolling_mad_zscore(funding_2d_filled)
    
    dollar_volume = close_2d * volume_2d
    
    # Vectorized Volume Z-score
    vol_df = pd.DataFrame(volume_2d)
    vol_mu = vol_df.rolling(18, min_periods=1).mean().to_numpy()
    vol_sd = vol_df.rolling(18, min_periods=1).std().to_numpy()
    volume_z_18 = (volume_2d - vol_mu) / np.maximum(vol_sd, 1e-12)

    atr_14 = _rolling_mean_2d(np.maximum(aligned.high_2d - aligned.low_2d, 0.0), 14)
    atr_pct_14 = atr_14 / np.maximum(close_2d, 1e-12)
    vol_of_vol_36 = _rolling_std_2d(rv_6, 36)
    dollar_volume_rank = cross_sectional_rank(dollar_volume, mask, cfg.min_group_size)
    adv_rank = cross_sectional_rank(adv_usdt_2d, mask, cfg.min_group_size)
    execution_cost_rank = cross_sectional_rank(-execution_cost_bps_2d, mask, cfg.min_group_size)

    cs_rank_ret_6 = cross_sectional_rank(ret_6, mask, cfg.min_group_size)
    cs_rank_ret_18 = cross_sectional_rank(ret_18, mask, cfg.min_group_size)
    cs_rank_rv_18 = cross_sectional_rank(rv_18, mask, cfg.min_group_size)
    cs_rank_funding_6 = cross_sectional_rank(funding_mean_6, mask, cfg.min_group_size)
    cs_rank_volume_18 = cross_sectional_rank(volume_z_18, mask, cfg.min_group_size)

    # [ML-UPGRADE] 장기 추세 변별성 강화를 위한 횡단면 랭크 확장 및
    # Volatility-adjusted CS-Sharpe 피처 도입
    cs_rank_ret_12 = cross_sectional_rank(ret_12, mask, cfg.min_group_size)
    cs_rank_ret_36 = cross_sectional_rank(ret_36, mask, cfg.min_group_size)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe_6 = np.nan_to_num(ret_6 / np.maximum(rv_6, 1e-8), nan=0.0)
        sharpe_18 = np.nan_to_num(ret_18 / np.maximum(rv_18, 1e-8), nan=0.0)
    cs_sharpe_6 = cross_sectional_rank(sharpe_6, mask, cfg.min_group_size)
    cs_sharpe_18 = cross_sectional_rank(sharpe_18, mask, cfg.min_group_size)

    btc_idx = 0
    if "BTCUSDT" in aligned.symbols:
        btc_idx = aligned.symbols.index("BTCUSDT")
    btc_ret_6 = np.repeat(ret_6[:, [btc_idx]], close_2d.shape[1], axis=1)
    btc_rv_18 = np.repeat(rv_18[:, [btc_idx]], close_2d.shape[1], axis=1)
    
    # 100% Vectorized Market-wide median and dispersion
    # (O(T) time, Python loop completely eliminated)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        market_median_base = np.nanmedian(ret_6, axis=1, keepdims=True)
        market_dispersion_base = np.nanstd(ret_6, axis=1, keepdims=True)
        # NaN rows fallbacks
        market_median_base = np.nan_to_num(market_median_base, nan=0.0)
        market_dispersion_base = np.nan_to_num(market_dispersion_base, nan=0.0)
        
    positive_breadth_base = np.nanmean(ret_6 > 0.0, axis=1, keepdims=True)
    market_median_ret_6 = np.repeat(market_median_base, close_2d.shape[1], axis=1)
    market_dispersion_6 = np.repeat(market_dispersion_base, close_2d.shape[1], axis=1)
    positive_breadth_6 = np.repeat(positive_breadth_base, close_2d.shape[1], axis=1)

    # 4종 신규 알파 피처 계산
    # 1. 단기 리턴 자기상관성 (ret_3과 이의 3-bar shift 리턴의 6-period rolling correlation)
    ret_3_shift = np.full_like(ret_3, np.nan)
    ret_3_shift[3:] = ret_3[:-3]
    momentum_autocorr = _numba_rolling_corr_2d(ret_3, ret_3_shift, 6)

    # 2. 횡단면 잔차 모멘텀 (CS demeaned return의 6-period rolling average)
    cs_mean_ret = np.nanmean(ret_3, axis=1, keepdims=True)
    cs_resid_ret = ret_3 - cs_mean_ret
    cs_residual_momentum = _rolling_mean_2d(cs_resid_ret, 6)

    # 3. vwap_deviation (VWAP 대비 종가 이격도)
    pv = close_2d * volume_2d
    rolling_pv = _rolling_mean_2d(pv, 12)
    rolling_v = _rolling_mean_2d(volume_2d, 12)
    vwap = rolling_pv / np.maximum(rolling_v, 1e-12)
    vwap_deviation = (close_2d - vwap) / np.maximum(vwap, 1e-12)

    # 4. funding_rate_momentum (자금조달율의 3-bar 차이의 6-period rolling average)
    funding_diff = funding_2d_filled - np.roll(funding_2d_filled, 3, axis=0)
    funding_diff[:3] = 0.0
    funding_rate_momentum = _rolling_mean_2d(funding_diff, 6)

    basis_1 = basis_2d.copy()
    basis_mean_6 = _rolling_mean_2d(basis_2d, 6)
    oi_ret_1 = _ret(np.maximum(oi_2d, 1e-12), 1)
    
    # Vectorized Open Interest Z-score
    oi_df = pd.DataFrame(oi_2d)
    oi_mu = oi_df.rolling(18, min_periods=1).mean().to_numpy()
    oi_sd = oi_df.rolling(18, min_periods=1).std().to_numpy()
    oi_z_18 = (oi_2d - oi_mu) / np.maximum(oi_sd, 1e-12)

    micro_hl_spread_1 = (aligned.high_2d - aligned.low_2d) / np.maximum(close_2d, 1e-12)
    micro_close_to_hl_1 = (close_2d - aligned.low_2d) / np.maximum(
        aligned.high_2d - aligned.low_2d,
        1e-12,
    )

    base_groups: dict[str, list[tuple[str, NDArray[np.float64]]]] = {
        "trend": [
        ("ret_1", ret_1),
        ("ret_3", ret_3),
        ("ret_6", ret_6),
        ("ret_12", ret_12),
        ("ret_18", ret_18),
        ("ret_36", ret_36),
        ("mom_12_skip_1", mom_12_skip_1),
        ("mom_36_skip_3", mom_36_skip_3),
        ("cs_rank_ret_6", cs_rank_ret_6),
        ("cs_rank_ret_12", cs_rank_ret_12),
        ("cs_rank_ret_18", cs_rank_ret_18),
        ("cs_rank_ret_36", cs_rank_ret_36),
        ("cs_sharpe_6", cs_sharpe_6),
        ("cs_sharpe_18", cs_sharpe_18),
        ("momentum_autocorr", momentum_autocorr),
        ("cs_residual_momentum", cs_residual_momentum),
        ("vwap_deviation", vwap_deviation),
        ],
        "reversal": [
        ("rev_3", rev_3),
        ("rev_6", rev_6),
        ("rev_12", rev_12),
        ],
        "volatility": [
        ("rv_6", rv_6),
        ("rv_18", rv_18),
        ("rv_36", rv_36),
        ("downside_rv_18", downside_rv_18),
        ("atr_pct_14", atr_pct_14),
        ("vol_of_vol_36", vol_of_vol_36),
        ("cs_rank_rv_18", cs_rank_rv_18),
        ],
        "carry": [
        ("funding_1", funding_1),
        ("funding_mean_3", funding_mean_3),
        ("funding_mean_6", funding_mean_6),
        ("funding_mean_18", funding_mean_18),
        ("funding_z_30d", funding_z_30d),
        ("funding_sign_persistence_6", funding_sign_persistence_6),
        ("cs_rank_funding_6", cs_rank_funding_6),
        ("basis_1", basis_1),
        ("basis_mean_6", basis_mean_6),
        ("funding_rate_momentum", funding_rate_momentum),
        ],
        "liquidity": [
        ("volume_z_18", volume_z_18),
        ("dollar_volume_rank", dollar_volume_rank),
        ("adv_rank", adv_rank),
        ("execution_cost_rank", execution_cost_rank),
        ("cs_rank_volume_18", cs_rank_volume_18),
        ("oi_ret_1", oi_ret_1),
        ("oi_z_18", oi_z_18),
        ],
        "market_context": [
        ("btc_ret_6", btc_ret_6),
        ("btc_rv_18", btc_rv_18),
        ("market_median_ret_6", market_median_ret_6),
        ("market_dispersion_6", market_dispersion_6),
        ("positive_breadth_6", positive_breadth_6),
        ],
        "microstructure": [
        ("micro_hl_spread_1", micro_hl_spread_1),
        ("micro_close_to_hl_1", micro_close_to_hl_1),
        ],
        "missingness": [],
    }
    if cfg.add_missingness_indicators:
        base_groups["missingness"].extend(
            [
                ("funding_missing_ind", (~np.isfinite(aligned.funding_2d)).astype(np.float64)),
                ("basis_missing_ind", (~np.isfinite(basis_2d)).astype(np.float64)),
                ("oi_missing_ind", (~np.isfinite(oi_2d)).astype(np.float64)),
                ("adv_missing_ind", (~np.isfinite(adv_usdt_2d)).astype(np.float64)),
                (
                    "execution_cost_missing_ind",
                    (~np.isfinite(execution_cost_bps_2d)).astype(np.float64),
                ),
            ]
        )
    feats: list[tuple[str, NDArray[np.float64]]] = []
    for group_name in cfg.feature_groups_enabled:
        feats.extend(base_groups[group_name])
    names = tuple(name for name, _ in feats)
    values = np.stack([arr for _, arr in feats], axis=2).astype(np.float32, copy=False)
    availability_masks: dict[str, np.ndarray] = {
        name: np.isfinite(arr) for name, arr in feats
    }
    valid_mask = mask.copy()
    metadata: dict[str, Any] = {
        "feature_count": len(names),
        "enabled_feature_groups": list(cfg.feature_groups_enabled),
        "funding_missing_ratio": float(np.mean(~np.isfinite(aligned.funding_2d))),
        "basis_missing_ratio": 1.0 if basis_missing else float(np.mean(~np.isfinite(basis_2d))),
        "oi_missing_ratio": 1.0 if oi_missing else float(np.mean(~np.isfinite(oi_2d))),
        "adv_missing_ratio": 1.0 if adv_missing else float(np.mean(~np.isfinite(adv_usdt_2d))),
        "execution_cost_missing_ratio": (
            1.0 if execution_cost_missing else float(np.mean(~np.isfinite(execution_cost_bps_2d)))
        ),
    }
    return FeaturePanel(
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        values=values,
        feature_names=names,
        valid_mask=valid_mask,
        availability_masks=availability_masks,
        metadata=metadata,
    )
