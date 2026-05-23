from __future__ import annotations

from typing import Any

import numpy as np
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


def _rolling_mean_2d(values: NDArray[np.float64], lb: int) -> NDArray[np.float64]:
    out: NDArray[np.float64] = np.full(values.shape, np.nan, dtype=np.float64)
    if lb <= 0:
        return out
    for t in range(lb - 1, values.shape[0]):
        out[t] = np.nanmean(values[t - lb + 1 : t + 1], axis=0)
    return out


def _rolling_std_2d(values: NDArray[np.float64], lb: int) -> NDArray[np.float64]:
    out: NDArray[np.float64] = np.full(values.shape, np.nan, dtype=np.float64)
    if lb <= 0:
        return out
    for t in range(lb - 1, values.shape[0]):
        out[t] = np.nanstd(values[t - lb + 1 : t + 1], axis=0)
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

    rv_6 = np.full(close_2d.shape, np.nan, dtype=np.float64)
    rv_18 = np.full(close_2d.shape, np.nan, dtype=np.float64)
    rv_36 = np.full(close_2d.shape, np.nan, dtype=np.float64)
    downside_rv_18 = np.full(close_2d.shape, np.nan, dtype=np.float64)
    for t in range(close_2d.shape[0]):
        if t >= 6:
            rv_6[t] = np.nanstd(ret_1[t - 5 : t + 1], axis=0)
        if t >= 18:
            w18 = ret_1[t - 17 : t + 1]
            rv_18[t] = np.nanstd(w18, axis=0)
            dn = np.where(w18 < 0.0, w18, np.nan)
            downside_rv_18[t] = np.nanstd(dn, axis=0)
        if t >= 36:
            rv_36[t] = np.nanstd(ret_1[t - 35 : t + 1], axis=0)

    funding_1 = np.nan_to_num(funding_2d, nan=0.0).copy()
    funding_mean_3 = np.full(close_2d.shape, np.nan, dtype=np.float64)
    funding_mean_6 = np.full(close_2d.shape, np.nan, dtype=np.float64)
    funding_mean_18 = np.full(close_2d.shape, np.nan, dtype=np.float64)
    funding_sign_persistence_6 = np.full(close_2d.shape, np.nan, dtype=np.float64)
    for t in range(close_2d.shape[0]):
        if t >= 3:
            funding_mean_3[t] = np.nanmean(
                np.nan_to_num(funding_2d[t - 2 : t + 1], nan=0.0),
                axis=0,
            )
        if t >= 6:
            w6 = np.nan_to_num(funding_2d[t - 5 : t + 1], nan=0.0)
            funding_mean_6[t] = np.nanmean(w6, axis=0)
            funding_sign_persistence_6[t] = np.mean(np.sign(w6), axis=0)
        if t >= 18:
            funding_mean_18[t] = np.nanmean(
                np.nan_to_num(funding_2d[t - 17 : t + 1], nan=0.0),
                axis=0,
            )

    funding_z_30d = np.full(close_2d.shape, np.nan, dtype=np.float64)
    for t in range(30, close_2d.shape[0]):
        win = np.nan_to_num(funding_2d[t - 29 : t + 1], nan=0.0)
        med = np.nanmedian(win, axis=0)
        mad = np.nanmedian(np.abs(win - med), axis=0) * 1.4826
        funding_z_30d[t] = (np.nan_to_num(funding_2d[t], nan=0.0) - med) / np.maximum(mad, 1e-12)
    dollar_volume = close_2d * volume_2d
    volume_z_18 = np.full(close_2d.shape, np.nan, dtype=np.float64)
    for t in range(18, close_2d.shape[0]):
        w = volume_2d[t - 17 : t + 1]
        mu = np.nanmean(w, axis=0)
        sd = np.nanstd(w, axis=0)
        volume_z_18[t] = (volume_2d[t] - mu) / np.maximum(sd, 1e-12)

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
    cs_rank_dollar_vol = cross_sectional_rank(dollar_volume, mask, cfg.min_group_size)

    btc_idx = 0
    if "BTCUSDT" in aligned.symbols:
        btc_idx = aligned.symbols.index("BTCUSDT")
    btc_ret_6 = np.repeat(ret_6[:, [btc_idx]], close_2d.shape[1], axis=1)
    btc_rv_18 = np.repeat(rv_18[:, [btc_idx]], close_2d.shape[1], axis=1)
    # Warmup rows can be all-NaN for ret_6; compute row-wise stats without warnings.
    market_median_base = np.zeros((close_2d.shape[0], 1), dtype=np.float64)
    market_dispersion_base = np.zeros((close_2d.shape[0], 1), dtype=np.float64)
    finite_ret6 = np.isfinite(ret_6)
    for t in range(close_2d.shape[0]):
        row = ret_6[t, finite_ret6[t]]
        if row.size == 0:
            continue
        market_median_base[t, 0] = float(np.median(row))
        market_dispersion_base[t, 0] = float(np.std(row))
    positive_breadth_base = np.nanmean(ret_6 > 0.0, axis=1, keepdims=True)
    market_median_ret_6 = np.repeat(market_median_base, close_2d.shape[1], axis=1)
    market_dispersion_6 = np.repeat(market_dispersion_base, close_2d.shape[1], axis=1)
    positive_breadth_6 = np.repeat(positive_breadth_base, close_2d.shape[1], axis=1)

    xs_reversal_prior_6 = -cs_rank_ret_6
    carry_prior_6 = cs_rank_funding_6

    basis_1 = basis_2d.copy()
    basis_mean_6 = _rolling_mean_2d(basis_2d, 6)
    oi_ret_1 = _ret(np.maximum(oi_2d, 1e-12), 1)
    oi_z_18 = np.full(close_2d.shape, np.nan, dtype=np.float64)
    for t in range(18, close_2d.shape[0]):
        w = oi_2d[t - 17 : t + 1]
        mu = np.nanmean(w, axis=0)
        sd = np.nanstd(w, axis=0)
        oi_z_18[t] = (oi_2d[t] - mu) / np.maximum(sd, 1e-12)

    micro_hl_spread_1 = (aligned.high_2d - aligned.low_2d) / np.maximum(close_2d, 1e-12)
    micro_close_to_hl_1 = (close_2d - aligned.low_2d) / np.maximum(
        aligned.high_2d - aligned.low_2d,
        1e-12,
    )

    feats = [
        ("ret_1", ret_1),
        ("ret_3", ret_3),
        ("ret_6", ret_6),
        ("ret_12", ret_12),
        ("ret_18", ret_18),
        ("ret_36", ret_36),
        ("rev_3", rev_3),
        ("rev_6", rev_6),
        ("rev_12", rev_12),
        ("mom_12_skip_1", mom_12_skip_1),
        ("mom_36_skip_3", mom_36_skip_3),
        ("rv_6", rv_6),
        ("rv_18", rv_18),
        ("rv_36", rv_36),
        ("downside_rv_18", downside_rv_18),
        ("atr_pct_14", atr_pct_14),
        ("vol_of_vol_36", vol_of_vol_36),
        ("funding_1", funding_1),
        ("funding_mean_3", funding_mean_3),
        ("funding_mean_6", funding_mean_6),
        ("funding_mean_18", funding_mean_18),
        ("funding_z_30d", funding_z_30d),
        ("funding_sign_persistence_6", funding_sign_persistence_6),
        ("volume_z_18", volume_z_18),
        ("dollar_volume_rank", dollar_volume_rank),
        ("adv_rank", adv_rank),
        ("execution_cost_rank", execution_cost_rank),
        ("cs_rank_ret_6", cs_rank_ret_6),
        ("cs_rank_ret_18", cs_rank_ret_18),
        ("cs_rank_rv_18", cs_rank_rv_18),
        ("cs_rank_funding_6", cs_rank_funding_6),
        ("cs_rank_volume_18", cs_rank_volume_18),
        ("cs_rank_dollar_volume", cs_rank_dollar_vol),
        ("btc_ret_6", btc_ret_6),
        ("btc_rv_18", btc_rv_18),
        ("market_median_ret_6", market_median_ret_6),
        ("market_dispersion_6", market_dispersion_6),
        ("positive_breadth_6", positive_breadth_6),
        ("xs_reversal_prior_6", xs_reversal_prior_6),
        ("carry_prior_6", carry_prior_6),
        ("basis_1", basis_1),
        ("basis_mean_6", basis_mean_6),
        ("oi_ret_1", oi_ret_1),
        ("oi_z_18", oi_z_18),
        ("micro_hl_spread_1", micro_hl_spread_1),
        ("micro_close_to_hl_1", micro_close_to_hl_1),
    ]
    names = tuple(name for name, _ in feats)
    values = np.stack([arr for _, arr in feats], axis=2).astype(np.float32, copy=False)
    availability_masks: dict[str, np.ndarray] = {
        name: np.isfinite(arr) for name, arr in feats
    }
    valid_mask = mask & np.all(np.isfinite(values), axis=2)
    metadata: dict[str, Any] = {
        "feature_count": len(names),
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
