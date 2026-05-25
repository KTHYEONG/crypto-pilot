from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.contracts import LabelPanel

_BETA_WINDOW: int = 120
_BETA_MIN_PERIODS: int = 20


def _build_relevance(
    signed_ret: NDArray[np.float32],
    eligible: NDArray[np.bool_],
    min_group_size: int,
) -> NDArray[np.int32]:
    rel = np.zeros(signed_ret.shape, dtype=np.int32)
    for t in range(signed_ret.shape[0]):
        idx = np.flatnonzero(eligible[t] & np.isfinite(signed_ret[t]))
        if idx.size < min_group_size:
            continue
        vals = signed_ret[t, idx]
        q15 = float(np.nanpercentile(vals, 15))
        q35 = float(np.nanpercentile(vals, 35))
        q65 = float(np.nanpercentile(vals, 65))
        q85 = float(np.nanpercentile(vals, 85))
        rel[t, idx] = np.where(
            vals >= q85,
            4,
            np.where(vals >= q65, 3, np.where(vals >= q35, 2, np.where(vals >= q15, 1, 0))),
        )
    return np.asarray(rel, dtype=np.int32)


def _compute_trailing_beta(
    close_2d: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute per-symbol rolling OLS beta against cross-sectional market return.

    All computation uses only past bars — no look-ahead.
    beta_2d defaults to 1.0 when insufficient history exists.

    Args:
        close_2d: Close price array of shape [T, N].

    Returns:
        beta_2d: Rolling beta array of shape [T, N].
        Time complexity: O(T * N). Space complexity: O(T * N).

    """
    t_len, n_len = close_2d.shape
    # 1-bar spot log returns: shape [T, N]
    spot_ret: NDArray[np.float64] = np.full((t_len, n_len), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = close_2d[1:] / close_2d[:-1]
        log_r = np.where((ratio > 0) & np.isfinite(ratio), np.log(ratio), np.nan)
    spot_ret[1:] = log_r

    # Equal-weighted cross-sectional market return per bar: shape [T]
    with np.errstate(all="ignore"):
        market_ret: NDArray[np.float64] = np.nanmean(spot_ret, axis=1)

    # Rolling OLS beta per symbol (loop over N ~20-50 symbols, not T). Zero-Loop policy compliance.
    mkt_series: pd.Series = pd.Series(market_ret)
    beta_2d: NDArray[np.float64] = np.ones((t_len, n_len), dtype=np.float64)
    for col in range(n_len):
        sym_series = pd.Series(spot_ret[:, col])
        cov = sym_series.rolling(_BETA_WINDOW, min_periods=_BETA_MIN_PERIODS).cov(mkt_series)
        var = mkt_series.rolling(_BETA_WINDOW, min_periods=_BETA_MIN_PERIODS).var()
        beta = (cov / var.clip(lower=1e-12)).fillna(1.0).clip(-5.0, 5.0)
        beta_2d[:, col] = beta.to_numpy()

    return beta_2d


def build_label_panel(aligned: AlignedMarketData, cfg: StrategyMLConfig) -> LabelPanel:
    """Build t+1 execution aligned label tensors with beta-residualized returns.

    Uses trailing rolling OLS beta (look-ahead free) to remove cross-sectional
    market factor from gross log returns before training signal generation.
    Cost deduction is deferred to the objectives layer (canonical B1 fix).

    Args:
        aligned: Aligned market data tensors.
        cfg: Strategy ML configuration.

    Returns:
        LabelPanel with beta-residualized net returns and relevance scores.

    """
    t_len, n_len = aligned.close_2d.shape
    horizon = cfg.label_horizon_bars
    long_net = np.full((t_len, n_len), np.nan, dtype=np.float32)
    short_net = np.full((t_len, n_len), np.nan, dtype=np.float32)
    eligible = (
        aligned.active_mask
        & aligned.warm_mask
        & ~aligned.entry_block_mask
        & ~aligned.kill_mask
    )
    # Gross alpha: cost deducted once at objectives layer (B1 canonical fix).
    cost = np.float64(0.0)

    # --- Beta-residualization (B2) ---
    # beta_2d[t, i]: trailing OLS beta at bar t — pure past data, no look-ahead.
    beta_2d: NDArray[np.float64] = _compute_trailing_beta(aligned.close_2d)

    # Market forward return for the same label horizon (equal-weighted, vectorized).
    # Computed for bars t using [t+1 .. t+horizon] — identical indexing to gross_long.
    market_fwd_ret: NDArray[np.float64] = np.full(t_len, np.nan, dtype=np.float64)
    if t_len > horizon:
        entry_mkt = aligned.open_2d[1 : t_len - horizon + 1]  # shape [T-h, N]
        exit_mkt = aligned.close_2d[horizon:t_len]  # shape [T-h, N]
        valid_mkt = (
            (entry_mkt > 0.0)
            & (exit_mkt > 0.0)
            & np.isfinite(entry_mkt)
            & np.isfinite(exit_mkt)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            log_rets = np.where(valid_mkt, np.log(exit_mkt / entry_mkt), np.nan)
        market_fwd_ret[: t_len - horizon] = np.nanmean(log_rets, axis=1)

    for t in range(t_len - horizon):
        entry = aligned.open_2d[t + 1]
        exit_ = aligned.close_2d[t + horizon]
        valid_px = (entry > 0.0) & (exit_ > 0.0) & np.isfinite(entry) & np.isfinite(exit_)
        row_ok = eligible[t] & valid_px
        if not np.any(row_ok):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            gross_long = np.log(exit_ / entry)
            gross_short = np.log(entry / exit_)
        funding = aligned.funding_2d[t]
        mkt_ret_t = float(market_fwd_ret[t]) if np.isfinite(market_fwd_ret[t]) else 0.0
        beta_t: NDArray[np.float64] = beta_2d[t]  # shape [N]
        # Residualize: remove estimated market beta component
        residual_adj = beta_t * mkt_ret_t
        long_net[t, row_ok] = (
            gross_long[row_ok] - residual_adj[row_ok] - cost - funding[row_ok]
        ).astype(np.float32)
        short_net[t, row_ok] = (
            gross_short[row_ok] + residual_adj[row_ok] - cost + funding[row_ok]
        ).astype(np.float32)

    # Snapshot pre-CS-demean beta-residualized return for calibrator (absolute EV).
    # CS-demean collapses cross-sectional mean to 0, losing absolute level needed by calibrator.
    # exec_net_ret preserves the pre-demean signal so calibrator can target real execution EV.
    # Shape: [T, N], NaN where not computed. Time complexity: O(T*N) copy.
    exec_net_ret: NDArray[np.float32] = long_net.copy()

    # CS-demean: per-timestep subtract cross-sectional mean from long_net and short_net labels.
    # OLS beta residualization does not guarantee E_i[long_net[t,i]] = 0 when beta != 1
    # or when eligible is a proper subset of all symbols (e.g., differential funding rates).
    # Vectorized: [T-h, N] masked nanmean → subtract per-row to zero-center CS distribution.
    # short_net adds the same mean (≈ -long_net before demean -> 0 after demean),
    # preserving the long/short anti-symmetry contract.
    _n_t: int = t_len - horizon
    _cs_mask: NDArray[np.bool_] = np.isfinite(long_net[:_n_t]) & eligible[:_n_t]  # [T-h, N]
    _cs_count: NDArray[np.intp] = _cs_mask.sum(axis=1)  # [T-h]
    _masked_long: NDArray[np.float32] = np.where(_cs_mask, long_net[:_n_t], np.float32(np.nan))
    with np.errstate(all="ignore"):
        _row_mean: NDArray[np.float32] = np.nanmean(_masked_long, axis=1, keepdims=True).astype(
            np.float32
        )  # [T-h, 1]
    # Skip rows with < 2 valid symbols (effective mean = 0 → no-op).
    _effective_mean: NDArray[np.float32] = np.where(
        (_cs_count >= 2)[:, np.newaxis], _row_mean, np.float32(0.0)
    )
    long_net[:_n_t] = np.where(_cs_mask, long_net[:_n_t] - _effective_mean, long_net[:_n_t])
    short_net[:_n_t] = np.where(_cs_mask, short_net[:_n_t] + _effective_mean, short_net[:_n_t])

    signed = long_net.copy()
    finite_long = np.isfinite(long_net)
    rel = _build_relevance(
        signed_ret=signed,
        eligible=eligible & finite_long,
        min_group_size=cfg.min_group_size,
    )

    liq_weight = np.clip(np.log1p(np.maximum(aligned.volume_2d, 0.0)), 0.25, 2.0)
    valid_mask = eligible & finite_long
    original_weight = np.where(valid_mask, liq_weight, 0.0).astype(np.float32)
    y_ev_abs = np.where(valid_mask, np.abs(signed), 0.0).astype(np.float32)
    sample_weight = (original_weight * (1.0 + 2.0 * y_ev_abs)).astype(np.float32)
    return LabelPanel(
        long_net_ret=long_net,
        short_net_ret=short_net,
        signed_net_ret=signed,
        exec_net_ret=exec_net_ret,
        relevance=rel,
        sample_weight=sample_weight,
        eligible_mask=eligible & finite_long,
    )
