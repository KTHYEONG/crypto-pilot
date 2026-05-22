from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.config import RegimeConfig


def _sigmoid(x: np.ndarray | float, k: float = 10.0) -> np.ndarray | float:
    """Helper sigmoid function for smooth threshold gating."""
    return 1.0 / (1.0 + np.exp(-k * x))


def compute_regime_posterior(
    close_2d: np.ndarray,
    cfg: RegimeConfig,
) -> dict[str, np.ndarray]:
    """Computes deterministic 5-state soft posterior regime probabilities.

    The 5 states are:
    - hmm_prob_bull_calm
    - hmm_prob_bull_vol_up
    - hmm_prob_bear_trend
    - hmm_prob_chop
    - hmm_prob_crisis

    Args:
        close_2d: [T, N] prices array.
        cfg: RegimeConfig settings.

    Returns:
        Dictionary mapping state column name to [T] probability array.
        Chronological and look-ahead free.
    """
    if close_2d.ndim != 2:
        raise ValueError("close_2d must be 2D")

    t_len, n_syms = close_2d.shape
    if t_len == 0:
        empty = np.zeros(0, dtype=np.float64)
        return {
            "hmm_prob_bull_calm": empty,
            "hmm_prob_bull_vol_up": empty,
            "hmm_prob_bear_trend": empty,
            "hmm_prob_chop": empty,
            "hmm_prob_crisis": empty,
        }

    # 1. Market Basket = equally-weighted log returns (NaN robust)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.diff(np.log(np.clip(close_2d, 1e-12, None)), axis=0)  # [T-1, N]
    mkt = np.nanmean(ret, axis=1)  # [T-1]
    mkt = np.concatenate([[0.0], mkt])  # [T]

    # Reconstruct close basket price panel
    close_basket = np.exp(np.cumsum(mkt))

    # 2. Computes rolling statistics
    rv = np.zeros(t_len, dtype=np.float64)
    rv_pct = np.zeros(t_len, dtype=np.float64)
    ma_f = np.zeros(t_len, dtype=np.float64)
    ma_s = np.zeros(t_len, dtype=np.float64)
    dd = np.zeros(t_len, dtype=np.float64)
    corr = np.full(t_len, 0.2, dtype=np.float64)

    running_max = np.maximum.accumulate(close_basket)

    for t in range(t_len):
        # Volatility
        vol_start = max(0, t - cfg.vol_window + 1)
        vol_window_data = mkt[vol_start : t + 1]
        rv[t] = np.std(vol_window_data) if len(vol_window_data) > 1 else 0.0

        # Percentile rank (PIT)
        pit_start = max(0, t - 252 + 1)
        pit_sub = rv[pit_start : t + 1]
        val = rv[t]
        less = np.sum(pit_sub < val)
        ties = np.sum(pit_sub == val)
        rv_pct[t] = (less + 0.5 * ties) / len(pit_sub)

        # Drawdown
        dd[t] = close_basket[t] / running_max[t] - 1.0

        # Moving Averages for trend
        ma_f_start = max(0, t - cfg.trend_ma_fast + 1)
        ma_s_start = max(0, t - cfg.trend_ma_slow + 1)
        ma_f[t] = np.mean(close_basket[ma_f_start : t + 1])
        ma_s[t] = np.mean(close_basket[ma_s_start : t + 1])

        # Pairwise Correlation
        corr_start = max(0, t - cfg.vol_window + 1)
        sub_ret = ret[max(0, corr_start - 1) : t, :]  # align with ret index
        if len(sub_ret) >= 5:
            # Filter out constants
            valid_cols = []
            for col_idx in range(sub_ret.shape[1]):
                col_data = sub_ret[:, col_idx]
                if np.any(np.isfinite(col_data)) and not np.all(col_data == col_data[0]):
                    valid_cols.append(col_idx)

            if len(valid_cols) >= 2:
                sub_clean = sub_ret[:, valid_cols]
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr_mat = np.corrcoef(sub_clean, rowvar=False)
                if corr_mat.ndim == 2:
                    tri = corr_mat[np.triu_indices_from(corr_mat, k=1)]
                    mean_corr = np.nanmean(tri)
                    corr[t] = mean_corr if np.isfinite(mean_corr) else 0.2

    # Trend Strength Calculation
    trend = (ma_f - ma_s) / np.maximum(ma_s, 1e-12)

    # 3. Softposterior Score Mappings
    score_crisis = (
        0.33 * _sigmoid(rv_pct - cfg.vol_crisis_pct)
        + 0.33 * _sigmoid(-(dd - cfg.dd_crisis_thr))
        + 0.34 * _sigmoid(corr - cfg.corr_crisis_thr)
    )
    score_bear = _sigmoid(-(trend - (-cfg.trend_thr))) * (1.0 - score_crisis)
    score_bull = _sigmoid(trend - cfg.trend_thr)
    score_bullvol = score_bull * _sigmoid(rv_pct - cfg.vol_high_pct)
    score_bullcalm = score_bull * (1.0 - _sigmoid(rv_pct - cfg.vol_high_pct))
    score_chop = 1.0 - score_bull - _sigmoid(-(trend + cfg.trend_thr))

    # 4. Softmax / L1 Normalization
    scores = np.stack(
        [score_bullcalm, score_bullvol, score_bear, score_chop, score_crisis], axis=1
    )
    scores = np.maximum(scores, 0.0)  # non-negative bounds
    sums = np.sum(scores, axis=1, keepdims=True)
    probs = np.where(sums > 1e-12, scores / sums, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))

    # 5. EWMA Smoothing
    smoothed = np.zeros_like(probs)
    alpha = 2.0 / (cfg.smooth_ewma_bars + 1.0)
    smoothed[0] = probs[0]
    for t in range(1, t_len):
        smoothed[t] = alpha * probs[t] + (1.0 - alpha) * smoothed[t - 1]

    # Re-normalize smoothed posteriors to strictly sum to 1.0
    s_sums = np.sum(smoothed, axis=1, keepdims=True)
    smoothed = np.where(s_sums > 1e-12, smoothed / s_sums, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))

    # Ensure precise rounding/sum limits (Fail-fast check)
    row_sums = np.sum(smoothed, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise RuntimeError("regime posterior sum != 1.0 within tolerance")

    return {
        "hmm_prob_bull_calm": smoothed[:, 0],
        "hmm_prob_bull_vol_up": smoothed[:, 1],
        "hmm_prob_bear_trend": smoothed[:, 2],
        "hmm_prob_chop": smoothed[:, 3],
        "hmm_prob_crisis": smoothed[:, 4],
    }
