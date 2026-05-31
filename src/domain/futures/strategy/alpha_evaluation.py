from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import scipy.stats
from numpy.typing import NDArray

from src.domain.futures.strategy.diagnostics import (
    _fast_rank1d,
    ic_lcb_hac,
    ic_summary,
    rolling_ic,
    top_bottom_spread_bps,
)

_logger = logging.getLogger(__name__)


def _measured_sigma_r_bps(
    values_2d: NDArray[np.float64],
    *,
    fallback_bps: float = 1.0,
) -> float:
    """Measure cross-sectional sigma_r in bps from finite panel values."""
    finite_vals: NDArray[np.float64] = values_2d[np.isfinite(values_2d)]
    if finite_vals.size == 0:
        return fallback_bps
    measured = float(np.nanstd(finite_vals)) * 1e4
    return measured if measured > 0.0 else fallback_bps


@dataclass(slots=True, frozen=True)
class AlphaEvaluationReport:
    """Comprehensive alpha quality report for Phase 0 evaluation gate.

    Attributes:
        net_ic: Spearman IC between predicted signed EV and realized gross fwd ret.
        net_icir: IC / IC_std (raw bar-level, not annualized).
        ic_t_stat_nw: Newey-West HAC corrected t-statistic (lag=horizon_bars).
        breakeven_ic: Cost floor / (sigma_r * sqrt(breadth_eff)).
        effective_breadth: Mean number of gate-passing symbols per bar.
        net_sharpe: Annualized Sharpe of net daily returns; NaN when not provided.
        quantile_coverage: Fraction of realized returns within [q10, q90].
        q50_sign_hit: Fraction where sign(q50) == sign(realized).
        per_regime_ic: Mean IC per regime {"bull", "bear", "chop"}.
        per_regime_breakeven: Breakeven IC per regime using regime-internal sigma_r
            and effective breadth {"bull", "bear", "chop"}.
        deflated_sharpe: DSR probability in [0, 1].
        cost_drag: PnL decomposition {"gross","fee","funding","slippage","net"};
            empty dict when backtest data not provided (deferred to Phase 1).
        passes: True when all PASS criteria are satisfied.
        fail_reasons: List of criterion names that failed.

    """

    net_ic: float
    net_icir: float
    ic_t_stat_nw: float
    breakeven_ic: float
    effective_breadth: float
    net_sharpe: float
    quantile_coverage: float
    q50_sign_hit: float
    per_regime_ic: dict[str, float]
    per_regime_breakeven: dict[str, float]
    deflated_sharpe: float
    cost_drag: dict[str, float]
    passes: bool
    fail_reasons: list[str]
    # Phase F: panel별 메트릭 분리 — {"inference": {...}, "trading": {...}}
    # inference panel: 학습에 사용된 전체 패널 (C1/C2)
    # trading panel: trading_mask=True 심볼만 (C3)
    metrics_by_panel: dict[str, dict[str, float]] = field(default_factory=dict)
    # Phase 1: gating IC = pre-clip signal IC (resid target).
    # NaN when inference_signed not provided.
    resid_ic: float = float("nan")
    # Phase 1: gating t-stat corresponding to resid_ic.
    resid_t_stat_nw: float = float("nan")
    # Phase 1: correlation-adjusted effective breadth N_eff = N/(1+(N-1)·rho_bar).
    n_eff: float = float("nan")
    # Phase 1: breakeven IC using N_eff and measured sigma_r.
    breakeven_ic_eff: float = float("nan")
    # Phase G1a: floor 전 corr-조정 N_eff (관측 보존용).
    n_eff_corr_raw: float = float("nan")
    # 진단용: emit breadth (게이트 판정에는 사용 안 함 — 대신 n_eff_universe 사용)
    n_eff_emit: float = float("nan")
    # Phase G2c: post-clip IC / pre-clip IC 비율. 클립 파괴 감지.
    clip_preservation_ratio: float = float("nan")
    # Robust rank skill and basket economics (24bps baseline).
    rank_ic_lcb: float = float("nan")
    basket_net_bps_lcb_24bps: float = float("nan")
    top_bottom_spread_bps: float = float("nan")
    turnover_proxy: float = float("nan")


def compute_net_ic(
    pred_2d: NDArray[np.float64],
    realized_fwd_2d: NDArray[np.float64],
    *,
    method: str = "spearman",
    horizon_bars: int = 6,
) -> dict[str, float]:
    """IC between predicted EV and realized forward returns.

    Applies Newey-West HAC correction with lag=horizon_bars for t-stat.
    Reuses diagnostics.rolling_ic and ic_summary.

    Args:
        pred_2d: Predicted signed EV [T, N].
        realized_fwd_2d: Realized forward gross log returns [T, N].
        method: Correlation method ("spearman" or "pearson").
        horizon_bars: Label horizon in bars, used as NW lag upper bound.

    Returns:
        Dict with keys: mean_ic, ic_std, icir, t_stat, t_stat_nw, hit_ratio, n_obs.

    Time complexity: O(T * N * log N) for spearman ranking.
    Space complexity: O(T) for ic_series.

    """
    ic_series: NDArray[np.float64] = rolling_ic(pred_2d, realized_fwd_2d, method=method)
    base: dict[str, float] = ic_summary(ic_series)

    valid: NDArray[np.float64] = ic_series[np.isfinite(ic_series)]
    n: int = len(valid)
    mean_ic: float = base["mean_ic"]

    if n < 2:
        return {**base, "t_stat_nw": 0.0}

    lag: int = min(horizon_bars, n // 4)
    demeaned: NDArray[np.float64] = valid - mean_ic

    # Newey-West HAC variance estimator
    s0: float = float(np.mean(demeaned**2))
    for j in range(1, lag + 1):
        cov_j: float = float(np.mean(demeaned[j:] * demeaned[:-j]))
        s0 += 2.0 * (1.0 - j / (lag + 1)) * cov_j

    se_nw: float = math.sqrt(max(s0, 1e-12) / n)
    t_nw: float = mean_ic / max(se_nw, 1e-12)

    return {**base, "t_stat_nw": t_nw}


def compute_breakeven_ic(
    *,
    cost_floor_bps: float,
    sigma_r_bps: float,
    breadth_eff: float,
) -> float:
    """Breakeven IC: minimum IC needed for positive expected edge.

    Derived from Fundamental Law: E[r|z] = IC * sigma_r * z > cost_floor
    => breakeven_IC = cost_floor_frac / (sigma_r_frac * sqrt(max(breadth, 1))).

    Args:
        cost_floor_bps: Total cost threshold in basis points.
        sigma_r_bps: Cross-sectional return volatility in basis points.
        breadth_eff: Effective number of independent bets per bar.

    Returns:
        Breakeven IC value as a float.

    Time complexity: O(1).
    Space complexity: O(1).

    """
    cost_floor_frac: float = cost_floor_bps / 1e4
    sigma_r_frac: float = sigma_r_bps / 1e4
    denominator: float = max(
        sigma_r_frac * math.sqrt(max(breadth_eff, 1.0)),
        1e-12,
    )
    return cost_floor_frac / denominator


def compute_effective_breadth(
    alpha_long_2d: NDArray[np.float64],
    alpha_short_2d: NDArray[np.float64],
    *,
    eps: float = 1e-12,
) -> float:
    """Mean number of symbols with non-zero alpha per bar.

    Args:
        alpha_long_2d: Long EV predictions [T, N].
        alpha_short_2d: Short EV predictions [T, N].
        eps: Threshold below which alpha is considered zero.

    Returns:
        Mean active symbol count per bar as float (0.0 if no valid bars).

    Time complexity: O(T * N).
    Space complexity: O(T).

    """
    # [T, N] bool — active if either side has non-trivial alpha
    active: NDArray[np.bool_] = (np.abs(alpha_long_2d) > eps) | (np.abs(alpha_short_2d) > eps)
    per_bar: NDArray[np.float64] = active.sum(axis=1).astype(np.float64)  # [T]

    # Keep only bars with at least one finite long alpha value
    valid_row_mask: NDArray[np.bool_] = np.any(np.isfinite(alpha_long_2d), axis=1)
    finite_rows: NDArray[np.float64] = per_bar[valid_row_mask]

    return float(np.mean(finite_rows)) if finite_rows.size > 0 else 0.0


def effective_breadth_corr(
    panel_2d: NDArray[np.float64],
    *,
    min_cofinite: int = 5,
) -> float:
    """Correlation-adjusted effective breadth ``N_eff = N / (1 + (N-1)·rho_bar)``.

    ``rho_bar`` is the mean off-diagonal pairwise Spearman correlation of the
    cross-sectional columns (assets), estimated over bars where both columns are
    finite (re-ranked within the overlap). It quantifies the diversification
    haircut from asset comovement — e.g. the common BTC factor in crypto — that a
    raw co-finite count ignores. Independent assets give ``N_eff ≈ N``; perfectly
    correlated assets give ``N_eff ≈ 1``.

    Args:
        panel_2d: Realized return or signal panel [T, N].
        min_cofinite: Minimum overlapping finite observations to count a pair.

    Returns:
        Effective breadth in [1.0, N]. Returns N (no haircut) when correlation
        cannot be estimated; callers must mask uninformative bars so this fallback
        does not silently lower the breakeven.

    Time complexity: O(N^2 * T * log T). Space complexity: O(T).

    """
    arr: NDArray[np.float64] = np.asarray(panel_2d, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return float(max(arr.shape[1], 1)) if arr.ndim == 2 else 1.0

    n_cols: int = arr.shape[1]
    corrs: list[float] = []
    for i in range(n_cols):
        col_i: NDArray[np.float64] = arr[:, i]
        for j in range(i + 1, n_cols):
            col_j: NDArray[np.float64] = arr[:, j]
            both: NDArray[np.bool_] = np.isfinite(col_i) & np.isfinite(col_j)
            if int(both.sum()) < min_cofinite:
                continue
            # Spearman == Pearson on ranks computed within the overlapping subset.
            rank_i: NDArray[np.float64] = _fast_rank1d(col_i[both].astype(np.float64))
            rank_j: NDArray[np.float64] = _fast_rank1d(col_j[both].astype(np.float64))
            if np.std(rank_i) < 1e-12 or np.std(rank_j) < 1e-12:
                continue
            c: float = float(np.corrcoef(rank_i, rank_j)[0, 1])
            if np.isfinite(c):
                corrs.append(c)

    if not corrs:
        return float(n_cols)

    # Negative mean correlation implies extra diversification; clamp to 0 so the
    # haircut never inflates breadth beyond the independent-bet count N.
    rho_bar: float = max(float(np.mean(corrs)), 0.0)
    n_eff: float = n_cols / (1.0 + (n_cols - 1) * rho_bar)
    return float(min(max(n_eff, 1.0), float(n_cols)))


def compute_quantile_coverage(
    q10_2d: NDArray[np.float64],
    q90_2d: NDArray[np.float64],
    realized_2d: NDArray[np.float64],
) -> float:
    """Fraction of realized returns that fall within [q10, q90] interval.

    Args:
        q10_2d: 10th percentile predictions [T, N].
        q90_2d: 90th percentile predictions [T, N].
        realized_2d: Realized forward returns [T, N].

    Returns:
        Coverage fraction in [0, 1], or 0.0 if no valid elements.

    Time complexity: O(T * N).
    Space complexity: O(T * N) for mask.

    """
    mask: NDArray[np.bool_] = (
        np.isfinite(q10_2d) & np.isfinite(q90_2d) & np.isfinite(realized_2d)
    )
    if not np.any(mask):
        return 0.0
    inside: NDArray[np.bool_] = (realized_2d[mask] >= q10_2d[mask]) & (
        realized_2d[mask] <= q90_2d[mask]
    )
    return float(np.mean(inside))


def compute_q50_sign_hit(
    q50_2d: NDArray[np.float64],
    realized_2d: NDArray[np.float64],
) -> float:
    """Fraction where sign(q50) == sign(realized).

    Args:
        q50_2d: Median (50th percentile) predictions [T, N].
        realized_2d: Realized forward returns [T, N].

    Returns:
        Sign hit rate in [0, 1], or 0.0 if no valid non-zero elements.

    Time complexity: O(T * N).
    Space complexity: O(T * N) for mask.

    """
    mask: NDArray[np.bool_] = (
        np.isfinite(q50_2d)
        & np.isfinite(realized_2d)
        & (np.abs(q50_2d) > 1e-12)
        & (np.abs(realized_2d) > 1e-12)
    )
    if not mask.any():
        return 0.0
    return float(np.mean(np.sign(q50_2d[mask]) == np.sign(realized_2d[mask])))


def _compute_regime_labels(
    btc_close_1d: NDArray[np.float64],
    *,
    trend_window: int = 30,
    vol_window: int = 20,
    vol_high_pct: float = 0.70,
) -> list[str | None]:
    """Classify each bar as bull/bear/chop using trailing BTC trend + vol. No look-ahead.

    Optimized using vectorized pre-computation of rolling standard deviations.
    """
    t_len: int = btc_close_1d.shape[0]
    labels: list[str | None] = [None] * t_len
    if t_len <= trend_window:
        return labels

    # Log returns for volatility computation; shape [T-1]
    log_ret_1d: NDArray[np.float64] = np.log(
        np.maximum(btc_close_1d[1:], 1e-12) / np.maximum(btc_close_1d[:-1], 1e-12)
    )

    rolling_vols = np.zeros(t_len, dtype=np.float64)
    
    # Pre-compute rolling standard deviations efficiently using pandas
    import pandas as pd
    series_ret = pd.Series(log_ret_1d)
    rolling_vols_slice = (
        series_ret.rolling(window=vol_window, min_periods=2)
        .std(ddof=0)
        .fillna(0.0)
        .to_numpy()
    )
    
    # Align indices: rolling_vols[s] represents standard deviation of log_ret_1d[s - vol_window : s]
    for s in range(vol_window, t_len):
        rolling_vols[s] = rolling_vols_slice[s - 1]

    for t in range(trend_window, t_len):
        trailing_ret: float = float(
            np.log(
                max(btc_close_1d[t], 1e-12) / max(btc_close_1d[t - trend_window], 1e-12)
            )
        )
        trailing_vol = rolling_vols[t]

        if trailing_ret < 0.0:
            labels[t] = "bear"
        else:
            # Vectorized percentile computation utilizing the pre-calculated slice
            hist_vols = rolling_vols[trend_window : t + 1]
            vol_threshold: float = float(np.percentile(hist_vols, vol_high_pct * 100.0))
            labels[t] = "chop" if trailing_vol > vol_threshold else "bull"

    return labels


def compute_per_regime_ic(
    pred_2d: NDArray[np.float64],
    realized_fwd_2d: NDArray[np.float64],
    btc_close_1d: NDArray[np.float64],
    *,
    trend_window: int = 30,
    vol_window: int = 20,
    vol_high_pct: float = 0.70,
) -> dict[str, float]:
    """Per-regime IC using BTC trailing trend+vol bucket (no look-ahead).

    Regime classification per bar t using only [t-trend_window..t] data:
      bull: trailing_ret > 0 AND trailing_vol <= vol_high_pct percentile
      bear: trailing_ret < 0
      chop: trailing_ret > 0 AND trailing_vol > vol_high_pct percentile

    Args:
        pred_2d: Predicted signed EV [T, N].
        realized_fwd_2d: Realized forward returns [T, N].
        btc_close_1d: BTC close prices [T], aligned with pred_2d rows.
        trend_window: Lookback bars for trailing return computation.
        vol_window: Lookback bars for rolling volatility computation.
        vol_high_pct: Percentile threshold separating chop from bull.

    Returns:
        Dict {"bull", "bear", "chop"} -> mean IC (float("nan") if < 5 bars).

    Time complexity: O(T^2) for regime labels + O(T*N*logN) for IC.
    Space complexity: O(T) for regime arrays.

    """
    regime_labels: list[str | None] = _compute_regime_labels(
        btc_close_1d,
        trend_window=trend_window,
        vol_window=vol_window,
        vol_high_pct=vol_high_pct,
    )

    ic_series: NDArray[np.float64] = rolling_ic(pred_2d, realized_fwd_2d, method="spearman")

    result: dict[str, float] = {}
    for regime in ("bull", "bear", "chop"):
        indices: list[int] = [t for t, lbl in enumerate(regime_labels) if lbl == regime]
        if len(indices) < 5:
            result[regime] = float("nan")
        else:
            regime_ic: NDArray[np.float64] = ic_series[np.array(indices, dtype=np.intp)]
            valid_regime_ic: NDArray[np.float64] = regime_ic[np.isfinite(regime_ic)]
            result[regime] = (
                float(np.mean(valid_regime_ic)) if valid_regime_ic.size > 0 else float("nan")
            )

    return result


def compute_per_regime_breakeven(
    alpha_long_2d: NDArray[np.float64],
    alpha_short_2d: NDArray[np.float64],
    realized_fwd_2d: NDArray[np.float64],
    btc_close_1d: NDArray[np.float64],
    *,
    cost_floor_bps: float,
    trend_window: int = 30,
    vol_window: int = 20,
    vol_high_pct: float = 0.70,
) -> dict[str, float]:
    """Breakeven IC per regime, using regime-internal sigma_r and effective breadth.

    For each regime r:
      - rows = bar indices labeled r (via _compute_regime_labels)
      - sigma_r_bps = nanstd(realized_fwd_2d[rows]) * 1e4
      - breadth_r = compute_effective_breadth(alpha_long_2d[rows], alpha_short_2d[rows])
      - breakeven[r] = compute_breakeven_ic(cost_floor_bps, sigma_r_bps, breadth_r)

    Args:
        alpha_long_2d: Long EV predictions [T, N].
        alpha_short_2d: Short EV predictions [T, N].
        realized_fwd_2d: Realized forward gross log returns [T, N].
        btc_close_1d: BTC close prices [T].
        cost_floor_bps: Total cost threshold in basis points.
        trend_window: Lookback bars for trailing return computation.
        vol_window: Lookback bars for rolling volatility computation.
        vol_high_pct: Percentile threshold separating chop from bull.

    Returns:
        Dict {"bull", "bear", "chop"} -> breakeven IC (float("nan") if < 5 bars).

    Time complexity: O(T^2) for regime labels + O(T*N) for breadth per regime.
    Space complexity: O(T) for label and row index lists.

    """
    labels: list[str | None] = _compute_regime_labels(
        btc_close_1d,
        trend_window=trend_window,
        vol_window=vol_window,
        vol_high_pct=vol_high_pct,
    )

    result: dict[str, float] = {}
    for regime in ("bull", "bear", "chop"):
        rows: list[int] = [t for t, lbl in enumerate(labels) if lbl == regime]
        if len(rows) < 5:
            result[regime] = float("nan")
            continue

        row_idx: NDArray[np.intp] = np.array(rows, dtype=np.intp)
        realized_slice: NDArray[np.float64] = realized_fwd_2d[row_idx]
        sigma_r_bps: float = _measured_sigma_r_bps(realized_slice)
        breadth_r: float = compute_effective_breadth(
            alpha_long_2d[row_idx], alpha_short_2d[row_idx]
        )
        result[regime] = compute_breakeven_ic(
            cost_floor_bps=cost_floor_bps,
            sigma_r_bps=sigma_r_bps,
            breadth_eff=breadth_r,
        )

    return result


def compute_deflated_sharpe(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio — P(observed SR is not spurious).

    Uses Bailey & Lopez de Prado (2014) approximation.

    Args:
        observed_sharpe: Sample Sharpe ratio (per-bar units, not annualized).
        n_trials: Number of hyperparameter trials (independent strategy tests).
        n_obs: Number of observations in the IC series.
        skew: Sample skewness of IC series.
        kurt: Sample excess kurtosis of IC series (NOT Fisher; 3.0 = normal).

    Returns:
        Probability in [0, 1] that the strategy is not spurious.

    Time complexity: O(1).
    Space complexity: O(1).

    """
    gamma: float = 0.5772156649  # Euler-Mascheroni constant
    safe_n_trials: int = max(n_trials, 2)

    z_ntrial: float = float(scipy.stats.norm.ppf(1.0 - 1.0 / safe_n_trials))
    e_max: float = (1.0 - gamma) * z_ntrial + gamma * float(
        scipy.stats.norm.ppf(1.0 - 1.0 / (safe_n_trials * math.e))
    )
    e_max = max(e_max, 0.0)

    # Non-normality adjustment from skewness and excess kurtosis
    non_normality_adj: float = math.sqrt(
        max(
            1.0 - skew / 6.0 * observed_sharpe + (kurt - 3.0) / 24.0 * observed_sharpe**2,
            0.0,
        )
    )
    non_normality_adj = max(non_normality_adj, 0.01)

    sr_star: float = e_max / math.sqrt(max(n_obs - 1, 1))
    numerator: float = (
        (observed_sharpe - sr_star) * math.sqrt(max(n_obs - 1, 1)) * non_normality_adj
    )
    return float(scipy.stats.norm.cdf(numerator))


def evaluate_alpha(
    *,
    alpha_long_2d: NDArray[np.float64],
    alpha_short_2d: NDArray[np.float64],
    realized_fwd_ret_2d: NDArray[np.float64],
    inference_signed_2d: NDArray[np.float64] | None = None,
    q10_2d: NDArray[np.float64] | None = None,
    q90_2d: NDArray[np.float64] | None = None,
    q50_2d: NDArray[np.float64] | None = None,
    btc_close_1d: NDArray[np.float64] | None = None,
    net_daily_returns: NDArray[np.float64] | None = None,
    cost_floor_bps: float = 24.0,
    n_trials: int = 1,
    horizon_bars: int = 6,
    basket_quantile: float = 0.35,
    # Phase F: trading_mask — True인 심볼(C3)만 trading panel 계산에 사용
    trading_mask: NDArray[np.bool_] | None = None,
) -> AlphaEvaluationReport:
    """Compute all alpha quality metrics in one call.

    Args:
        alpha_long_2d: Predicted long EV [T, N] (signed, in return units).
        alpha_short_2d: Predicted short EV [T, N].
        realized_fwd_ret_2d: Realized forward gross log returns [T, N].
        inference_signed_2d: Pre-clip continuous signed signal [T, N], optional.
            When provided, an ``inference_stat`` panel is added to
            ``metrics_by_panel`` reflecting ranking skill independent of the
            cost-threshold clipping.  The top-level report fields remain based
            on the clipped alpha to avoid breaking downstream consumers.
        q10_2d: 10th percentile predictions [T, N], optional.
        q90_2d: 90th percentile predictions [T, N], optional.
        q50_2d: Median predictions [T, N], optional.
        btc_close_1d: BTC close price series [T], optional for regime IC.
        net_daily_returns: Post-cost daily portfolio returns [D], optional.
            When provided, net_sharpe is computed. Otherwise NaN.
        cost_floor_bps: Total cost threshold (round-trip + hurdle), bps.
        n_trials: Number of hyperparameter trials (for DSR).
        horizon_bars: Label horizon in bars (for NW t-stat lag).
        basket_quantile: Top/bottom quantile used for basket spread diagnostics.
        trading_mask: Boolean mask [N] — True인 심볼(C3)만 trading panel 메트릭 계산에 사용.
            None이면 trading panel 계산 생략.

    Returns:
        AlphaEvaluationReport with all metrics and PASS/FAIL verdict.

    Time complexity: O(T * N * log N) dominated by rolling_ic.
    Space complexity: O(T * N).

    """
    # Signed net alpha: long minus short
    pred_2d: NDArray[np.float64] = alpha_long_2d - alpha_short_2d

    import time
    t_start = time.perf_counter()
    net_ic_dict: dict[str, float] = compute_net_ic(
        pred_2d, realized_fwd_ret_2d, horizon_bars=horizon_bars
    )
    _logger.debug("[micro-latency] compute_net_ic: %.4fs", time.perf_counter() - t_start)

    t_step = time.perf_counter()
    breadth: float = compute_effective_breadth(alpha_long_2d, alpha_short_2d)

    # Cross-sectional return volatility for breakeven IC
    sigma_r_bps: float = _measured_sigma_r_bps(realized_fwd_ret_2d)

    breakeven: float = compute_breakeven_ic(
        cost_floor_bps=cost_floor_bps,
        sigma_r_bps=sigma_r_bps,
        breadth_eff=breadth,
    )

    quantile_cov: float = (
        compute_quantile_coverage(q10_2d, q90_2d, realized_fwd_ret_2d)
        if q10_2d is not None and q90_2d is not None
        else float("nan")
    )

    q50_sign: float = (
        compute_q50_sign_hit(q50_2d, realized_fwd_ret_2d)
        if q50_2d is not None
        else float("nan")
    )
    _logger.debug(
        "[micro-latency] basic metrics (breadth, cov, sign): %.4fs",
        time.perf_counter() - t_step,
    )

    t_step = time.perf_counter()
    regime_labels = None
    if btc_close_1d is not None:
        regime_labels = _compute_regime_labels(btc_close_1d, trend_window=30)

    regime_ic: dict[str, float] = {
        "bull": float("nan"),
        "bear": float("nan"),
        "chop": float("nan"),
    }
    _gating_pred_2d: NDArray[np.float64] = (
        inference_signed_2d if inference_signed_2d is not None else pred_2d
    )
    if regime_labels is not None:
        ic_series = rolling_ic(_gating_pred_2d, realized_fwd_ret_2d, method="spearman")
        for regime in ("bull", "bear", "chop"):
            indices = [t for t, lbl in enumerate(regime_labels) if lbl == regime]
            if len(indices) < 5:
                regime_ic[regime] = float("nan")
            else:
                regime_ic_vals = ic_series[np.array(indices, dtype=np.intp)]
                valid_regime_ic = regime_ic_vals[np.isfinite(regime_ic_vals)]
                regime_ic[regime] = (
                    float(np.mean(valid_regime_ic))
                    if valid_regime_ic.size > 0
                    else float("nan")
                )

    regime_breakeven: dict[str, float] = {
        "bull": float("nan"),
        "bear": float("nan"),
        "chop": float("nan"),
    }
    if regime_labels is not None:
        for regime in ("bull", "bear", "chop"):
            rows = [t for t, lbl in enumerate(regime_labels) if lbl == regime]
            if len(rows) < 5:
                regime_breakeven[regime] = float("nan")
                continue
            row_idx = np.array(rows, dtype=np.intp)
            realized_slice = realized_fwd_ret_2d[row_idx]
            sigma_r_bps_reg = _measured_sigma_r_bps(realized_slice, fallback_bps=sigma_r_bps)
            breadth_r = compute_effective_breadth(alpha_long_2d[row_idx], alpha_short_2d[row_idx])
            regime_breakeven[regime] = compute_breakeven_ic(
                cost_floor_bps=cost_floor_bps,
                sigma_r_bps=sigma_r_bps_reg,
                breadth_eff=breadth_r,
            )
    _logger.debug("[micro-latency] regime metrics: %.4fs", time.perf_counter() - t_step)

    t_step = time.perf_counter()
    _logger.debug(
        "per_regime: ic=%s breakeven=%s",
        {r: f"{v:.4f}" for r, v in regime_ic.items()},
        {r: f"{v:.4f}" for r, v in regime_breakeven.items()},
    )

    # DSR computation
    ic_arr: NDArray[np.float64] = rolling_ic(pred_2d, realized_fwd_ret_2d)  # post-clip tradeable
    valid_ic: NDArray[np.float64] = ic_arr[np.isfinite(ic_arr)]

    if valid_ic.size > 1:
        ic_std_ddof1: float = float(np.std(valid_ic, ddof=1))
        observed_sharpe: float = float(np.mean(valid_ic)) / max(ic_std_ddof1, 1e-12)
    else:
        observed_sharpe = 0.0

    skew_v: float = float(scipy.stats.skew(valid_ic)) if valid_ic.size > 3 else 0.0
    kurt_v: float = (
        float(scipy.stats.kurtosis(valid_ic, fisher=False)) if valid_ic.size > 3 else 3.0
    )

    dsr: float = compute_deflated_sharpe(
        observed_sharpe,
        n_trials=n_trials,
        n_obs=max(len(valid_ic), 2),
        skew=skew_v,
        kurt=kurt_v,
    )
    _logger.debug("[micro-latency] deflated_sharpe: %.4fs", time.perf_counter() - t_step)

    # net_sharpe from post-cost daily returns (optional)
    net_sharpe: float
    if net_daily_returns is not None and len(net_daily_returns) > 1:
        rets = np.asarray(net_daily_returns, dtype=np.float64)
        rets = rets[np.isfinite(rets)]
        if len(rets) > 1:
            daily_std = float(np.std(rets, ddof=1))
            net_sharpe = float(np.mean(rets)) / max(daily_std, 1e-12) * math.sqrt(252.0)
        else:
            net_sharpe = float("nan")
    else:
        net_sharpe = float("nan")

    # cost_drag deferred (requires backtest PnL decomposition)
    cost_drag: dict[str, float] = {}

    # Phase 1: pre-clip signal IC (C3 fix) — inference_signed_2d is the unclipped dense signal.
    # Residualization (C1) is handled at the call site (realized_fwd_ret_2d should be resid).
    _resid_ic_dict: dict[str, float] | None = None
    _infer_ic_dict: dict[str, float] | None = None
    _infer_breadth: float = float("nan")
    if inference_signed_2d is not None:
        _infer_ic_dict = compute_net_ic(
            inference_signed_2d, realized_fwd_ret_2d, horizon_bars=horizon_bars
        )
        _resid_ic_dict = _infer_ic_dict
        _infer_breadth = float(
            np.mean(
                np.sum(
                    np.isfinite(inference_signed_2d) & (inference_signed_2d != 0.0),
                    axis=1,
                )
            )
        )

    # Gating IC: pre-clip when available (C3), else clipped net IC (backward compat).
    _gating_ic: float = (
        _resid_ic_dict["mean_ic"] if _resid_ic_dict is not None else net_ic_dict["mean_ic"]
    )
    _gating_t_nw: float = (
        _resid_ic_dict["t_stat_nw"] if _resid_ic_dict is not None else net_ic_dict["t_stat_nw"]
    )

    # IC skill gate: universe-level N_eff (emit breadth cap 제거).
    # G1은 신호 스킬을 측정 → sizing policy(emit breadth)에 독립적이어야 한다.
    # G2b(basket simulation)이 실제 emit P&L을 별도로 검증한다.
    _n_eff_corr_raw: float = effective_breadth_corr(realized_fwd_ret_2d)
    _n_eff_universe: float = float(max(_n_eff_corr_raw, 1.0))
    _breakeven_eff: float = compute_breakeven_ic(
        cost_floor_bps=cost_floor_bps,
        sigma_r_bps=sigma_r_bps,
        breadth_eff=_n_eff_universe,
    )
    _n_eff_emit: float = float(max(breadth, 1.0))  # 진단용 (게이트 판정 불사용)

    # Basket economics at fixed 24bps-style cost baseline.
    _spread_diag = top_bottom_spread_bps(
        score_2d=_gating_pred_2d,
        realized_2d=realized_fwd_ret_2d,
        eligible_2d=(
            np.isfinite(_gating_pred_2d)
            & np.isfinite(realized_fwd_ret_2d)
            & (np.isfinite(alpha_long_2d) | np.isfinite(alpha_short_2d))
        ),
        quantile=basket_quantile,
        cost_bps=cost_floor_bps,
    )
    _rank_ic_lcb = ic_lcb_hac(
        rolling_ic(_gating_pred_2d, realized_fwd_ret_2d, method="spearman"),
        horizon_bars=horizon_bars,
        z=1.0,
    )

    # PASS / FAIL verdict (Phase 1 criteria)
    fail_reasons: list[str] = []

    # G1a: IC LCB skill — universe breakeven 기준.
    if _rank_ic_lcb < _breakeven_eff:
        fail_reasons.append("signal_below_effective_breakeven")
    # G1b: statistical significance
    if _gating_t_nw < 3.0:
        fail_reasons.append("signal_t_stat_too_low")
    # G2: post-cost basket robustness
    if float(_spread_diag.get("net_spread_lcb_bps", -1e9)) <= 0.0:
        fail_reasons.append("basket_net_lcb_non_positive")
    # G1c: bear-regime IC non-negative (하락장에서 손실 불가 조건)
    _bear_ic: float = regime_ic.get("bear", float("nan"))
    if math.isfinite(_bear_ic) and _bear_ic < 0.0:
        fail_reasons.append("bear_regime_ic_negative")
    if not math.isnan(quantile_cov) and not (0.72 <= quantile_cov <= 0.88):
        fail_reasons.append("quantile_coverage_out_of_range")
    if dsr < 0.95:
        fail_reasons.append("deflated_sharpe_too_low")

    passes: bool = len(fail_reasons) == 0

    _logger.debug(
        "evaluate_alpha: gating_ic=%.4f n_eff=%.1f n_eff_emit=%.1f"
        " be_eff=%.4f lcb=%.4f basket_lcb=%.2fbps dsr=%.3f passes=%s fail=%s",
        _gating_ic,
        _n_eff_universe,
        _n_eff_emit,
        _breakeven_eff,
        _rank_ic_lcb,
        float(_spread_diag.get("net_spread_lcb_bps", float("nan"))),
        dsr,
        passes,
        fail_reasons,
    )

    # Phase F: dual panel 메트릭 분리 — inference(전체) vs trading(C3 마스크만)
    # Time complexity: O(T * N * log N). Space complexity: O(T * N).
    _metrics_by_panel: dict[str, dict[str, float]] = {
        "inference": {
            "net_ic": net_ic_dict["mean_ic"],
            "icir": net_ic_dict["icir"],
            "ic_t_stat_nw": net_ic_dict["t_stat_nw"],
            "effective_breadth": breadth,
        }
    }
    if trading_mask is not None and trading_mask.any():
        _pred_trading = pred_2d[:, trading_mask]
        _realized_trading = realized_fwd_ret_2d[:, trading_mask]
        _trading_ic_dict = compute_net_ic(
            _pred_trading, _realized_trading, horizon_bars=horizon_bars
        )
        _trading_breadth = compute_effective_breadth(
            alpha_long_2d[:, trading_mask], alpha_short_2d[:, trading_mask]
        )
        _metrics_by_panel["trading"] = {
            "net_ic": _trading_ic_dict["mean_ic"],
            "icir": _trading_ic_dict["icir"],
            "ic_t_stat_nw": _trading_ic_dict["t_stat_nw"],
            "effective_breadth": _trading_breadth,
        }

    # inference_stat panel: pre-clip signed signal (already computed above when provided).
    if _infer_ic_dict is not None:
        _metrics_by_panel["inference_stat"] = {
            "net_ic": _infer_ic_dict["mean_ic"],
            "icir": _infer_ic_dict["icir"],
            "ic_t_stat_nw": _infer_ic_dict["t_stat_nw"],
            "effective_breadth": _infer_breadth,
        }

    return AlphaEvaluationReport(
        net_ic=net_ic_dict["mean_ic"],
        net_icir=net_ic_dict["icir"],
        ic_t_stat_nw=net_ic_dict["t_stat_nw"],
        breakeven_ic=breakeven,
        effective_breadth=breadth,
        net_sharpe=net_sharpe,
        quantile_coverage=quantile_cov,
        q50_sign_hit=q50_sign,
        per_regime_ic=regime_ic,
        per_regime_breakeven=regime_breakeven,
        deflated_sharpe=dsr,
        cost_drag=cost_drag,
        passes=passes,
        fail_reasons=fail_reasons,
        metrics_by_panel=_metrics_by_panel,
        resid_ic=_gating_ic,
        resid_t_stat_nw=_gating_t_nw,
        n_eff=_n_eff_universe,
        breakeven_ic_eff=_breakeven_eff,
        n_eff_corr_raw=_n_eff_corr_raw,
        n_eff_emit=_n_eff_emit,
        rank_ic_lcb=float(_rank_ic_lcb),
        basket_net_bps_lcb_24bps=float(_spread_diag.get("net_spread_lcb_bps", float("nan"))),
        top_bottom_spread_bps=float(_spread_diag.get("gross_spread_bps", float("nan"))),
        turnover_proxy=float(_spread_diag.get("turnover_proxy", float("nan"))),
        clip_preservation_ratio=(
            # post_clip_IC / pre_clip_IC: 클립 후 스킬이 얼마나 보존됐는가 (Spec G2c).
            # net_ic_dict["mean_ic"] = post-clip (al-as), _gating_ic = pre-clip dense.
            float(net_ic_dict["mean_ic"] / _gating_ic)
            if abs(_gating_ic) > 1e-9
            else float("nan")
        ),
    )


def sweep_horizon_breakeven(
    realized_fwd_ret_map: dict[int, NDArray[np.float64]],
    alpha_long_map: dict[int, NDArray[np.float64]],
    alpha_short_map: dict[int, NDArray[np.float64]],
    *,
    cost_floor_bps: float = 24.0,
    cost_map: dict[int, float] | None = None,
) -> dict[int, dict[str, float]]:
    """Scan horizon candidates to find optimal cost/signal ratio.

    Pure function — accepts pre-computed realized return arrays keyed by
    horizon in bars, avoiding direct dependency on aligned market data.

    Args:
        realized_fwd_ret_map: {horizon_bars: realized_2d [T, N]} — gross fwd returns.
        alpha_long_map: {horizon_bars: alpha_long_2d [T, N]}.
        alpha_short_map: {horizon_bars: alpha_short_2d [T, N]}.
        cost_floor_bps: Default round-trip cost in bps (fallback when cost_map absent).
        cost_map: Optional per-horizon cost override {horizon: cost_bps}.
            Enables hold-period amortization (e.g., 24bps / (h // rebalance_bars)).
            When provided, overrides cost_floor_bps for the matching horizon.

    Returns:
        {horizon_bars: {"sigma_r_bps", "net_ic", "breakeven_ic", "breadth_eff",
                        "ic_exceeds_breakeven"}} for each horizon.

    Time complexity: O(H * T * N * log N) where H = number of horizons.

    """
    results: dict[int, dict[str, float]] = {}
    for horizon, realized_2d in realized_fwd_ret_map.items():
        alpha_long_2d = alpha_long_map.get(horizon)
        alpha_short_2d = alpha_short_map.get(horizon)
        if alpha_long_2d is None or alpha_short_2d is None:
            _logger.warning("[SWEEP] horizon=%d missing alpha maps — skipped", horizon)
            continue

        sigma_r_bps = _measured_sigma_r_bps(realized_2d)

        # per-horizon amortized cost: cost_map overrides cost_floor_bps for this horizon
        effective_cost = (
            float(cost_map[horizon]) if cost_map is not None and horizon in cost_map
            else cost_floor_bps
        )
        breadth = compute_effective_breadth(alpha_long_2d, alpha_short_2d)
        breakeven = compute_breakeven_ic(
            cost_floor_bps=effective_cost,
            sigma_r_bps=sigma_r_bps,
            breadth_eff=breadth,
        )

        pred_2d = alpha_long_2d - alpha_short_2d
        ic_dict = compute_net_ic(pred_2d, realized_2d, horizon_bars=horizon)

        results[horizon] = {
            "sigma_r_bps": sigma_r_bps,
            "net_ic": ic_dict["mean_ic"],
            "ic_t_stat_nw": ic_dict["t_stat_nw"],
            "breakeven_ic": breakeven,
            "breadth_eff": breadth,
            "ic_exceeds_breakeven": float(ic_dict["mean_ic"] > breakeven),
        }
        _logger.info(
            "[SWEEP] horizon=%d sigma_r=%.1fbps net_ic=%.4f breakeven=%.4f breadth=%.1f pass=%s",
            horizon,
            sigma_r_bps,
            ic_dict["mean_ic"],
            breakeven,
            breadth,
            ic_dict["mean_ic"] > breakeven,
        )

    return results


def diagnose_alpha_ic_decomposition(
    *,
    pred_dense_2d: NDArray[np.float64],
    realized_raw_2d: NDArray[np.float64],
    beta_2d: NDArray[np.float64] | None = None,
    market_fwd_1d: NDArray[np.float64] | None = None,
    trading_mask_1d: NDArray[np.bool_] | None = None,
    horizon_bars: int = 12,
) -> dict[str, float]:
    """Dense vs gated, raw vs residualized IC 분해 — acceptance 디버깅 전용.

    모든 IC는 관측만 하고 게이팅하지 않는다.

    Args:
        pred_dense_2d: Dense alpha signal [T, N] — EV hurdle/gate 미적용.
        realized_raw_2d: Raw forward log returns [T, N].
        beta_2d: Trailing OLS beta [T, N]; None이면 잔차화 생략.
        market_fwd_1d: Market forward return [T]; None이면 잔차화 생략.
        trading_mask_1d: C3 심볼 boolean mask [N]; None이면 C3 서브셋 생략.
        horizon_bars: NW t-stat lag.

    Returns:
        Dict with keys:
          dense_c1_raw_ic, dense_c1_raw_hit, dense_c1_raw_breadth,
          dense_c1_resid_ic, dense_c1_resid_hit,
          dense_c3_raw_ic, dense_c3_resid_ic.

    """
    result: dict[str, float] = {}

    # --- C1 dense, raw target ---
    ic_c1_raw = compute_net_ic(pred_dense_2d, realized_raw_2d, horizon_bars=horizon_bars)
    result["dense_c1_raw_ic"] = ic_c1_raw["mean_ic"]
    result["dense_c1_raw_hit"] = ic_c1_raw["hit_ratio"]
    result["dense_c1_raw_breadth"] = compute_effective_breadth(
        pred_dense_2d, np.zeros_like(pred_dense_2d)
    )

    # --- C1 dense, beta-residualized target ---
    realized_resid: NDArray[np.float64] | None = None
    if beta_2d is not None and market_fwd_1d is not None:
        realized_resid = realized_raw_2d - beta_2d * market_fwd_1d[:, np.newaxis]
        ic_c1_resid = compute_net_ic(pred_dense_2d, realized_resid, horizon_bars=horizon_bars)
        result["dense_c1_resid_ic"] = ic_c1_resid["mean_ic"]
        result["dense_c1_resid_hit"] = ic_c1_resid["hit_ratio"]
    else:
        result["dense_c1_resid_ic"] = float("nan")
        result["dense_c1_resid_hit"] = float("nan")

    # --- C3 subset dense, raw target ---
    if trading_mask_1d is not None and np.any(trading_mask_1d):
        pred_c3 = pred_dense_2d[:, trading_mask_1d]
        real_c3 = realized_raw_2d[:, trading_mask_1d]
        ic_c3_raw = compute_net_ic(pred_c3, real_c3, horizon_bars=horizon_bars)
        result["dense_c3_raw_ic"] = ic_c3_raw["mean_ic"]
        if realized_resid is not None:
            resid_c3 = realized_resid[:, trading_mask_1d]
            ic_c3_resid = compute_net_ic(pred_c3, resid_c3, horizon_bars=horizon_bars)
            result["dense_c3_resid_ic"] = ic_c3_resid["mean_ic"]
        else:
            result["dense_c3_resid_ic"] = float("nan")
    else:
        result["dense_c3_raw_ic"] = float("nan")
        result["dense_c3_resid_ic"] = float("nan")

    return result


def diagnose_selection_monotonicity(
    inference_signed_2d: NDArray[np.float64],
    realized_resid_2d: NDArray[np.float64],
    beta_2d: NDArray[np.float64] | None,
    *,
    n_deciles: int = 5,
    horizon_bars: int = 12,
) -> dict[str, float]:
    """Cross-sectional decile 단조성 + selection beta-tilt 분해.

    Args:
        inference_signed_2d: [T, N] dense NET rank signal (rank_score_long - short).
        realized_resid_2d: [T, N] beta-residualized forward returns.
        beta_2d: [T, N] trailing beta per (t, symbol). None이면 beta 지표 nan.
        n_deciles: 분위 수 (default 5 = quintile).
        horizon_bars: 확장용 보존 파라미터 (현재 미사용).

    Returns dict with:
        top_minus_bottom_bps: 최상-최하 decile 스프레드(bps). 양수=단조 방향 수익.
        monotonicity_spearman: decile_idx vs decile_mean_ret Spearman rho. 1=완전단조.
        decile_mean_ret_bps_{d}: 각 decile 평균 수익(bps), d=0최하 ~ d=k-1최상.
        long_decile_beta_mean: 최상 decile 평균 beta.
        short_decile_beta_mean: 최하 decile 평균 beta.
        beta_tilt: long_beta - short_beta. ≠0 → beta-loaded selection.
        n_obs: 유효 (t, s) 쌍 수.

    """
    t_size, _n_size = inference_signed_2d.shape

    decile_ret_sums = np.zeros(n_deciles, dtype=np.float64)
    decile_beta_sums = np.zeros(n_deciles, dtype=np.float64)
    decile_counts = np.zeros(n_deciles, dtype=np.int64)
    n_obs: int = 0

    for t in range(t_size):
        sig = inference_signed_2d[t]
        ret = realized_resid_2d[t]
        finite_mask = np.isfinite(sig) & np.isfinite(ret)
        if int(finite_mask.sum()) < n_deciles:
            continue

        sig_f = sig[finite_mask]
        ret_f = ret[finite_mask]
        beta_f: NDArray[np.float64] | None = None
        if beta_2d is not None:
            b_row = beta_2d[t][finite_mask]
            if np.isfinite(b_row).all():
                beta_f = b_row

        # fractional rank [0,1] → decile 인덱스 (0=최하, n_deciles-1=최상)
        # _fast_rank1d returns 1-based ranks (1..n); normalize to [0, 1]
        ranks = _fast_rank1d(sig_f)
        n_f = float(len(sig_f))
        ranks_norm = (ranks - 1.0) / max(n_f - 1.0, 1.0)
        decile_idx = np.clip(
            (ranks_norm * n_deciles).astype(np.int64), 0, n_deciles - 1
        )

        for d in range(n_deciles):
            mask_d = decile_idx == d
            if not mask_d.any():
                continue
            decile_ret_sums[d] += float(np.mean(ret_f[mask_d]))
            if beta_f is not None:
                decile_beta_sums[d] += float(np.mean(beta_f[mask_d]))
            decile_counts[d] += 1

        n_obs += int(finite_mask.sum())

    valid_counts = np.maximum(decile_counts, 1)
    decile_mean_ret = decile_ret_sums / valid_counts
    decile_mean_beta = decile_beta_sums / valid_counts

    decile_indices = np.arange(n_deciles, dtype=np.float64)
    if np.std(decile_mean_ret) > 1e-12:
        mono_rho = float(scipy.stats.spearmanr(decile_indices, decile_mean_ret).statistic)
    else:
        mono_rho = float("nan")

    top_idx = n_deciles - 1
    top_bps = float(decile_mean_ret[top_idx]) * 1e4
    bot_bps = float(decile_mean_ret[0]) * 1e4
    top_minus_bottom_bps = top_bps - bot_bps

    long_beta = float(decile_mean_beta[top_idx]) if beta_2d is not None else float("nan")
    short_beta = float(decile_mean_beta[0]) if beta_2d is not None else float("nan")
    beta_tilt = long_beta - short_beta if beta_2d is not None else float("nan")

    result: dict[str, float] = {
        "top_minus_bottom_bps": top_minus_bottom_bps,
        "monotonicity_spearman": mono_rho,
        "long_decile_beta_mean": long_beta,
        "short_decile_beta_mean": short_beta,
        "beta_tilt": beta_tilt,
        "n_obs": float(n_obs),
    }
    for d in range(n_deciles):
        result[f"decile_mean_ret_bps_{d}"] = float(decile_mean_ret[d]) * 1e4
    return result
