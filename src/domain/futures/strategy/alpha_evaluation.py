from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import scipy.stats
from numpy.typing import NDArray

from src.domain.futures.strategy.diagnostics import ic_summary, rolling_ic

_logger = logging.getLogger(__name__)


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

    Regime classification per bar t using only [t-trend_window..t] data:
      bear: trailing_ret < 0
      chop: trailing_ret >= 0 AND trailing_vol > vol_high_pct percentile of hist_vols
      bull: trailing_ret >= 0 AND trailing_vol <= vol_high_pct percentile of hist_vols

    Args:
        btc_close_1d: BTC close prices [T].
        trend_window: Lookback bars for trailing return computation.
        vol_window: Lookback bars for rolling volatility computation.
        vol_high_pct: Percentile threshold (0-1) separating chop from bull.

    Returns:
        List of length T with str labels or None for t < trend_window.

    Time complexity: O(T^2) for hist_vols accumulation.
    Space complexity: O(T) for label list.

    """
    t_len: int = btc_close_1d.shape[0]

    # Log returns for volatility computation; shape [T-1]
    log_ret_1d: NDArray[np.float64] = np.log(
        np.maximum(btc_close_1d[1:], 1e-12) / np.maximum(btc_close_1d[:-1], 1e-12)
    )

    labels: list[str | None] = [None] * t_len

    for t in range(trend_window, t_len):
        trailing_ret: float = float(
            np.log(
                max(btc_close_1d[t], 1e-12) / max(btc_close_1d[t - trend_window], 1e-12)
            )
        )
        # Rolling vol uses log returns up to bar t (log_ret_1d index t-1 corresponds to close[t])
        vol_start: int = max(0, t - vol_window)
        vol_slice: NDArray[np.float64] = log_ret_1d[vol_start:t]
        trailing_vol: float = float(np.std(vol_slice, ddof=0)) if vol_slice.size > 1 else 0.0

        if trailing_ret < 0.0:
            labels[t] = "bear"
        else:
            # Determine vol threshold from full available history up to t
            hist_vols: list[float] = []
            for s in range(trend_window, t + 1):
                v_start = max(0, s - vol_window)
                v_sl = log_ret_1d[v_start:s]
                hist_vols.append(float(np.std(v_sl, ddof=0)) if v_sl.size > 1 else 0.0)
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
        valid_vals: NDArray[np.float64] = realized_slice[np.isfinite(realized_slice)]
        sigma_r_bps: float = (
            float(np.nanstd(valid_vals)) * 1e4 if valid_vals.size > 0 else 400.0
        )
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
    q10_2d: NDArray[np.float64] | None = None,
    q90_2d: NDArray[np.float64] | None = None,
    q50_2d: NDArray[np.float64] | None = None,
    btc_close_1d: NDArray[np.float64] | None = None,
    net_daily_returns: NDArray[np.float64] | None = None,
    cost_floor_bps: float = 24.0,
    n_trials: int = 1,
    horizon_bars: int = 6,
    # Phase F: trading_mask — True인 심볼(C3)만 trading panel 계산에 사용
    trading_mask: NDArray[np.bool_] | None = None,
) -> AlphaEvaluationReport:
    """Compute all alpha quality metrics in one call.

    Args:
        alpha_long_2d: Predicted long EV [T, N] (signed, in return units).
        alpha_short_2d: Predicted short EV [T, N].
        realized_fwd_ret_2d: Realized forward gross log returns [T, N].
        q10_2d: 10th percentile predictions [T, N], optional.
        q90_2d: 90th percentile predictions [T, N], optional.
        q50_2d: Median predictions [T, N], optional.
        btc_close_1d: BTC close price series [T], optional for regime IC.
        net_daily_returns: Post-cost daily portfolio returns [D], optional.
            When provided, net_sharpe is computed. Otherwise NaN.
        cost_floor_bps: Total cost threshold (round-trip + hurdle), bps.
        n_trials: Number of hyperparameter trials (for DSR).
        horizon_bars: Label horizon in bars (for NW t-stat lag).
        trading_mask: Boolean mask [N] — True인 심볼(C3)만 trading panel 메트릭 계산에 사용.
            None이면 trading panel 계산 생략.

    Returns:
        AlphaEvaluationReport with all metrics and PASS/FAIL verdict.

    Time complexity: O(T * N * log N) dominated by rolling_ic.
    Space complexity: O(T * N).

    """
    # Signed net alpha: long minus short
    pred_2d: NDArray[np.float64] = alpha_long_2d - alpha_short_2d

    net_ic_dict: dict[str, float] = compute_net_ic(
        pred_2d, realized_fwd_ret_2d, horizon_bars=horizon_bars
    )

    breadth: float = compute_effective_breadth(alpha_long_2d, alpha_short_2d)

    # Cross-sectional return volatility for breakeven IC
    realized_flat: NDArray[np.float64] = realized_fwd_ret_2d[np.isfinite(realized_fwd_ret_2d)]
    sigma_r_bps: float = float(np.nanstd(realized_flat)) * 1e4 if realized_flat.size > 0 else 400.0

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

    regime_ic: dict[str, float] = (
        compute_per_regime_ic(pred_2d, realized_fwd_ret_2d, btc_close_1d, trend_window=30)
        if btc_close_1d is not None
        else {"bull": float("nan"), "bear": float("nan"), "chop": float("nan")}
    )

    regime_breakeven: dict[str, float] = (
        compute_per_regime_breakeven(
            alpha_long_2d,
            alpha_short_2d,
            realized_fwd_ret_2d,
            btc_close_1d,
            cost_floor_bps=cost_floor_bps,
            trend_window=30,
        )
        if btc_close_1d is not None
        else {"bull": float("nan"), "bear": float("nan"), "chop": float("nan")}
    )

    _logger.debug(
        "per_regime: ic=%s breakeven=%s",
        {r: f"{v:.4f}" for r, v in regime_ic.items()},
        {r: f"{v:.4f}" for r, v in regime_breakeven.items()},
    )

    # DSR computation
    ic_arr: NDArray[np.float64] = rolling_ic(pred_2d, realized_fwd_ret_2d)
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

    # cost_drag deferred to Phase 1 (requires backtest PnL decomposition)
    cost_drag: dict[str, float] = {}

    # PASS / FAIL verdict
    fail_reasons: list[str] = []

    if net_ic_dict["mean_ic"] < 0.03:
        fail_reasons.append("net_ic_below_0.03")
    if net_ic_dict["t_stat_nw"] < 2.0:
        fail_reasons.append("ic_t_stat_nw_below_2.0")
    if breadth < 3.0:
        fail_reasons.append("effective_breadth_below_3")
    if net_ic_dict["mean_ic"] < breakeven:
        fail_reasons.append("net_ic_below_breakeven")
    if not math.isnan(quantile_cov) and not (0.72 <= quantile_cov <= 0.88):
        fail_reasons.append("quantile_coverage_out_of_range")
    if dsr < 0.95:
        fail_reasons.append("deflated_sharpe_below_0.95")

    passes: bool = len(fail_reasons) == 0

    _logger.debug(
        "evaluate_alpha: net_ic=%.4f breadth=%.1f dsr=%.3f passes=%s fail=%s",
        net_ic_dict["mean_ic"],
        breadth,
        dsr,
        passes,
        fail_reasons,
    )

    # Phase F: dual panel 메트릭 계산 — inference(전체) vs trading(C3 마스크만)
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
    )


def sweep_horizon_breakeven(
    realized_fwd_ret_map: dict[int, NDArray[np.float64]],
    alpha_long_map: dict[int, NDArray[np.float64]],
    alpha_short_map: dict[int, NDArray[np.float64]],
    *,
    cost_floor_bps: float = 24.0,
) -> dict[int, dict[str, float]]:
    """Scan horizon candidates to find optimal cost/signal ratio.

    Pure function — accepts pre-computed realized return arrays keyed by
    horizon in bars, avoiding direct dependency on aligned market data.

    Args:
        realized_fwd_ret_map: {horizon_bars: realized_2d [T, N]} — gross fwd returns.
        alpha_long_map: {horizon_bars: alpha_long_2d [T, N]}.
        alpha_short_map: {horizon_bars: alpha_short_2d [T, N]}.
        cost_floor_bps: Round-trip cost + hurdle in bps (Taker fixed = 24.0).

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

        realized_flat = realized_2d[np.isfinite(realized_2d)]
        sigma_r_bps = float(np.nanstd(realized_flat)) * 1e4 if realized_flat.size > 0 else 400.0

        breadth = compute_effective_breadth(alpha_long_2d, alpha_short_2d)
        breakeven = compute_breakeven_ic(
            cost_floor_bps=cost_floor_bps,
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
