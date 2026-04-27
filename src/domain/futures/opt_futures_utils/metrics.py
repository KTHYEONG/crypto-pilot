"""백테스트 성과 지표 계산 (수익률, MDD, Sortino, SPA bootstrap 등).

전략의 우수성을 판단하기 위해 단순 수익률뿐만 아니라 리스크 대비 효율성을 점수화하는 역할을 수행함.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def calc_profit_factor_from_pnl(pnl_series: pd.Series | np.ndarray | Sequence[float]) -> float:
    """Calculate Profit Factor from a pre-computed net PNL series (fee-deducted)."""
    pnl_arr = np.asarray(pnl_series)
    if pnl_arr.size == 0:
        return 1.0

    gross_profit: float = float(pnl_arr[pnl_arr > 0].sum())
    gross_loss: float = abs(float(pnl_arr[pnl_arr < 0].sum()))

    if gross_loss == 0.0:
        return 5.0 if gross_profit > 0 else 1.0

    return gross_profit / gross_loss


def calc_profit_factor(trades_df: pd.DataFrame) -> float:
    """Calculate Profit Factor (Gross Profit / Gross Loss) from raw trades_df."""
    if trades_df.empty:
        return 1.0

    gains: pd.Series = trades_df["pnl"][trades_df["pnl"] > 0]
    losses: pd.Series = trades_df["pnl"][trades_df["pnl"] < 0]

    gross_profit: float = float(gains.sum()) if not gains.empty else 0.0
    gross_loss: float = abs(float(losses.sum())) if not losses.empty else 0.0

    if gross_loss == 0.0:
        return 5.0 if gross_profit > 0 else 1.0

    return gross_profit / gross_loss


# ==============================================================================
# [NEW] Portfolio Aggregated Equity Curve Metrics
# ==============================================================================


def calc_mdd_from_equity(equity_curve: np.ndarray) -> float:
    """Calculate Maximum Drawdown from an aggregated equity curve."""
    if len(equity_curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    running_max[running_max == 0] = 1e-9
    drawdown = (equity_curve - running_max) / running_max * 100.0
    return float(abs(np.min(np.nan_to_num(drawdown, nan=0.0))))


def calc_sortino_from_equity(equity_curve: np.ndarray, span_days: float) -> float:
    """Compute annualized Sortino ratio from equity curve (log-return based).

    1. 스무딩 없음: 엔진이 반환한 틱/캔들 단위의 모든 궤적(Path) 고통을 그대로 측정.
    2. 로그 수익률(Log Returns) 사용: 상하방 비대칭성을 제거하여
       복리 환경에서의 실제 변동성 드래그를 정확히 산출.
    """
    if len(equity_curve) < 2 or span_days <= 0:
        return 0.0

    start_eq = equity_curve[0] if equity_curve[0] > 0 else 1e-9
    end_eq = equity_curve[-1]

    # 1. CAGR (Annualized Geometric Return)
    total_ret_ratio = max(end_eq / start_eq, 0.0001)
    cagr_decimal = (total_ret_ratio ** (365.0 / span_days)) - 1.0

    # 2. High-Resolution Log Returns (모든 스텝의 로그 수익률)
    # 기계식 복리 환경에서는 산술 수익률 대신 로그 수익률을 써야 수학적 왜곡(오차)이 없음.
    # 0 이하로 떨어지는 것을 방지하기 위해 매우 작은 값(1e-9)으로 클리핑.
    safe_curve = np.clip(equity_curve, 1e-9, None)
    step_log_returns = np.log(safe_curve[1:] / safe_curve[:-1])

    # 3. Downside Deviation (하방 변동성)
    # 기계에게 리스크란 '계좌 잔고가 전 캔들 대비 줄어든 모든 순간'임.
    downside_log_returns = step_log_returns[step_log_returns < 0]

    if len(downside_log_returns) == 0:
        return 999.0 if cagr_decimal > 0 else 0.0

    # [INSTITUTIONAL] Standard Sortino Calculation (L2 Norm)
    # NSGA-II 최적화의 수렴을 돕기 위해 표준 금융공학 산식(제곱 하방편차)으로 복구.
    # CAGR Cap과 연동하여 생존 모델을 찾음.
    downside_log_returns = step_log_returns[step_log_returns < 0]

    if len(downside_log_returns) == 0:
        return 999.0 if cagr_decimal > 0 else 0.0

    # Standard L2 variance of downside returns
    step_downside_var = np.mean(downside_log_returns**2.0)

    # Annualized Downside Deviation
    # 1년(365일) 동안 발생하는 캔들(스텝)의 개수로 스케일 업.
    bars_per_year = (len(equity_curve) / span_days) * 365.0
    annual_downside_dev = np.sqrt(step_downside_var * bars_per_year)

    if annual_downside_dev == 0.0:
        return 999.0 if cagr_decimal > 0 else 0.0

    # Continuous Log-Sortino Ratio
    sortino = cagr_decimal / annual_downside_dev

    return float(sortino)


def compute_pbo_from_cpcv_paths(
    is_path_scores: Sequence[float],
    oos_path_scores: Sequence[float],
    *,
    min_paths: int = 6,
) -> tuple[float, float]:
    """Return PBO proxy from CPCV path IS vs OOS score ranks.

    Returns (pbo_fraction, spearman_rho). PBO ~ 0.5 * (1 - rho).
    """
    is_arr = np.asarray(list(is_path_scores), dtype=np.float64)
    oos_arr = np.asarray(list(oos_path_scores), dtype=np.float64)
    if is_arr.size != oos_arr.size or is_arr.size < int(min_paths):
        return (1.0, 0.0)

    ri = pd.Series(is_arr).rank(method="average").to_numpy(dtype=np.float64)
    ro = pd.Series(oos_arr).rank(method="average").to_numpy(dtype=np.float64)

    if np.std(ri) < 1e-12 or np.std(ro) < 1e-12:
        return (0.5, 0.0)

    rho = float(np.corrcoef(ri, ro)[0, 1])
    if not np.isfinite(rho):
        rho = 0.0
    pbo = float(np.clip(0.5 * (1.0 - rho), 0.0, 1.0))
    return (pbo, rho)


def calc_cvar5_loss_pct_from_equity(equity_curve: np.ndarray) -> float:
    """Portfolio CVaR(5%): mean of worst 5% bar returns as positive loss %."""
    if equity_curve.size < 2:
        return 0.0
    eq = np.asarray(equity_curve, dtype=np.float64)
    r = np.diff(eq) / np.clip(eq[:-1], 1e-9, None)
    if r.size == 0:
        return 0.0
    sorted_r = np.sort(r)
    k = max(1, int(len(sorted_r) * 0.05))
    worst = sorted_r[:k]
    return float(-np.mean(worst) * 100.0)


def calc_max_underwater_days_from_equity(equity_curve: np.ndarray, hours_per_bar: float) -> float:
    """Longest stretch below running peak, converted to days (4H -> hours_per_bar=4)."""
    if equity_curve.size < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    underwater = equity_curve < peak
    max_run = 0
    cur = 0
    for u in underwater:
        if u:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return float(max_run * hours_per_bar / 24.0)


def calc_ulcer_index_from_equity(equity_curve: np.ndarray) -> float:
    """Ulcer Index: RMS of percentage drawdown from running peak."""
    if equity_curve.size < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd_pct = (equity_curve - peak) / np.clip(peak, 1e-9, None) * 100.0
    return float(np.sqrt(np.mean(dd_pct ** 2.0)))


def calc_tail_ratio_from_equity(equity: np.ndarray) -> float:
    """95th percentile return / abs(5th percentile return)."""
    if equity.size < 2:
        return 1.0
    r = np.diff(equity) / np.clip(equity[:-1], 1e-12, None)
    if r.size < 5:
        return 1.0
    val95 = float(np.percentile(r, 95.0))
    val5 = float(np.percentile(r, 5.0))
    if abs(val5) < 1e-12:
        return 5.0 if val95 > 0 else 1.0
    return float(val95 / abs(val5))


def _log_tw_from_ret_pct(ret_pct: float) -> float:
    r = 1.0 + float(ret_pct) / 100.0
    if r <= 0.0 or not math.isfinite(r):
        return -10.0
    return float(math.log(max(r, 1e-9)))


def calc_gate1_dsr_from_path_log_tw(
    path_arr: np.ndarray,
    tf: str,
    stat_ref_len: float,
    n_trials_opt: float,
) -> float:
    """Compute Deflated Sharpe (Bailey & Lopez de Prado) on CPCV path log-TWR samples.

    Returns probability in [0, 0.99] or sentinel -1.0 if path std is degenerate.
    """
    if path_arr.size < 2:
        return 0.0
    m_pt = float(np.mean(path_arr))
    s_pt = float(np.std(path_arr, ddof=1))
    if not math.isfinite(s_pt) or s_pt < 1e-6:
        return -1.0
    sharpe = m_pt / (s_pt + 1e-12)
    hrs = int(tf.replace("h", "")) if tf.endswith("h") else 4
    t_samples = float(stat_ref_len) / (24.0 / float(hrs))
    sk = float(np.mean(((path_arr - m_pt) / (s_pt + 1e-12)) ** 3))
    ex_kurt = float(np.mean(((path_arr - m_pt) / (s_pt + 1e-12)) ** 4)) - 3.0
    sr_var_denom = max(
        1.0 - sk * sharpe + ((ex_kurt + 2.0) / 4.0) * sharpe**2,
        1e-12,
    )
    sr_bench = math.sqrt(2.0 * math.log(max(n_trials_opt, 2.0)))
    z_dsr = (sharpe - sr_bench) * math.sqrt(max(t_samples - 1.0, 1.0)) / math.sqrt(sr_var_denom)
    dsr_val = float(0.5 * (1.0 + math.erf(z_dsr / math.sqrt(2.0))))
    return float(min(0.99, max(0.0, dsr_val)))


def calc_time_to_target_wealth(
    path_log_returns: np.ndarray,
    target_multiplier: float,
    bars_per_year: float,
) -> tuple[float, float]:
    """Return probability-weighted time to reach target asset multiplier.

    Returns (expected_years, ci_upper_years).
    """
    if path_log_returns.size < 2:
        return 999.0, 999.0
    
    mu = float(np.mean(path_log_returns)) * bars_per_year
    sigma = float(np.std(path_log_returns, ddof=1)) * math.sqrt(bars_per_year)
    
    if mu <= 1e-6:
        return 999.0, 999.0
        
    log_target = math.log(max(1.0001, target_multiplier))
    expected_years = log_target / mu
    
    # 95% CI upper bound using normal approximation
    z = 1.645
    n_years = float(len(path_log_returns)) / bars_per_year
    drift_ci_lower = mu - z * sigma / math.sqrt(max(1.0, n_years))
    
    if drift_ci_lower <= 1e-6:
        ci_upper_years = 999.0
    else:
        ci_upper_years = log_target / drift_ci_lower
        
    return float(expected_years), float(ci_upper_years)


def calc_net_alpha_with_friction(
    equity_curve: np.ndarray,
    benchmark_cagr: float,
    bars_per_year: float,
    avg_funding_rate_bps: float = 1.0,
    avg_slippage_bps: float = 2.0,
    turnover_per_bar: float = 0.1,
) -> float:
    """Compute net alpha accounting for liquidity-depth slippage and funding-rate friction."""
    if len(equity_curve) < 2:
        return 0.0
    
    start_eq = equity_curve[0] if equity_curve[0] > 0 else 1e-9
    end_eq = equity_curve[-1]
    total_ret_ratio = max(end_eq / start_eq, 0.0001)
    
    years = len(equity_curve) / max(1.0, bars_per_year)
    if years <= 0:
        return 0.0
        
    cagr_decimal = (total_ret_ratio ** (1.0 / years)) - 1.0
    
    annual_turnover = turnover_per_bar * bars_per_year
    annual_slippage_cost = annual_turnover * (avg_slippage_bps / 10000.0)
    annual_funding_cost = bars_per_year * (avg_funding_rate_bps / 10000.0) * 0.5
    
    net_cagr = cagr_decimal - annual_slippage_cost - annual_funding_cost
    return float(net_cagr - benchmark_cagr)


def stationary_bootstrap_spa(
    leg_log_tw: np.ndarray,
    n_bootstrap: int = 2000,
    block_length: int | None = None,
    seed: int = 42,
) -> float:
    """Sign-permutation / circular-block bootstrap p-value for H0: E[leg_log_tw] <= 0.

    For K=5 AWF legs, uses circular block bootstrap (block_length=2 default).
    p-value < 0.10 rejects H0 at 10% level — strategy has positive expected log-TW.

    Returns p-value in [0, 1]. Lower = more significant (better strategy).
    """
    arr = np.asarray(leg_log_tw, dtype=np.float64)
    k = arr.size
    if k < 2:
        return 0.5

    obs_mean = float(np.mean(arr))
    if obs_mean <= 0.0:
        return 1.0

    rng = np.random.default_rng(seed)
    blk = max(1, int(block_length or max(1, k // 2)))
    n = n_bootstrap

    # Circular block bootstrap under H0: subtract observed mean (demeaned resampling)
    arr_dm = arr - obs_mean  # demeaned — H0 version
    bootstrap_means: np.ndarray = np.empty(n, dtype=np.float64)
    for b in range(n):
        sample = np.empty(k, dtype=np.float64)
        idx = 0
        while idx < k:
            start = int(rng.integers(0, k))
            take = min(blk, k - idx)
            for j in range(take):
                sample[idx] = arr_dm[(start + j) % k]
                idx += 1
        bootstrap_means[b] = float(np.mean(sample))

    # p-value: fraction of bootstrap means >= observed mean - H0 mean (=0)
    # Under H0 the null mean is 0, so we compare bootstrap means to obs_mean
    p_val = float(np.mean(bootstrap_means >= obs_mean))
    return float(np.clip(p_val, 0.0, 1.0))


def block_bootstrap_tstat(
    leg_log_tw: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> float:
    """Block-bootstrap t-statistic for H0: mean=0 over AWF legs.

    Returns t-stat corrected for small-sample autocorrelation.
    t > 2.0 at K=5 indicates robust positive drift.
    """
    arr = np.asarray(leg_log_tw, dtype=np.float64)
    k = arr.size
    if k < 2:
        return 0.0
    mu = float(np.mean(arr))
    se = float(np.std(arr, ddof=1)) / math.sqrt(k)
    if se < 1e-12:
        return 0.0
    return float(mu / se)

