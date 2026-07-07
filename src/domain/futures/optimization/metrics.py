# src/domain/futures/strategy/tiered_workflow/metrics.py

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.stats import norm

if TYPE_CHECKING:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import StrategySignal

_BARS_PER_YEAR: float = 2190.0  # 4h 기준


def _bars_per_year_for_tf(tf: str) -> float:
    """Timeframe 문자열로부터 연간 bar 수를 계산."""
    from src.domain.futures.portfolio.signal_composer import hours_per_bar_tf

    hours_per_bar = float(hours_per_bar_tf(tf))
    if not np.isfinite(hours_per_bar) or hours_per_bar <= 0.0:
        return _BARS_PER_YEAR
    return float((24.0 * 365.0) / hours_per_bar)


def _clean_rets_array(rets: list[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(rets, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return np.zeros((0,), dtype=np.float64)
    return arr


def _default_hac_lag(n_obs: int) -> int:
    if n_obs <= 1:
        return 0
    lag = int(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))
    return max(1, min(lag, n_obs - 1))


def _hac_long_run_variance(
    values: NDArray[np.float64],
    max_lag: int | None = None,
) -> float:
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        return 0.0
    demeaned = values - float(np.mean(values))
    gamma_0 = float(np.dot(demeaned, demeaned)) / float(values.size)
    lag = _default_hac_lag(values.size) if max_lag is None else max(0, min(int(max_lag), values.size - 1))
    long_run_var = gamma_0
    for j in range(1, lag + 1):
        weight = 1.0 - (j / float(lag + 1))
        gamma_j = float(np.dot(demeaned[j:], demeaned[:-j])) / float(values.size)
        long_run_var += 2.0 * weight * gamma_j
    return float(max(long_run_var, 0.0))


def _effective_sample_size_hac(
    rets: list[float] | NDArray[np.float64],
    max_lag: int | None = None,
) -> float:
    arr = _clean_rets_array(rets)
    if arr.size < 2:
        return 0.0
    var_iid = float(np.var(arr, ddof=1))
    if not np.isfinite(var_iid) or var_iid <= 1e-12:
        return 0.0
    long_run_var = _hac_long_run_variance(arr, max_lag=max_lag)
    if long_run_var <= 1e-12:
        return float(arr.size)
    n_eff = float(arr.size) * var_iid / long_run_var
    return float(np.clip(n_eff, 1.0, float(arr.size)))


def _hac_sharpe(
    rets: list[float] | NDArray[np.float64],
    *,
    bars_per_year: float = _BARS_PER_YEAR,
    max_lag: int | None = None,
) -> float:
    arr = _clean_rets_array(rets)
    if arr.size < 2:
        return 0.0
    sigma_hac = float(np.sqrt(max(_hac_long_run_variance(arr, max_lag=max_lag), 0.0)))
    if sigma_hac <= 1e-12:
        return 0.0
    return float((float(np.mean(arr)) / sigma_hac) * np.sqrt(bars_per_year))


def _annualized_log_growth(
    rets: list[float] | NDArray[np.float64],
    *,
    bars_per_year: float = _BARS_PER_YEAR,
) -> float:
    arr = _clean_rets_array(rets)
    if arr.size == 0 or np.any(arr <= -1.0):
        return float("nan")
    return float(bars_per_year * np.mean(np.log1p(arr)))


def _block_log_growth(
    rets: list[float] | NDArray[np.float64],
    *,
    bars_per_year: float = _BARS_PER_YEAR,
    block_size: int,
) -> NDArray[np.float64]:
    arr = _clean_rets_array(rets)
    if arr.size == 0 or block_size <= 0 or np.any(arr <= -1.0):
        return np.zeros((0,), dtype=np.float64)
    blocks: list[float] = []
    for start in range(0, arr.size, block_size):
        block = arr[start : start + block_size]
        if block.size == 0:
            continue
        blocks.append(float(bars_per_year * np.mean(np.log1p(block))))
    return np.asarray(blocks, dtype=np.float64)


def _growth_lcb(
    block_log_growth: list[float] | NDArray[np.float64],
    *,
    z_lcb: float = 1.0,
) -> float:
    arr = _clean_rets_array(block_log_growth)
    if arr.size == 0:
        return float("-1e6")
    if arr.size == 1:
        return float(arr[0])
    stderr = float(np.std(arr, ddof=1)) / float(np.sqrt(arr.size))
    return float(np.mean(arr) - (z_lcb * stderr))


def _cvar_95(rets: list[float] | NDArray[np.float64]) -> float:
    arr = _clean_rets_array(rets)
    if arr.size == 0:
        return float("inf")
    losses = -arr
    var_cut = float(np.quantile(losses, 0.95))
    tail = losses[losses >= var_cut]
    if tail.size == 0:
        return max(var_cut, 0.0)
    return float(np.maximum(np.mean(tail), 0.0))


def _sharpe_hac(
    rets: list[float] | NDArray[np.float64],
    *,
    bars_per_year: float,
    max_lag: int | None = None,
) -> float:
    return _hac_sharpe(rets, bars_per_year=bars_per_year, max_lag=max_lag)


def _contiguous_block_log_growth(
    rets: list[float] | NDArray[np.float64],
    *,
    block_bars: int,
) -> NDArray[np.float64]:
    arr = _clean_rets_array(rets)
    if arr.size == 0 or block_bars <= 0 or np.any(arr <= -1.0):
        return np.zeros((0,), dtype=np.float64)
    blocks: list[float] = []
    for start in range(0, arr.size, int(block_bars)):
        block = arr[start : start + int(block_bars)]
        if block.size == 0:
            continue
        blocks.append(float(np.sum(np.log1p(block), dtype=np.float64)))
    return np.asarray(blocks, dtype=np.float64)


def _growth_lower_confidence_bound(
    block_log_growth: NDArray[np.float64],
    *,
    blocks_per_year: float,
    z_value: float,
) -> float:
    if block_log_growth.size == 0 or not np.all(np.isfinite(block_log_growth)):
        return float("-1e6")
    annualized = block_log_growth * float(blocks_per_year)
    if annualized.size == 1:
        return float(np.expm1(annualized[0]))
    stderr = float(np.std(annualized, ddof=1)) / float(np.sqrt(annualized.size))
    return float(np.expm1(float(np.mean(annualized)) - (float(z_value) * stderr)))


def _cvar_loss(
    rets: list[float] | NDArray[np.float64],
    *,
    alpha: float = 0.95,
) -> float:
    arr = _clean_rets_array(rets)
    if arr.size == 0:
        return float("inf")
    losses = -arr
    var_cut = float(np.quantile(losses, alpha))
    tail = losses[losses >= var_cut]
    if tail.size == 0:
        return float(max(var_cut, 0.0))
    return float(np.maximum(np.mean(tail), 0.0))


def _sortino_hac_unit(
    rets: list[float] | NDArray[np.float64],
    *,
    bars_per_year: float = _BARS_PER_YEAR,
    target: float = 0.0,
    max_lag: int | None = None,
) -> float:
    """HAC 조정 downside deviation 기반 scale-invariant Sortino 비율.

    Scale-invariant 특성: leverage -> kL 변환 시 E[r]·sigma_down 동비율 변화 -> Sortino 불변.
    Sortino_HAC_unit은 unit-vol 정규화 book의 shape metric으로 사용.

    Args:
        rets: per-bar 수익률 리스트 또는 배열.
        bars_per_year: 연율화 팩터 (4h=2190).
        target: 하방편차 기준점 (기본 0.0).
        max_lag: HAC 최대 lag (None=자동).

    Returns:
        HAC 조정 연율화 Sortino. 데이터 부족·비수렴 시 0.0.

    Time Complexity: O(n·max_lag). Space Complexity: O(n).
    """
    arr = _clean_rets_array(rets)
    if arr.size < 2:
        return 0.0
    mean_r = float(np.mean(arr))

    # downside HAC: HAC long-run variance를 downside 표본에 적용
    downside_mask = arr < target
    if not np.any(downside_mask):
        return 0.0
    downside = arr[downside_mask] - target  # 양수 절대값 손실

    # HAC long-run variance on downside deviations (전표본 N 정규화 — Sortino & Price 1994)
    # downside.size << arr.size이므로 lag 자동 조정
    lrv = _hac_long_run_variance(
        downside,
        max_lag=_default_hac_lag(downside.size) if max_lag is None else max_lag,
    )
    # 전표본 N 정규화: TDD = sqrt(sum(d^2) / N_total)와 정합
    # HAC 분산은 downside 표본에서 계산하므로 N_down→N_total 보정
    n_ratio = float(downside.size) / float(arr.size)
    dd_hac = float(np.sqrt(max(lrv * n_ratio, 0.0)))
    if dd_hac < 1e-12:
        return 0.0
    return float((mean_r - target) / dd_hac * np.sqrt(bars_per_year))


def _deflated_sharpe_probability(
    *,
    selected_rets: list[float] | NDArray[np.float64],
    completed_trial_sharpes: NDArray[np.float64],
    effective_trial_count: float,
    bars_per_year: float,
    max_lag: int | None = None,
) -> float:
    arr = _clean_rets_array(selected_rets)
    if arr.size < 2 or effective_trial_count <= 0.0:
        return 0.0

    # 1. Annualized Sharpe Ratio
    observed_ann = _sharpe_hac(arr, bars_per_year=bars_per_year, max_lag=max_lag)
    if not np.isfinite(observed_ann):
        return 0.0

    # 2. Convert to per-bar scale
    observed_per_bar = observed_ann / np.sqrt(bars_per_year)

    # 3. Convert trial pool Sharpe ratios to per-bar scale
    sr_pool_ann = np.asarray(completed_trial_sharpes, dtype=np.float64)
    sr_pool_ann = sr_pool_ann[np.isfinite(sr_pool_ann)]
    if sr_pool_ann.size == 0:
        return _psr(arr.tolist(), bars_per_year=bars_per_year)

    sr_pool_per_bar = sr_pool_ann / np.sqrt(bars_per_year)

    # 4. Calculate per-bar benchmark — Bailey & López de Prado (2014) null SR=0 정론.
    # +mean(pool) 항 제거: 동일 신호셋 파라미터 섭동(독립 가설 불성립) → 자기참조 제거.
    # benchmark = std(pool) * sqrt(2 * ln(N_eff)) only.
    benchmark_per_bar = float(np.std(sr_pool_per_bar, ddof=0) * np.sqrt(2.0 * np.log(max(effective_trial_count, 1.0))))

    # 5. Effective sample size
    n_eff = _effective_sample_size_hac(arr, max_lag=max_lag)
    if n_eff <= 1.0:
        return 0.0

    # 6. Standard error under Bailey & López de Prado (2012)
    from scipy.stats import kurtosis as _kurt
    from scipy.stats import skew as _skew

    skew_val = float(_skew(arr))
    kurt_val = float(_kurt(arr, fisher=True))

    denom = 1.0 - skew_val * observed_per_bar + (kurt_val + 2.0) / 4.0 * observed_per_bar**2
    variance = 1.0 / (n_eff - 1.0) if denom <= 0.0 else denom / (n_eff - 1.0)

    variance = max(variance, 1e-12)
    z_score = (observed_per_bar - benchmark_per_bar) / float(np.sqrt(variance))
    return float(norm.cdf(z_score))


def _is_non_constant_finite_array(values: NDArray[np.float64]) -> bool:
    """배열이 모두 유한하고 상수(모든 원소가 동일)가 아닌지 여부 검증."""
    if values.size == 0:
        return False
    if not np.all(np.isfinite(values)):
        return False
    return bool(not np.all(values == values[0]))


def _sharpe(rets: list[float], bars_per_year: float = _BARS_PER_YEAR) -> float:
    """연율화 Sharpe 계산.

    Args:
        rets: per-bar 수익률 리스트.
        bars_per_year: 연율화 팩터.

    Returns:
        Sharpe Ratio (float). 데이터 부족 시 0.0.
    """
    if len(rets) < 2:
        return 0.0
    arr = np.asarray(rets, dtype=np.float64)
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd < 1e-9:
        return 0.0
    return float(mu * bars_per_year / (sd * np.sqrt(bars_per_year)))


def _sortino(
    rets: list[float] | NDArray[np.float64],
    *,
    bars_per_year: float = _BARS_PER_YEAR,
    target: float = 0.0,
) -> float:
    """연율화 Sortino 비율 계산 (하방편차 기준 위험효율).

    Args:
        rets: per-bar 수익률 리스트 또는 배열.
        bars_per_year: 연율화 팩터.
        target: 하방편차 기준점 (기본 0.0).

    Returns:
        연율화 Sortino. 데이터 부족 또는 무손실(dd≈0) degenerate 시 0.0.

    Time Complexity: O(n). Space Complexity: O(n).
    """
    arr = _clean_rets_array(rets)
    if arr.size < 2:
        return 0.0
    mean_r = float(np.mean(arr))
    downside = arr[arr < target]
    if downside.size == 0:
        return 0.0
    # 표준 Target Downside Deviation(TDD): 전표본 N으로 정규화 (Sortino & Price 1994)
    # ÷downside.size(비표준)가 아닌 ÷arr.size(전표본)으로 Sharpe와 분모 기준 일치
    dd = float(np.sqrt(np.sum(np.square(downside - target)) / arr.size))
    if dd < 1e-12:
        return 0.0
    return float((mean_r - target) / dd * np.sqrt(bars_per_year))


def _terminal_multiple(rets: list[float] | NDArray[np.float64]) -> float:
    """누적 복리 배수 ∏(1+r) 계산 ("PnL multiple").

    Args:
        rets: per-bar 수익률 리스트 또는 배열.

    Returns:
        복리 배수. 빈 배열 → 1.0. 전손(∏<=0) → 0.0.

    Time Complexity: O(n). Space Complexity: O(n).
    """
    arr = _clean_rets_array(rets)
    if arr.size == 0:
        return 1.0
    multiple = float(np.prod(1.0 + arr))
    if not np.isfinite(multiple) or multiple <= 0.0:
        return 0.0
    return multiple


def _mdd(rets: list[float]) -> float:
    """최대 낙폭 계산 (양수 반환).

    Args:
        rets: per-bar 수익률 리스트.

    Returns:
        최대 낙폭 절대값. 데이터 없으면 0.0.
    """
    if not rets:
        return 0.0
    arr = np.asarray(rets, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return float("nan")
    equity = np.cumprod(1.0 + arr)
    running_max = np.maximum.accumulate(equity)
    drawdown = 1.0 - np.divide(
        equity,
        np.maximum(running_max, 1e-12),
    )
    return float(np.max(drawdown))


def _cagr(rets: list[float], bars_per_year: float = _BARS_PER_YEAR) -> float:
    """연율화 CAGR 계산 (복리).

    Args:
        rets: per-bar 수익률 리스트.
        bars_per_year: 연율화 팩터.

    Returns:
        CAGR. 빈 리스트면 0.0, total loss(복리곱 <= 0)면 -1.0.
    """
    if not rets:
        return 0.0
    arr = np.asarray(rets, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return float("nan")
    n = len(arr)
    base = float(np.prod(1.0 + arr))
    if not np.isfinite(base) or base <= 0.0:
        return -1.0
    return float(base ** (bars_per_year / n) - 1.0)


def _nw_tstat_realized(r_sym: NDArray[np.float64]) -> float:
    """Bartlett NW HAC t-stat on a realized return series.

    Uses lag m = clip(n//20, 1, n-1) (5-percentile bandwidth).
    Returns 0.0 for n<4 or degenerate (std<1e-9) series.
    """
    n = len(r_sym)
    if n < 4:
        return 0.0
    if float(np.std(r_sym)) < 1e-9:
        return 0.0
    mu = float(np.mean(r_sym))
    demeaned = r_sym - mu
    m = min(n - 1, max(1, n // 20))
    gamma0 = float(np.dot(demeaned, demeaned)) / n
    gamma_sum = gamma0
    for j in range(1, m + 1):
        w = 1.0 - j / (m + 1)
        gamma_j = float(np.dot(demeaned[j:], demeaned[:-j])) / n
        gamma_sum += 2.0 * w * gamma_j
    se_hac = float(np.sqrt(max(gamma_sum, 1e-20) / n))
    return mu / se_hac if se_hac > 1e-20 else 0.0


def _newey_west_ic_tstat(
    pred: NDArray[np.float64],
    realized: NDArray[np.float64],
    max_lag: int | None = None,
) -> float:
    """Newey-West HAC t-stat for Spearman rank IC."""
    n_obs = len(pred)
    if n_obs < 4:
        return 0.0

    from scipy.stats import rankdata

    rp = rankdata(pred).astype(np.float64) / n_obs
    rr = rankdata(realized).astype(np.float64) / n_obs

    u = (rp - 0.5) * (rr - 0.5)
    ic_est = 12.0 * float(np.mean(u))

    nw_lag = max_lag if max_lag is not None else int(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))
    nw_lag = max(1, min(nw_lag, n_obs - 1))

    u_dm = u - np.mean(u)
    gamma_0 = float(np.dot(u_dm, u_dm)) / n_obs

    s_nw = gamma_0
    for lag in range(1, nw_lag + 1):
        gamma_l = float(np.dot(u_dm[lag:], u_dm[:-lag])) / n_obs
        w_l = 1.0 - lag / (nw_lag + 1.0)
        s_nw += 2.0 * w_l * gamma_l

    s_nw = max(s_nw, 1e-12)
    se_ic = 12.0 * np.sqrt(s_nw / n_obs)
    t_stat = ic_est / (se_ic + 1e-12)
    return float(t_stat)


def variance_ratio(rets: NDArray[np.float64], q: int) -> tuple[float, float]:
    """Lo-MacKinlay VR(q) and heteroskedasticity-robust M2 statistic.

    Returns (VR, M2). Returns (1.0, 0.0) if n < q*4.
    """
    n = len(rets)
    if n < q * 4:
        return (1.0, 0.0)
    mu = float(np.mean(rets))
    demeaned = rets - mu
    var_1 = float(np.dot(demeaned, demeaned)) / n

    # q-period overlapping returns variance
    q_rets = np.array([float(np.sum(rets[t : t + q])) for t in range(n - q + 1)])
    q_demeaned = q_rets - q * mu
    var_q = float(np.dot(q_demeaned, q_demeaned)) / (n - q + 1)

    vr = var_q / (q * var_1) if var_1 > 1e-20 else 1.0

    # Heteroskedasticity-robust phi(q)
    delta = np.zeros(q - 1)
    for j in range(1, q):
        numer = float(np.dot(demeaned[j:] ** 2, demeaned[:-j] ** 2))
        denom = (float(np.dot(demeaned, demeaned)) / n) ** 2
        delta[j - 1] = numer / (denom * n) if denom > 1e-40 else 0.0

    phi = float(np.sum([(2 * (q - j) / q) ** 2 * delta[j - 1] for j in range(1, q)]))
    m2 = (vr - 1.0) / float(np.sqrt(max(phi, 1e-20)))
    return (float(vr), float(m2))



def kaufman_efficiency_ratio(rets: NDArray[np.float64]) -> float:
    """Kaufman ER = |sum(rets)| / sum(|rets|). Returns 0.0 for n<4 or all-zero denom.

    [ADR_20260707_LTF_ENTRY_TIMING_LAYER]
    """
    if rets is None or rets.size < 4:
        return 0.0
    if not np.all(np.isfinite(rets)):
        return 0.0
    denom = float(np.sum(np.abs(rets)))
    if denom < 1e-20:
        return 0.0
    return float(np.abs(np.sum(rets))) / denom

def hurst_dfa(rets: NDArray[np.float64], *, min_scale: int = 8, max_scale: int | None = None) -> float:
    """Detrended Fluctuation Analysis Hurst exponent.

    Returns 0.5 for n<32, non-finite, or non-convergent fits.
    """
    n = len(rets)
    if n < 32 or not np.all(np.isfinite(rets)):
        return 0.5

    max_s = max_scale if max_scale is not None else n // 4
    max_s = min(max_s, n // 4)

    cumsum = np.cumsum(rets - np.mean(rets))

    scales: list[int] = []
    flucts: list[float] = []

    s = min_scale
    while s <= max_s:
        n_segments = n // s
        if n_segments < 2:
            break
        rms_vals: list[float] = []
        for seg in range(n_segments):
            seg_data = cumsum[seg * s : (seg + 1) * s]
            x = np.arange(s, dtype=np.float64)
            # Detrend by linear fit
            coeffs = np.polyfit(x, seg_data, 1)
            trend = np.polyval(coeffs, x)
            residual = seg_data - trend
            rms_vals.append(float(np.sqrt(np.mean(residual**2))))
        scales.append(s)
        flucts.append(float(np.mean(rms_vals)))
        s = int(s * 1.5) + 1

    if len(scales) < 4:
        return 0.5

    log_s = np.log(np.array(scales, dtype=np.float64))
    log_f = np.log(np.array(flucts, dtype=np.float64))

    if not np.all(np.isfinite(log_f)):
        return 0.5

    coeffs_fit = np.polyfit(log_s, log_f, 1)
    h = float(coeffs_fit[0])
    return float(np.clip(h, 0.0, 1.0))


def _series_tstat(values: NDArray[np.float64]) -> float:
    if values.size < 2:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return 0.0
    sigma = float(np.std(finite, ddof=1))
    if sigma <= 0.0:
        return 0.0
    return float(np.mean(finite) / (sigma / np.sqrt(finite.size)))


@njit(fastmath=False, cache=True)  # type: ignore[untyped-decorator]
def _numba_moving_block_bootstrap_mean(
    cluster_means: NDArray[np.float64],
    n_clusters: int,
    block: int,
    n_bootstrap: int,
    rand_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    boot = np.zeros(n_bootstrap, dtype=np.float64)
    num_blocks = rand_indices.shape[1]

    for boot_idx in range(n_bootstrap):
        sample_arr = np.zeros(n_clusters + block, dtype=np.float64)
        curr_len = 0
        for j in range(num_blocks):
            start = rand_indices[boot_idx, j]
            end = min(n_clusters, start + block)
            length = end - start
            if length > 0:
                sample_arr[curr_len : curr_len + length] = cluster_means[start:end]
                curr_len += length
            if curr_len >= n_clusters:
                break
        if curr_len > 0:
            use_len = min(curr_len, n_clusters)
            boot[boot_idx] = np.mean(sample_arr[:use_len])
    return boot


def moving_block_bootstrap_mean(
    values: NDArray[np.float64],
    decision_idx: NDArray[np.int64],
    *,
    block_bars: int,
    n_bootstrap: int,
    seed: int,
) -> NDArray[np.float64]:
    """Bootstrap mean distribution over decision-index blocks."""
    if values.size == 0 or decision_idx.size == 0 or values.size != decision_idx.size:
        return np.zeros((0,), dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(decision_idx)
    if int(mask.sum()) < 2:
        return np.zeros((0,), dtype=np.float64)
    ordered_idx = np.argsort(decision_idx[mask], kind="stable")
    x = values[mask][ordered_idx]
    d = decision_idx[mask][ordered_idx]
    unique_decisions, inverse = np.unique(d, return_inverse=True)
    cluster_means = np.zeros(unique_decisions.shape[0], dtype=np.float64)
    cluster_counts = np.zeros(unique_decisions.shape[0], dtype=np.float64)
    np.add.at(cluster_means, inverse, x)
    np.add.at(cluster_counts, inverse, 1.0)
    cluster_means = np.divide(
        cluster_means,
        np.maximum(cluster_counts, 1.0),
        out=np.zeros_like(cluster_means),
        where=cluster_counts > 0.0,
    )
    n_clusters = cluster_means.size
    if n_clusters < 2 or n_bootstrap < 1:
        return np.zeros((0,), dtype=np.float64)
    block = max(1, int(block_bars))
    num_blocks = max(1, (n_clusters + block - 1) // block)
    rng = np.random.default_rng(seed)
    rand_indices = rng.integers(0, n_clusters, size=(n_bootstrap, num_blocks), dtype=np.int64)

    return _numba_moving_block_bootstrap_mean(  # type: ignore[no-any-return]
        np.ascontiguousarray(cluster_means),
        n_clusters,
        block,
        n_bootstrap,
        np.ascontiguousarray(rand_indices),
    )


def _one_sided_p_value(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return 1.0
    return float(norm.sf(t_stat))


def compute_panel_diversity(panel: tuple[StrategySignal, ...]) -> float:
    """유효 전략 fold-edge 상관 기반 다양성."""
    valid_panel = [sig for sig in panel if sig.valid]
    if len(valid_panel) < 2:
        return 0.0

    pairwise_abs_corr: list[float] = []
    for idx, left in enumerate(valid_panel[:-1]):
        left_map = dict(left._fold_edges)
        for right in valid_panel[idx + 1 :]:
            right_map = dict(right._fold_edges)
            common_folds = sorted(set(left_map) & set(right_map))
            if len(common_folds) < 2:
                pairwise_abs_corr.append(1.0)
                continue
            left_vec = np.asarray([left_map[k] for k in common_folds], dtype=np.float64)
            right_vec = np.asarray([right_map[k] for k in common_folds], dtype=np.float64)
            if not _is_non_constant_finite_array(left_vec) or not _is_non_constant_finite_array(right_vec):
                pairwise_abs_corr.append(1.0)
                continue
            corr = float(np.corrcoef(left_vec, right_vec)[0, 1])
            pairwise_abs_corr.append(abs(corr) if np.isfinite(corr) else 1.0)

    if not pairwise_abs_corr:
        return 0.0
    return float(np.clip(1.0 - float(np.mean(pairwise_abs_corr)), 0.0, 1.0))


def _psr(
    rets: list[float],
    sr_benchmark: float = 0.0,
    bars_per_year: float = _BARS_PER_YEAR,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012).

    PSR = Φ( (SR_obs - SR_bench) * sqrt(n-1)
             / sqrt(1 - skew*SR_obs + (kurt-1)/4 * SR_obs^2) )
    SR_obs는 per-bar(비연율화) Sharpe. n<2이면 0.0.

    Args:
        rets: per-bar 수익률 리스트.
        sr_benchmark: 비교 기준 Sharpe (기본값 0.0).
        bars_per_year: 연율화 팩터 (미사용, 시그니처 일관성 유지).

    Returns:
        PSR ∈ [0, 1]. 데이터 부족 또는 비정상 입력 시 0.0.

    Time Complexity: O(n).
    Space Complexity: O(n).
    """
    if len(rets) < 2:
        return 0.0
    arr = np.asarray(rets, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return 0.0
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd < 1e-9:
        return 0.0
    # per-bar(비연율화) Sharpe
    sr_obs = mu / sd
    n = len(arr)
    from scipy.stats import kurtosis as _kurt
    from scipy.stats import skew as _skew

    skew_val = float(_skew(arr))
    kurt_val = float(_kurt(arr, fisher=True))  # excess kurtosis κ = γ₄ - 3 (normal=0)
    # Bailey & López de Prado (2012): (γ₄ - 1)/4 = (κ + 3 - 1)/4 = (κ + 2)/4
    denom = 1.0 - skew_val * sr_obs + (kurt_val + 2.0) / 4.0 * sr_obs**2
    if denom <= 0.0:
        return 0.0
    from scipy.special import ndtr

    z = (sr_obs - sr_benchmark) * float(np.sqrt(n - 1)) / float(np.sqrt(denom))
    return float(ndtr(z))


def compute_breadth_weighted_ic(
    per_symbol_ic: dict[str, float],
    per_symbol_n: dict[str, int],
) -> tuple[float, float]:
    """이벤트 가중 평균 per-symbol IC + cross-symbol IC IR t-stat."""
    if not per_symbol_ic:
        return 0.0, 0.0

    syms = list(per_symbol_ic.keys())
    ic_arr = np.array([per_symbol_ic[s] for s in syms], dtype=np.float64)
    n_arr = np.array([float(max(per_symbol_n.get(s, 1), 1)) for s in syms], dtype=np.float64)

    total_n = float(n_arr.sum())
    ic_weighted = float(np.dot(ic_arr, n_arr) / total_n) if total_n > 0 else 0.0

    s = len(syms)
    if s < 2:
        return ic_weighted, 0.0

    ic_mean = float(ic_arr.mean())
    ic_std = float(ic_arr.std(ddof=1))
    ic_ir = ic_mean / (ic_std / np.sqrt(s) + 1e-12)

    return ic_weighted, float(ic_ir)
