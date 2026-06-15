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
        for right in valid_panel[idx + 1:]:
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
