"""Statistical calibration primitives for the MHS fold-evidence gates.

Registered gates declare an error rate alpha against a stated null instead of
raw indicator thresholds: dispersion evidence bootstraps its null from the
strategy's own pooled daily returns, while level evidence uses closed-form
lower confidence bounds against registered absolute economic floors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from src.common.errors import DataIntegrityError
from src.mhs.params import (
    EVIDENCE_GATE_ALPHA,
    NULL_BOOTSTRAP_MEAN_BLOCK_DAYS,
    NULL_BOOTSTRAP_MIN_ROWS,
    NULL_BOOTSTRAP_SEED,
    NULL_BOOTSTRAP_TRIALS,
)

__all__ = [
    "NullShareCalibration",
    "calibrate_max_share_null",
    "sharpe_lower_confidence_bound",
    "stationary_block_bootstrap",
]


@dataclass(frozen=True, slots=True)
class NullShareCalibration:
    """One dispersion-gate calibration pass outcome.

    ``threshold`` is the (1-alpha) quantile of the strategy's own null
    max-fold-share distribution; ``observed_percentile`` locates the observed
    share inside that same null distribution.
    """

    threshold: float
    observed_percentile: float
    alpha: float
    n_folds: int
    fold_days: int
    trials: int


def stationary_block_bootstrap(
    source: np.ndarray,
    size: int,
    rng: np.random.Generator,
    mean_block_days: int = NULL_BOOTSTRAP_MEAN_BLOCK_DAYS,
) -> np.ndarray:
    """Circular stationary block resample preserving autocorrelation/fat tails."""
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    if mean_block_days < 1:
        raise ValueError(f"mean_block_days must be >= 1, got {mean_block_days}")
    if len(source) < 1:
        raise ValueError("source must be non-empty")
    n = len(source)
    out = np.empty(size, dtype="float64")
    filled = 0
    while filled < size:
        start = int(rng.integers(0, n))
        length = min(int(rng.geometric(1.0 / mean_block_days)), size - filled)
        out[filled : filled + length] = source[(start + np.arange(length)) % n]
        filled += length
    return out


def _log_growth(daily_returns: np.ndarray) -> float:
    return float(np.log1p(np.clip(daily_returns, -0.999, None)).sum())


def _percentile_rank(values: list[float], observed: float) -> float:
    below = sum(1 for value in values if value < observed)
    equal = sum(1 for value in values if value == observed)
    return 100.0 * (below + 0.5 * equal) / len(values)


def calibrate_max_share_null(
    pooled_daily_returns: np.ndarray,
    n_folds: int,
    fold_days: int,
    observed_share: float,
    alpha: float = EVIDENCE_GATE_ALPHA,
    *,
    trials: int = NULL_BOOTSTRAP_TRIALS,
    seed: int = NULL_BOOTSTRAP_SEED,
    mean_block_days: int = NULL_BOOTSTRAP_MEAN_BLOCK_DAYS,
    min_rows: int = NULL_BOOTSTRAP_MIN_ROWS,
) -> NullShareCalibration:
    """Derive the dispersion-gate threshold from the strategy's own null.

    Repeatedly draws ``n_folds`` x ``fold_days`` windows from the pooled daily
    returns and takes the (1-alpha) quantile of the resulting max-fold-share
    null as ``threshold``; trials whose total log growth is non-positive carry
    no share information and are excluded. Fail-closed: fewer than ``min_rows``
    finite rows or fewer than ``trials // 10`` usable trials raise
    ``DataIntegrityError`` -- never a silent fallback threshold.
    """
    finite = np.asarray(pooled_daily_returns, dtype="float64")
    finite = finite[np.isfinite(finite)]
    if len(finite) < min_rows:
        raise DataIntegrityError(
            f"null calibration requires >= {min_rows} finite pooled daily rows, "
            f"got {len(finite)}"
        )
    rng = np.random.default_rng(seed)
    shares: list[float] = []
    for _ in range(trials):
        growths = np.array([
            _log_growth(
                stationary_block_bootstrap(finite, fold_days, rng, mean_block_days)
            )
            for _ in range(n_folds)
        ])
        total = growths.sum()
        if total > 0.0:
            shares.append(float(growths.max() / total))
    if len(shares) < trials // 10:
        raise DataIntegrityError(
            f"null calibration produced only {len(shares)} usable trials "
            f"(need >= {trials // 10}) from {trials} attempts"
        )
    return NullShareCalibration(
        threshold=float(np.percentile(shares, 100.0 * (1.0 - alpha))),
        observed_percentile=_percentile_rank(shares, observed_share),
        alpha=float(alpha),
        n_folds=int(n_folds),
        fold_days=int(fold_days),
        trials=int(trials),
    )


def sharpe_lower_confidence_bound(
    daily_returns: np.ndarray,
    alpha: float = EVIDENCE_GATE_ALPHA,
    periods_per_year: float = 365.0,
) -> float:
    """Lo(2002) standard-error-based (1-alpha) lower bound of annualized Sharpe.

    Degenerate inputs (<2 rows, zero/non-finite dispersion or mean) return
    ``-inf`` so the caller's gate always rejects -- never fails open.
    """
    returns = np.asarray(daily_returns, dtype="float64").ravel()
    n = returns.size
    # 상수 배열은 부동소수점 오차로 std가 0이 아니게 되므로 전체 동일값을 먼저 검사한다.
    if n < 2 or bool(np.all(returns == returns[0])):
        return float("-inf")
    sd = float(returns.std(ddof=1))
    mean = float(returns.mean())
    if not math.isfinite(sd) or sd <= 0.0 or not math.isfinite(mean):
        return float("-inf")
    daily_sharpe = mean / sd
    se = math.sqrt((1.0 + daily_sharpe**2 / 2.0) / n) * math.sqrt(periods_per_year)
    z = NormalDist().inv_cdf(1.0 - alpha)
    return float(daily_sharpe * math.sqrt(periods_per_year) - z * se)
