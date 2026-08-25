"""Evaluation layer: phase diagnostics, cost/tail sensitivity, Sharpe, folds, readiness.

The target-weight proxy (``mhs_ledger_pnl``) is pre-screen only; Research GO,
OOS, capital, and participation numbers come from the simulated inventory
ledger. ``phase_diagnostic_metrics`` answers "is the result robust to an
arbitrary decision-clock offset?" and nothing else.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.mhs.execution import mhs_ledger_pnl
from src.mhs.params import (
    DEFAULT_SELECTION_WINDOW,
    PERIODS_PER_YEAR_1H,
    PNL_VOL_TARGET_BURN_IN_DAYS,
)
from src.mhs.types import DISCOVERY_START, MEASURED_EXECUTION_COST_TIERS_BPS

_EULER_GAMMA = 0.577215664901532860606512090082402431


def selection_overlap_fraction(
    report_start: pd.Timestamp, report_end: pd.Timestamp
) -> float:
    """Fraction of the report window inside ``DEFAULT_SELECTION_WINDOW``.

    Observational disclosure only (never a blocking gate): the CLI defaults
    were selected on that window, so any overlap means partially in-sample
    reporting. Returns ``|report ∩ selection| / |report|`` clipped to
    ``[0.0, 1.0]``; ``0.0`` for a zero-length or disjoint report window;
    raises ``ValueError`` when ``report_end < report_start``.
    """
    if report_end < report_start:
        raise ValueError(
            f"report_end ({report_end}) must not precede report_start ({report_start})"
        )
    span = report_end - report_start
    if span <= pd.Timedelta(0):
        return 0.0
    window_start, window_end = DEFAULT_SELECTION_WINDOW
    overlap = min(report_end, window_end) - max(report_start, window_start)
    if overlap <= pd.Timedelta(0):
        return 0.0
    return float(min(overlap / span, 1.0))


def holdout_tail_evidence(
    equity: pd.Series,
    cutoff: pd.Timestamp,
    min_days: int = PNL_VOL_TARGET_BURN_IN_DAYS,
) -> dict[str, Any] | None:
    """Summary of the equity ledger's tail strictly after ``cutoff``.

    Isolates the segment after the sealed evaluation boundary (typically
    ``HOLDOUT_CUTOFF``) so a report crossing it discloses the truly-novel
    segment's own performance separately from the dominant, already-tuned-
    against portion before it. Returns ``None`` -- never a near-empty or
    degenerate summary -- when fewer than ``min_days`` daily tail observations
    exist, including every ordinary run that never crosses the boundary at
    all. Pure function: the input ``equity`` series is never mutated.
    """
    daily_returns = equity.resample("1D").last().pct_change()
    tail = daily_returns.loc[daily_returns.index > cutoff].dropna()
    if len(tail) < min_days:
        return None
    curve = (1.0 + tail).cumprod()
    n_days = len(tail)
    years = n_days / 365.25  # n_days >= min_days >= 1 이므로 항상 양수.
    total_return = float(curve.iloc[-1] - 1.0)
    geometric_cagr = float(curve.iloc[-1] ** (1.0 / years) - 1.0)
    running_max = curve.cummax()
    max_drawdown = float((curve / running_max - 1.0).min())
    std = float(tail.std(ddof=1))
    naive_sharpe = (
        float(tail.mean() / std * np.sqrt(365.25)) if std > 0 else float("nan")
    )
    return {
        "start": str(tail.index[0]),
        "end": str(tail.index[-1]),
        "n_days": n_days,
        "total_return": total_return,
        "geometric_cagr": geometric_cagr,
        "max_drawdown": max_drawdown,
        "naive_sharpe": naive_sharpe,
    }


def _zero_variance(sd: float, mean: float) -> bool:
    """Treat numerically-zero sample dispersion as zero variance."""
    return sd <= 1e-12 * max(1.0, abs(mean))


def _sharpe(series: pd.Series, periods_per_year: float) -> float:
    """Annualized sample Sharpe, zero-variance-safe."""
    sd = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    mean = float(series.mean())
    if _zero_variance(sd, mean):
        return float("inf") if mean > 0 else float("-inf")
    return float(mean / sd * math.sqrt(periods_per_year))

def effective_breadth(returns: pd.DataFrame) -> tuple[float, float]:
    """Participation-ratio effective breadth of a return panel.

    ``n_eff = (sum(lambda))^2 / sum(lambda^2)`` over the eigenvalues of the
    return correlation matrix, plus the mean pairwise correlation. Non-finite
    correlation entries (zero-variance columns) become 0.0 before the
    eigendecomposition and the diagonal is forced to 1.0, matching quant.md's
    numerical-stability convention. Fails closed on fewer than 2 columns or 2
    rows; the final ``n_eff`` is clipped to ``[1.0, n_columns]``.
    """
    n_columns, n_rows = returns.shape[1], returns.shape[0]
    if n_columns < 2 or n_rows < 2:
        raise ValueError(
            f"effective_breadth requires >= 2 columns and >= 2 rows, got {returns.shape}"
        )
    corr = returns.corr().to_numpy()
    corr = np.where(np.isfinite(corr), corr, 0.0)
    np.fill_diagonal(corr, 1.0)
    eigenvalues = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    n_eff = float((eigenvalues.sum() ** 2) / (eigenvalues**2).sum())
    n_eff = float(np.clip(n_eff, 1.0, n_columns))
    mean_corr = float((corr.sum() - n_columns) / (n_columns * (n_columns - 1)))
    return n_eff, mean_corr


@dataclass(frozen=True, slots=True)
class PhaseDiagnosticResult:
    """Robustness diagnostic over independently-run, non-capital-shared phases.

    Not a tradable portfolio; ``degenerate`` flags a phase spread that exceeds
    the absolute mean performance, in which case the numbers must not be
    reported as a go/no-go input.
    """

    n_phases: int
    ensemble_ann: float
    ensemble_sharpe: float
    mean_phase_ann: float
    min_phase_ann: float
    max_phase_ann: float
    phase_spread_ann: float
    degenerate: bool


@dataclass(frozen=True, slots=True)
class CostResponsePoint:
    net_ann: float
    net_sharpe: float
    net_t: float


@dataclass(frozen=True, slots=True)
class TailSensitivityResult:
    base_net_ann: float
    base_sharpe: float
    winsor_curve: dict[int, tuple[float, float]]
    event_window_bars: int
    event_count: int
    top1_event_share: float
    top5_event_share: float
    top1pct_events_share: float
    leave_worst_event_out_sharpe: float


@dataclass(frozen=True, slots=True)
class BookEvidence:
    """Pre-screen + tail evidence for exactly one capital book.

    Carries both significance instruments (``prescreen`` and ``tail``) for a
    single weight book so reference and executed books can be measured under distinct labels.
    """

    prescreen: dict[float, CostResponsePoint]
    tail: TailSensitivityResult


def phase_diagnostic_metrics(
    phase_nets: Mapping[int, pd.Series],
    periods_per_year: float,
) -> PhaseDiagnosticResult:
    """Mean of N independently-run single-phase books, annualized.

    ``ensemble_ann`` is the annualized mean of the equal-weight average across
    the independent ``phase_nets``; per-phase annualized means are summarized
    by mean/min/max and ``phase_spread_ann`` is max minus min.
    """
    if not phase_nets:
        raise ValueError("phase_nets must not be empty")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
    series = list(phase_nets.values())
    first = series[0].index
    if any(not s.index.equals(first) for s in series):
        raise ValueError("all phase series must be identically indexed")
    per_phase_ann = [float(s.mean()) * periods_per_year for s in series]
    ensemble = pd.concat(series, axis=1).mean(axis=1)
    mean_phase_ann = float(np.mean(per_phase_ann))
    spread = float(max(per_phase_ann) - min(per_phase_ann))
    return PhaseDiagnosticResult(
        n_phases=len(series),
        ensemble_ann=float(ensemble.mean()) * periods_per_year,
        ensemble_sharpe=_sharpe(ensemble, periods_per_year),
        mean_phase_ann=mean_phase_ann,
        min_phase_ann=min(per_phase_ann),
        max_phase_ann=max(per_phase_ann),
        phase_spread_ann=spread,
        degenerate=spread > abs(mean_phase_ann),
    )


def cost_response_curve(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    cost_grid_bps: tuple[float, ...],
    periods_per_year: float,
) -> dict[float, CostResponsePoint]:
    """Target-weight pre-screen cost sensitivity at every one-way rate.

    Pre-screen only; the simulated inventory ledger remains the Research-GO
    PnL. The grid must include the three ``MEASURED_EXECUTION_COST_TIERS_BPS``
    values in addition to any diagnostic literals.
    """
    if not cost_grid_bps:
        raise ValueError("cost_grid_bps must not be empty")
    if any(c < 0 for c in cost_grid_bps):
        raise ValueError("cost rates must be >= 0")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
    out: dict[float, CostResponsePoint] = {}
    for rate in cost_grid_bps:
        net, _turnover = mhs_ledger_pnl(weights, opens, bar_funding, rate)
        sd = float(net.std(ddof=1)) if len(net) > 1 else 0.0
        t_stat = (
            float(net.mean() / sd * math.sqrt(len(net))) if sd > 0 else float("nan")
        )
        out[rate] = CostResponsePoint(
            net_ann=float(net.mean()) * periods_per_year,
            net_sharpe=_sharpe(net, periods_per_year),
            net_t=t_stat,
        )
    return out


def year_restricted_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    years: Sequence[int],
) -> float:
    """Pearson correlation of two return series restricted to calendar years.

    Filters both series to rows whose index year is in ``years``, aligns on the
    intersection, and returns pandas ``.corr()``. An intersection with fewer
    than 3 points returns ``float('nan')`` (a correlation on 1-2 points is not
    meaningfully interpretable and would otherwise surface a spurious +/-1.0) --
    never raises, never returns a made-up value.
    """
    a = series_a[series_a.index.year.isin(years)]
    b = series_b[series_b.index.year.isin(years)]
    joined = pd.concat([a, b], axis=1, join="inner")
    if len(joined) < 3:
        return float("nan")
    return float(joined.corr().iloc[0, 1])


def _event_clusters(
    base: pd.Series, window_bars: int,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Coalesce holding-horizon windows around the top positive bars.

    Seeds are the highest ``ceil(1% * n)`` positive-contribution bars; each
    seed maps to its inclusive ``[i - window, i + window]`` positional window
    and overlapping windows are merged, so adjacent bars from one market move
    become one event rather than independent samples.
    """
    n = len(base)
    if n == 0:
        return [], []
    sorted_desc = base.sort_values(ascending=False)
    positive = sorted_desc[sorted_desc > 0]
    n_seeds = max(1, math.ceil(0.01 * n))
    seeds = list(positive.index[:n_seeds])
    positions = {t: i for i, t in enumerate(base.index)}
    intervals = [
        (max(0, positions[t] - window_bars), min(n - 1, positions[t] + window_bars))
        for t in seeds
    ]
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    contributions = [float(base.iloc[lo : hi + 1].sum()) for lo, hi in merged]
    return merged, contributions


def tail_sensitivity_curve(
    weights: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    turnover: pd.Series,
    one_way_bps: float,
    periods_per_year: float,
    event_window_bars: int,
) -> TailSensitivityResult:
    """Winsor curve + deterministic event-cluster tail diagnostics.

    The winsor curve clips the PER-SYMBOL forward-return panel at each cap in
    (0.50, 0.30, 0.20, 0.10) before aggregating with weights; event clusters
    are computed on the uncapped base net-return series with the frozen
    holding-horizon radius ``event_window_bars``.
    """
    if not weights.index.equals(fwd_returns.index) or list(weights.columns) != list(fwd_returns.columns):
        raise ValueError("weights and fwd_returns must be identically indexed and columned")
    if not weights.index.equals(turnover.index):
        raise ValueError("weights and turnover must share an index")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
    if event_window_bars < 1:
        raise ValueError(f"event_window_bars must be >= 1, got {event_window_bars}")

    def net_from(fwd: pd.DataFrame) -> pd.Series:
        gross = (weights * fwd).sum(axis=1)
        return gross - turnover * (one_way_bps * 1e-4)

    base = net_from(fwd_returns)
    curve: dict[int, tuple[float, float]] = {}
    for cap in (0.50, 0.30, 0.20, 0.10):
        capped = net_from(fwd_returns.clip(-cap, cap))
        curve[int(cap * 100)] = (
            float(capped.mean() * periods_per_year),
            _sharpe(capped, periods_per_year),
        )

    total = float(base.sum())
    _intervals, contributions = _event_clusters(base, event_window_bars)
    event_count = len(contributions)
    if not contributions or total == 0:
        top1 = top5 = top1pct = float("nan")
    else:
        ordered = sorted(contributions, reverse=True)
        n_top1pct = max(1, math.ceil(0.01 * event_count))
        top1 = ordered[0] / total
        top5 = float(np.sum(ordered[:5])) / total
        top1pct = float(np.sum(ordered[:n_top1pct])) / total

    worst_idx = int(np.argmin(base.to_numpy(dtype="float64"))) if len(base) else 0
    n = len(base)
    lo = max(0, worst_idx - event_window_bars)
    hi = min(n - 1, worst_idx + event_window_bars)
    mask = np.ones(n, dtype=bool)
    mask[lo : hi + 1] = False
    without_worst = base.iloc[mask]

    return TailSensitivityResult(
        base_net_ann=float(base.mean() * periods_per_year),
        base_sharpe=_sharpe(base, periods_per_year),
        winsor_curve=curve,
        event_window_bars=event_window_bars,
        event_count=event_count,
        top1_event_share=top1,
        top5_event_share=top5,
        top1pct_events_share=top1pct,
        leave_worst_event_out_sharpe=_sharpe(without_worst, periods_per_year),
    )


def book_evidence(
    weights_1h: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    cost_grid_bps: tuple[float, ...],
    periods_per_year: float,
    event_window_bars: int,
    tail_one_way_bps: float = 8.0,
) -> BookEvidence:
    """Pre-screen + tail significance evidence for one weight book.

    Pure extraction of the inline construction the MHS orchestrator previously
    reserved for the reference (zero-capital) book, so the identical instruments
    can also point at the executed book (roster + ensemble + tilt + regime
    scale) that actually carries capital. ``cost_response_curve``,
    ``tail_sensitivity_curve``, and ``mhs_ledger_pnl`` are reused unchanged; no
    statistic is reimplemented. Intermediates are released before returning to minimize resident memory.
    """
    if not cost_grid_bps:
        raise ValueError("cost_grid_bps must not be empty")
    if any(c < 0 for c in cost_grid_bps):
        raise ValueError("cost rates must be >= 0")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
    if event_window_bars < 1:
        raise ValueError(f"event_window_bars must be >= 1, got {event_window_bars}")
    if tail_one_way_bps < 0:
        raise ValueError(f"tail_one_way_bps must be >= 0, got {tail_one_way_bps}")

    prescreen = cost_response_curve(
        weights_1h, opens, bar_funding, cost_grid_bps, periods_per_year,
    )

    effective_weights = weights_1h.shift(2).fillna(0.0)
    fwd = opens.pct_change()
    _net, turnover = mhs_ledger_pnl(weights_1h, opens, bar_funding, tail_one_way_bps)
    tail = tail_sensitivity_curve(
        effective_weights, fwd, turnover, tail_one_way_bps, periods_per_year, event_window_bars,
    )
    del effective_weights, fwd, _net, turnover
    return BookEvidence(prescreen=prescreen, tail=tail)


def _bartlett_weighted_acf_sum(demeaned: np.ndarray, var: float, max_lag: int) -> float:
    """Bartlett-weighted lag autocorrelation sum over ``k = 1..max_lag``.

    Shared by the autocorrelation-adjusted Sharpe denominator and the
    effective-sample-size estimator so both statistics penalize exactly the
    same serial-dependence estimate.
    """
    n = len(demeaned)
    acf_sum = 0.0
    for k in range(1, max_lag + 1):
        rho = float(np.dot(demeaned[k:], demeaned[: n - k])) / var if var > 0 else 0.0
        acf_sum += (1.0 - k / (max_lag + 1)) * rho
    return acf_sum


def autocorrelation_adjusted_sharpe(
    daily_net_returns: pd.Series,
    annualization_days: int = 365,
    max_lag_days: int = 7,
) -> float:
    """Daily-compounded annualized Sharpe with the frozen autocorrelation adjustment.

    Divides the annualized sample Sharpe by
    ``sqrt(1 + 2 * sum((1 - k/(max_lag_days + 1)) * rho_k))`` over lags
    ``k = 1..max_lag_days``. Phase 1 fixes ``annualization_days=365`` and
    ``max_lag_days=7`` (the longest frozen holding horizon is 168h).
    """
    if daily_net_returns.index.tz is None:
        raise ValueError("daily_net_returns must be tz-aware")
    if not daily_net_returns.index.is_monotonic_increasing:
        raise ValueError("daily_net_returns must be monotonic in time")
    if len(daily_net_returns) < max_lag_days + 2:
        raise ValueError(
            f"need at least {max_lag_days + 2} observations, got {len(daily_net_returns)}"
        )
    if annualization_days < 1:
        raise ValueError(f"annualization_days must be >= 1, got {annualization_days}")
    if max_lag_days < 1:
        raise ValueError(f"max_lag_days must be >= 1, got {max_lag_days}")

    mean = float(daily_net_returns.mean())
    std = float(daily_net_returns.std(ddof=1))
    if _zero_variance(std, mean):
        if mean > 0:
            return float("inf")
        if mean < 0:
            return float("-inf")
        return float("nan")
    sample_sharpe = mean / std * math.sqrt(annualization_days)
    demeaned = daily_net_returns.to_numpy(dtype="float64") - mean
    var = float(np.dot(demeaned, demeaned))
    acf_sum = _bartlett_weighted_acf_sum(demeaned, var, max_lag_days)
    denom = max(1.0 + 2.0 * acf_sum, 1e-12)
    return sample_sharpe / math.sqrt(denom)


def effective_observation_count(returns: pd.Series, max_lag: int = 24) -> int:
    """Effective sample size under the Bartlett-weighted autocorrelation sum.

    ``n_eff = n / (1 + 2 * sum((1 - k/(max_lag+1)) * rho_k))`` over lags
    ``k = 1..max_lag``, sharing the exact autocorrelation estimate of
    ``autocorrelation_adjusted_sharpe`` so both statistics penalize the same
    serial dependence. The result is clipped to ``[1, n]``: negative
    autocorrelation can never inflate the sample above the raw count. Raises
    ``ValueError`` when ``max_lag < 1`` or fewer than ``max_lag + 2``
    observations are supplied.
    """
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")
    n = len(returns)
    if n < max_lag + 2:
        raise ValueError(f"need at least {max_lag + 2} observations, got {n}")
    arr = returns.to_numpy(dtype="float64")
    mean = float(arr.mean())
    demeaned = arr - mean
    var = float(np.dot(demeaned, demeaned))
    denom = 1.0 + 2.0 * _bartlett_weighted_acf_sum(demeaned, var, max_lag)
    n_eff = int(n / denom) if denom > 0.0 else 0
    return int(min(max(n_eff, 1), n))


def _psr_radicand(observed_sr: float, skew: float, kurtosis: float) -> float:
    """PSR variance correction ``1 - skew*SR + ((kurt-1)/4)*SR^2`` (shared form)."""
    return 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr**2


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado): normal CDF of the
    probability that the true (non-annualized) Sharpe exceeds ``benchmark_sr``.

    All Sharpe inputs are per-observation (non-annualized): the raw ``mean/std``
    of the return series, never scaled by ``sqrt(periods_per_year)``.
    ``kurtosis`` is the full (non-excess) fourth standardized moment, so pass
    ``excess_kurtosis + 3.0``. Returns NaN (never ``inf`` or a complex value)
    when the denominator radicand is not strictly positive.
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}")
    radicand = _psr_radicand(observed_sr, skew, kurtosis)
    if radicand <= 0.0:
        return float("nan")
    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1.0) / math.sqrt(radicand)
    return float(norm.cdf(z))


def _expected_max_trial_sr(trial_sr_variance: float, n_trials: int) -> float:
    """Expected maximum Sharpe over ``n_trials`` independent zero-edge trials."""
    sd = math.sqrt(trial_sr_variance)
    return float(
        sd
        * (
            (1.0 - _EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials)
            + _EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        )
    )


def deflated_sharpe_ratio(
    observed_sr: float,
    trial_sr_variance: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado): PSR against the expected
    maximum Sharpe over ``n_trials`` independent trials under the null.

    The benchmark is ``sqrt(trial_sr_variance) * ((1 - gamma) * Phi_inv(1 - 1/N)
    + gamma * Phi_inv(1 - 1/(N*e)))`` with ``gamma`` the Euler-Mascheroni
    constant. All Sharpe inputs are per-observation (non-annualized);
    ``trial_sr_variance`` is the variance of the per-observation Sharpe across
    trials. With zero trial dispersion the benchmark collapses to zero and the
    result equals the plain PSR against a zero benchmark.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}")
    if trial_sr_variance < 0.0:
        raise ValueError(
            f"trial_sr_variance must be >= 0, got {trial_sr_variance}"
        )
    if trial_sr_variance == 0.0:
        return probabilistic_sharpe_ratio(observed_sr, 0.0, n_obs, skew, kurtosis)
    benchmark_sr = _expected_max_trial_sr(trial_sr_variance, n_trials)
    return probabilistic_sharpe_ratio(observed_sr, benchmark_sr, n_obs, skew, kurtosis)


TRIAL_SHARPE_DEDUP_DECIMALS: int = 6
MIN_TRIAL_SHARPE_OUTCOMES: int = 8


def distinct_trial_sr_variance(
    trial_sharpes: Sequence[float],
    observed_sr: float,
    *,
    dedup_decimals: int = TRIAL_SHARPE_DEDUP_DECIMALS,
    min_outcomes: int = MIN_TRIAL_SHARPE_OUTCOMES,
) -> float:
    """ddof=1 variance over the DISTINCT pooled per-observation trial Sharpes.

    The pool is ``[observed_sr] + every finite element of trial_sharpes`` --
    the current run's own Sharpe is always a member -- deduplicated by
    ``round(x, dedup_decimals)`` so bit-identical re-runs of one configuration
    can neither shrink V nor game the deflation benchmark. Raises ValueError
    naming both counts when fewer than ``min_outcomes`` distinct outcomes
    survive; callers must fail closed rather than substitute another variance.
    """
    pooled = [float(observed_sr)]
    for sharpe in trial_sharpes:
        value = float(sharpe)
        if np.isfinite(value):
            pooled.append(value)
    distinct = sorted({round(value, dedup_decimals) for value in pooled})
    if len(distinct) < min_outcomes:
        raise ValueError(
            f"only {len(distinct)} distinct trial Sharpe outcomes pooled "
            f"(observed_sr included), need at least {min_outcomes}"
        )
    return float(np.var(np.asarray(distinct, dtype="float64"), ddof=1))


@dataclass(frozen=True, slots=True)
class DsrDecomposition:
    """Descriptive decomposition of one deflated Sharpe Ratio evaluation.

    Purely observational -- it never feeds back into the reported DSR value.
    ``margin = observed_sr - benchmark_sr``, ``radicand`` is the PSR variance
    correction. ``fold_sharpes`` holds the per-fold Sharpes of the replayed
    anchored folds (observational context only: the benchmark dispersion now
    comes from the distinct trial outcomes recorded in the run history).
    ``n_obs_effective_holding_horizon`` is the lag-168h sensitivity reading of
    the effective sample size, published so no reader has to rediscover it.
    """

    observed_sr: float
    benchmark_sr: float
    margin: float
    trial_sr_sqrt_variance: float
    n_obs_raw: int
    n_obs_effective: int
    radicand: float
    n_trials: int
    fold_sharpes: tuple[float, ...]
    trial_sr_source: str = "distinct_trial_outcomes"
    distinct_trial_outcomes: int = 0
    n_obs_effective_holding_horizon: int = 0


def deflated_sharpe_decomposition(
    observed_sr: float,
    trial_sr_variance: float,
    n_trials: int,
    n_obs_raw: int,
    n_obs_effective: int,
    skew: float,
    kurtosis: float,
    fold_sharpes: tuple[float, ...],
    *,
    trial_sr_source: str = "distinct_trial_outcomes",
    distinct_trial_outcomes: int = 0,
    n_obs_effective_holding_horizon: int = 0,
) -> DsrDecomposition:
    """Decompose one DSR evaluation into its registered-formula components.

    The benchmark and radicand are recomputed with exactly the expressions
    ``deflated_sharpe_ratio`` uses, so
    ``norm.cdf(margin * sqrt(n_obs_effective - 1) / sqrt(radicand))``
    reproduces the reported statistic. Validation mirrors
    ``deflated_sharpe_ratio`` for both sample counts.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_obs_raw < 2:
        raise ValueError(f"n_obs_raw must be >= 2, got {n_obs_raw}")
    if n_obs_effective < 2:
        raise ValueError(f"n_obs_effective must be >= 2, got {n_obs_effective}")
    if trial_sr_variance < 0.0:
        raise ValueError(
            f"trial_sr_variance must be >= 0, got {trial_sr_variance}"
        )
    benchmark_sr = (
        0.0
        if trial_sr_variance == 0.0
        else _expected_max_trial_sr(trial_sr_variance, n_trials)
    )
    margin = observed_sr - benchmark_sr
    return DsrDecomposition(
        observed_sr=observed_sr,
        benchmark_sr=benchmark_sr,
        margin=margin,
        trial_sr_sqrt_variance=math.sqrt(trial_sr_variance),
        n_obs_raw=n_obs_raw,
        n_obs_effective=n_obs_effective,
        radicand=_psr_radicand(observed_sr, skew, kurtosis),
        n_trials=n_trials,
        fold_sharpes=tuple(fold_sharpes),
        trial_sr_source=trial_sr_source,
        distinct_trial_outcomes=distinct_trial_outcomes,
        n_obs_effective_holding_horizon=n_obs_effective_holding_horizon,
    )


# Causal regime stratification constants (report-only evidence, never gates):
# one-year trailing high for the BTC drawdown state, 30-day window for the
# book realized vol, both labelled strictly backward-looking.
_REGIME_TRAILING_HIGH_HOURS: int = 8760
_REGIME_TRAILING_HIGH_MIN_HOURS: int = 24
_REGIME_VOL_LOOKBACK_HOURS: int = 720
_REGIME_VOL_MIN_HOURS: int = 24
_REGIME_TERCILE_MIN_HOURS: int = 48
_BTC_DRAWDOWN_BULL_FLOOR: float = -0.20
_BTC_DRAWDOWN_BEAR_FLOOR: float = -0.50


def causal_regime_labels(returns: pd.Series, btc_close: pd.Series) -> pd.DataFrame:
    """Strictly causal per-bar regime labels over an hourly return series.

    The BTC drawdown state divides ``btc_close`` -- an independent, exogenous
    price level reindexed to ``returns.index`` -- by its own trailing rolling
    max **shifted by one bar**; the book trailing-vol terciles compare the
    trailing realized vol of ``returns`` against expanding-window quantiles
    **shifted by one bar**. Deriving the drawdown state from the book's own
    equity would make "Sharpe conditional on our own drawdown" close to
    tautological; ``btc_close`` keeps the two regime axes independent. No
    full-sample statistic enters any label, so each label at index t is
    bit-identical whether computed on the prefix ``[:t+1]`` of both series or
    the full series. Warm-up rows carry an empty string (unlabelled); a
    non-finite ``btc_close`` bar carries an unlabelled drawdown state only.
    """
    btc_level = btc_close.reindex(returns.index).astype("float64")
    trailing_high = (
        btc_level.rolling(
            _REGIME_TRAILING_HIGH_HOURS, min_periods=_REGIME_TRAILING_HIGH_MIN_HOURS
        ).max().shift(1)
    )
    drawdown = btc_level / trailing_high - 1.0
    bull = (drawdown > _BTC_DRAWDOWN_BULL_FLOOR).fillna(False).to_numpy(dtype=bool)
    bear = (drawdown < _BTC_DRAWDOWN_BEAR_FLOOR).fillna(False).to_numpy(dtype=bool)
    states: np.ndarray = np.full(len(returns), "", dtype=object)
    states[bull] = "bull"
    states[~bull & ((drawdown >= _BTC_DRAWDOWN_BEAR_FLOOR).fillna(False).to_numpy(dtype=bool))] = "correction"
    states[bear] = "bear"

    realized_vol = (
        returns.rolling(_REGIME_VOL_LOOKBACK_HOURS, min_periods=_REGIME_VOL_MIN_HOURS)
        .std().shift(1)
    )
    quantile_low = (
        realized_vol.expanding(min_periods=_REGIME_TERCILE_MIN_HOURS)
        .quantile(1.0 / 3.0).shift(1)
    )
    quantile_high = (
        realized_vol.expanding(min_periods=_REGIME_TERCILE_MIN_HOURS)
        .quantile(2.0 / 3.0).shift(1)
    )
    is_low = (realized_vol <= quantile_low).fillna(False).to_numpy(dtype=bool)
    is_mid = (
        (realized_vol > quantile_low) & (realized_vol <= quantile_high)
    ).fillna(False).to_numpy(dtype=bool)
    is_high = (realized_vol > quantile_high).fillna(False).to_numpy(dtype=bool)
    terciles: np.ndarray = np.full(len(returns), "", dtype=object)
    terciles[is_low] = "low"
    terciles[is_mid] = "mid"
    terciles[is_high] = "high"

    return pd.DataFrame(
        {"btc_drawdown_state": states, "book_vol_tercile": terciles},
        index=returns.index,
    )


def regime_conditional_sharpe_blocks(
    returns: pd.Series, btc_close: pd.Series,
) -> dict[str, dict[str, float]]:
    """Causal regime stratification of hourly returns (observational only).

    Consumes :func:`causal_regime_labels` and reports, per block, ``n_hours``,
    naive annualized ``sharpe`` and ``ann_vol``. Warm-up hours remain
    unlabelled, so the summed ``n_hours`` of one labelling never exceeds
    ``len(returns)``.
    """
    if len(returns) == 0:
        return {}
    labels = causal_regime_labels(returns, btc_close)
    values = returns.to_numpy(dtype="float64")
    annualization = math.sqrt(PERIODS_PER_YEAR_1H)
    blocks: dict[str, dict[str, float]] = {}
    for column, prefix in (
        ("btc_drawdown_state", "btc_drawdown_"),
        ("book_vol_tercile", "book_vol_"),
    ):
        column_values = labels[column].to_numpy()
        for label in sorted({value for value in column_values if value}):
            sample = values[column_values == label]
            count = int(sample.size)
            mean = float(sample.mean()) if count else 0.0
            sd = float(sample.std(ddof=1)) if count >= 2 else 0.0
            blocks[f"{prefix}{label}"] = {
                "n_hours": count,
                "sharpe": float(mean / sd * annualization) if sd > 0.0 else 0.0,
                "ann_vol": float(sd * annualization),
            }
    return blocks


@dataclass(frozen=True, slots=True)
class AnchoredPurgedFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    forward_dependency_hours: int
    purge_hours: int

    def __post_init__(self) -> None:
        bounds = (self.train_start, self.train_end, self.validation_start, self.validation_end)
        if any(b.tzinfo is None for b in bounds):
            raise ValueError("fold bounds must be tz-aware")
        if not (
            self.train_start < self.train_end < self.validation_start < self.validation_end
        ):
            raise ValueError("fold bounds must be strictly ascending")
        if self.forward_dependency_hours < 1:
            raise ValueError("forward_dependency_hours must be >= 1")
        if self.purge_hours < self.forward_dependency_hours:
            raise ValueError("purge_hours must be >= forward_dependency_hours")
        if self.validation_start <= self.train_end:
            raise ValueError("the purge embargo must be positive")


def phase_1_anchored_purged_folds() -> tuple[AnchoredPurgedFold, ...]:
    """The four preregistered Level 2 anchored purged folds.

    ``purge_hours`` derives from the maximum forward dependency (frozen at
    168h for Phase 1); these folds are internal historical robustness, never
    labelled OOS.  The 2022 fold is the only bear-regime fold available and
    adds leak-free evidence; the concentration gate denominator widens from 3
    to 4.
    """
    purge = 168
    return (
        AnchoredPurgedFold(
            DISCOVERY_START,
            pd.Timestamp("2021-12-31", tz="UTC"),
            pd.Timestamp("2022-01-08", tz="UTC"),
            pd.Timestamp("2022-12-31", tz="UTC"),
            168,
            purge,
        ),
        AnchoredPurgedFold(
            DISCOVERY_START,
            pd.Timestamp("2022-12-31", tz="UTC"),
            pd.Timestamp("2023-01-08", tz="UTC"),
            pd.Timestamp("2023-12-31", tz="UTC"),
            168,
            purge,
        ),
        AnchoredPurgedFold(
            DISCOVERY_START,
            pd.Timestamp("2023-12-31", tz="UTC"),
            pd.Timestamp("2024-01-08", tz="UTC"),
            pd.Timestamp("2024-12-31", tz="UTC"),
            168,
            purge,
        ),
        AnchoredPurgedFold(
            DISCOVERY_START,
            pd.Timestamp("2024-12-31", tz="UTC"),
            pd.Timestamp("2025-01-08", tz="UTC"),
            pd.Timestamp("2025-12-31", tz="UTC"),
            168,
            purge,
        ),
    )


@dataclass(frozen=True, slots=True)
class SyntheticStressScenario:
    name: str
    description: str


def synthetic_stress_scenarios() -> tuple[SyntheticStressScenario, ...]:
    """The nine deterministic, parameter-free preregistered stress scenarios.

    Distinct from historical event removal. Unsupported components are reported
    rather than fabricated from OHLCV, and no scenario may tune Phase 1
    parameters.
    """
    return (
        SyntheticStressScenario("BTC_DOWN_10", "BTC spot falls 10% intraday"),
        SyntheticStressScenario("BTC_DOWN_20", "BTC spot falls 20% intraday"),
        SyntheticStressScenario("ALT_BETA_UP", "altcoin beta to BTC spikes"),
        SyntheticStressScenario("XS_CORRELATION_ONE", "cross-sectional correlation equals 1"),
        SyntheticStressScenario("SPREAD_AND_COST_X3", "spread and costs triple"),
        SyntheticStressScenario("PASSIVE_FILL_DEGRADATION", "passive fill rate degrades"),
        SyntheticStressScenario("FUNDING_EXTREME", "funding rate reaches an extreme"),
        SyntheticStressScenario(
            "LIQUIDITY_DETERIORATION_50PCT", "50% of symbols face liquidity deterioration",
        ),
        SyntheticStressScenario("VENUE_API_OUTAGE_30M", "venue/API outage of 30 minutes"),
    )


@dataclass(frozen=True, slots=True)
class DeploymentReadinessResult:
    geometric_cagr: float
    max_drawdown: float
    calmar: float
    expected_shortfall: float
    worst_1d: float
    worst_7d: float
    worst_event: float
    time_under_water_bars: int
    recovery_bars: int | None
    probability_final_wealth_below_initial: float
    probability_mdd_over_20pct: float
    probability_mdd_over_30pct: float
    leverage_ruin_probabilities: Mapping[float, float]
    concentration: Mapping[str, float]
    participation_warnings: Mapping[str, float]
    research_go_eligible: bool
    execution_go_eligible: bool
    pilot_go_eligible: bool
    scale_go_eligible: bool


def _max_drawdown_from_equity(equity: pd.Series) -> float:
    running_max = equity.cummax()
    return float((equity / running_max - 1.0).min())


def _time_under_water(equity: pd.Series) -> int:
    running_max = equity.cummax()
    underwater = (equity < running_max).to_numpy()
    if not underwater.any():
        return 0
    # Vectorized run-length encoding of contiguous True runs: diff over a
    # zero-padded boolean array marks each run start (+1) and end (-1).
    padded = np.concatenate(([0], underwater.astype(np.int8), [0]))
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int((ends - starts).max())


def _recovery_bars(equity: pd.Series) -> int | None:
    if equity.empty:
        return None
    running_max = equity.cummax()
    ratio = equity / running_max
    trough_idx = int(np.argmin(ratio.to_numpy(dtype="float64")))
    peak_before = running_max.iloc[trough_idx]
    after = equity.iloc[trough_idx:]
    recovered = after[after >= peak_before]
    if len(recovered):
        return int(recovered.index.get_loc(recovered.index[0]))
    return None


def _bootstrap_chunk_size(n: int) -> int:
    """Per-chunk replicate count keeping the (chunk, n) sample matrix bounded.

    The vectorized bootstrap materialises one ``(chunk, n)`` float64 sample
    matrix plus a few same-shaped temporaries per chunk.  At production scale
    the equity is the 5-minute grid (~525,600 bars), so a fixed ``chunk=500``
    would allocate ~2.1GB per array and spike RSS far above the 8GB soft
    budget.  ``chunk`` is capped to keep a single sample matrix <= 128MB
    (spec O10); the MDD path also allocates a same-sized running-max temporary,
    so a 128MB sample keeps the combined transient ~1.1GB.
    """
    if n <= 0:
        return 500
    return max(1, min(500, int((128 * 2**20) // (n * 8))))


def _stationary_block_bootstrap_paths(
    net_returns: np.ndarray, n_replicates: int, mean_block: int, seed: int,
) -> np.ndarray:
    """Wealth multipliers (final wealth / initial) per replicate, vectorized.

    Statistically equivalent to the scalar while-loop block composition
    (PERF_OPT_003 precedent): block lengths are ``geometric(p_block)`` and
    block starts are uniform, matching the scalar length law.  A 6x block-count
    safety margin makes running short effectively impossible; any shortfall
    still falls back to the scalar replicate path.
    """
    if n_replicates <= 0:
        raise ValueError(f"n_replicates must be > 0, got {n_replicates}")
    if mean_block < 0:
        raise ValueError(f"mean_block must be >= 0, got {mean_block}")
    rng = np.random.default_rng(seed)
    n = len(net_returns)
    if n == 0:
        return np.ones(n_replicates)
    p_block = 1.0 / mean_block if mean_block > 0 else 0.0
    if p_block <= 0.0:
        # Degenerate mean_block == 0: every block is a single element (the
        # scalar ``while length < n and rng.random() > 0`` never advances).
        starts = rng.integers(0, n, size=n_replicates)
        return np.asarray(np.prod(1.0 + net_returns[starts], axis=0), dtype="float64")
    outcomes = np.empty(n_replicates, dtype="float64")
    chunk = _bootstrap_chunk_size(n)
    for r0 in range(0, n_replicates, chunk):
        r1 = min(r0 + chunk, n_replicates)
        k = r1 - r0
        max_blocks = min(n, int(np.ceil(n * 6.0 / mean_block)) + 16)
        lengths = rng.geometric(p_block, size=(k, max_blocks))
        starts = rng.integers(0, n, size=(k, max_blocks))
        ends = np.cumsum(lengths, axis=1)
        short = ends[:, -1] < n
        for r in np.flatnonzero(short).tolist():
            outcomes[r0 + r] = _block_bootstrap_replicate_wealth(
                net_returns, n, p_block, rng,
            )
        valid = ~short
        if valid.any():
            ends_trunc = np.minimum(ends, n)
            used = ends_trunc - np.concatenate(
                [np.zeros((k, 1), dtype=np.int64), ends_trunc[:, :-1]], axis=1,
            )
            u = used[valid].ravel()
            s = starts[valid].ravel()
            keep = u > 0
            u = u[keep]
            s = s[keep]
            block_start = np.cumsum(u) - u
            offsets = np.arange(int(u.sum()), dtype=np.int64) - np.repeat(block_start, u)
            arr_idx = (np.repeat(s, u) + offsets) % n
            sample = net_returns[arr_idx].reshape(int(valid.sum()), n)
            outcomes[r0 + np.flatnonzero(valid)] = np.prod(1.0 + sample, axis=1)
    return outcomes


def _block_bootstrap_replicate_wealth(
    arr: np.ndarray, n: int, p_block: float, rng: np.random.Generator,
) -> float:
    """Wealth multiplier of one scalar block-bootstrap replicate (fallback)."""
    blocks: list[float] = []
    while len(blocks) < n:
        start = int(rng.integers(0, n))
        length = 1
        while length < n and rng.random() > p_block:
            length += 1
        length = min(length, n - len(blocks))
        blocks.extend(arr[start : start + length].tolist())
    path = np.array(blocks[:n], dtype="float64")
    return float(np.prod(1.0 + path))


def _bootstrap_mdd_paths(
    net_returns: np.ndarray, n_replicates: int, mean_block: int, seed: int,
) -> np.ndarray:
    """Per-replicate max drawdown of the block-bootstrap equity path, vectorized."""
    if n_replicates <= 0:
        raise ValueError(f"n_replicates must be > 0, got {n_replicates}")
    if mean_block < 0:
        raise ValueError(f"mean_block must be >= 0, got {mean_block}")
    rng = np.random.default_rng(seed + 1)
    n = len(net_returns)
    if n == 0:
        return np.zeros(n_replicates)
    p_block = 1.0 / mean_block if mean_block > 0 else 0.0
    if p_block <= 0.0:
        starts = rng.integers(0, n, size=n_replicates)
        equity = np.cumprod(1.0 + net_returns[starts], axis=0)
        running_max = np.maximum.accumulate(equity, axis=0)
        return np.asarray((equity / running_max - 1.0).min(axis=0), dtype="float64")
    mdd = np.empty(n_replicates, dtype="float64")
    chunk = _bootstrap_chunk_size(n)
    for r0 in range(0, n_replicates, chunk):
        r1 = min(r0 + chunk, n_replicates)
        k = r1 - r0
        max_blocks = min(n, int(np.ceil(n * 6.0 / mean_block)) + 16)
        lengths = rng.geometric(p_block, size=(k, max_blocks))
        starts = rng.integers(0, n, size=(k, max_blocks))
        ends = np.cumsum(lengths, axis=1)
        short = ends[:, -1] < n
        for r in np.flatnonzero(short).tolist():
            mdd[r0 + r] = _block_bootstrap_replicate_mdd(
                net_returns, n, p_block, rng,
            )
        valid = ~short
        if valid.any():
            ends_trunc = np.minimum(ends, n)
            used = ends_trunc - np.concatenate(
                [np.zeros((k, 1), dtype=np.int64), ends_trunc[:, :-1]], axis=1,
            )
            u = used[valid].ravel()
            s = starts[valid].ravel()
            keep = u > 0
            u = u[keep]
            s = s[keep]
            block_start = np.cumsum(u) - u
            offsets = np.arange(int(u.sum()), dtype=np.int64) - np.repeat(block_start, u)
            arr_idx = (np.repeat(s, u) + offsets) % n
            sample = net_returns[arr_idx].reshape(int(valid.sum()), n)
            sample += 1.0
            np.cumprod(sample, axis=1, out=sample)
            running_max = np.maximum.accumulate(sample, axis=1)
            mdd[r0 + np.flatnonzero(valid)] = (sample / running_max - 1.0).min(axis=1)
    return mdd


def _block_bootstrap_replicate_mdd(
    arr: np.ndarray, n: int, p_block: float, rng: np.random.Generator,
) -> float:
    """Max drawdown of one scalar block-bootstrap replicate (fallback)."""
    blocks: list[float] = []
    while len(blocks) < n:
        start = int(rng.integers(0, n))
        length = 1
        while length < n and rng.random() > p_block:
            length += 1
        length = min(length, n - len(blocks))
        blocks.extend(arr[start : start + length].tolist())
    path = np.array(blocks[:n], dtype="float64")
    equity = np.cumprod(1.0 + path)
    running_max = np.maximum.accumulate(equity)
    return float((equity / running_max - 1.0).min())


def compute_deployment_readiness(
    equity: pd.Series,
    periods_per_year: float,
    concentration: Mapping[str, float] | None = None,
    participation_warnings: Mapping[str, float] | None = None,
    primary_valid: bool = True,
    research_go_eligible: bool | None = None,
    mean_block_bars: int = 168,
    n_bootstrap: int = 2000,
    seed: int = 20260807,
    leverage_grid: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> DeploymentReadinessResult:
    """Research/execution/pilot/scale readiness from strict-proxy simulated equity.

    Research GO can be true from historical strict-proxy evidence only when the
    primary ledger is valid (``primary_valid``). When ``research_go_eligible``
    is passed, it is the explicit gate decision and overrides the ``primary_valid``
    shortcut entirely, so a caller can route a fold-based Research-GO decision
    without weakening the fail-closed default. Execution, Pilot, and Scale GO
    require forward evidence that is absent by construction until the respective
    calibration data exists.
    """
    if equity.index.tz is None:
        raise ValueError("equity must be tz-aware")
    if not equity.index.is_monotonic_increasing:
        raise ValueError("equity must be monotonic in time")
    if (equity <= 0).any():
        raise ValueError("equity must be strictly positive")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")
    if not isinstance(primary_valid, bool):
        raise ValueError("primary_valid must be a bool")
    if research_go_eligible is not None and not isinstance(research_go_eligible, bool):
        raise ValueError("research_go_eligible must be a bool or None")
    research_go = primary_valid if research_go_eligible is None else research_go_eligible

    net = equity.pct_change().dropna()
    if len(net) == 0:
        raise ValueError("equity must contain at least two observations")
    final = float(equity.iloc[-1])
    initial = float(equity.iloc[0])
    n = len(net)
    geometric_cagr = (final / initial) ** (periods_per_year / n) - 1.0
    max_drawdown = _max_drawdown_from_equity(equity)
    calmar = geometric_cagr / abs(max_drawdown) if max_drawdown < 0 else float("nan")
    sorted_net = np.sort(net.to_numpy(dtype="float64"))
    tail_count = max(1, math.ceil(0.05 * n))
    expected_shortfall = float(np.mean(sorted_net[:tail_count]))
    worst_1d = float(net.min())

    k7 = min(7, n)
    # Vectorized sliding-window 7-bar sum: identical arithmetic to the scalar
    # slice loop (each window is summed in the same element order).
    if k7 > 1:
        from numpy.lib.stride_tricks import sliding_window_view

        worst_7d = float(
            sliding_window_view(net.to_numpy(dtype="float64"), k7).sum(axis=1).min()
        )
    else:
        worst_7d = worst_1d
    worst_event = worst_7d if k7 > 1 else worst_1d

    net_arr = net.to_numpy(dtype="float64")
    wealth = _stationary_block_bootstrap_paths(net_arr, n_bootstrap, mean_block_bars, seed)
    mdd_paths = _bootstrap_mdd_paths(net_arr, n_bootstrap, mean_block_bars, seed)
    probability_final_wealth_below_initial = float(np.mean(wealth < 1.0))
    probability_mdd_over_20pct = float(np.mean(mdd_paths < -0.20))
    probability_mdd_over_30pct = float(np.mean(mdd_paths < -0.30))

    # The scalar leverage-ruin loop recomputed the identical deterministic
    # leveraged path ``n_bootstrap`` times; a single cumulative-product check
    # per leverage is exactly equivalent (ruin_probs[lev] is 0.0 or 1.0).
    ruin_probs: dict[float, float] = {}
    for lev in leverage_grid:
        ruin_probs[lev] = float(
            bool((np.cumprod(1.0 + lev * net_arr) <= 0.0).any())
        )

    return DeploymentReadinessResult(
        geometric_cagr=geometric_cagr,
        max_drawdown=max_drawdown,
        calmar=calmar,
        expected_shortfall=expected_shortfall,
        worst_1d=worst_1d,
        worst_7d=worst_7d,
        worst_event=worst_event,
        time_under_water_bars=_time_under_water(equity),
        recovery_bars=_recovery_bars(equity),
        probability_final_wealth_below_initial=probability_final_wealth_below_initial,
        probability_mdd_over_20pct=probability_mdd_over_20pct,
        probability_mdd_over_30pct=probability_mdd_over_30pct,
        leverage_ruin_probabilities=ruin_probs,
        concentration=dict(concentration or {}),
        participation_warnings=dict(participation_warnings or {}),
        research_go_eligible=research_go,
        execution_go_eligible=False,
        pilot_go_eligible=False,
        scale_go_eligible=False,
    )


def required_cost_tiers() -> tuple[float, ...]:
    """The three named measured execution tiers, always reported together."""
    return tuple(
        MEASURED_EXECUTION_COST_TIERS_BPS[tier]
        for tier in ("optimistic", "base", "stress")
    )
