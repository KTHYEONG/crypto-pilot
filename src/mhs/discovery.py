"""Discovery/qualification horizon-selection gate (spec §2).

The gate picks one horizon candidate on a worst-year-robust discovery score and
re-confirms that single candidate on a disjoint qualification window, reusing
the project's existing ``horizon_log_return`` / ``rank_weight_book`` /
``phase_tranche_book`` / ``cost_response_curve`` building blocks and the
existing ``DISCOVERY_END`` / ``QUALIFICATION_START`` / ``QUALIFICATION_END``
date convention (``src/research/technical_experts/trend_screen_catalog.py``).
No new algorithm is introduced and no aggregation over the candidate grid
happens on the qualification window (a re-scan there would be a one-step-later
p-hack).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.mhs.books import phase_tranche_book, rank_weight_book
from src.mhs.contracts import MEASURED_EXECUTION_COST_TIERS_BPS
from src.mhs.evaluation import AnchoredPurgedFold, cost_response_curve
from src.mhs.execution import mhs_ledger_pnl
from src.mhs.horizons import horizon_log_return, realized_vol, vol_normalized_horizon_signal

_PERIODS_PER_YEAR_1H = 365.0 * 24.0
_ADMISSION_T = 2.0


@dataclass(frozen=True, slots=True)
class DiscoveryQualificationResult:
    """Outcome of one discovery/qualification gate run for one sign family.

    ``selected_horizon`` is the discovery-selected candidate (None only when no
    candidate's worst-year score clears the admission floor). ``admitted`` is
    True only when a candidate was selected AND the single-candidate
    qualification re-check cleared |t| >= 2.0 with the same sign. Discovery
    scores are ``(horizon, worst-year net_t)`` pairs; the qualification
    aggregate ``net_t`` and sign-consistency flag are ``None`` when the gate
    failed closed before any qualification evaluation. ``yearly_net_t`` maps
    each candidate horizon to its full per-calendar-year ``(year, net_t)``
    series -- including non-finite years -- the raw values discovery_scores'
    worst-year min was computed from.
    """

    selected_horizon: int | None
    admitted: bool
    discovery_scores: tuple[tuple[int, float], ...]
    discovery_aggregate_net_t: float | None
    qualification_net_t: float | None
    qualification_sign_consistent: bool | None
    yearly_net_t: tuple[tuple[int, tuple[tuple[int, float], ...]], ...] = ()
    yearly_adjusted_net_t: tuple[tuple[int, tuple[tuple[int, float], ...]], ...] = ()
    discovery_scores_adjusted: tuple[tuple[int, float], ...] = ()
    discovery_aggregate_adjusted_net_t: float | None = None
    qualification_adjusted_net_t: float | None = None
    adjusted_admitted: bool | None = None
    yearly_regime_scaled_net_t: tuple[tuple[int, tuple[tuple[int, float], ...]], ...] = ()
    discovery_scores_regime_scaled: tuple[tuple[int, float], ...] = ()
    discovery_aggregate_regime_scaled_net_t: float | None = None
    qualification_regime_scaled_net_t: float | None = None
    regime_scaled_admitted: bool | None = None


def _year_mask(index: pd.DatetimeIndex, year: int) -> np.ndarray:
    return np.asarray(index.year == year, dtype=bool)


def _horizon_weights(
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    sign: int,
    horizon: int,
    min_symbols: int,
    tranche_count: int,
) -> pd.DataFrame:
    """Build the combined weight book for one horizon candidate.

    The signal → rank-weight → phase-tranche chain depends only on
    ``(sign, horizon, min_symbols, tranche_count)``, never on the year/qualification
    mask, so it is computed once per candidate and shared across every mask the
    gate evaluates.
    """
    signal = vol_normalized_horizon_signal(log_close, horizon) if sign == 1 else horizon_log_return(log_close, horizon)
    weights = rank_weight_book(signal, eligible, sign, min_symbols)
    return phase_tranche_book(weights, tranche_count)


def _score_masked_net_t(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    mask: np.ndarray,
    cost_bps: float,
    periods_per_year: float,
) -> float:
    """Aggregate prescreen net t-stat of an already-built weight book over ``mask``."""
    curve = cost_response_curve(
        weights[mask], opens[mask], bar_funding[mask], (cost_bps,), periods_per_year,
    )
    return float(curve[cost_bps].net_t)


def _bartlett_hac_denom(demeaned: np.ndarray, max_lag: int) -> float:
    """Bartlett-kernel long-run-variance ratio denominator.

    ``1 + 2 * sum_{k=1}^{max_lag}(1 - k/(max_lag + 1)) * rho_k`` over the
    sample lag autocorrelations ``rho_k`` of the demeaned series -- the same
    kernel form frozen in ``autocorrelation_adjusted_sharpe``
    (``src/mhs/evaluation.py``), generalized to arbitrary bar frequency.
    Autocovariances use ``np.dot`` on numpy slices, never a per-row Python
    loop. ``max_lag < 1`` is treated as no adjustment (returns 1.0), and a
    zero-variance series also returns 1.0 -- a defensive internal degenerate
    case, never a raise.
    """
    if max_lag < 1:
        return 1.0
    n = len(demeaned)
    var = float(np.dot(demeaned, demeaned))
    if n < 2 or var <= 0.0:
        return 1.0
    acf_sum = 0.0
    for k in range(1, max_lag + 1):
        rho = float(np.dot(demeaned[k:], demeaned[: n - k])) / var
        acf_sum += (1.0 - k / (max_lag + 1)) * rho
    return float(max(1.0 + 2.0 * acf_sum, 1e-12))


def _score_masked_adjusted_net_t(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    mask: np.ndarray,
    cost_bps: float,
    periods_per_year: float,
    max_lag_periods: int,
) -> float:
    """Bartlett/HAC-adjusted prescreen net t-stat of a weight book over ``mask``.

    Computes the identical raw t-stat as ``_score_masked_net_t`` (same
    ``mhs_ledger_pnl`` primitive, same ``mean/std * sqrt(n)`` formula), then
    divides by ``sqrt(_bartlett_hac_denom(...))`` with ``max_lag_periods`` set
    to the candidate's own lookback horizon (the overlap length of its signal
    window) -- correcting the naive i.i.d. t-stat for the serial correlation
    overlapping windows structurally induce. Returns ``nan`` when the masked
    window is too short (``len(net) < max(3, max_lag_periods + 2)``) or the net
    series has zero variance. Diagnostic-only -- never feeds ``admitted`` /
    ``selected_horizon``.
    """
    net, _turnover = mhs_ledger_pnl(weights[mask], opens[mask], bar_funding[mask], cost_bps)
    if len(net) < max(3, max_lag_periods + 2):
        return float("nan")
    sd = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    if not (sd > 0.0):
        return float("nan")
    mean = float(net.mean())
    raw_t = mean / sd * math.sqrt(len(net))
    demeaned = net.to_numpy(dtype="float64") - mean
    denom = _bartlett_hac_denom(demeaned, max_lag_periods)
    return raw_t / math.sqrt(denom)

def _discovery_regime_cash_scale(vol_mean: pd.Series) -> pd.Series:
    """Per-bar gross-exposure scale that raises cash in high-vol regimes.

    Duplicates the exact kernel of the application layer's ``_regime_cash_scale``
    (``src/application/research/mhs/evaluation.py``): ``median(vol) / vol``
    clipped to ``[floor, 1.0]`` with a ``fillna(1.0)`` for
    insufficient-history bars -- same local constants (floor ``0.5`` =
    ``MHS_REGIME_CASH_SCALE_FLOOR``, median window ``720``h =
    ``MHS_REGIME_CASH_MEDIAN_WINDOW_HOURS``, ``min_periods`` 48), deliberately
    NOT imported because ``src/mhs/discovery.py`` is domain layer and the
    original lives in the application layer (importing it would invert the
    dependency direction -- same pattern as ``_bartlett_hac_denom``). The
    market-vol input it consumes is an approximation of production's
    ``vol_mean``; see ``_discovery_market_vol_mean``.
    """
    if vol_mean.empty:
        return pd.Series(1.0, index=vol_mean.index)
    median = vol_mean.rolling(720, min_periods=min(48, 720)).median()
    scale = median.div(vol_mean.clip(lower=1e-12))
    scale = scale.clip(lower=0.5, upper=1.0)
    return scale.fillna(1.0)


def _discovery_market_vol_mean(
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    horizon_bars: int = 48,
) -> pd.Series:
    """Approximate market-wide realized-vol series used for regime de-risking.

    ``realized_vol(log_close, horizon_bars).where(eligible).mean(axis=1)`` --
    the cross-sectional mean of the liquid-half-eligible panel's realized vol.
    This is an APPROXIMATION of the production regime signal
    (``evaluation.py``'s ``vol_mean`` masks to the PIT ``execution_mask`` /
    Top-30 roster, data ``discovery.py`` does not have and this spec does not
    attempt to thread in) -- never a bit-identical replica of production's
    ``_regime_cash_scale`` input.
    """
    return realized_vol(log_close, horizon_bars).where(eligible).mean(axis=1)


def _score_masked_regime_scaled_net_t(
    weights: pd.DataFrame,
    regime_scale: pd.Series,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    mask: np.ndarray,
    cost_bps: float,
    periods_per_year: float,
) -> float:
    """Aggregate prescreen net t-stat of a regime-de-risked weight book over ``mask``.

    Scales each bar's gross exposure by ``regime_scale`` (``weights.mul(
    regime_scale, axis=0)``) then applies the identical raw t-stat formula as
    ``_score_masked_net_t``. With ``regime_scale`` identically 1.0 this
    degenerates exactly to ``_score_masked_net_t``'s value. Diagnostic-only --
    never feeds ``admitted`` / ``selected_horizon``.
    """
    scaled = weights.mul(regime_scale, axis=0)
    curve = cost_response_curve(
        scaled[mask], opens[mask], bar_funding[mask], (cost_bps,), periods_per_year,
    )
    return float(curve[cost_bps].net_t)


def _candidate_net_t(
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    sign: int,
    horizon: int,
    mask: np.ndarray,
    min_symbols: int,
    tranche_count: int,
    cost_bps: float,
    periods_per_year: float,
) -> float:
    """Aggregate prescreen net t-stat of one horizon candidate over ``mask``."""
    weights = _horizon_weights(log_close, eligible, sign, horizon, min_symbols, tranche_count)
    return _score_masked_net_t(weights, opens, bar_funding, mask, cost_bps, periods_per_year)


def build_candidate_weights(
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    sign: int,
    horizon_candidates: tuple[int, ...],
    min_symbols: int = 8,
    tranche_count: int = 1,
) -> dict[int, pd.DataFrame]:
    """Precompute every horizon candidate's combined weight book once.

    A thin dict-comprehension wrapper around the unchanged ``_horizon_weights``
    building block: ``{h: _horizon_weights(log_close, eligible, sign, h,
    min_symbols, tranche_count) for h in horizon_candidates}``.  ``_horizon_weights``
    is a pure function of ``(log_close, eligible, sign, horizon, min_symbols,
    tranche_count)`` -- it never depends on the discovery/qualification window
    bounds -- so building the full candidate grid once and reusing it across
    multiple window scans (e.g. one per anchored fold) yields values
    mathematically identical to rebuilding per scan.
    """
    return {
        h: _horizon_weights(log_close, eligible, sign, h, min_symbols, tranche_count)
        for h in horizon_candidates
    }


def yearly_net_t_diagnostic(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    years: Sequence[int],
    cost_bps: float,
    periods_per_year: float,
) -> dict[int, float]:
    """Retrospective full-history per-calendar-year net t-stat (REPORT-ONLY).

    This function NEVER feeds ``admission_t``, ``DiscoveryQualificationResult``,
    or any capital/gate decision -- unlike ``select_horizon_by_discovery_qualification``
    it is not confined to a discovery window and is a purely reporting statistic,
    so a future reader must not mistake it for an admission input. Each
    requested calendar year is scored with the same ``_score_masked_net_t``
    primitive the gate itself uses, over the full ``weights`` panel. A year
    with zero rows in the panel, or a year whose score is non-finite, is
    returned as ``float('nan')`` -- never silently dropped, never ``0.0``.
    """
    if not years:
        raise ValueError("years must not be empty")
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
    if not weights.index.equals(opens.index) or list(weights.columns) != list(opens.columns):
        raise ValueError("weights and opens must be identically indexed and columned")
    if not weights.index.equals(bar_funding.index) or list(weights.columns) != list(bar_funding.columns):
        raise ValueError("weights and bar_funding must be identically indexed and columned")
    index = weights.index
    out: dict[int, float] = {}
    for year in years:
        mask = _year_mask(index, year)
        if not mask.any():
            out[year] = float("nan")
            continue
        out[year] = _score_masked_net_t(
            weights, opens, bar_funding, mask, cost_bps, periods_per_year,
        )
    return out


def select_horizon_by_discovery_qualification(
    sign: int,
    horizon_candidates: tuple[int, ...],
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    discovery_start: pd.Timestamp,
    discovery_end: pd.Timestamp,
    qualification_end: pd.Timestamp,
    min_symbols: int = 8,
    tranche_count: int = 1,
    cost_bps: float = MEASURED_EXECUTION_COST_TIERS_BPS["optimistic"],
    periods_per_year: float = _PERIODS_PER_YEAR_1H,
    admission_t: float = _ADMISSION_T,
    precomputed_candidate_weights: Mapping[int, pd.DataFrame] | None = None,
    compute_adjusted_net_t: bool = False,
    compute_regime_scaled_net_t: bool = False,
) -> DiscoveryQualificationResult:
    """Run the worst-year-robust discovery/qualification gate for one sign family.

    For every candidate horizon the discovery window (calendar years within
    ``[discovery_start, discovery_end]``) is scored in a sign-consistent
    oriented space ``oriented = sign * net_t`` (larger means stronger evidence
    in the preregistered direction) by the MINIMUM of its yearly oriented
    prescreen scores at ``cost_bps`` -- a single strong year can never admit a
    candidate (spec §2.2). The highest-scoring candidate is selected (ties
    break toward the smaller horizon, the project's existing wide-candidate
    tie-break convention); if its worst-year oriented score is below
    ``admission_t`` the gate fails closed with ``selected_horizon=None`` and
    qualification is never evaluated. Otherwise that one candidate is
    re-computed on the disjoint qualification window
    ``(discovery_end, qualification_end]`` and admitted only when
    ``|net_t| >= admission_t`` AND the qualification net_t keeps the discovery
    aggregate's sign.

    ``compute_adjusted_net_t`` is a DIAGNOSTIC-ONLY opt-in: when True it also
    computes the Bartlett/HAC-adjusted net t-stat per (horizon, year), the
    adjusted worst-year ``discovery_scores_adjusted`` table, and -- for the
    selected candidate only -- the adjusted aggregate/qualification t-stats and
    ``adjusted_admitted``. ``compute_regime_scaled_net_t`` is a second,
    independent DIAGNOSTIC-ONLY opt-in: when True it also computes a vol-regime
    cash-scale-adjusted net t-stat per (horizon, year) from ONE shared
    ``regime_scale`` series built once per call, the regime-scaled worst-year
    ``discovery_scores_regime_scaled`` table, and -- for the selected candidate
    only -- the regime-scaled aggregate/qualification t-stats and
    diagnostic ever changes ``selected_horizon``/``admitted``/the raw scores,
    which stay driven solely by the unadjusted path.
    """
    if sign not in (-1, 1):
        raise ValueError(f"sign must be -1 or +1, got {sign}")
    if not horizon_candidates:
        raise ValueError("horizon_candidates must not be empty")
    if min_symbols < 2:
        raise ValueError(f"min_symbols must be >= 2, got {min_symbols}")
    if tranche_count < 1:
        raise ValueError(f"tranche_count must be >= 1, got {tranche_count}")
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    if admission_t <= 0:
        raise ValueError(f"admission_t must be > 0, got {admission_t}")
    if not log_close.index.equals(opens.index) or list(log_close.columns) != list(opens.columns):
        raise ValueError("log_close and opens must be identically indexed and columned")
    if not log_close.index.equals(eligible.index) or list(log_close.columns) != list(eligible.columns):
        raise ValueError("log_close and eligible must be identically indexed and columned")
    if not grid_1h.equals(log_close.index):
        raise ValueError("grid_1h must exactly match the log_close index")

    index = log_close.index
    discovery_mask = (index >= discovery_start) & (index <= discovery_end)
    discovery_years = sorted({t.year for t in index[discovery_mask]})
    qualification_mask = (index > discovery_end) & (index <= qualification_end)

    oriented_scores: dict[int, float] = {}
    raw_worst_year: dict[int, float] = {}
    raw_worst_year_adjusted: dict[int, float] = {}
    raw_worst_year_regime_scaled: dict[int, float] = {}
    yearly_net_t_by_horizon: dict[int, dict[int, float]] = {}
    yearly_adjusted_net_t_by_horizon: dict[int, dict[int, float]] = {}
    yearly_regime_scaled_net_t_by_horizon: dict[int, dict[int, float]] = {}
    regime_scale: pd.Series | None = (
        _discovery_regime_cash_scale(_discovery_market_vol_mean(log_close, eligible))
        if compute_regime_scaled_net_t
        else None
    )
    for horizon in horizon_candidates:
        weights = (
            precomputed_candidate_weights[horizon]
            if precomputed_candidate_weights is not None
            else _horizon_weights(log_close, eligible, sign, horizon, min_symbols, tranche_count)
        )
        yearly: list[float] = []
        yearly_by_year: dict[int, float] = {}
        adjusted_by_year: dict[int, float] = {}
        regime_scaled_by_year: dict[int, float] = {}
        for year in discovery_years:
            year_mask = discovery_mask & _year_mask(index, year)
            net_t = _score_masked_net_t(
                weights, opens, bar_funding, year_mask, cost_bps, periods_per_year,
            )
            yearly_by_year[year] = net_t
            if np.isfinite(net_t):
                yearly.append(net_t)
            if compute_regime_scaled_net_t:
                assert regime_scale is not None
                regime_scaled_by_year[year] = _score_masked_regime_scaled_net_t(
                    weights, regime_scale, opens, bar_funding, year_mask,
                    cost_bps, periods_per_year,
                )
            if compute_adjusted_net_t:
                adjusted_by_year[year] = _score_masked_adjusted_net_t(
                    weights, opens, bar_funding, year_mask, cost_bps, periods_per_year,
                    max_lag_periods=horizon,
                )
        yearly_net_t_by_horizon[horizon] = yearly_by_year
        if compute_regime_scaled_net_t:
            yearly_regime_scaled_net_t_by_horizon[horizon] = regime_scaled_by_year
        if compute_adjusted_net_t:
            yearly_adjusted_net_t_by_horizon[horizon] = adjusted_by_year
        if not yearly:
            oriented_scores[horizon] = float("-inf")
            raw_worst_year[horizon] = float("nan")
            if compute_regime_scaled_net_t:
                raw_worst_year_regime_scaled[horizon] = float("nan")
            if compute_adjusted_net_t:
                raw_worst_year_adjusted[horizon] = float("nan")
            continue
        worst_oriented = min(sign * t for t in yearly)
        oriented_scores[horizon] = worst_oriented
        raw_worst_year[horizon] = worst_oriented * sign
        if compute_regime_scaled_net_t:
            regime_scaled_finite = [t for t in regime_scaled_by_year.values() if np.isfinite(t)]
            if regime_scaled_finite:
                worst_regime_scaled = min(sign * t for t in regime_scaled_finite)
                raw_worst_year_regime_scaled[horizon] = worst_regime_scaled * sign
            else:
                raw_worst_year_regime_scaled[horizon] = float("nan")
        if compute_adjusted_net_t:
            adjusted_finite = [t for t in adjusted_by_year.values() if np.isfinite(t)]
            if adjusted_finite:
                worst_adjusted = min(sign * t for t in adjusted_finite)
                raw_worst_year_adjusted[horizon] = worst_adjusted * sign
            else:
                raw_worst_year_adjusted[horizon] = float("nan")

    yearly_net_t = tuple(
        (h, tuple(sorted(yearly_net_t_by_horizon[h].items())))
        for h in sorted(yearly_net_t_by_horizon)
    )
    yearly_adjusted_net_t = tuple(
        (h, tuple(sorted(yearly_adjusted_net_t_by_horizon[h].items())))
        for h in sorted(yearly_adjusted_net_t_by_horizon)
    )
    yearly_regime_scaled_net_t = tuple(
        (h, tuple(sorted(yearly_regime_scaled_net_t_by_horizon[h].items())))
        for h in sorted(yearly_regime_scaled_net_t_by_horizon)
    )

    best_horizon = max(
        horizon_candidates,
        key=lambda h: (oriented_scores[h], -h),
    )
    best_oriented = oriented_scores[best_horizon]

    discovery_scores = tuple(sorted(raw_worst_year.items()))
    discovery_scores_adjusted = tuple(sorted(raw_worst_year_adjusted.items()))
    discovery_scores_regime_scaled = tuple(sorted(raw_worst_year_regime_scaled.items()))
    if best_oriented < admission_t:
        return DiscoveryQualificationResult(
            selected_horizon=None,
            admitted=False,
            discovery_scores=discovery_scores,
            discovery_aggregate_net_t=None,
            qualification_net_t=None,
            qualification_sign_consistent=None,
            yearly_net_t=yearly_net_t,
            yearly_adjusted_net_t=yearly_adjusted_net_t,
            discovery_scores_adjusted=discovery_scores_adjusted,
            yearly_regime_scaled_net_t=yearly_regime_scaled_net_t,
            discovery_scores_regime_scaled=discovery_scores_regime_scaled,
        )

    best_weights = (
        precomputed_candidate_weights[best_horizon]
        if precomputed_candidate_weights is not None
        else _horizon_weights(log_close, eligible, sign, best_horizon, min_symbols, tranche_count)
    )
    discovery_aggregate_net_t = _score_masked_net_t(
        best_weights, opens, bar_funding, discovery_mask, cost_bps, periods_per_year,
    )
    if not np.isfinite(discovery_aggregate_net_t):
        return DiscoveryQualificationResult(
            selected_horizon=best_horizon,
            admitted=False,
            discovery_scores=discovery_scores,
            discovery_aggregate_net_t=None,
            qualification_net_t=None,
            qualification_sign_consistent=None,
            yearly_net_t=yearly_net_t,
            yearly_adjusted_net_t=yearly_adjusted_net_t,
            discovery_scores_adjusted=discovery_scores_adjusted,
            yearly_regime_scaled_net_t=yearly_regime_scaled_net_t,
            discovery_scores_regime_scaled=discovery_scores_regime_scaled,
        )
    qualification_net_t = _score_masked_net_t(
        best_weights, opens, bar_funding, qualification_mask, cost_bps, periods_per_year,
    )
    sign_consistent = bool(
        np.isfinite(qualification_net_t)
        and (qualification_net_t >= 0.0) == (discovery_aggregate_net_t >= 0.0)
    )
    admitted = sign_consistent and abs(qualification_net_t) >= admission_t
    if compute_adjusted_net_t:
        discovery_aggregate_adjusted_net_t = _score_masked_adjusted_net_t(
            best_weights, opens, bar_funding, discovery_mask, cost_bps, periods_per_year,
            max_lag_periods=best_horizon,
        )
        qualification_adjusted_net_t = _score_masked_adjusted_net_t(
            best_weights, opens, bar_funding, qualification_mask, cost_bps, periods_per_year,
            max_lag_periods=best_horizon,
        )
        adjusted_admitted = bool(
            sign_consistent
            and qualification_adjusted_net_t is not None
            and np.isfinite(qualification_adjusted_net_t)
            and abs(qualification_adjusted_net_t) >= admission_t
        )
    else:
        discovery_aggregate_adjusted_net_t = None
        qualification_adjusted_net_t = None
        adjusted_admitted = None
    if compute_regime_scaled_net_t:
        assert regime_scale is not None
        discovery_aggregate_regime_scaled_net_t = _score_masked_regime_scaled_net_t(
            best_weights, regime_scale, opens, bar_funding, discovery_mask,
            cost_bps, periods_per_year,
        )
        qualification_regime_scaled_net_t = _score_masked_regime_scaled_net_t(
            best_weights, regime_scale, opens, bar_funding, qualification_mask,
            cost_bps, periods_per_year,
        )
        regime_scaled_admitted = bool(
            sign_consistent
            and qualification_regime_scaled_net_t is not None
            and math.isfinite(qualification_regime_scaled_net_t)
            and abs(qualification_regime_scaled_net_t) >= admission_t
        )
    else:
        discovery_aggregate_regime_scaled_net_t = None
        qualification_regime_scaled_net_t = None
        regime_scaled_admitted = None
    return DiscoveryQualificationResult(
        selected_horizon=best_horizon,
        admitted=admitted,
        discovery_scores=discovery_scores,
        discovery_aggregate_net_t=discovery_aggregate_net_t,
        qualification_net_t=qualification_net_t,
        qualification_sign_consistent=sign_consistent,
        yearly_net_t=yearly_net_t,
        yearly_adjusted_net_t=yearly_adjusted_net_t,
        discovery_scores_adjusted=discovery_scores_adjusted,
        discovery_aggregate_adjusted_net_t=discovery_aggregate_adjusted_net_t,
        qualification_adjusted_net_t=qualification_adjusted_net_t,
        adjusted_admitted=adjusted_admitted,
        yearly_regime_scaled_net_t=yearly_regime_scaled_net_t,
        discovery_scores_regime_scaled=discovery_scores_regime_scaled,
        discovery_aggregate_regime_scaled_net_t=discovery_aggregate_regime_scaled_net_t,
        qualification_regime_scaled_net_t=qualification_regime_scaled_net_t,
        regime_scaled_admitted=regime_scaled_admitted,
    )


def fold_train_only_discovery_qualification(
    sign: int,
    horizon_candidates: tuple[int, ...],
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    fold: AnchoredPurgedFold,
    min_symbols: int = 8,
    tranche_count: int = 1,
    cost_bps: float = MEASURED_EXECUTION_COST_TIERS_BPS["optimistic"],
    periods_per_year: float = _PERIODS_PER_YEAR_1H,
    admission_t: float = _ADMISSION_T,
    precomputed_candidate_weights: Mapping[int, pd.DataFrame] | None = None,
    compute_adjusted_net_t: bool = False,
    compute_regime_scaled_net_t: bool = False,
) -> DiscoveryQualificationResult:
    """Run the discovery/qualification gate on one anchored fold's own train data.

    A leak-free, fold-scoped wrapper around
    ``select_horizon_by_discovery_qualification``: the qualification window is
    the single calendar year containing ``fold.train_end``, discovery is
    everything in ``[fold.train_start, discovery_end]``, and both bounds are
    ``<= fold.train_end`` (which ``AnchoredPurgedFold.__post_init__`` already
    guarantees is ``< fold.validation_start``), so the gate never reads a bar
    the fold's validation replay could see. When the train window spans less
    than one extra calendar year beyond its first (``qualification_start <=
    fold.train_start``) there is no room for a disjoint split and the gate
    fails closed without evaluating any candidate. ``compute_adjusted_net_t``
    and ``compute_regime_scaled_net_t`` are forwarded unchanged to the
    underlying selection; the closed-gate early return leaves the diagnostic
    fields at their empty/None defaults.
    """
    qualification_start = pd.Timestamp(year=fold.train_end.year, month=1, day=1, tz="UTC")
    if qualification_start <= fold.train_start:
        return DiscoveryQualificationResult(
            selected_horizon=None,
            admitted=False,
            discovery_scores=(),
            discovery_aggregate_net_t=None,
            qualification_net_t=None,
            qualification_sign_consistent=None,
            yearly_net_t=(),
        )
    return select_horizon_by_discovery_qualification(
        sign=sign,
        horizon_candidates=horizon_candidates,
        log_close=log_close,
        eligible=eligible,
        opens=opens,
        bar_funding=bar_funding,
        grid_1h=grid_1h,
        discovery_start=fold.train_start,
        discovery_end=qualification_start - pd.Timedelta(microseconds=1),
        qualification_end=fold.train_end,
        min_symbols=min_symbols,
        tranche_count=tranche_count,
        cost_bps=cost_bps,
        periods_per_year=periods_per_year,
        admission_t=admission_t,
        precomputed_candidate_weights=precomputed_candidate_weights,
        compute_adjusted_net_t=compute_adjusted_net_t,
        compute_regime_scaled_net_t=compute_regime_scaled_net_t,
    )


__all__ = [
    "DiscoveryQualificationResult",
    "fold_train_only_discovery_qualification",
    "select_horizon_by_discovery_qualification",
    "yearly_net_t_diagnostic",
]
