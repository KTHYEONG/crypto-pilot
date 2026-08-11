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

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.mhs.books import phase_tranche_book, rank_weight_book
from src.mhs.contracts import MEASURED_EXECUTION_COST_TIERS_BPS
from src.mhs.evaluation import cost_response_curve
from src.mhs.horizons import horizon_log_return, vol_normalized_horizon_signal

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
    failed closed before any qualification evaluation.
    """

    selected_horizon: int | None
    admitted: bool
    discovery_scores: tuple[tuple[int, float], ...]
    discovery_aggregate_net_t: float | None
    qualification_net_t: float | None
    qualification_sign_consistent: bool | None


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
    for horizon in horizon_candidates:
        weights = _horizon_weights(log_close, eligible, sign, horizon, min_symbols, tranche_count)
        yearly: list[float] = []
        for year in discovery_years:
            net_t = _score_masked_net_t(
                weights, opens, bar_funding,
                discovery_mask & _year_mask(index, year),
                cost_bps, periods_per_year,
            )
            if np.isfinite(net_t):
                yearly.append(net_t)
        if not yearly:
            oriented_scores[horizon] = float("-inf")
            raw_worst_year[horizon] = float("nan")
            continue
        worst_oriented = min(sign * t for t in yearly)
        oriented_scores[horizon] = worst_oriented
        raw_worst_year[horizon] = worst_oriented * sign

    best_horizon = max(
        horizon_candidates,
        key=lambda h: (oriented_scores[h], -h),
    )
    best_oriented = oriented_scores[best_horizon]

    discovery_scores = tuple(sorted(raw_worst_year.items()))
    if best_oriented < admission_t:
        return DiscoveryQualificationResult(
            selected_horizon=None,
            admitted=False,
            discovery_scores=discovery_scores,
            discovery_aggregate_net_t=None,
            qualification_net_t=None,
            qualification_sign_consistent=None,
        )

    best_weights = _horizon_weights(log_close, eligible, sign, best_horizon, min_symbols, tranche_count)
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
        )
    qualification_net_t = _score_masked_net_t(
        best_weights, opens, bar_funding, qualification_mask, cost_bps, periods_per_year,
    )
    sign_consistent = bool(
        np.isfinite(qualification_net_t)
        and (qualification_net_t >= 0.0) == (discovery_aggregate_net_t >= 0.0)
    )
    admitted = sign_consistent and abs(qualification_net_t) >= admission_t
    return DiscoveryQualificationResult(
        selected_horizon=best_horizon,
        admitted=admitted,
        discovery_scores=discovery_scores,
        discovery_aggregate_net_t=discovery_aggregate_net_t,
        qualification_net_t=qualification_net_t,
        qualification_sign_consistent=sign_consistent,
    )


__all__ = [
    "DiscoveryQualificationResult",
    "select_horizon_by_discovery_qualification",
]
