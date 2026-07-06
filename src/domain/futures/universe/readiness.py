"""Strategy readiness cube builder for PIT universe.

Evaluates per-strategy, per-instrument, per-bar readiness based on:
- Execution eligibility (from UniverseStateCube)
- Sufficient trailing finite close bars (rolling cumsum approach)
- Optional funding_rate and open_interest availability windows

Complexity:
    Time:  O(S * T * N) vectorized — inner T x N done via numpy cumsum (no Python loop)
    Space: O(S * T * N) for the output cubes; intermediate arrays are O(T * N)
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.universe.contracts import (
    StrategyReadinessCube,
    StrategyRequirement,
    UniverseStateCube,
)

__all__ = ["evaluate_strategy_readiness"]

_LOG = logging.getLogger(__name__)

# Reason code constants (dtype=object arrays use these string values)
_REASON_NOT_READY: str = "not_ready"
_REASON_NOT_ELIGIBLE: str = "not_eligible"
_REASON_INSUFFICIENT_LOOKBACK: str = "insufficient_lookback"
_REASON_MISSING_FUNDING: str = "missing_funding"
_REASON_INSUFFICIENT_FUNDING: str = "insufficient_funding"
_REASON_MISSING_OI: str = "missing_open_interest"
_REASON_INSUFFICIENT_OI: str = "insufficient_open_interest"
_REASON_READY: str = "ready"


def _rolling_finite_count(arr: NDArray[np.float64], lookback: int) -> NDArray[np.float64]:
    """Compute rolling count of finite values over a lookback window.

    Args:
        arr: Shape [T, N] float64 array; NaN denotes missing/out-of-lifecycle bars.
        lookback: Number of bars in the rolling window [t-lookback+1 : t+1].

    Returns:
        Shape [T, N] float64 array where each cell is the count of finite values
        in the trailing ``lookback`` bars (inclusive of bar t).

    Notes:
        Uses cumulative sum trick to avoid O(T) per-cell Python loop:
            rolling_count[t] = cum[t] - cum[t - lookback]  (if t >= lookback)
        Time: O(T*N), Space: O(T*N)
    """
    n_t = arr.shape[0]
    finite_mask: NDArray[np.float64] = np.isfinite(arr).astype(np.float64)  # [T, N]
    cum: NDArray[np.float64] = np.cumsum(finite_mask, axis=0)  # [T, N]
    # Prepend zero row → cum_padded[t+1] == cum[t]
    pad = np.zeros((1, arr.shape[1]), dtype=np.float64)
    cum_padded: NDArray[np.float64] = np.concatenate([pad, cum], axis=0)  # [T+1, N]
    # rolling[t] = cum_padded[t+1] - cum_padded[max(0, t+1-lookback)]
    rows_end = np.arange(1, n_t + 1)  # [T,]  values 1..T
    rows_start = np.maximum(0, rows_end - lookback)  # [T,]  clipped at 0
    rolling: NDArray[np.float64] = cum_padded[rows_end] - cum_padded[rows_start]  # [T, N]
    return rolling


def evaluate_strategy_readiness(
    *,
    aligned: object,
    requirements: Mapping[str, StrategyRequirement],
    eligibility: UniverseStateCube,
) -> StrategyReadinessCube:
    """Build a dense readiness cube for all strategies.

    For each strategy ``s``, instrument ``n``, bar ``t``:
    1. If ``eligibility.eligible[t, n]`` is False  → ``not_eligible``.
    2. If rolling finite close count < ``req.required_lookback_bars`` → ``insufficient_lookback``.
    3. If ``req.requires_funding`` and funding array missing → ``missing_funding``.
       If funding array present but insufficient finite values → ``insufficient_funding``.
    4. If ``req.requires_open_interest`` and OI array missing → ``missing_open_interest``.
       If OI array present but insufficient finite values → ``insufficient_open_interest``.
    5. Otherwise → ``ready``.

    Args:
        aligned: Object with ``.close`` NDArray[float64] of shape [T, N].
            Optionally has ``.funding_rate`` and ``.open_interest``, both [T, N].
        requirements: Mapping of strategy name → StrategyRequirement.
        eligibility: UniverseStateCube with ``.eligible`` bool mask [T, N].

    Returns:
        StrategyReadinessCube with ``ready[S, T, N]`` (bool_) and
        ``reason_code[S, T, N]`` (object, default "not_ready").

    Raises:
        ValueError: If aligned.close shape is inconsistent with eligibility cube.
    """
    n_cal: int = len(eligibility.calendar)
    n_inst: int = len(eligibility.instrument_ids)

    close_arr: NDArray[np.float64] = getattr(aligned, "close", np.full((n_cal, n_inst), np.nan))

    n_t, n_n = close_arr.shape

    if n_t != n_cal or n_n != n_inst:
        raise ValueError(f"aligned.close shape {(n_t, n_n)} inconsistent with eligibility cube ({n_cal}, {n_inst})")

    # Optional arrays — None if the attribute is absent
    funding_arr: NDArray[np.float64] | None = getattr(aligned, "funding_rate", None)
    oi_arr: NDArray[np.float64] | None = getattr(aligned, "open_interest", None)

    strategies: tuple[str, ...] = tuple(requirements.keys())
    n_s: int = len(strategies)

    # Allocate output arrays: shape [n_s, n_t, n_n]
    ready_cube: NDArray[np.bool_] = np.zeros((n_s, n_t, n_n), dtype=np.bool_)
    reason_cube: NDArray[np.object_] = np.full((n_s, n_t, n_n), _REASON_NOT_READY, dtype=object)

    # Precompute rolling finite counts for close (shared across strategies with same lookback)
    _close_rolling_cache: dict[int, NDArray[np.float64]] = {}

    def _get_close_rolling(lookback: int) -> NDArray[np.float64]:
        if lookback not in _close_rolling_cache:
            _close_rolling_cache[lookback] = _rolling_finite_count(close_arr, lookback)
        return _close_rolling_cache[lookback]

    # eligible mask broadcast-ready: [T, N] bool_
    eligible_mask: NDArray[np.bool_] = eligibility.eligible  # [T, N]

    for s_idx, strat_name in enumerate(strategies):
        req: StrategyRequirement = requirements[strat_name]
        lookback: int = req.required_lookback_bars

        # --- Gate 1: not_eligible ---
        # Start with not_eligible as the default for ineligible cells
        reason_cube[s_idx, ~eligible_mask] = _REASON_NOT_ELIGIBLE

        # Work only on eligible cells going forward
        # Build a working boolean mask: shape [n_t, n_n]
        # True = still a candidate for "ready"
        candidate: NDArray[np.bool_] = eligible_mask.copy()

        # --- Gate 2: insufficient_lookback ---
        close_rolling: NDArray[np.float64] = _get_close_rolling(lookback)
        insufficient_lb: NDArray[np.bool_] = candidate & (close_rolling < lookback)
        reason_cube[s_idx, insufficient_lb] = _REASON_INSUFFICIENT_LOOKBACK
        candidate &= ~insufficient_lb

        # --- Gate 3: funding_rate ---
        if req.requires_funding:
            if funding_arr is None:
                # Entire candidate set fails: missing array
                reason_cube[s_idx, candidate] = _REASON_MISSING_FUNDING
                candidate[:] = False
            else:
                funding_rolling: NDArray[np.float64] = _rolling_finite_count(funding_arr, lookback)
                insufficient_funding: NDArray[np.bool_] = candidate & (funding_rolling < lookback)
                reason_cube[s_idx, insufficient_funding] = _REASON_INSUFFICIENT_FUNDING
                candidate &= ~insufficient_funding

        # --- Gate 4: open_interest ---
        if req.requires_open_interest:
            if oi_arr is None:
                reason_cube[s_idx, candidate] = _REASON_MISSING_OI
                candidate[:] = False
            else:
                oi_rolling: NDArray[np.float64] = _rolling_finite_count(oi_arr, lookback)
                insufficient_oi: NDArray[np.bool_] = candidate & (oi_rolling < lookback)
                reason_cube[s_idx, insufficient_oi] = _REASON_INSUFFICIENT_OI
                candidate &= ~insufficient_oi

        # --- All gates passed: ready ---
        ready_cube[s_idx, candidate] = True
        reason_cube[s_idx, candidate] = _REASON_READY

        _LOG.debug(
            "strategy=%s ready_fraction=%.4f",
            strat_name,
            ready_cube[s_idx].mean(),
        )

    return StrategyReadinessCube(
        strategies=strategies,
        calendar=eligibility.calendar,
        instrument_ids=eligibility.instrument_ids,
        ready=ready_cube,
        reason_code=reason_cube,
    )
