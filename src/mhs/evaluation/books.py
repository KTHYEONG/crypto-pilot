from __future__ import annotations

import gc
from typing import Literal

import numpy as np
import pandas as pd

from src.mhs import scaling as _scaling
from src.mhs.books import (
    equal_weight_book_ensemble,
    inverse_realized_vol_tilt,
    phase_tranche_book,
    rank_weight_book,
    renormalize_within_mask,
)
from src.mhs.discovery import build_candidate_weights
from src.mhs.funding import build_funding_carry_candidate_weights
from src.mhs.horizons import horizon_log_return, realized_vol, vol_normalized_horizon_signal
from src.mhs.params import (
    DISCOVERY_GATE_TRANCHE_COUNT,
    DISCOVERY_MOMENTUM_CANDIDATES,
    DISCOVERY_REVERSAL_CANDIDATES,
    FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
)
from src.mhs.types import BOOK_BLEND_WEIGHTS, BookSpec  # noqa: F401 - re-exported for monkeypatch seams


def _book_structure_trace(target_weights: pd.DataFrame) -> dict[str, float]:
    """Observational book-structure trace of a post-deadband decision book.

    ``holdings_growth_slope`` is the OLS slope of per-row holdings against the
    normalized row position ``[0, 1]``, divided by ``holdings_mean``: a
    dimensionless growth rate over the window (0 = stationary, 1 = doubles).
    """
    if target_weights.empty:
        return {
            "n_rows": 0.0,
            "gross_mean": 0.0,
            "holdings_mean": 0.0,
            "holdings_max": 0.0,
            "holdings_growth_slope": 0.0,
        }
    n_rows = float(len(target_weights))
    gross = target_weights.abs().sum(axis=1)
    holdings = (target_weights != 0.0).sum(axis=1)
    holdings_mean = float(holdings.mean())
    if n_rows < 2 or holdings_mean <= 0.0:
        growth_slope = 0.0
    else:
        x = np.linspace(0.0, 1.0, int(n_rows))
        y = holdings.to_numpy(dtype="float64")
        x_mean = float(x.mean())
        slope = float(np.dot(x - x_mean, y - y.mean()) / np.dot(x - x_mean, x - x_mean))
        growth_slope = slope / holdings_mean
    return {
        "n_rows": n_rows,
        "gross_mean": float(gross.mean()),
        "holdings_mean": holdings_mean,
        "holdings_max": float(holdings.max()),
        "holdings_growth_slope": growth_slope,
    }

def _book_weights(
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    spec: BookSpec,
    step_grid: pd.DatetimeIndex,
    ema_span: int | None = None,
) -> pd.DataFrame:
    # Raw horizon_log_return is used for live book weights.
    sig = horizon_log_return(log_close, spec.horizon_hours)
    if ema_span is not None:
        sig = _scaling._smooth_signal_ema(sig, ema_span)
    sig_step = sig.reindex(step_grid)
    el_step = eligible.reindex(step_grid)
    weights = rank_weight_book(sig_step, el_step, spec.band.sign, spec.min_symbols)
    return phase_tranche_book(weights, spec.tranche_count())


def _horizon_ensemble_execution_weights(
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    execution_mask: pd.DataFrame,
    spec: BookSpec,
    step_grid: pd.DatetimeIndex,
    mode: Literal["single_horizon", "horizon_ensemble"],
    signal_kind: Literal["raw", "vol_normalized"],
    ema_span: int | None,
) -> pd.DataFrame:
    """Shared execution-book builder for BOTH the slow and fast bands.

    The same generic chain (``spec: BookSpec``, never momentum-specific)
    constructs the top-level diagnostic and fold-replay execution books, so it
    is wired to whichever band asks for it.

    ``mode='single_horizon'`` reproduces the frozen production chain
    byte-identically (``horizon_log_return`` -> EMA -> ``rank_weight_book`` ->
    ``phase_tranche_book`` -> ``inverse_realized_vol_tilt`` ->
    ``renormalize_within_mask``). ``mode='horizon_ensemble'`` runs that same
    chain once per candidate horizon in ``spec.band.horizons_hours`` and
    combines the execution books with ``equal_weight_book_ensemble``, removing
    the discovery argmax from the capital path (RC-2). Each horizon's book is
    built on the same ``step_grid`` so the combination is a plain row-wise mean;
    each horizon's intermediates are released before the next is built (bounded
    RSS on the 43k-bar, 450-symbol panel).
    """
    if mode not in ("single_horizon", "horizon_ensemble"):
        raise ValueError(f"unknown mode '{mode}'")
    if signal_kind not in ("raw", "vol_normalized"):
        raise ValueError(f"unknown signal_kind '{signal_kind}'")
    mask = execution_mask.reindex(step_grid).fillna(False)
    horizons = (
        (spec.horizon_hours,) if mode == "single_horizon" else spec.band.horizons_hours
    )
    books: dict[int, pd.DataFrame] = {}
    for h in horizons:
        sig = (
            vol_normalized_horizon_signal(log_close, h)
            if signal_kind == "vol_normalized"
            else horizon_log_return(log_close, h)
        )
        if ema_span is not None:
            sig = _scaling._smooth_signal_ema(sig, ema_span)
        sig_step = sig.reindex(step_grid)
        weights = rank_weight_book(
            sig_step, eligible.reindex(step_grid), spec.band.sign, spec.min_symbols,
        )
        book = phase_tranche_book(weights, h // spec.step_hours)
        tilted = inverse_realized_vol_tilt(
            book, realized_vol(log_close, h).reindex(step_grid),
        )
        books[h] = renormalize_within_mask(tilted, mask, spec.min_symbols)
        del sig, sig_step, weights, book, tilted
        gc.collect()
    if mode == "single_horizon":
        return books[spec.horizon_hours]
    return equal_weight_book_ensemble(books)


def _active_blend_book_and_grid(
    fast: BookSpec,
    slow: BookSpec,
    fast_grid: pd.DatetimeIndex,
    slow_grid: pd.DatetimeIndex,
) -> tuple[BookSpec, pd.DatetimeIndex]:
    """Select the blend's active book spec and execution grid from the capital contract.

    The blend's decision cadence must derive from the same contract that
    allocates capital (``BOOK_BLEND_WEIGHTS``), never from a hardcoded
    book name: with only ``slow_momentum`` weighted the blend replays on slow's
    native 24h grid, while a nonzero ``fast_reversal`` weight (e.g. the
    historical 50/50) admits fast's 6h grid -- a superset of slow's from the
    same origin -- reproducing the pre-fix behavior byte-for-byte.  If no book
    carries capital the allocation invariant is violated and we fail closed.
    """
    import src.mhs.evaluation as ev
    if ev.BOOK_BLEND_WEIGHTS["fast_reversal"] != 0.0:
        return fast, fast_grid
    if ev.BOOK_BLEND_WEIGHTS["slow_momentum"] != 0.0:
        return slow, slow_grid
    raise ValueError(
        "BOOK_BLEND_WEIGHTS allocates no capital to either book; "
        "blend has no active execution grid"
    )


def _ordered_union(*tuples: tuple[int, ...]) -> tuple[int, ...]:
    """Ordered set union of horizon tuples (first-seen order preserved)."""
    result: list[int] = []
    seen: set[int] = set()
    for item in (*tuples,):
        for h in item:
            if h not in seen:
                seen.add(h)
                result.append(h)
    return tuple(result)


def _candidate_weight_books(
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    bar_funding: pd.DataFrame,
    specs: dict[str, BookSpec],
) -> dict[str, dict[int, pd.DataFrame]]:
    """Build every discovery candidate weight book exactly once.

    ``fold_train_only_discovery_qualification``/``select_horizon_by_discovery_qualification``
    never depend on window bounds for their candidate weights
    (``discovery.build_candidate_weights``), so the full candidate grid is built
    once in the parent and shared by both consumers: every fold's
    slow/fast/funding-carry scan and the top-level discovery gate. The slow/fast
    horizon key sets cover the union of the fold-safe ``BookSpec`` band horizons
    and the top-level ``DISCOVERY_MOMENTUM_CANDIDATES``/
    ``DISCOVERY_REVERSAL_CANDIDATES`` gate sets (currently identical), so a
    single build satisfies both. Returns a ``{"slow", "fast", "funding_long",
    "funding_short"}`` mapping of horizon-keyed weight books.
    """
    slow_horizons = _ordered_union(
        specs["slow_momentum"].band.horizons_hours,
        DISCOVERY_MOMENTUM_CANDIDATES,
    )
    fast_horizons = _ordered_union(
        specs["fast_reversal"].band.horizons_hours,
        DISCOVERY_REVERSAL_CANDIDATES,
    )
    return {
        "slow": build_candidate_weights(
            log_close, eligible, 1, slow_horizons,
            tranche_count=DISCOVERY_GATE_TRANCHE_COUNT,
        ),
        "fast": build_candidate_weights(
            log_close, eligible, -1, fast_horizons,
            tranche_count=DISCOVERY_GATE_TRANCHE_COUNT,
        ),
        "funding_long": build_funding_carry_candidate_weights(
            bar_funding, eligible, 1, FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
            tranche_count=DISCOVERY_GATE_TRANCHE_COUNT,
        ),
        "funding_short": build_funding_carry_candidate_weights(
            bar_funding, eligible, -1, FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
            tranche_count=DISCOVERY_GATE_TRANCHE_COUNT,
        ),
    }


