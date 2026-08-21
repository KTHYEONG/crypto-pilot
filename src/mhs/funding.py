"""Funding-rate carry signal builders (P0 return-source breadth diagnostics).

``bar_funding_panel`` (src/mhs/execution.py) is already causally aligned per
bar (its own docstring: "a symbol whose funding series cannot be causally
aligned is excluded"), so a trailing rolling mean over it is automatically
causal. This mirrors horizons.py's ``realized_vol`` construction exactly (same
rolling/min_periods/fail-closed-to-NaN pattern), just applied to funding
instead of price. This module is a NEW read-only consumer of the existing
panel -- it never modifies how funding is applied as a carry/cost term in the
ledger.
"""

from __future__ import annotations

import pandas as pd

from src.mhs.books import phase_tranche_book, rank_weight_book

# Matches src/application/research/mhs/evaluation.py
# ``DISCOVERY_GATE_TRANCHE_COUNT`` (the application layer passes that
# constant explicitly at the wiring site); the domain layer cannot import it
# without inverting the src.mhs <- src.application layering.
_DEFAULT_TRANCHE_COUNT = 8


def funding_carry_signal(bar_funding: pd.DataFrame, lookback_hours: int) -> pd.DataFrame:
    """Trailing mean funding rate over ``lookback_hours`` (causal by construction).

    ``min_periods`` equals ``lookback_hours`` so a short window fails closed to
    NaN rather than reporting an unreliable estimate, matching
    ``horizons.realized_vol``'s convention.
    """
    if lookback_hours < 1:
        raise ValueError(f"lookback_hours must be >= 1, got {lookback_hours}")
    return bar_funding.rolling(lookback_hours, min_periods=lookback_hours).mean()


def build_funding_carry_candidate_weights(
    bar_funding: pd.DataFrame,
    eligible: pd.DataFrame,
    sign: int,
    lookback_candidates: tuple[int, ...],
    min_symbols: int = 8,
    tranche_count: int = _DEFAULT_TRANCHE_COUNT,
) -> dict[int, pd.DataFrame]:
    """Precompute every funding-carry lookback candidate's weight book once.

    For each lookback the chain ``funding_carry_signal`` -> ``rank_weight_book``
    -> ``phase_tranche_book`` reuses the existing book builders unchanged
    (src/mhs/books.py) -- no new normalization or book logic. Mirrors
    ``discovery.build_candidate_weights``'s shape (same sign/min_symbols/
    tranche_count parameters), so the result plugs straight into
    ``select_horizon_by_discovery_qualification``'s
    ``precomputed_candidate_weights: Mapping[int, pd.DataFrame]`` parameter.
    """
    if not lookback_candidates:
        raise ValueError("lookback_candidates must not be empty")
    return {
        lookback: phase_tranche_book(
            rank_weight_book(
                funding_carry_signal(bar_funding, lookback),
                eligible,
                sign,
                min_symbols,
            ),
            tranche_count,
        )
        for lookback in lookback_candidates
    }


def funding_carry_execution_book(bar_funding: pd.DataFrame, execution_mask: pd.DataFrame, lookback_hours: int, decision_grid: pd.DatetimeIndex, tranche_count: int, min_symbols: int = 8) -> pd.DataFrame:
    """캐리 슬리브 execution book: 고펀딩 숏 / 저펀딩 롱 (sign=-1 측정값).

    Chain: funding_carry_signal -> rank_weight_book(sign=-1) -> reindex(decision_grid)
    -> phase_tranche_book -> reindex(bar_funding.index, method='ffill').
    sign=-1은 측정된 밴드 속성(가장 높은 trailing funding 숏, 가장 낮은 롱)이며
    호출부에서 추론하지 않는다. shift(1)을 의도적으로 추가하지 않음:
    기존 build_funding_carry_candidate_weights와 동일하게 재사용.
    """
    if tranche_count < 1:
        raise ValueError(f"tranche_count must be >= 1, got {tranche_count}")
    if min_symbols < 2:
        raise ValueError(f"min_symbols must be >= 2, got {min_symbols}")
    signal = funding_carry_signal(bar_funding, lookback_hours)
    book = rank_weight_book(signal, execution_mask, sign=-1, min_symbols=min_symbols)
    book_grid = book.reindex(decision_grid).fillna(0.0)
    stepped = phase_tranche_book(book_grid, tranche_count)
    return stepped.reindex(bar_funding.index, method="ffill").fillna(0.0)
