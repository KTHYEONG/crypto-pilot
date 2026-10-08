"""MHS input data policy: single source of truth (INV-POLICY-SINGLE-SOURCE).

Every MHS entrypoint (``MhsDiagnosticRequest``, ``LiveStrategyParams``,
the live runtime) shares ``MHS_DATA_POLICY_DEFAULT``.
The new default is ``zombie_mask_v1``; artifacts that predate the policy field
keep their legacy interpretation and are never auto-upgraded.
"""

from __future__ import annotations

from collections.abc import Iterator, Set
from enum import StrEnum
from typing import Final, Literal

from src.core.instrument_settlements import load_instrument_settlement_registry
from src.core.source_gaps import SourceGapExtent, active_intervals


class MhsDataPolicy(StrEnum):
    """Registered MHS input-data contracts."""

    LEGACY = "legacy"
    ZOMBIE_MASK_V1 = "zombie_mask_v1"


MHS_DATA_POLICY_DEFAULT: Final[Literal["zombie_mask_v1"]] = "zombie_mask_v1"

# Source-gap exclusions are derived from the single interval registry at
# `src/core/policy/source_gaps.jsonl` (see `src.core.source_gaps`). The registry
# scopes each absence to its evidenced interval; the symbol-level view below
# exists only for legacy consumers that filter a universe before they know
# their evaluation grid. Interval-aware callers must consult
# `src.core.source_gaps.blocked_mask` instead.


# Exclusion applies only to confirmed delistings or unscoped legacy records.
_SYMBOL_EXCLUDING_REASONS: Final[frozenset[str]] = frozenset({"DELISTED"})
_SYMBOL_EXCLUDING_EXTENTS: Final[frozenset[SourceGapExtent]] = frozenset({"UNSCOPED"})


def source_gap_excluded_symbols() -> frozenset[str]:
    """Return symbols the legacy symbol-level view must drop for their whole history.

    A symbol is excluded when at least one active interval that is not superseded by an evidenced
    settlement is ``DELISTED`` or has ``UNSCOPED`` extent. A delisting explained by the settlement
    registry is replayed causally (announcement policy, delivery settlement) instead of hiding the
    symbol's whole life, which would select the universe on its future. Measured ``LISTING_EDGE``,
    ``OPEN_EDGE`` and ``INTERIOR`` spans never exclude the symbol. Interval-aware callers must
    consult ``src.core.source_gaps.blocked_mask``.

    Returns:
        Symbols with at least one active, unsuperseded DELISTED or UNSCOPED interval.
    Raises:
        DataIntegrityError: the settlement or source-gap registry is invalid.
    """
    from src.core.instrument_settlements import source_gap_superseded_by_settlement

    registry = load_instrument_settlement_registry()
    return frozenset(
        iv.symbol
        for iv in active_intervals()
        if (iv.reason in _SYMBOL_EXCLUDING_REASONS or iv.extent in _SYMBOL_EXCLUDING_EXTENTS)
        and not source_gap_superseded_by_settlement(iv, registry)
    )


class _SourceGapExcludedSymbolsView(Set[str]):
    """Lazily derived legacy view over the single source-gap registry."""

    __slots__ = ()

    def __contains__(self, item: object) -> bool:
        return item in source_gap_excluded_symbols()

    def __iter__(self) -> Iterator[str]:
        return iter(source_gap_excluded_symbols())

    def __len__(self) -> int:
        return len(source_gap_excluded_symbols())


SOURCE_GAP_EXCLUDED_SYMBOLS: Final[_SourceGapExcludedSymbolsView] = _SourceGapExcludedSymbolsView()
