"""MHS input data policy: single source of truth (INV-POLICY-SINGLE-SOURCE).

Every MHS entrypoint (``MhsDiagnosticRequest``, ``MhsRunConfig``,
``LiveStrategyParams``, the live runtime) shares ``MHS_DATA_POLICY_DEFAULT``.
The new default is ``zombie_mask_v1``; artifacts that predate the policy field
keep their legacy interpretation and are never auto-upgraded.
"""

from __future__ import annotations

from collections.abc import Iterator, Set
from enum import StrEnum
from typing import Final, Literal

from src.mhs.source_gaps import SourceGapExtent, active_intervals


class MhsDataPolicy(StrEnum):
    """Registered MHS input-data contracts."""

    LEGACY = "legacy"
    ZOMBIE_MASK_V1 = "zombie_mask_v1"


MHS_DATA_POLICY_DEFAULT: Final[Literal["zombie_mask_v1"]] = "zombie_mask_v1"

# Source-gap exclusions are derived from the single interval registry at
# `src/mhs/policy/source_gaps.jsonl` (see `src.mhs.source_gaps`). The registry
# scopes each absence to its evidenced interval; the symbol-level view below
# exists only for legacy consumers that filter a universe before they know
# their evaluation grid. Interval-aware callers must consult
# `src.mhs.source_gaps.blocked_mask` instead.


# 심볼 전체 이력을 버리는 근거: 상폐 확정이거나 범위가 측정되지 않은 레거시 기록일 때만.
_SYMBOL_EXCLUDING_REASONS: Final[frozenset[str]] = frozenset({"DELISTED"})
_SYMBOL_EXCLUDING_EXTENTS: Final[frozenset[SourceGapExtent]] = frozenset({"UNSCOPED"})


def source_gap_excluded_symbols() -> frozenset[str]:
    """Return symbols the legacy symbol-level view must drop for their whole history.

    A symbol is excluded when at least one active interval is ``DELISTED`` or has
    ``UNSCOPED`` extent (a legacy or manual record whose scope was never measured).
    Measured ``LISTING_EDGE``, ``OPEN_EDGE`` and ``INTERIOR`` spans do not exclude the
    symbol: an absence before listing, after the last observed bar or between observed
    bars is not evidence that the symbol was untradeable while it traded, so dropping
    its entire history would be survivorship-style selection. This view exists only for
    legacy consumers that filter a universe before they know their evaluation grid;
    interval-aware callers must consult `src.mhs.source_gaps.blocked_mask`, which
    blocks exactly the measured bars.

    Returns:
        Symbols with at least one active DELISTED or UNSCOPED interval across all planes.
    """
    return frozenset(
        iv.symbol
        for iv in active_intervals()
        if iv.reason in _SYMBOL_EXCLUDING_REASONS or iv.extent in _SYMBOL_EXCLUDING_EXTENTS
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
