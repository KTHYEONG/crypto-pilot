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

from src.mhs.source_gaps import active_intervals


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


def source_gap_excluded_symbols() -> frozenset[str]:
    """Return every symbol carrying at least one unresolved source-gap interval.

    This symbol-level view exists only for legacy consumers that filter a universe
    before they know their evaluation grid. Interval-aware callers must consult
    `src.mhs.source_gaps.blocked_mask` instead, because collapsing an interval to a
    symbol discards the point-in-time reality that the symbol traded normally outside
    the gap.

    Returns:
        Symbols with at least one active interval across all planes.
    """
    return frozenset(iv.symbol for iv in active_intervals())


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
