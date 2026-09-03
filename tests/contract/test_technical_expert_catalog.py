from __future__ import annotations

import pytest

from src.quant.technical_experts.catalog import (
    ADMITTED_FAMILY_MATRIX,
    TECHNICAL_CANDIDATES,
    TECHNICAL_EXPERT_FAMILIES,
    resolve_technical_candidate,
)

_NEW_FAMILIES = {"supertrend", "parabolic_sar", "keltner_channel_breakout"}


class TestPrunedFamilyMatrixDocumented:
    def test_pruned_family_matrix_documented(self) -> None:
        # pruned_family_matrix_documented: every family remaining in the catalog
        # has at least one recorded admission pass in the timeframe-census
        # matrix; the catalog contains no all-fail families.
        for family in TECHNICAL_EXPERT_FAMILIES:
            assert family in ADMITTED_FAMILY_MATRIX
            assert any(ADMITTED_FAMILY_MATRIX[family].values()), family

    def test_matrix_only_covers_frozen_families(self) -> None:
        # Every matrix key is a frozen catalog family; a pruned family would
        # have been removed from the matrix at the same time.
        assert set(ADMITTED_FAMILY_MATRIX) == set(TECHNICAL_EXPERT_FAMILIES)

    def test_catalog_has_no_all_fail_or_unmeasured_family(self) -> None:
        # The 18 frozen candidates are exactly two sides per matrix-admitted
        # family; no candidate family is missing from the matrix.
        catalog_families = {candidate.family for candidate in TECHNICAL_CANDIDATES}
        assert catalog_families == set(TECHNICAL_EXPERT_FAMILIES)
        assert {candidate.side for candidate in TECHNICAL_CANDIDATES} == {
            "LONG", "SHORT",
        }
        for candidate in TECHNICAL_CANDIDATES:
            assert candidate.family in ADMITTED_FAMILY_MATRIX

    def test_new_families_are_not_freebied_into_the_catalog(self) -> None:
        # new_family_gated_not_freebied: the new candidate families are not
        # admitted without passing the gate, so none of their candidate ids
        # resolve from the catalog yet.
        catalog_families = {candidate.family for candidate in TECHNICAL_CANDIDATES}
        assert not (_NEW_FAMILIES & catalog_families)
        for family in _NEW_FAMILIES:
            with pytest.raises(ValueError, match="unknown or retired"):
                resolve_technical_candidate(f"technical_{family}_long_v1")
