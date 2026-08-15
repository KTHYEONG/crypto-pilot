"""Unit tests for build_cpcv_folds (Combinatorial Purged CV fold builder)."""

from __future__ import annotations

import math

import pytest

from src.domain.futures.strategy.walk_forward import CPCVFold, build_cpcv_folds

# ---------------------------------------------------------------------------
# TI1: Fold count and structural invariants
# ---------------------------------------------------------------------------


class TestCPCVFoldCountAndStructure:
    """C(6,2)=15 folds, disjoint fit/test groups, correct group cardinality."""

    def test_fold_count_equals_combination(self) -> None:
        # Arrange
        n_groups = 6
        n_test_groups = 2
        expected_count = math.comb(n_groups, n_test_groups)  # 15

        # Act
        folds = build_cpcv_folds(
            n_bars=600,
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            embargo_bars=18,
            purge_bars=6,
        )

        # Assert
        assert len(folds) == expected_count

    def test_each_fold_has_correct_test_group_count(self) -> None:
        # Arrange / Act
        folds = build_cpcv_folds(
            n_bars=600,
            n_groups=6,
            n_test_groups=2,
            embargo_bars=18,
            purge_bars=6,
        )

        # Assert
        for fold in folds:
            assert len(fold.test_groups) == 2

    def test_each_fold_has_correct_fit_group_count(self) -> None:
        # Arrange / Act
        folds = build_cpcv_folds(
            n_bars=600,
            n_groups=6,
            n_test_groups=2,
            embargo_bars=18,
            purge_bars=6,
        )

        # Assert
        for fold in folds:
            assert len(fold.fit_groups) == 4

    def test_fit_and_test_groups_are_disjoint(self) -> None:
        # Arrange / Act
        folds = build_cpcv_folds(
            n_bars=600,
            n_groups=6,
            n_test_groups=2,
            embargo_bars=18,
            purge_bars=6,
        )

        # Assert: intersection must be empty
        for fold in folds:
            assert set(fold.fit_groups) & set(fold.test_groups) == set()

    def test_fit_and_test_groups_cover_all_groups(self) -> None:
        # Arrange / Act
        folds = build_cpcv_folds(
            n_bars=600,
            n_groups=6,
            n_test_groups=2,
            embargo_bars=18,
            purge_bars=6,
        )

        # Assert: union == all group indices
        for fold in folds:
            assert set(fold.fit_groups) | set(fold.test_groups) == set(range(6))


# ---------------------------------------------------------------------------
# TI2: Purge / embargo boundary values
# ---------------------------------------------------------------------------


class TestPurgeEmbargoBoundaries:
    """Verify trimmed fit_spans match expected purge/embargo arithmetic."""

    @pytest.fixture
    def folds_4groups(self) -> tuple[CPCVFold, ...]:
        """C(4,1)=4 folds; groups of 100 bars each, purge=10, embargo=20."""
        return build_cpcv_folds(
            n_bars=400,
            n_groups=4,
            n_test_groups=1,
            purge_bars=10,
            embargo_bars=20,
        )

    def _find_fold_with_test_group(
        self,
        folds: tuple[CPCVFold, ...],
        test_group: int,
    ) -> CPCVFold:
        for fold in folds:
            if fold.test_groups == (test_group,):
                return fold
        raise AssertionError(f"No fold with test_group={test_group}")

    def test_fit_group_before_test_has_purge_applied(self, folds_4groups: tuple[CPCVFold, ...]) -> None:
        """fit_group 0 (bars 0-100) precedes test_group 1 (bars 100-200).
        Purge trims ge: ge = group_spans[1][0] - purge = 100 - 10 = 90.
        """
        # Arrange
        fold = self._find_fold_with_test_group(folds_4groups, test_group=1)

        # Act: find span belonging to fit_group 0 → starts at 0
        spans_starting_at_zero = [s for s in fold.fit_spans if s[0] == 0]

        # Assert
        assert len(spans_starting_at_zero) == 1
        assert spans_starting_at_zero[0][1] == 90  # 100 - purge(10)

    def test_fit_group_after_test_has_embargo_applied(self, folds_4groups: tuple[CPCVFold, ...]) -> None:
        """fit_group 2 (bars 200-300) follows test_group 1 (bars 100-200).
        Embargo trims gs: gs = group_spans[1][1] + embargo = 200 + 20 = 220.
        """
        # Arrange
        fold = self._find_fold_with_test_group(folds_4groups, test_group=1)

        # Act: find span for fit_group 2 → ends at 300
        spans_ending_at_300 = [s for s in fold.fit_spans if s[1] == 300]

        # Assert
        assert len(spans_ending_at_300) == 1
        assert spans_ending_at_300[0][0] == 220  # 200 + embargo(20)

    def test_untouched_fit_group_keeps_original_span(self, folds_4groups: tuple[CPCVFold, ...]) -> None:
        """fit_group 3 (bars 300-400) is not adjacent to test_group 1.
        Its span must remain (300, 400) - no trimming.
        """
        # Arrange
        fold = self._find_fold_with_test_group(folds_4groups, test_group=1)

        # Assert
        assert (300, 400) in fold.fit_spans


# ---------------------------------------------------------------------------
# TI3: Degenerate — n_bars very small relative to n_groups
# ---------------------------------------------------------------------------


class TestDegenerateSmallNBars:
    """When purge+embargo consume all fit bars, fallback must be returned."""

    def test_small_n_bars_returns_at_least_one_fold(self) -> None:
        # Arrange / Act
        result = build_cpcv_folds(
            n_bars=3,
            n_groups=6,
            n_test_groups=2,
            embargo_bars=5,
            purge_bars=5,
        )

        # Assert: must not be empty
        assert len(result) >= 1

    def test_small_n_bars_result_is_tuple_of_cpcv_folds(self) -> None:
        # Arrange / Act
        result = build_cpcv_folds(
            n_bars=3,
            n_groups=6,
            n_test_groups=2,
            embargo_bars=5,
            purge_bars=5,
        )

        # Assert: correct type
        assert all(isinstance(f, CPCVFold) for f in result)


# ---------------------------------------------------------------------------
# TI4: n_test_groups >= n_groups → immediate fallback
# ---------------------------------------------------------------------------


class TestNTestGroupsExceedsNGroups:
    """n_test_groups >= n_groups must trigger single-fold fallback."""

    @pytest.mark.parametrize(
        ("n_groups", "n_test_groups"),
        [
            (3, 3),  # equal
            (3, 4),  # exceeds
            (1, 2),  # edge: single group
        ],
    )
    def test_exceeds_returns_fallback_fold(self, n_groups: int, n_test_groups: int) -> None:
        # Arrange / Act
        result = build_cpcv_folds(
            n_bars=300,
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            embargo_bars=5,
            purge_bars=5,
        )

        # Assert: fallback → at least 1 fold, no OOS test groups
        assert len(result) >= 1
        assert result[0].test_groups == ()
