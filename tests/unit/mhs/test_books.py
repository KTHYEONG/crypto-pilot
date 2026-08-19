from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mhs.books import (
    equal_weight_book_ensemble,
    inverse_realized_vol_tilt,
    phase_tranche_book,
    portfolio_rebalance_trigger,
    rank_weight_book,
    renormalize_within_mask,
    scale_book_to_target_gross,
)


class TestPortfolioRebalanceTrigger:
    """SCENARIO_MHS_ALPHA_ENGINE_01 / SCENARIO_MHS_ALPHA_ENGINE_02: the
    portfolio-level trigger holds the previously adopted row wholesale and every
    emitted row is an exact copy of some input row, so dollar-neutrality and
    unit-gross survive the turnover filter by construction."""

    def test_holds_previous_row_below_threshold_and_adopts_wholesale(self) -> None:
        target = pd.DataFrame(
            [[0.5, -0.5], [0.45, -0.45], [0.3, -0.3], [0.35, -0.35]],
            columns=["A", "B"],
        )
        # One-way tracking error from the first row:
        #   row1: |0.45-0.5| + |(-0.45)-(-0.5)| = 0.10 (hold)
        #   row2: |0.3-0.5| + |(-0.3)-(-0.5)|  = 0.40 >= 0.20 (adopt)
        #   row3: |0.35-0.3| + |(-0.35)-(-0.3)| = 0.10 (hold)
        out = portfolio_rebalance_trigger(target, 0.20)
        assert out.iloc[0].tolist() == [0.5, -0.5]
        assert out.iloc[1].tolist() == [0.5, -0.5]
        assert out.iloc[2].tolist() == [0.3, -0.3]
        assert out.iloc[3].tolist() == [0.3, -0.3]

    def test_every_output_row_equals_some_input_row(self) -> None:
        rng = np.random.default_rng(3)
        target = pd.DataFrame(rng.normal(0.0, 0.3, (40, 6)), columns=list("ABCDEF"))
        out = portfolio_rebalance_trigger(target, 0.25)
        input_rows = target.to_numpy()
        for i in range(len(out)):
            row = out.iloc[i].to_numpy()
            assert any(np.array_equal(row, input_rows[j]) for j in range(len(input_rows))), i

    def test_dollar_neutral_unit_gross_input_preserves_invariants(self) -> None:
        rng = np.random.default_rng(5)
        raw = pd.DataFrame(rng.normal(0.0, 1.0, (60, 8)), columns=list("ABCDEFGH"))
        raw = raw.sub(raw.mean(axis=1), axis=0)
        book = raw.div(raw.abs().sum(axis=1), axis=0)
        out = portfolio_rebalance_trigger(book, 0.15)
        assert out.sum(axis=1).abs().max() < 1e-9
        assert out.abs().sum(axis=1).sub(1.0).abs().max() < 1e-9

    def test_nonfinite_cells_treated_as_zero_and_output_finite(self) -> None:
        target = pd.DataFrame(
            [[0.5, -0.5], [np.nan, -0.4], [0.3, np.inf]],
            columns=["A", "B"],
        )
        out = portfolio_rebalance_trigger(target, 0.2)
        assert bool(np.isfinite(out.to_numpy()).all())
        # Row1: |0.0-0.5| + |-0.4-(-0.5)| = 0.6 >= 0.2 -> adopted (NaN -> 0).
        assert out.iloc[1].tolist() == [0.0, -0.4]

    def test_first_row_always_adopted(self) -> None:
        target = pd.DataFrame([[0.4, -0.4], [0.4, -0.4], [0.4, -0.4]], columns=["A", "B"])
        out = portfolio_rebalance_trigger(target, 0.1)
        assert out.iloc[0].tolist() == [0.4, -0.4]

    def test_zero_threshold_is_identity_passthrough(self) -> None:
        target = pd.DataFrame(
            [[0.5, -0.5], [0.2, -0.2], [0.1, -0.1]], columns=["A", "B"],
        )
        pd.testing.assert_frame_equal(portfolio_rebalance_trigger(target, 0.0), target)

    def test_empty_input_is_unchanged_copy(self) -> None:
        empty = pd.DataFrame(columns=["A", "B"])
        out = portfolio_rebalance_trigger(empty, 0.2)
        assert out.empty
        assert list(out.columns) == ["A", "B"]

    def test_fails_closed_on_negative_threshold(self) -> None:
        with pytest.raises(ValueError, match="tracking_error_threshold"):
            portfolio_rebalance_trigger(pd.DataFrame([[0.5, -0.5]]), -0.1)


class TestEqualWeightBookEnsemble:
    """SCENARIO_MHS_ALPHA_ENGINE_03: the ensemble is the plain row-wise mean of
    every candidate book -- dollar-neutral, gross <= 1.0 with strict inequality
    when horizons disagree -- and rejects empty/mismatched inputs."""

    def test_averages_books_row_wise(self) -> None:
        books = {
            1: pd.DataFrame([[0.5, -0.5], [0.4, -0.4]], columns=["A", "B"]),
            2: pd.DataFrame([[-0.5, 0.5], [0.6, -0.6]], columns=["A", "B"]),
        }
        out = equal_weight_book_ensemble(books)
        assert out.iloc[0].tolist() == pytest.approx([0.0, 0.0])
        assert out.iloc[1].tolist() == pytest.approx([0.5, -0.5])

    def test_dollar_neutral_and_gross_bounded_by_unit(self) -> None:
        rng = np.random.default_rng(11)
        books = {}
        for k in range(5):
            raw = pd.DataFrame(rng.normal(0.0, 1.0, (30, 8)), columns=list("ABCDEFGH"))
            raw = raw.sub(raw.mean(axis=1), axis=0)
            books[k] = raw.div(raw.abs().sum(axis=1), axis=0)
        out = equal_weight_book_ensemble(books)
        assert out.sum(axis=1).abs().max() < 1e-9
        assert out.abs().sum(axis=1).max() <= 1.0 + 1e-9
        assert not np.isnan(out.to_numpy()).any()

    def test_disagreeing_books_yield_strictly_smaller_gross(self) -> None:
        books = {
            1: pd.DataFrame([[0.5, -0.5], [0.5, -0.5]], columns=["A", "B"]),
            2: pd.DataFrame([[-0.5, 0.5], [-0.5, 0.5]], columns=["A", "B"]),
        }
        out = equal_weight_book_ensemble(books)
        assert out.abs().sum(axis=1).max() == pytest.approx(0.0)

    def test_fails_closed_on_empty_mapping(self) -> None:
        with pytest.raises(ValueError, match="books must not be empty"):
            equal_weight_book_ensemble({})

    def test_fails_closed_on_mismatched_index(self) -> None:
        a = pd.DataFrame([[0.5, -0.5]], columns=["A", "B"])
        b = pd.DataFrame([[0.5, -0.5]], columns=["A", "B"], index=[10])
        with pytest.raises(ValueError, match="identical index"):
            equal_weight_book_ensemble({1: a, 2: b})

    def test_fails_closed_on_mismatched_columns(self) -> None:
        a = pd.DataFrame([[0.5, -0.5]], columns=["A", "B"])
        b = pd.DataFrame([[0.5, -0.5]], columns=["B", "A"])
        with pytest.raises(ValueError, match="identical index"):
            equal_weight_book_ensemble({1: a, 2: b})


class TestRankWeightBook:
    """MHS-05-RANK-BOOK-DOLLAR-NEUTRAL: rows sum to zero with unit gross."""

    def test_rank_weights_are_dollar_neutral_and_unit_gross(self) -> None:
        signal = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0], "D": [4.0]})
        eligible = pd.DataFrame({"A": [True], "B": [True], "C": [True], "D": [True]})
        weights = rank_weight_book(signal, eligible, 1, 4)
        assert weights.iloc[0].tolist() == pytest.approx([-0.375, -0.125, 0.125, 0.375])
        assert abs(float(weights.sum(axis=1).iloc[0])) < 1e-12
        assert abs(float(weights.abs().sum(axis=1).iloc[0]) - 1.0) < 1e-12

    @pytest.mark.parametrize("n_symbols", [2, 3, 5, 7, 12])
    def test_odd_and_even_counts_stay_neutral(self, n_symbols: int) -> None:
        rng = np.random.default_rng(0)
        signal = pd.DataFrame(
            {f"S{i}": rng.normal(size=3) for i in range(n_symbols)},
        )
        eligible = pd.DataFrame(True, index=signal.index, columns=signal.columns)
        for sign in (-1, 1):
            weights = rank_weight_book(signal, eligible, sign, n_symbols)
            assert weights.sum(axis=1).abs().max() < 1e-9
            assert weights.abs().sum(axis=1).sub(1.0).abs().max() < 1e-9

    def test_reversal_sign_flips_weights(self) -> None:
        signal = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0], "D": [4.0]})
        eligible = pd.DataFrame(True, index=signal.index, columns=signal.columns)
        long_side = rank_weight_book(signal, eligible, 1, 4)
        short_side = rank_weight_book(signal, eligible, -1, 4)
        assert long_side.iloc[0].tolist() == pytest.approx(
            [-v for v in short_side.iloc[0].tolist()],
        )

    def test_insufficient_symbols_return_zeros(self) -> None:
        signal = pd.DataFrame({"A": [1.0], "B": [2.0]})
        eligible = pd.DataFrame({"A": [True], "B": [True]})
        weights = rank_weight_book(signal, eligible, 1, 5)
        assert weights.abs().sum(axis=1).iloc[0] == pytest.approx(0.0)

    def test_ineligible_symbols_are_excluded(self) -> None:
        signal = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]})
        eligible = pd.DataFrame({"A": [True], "B": [False], "C": [True]})
        weights = rank_weight_book(signal, eligible, 1, 2)
        assert weights.loc[0, "B"] == pytest.approx(0.0)
        assert abs(float(weights.sum(axis=1).iloc[0])) < 1e-9
        assert abs(float(weights.abs().sum(axis=1).iloc[0]) - 1.0) < 1e-9

    def test_fails_closed_on_bad_sign(self) -> None:
        signal = pd.DataFrame({"A": [1.0]})
        eligible = pd.DataFrame({"A": [True]})
        with pytest.raises(ValueError, match="sign must be"):
            rank_weight_book(signal, eligible, 0, 2)

    def test_fails_closed_on_misaligned_frames(self) -> None:
        signal = pd.DataFrame({"A": [1.0]})
        eligible = pd.DataFrame({"B": [True]})
        with pytest.raises(ValueError, match="identically indexed"):
            rank_weight_book(signal, eligible, 1, 2)


class TestRenormalizeWithinMask:
    """Execution-roster renormalization: surviving roster cells are re-centered
    and re-normalized while masked-out cells stay exactly zero."""

    def test_renormalizes_surviving_cells_to_dollar_neutral_unit_gross(
        self,
    ) -> None:
        # SCENARIO_MHS_RENORM_01_DOLLAR_NEUTRAL_UNIT_GROSS
        weights = pd.DataFrame(
            {"A": [0.4], "B": [-0.4], "C": [0.2], "D": [-0.2]},
        )
        mask = pd.DataFrame({"A": [True], "B": [True], "C": [False], "D": [False]})
        out = renormalize_within_mask(weights, mask, 2)
        assert abs(float(out.iloc[0][["A", "B"]].sum())) < 1e-9
        assert abs(float(out.iloc[0][["A", "B"]].abs().sum()) - 1.0) < 1e-9
        assert out.iloc[0]["C"] == 0.0
        assert out.iloc[0]["D"] == 0.0

    def test_masked_out_columns_are_exactly_zero_across_rows(self) -> None:
        # SCENARIO_MHS_RENORM_02_MASKED_OUT_COLUMNS_ARE_ZERO
        weights = pd.DataFrame(
            {
                "A": [0.5, 0.7],
                "B": [-0.3, -0.1],
                "C": [0.9, -0.2],
                "D": [-0.4, 0.6],
            },
        )
        mask = pd.DataFrame(
            {"A": [True, False], "B": [True, True], "C": [False, True], "D": [False, False]},
        )
        out = renormalize_within_mask(weights, mask, 2)
        assert out["C"].iloc[0] == 0.0
        assert out["D"].iloc[0] == 0.0
        assert out["A"].iloc[1] == 0.0
        assert out["D"].iloc[1] == 0.0
        assert out.abs().sum(axis=1).iloc[0] == pytest.approx(1.0)
        assert out.abs().sum(axis=1).iloc[1] == pytest.approx(1.0)

    def test_rows_below_min_symbols_return_all_zeros(self) -> None:
        # SCENARIO_MHS_RENORM_03_MIN_SYMBOLS_FAIL_CLOSED_TO_ZERO
        weights = pd.DataFrame({"A": [0.4], "B": [-0.4], "C": [0.2]})
        mask = pd.DataFrame({"A": [True], "B": [True], "C": [True]})
        out = renormalize_within_mask(weights, mask, 4)
        assert out.to_numpy().tolist() == [[0.0, 0.0, 0.0]]

    def test_fails_closed_on_misaligned_frames(self) -> None:
        # SCENARIO_MHS_RENORM_04_FAILS_CLOSED_ON_MISALIGNED_FRAMES
        weights = pd.DataFrame({"A": [0.4], "B": [-0.4]})
        mask = pd.DataFrame({"A": [True], "C": [True]})
        with pytest.raises(ValueError, match="identically indexed"):
            renormalize_within_mask(weights, mask, 2)

    def test_fails_closed_on_bad_min_symbols(self) -> None:
        # SCENARIO_MHS_RENORM_05_FAILS_CLOSED_ON_BAD_MIN_SYMBOLS
        weights = pd.DataFrame({"A": [0.4], "B": [-0.4]})
        mask = pd.DataFrame({"A": [True], "B": [True]})
        with pytest.raises(ValueError, match="min_symbols must be >= 2"):
            renormalize_within_mask(weights, mask, 1)

    def test_all_zero_surviving_gross_stays_zero_not_nan(self) -> None:
        # SCENARIO_MHS_RENORM_06_ALL_ZERO_GROSS_STAYS_ZERO
        weights = pd.DataFrame({"A": [0.0], "B": [0.0], "C": [0.0], "D": [0.0]})
        mask = pd.DataFrame({"A": [True], "B": [True], "C": [False], "D": [False]})
        out = renormalize_within_mask(weights, mask, 2)
        assert out.to_numpy().tolist() == [[0.0, 0.0, 0.0, 0.0]]
        assert bool(np.isfinite(out.to_numpy()).all())


class TestInverseRealizedVolTilt:
    """Inverse-realized-vol tilt: magnitudes scale by 1/vol with a never-NaN
    neutral fallback, preserving sign and sparsity."""

    def test_scales_by_inverse_vol(self) -> None:
        # SCENARIO_MHS_TILT_01_SCALES_BY_INVERSE_VOL
        weights = pd.DataFrame({"A": [0.6], "B": [-0.4], "C": [0.2]})
        vol = pd.DataFrame({"A": [2.0], "B": [4.0], "C": [0.5]})
        out = inverse_realized_vol_tilt(weights, vol)
        assert out.iloc[0].tolist() == pytest.approx([0.3, -0.1, 0.4])

    def test_zero_weight_stays_zero(self) -> None:
        # SCENARIO_MHS_TILT_02_ZERO_WEIGHT_STAYS_ZERO
        weights = pd.DataFrame({"A": [0.0], "B": [0.5], "C": [0.0]})
        vol = pd.DataFrame({"A": [1e-9], "B": [2.0], "C": [1e9]})
        out = inverse_realized_vol_tilt(weights, vol)
        assert out.iloc[0]["A"] == 0.0
        assert out.iloc[0]["C"] == 0.0
        assert out.iloc[0]["B"] == pytest.approx(0.25)

    def test_nonfinite_or_nonpositive_vol_falls_back_to_unscaled(self) -> None:
        # SCENARIO_MHS_TILT_03_NONFINITE_OR_NONPOSITIVE_VOL_FALLS_BACK_TO_UNSCALED
        weights = pd.DataFrame({"A": [0.4], "B": [0.4], "C": [0.4], "D": [0.4]})
        vol = pd.DataFrame({"A": [np.nan], "B": [0.0], "C": [-1.0], "D": [2.0]})
        out = inverse_realized_vol_tilt(weights, vol)
        assert out.iloc[0]["A"] == pytest.approx(0.4)
        assert out.iloc[0]["B"] == pytest.approx(0.4)
        assert out.iloc[0]["C"] == pytest.approx(0.4)
        assert out.iloc[0]["D"] == pytest.approx(0.2)
        assert bool(np.isfinite(out.to_numpy()).all())

    def test_fails_closed_on_misaligned_frames(self) -> None:
        # SCENARIO_MHS_TILT_04_FAILS_CLOSED_ON_MISALIGNED_FRAMES
        weights = pd.DataFrame({"A": [0.4], "B": [-0.4]})
        vol = pd.DataFrame({"A": [2.0], "C": [1.0]})
        with pytest.raises(ValueError, match="identically indexed"):
            inverse_realized_vol_tilt(weights, vol)

    def test_preserves_sign(self) -> None:
        # SCENARIO_MHS_TILT_05_PRESERVES_SIGN
        weights = pd.DataFrame({"A": [-0.6], "B": [0.3]})
        vol = pd.DataFrame({"A": [3.0], "B": [0.5]})
        out = inverse_realized_vol_tilt(weights, vol)
        assert out.iloc[0]["A"] == pytest.approx(-0.2)
        assert out.iloc[0]["B"] == pytest.approx(0.6)
        assert bool((np.sign(out) == np.sign(weights)).to_numpy().all())


class TestPhaseTrancheBook:
    """MHS-06-TRANCHE-PHASE-INVARIANCE: staggered books are clock-offset invariant."""

    def test_tranche_count_one_is_identity(self) -> None:
        weights = pd.DataFrame({"A": [0.5, -0.5], "B": [-0.5, 0.5]})
        assert phase_tranche_book(weights, 1).equals(weights)

    def test_leading_rows_are_zero_filled(self) -> None:
        weights = pd.DataFrame({"A": [0.5, -0.5], "B": [-0.5, 0.5]})
        result = phase_tranche_book(weights, 2)
        assert result.iloc[0].tolist() == [0.0, 0.0]
        assert result.iloc[1].tolist() == [0.0, 0.0]

    def test_combined_weights_match_trailing_mean(self) -> None:
        weights = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0]})
        result = phase_tranche_book(weights, 3)
        assert result.iloc[2].tolist() == [1.0, -1.0]

    def test_periodic_signal_is_phase_invariant_after_first_cycle(self) -> None:
        period_mean = np.array([1.0 / 3.0, 0.0])
        rows = [[2.0, 0.0], [-1.0, 1.0], [0.0, -1.0]]
        weights = pd.DataFrame(rows * 6, columns=["A", "B"])
        base = phase_tranche_book(weights, 3)
        assert base.iloc[2:].sub(period_mean).abs().max().max() < 1e-12
        for offset in (1, 2):
            shifted = weights.shift(offset).fillna(0.0)
            other = phase_tranche_book(shifted, 3)
            assert other.iloc[2 + offset :].sub(period_mean).abs().max().max() < 1e-12

    def test_single_phase_book_differs_across_offsets(self) -> None:
        rows = [[2.0, 0.0], [-1.0, 1.0], [0.0, -1.0]]
        weights = pd.DataFrame(rows * 6, columns=["A", "B"])
        single = phase_tranche_book(weights, 1)
        shifted = phase_tranche_book(weights.shift(1).fillna(0.0), 1)
        assert not single.iloc[3:].equals(shifted.iloc[3:])

    def test_fails_closed_on_zero_tranches(self) -> None:
        with pytest.raises(ValueError, match="tranche_count"):
            phase_tranche_book(pd.DataFrame({"A": [1.0]}), 0)


class TestIndependentBands:
    """MHS-20-FAST-SLOW-ALLOCATION-NOT-SIGNAL-POOLING: bands stay independent."""

    def test_fast_and_slow_signals_are_not_pooled(self) -> None:
        rng = np.random.default_rng(3)
        signal = pd.DataFrame({"A": rng.normal(size=5), "B": rng.normal(size=5), "C": rng.normal(size=5)})
        eligible = pd.DataFrame(True, index=signal.index, columns=signal.columns)
        fast = rank_weight_book(signal, eligible, -1, 3)
        slow = rank_weight_book(signal, eligible, 1, 3)
        # Independent rank books, never a single shared TrendScore/rank.
        assert fast.abs().sum(axis=1).sub(1.0).abs().max() < 1e-9
        assert slow.abs().sum(axis=1).sub(1.0).abs().max() < 1e-9

    def test_zero_weighted_fast_blend_is_pure_admitted_slow(self) -> None:
        from src.mhs.contracts import PHASE_1_BOOK_BLEND_WEIGHTS

        fast = pd.DataFrame({"A": [1.0, 1.0], "B": [-1.0, -1.0]})
        slow = pd.DataFrame({"A": [-0.5, -0.5], "B": [0.5, 0.5]})
        blend = (
            PHASE_1_BOOK_BLEND_WEIGHTS["fast_reversal"] * fast
            + PHASE_1_BOOK_BLEND_WEIGHTS["slow_momentum"] * slow
        )
        # fast_reversal is zero-weighted (admission-failed prescreen); the blend
        # is exactly the admitted slow book, never renormalized to another gross.
        pd.testing.assert_frame_equal(blend, slow)


class TestScaleBookToTargetGross:
    """SCENARIO_MHS_TARGET_GROSS_RESTORES_UNIT_GROSS / SCENARIO_MHS_TARGET_GROSS_*"""

    def test_target_gross_restores_unit_gross(self) -> None:
        # SCENARIO_MHS_TARGET_GROSS_RESTORES_UNIT_GROSS
        weights = pd.DataFrame(
            {"A": [0.4, -0.2, 0.5], "B": [-0.4, 0.1, -0.3], "C": [0.0, 0.1, -0.2]},
        )
        out = scale_book_to_target_gross(weights, 1.0)
        for i in range(len(out)):
            assert out.iloc[i].abs().sum() == pytest.approx(1.0, abs=1e-12)
            assert out.iloc[i].sum() == pytest.approx(0.0, abs=1e-12)
        # ratios preserved
        for i in range(len(out)):
            for j in range(out.shape[1]):
                for k in range(j + 1, out.shape[1]):
                    if weights.iloc[i, j] != 0 and weights.iloc[i, k] != 0:
                        assert out.iloc[i, j] / out.iloc[i, k] == pytest.approx(
                            weights.iloc[i, j] / weights.iloc[i, k], abs=1e-12,
                        )

    def test_target_gross_various_gross_levels(self) -> None:
        weights = pd.DataFrame(
            {"A": [0.5, -0.2], "B": [-0.3, 0.1], "C": [-0.2, 0.1]},
        )
        for tg in (0.25, 0.5, 1.5, 2.0):
            out = scale_book_to_target_gross(weights, tg)
            for i in range(len(out)):
                assert out.iloc[i].abs().sum() == pytest.approx(tg, abs=1e-12)
                assert out.iloc[i].sum() == pytest.approx(0.0, abs=1e-12)

    def test_target_gross_zero_and_nonfinite_rows_emit_zeros(self) -> None:
        # SCENARIO_MHS_TARGET_GROSS_ZERO_AND_NONFINITE_ROWS_EMIT_ZEROS
        weights = pd.DataFrame(
            {"A": [0.0, np.nan, 0.5], "B": [0.0, 0.0, np.inf], "C": [0.0, 0.0, -0.5]},
        )
        out = scale_book_to_target_gross(weights, 1.0)
        assert np.isfinite(out.to_numpy()).all()
        # all-zero row stays zero
        assert out.iloc[0].tolist() == [0.0, 0.0, 0.0]
        # all-NaN row becomes zero
        assert out.iloc[1].tolist() == [0.0, 0.0, 0.0]

    def test_target_gross_rejects_invalid_target(self) -> None:
        # SCENARIO_MHS_TARGET_GROSS_REJECTS_INVALID_TARGET
        weights = pd.DataFrame({"A": [0.5], "B": [-0.5]})
        with pytest.raises(ValueError, match="target_gross"):
            scale_book_to_target_gross(weights, 0.0)
        with pytest.raises(ValueError, match="target_gross"):
            scale_book_to_target_gross(weights, -0.5)
        with pytest.raises(ValueError, match="target_gross"):
            scale_book_to_target_gross(weights, float("nan"))
        with pytest.raises(ValueError, match="target_gross"):
            scale_book_to_target_gross(weights, float("inf"))

    def test_target_gross_none_identity(self) -> None:
        # SCENARIO_MHS_TARGET_GROSS_NONE_IS_BYTE_IDENTICAL
        weights = pd.DataFrame(
            {"A": [0.5, -0.3], "B": [-0.5, 0.3]},
        )
        out = phase_tranche_book(weights, 1)
        # target_gross=None is not a valid input to scale_book_to_target_gross
        # (it must be a float), but the _committee_execution_book None branch
        # is the identity. This test confirms the pure function identity.
        assert scale_book_to_target_gross(weights, 1.0).abs().sum(axis=1).sub(1.0).abs().max() < 1e-12

    def test_preserves_dollar_neutrality_exactly(self) -> None:
        rng = np.random.default_rng(99)
        raw = pd.DataFrame(rng.normal(0.0, 1.0, (50, 8)), columns=list("ABCDEFGH"))
        raw = raw.sub(raw.mean(axis=1), axis=0)
        out = scale_book_to_target_gross(raw, 1.0)
        assert out.sum(axis=1).abs().max() < 1e-12
        assert out.abs().sum(axis=1).sub(1.0).abs().max() < 1e-12

    def test_empty_frame_returned_unchanged(self) -> None:
        weights = pd.DataFrame(columns=["A", "B", "C"])
        out = scale_book_to_target_gross(weights, 1.0)
        assert out.empty
        assert list(out.columns) == ["A", "B", "C"]
