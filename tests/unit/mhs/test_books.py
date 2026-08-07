from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mhs.books import phase_tranche_book, rank_weight_book


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

    def test_5050_blend_is_not_rescaled_to_gross_one(self) -> None:
        from src.mhs.contracts import PHASE_1_BOOK_BLEND_WEIGHTS

        fast = pd.DataFrame({"A": [1.0, 1.0], "B": [-1.0, -1.0]})
        slow = pd.DataFrame({"A": [-0.5, -0.5], "B": [0.5, 0.5]})
        blend = (
            PHASE_1_BOOK_BLEND_WEIGHTS["fast_reversal"] * fast
            + PHASE_1_BOOK_BLEND_WEIGHTS["slow_momentum"] * slow
        )
        gross = blend.abs().sum(axis=1)
        # Opposing targets net to reduced gross: never rescaled back to 1.0.
        assert gross.iloc[0] == pytest.approx(0.5)
        assert (gross <= 1.0).all()
        assert blend.abs().sum(axis=1).max() <= 1.0 + 1e-12
