"""Funding-rate carry signal builders (P0 return-source breadth diagnostics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mhs.funding import (
    build_funding_carry_candidate_weights,
    funding_carry_execution_book,
    funding_carry_signal,
)


class TestFundingCarrySignal:
    """SCENARIO_MHS_FUNDING_CARRY_SIGNAL_IS_CAUSAL_01."""

    def test_future_perturbation_leaves_past_rows_unchanged(self) -> None:
        # Causal-perturbation test: the signal's value at row t depends only on
        # bar_funding rows at or before t, so perturbing a future row leaves
        # every earlier output row unchanged (the standard realized_vol test
        # pattern from horizons.py).
        idx = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
        rng = np.random.default_rng(1)
        bf = pd.DataFrame(
            rng.standard_normal((200, 2)) * 0.0001, index=idx, columns=["A", "B"],
        )
        baseline = funding_carry_signal(bf, 72)
        perturbed = bf.copy()
        perturbed.iloc[-1] = 1e6
        after = funding_carry_signal(perturbed, 72)
        pd.testing.assert_frame_equal(baseline.iloc[:-1], after.iloc[:-1])

    def test_short_window_fails_closed_to_nan(self) -> None:
        # The first lookback_hours - 1 rows are NaN (fail-closed short window),
        # matching realized_vol's min_periods=horizon_bars convention.
        idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
        bf = pd.DataFrame(np.ones((100, 2)), index=idx, columns=["A", "B"])
        sig = funding_carry_signal(bf, 72)
        assert sig.shape == bf.shape
        assert sig.iloc[:71].isna().all().all()
        assert sig.iloc[71:].notna().any().any()

    def test_fails_closed_on_zero_lookback(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
        bf = pd.DataFrame({"A": [0.0] * 10}, index=idx)
        with pytest.raises(ValueError, match="lookback_hours"):
            funding_carry_signal(bf, 0)


class TestFundingCarryCandidateWeights:
    """SCENARIO_MHS_FUNDING_CARRY_CANDIDATE_WEIGHTS_REUSE_BOOKS_02."""

    def test_returns_dollar_neutral_unit_gross_book_per_lookback(self) -> None:
        # Delegating to rank_weight_book/phase_tranche_book must preserve the
        # dollar-neutral (row sum ~ 0) and unit-gross (row abs-sum ~ 1) book
        # invariants on qualifying rows -- proving no normalization logic is
        # reimplemented here.
        idx = pd.date_range("2024-01-01", periods=400, freq="1h", tz="UTC")
        cols = [f"S{i}" for i in range(10)]
        rng = np.random.default_rng(5)
        bf = pd.DataFrame(rng.standard_normal((400, 10)) * 0.0001, index=idx, columns=cols)
        eligible = pd.DataFrame(True, index=idx, columns=cols)
        out = build_funding_carry_candidate_weights(
            bf, eligible, 1, (72, 168), min_symbols=4, tranche_count=1,
        )
        assert set(out) == {72, 168}
        for book in out.values():
            assert book.shape == (400, 10)
            qualifying = book.abs().sum(axis=1) > 0
            assert (book.loc[qualifying].sum(axis=1).abs() < 1e-9).all()
            assert ((book.loc[qualifying].abs().sum(axis=1) - 1.0).abs() < 1e-9).all()

    def test_sign_flips_weights_on_identical_input(self) -> None:
        # The two sign values produce sign-flipped books on the same input.
        idx = pd.date_range("2024-01-01", periods=400, freq="1h", tz="UTC")
        cols = [f"S{i}" for i in range(10)]
        rng = np.random.default_rng(5)
        bf = pd.DataFrame(rng.standard_normal((400, 10)) * 0.0001, index=idx, columns=cols)
        eligible = pd.DataFrame(True, index=idx, columns=cols)
        long_book = build_funding_carry_candidate_weights(
            bf, eligible, 1, (72,), min_symbols=4, tranche_count=1,
        )[72]
        short_book = build_funding_carry_candidate_weights(
            bf, eligible, -1, (72,), min_symbols=4, tranche_count=1,
        )[72]
        pd.testing.assert_frame_equal(long_book, -short_book)


class TestFundingCarryExecutionBook:
    """SCENARIO_CARRY_BOOK_SHORTS_HIGHEST_FUNDING."""

    def test_shorts_highest_funding(self) -> None:
        idx = pd.date_range("2024-01-01", periods=400, freq="1h", tz="UTC")
        cols = ["A", "B", "C", "D"]
        rates = [0.003, 0.001, -0.001, -0.003]
        bf = pd.DataFrame({c: [r] * 400 for c, r in zip(cols, rates, strict=True)}, index=idx)
        mask = pd.DataFrame(True, index=idx, columns=cols)
        decision_grid = idx[::24]
        book = funding_carry_execution_book(
            bf, mask, lookback_hours=168,
            decision_grid=decision_grid, tranche_count=1, min_symbols=2,
        )
        post_warmup = book.iloc[168:]
        # Row sum == 0 (dollar-neutral)
        assert (post_warmup.sum(axis=1).abs() < 1e-12).all()
        # Row abs-sum == 1 (unit-gross)
        assert ((post_warmup.abs().sum(axis=1) - 1.0).abs() < 1e-12).all()
        # Short the highest funding (A=0.003), long the lowest (D=-0.003)
        assert (post_warmup["A"] <= 0.0).all()
        assert (post_warmup["D"] >= 0.0).all()

    def test_fails_closed_on_short_history(self) -> None:
        idx = pd.date_range("2024-01-01", periods=400, freq="1h", tz="UTC")
        cols = ["A", "B", "C", "D"]
        bf = pd.DataFrame({c: [0.001] * 400 for c in cols}, index=idx)
        mask = pd.DataFrame(True, index=idx, columns=cols)
        decision_grid = idx[::24]
        book = funding_carry_execution_book(
            bf, mask, lookback_hours=168,
            decision_grid=decision_grid, tranche_count=1, min_symbols=2,
        )
        # Rows before lookback_hours are exactly 0.0
        assert (book.iloc[:168] == 0.0).all().all()

    def test_fails_closed_on_invalid_params(self) -> None:
        idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
        bf = pd.DataFrame({"A": [0.0] * 100, "B": [0.0] * 100}, index=idx)
        mask = pd.DataFrame(True, index=idx, columns=["A", "B"])
        decision_grid = idx[::24]
        with pytest.raises(ValueError, match="tranche_count"):
            funding_carry_execution_book(bf, mask, 168, decision_grid, tranche_count=0)
        with pytest.raises(ValueError, match="min_symbols"):
            funding_carry_execution_book(bf, mask, 168, decision_grid, tranche_count=1, min_symbols=1)


# SCENARIO_CARRY_BOOK_FAILS_CLOSED_ON_SHORT_HISTORY: covered by test_fails_closed_on_short_history above
