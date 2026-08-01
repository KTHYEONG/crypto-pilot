from __future__ import annotations

import pandas as pd
import pytest

from src.research.expert_portfolio.allocator import compute_causal_lcb_weights
from src.research.expert_portfolio.contextual_router import (
    _UNAVAILABLE,
    build_causal_context_labels,
    compute_causal_contextual_winner_weights,
    state_labels,
)
from src.research.expert_portfolio.models import (
    ContextualRouterSpec,
    ExpertDefinition,
    ExpertPortfolioSpec,
)


def _expert(
    expert_id: str,
    family: str = "f1",
    symbols: tuple[str, ...] = ("S1",),
) -> ExpertDefinition:
    return ExpertDefinition(expert_id, "return_source", family, symbols, "run_backtest", "hash")


def _router(min_history_bars: int = 2) -> ContextualRouterSpec:
    return ContextualRouterSpec("BTCUSDT", 1, 1, min_history_bars)


def _panel(
    a: list[float],
    b: list[float],
    start: str = "2024-01-01",
) -> pd.DataFrame:
    n = len(a)
    idx = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def _spec() -> ExpertPortfolioSpec:
    return ExpertPortfolioSpec(experts=(_expert("A", "f1", ("S1",)), _expert("B", "f2", ("S2",))))


class TestContextLabels:
    def test_labels_preserve_index_and_are_causally_classified(self) -> None:
        # Trend uses only closes no later than t; the first row cannot know a
        # trend and volatility, so it must be 'unavailable', and every labelled
        # row is one of the six pre-registered states.
        idx = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
        close = pd.Series([100.0, 102.0, 101.0, 101.5, 100.0], index=idx)
        labels = build_causal_context_labels(close, _router())
        assert labels.index.equals(idx)
        assert labels.iloc[0] == _UNAVAILABLE
        assert set(labels.iloc[1:]).issubset(set(state_labels()))

    def test_labels_never_use_future_rows(self) -> None:
        # Appending future closes must never change an already-labelled row.
        idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
        base = pd.Series([100.0, 102.0, 101.0, 99.0], index=idx)
        short = build_causal_context_labels(base, _router())

        idx_long = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
        long_close = pd.Series([100.0, 102.0, 101.0, 99.0, 500.0, 1.0], index=idx_long)
        long_labels = build_causal_context_labels(long_close, _router())
        pd.testing.assert_series_equal(short, long_labels.iloc[:4])


class TestCausalAttribution:
    def test_changing_r_t_leaves_target_row_t_unchanged(self) -> None:
        # ECR-01: the target decided at close t may use only completed round
        # trips (contexts strictly before t-1); r[t] is attributed to context
        # t-1 and affects decisions strictly later than t.
        idx = pd.date_range("2024-01-01", periods=7, freq="4h", tz="UTC")
        a = [0.0, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02]
        b = [0.0, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01]
        panel = _panel(a, b)
        context = pd.Series(["up_low_vol"] * 7, index=idx)

        original = compute_causal_contextual_winner_weights(
            panel, context, _spec(), _router(),
        )
        panel_perturbed = panel.copy()
        panel_perturbed.iloc[5, panel_perturbed.columns.get_loc("A")] = -0.5
        perturbed = compute_causal_contextual_winner_weights(
            panel_perturbed, context, _spec(), _router(),
        )
        # rows 0..5 must be identical: r[5] must not reach the row-5 decision
        pd.testing.assert_frame_equal(original.iloc[:6], perturbed.iloc[:6])
        # row 6 does use r[5] (attributed to context at row 5), so it changes
        assert original.loc[idx[6], "A"] != perturbed.loc[idx[6], "A"]


class TestStateSpecialists:
    def test_each_state_picks_its_positive_lcb_specialist(self) -> None:
        # ECR-02: with sufficient completed observations each context allocates
        # the full gross exposure to its own positive-LCB specialist; unseen and
        # under-sampled contexts are exactly all CASH.
        idx = pd.date_range("2024-01-01", periods=20, freq="4h", tz="UTC")
        a = [0.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
             -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01]
        b = [0.0, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01,
             0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
        context = pd.Series(
            ["up_low_vol"] * 10 + ["down_high_vol"] * 10, index=idx,
        )
        weights = compute_causal_contextual_winner_weights(
            _panel(a, b), context, _spec(), _router(),
        )
        cash = pd.Series({"A": 0.0, "B": 0.0, "CASH": 1.0})
        # warm-up rows have fewer than min_context_history_bars completed
        # samples and are exactly all CASH
        for t in range(3):
            pd.testing.assert_series_equal(weights.iloc[t], cash, check_names=False)
        # up_low_vol with 2+ completed samples -> specialist A
        for t in range(3, 10):
            pd.testing.assert_series_equal(
                weights.iloc[t],
                pd.Series({"A": 1.0, "B": 0.0, "CASH": 0.0}),
                check_names=False,
            )
        # down_high_vol is unseen at row 10 and under-sampled at rows 11-12 -> CASH
        for t in (10, 11, 12):
            pd.testing.assert_series_equal(weights.iloc[t], cash, check_names=False)
        # down_high_vol with 2+ completed samples -> specialist B
        for t in range(13, 20):
            pd.testing.assert_series_equal(
                weights.iloc[t],
                pd.Series({"A": 0.0, "B": 1.0, "CASH": 0.0}),
                check_names=False,
            )


class TestTieAndIntegrity:
    def test_tied_positive_scores_select_first_declared_expert(self) -> None:
        # ECR-03: identical expert evidence yields an exact tie that is broken
        # by declaration order (A first), never by index or by CASH.
        idx = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
        panel = _panel([0.0, 0.02, 0.02, 0.02, 0.02, 0.02], [0.0, 0.02, 0.02, 0.02, 0.02, 0.02])
        context = pd.Series(["up_low_vol"] * 6, index=idx)
        weights = compute_causal_contextual_winner_weights(
            panel, context, _spec(), _router(),
        )
        for t in range(3, 6):
            assert weights.iloc[t]["A"] == 1.0
            assert weights.iloc[t]["B"] == 0.0
            assert weights.iloc[t]["CASH"] == 0.0

    def test_malformed_contexts_fail_closed(self) -> None:
        idx = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
        panel = _panel([0.0, 0.02, 0.02, 0.02, 0.02, 0.02], [0.0, 0.02, 0.02, 0.02, 0.02, 0.02])
        valid = pd.Series(["up_low_vol"] * 6, index=idx)

        misaligned = pd.Series(["up_low_vol"] * 5, index=idx[:5])
        with pytest.raises(ValueError, match="aligned"):
            compute_causal_contextual_winner_weights(panel, misaligned, _spec(), _router())

        missing = valid.copy()
        missing.iloc[2] = None
        with pytest.raises(ValueError, match="missing"):
            compute_causal_contextual_winner_weights(panel, missing, _spec(), _router())

        non_string = pd.Series([1] * 6, index=idx)
        with pytest.raises(ValueError, match="string"):
            compute_causal_contextual_winner_weights(panel, non_string, _spec(), _router())

        unknown = pd.Series(["mystery_state"] * 6, index=idx)
        with pytest.raises(ValueError, match="unknown"):
            compute_causal_contextual_winner_weights(panel, unknown, _spec(), _router())

        duplicate = pd.Series(
            ["up_low_vol", "up_low_vol", "up_low_vol", "up_low_vol", "up_low_vol", "up_low_vol"],
            index=pd.DatetimeIndex([idx[0], idx[0], idx[1], idx[2], idx[3], idx[4]]),
        )
        with pytest.raises(ValueError, match="aligned"):
            compute_causal_contextual_winner_weights(panel, duplicate, _spec(), _router())

    def test_lcb_mix_path_is_left_untouched(self) -> None:
        # The router must not modify the existing causal LCB weights: the same
        # panel still produces the identical lcb_mix target and per-bar score.
        idx = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
        panel = _panel([0.01] * 10, [-0.01] * 10)
        spec = ExpertPortfolioSpec(
            experts=(_expert("A", "f1", ("S1",)), _expert("B", "f2", ("S2",))),
            min_history_bars=3,
        )
        before = compute_causal_lcb_weights(
            panel, spec, as_of=idx[5],
            previous_weights=pd.Series({"A": 0.0, "B": 0.0, "CASH": 1.0}),
        )
        context = pd.Series(["up_low_vol"] * 10, index=idx)
        compute_causal_contextual_winner_weights(panel, context, spec, _router())
        after = compute_causal_lcb_weights(
            panel, spec, as_of=idx[5],
            previous_weights=pd.Series({"A": 0.0, "B": 0.0, "CASH": 1.0}),
        )
        pd.testing.assert_series_equal(before, after)
