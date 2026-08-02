from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.expert_portfolio.allocator import compute_causal_lcb_weights
from src.research.expert_portfolio.contextual_router import (
    _UNAVAILABLE,
    build_causal_context_labels,
    compute_causal_contextual_winner_weights,
    compute_causal_per_symbol_contextual_weights,
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
    # volatility_lookback_bars must be >= 2 so the completed volatility of a
    # non-constant series is strictly positive and the normalized trend is defined.
    return ContextualRouterSpec("BTCUSDT", 1, 2, min_history_bars)


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
        # Trend uses only closes no later than t; the first rows cannot know a
        # trend and volatility, so they must be 'unavailable', and every labelled
        # row is one of the six pre-registered states.
        idx = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
        close = pd.Series([100.0, 102.0, 101.0, 101.5, 100.0], index=idx)
        labels = build_causal_context_labels(close, _router())
        assert labels.index.equals(idx)
        assert labels.iloc[0] == _UNAVAILABLE
        labeled = labels[labels != _UNAVAILABLE]
        assert not labeled.empty
        assert set(labeled).issubset(set(state_labels()))

    def test_labels_never_use_future_rows(self) -> None:
        # Appending future closes must never change an already-labelled row.
        idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
        base = pd.Series([100.0, 102.0, 101.0, 99.0], index=idx)
        short = build_causal_context_labels(base, _router())

        idx_long = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
        long_close = pd.Series([100.0, 102.0, 101.0, 99.0, 500.0, 1.0], index=idx_long)
        long_labels = build_causal_context_labels(long_close, _router())
        pd.testing.assert_series_equal(short, long_labels.iloc[:4])

    def test_positive_scaling_and_appended_bars_do_not_alter_labels(self) -> None:
        # LAP-06: the volatility-normalized quantile trend is scale invariant and
        # strictly causal, so a positive multiplicative rescaling and appended
        # future bars leave every existing label unchanged.
        idx = pd.date_range("2024-01-01", periods=8, freq="4h", tz="UTC")
        close = pd.Series([100.0, 102.0, 101.0, 99.0, 103.0, 100.5, 101.0, 98.0], index=idx)
        router = _router()
        base = build_causal_context_labels(close, router)
        scaled = build_causal_context_labels(close * 1_000.0, router)
        pd.testing.assert_series_equal(base, scaled)

        longer = pd.Series(
            [*close.tolist(), 500.0, 1.0, 700.0],
            index=pd.date_range("2024-01-01", periods=11, freq="4h", tz="UTC"),
        )
        pd.testing.assert_series_equal(base, build_causal_context_labels(longer, router).iloc[:8])

    def test_low_movement_high_variation_data_yields_both_flat_states(self) -> None:
        # LAP-07: causal quantile labeling produces both flat volatility states
        # after readiness on low-movement/high-variation data while insufficient
        # history stays unavailable.
        router = ContextualRouterSpec("BTCUSDT", 1, 4, 1)
        low_movement = [100.0 + 0.001 * (i % 3) for i in range(40)]
        volatile = [
            100.0, 115.0, 90.0, 105.0, 101.0, 95.0, 110.0, 85.0, 103.0, 92.0,
            108.0, 98.0, 99.0, 104.0, 97.0, 102.0,
        ]
        close = pd.Series(
            [*low_movement, *volatile],
            index=pd.date_range(
                "2024-01-01", periods=len(low_movement) + len(volatile), freq="4h", tz="UTC",
            ),
        )
        labels = build_causal_context_labels(close, router)
        present = set(labels[labels != _UNAVAILABLE])
        assert "flat_low_vol" in present
        assert "flat_high_vol" in present
        assert int((labels == _UNAVAILABLE).sum()) >= router.volatility_lookback_bars


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


class TestPerSymbolRouter:
    def test_two_same_symbol_specialists_never_concurrent_but_cross_symbols_can(
        self,
    ) -> None:
        # RAP-07: two BTC specialists (A and C on S1) never receive concurrent
        # weight, while a positive BTC winner and a positive ETH winner (B on
        # S2) can hold simultaneous non-zero weight within the gross cap.
        idx = pd.date_range("2024-01-01", periods=20, freq="4h", tz="UTC")
        panel = pd.DataFrame(
            {
                "A": [0.0, 0.02] * 10,
                "B": [0.0, 0.02] * 10,
                "C": [0.0, 0.015] * 10,
            },
            index=idx,
        )
        context = pd.Series(["up_low_vol"] * 20, index=idx)
        spec = ExpertPortfolioSpec(
            experts=(
                _expert("A", "f1", ("S1",)),
                _expert("B", "f2", ("S2",)),
                _expert("C", "f3", ("S1",)),
            ),
            router=_router(),
            router_kind="per_symbol_winner_v2",
        )
        weights = compute_causal_per_symbol_contextual_weights(
            panel, context, spec, spec.router,
        )
        assert weights.columns.tolist() == ["A", "B", "C", "CASH"]
        for t in range(3, 20):
            row = weights.iloc[t]
            # never both same-symbol specialists at once
            assert not (row["A"] > 0.0 and row["C"] > 0.0)
            if t >= 10:
                # the higher-LCB BTC winner and the ETH winner coexist
                assert row["A"] > 0.0
                assert row["B"] > 0.0
                assert row["C"] == 0.0
                assert row["A"] + row["B"] == pytest.approx(1.0)

    def test_changing_a_future_return_cannot_alter_prior_weights(self) -> None:
        # RAP-08: the decision at close t uses only completed round trips
        # strictly before t; perturbing a future return cannot change weights
        # at or before that timestamp.
        idx = pd.date_range("2024-01-01", periods=7, freq="4h", tz="UTC")
        panel = pd.DataFrame(
            {"A": [0.0, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02], "B": [0.0] * 7},
            index=idx,
        )
        context = pd.Series(["up_low_vol"] * 7, index=idx)
        spec = ExpertPortfolioSpec(
            experts=(_expert("A", "f1", ("S1",)), _expert("B", "f2", ("S2",))),
            router=_router(),
            router_kind="per_symbol_winner_v2",
        )
        original = compute_causal_per_symbol_contextual_weights(
            panel, context, spec, spec.router,
        )
        perturbed_panel = panel.copy()
        perturbed_panel.iloc[5, 0] = -0.5
        perturbed = compute_causal_per_symbol_contextual_weights(
            perturbed_panel, context, spec, spec.router,
        )
        pd.testing.assert_frame_equal(original.iloc[:6], perturbed.iloc[:6])
        assert original.loc[idx[6], "A"] != perturbed.loc[idx[6], "A"]

    def test_no_positive_lcb_states_are_all_cash(self) -> None:
        # RAP-08: a state where no symbol has a strictly positive eligible LCB
        # returns exact all-CASH weights.
        idx = pd.date_range("2024-01-01", periods=20, freq="4h", tz="UTC")
        panel = pd.DataFrame(
            {"A": [0.0, -0.01] * 10, "B": [0.0, -0.01] * 10}, index=idx,
        )
        context = pd.Series(["up_low_vol"] * 20, index=idx)
        spec = ExpertPortfolioSpec(
            experts=(_expert("A", "f1", ("S1",)), _expert("B", "f2", ("S2",))),
            router=_router(),
            router_kind="per_symbol_winner_v2",
        )
        weights = compute_causal_per_symbol_contextual_weights(
            panel, context, spec, spec.router,
        )
        for t in range(3, 20):
            pd.testing.assert_series_equal(
                weights.iloc[t],
                pd.Series({"A": 0.0, "B": 0.0, "CASH": 1.0}),
                check_names=False,
            )

    def test_malformed_inputs_fail_closed(self) -> None:
        # RAP-08: duplicate expert ids, unknown contexts, and misalignment fail.
        idx = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
        panel = _panel([0.0, 0.02] * 3, [0.0, 0.02] * 3)
        spec = ExpertPortfolioSpec(
            experts=(_expert("A", "f1", ("S1",)), _expert("B", "f2", ("S2",))),
            router=_router(),
            router_kind="per_symbol_winner_v2",
        )
        unknown = pd.Series(["mystery_state"] * 6, index=idx)
        with pytest.raises(ValueError, match="unknown"):
            compute_causal_per_symbol_contextual_weights(
                panel, unknown, spec, spec.router,
            )
        with pytest.raises(ValueError, match=r"pd\.Series"):
            compute_causal_per_symbol_contextual_weights(
                panel, ["up_low_vol"] * 6, spec, spec.router,
            )
        non_string = pd.Series([1] * 6, index=idx)
        with pytest.raises(ValueError, match="string"):
            compute_causal_per_symbol_contextual_weights(
                panel, non_string, spec, spec.router,
            )
        missing_label = pd.Series(["up_low_vol"] * 6, index=idx)
        missing_label.iloc[2] = None
        with pytest.raises(ValueError, match="missing labels"):
            compute_causal_per_symbol_contextual_weights(
                panel, missing_label, spec, spec.router,
            )
        misaligned = pd.Series(["up_low_vol"] * 5, index=idx[:5])
        with pytest.raises(ValueError, match="aligned"):
            compute_causal_per_symbol_contextual_weights(
                panel, misaligned, spec, spec.router,
            )
        duplicate = pd.DataFrame(
            {"A": [0.0, 0.02] * 3, "B": [0.0, 0.02] * 3}, index=idx,
        )
        duplicate.columns = ["A", "A"]
        with pytest.raises(ValueError, match="columns must be unique"):
            compute_causal_per_symbol_contextual_weights(
                duplicate,
                pd.Series(["up_low_vol"] * 6, index=idx),
                spec,
                spec.router,
            )

    def test_non_finite_returns_fail_closed(self) -> None:
        # RAP-08: a missing completed return is never zero-filled.
        idx = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
        panel = pd.DataFrame(
            {"A": [0.0, 0.02, np.nan, 0.02, 0.02, 0.02], "B": [0.0] * 6}, index=idx,
        )
        spec = ExpertPortfolioSpec(
            experts=(_expert("A", "f1", ("S1",)), _expert("B", "f2", ("S2",))),
            router=_router(),
            router_kind="per_symbol_winner_v2",
        )
        context = pd.Series(["up_low_vol"] * 6, index=idx)
        with pytest.raises(ValueError, match="finite"):
            compute_causal_per_symbol_contextual_weights(
                panel, context, spec, spec.router,
            )

    def test_duplicate_family_within_symbol_fails_closed(self) -> None:
        # RAP-07: two experts of one family on the same symbol cannot both be
        # candidates for the per-symbol winner router.
        idx = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
        panel = pd.DataFrame(
            {"A": [0.0, 0.02] * 3, "B": [0.0, 0.02] * 3}, index=idx,
        )
        spec = ExpertPortfolioSpec(
            experts=(_expert("A", "f1", ("S1",)), _expert("B", "f1", ("S1",))),
            router=_router(),
            router_kind="per_symbol_winner_v2",
        )
        context = pd.Series(["up_low_vol"] * 6, index=idx)
        with pytest.raises(ValueError, match="duplicate family within symbol"):
            compute_causal_per_symbol_contextual_weights(
                panel, context, spec, spec.router,
            )

    def test_multi_symbol_expert_fails_closed(self) -> None:
        # RAP-07: the per-symbol router only admits single-symbol experts.
        idx = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
        panel = pd.DataFrame(
            {"A": [0.0, 0.02] * 3, "B": [0.0, 0.02] * 3}, index=idx,
        )
        spec = ExpertPortfolioSpec(
            experts=(_expert("A", "f1", ("S1",)), _expert("B", "f2", ("S1", "S2"))),
            router=_router(),
            router_kind="per_symbol_winner_v2",
        )
        context = pd.Series(["up_low_vol"] * 6, index=idx)
        with pytest.raises(ValueError, match="single-symbol"):
            compute_causal_per_symbol_contextual_weights(
                panel, context, spec, spec.router,
            )

    def test_missing_expert_column_fails_closed(self) -> None:
        # RAP-07: an expert absent from the panel is never silently skipped.
        idx = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
        panel = pd.DataFrame({"A": [0.0, 0.02] * 3}, index=idx)
        spec = ExpertPortfolioSpec(
            experts=(_expert("A", "f1", ("S1",)), _expert("B", "f2", ("S2",))),
            router=_router(),
            router_kind="per_symbol_winner_v2",
        )
        context = pd.Series(["up_low_vol"] * 6, index=idx)
        with pytest.raises(ValueError, match="missing from component_returns"):
            compute_causal_per_symbol_contextual_weights(
                panel, context, spec, spec.router,
            )
