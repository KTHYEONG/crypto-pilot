"""Contract scenarios XSV3-02, XSV3-03, and XSV4-01..05 for the XS contextual router.

XSV3-02-CASH-BEFORE-EVIDENCE, XSV3-03-CAUSAL-ATTRIBUTION, and the score-layer
relocation scenarios XSV4-01-CASH-BEFORE-EVIDENCE, XSV4-02-SCORE-ROW-SELECTION,
XSV4-03-CAUSAL-ATTRIBUTION, XSV4-04-VALIDATION-PARITY,
XSV4-05-SHARED-SELECTION-HELPER.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.research.expert_portfolio.contextual_router import state_labels
from src.research.expert_portfolio.models import ContextualRouterSpec
from src.research.technical_experts.xs_contextual_router import (
    build_xs_causal_contextual_allocation,
    build_xs_causal_score_selection,
    build_xs_context_market,
)

_FAMILIES = ("trend", "funding_contrarian", "taker_imbalance")
_CASH = "CASH"


def _router_spec(min_history: int = 168) -> ContextualRouterSpec:
    return ContextualRouterSpec("XS_EQUAL_WEIGHT_MARKET", 42, 42, min_history, 0.90)


def _make_sleeves(
    rows: int, cols: int = 4, seed: int = 0,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DatetimeIndex]:
    index = pd.date_range("2024-01-01", periods=rows, freq="4h", tz="UTC")
    columns = [f"S{i}" for i in range(cols)]
    rng = np.random.default_rng(seed)
    weights = {
        name: pd.DataFrame(
            rng.uniform(-0.3, 0.3, size=(rows, cols)),
            index=index, columns=columns,
        )
        for name in _FAMILIES
    }
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(rows, len(_FAMILIES))),
        index=index, columns=list(_FAMILIES),
    )
    return weights, returns, index


class TestContextMarket:
    def test_xsv3_02_context_market_is_equal_weight_geometric(self) -> None:
        closes = pd.DataFrame(
            {
                "A": [100.0, 110.0],
                "B": [200.0, 220.0],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC"),
        )
        market = build_xs_context_market(closes)
        expected = np.exp(np.log(closes).mean(axis=1))
        assert np.allclose(market.to_numpy(), expected.to_numpy(), atol=1e-12)
        assert float(market.iloc[0]) == pytest.approx(np.sqrt(100.0 * 200.0))

    def test_xsv3_02_malformed_closes_fail_closed(self) -> None:
        closes = pd.DataFrame(
            {"A": [100.0, np.nan]},
            index=pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC"),
        )
        with pytest.raises(DataIntegrityError, match="strictly positive"):
            build_xs_context_market(closes)


class TestCashBeforeEvidence:
    def test_xsv3_02_unavailable_label_is_cash(self) -> None:
        weights, returns, index = _make_sleeves(rows=50)
        context = pd.Series(["unavailable"] * len(index), index=index)
        allocation = build_xs_causal_contextual_allocation(
            weights, returns, context, _router_spec(),
        )
        assert (allocation.selected_sleeve == _CASH).all()
        assert float(allocation.target_weights.abs().to_numpy().max()) == 0.0

    def test_xsv3_02_under_sampled_state_is_cash(self) -> None:
        weights, returns, index = _make_sleeves(rows=100)
        context = pd.Series(["up_low_vol"] * len(index), index=index)
        allocation = build_xs_causal_contextual_allocation(
            weights, returns, context, _router_spec(min_history=168),
        )
        assert (allocation.selected_sleeve == _CASH).all()
        assert float(allocation.target_weights.abs().to_numpy().max()) == 0.0

    def test_xsv3_02_non_positive_winner_is_cash(self) -> None:
        weights, returns, index = _make_sleeves(rows=300, seed=5)
        returns = pd.DataFrame(-0.05, index=index, columns=returns.columns)
        context = pd.Series(["up_low_vol"] * len(index), index=index)
        allocation = build_xs_causal_contextual_allocation(
            weights, returns, context, _router_spec(),
        )
        assert (allocation.selected_sleeve == _CASH).all()

    def test_xsv3_02_positive_winner_selects_exact_family_target_row(self) -> None:
        weights, returns, index = _make_sleeves(rows=300, seed=3)
        returns = pd.DataFrame(0.001, index=index, columns=returns.columns)
        context = pd.Series(["up_low_vol"] * len(index), index=index)
        allocation = build_xs_causal_contextual_allocation(
            weights, returns, context, _router_spec(),
        )
        invested = allocation.selected_sleeve != _CASH
        assert bool(invested.any())
        assert (allocation.selected_sleeve[invested] == "trend").all()
        for t in np.flatnonzero(invested.to_numpy()):
            assert np.allclose(
                allocation.target_weights.iloc[t].to_numpy(),
                weights["trend"].iloc[t].to_numpy(),
            )


class TestCausalAttribution:
    def test_xsv3_03_current_and_future_returns_cannot_alter_current_allocation(self) -> None:
        weights, returns, index = _make_sleeves(rows=400, seed=7)
        returns = pd.DataFrame(0.002, index=index, columns=returns.columns)
        context = pd.Series(["up_low_vol"] * len(index), index=index)
        spec = _router_spec()
        base = build_xs_causal_contextual_allocation(weights, returns, context, spec)
        assert bool((base.selected_sleeve != _CASH).any())

        t = 250
        perturbed = returns.copy()
        perturbed.iloc[t:] = -0.05
        alt = build_xs_causal_contextual_allocation(
            weights, perturbed, context, spec,
        )
        assert alt.selected_sleeve.iloc[: t + 1].equals(base.selected_sleeve.iloc[: t + 1])
        assert alt.target_weights.iloc[: t + 1].equals(base.target_weights.iloc[: t + 1])
        assert not alt.selected_sleeve.iloc[t:].equals(base.selected_sleeve.iloc[t:])


class TestValidation:
    def test_xsv3_03_misaligned_indices_fail_closed(self) -> None:
        weights, returns, index = _make_sleeves(rows=60)
        shifted = returns.iloc[1:]
        context = pd.Series(["unavailable"] * len(index), index=index)
        with pytest.raises(DataIntegrityError, match="index"):
            build_xs_causal_contextual_allocation(weights, shifted, context, _router_spec())

    def test_xsv3_03_unknown_family_fails_closed(self) -> None:
        weights, returns, index = _make_sleeves(rows=60)
        weights["mystery"] = weights["trend"].copy()
        context = pd.Series(["unavailable"] * len(index), index=index)
        with pytest.raises(DataIntegrityError, match="families"):
            build_xs_causal_contextual_allocation(weights, returns, context, _router_spec())

    def test_xsv3_03_non_finite_returns_fail_closed(self) -> None:
        weights, returns, index = _make_sleeves(rows=60)
        bad = returns.copy()
        bad.iloc[10, 0] = np.nan
        context = pd.Series(["unavailable"] * len(index), index=index)
        with pytest.raises(DataIntegrityError, match="finite"):
            build_xs_causal_contextual_allocation(weights, bad, context, _router_spec())

    def test_xsv3_03_invalid_label_fails_closed(self) -> None:
        weights, returns, index = _make_sleeves(rows=60)
        context = pd.Series(["not_a_state"] * len(index), index=index)
        with pytest.raises(DataIntegrityError, match="unknown labels"):
            build_xs_causal_contextual_allocation(weights, returns, context, _router_spec())

    def test_xsv3_03_column_mismatch_fails_closed(self) -> None:
        weights, returns, index = _make_sleeves(rows=60)
        bad = returns.rename(columns={"trend": "x"})
        context = pd.Series(["unavailable"] * len(index), index=index)
        with pytest.raises(DataIntegrityError, match="columns"):
            build_xs_causal_contextual_allocation(weights, bad, context, _router_spec())


class TestScoreSelectionCashBeforeEvidence:
    def test_xsv4_01_unavailable_label_is_cash(self) -> None:
        scores, returns, index = _make_sleeves(rows=50)
        context = pd.Series(["unavailable"] * len(index), index=index)
        allocation = build_xs_causal_score_selection(
            scores, returns, context, _router_spec(),
        )
        assert (allocation.selected_sleeve == _CASH).all()
        assert float(allocation.combined_score.abs().to_numpy().max()) == 0.0

    def test_xsv4_01_under_sampled_state_is_cash(self) -> None:
        scores, returns, index = _make_sleeves(rows=100)
        context = pd.Series(["up_low_vol"] * len(index), index=index)
        allocation = build_xs_causal_score_selection(
            scores, returns, context, _router_spec(min_history=168),
        )
        assert (allocation.selected_sleeve == _CASH).all()
        assert float(allocation.combined_score.abs().to_numpy().max()) == 0.0

    def test_xsv4_01_non_positive_winner_is_cash(self) -> None:
        scores, returns, index = _make_sleeves(rows=300, seed=5)
        returns = pd.DataFrame(-0.05, index=index, columns=returns.columns)
        context = pd.Series(["up_low_vol"] * len(index), index=index)
        allocation = build_xs_causal_score_selection(
            scores, returns, context, _router_spec(),
        )
        assert (allocation.selected_sleeve == _CASH).all()


class TestScoreSelectionScoreRow:
    def test_xsv4_02_positive_winner_selects_exact_family_score_row(self) -> None:
        index = pd.date_range("2024-01-01", periods=300, freq="4h", tz="UTC")
        columns = ["S1", "S2", "S3", "S4"]
        rng = np.random.default_rng(11)
        base = rng.uniform(-2.0, 2.0, size=(300, len(columns)))
        base[50, 0] = np.nan
        base[120, 2] = np.nan
        scores = {
            name: pd.DataFrame(base, index=index, columns=columns)
            for name in _FAMILIES
        }
        returns = pd.DataFrame(
            {"trend": 0.001, "funding_contrarian": 0.0, "taker_imbalance": 0.0},
            index=index,
        )
        context = pd.Series(["up_low_vol"] * len(index), index=index)
        allocation = build_xs_causal_score_selection(
            scores, returns, context, _router_spec(),
        )
        invested = allocation.selected_sleeve != _CASH
        assert bool(invested.any())
        assert (allocation.selected_sleeve[invested] == "trend").all()
        for t in np.flatnonzero(invested.to_numpy()):
            assert np.allclose(
                allocation.combined_score.iloc[t].to_numpy(),
                scores["trend"].iloc[t].to_numpy(),
                rtol=0.0, atol=0.0, equal_nan=True,
            )


class TestScoreSelectionCausalAttribution:
    def test_xsv4_03_future_returns_cannot_alter_current_selection(self) -> None:
        scores, returns, index = _make_sleeves(rows=400, seed=7)
        returns = pd.DataFrame(
            {"trend": 0.002, "funding_contrarian": -0.001, "taker_imbalance": -0.001},
            index=index,
        )
        context = pd.Series(["up_low_vol"] * len(index), index=index)
        spec = _router_spec()
        base = build_xs_causal_score_selection(scores, returns, context, spec)
        assert bool((base.selected_sleeve != _CASH).any())

        t = 250
        perturbed = returns.copy()
        perturbed.iloc[t:] = -0.05
        alt = build_xs_causal_score_selection(scores, perturbed, context, spec)
        assert alt.selected_sleeve.iloc[: t + 1].equals(base.selected_sleeve.iloc[: t + 1])
        assert alt.combined_score.iloc[: t + 1].equals(base.combined_score.iloc[: t + 1])
        assert not alt.selected_sleeve.iloc[t:].equals(base.selected_sleeve.iloc[t:])


class TestScoreSelectionValidation:
    def test_xsv4_04_unknown_family_fails_closed(self) -> None:
        scores, returns, index = _make_sleeves(rows=60)
        scores["mystery"] = scores["trend"].copy()
        context = pd.Series(["unavailable"] * len(index), index=index)
        with pytest.raises(DataIntegrityError, match="families"):
            build_xs_causal_score_selection(scores, returns, context, _router_spec())

    def test_xsv4_04_misaligned_index_fails_closed(self) -> None:
        scores, returns, index = _make_sleeves(rows=60)
        shifted = returns.iloc[1:]
        context = pd.Series(["unavailable"] * len(index), index=index)
        with pytest.raises(DataIntegrityError, match="index"):
            build_xs_causal_score_selection(scores, shifted, context, _router_spec())

    def test_xsv4_04_unknown_label_fails_closed(self) -> None:
        scores, returns, index = _make_sleeves(rows=60)
        context = pd.Series(["not_a_state"] * len(index), index=index)
        with pytest.raises(DataIntegrityError, match="unknown labels"):
            build_xs_causal_score_selection(scores, returns, context, _router_spec())

    def test_xsv4_04_non_finite_returns_fail_closed(self) -> None:
        scores, returns, index = _make_sleeves(rows=60)
        bad = returns.copy()
        bad.iloc[10, 0] = np.nan
        context = pd.Series(["unavailable"] * len(index), index=index)
        with pytest.raises(DataIntegrityError, match="finite"):
            build_xs_causal_score_selection(scores, bad, context, _router_spec())

    def test_xsv4_04_column_mismatch_fails_closed(self) -> None:
        scores, returns, index = _make_sleeves(rows=60)
        bad = returns.rename(columns={"trend": "x"})
        context = pd.Series(["unavailable"] * len(index), index=index)
        with pytest.raises(DataIntegrityError, match="columns"):
            build_xs_causal_score_selection(scores, bad, context, _router_spec())


class TestSharedSelectionCore:
    def test_xsv4_05_selected_sleeve_identical_across_layers(self) -> None:
        weights, _returns, index = _make_sleeves(rows=300, seed=13)
        scores = {
            name: pd.DataFrame(
                np.tile(
                    np.linspace(-2.0, 2.0, len(index)),
                    (weights[name].shape[1], 1),
                ).T
                + np.arange(weights[name].shape[1]),
                index=index, columns=list(weights[name].columns),
            )
            for name in _FAMILIES
        }
        states = state_labels()
        context = pd.Series(list(states) * 50, index=index)
        returns = pd.DataFrame(
            {"trend": 0.002, "funding_contrarian": -0.001, "taker_imbalance": -0.001},
            index=index,
        )
        spec = _router_spec(min_history=2)
        weight_allocation = build_xs_causal_contextual_allocation(
            weights, returns, context, spec,
        )
        score_allocation = build_xs_causal_score_selection(
            scores, returns, context, spec,
        )
        assert bool((score_allocation.selected_sleeve != _CASH).any())
        assert score_allocation.selected_sleeve.equals(
            weight_allocation.selected_sleeve,
        )
