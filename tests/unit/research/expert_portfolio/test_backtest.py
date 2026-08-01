from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.research.baseline.backtest import BacktestResult
from src.research.contracts import CostModel
from src.research.expert_portfolio.backtest import (
    ExpertPortfolioBacktestResult,
    run_expert_portfolio,
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
    code_hash: str = "hash",
) -> ExpertDefinition:
    return ExpertDefinition(expert_id, "return_source", family, symbols, "run_backtest", code_hash)


def _panel(
    returns: dict[str, list[float]],
    start: str = "2024-01-01",
) -> pd.DataFrame:
    n = len(next(iter(returns.values())))
    idx = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(returns, index=idx)


def _cash_weights(spec: ExpertPortfolioSpec, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {e.expert_id: 0.0 for e in spec.experts} | {"CASH": 1.0},
        index=index,
    )


def test_weight_swap_charges_one_way_allocation_cost() -> None:
    # EP-04: a full A-to-B weight swap charges one one-way allocation cost
    # exactly once, never zero and never twice.
    spec = ExpertPortfolioSpec(experts=(
        _expert("A", "f1", ("S1",)),
        _expert("B", "f1", ("S2",)),
    ))
    two_bar_panel = _panel({"A": [0.01, 0.02], "B": [0.01, -0.01]})
    swap_weights = pd.DataFrame(
        {"A": [1.0, 0.0], "B": [0.0, 1.0], "CASH": [0.0, 0.0]},
        index=two_bar_panel.index,
    )
    result = run_expert_portfolio(
        two_bar_panel, spec, CostModel(),
        initial_equity=10_000.0, fixed_weights=swap_weights,
    )
    assert result.allocation_cost.iloc[1] == CostModel().fee_rate + CostModel().slippage_rate
    assert result.allocation_cost.iloc[0] == 0.0


class TestMasterLedger:
    def test_decision_at_close_applies_to_next_bar_only(self) -> None:
        # The target decided at bar t must apply to component return at bar t+1
        # only; bar t's own return is unaffected by its own target.
        spec = ExpertPortfolioSpec(experts=(_expert("e1"),))
        c_alloc = CostModel().fee_rate + CostModel().slippage_rate
        panel = _panel({"e1": [0.01, 0.02, 0.03]})
        fixed_weights = pd.DataFrame(
            {"e1": [1.0, 0.5, 0.5], "CASH": [0.0, 0.5, 0.5]},
            index=panel.index,
        )
        result = run_expert_portfolio(
            panel, spec, CostModel(), initial_equity=10_000.0, fixed_weights=fixed_weights,
        )
        # bar0: cash applied -> 0 return
        assert result.backtest_result.equity.iloc[0] == pytest.approx(10_000.0)
        # bar1: w_0=(1,0) on r_1=0.02, minus one one-way turnover
        expected_1 = 10_000.0 * (1.0 + 0.02 - c_alloc)
        assert result.backtest_result.equity.iloc[1] == pytest.approx(expected_1)
        # bar2: w_1=(0.5,0.5) on r_2=0.03, minus half a one-way turnover
        expected_2 = expected_1 * (1.0 + 0.5 * 0.03 - 0.5 * c_alloc)
        assert result.backtest_result.equity.iloc[2] == pytest.approx(expected_2)

    def test_causal_weights_are_used_when_no_fixed_weights(self) -> None:
        # without pre-computed weights the ledger targets come from the causal series
        # and the result still returns target_weights + allocation cost.
        spec = ExpertPortfolioSpec(
            experts=(_expert("e1", "f1", ("S1",)), _expert("e2", "f2", ("S2",))),
            min_history_bars=20,
        )
        idx = pd.date_range("2024-01-01", periods=100, freq="4h", tz="UTC")
        rng = np.random.default_rng(3)
        panel = pd.DataFrame(
            {"e1": rng.normal(0.002, 0.005, 100), "e2": rng.normal(0.002, 0.005, 100)},
            index=idx,
        )
        result = run_expert_portfolio(panel, spec, CostModel(), initial_equity=10_000.0)
        assert isinstance(result, ExpertPortfolioBacktestResult)
        assert list(result.target_weights.columns) == ["e1", "e2", "CASH"]
        assert len(result.allocation_cost) == len(panel)
        assert result.allocation_cost.iloc[:20].sum() == 0.0  # all-cash until history is ready
        assert np.isfinite(result.backtest_result.equity.to_numpy()).all()
        assert (result.backtest_result.equity > 0).all()

    def test_signal_delay_bars_shift_application(self) -> None:
        # with one extra delay bar, a target decided at close t is applied at
        # t+2, so the first risky return is deferred by one more bar.
        spec = ExpertPortfolioSpec(experts=(_expert("e1"),))
        c_alloc = CostModel().fee_rate + CostModel().slippage_rate
        panel = _panel({"e1": [0.01, 0.02, 0.03, 0.01]})
        fixed_weights = pd.DataFrame(
            {"e1": [1.0, 1.0, 1.0, 1.0], "CASH": [0.0, 0.0, 0.0, 0.0]},
            index=panel.index,
        )
        result = run_expert_portfolio(
            panel, spec, CostModel(), initial_equity=10_000.0,
            fixed_weights=fixed_weights, signal_delay_bars=1,
        )
        # applied: bar0 cash, bar1 cash, bar2 w_0, bar3 w_1
        assert result.backtest_result.equity.iloc[0] == pytest.approx(10_000.0)
        assert result.backtest_result.equity.iloc[1] == pytest.approx(10_000.0)
        expected_2 = 10_000.0 * (1.0 + 0.03 - c_alloc)
        assert result.backtest_result.equity.iloc[2] == pytest.approx(expected_2)


class TestStressFreeze:
    def test_fixed_weights_are_reused_verbatim_not_recomputed(self) -> None:
        # EP-06: the stress runner must receive identical base targets and never
        # recompute them; even if the panel data changes, the reused targets win.
        spec = ExpertPortfolioSpec(experts=(_expert("e1"), _expert("e2", "f2", ("S2",))))
        base_index = pd.date_range("2024-01-01", periods=60, freq="4h", tz="UTC")
        base_weights = pd.DataFrame(
            {"e1": [1.0] * 60, "e2": [0.0] * 60, "CASH": [0.0] * 60},
            index=base_index,
        )
        panel_a = pd.DataFrame(
            {"e1": [0.001] * 60, "e2": [-0.001] * 60}, index=base_index,
        )
        result = run_expert_portfolio(
            panel_a, spec, CostModel(), initial_equity=10_000.0,
            fixed_weights=base_weights,
        )
        pd.testing.assert_frame_equal(result.target_weights, base_weights)

        # a completely different return panel must not change the reused targets
        panel_b = pd.DataFrame(
            {"e1": [-0.05] * 60, "e2": [-0.05] * 60}, index=base_index,
        )
        result_b = run_expert_portfolio(
            panel_b, spec, CostModel(), initial_equity=10_000.0,
            fixed_weights=base_weights,
        )
        pd.testing.assert_frame_equal(result_b.target_weights, base_weights)

    def test_non_aligned_fixed_weights_are_rejected(self) -> None:
        spec = ExpertPortfolioSpec(experts=(_expert("e1"), _expert("e2", "f2", ("S2",))))
        index = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
        panel = pd.DataFrame({"e1": [0.01, 0.02, 0.03], "e2": [0.01, 0.01, 0.01]}, index=index)
        with pytest.raises(ValueError, match="aligned"):
            run_expert_portfolio(
                panel, spec, CostModel(), initial_equity=10_000.0,
                fixed_weights=_cash_weights(spec, index[:-1]),
            )
        wrong_cols = pd.DataFrame(
            {"e1": [1.0, 1.0, 1.0], "e2": [0.0, 0.0, 0.0], "EXTRA": [0.0, 0.0, 0.0]},
            index=index,
        )
        with pytest.raises(ValueError, match="exactly"):
            run_expert_portfolio(
                panel, spec, CostModel(), initial_equity=10_000.0,
                fixed_weights=wrong_cols,
            )
        with pytest.raises(ValueError, match="finite"):
            run_expert_portfolio(
                panel, spec, CostModel(), initial_equity=10_000.0,
                fixed_weights=_cash_weights(spec, index).assign(e1=np.nan),
            )
        with pytest.raises(ValueError, match="non-negative"):
            run_expert_portfolio(
                panel, spec, CostModel(), initial_equity=10_000.0,
                fixed_weights=_cash_weights(spec, index).assign(e1=-0.1),
            )


class TestIntegrity:
    def test_missing_component_column_raises_data_integrity_error(self) -> None:
        spec = ExpertPortfolioSpec(experts=(_expert("e1"), _expert("e2", "f2", ("S2",))))
        index = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
        panel = pd.DataFrame({"e1": [0.01, 0.02, 0.03]}, index=index)
        with pytest.raises(DataIntegrityError, match="match expert ids"):
            run_expert_portfolio(panel, spec, CostModel())

    def test_missing_component_return_on_exposed_position_fails_closed(self) -> None:
        # EP-05: a missing return on an exposed position raises rather than
        # being zero-filled; the ledger never fabricates a return.
        spec = ExpertPortfolioSpec(experts=(_expert("e1"),))
        index = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
        panel = pd.DataFrame({"e1": [0.01, np.nan, 0.03]}, index=index)
        fixed_weights = pd.DataFrame(
            {"e1": [1.0, 1.0, 1.0], "CASH": [0.0, 0.0, 0.0]}, index=index,
        )
        with pytest.raises(DataIntegrityError, match="missing"):
            run_expert_portfolio(
                panel, spec, CostModel(), initial_equity=10_000.0,
                fixed_weights=fixed_weights,
            )

    def test_non_finite_panel_and_short_panel_fail_closed(self) -> None:
        spec = ExpertPortfolioSpec(experts=(_expert("e1"),))
        index = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
        panel = pd.DataFrame({"e1": [0.01, 0.02, np.inf]}, index=index)
        fixed_weights = pd.DataFrame(
            {"e1": [1.0, 1.0, 1.0], "CASH": [0.0, 0.0, 0.0]}, index=index,
        )
        with pytest.raises(DataIntegrityError, match="non-finite"):
            run_expert_portfolio(
                panel, spec, CostModel(), initial_equity=10_000.0,
                fixed_weights=fixed_weights,
            )
        single = pd.DataFrame({"e1": [0.01]}, index=index[:1])
        with pytest.raises(DataIntegrityError, match="at least 2"):
            run_expert_portfolio(single, spec, CostModel())

    def test_invalid_arguments_fail_closed(self) -> None:
        spec = ExpertPortfolioSpec(experts=(_expert("e1"),))
        index = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
        panel = pd.DataFrame({"e1": [0.01, 0.02, 0.03]}, index=index)
        with pytest.raises(ValueError, match="initial_equity"):
            run_expert_portfolio(panel, spec, CostModel(), initial_equity=0.0)
        with pytest.raises(ValueError, match="signal_delay_bars"):
            run_expert_portfolio(panel, spec, CostModel(), signal_delay_bars=-1)


class TestContextualRouterLedger:
    def test_context_induced_winner_change_applies_one_bar_later(self) -> None:
        # ECR-04: a context-induced A-to-B winner change is applied one bar
        # later and pays the existing one-way allocation cost exactly once.
        c_alloc = CostModel().fee_rate + CostModel().slippage_rate
        spec = ExpertPortfolioSpec(
            experts=(_expert("A", "f1", ("S1",)), _expert("B", "f2", ("S2",))),
            router=ContextualRouterSpec("BTCUSDT", 1, 1, 3),
        )
        idx = pd.date_range("2024-01-01", periods=12, freq="4h", tz="UTC")
        panel = pd.DataFrame({
            "A": [0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, -0.2, -0.2, -0.2],
            "B": [0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2, 0.2, 0.2],
        }, index=idx)
        context = pd.Series(["up_low_vol"] * 12, index=idx)
        result = run_expert_portfolio(panel, spec, CostModel(), decision_context=context)
        target = result.target_weights
        # warm-up CASH until min_context_history_bars, then specialist A
        assert target.iloc[0]["CASH"] == pytest.approx(1.0)
        assert target.iloc[9]["A"] == pytest.approx(1.0)
        # the contextual winner flips to B at decision row 10
        assert target.iloc[10]["B"] == pytest.approx(1.0)
        assert target.iloc[10]["A"] == pytest.approx(0.0)
        # the flip decided at bar 10 is applied at bar 11, one bar later
        assert result.allocation_cost.iloc[10] == pytest.approx(0.0)
        assert result.allocation_cost.iloc[11] == pytest.approx(c_alloc)
        # bar 11 earns the new B target's return, minus exactly one-way cost
        expected_11 = result.backtest_result.equity.iloc[10] * (1.0 + 0.2 - c_alloc)
        assert result.backtest_result.equity.iloc[11] == pytest.approx(expected_11)

    def test_routed_run_requires_decision_context(self) -> None:
        spec = ExpertPortfolioSpec(
            experts=(_expert("A", "f1", ("S1",)), _expert("B", "f2", ("S2",))),
            router=ContextualRouterSpec("BTCUSDT", 1, 1, 1),
        )
        idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
        panel = pd.DataFrame({"A": [0.01, 0.01, 0.01, 0.01], "B": [0.01, 0.01, 0.01, 0.01]}, index=idx)
        with pytest.raises(ValueError, match="decision_context is required"):
            run_expert_portfolio(panel, spec, CostModel())

    def test_fixed_weights_stay_dominant_over_router(self) -> None:
        # passing pre-computed weights alongside decision_context must never recompute a
        # router target; the frozen frame is reused verbatim.
        spec = ExpertPortfolioSpec(
            experts=(_expert("A", "f1", ("S1",)), _expert("B", "f2", ("S2",))),
            router=ContextualRouterSpec("BTCUSDT", 1, 1, 1),
        )
        idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
        panel = pd.DataFrame({"A": [0.01, 0.01, 0.01, 0.01], "B": [0.01, 0.01, 0.01, 0.01]}, index=idx)
        frozen = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0], "B": [0.0, 0.0, 0.0, 0.0], "CASH": [0.0, 0.0, 0.0, 0.0]}, index=idx)
        context = pd.Series(["up_low_vol"] * 4, index=idx)
        result = run_expert_portfolio(
            panel, spec, CostModel(), fixed_weights=frozen, decision_context=context,
        )
        pd.testing.assert_frame_equal(result.target_weights, frozen)


def test_result_carries_component_evidence() -> None:
    spec = ExpertPortfolioSpec(experts=(_expert("e1"), _expert("e2", "f2", ("S2",))))
    idx = pd.date_range("2024-01-01", periods=40, freq="4h", tz="UTC")
    rng = np.random.default_rng(4)
    panel = pd.DataFrame(
        {"e1": rng.normal(0.002, 0.005, 40), "e2": rng.normal(0.002, 0.005, 40)},
        index=idx,
    )
    result = run_expert_portfolio(panel, spec, CostModel(), initial_equity=10_000.0)
    pd.testing.assert_frame_equal(result.component_returns, panel)
    assert isinstance(result.backtest_result, BacktestResult)
    assert result.backtest_result.trades.empty
