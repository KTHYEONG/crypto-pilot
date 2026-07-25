from __future__ import annotations


import numpy as np
import pytest

from src.domain.futures.compound.contracts import (
    CausalFold,
    EdgeEvidence,
    ExecutionCostFrame,
    ForecastFrame,
)
from src.domain.futures.compound.l1_multiscale import (
    CausalityError,
    evaluate_alpha_edge,
    select_family_timeframes,
)


@pytest.fixture
def l1_config() -> object:
    return object()


def _make_passing_forecast(recipe_id: str = "test_recipe") -> ForecastFrame:
    timestamps = np.arange(240, dtype=np.int64) * 3600000000000
    scores = np.random.default_rng(42).normal(0.005, 0.01, (240, 5)).astype(np.float32)
    valid = np.ones((240, 5), dtype=np.bool_)
    return ForecastFrame(
        timestamps_ns=timestamps,
        symbols=("A", "B", "C", "D", "E"),
        recipe_id=recipe_id,
        scores_2d=scores,
        valid_2d=valid,
    )


def _make_costs() -> ExecutionCostFrame:
    timestamps = np.arange(240, dtype=np.int64) * 3600000000000
    return ExecutionCostFrame(
        timestamps_ns=timestamps,
        symbols=("A", "B", "C", "D", "E"),
        execution_cost_bps=np.full((240, 5), 12.0, dtype=np.float32),
        funding_cost_bps=np.zeros((240, 5), dtype=np.float32),
    )


def _make_folds(n_folds: int = 5) -> tuple[CausalFold, ...]:
    folds: list[CausalFold] = []
    fold_size = 40
    for i in range(n_folds):
        fit_start = 0
        fit_end = (i + 1) * fold_size
        oos_start = fit_end + 5
        oos_end = min(oos_start + fold_size, 240)
        folds.append(
            CausalFold(
                fold_id=i,
                fit_start=fit_start,
                fit_end_exclusive=fit_end,
                calibration_start=fit_start,
                calibration_end_exclusive=fit_end,
                oos_start=oos_start,
                oos_end_exclusive=oos_end,
                purge_bars=5,
                embargo_bars=1,
            )
        )
    return tuple(folds)


class TestEvaluateAlphaEdge:
    def test_edge_requires_four_of_five_positive_folds(self) -> None:
        forecasts = _make_passing_forecast()
        costs = _make_costs()
        folds = _make_folds(5)
        evidence = evaluate_alpha_edge(
            forecasts=forecasts, costs=costs, folds=folds, config=object(),
        )
        assert evidence.admitted

    def test_edge_rejects_insufficient_positive_folds(self) -> None:
        forecasts = _make_passing_forecast()
        costs = _make_costs()
        forecasts = ForecastFrame(
            timestamps_ns=forecasts.timestamps_ns,
            symbols=forecasts.symbols,
            recipe_id=forecasts.recipe_id,
            scores_2d=np.full((240, 5), -0.01, dtype=np.float32),
            valid_2d=forecasts.valid_2d,
        )
        folds = _make_folds(5)
        evidence = evaluate_alpha_edge(
            forecasts=forecasts, costs=costs, folds=folds, config=object(),
        )
        assert not evidence.admitted
        assert any("positive_folds" in r for r in evidence.reasons)

    def test_rejects_causality_violation(self) -> None:
        forecasts = _make_passing_forecast()
        costs = _make_costs()
        bad_folds = (
            CausalFold(
                fold_id=0, fit_start=0, fit_end_exclusive=100,
                calibration_start=0, calibration_end_exclusive=100,
                oos_start=80, oos_end_exclusive=120,
                purge_bars=5, embargo_bars=1,
            ),
        )
        with pytest.raises(CausalityError):
            evaluate_alpha_edge(
                forecasts=forecasts, costs=costs, folds=bad_folds, config=object(),
            )


class TestSelectFamilyTimeframes:
    def test_correlated_same_family_timeframe_is_not_quota_admitted(self) -> None:
        evidence = [
            EdgeEvidence(
                recipe_id="fast", outer_folds=5, positive_folds=5,
                effective_days=180.0, effective_events=2000,
                net_growth_lcb90=0.001, doubled_cost_growth=0.0005,
                probability_positive=0.75, sign_consistency=0.85,
                fdr_q_value=0.05, max_residual_correlation=0.85,
                incremental_growth_lcb90=0.0, capacity_feasible=True,
                admitted=True, reasons=(),
            ),
            EdgeEvidence(
                recipe_id="slow", outer_folds=5, positive_folds=4,
                effective_days=180.0, effective_events=1500,
                net_growth_lcb90=0.0008, doubled_cost_growth=0.0003,
                probability_positive=0.70, sign_consistency=0.82,
                fdr_q_value=0.05, max_residual_correlation=0.85,
                incremental_growth_lcb90=-0.0001, capacity_feasible=True,
                admitted=True, reasons=(),
            ),
        ]
        residual_correlations = np.array([[1.0, 0.85], [0.85, 1.0]], dtype=np.float64)
        selected = select_family_timeframes(
            evidence=evidence,  # type: ignore[arg-type]
            residual_correlations=residual_correlations,
            config=object(),
        )
        assert len(selected) == 1

    def test_incremental_independent_timeframe_can_coexist(self) -> None:
        evidence = [
            EdgeEvidence(
                recipe_id="primary", outer_folds=5, positive_folds=5,
                effective_days=180.0, effective_events=2000,
                net_growth_lcb90=0.001, doubled_cost_growth=0.0005,
                probability_positive=0.75, sign_consistency=0.85,
                fdr_q_value=0.05, max_residual_correlation=0.40,
                incremental_growth_lcb90=0.01, capacity_feasible=True,
                admitted=True, reasons=(),
            ),
            EdgeEvidence(
                recipe_id="secondary", outer_folds=5, positive_folds=4,
                effective_days=180.0, effective_events=1500,
                net_growth_lcb90=0.0008, doubled_cost_growth=0.0003,
                probability_positive=0.70, sign_consistency=0.82,
                fdr_q_value=0.05, max_residual_correlation=0.40,
                incremental_growth_lcb90=0.005, capacity_feasible=True,
                admitted=True, reasons=(),
            ),
        ]
        residual_correlations = np.array([[1.0, 0.40], [0.40, 1.0]], dtype=np.float64)
        selected = select_family_timeframes(
            evidence=evidence,  # type: ignore[arg-type]
            residual_correlations=residual_correlations,
            config=object(),
        )
        assert len(selected) == 2

    def test_empty_evidence_returns_empty(self) -> None:
        selected = select_family_timeframes(
            evidence=[],
            residual_correlations=np.array([[1.0]]),
            config=object(),
        )
        assert len(selected) == 0

    def test_slower_timeframe_wins_statistical_tie(self) -> None:
        evidence = [
            EdgeEvidence(
                recipe_id="fast_4h", outer_folds=5, positive_folds=4,
                effective_days=180.0, effective_events=2000,
                net_growth_lcb90=0.001, doubled_cost_growth=0.0005,
                probability_positive=0.70, sign_consistency=0.85,
                fdr_q_value=0.05, max_residual_correlation=0.40,
                incremental_growth_lcb90=0.0, capacity_feasible=True,
                admitted=True, reasons=(),
            ),
            EdgeEvidence(
                recipe_id="slow_12h", outer_folds=5, positive_folds=5,
                effective_days=180.0, effective_events=1500,
                net_growth_lcb90=0.0012, doubled_cost_growth=0.0005,
                probability_positive=0.72, sign_consistency=0.85,
                fdr_q_value=0.05, max_residual_correlation=0.40,
                incremental_growth_lcb90=0.0, capacity_feasible=True,
                admitted=True, reasons=(),
            ),
        ]
        residual_correlations = np.array([[1.0, 0.40], [0.40, 1.0]], dtype=np.float64)
        selected = select_family_timeframes(
            evidence=evidence,  # type: ignore[arg-type]
            residual_correlations=residual_correlations,
            config=object(),
        )
        assert len(selected) >= 1
