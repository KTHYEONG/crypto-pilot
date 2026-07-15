"""Contract tests for tiered-workflow result dataclasses."""

from __future__ import annotations

from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer1Result


def _minimal_layer1_result(*, selected_timeframe: str | None = None) -> Layer1Result:
    return Layer1Result(
        signals_per_fold=(),
        oos_stacked={},
        pooled_ic=0.0,
        pooled_tstat=0.0,
        breadth=0.0,
        valid_coverage=0.0,
        fold_pass_ratio=0.0,
        gate_passed=False,
        n_valid=0,
        n_total=0,
        selected_timeframe=selected_timeframe,
    )


def test_selected_timeframe_defaults_to_none() -> None:
    result = _minimal_layer1_result()

    assert result.selected_timeframe is None


def test_selected_timeframe_is_preserved() -> None:
    result = _minimal_layer1_result(selected_timeframe="4h")

    assert result.selected_timeframe == "4h"
