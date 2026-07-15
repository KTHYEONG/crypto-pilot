from __future__ import annotations

from src.domain.futures.strategy.candidate_contracts import Layer1FoldReadiness


def test_layer1_fold_readiness_diagnostic_fields_default_to_zero() -> None:
    # Arrange / Act
    report = Layer1FoldReadiness(
        fold_id=0,
        registry_source_end_idx=10,
        outer_oos_start_idx=0,
        outer_oos_end_idx=100,
        ready_symbols=("BTCUSDT",),
    )

    # Assert
    assert report.bars_per_fold_native == 0
    assert report.decision_points_per_calendar_year == 0.0


def test_layer1_fold_readiness_diagnostic_fields_accept_explicit_values() -> None:
    # Arrange / Act
    report = Layer1FoldReadiness(
        fold_id=1,
        registry_source_end_idx=20,
        outer_oos_start_idx=100,
        outer_oos_end_idx=200,
        ready_symbols=("BTCUSDT", "ETHUSDT"),
        bars_per_fold_native=100,
        decision_points_per_calendar_year=365.0,
    )

    # Assert
    assert report.bars_per_fold_native == 100
    assert report.decision_points_per_calendar_year == 365.0
