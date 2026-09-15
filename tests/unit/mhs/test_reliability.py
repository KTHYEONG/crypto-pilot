

def test_reliability_blocks_invalid_primary_and_overlap() -> None:
    from src.mhs.data_provenance import DataEvidenceTier, DataProvenanceResult
    from src.mhs.reliability import BacktestCertificationLevel, evaluate_backtest_reliability
    provenance = DataProvenanceResult(DataEvidenceTier.REPRODUCIBLE_ARCHIVE, True, (), 'a'*64, 3)
    result = evaluate_backtest_reliability(primary_valid=False, primary_invalid_reasons=('MISSING_ORDER_OHLCV',), selection_overlap_fraction=0.5, fold_committee_weight_leak=None, input_provenance=provenance, data_limitations=())
    assert not result.eligible
    assert result.certification_level is BacktestCertificationLevel.UNCERTIFIED
    assert 'PRIMARY_EXECUTION_INVALID' in result.reason_codes
    assert 'SELECTION_WINDOW_OVERLAP' in result.reason_codes


def test_reliability_separates_historical_from_forward_validation() -> None:
    from src.mhs.data_provenance import DataEvidenceTier, DataProvenanceResult
    from src.mhs.reliability import BacktestCertificationLevel, evaluate_backtest_reliability
    archive = DataProvenanceResult(DataEvidenceTier.REPRODUCIBLE_ARCHIVE, True, (), 'a'*64, 3)
    observed = DataProvenanceResult(DataEvidenceTier.FORWARD_OBSERVED, True, (), None, 100)
    base = {'primary_valid': True, 'primary_invalid_reasons': (), 'selection_overlap_fraction': 0.0, 'fold_committee_weight_leak': {'fold_0': 0.0}, 'input_provenance': archive, 'data_limitations': ('NO_HISTORICAL_L2',)}
    historical = evaluate_backtest_reliability(**base)
    forward = evaluate_backtest_reliability(**base, forward_provenance=observed)
    assert historical.certification_level is BacktestCertificationLevel.REPRODUCIBLE
    assert forward.certification_level is BacktestCertificationLevel.FORWARD_VALIDATED


def test_validation_tracks_never_label_retrospective_as_oos() -> None:
    import pandas as pd
    from src.mhs.reliability import build_validation_track_disclosure
    top = pd.Timestamp('2025-01-01', tz='UTC')
    tracks = build_validation_track_disclosure(selection_overlap_fraction=0.25, fold_committee_weight_leak={'fold_0': 0.0}, top_level_boundary=top, fold_boundaries=[pd.Timestamp('2023-01-01', tz='UTC')])
    assert tracks['retrospective_deployed_config']['independent_oos'] is False
    assert tracks['independent_walk_forward']['independent_oos'] is False
    assert 'SELECTION_WINDOW_OVERLAP' in tracks['independent_walk_forward']['limitations']


def test_reliability_unsealed_inputs_and_fold_leak_cap_history() -> None:
    import pandas as pd
    from src.mhs.data_provenance import DataEvidenceTier, DataProvenanceResult
    from src.mhs.reliability import (
        BacktestCertificationLevel,
        build_validation_track_disclosure,
        evaluate_backtest_reliability,
    )
    stale = DataProvenanceResult(DataEvidenceTier.UNSEALED_ARCHIVE, False, ('STALE_INPUT_MANIFEST',), None, 2)
    result = evaluate_backtest_reliability(primary_valid=True, primary_invalid_reasons=(), selection_overlap_fraction=0.0, fold_committee_weight_leak={'fold_0': 0.5}, input_provenance=stale, data_limitations=())
    assert not result.eligible
    assert result.certification_level is BacktestCertificationLevel.HISTORICAL_ROBUSTNESS
    assert 'FOLD_COMMITTEE_WEIGHT_LEAK' in result.reason_codes
    assert 'STALE_INPUT_MANIFEST' in result.reason_codes
    tracks = build_validation_track_disclosure(selection_overlap_fraction=0.0, fold_committee_weight_leak={'fold_0': 0.5}, top_level_boundary=pd.Timestamp('2025-01-01', tz='UTC'), fold_boundaries=[pd.Timestamp('2023-01-01', tz='UTC')])
    assert tracks['independent_walk_forward']['independent_oos'] is False
    assert 'FOLD_COMMITTEE_WEIGHT_LEAK' in tracks['independent_walk_forward']['limitations']
