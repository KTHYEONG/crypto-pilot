"""Contract coverage for the MHS application evaluation resource telemetry."""

from src.application.research.mhs.evaluation import MhsResourceMeasurement, _StageRecorder


def test_mhs_resource_measurement_records_ordered_stage_data() -> None:
    recorder = _StageRecorder(log_run=False)
    recorder.record("unit_stage", grid_bars=3, n_symbols=2, fill_count=1)

    records = recorder.records
    assert len(records) == 1
    assert records[0] == MhsResourceMeasurement(
        stage="unit_stage",
        elapsed_ms=records[0].elapsed_ms,
        rss_bytes=records[0].rss_bytes,
        grid_bars=3,
        n_symbols=2,
        fill_count=1,
    )
