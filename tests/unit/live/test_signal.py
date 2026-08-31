"""SCENARIO_LIVE_07: 인과성 게이트와 아티팩트 행 정확 조회."""

from __future__ import annotations

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.live.errors import CausalityViolation
from src.live.signal import assert_signal_available, latest_target_weights


@pytest.fixture
def artifact(tmp_path):
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-08-23 00:00Z"),
            pd.Timestamp("2026-08-24 00:00Z"),
        ]
    )
    frame = pd.DataFrame({"AAAUSDT": [0.1, 0.2], "BBBUSDT": [-0.1, -0.2]}, index=index)
    path = tmp_path / "deployed_target_weights.parquet"
    frame.to_parquet(path, index=True)
    return path


def test_SCENARIO_LIVE_07_causality_gate(artifact) -> None:
    decision_time = pd.Timestamp("2026-08-24 00:00Z")

    with pytest.raises(CausalityViolation):
        assert_signal_available(decision_time, decision_time)
    with pytest.raises(CausalityViolation):
        assert_signal_available(decision_time, decision_time + pd.Timedelta(minutes=59))

    assert_signal_available(decision_time, decision_time + pd.Timedelta(hours=1))
    assert_signal_available(decision_time, decision_time + pd.Timedelta(hours=2))


def test_latest_target_weights_exact_row_or_fail(artifact, tmp_path) -> None:
    decision_time = pd.Timestamp("2026-08-24 00:00Z")
    weights = latest_target_weights(artifact, decision_time)
    assert weights["AAAUSDT"] == pytest.approx(0.2)

    # v2: missing exact date returns most recent prior row within staleness
    missing_on_grid = pd.Timestamp("2026-08-25 00:00Z")
    held = latest_target_weights(artifact, missing_on_grid)
    assert pd.Timestamp(held.name) == pd.Timestamp("2026-08-24 00:00Z")
    assert held["AAAUSDT"] == pytest.approx(0.2)

    off_grid = pd.Timestamp("2026-08-24 03:00Z")
    with pytest.raises(ValueError, match="24h grid"):
        latest_target_weights(artifact, off_grid)


def test_naive_index_artifact_fails_closed(tmp_path) -> None:
    frame = pd.DataFrame({"AAAUSDT": [0.1]}, index=pd.DatetimeIndex([pd.Timestamp("2026-08-24")]))
    path = tmp_path / "naive.parquet"
    frame.to_parquet(path, index=True)
    with pytest.raises(DataIntegrityError):
        latest_target_weights(path, pd.Timestamp("2026-08-24 00:00Z"))

#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_07_CAUSALITY_GATE",
)

def test_SCENARIO_PARITY_08_decision_marks_fail_closed(tmp_path):
    """SCENARIO_PARITY_08-decision-marks-fail-closed"""
    import pandas as pd
    from src.live.signal import latest_decision_marks
    from src.common.errors import DataIntegrityError
    # case 1: file missing -> None
    artifact_path = tmp_path / "deployed_target_weights.parquet"
    # create a dummy weights file to avoid missing but marks missing
    weights_df = pd.DataFrame({"AAA": [0.1]}, index=pd.DatetimeIndex([pd.Timestamp("2026-01-01", tz="UTC")]))
    weights_df.to_parquet(artifact_path)
    assert latest_decision_marks(artifact_path, pd.Timestamp("2026-01-01", tz="UTC")) is None
    # case 2: file exists but row missing -> DataIntegrityError
    marks_path = tmp_path / "deployed_decision_marks.parquet"
    marks_df = pd.DataFrame({"AAA": [100.0]}, index=pd.DatetimeIndex([pd.Timestamp("2026-01-02", tz="UTC")]))
    marks_df.to_parquet(marks_path)
    try:
        latest_decision_marks(artifact_path, pd.Timestamp("2026-01-01", tz="UTC"))
        pytest.fail("should have raised")
    except DataIntegrityError:
        pass
    # case 3: row exists -> Series float64
    marks_df2 = pd.DataFrame({"AAA": [100.0], "BBB": [200.0]}, index=pd.DatetimeIndex([pd.Timestamp("2026-01-01", tz="UTC")]))
    marks_df2.to_parquet(marks_path)
    s = latest_decision_marks(artifact_path, pd.Timestamp("2026-01-01", tz="UTC"))
    assert s is not None
    assert s.dtype == "float64"
    assert "AAA" in s.index
