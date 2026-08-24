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

    missing_on_grid = pd.Timestamp("2026-08-25 00:00Z")
    with pytest.raises(DataIntegrityError):
        latest_target_weights(artifact, missing_on_grid)

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
