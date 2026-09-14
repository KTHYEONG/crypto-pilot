

# --- auto appended from contract: live_alert_gaps ---
def test_signal_step_result_roundtrip_and_decision_time_match(tmp_path) -> None:
    import json

    import pandas as pd

    from src.live.signal_step_result import (
        SIGNAL_STEP_REASON_MAX_CHARS,
        SIGNAL_STEP_RESULT_NAME,
        SIGNAL_STEP_STATUS_FAILED,
        SIGNAL_STEP_STATUS_OK,
        SignalStepResult,
        read_signal_step_result,
        signal_step_result_path,
        write_signal_step_result,
    )

    target = pd.Timestamp("2026-09-15 00:00Z")
    path = signal_step_result_path(tmp_path / "deployed_target_weights.parquet.enc")
    assert path == tmp_path / SIGNAL_STEP_RESULT_NAME
    assert read_signal_step_result(path, target) is None

    # When: OK + 격리 목록 기록
    quarantine = (("AAAUSDT", "decision_bar_missing"), ("BBBUSDT", "unreadable:ArrowInvalid"))
    write_signal_step_result(path, SignalStepResult(decision_time=target, status=SIGNAL_STEP_STATUS_OK, quarantine=quarantine))

    # Then
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "decision_time": "2026-09-15T00:00:00+00:00",
        "status": "OK",
        "error_type": "",
        "reason": "",
        "quarantine": [
            {"symbol": "AAAUSDT", "reason": "decision_bar_missing"},
            {"symbol": "BBBUSDT", "reason": "unreadable:ArrowInvalid"},
        ],
    }
    assert read_signal_step_result(path, target) == SignalStepResult(decision_time=target, status="OK", quarantine=quarantine)
    assert read_signal_step_result(path, target + pd.Timedelta(days=1)) is None

    # When: FAILED + 긴 사유는 잘린다
    write_signal_step_result(
        path,
        SignalStepResult(decision_time=target, status=SIGNAL_STEP_STATUS_FAILED, error_type="DataIntegrityError", reason="x" * (SIGNAL_STEP_REASON_MAX_CHARS + 50)),
    )
    failed = read_signal_step_result(path, target)
    assert failed is not None
    assert (failed.status, failed.error_type) == ("FAILED", "DataIntegrityError")
    assert len(failed.reason) == SIGNAL_STEP_REASON_MAX_CHARS
    assert not (tmp_path / (SIGNAL_STEP_RESULT_NAME + ".tmp")).exists()

    # 깨진/불완전 파일은 관측 전용이라 None
    path.write_text("{", encoding="utf-8")
    assert read_signal_step_result(path, target) is None
    path.write_text('{"status": "OK"}', encoding="utf-8")
    assert read_signal_step_result(path, target) is None
    path.write_text("[]", encoding="utf-8")
    assert read_signal_step_result(path, target) is None


def test_write_signal_step_result_rejects_invalid_payload(tmp_path) -> None:
    import pandas as pd
    import pytest

    from src.live.signal_step_result import SignalStepResult, write_signal_step_result

    path = tmp_path / "result.json"
    with pytest.raises(ValueError, match="tz-aware"):
        write_signal_step_result(path, SignalStepResult(decision_time=pd.Timestamp("2026-09-15"), status="OK"))
    with pytest.raises(ValueError, match="status"):
        write_signal_step_result(path, SignalStepResult(decision_time=pd.Timestamp("2026-09-15 00:00Z"), status="MAYBE"))
    assert not path.exists()


def test_load_quarantine_records_matches_decision_time(tmp_path) -> None:
    import json

    import pandas as pd

    from src.live.signal_step_result import load_quarantine_records

    target = pd.Timestamp("2026-09-15 00:00Z")
    path = tmp_path / "signal_quarantine.json"
    assert load_quarantine_records(path, target) == ()

    # Given: write_quarantine_sidecar 와 같은 포맷
    path.write_text(
        json.dumps({"decision_time": "2026-09-15T00:00:00+00:00", "records": [{"symbol": "AAAUSDT", "reason": "decision_bar_missing"}]}, sort_keys=True),
        encoding="utf-8",
    )
    assert load_quarantine_records(path, target) == (("AAAUSDT", "decision_bar_missing"),)
    assert load_quarantine_records(path, target - pd.Timedelta(days=1)) == ()
    path.write_text("[1", encoding="utf-8")
    assert load_quarantine_records(path, target) == ()
    path.write_text("[]", encoding="utf-8")
    assert load_quarantine_records(path, target) == ()


