from __future__ import annotations

import json

import pytest

import src.live.runner as runner_mod
from src.live.runner import run_shadow_cycle
from src.live.settings import LiveSettings

from tests.unit.live._runner_stubs import DECISION_TIME, NOW
from tests.unit.live import test_runner_shadow_cycle as _base

artifact = _base.artifact
live_env = _base.live_env


@pytest.mark.parametrize("audit_failed", [False, True])
def test_depth_candidates_expand_capture_without_affecting_execution(
    artifact, live_env, tmp_path, monkeypatch, audit_failed,
) -> None:
    from src.live.audit import AuditLog
    from src.live.depth_candidates import DEPTH_CANDIDATES_NAME, write_depth_candidates

    write_depth_candidates(
        artifact.parent / DEPTH_CANDIDATES_NAME, DECISION_TIME,
        ["CUSDT", "AAAUSDT", "DUSDT"],
    )
    requested = []
    base = runner_mod.ExecutionDepthRecorder

    class Recorder(base):
        def __init__(self, symbols, **kwargs):
            requested.extend(symbols)
            super().__init__(symbols, **kwargs)

    monkeypatch.setattr(runner_mod, "ExecutionDepthRecorder", Recorder)
    original_record = AuditLog.record

    def record(self, event, **kwargs):
        if event == "exec_depth_symbols" and audit_failed:
            raise OSError("audit unavailable")
        return original_record(self, event, **kwargs)

    monkeypatch.setattr(AuditLog, "record", record)
    settings = LiveSettings(
        ledger_path=str(tmp_path / "ledger.json"), exec_depth_candidate_breadth=3,
        exec_depth_max_symbols=3, fills_dir=str(tmp_path / "fills"),
    )
    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert report.status == "COMPLETE"
    assert requested == ["AAAUSDT", "BUSDT", "CUSDT"]
    assert [intent.symbol for intent in live_env] == ["AAAUSDT", "BUSDT"]
    if not audit_failed:
        events = [json.loads(line) for line in (tmp_path / "shadow_cycle.jsonl").read_text().splitlines()]
        event = next(e for e in events if e.get("event") == "exec_depth_symbols")
        assert event["traded"] == 2
        assert event["candidates"] == 1
