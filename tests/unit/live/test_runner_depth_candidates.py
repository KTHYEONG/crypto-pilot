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


@pytest.mark.parametrize("ready", [True, False])
def test_depth_readiness_precedes_submission_and_never_filters_intents(
    artifact, live_env, tmp_path, monkeypatch, ready,
) -> None:
    from src.live.depth_capture import DepthReadiness
    from src.live.depth_candidates import DEPTH_CANDIDATES_NAME, write_depth_candidates
    from src.live.lifecycle import ShutdownFlag

    write_depth_candidates(artifact.parent / DEPTH_CANDIDATES_NAME, DECISION_TIME, ["CUSDT"])
    base = runner_mod.ExecutionDepthRecorder
    received = {}

    class Recorder(base):
        def start(self, *, ready_timeout_s, required, shutdown):
            received.update(timeout=ready_timeout_s, required=required, shutdown=shutdown)
            return DepthReadiness(
                ready, "ready" if ready else "timeout", 0.1, 2, 2 if ready else 1,
                () if ready else ("BUSDT",),
            )

    original_execute = runner_mod.execute_intents

    def execute(client, intents, filters, policy, audit, *args, **kwargs):
        audit.record("first_order_post")
        return original_execute(client, intents, filters, policy, audit, *args, **kwargs)

    monkeypatch.setattr(runner_mod, "ExecutionDepthRecorder", Recorder)
    monkeypatch.setattr(runner_mod, "execute_intents", execute)
    settings = LiveSettings(ledger_path=str(tmp_path / "ledger.json"), exec_depth_ready_timeout_s=1.5)
    shutdown = ShutdownFlag()
    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW, shutdown=shutdown)
    assert report.status == "COMPLETE"
    assert [intent.symbol for intent in live_env] == ["AAAUSDT", "BUSDT"]
    assert received == {"timeout": 1.5, "required": ["AAAUSDT", "BUSDT"], "shutdown": shutdown}
    events = [json.loads(line) for line in (tmp_path / "shadow_cycle.jsonl").read_text().splitlines()]
    names = [event["event"] for event in events]
    assert names.index("exec_depth_ready") < names.index("first_order_post")
    outcome = events[names.index("exec_depth_ready")]
    assert outcome["ready"] is ready
    assert outcome["reason"] == ("ready" if ready else "timeout")
    assert outcome["waited_s"] == 0.1
    assert outcome["required"] == 2
    assert outcome["seen"] == (2 if ready else 1)
    assert outcome["missing"] == ("" if ready else "BUSDT")


def test_depth_readiness_audit_failure_does_not_halt(artifact, live_env, tmp_path, monkeypatch) -> None:
    from src.live.audit import AuditLog

    original = AuditLog.record

    def record(self, event, **kwargs):
        if event == "exec_depth_ready":
            raise OSError("readiness audit unavailable")
        return original(self, event, **kwargs)

    monkeypatch.setattr(AuditLog, "record", record)
    settings = LiveSettings(ledger_path=str(tmp_path / "ledger.json"))
    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert report.status == "COMPLETE"
    assert [intent.symbol for intent in live_env] == ["AAAUSDT", "BUSDT"]
