"""Groups D/E/F executed end-to-end against the two-account HTTP venue: evidence must come from real faults."""

from __future__ import annotations

import io
import json
from decimal import Decimal
from contextlib import redirect_stdout

import pytest

from src.live.testnet_drill import worker
from src.live.testnet_drill.registry import SCENARIOS
from src.live.testnet_drill.report import Verdict
from src.live.testnet_drill.scenarios import restart_recovery


class _Killed(BaseException):
    pass


class _DeadSwitch:
    """Inner transport that refuses every call once the child has been killed."""

    def __init__(self, venue, dead):
        self._venue, self._dead = venue, dead

    def call(self, method, url, headers):
        if self._dead:
            raise _Killed
        return self._venue.call(method, url, headers)

    def close(self):
        self._venue.close()


@pytest.fixture
def in_process_worker(monkeypatch):
    def bind(session, venue):
        def launch(session_dir, run_id, script_path, rules_path, timeout_s):
            dead: list[int] = []

            def kill(code):
                dead.append(code)
                raise _Killed

            argv = ["--session-dir", str(session_dir), "--run-id", run_id, "--script", str(script_path),
                    "--rules", str(rules_path)]
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                try:
                    code = worker.main(argv, kill=kill, inner=_DeadSwitch(venue, dead), settings=session.settings)
                except _Killed:
                    code = dead[0]
            lines = tuple(json.loads(line) for line in buffer.getvalue().splitlines())
            return restart_recovery.WorkerRun(code, lines)

        monkeypatch.setattr(restart_recovery, "launch_worker", launch)

    return bind


def _run(session, scenario_id):
    spec = SCENARIOS[scenario_id]
    evidence = spec.run(session, session.config)
    return evidence, spec.evaluate(evidence)


_EXPECTED = {
    **dict.fromkeys(("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"), Verdict.PASS),
    **dict.fromkeys(("E1", "E2", "E3", "E4", "E5", "E6", "E8", "E9", "E10"), Verdict.PASS),
    **dict.fromkeys(("F1", "F2", "F3", "F6", "F7"), Verdict.PASS),
    "F4": Verdict.OBSERVED,
    **dict.fromkeys(("G1", "G2", "G3", "G4"), Verdict.PASS),
}


@pytest.mark.parametrize("scenario_id", sorted(_EXPECTED))
def test_scenarios_run_against_real_faults(exchange_session, in_process_worker, scenario_id):
    session, venue, _ = exchange_session(config={"symbols": ("BTCUSDT", "ETHUSDT")})
    in_process_worker(session, venue)
    evidence, result = _run(session, scenario_id)
    assert evidence.payload["primary_clean"] is True
    assert not any(name in ("complete_evidence", "valid_evidence") for name, _ in result.expectations), result
    assert result.verdict is _EXPECTED[scenario_id], (result, evidence.payload)


def test_corrupt_success_body_refutes_h11(exchange_session):
    session, _, _ = exchange_session(config={"symbols": ("BTCUSDT", "ETHUSDT")})
    evidence, result = _run(session, "D8")
    assert result.outcome_for("H11") == "REFUTED"
    assert {v["variant"] for v in evidence.payload["variants"]} == {"corrupt", "truncated", "empty"}
    assert all(v["venue_accepted"] for v in evidence.payload["variants"])
    assert all(v["path"] == "resolved_by_query" for v in evidence.payload["variants"])


def test_dropped_request_waits_for_horizon_and_misses(exchange_session):
    from src.live.executor import UNKNOWN_SUBMISSION_MISS_LIMIT

    session, _, _ = exchange_session(config={"symbols": ("BTCUSDT", "ETHUSDT")})
    evidence, _ = _run(session, "D3")
    assert evidence.payload["miss_count"] >= UNKNOWN_SUBMISSION_MISS_LIMIT
    assert len(evidence.payload["post_seqs"]) >= 2


def test_ban_stops_the_wire_in_both_phases(exchange_session):
    session, _, _ = exchange_session(config={"symbols": ("BTCUSDT", "ETHUSDT")})
    evidence, result = _run(session, "D6")
    assert [phase["records_after"] for phase in evidence.payload["phases"]] == [0, 0]
    assert evidence.payload["ban_horizon_captured"] is True
    assert evidence.payload["subsequent_blocked"] is True
    assert result.outcome_for("H4") == "REFUTED"


def test_repeg_kill_without_repeg_is_refused_not_passed(exchange_session, in_process_worker):
    from src.live.testnet_drill.guard import DrillRefused
    from src.live.testnet_drill.scenarios.restart_recovery import _killed_and_restarted

    session, venue, _ = exchange_session(config={"symbols": ("BTCUSDT", "ETHUSDT")})
    in_process_worker(session, venue)
    never = {"action": "cycle", "kill_on_audit_event": {"event": "event_that_never_occurs", "fields": {}}}
    with pytest.raises(DrillRefused, match="kill_point_not_reached"):
        _killed_and_restarted(session, never, [])


def test_repeg_kill_is_reached_when_the_anchor_keeps_the_order_unfilled(exchange_session, in_process_worker):
    session, venue, _ = exchange_session(config={"symbols": ("BTCUSDT", "ETHUSDT")})
    in_process_worker(session, venue)
    evidence, result = _run(session, "E7")
    assert result.verdict is Verdict.PASS, evidence.payload


def test_external_position_adoption_conserves_with_journaled_resync(exchange_session):
    session, _, _ = exchange_session(config={"symbols": ("BTCUSDT", "ETHUSDT")})
    evidence, result = _run(session, "F2")
    assert all(evidence.payload[k] is True for k in ("breach_detected", "derisk_or_halt", "adoption_listed", "breach_cleared"))
    assert all(check.passed for check in evidence.payload["reconciliation"])
    assert result.verdict is Verdict.PASS


def test_funding_soak_books_venue_funding_with_long_pays_sign(exchange_session):
    session, venue, _ = exchange_session(
        config={"symbols": ("BTCUSDT", "ETHUSDT"), "funding_soak_s": 30.0, "funding_poll_s": 5.0}
    )
    venue.funding_interval_ms = 10_000
    evidence, result = _run(session, "F5")
    assert evidence.payload["funding_events"] >= 2
    assert Decimal(evidence.payload["funding_sum"]) < 0
    assert evidence.payload["sign_convention_ok"] is True
    assert result.verdict is Verdict.PASS


def test_funding_soak_flags_inverted_sign(exchange_session):
    session, venue, _ = exchange_session(
        config={"symbols": ("BTCUSDT", "ETHUSDT"), "funding_soak_s": 30.0, "funding_poll_s": 5.0}
    )
    venue.funding_interval_ms = 10_000
    original = venue.settle_funding

    def inverted():
        before = {k: len(a["income"]) for k, a in venue.accounts.items()}
        original()
        for key, account in venue.accounts.items():
            for row in account["income"][before.get(key, 0):]:
                if row["incomeType"] == "FUNDING_FEE":
                    row["income"] = str(-Decimal(row["income"]))

    venue.settle_funding = inverted
    evidence, result = _run(session, "F5")
    assert evidence.payload["sign_convention_ok"] is False
    assert result.verdict is Verdict.FAIL


def test_launch_worker_runs_a_real_child_process(tmp_path):
    from src.live.testnet_drill.scenarios.restart_recovery import launch_worker

    missing = tmp_path / "missing.json"
    run = launch_worker(tmp_path / "ws", "w1", missing, missing, timeout_s=120)
    assert run.exit_code == 1
    assert [m["event"] for m in run.milestones] == ["started", "finished"]
