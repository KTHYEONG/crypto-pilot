"""Group D transport-fault pure-evaluate invariants."""

from __future__ import annotations

from src.live.testnet_drill.report import Verdict
from src.live.testnet_drill.scenarios import transport_faults
from src.live.testnet_drill.scenarios.base import ev

_SPECS = {spec.scenario_id: spec for spec in transport_faults.scenario_specs()}


def test_dropped_request_not_reposted_early() -> None:
    failing = ev("D3", reposted_early=True, reposted_after_horizon=False, miss_limit_reached=False,
                 journal_not_placed=False, never_two_live=True, post_count=2, post_seqs=[3, 5])
    result = _SPECS["D3"].evaluate(failing)
    assert result.verdict is Verdict.FAIL
    assert "horizon" in result.proposal_rationale or "wire" in result.proposal_rationale
    passing = ev("D3", reposted_early=False, reposted_after_horizon=True, miss_limit_reached=True,
                 journal_not_placed=True, never_two_live=True, post_count=2, conservation_passed=True)
    assert _SPECS["D3"].evaluate(passing).verdict is Verdict.PASS


def test_single_cancel_per_id() -> None:
    failing = ev("D4", delete_count=2, confirmed_by_query=True, terminal_once=True)
    assert _SPECS["D4"].evaluate(failing).verdict is Verdict.FAIL
    passing = ev("D4", delete_count=1, confirmed_by_query=True, terminal_once=True, conservation_passed=True)
    assert _SPECS["D4"].evaluate(passing).verdict is Verdict.PASS


def test_resend_limited_to_clock_case() -> None:
    passing = ev("D9", post_count=2, first_code=-1021, resync_count=1, no_duplicate=True, conservation_passed=True)
    assert _SPECS["D9"].evaluate(passing).verdict is Verdict.PASS
    failing = ev("D9", post_count=2, first_code=None, resync_count=0, no_duplicate=True)
    assert _SPECS["D9"].evaluate(failing).verdict is Verdict.FAIL


def test_blind_post_on_5xx() -> None:
    failing = ev("D7", get_retries_bounded=True, post_unknown_no_repost=True, post_fail_closed=True,
                 no_duplicate_exposure=True, post_resent_after_5xx=True)
    assert _SPECS["D7"].evaluate(failing).verdict is Verdict.FAIL


def test_corrupt_success_body_flagged() -> None:
    result = _SPECS["D8"].evaluate(ev("D8", resolved_by_query=False, treated_as_active_without_query=True))
    assert result.verdict is Verdict.FAIL
    assert "H11" in result.hypothesis_ids
    assert result.hypothesis_outcome == "CONFIRMED"


_D5 = {"not_placed": True, "retried_later": True, "no_halt": True, "cancel_outcome": "retried", "resting_left": False,
       "rate_limited_read": True, "retry_after_s": 2, "sleep_s": 30.0, "remaining_window_s": 5.0,
       "sleep_exceeds_window": True, "conservation_passed": True}


def test_retry_after_beyond_window() -> None:
    result = _SPECS["D5"].evaluate(ev("D5", **_D5))
    assert result.verdict is Verdict.PASS
    assert result.outcome_for("H10") == "CONFIRMED"
    assert result.outcome_for("H3") == "REFUTED"
    short = _SPECS["D5"].evaluate(ev("D5", **{**_D5, "sleep_s": 1.0, "sleep_exceeds_window": False}))
    assert short.verdict is Verdict.FAIL


def test_ban_handling() -> None:
    failing = ev("D6", records_after_ban=2, ban_horizon_captured=False, halted=True)
    assert _SPECS["D6"].evaluate(failing).verdict is Verdict.FAIL
    not_halted = ev("D6", records_after_ban=0, ban_horizon_captured=False, halted=False)
    assert _SPECS["D6"].evaluate(not_halted).verdict is Verdict.FAIL
    uncaptured = _SPECS["D6"].evaluate(ev("D6", records_after_ban=0, ban_horizon_captured=False, halted=True,
                                             conservation_passed=True))
    assert uncaptured.verdict is Verdict.FAIL
    result = _SPECS["D6"].evaluate(ev("D6", records_after_ban=0, ban_horizon_captured=True, halted=True,
                                      subsequent_blocked=True, conservation_passed=True))
    assert result.verdict is Verdict.PASS
    assert result.outcome_for("H4") == "REFUTED"
    missing_guard = _SPECS["D6"].evaluate(ev(
        "D6", records_after_ban=0, ban_horizon_captured=True,
        halted=True, conservation_passed=True,
    ))
    assert missing_guard.verdict is Verdict.FAIL


def test_rate_limited_cancel() -> None:
    result = _SPECS["D5"].evaluate(ev("D5", **{**_D5, "cancel_outcome": "aborted", "resting_left": True,
                                               "sleep_exceeds_window": False}))
    assert result.outcome_for("H3") == "CONFIRMED"
    assert result.outcome_for("H10") == "REFUTED"


def test_order_read_failure_must_be_contained() -> None:
    base = {"tick_skipped": True, "counters_reset": True, "aborted_with_cleanup": True, "no_live_order": True,
            "max_failures_exceeded": True, "conservation_passed": True}
    assert _SPECS["D10"].evaluate(ev("D10", **base, order_read_contained=True)).verdict is Verdict.PASS
    assert _SPECS["D10"].evaluate(ev("D10", **base, order_read_contained=False)).verdict is Verdict.FAIL


def test_registry_covers_all_d_scenarios() -> None:
    assert set(_SPECS) == {f"D{i}" for i in range(1, 11)}
