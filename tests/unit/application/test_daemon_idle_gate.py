from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.application.ops.daemon_idle_gate import (
    DECISION_RELEASE_HOUR_UTC,
    DECISION_SIGNAL_STALENESS_HOURS,
    DECISION_WINDOW_LEAD_MINUTES,
    decide_deploy,
)

D = datetime(2026, 9, 10, 0, 0, tzinfo=UTC)


def _hb(status: str, stage: str, decision: datetime, ts: datetime) -> dict[str, object]:
    return {
        "status": status,
        "stage": stage,
        "decision_time": decision.isoformat(),
        "ts": ts.isoformat(),
    }


def test_deploy_waits_during_retry_backoff_inside_window() -> None:
    now = D.replace(hour=23, minute=40)
    hb = _hb("HALT", "idle", D, now)
    decision = decide_deploy(hb, now=now, waited_s=0.0, max_wait_s=10800.0, stale_after_s=2700.0)
    assert decision.action == "wait"


def test_deploy_waits_while_awaiting_data() -> None:
    now = D.replace(hour=23, minute=20)
    hb = _hb("AWAITING_DATA", "idle", D, now)
    decision = decide_deploy(hb, now=now, waited_s=0.0, max_wait_s=10800.0, stale_after_s=2700.0)
    assert decision.action == "wait"


def test_deploy_proceeds_once_day_completed() -> None:
    now = D + timedelta(days=1, minutes=40)
    hb = _hb("COMPLETE", "idle", D, now)
    decision = decide_deploy(hb, now=now, waited_s=0.0, max_wait_s=10800.0, stale_after_s=2700.0)
    assert decision.action == "proceed"
    assert decision.reason == "cycle_complete"


def test_previous_day_complete_does_not_unlock_today() -> None:
    now = D.replace(hour=23, minute=10)
    hb = _hb("COMPLETE", "idle", D - timedelta(days=1), now)
    decision = decide_deploy(hb, now=now, waited_s=0.0, max_wait_s=10800.0, stale_after_s=2700.0)
    assert decision.action == "wait"


def test_window_lead_starts_before_release() -> None:
    hb = _hb("COMPLETE", "idle", D - timedelta(days=1), D.replace(hour=22, minute=46))
    in_window = decide_deploy(
        hb, now=D.replace(hour=22, minute=46), waited_s=0.0, max_wait_s=10800.0, stale_after_s=2700.0
    )
    assert in_window.action == "wait"
    out_window = decide_deploy(
        hb, now=D.replace(hour=22, minute=44), waited_s=0.0, max_wait_s=10800.0, stale_after_s=2700.0
    )
    assert out_window.action == "proceed"


def test_window_ends_at_staleness_limit() -> None:
    now = D + timedelta(hours=26)
    hb = _hb("HALT", "idle", D, now)
    decision = decide_deploy(hb, now=now, waited_s=0.0, max_wait_s=10800.0, stale_after_s=2700.0)
    assert decision.action == "proceed"


def test_missing_heartbeat_in_window_waits() -> None:
    in_window = decide_deploy(
        None, now=D.replace(hour=23, minute=30), waited_s=0.0, max_wait_s=10800.0, stale_after_s=2700.0
    )
    assert in_window.action == "wait"
    out_window = decide_deploy(
        None, now=D.replace(hour=12, minute=0), waited_s=0.0, max_wait_s=10800.0, stale_after_s=2700.0
    )
    assert out_window.action == "proceed"
    assert out_window.reason == "no_heartbeat"


def test_max_wait_still_escapes() -> None:
    now = D.replace(hour=23, minute=40)
    hb = _hb("HALT", "idle", D, now)
    decision = decide_deploy(hb, now=now, waited_s=10800.0, max_wait_s=10800.0, stale_after_s=2700.0)
    assert decision.action == "proceed_timeout"


def test_constants_mirror_strategy_and_settings() -> None:
    from src.live.settings import LiveSettings
    from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP20_V2

    assert FROZEN_MHS_TOP20_V2.release_hour_utc == DECISION_RELEASE_HOUR_UTC
    assert LiveSettings().max_signal_staleness_hours == DECISION_SIGNAL_STALENESS_HOURS
    assert DECISION_WINDOW_LEAD_MINUTES == 15  # noqa: SIM300 - spec pins the literal


def test_naive_now_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="tz-aware"):
        decide_deploy(None, now=datetime(2026, 9, 10, 23, 0), waited_s=0.0, max_wait_s=1.0, stale_after_s=1.0)


def test_stale_heartbeat_escapes_inside_window() -> None:
    from datetime import timedelta

    now = D.replace(hour=23, minute=40)
    old_ts = now - timedelta(hours=2)
    hb = _hb("HALT", "idle", D, old_ts)
    decision = decide_deploy(hb, now=now, waited_s=0.0, max_wait_s=10800.0, stale_after_s=2700.0)
    assert decision.action == "proceed_stale"


def test_missing_heartbeat_max_wait_escapes_inside_window() -> None:
    decision = decide_deploy(
        None, now=D.replace(hour=23, minute=30), waited_s=10800.0, max_wait_s=10800.0, stale_after_s=2700.0
    )
    assert decision.action == "proceed_timeout"


def test_unparseable_and_naive_ts_wait_inside_window() -> None:
    now = D.replace(hour=23, minute=30)
    bad = {"status": "HALT", "stage": "idle", "decision_time": D.isoformat(), "ts": "not-a-time"}
    assert decide_deploy(bad, now=now, waited_s=0.0, max_wait_s=10800.0, stale_after_s=2700.0).action == "wait"
    naive = {"status": "HALT", "stage": "idle", "decision_time": D.isoformat(), "ts": "2026-09-10T23:00:00"}
    assert decide_deploy(naive, now=now, waited_s=0.0, max_wait_s=10800.0, stale_after_s=2700.0).action == "wait"
