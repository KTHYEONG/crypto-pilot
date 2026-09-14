def test_decide_deploy_proceeds_without_heartbeat() -> None:
    from datetime import datetime
    from tools.devops.daemon_idle_gate import decide_deploy

    now = datetime.fromisoformat("2026-09-15T01:30:00+00:00")
    for heartbeat in (
        None,
        {},
        {"stage": "signal"},
        {"stage": "signal", "ts": "not-a-time"},
        {"stage": "signal", "ts": "2026-09-15T01:29:00"},
    ):
        decision = decide_deploy(heartbeat, now=now, waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0)
        assert decision.action == "proceed"
        assert decision.reason == "no_heartbeat"


def test_decide_deploy_proceeds_when_idle_or_legacy_heartbeat() -> None:
    from datetime import datetime
    from tools.devops.daemon_idle_gate import decide_deploy

    now = datetime.fromisoformat("2026-09-15T01:30:00+00:00")
    fresh_ts = "2026-09-15T01:29:00+00:00"
    for heartbeat in (
        {"stage": "idle", "status": "COMPLETE", "ts": fresh_ts},
        {"status": "HALT", "ts": fresh_ts},
    ):
        decision = decide_deploy(heartbeat, now=now, waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0)
        assert decision.action == "proceed"
        assert decision.reason == "idle"


def test_decide_deploy_waits_on_fresh_busy_stage() -> None:
    from datetime import datetime
    from tools.devops.daemon_idle_gate import BUSY_STAGES, decide_deploy

    now = datetime.fromisoformat("2026-09-15T01:30:00+00:00")
    assert frozenset({"refresh", "signal", "execute"}) == BUSY_STAGES
    for stage in ("refresh", "signal", "execute"):
        heartbeat = {"stage": stage, "status": "RUNNING", "ts": "2026-09-15T01:26:24+00:00"}
        decision = decide_deploy(heartbeat, now=now, waited_s=600.0, max_wait_s=3600.0, stale_after_s=2700.0)
        assert decision.action == "wait"
        assert decision.reason == f"busy:{stage}"


def test_decide_deploy_proceeds_stale_when_busy_heartbeat_too_old() -> None:
    from datetime import datetime
    from tools.devops.daemon_idle_gate import decide_deploy

    now = datetime.fromisoformat("2026-09-15T01:30:00+00:00")
    heartbeat = {"stage": "execute", "status": "RUNNING", "ts": "2026-09-15T00:40:00+00:00"}

    decision = decide_deploy(heartbeat, now=now, waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0)

    assert decision.action == "proceed_stale"
    assert decision.reason == "stale:execute age_s=3000"


def test_decide_deploy_proceeds_after_max_wait() -> None:
    from datetime import datetime
    from tools.devops.daemon_idle_gate import decide_deploy

    now = datetime.fromisoformat("2026-09-15T01:30:00+00:00")
    heartbeat = {"stage": "signal", "status": "RUNNING", "ts": "2026-09-15T01:29:30+00:00"}

    decision = decide_deploy(heartbeat, now=now, waited_s=3600.0, max_wait_s=3600.0, stale_after_s=2700.0)

    assert decision.action == "proceed_timeout"
    assert decision.reason == "max_wait:signal waited_s=3600"


def test_decide_deploy_rejects_naive_now() -> None:
    from datetime import datetime
    import pytest
    from tools.devops.daemon_idle_gate import decide_deploy

    with pytest.raises(ValueError, match="tz-aware"):
        decide_deploy({"stage": "idle", "ts": "2026-09-15T01:29:00+00:00"}, now=datetime(2026, 9, 15, 1, 30), waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0)


def test_gate_main_exit_codes_and_output(tmp_path, capsys) -> None:
    import json
    from tools.devops.daemon_idle_gate import EXIT_PROCEED, EXIT_WAIT, main

    assert (EXIT_PROCEED, EXIT_WAIT) == (0, 10)
    busy = tmp_path / "busy.json"
    busy.write_text(json.dumps({"stage": "signal", "status": "RUNNING", "ts": "2026-09-15T01:29:00+00:00"}), encoding="utf-8")
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    garbage = tmp_path / "garbage.json"
    garbage.write_text("<html>", encoding="utf-8")
    listing = tmp_path / "list.json"
    listing.write_text("[1, 2]", encoding="utf-8")
    base = ["--now", "2026-09-15T01:30:00+00:00", "--max-wait-s", "3600", "--stale-after-s", "2700"]

    assert main(["--heartbeat-file", str(busy), "--waited-s", "0", *base]) == EXIT_WAIT
    assert capsys.readouterr().out.strip() == "action=wait reason=busy:signal"

    assert main(["--heartbeat-file", str(busy), "--waited-s", "3600", *base]) == EXIT_PROCEED
    assert capsys.readouterr().out.strip() == "action=proceed_timeout reason=max_wait:signal waited_s=3600"

    for unusable in (empty, garbage, listing):
        assert main(["--heartbeat-file", str(unusable), "--waited-s", "0", *base]) == EXIT_PROCEED
        assert capsys.readouterr().out.strip() == "action=proceed reason=no_heartbeat"


def test_gate_main_uses_utc_clock_when_now_omitted(tmp_path, capsys) -> None:
    import json
    from datetime import UTC, datetime, timedelta
    from tools.devops.daemon_idle_gate import DEFAULT_MAX_WAIT_S, DEFAULT_STALE_AFTER_S, EXIT_WAIT, main

    assert (DEFAULT_MAX_WAIT_S, DEFAULT_STALE_AFTER_S) == (3600.0, 2700.0)
    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({"stage": "execute", "ts": (datetime.now(UTC) - timedelta(minutes=5)).isoformat()}), encoding="utf-8")

    assert main(["--heartbeat-file", str(hb), "--waited-s", "0"]) == EXIT_WAIT
    assert capsys.readouterr().out.strip() == "action=wait reason=busy:execute"


def test_gate_script_runs_as_stdlib_entry_point(tmp_path) -> None:
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({"stage": "refresh", "ts": "2026-09-15T01:29:00+00:00"}), encoding="utf-8")

    result = subprocess.run(  # noqa: S603 - fixed argv: this interpreter + the repo gate script
        [sys.executable, "-I", str(root / "tools" / "devops" / "daemon_idle_gate.py"), "--heartbeat-file", str(hb), "--waited-s", "0", "--now", "2026-09-15T01:30:00+00:00"],
        capture_output=True, text=True, check=False, cwd=tmp_path,
    )

    assert result.returncode == 10
    assert result.stdout.strip() == "action=wait reason=busy:refresh"
