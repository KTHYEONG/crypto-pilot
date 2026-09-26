"""불변 테스트: daemon liveness evaluation + episode alerting + exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.live.liveness import ContainerObservation, evaluate_daemon_liveness, run_liveness_check
from src.live.settings import LiveSettings


def _container(**overrides) -> ContainerObservation:
    base: dict[str, object] = {"running": True, "restarting": False, "oom_killed": False, "restart_count": 0, "started_at": None}
    base.update(overrides)
    return ContainerObservation(**base)  # type: ignore[arg-type]


def _heartbeat(ts: pd.Timestamp, expected_by=None, stage: str = "execute") -> dict:
    payload = {"ts": ts.isoformat(), "stage": stage, "decision_time": ts.isoformat(), "status": "RUNNING"}
    if expected_by is not None:
        payload["expected_by"] = expected_by.isoformat()
    return payload


def _settings(tmp_path: Path, **overrides) -> LiveSettings:
    base: dict[str, object] = {"alert_outbox_path": str(tmp_path / "outbox.json")}
    base.update(overrides)
    return LiveSettings(**base)  # type: ignore[arg-type]


def test_healthy_daemon_has_no_findings() -> None:
    """신선한 하트비트·실행 중 컨테이너는 finding이 없다."""
    now = pd.Timestamp("2026-08-24 01:10Z")
    hb = _heartbeat(now - pd.Timedelta(seconds=60), expected_by=now + pd.Timedelta(seconds=600))
    assert evaluate_daemon_liveness(hb, _container(), now=now, heartbeat_stale_s=900.0) == ()


def test_stale_heartbeat_detected() -> None:
    """ts가 임계를 초과하면 heartbeat_stale → daemon_unresponsive이다."""
    now = pd.Timestamp("2026-08-24 01:10Z")
    hb = _heartbeat(now - pd.Timedelta(seconds=901), expected_by=now + pd.Timedelta(seconds=600))
    findings = evaluate_daemon_liveness(hb, _container(), now=now, heartbeat_stale_s=900.0)
    assert [f.key for f in findings] == ["heartbeat_stale"]
    assert findings[0].event == "daemon_unresponsive"


def test_hung_stage_detected_despite_fresh_pulse() -> None:
    """ts는 신선해도 expected_by 경과 시 stage_overrun이다."""
    now = pd.Timestamp("2026-08-24 01:10Z")
    hb = _heartbeat(now - pd.Timedelta(seconds=60), expected_by=now - pd.Timedelta(seconds=1), stage="execute")
    findings = evaluate_daemon_liveness(hb, _container(), now=now, heartbeat_stale_s=900.0)
    assert [f.key for f in findings] == ["stage_overrun"]
    assert findings[0].event == "daemon_stage_overrun"


def test_legacy_heartbeat_has_no_false_overrun() -> None:
    """expected_by 없는 구 하트비트는 overrun을 내지 않는다."""
    now = pd.Timestamp("2026-08-24 01:10Z")
    hb = _heartbeat(now - pd.Timedelta(seconds=60))
    assert evaluate_daemon_liveness(hb, _container(), now=now, heartbeat_stale_s=900.0) == ()


def test_container_down_detected() -> None:
    """정지·재시작 중·OOM은 container_down이다."""
    now = pd.Timestamp("2026-08-24 01:10Z")
    hb = _heartbeat(now - pd.Timedelta(seconds=60), expected_by=now + pd.Timedelta(seconds=600))
    for override in ({"running": False}, {"restarting": True}, {"oom_killed": True}):
        findings = evaluate_daemon_liveness(hb, _container(**override), now=now, heartbeat_stale_s=900.0)
        assert [f.key for f in findings] == ["container_down"]
        assert findings[0].event == "daemon_container_down"


def test_missing_or_malformed_heartbeat_is_finding() -> None:
    """None·비-dict·깨진 ts는 heartbeat_missing이다."""
    now = pd.Timestamp("2026-08-24 01:10Z")
    assert evaluate_daemon_liveness(None, _container(), now=now, heartbeat_stale_s=900.0)[0].key == "heartbeat_missing"
    assert evaluate_daemon_liveness("nope", _container(), now=now, heartbeat_stale_s=900.0)[0].key == "heartbeat_missing"  # type: ignore[arg-type]
    assert evaluate_daemon_liveness({"ts": "garbage"}, _container(), now=now, heartbeat_stale_s=900.0)[0].key == "heartbeat_missing"


def test_alert_once_per_episode_then_recover_once(tmp_path, monkeypatch) -> None:
    """연속 장애는 1회만 알리고 복구 시 1회 복구 알림을 보낸다."""
    import src.live.liveness as liveness_mod

    monkeypatch.setattr("src.live.alerting.post_alert", lambda *a, **k: True)
    monkeypatch.setattr("src.live.alerting.send_email_alert", lambda **k: True)
    settings = _settings(tmp_path, alert_webhook_url="https://h.example")
    state_path = tmp_path / "liveness.json"
    now = pd.Timestamp("2026-08-24 01:10Z")
    stale = _heartbeat(now - pd.Timedelta(seconds=5000))
    container = _container()

    monkeypatch.setattr(liveness_mod, "_read_heartbeat_file", lambda settings: stale)
    assert run_liveness_check(settings, container, now=now, state_path=state_path) == 0
    assert run_liveness_check(settings, container, now=now + pd.Timedelta(seconds=300), state_path=state_path) == 0
    assert run_liveness_check(settings, container, now=now + pd.Timedelta(seconds=600), state_path=state_path) == 0

    outbox = json.loads((tmp_path / "outbox.json").read_text(encoding="utf-8"))
    unresponsive = [r for r in outbox["records"] if r["event"] == "daemon_unresponsive"]
    assert len(unresponsive) == 1

    monkeypatch.setattr(liveness_mod, "_read_heartbeat_file", lambda settings: _heartbeat(now + pd.Timedelta(seconds=900), expected_by=now + pd.Timedelta(seconds=3600)))
    assert run_liveness_check(settings, container, now=now + pd.Timedelta(seconds=900), state_path=state_path) == 0
    outbox = json.loads((tmp_path / "outbox.json").read_text(encoding="utf-8"))
    recovered = [r for r in outbox["records"] if r["event"] == "daemon_liveness_recovered"]
    assert len(recovered) == 1


def test_exit_code_escalates_when_critical_undelivered(tmp_path, monkeypatch) -> None:
    """CRITICAL을 어떤 채널로도 전송 못하면 exit 2이다."""
    import src.live.liveness as liveness_mod

    monkeypatch.setattr("src.live.alerting.post_alert", lambda *a, **k: False)
    monkeypatch.setattr("src.live.alerting.send_email_alert", lambda **k: False)
    monkeypatch.setattr(liveness_mod, "_any_channel_delivered", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("lookup down")))
    settings = _settings(tmp_path, alert_webhook_url="https://h.example")
    now = pd.Timestamp("2026-08-24 01:10Z")
    monkeypatch.setattr(liveness_mod, "_read_heartbeat_file", lambda settings: _heartbeat(now - pd.Timedelta(seconds=5000)))
    assert run_liveness_check(settings, _container(), now=now, state_path=tmp_path / "liveness.json") == 2


def test_checker_drains_daemon_backlog(tmp_path, monkeypatch) -> None:
    """죽은 데몬이 남긴 pending 알림을 체커가 배달한다."""
    import src.live.liveness as liveness_mod

    from src.live.alerting import dispatch_alert

    delivered: list[str] = []
    monkeypatch.setattr("src.live.alerting.post_alert", lambda url, *, event, detail, decision_time, now: delivered.append(event) or False)
    settings = _settings(tmp_path, alert_webhook_url="https://h.example")
    now = pd.Timestamp("2026-08-24 01:10Z")
    assert dispatch_alert(settings, event="day_skipped", detail="d", decision_time=None, dedupe_key="day_skipped:none", now=now) is True
    assert delivered == ["day_skipped"]
    monkeypatch.setattr("src.live.alerting.post_alert", lambda url, *, event, detail, decision_time, now: delivered.append(event) or True)
    monkeypatch.setattr(liveness_mod, "_read_heartbeat_file", lambda settings: _heartbeat(now - pd.Timedelta(seconds=60), expected_by=now + pd.Timedelta(seconds=3600)))
    assert run_liveness_check(settings, _container(), now=now + pd.Timedelta(seconds=3600), state_path=tmp_path / "liveness.json") == 0
    assert delivered.count("day_skipped") >= 2


def test_liveness_check_cli_maps_container_flags_and_exits(tmp_path, monkeypatch) -> None:
    """`live liveness-check`는 컨테이너 플래그를 관측값으로 매핑하고 exit 코드를 전달한다."""
    import argparse

    import pytest

    import src.cli.commands.live as live_mod

    seen: dict = {}

    def _fake_run(settings, container, *, now, state_path):
        seen["running"] = container.running
        seen["restarting"] = container.restarting
        seen["oom"] = container.oom_killed
        seen["restarts"] = container.restart_count
        seen["started"] = container.started_at
        seen["state"] = state_path
        return 2

    monkeypatch.setattr("src.live.liveness.run_liveness_check", _fake_run)
    args = argparse.Namespace(
        container_running=0, container_restarting=1, container_oom=0,
        restart_count=3, container_started_at="2026-08-24T00:00:00+00:00",
        state_path=str(tmp_path / "ep.json"), mode=None,
    )
    with pytest.raises(SystemExit) as exc:
        live_mod._run_liveness_check(args)
    assert exc.value.code == 2
    assert seen == {
        "running": False, "restarting": True, "oom": False, "restarts": 3,
        "started": seen["started"], "state": tmp_path / "ep.json",
    }
    assert seen["started"] is not None

    from src.live.liveness import _default_state_path

    args.state_path = None
    with pytest.raises(SystemExit):
        live_mod._run_liveness_check(args)
    assert seen["state"] == _default_state_path()


def test_evaluate_handles_naive_and_unparseable_now() -> None:
    """naive now는 UTC로 간주하고 파싱 불가는 missing finding이다."""
    now = pd.Timestamp("2026-08-24 01:10Z")
    hb = _heartbeat(now - pd.Timedelta(seconds=60), expected_by=now + pd.Timedelta(seconds=600))
    assert evaluate_daemon_liveness(hb, _container(), now=pd.Timestamp("2026-08-24 01:10:00"), heartbeat_stale_s=900.0) == ()
    findings = evaluate_daemon_liveness(hb, _container(), now="garbage", heartbeat_stale_s=900.0)  # type: ignore[arg-type]
    assert [f.key for f in findings] == ["heartbeat_missing"]
    naive_hb = {"ts": "2026-08-24T01:09:00", "stage": "idle"}
    assert evaluate_daemon_liveness(naive_hb, _container(), now=now, heartbeat_stale_s=900.0) == ()


def test_evaluate_outer_guard_catches_mapping_errors() -> None:
    """heartbeat 접근 예외는 missing finding으로 흡수된다."""

    class _Exploding(dict):
        def get(self, key, default=None):
            if key == "expected_by":
                raise RuntimeError("dict down")
            return super().get(key, default)

    now = pd.Timestamp("2026-08-24 01:10Z")
    hb = _Exploding(ts=(now - pd.Timedelta(seconds=60)).isoformat(), stage="idle")
    findings = evaluate_daemon_liveness(hb, _container(), now=now, heartbeat_stale_s=900.0)
    assert [f.key for f in findings] == ["heartbeat_missing"]


def test_episode_state_tolerates_corrupt_and_shapeless_files(tmp_path, monkeypatch) -> None:
    """깨진 episode 파일은 빈 에피소드로 시작하고 검사를 막지 않는다."""
    import src.live.liveness as liveness_mod

    monkeypatch.setattr("src.live.alerting.post_alert", lambda *a, **k: True)
    settings = _settings(tmp_path, alert_webhook_url="https://h.example")
    state_path = tmp_path / "liveness.json"
    now = pd.Timestamp("2026-08-24 01:10Z")
    container = _container()
    monkeypatch.setattr(liveness_mod, "_read_heartbeat_file", lambda settings: _heartbeat(now - pd.Timedelta(seconds=60), expected_by=now + pd.Timedelta(seconds=3600)))
    state_path.write_text("{bad", encoding="utf-8")
    assert run_liveness_check(settings, container, now=now, state_path=state_path) == 0
    state_path.write_text("[1,2]", encoding="utf-8")
    assert run_liveness_check(settings, container, now=now, state_path=state_path) == 0
    state_path.write_text(json.dumps({"open_keys": 5, "episode_started_at": None}), encoding="utf-8")
    assert run_liveness_check(settings, container, now=now, state_path=state_path) == 0


def test_read_heartbeat_file_roundtrip_and_failure(tmp_path, monkeypatch) -> None:
    """실제 하트비트 파일을 읽고 실패 시 None을 반환한다."""
    import src.live.liveness as liveness_mod

    real_reader = liveness_mod._read_heartbeat_file
    assert real_reader is not None
    hb_path = tmp_path / "hb.json"
    hb_path.write_text(json.dumps({"ts": "2026-08-24T01:09:00+00:00", "stage": "idle"}), encoding="utf-8")
    settings = _settings(tmp_path, heartbeat_path=str(hb_path))
    assert real_reader(settings)["stage"] == "idle"  # type: ignore[index]
    missing = _settings(tmp_path, heartbeat_path=str(tmp_path / "missing.json"))
    assert real_reader(missing) is None
    broken = tmp_path / "broken.json"
    broken.write_text("[1,2]", encoding="utf-8")
    assert real_reader(_settings(tmp_path, heartbeat_path=str(broken))) is None


def test_run_liveness_check_rejects_bad_now_and_survives_drain_failure(tmp_path, monkeypatch) -> None:
    """파싱 불가 now는 exit 1, drain 예외는 검사를 막지 않는다."""
    import src.live.liveness as liveness_mod

    settings = _settings(tmp_path)
    assert run_liveness_check(settings, _container(), now="garbage", state_path=tmp_path / "s.json") == 1  # type: ignore[arg-type]
    now = pd.Timestamp("2026-08-24 01:10Z")
    monkeypatch.setattr(liveness_mod, "_read_heartbeat_file", lambda settings: _heartbeat(now - pd.Timedelta(seconds=60), expected_by=now + pd.Timedelta(seconds=3600)))
    monkeypatch.setattr("src.live.alerting.drain_alerts", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("drain down")))
    assert run_liveness_check(settings, _container(), now=now, state_path=tmp_path / "s.json") == 0


def test_run_liveness_check_outer_failure_returns_one(tmp_path, monkeypatch) -> None:
    """평가 예외는 exit 1이다."""
    import src.live.liveness as liveness_mod

    settings = _settings(tmp_path)
    monkeypatch.setattr(liveness_mod, "evaluate_daemon_liveness", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("eval down")))
    assert run_liveness_check(settings, _container(), now=pd.Timestamp("2026-08-24 01:10Z"), state_path=tmp_path / "s.json") == 1


def test_recovery_acceptance_failure_is_not_critical(tmp_path, monkeypatch) -> None:
    """복구 알림 미접수는 CRITICAL 실패로 취급하지 않는다."""
    import src.live.liveness as liveness_mod

    settings = _settings(tmp_path)
    now = pd.Timestamp("2026-08-24 01:10Z")
    state_path = tmp_path / "s.json"
    state_path.write_text(json.dumps({"open_keys": ["heartbeat_stale"], "episode_started_at": now.isoformat()}), encoding="utf-8")
    monkeypatch.setattr(liveness_mod, "_read_heartbeat_file", lambda settings: _heartbeat(now - pd.Timedelta(seconds=60), expected_by=now + pd.Timedelta(seconds=3600)))
    monkeypatch.setattr("src.live.alerting.dispatch_alert", lambda *a, **k: False)
    assert run_liveness_check(settings, _container(), now=now, state_path=state_path) == 0


def test_episode_state_write_failure_returns_one(tmp_path, monkeypatch) -> None:
    """episode 상태 기록 실패는 exit 1이다."""
    import src.live.liveness as liveness_mod

    settings = _settings(tmp_path)
    now = pd.Timestamp("2026-08-24 01:10Z")
    monkeypatch.setattr(liveness_mod, "_read_heartbeat_file", lambda settings: _heartbeat(now - pd.Timedelta(seconds=60), expected_by=now + pd.Timedelta(seconds=3600)))
    monkeypatch.setattr(liveness_mod, "_write_episode_state", lambda *a, **k: (_ for _ in ()).throw(OSError("ro fs")))
    assert run_liveness_check(settings, _container(), now=now, state_path=tmp_path / "s.json") == 1


def test_any_channel_delivered_handles_missing_and_shapeless_outbox(tmp_path) -> None:
    """아웃박스 부재·형식 오류·키 불일치는 미전송으로 판단한다."""
    import src.live.liveness as liveness_mod

    from src.live.settings import LiveSettings

    settings = LiveSettings(alert_outbox_path=str(tmp_path / "nope.json"))
    assert liveness_mod._any_channel_delivered(settings, "day_skipped", "k") is False
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    settings = LiveSettings(alert_outbox_path=str(bad))
    assert liveness_mod._any_channel_delivered(settings, "day_skipped", "k") is False
    bad.write_text(json.dumps({"records": [{"dedupe_key": "other", "delivered_channels": ["webhook"], "pending_channels": [], "expired": False}]}), encoding="utf-8")
    assert liveness_mod._any_channel_delivered(settings, "day_skipped", "k") is False
    bad.write_text(json.dumps({"records": [{"dedupe_key": "k", "delivered_channels": [], "pending_channels": [], "expired": False}]}), encoding="utf-8")
    assert liveness_mod._any_channel_delivered(settings, "day_skipped", "k") is True
    bad.write_text(json.dumps({"records": "shapeless"}), encoding="utf-8")
    assert liveness_mod._any_channel_delivered(settings, "day_skipped", "k") is False
