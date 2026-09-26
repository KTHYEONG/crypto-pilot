"""불변 테스트: durable alert outbox (enqueue/drain/dedupe/backoff/expiry/capacity)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.live.alert_outbox import AlertOutbox, default_dedupe_key, resolve_outbox_path
from src.live.alerting import dispatch_alert, drain_alerts
from src.live.settings import LiveSettings


def _settings(tmp_path: Path, **overrides) -> LiveSettings:
    base: dict[str, object] = {"alert_outbox_path": str(tmp_path / "outbox.json")}
    base.update(overrides)
    return LiveSettings(**base)  # type: ignore[arg-type]


def _box(tmp_path: Path, **overrides) -> AlertOutbox:
    settings = _settings(tmp_path, **overrides)
    return AlertOutbox.from_settings(resolve_outbox_path(settings), settings)


def _channels(webhook: bool = True, email: bool = True) -> frozenset:
    channels = set()
    if webhook:
        channels.add("webhook")
    if email:
        channels.add("email")
    return frozenset(channels)


def test_failed_first_send_is_retried_later(tmp_path) -> None:
    """첫 전송 실패분은 백오프 후 drain으로 재전송되어 완료된다."""
    box = _box(tmp_path)
    now = pd.Timestamp("2026-08-24 01:10Z")
    assert box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k1", channels=_channels(), now=now) is True
    calls: list[tuple[str, str]] = []
    state = {"webhook_ok": False}

    def deliver(record, channel):
        calls.append((record.dedupe_key, channel))
        if channel == "webhook":
            return state["webhook_ok"]
        return True

    report = box.drain(deliver, now=now, blocking=True)
    assert report.attempted == 2
    state["webhook_ok"] = True
    later = now + pd.Timedelta(seconds=3600)
    report = box.drain(deliver, now=later, blocking=True)
    assert report.pending == 0
    assert ("k1", "webhook") in calls
    assert sum(1 for _, channel in calls if channel == "email") == 1


def test_per_channel_retry_never_resends_delivered_channel(tmp_path) -> None:
    """성공한 채널은 재전송하지 않고 실패 채널만 재시도한다."""
    box = _box(tmp_path)
    now = pd.Timestamp("2026-08-24 01:10Z")
    box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k1", channels=_channels(), now=now)
    counts = {"webhook": 0, "email": 0}

    def deliver(record, channel):
        counts[channel] += 1
        return channel == "email"

    box.drain(deliver, now=now, blocking=True)
    box.drain(deliver, now=now + pd.Timedelta(seconds=3600), blocking=True)
    box.drain(deliver, now=now + pd.Timedelta(seconds=7200), blocking=True)
    assert counts["email"] == 1
    assert counts["webhook"] >= 2


def test_dedupe_across_process_restarts(tmp_path) -> None:
    """완료된 키를 새 인스턴스가 다시 enqueue해도 중복 전송이 없다."""
    now = pd.Timestamp("2026-08-24 01:10Z")
    box = _box(tmp_path)
    box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k1", channels=_channels(), now=now)
    box.drain(lambda record, channel: True, now=now, blocking=True)
    reopened = _box(tmp_path)
    assert reopened.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k1", channels=_channels(), now=now) is True
    calls: list[str] = []
    reopened.drain(lambda record, channel: calls.append(channel) or True, now=now + pd.Timedelta(seconds=3600), blocking=True)
    assert calls == []


def test_backoff_doubles_and_is_capped(tmp_path) -> None:
    """재시도 간격은 2배씩 증가하고 상한에서 멈춘다."""
    box = _box(tmp_path, alert_retry_backoff_s=60.0, alert_retry_backoff_max_s=200.0)
    now = pd.Timestamp("2026-08-24 01:10Z")
    box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k1", channels=frozenset({"webhook"}), now=now)
    gaps: list[float] = []
    tick = [now]

    def deliver(record, channel):
        gaps.append((record.next_attempt_at - tick[0]).total_seconds())
        return False

    for _ in range(4):
        box.drain(deliver, now=tick[0], blocking=True)
        raw = json.loads((tmp_path / "outbox.json").read_text(encoding="utf-8"))
        nxt = pd.Timestamp(raw["records"][0]["next_attempt_at"])
        tick[0] = nxt
    assert gaps[0] == pytest.approx(0.0)
    raw = json.loads((tmp_path / "outbox.json").read_text(encoding="utf-8"))
    record = raw["records"][0]
    assert record["attempts"] == 4
    assert (pd.Timestamp(record["next_attempt_at"]) - now).total_seconds() == pytest.approx(60 + 120 + 200 + 200, abs=1.0)


def test_expiry_is_loud_never_silent(tmp_path, caplog) -> None:
    """보존 기한 초과분은 만료 표시와 ERROR 로그를 남기고 조용히 사라지지 않는다."""
    import logging

    box = _box(tmp_path, alert_outbox_max_age_s=100.0)
    now = pd.Timestamp("2026-08-24 01:10Z")
    box.enqueue(event="day_skipped", detail="stale-detail", decision_time=None, dedupe_key="k1", channels=_channels(), now=now)
    with caplog.at_level(logging.ERROR, logger="LiveAlertOutbox"):
        report = box.drain(lambda record, channel: False, now=now + pd.Timedelta(seconds=500), blocking=True)
    assert report.expired >= 1
    assert any("day_skipped" in r.getMessage() and "EXPIRED" in r.getMessage() for r in caplog.records)
    raw = json.loads((tmp_path / "outbox.json").read_text(encoding="utf-8"))
    assert raw["records"][0]["expired"] is True


def test_no_configured_channel_expires_with_error_log(tmp_path, caplog) -> None:
    """채널 미설정 시 만료 기록과 ERROR 로그를 남기고 재시도하지 않는다."""
    import logging

    settings = _settings(tmp_path)
    now = pd.Timestamp("2026-08-24 01:10Z")
    with caplog.at_level(logging.ERROR, logger="LiveAlertOutbox"):
        assert dispatch_alert(settings, event="day_skipped", detail="d", decision_time=None, dedupe_key="k1", now=now) is True
    assert any("NO_CHANNEL" in r.getMessage() for r in caplog.records)
    raw = json.loads((tmp_path / "outbox.json").read_text(encoding="utf-8"))
    assert raw["records"][0]["expired"] is True
    report = drain_alerts(settings, now=now + pd.Timedelta(seconds=3600), blocking=True)
    assert report.pending == 0


def test_capacity_keeps_critical_alerts(tmp_path) -> None:
    """가득 찬 아웃박스는 NOTICE를 거부하고 CRITICAL을 받아 overflow를 함께 적재한다."""
    box = _box(tmp_path, alert_outbox_max_records=3)
    now = pd.Timestamp("2026-08-24 01:10Z")
    for i in range(3):
        assert box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key=f"k{i}", channels=_channels(), now=now) is True
    assert box.enqueue(event="cycle_complete", detail="n", decision_time=None, dedupe_key="notice1", channels=_channels(), now=now) is False
    assert box.enqueue(event="halt_streak", detail="c", decision_time=None, dedupe_key="crit1", channels=_channels(), now=now) is True
    raw = json.loads((tmp_path / "outbox.json").read_text(encoding="utf-8"))
    events = [r["event"] for r in raw["records"]]
    assert "halt_streak" in events
    assert "alert_outbox_overflow" in events


def test_corrupt_outbox_quarantined(tmp_path) -> None:
    """깨진 아웃박스 파일은 격리 후 빈 큐로 교체되고 enqueue가 성공한다."""
    path = tmp_path / "outbox.json"
    path.write_text("{not json", encoding="utf-8")
    box = _box(tmp_path)
    now = pd.Timestamp("2026-08-24 01:10Z")
    assert box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k1", channels=_channels(), now=now) is True
    leftovers = list(tmp_path.glob("outbox.json.corrupt-*"))
    assert len(leftovers) == 1
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert [r["dedupe_key"] for r in raw["records"]] == ["k1"]


def test_concurrent_writers_merge_delivered_channels(tmp_path) -> None:
    """두 프로세스(데몬·체커)가 각자 전송한 채널은 합집합으로 병합된다."""
    now = pd.Timestamp("2026-08-24 01:10Z")
    first = _box(tmp_path)
    first.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k1", channels=_channels(), now=now)
    second = _box(tmp_path)
    first.drain(lambda record, channel: channel == "webhook", now=now, blocking=True)
    second.drain(lambda record, channel: channel == "email", now=now + pd.Timedelta(seconds=3600), blocking=True)
    raw = json.loads((tmp_path / "outbox.json").read_text(encoding="utf-8"))
    record = raw["records"][0]
    assert sorted(record["delivered_channels"]) == ["email", "webhook"]
    assert record["completed_at"] is not None


def test_secrets_never_persisted(tmp_path, monkeypatch) -> None:
    """아웃박스 파일에 웹훅 URL·앱 비밀번호가 남지 않는다."""
    webhook = "https://hooks.example.com/secret-token-abc123"
    password = "app-password-xyz789"  # noqa: S105
    settings = _settings(tmp_path, alert_webhook_url=webhook, alert_gmail_user="bot@gmail.com", alert_gmail_app_password=password)  # noqa: S106
    monkeypatch.setattr("src.live.alerting.post_alert", lambda *a, **k: False)
    monkeypatch.setattr("src.live.alerting.send_email_alert", lambda **k: False)
    dispatch_alert(settings, event="day_skipped", detail="d", decision_time=None, dedupe_key="k1", now=pd.Timestamp("2026-08-24 01:10Z"))
    text = (tmp_path / "outbox.json").read_text(encoding="utf-8")
    assert webhook not in text
    assert password not in text


def test_default_dedupe_key_format() -> None:
    assert default_dedupe_key("day_skipped", None) == "day_skipped:none"
    assert default_dedupe_key("day_skipped", pd.Timestamp("2026-08-24 00:00Z")).startswith("day_skipped:2026-08-24")


def test_dispatch_then_drain_end_to_end(tmp_path, monkeypatch) -> None:
    """첫 전송 실패 후 채널이 복구되면 drain으로 정확히 1회 전송된다."""
    settings = _settings(tmp_path, alert_webhook_url="https://h.example", alert_gmail_user="bot@gmail.com", alert_gmail_app_password="pw")  # noqa: S106
    now = pd.Timestamp("2026-08-24 01:10Z")
    state = {"webhook_ok": False}
    webhook_calls: list[str] = []
    monkeypatch.setattr(
        "src.live.alerting.post_alert",
        lambda url, *, event, detail, decision_time, now: webhook_calls.append(event) or state["webhook_ok"],
    )
    monkeypatch.setattr("src.live.alerting.send_email_alert", lambda **k: True)
    assert dispatch_alert(settings, event="day_skipped", detail="d", decision_time=None, dedupe_key="k1", now=now) is True
    assert webhook_calls == ["day_skipped"]
    state["webhook_ok"] = True
    report = drain_alerts(settings, now=now + pd.Timedelta(seconds=3600), blocking=True)
    assert report.pending == 0
    assert webhook_calls == ["day_skipped", "day_skipped"]


def test_dedupe_key_falls_back_for_unparseable_decision_time() -> None:
    """파싱 불가 decision_time은 원문 그대로 키에 사용한다."""
    assert default_dedupe_key("day_skipped", "not-a-time") == "day_skipped:not-a-time"  # type: ignore[arg-type]


def test_resolve_outbox_path_defaults_to_data_dir(monkeypatch) -> None:
    """alert_outbox_path 미설정 시 DATA_DIR 기본값을 사용한다."""
    from src.common.paths import DATA_DIR
    from src.live.settings import LiveSettings

    monkeypatch.delenv("LIVE_ALERT_OUTBOX_PATH", raising=False)
    assert resolve_outbox_path(LiveSettings()) == DATA_DIR / "state" / "alert_outbox.json"


def test_enqueue_rejects_unparseable_now(tmp_path, caplog) -> None:
    """파싱 불가 now는 ERROR 로그와 함께 거부된다."""
    import logging

    box = _box(tmp_path)
    with caplog.at_level(logging.ERROR, logger="LiveAlertOutbox"):
        assert box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k", channels=_channels(), now="garbage") is False  # type: ignore[arg-type]
    assert any("ENQUEUE_REJECTED" in r.getMessage() for r in caplog.records)


def test_enqueue_with_naive_timestamps(tmp_path) -> None:
    """naive 타임스탬프는 UTC로 간주되어 적재된다."""
    box = _box(tmp_path)
    now = pd.Timestamp("2026-08-24 01:10:00")
    assert box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k", channels=_channels(), now=now) is True
    raw = json.loads((tmp_path / "outbox.json").read_text(encoding="utf-8"))
    assert raw["records"][0]["created_at"].endswith("+00:00")


def test_load_skips_malformed_records_and_schema(tmp_path) -> None:
    """스키마 불일치·비-dict 항목·불완전 레코드는 격리·건너뛰기로 처리된다."""
    path = tmp_path / "outbox.json"
    now = pd.Timestamp("2026-08-24 01:10Z")
    path.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
    box = _box(tmp_path)
    assert box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k1", channels=_channels(), now=now) is True
    assert list(tmp_path.glob("outbox.json.corrupt-*"))
    path.write_text(
        json.dumps({"records": [
            "not-a-dict",
            {"dedupe_key": "bad", "event": "x"},
            {"dedupe_key": "naive", "event": "day_skipped", "severity": "CRITICAL", "detail": "d",
             "decision_time": "2026-08-24 01:10:00", "created_at": "2026-08-24 01:10:00",
             "pending_channels": ["webhook"], "delivered_channels": [],
             "attempts": 0, "next_attempt_at": "garbage",
             "completed_at": None, "expired": False},
            {"dedupe_key": "badts", "event": "day_skipped", "severity": "CRITICAL", "detail": "d",
             "decision_time": "garbage", "created_at": "2026-08-24T01:10:00+00:00",
             "pending_channels": ["webhook"], "delivered_channels": [],
             "attempts": 0, "next_attempt_at": "2026-08-24T01:10:00+00:00",
             "completed_at": None, "expired": False},
        ]}),
        encoding="utf-8",
    )
    box.drain(lambda record, channel: True, now=now + pd.Timedelta(seconds=3600), blocking=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert [r["dedupe_key"] for r in raw["records"]] == ["badts"]


def test_read_failure_returns_empty_and_write_failure_returns_false(tmp_path, monkeypatch) -> None:
    """읽기 OSError는 빈 큐로, 쓰기 실패는 False로 흡수된다."""
    from pathlib import Path as _Path

    box = _box(tmp_path)
    real_read = _Path.read_text

    def _boom(self, *a, **k):
        if self.name == "outbox.json":
            raise OSError("read down")
        return real_read(self, *a, **k)

    monkeypatch.setattr(_Path, "read_text", _boom)
    assert box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k", channels=_channels(), now=pd.Timestamp("2026-08-24 01:10Z")) is True
    monkeypatch.undo()
    blocker = tmp_path / "blocker"
    blocker.write_text("not-a-dir", encoding="utf-8")
    bad_box = AlertOutbox(
        blocker / "outbox.json", retry_backoff_s=60.0, retry_backoff_max_s=1800.0,
        max_age_s=259200.0, retention_s=604800.0, max_records=1000,
    )
    assert bad_box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k", channels=_channels(), now=pd.Timestamp("2026-08-24 01:10Z")) is False


def test_quarantine_failure_never_raises(tmp_path, monkeypatch) -> None:
    """격리 중 예외도 호출자에게 전파되지 않는다."""
    import os

    path = tmp_path / "outbox.json"
    path.write_text("{bad", encoding="utf-8")
    real_replace = os.replace

    def _boom(src, dst, *a, **k):
        if "corrupt" in str(dst):
            raise OSError("ro fs")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", _boom)
    box = _box(tmp_path)
    assert box.enqueue(event="x", detail="d", decision_time=None, dedupe_key="k", channels=_channels(), now=pd.Timestamp("2026-08-24 01:10Z")) is True


def test_dir_fsync_failure_still_persists(tmp_path, monkeypatch) -> None:
    """디렉터리 fsync 실패에도 기록은 유지된다."""
    import os

    box = _box(tmp_path)
    monkeypatch.setattr(os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("no dir fsync")))
    assert box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k", channels=_channels(), now=pd.Timestamp("2026-08-24 01:10Z")) is True
    assert (tmp_path / "outbox.json").exists()


def test_severity_fallback_when_registry_raises(tmp_path, monkeypatch) -> None:
    """severity 조회 실패 시 CRITICAL로 적재된다."""
    box = _box(tmp_path)

    def _boom(event: str) -> str:
        raise RuntimeError("registry down")

    monkeypatch.setattr("src.live.alerting.event_severity", _boom)
    assert box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k", channels=_channels(), now=pd.Timestamp("2026-08-24 01:10Z")) is True
    raw = json.loads((tmp_path / "outbox.json").read_text(encoding="utf-8"))
    assert raw["records"][0]["severity"] == "CRITICAL"


def test_capacity_evicts_oldest_completed_first(tmp_path) -> None:
    """용량 초과 시 가장 오래된 완료 기록부터 비운다."""
    box = _box(tmp_path, alert_outbox_max_records=2)
    base = pd.Timestamp("2026-08-24 01:10Z")
    box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="old", channels=frozenset({"webhook"}), now=base)
    box.drain(lambda record, channel: True, now=base, blocking=True)
    box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="p1", channels=_channels(), now=base)
    assert box.enqueue(event="halt_streak", detail="c", decision_time=None, dedupe_key="crit", channels=_channels(), now=base) is True
    raw = json.loads((tmp_path / "outbox.json").read_text(encoding="utf-8"))
    keys = [r["dedupe_key"] for r in raw["records"]]
    assert "old" not in keys
    assert "crit" in keys


def test_enqueue_lock_failure_returns_false(tmp_path, monkeypatch) -> None:
    """락 획득 실패는 False로 보고된다."""
    import fcntl

    box = _box(tmp_path)
    monkeypatch.setattr(fcntl, "flock", lambda *a, **k: (_ for _ in ()).throw(OSError("lock down")))
    assert box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k", channels=_channels(), now=pd.Timestamp("2026-08-24 01:10Z")) is False


def test_drain_rejects_bad_now_and_handles_busy_lock(tmp_path) -> None:
    """파싱 불가 now는 빈 리포트를, 사용 중 락은 비차단 호출에 빈 리포트를 반환한다."""
    import fcntl

    box = _box(tmp_path)
    report = box.drain(lambda record, channel: True, now="garbage", blocking=True)  # type: ignore[arg-type]
    assert (report.attempted, report.pending) == (0, 0)
    now = pd.Timestamp("2026-08-24 01:10Z")
    box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k", channels=_channels(), now=now)
    lock_path = tmp_path / "outbox.json.lock"
    with open(lock_path, "a+", encoding="utf-8") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        report = box.drain(lambda record, channel: True, now=now, blocking=False)
        assert (report.attempted, report.pending) == (0, 0)
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)


def test_drain_load_and_write_failures_are_contained(tmp_path, monkeypatch) -> None:
    """로드·병합 쓰기 실패는 리포트로 흡수된다."""
    from src.live.alert_outbox import AlertOutbox

    box = _box(tmp_path)
    now = pd.Timestamp("2026-08-24 01:10Z")
    box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k", channels=_channels(), now=now)
    monkeypatch.setattr(AlertOutbox, "_load_locked", lambda self: (_ for _ in ()).throw(OSError("read down")))
    report = box.drain(lambda record, channel: True, now=now, blocking=True)
    assert (report.attempted, report.pending) == (0, 0)
    monkeypatch.undo()
    monkeypatch.setattr(AlertOutbox, "_write_locked", lambda self, records: (_ for _ in ()).throw(OSError("write down")))
    report = box.drain(lambda record, channel: True, now=now + pd.Timedelta(seconds=3600), blocking=True)
    assert report.attempted == 2


def test_drain_deliver_exception_counts_as_failure(tmp_path) -> None:
    """deliver 예외는 실패로 처리되고 재시도된다."""
    box = _box(tmp_path)
    now = pd.Timestamp("2026-08-24 01:10Z")
    box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k", channels=frozenset({"webhook"}), now=now)

    def _boom(record, channel):
        raise RuntimeError("transport down")

    report = box.drain(_boom, now=now, blocking=True)
    assert report.attempted == 1
    assert report.pending == 1


def test_drain_merge_skips_vanished_record(tmp_path, monkeypatch) -> None:
    """병합 시점에 사라진 레코드는 조용히 건너뛴다."""
    from src.live.alert_outbox import AlertOutbox

    box = _box(tmp_path)
    now = pd.Timestamp("2026-08-24 01:10Z")
    box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k", channels=_channels(), now=now)
    real_load = AlertOutbox._load_locked
    calls = {"n": 0}

    def _vanishing(self):
        calls["n"] += 1
        records = real_load(self)
        if calls["n"] >= 2:
            return []
        return records

    monkeypatch.setattr(AlertOutbox, "_load_locked", _vanishing)
    report = box.drain(lambda record, channel: True, now=now, blocking=True)
    assert report.attempted == 2


def test_finalize_failure_returns_empty_report(tmp_path, monkeypatch) -> None:
    """finalize 경로 예외는 빈 리포트로 흡수된다."""
    from src.live.alert_outbox import AlertOutbox

    box = _box(tmp_path)
    monkeypatch.setattr(AlertOutbox, "_load_locked", lambda self: (_ for _ in ()).throw(OSError("read down")))
    report = box.drain(lambda record, channel: True, now=pd.Timestamp("2026-08-24 01:10Z"), blocking=True)
    assert (report.attempted, report.pending) == (0, 0)


def test_write_failure_when_outbox_path_is_directory(tmp_path) -> None:
    """아웃박스 경로가 디렉터리면 쓰기 실패로 False를 반환한다."""
    path = tmp_path / "outbox.json"
    path.mkdir()
    box = _box(tmp_path)
    assert box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k", channels=_channels(), now=pd.Timestamp("2026-08-24 01:10Z")) is False


def test_record_with_unparseable_attempts_is_skipped(tmp_path) -> None:
    """형식은 맞으나 필드 파싱이 깨진 레코드는 건너뛴다."""
    path = tmp_path / "outbox.json"
    path.write_text(
        json.dumps({"records": [{
            "dedupe_key": "weird", "event": "day_skipped", "severity": "CRITICAL", "detail": "d",
            "decision_time": None, "created_at": "2026-08-24T01:10:00+00:00",
            "pending_channels": ["webhook"], "delivered_channels": [],
            "attempts": "many", "next_attempt_at": "2026-08-24T01:10:00+00:00",
            "completed_at": None, "expired": False,
        }]}),
        encoding="utf-8",
    )
    box = _box(tmp_path)
    calls: list[str] = []
    box.drain(lambda record, channel: calls.append(channel) or True, now=pd.Timestamp("2026-08-24 02:10Z"), blocking=True)
    assert calls == []


def test_finalize_load_failure_returns_empty_report(tmp_path, monkeypatch) -> None:
    """finalize 중 로드 실패는 빈 리포트로 흡수된다."""
    from src.live.alert_outbox import AlertOutbox

    box = _box(tmp_path)
    now = pd.Timestamp("2026-08-24 01:10Z")
    box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k", channels=_channels(), now=now)
    real_load = AlertOutbox._load_locked
    calls = {"n": 0}

    def _fail_on_finalize(self):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise OSError("read down")
        return real_load(self)

    monkeypatch.setattr(AlertOutbox, "_load_locked", _fail_on_finalize)
    report = box.drain(lambda record, channel: False, now=now, blocking=True)
    assert (report.attempted, report.completed, report.expired, report.pending) == (2, 0, 0, 0)
