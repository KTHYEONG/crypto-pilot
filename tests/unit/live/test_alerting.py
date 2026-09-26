

# ruff: noqa

# --- auto appended from contract ---
def test_post_alert_noop_without_url(monkeypatch) -> None:
    import pandas as pd
    import src.live.alerting as a

    def _boom(*_a, **_k):
        raise AssertionError("network must not be touched")

    monkeypatch.setattr(a.urllib.request, "urlopen", _boom)

    out = a.post_alert(
        None, event="halt_streak", detail="x",
        decision_time=pd.Timestamp("2026-08-31", tz="UTC"),
        now=pd.Timestamp("2026-08-31T01:00:00", tz="UTC"),
    )

    assert out is False


def test_post_alert_posts_json_payload(monkeypatch) -> None:
    import json
    import pandas as pd
    import src.live.alerting as a

    captured = {}

    class _Resp:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(a.urllib.request, "urlopen", _fake_urlopen)

    out = a.post_alert(
        "https://hook.example/abc", event="halt_streak", detail="consecutive_halts=3",
        decision_time=pd.Timestamp("2026-08-31", tz="UTC"),
        now=pd.Timestamp("2026-08-31T01:00:00", tz="UTC"),
    )

    assert out is True
    assert captured["url"] == "https://hook.example/abc"
    assert captured["body"]["event"] == "halt_streak"
    assert captured["body"]["detail"] == "consecutive_halts=3"
    assert captured["body"]["decision_time"].startswith("2026-08-31")
    assert captured["body"]["source"] == "mhs-live"
    assert captured["timeout"] == a.ALERT_WEBHOOK_TIMEOUT_S


def test_post_alert_swallows_transport_errors(monkeypatch) -> None:
    import pandas as pd
    import src.live.alerting as a

    def _raise(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr(a.urllib.request, "urlopen", _raise)

    out = a.post_alert(
        "https://hook.example/abc", event="awaiting_params", detail="x",
        decision_time=None, now=pd.Timestamp("2026-08-31T01:00:00", tz="UTC"),
    )

    assert out is False



def test_send_email_alert_noop_without_credentials(monkeypatch) -> None:
    import pandas as pd
    import src.live.alerting as a

    def _boom(*_a, **_k):
        raise AssertionError("SMTP must not be constructed")

    monkeypatch.setattr(a.smtplib, "SMTP", _boom)

    out = a.send_email_alert(
        gmail_user="bot@gmail.com",
        gmail_app_password=None,
        event="halt_streak",
        detail="consecutive_halts=3",
        decision_time=pd.Timestamp("2026-08-31", tz="UTC"),
        now=pd.Timestamp("2026-08-31T01:00:00", tz="UTC"),
    )

    assert out is False

def test_send_email_alert_sends_via_gmail_smtp(monkeypatch) -> None:
    import pandas as pd
    import src.live.alerting as a

    captured = {}

    class _FakeSMTP:
        def __init__(self, host, port, timeout=None):
            captured["host"] = host
            captured["port"] = port
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def starttls(self):
            captured["starttls"] = True

        def login(self, user, password):
            captured["login"] = (user, password)

        def send_message(self, msg):
            captured["from"] = msg["From"]
            captured["to"] = msg["To"]
            captured["subject"] = msg["Subject"]
            captured["body"] = msg.get_content()

    monkeypatch.setattr(a.smtplib, "SMTP", _FakeSMTP)

    out = a.send_email_alert(
        gmail_user="bot@gmail.com",
        gmail_app_password="abcd efgh ijkl mnop",
        event="halt_streak",
        detail="consecutive_halts=3",
        decision_time=pd.Timestamp("2026-08-31", tz="UTC"),
        now=pd.Timestamp("2026-08-31T01:00:00", tz="UTC"),
    )

    assert out is True
    assert captured["host"] == a.SMTP_HOST == "smtp.gmail.com"
    assert captured["port"] == a.SMTP_PORT == 587
    assert captured["timeout"] == a.ALERT_EMAIL_TIMEOUT_S
    assert captured["starttls"] is True
    assert captured["login"] == ("bot@gmail.com", "abcd efgh ijkl mnop")
    assert captured["from"] == "bot@gmail.com"
    assert captured["to"] == "bot@gmail.com"
    assert "halt_streak" in captured["subject"]
    assert "consecutive_halts=3" in captured["body"]

def test_send_email_alert_swallows_smtp_errors(monkeypatch) -> None:
    import pandas as pd
    import src.live.alerting as a

    def _raise(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr(a.smtplib, "SMTP", _raise)

    out = a.send_email_alert(
        gmail_user="bot@gmail.com",
        gmail_app_password="pw",
        event="data_refresh_failed",
        detail="boom",
        decision_time=None,
        now=pd.Timestamp("2026-08-31T01:00:00", tz="UTC"),
    )

    assert out is False

def test_live_settings_accepts_gmail_alert_env(monkeypatch) -> None:
    from pydantic import SecretStr
    from src.live.settings import LiveSettings

    monkeypatch.setenv("LIVE_ALERT_GMAIL_USER", "bot@gmail.com")
    monkeypatch.setenv("LIVE_ALERT_GMAIL_APP_PASSWORD", "abcd efgh ijkl mnop")

    settings = LiveSettings()

    assert settings.alert_gmail_user == "bot@gmail.com"
    assert isinstance(settings.alert_gmail_app_password, SecretStr)
    assert settings.alert_gmail_app_password.get_secret_value() == "abcd efgh ijkl mnop"
    assert "abcd" not in repr(settings.alert_gmail_app_password)


def test_send_email_alert_orderbook_backup_impending(monkeypatch) -> None:
    import pandas as pd
    import src.live.alerting as a

    captured = {}

    class _FakeSMTP:
        def __init__(self, host, port, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def starttls(self):
            pass

        def login(self, user, password):
            pass

        def send_message(self, msg):
            captured["subject"] = msg["Subject"]
            captured["body"] = msg.get_content()

    monkeypatch.setattr(a.smtplib, "SMTP", _FakeSMTP)

    out = a.send_email_alert(
        gmail_user="bot@gmail.com",
        gmail_app_password="pw",
        event="orderbook_backup_impending",
        detail="earliest_date=2025-09-05 days_left=4",
        decision_time=None,
        now=pd.Timestamp("2026-09-01T01:00:00", tz="UTC"),
    )

    assert out is True
    assert "백업 권장" in captured["subject"]
    assert "orderbook_backup_impending" in captured["subject"]
    assert "rsync" in captured["body"]
    assert "earliest_date=2025-09-05" in captured["body"]


def test_send_email_alert_day_skipped_uses_event_info(monkeypatch) -> None:
    import pandas as pd
    import src.live.alerting as a

    captured = {}

    class _FakeSMTP:
        def __init__(self, host, port, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def starttls(self):
            pass

        def login(self, user, password):
            pass

        def send_message(self, msg):
            captured["subject"] = msg["Subject"]
            captured["body"] = msg.get_content()

    monkeypatch.setattr(a.smtplib, "SMTP", _FakeSMTP)

    out = a.send_email_alert(
        gmail_user="bot@gmail.com",
        gmail_app_password="pw",
        event="day_skipped",
        detail="attempts=5 cause=signal_step exit=1",
        decision_time=pd.Timestamp("2026-09-14", tz="UTC"),
        now=pd.Timestamp("2026-09-14T02:05:00", tz="UTC"),
    )

    assert out is True
    assert a.EVENT_INFO["day_skipped"]["severity_label"] == "CRITICAL"
    assert a.EVENT_INFO["day_skipped"]["title"] in captured["subject"]
    assert "docker logs --tail 200 mhs-live-daemon" in captured["body"]
    assert "attempts=5 cause=signal_step exit=1" in captured["body"]


def test_event_info_actions_reference_runnable_operator_commands() -> None:
    from src.cli.main import build_root_parser
    from src.live.alerting import EVENT_INFO

    for event, info in EVENT_INFO.items():
        assert "logs/live_daemon.log" not in info["action"], event
        assert "refresh-live-market-data" not in info["action"], event
    for event in ("halt_streak", "day_skipped", "data_refresh_failed"):
        assert "docker logs --tail 200 mhs-live-daemon" in EVENT_INFO[event]["action"]
    assert "src.cli.main live status" in EVENT_INFO["halt_streak"]["action"]
    assert "src.cli.main data refresh-live-universe" in EVENT_INFO["data_refresh_failed"]["action"]

    parser = build_root_parser()
    assert parser.parse_args(["live", "status"]).handler is not None
    assert parser.parse_args(["data", "refresh-live-universe"]).handler is not None


def test_event_info_state_corrupt_is_critical_with_docker_action() -> None:
    from src.live.alerting import EVENT_INFO

    info = EVENT_INFO["state_corrupt"]

    assert info["severity_label"] == "CRITICAL"
    assert "docker logs --tail 200 mhs-live-daemon" in info["action"]
    assert "live_daemon_last_run.json" in info["action"]


# --- auto appended from contract: signal_input_quarantine ---


def test_event_info_data_quarantine_is_warning_with_sidecar_and_repair_action() -> None:
    from src.cli.main import build_root_parser
    from src.live.alerting import EVENT_INFO

    info = EVENT_INFO["data_quarantine"]

    assert info["severity_label"] == "WARNING"
    assert "signal_quarantine.json" in info["action"]
    assert "src.cli.main data repair-ohlcv --symbol" in info["action"]
    args = build_root_parser().parse_args(["data", "repair-ohlcv", "--symbol", "BTCUSDT"])
    assert args.handler is not None


def test_event_info_paper_ledger_events() -> None:
    from src.live.alerting import EVENT_INFO

    lag = EVENT_INFO["paper_funding_lag"]
    close = EVENT_INFO["paper_delisted_unresolved"]
    assert lag["severity_label"] == "WARNING"
    assert close["severity_label"] == "CRITICAL"
    for info in (lag, close):
        assert "docker" in info["action"]
        assert set(info) == {"title", "severity_badge", "severity_label", "header_color", "bg_color", "impact", "action"}


# --- auto appended from contract: live_alert_gaps ---
def test_alert_gap_events_registered_and_digest_default_on() -> None:
    from src.live.alerting import EVENT_INFO
    from src.live.settings import LiveSettings

    assert EVENT_INFO["cycle_interrupted"]["severity_label"] == "WARNING"
    assert EVENT_INFO["daemon_crashed"]["severity_label"] == "CRITICAL"
    assert EVENT_INFO["cycle_complete"]["severity_label"] == "NOTICE"
    for event in ("cycle_interrupted", "daemon_crashed", "cycle_complete", "data_quarantine"):
        for key in ("title", "severity_badge", "severity_label", "header_color", "bg_color", "impact", "action"):
            assert EVENT_INFO[event][key]
    assert LiveSettings().alert_daily_digest is True
    assert LiveSettings(alert_daily_digest=False).alert_daily_digest is False




def test_recorder_events_registered_with_correct_severity() -> None:
    from src.live.alerting import EVENT_INFO

    unhealthy = EVENT_INFO["recorder_unhealthy"]
    assert unhealthy["severity_label"] == "CRITICAL"
    assert "market-recorder" not in unhealthy["action"]
    assert "market-normalizer" in unhealthy["action"]
    assert "market-capture-blue" in unhealthy["action"]
    assert "logs/capture" in unhealthy["action"]
    recovered = EVENT_INFO["recorder_recovered"]
    assert recovered["severity_label"] == "NOTICE"


def test_reconcile_mismatch_is_critical_on_both_channels(monkeypatch) -> None:
    """ledger_reconcile_mismatch는 웹훅 CRITICAL·이메일 CRITICAL 배지로 전송된다."""
    import json
    import pandas as pd
    import src.live.alerting as a

    payloads: dict = {}

    class _Resp:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False

    def _fake_urlopen(req, timeout=None):
        payloads.update(json.loads(req.data.decode("utf-8")))
        return _Resp()

    monkeypatch.setattr(a.urllib.request, "urlopen", _fake_urlopen)
    captured: dict = {}

    class _SMTP:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False
        def starttls(self):
            pass
        def login(self, user, password):
            captured["user"] = user
        def send_message(self, msg):
            captured["subject"] = str(msg["Subject"])

    monkeypatch.setattr(a.smtplib, "SMTP", _SMTP)
    now = pd.Timestamp("2026-08-24T01:00:00", tz="UTC")
    assert a.post_alert("https://hook.example/abc", event="ledger_reconcile_mismatch", detail="difference=1", decision_time=None, now=now) is True
    assert payloads["severity"] == "CRITICAL"
    assert a.send_email_alert(gmail_user="bot@gmail.com", gmail_app_password="pw", event="ledger_reconcile_mismatch", detail="difference=1", decision_time=None, now=now) is True
    assert "🚨 긴급" in captured["subject"]


def test_unregistered_event_renders_critical_not_info(monkeypatch) -> None:
    """미등록 이벤트는 INFO가 아닌 CRITICAL으로 렌더링된다."""
    import pandas as pd
    import src.live.alerting as a

    assert a.event_severity("made_up_event") == "CRITICAL"
    captured: dict = {}

    class _SMTP:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False
        def starttls(self):
            pass
        def login(self, user, password):
            pass
        def send_message(self, msg):
            captured["subject"] = str(msg["Subject"])

    monkeypatch.setattr(a.smtplib, "SMTP", _SMTP)
    assert a.send_email_alert(gmail_user="bot@gmail.com", gmail_app_password="pw", event="made_up_event", detail="x", decision_time=None, now=pd.Timestamp("2026-08-24T01:00:00", tz="UTC")) is True
    assert "🚨 긴급" in captured["subject"]
    assert "🔔 알림" not in captured["subject"]


def test_event_registry_covers_all_dispatched_events() -> None:
    """src/에서 dispatch/_daemon_alert/_notify_event로 전송되는 모든 이벤트는 EVENT_INFO에 등록된다."""
    import re
    from pathlib import Path

    import src.live.alerting as a

    root = Path("src")
    pattern = re.compile(r"""(?:dispatch_alert|_daemon_alert|_notify_event)\s*\([^)]*?event\s*=\s*["']([^"']+)["']""", re.DOTALL)
    found: set[str] = set()
    for path in root.rglob("*.py"):
        found |= set(pattern.findall(path.read_text(encoding="utf-8")))
    literal_pattern = re.compile(r"""event\s*=\s*["']([a-z_]+)["']""")
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "dispatch_alert" in text or "_notify_event" in text:
            found |= set(literal_pattern.findall(text))
    missing = {event for event in found if event not in a.EVENT_INFO and event != "event"}
    assert missing == set(), f"unregistered alert events: {sorted(missing)}"


def test_dispatch_returns_false_when_outbox_rejects_notice(tmp_path, monkeypatch) -> None:
    """용량 초과로 NOTICE가 거부되면 dispatch는 False를 반환한다."""
    import pandas as pd
    from src.live.alert_outbox import AlertOutbox, resolve_outbox_path
    from src.live.settings import LiveSettings
    import src.live.alerting as a

    settings = LiveSettings(alert_outbox_path=str(tmp_path / "outbox.json"), alert_outbox_max_records=1, alert_webhook_url="https://h.example")
    box = AlertOutbox.from_settings(resolve_outbox_path(settings), settings)
    now = pd.Timestamp("2026-08-24 01:10Z")
    assert box.enqueue(event="day_skipped", detail="d", decision_time=None, dedupe_key="k0", channels=frozenset({"webhook"}), now=now) is True
    monkeypatch.setattr(a, "event_severity", lambda event: "NOTICE")
    assert a.dispatch_alert(settings, event="cycle_complete", detail="n", decision_time=None, dedupe_key="notice1", now=now) is False


def test_dispatch_survives_immediate_drain_failure(tmp_path, monkeypatch) -> None:
    """즉시 전송 시 drain 예외는 삼키고 durable hand-off(True)를 반환한다."""
    import pandas as pd
    from src.live.settings import LiveSettings
    import src.live.alerting as a

    settings = LiveSettings(alert_outbox_path=str(tmp_path / "outbox.json"), alert_webhook_url="https://h.example")
    monkeypatch.setattr("src.live.alert_outbox.AlertOutbox.drain", lambda self, deliver, *, now, blocking: (_ for _ in ()).throw(RuntimeError("drain down")))
    assert a.dispatch_alert(settings, event="day_skipped", detail="d", decision_time=None, dedupe_key="k1", now=pd.Timestamp("2026-08-24 01:10Z")) is True


def test_dispatch_never_raises_on_enqueue_failure(monkeypatch) -> None:
    """enqueue 경로 예외도 False로 흡수된다."""
    import pandas as pd
    from src.live.settings import LiveSettings
    import src.live.alerting as a

    settings = LiveSettings(alert_webhook_url="https://h.example")
    monkeypatch.setattr("src.live.alert_outbox.AlertOutbox.enqueue", lambda self, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert a.dispatch_alert(settings, event="day_skipped", detail="d", decision_time=None, dedupe_key="k1", now=pd.Timestamp("2026-08-24 01:10Z")) is False


def test_drain_alerts_never_raises_on_outbox_failure(monkeypatch) -> None:
    """drain 경로 예외는 빈 리포트로 흡수된다."""
    import pandas as pd
    from src.live.settings import LiveSettings
    import src.live.alerting as a

    settings = LiveSettings(alert_webhook_url="https://h.example")

    def _boom(*a, **k):
        raise OSError("lock down")

    monkeypatch.setattr("src.live.alert_outbox.resolve_outbox_path", _boom)
    report = a.drain_alerts(settings, now=pd.Timestamp("2026-08-24 01:10Z"), blocking=True)
    assert (report.attempted, report.completed, report.expired, report.pending) == (0, 0, 0, 0)


def test_deliver_record_failure_isolated_per_channel(tmp_path, monkeypatch) -> None:
    """채널 전송 예외는 False로 격리되고 로그 후 계속된다."""
    import pandas as pd
    from src.live.alert_outbox import AlertRecord
    from src.live.settings import LiveSettings
    import src.live.alerting as a

    settings = LiveSettings(alert_outbox_path=str(tmp_path / "outbox.json"), alert_webhook_url="https://h.example")
    record = AlertRecord(
        dedupe_key="k", event="day_skipped", severity="CRITICAL", detail="d", decision_time=None,
        created_at=pd.Timestamp("2026-08-24 01:10Z"), pending_channels=frozenset({"webhook"}),
        delivered_channels=frozenset(), attempts=0, next_attempt_at=pd.Timestamp("2026-08-24 01:10Z"),
        completed_at=None, expired=False,
    )

    def _boom(*a, **k):
        raise RuntimeError("transport down")

    monkeypatch.setattr(a, "post_alert", _boom)
    assert a._deliver_record(settings, record, "webhook", pd.Timestamp("2026-08-24 01:10Z")) is False
    assert a._deliver_record(settings, record, "bogus", pd.Timestamp("2026-08-24 01:10Z")) is False
