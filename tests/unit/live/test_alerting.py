

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
        email_to="me@gmail.com",
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
        email_to="me@gmail.com",
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
    assert captured["to"] == "me@gmail.com"
    assert "halt_streak" in captured["subject"]
    assert "consecutive_halts=3" in captured["body"]

def test_send_email_alert_defaults_recipient_to_sender(monkeypatch) -> None:
    import pandas as pd
    import src.live.alerting as a

    captured = {}

    class _FakeSMTP:
        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def starttls(self):
            pass

        def login(self, *_a):
            pass

        def send_message(self, msg):
            captured["to"] = msg["To"]

    monkeypatch.setattr(a.smtplib, "SMTP", _FakeSMTP)

    out = a.send_email_alert(
        gmail_user="bot@gmail.com",
        gmail_app_password="pw",
        email_to=None,
        event="awaiting_params",
        detail="strategy_params missing",
        decision_time=None,
        now=pd.Timestamp("2026-08-31T01:00:00", tz="UTC"),
    )

    assert out is True
    assert captured["to"] == "bot@gmail.com"

def test_send_email_alert_swallows_smtp_errors(monkeypatch) -> None:
    import pandas as pd
    import src.live.alerting as a

    def _raise(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr(a.smtplib, "SMTP", _raise)

    out = a.send_email_alert(
        gmail_user="bot@gmail.com",
        gmail_app_password="pw",
        email_to="me@gmail.com",
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
    monkeypatch.setenv("LIVE_ALERT_EMAIL_TO", "me@gmail.com")

    settings = LiveSettings()

    assert settings.alert_gmail_user == "bot@gmail.com"
    assert isinstance(settings.alert_gmail_app_password, SecretStr)
    assert settings.alert_gmail_app_password.get_secret_value() == "abcd efgh ijkl mnop"
    assert "abcd" not in repr(settings.alert_gmail_app_password)
    assert settings.alert_email_to == "me@gmail.com"
