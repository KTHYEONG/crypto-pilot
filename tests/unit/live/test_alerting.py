

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


