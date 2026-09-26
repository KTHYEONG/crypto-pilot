"""불변 테스트: dead-man's-switch pinger (rate limit, /fail, never-raise)."""

from __future__ import annotations

import pandas as pd
from pydantic import SecretStr

from src.live.deadman import DeadmanPinger


def _pinger(url: str | None = "https://hc.example.com/check-token", interval_s: float = 300.0, transport=None) -> tuple[DeadmanPinger, list[str]]:
    calls: list[str] = []

    def fake(url: str, timeout_s: float) -> int:
        calls.append(url)
        return 200

    secret = SecretStr(url) if url is not None else None
    pinger = DeadmanPinger(url=secret, interval_s=interval_s, timeout_s=10.0, transport=transport or fake)
    return pinger, calls


def test_disabled_without_url_never_calls_transport() -> None:
    """URL 미설정 시 전송을 시도하지 않는다."""
    pinger, calls = _pinger(url=None)
    now = pd.Timestamp("2026-08-24 01:10Z")
    assert pinger.maybe_ping(now=now, failing=False) is False
    assert pinger.maybe_ping(now=now + pd.Timedelta(seconds=600), failing=True) is False
    assert calls == []


def test_pings_are_rate_limited() -> None:
    """간격 내 호출은 건너뛰고 간격 경과 후에만 전송한다."""
    pinger, calls = _pinger(interval_s=300.0)
    base = pd.Timestamp("2026-08-24 01:10Z")
    assert pinger.maybe_ping(now=base, failing=False) is True
    assert pinger.maybe_ping(now=base + pd.Timedelta(seconds=100), failing=False) is False
    assert pinger.maybe_ping(now=base + pd.Timedelta(seconds=299), failing=False) is False
    assert pinger.maybe_ping(now=base + pd.Timedelta(seconds=300), failing=False) is True
    assert len(calls) == 2


def test_failing_state_pings_fail_endpoint_immediately() -> None:
    """healthy→failing 전이는 rate limit과 무관하게 즉시 /fail로 전송한다."""
    pinger, calls = _pinger(interval_s=300.0)
    base = pd.Timestamp("2026-08-24 01:10Z")
    assert pinger.maybe_ping(now=base, failing=False) is True
    assert pinger.maybe_ping(now=base + pd.Timedelta(seconds=10), failing=True) is True
    assert calls[0] == "https://hc.example.com/check-token"
    assert calls[1] == "https://hc.example.com/check-token/fail"


def test_vendor_failure_never_raises_and_never_logs_url(caplog) -> None:
    """벤더 타임아웃은 False 반환·WARNING 로그이며 URL을 기록하지 않는다."""
    import logging

    def _timeout(url: str, timeout_s: float) -> int:
        raise TimeoutError("vendor down")

    pinger, _ = _pinger(transport=_timeout)
    with caplog.at_level(logging.WARNING, logger="LiveDeadman"):
        assert pinger.maybe_ping(now=pd.Timestamp("2026-08-24 01:10Z"), failing=False) is False
    assert any("PING_FAILED" in r.getMessage() for r in caplog.records)
    assert not any("check-token" in r.getMessage() for r in caplog.records)


def test_non_2xx_is_failure_without_url_leak(caplog) -> None:
    """비-2xx 응답은 실패이며 URL을 로그에 남기지 않는다."""
    import logging

    pinger, _ = _pinger(transport=lambda url, timeout: 500)
    with caplog.at_level(logging.WARNING, logger="LiveDeadman"):
        assert pinger.maybe_ping(now=pd.Timestamp("2026-08-24 01:10Z"), failing=False) is False
    assert not any("check-token" in r.getMessage() for r in caplog.records)


def test_daemon_pulse_skips_ping_when_heartbeat_write_fails(tmp_path, monkeypatch) -> None:
    """하트비트 기록 실패 시 핑을 보내지 않는다."""
    import threading
    import time

    import src.live.scheduler as sched

    pings: list[bool] = []

    class _Pinger:
        def maybe_ping(self, *, now, failing):
            pings.append(failing)
            return True

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(sched, "write_heartbeat", _boom)
    stop = threading.Event()
    thread = threading.Thread(
        target=sched._run_heartbeat_pulse,
        kwargs={
            "stop": stop,
            "heartbeat_path": tmp_path / "hb.json",
            "decision_time": pd.Timestamp("2026-08-24 00:00Z"),
            "attempts": 0,
            "consecutive_halts": 0,
            "interval_s": 0.01,
            "pinger": _Pinger(),
        },
        daemon=True,
    )
    thread.start()
    time.sleep(0.05)
    stop.set()
    thread.join(timeout=5)
    assert pings == []


def test_halt_heartbeat_maps_to_fail_endpoint() -> None:
    """HALT 상태의 핑은 /fail 경로로 전송한다."""
    pinger, calls = _pinger(interval_s=0.0)
    now = pd.Timestamp("2026-08-24 01:10Z")
    assert pinger.maybe_ping(now=now, failing=True) is True
    assert calls == ["https://hc.example.com/check-token/fail"]


def test_plain_string_url_supported() -> None:
    """SecretStr가 아닌 평문 URL도 허용한다."""
    calls: list[str] = []
    pinger = DeadmanPinger(url="https://hc.example.com/plain", interval_s=300.0, timeout_s=10.0, transport=lambda url, t: calls.append(url) or 200)
    assert pinger.maybe_ping(now=pd.Timestamp("2026-08-24 01:10Z"), failing=False) is True
    assert calls == ["https://hc.example.com/plain"]












def test_daemon_pulse_drains_and_pings_with_settings(tmp_path) -> None:
    """펄스에 settings·pinger를 주면 drain과 ping이 함께 수행된다."""
    import threading
    import time

    import pandas as pd

    import src.live.scheduler as sched
    from src.live.settings import LiveSettings

    pings: list[bool] = []

    class _Pinger:
        def maybe_ping(self, *, now, failing):
            pings.append(failing)
            return True

    settings = LiveSettings(alert_outbox_path=str(tmp_path / "outbox.json"))
    stop = threading.Event()
    thread = threading.Thread(
        target=sched._run_heartbeat_pulse,
        kwargs={
            "stop": stop,
            "heartbeat_path": tmp_path / "hb.json",
            "decision_time": pd.Timestamp("2026-08-24 00:00Z"),
            "attempts": 0,
            "consecutive_halts": 0,
            "interval_s": 0.02,
            "stage": "refresh",
            "settings": settings,
            "pinger": _Pinger(),
        },
        daemon=True,
    )
    thread.start()
    time.sleep(0.08)
    stop.set()
    thread.join(timeout=5)
    assert pings
    assert all(f is False for f in pings)


def test_daemon_pulse_contained_when_drain_and_ping_raise(tmp_path, monkeypatch) -> None:
    """펄스의 drain·ping 예외는 스레드를 멈추지 않는다."""
    import threading
    import time

    import pandas as pd

    import src.live.scheduler as sched
    from src.live.settings import LiveSettings

    def _boom_drain(*a, **k):
        raise RuntimeError("drain down")

    monkeypatch.setattr(sched, "drain_alerts", _boom_drain)

    class _BoomPinger:
        def maybe_ping(self, *, now, failing):
            raise RuntimeError("deadman down")

    settings = LiveSettings(alert_outbox_path=str(tmp_path / "outbox.json"))
    stop = threading.Event()
    thread = threading.Thread(
        target=sched._run_heartbeat_pulse,
        kwargs={
            "stop": stop,
            "heartbeat_path": tmp_path / "hb.json",
            "decision_time": pd.Timestamp("2026-08-24 00:00Z"),
            "attempts": 0,
            "consecutive_halts": 0,
            "interval_s": 0.02,
            "settings": settings,
            "pinger": _BoomPinger(),
        },
        daemon=True,
    )
    thread.start()
    time.sleep(0.08)
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_default_transport_reports_status_and_getcode_fallback(monkeypatch) -> None:
    """기본 전송은 status·getcode 순으로 확인하고 예외 시 200으로 간주한다."""
    import urllib.request

    import src.live.deadman as deadman_mod

    class _Resp:
        status = 204
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp())
    assert deadman_mod._default_transport("https://hc.example.com/x", 10.0) == 204

    class _Legacy:
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False
        def getcode(self):
            return 200

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Legacy())
    assert deadman_mod._default_transport("https://hc.example.com/x", 10.0) == 200

    class _Broken:
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False
        def getcode(self):
            raise RuntimeError("no code")

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Broken())
    assert deadman_mod._default_transport("https://hc.example.com/x", 10.0) == 200


def test_secret_value_tolerates_broken_secret() -> None:
    """get_secret_value 예외·빈 값은 비활성화로 처리된다."""
    import pandas as pd

    from src.live.deadman import DeadmanPinger

    class _Broken:
        def get_secret_value(self):
            raise RuntimeError("vault down")

    pinger = DeadmanPinger(url=_Broken(), interval_s=300.0, timeout_s=10.0, transport=lambda url, t: 200)
    assert pinger.maybe_ping(now=pd.Timestamp("2026-08-24 01:10Z"), failing=False) is False

    class _Empty:
        def get_secret_value(self):
            return ""

    pinger = DeadmanPinger(url=_Empty(), interval_s=300.0, timeout_s=10.0, transport=lambda url, t: 200)
    assert pinger.maybe_ping(now=pd.Timestamp("2026-08-24 01:10Z"), failing=False) is False


def test_maybe_ping_rejects_unparseable_now_and_clock() -> None:
    """파싱 불가 시각·시계는 False로 흡수된다."""
    import pandas as pd

    from src.live.deadman import DeadmanPinger

    pinger = DeadmanPinger(url="https://hc.example.com/x", interval_s=300.0, timeout_s=10.0, transport=lambda url, t: 200)
    assert pinger.maybe_ping(now="garbage", failing=False) is False  # type: ignore[arg-type]
    pinger.maybe_ping(now=pd.Timestamp("2026-08-24 01:10Z"), failing=False)
    pinger._last_attempt_at = "garbage"  # type: ignore[assignment]
    assert pinger.maybe_ping(now=pd.Timestamp("2026-08-24 01:11Z"), failing=False) is False






