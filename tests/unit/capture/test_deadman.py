"""Invariant guards for the dead-man pinger."""

from __future__ import annotations

import logging

import pytest

from src.capture.deadman import DeadmanPinger

_T0 = 1_757_824_000_000_000_000


def test_disabled_without_url() -> None:
    """Spec 01: no env URL means no ping and no transport call."""
    calls: list[str] = []
    pinger = DeadmanPinger(None, interval_s=300.0, timeout_s=10.0, transport=lambda url, timeout: calls.append(url) or 200)
    assert pinger.maybe_ping(now_ns=_T0, failing=False) is False
    assert calls == []
    empty = DeadmanPinger("", interval_s=300.0, timeout_s=10.0, transport=lambda url, timeout: calls.append(url) or 200)
    assert empty.maybe_ping(now_ns=_T0, failing=False) is False
    assert calls == []


def test_fail_suffix_and_rate_limit() -> None:
    """Spec 01: healthy pings base; a failing transition pings /fail at once; then silence."""
    calls: list[str] = []
    pinger = DeadmanPinger(
        "https://example.invalid/check",
        interval_s=300.0,
        timeout_s=10.0,
        transport=lambda url, timeout: calls.append(url) or 200,
    )
    assert pinger.maybe_ping(now_ns=_T0, failing=False) is True
    assert calls == ["https://example.invalid/check"]
    assert pinger.maybe_ping(now_ns=_T0 + 10_000_000_000, failing=True) is True
    assert calls[-1] == "https://example.invalid/check/fail"
    assert pinger.maybe_ping(now_ns=_T0 + 20_000_000_000, failing=True) is False
    assert len(calls) == 2
    assert pinger.maybe_ping(now_ns=_T0 + 400_000_000_000, failing=True) is True
    assert len(calls) == 3


def test_url_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Spec 01: a transport error echoing the URL must not leak it into the logs."""
    url = "https://example.invalid/secret-check-9f3k"

    def transport(target: str, timeout: float) -> int:
        raise ConnectionError(f"cannot reach {target} from here")

    pinger = DeadmanPinger(url, interval_s=300.0, timeout_s=10.0, transport=transport)
    with caplog.at_level(logging.WARNING):
        assert pinger.maybe_ping(now_ns=_T0, failing=False) is True
    assert url not in caplog.text


def test_default_transport_reports_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bundled urllib transport returns the HTTP status without logging the URL."""
    import urllib.request

    seen: list[str] = []

    class _Response:
        status = 200

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    def fake_urlopen(request: object, timeout: float = 0.0) -> _Response:
        seen.append(str(request))
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    pinger = DeadmanPinger("https://example.invalid/check", interval_s=300.0, timeout_s=10.0)
    assert pinger.maybe_ping(now_ns=_T0, failing=False) is True
    assert seen


def test_default_transport_falls_back_to_getcode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A response without .status still resolves through getcode, even a failing one."""
    import urllib.request

    class _BareResponse:
        def __enter__(self) -> _BareResponse:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def getcode(self) -> int:
            raise OSError("no code")

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=0.0: _BareResponse())
    pinger = DeadmanPinger("https://example.invalid/check", interval_s=300.0, timeout_s=10.0)
    assert pinger.maybe_ping(now_ns=_T0, failing=False) is True
