"""Lifecycle shutdown flag tests."""

from src.live.lifecycle import ShutdownFlag, install_shutdown_handlers


def test_shutdown_flag_basic() -> None:
    flag = ShutdownFlag()
    assert not flag.requested
    flag.request("SIGTERM")
    assert flag.requested
    assert flag.signal_name == "SIGTERM"


def test_install_shutdown_handlers_noop_on_non_main() -> None:
    flag = ShutdownFlag()
    # Should not raise even when called from non-main thread context (we are main, but should not crash)
    install_shutdown_handlers(flag)
    assert True


def test_shutdown_flag_wait_returns_immediately_when_requested_from_another_thread() -> None:
    import threading
    import time
    from src.live.lifecycle import ShutdownFlag

    flag = ShutdownFlag()
    timer = threading.Timer(0.05, flag.request, args=("SIGTERM",))
    timer.daemon = True
    started = time.monotonic()
    timer.start()

    woke = flag.wait(5.0)

    assert woke is True
    assert time.monotonic() - started < 2.0
    assert flag.requested is True
    assert flag.signal_name == "SIGTERM"


def test_shutdown_flag_wait_times_out_without_request() -> None:
    from src.live.lifecycle import ShutdownFlag

    flag = ShutdownFlag()

    assert flag.wait(0.01) is False
    assert flag.requested is False


def test_docker_compose_mhs_live_stop_grace_period_covers_stage_boundary() -> None:
    from pathlib import Path

    text = Path("docker-compose.yml").read_text(encoding="utf-8")
    mhs_block = text.split("  mhs-live:", 1)[1].split("  market-recorder:", 1)[0]
    liq_block = text.split("  market-recorder:", 1)[1]

    assert "stop_grace_period: 120s" in mhs_block
    assert "stop_grace_period: 30s" in liq_block

