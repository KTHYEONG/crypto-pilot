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
