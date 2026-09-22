"""tests/unit/live 공통 픽스처. 어떤 단위 테스트도 실제 네트워크를 열지 않는다."""

from __future__ import annotations

import contextlib
import os
import socket

import pytest


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch):
    """socket.connect를 예외 스텁으로 대체해 네트워크 사용을 구조적으로 차단한다."""

    def _raise(*args: object, **kwargs: object) -> None:
        raise AssertionError("unit tests must not open network sockets")

    monkeypatch.setattr(socket.socket, "connect", _raise)


@pytest.fixture(autouse=True)
def _isolate_exchange_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep developer-shell exchange credentials out of hermetic unit tests.

    ``LiveSettings`` intentionally accepts legacy ``BINANCE_*`` aliases and
    ``LIVE_*`` settings for real deployments. A host shell may export those,
    however, and silently change a test's default settings or credentials.
    Scrub all matching keys by prefix so every test starts completely hermetic
    unless it explicitly opts in via monkeypatch.setenv.
    """
    for key in list(os.environ):
        if key.startswith(("LIVE_", "BINANCE_", "UPBIT_")):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _isolate_signal_step_sidecars(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import src.cli.commands.live as live_cli

    monkeypatch.setattr(live_cli, "default_weights_path", lambda: tmp_path / "state" / "deployed_target_weights.parquet.enc")


@pytest.fixture(autouse=True)
def _isolate_live_process_logs(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """src.cli.commands.live 로그 디렉터리를 테스트 격리 경로로 돌린다."""
    import logging

    import src.cli.commands.live as live_cli

    monkeypatch.setattr(live_cli, "_LIVE_LOG_DIR", tmp_path / "live_logs")
    before = list(logging.getLogger().handlers)
    yield
    for handler in [h for h in logging.getLogger().handlers if h not in before]:
        logging.getLogger().removeHandler(handler)
        with contextlib.suppress(Exception):
            handler.close()


@pytest.fixture(autouse=True)
def _neutralize_execution_depth_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """실행 윈도우 깊이 캡처는 단위 테스트에서 실제 스레드/소켓을 열지 않는다.

    ``run_shadow_cycle`` 은 기본적으로 WS 레코더를 기동하므로, 스텁하지 않으면
    실제 aiohttp 세션과 30분 post-window 대기로 테스트가 멈춘다. 깊이 캡처 자체를
    검증하는 테스트는 자체 스텁으로 덮어쓴다.
    """
    import src.live.runner as runner_mod
    from src.live.depth_capture import DepthCaptureSummary

    class _NullDepthRecorder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def stop(self, *, post_window_s: float, shutdown: object = None) -> DepthCaptureSummary:
            return DepthCaptureSummary(rows=0, symbols_requested=0, symbols_seen=0, reconnects=0, parts=0)

    monkeypatch.setattr(runner_mod, "ExecutionDepthRecorder", _NullDepthRecorder)
