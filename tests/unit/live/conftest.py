"""tests/unit/live 공통 픽스처. 어떤 단위 테스트도 실제 네트워크를 열지 않는다."""

from __future__ import annotations

import contextlib
import socket

import pytest


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch):
    """socket.connect를 예외 스텁으로 대체해 네트워크 사용을 구조적으로 차단한다."""

    def _raise(*args: object, **kwargs: object) -> None:
        raise AssertionError("unit tests must not open network sockets")

    monkeypatch.setattr(socket.socket, "connect", _raise)


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