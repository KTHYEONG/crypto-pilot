"""tests/unit/live 공통 픽스처. 어떤 단위 테스트도 실제 네트워크를 열지 않는다."""

from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch):
    """socket.connect를 예외 스텁으로 대체해 네트워크 사용을 구조적으로 차단한다."""

    def _raise(*args: object, **kwargs: object) -> None:
        raise AssertionError("unit tests must not open network sockets")

    monkeypatch.setattr(socket.socket, "connect", _raise)
