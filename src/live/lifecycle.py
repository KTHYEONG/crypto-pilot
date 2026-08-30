"""Shutdown coordination boundary (NO-SIGNAL-IMPORT-LEAK).

signal 모듈 import는 이 파일에만 존재한다.
"""

from __future__ import annotations

import signal
import threading
from dataclasses import dataclass


@dataclass(slots=True)
class ShutdownFlag:
    requested: bool = False
    signal_name: str | None = None

    def request(self, signal_name: str) -> None:
        self.requested = True
        self.signal_name = signal_name


def install_shutdown_handlers(
    flag: ShutdownFlag, *, signals: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT)
) -> None:
    if threading.current_thread() is not threading.main_thread():
        return

    def _handler(signum: int, _frame: object) -> None:
        try:
            name = signal.Signals(signum).name
        except (ValueError, AttributeError):
            name = str(signum)
        flag.request(name)

    for sig in signals:
        try:
            signal.signal(sig, _handler)
        except (ValueError, AttributeError, OSError):
            continue
