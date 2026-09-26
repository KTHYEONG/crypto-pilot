"""Invariant guards for capture logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.capture.logsetup import configure_capture_logging


def test_configures_rotating_file_and_stdout(tmp_path: Path) -> None:
    """A first call adds a bounded rotating file handler and returns its path."""
    path = configure_capture_logging(tmp_path / "logs", "bluelog")
    assert path is not None
    assert path.name == "capture_bluelog.log"
    assert path.exists()


def test_idempotent_second_call(tmp_path: Path) -> None:
    """A repeated call for the same directory and slot adds no duplicate handler."""
    root = logging.getLogger()
    before = len(root.handlers)
    configure_capture_logging(tmp_path / "logs", "greenlog")
    configure_capture_logging(tmp_path / "logs", "greenlog")
    names = [type(item).__name__ for item in root.handlers]
    assert names.count("RotatingFileHandler") <= before + 1


def test_file_failure_falls_back_to_stdout(tmp_path: Path) -> None:
    """An unusable log directory falls back to stdout-only with a None return."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not-a-directory")
    assert configure_capture_logging(blocker / "child", "blue") is None


def test_adds_stdout_handler_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare root logger gains a stdout handler alongside the rotating file."""
    root = logging.getLogger()
    saved = root.handlers[:]
    for handler in saved:
        root.removeHandler(handler)
    try:
        path = configure_capture_logging(tmp_path / "logs", "nostreamlog")
        added = [item for item in root.handlers if item not in saved]
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in saved:
            root.addHandler(handler)
    assert path is not None
    assert any(isinstance(item, logging.StreamHandler) for item in added)


def test_reuses_existing_file_handler(tmp_path: Path) -> None:
    """A forgotten marker still resolves to the already-open rotating file."""
    from src.capture import logsetup as _logsetup

    log_dir = tmp_path / "logs"
    first = configure_capture_logging(log_dir, "reuselog")
    _logsetup._configured_for.discard(f"{log_dir}:reuselog")
    second = configure_capture_logging(log_dir, "reuselog")
    assert first == second
