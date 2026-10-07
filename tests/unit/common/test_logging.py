"""Tests for src/common/logging.py — D3 fix verification."""

from __future__ import annotations

import contextlib
import logging
from io import StringIO
from pathlib import Path

import pytest

from src.common.logging import setup_logger, _TagDefaultFormatter, _FORMAT, _DATE_FMT


def _make_capture_handler() -> logging.StreamHandler:
    """Create a handler that uses the same _TagDefaultFormatter as setup_logger."""
    handler = logging.StreamHandler(StringIO())
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_TagDefaultFormatter(_FORMAT, datefmt=_DATE_FMT))
    return handler


def _close_logger_handlers(name: str) -> None:
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        with contextlib.suppress(Exception):
            handler.close()


def test_plain_message_no_extra_tag(tmp_path: Path):
    """SCENARIO_ANALYSIS_ARCHITECTURE_01: setup_logger('x', log_dir=...).info('plain message')

    Before the fix this path raises ValueError("Formatting field not found in
    record: 'tag'"); after it, the logger emits normally and a StreamHandler
    captures exactly 1 record whose getMessage() == 'plain message'.
    """
    logger = setup_logger("test_plain_tag", log_dir=tmp_path, level=logging.DEBUG)
    handler = _make_capture_handler()
    logger.addHandler(handler)
    try:
        logger.info("plain message")
        output = handler.stream.getvalue()
        assert "plain message" in output
        assert "[SYS]" in output  # default tag injected
    finally:
        logger.removeHandler(handler)
        _close_logger_handlers("test_plain_tag")


def test_message_with_extra_tag(tmp_path: Path):
    """Verify that explicit extra={'tag': ...} still works."""
    logger = setup_logger("test_extra_tag", log_dir=tmp_path, level=logging.DEBUG)
    handler = _make_capture_handler()
    logger.addHandler(handler)
    try:
        logger.info("tagged message", extra={"tag": "ALGO"})
        output = handler.stream.getvalue()
        assert "tagged message" in output
        assert "[ALGO]" in output
    finally:
        logger.removeHandler(handler)
        _close_logger_handlers("test_extra_tag")


def test_default_tag_is_sys(tmp_path: Path):
    """Without extra, the default tag should be SYS."""
    logger = setup_logger("test_default_sys", log_dir=tmp_path, level=logging.DEBUG)
    handler = _make_capture_handler()
    logger.addHandler(handler)
    try:
        logger.info("check tag")
        output = handler.stream.getvalue()
        assert "[SYS]" in output
    finally:
        logger.removeHandler(handler)
        _close_logger_handlers("test_default_sys")


def test_setup_logger_requires_explicit_log_dir(tmp_path: Path):
    name = "test_explicit_log_dir_required"
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    sentinel = _make_capture_handler()
    logger.addHandler(sentinel)
    try:
        with pytest.raises(TypeError):
            setup_logger(name)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            setup_logger(name, log_dir=None)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            setup_logger(name, log_dir=str(tmp_path))  # type: ignore[arg-type]
        assert list(tmp_path.iterdir()) == []
        assert not sentinel._closed
        assert list(logger.handlers) == [sentinel]
    finally:
        logger.removeHandler(sentinel)
        sentinel.close()
        _close_logger_handlers(name)


def test_setup_logger_writes_only_under_given_dir(tmp_path: Path):
    dest = tmp_path / "x"
    assert not dest.exists()
    logger = setup_logger("n_explicit_dir_probe", log_dir=dest)
    try:
        logger.info("m")
    finally:
        _close_logger_handlers("n_explicit_dir_probe")
    assert (dest / "n_explicit_dir_probe.log").exists()
    assert "m" in (dest / "n_explicit_dir_probe.log").read_text(encoding="utf-8")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x"]
    assert sorted(p.name for p in dest.iterdir()) == ["n_explicit_dir_probe.log"]


def test_setup_logger_reinvocation_closes_replaced_file_handler(tmp_path: Path):
    name = "n_reinvoke_close_probe"
    first = setup_logger(name, log_dir=tmp_path)
    first_file = next(h for h in first.handlers if isinstance(h, logging.FileHandler))
    assert len(first.handlers) == 2
    second = setup_logger(name, log_dir=tmp_path)
    assert second is first
    file_handlers = [h for h in second.handlers if isinstance(h, logging.FileHandler)]
    stream_handlers = [h for h in second.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert len(stream_handlers) == 1
    stream = getattr(first_file, "stream", None)
    assert stream is None or getattr(stream, "closed", False)
    _close_logger_handlers(name)


def test_tagged_message_emits_single_tag(tmp_path: Path):
    logger = setup_logger("test_single_tag_probe", log_dir=tmp_path, level=logging.DEBUG)
    handler = _make_capture_handler()
    logger.addHandler(handler)
    try:
        logger.info("[ALGO] stage=committee_book")
        logger.info("[SYS] stage=base_1h_panel rss=1 elapsed_ms=2")
        lines = handler.stream.getvalue().strip().splitlines()
        assert len(lines) == 2
        assert lines[0].count("[ALGO]") == 1
        assert "[SYS]" not in lines[0].split("[ALGO]")[0]
        assert lines[1].count("[SYS]") == 1
    finally:
        logger.removeHandler(handler)
        _close_logger_handlers("test_single_tag_probe")
