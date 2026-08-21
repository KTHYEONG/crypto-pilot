"""Tests for src/common/logging.py — D3 fix verification."""

from __future__ import annotations

import logging
from io import StringIO

from src.common.logging import setup_logger, _TagDefaultFormatter, _FORMAT, _DATE_FMT


def _make_capture_handler() -> logging.StreamHandler:
    """Create a handler that uses the same _TagDefaultFormatter as setup_logger."""
    handler = logging.StreamHandler(StringIO())
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_TagDefaultFormatter(_FORMAT, datefmt=_DATE_FMT))
    return handler


def test_plain_message_no_extra_tag():
    """SCENARIO_ANALYSIS_ARCHITECTURE_01: setup_logger('x').info('plain message')

    Before the fix this path raises ValueError("Formatting field not found in
    record: 'tag'"); after it, the logger emits normally and a StreamHandler
    captures exactly 1 record whose getMessage() == 'plain message'.
    """
    logger = setup_logger("test_plain_tag", level=logging.DEBUG)
    handler = _make_capture_handler()
    logger.addHandler(handler)
    try:
        logger.info("plain message")
        output = handler.stream.getvalue()
        assert "plain message" in output
        assert "[SYS]" in output  # default tag injected
    finally:
        logger.removeHandler(handler)


def test_message_with_extra_tag():
    """Verify that explicit extra={'tag': ...} still works."""
    logger = setup_logger("test_extra_tag", level=logging.DEBUG)
    handler = _make_capture_handler()
    logger.addHandler(handler)
    try:
        logger.info("tagged message", extra={"tag": "ALGO"})
        output = handler.stream.getvalue()
        assert "tagged message" in output
        assert "[ALGO]" in output
    finally:
        logger.removeHandler(handler)


def test_default_tag_is_sys():
    """Without extra, the default tag should be SYS."""
    logger = setup_logger("test_default_sys", level=logging.DEBUG)
    handler = _make_capture_handler()
    logger.addHandler(handler)
    try:
        logger.info("check tag")
        output = handler.stream.getvalue()
        assert "[SYS]" in output
    finally:
        logger.removeHandler(handler)
