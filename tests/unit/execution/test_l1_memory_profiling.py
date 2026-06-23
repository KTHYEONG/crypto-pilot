from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from src.execution.opt_main_futures import _get_rss_mb


def test_get_rss_mb_returns_positive_on_linux() -> None:
    result = _get_rss_mb()
    assert isinstance(result, float)
    assert result > 0.0


def test_get_rss_mb_returns_neg_one_on_missing_proc() -> None:
    with patch("builtins.open", side_effect=FileNotFoundError):
        result = _get_rss_mb()
    assert result == -1.0


def test_log_mem_emits_debug_record(caplog: pytest.LogCaptureFixture) -> None:
    from src.execution.opt_main_futures import _log_mem
    logger = logging.getLogger("opt_main_futures")
    caplog.set_level(logging.DEBUG, logger="opt_main_futures")
    old_propagate = logger.propagate
    logger.propagate = True
    try:
        old_handler_levels = [(h, h.level) for h in logger.handlers]
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        _log_mem("test_stage", 100.0, extra="n_syms=5")
        found = any(
            "[MEM] stage=test_stage" in r.message and "n_syms=5" in r.message
            for r in caplog.records
        )
        assert found, f"[MEM] log not found in caplog records: {[r.message for r in caplog.records]}"
    finally:
        logger.propagate = old_propagate
        for h, lvl in old_handler_levels:
            h.setLevel(lvl)
