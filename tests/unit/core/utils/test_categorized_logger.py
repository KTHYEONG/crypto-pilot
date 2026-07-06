import logging
from typing import cast

import pytest

from src.core.utils.utils import CategorizedLogger, setup_logger


def test_categorized_logger_methods(caplog: pytest.LogCaptureFixture) -> None:
    """Verify logger.perf(), logger.data(), logger.opt(), logger.strat() methods work."""
    logger = cast(CategorizedLogger, setup_logger("TestLogger_Methods", write_file=False))
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    caplog.set_level(logging.DEBUG)

    # Trigger different categorized methods
    logger.perf("simulated pipeline step")
    logger.data("loaded universe assets")
    logger.opt("optuna objective evaluated")
    logger.strat("gate checks passed")
    logger.debug("generic debug log message")

    records = caplog.records
    assert len(records) >= 5

    # Check level is standard DEBUG (10) for all
    for rec in records:
        assert rec.levelno == logging.DEBUG

    # Check automatic category tagging
    assert any("[PERF] simulated pipeline step" in rec.message for rec in records)
    assert any("[DATA] loaded universe assets" in rec.message for rec in records)
    assert any("[OPT] optuna objective evaluated" in rec.message for rec in records)
    assert any("[STRAT] gate checks passed" in rec.message for rec in records)
    assert any("[SYS] generic debug log message" in rec.message for rec in records)


def test_categorized_logger_fallback_to_sys(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that any debug logs without a valid tag automatically fall back to [SYS]."""
    logger = setup_logger("TestLogger_Fallback", write_file=False)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    caplog.set_level(logging.DEBUG)

    # These do not start with a valid tag
    logger.debug("[PROFILE] database operation completed in 0.5s")
    logger.debug("[DATASET] loaded 100 entries")
    logger.debug("study run started")

    records = caplog.records
    assert len(records) >= 3

    # All should fall back to [SYS]
    assert any("[SYS] [PROFILE] database operation completed in 0.5s" in rec.message for rec in records)
    assert any("[SYS] [DATASET] loaded 100 entries" in rec.message for rec in records)
    assert any("[SYS] study run started" in rec.message for rec in records)


def test_categorized_logger_no_double_prefixing(caplog: pytest.LogCaptureFixture) -> None:
    """Verify logger does not prepend category prefix if it already starts with it."""
    logger = cast(CategorizedLogger, setup_logger("TestLogger_NoDouble", write_file=False))
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    caplog.set_level(logging.DEBUG)

    logger.perf("already prefixed performance")
    logger.debug("[OPT] already prefixed optimization")

    records = caplog.records
    assert any(rec.message == "[PERF] already prefixed performance" for rec in records)
    assert any(rec.message == "[OPT] already prefixed optimization" for rec in records)
    # Ensure there is no duplicated prefix like "[PERF] [PERF]"
    assert not any("[PERF] [PERF]" in rec.message for rec in records)
    assert not any("[OPT] [OPT]" in rec.message for rec in records)
