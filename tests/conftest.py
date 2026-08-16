from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from src.research.contracts import CostModel, StrategySpec

# Project-only temp policy (AGENTS.md §4): pytest's ``tmp_path``/``tmpdir``
# fixtures, the ``tempfile`` module, and ``TMPDIR``-honoring subprocesses all
# resolve their temp root to the repo-local ``tmp/pytest/`` instead of the
# system ``/tmp``, so test artifacts never escape the project. The Sync skill
# purges the ``tmp/`` directory at task finalization.
_PROJECT_TEMP_ROOT = Path(__file__).resolve().parents[1] / "tmp" / "pytest"


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    _PROJECT_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(_PROJECT_TEMP_ROOT)
    os.environ["TMPDIR"] = str(_PROJECT_TEMP_ROOT)
    tempfile.tempdir = str(_PROJECT_TEMP_ROOT)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
    """Clean up temporary test artifacts on session finish.

    Time Complexity: O(N) where N is the number of temporary entries in the session temp root.
    Space Complexity: O(1) auxiliary space.
    """
    if hasattr(session.config, "workerinput"):
        # pytest-xdist worker process: sessionfinish fires independently per
        # worker, so sweeping the shared temp root here would race sibling
        # workers still creating their own tmp_path directories. Only the
        # controller process (no ``workerinput``) may clean up.
        return
    if _PROJECT_TEMP_ROOT.exists():
        for child in _PROJECT_TEMP_ROOT.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            except OSError:
                pass


pytest_plugins = [
    "tests.fixtures.bars",
    "tests.fixtures.market_data",
]


@pytest.fixture
def spec() -> StrategySpec:
    return StrategySpec()


@pytest.fixture
def costs() -> CostModel:
    return CostModel()
