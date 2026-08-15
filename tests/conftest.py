from __future__ import annotations

import os
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
def pytest_configure(config) -> None:
    _PROJECT_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(_PROJECT_TEMP_ROOT)
    os.environ["TMPDIR"] = str(_PROJECT_TEMP_ROOT)
    tempfile.tempdir = str(_PROJECT_TEMP_ROOT)


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
