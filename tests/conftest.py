from __future__ import annotations

import os
# 다중 프로젝트 및 로컬 동시성 환경 리소스 안전 가드:
# Polars, NumPy, OpenBLAS, MKL, Numba 등이 8코어 머신에서 스레드를 과도하게 점유하지 못하도록 상한선 강제
for _key, _val in (
    ("POLARS_MAX_THREADS", "2"),
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMBA_NUM_THREADS", "1"),
    ("RAY_ACCEL_NUM_WORKERS", "1"),
    ("PYTHONDONTWRITEBYTECODE", "1"),
):
    os.environ.setdefault(_key, _val)


import os
import shutil
import tempfile
from pathlib import Path

import pytest

import uuid

from src.quant.contracts import CostModel, StrategySpec

# Project-only temp policy (AGENTS.md §4): pytest's ``tmp_path``/``tmpdir``
# fixtures, the ``tempfile`` module, and ``TMPDIR``-honoring subprocesses all
# resolve their temp root to the repo-local ``tmp/pytest/`` instead of the
# system ``/tmp``, so test artifacts never escape the project.
#
# Process-isolated partitioning: Each pytest session creates an isolated
# run directory (`proc_{pid}_{uuid}`) inside ``tmp/pytest/``. Sibling test runs
# (e.g. concurrent agents or terminal checks) never touch or delete each other's
# temporary directories, preventing race condition crashes.
_PROJECT_TEMP_ROOT = Path(__file__).resolve().parents[1] / "tmp" / "pytest"
_PROC_RUN_ID = f"proc_{os.getpid()}_{uuid.uuid4().hex[:8]}"
_PROC_TEMP_ROOT = _PROJECT_TEMP_ROOT / _PROC_RUN_ID


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    _PROC_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(_PROC_TEMP_ROOT)
    os.environ["TMPDIR"] = str(_PROC_TEMP_ROOT)
    tempfile.tempdir = str(_PROC_TEMP_ROOT)
    if hasattr(config.option, "basetemp") and not config.option.basetemp:
        config.option.basetemp = str(_PROC_TEMP_ROOT / "basetemp")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
    """Clean up this process's temporary test artifacts on session finish.

    Time Complexity: O(N) where N is the number of temporary entries in this process's temp root.
    Space Complexity: O(1) auxiliary space.
    """
    if hasattr(session.config, "workerinput"):
        # pytest-xdist worker process: sessionfinish fires independently per
        # worker; only the controller process cleans up.
        return
    if _PROC_TEMP_ROOT.exists():
        try:
            shutil.rmtree(_PROC_TEMP_ROOT, ignore_errors=True)
        except OSError:
            pass


@pytest.fixture(autouse=True, scope="session")
def _sanitize_host_environment() -> None:
    """Sanitize business environment variables across all test sessions.

    Host developer shells frequently export domain variables (e.g. LIVE_*, BINANCE_*, UPBIT_*).
    Without session-level sanitization, unit tests testing default fallbacks or negative assertions
    leak host credentials and fail unexpectedly. Individual tests that exercise env-loading
    opt-in explicitly via monkeypatch.setenv.
    """
    prefixes = ("LIVE_", "BINANCE_", "UPBIT_")
    saved: dict[str, str] = {}
    for key in list(os.environ):
        if key.startswith(prefixes):
            saved[key] = os.environ.pop(key)
    try:
        yield
    finally:
        os.environ.update(saved)


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
