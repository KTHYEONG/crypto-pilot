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


import contextlib
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

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

# Route process logs and the backtest registry to this run's temp root *before* any ``src`` import:
# ``setup_logger`` opens its log files and registry defaults resolve at import time, so an
# unconditional override here keeps tests from writing into the developer's real ``logs/`` and
# ``data/backtests/registry.sqlite3`` regardless of the host shell.
os.environ["CRYPTO_PILOT_LOG_DIR"] = str(_PROC_TEMP_ROOT / "logs")
os.environ["CRYPTO_PILOT_BACKTESTS_DIR"] = str(_PROC_TEMP_ROOT / "backtests")

from src.quant.contracts import CostModel, StrategySpec  # noqa: E402

# Developer-tree paths that must be byte-identical before and after a test session. A test that
# writes here pollutes real state, logs or the backtest registry, so the session fails loudly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_GUARDED_TREES: tuple[str, ...] = ("data/state", "data/live_capture", "data/backtests", "logs")


def _snapshot_guarded_trees() -> dict[str, tuple[int, int]]:
    """Return ``{relative_path: (size, mtime_ns)}`` for every file under the guarded trees."""
    snapshot: dict[str, tuple[int, int]] = {}
    for tree in _GUARDED_TREES:
        root = _REPO_ROOT / tree
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                stat = path.stat()
                snapshot[str(path.relative_to(_REPO_ROOT))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


_GUARDED_BEFORE = _snapshot_guarded_trees()


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
    after = _snapshot_guarded_trees()
    leaked = sorted(path for path in after.keys() | _GUARDED_BEFORE.keys() if after.get(path) != _GUARDED_BEFORE.get(path))
    if leaked:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        shown = "\n  ".join(leaked[:20])
        sys.stderr.write(
            f"\nHERMETIC GUARD: tests changed {len(leaked)} file(s) under {', '.join(_GUARDED_TREES)}:\n  {shown}\n"
            "Route the writer to tmp_path (see tests/conftest.py and tests/unit/live/conftest.py).\n"
        )
    if _PROC_TEMP_ROOT.exists():
        with contextlib.suppress(OSError):
            shutil.rmtree(_PROC_TEMP_ROOT, ignore_errors=True)


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
