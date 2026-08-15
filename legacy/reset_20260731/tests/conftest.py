"""pytest root conftest — project root를 sys.path에 추가."""

from __future__ import annotations

import sys
from pathlib import Path

# project root: tests/ 의 상위 디렉토리
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pytest

from src.domain.futures.compound.contracts import TimeframeBarCube


@pytest.fixture
def light_test_bars() -> TimeframeBarCube:
    """200-bar lightweight fixture for unit tests (capped from 5,442 full bars)."""
    t = 200
    n = 5
    close = np.column_stack(
        [np.linspace(100.0, 100.0 + 20.0 * i / n, t) for i in range(n)]
    ).astype(np.float32)
    return TimeframeBarCube(
        "4h",
        np.arange(t, dtype=np.int64),
        tuple(f"S{i}" for i in range(n)),
        close,
        close + 2.0,
        close - 2.0,
        close,
        np.ones((t, n), dtype=np.float32),
        np.ones((t, n), dtype=bool),
    )
