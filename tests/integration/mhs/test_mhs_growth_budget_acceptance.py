"""Acceptance runs for MHS growth-budget boundary propagation (SCENARIO 07/08).

Each test drives the full-pipeline CLI over the real multi-year MHS lake at
[redacted]
asserts on the per-run persisted compact report at
[redacted]
default suite via the ``slow`` marker and skipped without the lake via
``requires_market_lake``; run explicitly with
``uv run pytest tests/integration/mhs/test_mhs_growth_budget_acceptance.py -m slow``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from src.common.paths import DATA_DIR

pytestmark = [pytest.mark.slow, pytest.mark.requires_market_lake]

[redacted]
[redacted]
[redacted]


@contextmanager
def _run_cli(extra_args: list[str]) -> Iterator[Path]:
    run_id = f"pytest-growth-acceptance-{uuid.uuid4().hex}"
    cmd = [
        sys.executable, "-m", "src.cli.main", "research", "run", "portfolio",
        "mhs-horizon-diagnostic", "--no-log-run", "--run-id", run_id, *extra_args,
    ]
    run_dir = DATA_DIR / "research" / "mhs" / run_id
    try:
        subprocess.run(cmd, check=True, timeout=1800)  # noqa: S603
        yield run_dir / "mhs_horizon_diagnostic.json"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


@pytest.mark.slow
[redacted]
    # SCENARIO_MHS_GROWTH_BUDGET_ACCEPTANCE_07: with balanced/growth_budget the
    [redacted]
    [redacted]
    [redacted]
    with _run_cli(["--growth-envelope", "balanced", "--pnl-vol-target-mode", "growth_budget"]) as report_path:
        report = json.loads(report_path.read_text())

        fold_cagrs = [f["primary_geometric_cagr"] for f in report["folds"]]
        moved = sum(
            [redacted]
            [redacted]
        )
        [redacted]

        blend = report["blend"]
        [redacted]
        [redacted]
        [redacted]
        assert blend["stress_naive_sharpe"] > 0
        assert "FOLD_BLEND_PATH_DIVERGENCE" not in report["research_go"]["reason_codes"]


@pytest.mark.slow
[redacted]
    # Regression anchor for the original (pre-2026-08-22) production default,
    # now reachable only via explicit flags since --growth-envelope/
    # --pnl-vol-target-mode/--committee-evidence-weighting main-logic
    # defaults changed. Bit-identical to the historically registered numbers.
    with _run_cli([
        "--growth-envelope", "conservative", "--pnl-vol-target-mode", "exante_target",
        "--no-committee-evidence-weighting",
    ]) as report_path:
        report = json.loads(report_path.read_text())
        blend = report["blend"]
        [redacted]
        [redacted]
        [redacted]
        folds = report["folds"]
        [redacted]
        [redacted]
            [redacted]


@pytest.mark.slow
[redacted]
    # SCENARIO_MHS_DEFAULT_BYTE_IDENTICAL_08 (superseded 2026-08-22): the
    # main-logic default is now growth-envelope + growth_budget +
    # evidence-weighting -- explicit user decision to maximize compounding
    # growth without regard to drawdown magnitude, bound only by the
    [redacted]
    [redacted]
    [redacted]
    with _run_cli([]) as report_path:
        report = json.loads(report_path.read_text())
        assert report["growth_envelope"]["name"] == "growth"
        blend = report["blend"]
        [redacted]
        [redacted]
        [redacted]
        [redacted]
        folds = report["folds"]
        [redacted]
        [redacted]
        [redacted]
            [redacted]
