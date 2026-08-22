"""Acceptance runs for MHS growth-budget boundary propagation (SCENARIO 07/08).

Each test drives the full-pipeline CLI over real multi-year data (~5-7 minutes
per run on the measurement machine) and asserts on the persisted compact
report at ``docs/results/mhs_horizon_diagnostic.json``. Excluded from the
default suite via the ``slow`` marker; run explicitly with
``uv run pytest tests/integration/mhs/test_mhs_growth_budget_acceptance.py -m slow``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPORT = Path("docs/results/mhs_horizon_diagnostic.json")
[redacted]
[redacted]
[redacted]


def _run_cli(extra_args: list[str]) -> None:
    cmd = [
        sys.executable, "-m", "src.cli.main", "research", "run", "portfolio",
        "mhs-horizon-diagnostic", "--no-log-run", *extra_args,
    ]
    subprocess.run(cmd, check=True, timeout=1800)  # noqa: S603


def _load_report() -> dict:
    return json.loads(_REPORT.read_text())


@pytest.mark.slow
[redacted]
    # SCENARIO_MHS_GROWTH_BUDGET_ACCEPTANCE_07: with balanced/growth_budget the
    [redacted]
    [redacted]
    [redacted]
    _run_cli(["--growth-envelope", "balanced", "--pnl-vol-target-mode", "growth_budget"])
    report = _load_report()

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

    # Restore the committed baseline report state.
    _run_cli([])


@pytest.mark.slow
[redacted]
    # Regression anchor for the original (pre-2026-08-22) production default,
    # now reachable only via explicit flags since --growth-envelope/
    # --pnl-vol-target-mode/--committee-evidence-weighting main-logic
    # defaults changed. Bit-identical to the historically registered numbers.
    _run_cli([
        "--growth-envelope", "conservative", "--pnl-vol-target-mode", "exante_target",
        "--no-committee-evidence-weighting",
    ])
    report = _load_report()
    blend = report["blend"]
    [redacted]
    [redacted]
    [redacted]
    folds = report["folds"]
    [redacted]
    [redacted]
        [redacted]

    # Restore the committed baseline report state.
    _run_cli([])


@pytest.mark.slow
[redacted]
    # SCENARIO_MHS_DEFAULT_BYTE_IDENTICAL_08 (superseded 2026-08-22): the
    # main-logic default is now growth-envelope + growth_budget +
    # evidence-weighting -- explicit user decision to maximize compounding
    # growth without regard to drawdown magnitude, bound only by the
    [redacted]
    [redacted]
    [redacted]
    _run_cli([])
    report = _load_report()
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
