"""Acceptance runs for MHS growth-budget boundary propagation (SCENARIO 07/08).

Each test drives the full-pipeline CLI over the real multi-year MHS lake at
``data/futures/ohlcv`` and asserts on the per-run persisted compact report at
``data/research/mhs/<run_id>/mhs_horizon_diagnostic.json``. The registered
expected numbers are research results and live in the private strategy store
(``crypto-pilot/research/acceptance/mhs_growth_budget.json``); the tests skip
when the active store is the public example store or lacks that file. Excluded
from the default suite via the ``slow`` marker and skipped without the lake via
``requires_market_lake``; run explicitly with
``uv run pytest tests/lab/mhs/integration/test_mhs_growth_budget_acceptance.py -m slow``.
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
from typing import Any

import pytest

from src.common.paths import DATA_DIR
from src.strategy.store import strategy_values

pytestmark = [pytest.mark.slow, pytest.mark.requires_market_lake]

_ACCEPTANCE_FILE = Path("crypto-pilot") / "research" / "acceptance" / "mhs_growth_budget.json"
_ACCEPTANCE_SCHEMA = "crypto.research.mhs_growth_budget_acceptance.v1"
_GROWTH_BLEND_ABS_TOL = {
    "primary_geometric_cagr": 1e-5,
    "primary_max_drawdown": 1e-5,
    "primary_annualized_turnover": 1e-4,
    "primary_naive_sharpe": 1e-4,
}


def _load_expected(root: Path, *, example: bool) -> dict[str, Any]:
    """Registered acceptance numbers from the store ``root``; skips on the example store or a missing file."""
    path = root / _ACCEPTANCE_FILE
    if example or not path.is_file():
        pytest.skip("registered acceptance numbers live in the private strategy store")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != _ACCEPTANCE_SCHEMA:
        pytest.fail(f"unsupported acceptance research file: {_ACCEPTANCE_FILE}")
    return data


@pytest.fixture(scope="module")
def expected() -> dict[str, Any]:
    values = strategy_values()
    return _load_expected(values.root, example=values.example)


@contextmanager
def _run_cli(extra_args: list[str]) -> Iterator[Path]:
    run_id = f"pytest-growth-acceptance-{uuid.uuid4().hex}"
    cmd = [
        sys.executable, "-m", "src.cli.main", "lab",
        "horizon-diagnostic", "--no-log-run", "--run-id", run_id, *extra_args,
    ]
    run_dir = DATA_DIR / "research" / "mhs" / run_id
    try:
        subprocess.run(cmd, check=True, timeout=1800)  # noqa: S603
        yield run_dir / "mhs_horizon_diagnostic.json"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


@pytest.mark.slow
def test_balanced_growth_budget_propagates_to_folds(expected: dict[str, Any]) -> None:
    # SCENARIO_MHS_GROWTH_BUDGET_ACCEPTANCE_07: with balanced/growth_budget the
    # fold evidence must move off the registered baseline vector -- bit-identical
    # folds are the D1 defect -- while the blend stays inside the registered
    # balanced drawdown budget and the parity guard stays silent.
    with _run_cli(["--growth-envelope", "balanced", "--pnl-vol-target-mode", "growth_budget"]) as report_path:
        report = json.loads(report_path.read_text())

        fold_cagrs = [f["primary_geometric_cagr"] for f in report["folds"]]
        moved = sum(
            abs(actual - baseline) > 1e-9
            for actual, baseline in zip(fold_cagrs, expected["baseline_fold_cagrs"], strict=True)
        )
        balanced = expected["balanced_growth_budget"]
        assert moved >= balanced["min_folds_moved"], f"fold evidence still frozen at baseline: {fold_cagrs}"

        blend = report["blend"]
        for key, floor in balanced["blend_floor"].items():
            assert blend[key] >= floor, key
        assert blend["stress_naive_sharpe"] > 0
        assert "FOLD_BLEND_PATH_DIVERGENCE" not in report["research_go"]["reason_codes"]


@pytest.mark.slow
def test_default_run_reproduces_original_conservative_baseline_via_explicit_flags(
    expected: dict[str, Any],
) -> None:
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
        for key, value in expected["conservative_baseline"]["blend"].items():
            assert blend[key] == pytest.approx(value, abs=1e-6), key
        folds = report["folds"]
        baseline = expected["baseline_fold_cagrs"]
        assert len(folds) == len(baseline)
        for fold, value in zip(folds, baseline, strict=True):
            assert fold["primary_geometric_cagr"] == pytest.approx(value, abs=1e-9)


@pytest.mark.slow
def test_default_run_reproduces_growth_main_logic_metrics(expected: dict[str, Any]) -> None:
    # SCENARIO_MHS_DEFAULT_BYTE_IDENTICAL_08 (superseded 2026-08-22): the
    # main-logic default is now growth-envelope + growth_budget +
    # evidence-weighting -- explicit user decision to maximize compounding
    # growth without regard to drawdown magnitude, bound only by the
    # registered ruin-probability frontier (GROWTH_RISK_ENVELOPES["growth"]).
    # A bare CLI invocation (no flags) must reproduce the registered numbers.
    with _run_cli([]) as report_path:
        report = json.loads(report_path.read_text())
        assert report["growth_envelope"]["name"] == "growth"
        growth = expected["growth_default"]
        blend = report["blend"]
        for key, value in growth["blend"].items():
            assert blend[key] == pytest.approx(value, abs=_GROWTH_BLEND_ABS_TOL[key]), key
        folds = report["folds"]
        assert len(folds) == len(growth["fold_cagrs"])
        for fold, value in zip(folds, growth["fold_cagrs"], strict=True):
            assert fold["primary_geometric_cagr"] == pytest.approx(value, abs=1e-6)
