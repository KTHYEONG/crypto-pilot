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
# Pre-change fold CAGR vector: folds 1-3 were bit-identical to baseline under
# the D1 defect (the boundary-resolved target vol never reached them).
_BASELINE_FOLD_CAGRS = [0.222784719, 0.351131440, 0.246587046, 1.617203362]


def _run_cli(extra_args: list[str]) -> None:
    cmd = [
        sys.executable, "-m", "src.cli.main", "research", "run", "portfolio",
        "mhs-horizon-diagnostic", "--no-log-run", *extra_args,
    ]
    subprocess.run(cmd, check=True, timeout=1800)  # noqa: S603


def _load_report() -> dict:
    return json.loads(_REPORT.read_text())


@pytest.mark.slow
def test_balanced_growth_budget_propagates_to_folds() -> None:
    # SCENARIO_MHS_GROWTH_BUDGET_ACCEPTANCE_07: with balanced/growth_budget the
    # fold evidence must move off the baseline vector (>= 3 of 4 entries) --
    # bit-identical folds are the D1 defect -- while the blend stays inside the
    # registered balanced drawdown budget and the parity guard stays silent.
    _run_cli(["--growth-envelope", "balanced", "--pnl-vol-target-mode", "growth_budget"])
    report = _load_report()

    fold_cagrs = [f["primary_geometric_cagr"] for f in report["folds"]]
    moved = sum(
        abs(actual - expected) > 1e-9
        for actual, expected in zip(fold_cagrs, _BASELINE_FOLD_CAGRS, strict=True)
    )
    assert moved >= 3, f"fold evidence still frozen at baseline: {fold_cagrs}"

    blend = report["blend"]
    assert blend["primary_geometric_cagr"] >= 0.90
    assert blend["primary_max_drawdown"] >= -0.35
    assert blend["primary_naive_sharpe"] >= 2.10
    assert blend["stress_naive_sharpe"] > 0
    assert "FOLD_BLEND_PATH_DIVERGENCE" not in report["research_go"]["reason_codes"]

    # Restore the committed baseline report state.
    _run_cli([])


@pytest.mark.slow
def test_default_run_reproduces_original_conservative_baseline_via_explicit_flags() -> None:
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
    assert blend["primary_geometric_cagr"] == pytest.approx(0.560204, abs=1e-6)
    assert blend["primary_max_drawdown"] == pytest.approx(-0.176059, abs=1e-6)
    assert blend["primary_annualized_turnover"] == pytest.approx(78.069696, abs=1e-6)
    folds = report["folds"]
    assert len(folds) == len(_BASELINE_FOLD_CAGRS)
    for fold, expected in zip(folds, _BASELINE_FOLD_CAGRS, strict=True):
        assert fold["primary_geometric_cagr"] == pytest.approx(expected, abs=1e-9)

    # Restore the committed baseline report state.
    _run_cli([])


@pytest.mark.slow
def test_default_run_reproduces_growth_main_logic_metrics() -> None:
    # SCENARIO_MHS_DEFAULT_BYTE_IDENTICAL_08 (superseded 2026-08-22): the
    # main-logic default is now growth-envelope + growth_budget +
    # evidence-weighting -- explicit user decision to maximize compounding
    # growth without regard to drawdown magnitude, bound only by the
    # registered 1% ruin-probability frontier
    # (GROWTH_RISK_ENVELOPES["growth"]: max_ruin_prob=0.01, ruin_fraction=0.60).
    # A bare CLI invocation (no flags) must reproduce these measured numbers.
    _run_cli([])
    report = _load_report()
    assert report["growth_envelope"]["name"] == "growth"
    blend = report["blend"]
    assert blend["primary_geometric_cagr"] == pytest.approx(1.124586, abs=1e-5)
    assert blend["primary_max_drawdown"] == pytest.approx(-0.264844, abs=1e-5)
    assert blend["primary_annualized_turnover"] == pytest.approx(117.912931, abs=1e-4)
    assert blend["primary_naive_sharpe"] == pytest.approx(2.331173, abs=1e-4)
    folds = report["folds"]
    expected_fold_cagrs = [0.31600648, 0.38799087, 0.15449526, 1.92949683]
    assert len(folds) == len(expected_fold_cagrs)
    for fold, expected in zip(folds, expected_fold_cagrs, strict=True):
        assert fold["primary_geometric_cagr"] == pytest.approx(expected, abs=1e-6)
