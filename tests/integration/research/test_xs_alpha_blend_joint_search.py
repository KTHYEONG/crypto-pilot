"""Contract scenarios XABJS-03 and XABJS-04 for the joint search dev tool.

The search script is the only component allowed to import ``optuna``
(transitively, via ``run_structural_search``'s own lazy import), so it must be
importable without the ``tuning`` extra -- which this module's top-level import
already proves -- and it must never touch qualification/holdout data nor call
``evaluate_xs_admission``. Both are asserted structurally over the module
source: it references none of the OOS window helpers at all.

XABJS-04: the search must additionally report a turnover diagnostic -- the
winning point's worst-6-month-fold discovery-window turnover vs
``XsAdmissionConfig().turnover_max`` -- clearly labeled informational, computed
strictly after the search returns so it can never influence which point is
selected. The functional test monkeypatches the data loader and the optuna
entry point so the real diagnostic pipeline runs on synthetic discovery-window
data without the ``tuning`` extra or any data lake.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow

from src.research.technical_experts.cross_sectional import XsAdmissionConfig
from tools.research import xs_alpha_blend_joint_search as _search_module
from tools.research.structural_tuner import StructuralSearchResult
from tools.research.xs_alpha_blend_joint_search import run

_MODULE = Path("tools/research/xs_alpha_blend_joint_search.py")


def test_xabjs_03_module_never_touches_oos() -> None:
    # XABJS-03: no reference to any qualification/holdout window helper and no
    # call to evaluate_xs_admission anywhere in the module -- its objective is
    # restricted to [XS_DISCOVERY_START, DISCOVERY_END] before scoring.
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "QUALIFICATION_START" not in names
    assert "QUALIFICATION_END" not in names
    assert "HOLDOUT_CUTOFF" not in names
    assert "evaluate_xs_admission" not in names


def test_xabjs_03_run_signature_is_keyword_only() -> None:
    # Contract python_assertion: keyword-only surface with the grid-comparable
    # trial budget default (30) and deterministic seed default (0).
    params = inspect.signature(run).parameters
    assert set(params) == {"max_trials", "seed"}
    assert all(p.kind == p.KEYWORD_ONLY for p in params.values())
    assert params["max_trials"].default == 30
    assert params["seed"].default == 0


def test_xabjs_04_diagnostic_block_is_strictly_post_search() -> None:
    # XABJS-04: the turnover diagnostic lives inside run() but only ever
    # *after* run_structural_search returns -- it recomputes the winning
    # point's scaled ledger for reporting, never feeds back into the search or
    # re-enters optuna -- and run() keeps its keyword-only contract surface.
    src = inspect.getsource(run)
    search_pos = src.find("run_structural_search(")
    diag_pos = src.find("discovery_worst_fold_turnover=")
    assert search_pos != -1
    assert diag_pos > search_pos
    assert "compute_turnover_fold_upper_bound" in src
    assert "diagnostic" in src.lower()
    assert "turnover_max=" in src


def test_xabjs_04_reports_turnover_diagnostic_without_altering_selection(
    monkeypatch, capsys,
) -> None:
    # XABJS-04 (functional): running run() end-to-end on synthetic
    # discovery-window data (with the data loader and the optuna entry point
    # monkeypatched out) must (1) return the exact StructuralSearchResult the
    # search produced -- adding the diagnostic must not alter which point is
    # selected -- and (2) print the informational worst-fold turnover line with
    # the XsAdmissionConfig().turnover_max comparison.
    start = pd.Timestamp("2022-04-03", tz="UTC")
    end = pd.Timestamp("2023-04-30 23:59:59", tz="UTC")
    idx = pd.date_range(start, end, freq="4h", tz="UTC")
    rng = np.random.default_rng(0)
    xs_net = pd.Series(
        rng.normal(0.0004, 0.008, len(idx)), index=idx, name="xs_alpha_net",
    )
    xs_w = pd.DataFrame(
        rng.uniform(0.5, 1.0, (len(idx), 2)), index=idx, columns=["a", "b"],
    )
    bl_net = pd.Series(
        rng.normal(0.0002, 0.004, len(idx)), index=idx, name="baseline_net",
    )
    bl_w = pd.Series(
        rng.uniform(0.8, 1.2, len(idx)), index=idx, name="baseline_realized_weight",
    )

    monkeypatch.setattr(
        _search_module, "_load_net_returns", lambda: (xs_net, xs_w, bl_net, bl_w),
    )
    monkeypatch.setattr(_search_module, "XS_DISCOVERY_START", start)
    monkeypatch.setattr(_search_module, "DISCOVERY_END", end)
    best = {"xs_alpha_weight": 0.5, "leverage_scale": 2.0}
    fake_result = StructuralSearchResult(
        best_params=best, best_is_score=10.0,
        plateau_neighbor_ratio=0.8, plateau_passed=True, n_trials=4,
    )
    monkeypatch.setattr(
        _search_module, "run_structural_search",
        lambda objective, search_space, config: fake_result,
    )

    result = _search_module.run(max_trials=4, seed=0)
    assert result is fake_result
    assert result.best_params == best

    out = capsys.readouterr().out
    assert "discovery_worst_fold_turnover=" in out
    assert f"turnover_max={XsAdmissionConfig().turnover_max:.1f}" in out
    assert "diagnostic" in out
