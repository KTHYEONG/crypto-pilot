"""Contract scenario XABJS-03 for the joint search dev tool.

The search script is the only component allowed to import ``optuna``
(transitively, via ``run_structural_search``'s own lazy import), so it must be
importable without the ``tuning`` extra -- which this module's top-level import
already proves -- and it must never touch qualification/holdout data nor call
``evaluate_xs_admission``. Both are asserted structurally over the module
source: it references none of the OOS window helpers at all.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

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
