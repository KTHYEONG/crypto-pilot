"""SCENARIO_MHS_PERF_P4_01_DEPENDENCY_DIRECTION: module boundary contract.

An AST scan enforces the blueprint P4 dependency direction:

* ``src/application/research/mhs/evaluation.py`` imports nothing from
  ``src.mhs.pipeline.*`` at module level (the two function-scoped imports in
  ``run_mhs_horizon_diagnostic`` are the repo's documented cycle-breaker seam
  and are asserted to stay exactly there);
* no module under ``src/mhs/pipeline/stages/`` imports a private symbol from
  ``evaluation.py`` directly, NOR reaches one via attribute access on an
  ``evaluation``-aliased module object (``_evaluation._foo(...)``) -- both are
  the same coupling; only the ``ImportFrom`` form is visible to a naive
  scanner, so both are checked. Stage-facing helpers resolve exclusively
  through the ``stage_services`` seam (an identity re-export of the same
  function objects, pinned by
  tests/unit/application/research/mhs/test_stage_services.py);
* ``src/mhs/report/schema.py`` imports ``_jsonable`` from
  ``src.mhs.report.artifacts`` rather than via ``evaluation.py``;
* every public name previously importable from ``src.mhs.execution`` remains
  importable;
* SCENARIO_MHS_PERF_P4_02_TEST_FILE_SIZE_BUDGET: no file under ``tests/``
  exceeds 60 KB (the giant-test-file split of docs/specs/mhs_perf_refactor.md
  §7.3).
"""

from __future__ import annotations

import ast
from pathlib import Path

EVALUATION = Path("src/application/research/mhs/evaluation.py")
STAGES_DIR = Path("src/mhs/pipeline/stages")
SCHEMA = Path("src/mhs/report/schema.py")
ARTIFACTS = Path("src/mhs/report/artifacts.py")


def _import_targets(tree: ast.Module) -> list[tuple[str, bool]]:
    """(module, is_module_level) for every Import/ImportFrom in ``tree``."""
    targets: list[tuple[str, bool]] = []
    top_level_lines = {node.lineno for node in tree.body}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(
                (alias.name, node.lineno in top_level_lines) for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets.append((node.module, node.lineno in top_level_lines))
    return targets


def test_evaluation_has_no_module_level_pipeline_imports() -> None:
    """The application layer never depends on the pipeline at module scope."""
    tree = ast.parse(EVALUATION.read_text(encoding="utf-8"))
    violations = [
        module
        for module, is_module_level in _import_targets(tree)
        if is_module_level and module.startswith("src.mhs.pipeline")
    ]
    assert violations == []


def test_evaluation_function_scoped_cycle_breakers_are_pinned() -> None:
    """Exactly the two documented function-scoped cycle-breaker imports exist."""
    tree = ast.parse(EVALUATION.read_text(encoding="utf-8"))
    scoped = [
        module
        for module, is_module_level in _import_targets(tree)
        if not is_module_level and module.startswith("src.mhs.pipeline")
    ]
    assert sorted(scoped) == [
        "src.mhs.pipeline.context",
        "src.mhs.pipeline.runner",
    ]


def _evaluation_aliases(tree: ast.Module) -> set[str]:
    """Local names bound to the ``evaluation`` module object itself."""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.application.research.mhs":
            aliases.update(
                (alias.asname or alias.name)
                for alias in node.names
                if alias.name == "evaluation"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "src.application.research.mhs.evaluation":
                    aliases.add(alias.asname or alias.name)
    return aliases


def test_no_stage_imports_private_symbols_from_evaluation() -> None:
    """Stage modules consume private helpers via the stage_services seam only.

    Both coupling forms are checked: ``from evaluation import _foo`` (an
    ``ImportFrom`` a naive scanner would catch) and ``_evaluation._foo(...)``
    (attribute access on an evaluation-aliased module object, which a scanner
    restricted to ``ImportFrom`` names would miss entirely).
    """
    offenders: list[str] = []
    for path in sorted(STAGES_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "src.application.research.mhs.evaluation"
            ):
                offenders.extend(
                    f"{path.name}:{alias.name}"
                    for alias in node.names
                    if alias.name.startswith("_")
                )
        aliases = _evaluation_aliases(tree)
        offenders.extend(
            f"{path.name}:{node.value.id}.{node.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr.startswith("_")
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
        )
    assert offenders == []


def test_schema_imports_jsonable_from_artifacts_not_evaluation() -> None:
    """schema.py -> report.artifacts._jsonable directly (no evaluation detour)."""
    schema_tree = ast.parse(SCHEMA.read_text(encoding="utf-8"))
    evaluation_jsonable = [
        node.lineno
        for node in ast.walk(schema_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "src.application.research.mhs.evaluation"
        and any(alias.name == "_jsonable" for alias in node.names)
    ]
    artifacts_jsonable = [
        node.lineno
        for node in ast.walk(schema_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "src.mhs.report.artifacts"
        and any(alias.name == "_jsonable" for alias in node.names)
    ]
    assert evaluation_jsonable == []
    assert artifacts_jsonable, "schema must import _jsonable from report.artifacts"
    # ...and the artifact module actually defines it.
    artifact_tree = ast.parse(ARTIFACTS.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "_jsonable"
        for node in artifact_tree.body
    )


def test_execution_public_import_surface_stable() -> None:
    """Every public name historically importable from src.mhs.execution still is."""
    import importlib

    module = importlib.import_module("src.mhs.execution")
    expected_public = (
        "ExecutionDataGap",
        "ExecutionReplayWindow",
        "ExecutionSpec",
        "ForwardExecutionObservation",
        "IsolatedBoundFailure",
        "SimulatedInventoryLedgerResult",
        "StrategyExecutionReplayResult",
        "BatchReplayOutcome",
        "bar_funding_panel",
        "laddered_fill_schedule",
        "mhs_ledger_pnl",
        "mhs_ledger_pnl_multi_tier",
        "notional_weighted_shortfall_bps",
        "passive_fill_shortfall_bps",
        "replay_execution_window_batch",
        "replay_execution_window_batch_isolated",
        "replay_execution_window_pair",
        "replay_execution_windows",
        "replay_execution_windows_coupled",
        "ruin_guard_equity",
        "simulated_inventory_ledger",
        "strategy_aware_execution_replay",
    )
    missing = [name for name in expected_public if not hasattr(module, name)]
    assert missing == []


def test_stage_services_seam_reexports_match_stage_usage() -> None:
    """Every symbol stages import from the seam actually exists on it.

    ``stage_services`` re-exports the SAME function objects evaluation
    defines (an identity re-export, pinned by
    ``tests/unit/application/research/mhs/test_stage_services.py``'s ``is``
    check) so existing ``monkeypatch.setattr(evaluation, "_foo", ...)`` call
    sites keep working whenever a stage's own import binds first; test call
    sites that need the injection to hold regardless of import order patch
    both modules explicitly.
    """
    seam_path = Path("src/application/research/mhs/stage_services.py")
    seam_tree = ast.parse(seam_path.read_text(encoding="utf-8"))
    seam_names: set[str] = set()
    for node in seam_tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "src.application.research.mhs.evaluation"
        ):
            seam_names.update(alias.name for alias in node.names)

    used: set[str] = set()
    for path in sorted(STAGES_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "src.application.research.mhs.stage_services"
            ):
                used.update(alias.name for alias in node.names)
    assert used <= seam_names, f"seam missing: {sorted(used - seam_names)}"


def test_file_size_budget() -> None:
    """SCENARIO_MHS_PERF_P4_02_TEST_FILE_SIZE_BUDGET: no file under ``tests/``
    exceeds 60 KB, so an AI changing one behavior never has to read a
    hundreds-of-KB test module for unrelated context."""
    budget_bytes = 60 * 1024
    offenders = [
        str(path)
        for path in Path("tests").rglob("*.py")
        if "__pycache__" not in path.parts and path.stat().st_size > budget_bytes
    ]
    assert offenders == [], f"files over the {budget_bytes}-byte test budget: {offenders}"
