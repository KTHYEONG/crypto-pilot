"""SCENARIO_MHS_PERF_P4_01_DEPENDENCY_DIRECTION: module boundary contract."""

from __future__ import annotations

import ast
from pathlib import Path

STAGES_DIR = Path("src/mhs/pipeline/stages")
SCHEMA = Path("src/mhs/report/schema.py")
ARTIFACTS = Path("src/mhs/report/artifacts.py")


def test_schema_imports_jsonable_from_artifacts_not_evaluation() -> None:
    """schema.py -> report.artifacts._jsonable directly (no evaluation detour)."""
    schema_tree = ast.parse(SCHEMA.read_text(encoding="utf-8"))
    evaluation_jsonable = [
        node.lineno
        for node in ast.walk(schema_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "src.mhs.evaluation"
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


def test_baseline_regression_gates_are_green() -> None:
    """Marker: G2-G7 are covered by existing contract tests, not new ones.

    The real gate is the execution_command. This assertion only pins the
    files that must exist and stay green.
    """
    from pathlib import Path

    gates = [
        "tests/contract/test_code_map.py",
        "tests/contract/test_param_single_source.py",
        "tests/contract/test_module_boundaries.py",
        "tests/contract/test_request_cli_parity.py",
        "tests/unit/test_deployment_assets.py",
    ]
    missing = [g for g in gates if not Path(g).exists()]
    assert missing == [], f"missing baseline gate files: {missing}"


def test_evaluation_package_has_no_pipeline_dependency() -> None:
    """Layer A never depends on the pipeline that consumes it."""
    import ast
    from pathlib import Path

    package = Path("src/mhs/evaluation")
    assert package.is_dir(), "evaluation must be a package after P2"

    offenders: list[str] = []
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            elif isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            if any(n.startswith("src.mhs.pipeline") for n in names):
                offenders.append(f"{path}:{node.lineno}")

    assert offenders == [], f"evaluation package imports pipeline: {offenders}"


def test_stage_services_seam_is_deleted() -> None:
    """The cycle-hiding seam must not survive in any form."""
    from pathlib import Path

    assert not Path(
        "src/mhs/stage_services.py"
    ).exists()
    assert not Path(
        "tests/unit/mhs/test_stage_services.py"
    ).exists()


def test_evaluation_facade_preserves_public_surface() -> None:
    """The split must not break a single existing import site."""
    import ast
    from pathlib import Path

    import src.mhs.evaluation as ev

    wanted: set[str] = set()
    for root in ("src", "tests", "tools"):
        for path in Path(root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "src.mhs.evaluation"
                ):
                    wanted.update(a.name for a in node.names)

    wanted.discard("run_mhs_horizon_diagnostic")  # moved to diagnostic_run (P2)
    missing = sorted(n for n in wanted if not hasattr(ev, n))
    assert missing == [], f"facade dropped names: {missing}"


def test_composition_root_owns_the_pipeline_edge() -> None:
    """Layer C imports the pipeline eagerly; no function-scoped seam remains."""
    import ast
    import inspect
    from pathlib import Path

    from src.mhs.diagnostic_run import (
        run_mhs_horizon_diagnostic,
    )

    assert callable(run_mhs_horizon_diagnostic)

    path = Path("src/mhs/diagnostic_run.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level_lines = {node.lineno for node in tree.body}
    pipeline_imports = [
        (node.module, node.lineno in top_level_lines)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("src.mhs.pipeline")
    ]
    assert pipeline_imports, "composition root must import the pipeline"
    assert all(is_top for _, is_top in pipeline_imports), (
        "pipeline imports must be module level, not function scoped"
    )
    assert inspect.isfunction(run_mhs_horizon_diagnostic)


def test_evaluation_modules_respect_size_budget() -> None:
    """No module in the split package may re-accrete into a monolith."""
    from pathlib import Path

    budget = 700
    offenders = {
        str(path): len(path.read_text(encoding="utf-8").splitlines())
        for path in Path("src/mhs/evaluation").rglob("*.py")
        if len(path.read_text(encoding="utf-8").splitlines()) > budget
    }
    assert offenders == {}, f"modules over {budget} lines: {offenders}"


def test_execution_public_surface_preserved() -> None:
    """The split must not break a single existing execution import site."""
    import ast
    from pathlib import Path

    import src.mhs.execution as execution

    wanted: set[str] = set()
    for root in ("src", "tests", "tools"):
        for path in Path(root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "src.mhs.execution"
                ):
                    wanted.update(a.name for a in node.names)

    missing = sorted(n for n in wanted if not hasattr(execution, n))
    assert missing == [], f"execution facade dropped names: {missing}"


def test_no_method_exceeds_length_budget() -> None:
    """A 700-line method is unreadable; consume() must stay decomposed.

    Scoped to accumulator.py, the sole module this phase authorizes
    decomposing methods in. Every other execution/ module is a verbatim
    move (target_layout) and may keep whatever length its original
    function had -- e.g. strategy_aware_execution_replay at 576 lines.
    """
    import ast
    from pathlib import Path

    budget = 250
    path = Path("src/mhs/execution/accumulator.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        span = (node.end_lineno or node.lineno) - node.lineno
        if span > budget:
            offenders[f"{path}::{node.name}"] = span

    assert offenders == {}, f"accumulator methods over {budget} lines: {offenders}"


def test_execution_module_size_budget_with_allowlist() -> None:
    """One documented exemption: the cohesive stateful accumulator class."""
    from pathlib import Path

    default_budget = 700
    allowlist = {"src/mhs/execution/accumulator.py": 1200}

    offenders: dict[str, int] = {}
    for path in Path("src/mhs/execution").rglob("*.py"):
        lines = len(path.read_text(encoding="utf-8").splitlines())
        budget = allowlist.get(str(path), default_budget)
        if lines > budget:
            offenders[str(path)] = lines

    assert offenders == {}, f"modules over budget: {offenders}"
    assert not Path("src/mhs/execution.py").exists(), "monolith must be gone"


def test_source_module_size_budget() -> None:
    """Standing guard against monolith regrowth (see ADR_20260902)."""
    from pathlib import Path

    default_budget = 700
    allowlist = {
        "src/mhs/execution/accumulator.py": 1200,  # P3: cohesive stateful accumulator
        # P5 amendment: pre-existing modules outside P2/P3 scope, frozen at
        # measured lines. Growth must split the module or re-justify the cap.
        "src/live/runner.py": 718,
        "src/live/executor.py": 894,
        "src/mhs/evidence.py": 1241,
        "src/mhs/scaling.py": 794,
        "src/market_data/services/futures_collection.py": 1190,
        "src/quant/technical_experts/cross_sectional.py": 1267,
        "src/quant/evaluation/reliability.py": 816,
        "src/mhs/report/persist.py": 780,
    }

    offenders: dict[str, int] = {}
    for path in Path("src").rglob("*.py"):
        lines = len(path.read_text(encoding="utf-8").splitlines())
        if lines > allowlist.get(str(path), default_budget):
            offenders[str(path)] = lines

    assert offenders == {}, (
        f"modules over budget: {offenders}. Split it, or add a documented "
        f"allowlist entry with a stated reason."
    )


def test_no_import_cycles_between_packages() -> None:
    """Standing guard: no NEW cycle may appear at package granularity."""
    import ast
    from collections import defaultdict
    from pathlib import Path

    # P5 amendment (ADR_20260902): parent/child edges are facade re-exports,
    # not cycles. The two sanctioned pairs pre-date P5 (live handoff, market
    # data collection) and are frozen; evaluation<->pipeline stays forbidden.
    sanctioned = {
        frozenset({"live", "mhs"}),
        frozenset({"mhs", "market_data.services"}),
    }

    edges: dict[str, set[str]] = defaultdict(set)
    for path in Path("src").rglob("*.py"):
        pkg = ".".join(path.relative_to("src").parts[:-1]) or "root"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
            elif isinstance(node, ast.Import):
                mods.extend(a.name for a in node.names)
            for mod in mods:
                if not mod.startswith("src."):
                    continue
                target = ".".join(mod.split(".")[1:-1]) or "root"
                if not target or target == pkg:
                    continue
                if target.startswith(pkg + ".") or pkg.startswith(target + "."):
                    continue  # facade re-export between parent and child
                edges[pkg].add(target)

    cycles: list[tuple[str, str]] = [
        (a, b)
        for a, deps in edges.items()
        for b in deps
        if a in edges.get(b, set()) and frozenset({a, b}) not in sanctioned
    ]
    assert cycles == [], f"package import cycles: {sorted(cycles)}"


def test_no_function_exceeds_length_budget() -> None:
    """Standing guard against unreviewable mega-functions."""
    import ast
    from pathlib import Path

    budget = 250
    # P5 amendment (ADR_20260902): pre-existing functions outside P3 scope
    # (I-P3-METHOD-BUDGET-SCOPE), frozen at measured spans. New code over
    # budget and any growth beyond a frozen span both fail.
    frozen = {
        "src/live/runner.py::run_shadow_cycle": 424,
        "src/mhs/discovery.py::select_horizon_by_discovery_qualification": 270,
        "src/cli/commands/research/mhs.py::add_mhs_commands": 567,
        "src/mhs/evaluation/committee.py::_committee_diagnostic": 282,
        "src/mhs/evaluation/windows.py::_book_outcome": 403,
        "src/mhs/execution/strategy_replay.py::strategy_aware_execution_replay": 576,
        "src/mhs/pipeline/stages/committee.py::build_committee": 291,
        "src/mhs/pipeline/stages/fold.py::run_folds": 251,
    }
    offenders: dict[str, int] = {}
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            span = (node.end_lineno or node.lineno) - node.lineno
            key = f"{path}::{node.name}"
            if span > frozen.get(key, budget):
                offenders[key] = span

    assert offenders == {}, f"functions over {budget} lines: {offenders}"


def test_docs_reference_no_ephemeral_spec_paths() -> None:
    """Specs are purged at sync; code must cite ADR ids instead."""
    from pathlib import Path

    offenders = [
        str(path)
        for path in Path("src").rglob("*.py")
        if "docs/specs/" in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"src cites ephemeral spec paths: {offenders}"


def test_architecture_docs_within_line_limit() -> None:
    """.agents/rules/documentation.md §4: 300-line ceiling per doc."""
    from pathlib import Path

    offenders = {
        str(path): len(path.read_text(encoding="utf-8").splitlines())
        for path in Path("docs/architecture").glob("*.md")
        if len(path.read_text(encoding="utf-8").splitlines()) > 300
    }
    assert offenders == {}, f"architecture docs over 300 lines: {offenders}"


def test_deleted_trees_stay_deleted() -> None:
    """P1/P4 removed legacy/, src/application/, src/core/; nothing may reintroduce them.

    Standing guard consolidated from the retired throwaway
    tests/contract/test_refactor_p1.py and test_refactor_p4.py.
    """
    import ast
    from pathlib import Path

    for name in ("legacy", "src/application", "src/core"):
        assert not Path(name).exists(), f"deleted tree reappeared: {name}"

    stale_prefixes = ("legacy", "src.application", "src.core")
    offenders: list[str] = []
    for root in ("src", "tests", "tools"):
        for path in Path(root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                mods: list[str] = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    mods.append(node.module)
                elif isinstance(node, ast.Import):
                    mods.extend(a.name for a in node.names)
                if any(
                    m == prefix or m.startswith(prefix + ".")
                    for m in mods
                    for prefix in stale_prefixes
                ):
                    offenders.append(str(path))
                    break

    assert offenders == [], f"modules still import a deleted tree: {offenders}"
