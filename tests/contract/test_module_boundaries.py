"""SCENARIO_MHS_PERF_P4_01_DEPENDENCY_DIRECTION: module boundary contract."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

STAGES_DIR = Path("src/mhs/pipeline/stages")
SCHEMA = Path("src/mhs/report/schema.py")
ARTIFACTS = Path("src/mhs/report/artifacts.py")


def _assert_frozen_entries_live(
    frozen: Mapping[str, int], measured: Mapping[str, int], default_budget: int, label: str
) -> None:
    """Fail when a frozen budget exemption no longer exempts anything.

    Each frozen key must still be present in ``measured`` and its measured size
    must still exceed ``default_budget``; otherwise the exemption is dead weight
    that would silently re-admit growth up to a stale ceiling. ``label`` names
    the budget in the assertion message.
    """
    stale = sorted(
        key
        for key in frozen
        if key not in measured or measured[key] <= default_budget
    )
    assert stale == [], f"stale frozen entries in {label} (shrink or delete): {stale}"


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
    # frozen at measured size; growth fails, shrink requires deleting/lowering the entry.
    frozen_oversized = {
        "tests/unit/live/test_scheduler.py": 170554,  # spec 17: mainnet refuse / testnet warn gate tests.
        "tests/unit/live/test_runner_shadow_cycle.py": 99508,
        "tests/unit/live/test_runner_ledger.py": 63891,
        "tests/unit/live/test_executor.py": 193626,
        "tests/unit/live/test_data_refresh.py": 71709,
        "tests/unit/mhs/test_process_backtest.py": 140056,
        "tests/unit/cli/commands/test_backtest.py": 81671,
        "tests/unit/mhs/evaluation/test_windows.py": 87424,
        # Existing fold replay and shared-reference regression suite; freeze its current size.
        "tests/unit/mhs/test_evaluation_folds.py": 62668,
    }
    measured = {
        str(path): path.stat().st_size
        for path in Path("tests").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    offenders = [
        str(path)
        for path in Path("tests").rglob("*.py")
        if "__pycache__" not in path.parts
        and path.stat().st_size > frozen_oversized.get(str(path), budget_bytes)
    ]
    assert offenders == [], f"files over the {budget_bytes}-byte test budget: {offenders}"
    _assert_frozen_entries_live(frozen_oversized, measured, budget_bytes, "test file size budget")


def test_baseline_regression_gates_are_green() -> None:
    """Marker: G2-G7 are covered by existing contract tests, not new ones.

    The real gate is the execution_command. This assertion only pins the
    files that must exist and stay green.
    """
    from pathlib import Path

    gates = [
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
    # Submodules in evaluation package (P2 split); callers import concrete component owners
    wanted.difference_update({"books", "committee", "concurrency", "fold_weights", "folds"})
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
    # frozen at measured size; growth fails, shrink requires deleting/lowering the entry.
    allowlist: dict[str, int] = {
        "src/mhs/evaluation/windows.py": 712,
        # Fold validation, shared train-reference reuse and fork scheduling form one lifecycle.
        "src/mhs/evaluation/folds.py": 1262,
    }
    measured = {
        str(path): len(path.read_text(encoding="utf-8").splitlines())
        for path in Path("src/mhs/evaluation").rglob("*.py")
    }
    offenders = {
        key: lines for key, lines in measured.items() if lines > allowlist.get(key, budget)
    }
    assert offenders == {}, f"modules over {budget} lines: {offenders}"
    _assert_frozen_entries_live(allowlist, measured, budget, "evaluation module size budget")


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


def test_inventory_oracle_stays_out_of_production() -> None:
    """The single-panel ledger oracle must never be imported by ``src/``.

    It exists to certify the streamed accumulator ledger. A production import
    would let the thing being checked become the source it is checked against,
    so only the facade re-export (kept for test imports) is allowed.
    """
    import ast
    from pathlib import Path

    facade = Path("src/mhs/execution/__init__.py")
    oracle = "simulated_inventory_ledger"
    offenders: list[tuple[str, int]] = []
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                imported = [alias.asname or alias.name.split(".")[0] for alias in node.names]
            else:
                modules = [node.module or ""]
                imported = [alias.name for alias in node.names]
                if node.level:
                    base = "src.mhs.execution" if path.parent.name == "execution" else ""
                    modules = [f"{base}{'.' * node.level}{modules[0]}"]
            reaches_ledger = any(
                module == "src.mhs.execution.ledger"
                or (path.parent == facade.parent and module == "ledger")
                for module in modules
            )
            if path != facade and (reaches_ledger or oracle in imported):
                offenders.append((str(path), node.lineno))

    assert offenders == [], (
        f"production modules import the inventory oracle {oracle!r}: {sorted(offenders)}"
    )


def test_no_method_exceeds_length_budget() -> None:
    """A 700-line method is unreadable; consume() must stay decomposed.

    Scoped to accumulator.py, the sole module this phase authorizes
    decomposing methods in.
    """
    import ast
    from pathlib import Path

    budget = 260
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
    """One documented exemption: the cohesive stateful accumulator class.

    The state machine stays in one class by design.
    """
    from pathlib import Path

    default_budget = 700
    # frozen at measured size; growth fails, shrink requires deleting/lowering the entry.
    allowlist = {"src/mhs/execution/accumulator.py": 1897}

    measured = {
        str(path): len(path.read_text(encoding="utf-8").splitlines())
        for path in Path("src/mhs/execution").rglob("*.py")
    }
    offenders: dict[str, int] = {
        key: lines for key, lines in measured.items() if lines > allowlist.get(key, default_budget)
    }

    assert offenders == {}, f"modules over budget: {offenders}"
    assert not Path("src/mhs/execution.py").exists(), "monolith must be gone"
    _assert_frozen_entries_live(allowlist, measured, default_budget, "execution module size budget")


def test_source_module_size_budget() -> None:
    """Standing guard against monolith regrowth (see ADR_20260902)."""
    from pathlib import Path

    default_budget = 700
    # frozen at measured size; growth fails, shrink requires deleting/lowering the entry.
    allowlist = {
        "src/mhs/execution/accumulator.py": 1897,
        # Cycle phases stay co-located to preserve runtime module-global test seams;
        # explicit phase contracts add lines while reducing orchestration complexity.
        # spec 17: venue snapshot passthrough, disabled-collection audit, genesis alert.
        "src/live/runner.py": 1916,
        # spec 17: mainnet refuse-to-start gate (fail loud before any venue call).
        "src/live/scheduler.py": 1007,
        # Shared post/unknown-submission primitives retain the executor lifecycle and contracts.
        "src/live/executor.py": 1980,
        "src/live/tax_ledger.py": 1209,
        "src/live/rest.py": 815,
        "src/mhs/resources.py": 917,
        "src/mhs/evidence.py": 1267,
        "src/mhs/deploy_gate.py": 723,
        "src/mhs/scaling.py": 892,
        "src/application/mhs_supervisor.py": 1224,
        "src/cli/commands/backtest.py": 1133,
        "src/mhs/reporting/inventory.py": 741,
        "src/mhs/backtest/paths.py": 849,
        "src/mhs/backtest/journal.py": 1091,
        "src/mhs/backtest/inventory.py": 873,
        "src/mhs/evaluation/windows.py": 712,
        # Unified MhsDiagnosticRequest carries per-field CLI/validation metadata as the single schema source (spec 10 parts 2-3).
        # Freeze the existing fold lifecycle; further growth requires decomposition.
        "src/mhs/evaluation/folds.py": 1262,
        # Declare-once request schema: each MHS option is exactly one field plus CLI metadata.
        "src/mhs/contracts.py": 898,
        # Checkpoint advancement, retention and the loop stay co-located for review;
        # _run_retention_pass persists the checkpoint and is not split out.
        "src/market_data/streams/normalizer.py": 1112,
        # LiveSettings resolved path accessors (spec 20) stay on the settings model
        # so call-site default seams keep resolving lazily at call time.
        # spec 17: live modes skip run-scoped tax ledger derivation (account-scoped venue ledger).
        "src/live/settings.py": 751,
    }
    measured = {
        str(path): len(path.read_text(encoding="utf-8").splitlines())
        for path in Path("src").rglob("*.py")
    }
    offenders: dict[str, int] = {
        key: lines for key, lines in measured.items() if lines > allowlist.get(key, default_budget)
    }

    assert offenders == {}, (
        f"modules over budget: {offenders}. Split it, or add a documented "
        f"allowlist entry with a stated reason."
    )
    _assert_frozen_entries_live(allowlist, measured, default_budget, "source module size budget")


def test_no_import_cycles_between_packages() -> None:
    """Standing guard: no NEW cycle may appear at package granularity."""
    import ast
    from collections import defaultdict
    from pathlib import Path

    # P5 amendment (ADR_20260902): parent/child edges are facade re-exports,
    # not cycles. Sanctioned pairs pre-date P5 or represent intentional handoff
    # boundaries (live handoff, market data collection, backtest evidence).
    sanctioned = {
        frozenset({"live", "mhs"}),
        frozenset({"mhs", "market_data.services"}),
        frozenset({"backtests", "mhs"}),
        frozenset({"live", "market_data.streams"}),
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
    # frozen at measured size; growth fails, shrink requires deleting/lowering the entry.
    frozen = {
        "src/live/runner.py::run_shadow_cycle": 348,
        # spec 17: mainnet refuse-to-start gate (fail loud before any venue call).
        "src/live/scheduler.py::run_daemon": 354,
        "src/live/frozen_signal.py::run_frozen_signal_step": 312,
        "src/mhs/account_ledger.py::replay_account": 308,
        "src/cli/commands/backtest.py::run_frozen_account_command": 284,
        "src/mhs/evaluation/windows.py::_book_outcome": 368,
        "src/mhs/execution/accumulator.py::_consume_append_ledger": 252,
        "src/mhs/execution/window_stream.py::_iter_mhs_execution_windows": 382,
        "src/mhs/backtest/paths.py::run_process_paths": 261,
        "src/mhs/backtest/inventory.py::evaluate_process_inventory_backtest": 289,
        "src/mhs/discovery.py::select_horizon_by_discovery_qualification": 270,
        "src/mhs/evaluation/committee.py::_committee_diagnostic": 282,
        "src/mhs/pipeline/stages/committee.py::build_committee": 278,
    }
    measured: dict[str, int] = {}
    offenders: dict[str, int] = {}
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            span = (node.end_lineno or node.lineno) - node.lineno
            key = f"{path}::{node.name}"
            measured[key] = max(measured.get(key, 0), span)
            if span > frozen.get(key, budget):
                offenders[key] = span

    assert offenders == {}, f"functions over {budget} lines: {offenders}"
    _assert_frozen_entries_live(frozen, measured, budget, "function length budget")


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
    """P1/P4 removed legacy/, src/core/; nothing may reintroduce them.

    Standing guard consolidated from the retired throwaway
    tests/contract/test_refactor_p1.py and test_refactor_p4.py.
    """
    import ast
    from pathlib import Path

    for name in ("legacy", "src/core"):
        assert not Path(name).exists(), f"deleted tree reappeared: {name}"

    stale_prefixes = ("legacy", "src.core")
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


def test_frozen_entries_live_rejects_missing_key() -> None:
    """Live-entry helper rejects a frozen key absent from the measurement."""
    import pytest

    with pytest.raises(AssertionError, match=r"a\.py::f"):
        _assert_frozen_entries_live({"a.py::f": 300}, {}, 250, "function length budget")


def test_frozen_entries_live_rejects_compliant_key() -> None:
    """Live-entry helper rejects a frozen key that no longer exceeds the budget."""
    import pytest

    with pytest.raises(AssertionError, match=r"a\.py"):
        _assert_frozen_entries_live({"a.py": 900}, {"a.py": 650}, 700, "source module size budget")


def test_frozen_entries_live_accepts_oversized_key() -> None:
    """Live-entry helper accepts a frozen key that still exceeds the budget."""
    _assert_frozen_entries_live({"a.py": 900}, {"a.py": 850}, 700, "source module size budget")
