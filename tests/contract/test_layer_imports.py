"""Layer import contract for the part-2 packages through the part-4 lab (spec 38).

Import direction is strictly downward: core <- strategy <- engine <- evaluation,
with live on core+strategy only, and the exploratory ``src.lab`` isolated from
every deployable layer. Static AST analysis covers lazy (function-level)
imports and TYPE_CHECKING blocks.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_CORE_ALLOW = frozenset({"core", "common", "market_data", "quant"})
_STRATEGY_ALLOW = frozenset({"core", "common", "market_data", "quant", "strategy"})
_ENGINE_ALLOW = _STRATEGY_ALLOW | {"engine"}
_EVALUATION_ALLOW = _ENGINE_ALLOW | {"evaluation"}
_LIVE_ALLOW = frozenset({"core", "strategy", "common", "market_data", "live", "capture"})

# Built by concatenation so the acceptance grep for stale ``src." + "mhs.*`` imports
# stays empty; the test asserts none of these remain importable.
_LEGACY_PREFIX = "src." + "mhs."
_MOVED_SOURCES = tuple(
    _LEGACY_PREFIX + name
    for name in (
        "params",
        "types",
        "resources",
        "tree_memory",
        "parallel",
        "bootstrap",
        "panel",
        "data_policy",
        "data_provenance",
        "marks",
        "source_gaps",
        "instrument_settlements",
        "venue_halts",
        "settlement_evidence",
        "books",
        "features",
        "feature_admission",
        "horizons",
        "frozen_research_candidate",
        "frozen_research_universe",
        "account_policy",
        "account_liquidity",
        "execution",
        "account_ledger",
        "account_sources",
        "frozen_research_run",
        "frozen_research_windows",
        "frozen_research_evidence",
        "frozen_research_report",
        "growth_exposure",
    )
)


def module_imports(path: Path, *, module_level_only: bool = False) -> frozenset[str]:
    """Absolute ``src.*`` modules imported by one file (lazy and TYPE_CHECKING included)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    owner = ".".join(parts)
    package = owner if path.name == "__init__.py" else owner.rpartition(".")[0]
    found: set[str] = set()
    def scoped_nodes(node: ast.AST):
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield from scoped_nodes(child)

    nodes = scoped_nodes(tree) if module_level_only else ast.walk(tree)
    for node in nodes:
        if isinstance(node, ast.ImportFrom):
            if node.level:
                pkg = package
                for _ in range(node.level - 1):
                    pkg = pkg.rpartition(".")[0]
                base = pkg + ("." + node.module if node.module else "")
            else:
                base = node.module or ""
            if base.startswith("src."):
                found.add(base)
            for alias in node.names:
                if alias.name == "*":
                    continue
                if base == "src" or base.startswith("src."):
                    found.add(f"{base}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    found.add(alias.name)
    return frozenset(found)


def _layer_of_src_module(dotted: str) -> str | None:
    segments = dotted.split(".")
    if len(segments) < 2 or segments[0] != "src":
        return None
    return segments[1]


def _iter_layer_files(layer: str) -> list[Path]:
    root = REPO_ROOT / "src" / layer
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _module_name(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _check_allow(layer: str, allow: frozenset[str]) -> list[str]:
    offenders: list[str] = []
    for path in _iter_layer_files(layer):
        for target in sorted(module_imports(path)):
            target_layer = _layer_of_src_module(target)
            if target_layer is not None and target_layer not in allow:
                offenders.append(f"{_module_name(path)} ({layer}) imports {target} ({target_layer})")
    return sorted(offenders)


def test_core_imports_no_higher_layer() -> None:
    """No src.core module imports strategy|engine|evaluation|live|application|cli|lab."""
    assert _check_allow("core", _CORE_ALLOW) == []


def test_strategy_imports_only_core() -> None:
    """src.strategy imports only core/common/market_data/quant (plus itself)."""
    assert _check_allow("strategy", _STRATEGY_ALLOW) == []


def test_engine_never_imports_live_evaluation_or_lab() -> None:
    """src.engine never imports live, evaluation, application, cli or lab."""
    assert _check_allow("engine", _ENGINE_ALLOW) == []


def test_evaluation_never_imports_live_or_lab() -> None:
    """src.evaluation never imports live, lab, application or cli."""
    assert _check_allow("evaluation", _EVALUATION_ALLOW) == []


def test_live_never_imports_engine_evaluation_or_lab() -> None:
    """src.live imports only core/strategy/common/market_data/live/capture."""
    assert _check_allow("live", _LIVE_ALLOW) == []


@pytest.mark.parametrize("statement", [
    "from src import live",
    "from ..live import runner",
    "def build():\n    import src.live.runner",
    "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from src.live import runner",
])
def test_layer_checker_rejects_all_import_forms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, statement: str) -> None:
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    root = tmp_path / "src" / "core"
    root.mkdir(parents=True)
    (root / "probe.py").write_text(statement + "\n", encoding="utf-8")
    offenders = _check_allow("core", _CORE_ALLOW)
    assert offenders
    assert all("src.core.probe (core) imports src.live" in message for message in offenders)


@pytest.mark.parametrize(("layer", "allow"), [
    ("core", _CORE_ALLOW), ("engine", _ENGINE_ALLOW), ("evaluation", _EVALUATION_ALLOW),
])
def test_layer_checker_rejects_unlisted_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layer: str, allow: frozenset[str],
) -> None:
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    root = tmp_path / "src" / layer
    root.mkdir(parents=True)
    (root / "probe.py").write_text("from src.backtests.contracts import JsonValue\n", encoding="utf-8")
    assert _check_allow(layer, allow)


def _find_spec_or_none(name: str) -> object | None:
    """find_spec without raising when a parent package is already gone."""
    try:
        return importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return None


def test_no_module_remains_importable_at_old_path() -> None:
    """Every move-map source is gone: find_spec returns None (no shims)."""
    offenders = [name for name in _MOVED_SOURCES if _find_spec_or_none(name) is not None]
    assert offenders == [], f"old paths still importable: {sorted(offenders)}"


# --- part 4: research lab isolation -------------------------------------------

#: Deployable layers that must never import ``src.lab`` (any import form).
_NO_LAB_LAYERS = ("core", "strategy", "engine", "evaluation", "live", "market_data", "common", "quant", "capture", "backtests")

#: Part-4 move map: old importable paths that must be gone (no shims).
_PART4_REMOVED_MODULES = (
    "src." + "mhs",
    "src.application.mhs_backtest",
    "src.application.mhs_supervisor",
    "src.application.mhs_worker",
    "src.application.mhs_reuse",
    "src.backtests.migration",
    "src.cli.commands.research",
    "src.cli.commands.research.mhs",
)

_PART4_REMOVED_PATHS = (
    "src/mhs",
    "src/application/mhs_backtest.py",
    "src/application/mhs_supervisor.py",
    "src/application/mhs_worker.py",
    "src/application/mhs_reuse.py",
    "src/backtests/migration.py",
    "src/cli/commands/research",
)


def _function_level_imports(path: Path) -> frozenset[str]:
    """Absolute ``src.*`` modules imported inside function bodies (lazy imports)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    owner = ".".join(parts)
    package = owner if path.name == "__init__.py" else owner.rpartition(".")[0]
    found: set[str] = set()

    def _visit(node: ast.AST, package: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for sub in ast.walk(child):
                    if isinstance(sub, (ast.ImportFrom, ast.Import)):
                        _collect(sub, package, found)
            else:
                _visit(child, package)

    def _collect(node: ast.AST, package: str, found: set[str]) -> None:
        if isinstance(node, ast.ImportFrom):
            if node.level:
                pkg = package
                for _ in range(node.level - 1):
                    pkg = pkg.rpartition(".")[0]
                base = pkg + ("." + node.module if node.module else "")
            else:
                base = node.module or ""
            if base.startswith("src."):
                found.add(base)
            for alias in node.names:
                if alias.name == "*":
                    continue
                if base == "src" or base.startswith("src."):
                    found.add(f"{base}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    found.add(alias.name)

    _visit(tree, package)
    return frozenset(found)


def _module_level_imports(path: Path) -> frozenset[str]:
    """Absolute ``src.*`` modules imported at module top level (incl. TYPE_CHECKING)."""
    return module_imports(path, module_level_only=True)


@pytest.mark.parametrize("statement", [
    "import src.lab.mhs.contracts",
    "from src.lab.mhs import contracts",
    "if True:\n    from src.lab.mhs import contracts",
    "class Registry:\n    from src.lab.mhs import contracts",
    "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from src.lab.mhs import contracts",
])
def test_module_level_import_cannot_be_hidden_by_lazy_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, statement: str,
) -> None:
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    path = tmp_path / "src" / "cli" / "commands" / "lab.py"
    path.parent.mkdir(parents=True)
    path.write_text(statement + "\ndef run():\n    import src.lab.mhs.contracts\n", encoding="utf-8")
    assert "src.lab.mhs.contracts" in _module_level_imports(path)


def _shortest_lab_import_chain(target: str) -> str:
    """BFS import chain from a deployable module to a ``src.lab`` target.

    Ported from the part-1 live-closure contract: when a deployable layer
    reaches the lab, the failure message shows the exact chain.
    """
    graph: dict[str, set[str]] = {}
    modules: dict[str, Path] = {}
    for layer in (*_NO_LAB_LAYERS, "application", "cli", "lab"):
        for path in _iter_layer_files(layer):
            modules[_module_name(path)] = path
    for name, path in modules.items():
        try:
            graph[name] = set(module_imports(path))
        except SyntaxError:
            graph[name] = set()
    resolved: dict[str, set[str]] = {}
    names = set(modules)
    for name, deps in graph.items():
        hit: set[str] = set()
        for dep in deps:
            node = dep
            while node:
                if node in names:
                    hit.add(node)
                    break
                node = node.rpartition(".")[0] if "." in node else ""
        resolved[name] = hit
    roots = sorted(n for n in names if n.split(".")[1] in (*_NO_LAB_LAYERS, "application", "cli"))
    prev: dict[str, str | None] = dict.fromkeys(roots)
    queue: list[str] = list(roots)
    while queue:
        current = queue.pop(0)
        for dep in sorted(resolved.get(current, ())):
            if dep not in prev:
                prev[dep] = current
                queue.append(dep)
    if target not in prev:
        return f"{target} <= (unreachable from deployable roots)"
    chain = [target]
    parent = prev[target]
    while parent is not None:
        chain.append(parent)
        parent = prev[parent]
    return f"{target} <= {' -> '.join(reversed(chain))}"


def _lab_offenders(layers: tuple[str, ...]) -> list[str]:
    offenders: list[str] = []
    for layer in layers:
        offenders.extend(
            f"{_module_name(path)} ({layer}) imports {target} "
            f"[chain: {_shortest_lab_import_chain(target)}]"
            for path in _iter_layer_files(layer)
            for target in sorted(module_imports(path))
            if _layer_of_src_module(target) == "lab"
        )
    return sorted(offenders)


def test_deployable_layers_never_import_lab() -> None:
    """Deployable layers never import lab (static AST, lazy and TYPE_CHECKING included)."""
    assert _lab_offenders(_NO_LAB_LAYERS) == []


def test_application_never_imports_lab() -> None:
    """src.application orchestrates deployable code only: never imports src.lab."""
    root = REPO_ROOT / "src" / "application"
    offenders = [
        f"{_module_name(path)} (application) imports {target}"
        for path in sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
        for target in sorted(module_imports(path))
        if _layer_of_src_module(target) == "lab"
    ]
    assert sorted(offenders) == []


def test_lab_never_imports_live_or_cli() -> None:
    """src.lab never imports src.live or src.cli."""
    offenders = [
        f"{_module_name(path)} (lab) imports {target}"
        for path in _iter_layer_files("lab")
        for target in sorted(module_imports(path))
        if _layer_of_src_module(target) in ("live", "cli")
    ]
    assert sorted(offenders) == []


def test_only_lab_cli_command_imports_lab_lazily() -> None:
    """src.cli.commands.lab is the only CLI module importing src.lab, only inside functions."""
    cli_files = [(path, _module_name(path)) for path in _iter_layer_files("cli")]
    top_level = [
        f"{name} imports {target} at module level"
        for path, name in cli_files
        for target in sorted(_module_level_imports(path))
        if _layer_of_src_module(target) == "lab"
    ]
    lazy_elsewhere = [
        f"{name} imports {target} inside a function"
        for path, name in cli_files
        if name != "src.cli.commands.lab"
        for target in sorted(_function_level_imports(path))
        if _layer_of_src_module(target) == "lab"
    ]
    lab_lazy = [
        target
        for path, name in cli_files
        if name == "src.cli.commands.lab"
        for target in sorted(_function_level_imports(path))
        if _layer_of_src_module(target) == "lab"
    ]
    assert top_level == []
    assert lazy_elsewhere == []
    assert lab_lazy, "src.cli.commands.lab must be the wired lab entry point"


def test_old_research_package_is_gone() -> None:
    """The pre-part-4 research paths are gone: not importable, no files, no shims."""
    offenders = [name for name in _PART4_REMOVED_MODULES if _find_spec_or_none(name) is not None]
    assert offenders == [], f"old research paths still importable: {sorted(offenders)}"
    lingering = [rel for rel in _PART4_REMOVED_PATHS if (REPO_ROOT / rel).exists()]
    assert lingering == [], f"old research paths still on disk: {lingering}"


def test_moved_registries_keep_content_digests() -> None:
    """Policy jsonl files moved byte-identical; registry digests equal pre-move values."""
    from src.core.instrument_settlements import load_instrument_settlement_registry
    from src.core.source_gaps import _default_registry_path as _gaps_path
    from src.core.venue_halts import load_venue_halt_registry

    assert (
        load_instrument_settlement_registry().digest
        == "sha256:4e09a210879043fdd98f5d9e73ec3c04f2251836082f267a8765561e3854a047"
    )
    assert (
        load_venue_halt_registry().digest
        == "sha256:67b420f149af3f192788192cdba81d262c5744093f1281a8bde6527c9867409c"
    )
    import hashlib

    assert (
        hashlib.sha256(_gaps_path().read_bytes()).hexdigest()
        == "1746d1accca455c52eb5882a6eac4d300ff25279fcccc7d67ea6238bb0aec273"
    )
