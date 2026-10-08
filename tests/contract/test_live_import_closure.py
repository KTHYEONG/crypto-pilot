"""Live daemon import-closure contract: research code must never reach the daemon."""

from __future__ import annotations

import ast
import fnmatch
from collections import deque
from pathlib import Path

import pytest

_LIVE_DENYLIST: tuple[str, ...] = (
    "src.mhs.deploy_gate",
    "src.mhs.research_go",
    "src.mhs.preregistration",
    "src.mhs.backtest*",
    "src.mhs.discovery",
    "src.mhs.run_history",
    "src.mhs.evidence",
    "src.mhs.validation",
    "src.mhs.process*",
    "src.mhs.pipeline*",
    "src.mhs.evaluation*",
    "src.mhs.report*",
    "src.mhs.reporting*",
    "src.mhs.contracts",
    "src.engine.strategy_backtest",
    "src.engine.backtest_evidence",
    "src.engine.backtest_persist",
    "src.engine.backtest_windows",
    "src.engine.execution*",
    "src.engine.account_sources",
    "src.engine.account_ledger",
    "src.engine.daily_evidence",
    "src.evaluation*",
)


def _module_name(path: Path, anchor: Path) -> str | None:
    try:
        rel = path.relative_to(anchor)
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if not parts or parts[0] != "src":
        return None
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _resolve(base: str, modules: set[str]) -> str | None:
    name = base
    while name:
        if name in modules:
            return name
        name = name.rpartition(".")[0] if "." in name else ""
    return None


def _import_graph(repo_root: Path) -> tuple[dict[str, set[str]], set[str]]:
    anchor = Path(repo_root)
    files: dict[str, Path] = {}
    for path in sorted((anchor / "src").rglob("*.py")):
        name = _module_name(path, anchor)
        if name is not None:
            files[name] = path
    modules = set(files)
    graph: dict[str, set[str]] = {name: set() for name in modules}
    for name, path in files.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        is_pkg = path.name == "__init__.py"
        package = name if is_pkg else name.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    pkg = package
                    for _ in range(node.level - 1):
                        pkg = pkg.rpartition(".")[0]
                    base = pkg + ("." + node.module if node.module else "")
                else:
                    base = node.module or ""
                base_hit = _resolve(base, modules) if base else None
                if base_hit is not None:
                    graph[name].add(base_hit)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    hit = _resolve(f"{base}.{alias.name}", modules) if base else None
                    if hit is not None:
                        graph[name].add(hit)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    hit = _resolve(alias.name, modules)
                    if hit is not None:
                        graph[name].add(hit)
        parts = name.split(".")
        for depth in range(1, len(parts)):
            parent = ".".join(parts[:depth])
            if parent in modules:
                graph[name].add(parent)
    return graph, modules


def live_import_closure(repo_root: Path) -> frozenset[str]:
    """Modules reachable from the live daemon by static import analysis, including function-level imports. Roots are every module under ``src/live`` and ``src/cli/commands/live.py``. The live process must never load research or evaluation machinery: a research edit must not be able to change, slow, or break the daemon."""
    anchor = Path(repo_root)
    graph, modules = _import_graph(anchor)
    roots = [m for m in modules if m == "src.cli.commands.live" or m == "src.live" or m.startswith("src.live.")]
    assert roots, f"no live import roots found under {repo_root}"
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        current = stack.pop()
        if current in seen or current not in graph:
            continue
        seen.add(current)
        stack.extend(graph[current])
    return frozenset(seen)


def _matches(pattern: str, module: str) -> bool:
    return fnmatch.fnmatchcase(module, pattern) or (
        not pattern.endswith("*") and module.startswith(pattern + ".")
    )


def _shortest_paths(
    repo_root: Path, targets: set[str],
) -> dict[str, str]:
    graph, modules = _import_graph(Path(repo_root))
    roots = sorted(m for m in modules if m == "src.cli.commands.live" or m == "src.live" or m.startswith("src.live."))
    prev: dict[str, str | None] = {root: None for root in roots if root in graph}
    queue: deque[str] = deque(prev)
    while queue:
        current = queue.popleft()
        for dep in sorted(graph.get(current, ())):
            if dep not in prev:
                prev[dep] = current
                queue.append(dep)
    paths: dict[str, str] = {}
    for target in targets:
        if target not in prev:
            paths[target] = f"{target} <= (unreachable from roots)"
            continue
        chain = [target]
        parent = prev[target]
        while parent is not None:
            chain.append(parent)
            parent = prev[parent]
        paths[target] = f"{target} <= {' -> '.join(reversed(chain))}"
    return paths


def _offenders(closure: frozenset[str]) -> set[str]:
    return {
        module for module in closure
        if any(_matches(pattern, module) for pattern in _LIVE_DENYLIST)
    }


def test_live_closure_excludes_research_machinery() -> None:
    closure = live_import_closure(Path("."))
    bad = _offenders(closure)
    assert bad == set(), (
        "live daemon reaches research/evaluation machinery:\n"
        + "\n".join(sorted(_shortest_paths(Path("."), bad).values()))
    )


def test_closure_helper_sees_lazy_imports(tmp_path: Path) -> None:
    src = tmp_path / "src" / "live"
    src.mkdir(parents=True)
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "root.py").write_text(
        "def build():\n    import src.lazy_mod\n    return src.lazy_mod.VALUE\n",
        encoding="utf-8",
    )
    (src.parent / "lazy_mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert "src.lazy_mod" in live_import_closure(tmp_path)


def test_closure_helper_sees_type_checking_imports(tmp_path: Path) -> None:
    src = tmp_path / "src" / "live"
    src.mkdir(parents=True)
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "root.py").write_text(
        "from __future__ import annotations\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from src.anno_types import Alias\n"
        "def build() -> Alias:\n    raise NotImplementedError\n",
        encoding="utf-8",
    )
    (src.parent / "anno_types.py").write_text("Alias = int\n", encoding="utf-8")
    assert "src.anno_types" in live_import_closure(tmp_path)


@pytest.mark.parametrize("statement", [
    "import src.mhs.bridge",
    "from src.mhs import bridge",
    "from ..mhs import bridge",
    "from src.mhs.bridge import *",
])
def test_closure_resolves_import_forms_and_parent_packages(tmp_path: Path, statement: str) -> None:
    live = tmp_path / "src" / "live"
    mhs = tmp_path / "src" / "mhs"
    live.mkdir(parents=True)
    mhs.mkdir()
    (live / "root.py").write_text(statement + "\n", encoding="utf-8")
    (mhs / "__init__.py").write_text("from . import validation\n", encoding="utf-8")
    (mhs / "bridge.py").write_text("from . import evidence\n", encoding="utf-8")
    (mhs / "evidence.py").write_text("VALUE = 1\n", encoding="utf-8")
    (mhs / "validation.py").write_text("VALUE = 2\n", encoding="utf-8")
    closure = live_import_closure(tmp_path)
    assert {"src.mhs", "src.mhs.bridge", "src.mhs.evidence", "src.mhs.validation"} <= closure
    assert _offenders(closure) == {"src.mhs.evidence", "src.mhs.validation"}
    assert _shortest_paths(tmp_path, {"src.mhs.evidence"})["src.mhs.evidence"] == (
        "src.mhs.evidence <= src.live.root -> src.mhs.bridge -> src.mhs.evidence"
    )


def test_closure_rejects_unparseable_source(tmp_path: Path) -> None:
    live = tmp_path / "src" / "live"
    live.mkdir(parents=True)
    (live / "root.py").write_text("def broken(\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        live_import_closure(tmp_path)


def test_closure_rejects_missing_roots(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "liveness.py").write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="no live import roots"):
        live_import_closure(tmp_path)
