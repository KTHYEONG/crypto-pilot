"""Legacy isolation contract.

Verifies that no module under ``src/`` imports a ``legacy.*`` module, and
the AST import closure of active entry points contains no legacy modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

# Entry points whose import closure must be contained in the KEEP tree.
_KEEP_ENTRY_POINTS = (
    "src.cli.commands.research.mhs",
    "src.mhs",
    "src.cli.commands.data",
    "src.application.data",
    "src.core.settings",
)


def _src_imports(path: Path) -> set[str]:
    """Direct ``src.*`` module imports in one file (top-level + nested AST)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods.update(a.name for a in node.names if a.name.startswith("src."))
    return mods


def _module_of(path: Path) -> str:
    rel = path.relative_to(_ROOT)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts.pop()
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def _legacy_src_modules() -> set[str]:
    """Every ``legacy/src/**.py`` module path (the migrated-to-legacy set)."""
    mods: set[str] = set()
    root = _ROOT / "legacy" / "src"
    if not root.exists():
        return mods
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts.pop()
        else:
            parts[-1] = parts[-1].removesuffix(".py")
        mods.add(".".join(parts))
    return mods


def _keep_src_modules() -> set[str]:
    mods: set[str] = set()
    root = _ROOT / "src"
    for path in root.rglob("*.py"):
        mods.add(_module_of(path))
    return mods


def test_no_src_module_imports_legacy() -> None:
    """No module under ``src/`` imports a ``legacy.*`` module."""
    legacy_prefix = ("legacy.",)
    offenders: list[str] = []
    for path in (_ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(a.name for a in node.names)
            if any(m.startswith(legacy_prefix) for m in modules):
                offenders.append(str(path.relative_to(_ROOT)))
                break
    assert offenders == []


def test_keep_closure_contains_no_legacy_module() -> None:
    """The KEEP entry-point closure is fully contained in the KEEP src tree."""
    legacy_mods = _legacy_src_modules()
    seen: set[str] = set()

    def visit(module: str) -> None:
        if module in seen or module == "src":
            return
        seen.add(module)
        rel = module.replace(".", "/") + ".py"
        path = _ROOT / "src" / rel
        if not path.exists():
            path = _ROOT / "src" / module.replace(".", "/") / "__init__.py"
        if not path.exists():
            # Namespace packages / modules without a file (e.g. builtins) are
            # tolerated only if they are not legacy.
            assert module not in legacy_mods, f"closure hits legacy module {module}"
            return
        for dep in _src_imports(path):
            visit(dep)

    for entry in _KEEP_ENTRY_POINTS:
        visit(entry)

    closure_hits_legacy = sorted(
        {m for m in seen if m in legacy_mods or any(m == l or m.startswith(l + ".") for l in legacy_mods)}
    )
    assert closure_hits_legacy == []


def test_keep_src_modules_do_not_import_migrated_modules() -> None:
    """Every ``src/`` module's direct imports stay inside ``src/`` (KEEP)."""
    keep_mods = _keep_src_modules()

    def is_migrated(m: str) -> bool:
        return any(m == k or m.startswith(k + ".") for k in keep_mods) is False and m.startswith("src.")

    offenders: list[tuple[str, str]] = []
    for path in (_ROOT / "src").rglob("*.py"):
        offenders.extend(
            (str(path.relative_to(_ROOT)), dep)
            for dep in _src_imports(path)
            if is_migrated(dep)
        )
    assert offenders == []
