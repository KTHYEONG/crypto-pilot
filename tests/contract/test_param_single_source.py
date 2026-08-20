"""I2 PARAM SINGLE SOURCE contract.

For every module-level uppercase constant name ``N`` in ``src/**``, exactly one
file may assign ``N`` at module level. Violations are reported as
``{name: [files]}``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

_CONSTANT_RE = re.compile(r"^_?[A-Z][A-Z0-9_]*$")


def _src_py_files() -> list[Path]:
    return sorted((_ROOT / "src").rglob("*.py"))


def _module_level_constants(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and _CONSTANT_RE.match(target.id):
                    names.add(target.id)
    return names


def test_module_level_constants_declared_in_exactly_one_file() -> None:
    declared: dict[str, list[str]] = {}
    for path in _src_py_files():
        rel = str(path.relative_to(_ROOT))
        for name in _module_level_constants(path):
            declared.setdefault(name, []).append(rel)
    violations = {name: files for name, files in declared.items() if len(files) != 1}
    assert violations == {}, f"constants declared in multiple files: {violations}"
