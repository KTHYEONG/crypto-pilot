"""Contract: the capture package imports only stdlib, aiohttp and itself."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

CAPTURE_ROOT = Path(__file__).resolve().parents[2] / "src" / "capture"


def _allowed_top_level() -> set[str]:
    return set(sys.stdlib_module_names)


def test_capture_package_imports_only_stdlib_aiohttp_and_itself() -> None:
    """AST-scan every module under src/capture; each Import/ImportFrom top-level name must be in
    ``sys.stdlib_module_names`` or be ``aiohttp`` or start with ``src.capture``. Relative imports
    are allowed. Fails listing ``file:line module``.
    """
    assert CAPTURE_ROOT.is_dir(), f"capture package missing at {CAPTURE_ROOT}"
    allowed = _allowed_top_level()
    violations: list[str] = []
    for path in sorted(CAPTURE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in allowed and top != "aiohttp":
                        violations.append(f"{path}:{node.lineno} {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                module = (node.module or "").split(".")[0]
                if not module:
                    continue
                if module in allowed or module == "aiohttp" or (node.module or "").startswith("src.capture"):
                    continue
                violations.append(f"{path}:{node.lineno} {node.module}")
    assert not violations, "forbidden imports under src/capture/:\n" + "\n".join(violations)
