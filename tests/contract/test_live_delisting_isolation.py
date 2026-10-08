"""Live never uses research-only delisting exclusions (spec 34 part 4)."""

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN = (
    "strategy_blocked_decisions",
    "source_gap_excluded_symbols",
    "SOURCE_GAP_EXCLUDED_SYMBOLS",
    "structurally_excluded_symbols",
    "instrument_settlements",
    "venue_halts",
    "src.engine.execution.settlement",
)


def _live_files() -> list[Path]:
    return [p for p in Path("src/live").rglob("*.py") if "__pycache__" not in p.parts]


def test_live_never_uses_research_exclusions() -> None:
    offenders: list[str] = []
    for path in _live_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                haystack = f"{node.module} {' '.join(a.name for a in node.names)}"
                if any(token in haystack for token in _FORBIDDEN):
                    offenders.append(f"{path}:{node.lineno}")
            elif isinstance(node, ast.Import):
                haystack = " ".join(a.name for a in node.names)
                if any(token in haystack for token in _FORBIDDEN):
                    offenders.append(f"{path}:{node.lineno}")
            elif (isinstance(node, ast.Name) and node.id in _FORBIDDEN) or (isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN):
                offenders.append(f"{path}:{node.lineno}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(token in node.value for token in _FORBIDDEN):
                    offenders.append(f"{path}:{node.lineno}")
    assert offenders == [], f"live imports research exclusions: {offenders}"


def test_no_import_time_registry_reads() -> None:
    import subprocess
    import sys

    probe = (
        "import unittest.mock as mock\n"
        "def _boom(*a, **k):\n"
        "    raise AssertionError('registry read at import')\n"
        "with mock.patch('src.core.instrument_settlements.load_instrument_settlement_registry', side_effect=_boom):\n"
        "    with mock.patch('src.core.venue_halts.load_venue_halt_registry', side_effect=_boom):\n"
        "        with mock.patch('src.core.source_gaps.load_source_gap_registry', side_effect=_boom):\n"
        "            import src.live.strategy_signal\n"
        "print('ok')"
    )
    completed = subprocess.run(  # noqa: S603 - fixed sys.executable argv in contract probe
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=".",
    )
    assert completed.returncode == 0, f"live import reads registries: {completed.stderr}"
    assert "ok" in completed.stdout
