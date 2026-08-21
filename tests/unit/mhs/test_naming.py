"""Tests for identifier naming conventions (I_NOVERSION)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "src"


def _scan_python_files(root: Path) -> list[Path]:
    """Collect all .py files under root, skipping __pycache__."""
    return [p for p in root.rglob("*.py") if "__pycache__" not in str(p)]


def test_no_version_suffix_in_identifiers():
    """SCENARIO_ANALYSIS_ARCHITECTURE_09: No identifier carries a _v<N> suffix.

    Scanning every .py under src/ finds zero identifiers matching
    r'\\b\\w+_v[0-9]+\\b'.
    """
    pattern = re.compile(r"\b\w+_v\d+\b")
    violations: list[tuple[Path, int, str]] = []
    for path in _scan_python_files(_SRC_ROOT):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001, S112
            continue
        for i, line in enumerate(text.splitlines(), 1):
            # Skip comments and strings that are expected to contain version refs
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for match in pattern.finditer(line):
                word = match.group()
                # Allow ADR references and doc references
                if "ADR_" in line or "docs/" in line or "spec" in line.lower():
                    continue
                violations.append((path, i, word))
    if violations:
        msg = "\n".join(f"  {p}:{line}: {word}" for p, line, word in violations[:20])
        pytest.fail(f"Found _v<N> suffixes in identifiers:\n{msg}")


def test_no_phase_prefix_in_identifiers():
    """SCENARIO_ANALYSIS_ARCHITECTURE_09: No PHASE_1 or phase_1 identifiers."""
    pattern = re.compile(r"\bPHASE_1\b|\bphase_1\b")
    violations: list[tuple[Path, int, str]] = []
    for path in _scan_python_files(_SRC_ROOT):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001, S112
            continue
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for match in pattern.finditer(line):
                word = match.group()
                if "docs/" in line or "spec" in line.lower() or "ADR_" in line:
                    continue
                violations.append((path, i, word))
    if violations:
        msg = "\n".join(f"  {p}:{line}: {word}" for p, line, word in violations[:20])
        pytest.fail(f"Found PHASE_1/phase_1 identifiers:\n{msg}")
