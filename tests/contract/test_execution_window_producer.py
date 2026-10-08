"""Single window producer: settlement events have exactly one production source."""

from __future__ import annotations

import ast
from pathlib import Path


def _call_sites(path: Path, func: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            name = ""
            if isinstance(target, ast.Name):
                name = target.id
            elif isinstance(target, ast.Attribute):
                name = target.attr
            if name == func:
                lines.append(node.lineno)
    return lines


def test_single_window_producer() -> None:
    src = Path("src")
    window_ctors = []
    event_ctors = []
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "ExecutionReplayWindow(" not in text and "InstrumentSettlementEvent(" not in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            name = target.id if isinstance(target, ast.Name) else (target.attr if isinstance(target, ast.Attribute) else "")
            rel = str(path)
            if name == "ExecutionReplayWindow":
                window_ctors.append(rel)
            if name == "InstrumentSettlementEvent":
                event_ctors.append(rel)
    assert sorted(set(window_ctors)) == sorted([
        "src/engine/execution/window_stream.py",
        "src/lab/mhs/evaluation/windows.py",
    ])
    assert set(event_ctors) <= {
        "src/engine/execution/settlement.py",
        "src/engine/execution/contracts.py",
    }
    assert "src/engine/execution/settlement.py" in set(event_ctors)
