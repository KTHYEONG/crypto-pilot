"""Confidentiality tests."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def test_docs_results_tracks_only_sealed_or_markdown() -> None:
    # skip if not inside git work tree
    try:
        subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree"], stderr=subprocess.DEVNULL)  # noqa: S603, S607
    except Exception:
        import pytest

        pytest.skip("not inside git work tree")
    out = subprocess.check_output(["git", "ls-files", "docs/results/"], text=True)  # noqa: S603, S607
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    violations = [l for l in lines if not (l.endswith(".enc") or l.endswith(".md"))]
    assert violations == [], f"violations: {violations}"


def test_daily_ledger_and_run_history_are_untracked() -> None:
    try:
        subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree"], stderr=subprocess.DEVNULL)  # noqa: S603, S607
    except Exception:
        import pytest

        pytest.skip("not inside git work tree")
    out = subprocess.check_output(["git", "ls-files"], text=True)  # noqa: S603, S607
    tracked = set(out.splitlines())
    assert "docs/results/mhs_horizon_diagnostic.json" not in tracked
    assert "docs/results/mhs_refactor_baseline.json" not in tracked
    assert "docs/results/mhs_horizon_diagnostic_artifacts/daily_ledger.parquet" not in tracked
    assert not any(p.startswith("docs/results/mhs_run_history/") for p in tracked)


def test_gitignore_seals_research_outputs() -> None:
    content = Path(".gitignore").read_text(encoding="utf-8")
    for line in ["docs/results/**/*.json", "docs/results/**/*.jsonl", "docs/results/**/*.parquet", "!docs/results/**/*.enc"]:
        assert line in content, f"missing {line}"


def test_dockerignore_ships_only_sealed_results() -> None:
    content = Path(".dockerignore").read_text(encoding="utf-8")
    assert "!docs/results/mhs_horizon_diagnostic_artifacts/*.parquet" not in content
    assert "!docs/results/mhs_horizon_diagnostic_artifacts/*.enc" in content


def test_source_tree_has_no_measured_performance_metrics() -> None:
    patterns = [
        re.compile(r"measured\s+CAGR"),
        re.compile(r"gave\s+CAGR"),
        re.compile(r"replay\s+measured"),
        re.compile(r"CAGR\s+[+-]?\d"),
        re.compile(r"Calmar\s+\d"),
        re.compile(r"MDD\s+-?\d"),
        re.compile(r"Sharpe\s+\+?\d"),
    ]
    violations: list[tuple[str, int, str]] = []
    for root, dirs, files in os.walk("src"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if fn.endswith(".py"):
                fp = os.path.join(root, fn)
                with open(fp, encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        for pat in patterns:
                            if pat.search(line):
                                violations.append((fp, i, line.strip()))
                                break
    assert violations == [], f"violations: {violations}"
