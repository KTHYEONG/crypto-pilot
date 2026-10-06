"""Identifier naming conventions with a frozen persisted-identifier allowlist."""

from __future__ import annotations

import io
import re
import tokenize
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_SRC_ROOT: Final[Path] = _REPO_ROOT / "src"

# persisted contract identifiers; renaming requires data migration
PERSISTED_VERSIONED_IDENTIFIERS: Final[dict[tuple[str, str], int]] = {
    ("src/cli/commands/backtest.py", "_control_v2"): 1,
    ("src/live/frozen_book.py", "frozen_mhs_top20_v2"): 1,
    ("src/live/recorder_watch.py", "heartbeat_v3"): 1,
    ("src/market_data/services/spot_collection.py", "log1p_geometric_bridge_v1"): 1,
    ("src/market_data/streams/heartbeat_v3.py", "heartbeat_v3"): 1,
    ("src/market_data/streams/normalizer.py", "heartbeat_v3"): 1,
    ("src/mhs/contracts.py", "zombie_mask_v1"): 3,
    ("src/mhs/data_policy.py", "zombie_mask_v1"): 4,
    ("src/mhs/frozen_research_candidate.py", "frozen_mhs_top20_growth_v2"): 1,
    ("src/mhs/frozen_research_candidate.py", "frozen_mhs_top20_v2"): 2,
    ("src/mhs/frozen_research_candidate.py", "frozen_mhs_top40_control_v2"): 1,
    ("src/mhs/panel.py", "zombie_mask_v1"): 2,
    ("src/mhs/run_history.py", "zombie_mask_v1"): 1,
}

_VERSION_PATTERN = re.compile(r"\b\w+_v\d+\b")
_PHASE_PATTERN = re.compile(r"\bPHASE_1\b|\bphase_1\b")


def _code_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (line number, line text with every comment token removed) for one source file.

    Comments are not identifiers, so full-line and trailing comments are excluded via
    ``tokenize``; string literals and docstrings stay scanned because persisted
    identifiers live in them. Raises on unreadable or untokenizable files instead of
    skipping them, so a scan can never silently cover fewer files.
    """
    text = path.read_text(encoding="utf-8")
    comment_cols: dict[int, int] = {}
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            comment_cols[tok.start[0]] = tok.start[1]
    lines = text.splitlines()
    for lineno, line in enumerate(lines, 1):
        yield lineno, line[: comment_cols[lineno]] if lineno in comment_cols else line


def _scan_python_files(root: Path) -> list[Path]:
    """Collect all .py files under root, skipping __pycache__."""
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _rel(path: Path, src_root: Path) -> str:
    return path.relative_to(src_root.parent).as_posix()


def _collect_versioned(
    src_root: Path,
) -> tuple[Counter[tuple[str, str]], list[tuple[str, int, str]]]:
    """Count ``_v<N>`` identifier occurrences per (path, identifier)."""
    counts: Counter[tuple[str, str]] = Counter()
    details: list[tuple[str, int, str]] = []
    for path in sorted(_scan_python_files(src_root)):
        rel = _rel(path, src_root)
        for lineno, line in _code_lines(path):
            if "ADR_" in line or "docs/" in line or "spec" in line.lower():
                continue
            for match in _VERSION_PATTERN.finditer(line):
                counts[(rel, match.group())] += 1
                details.append((rel, lineno, match.group()))
    return counts, details


def _check_versioned(src_root: Path, expected: dict[tuple[str, str], int]) -> None:
    """Enforce the frozen versioned-identifier allowlist against a source root."""
    counts, details = _collect_versioned(src_root)
    new_lines = [
        f"  {rel}:{lineno}: {word}"
        for rel, lineno, word in details
        if (rel, word) not in expected or counts[(rel, word)] > expected[(rel, word)]
    ]
    stale = sorted(
        f"  {rel} {word}: measured {counts.get((rel, word), 0)} vs frozen {frozen}"
        for (rel, word), frozen in expected.items()
        if counts.get((rel, word), 0) != frozen
        and ((rel, word) not in counts or counts[(rel, word)] < frozen)
    )
    parts = []
    if new_lines:
        parts.append("NEW versioned identifiers:\n" + "\n".join(new_lines))
    if stale:
        parts.append("STALE allowlist entries (shrink or delete):\n" + "\n".join(stale))
    assert not parts, "\n".join(parts)


def _collect_phase(src_root: Path) -> list[tuple[str, int, str]]:
    """Collect ``PHASE_1``/``phase_1`` identifier occurrences."""
    found: list[tuple[str, int, str]] = []
    for path in sorted(_scan_python_files(src_root)):
        rel = _rel(path, src_root)
        for lineno, line in _code_lines(path):
            if "ADR_" in line or "docs/" in line or "spec" in line.lower():
                continue
            found.extend((rel, lineno, match.group()) for match in _PHASE_PATTERN.finditer(line))
    return found


def _check_phase(src_root: Path) -> None:
    """Forbid every ``PHASE_1``/``phase_1`` identifier under a source root."""
    found = _collect_phase(src_root)
    assert found == [], "PHASE_1/phase_1 identifiers:\n" + "\n".join(
        f"  {rel}:{lineno}: {word}" for rel, lineno, word in found
    )


def test_naming_scan_covers_real_source_tree() -> None:
    """The naming scan covers the repository's real source tree."""
    assert _SRC_ROOT == _REPO_ROOT / "src"
    assert _SRC_ROOT.is_dir()
    scanned = _scan_python_files(_SRC_ROOT)
    assert scanned, "naming scan found no files"
    assert (_SRC_ROOT / "mhs" / "data_policy.py") in scanned


def test_no_version_suffix_in_identifiers() -> None:
    """SCENARIO_ANALYSIS_ARCHITECTURE_09: only frozen persisted _v<N> identifiers remain."""
    _check_versioned(_SRC_ROOT, PERSISTED_VERSIONED_IDENTIFIERS)


def test_no_phase_prefix_in_identifiers() -> None:
    """SCENARIO_ANALYSIS_ARCHITECTURE_09: No PHASE_1 or phase_1 identifiers."""
    _check_phase(_SRC_ROOT)


def test_new_versioned_identifier_fails(tmp_path: Path) -> None:
    """A new _v<N> identifier fails as NEW while absent pairs report STALE."""
    src_root = tmp_path / "src"
    (src_root / "mhs").mkdir(parents=True)
    real = _SRC_ROOT / "mhs" / "data_policy.py"
    (src_root / "mhs" / "data_policy.py").write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
    (src_root / "new_mod.py").write_text("foo_v2 = 1\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="foo_v2"):
        _check_versioned(src_root, PERSISTED_VERSIONED_IDENTIFIERS)
    with pytest.raises(AssertionError, match="STALE"):
        _check_versioned(src_root, PERSISTED_VERSIONED_IDENTIFIERS)


def test_stale_allowlist_entry_fails(tmp_path: Path) -> None:
    """A shrunk persisted identifier fails as STALE with measured vs frozen counts."""
    src_root = tmp_path / "src"
    (src_root / "mhs").mkdir(parents=True)
    (src_root / "mhs" / "panel.py").write_text("zombie_mask_v1 = 1\n", encoding="utf-8")
    with pytest.raises(
        AssertionError,
        match=r"src/mhs/panel\.py.*zombie_mask_v1.*measured 1 vs frozen 2",
    ):
        _check_versioned(
            src_root, {("src/mhs/panel.py", "zombie_mask_v1"): 2}
        )


def test_comments_are_not_scanned(tmp_path: Path) -> None:
    """Comment-only version tokens never count as identifiers."""
    src_root = tmp_path / "src"
    src_root.mkdir(parents=True)
    (src_root / "mod.py").write_text("x = 1  # was foo_v1\n", encoding="utf-8")
    counts, _ = _collect_versioned(src_root)
    assert counts == {}
    _check_versioned(src_root, {})


def test_phase_prefix_stays_forbidden(tmp_path: Path) -> None:
    """PHASE_1 in a fresh tree fails the phase check."""
    src_root = tmp_path / "src"
    src_root.mkdir(parents=True)
    (src_root / "mod.py").write_text("PHASE_1 = 1\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="PHASE_1"):
        _check_phase(src_root)
