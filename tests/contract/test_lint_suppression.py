"""Contract: no file-level blanket linter/type-checker suppressions under src/."""

from __future__ import annotations

from pathlib import Path


def _find_blanket_suppressions(root: Path) -> list[str]:
    """Scan every ``src/**/*.py`` for wholesale suppressions.

    Reports ``path:line`` for a bare file-level ``# ruff: noqa`` (no ``:``
    codes) or any line containing ``mypy: ignore-errors``. Code-scoped
    headers such as ``# ruff: noqa: RUF002 -- rationale`` are allowed.
    """
    offenders: list[str] = []
    for path in sorted((root / "src").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            bare_noqa = stripped.startswith("# ruff: noqa") and ":" not in stripped.split("# ruff: noqa", 1)[1].split("--", 1)[0]
            if bare_noqa or "mypy: ignore-errors" in line:
                offenders.append(f"{path}:{lineno}")
    return offenders


def test_src_has_no_blanket_suppression_headers() -> None:
    """No module under src/ disables a linter or type checker wholesale.

    File-level ``# ruff: noqa`` without codes and ``# mypy: ignore-errors`` hid
    unused imports, undefined names and a type-contract lie in production code;
    exemptions must be line-scoped and code-scoped with a stated rationale.
    """
    offenders = _find_blanket_suppressions(Path("."))
    assert offenders == [], f"blanket suppressions: {offenders}"


def test_detector_flags_bare_header(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("# ruff: noqa\nx = 1\n", encoding="utf-8")
    assert _find_blanket_suppressions(tmp_path) == [f"{src / 'a.py'}:1"]


def test_detector_allows_code_scoped_header(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("# ruff: noqa: RUF002 -- rationale\nx = 1\n", encoding="utf-8")
    assert _find_blanket_suppressions(tmp_path) == []


def test_detector_flags_mypy_ignore_errors(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("from __future__ import annotations  # mypy: ignore-errors\n", encoding="utf-8")
    assert _find_blanket_suppressions(tmp_path) == [f"{src / 'a.py'}:1"]
