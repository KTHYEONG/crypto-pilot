"""Invariant guards for the raw-first capture deploy fingerprint."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.ops.capture_fingerprint import (
    CAPTURE_LOCK_ROOT_PACKAGE,
    FingerprintError,
    capture_source_files,
    compute_capture_fingerprint,
    locked_dependency_blocks,
    main,
)

ROOT = Path(__file__).resolve().parents[3]

AIOHTTP_LOCK = """[[package]]
name = "aiohttp"
version = "3.14.1"
dependencies = [
    { name = "yarl" },
]

[[package]]
name = "yarl"
version = "1.24.2"

[[package]]
name = "pandas"
version = "2.2.0"
"""


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _minimal_tree(root: Path, files: dict[str, str]) -> None:
    payload = {"uv.lock": AIOHTTP_LOCK, "src/__init__.py": ""}
    payload.update(files)
    _write_tree(root, payload)


def test_shared_file_edit_keeps_fingerprint(tmp_path: Path) -> None:
    """Untracked shared files and non-aiohttp lock entries leave the fingerprint unchanged."""
    _minimal_tree(
        tmp_path,
        {
            "src/capture/__init__.py": "",
            "src/capture/main.py": "VALUE = 1\n",
            "src/common/paths.py": "VALUE = 1\n",
        },
    )
    before = compute_capture_fingerprint(tmp_path)
    (tmp_path / "src" / "common" / "paths.py").write_text("VALUE = 2\n", encoding="utf-8")
    lock = (tmp_path / "uv.lock").read_text(encoding="utf-8").replace('version = "2.2.0"', 'version = "2.3.0"')
    (tmp_path / "uv.lock").write_text(lock, encoding="utf-8")
    assert compute_capture_fingerprint(tmp_path) == before


def test_capture_source_edit_changes_fingerprint(tmp_path: Path) -> None:
    """One byte changed in a capture source alters the fingerprint."""
    _minimal_tree(tmp_path, {"src/capture/__init__.py": "", "src/capture/main.py": "VALUE = 1\n"})
    before = compute_capture_fingerprint(tmp_path)
    (tmp_path / "src" / "capture" / "main.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert compute_capture_fingerprint(tmp_path) != before


def test_aiohttp_transitive_lock_change_changes_fingerprint(tmp_path: Path) -> None:
    """A transitive aiohttp dependency version change alters the fingerprint."""
    _minimal_tree(tmp_path, {"src/capture/__init__.py": "", "src/capture/main.py": "VALUE = 1\n"})
    before = compute_capture_fingerprint(tmp_path)
    lock = (tmp_path / "uv.lock").read_text(encoding="utf-8").replace('version = "1.24.2"', 'version = "1.25.0"')
    (tmp_path / "uv.lock").write_text(lock, encoding="utf-8")
    assert compute_capture_fingerprint(tmp_path) != before


def test_first_party_import_outside_capture_fails_build(tmp_path: Path) -> None:
    """A capture file importing outside src.capture raises instead of emitting a fingerprint."""
    _minimal_tree(
        tmp_path,
        {"src/capture/__init__.py": "", "src/capture/main.py": "from src.common.paths import X\n"},
    )
    with pytest.raises(FingerprintError):
        compute_capture_fingerprint(tmp_path)


def test_plain_import_outside_capture_fails_build(tmp_path: Path) -> None:
    """A plain `import src.common...` is also a boundary breach."""
    _minimal_tree(
        tmp_path,
        {"src/capture/__init__.py": "", "src/capture/main.py": "import src.common.paths\n"},
    )
    with pytest.raises(FingerprintError):
        compute_capture_fingerprint(tmp_path)


def test_relative_import_inside_capture_is_allowed(tmp_path: Path) -> None:
    """Relative imports resolving inside src.capture are part of the closure."""
    _minimal_tree(
        tmp_path,
        {"src/capture/__init__.py": "", "src/capture/main.py": "from .config import X\n", "src/capture/config.py": "X = 1\n"},
    )
    compute_capture_fingerprint(tmp_path)


def test_relative_import_escaping_capture_fails_build(tmp_path: Path) -> None:
    """A relative import escaping src.capture raises instead of emitting a fingerprint."""
    _minimal_tree(
        tmp_path,
        {"src/capture/__init__.py": "", "src/capture/main.py": "from ..common import paths\n"},
    )
    with pytest.raises(FingerprintError):
        compute_capture_fingerprint(tmp_path)


def test_missing_capture_package_fails_build(tmp_path: Path) -> None:
    """A root without src/capture raises instead of emitting a fingerprint."""
    (tmp_path / "uv.lock").write_text(AIOHTTP_LOCK, encoding="utf-8")
    with pytest.raises(FingerprintError):
        compute_capture_fingerprint(tmp_path)


def test_unparseable_capture_file_fails_build(tmp_path: Path) -> None:
    """A syntactically broken capture file raises instead of emitting a fingerprint."""
    _minimal_tree(
        tmp_path,
        {"src/capture/__init__.py": "", "src/capture/main.py": "def broken(:\n"},
    )
    with pytest.raises(FingerprintError):
        compute_capture_fingerprint(tmp_path)


def test_missing_lock_block_fails_build(tmp_path: Path) -> None:
    """A lock without an aiohttp block raises instead of emitting a fingerprint."""
    _minimal_tree(tmp_path, {"src/capture/__init__.py": "", "src/capture/main.py": "VALUE = 1\n"})
    (tmp_path / "uv.lock").write_text('[[package]]\nname = "pandas"\nversion = "2.2.0"\n', encoding="utf-8")
    with pytest.raises(FingerprintError):
        compute_capture_fingerprint(tmp_path)


def test_missing_transitive_lock_block_fails_build(tmp_path: Path) -> None:
    """An aiohttp dependency without a lock block raises instead of emitting a fingerprint."""
    _minimal_tree(tmp_path, {"src/capture/__init__.py": "", "src/capture/main.py": "VALUE = 1\n"})
    dangling = AIOHTTP_LOCK.replace('name = "yarl"\nversion = "1.24.2"\n', "")
    (tmp_path / "uv.lock").write_text(dangling, encoding="utf-8")
    with pytest.raises(FingerprintError, match="yarl"):
        compute_capture_fingerprint(tmp_path)


def test_missing_lock_file_fails_build(tmp_path: Path) -> None:
    """A root without uv.lock raises instead of emitting a fingerprint."""
    _write_tree(tmp_path, {"src/__init__.py": "", "src/capture/__init__.py": "", "src/capture/main.py": "VALUE = 1\n"})
    with pytest.raises(FingerprintError, match=r"uv\.lock"):
        compute_capture_fingerprint(tmp_path)


def test_compute_capture_fingerprint_is_deterministic(tmp_path: Path) -> None:
    """The same tree always yields the identical fingerprint string."""
    _minimal_tree(tmp_path, {"src/capture/__init__.py": "", "src/capture/main.py": "VALUE = 1\n"})
    first = compute_capture_fingerprint(tmp_path)
    second = compute_capture_fingerprint(tmp_path)
    assert first == second
    assert first.startswith("sha256:")


def test_capture_sources_cover_package_and_root_init(tmp_path: Path) -> None:
    """Closure is every *.py under src/capture plus src/__init__.py, sorted."""
    _minimal_tree(
        tmp_path,
        {
            "src/capture/__init__.py": "",
            "src/capture/main.py": "VALUE = 1\n",
            "src/capture/sub.py": "VALUE = 2\n",
        },
    )
    relative = [path.relative_to(tmp_path).as_posix() for path in capture_source_files(tmp_path)]
    assert relative == sorted(relative)
    assert "src/__init__.py" in relative
    assert "src/capture/main.py" in relative
    assert "src/capture/sub.py" in relative


def test_locked_blocks_exclude_non_aiohttp_packages() -> None:
    """Only aiohttp and its transitive deps are fingerprinted."""
    blocks = locked_dependency_blocks(AIOHTTP_LOCK, CAPTURE_LOCK_ROOT_PACKAGE)
    names = [
        next(line.split("=", 1)[1].strip().strip('"') for line in block.splitlines() if line.startswith("name "))
        for block in blocks
    ]
    assert "aiohttp" in names
    assert "yarl" in names
    assert "pandas" not in names


def test_capture_fingerprint_cli_lists_closure_and_prints_fingerprint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`--list` prints closure paths before the single fingerprint line."""
    _minimal_tree(tmp_path, {"src/capture/__init__.py": "", "src/capture/main.py": "VALUE = 1\n"})
    assert main(["--root", str(tmp_path), "--list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1].startswith("sha256:")
    assert "src/capture/main.py" in lines
