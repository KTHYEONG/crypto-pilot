"""Invariant guards for the market-recorder deploy fingerprint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.application.ops.recorder_fingerprint import (
    FINGERPRINT_EXTRA_FILES,
    FingerprintError,
    compute_recorder_fingerprint,
    main,
    recorder_source_closure,
)

ROOT = Path(__file__).resolve().parents[3]


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _minimal_tree(root: Path, files: dict[str, str]) -> None:
    payload = {"uv.lock": "lock-1\n", "pyproject.toml": "[project]\n", "Dockerfile": "FROM x\n"}
    payload.update(files)
    _write_tree(root, payload)


ENTRY_FILES = "src/market_data/streams/recorder_main.py"


def test_recorder_source_closure_excludes_cli_and_strategy() -> None:
    """Closure covers the recorder entrypoint without CLI or strategy code."""
    closure = recorder_source_closure(ROOT)
    relative = sorted(path.relative_to(ROOT).as_posix() for path in closure)
    for required in (
        "src/market_data/streams/recorder_main.py",
        "src/market_data/streams/recorder.py",
        "src/market_data/streams/liquidations.py",
        "src/market_data/streams/snapshots.py",
        "src/market_data/streams/coverage.py",
        "src/live/lifecycle.py",
        "src/common/paths.py",
        "src/__init__.py",
        "src/common/__init__.py",
        "src/market_data/__init__.py",
        "src/market_data/streams/__init__.py",
        "src/live/__init__.py",
    ):
        assert required in relative
    assert [path for path in relative if path.startswith(("src/cli/", "src/mhs/", "src/backtests/"))] == []


def test_recorder_source_closure_includes_nested_and_type_checking_imports(tmp_path: Path) -> None:
    """Function-local and TYPE_CHECKING imports are part of the closure."""
    _write_tree(
        tmp_path,
        {
            "src/entry.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import src.b\n\ndef run():\n    import src.a\n",
            "src/a.py": "VALUE = 1\n",
            "src/b.py": "VALUE = 2\n",
        },
    )
    closure = recorder_source_closure(tmp_path, entry_module="src.entry")
    relative = sorted(path.relative_to(tmp_path).as_posix() for path in closure)
    assert "src/a.py" in relative
    assert "src/b.py" in relative


def test_recorder_source_closure_resolves_from_import_submodule(tmp_path: Path) -> None:
    """`from src.pkg import sub` includes the submodule and its package init."""
    _write_tree(
        tmp_path,
        {
            "src/entry.py": "from src.pkg import sub\n",
            "src/pkg/__init__.py": "VALUE = 1\n",
            "src/pkg/sub.py": "VALUE = 2\n",
        },
    )
    closure = recorder_source_closure(tmp_path, entry_module="src.entry")
    relative = sorted(path.relative_to(tmp_path).as_posix() for path in closure)
    assert "src/pkg/sub.py" in relative
    assert "src/pkg/__init__.py" in relative


def test_recorder_source_closure_fails_closed_on_unresolvable_module(tmp_path: Path) -> None:
    """An entry importing a missing src module raises instead of emitting a fingerprint."""
    _write_tree(tmp_path, {"src/entry.py": "import src.missing\n"})
    with pytest.raises(FingerprintError):
        recorder_source_closure(tmp_path, entry_module="src.entry")


def test_compute_recorder_fingerprint_is_deterministic(tmp_path: Path) -> None:
    """The same tree always yields the identical fingerprint string."""
    _minimal_tree(tmp_path, {ENTRY_FILES: "import src.a\n", "src/a.py": "VALUE = 1\n"})
    first = compute_recorder_fingerprint(tmp_path)
    second = compute_recorder_fingerprint(tmp_path)
    assert first == second
    assert first.startswith("sha256:")


def test_compute_recorder_fingerprint_changes_on_closure_byte_change(tmp_path: Path) -> None:
    """One byte changed in an included module alters the fingerprint."""
    _minimal_tree(tmp_path, {ENTRY_FILES: "import src.a\n", "src/a.py": "VALUE = 1\n"})
    before = compute_recorder_fingerprint(tmp_path)
    (tmp_path / "src" / "a.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert compute_recorder_fingerprint(tmp_path) != before


def test_compute_recorder_fingerprint_ignores_unrelated_files(tmp_path: Path) -> None:
    """Changes outside the closure (CLI code) leave the fingerprint unchanged."""
    _minimal_tree(
        tmp_path,
        {
            ENTRY_FILES: "import src.a\n",
            "src/a.py": "VALUE = 1\n",
            "src/cli/unrelated.py": "VALUE = 1\n",
        },
    )
    before = compute_recorder_fingerprint(tmp_path)
    assert "src/cli/unrelated.py" not in [
        path.relative_to(tmp_path).as_posix() for path in recorder_source_closure(tmp_path)
    ]
    (tmp_path / "src" / "cli" / "unrelated.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert compute_recorder_fingerprint(tmp_path) == before


def test_compute_recorder_fingerprint_covers_lock_and_image_recipe(tmp_path: Path) -> None:
    """Dependency lock changes alter the fingerprint; a missing Dockerfile fails closed."""
    _minimal_tree(tmp_path, {ENTRY_FILES: "VALUE = 1\n"})
    before = compute_recorder_fingerprint(tmp_path)
    assert set(FINGERPRINT_EXTRA_FILES) == {"uv.lock", "pyproject.toml", "Dockerfile"}
    (tmp_path / "uv.lock").write_text("lock-2\n", encoding="utf-8")
    assert compute_recorder_fingerprint(tmp_path) != before
    (tmp_path / "Dockerfile").unlink()
    with pytest.raises(FingerprintError):
        compute_recorder_fingerprint(tmp_path)


def test_recorder_entrypoint_stays_import_light() -> None:
    """Importing the recorder entrypoint pulls in no CLI, strategy, or ccxt modules."""
    result = subprocess.run(  # noqa: S603 - fixed argv: this interpreter + a static import probe
        [
            sys.executable,
            "-c",
            "import src.market_data.streams.recorder_main, sys; "
            "print(sorted(m for m in sys.modules if m.startswith(('src.cli','ccxt','src.mhs'))))",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "[]"


def test_recorder_fingerprint_cli_lists_closure_and_prints_fingerprint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`--list` prints closure paths before the single fingerprint line."""
    _minimal_tree(tmp_path, {ENTRY_FILES: "import src.a\n", "src/a.py": "VALUE = 1\n"})
    assert main(["--root", str(tmp_path), "--list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1].startswith("sha256:")
    assert "src/market_data/streams/recorder_main.py" in lines
    assert "src/a.py" in lines


def _rel_tree(tmp_path: Path, entry_body: str) -> Path:
    _minimal_tree(tmp_path, {
        "src/__init__.py": "",
        "src/pkg/__init__.py": "",
        "src/pkg/sib.py": "X = 1\n",
        "src/pkg/sub/__init__.py": "from . import leaf\n",
        "src/pkg/sub/leaf.py": "",
        "src/market_data/__init__.py": "",
        "src/market_data/streams/__init__.py": "",
        "src/market_data/streams/recorder_main.py": entry_body,
    })
    return tmp_path


def test_closure_follows_relative_imports_from_module_and_package(tmp_path: Path) -> None:
    root = _rel_tree(tmp_path, "from ...pkg import sub\nfrom ...pkg import sib\n")
    rels = {p.relative_to(root).as_posix() for p in recorder_source_closure(root)}
    # __init__ 안의 `from . import leaf` 도 패키지 기준으로 해석되어야 한다.
    assert {"src/pkg/sib.py", "src/pkg/sub/__init__.py", "src/pkg/sub/leaf.py"} <= rels


def test_closure_relative_import_escaping_src_fails_closed(tmp_path: Path) -> None:
    root = _rel_tree(tmp_path, "from .... import nothing\n")
    with pytest.raises(FingerprintError, match="escapes src"):
        recorder_source_closure(root)


def test_closure_unparseable_module_fails_closed(tmp_path: Path) -> None:
    _minimal_tree(tmp_path, {
        "src/__init__.py": "",
        "src/market_data/__init__.py": "",
        "src/market_data/streams/__init__.py": "",
        "src/market_data/streams/recorder_main.py": "def broken(:\n",
    })
    with pytest.raises(FingerprintError, match="cannot parse"):
        recorder_source_closure(tmp_path)


def test_closure_from_import_of_plain_symbol_keeps_only_its_module(tmp_path: Path) -> None:
    _minimal_tree(tmp_path, {
        "src/__init__.py": "",
        "src/pkg/__init__.py": "",
        "src/pkg/sib.py": "VALUE = 1\n",
        "src/market_data/__init__.py": "",
        "src/market_data/streams/__init__.py": "",
        "src/market_data/streams/recorder_main.py": "from src.pkg.sib import VALUE\n",
    })
    rels = {p.relative_to(tmp_path).as_posix() for p in recorder_source_closure(tmp_path)}
    assert "src/pkg/sib.py" in rels
