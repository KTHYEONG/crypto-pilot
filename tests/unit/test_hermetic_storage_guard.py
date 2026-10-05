"""Invariant scenarios for the hermetic storage-root guard."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.fixtures.hermetic import assert_storage_roots_hermetic


def test_contained_roots_pass(tmp_path: Path) -> None:
    assert assert_storage_roots_hermetic(
        {"BACKTESTS_DIR": tmp_path / "backtests", "LOG_DIR": tmp_path / "logs", "ROOT": tmp_path},
        tmp_path,
        ("src", "src.common.paths"),
    ) is None


def test_escaped_root_fails_loudly_with_diagnosis(tmp_path: Path) -> None:
    offending = tmp_path / "data" / "backtests"
    temp_root = tmp_path / "proc"
    temp_root.mkdir()
    with pytest.raises(pytest.UsageError) as excinfo:
        assert_storage_roots_hermetic(
            {"BACKTESTS_DIR": offending, "LOG_DIR": temp_root / "logs"},
            temp_root,
            ("src.common.paths",),
        )
    message = str(excinfo.value)
    assert "BACKTESTS_DIR" in message
    assert str(offending.resolve()) in message
    assert str(temp_root.resolve()) in message
    assert "src.common.paths" in message
    assert "LOG_DIR=" not in message


def test_all_offenders_reported_together(tmp_path: Path) -> None:
    temp_root = tmp_path / "proc"
    temp_root.mkdir()
    with pytest.raises(pytest.UsageError) as excinfo:
        assert_storage_roots_hermetic(
            {
                "BACKTESTS_DIR": tmp_path / "data" / "backtests",
                "LOG_DIR": tmp_path / "other" / "logs",
            },
            temp_root,
            (),
        )
    message = str(excinfo.value)
    assert "BACKTESTS_DIR" in message
    assert "LOG_DIR=" in message
    assert message.index("BACKTESTS_DIR=") < message.index("LOG_DIR=")
    assert "CRYPTO_PILOT_BACKTESTS_DIR/CRYPTO_PILOT_LOG_DIR" in message
    assert "preimported src modules: none" in message


def test_preimported_module_diagnosis_is_bounded(tmp_path: Path) -> None:
    modules = tuple(f"src.module_{index:02d}" for index in range(23))
    with pytest.raises(pytest.UsageError) as excinfo:
        assert_storage_roots_hermetic({"BACKTESTS_DIR": tmp_path / "outside"}, tmp_path / "proc", modules)
    message = str(excinfo.value)
    assert ", ".join(modules[:20]) in message
    assert "23 module(s)" in message
    assert "(+3 more)" in message
    assert all(name not in message for name in modules[20:])


def test_symlinked_root_resolves_before_containment(tmp_path: Path) -> None:
    temp_root = tmp_path / "proc"
    temp_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(pytest.UsageError):
        assert_storage_roots_hermetic(
            {"BACKTESTS_DIR": link},
            temp_root,
            (),
        )


def test_live_session_roots_are_partitioned() -> None:
    from src.common import logging as _src_logging
    from src.common import paths as _src_paths

    temp_root = Path(os.environ["PYTEST_DEBUG_TEMPROOT"]).resolve()
    for root in (_src_paths.BACKTESTS_DIR, _src_paths.FROZEN_BACKTESTS_DIR, _src_logging.LOG_DIR):
        assert root.resolve().is_relative_to(temp_root)
